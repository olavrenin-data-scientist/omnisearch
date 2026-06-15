"""
Load a trained HAPPO checkpoint and expose it as a VMAS action_fn.

HARL saves one ``actor_agentN.pt`` file per agent after training. This module
reconstructs those actors (the network shape is determined by `algo_args` and
the env's spaces), restores their state dicts, and wraps them in a callable
that ``export_trajectory`` / ``compare_baselines`` can use as the policy.

Usage::

    from agents.happo_policy import HappoPolicy
    policy = HappoPolicy.from_checkpoint(
        checkpoint_dir="results/harl_runs/.../models",
        algo_args=None,         # uses default smoke args if None
        deterministic=True,
    )
    # `policy` is callable: policy(env) -> per-agent action list
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from gymnasium.spaces import Box

from agents.happo_checkpoint import load_training_manifest
from agents.harl_runner import default_algo_args


def _scenario_kwargs_from_manifest(manifest: Optional[dict]) -> dict:
    if not manifest:
        return {}
    env_args = manifest.get("env_args", {})
    scenario_kwargs = env_args.get("scenario_kwargs", {})
    return copy.deepcopy(scenario_kwargs) if isinstance(scenario_kwargs, dict) else {}


class HappoPolicy:
    """Trained HAPPO actor wrapped as a VMAS-style ``policy(env) -> actions``."""

    def __init__(
        self,
        checkpoint_dir: Path,
        algo_args: dict,
        deterministic: bool = True,
        scenario_kwargs: Optional[dict] = None,
    ):
        from harl.algorithms.actors.happo import HAPPO

        self.checkpoint_dir = Path(checkpoint_dir)
        self.algo_args      = algo_args
        self.deterministic  = deterministic
        self._device        = torch.device("cpu")

        # The actor expects a flat namespace with model + algo + train sections.
        # Build the input by merging the trio into one dict (it just .gets() them).
        merged = {
            **algo_args["model"],
            **algo_args["algo"],
            **algo_args["train"],
        }

        # We need obs_space and action_space PER AGENT. Build a temporary VMAS
        # env to read them off (they're scenario-dependent).
        import vmas
        from envs.wildfire_search import WildfireSearchScenario
        tmp = vmas.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            seed=0,
            **copy.deepcopy(scenario_kwargs or {}),
        )
        tmp.reset()
        self.agent_names: List[str] = [a.name for a in tmp.agents]
        action_spaces: List[Box] = []
        for a in tmp.agents:
            sz = tmp.get_agent_action_size(a)
            action_spaces.append(Box(-1.0, 1.0, (sz,), dtype=np.float32))
        obs_spaces: List[Box] = list(tmp.observation_space.spaces)

        # Build one HAPPO actor per agent, restore the state dict.
        self.actors = []
        for i, (obs_s, act_s) in enumerate(zip(obs_spaces, action_spaces)):
            actor = HAPPO(merged, obs_s, act_s, self._device)
            state = torch.load(
                self.checkpoint_dir / f"actor_agent{i}.pt",
                map_location=self._device,
                weights_only=True,
            )
            actor.actor.load_state_dict(state)
            actor.actor.eval()
            self.actors.append(actor)

        # Recurrent params (we don't use recurrence — non-RNN actors still
        # require the rnn_state/masks tensors to be passed in).
        self._recurrent_n     = algo_args["model"]["recurrent_n"]
        self._rnn_hidden_size = algo_args["model"]["hidden_sizes"][-1]

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        algo_args: Optional[dict] = None,
        deterministic: bool = True,
    ) -> "HappoPolicy":
        manifest = load_training_manifest(checkpoint_dir)
        scenario_kwargs = _scenario_kwargs_from_manifest(manifest)
        if algo_args is None:
            if manifest is not None:
                algo_args = manifest["algo_args"]
            else:
                algo_args = default_algo_args()
                # HARL's config is a fallback for checkpoints created before the
                # OmniSearch manifest existed, including recurrent policies.
                cfg_path = Path(checkpoint_dir).parent / "config.json"
                if cfg_path.exists():
                    try:
                        import json

                        with cfg_path.open(encoding="utf-8") as config_file:
                            saved = json.load(config_file)
                        saved_model = saved.get("algo_args", {}).get("model", {})
                        algo_args = {
                            **algo_args,
                            "model": {**algo_args["model"], **saved_model},
                        }
                    except (OSError, TypeError, ValueError):
                        pass
        return cls(checkpoint_dir, algo_args, deterministic, scenario_kwargs=scenario_kwargs)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear recurrent hidden state (call at episode boundaries)."""
        self._rnn_states = None

    def __call__(self, env) -> List[torch.Tensor]:
        """Return per-agent action tensors of shape (num_envs, action_dim)."""
        out: List[torch.Tensor] = []
        B = env.scenario.world.batch_dim
        # Persist per-agent RNN hidden state across steps — otherwise a recurrent
        # policy is reset to zero memory every step and behaves like a
        # feedforward one. (No-op for non-recurrent policies.)
        if getattr(self, "_rnn_states", None) is None or self._rnn_states[0].shape[0] != B:
            self._rnn_states = [
                np.zeros((B, self._recurrent_n, self._rnn_hidden_size), dtype=np.float32)
                for _ in env.agents
            ]
        for i, agent in enumerate(env.agents):
            # Pull current obs via the scenario hook — this is the same
            # path that training used.
            obs = env.scenario.observation(agent).cpu().numpy()  # (B, obs_dim)
            masks = np.ones((B, 1), dtype=np.float32)
            with torch.no_grad():
                # HAPPO.act returns (actions, rnn_states_actor)
                actions, rnn_out = self.actors[i].act(
                    obs, self._rnn_states[i], masks,
                    available_actions=None,
                    deterministic=self.deterministic,
                )
            self._rnn_states[i] = (
                rnn_out.cpu().numpy() if isinstance(rnn_out, torch.Tensor) else np.asarray(rnn_out)
            )
            if isinstance(actions, torch.Tensor):
                a_np = actions.cpu().numpy()
            else:
                a_np = np.asarray(actions)
            # Match the train-time HARL adapter's bounded action command.
            a_np = np.clip(a_np, -1.0, 1.0).astype(np.float32)
            out.append(torch.from_numpy(a_np))
        return out


# ----------------------------------------------------------------------
# Helper: locate the most recent HARL run directory
# ----------------------------------------------------------------------
def find_latest_happo_checkpoint(root: str | Path = None) -> Path:
    """
    Return the most-recently-written models/ dir under results/harl_runs/.

    HARL stores each run in
        <log_dir>/wildfire/wildfire_search/happo/<exp_name>/seed-<N>-<ts>/models/
    We pick the newest one by directory mtime.
    """
    root = Path(root or "results/harl_runs")
    candidates = list(root.rglob("models"))
    if not candidates:
        raise FileNotFoundError(
            f"No HAPPO checkpoint found under {root}. "
            "Run scripts/train_happo_smoke.py first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)
