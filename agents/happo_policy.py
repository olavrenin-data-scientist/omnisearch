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
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from gymnasium.spaces import Box

from agents.action_transform import transform_continuous_action
from agents.happo_checkpoint import load_training_manifest
from agents.harl_runner import default_algo_args


def _scenario_kwargs_from_manifest(manifest: Optional[dict]) -> dict:
    if not manifest:
        return {}
    env_args = manifest.get("env_args", {})
    scenario_kwargs = env_args.get("scenario_kwargs", {})
    if not isinstance(scenario_kwargs, dict):
        return {}
    scenario_kwargs = copy.deepcopy(scenario_kwargs)
    # Older diagnostic manifests may contain this removed experimental option.
    scenario_kwargs.pop("ugv_action_mode", None)
    return scenario_kwargs


def _action_transform_from_manifest(manifest: Optional[dict]) -> str:
    if not manifest:
        return "clip"
    env_args = manifest.get("env_args", {})
    return str(env_args.get("action_transform", "clip"))


def _actor_args(algo_args: dict) -> dict:
    """Return the flat HARL actor config expected by HAPPO."""
    return {
        **algo_args["model"],
        **algo_args["algo"],
        **algo_args["train"],
    }


def _legacy_algo_args(checkpoint_dir: str | Path) -> dict:
    """Best-effort config for pre-manifest checkpoints."""
    algo_args = default_algo_args()
    cfg_path = Path(checkpoint_dir).parent / "config.json"
    if not cfg_path.exists():
        return algo_args

    try:
        with cfg_path.open(encoding="utf-8") as config_file:
            saved = json.load(config_file)
        saved_model = saved.get("algo_args", {}).get("model", {})
    except (OSError, TypeError, ValueError):
        return algo_args

    return {
        **algo_args,
        "model": {**algo_args["model"], **saved_model},
    }


def _algo_args_from_manifest_or_legacy(
    checkpoint_dir: str | Path,
    manifest: Optional[dict],
    override: Optional[dict],
) -> dict:
    if override is not None:
        return override
    if manifest is not None:
        return manifest["algo_args"]
    return _legacy_algo_args(checkpoint_dir)


def _build_policy_spaces(scenario_kwargs: Optional[dict]) -> tuple[list[str], list[Box], list[Box]]:
    """Build agent names, obs spaces, and action spaces for a scenario config."""
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
    agent_names: list[str] = [a.name for a in tmp.agents]
    obs_spaces: list[Box] = list(tmp.observation_space.spaces)
    action_spaces: list[Box] = [
        Box(-1.0, 1.0, (tmp.get_agent_action_size(a),), dtype=np.float32)
        for a in tmp.agents
    ]
    return agent_names, obs_spaces, action_spaces


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class HappoPolicy:
    """Trained HAPPO actor wrapped as a VMAS-style ``policy(env) -> actions``."""

    def __init__(
        self,
        checkpoint_dir: Path,
        algo_args: dict,
        deterministic: bool = True,
        scenario_kwargs: Optional[dict] = None,
        action_transform: str = "clip",
    ):
        from harl.algorithms.actors.happo import HAPPO

        self.checkpoint_dir = Path(checkpoint_dir)
        self.algo_args      = algo_args
        self.deterministic  = deterministic
        self._device        = torch.device("cpu")
        self.action_transform = action_transform

        actor_args = _actor_args(algo_args)
        self.agent_names, obs_spaces, action_spaces = _build_policy_spaces(scenario_kwargs)

        # Build one HAPPO actor per agent, restore the state dict.
        self.actors = []
        for i, (obs_s, act_s) in enumerate(zip(obs_spaces, action_spaces)):
            actor = HAPPO(actor_args, obs_s, act_s, self._device)
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
        action_transform = _action_transform_from_manifest(manifest)
        algo_args = _algo_args_from_manifest_or_legacy(checkpoint_dir, manifest, algo_args)
        return cls(
            checkpoint_dir,
            algo_args,
            deterministic,
            scenario_kwargs=scenario_kwargs,
            action_transform=action_transform,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear recurrent hidden state (call at episode boundaries)."""
        self._rnn_states = None

    def _ensure_rnn_states(self, batch_dim: int) -> None:
        if getattr(self, "_rnn_states", None) is None or self._rnn_states[0].shape[0] != batch_dim:
            self._rnn_states = [
                np.zeros((batch_dim, self._recurrent_n, self._rnn_hidden_size), dtype=np.float32)
                for _ in self.actors
            ]

    def act_from_observations(self, observations: Sequence[np.ndarray | torch.Tensor]) -> List[torch.Tensor]:
        """Return per-agent actions for pre-collected observations.

        Separating observation collection from actor inference makes the
        checkpoint wrapper easier to test and avoids hiding environment side
        effects inside the action loop.
        """
        if len(observations) != len(self.actors):
            raise ValueError(f"expected {len(self.actors)} observations, got {len(observations)}")

        obs_arrays = [_as_numpy(obs).astype(np.float32, copy=False) for obs in observations]
        batch_dim = obs_arrays[0].shape[0]
        self._ensure_rnn_states(batch_dim)

        out: List[torch.Tensor] = []
        masks = np.ones((batch_dim, 1), dtype=np.float32)
        for i, obs in enumerate(obs_arrays):
            if obs.shape[0] != batch_dim:
                raise ValueError("all observations must share the same batch dimension")
            with torch.no_grad():
                actions, rnn_out = self.actors[i].act(
                    obs,
                    self._rnn_states[i],
                    masks,
                    available_actions=None,
                    deterministic=self.deterministic,
                )
            self._rnn_states[i] = _as_numpy(rnn_out)
            a_np = transform_continuous_action(_as_numpy(actions), self.action_transform)
            out.append(torch.from_numpy(a_np))
        return out

    def __call__(self, env) -> List[torch.Tensor]:
        """Return per-agent action tensors of shape (num_envs, action_dim)."""
        observations = [env.scenario.observation(agent) for agent in env.agents]
        return self.act_from_observations(observations)


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
