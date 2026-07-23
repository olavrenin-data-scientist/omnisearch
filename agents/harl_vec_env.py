"""
Batched VMAS vec-env for HARL — the proper-parallelism version.

The simpler ``WildfireHARLEnv`` (single-env adapter) loses VMAS's main
feature: one tensor-batched physics step processing many parallel envs.
This module ships ``BatchedVMASVecEnv`` which subclasses HARL's
``ShareVecEnv`` directly and wraps a single VMAS instance with
``num_envs=N``, so HARL gets vectorised rollouts without subprocess
fan-out.

Throughput comparison on CPU (rough):

    Approach                        Effective parallel envs   Notes
    ------------------------------  ------------------------  ------------
    SingleEnv x ShareSubprocVecEnv  N (= worker count)        subprocess
                                                              overhead per
                                                              step
    BatchedVMASVecEnv (this)        N (single VMAS batch)     one tensor
                                                              op per step,
                                                              GPU-friendly

On a GPU this is ~10-50x faster than the subprocess approach for the
same N. On CPU it's comparable in throughput but uses one process.

Auto-reset semantics match HARL's ShareDummyVecEnv contract: when an
env is done we save terminal obs/state into info[i][0] then call
``scenario.reset_world_at(env_index=i)`` to refresh just that env's
state and re-collect its observation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
from gymnasium.spaces import Box

import vmas

from agents.action_transform import transform_continuous_action
from agents.harl_metrics import ENV_METRICS
from envs.wildfire_search import WildfireSearchScenario
from harl.envs.env_wrappers import ShareVecEnv


_TRAINING_INFO_KEYS = frozenset(ENV_METRICS)


class BatchedVMASVecEnv(ShareVecEnv):
    """Single batched VMAS env exposed as a HARL ShareVecEnv."""

    def __init__(
        self,
        num_envs:        int,
        seed:            int = 0,
        max_cycles:      int = 200,
        scenario_kwargs: Optional[Dict[str, Any]] = None,
        action_transform: str = "clip",
        device:          str = "cpu",
    ):
        self._num_envs       = int(num_envs)
        self._seed_val       = int(seed)
        self.max_cycles      = int(max_cycles)
        self._scenario_kwargs = copy.deepcopy(scenario_kwargs or {})
        self.action_transform = action_transform
        self._device         = device

        self._build_env()

        # Build per-agent space lists
        obs_space_list:    list[Box] = list(self._env.observation_space.spaces)
        action_space_list: list[Box] = []
        for a in self._env.agents:
            sz = self._env.get_agent_action_size(a)
            action_space_list.append(
                Box(low=-1.0, high=1.0, shape=(sz,), dtype=np.float32),
            )
        shared_dim  = sum(s.shape[0] for s in obs_space_list)
        share_space = Box(low=-np.inf, high=np.inf, shape=(shared_dim,), dtype=np.float32)
        share_space_list = [share_space for _ in range(self.n_agents)]

        super().__init__(self._num_envs, obs_space_list, share_space_list, action_space_list)

        self._pending_actions: Optional[np.ndarray] = None
        self._step_counts = np.zeros(self._num_envs, dtype=np.int64)
        self._policy_class_obs_masks: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_env(self):
        self._env = WildfireSearchScenario.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=self._num_envs,
            device=self._device,
            continuous_actions=True,
            seed=self._seed_val,
            **self._scenario_kwargs,
        )
        self._env.reset()
        self.agents:   list[str] = [a.name for a in self._env.agents]
        self.n_agents: int       = len(self.agents)
        self._policy_class_obs_masks = {}

    def _stack_per_agent(self, tensor_list) -> np.ndarray:
        """Stack list of (N, ...) per-agent tensors into (N, A, ...) ndarray."""
        np_list = [t.cpu().numpy() for t in tensor_list]
        return np.stack(np_list, axis=1)

    def _collect_obs(self) -> np.ndarray:
        """Re-collect per-agent obs from the scenario (used after partial resets).
        Returns shape (num_envs, n_agents, obs_dim)."""
        per_agent = [self._env.scenario.observation(a) for a in self._env.agents]
        return self._stack_per_agent(per_agent)

    def _share_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """Build (N, A, A*obs_dim) share_obs: concat-of-locals tiled per agent."""
        N, A, D = obs.shape
        flat = obs.reshape(N, A * D)
        # Repeat across the agent axis so every actor sees the same shared state.
        return np.broadcast_to(flat[:, None, :], (N, A, A * D)).copy()

    def _observation_mask_for_policy_class(self, policy_class: str, obs_dim: int) -> np.ndarray:
        policy_class = str(policy_class).replace("-", "_").lower()
        cached = self._policy_class_obs_masks.get(policy_class)
        if cached is not None:
            return cached
        if not hasattr(self._env.scenario, "observation_mask_for_policy_class"):
            mask = np.ones(obs_dim, dtype=np.float32)
        else:
            mask = np.asarray(
                self._env.scenario.observation_mask_for_policy_class(policy_class),
                dtype=np.float32,
            )
        if mask.shape != (obs_dim,):
            raise ValueError(
                f"{policy_class} observation mask shape {mask.shape} does not match "
                f"agent observation width {obs_dim}"
            )
        self._policy_class_obs_masks[policy_class] = mask
        return mask

    def share_obs_for_policy_class(
        self,
        obs: np.ndarray,
        policy_class: str,
        member_ids: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray:
        """Build class-masked centralized observations with each member first."""
        N, A, D = obs.shape
        member_ids = [int(agent_id) for agent_id in member_ids]
        mask = self._observation_mask_for_policy_class(policy_class, D)
        masked_obs = obs * mask.reshape(1, 1, D)
        out = np.zeros((N, len(member_ids), A * D), dtype=np.float32)
        all_ids = list(range(A))
        for slot, agent_id in enumerate(member_ids):
            if agent_id < 0 or agent_id >= A:
                raise IndexError(f"agent id {agent_id} outside centralized obs with {A} agents")
            order = [agent_id] + [other_id for other_id in all_ids if other_id != agent_id]
            out[:, slot] = masked_obs[:, order, :].reshape(N, A * D)
        return out

    # ------------------------------------------------------------------
    # ShareVecEnv interface
    # ------------------------------------------------------------------
    def reset(self):
        # Bump seed each episode-batch to randomise spawns — same approach
        # as HARL's PettingZooMPE wrapper.
        self._seed_val += 1
        self._build_env()
        self._step_counts.fill(0)

        obs_tensors = self._env.reset()
        obs = self._stack_per_agent(obs_tensors)
        share_obs = self._share_from_obs(obs)
        # HARL indexes available_actions per-env even in the continuous case
        # (it does `available_actions[0] is not None` to detect discrete).
        # So return an ndarray of Nones with shape (num_envs,), not None.
        avail = np.array([None] * self._num_envs, dtype=object)
        return obs, share_obs, avail

    def step_async(self, actions: np.ndarray):
        # actions: shape (N, A, action_dim)
        self._pending_actions = actions

    def step_wait(self):
        actions = self._pending_actions
        self._pending_actions = None

        # Convert (N, A, action_dim) → list of A tensors each (N, action_dim)
        action_list = []
        for i in range(self.n_agents):
            a_i = transform_continuous_action(actions[:, i, :], self.action_transform)
            action_list.append(torch.from_numpy(a_i))

        obs_tensors, rew_tensors, dones_t, raw_infos = self._env.step(action_list)

        obs       = self._stack_per_agent(obs_tensors)               # (N, A, obs_dim)
        share_obs = self._share_from_obs(obs)                        # (N, A, A*obs_dim)
        rews      = self._stack_per_agent(rew_tensors)[..., None]    # (N, A, 1)

        # Done = natural OR truncated-at-max-cycles
        self._step_counts += 1
        natural = dones_t.cpu().numpy().astype(bool)                 # (N,)
        truncated = self._step_counts >= self.max_cycles
        done_per_env = natural | truncated                           # (N,)
        dones = np.broadcast_to(done_per_env[:, None], (self._num_envs, self.n_agents)).copy()

        # Per-env, per-agent infos. HARL needs bad_transition for every
        # agent, while our logger reads episode metrics from infos[i][0].
        infos = np.empty((self._num_envs, self.n_agents), dtype=object)
        for i in range(self._num_envs):
            bad = bool(truncated[i] and not natural[i])
            metric_info = self._info_for_env_agent(raw_infos, i, 0, keys=_TRAINING_INFO_KEYS)
            metric_info["bad_transition"] = bad
            infos[i, 0] = metric_info
            for j in range(1, self.n_agents):
                infos[i, j] = {"bad_transition": bad}

        # Auto-reset any done envs and re-collect their observations
        done_idx = np.where(done_per_env)[0]
        if len(done_idx):
            for i in done_idx:
                infos[i, 0]["original_obs"]           = obs[i].copy()
                infos[i, 0]["original_state"]         = share_obs[i].copy()
                infos[i, 0]["original_avail_actions"] = None
                self._env.reset_at(index=int(i), return_observations=False)
                self._step_counts[i] = 0
            # Re-collect obs (batched call; we'll overwrite only the rows we reset)
            fresh = self._collect_obs()
            obs[done_idx] = fresh[done_idx]
            share_obs[done_idx] = self._share_from_obs(obs)[done_idx]

        avail = np.array([None] * self._num_envs, dtype=object)
        return obs, share_obs, rews, dones, infos, avail

    def close_extras(self):
        pass

    def _info_for_env_agent(
        self,
        raw_infos: Any,
        env_index: int,
        agent_id: int,
        *,
        keys: frozenset[str] | None = None,
    ) -> Dict[str, Any]:
        if not raw_infos:
            return {}
        raw = raw_infos[agent_id] if isinstance(raw_infos, list) else raw_infos
        info: Dict[str, Any] = {}
        for key, value in raw.items():
            if keys is not None and key not in keys:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.ndim == 0:
                    info[key] = float(value.item())
                else:
                    item = value[env_index]
                    if item.numel() == 1:
                        info[key] = float(item.reshape(-1)[0])
                    else:
                        info[key] = item.numpy()
            else:
                info[key] = value
        return info


# ----------------------------------------------------------------------
# Factory: registered in train_happo_smoke.py's monkey-patch path so
# HARL gets BatchedVMASVecEnv when it asks for the "wildfire" env, in
# place of ShareDummyVecEnv-of-WildfireHARLEnv-singletons.
# ----------------------------------------------------------------------
def make_batched_wildfire_vec_env(
    num_envs:        int,
    seed:            int,
    env_args:        Dict[str, Any],
) -> BatchedVMASVecEnv:
    return BatchedVMASVecEnv(
        num_envs        = num_envs,
        seed            = seed,
        max_cycles      = env_args.get("max_cycles", 200),
        scenario_kwargs = env_args.get("scenario_kwargs", {}),
        action_transform = env_args.get("action_transform", "clip"),
        device          = env_args.get("device", "cpu"),
    )
