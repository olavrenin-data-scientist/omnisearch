"""
HARL-compatible adapter around the VMAS WildfireSearchScenario.

HARL expects a single-env interface (not batched) with the following contract,
matching their PettingZooMPEEnv reference impl in
harl/envs/pettingzoo_mpe/pettingzoo_mpe_env.py:

    attributes
        n_agents                  : int
        agents                    : List[str]
        observation_space         : List[gymnasium.spaces.Space]  (one per agent)
        share_observation_space   : List[gymnasium.spaces.Space]  (centralized critic)
        action_space              : List[gymnasium.spaces.Space]

    methods
        reset()  -> (obs_list, share_obs_list, available_actions)
        step(actions: ndarray[n_agents, action_dim])
                -> (obs_list, share_obs_list, reward_list, done_list, info_list, available_actions)
        seed(seed: int)
        render()
        close()

Our scenario is heterogeneous (drones + ground robots) but action dim happens
to match (2 per agent — VMAS continuous holonomic control). If we later split
actions per agent type, HARL handles that via per-agent space lists already.

For the shared observation (centralized critic input) we concatenate per-agent
local observations. That gives HAPPO's critic full visibility while keeping
each actor decentralised (its own slice). A more sophisticated share-obs
(e.g. full scenario state including survivor positions, fire grid) is a
TODO — concat-of-locals is the standard MAPPO baseline.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch
from gymnasium.spaces import Box

import vmas

from envs.wildfire_search import WildfireSearchScenario


class WildfireHARLEnv:
    """Single-env HARL adapter for the WildfireSearchScenario."""

    def __init__(self, args: Dict[str, Any]):
        args = copy.deepcopy(args)
        self.scenario_kwargs: Dict[str, Any] = args.get("scenario_kwargs", {}) or {}
        self.seed_val: int = args.get("seed", 0)
        self.max_cycles: int = args.get("max_cycles", 200)

        self._build_env()

    # ------------------------------------------------------------------
    # HARL contract — attributes
    # ------------------------------------------------------------------
    def _build_env(self):
        self._env = vmas.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            seed=self.seed_val,
            **self.scenario_kwargs,
        )
        self._env.reset()

        self.agents:   List[str] = [a.name for a in self._env.agents]
        self.n_agents: int       = len(self.agents)

        # Per-agent action space: Box(-1, 1, (2,))
        self.action_space: List[Box] = []
        for agent in self._env.agents:
            a_size = self._env.get_agent_action_size(agent)
            self.action_space.append(
                Box(low=-1.0, high=1.0, shape=(a_size,), dtype=np.float32),
            )

        # Per-agent local observation space — pulled from VMAS directly
        self.observation_space: List[Box] = list(self._env.observation_space.spaces)

        # Centralised-critic input: concat of every agent's local obs.
        shared_dim   = sum(s.shape[0] for s in self.observation_space)
        shared_space = Box(low=-np.inf, high=np.inf, shape=(shared_dim,), dtype=np.float32)
        self.share_observation_space: List[Box] = [shared_space] * self.n_agents

        self._cur_step = 0

    # ------------------------------------------------------------------
    # HARL contract — step / reset
    # ------------------------------------------------------------------
    def reset(self):
        # VMAS's reset uses the seed it was constructed with — to actually
        # randomize episodes we rebuild on a bumped seed each time. That's
        # consistent with HARL's PettingZoo wrapper (it bumps _seed too).
        self.seed_val += 1
        self._build_env()
        obs = self._env.reset()
        obs_list   = [o.cpu().numpy()[0] for o in obs]
        share_obs  = self._make_share_obs(obs_list)
        return obs_list, share_obs, self.get_avail_actions()

    def step(self, actions: np.ndarray):
        """
        actions : ndarray shape (n_agents, action_dim)
        returns (local_obs, share_obs, rewards, dones, infos, avail_actions)
        """
        action_list = []
        for i, agent in enumerate(self._env.agents):
            a_size = self._env.get_agent_action_size(agent)
            a = np.asarray(actions[i][:a_size], dtype=np.float32)
            # VMAS asserts actions are within range.
            a = np.clip(a, -1.0, 1.0)
            action_list.append(torch.from_numpy(a).unsqueeze(0))

        obs, rewards, dones, raw_infos = self._env.step(action_list)

        obs_list    = [o.cpu().numpy()[0] for o in obs]
        share_obs   = self._make_share_obs(obs_list)
        # Per-agent reward (HARL accepts shape [[r], ...])
        reward_list = [[float(r.cpu().numpy()[0])] for r in rewards]

        done_bool = bool(dones[0].item())
        self._cur_step += 1
        # HARL's PettingZoo wrapper truncates at max_cycles; mirror that.
        if not done_bool and self._cur_step >= self.max_cycles:
            done_bool = True

        done_list = [done_bool] * self.n_agents
        info_list = []
        for agent_id in range(self.n_agents):
            info = self._info_for_agent(raw_infos, agent_id)
            info["bad_transition"] = (self._cur_step >= self.max_cycles and done_bool)
            info_list.append(info)
        return obs_list, share_obs, reward_list, done_list, info_list, self.get_avail_actions()

    # ------------------------------------------------------------------
    # HARL contract — helpers
    # ------------------------------------------------------------------
    def get_avail_actions(self):
        # Continuous action space — no availability mask.
        return None

    def get_avail_agent_actions(self, agent_id):
        return None

    def render(self):
        # VMAS has its own viewer; we don't drive it from HARL.
        pass

    def close(self):
        # VMAS doesn't expose a close() — letting the env get garbage-collected
        # is fine for our use case.
        pass

    def seed(self, seed: int):
        self.seed_val = int(seed)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _make_share_obs(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        shared = np.concatenate(obs_list, axis=0).astype(np.float32)
        return [shared for _ in range(self.n_agents)]

    def _info_for_agent(self, raw_infos: Any, agent_id: int) -> Dict[str, Any]:
        if not raw_infos:
            return {}
        raw = raw_infos[agent_id] if isinstance(raw_infos, list) else raw_infos
        info: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    info[key] = float(value.reshape(-1)[0])
                else:
                    info[key] = value.numpy()
            else:
                info[key] = value
        return info
