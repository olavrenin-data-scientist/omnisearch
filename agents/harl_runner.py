"""
Reusable HARL/HAPPO training entry point.

Used by both ``scripts/train_happo_smoke.py`` (single training run) and
``scripts/comms_dropout_sweep.py`` (sweep across dropout × seed). Wraps:

  1. The HARL env-registry monkey-patches (idempotent, safe to call many times)
  2. A custom logger that captures the *last* per-episode mean reward into
     an attribute so callers can read it after ``runner.run()`` — HARL's
     default logger clears it.
  3. ``train_happo()`` — one call, returns a metrics dict.

If you want HATRPO / HASAC instead of HAPPO, swap the runner class in
``train_happo()``: HARL ships ``OnPolicyHARunner`` for HAPPO/HATRPO/HAA2C
and ``OffPolicyHARunner`` for HASAC/HATD3/HADDPG.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agents.harl_metrics import (
    accumulate_env_metrics,
    init_env_metric_storage,
    log_done_env_metrics,
)

# Module-level flag so monkey-patches run only once
_REGISTERED = False


# ----------------------------------------------------------------------
# Logger override — keep the last computed per-episode mean reward
# ----------------------------------------------------------------------
def _build_wildfire_logger_class():
    from harl.common.base_logger import BaseLogger

    class WildfireLogger(BaseLogger):
        """BaseLogger that stashes the last per-episode mean reward."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.last_aver_episode_rewards: Optional[float] = None
            self.last_average_step_rewards: Optional[float] = None

        def get_task_name(self):
            return "wildfire_search"

        def init(self, episodes):
            super().init(episodes)
            init_env_metric_storage(self)

        def per_step(self, data):
            accumulate_env_metrics(self, data[4], data[3])
            super().per_step(data)

        def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
            # Snapshot before super() clears done_episodes_rewards
            if len(self.done_episodes_rewards) > 0:
                self.last_aver_episode_rewards = float(np.mean(self.done_episodes_rewards))
            super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
            log_done_env_metrics(self)
            self.last_average_step_rewards = float(critic_train_info["average_step_rewards"])

    return WildfireLogger


# ----------------------------------------------------------------------
# HARL env registry monkey-patch (idempotent)
# ----------------------------------------------------------------------
def register_wildfire_with_harl():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    import harl.envs as harl_envs_pkg
    import harl.utils.envs_tools as envs_tools
    import harl.utils.configs_tools as configs_tools

    from agents.harl_env import WildfireHARLEnv
    from agents.harl_vec_env import make_batched_wildfire_vec_env

    _orig_train  = envs_tools.make_train_env
    _orig_eval   = envs_tools.make_eval_env
    _orig_render = envs_tools.make_render_env
    _orig_nagent = envs_tools.get_num_agents
    _orig_task   = configs_tools.get_task_name

    def make_train_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            return make_batched_wildfire_vec_env(n_threads, seed, env_args)
        return _orig_train(env_name, seed, n_threads, env_args)

    def make_eval_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            return make_batched_wildfire_vec_env(n_threads, seed + 10_000, env_args)
        return _orig_eval(env_name, seed, n_threads, env_args)

    def make_render_env(env_name, seed, env_args):
        if env_name == "wildfire":
            env = WildfireHARLEnv({**env_args, "seed": seed})
            return env, env.n_agents, env.agents
        return _orig_render(env_name, seed, env_args)

    def get_num_agents(env_name, env_args, envs):
        if env_name == "wildfire":
            return envs.n_agents
        return _orig_nagent(env_name, env_args, envs)

    def get_task_name(env_name, env_args):
        if env_name == "wildfire":
            return "wildfire_search"
        return _orig_task(env_name, env_args)

    envs_tools.make_train_env       = make_train_env
    envs_tools.make_eval_env        = make_eval_env
    envs_tools.make_render_env      = make_render_env
    envs_tools.get_num_agents       = get_num_agents
    configs_tools.get_task_name     = get_task_name

    harl_envs_pkg.LOGGER_REGISTRY["wildfire"] = _build_wildfire_logger_class()


# ----------------------------------------------------------------------
# Default HARL config builders
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


def default_algo_args() -> Dict[str, Any]:
    """Smoke-budget HAPPO algo_args. Override fields before passing to train_happo."""
    return {
        "seed":   {"seed_specify": True, "seed": 1},
        "device": {"cuda": False, "cuda_deterministic": True, "torch_threads": 4},
        "train": {
            "n_rollout_threads":     8,
            "num_env_steps":         8_000,
            "episode_length":        100,
            "log_interval":          1,
            "eval_interval":         1,
            "use_valuenorm":         True,
            "use_linear_lr_decay":   False,
            "use_proper_time_limits": True,
            "model_dir":             None,
        },
        "eval":   {"use_eval": False, "n_eval_rollout_threads": 1, "eval_episodes": 2},
        "render": {"use_render": False, "render_episodes": 1},
        "model": {
            "hidden_sizes": [128, 128], "activation_func": "relu",
            "use_feature_normalization": True, "initialization_method": "orthogonal_",
            "gain": 0.01, "use_naive_recurrent_policy": False,
            "use_recurrent_policy": False, "recurrent_n": 1, "data_chunk_length": 10,
            "lr": 5e-4, "critic_lr": 5e-4, "opti_eps": 1e-5, "weight_decay": 0,
            "std_x_coef": 1, "std_y_coef": 0.5,
        },
        "algo": {
            "ppo_epoch": 2, "critic_epoch": 2,
            "use_clipped_value_loss": True, "clip_param": 0.2,
            "actor_num_mini_batch": 1, "critic_num_mini_batch": 1,
            "entropy_coef": 0.01, "value_loss_coef": 1,
            "use_max_grad_norm": True, "max_grad_norm": 10.0,
            "use_gae": True, "gamma": 0.99, "gae_lambda": 0.95,
            "use_huber_loss": True, "use_policy_active_masks": True, "huber_delta": 10.0,
            "action_aggregation": "prod",
            "share_param": False, "fixed_order": False,
        },
        "logger": {"log_dir": str(ROOT / "results" / "harl_runs")},
    }


def default_env_args() -> Dict[str, Any]:
    return {
        "max_cycles":      100,
        "scenario_kwargs": {"max_steps": 100, "n_drones": 3, "n_ground": 2},
    }


# ----------------------------------------------------------------------
# Public API — train HAPPO once, return metrics
# ----------------------------------------------------------------------
def train_happo(
    seed:           int   = 1,
    num_env_steps:  int   = 8_000,
    comms_dropout:  float = 0.0,
    n_rollout_threads: int = 8,
    exp_name:       str   = "happo",
    entropy_coef:   float = 0.01,
) -> Dict[str, Any]:
    """
    Train HAPPO at the given seed + comms_dropout, return final metrics.

    Returns
    -------
    dict with keys:
        mean_episode_reward    : float — last logged per-episode mean reward
        mean_step_reward       : float — last logged per-step mean reward
        num_env_steps          : int
        seed                   : int
        comms_dropout          : float
        wall_sec               : float
    """
    register_wildfire_with_harl()
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner

    args = {
        "algo": "happo", "env": "wildfire",
        "exp_name": exp_name, "load_config": "",
    }
    algo_args = default_algo_args()
    algo_args["seed"]["seed"] = seed
    algo_args["train"]["num_env_steps"]     = num_env_steps
    algo_args["train"]["n_rollout_threads"] = n_rollout_threads
    algo_args["algo"]["entropy_coef"]       = entropy_coef

    env_args = default_env_args()
    env_args["scenario_kwargs"] = {**env_args["scenario_kwargs"],
                                   "comms_dropout": comms_dropout}

    t0 = time.time()
    runner = OnPolicyHARunner(args, algo_args, env_args)
    runner.run()
    wall = time.time() - t0

    mean_ep = runner.logger.last_aver_episode_rewards
    mean_step = runner.logger.last_average_step_rewards
    runner.close()

    return {
        "mean_episode_reward": float(mean_ep) if mean_ep is not None else float("nan"),
        "mean_step_reward":    float(mean_step) if mean_step is not None else float("nan"),
        "num_env_steps":       int(num_env_steps),
        "seed":                int(seed),
        "comms_dropout":       float(comms_dropout),
        "wall_sec":            round(wall, 2),
    }
