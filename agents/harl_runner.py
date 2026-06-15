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

        def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
            # Snapshot before super() clears done_episodes_rewards
            if len(self.done_episodes_rewards) > 0:
                self.last_aver_episode_rewards = float(np.mean(self.done_episodes_rewards))
            super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
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


def default_algo_args(profile: str = "smoke") -> Dict[str, Any]:
    """
    Return HAPPO algo_args for a named profile.

    Profiles
    --------
    smoke:
        Tiny budget for "does this run?" checks.
    research:
        Larger budget and stronger optimization defaults for better convergence.
    """
    if profile not in {"smoke", "research"}:
        raise ValueError(f"Unsupported HAPPO profile: {profile!r}")

    is_research = profile == "research"
    return {
        "seed":   {"seed_specify": True, "seed": 1},
        "device": {"cuda": False, "cuda_deterministic": True, "torch_threads": 4},
        "train": {
            "n_rollout_threads":     16 if is_research else 8,
            "num_env_steps":         400_000 if is_research else 8_000,
            "episode_length":        500 if is_research else 100,
            "log_interval":          1,
            "eval_interval":         10 if is_research else 1,
            "use_valuenorm":         True,
            "use_linear_lr_decay":   is_research,
            "use_proper_time_limits": True,
            "model_dir":             None,
        },
        "eval": {
            "use_eval": is_research,
            "n_eval_rollout_threads": 2 if is_research else 1,
            "eval_episodes": 8 if is_research else 2,
        },
        "render": {"use_render": False, "render_episodes": 1},
        "model": {
            "hidden_sizes": [128, 128], "activation_func": "relu",
            "use_feature_normalization": True, "initialization_method": "orthogonal_",
            "gain": 0.01, "use_naive_recurrent_policy": False,
            "use_recurrent_policy": is_research,
            "recurrent_n": 1,
            "data_chunk_length": 20 if is_research else 10,
            "lr": 5e-4, "critic_lr": 5e-4, "opti_eps": 1e-5, "weight_decay": 0,
            "std_x_coef": 1, "std_y_coef": 0.5,
        },
        "algo": {
            "ppo_epoch": 5 if is_research else 2,
            "critic_epoch": 5 if is_research else 2,
            "use_clipped_value_loss": True, "clip_param": 0.2,
            "actor_num_mini_batch": 4 if is_research else 1,
            "critic_num_mini_batch": 4 if is_research else 1,
            "entropy_coef": 0.02 if is_research else 0.01,
            "value_loss_coef": 1,
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
    max_steps:      int   = 100,
    n_rollout_threads: int = 8,
    exp_name:       str   = "happo",
    entropy_coef:   float = 0.01,
    profile:        str   = "smoke",
    recurrent:      Optional[bool] = None,
    reward_search:  bool  = False,
    drone_min_footprint: float = 0.0,
    ground_confirm_min: float = 0.0,
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
        tensorboard_log_dir    : str  — TensorBoard events directory
        tensorboard_cmd        : str  — ready-to-run TensorBoard command
    """
    register_wildfire_with_harl()
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner
    from agents.happo_checkpoint import save_training_manifest

    args = {
        "algo": "happo", "env": "wildfire",
        "exp_name": exp_name, "load_config": "",
    }
    # HARL computes episode count as:
    #   episodes = num_env_steps // episode_length // n_rollout_threads
    # If this is 0, training never runs and no checkpoint is produced.
    min_env_steps = int(max_steps) * int(n_rollout_threads)
    if num_env_steps < min_env_steps:
        num_env_steps = min_env_steps

    algo_args = default_algo_args(profile=profile)
    algo_args["seed"]["seed"] = seed
    algo_args["train"]["num_env_steps"]     = num_env_steps
    algo_args["train"]["n_rollout_threads"] = n_rollout_threads
    algo_args["algo"]["entropy_coef"]       = entropy_coef
    algo_args["train"]["episode_length"]    = max_steps
    if recurrent is not None:
        algo_args["model"]["use_recurrent_policy"] = bool(recurrent)

    env_args = default_env_args()
    env_args["max_cycles"] = max_steps
    scenario_kwargs = {
        **env_args["scenario_kwargs"],
        "max_steps": max_steps,
        "comms_dropout": comms_dropout,
        "drone_min_footprint": float(max(drone_min_footprint, 0.0)),
        "ground_confirm_min": float(max(ground_confirm_min, 0.0)),
    }
    if reward_search:
        # Search-dominant shaping: promotes exploration and confirmation
        # while reducing degeneracy toward low-movement policies.
        scenario_kwargs.update({
            "r_found_survivor": 10.0,
            "r_drone_scout": 2.0,
            "r_ground_confirm": 4.0,
            "r_drone_shaping": 0.30,
            "r_ground_shaping": 0.50,
            "r_ground_approach": 0.10,
            "ground_approach_radius": 0.4,
            "r_fire_penalty": -0.20,
            "r_ground_travel_cost": -0.01,
            "r_drone_climb_cost": -0.005,
            "r_time_penalty": -0.0005,
            "r_coverage": 5.0,
        })
    env_args["scenario_kwargs"] = scenario_kwargs

    t0 = time.time()
    runner = OnPolicyHARunner(args, algo_args, env_args)
    tb_log_dir = str(Path(getattr(runner, "log_dir", algo_args["logger"]["log_dir"])))
    tb_cmd = f"tensorboard --logdir \"{tb_log_dir}\" --port 6006"
    manifest_path = save_training_manifest(
        runner,
        harl_args=args,
        algo_args=algo_args,
        env_args=env_args,
    )
    runner.run()
    wall = time.time() - t0

    mean_ep = runner.logger.last_aver_episode_rewards
    mean_step = runner.logger.last_average_step_rewards
    runner.close()

    return {
        "mean_episode_reward": float(mean_ep) if mean_ep is not None else float("nan"),
        "mean_step_reward":    float(mean_step) if mean_step is not None else float("nan"),
        "num_env_steps":       int(num_env_steps),
        "max_steps":           int(max_steps),
        "seed":                int(seed),
        "comms_dropout":       float(comms_dropout),
        "wall_sec":            round(wall, 2),
        "manifest_path":       str(manifest_path),
        "checkpoint_dir":      str(manifest_path.parent / "models"),
        "tensorboard_log_dir": tb_log_dir,
        "tensorboard_cmd":     tb_cmd,
    }
