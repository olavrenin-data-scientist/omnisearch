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
import torch

from agents.harl_metrics import (
    accumulate_env_metrics,
    init_env_metric_storage,
    log_done_env_metrics,
)
from agents.action_transform import transform_continuous_action

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
    from agents.harl_terrain_cnn import install_harl_terrain_cnn_patch

    install_harl_terrain_cnn_patch()
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


def _target_bucket_names() -> tuple[str, ...]:
    return ("E", "NE", "N", "NW", "W", "SW", "S", "SE")


def _target_angle_bucket(unit_xy: np.ndarray) -> np.ndarray:
    angles = (np.degrees(np.arctan2(unit_xy[:, 1], unit_xy[:, 0])) + 360.0) % 360.0
    buckets = np.zeros(unit_xy.shape[0], dtype=np.int64)
    buckets[(angles >= 22.5) & (angles < 67.5)] = 1
    buckets[(angles >= 67.5) & (angles < 112.5)] = 2
    buckets[(angles >= 112.5) & (angles < 157.5)] = 3
    buckets[(angles >= 157.5) & (angles < 202.5)] = 4
    buckets[(angles >= 202.5) & (angles < 247.5)] = 5
    buckets[(angles >= 247.5) & (angles < 292.5)] = 6
    buckets[(angles >= 292.5) & (angles < 337.5)] = 7
    return buckets


def _advantage_alignment_diagnostics(
    actor_buffer,
    advantages: np.ndarray,
    *,
    n_survivors: int,
    action_transform: str,
    survivor_message_distance_scale_m: float,
) -> dict[str, float]:
    """Summarize whether target-aligned sampled actions get positive advantage.

    The actor observation appends one 7-value survivor message per survivor:
    [known, dx, dy, ux, uy, distance_norm, confirmed]. This diagnostic uses the
    nearest known, unconfirmed survivor message and compares the sampled action
    with that target unit vector.
    """
    if n_survivors <= 0:
        return {}
    obs = actor_buffer.obs[:-1]
    next_obs = actor_buffer.obs[1:]
    actions = actor_buffer.actions
    if obs.shape[-1] < 7 * n_survivors or actions.shape[-1] < 2:
        return {}

    flat_obs = obs.reshape(-1, obs.shape[-1])
    flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    flat_adv = advantages.reshape(-1)
    survivor_messages = flat_obs[:, -7 * n_survivors:].reshape(-1, n_survivors, 7)
    next_survivor_messages = flat_next_obs[:, -7 * n_survivors:].reshape(-1, n_survivors, 7)

    known = survivor_messages[:, :, 0] > 0.5
    confirmed = survivor_messages[:, :, 6] > 0.5
    valid = known & ~confirmed
    if not np.any(valid):
        return {"diag/target_known_frac": 0.0}

    distance = survivor_messages[:, :, 5]
    masked_distance = np.where(valid, distance, np.inf)
    target_idx = np.argmin(masked_distance, axis=1)
    row_idx = np.arange(flat_obs.shape[0])
    has_target = np.isfinite(masked_distance[row_idx, target_idx])
    if not np.any(has_target):
        return {"diag/target_known_frac": 0.0}

    target_unit = survivor_messages[row_idx[has_target], target_idx[has_target], 3:5]
    next_target_message = next_survivor_messages[row_idx[has_target], target_idx[has_target]]
    action = transform_continuous_action(flat_actions[has_target, :2], action_transform)
    adv = flat_adv[has_target]
    displacement = flat_next_obs[has_target, :2] - flat_obs[has_target, :2]
    progress_m = (
        survivor_messages[row_idx[has_target], target_idx[has_target], 5]
        - next_target_message[:, 5]
    ) * max(float(survivor_message_distance_scale_m), 1e-6)

    target_norm = np.linalg.norm(target_unit, axis=1)
    action_norm = np.linalg.norm(action, axis=1)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    usable = (
        (target_norm > 1e-6)
        & (action_norm > 1e-6)
        & np.isfinite(adv)
        & np.isfinite(progress_m)
    )
    if not np.any(usable):
        return {"diag/target_known_frac": float(np.mean(has_target))}

    target_unit = target_unit[usable] / target_norm[usable, None]
    action = action[usable]
    displacement = displacement[usable]
    displacement_norm = displacement_norm[usable]
    adv = adv[usable]
    progress_m = progress_m[usable]
    alignment = np.sum(action * target_unit, axis=1) / np.linalg.norm(action, axis=1).clip(min=1e-6)
    displacement_alignment = np.full_like(alignment, np.nan, dtype=np.float64)
    moved = displacement_norm > 1e-9
    displacement_alignment[moved] = (
        np.sum(displacement[moved] * target_unit[moved], axis=1)
        / displacement_norm[moved].clip(min=1e-6)
    )
    toward = alignment > 0.0
    away = alignment < 0.0
    displacement_toward = displacement_alignment > 0.0

    out: dict[str, float] = {
        "diag/target_known_frac": float(np.mean(has_target)),
        "diag/action_target_alignment_mean": float(np.mean(alignment)),
        "diag/action_toward_frac": float(np.mean(toward)),
        "diag/adv_mean_all": float(np.mean(adv)),
    }
    if np.any(toward):
        out["diag/adv_mean_toward"] = float(np.mean(adv[toward]))
    if np.any(away):
        out["diag/adv_mean_away"] = float(np.mean(adv[away]))
    if np.any(toward) and np.any(away):
        out["diag/adv_toward_minus_away"] = float(
            out["diag/adv_mean_toward"] - out["diag/adv_mean_away"],
        )
    if np.std(alignment) > 1e-8 and np.std(adv) > 1e-8:
        out["diag/adv_alignment_corr"] = float(np.corrcoef(alignment, adv)[0, 1])

    bucket = _target_angle_bucket(target_unit)
    for bucket_idx, bucket_name in enumerate(_target_bucket_names()):
        mask = bucket == bucket_idx
        if not np.any(mask):
            continue
        prefix = f"diag/bucket_{bucket_name.lower()}"
        out[f"{prefix}_frac"] = float(np.mean(mask))
        out[f"{prefix}_alignment"] = float(np.mean(alignment[mask]))
        out[f"{prefix}_adv"] = float(np.mean(adv[mask]))
        out[f"{prefix}_toward_frac"] = float(np.mean(toward[mask]))

        table_prefix = f"diag_train/{bucket_name.lower()}"
        bucket_alignment = alignment[mask]
        bucket_progress = progress_m[mask]
        bucket_adv = adv[mask]
        bucket_toward = toward[mask]
        bucket_away = away[mask]
        bucket_disp_toward = displacement_toward[mask]
        bucket_disp_valid = np.isfinite(displacement_alignment[mask])

        out[f"{table_prefix}/n_steps"] = float(np.sum(mask))
        out[f"{table_prefix}/pct_action_toward"] = float(np.mean(bucket_toward))
        out[f"{table_prefix}/pct_displacement_toward"] = (
            float(np.mean(bucket_disp_toward[bucket_disp_valid]))
            if np.any(bucket_disp_valid)
            else 0.0
        )
        out[f"{table_prefix}/pct_toward_positive_progress"] = (
            float(np.mean(bucket_progress[bucket_toward] > 0.0))
            if np.any(bucket_toward)
            else 0.0
        )
        out[f"{table_prefix}/pct_away_negative_progress"] = (
            float(np.mean(bucket_progress[bucket_away] < 0.0))
            if np.any(bucket_away)
            else 0.0
        )
        out[f"{table_prefix}/pct_toward_positive_advantage"] = (
            float(np.mean(bucket_adv[bucket_toward] > 0.0))
            if np.any(bucket_toward)
            else 0.0
        )
        out[f"{table_prefix}/pct_away_positive_advantage"] = (
            float(np.mean(bucket_adv[bucket_away] > 0.0))
            if np.any(bucket_away)
            else 0.0
        )
        out[f"{table_prefix}/alignment_p10"] = float(np.percentile(bucket_alignment, 10))
        out[f"{table_prefix}/alignment_p50"] = float(np.percentile(bucket_alignment, 50))
        out[f"{table_prefix}/alignment_p90"] = float(np.percentile(bucket_alignment, 90))
        out[f"{table_prefix}/progress_m_p10"] = float(np.percentile(bucket_progress, 10))
        out[f"{table_prefix}/progress_m_p50"] = float(np.percentile(bucket_progress, 50))
        out[f"{table_prefix}/progress_m_p90"] = float(np.percentile(bucket_progress, 90))
    return out


def _build_diagnostic_happo_runner_class():
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner
    from harl.utils.trans_tools import _t2n

    class DiagnosticHAPPORunner(OnPolicyHARunner):
        """OnPolicyHARunner plus per-update actor advantage diagnostics."""

        def train(self):
            actor_train_infos = []
            factor = np.ones(
                (
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                ),
                dtype=np.float32,
            )

            if self.value_normalizer is not None:
                advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                    self.critic_buffer.value_preds[:-1],
                )
            else:
                advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

            if self.state_type == "FP":
                active_masks_collector = [
                    self.actor_buffer[i].active_masks for i in range(self.num_agents)
                ]
                active_masks_array = np.stack(active_masks_collector, axis=2)
                advantages_copy = advantages.copy()
                advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(torch.randperm(self.num_agents).numpy())

            scenario_kwargs = self.env_args.get("scenario_kwargs", {})
            n_survivors = int(scenario_kwargs.get("n_survivors", 0))
            action_transform = str(self.env_args.get("action_transform", "clip"))
            survivor_message_distance_scale_m = float(
                scenario_kwargs.get("survivor_message_distance_scale_m", 100.0),
            )

            for agent_id in agent_order:
                self.actor_buffer[agent_id].update_factor(factor)
                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

                old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )

                if self.state_type == "EP":
                    agent_advantages = advantages.copy()
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        agent_advantages,
                        "EP",
                    )
                elif self.state_type == "FP":
                    agent_advantages = advantages[:, :, agent_id].copy()
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        agent_advantages,
                        "FP",
                    )
                else:
                    raise ValueError(f"unsupported state_type: {self.state_type}")

                actor_train_info.update(
                    _advantage_alignment_diagnostics(
                        self.actor_buffer[agent_id],
                        agent_advantages,
                        n_survivors=n_survivors,
                        action_transform=action_transform,
                        survivor_message_distance_scale_m=survivor_message_distance_scale_m,
                    ),
                )

                new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )

                factor = factor * _t2n(
                    getattr(torch, self.action_aggregation)(
                        torch.exp(new_actions_logprob - old_actions_logprob),
                        dim=-1,
                    ).reshape(
                        self.algo_args["train"]["episode_length"],
                        self.algo_args["train"]["n_rollout_threads"],
                        1,
                    )
                )
                actor_train_infos.append(actor_train_info)

            critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
            return actor_train_infos, critic_train_info

    return DiagnosticHAPPORunner


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
    OnPolicyHARunner = _build_diagnostic_happo_runner_class()
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
