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
            "warmstart_uav_model_dir": None,
            "warmstart_ugv_model_dir": None,
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
            "share_param": False,
            "share_param_by_agent_class": False,
            "share_param_groups": [],
            "share_param_group_names": [],
            "fixed_order": False,
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

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._configure_class_shared_actors()
            self._restore_class_warmstart_actors()

        def _use_class_shared_policy(self) -> bool:
            return bool(self.algo_args.get("algo", {}).get("share_param_by_agent_class", False))

        def _configure_class_shared_actors(self):
            if getattr(self, "_class_shared_policy_configured", False):
                return

            if not self._use_class_shared_policy():
                self._policy_update_groups = [[agent_id] for agent_id in range(self.num_agents)]
                self._policy_group_names = [f"agent{agent_id}" for agent_id in range(self.num_agents)]
                self._policy_group_representatives = list(range(self.num_agents))
                self._agent_to_policy_group = {
                    agent_id: agent_id for agent_id in range(self.num_agents)
                }
                self._class_shared_policy_configured = True
                return

            if self.share_param:
                raise ValueError("share_param_by_agent_class cannot be combined with share_param")

            raw_groups = list(self.algo_args["algo"].get("share_param_groups", []))
            raw_names = list(self.algo_args["algo"].get("share_param_group_names", []))
            if len(raw_groups) != self.num_agents:
                raise ValueError(
                    "share_param_by_agent_class requires one share_param_groups entry "
                    f"per agent; got {len(raw_groups)} for {self.num_agents} agents"
                )

            ordered_group_ids: list[int] = []
            for group_id in raw_groups:
                group_id = int(group_id)
                if group_id not in ordered_group_ids:
                    ordered_group_ids.append(group_id)

            update_groups: list[list[int]] = []
            group_names: list[str] = []
            representatives: list[int] = []
            agent_to_group: dict[int, int] = {}
            for group_index, group_id in enumerate(ordered_group_ids):
                members = [
                    agent_id for agent_id, raw_group_id in enumerate(raw_groups)
                    if int(raw_group_id) == group_id
                ]
                if not members:
                    continue
                representative = members[0]
                for agent_id in members[1:]:
                    if self.envs.observation_space[agent_id] != self.envs.observation_space[representative]:
                        raise ValueError(
                            "Agents in a class-shared policy group must have identical "
                            f"observation spaces; group {group_id} has agents {members}"
                        )
                    if self.envs.action_space[agent_id] != self.envs.action_space[representative]:
                        raise ValueError(
                            "Agents in a class-shared policy group must have identical "
                            f"action spaces; group {group_id} has agents {members}"
                        )
                    self.actor[agent_id] = self.actor[representative]
                group_name = raw_names[group_id] if 0 <= group_id < len(raw_names) else f"group{group_id}"
                update_groups.append(members)
                group_names.append(str(group_name))
                representatives.append(representative)
                for agent_id in members:
                    agent_to_group[agent_id] = group_index

            self._policy_update_groups = update_groups
            self._policy_group_names = group_names
            self._policy_group_representatives = representatives
            self._agent_to_policy_group = agent_to_group
            self._class_shared_policy_configured = True

        def _unique_actor_agent_ids(self) -> list[int]:
            self._configure_class_shared_actors()
            if self._use_class_shared_policy():
                return list(self._policy_group_representatives)
            if self.share_param:
                return [0]
            return list(range(self.num_agents))

        def _class_warmstart_model_dirs(self) -> dict[str, str]:
            train_args = self.algo_args.get("train", {})
            out: dict[str, str] = {}
            uav_dir = train_args.get("warmstart_uav_model_dir")
            ugv_dir = train_args.get("warmstart_ugv_model_dir")
            if uav_dir:
                out["uav"] = str(uav_dir)
            if ugv_dir:
                out["ugv"] = str(ugv_dir)
            return out

        def _restore_class_warmstart_actors(self) -> None:
            """Warm-start class-shared actors from per-class checkpoint dirs.

            This intentionally loads only actor weights. The joint critic and
            value normalizer remain freshly initialized because separate UAV
            and UGV warmup critics do not describe the joint task.
            """
            import os

            warmstart_dirs = self._class_warmstart_model_dirs()
            if not warmstart_dirs:
                return
            if self.algo_args["train"].get("model_dir") is not None:
                raise ValueError(
                    "class warm-start dirs cannot be combined with model_dir; "
                    "use either --model-dir or --warmstart-*-model-dir"
                )
            if not self._use_class_shared_policy():
                raise ValueError(
                    "class warm-start dirs require share_param_by_agent_class=True"
                )

            configured_groups = set(self._policy_group_names)
            missing_groups = sorted(set(warmstart_dirs) - configured_groups)
            if missing_groups:
                raise ValueError(
                    "class warm-start requested for missing policy group(s): "
                    + ", ".join(missing_groups)
                )

            for group_index, group_name in enumerate(self._policy_group_names):
                model_dir = warmstart_dirs.get(group_name)
                if model_dir is None:
                    continue
                actor_path = os.path.join(model_dir, "actor_agent0.pt")
                if not os.path.isfile(actor_path):
                    raise FileNotFoundError(
                        f"class warm-start checkpoint for {group_name!r} is missing "
                        f"actor_agent0.pt: {model_dir}"
                    )
                representative = self._policy_group_representatives[group_index]
                try:
                    state = torch.load(
                        actor_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                    self.actor[representative].actor.load_state_dict(state)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Could not warm-start {group_name!r} actor from {actor_path}. "
                        "Check that the warmup checkpoint was trained with the same "
                        "class observation and action shape as the joint run."
                    ) from exc
                print(
                    f"[warmstart] Loaded {group_name} actor for joint agent "
                    f"{representative} from {actor_path}; critic starts fresh."
                )

        def _evaluate_agent_actions(self, agent_id: int):
            actor_buffer = self.actor_buffer[agent_id]
            available_actions = (
                None
                if actor_buffer.available_actions is None
                else actor_buffer.available_actions[:-1].reshape(
                    -1, *actor_buffer.available_actions.shape[2:]
                )
            )
            return self.actor[agent_id].evaluate_actions(
                actor_buffer.obs[:-1].reshape(-1, *actor_buffer.obs.shape[2:]),
                actor_buffer.rnn_states[0:1].reshape(
                    -1, *actor_buffer.rnn_states.shape[2:]
                ),
                actor_buffer.actions.reshape(-1, *actor_buffer.actions.shape[2:]),
                actor_buffer.masks[:-1].reshape(-1, *actor_buffer.masks.shape[2:]),
                available_actions,
                actor_buffer.active_masks[:-1].reshape(
                    -1, *actor_buffer.active_masks.shape[2:]
                ),
            )[0]

        def _agent_action_ratio(self, agent_id: int, old_actions_logprob):
            new_actions_logprob = self._evaluate_agent_actions(agent_id)
            return _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob),
                    dim=-1,
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )

        def _group_actor_buffer(self, members: list[int], factor: np.ndarray):
            if len(members) == 1:
                actor_buffer = self.actor_buffer[members[0]]
                actor_buffer.update_factor(factor)
                return actor_buffer

            buffers = [self.actor_buffer[agent_id] for agent_id in members]
            grouped = copy.copy(buffers[0])
            for attr in ("obs", "rnn_states", "masks", "active_masks"):
                setattr(grouped, attr, np.concatenate([getattr(buf, attr) for buf in buffers], axis=1))
            for attr in ("actions", "action_log_probs"):
                setattr(grouped, attr, np.concatenate([getattr(buf, attr) for buf in buffers], axis=1))
            if buffers[0].available_actions is None:
                grouped.available_actions = None
            else:
                grouped.available_actions = np.concatenate(
                    [buf.available_actions for buf in buffers],
                    axis=1,
                )
            grouped.factor = np.concatenate([factor for _ in members], axis=1)
            grouped.n_rollout_threads = int(grouped.actions.shape[1])
            return grouped

        def _group_advantages(self, advantages: np.ndarray, members: list[int]) -> np.ndarray:
            if self.state_type == "EP":
                if len(members) == 1:
                    return advantages.copy()
                return np.concatenate([advantages.copy() for _ in members], axis=1)
            if self.state_type == "FP":
                return np.concatenate(
                    [advantages[:, :, agent_id].copy() for agent_id in members],
                    axis=1,
                )
            raise ValueError(f"unsupported state_type: {self.state_type}")

        def _annotate_policy_group_info(
            self,
            train_info: dict,
            group_index: int,
            representative: int,
            group_size: int,
        ) -> dict:
            if not self._use_class_shared_policy():
                return dict(train_info)
            group_name = self._policy_group_names[group_index]
            out = dict(train_info)
            out["policy_group/id"] = float(group_index)
            out["policy_group/size"] = float(group_size)
            out["policy_group/representative"] = float(representative)
            out["policy_group/is_uav"] = 1.0 if group_name == "uav" else 0.0
            out["policy_group/is_ugv"] = 1.0 if group_name == "ugv" else 0.0
            return out

        def run(self):
            """Run the training pipeline with class-shared actors decayed once."""
            if self.algo_args["render"]["use_render"] is True:
                self.render()
                return
            print("start running")
            self.warmup()

            episodes = (
                int(self.algo_args["train"]["num_env_steps"])
                // self.algo_args["train"]["episode_length"]
                // self.algo_args["train"]["n_rollout_threads"]
            )

            self.logger.init(episodes)

            for episode in range(1, episodes + 1):
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    for agent_id in self._unique_actor_agent_ids():
                        self.actor[agent_id].lr_decay(episode, episodes)
                    self.critic.lr_decay(episode, episodes)

                self.logger.episode_init(episode)

                self.prep_rollout()
                for step in range(self.algo_args["train"]["episode_length"]):
                    (
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    ) = self.collect(step)
                    (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                    ) = self.envs.step(actions)
                    data = (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )

                    self.logger.per_step(data)
                    self.insert(data)

                self.compute()
                self.prep_training()

                actor_train_infos, critic_train_info = self.train()

                if episode % self.algo_args["train"]["log_interval"] == 0:
                    self.logger.episode_log(
                        actor_train_infos,
                        critic_train_info,
                        self.actor_buffer,
                        self.critic_buffer,
                    )

                if episode % self.algo_args["train"]["eval_interval"] == 0:
                    if self.algo_args["eval"]["use_eval"]:
                        self.prep_rollout()
                        self.eval()
                    self.save()

                self.after_update()

        def restore(self):
            """Restore actors from model_dir; restore critic + value_normalizer
            only if those files exist.

            A behavioural-cloning warm-start dir (scripts/train_bc_happo.py) ships
            actor_agent*.pt but no critic/normalizer — in that case we want to load
            the cloned actors and let the critic learn from scratch, rather than
            crashing on a missing critic_agent.pt.
            """
            import os

            self._configure_class_shared_actors()
            model_dir = str(self.algo_args["train"]["model_dir"])
            for agent_id in self._unique_actor_agent_ids():
                actor_path = os.path.join(model_dir, f"actor_agent{agent_id}.pt")
                self.actor[agent_id].actor.load_state_dict(torch.load(actor_path))

            if self.algo_args["render"]["use_render"]:
                return

            critic_path = os.path.join(model_dir, "critic_agent.pt")
            if os.path.isfile(critic_path):
                self.critic.critic.load_state_dict(torch.load(critic_path))
                vn_path = os.path.join(model_dir, "value_normalizer.pt")
                if self.value_normalizer is not None and os.path.isfile(vn_path):
                    self.value_normalizer.load_state_dict(torch.load(vn_path))
            else:
                print(
                    f"[restore] No critic_agent.pt in {model_dir}; "
                    "loaded BC actors only, critic will train from scratch."
                )

        def train(self):
            actor_train_infos: list[dict | None] = [None] * self.num_agents
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

            self._configure_class_shared_actors()
            update_groups = (
                self._policy_update_groups
                if self._use_class_shared_policy()
                else [[agent_id] for agent_id in range(self.num_agents)]
            )
            if self.fixed_order:
                group_order = list(range(len(update_groups)))
            else:
                group_order = list(torch.randperm(len(update_groups)).numpy())

            scenario_kwargs = self.env_args.get("scenario_kwargs", {})
            n_survivors = int(scenario_kwargs.get("n_survivors", 0))
            action_transform = str(self.env_args.get("action_transform", "clip"))
            survivor_message_distance_scale_m = float(
                scenario_kwargs.get("survivor_message_distance_scale_m", 100.0),
            )

            for group_index in group_order:
                members = list(update_groups[group_index])
                representative = members[0]
                old_actions_logprob_by_agent = {
                    agent_id: self._evaluate_agent_actions(agent_id)
                    for agent_id in members
                }
                group_buffer = self._group_actor_buffer(members, factor)
                group_advantages = self._group_advantages(advantages, members)
                actor_train_info = self.actor[representative].train(
                    group_buffer,
                    group_advantages,
                    self.state_type,
                )

                actor_train_info.update(
                    _advantage_alignment_diagnostics(
                        group_buffer,
                        group_advantages,
                        n_survivors=n_survivors,
                        action_transform=action_transform,
                        survivor_message_distance_scale_m=survivor_message_distance_scale_m,
                    ),
                )
                annotated_info = self._annotate_policy_group_info(
                    actor_train_info,
                    group_index,
                    representative,
                    len(members),
                )

                group_ratio = np.ones_like(factor)
                for agent_id in members:
                    group_ratio = group_ratio * self._agent_action_ratio(
                        agent_id,
                        old_actions_logprob_by_agent[agent_id],
                    )
                    actor_train_infos[agent_id] = dict(annotated_info)
                factor = factor * group_ratio

            critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
            return [
                info if info is not None else {}
                for info in actor_train_infos
            ], critic_train_info

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
