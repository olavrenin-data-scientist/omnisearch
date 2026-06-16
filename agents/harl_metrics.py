"""Helpers for logging OmniSearch environment metrics through HARL."""

from __future__ import annotations

from typing import Any

import numpy as np


ADDITIVE_ENV_METRICS = (
    "mission/new_scouts",
    "mission/new_confirmations",
    "reward/team",
    "reward/drone_scout",
    "reward/drone_progress",
    "reward/ugv_progress",
    "reward/ugv_approach",
    "reward/ugv_movement_alignment",
    "reward/ground_confirm",
    "reward/coverage",
    "cost/ugv_fire_exposure",
    "cost/ugv_travel",
    "cost/drone_energy",
    "cost/drone_climb",
)

FINAL_ENV_METRICS = (
    "mission/n_scouted",
    "mission/n_confirmed",
    "mission/full_success",
    "diagnostic/ugv_final_known_target_distance_m",
    "diagnostic/ugv_confirm_range_m",
)

MIN_ENV_METRICS = (
    "diagnostic/ugv_min_known_target_distance_m",
)

ADDITIVE_DIAGNOSTIC_METRICS = (
    "diagnostic/ugv_steps_within_confirm_range",
    "diagnostic/ugv_steps_within_12m",
    "diagnostic/ugv_steps_within_15m",
    "diagnostic/ugv_known_target_valid",
    "diagnostic/ugv_same_target",
    "diagnostic/ugv_prev_distance_valid",
    "diagnostic/ugv_progress_gate_active",
    "diagnostic/ugv_ground_progress_m",
    "diagnostic/ugv_ground_progress_scaled",
    "diagnostic/ugv_action_alignment",
    "diagnostic/ugv_movement_alignment",
)

FINAL_DIAGNOSTIC_METRICS = (
    "diagnostic/ugv_target_index",
)

ENV_METRICS = (
    ADDITIVE_ENV_METRICS
    + ADDITIVE_DIAGNOSTIC_METRICS
    + FINAL_ENV_METRICS
    + FINAL_DIAGNOSTIC_METRICS
    + MIN_ENV_METRICS
)


def init_env_metric_storage(logger: Any) -> None:
    n_threads = int(logger.algo_args["train"]["n_rollout_threads"])
    logger.env_metric_episode_sums = {
        key: np.zeros(n_threads, dtype=np.float64)
        for key in ADDITIVE_ENV_METRICS + ADDITIVE_DIAGNOSTIC_METRICS
    }
    logger.env_metric_episode_mins = {
        key: np.full(n_threads, np.inf, dtype=np.float64)
        for key in MIN_ENV_METRICS
    }
    logger.done_env_metrics = {key: [] for key in ENV_METRICS}


def accumulate_env_metrics(logger: Any, infos: Any, dones: np.ndarray) -> None:
    if not hasattr(logger, "env_metric_episode_sums"):
        init_env_metric_storage(logger)

    dones_env = np.all(dones, axis=1)
    for env_index in range(len(dones_env)):
        info = _first_agent_info(infos, env_index)
        if not info:
            continue

        for key in ADDITIVE_ENV_METRICS + ADDITIVE_DIAGNOSTIC_METRICS:
            logger.env_metric_episode_sums[key][env_index] += _as_float(info.get(key, 0.0))
        for key in MIN_ENV_METRICS:
            value = _as_float(info.get(key, np.inf))
            if np.isfinite(value):
                logger.env_metric_episode_mins[key][env_index] = min(
                    logger.env_metric_episode_mins[key][env_index],
                    value,
                )

        if dones_env[env_index]:
            for key in ADDITIVE_ENV_METRICS + ADDITIVE_DIAGNOSTIC_METRICS:
                logger.done_env_metrics[key].append(
                    float(logger.env_metric_episode_sums[key][env_index]),
                )
                logger.env_metric_episode_sums[key][env_index] = 0.0
            for key in FINAL_ENV_METRICS + FINAL_DIAGNOSTIC_METRICS:
                logger.done_env_metrics[key].append(_as_float(info.get(key, 0.0)))
            for key in MIN_ENV_METRICS:
                value = logger.env_metric_episode_mins[key][env_index]
                logger.done_env_metrics[key].append(float(value) if np.isfinite(value) else 0.0)
                logger.env_metric_episode_mins[key][env_index] = np.inf


def log_done_env_metrics(logger: Any) -> None:
    if not hasattr(logger, "done_env_metrics"):
        return
    env_infos = {
        key: values
        for key, values in logger.done_env_metrics.items()
        if len(values) > 0
    }
    if not env_infos:
        return
    logger.log_env(env_infos)
    logger.done_env_metrics = {key: [] for key in ENV_METRICS}


def _first_agent_info(infos: Any, env_index: int) -> dict[str, Any]:
    arr = np.asarray(infos, dtype=object)
    if arr.ndim == 2:
        item = arr[env_index, 0]
    elif arr.ndim == 1:
        item = arr[env_index]
        if isinstance(item, (list, tuple, np.ndarray)):
            item = item[0]
    else:
        item = {}
    return item if isinstance(item, dict) else {}


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])
