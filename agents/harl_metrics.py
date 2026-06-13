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
)

ENV_METRICS = ADDITIVE_ENV_METRICS + FINAL_ENV_METRICS


def init_env_metric_storage(logger: Any) -> None:
    n_threads = int(logger.algo_args["train"]["n_rollout_threads"])
    logger.env_metric_episode_sums = {
        key: np.zeros(n_threads, dtype=np.float64) for key in ADDITIVE_ENV_METRICS
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

        for key in ADDITIVE_ENV_METRICS:
            logger.env_metric_episode_sums[key][env_index] += _as_float(info.get(key, 0.0))

        if dones_env[env_index]:
            for key in ADDITIVE_ENV_METRICS:
                logger.done_env_metrics[key].append(
                    float(logger.env_metric_episode_sums[key][env_index]),
                )
                logger.env_metric_episode_sums[key][env_index] = 0.0
            for key in FINAL_ENV_METRICS:
                logger.done_env_metrics[key].append(_as_float(info.get(key, 0.0)))


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
