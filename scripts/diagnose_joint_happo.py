#!/usr/bin/env python3
"""Diagnostics for joint UAV scouting plus UGV confirmation HAPPO checkpoints."""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import (
    HappoPolicy,
    actor_file_indices_for_scenario,
    find_latest_happo_checkpoint,
)
from envs.wildfire_defaults import COMMS_DROPOUT_MODE
from envs.wildfire_search import WildfireSearchScenario
from scripts.diagnostic_json import (
    partial_json_path,
    write_final_json,
    write_partial_json,
)
from scripts.train_happo_smoke import build_args


UAV_REWARD_COMPONENTS = (
    ("team", "reward/team"),
    ("team_scout", "reward/team_scout"),
    ("drone_scout", "reward/drone_scout"),
    ("uav_move_cov", "reward/uav_move_coverage"),
    ("uav_inefficient", "reward/uav_inefficient_move"),
    ("uav_coverage95", "reward/uav_coverage_threshold"),
    ("uav_conf", "reward/uav_confidence"),
    ("uav_team_conf", "reward/uav_team_confidence"),
    ("uav_team_conf_ov", "reward/uav_team_confidence_overlap"),
    ("uav_conf_move", "reward/uav_confidence_move"),
    ("uav_conf_overlap", "reward/uav_confidence_overlap"),
    ("uav_frontier", "reward/uav_frontier_alignment"),
    ("uav_cleanup", "reward/uav_cleanup_target_progress"),
    ("uav_astar", "reward/uav_astar_progress"),
    ("uav_overlap", "reward/uav_overlap"),
    ("uav_inter_overlap", "reward/uav_inter_uav_overlap"),
    ("uav_outside", "reward/uav_outside_footprint"),
    ("uav_fire", "reward/uav_fire_footprint"),
)

UGV_REWARD_COMPONENTS = (
    ("team", "reward/team"),
    ("ground_confirm", "reward/ground_confirm"),
    ("ugv_progress", "reward/ugv_progress"),
    ("ugv_approach", "reward/ugv_approach"),
    ("ugv_align", "reward/ugv_movement_alignment"),
    ("ugv_planner", "reward/ugv_planner_progress"),
    ("ugv_stall", "reward/ugv_stall_penalty"),
    ("ugv_route_floor", "reward/ugv_route_progress_floor_penalty"),
    ("ugv_route_shortfall", "reward/ugv_route_progress_shortfall_penalty"),
    ("pending", "reward/pending_penalty"),
    ("ugv_travel", "cost/ugv_travel"),
)

RECALL_TIME_THRESHOLDS = (0.50, 0.80, 0.90, 1.00)


def _checkpoint_path(path: str | None) -> Path:
    return Path(path) if path else find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _normalize_active_range(
    scenario_kwargs: dict[str, Any],
    *,
    slot_key: str,
    min_key: str,
    max_key: str,
    explicit_min: int | None,
    explicit_max: int | None,
) -> None:
    slots = max(int(scenario_kwargs.get(slot_key, 0)), 0)
    min_value = int(scenario_kwargs.get(min_key, slots)) if explicit_min is None else int(explicit_min)
    max_value = int(scenario_kwargs.get(max_key, slots)) if explicit_max is None else int(explicit_max)
    explicit = explicit_min is not None or explicit_max is not None
    if min_value < 0 or max_value < min_value or max_value > slots:
        if explicit:
            raise ValueError(f"{min_key}/{max_key} must satisfy 0 <= min <= max <= {slot_key}")
        min_value = slots
        max_value = slots
    scenario_kwargs[min_key] = min_value
    scenario_kwargs[max_key] = max_value


def _active_survivor_mask_for_env(scenario: WildfireSearchScenario, env_index: int = 0) -> np.ndarray:
    slots = int(getattr(scenario, "n_survivors", 0))
    active = getattr(scenario, "active_survivors", None)
    if active is None:
        return np.ones(slots, dtype=bool)
    return active[env_index].detach().cpu().numpy().astype(bool)


def _joint_defaults() -> dict[str, Any]:
    _, _algo_args, env_args = build_args(
        num_env_steps=100,
        episode_length=100,
        seed=1,
        comms_dropout=0.0,
        entropy_coef=0.01,
        exp_name="joint_diag_eval_defaults",
        joint_survivor_diagnostic=True,
    )
    return copy.deepcopy(env_args["scenario_kwargs"])


def _joint_schema_ugv_defaults() -> dict[str, Any]:
    _, _algo_args, env_args = build_args(
        num_env_steps=100,
        episode_length=100,
        seed=1,
        comms_dropout=0.0,
        entropy_coef=0.01,
        exp_name="joint_schema_ugv_diag_eval_defaults",
        joint_schema_ugv_diagnostic=True,
    )
    return copy.deepcopy(env_args["scenario_kwargs"])


def _scenario_kwargs(checkpoint_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(checkpoint_dir)
    scenario_kwargs: dict[str, Any] = {}
    if manifest is not None:
        scenario_kwargs.update(copy.deepcopy(manifest.get("env_args", {}).get("scenario_kwargs", {})))
    if args.joint_survivor_diagnostic or args.joint_schema_ugv_diagnostic or not scenario_kwargs:
        defaults = _joint_schema_ugv_defaults() if args.joint_schema_ugv_diagnostic else _joint_defaults()
        defaults.update(scenario_kwargs)
        scenario_kwargs = defaults

    scenario_kwargs["max_steps"] = int(args.steps)
    scenario_kwargs.setdefault("n_drones", 3)
    scenario_kwargs.setdefault("n_ground", int(args.joint_diagnostic_ugvs))
    scenario_kwargs.setdefault("n_survivors", 5)
    scenario_kwargs.setdefault("obs_schema_n_drones", scenario_kwargs.get("n_drones", 3))
    scenario_kwargs.setdefault("obs_schema_n_ground", scenario_kwargs.get("n_ground", int(args.joint_diagnostic_ugvs)))
    scenario_kwargs.setdefault("obs_schema_n_survivors", scenario_kwargs.get("n_survivors", 5))
    if args.n_drones is not None:
        if args.joint_schema_ugv_diagnostic:
            scenario_kwargs["obs_schema_n_drones"] = int(args.n_drones)
        else:
            scenario_kwargs["n_drones"] = int(args.n_drones)
            scenario_kwargs["obs_schema_n_drones"] = int(args.n_drones)
    if args.n_ugvs is not None:
        scenario_kwargs["n_ground"] = int(args.n_ugvs)
        scenario_kwargs["obs_schema_n_ground"] = int(args.n_ugvs)
    if args.n_survivors is not None:
        scenario_kwargs["n_survivors"] = int(args.n_survivors)
        scenario_kwargs["obs_schema_n_survivors"] = int(args.n_survivors)
    if getattr(args, "n_decoys", None) is not None:
        scenario_kwargs["n_decoys"] = max(int(args.n_decoys), 0)
    scenario_kwargs.setdefault("known_survivors_at_reset", False)
    scenario_kwargs.setdefault("drone_can_confirm", False)
    scenario_kwargs.setdefault("comms_dropout", 0.0)
    scenario_kwargs.setdefault("ugv_target_assignment_mode", "route_cost_sticky")
    scenario_kwargs.setdefault("ugv_planner_hint", "global_astar")
    scenario_kwargs.setdefault("ugv_dense_reward_mode", "planner_follow")
    scenario_kwargs["uav_confidence_diagnostics"] = True

    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if getattr(args, "fire_grid_size", None) is not None:
        scenario_kwargs["fire_grid_size"] = int(args.fire_grid_size)
    if getattr(args, "drone_perception_mode", None) is not None:
        scenario_kwargs["drone_perception_mode"] = (
            str(args.drone_perception_mode).replace("+", "_").replace("-", "_")
        )
    if getattr(args, "uav_fire_block_threshold", None) is not None:
        scenario_kwargs["uav_fire_block_threshold"] = float(args.uav_fire_block_threshold)
    if getattr(args, "uav_fire_footprint_penalty", None) is not None:
        scenario_kwargs["r_uav_fire_footprint"] = float(args.uav_fire_footprint_penalty)
    if getattr(args, "uav_fire_penalty_threshold", None) is not None:
        scenario_kwargs["uav_fire_penalty_threshold"] = float(args.uav_fire_penalty_threshold)
    if getattr(args, "no_variable_drone_clearance", False):
        for key in (
            "drone_safety_clearance_by_land_cover_m",
            "drone_safety_clearance_by_object_m",
            "drone_fire_safety_clearance_m",
            "drone_smoke_safety_clearance_m",
            "drone_smoke_clearance_threshold",
        ):
            scenario_kwargs.pop(key, None)
    else:
        if getattr(args, "drone_safety_clearance_by_land_cover_m", None) is not None:
            scenario_kwargs["drone_safety_clearance_by_land_cover_m"] = tuple(
                float(v) for v in args.drone_safety_clearance_by_land_cover_m
            )
        if getattr(args, "drone_safety_clearance_by_object_m", None) is not None:
            scenario_kwargs["drone_safety_clearance_by_object_m"] = tuple(
                float(v) for v in args.drone_safety_clearance_by_object_m
            )
        if getattr(args, "drone_fire_safety_clearance_m", None) is not None:
            scenario_kwargs["drone_fire_safety_clearance_m"] = float(args.drone_fire_safety_clearance_m)
        if getattr(args, "drone_smoke_safety_clearance_m", None) is not None:
            scenario_kwargs["drone_smoke_safety_clearance_m"] = float(args.drone_smoke_safety_clearance_m)
        if getattr(args, "drone_smoke_clearance_threshold", None) is not None:
            scenario_kwargs["drone_smoke_clearance_threshold"] = float(args.drone_smoke_clearance_threshold)
    fire_override = getattr(args, "enable_fire", None)
    if fire_override is not None:
        scenario_kwargs["disable_fire"] = not bool(fire_override)
    if getattr(args, "comms_dropout", None) is not None:
        scenario_kwargs["comms_dropout"] = float(args.comms_dropout)
    if getattr(args, "comms_dropout_mode", None) is not None:
        scenario_kwargs["comms_dropout_mode"] = str(args.comms_dropout_mode).replace("-", "_")
    if getattr(args, "comms_map_mode", None) is not None:
        scenario_kwargs["comms_map_mode"] = str(args.comms_map_mode).replace("-", "_")
    if getattr(args, "comms_dropout_min_steps", None) is not None:
        scenario_kwargs["comms_dropout_min_steps"] = int(args.comms_dropout_min_steps)
    if getattr(args, "comms_dropout_max_steps", None) is not None:
        scenario_kwargs["comms_dropout_max_steps"] = int(args.comms_dropout_max_steps)
    if args.ugv_target_assignment_mode is not None:
        scenario_kwargs["ugv_target_assignment_mode"] = args.ugv_target_assignment_mode.replace("-", "_")
    for attr in (
        "uav_decision_grid",
        "uav_confidence_reward_grid",
        "uav_frontier_global_grid",
        "uav_coverage_reward_grid",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            scenario_kwargs[attr] = int(value)
    _normalize_active_range(
        scenario_kwargs,
        slot_key="n_survivors",
        min_key="active_survivors_min",
        max_key="active_survivors_max",
        explicit_min=getattr(args, "active_survivors_min", None),
        explicit_max=getattr(args, "active_survivors_max", None),
    )
    _normalize_active_range(
        scenario_kwargs,
        slot_key="n_decoys",
        min_key="active_decoys_min",
        max_key="active_decoys_max",
        explicit_min=getattr(args, "active_decoys_min", None),
        explicit_max=getattr(args, "active_decoys_max", None),
    )
    scenario_kwargs["comms_map_mode"] = "per_agent"
    return scenario_kwargs


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(finite)) if finite else float("nan")


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "std": float(np.std(finite)) if finite else float("nan"),
        "count": float(len(finite)),
    }


def _recall_threshold_time_stats(
    rows: list[dict[str, Any]],
    *,
    key: str,
    threshold: float,
    max_survivors: int | None = None,
) -> dict[str, Any]:
    if max_survivors is None:
        max_survivors = max((int(row.get("survivors", 0)) for row in rows), default=0)
    max_survivors = max(int(max_survivors), 0)
    required = max(1, int(math.ceil(float(threshold) * max_survivors - 1e-9)))
    times_s: list[float] = []
    eligible_count = 0
    for row in rows:
        survivors = int(row.get("survivors", 0))
        if survivors < required:
            continue
        eligible_count += 1
        event_steps = sorted(
            float(step)
            for step in row.get(key, [])
            if step is not None and math.isfinite(float(step))
        )
        if len(event_steps) < required:
            continue
        step_seconds = max(float(row.get("step_seconds", 1.0)), 1e-9)
        times_s.append(event_steps[required - 1] * step_seconds)
    reached_count = len(times_s)
    std_s = _std(times_s)
    ci95_s = (
        1.96 * std_s / math.sqrt(reached_count)
        if reached_count > 0 and math.isfinite(std_s)
        else float("nan")
    )
    mean_s = _mean(times_s)
    return {
        "threshold": float(threshold),
        "threshold_basis": "configured_max_survivors",
        "max_survivors": float(max_survivors),
        "required_count": float(required),
        "total_count": float(len(rows)),
        "eligible_count": float(eligible_count),
        "eligible_fraction": float(eligible_count / len(rows)) if rows else float("nan"),
        "ineligible_count": float(len(rows) - eligible_count),
        "reached_count": float(reached_count),
        "valid_count": float(reached_count),
        "reached_fraction": (
            float(reached_count / eligible_count) if eligible_count > 0 else float("nan")
        ),
        "mean_s": mean_s,
        "std_s": std_s,
        "ci95_s": ci95_s,
        "ci95_lower_s": mean_s - ci95_s,
        "ci95_upper_s": mean_s + ci95_s,
    }


def _threshold_time_summary(
    rows: list[dict[str, Any]],
    key: str,
    *,
    max_survivors: int | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        f"recall_{int(round(threshold * 100)):03d}": _recall_threshold_time_stats(
            rows,
            key=key,
            threshold=threshold,
            max_survivors=max_survivors,
        )
        for threshold in RECALL_TIME_THRESHOLDS
    }


def _positions_m(
    agents: list[Any],
    *,
    meters_per_sim: float,
) -> list[list[float]]:
    return [
        (agent.state.pos[0].detach().cpu().numpy().astype(float) * meters_per_sim).tolist()
        for agent in agents
    ]


def _metric_array(
    scenario: WildfireSearchScenario,
    name: str,
    size: int,
) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=float)
    value = getattr(scenario, name, None)
    if value is None:
        return np.zeros(size, dtype=float)
    array = value.detach().cpu().numpy().astype(float).reshape(-1)
    if array.size == size:
        return array
    if array.size == 1:
        return np.repeat(array, size)
    return np.resize(array, size)


def _event_time_bins(
    rows: list[dict[str, Any]],
    *,
    key: str,
    bins: int,
) -> list[dict[str, float]]:
    bins = max(int(bins), 1)
    output: list[dict[str, float]] = []
    for bin_index in range(bins):
        start_fraction = bin_index / bins
        end_fraction = (bin_index + 1) / bins
        cumulative: list[float] = []
        new_recall: list[float] = []
        for row in rows:
            steps = row.get(key, [])
            episode_steps = max(int(row.get("episode_steps", 0)), 1)
            survivors = max(int(row.get("survivors", 0)), 1)
            start_step = start_fraction * episode_steps
            end_step = end_fraction * episode_steps
            cumulative.append(
                sum(step is not None and float(step) <= end_step for step in steps) / survivors
            )
            new_recall.append(
                sum(
                    step is not None and start_step < float(step) <= end_step
                    for step in steps
                )
                / survivors
            )
        output.append(
            {
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
                "mean_new_recall": _mean(new_recall),
                "mean_cumulative_recall": _mean(cumulative),
                "median_cumulative_recall": (
                    float(np.median(cumulative)) if cumulative else float("nan")
                ),
            }
        )
    return output


def _bin_index(step: int, steps: int, bins: int) -> int:
    if bins <= 1:
        return 0
    return min(int(step / max(steps, 1) * bins), bins - 1)


def _new_time_bins(bins: int) -> list[dict[str, float]]:
    return [
        {
            "count": 0.0,
            "uav_new_cells": 0.0,
            "uav_conf_gain": 0.0,
            "uav_displacement_m": 0.0,
            "uav_overlap": 0.0,
            "uav_excess_overlap": 0.0,
            "uav_edge_step": 0.0,
            "uav_moving_no_new": 0.0,
            "ugv_progress_m": 0.0,
            "ugv_displacement_m": 0.0,
            "ugv_route_active": 0.0,
            "pending": 0.0,
            "new_scouts": 0.0,
            "new_oracle_reveals": 0.0,
            "new_confirmations": 0.0,
            "duplicate_assignment": 0.0,
            "assignment_switches": 0.0,
            **{f"uav_reward_{name}": 0.0 for name, _key in UAV_REWARD_COMPONENTS},
            **{f"ugv_reward_{name}": 0.0 for name, _key in UGV_REWARD_COMPONENTS},
        }
        for _ in range(bins)
    ]


def _positions(scenario: WildfireSearchScenario) -> torch.Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def run_rollout(
    policy: HappoPolicy,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    time_bins: int,
) -> dict[str, Any]:
    env = WildfireSearchScenario.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset(seed=seed)
    policy.reset()
    scenario = env.scenario
    n_drones = int(scenario.n_drones)
    n_ground = int(scenario.n_ground)
    n_agents = int(scenario.n_agents)
    survivor_slots = int(scenario.n_survivors)
    active_survivor_mask = _active_survivor_mask_for_env(scenario)
    active_survivor_indices = np.flatnonzero(active_survivor_mask)
    n_active_survivors = int(active_survivor_mask.sum())
    max_steps = int(scenario_kwargs["max_steps"])
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    meters_per_sim = 1.0 / max(
        float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()),
        1e-12,
    )
    x_half_m = float(scenario.x_semidim) * meters_per_sim
    y_half_m = float(scenario.y_semidim) * meters_per_sim

    uav_agents = list(scenario.world.agents[:n_drones])
    ugv_agents = list(scenario.world.agents[n_drones:n_drones + n_ground])
    uav_start_positions_m = _positions_m(uav_agents, meters_per_sim=meters_per_sim)
    ugv_start_positions_m = _positions_m(ugv_agents, meters_per_sim=meters_per_sim)
    survivor_positions_m = _positions_m(
        [scenario._survivors[index] for index in active_survivor_indices],
        meters_per_sim=meters_per_sim,
    )

    first_scout_steps: list[int | None] = [None] * survivor_slots
    first_confirm_steps: list[int | None] = [None] * survivor_slots
    path_lengths_m = np.zeros(n_agents, dtype=float)
    time_series = _new_time_bins(max(int(time_bins), 1))
    pending_counts: list[float] = []
    duplicate_assignment: list[float] = []
    assignment_switches: list[float] = []
    uav_reward_terms = {name: [] for name, _key in UAV_REWARD_COMPONENTS}
    ugv_reward_terms = {name: [] for name, _key in UGV_REWARD_COMPONENTS}
    uav_overlap_values: list[float] = []
    uav_excess_overlap_values: list[float] = []
    uav_edge_values: list[float] = []
    uav_moving_no_new_values: list[float] = []
    uav_fire_footprint_values: list[float] = []
    ugv_fire_exposure_values: list[float] = []
    hazard_exposure_values: list[float] = []
    scout_auc_sum = 0.0
    confirm_auc_sum = 0.0
    coverage_auc_sum = 0.0
    confidence_auc_sum = 0.0
    auc_steps = 0

    prev_pos = _positions(scenario).clone()
    for step in range(max_steps):
        actions = policy(env)
        env.step(actions)

        pos = _positions(scenario).clone()
        displacement_m = (
            torch.linalg.norm(pos[0] - prev_pos[0], dim=-1).detach().cpu().numpy()
            * meters_per_sim
        )
        path_lengths_m += displacement_m
        prev_pos = pos

        scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        confirmed = scenario.found_survivors[0].detach().cpu().numpy().astype(bool)
        if n_active_survivors > 0:
            scout_auc_sum += float(np.logical_and(active_survivor_mask, scouted).sum() / n_active_survivors)
            confirm_auc_sum += float(np.logical_and(active_survivor_mask, confirmed).sum() / n_active_survivors)
        else:
            scout_auc_sum += 1.0
            confirm_auc_sum += 1.0
        coverage_auc_sum += float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
        confidence_auc_sum += float(scenario.uav_confidence_grid[0].float().mean().detach().cpu().item())
        auc_steps += 1
        for survivor_idx in active_survivor_indices:
            if scouted[survivor_idx] and first_scout_steps[survivor_idx] is None:
                first_scout_steps[survivor_idx] = step + 1
            if confirmed[survivor_idx] and first_confirm_steps[survivor_idx] is None:
                first_confirm_steps[survivor_idx] = step + 1

        info = scenario.info(env.agents[0])
        pending = float(np.logical_and(active_survivor_mask & scouted, ~confirmed).sum())
        pending_counts.append(pending)
        duplicate = _to_float(info.get("diagnostic/ugv_duplicate_assignment_fraction"))
        switches = _to_float(info.get("diagnostic/ugv_assignment_switches"))
        duplicate_assignment.append(duplicate)
        assignment_switches.append(switches)
        uav_fire_footprint = _to_float(info.get("diagnostic/uav_fire_footprint_fraction"))
        ugv_fire_exposure = _to_float(info.get("cost/ugv_fire_exposure"))
        ugv_fire_fraction = ugv_fire_exposure / max(n_ground, 1)
        hazard_exposure = (
            (uav_fire_footprint * n_drones) + ugv_fire_exposure
        ) / max(n_drones + n_ground, 1)
        uav_fire_footprint_values.append(uav_fire_footprint)
        ugv_fire_exposure_values.append(ugv_fire_fraction)
        hazard_exposure_values.append(hazard_exposure)

        bin_row = time_series[_bin_index(step, max_steps, len(time_series))]
        bin_row["count"] += 1.0
        new_cells_by_drone = _metric_array(
            scenario,
            "metric_uav_new_coverage_cells_by_drone",
            n_drones,
        )
        overlap_by_drone = _metric_array(
            scenario,
            "metric_uav_overlap_fraction_by_drone",
            n_drones,
        )
        excess_overlap_by_drone = _metric_array(
            scenario,
            "metric_uav_excess_overlap_fraction_by_drone",
            n_drones,
        )
        boundary_distance_by_drone = _metric_array(
            scenario,
            "metric_uav_boundary_distance_m_by_drone",
            n_drones,
        )
        footprint_radius_by_drone = _metric_array(
            scenario,
            "metric_uav_footprint_radius_m_by_drone",
            n_drones,
        )
        if n_drones > 0:
            uav_displacements = displacement_m[:n_drones]
            edge_steps = boundary_distance_by_drone <= np.where(
                footprint_radius_by_drone > 0.0,
                footprint_radius_by_drone,
                25.0,
            )
            moving_no_new = np.logical_and(uav_displacements > 1.0, new_cells_by_drone < 1.0)
            mean_new_cells = float(new_cells_by_drone.mean())
            mean_uav_displacement = float(uav_displacements.mean())
            mean_overlap = float(overlap_by_drone.mean())
            mean_excess_overlap = float(excess_overlap_by_drone.mean())
            mean_edge = float(edge_steps.mean())
            mean_moving_no_new = float(moving_no_new.mean())
            uav_overlap_values.extend(overlap_by_drone.tolist())
            uav_excess_overlap_values.extend(excess_overlap_by_drone.tolist())
            uav_edge_values.extend(edge_steps.astype(float).tolist())
            uav_moving_no_new_values.extend(moving_no_new.astype(float).tolist())
        else:
            mean_new_cells = 0.0
            mean_uav_displacement = 0.0
            mean_overlap = 0.0
            mean_excess_overlap = 0.0
            mean_edge = 0.0
            mean_moving_no_new = 0.0

        bin_row["uav_new_cells"] += mean_new_cells
        bin_row["uav_conf_gain"] += _to_float(info.get("diagnostic/uav_confidence_gain"))
        bin_row["uav_displacement_m"] += mean_uav_displacement
        bin_row["uav_overlap"] += mean_overlap
        bin_row["uav_excess_overlap"] += mean_excess_overlap
        bin_row["uav_edge_step"] += mean_edge
        bin_row["uav_moving_no_new"] += mean_moving_no_new
        bin_row["ugv_progress_m"] += _to_float(info.get("diagnostic/ugv_global_route_progress_m"))
        bin_row["ugv_displacement_m"] += (
            float(displacement_m[n_drones:].mean()) if n_ground > 0 else 0.0
        )
        bin_row["ugv_route_active"] += _to_float(info.get("diagnostic/ugv_global_route_active"))
        bin_row["pending"] += pending
        bin_row["new_scouts"] += _to_float(info.get("mission/new_scouts"))
        bin_row["new_oracle_reveals"] += _to_float(info.get("mission/new_oracle_reveals"))
        bin_row["new_confirmations"] += _to_float(info.get("mission/new_confirmations"))
        bin_row["duplicate_assignment"] += duplicate
        bin_row["assignment_switches"] += switches
        for name, key in UAV_REWARD_COMPONENTS:
            value = abs(_to_float(info.get(key)))
            uav_reward_terms[name].append(value)
            bin_row[f"uav_reward_{name}"] += value
        for name, key in UGV_REWARD_COMPONENTS:
            value = abs(_to_float(info.get(key)))
            ugv_reward_terms[name].append(value)
            bin_row[f"ugv_reward_{name}"] += value

    active_scout_steps = [first_scout_steps[idx] for idx in active_survivor_indices]
    active_confirm_steps = [first_confirm_steps[idx] for idx in active_survivor_indices]
    scout_count = sum(step is not None for step in active_scout_steps)
    confirm_count = sum(step is not None for step in active_confirm_steps)
    latencies = [
        float(confirm_step - scout_step)
        for scout_step, confirm_step in zip(first_scout_steps, first_confirm_steps)
        if scout_step is not None and confirm_step is not None and confirm_step >= scout_step
    ]
    time_bin_rows = []
    for idx, bucket in enumerate(time_series):
        count = max(bucket["count"], 1.0)
        row = {
            "episode_fraction": (idx + 0.5) / len(time_series),
            "uav_new_cells_per_step": bucket["uav_new_cells"] / count,
            "uav_confidence_gain_per_step": bucket["uav_conf_gain"] / count,
            "uav_displacement_m_per_step": bucket["uav_displacement_m"] / count,
            "uav_overlap_fraction": bucket["uav_overlap"] / count,
            "uav_excess_overlap_fraction": bucket["uav_excess_overlap"] / count,
            "uav_edge_step_fraction": bucket["uav_edge_step"] / count,
            "uav_moving_no_new_fraction": bucket["uav_moving_no_new"] / count,
            "ugv_route_progress_m_per_step": bucket["ugv_progress_m"] / count,
            "ugv_displacement_m_per_step": bucket["ugv_displacement_m"] / count,
            "ugv_route_active_fraction": bucket["ugv_route_active"] / count,
            "pending_known_survivors": bucket["pending"] / count,
            "new_scouts_per_step": bucket["new_scouts"] / count,
            "new_oracle_reveals_per_step": bucket["new_oracle_reveals"] / count,
            "new_confirmations_per_step": bucket["new_confirmations"] / count,
            "duplicate_assignment_fraction": bucket["duplicate_assignment"] / count,
            "assignment_switches_per_step": bucket["assignment_switches"] / count,
        }
        for name, _key in UAV_REWARD_COMPONENTS:
            row[f"uav_reward_{name}"] = bucket[f"uav_reward_{name}"] / count
        for name, _key in UGV_REWARD_COMPONENTS:
            row[f"ugv_reward_{name}"] = bucket[f"ugv_reward_{name}"] / count
        time_bin_rows.append(row)

    confirmed = scenario.found_survivors[0].detach().cpu().numpy().astype(bool)
    final_pending_distances_m: list[float] = []
    if n_ground > 0:
        ugv_positions = np.asarray(_positions_m(ugv_agents, meters_per_sim=meters_per_sim))
        survivor_positions = np.asarray(survivor_positions_m, dtype=float)
        for local_index, survivor_index in enumerate(active_survivor_indices):
            if confirmed[survivor_index]:
                continue
            final_pending_distances_m.append(
                float(np.linalg.norm(ugv_positions - survivor_positions[local_index], axis=1).min())
            )

    active_confirm_values = [step for step in active_confirm_steps if step is not None]
    uav_path_lengths = path_lengths_m[:n_drones]
    ugv_path_lengths = path_lengths_m[n_drones:]
    uav_failure_labels: list[str] = []
    if scout_count == n_active_survivors:
        uav_failure_labels.append("success")
    else:
        uav_failure_labels.append("partial_search")
        if float(scenario.coverage_grid[0].float().mean().item()) < 0.70:
            uav_failure_labels.append("low_coverage")
        if float(scenario.uav_confidence_grid[0].float().mean().item()) < 0.75:
            uav_failure_labels.append("low_confidence")
        if n_drones > 0 and float(uav_path_lengths.sum() / max(n_drones * max_steps, 1)) < 5.0:
            uav_failure_labels.append("slow_search")
        if _mean(uav_excess_overlap_values) > 0.20:
            uav_failure_labels.append("high_excess_overlap")
        if _mean(uav_edge_values) > 0.30:
            uav_failure_labels.append("boundary_heavy")
        scout_events = [value for value in active_scout_steps if value is not None]
        if scout_events and max(scout_events) > 0.80 * max_steps:
            uav_failure_labels.append("late_discovery")

    ugv_speed_mps = (
        float(ugv_path_lengths.sum() / max(n_ground * max_steps * step_seconds, 1e-9))
        if n_ground > 0 else 0.0
    )
    ugv_failure_labels: list[str] = []
    if confirm_count == n_active_survivors:
        ugv_failure_labels.append("success")
    else:
        if scout_count < n_active_survivors:
            ugv_failure_labels.append("unscouted_survivors")
        if confirm_count < scout_count:
            ugv_failure_labels.append("confirmation_backlog")
        if ugv_speed_mps < 0.50:
            ugv_failure_labels.append("slow_motion")
        if final_pending_distances_m and min(final_pending_distances_m) <= float(
            getattr(scenario, "ground_confirmation_range_m", 10.0)
        ):
            ugv_failure_labels.append("close_miss")
        if active_confirm_values and max(active_confirm_values) > 0.80 * max_steps:
            ugv_failure_labels.append("late_confirmation")
        if confirm_count < scout_count and ugv_speed_mps >= 0.50:
            ugv_failure_labels.append("unfinished_route")

    return {
        "seed": int(seed),
        "survivors": n_active_survivors,
        "active_survivors": n_active_survivors,
        "survivor_slots": survivor_slots,
        "scouted": int(scout_count),
        "confirmed": int(confirm_count),
        "scout_recall": float(scout_count / n_active_survivors) if n_active_survivors else 1.0,
        "confirm_recall": float(confirm_count / n_active_survivors) if n_active_survivors else 1.0,
        "scout_auc": float(scout_auc_sum / max(auc_steps, 1)),
        "confirm_auc": float(confirm_auc_sum / max(auc_steps, 1)),
        "coverage_auc": float(coverage_auc_sum / max(auc_steps, 1)),
        "confidence_auc": float(confidence_auc_sum / max(auc_steps, 1)),
        "overall_success": bool(confirm_count == n_active_survivors),
        "full_confirm_success": bool(confirm_count == n_active_survivors),
        "first_scout_steps": first_scout_steps,
        "first_confirm_steps": first_confirm_steps,
        "episode_steps": max_steps,
        "step_seconds": step_seconds,
        "scout_to_confirm_latencies_steps": latencies,
        "scout_to_confirm_latency_count": int(len(latencies)),
        "avg_scout_to_confirm_latency_steps": _mean(latencies),
        "avg_scout_to_confirm_latency_s": _mean(latencies) * step_seconds,
        "final_coverage_fraction": float(scenario.coverage_grid[0].float().mean().item()),
        "final_confidence_mean": float(scenario.uav_confidence_grid[0].float().mean().item()),
        "uav_path_length_m": float(path_lengths_m[:n_drones].sum()) if n_drones > 0 else 0.0,
        "uav_path_length_by_drone_m": [float(value) for value in uav_path_lengths],
        "uav_movement_m_per_drone_step": (
            float(path_lengths_m[:n_drones].sum() / max(n_drones * max_steps, 1))
            if n_drones > 0 else 0.0
        ),
        "ugv_path_length_m": float(path_lengths_m[n_drones:].sum()) if n_ground > 0 else 0.0,
        "ugv_path_length_by_ground_m": [float(value) for value in ugv_path_lengths],
        "ugv_movement_m_per_ground_step": (
            float(ugv_path_lengths.sum() / max(n_ground * max_steps, 1))
            if n_ground > 0 else 0.0
        ),
        "ugv_speed_mps": ugv_speed_mps,
        "ugv_travel_cost_per_step": _mean(ugv_reward_terms["ugv_travel"]),
        "ugv_travel_cost_per_ground_step": (
            _mean([value / max(n_ground, 1) for value in ugv_reward_terms["ugv_travel"]])
            if n_ground > 0 else 0.0
        ),
        "ugv_travel_cost_total": float(np.sum(ugv_reward_terms["ugv_travel"])),
        "uav_fire_footprint_fraction": _mean(uav_fire_footprint_values),
        "ugv_fire_exposure_fraction": _mean(ugv_fire_exposure_values),
        "hazard_exposure": _mean(hazard_exposure_values),
        "ugv_final_pending_distance_m": (
            float(np.mean(final_pending_distances_m)) if final_pending_distances_m else 0.0
        ),
        "ugv_final_pending_distances_m": final_pending_distances_m,
        "avg_confirm_step": _mean([float(value) for value in active_confirm_values]),
        "uav_start_positions_m": uav_start_positions_m,
        "ugv_start_positions_m": ugv_start_positions_m,
        "survivor_positions_m": survivor_positions_m,
        "map_width_m": 2.0 * x_half_m,
        "map_height_m": 2.0 * y_half_m,
        "avg_uav_overlap_fraction": _mean(uav_overlap_values),
        "avg_uav_excess_overlap_fraction": _mean(uav_excess_overlap_values),
        "uav_edge_step_fraction": _mean(uav_edge_values),
        "uav_moving_no_new_fraction": _mean(uav_moving_no_new_values),
        "uav_failure_labels": uav_failure_labels,
        "ugv_failure_labels": ugv_failure_labels,
        "path_length_by_agent_m": [float(v) for v in path_lengths_m],
        "pending_target_time_mean": _mean(pending_counts),
        "pending_target_time_fraction": float(np.count_nonzero(np.asarray(pending_counts) > 0) / max(len(pending_counts), 1)),
        "duplicate_ugv_assignment_rate": _mean(duplicate_assignment),
        "ugv_assignment_switches_per_episode": float(np.sum(assignment_switches)),
        "avg_uav_reward_components_abs": {
            name: _mean(values) for name, values in uav_reward_terms.items()
        },
        "avg_ugv_reward_components_abs": {
            name: _mean(values) for name, values in ugv_reward_terms.items()
        },
        "avg_reward_components_abs": {
            **{name: _mean(values) for name, values in uav_reward_terms.items()},
            **{name: _mean(values) for name, values in ugv_reward_terms.items()},
        },
        "time_bins": time_bin_rows,
    }


def _label_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for label in row.get(key, []):
            if label == "success":
                continue
            counts[str(label)] = counts.get(str(label), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _mean_path_by_agent(rows: list[dict[str, Any]], key: str) -> list[float]:
    count = max((len(row.get(key, [])) for row in rows), default=0)
    return [
        _mean([
            float(row[key][index])
            for row in rows
            if index < len(row.get(key, []))
        ])
        for index in range(count)
    ]


def summarize(
    rows: list[dict[str, Any]],
    bins: int = 5,
    *,
    max_survivors: int | None = None,
) -> dict[str, Any]:
    success_count = int(sum(bool(row["full_confirm_success"]) for row in rows))
    latency_values_s: list[float] = []
    for row in rows:
        step_seconds = max(float(row.get("step_seconds", 1.0)), 1e-9)
        latency_values_s.extend(
            float(value) * step_seconds
            for value in row.get("scout_to_confirm_latencies_steps", [])
            if math.isfinite(float(value))
        )
    ugv_travel_cost_per_ground_step = [
        row.get("ugv_travel_cost_per_ground_step", row.get("ugv_travel_cost_per_step", float("nan")))
        for row in rows
    ]
    hazard_exposure = [row.get("hazard_exposure", float("nan")) for row in rows]
    return {
        "episodes": float(len(rows)),
        "mean_scout_recall": _mean([row["scout_recall"] for row in rows]),
        "std_scout_recall": _std([row["scout_recall"] for row in rows]),
        "mean_confirm_recall": _mean([row["confirm_recall"] for row in rows]),
        "std_confirm_recall": _std([row["confirm_recall"] for row in rows]),
        "mean_scout_auc": _mean([row["scout_auc"] for row in rows]),
        "std_scout_auc": _std([row["scout_auc"] for row in rows]),
        "mean_confirm_auc": _mean([row["confirm_auc"] for row in rows]),
        "std_confirm_auc": _std([row["confirm_auc"] for row in rows]),
        "mean_coverage_auc": _mean([row["coverage_auc"] for row in rows]),
        "std_coverage_auc": _std([row["coverage_auc"] for row in rows]),
        "mean_confidence_auc": _mean([row["confidence_auc"] for row in rows]),
        "std_confidence_auc": _std([row["confidence_auc"] for row in rows]),
        "full_confirm_success_count": float(success_count),
        "full_confirm_success_rate": _mean([float(row["full_confirm_success"]) for row in rows]),
        "full_confirm_success_percent": (
            100.0 * success_count / max(len(rows), 1)
        ),
        "mean_final_coverage_fraction": _mean([row["final_coverage_fraction"] for row in rows]),
        "std_final_coverage_fraction": _std([row["final_coverage_fraction"] for row in rows]),
        "mean_final_confidence": _mean([row["final_confidence_mean"] for row in rows]),
        "std_final_confidence": _std([row["final_confidence_mean"] for row in rows]),
        "mean_uav_path_length_m": _mean([row["uav_path_length_m"] for row in rows]),
        "std_uav_path_length_m": _std([row["uav_path_length_m"] for row in rows]),
        "mean_uav_movement_m_per_drone_step": _mean([
            row["uav_movement_m_per_drone_step"] for row in rows
        ]),
        "mean_ugv_path_length_m": _mean([row["ugv_path_length_m"] for row in rows]),
        "std_ugv_path_length_m": _std([row["ugv_path_length_m"] for row in rows]),
        "mean_ugv_speed_mps": _mean([row["ugv_speed_mps"] for row in rows]),
        "std_ugv_speed_mps": _std([row["ugv_speed_mps"] for row in rows]),
        "mean_ugv_travel_cost_per_ground_step": _mean(ugv_travel_cost_per_ground_step),
        "std_ugv_travel_cost_per_ground_step": _std(ugv_travel_cost_per_ground_step),
        "mean_ugv_travel_cost_total": _mean([
            row.get("ugv_travel_cost_total", float("nan")) for row in rows
        ]),
        "std_ugv_travel_cost_total": _std([
            row.get("ugv_travel_cost_total", float("nan")) for row in rows
        ]),
        "mean_uav_fire_footprint_fraction": _mean([
            row.get("uav_fire_footprint_fraction", float("nan")) for row in rows
        ]),
        "std_uav_fire_footprint_fraction": _std([
            row.get("uav_fire_footprint_fraction", float("nan")) for row in rows
        ]),
        "mean_ugv_fire_exposure_fraction": _mean([
            row.get("ugv_fire_exposure_fraction", float("nan")) for row in rows
        ]),
        "std_ugv_fire_exposure_fraction": _std([
            row.get("ugv_fire_exposure_fraction", float("nan")) for row in rows
        ]),
        "mean_hazard_exposure": _mean(hazard_exposure),
        "std_hazard_exposure": _std(hazard_exposure),
        "mean_ugv_final_pending_distance_m": _mean([
            row["ugv_final_pending_distance_m"] for row in rows
        ]),
        "mean_uav_excess_overlap_fraction": _mean([
            row["avg_uav_excess_overlap_fraction"] for row in rows
        ]),
        "mean_uav_edge_step_fraction": _mean([
            row["uav_edge_step_fraction"] for row in rows
        ]),
        "mean_uav_moving_no_new_fraction": _mean([
            row["uav_moving_no_new_fraction"] for row in rows
        ]),
        "mean_confirm_step": _mean([row["avg_confirm_step"] for row in rows]),
        "mean_scout_to_confirm_latency_count": _mean([
            float(row["scout_to_confirm_latency_count"]) for row in rows
        ]),
        "total_scout_to_confirm_latency_count": float(
            sum(int(row["scout_to_confirm_latency_count"]) for row in rows)
        ),
        "mean_scout_to_confirm_latency_steps": _mean([
            row["avg_scout_to_confirm_latency_steps"] for row in rows
        ]),
        "mean_scout_to_confirm_latency_s": _mean([
            row["avg_scout_to_confirm_latency_s"] for row in rows
        ]),
        "std_scout_to_confirm_latency_s": _std(latency_values_s),
        "mean_pending_target_time_fraction": _mean([
            row["pending_target_time_fraction"] for row in rows
        ]),
        "mean_duplicate_ugv_assignment_rate": _mean([
            row["duplicate_ugv_assignment_rate"] for row in rows
        ]),
        "mean_ugv_assignment_switches_per_episode": _mean([
            row["ugv_assignment_switches_per_episode"] for row in rows
        ]),
        "mean_uav_path_length_by_drone_m": _mean_path_by_agent(
            rows,
            "uav_path_length_by_drone_m",
        ),
        "mean_ugv_path_length_by_ground_m": _mean_path_by_agent(
            rows,
            "ugv_path_length_by_ground_m",
        ),
        "uav_failure_label_counts": _label_counts(rows, "uav_failure_labels"),
        "ugv_failure_label_counts": _label_counts(rows, "ugv_failure_labels"),
        "scout_time_bins": _event_time_bins(
            rows,
            key="first_scout_steps",
            bins=bins,
        ),
        "confirm_time_bins": _event_time_bins(
            rows,
            key="first_confirm_steps",
            bins=bins,
        ),
        "time_to_scout_s": _threshold_time_summary(
            rows,
            "first_scout_steps",
            max_survivors=max_survivors,
        ),
        "time_to_confirm_s": _threshold_time_summary(
            rows,
            "first_confirm_steps",
            max_survivors=max_survivors,
        ),
        "fast_metrics": {
            "scout_recall": _mean_std([row["scout_recall"] for row in rows]),
            "confirm_recall": _mean_std([row["confirm_recall"] for row in rows]),
            "scout_auc": _mean_std([row["scout_auc"] for row in rows]),
            "confirm_auc": _mean_std([row["confirm_auc"] for row in rows]),
            "coverage_auc": _mean_std([row["coverage_auc"] for row in rows]),
            "confidence_auc": _mean_std([row["confidence_auc"] for row in rows]),
            "coverage": _mean_std([row["final_coverage_fraction"] for row in rows]),
            "confidence": _mean_std([row["final_confidence_mean"] for row in rows]),
            "uav_path_length_m": _mean_std([row["uav_path_length_m"] for row in rows]),
            "ugv_travel_cost_per_ground_step": _mean_std(ugv_travel_cost_per_ground_step),
            "hazard_exposure": _mean_std(hazard_exposure),
            "scout_to_confirm_latency_s": _mean_std(latency_values_s),
        },
    }


def _format_value(value: float, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def _format_duration(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    minutes, sec = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {sec:02d}s"


def _print_mean_std(label: str, mean: float, std: float, *, digits: int = 3) -> None:
    print(f"{label:<34} mean={_format_value(mean, digits)} std={_format_value(std, digits)}")


def _print_threshold_times(title: str, entries: dict[str, dict[str, float]]) -> None:
    print(title)
    for key in ("recall_050", "recall_080", "recall_090", "recall_100"):
        row = entries.get(key, {})
        threshold = 100.0 * float(row.get("threshold", 0.0))
        required = int(row.get("required_count", 0.0))
        print(
            f"  {required:>2} events ({threshold:>5.1f}% of max): "
            f"reached={_format_value(row.get('reached_fraction', float('nan')), 3)} "
            f"({int(row.get('reached_count', 0.0))}/"
            f"{int(row.get('eligible_count', 0.0))} eligible), "
            f"time={_format_value(row.get('mean_s', float('nan')), 1)}s "
            f"+/- {_format_value(row.get('ci95_s', float('nan')), 1)}s (95% CI)"
        )


def _print_core_joint_metrics(summary: dict[str, Any]) -> None:
    episodes = int(summary.get("episodes", 0.0))
    print("CORE JOINT METRICS")
    print("-" * 88)
    print(
        "success".ljust(34)
        + f"{int(summary.get('full_confirm_success_count', 0.0))}/{episodes} "
        + f"({_format_value(summary.get('full_confirm_success_percent', float('nan')), 1)}%)"
    )
    _print_mean_std(
        "scout recall",
        summary.get("mean_scout_recall", float("nan")),
        summary.get("std_scout_recall", float("nan")),
    )
    _print_mean_std(
        "confirm recall",
        summary.get("mean_confirm_recall", float("nan")),
        summary.get("std_confirm_recall", float("nan")),
    )
    _print_mean_std(
        "final confidence",
        summary.get("mean_final_confidence", float("nan")),
        summary.get("std_final_confidence", float("nan")),
    )
    _print_mean_std(
        "final coverage",
        summary.get("mean_final_coverage_fraction", float("nan")),
        summary.get("std_final_coverage_fraction", float("nan")),
    )


def _print_fast_summary(summary: dict[str, Any]) -> None:
    print("FAST JOINT DETAILS")
    print("-" * 88)
    _print_mean_std(
        "UAV path length (m)",
        summary.get("mean_uav_path_length_m", float("nan")),
        summary.get("std_uav_path_length_m", float("nan")),
        digits=1,
    )
    _print_mean_std(
        "UGV travel cost / UGV-step",
        summary.get("mean_ugv_travel_cost_per_ground_step", float("nan")),
        summary.get("std_ugv_travel_cost_per_ground_step", float("nan")),
        digits=4,
    )
    _print_mean_std(
        "hazard exposure",
        summary.get("mean_hazard_exposure", float("nan")),
        summary.get("std_hazard_exposure", float("nan")),
    )
    _print_mean_std(
        "scout-to-confirm latency (s)",
        summary.get("mean_scout_to_confirm_latency_s", float("nan")),
        summary.get("std_scout_to_confirm_latency_s", float("nan")),
        digits=1,
    )
    print(
        "scout-to-confirm latency count".ljust(34)
        + f"{int(summary.get('total_scout_to_confirm_latency_count', 0.0))} events "
        + f"(mean {summary.get('mean_scout_to_confirm_latency_count', float('nan')):.2f}/episode)"
    )
    print("-" * 88)
    _print_threshold_times("time to scout", summary.get("time_to_scout_s", {}))
    _print_threshold_times("time to confirmation", summary.get("time_to_confirm_s", {}))


def _plot(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 4, figsize=(18, 26), constrained_layout=True)
    axes = axes.ravel()

    def hist(ax, title: str, values: list[float], xlabel: str, xlim: tuple[float, float] | None = None):
        values = [float(v) for v in values if math.isfinite(float(v))]
        if values:
            ax.hist(values, bins=min(max(len(values) // 2, 5), 20), color="#4f7df3", alpha=0.78)
        else:
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        mean = _mean(values)
        median = float(np.nanmedian(values)) if values else float("nan")
        if math.isfinite(mean):
            ax.axvline(mean, color="#ef4444", label=f"mean {mean:.2f}")
        if math.isfinite(median):
            ax.axvline(median, color="#111827", linestyle="--", label=f"med {median:.2f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("episodes")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    def label_bars(ax, title: str, counts: dict[str, int]) -> None:
        if counts:
            labels = list(counts)
            values = [counts[label] for label in labels]
            y = np.arange(len(labels))
            ax.barh(y, values, color="#36a269", alpha=0.82)
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set_xlabel("episodes")
        else:
            ax.text(0.5, 0.5, "no labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    def survivor_count_hist(ax) -> None:
        values = [
            int(row.get("active_survivors", row.get("survivors", 0)))
            for row in rows
            if int(row.get("active_survivors", row.get("survivors", 0))) > 0
        ]
        if values:
            counts = Counter(values)
            success_counts = Counter(
                int(row.get("active_survivors", row.get("survivors", 0)))
                for row in rows
                if int(row.get("active_survivors", row.get("survivors", 0))) > 0
                and bool(row.get("full_confirm_success", row.get("overall_success", False)))
            )
            xs = sorted(counts)
            ys = [counts[x] for x in xs]
            success_ys = [success_counts.get(x, 0) for x in xs]
            ax.bar(xs, ys, color="#4f7df3", alpha=0.72, width=0.75, label="episodes")
            ax.bar(xs, success_ys, color="#22c55e", alpha=0.82, width=0.45, label="successful")
            ax.set_xticks(xs)
            ymax = max(ys) if ys else 1
            ax.set_ylim(0.0, ymax * 1.18 + 0.5)
            for x, total, successful in zip(xs, ys, success_ys):
                ax.text(
                    x,
                    total + max(ymax * 0.025, 0.25),
                    f"{successful}/{total}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#111827",
                )
            mean = _mean(values)
            median = float(np.nanmedian(values))
            if math.isfinite(mean):
                ax.axvline(mean, color="#ef4444", label=f"mean {mean:.2f}")
            if math.isfinite(median):
                ax.axvline(median, color="#111827", linestyle="--", label=f"med {median:.2f}")
        else:
            ax.text(0.5, 0.5, "no survivor counts", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Active Survivors / Success", fontsize=10)
        ax.set_xlabel("survivors / episode")
        ax.set_ylabel("episodes")
        ax.grid(axis="y", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=8)

    def heatmap(ax, title: str, key: str, cmap: str) -> None:
        points = [point for row in rows for point in row.get(key, []) if len(point) >= 2]
        if points:
            array = np.asarray(points, dtype=float)
            width = _mean([float(row.get("map_width_m", math.nan)) for row in rows])
            height = _mean([float(row.get("map_height_m", math.nan)) for row in rows])
            image = ax.hist2d(
                array[:, 0],
                array[:, 1],
                bins=12,
                range=[[-width / 2.0, width / 2.0], [-height / 2.0, height / 2.0]],
                cmap=cmap,
            )
            fig.colorbar(image[3], ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        else:
            ax.text(0.5, 0.5, "no positions", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)

    def event_plot(ax, title: str, key: str) -> None:
        event_bins = summary.get(key, [])
        centers = [
            0.5 * (float(row["start_fraction"]) + float(row["end_fraction"]))
            for row in event_bins
        ]
        widths = [
            0.72 * (float(row["end_fraction"]) - float(row["start_fraction"]))
            for row in event_bins
        ]
        ax.bar(
            centers,
            [float(row["mean_new_recall"]) for row in event_bins],
            width=widths,
            color="#80bfff",
            alpha=0.40,
            label="new recall/bin",
        )
        ax.plot(
            centers,
            [float(row["mean_cumulative_recall"]) for row in event_bins],
            marker="o",
            color="#d44a3a",
            label="cumulative mean",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("episode fraction")
        ax.set_ylabel("recall fraction")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    hist(axes[0], "Scout Recall", [row["scout_recall"] for row in rows], "fraction", (0, 1))
    hist(axes[1], "Confirm Recall", [row["confirm_recall"] for row in rows], "fraction", (0, 1))
    hist(axes[2], "Final Coverage", [row["final_coverage_fraction"] for row in rows], "covered fraction", (0, 1))
    hist(axes[3], "Final Confidence", [row["final_confidence_mean"] for row in rows], "mean confidence", (0, 1))
    heatmap(axes[4], "Survivor Heatmap", "survivor_positions_m", "YlOrRd")
    hist(axes[5], "UAV Movement / Step", [row["uav_movement_m_per_drone_step"] for row in rows], "m / UAV-step")
    event_plot(axes[6], "Survivor Discovery Over Time", "scout_time_bins")

    if rows and rows[0].get("time_bins"):
        xs = [b["episode_fraction"] for b in rows[0]["time_bins"]]

        def mean_series(key: str) -> list[float]:
            return [
                _mean([row["time_bins"][i][key] for row in rows if i < len(row["time_bins"])])
                for i in range(len(xs))
            ]

        ax = axes[7]
        for name, _key in UAV_REWARD_COMPONENTS:
            values = mean_series(f"uav_reward_{name}")
            if any(math.isfinite(value) and value > 1e-10 for value in values):
                ax.plot(xs, values, marker="o", label=name.replace("_", " "))
        ax.set_title("UAV Reward Scale (mean abs)", fontsize=10)
        ax.set_xlabel("episode fraction")
        ax.set_ylabel("abs reward / step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[10]
        line_new = ax.plot(xs, mean_series("uav_new_cells_per_step"), marker="o", label="new cells", color="#4f7cff")
        ax.set_xlabel("episode fraction")
        ax.set_ylabel("new cells / drone-step")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        fraction_lines = []
        for label, key, color in (
            ("overlap", "uav_overlap_fraction", "#36a269"),
            ("excess", "uav_excess_overlap_fraction", "#d44a3a"),
            ("edge", "uav_edge_step_fraction", "#20242c"),
            ("moving no new", "uav_moving_no_new_fraction", "#8a5cf6"),
        ):
            fraction_lines += ax2.plot(xs, mean_series(key), marker="o", label=label, color=color)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("fraction")
        lines = line_new + fraction_lines
        ax.legend(lines, [line.get_label() for line in lines], fontsize=7)
        ax.set_title("UAV Time-Bin Search Efficiency", fontsize=10)

        ax = axes[17]
        for name, _key in UGV_REWARD_COMPONENTS:
            values = mean_series(f"ugv_reward_{name}")
            if any(math.isfinite(value) and value > 1e-10 for value in values):
                ax.plot(xs, values, marker="o", label=name.replace("_", " "))
        ax.set_title("UGV Reward Scale (mean abs)", fontsize=10)
        ax.set_xlabel("episode fraction")
        ax.set_ylabel("abs reward / step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    uav_paths = summary.get("mean_uav_path_length_by_drone_m", [])
    axes[8].bar([f"d{i}" for i in range(len(uav_paths))], uav_paths, color="#36a269")
    axes[8].set_title("Per-Drone Path Length", fontsize=10)
    axes[8].set_ylabel("m")
    axes[8].grid(axis="y", alpha=0.25)
    heatmap(axes[9], "UAV Start Heatmap", "uav_start_positions_m", "Blues")
    label_bars(axes[11], "UAV Failure Labels", summary.get("uav_failure_label_counts", {}))

    event_plot(axes[12], "Survivor Confirmation Over Time", "confirm_time_bins")
    hist(axes[13], "UGV Final Distance", [row["ugv_final_pending_distance_m"] for row in rows], "m to unconfirmed survivor")
    hist(axes[14], "Time To Confirm", [row["avg_confirm_step"] for row in rows], "step")
    hist(axes[15], "UGV Path Length", [row["ugv_path_length_m"] for row in rows], "total m / episode")
    hist(axes[16], "UGV Speed", [row["ugv_speed_mps"] for row in rows], "m/s")
    heatmap(axes[18], "UGV Start Heatmap", "ugv_start_positions_m", "Greens")
    label_bars(axes[19], "UGV Failure Labels", summary.get("ugv_failure_label_counts", {}))
    survivor_count_hist(axes[20])
    for ax in axes[21:]:
        ax.axis("off")

    fig.suptitle(
        "Joint UAV+UGV HAPPO Diagnostics "
        f"(n={len(rows)}, success={summary['full_confirm_success_rate']:.2f})",
        fontsize=14,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", "--checkpoint", dest="checkpoint_dir", default=None)
    parser.add_argument("--joint-survivor-diagnostic", action="store_true",
                        help="Use joint diagnostic defaults when the checkpoint has no manifest.")
    parser.add_argument("--joint-schema-ugv-diagnostic", action="store_true",
                        help="Use 2-UGV delayed-knowledge joint-schema curriculum defaults.")
    parser.add_argument("--joint-diagnostic-ugvs", type=int, default=1)
    parser.add_argument("--n-drones", "--n-uavs", dest="n_drones", type=int, default=None,
                        help="Override UAV count. Default preserves the checkpoint manifest or uses 3.")
    parser.add_argument("--n-ugvs", "--n-ground", dest="n_ugvs", type=int, default=None,
                        help="Override UGV count. Default preserves the checkpoint manifest or uses --joint-diagnostic-ugvs.")
    parser.add_argument("--n-survivors", type=int, default=None,
                        help="Override survivor count. Default preserves the checkpoint manifest or uses 5.")
    parser.add_argument("--active-survivors-min", type=int, default=None,
                        help="Minimum active true survivors sampled per episode. Default preserves the checkpoint manifest.")
    parser.add_argument("--active-survivors-max", type=int, default=None,
                        help="Maximum active true survivors sampled per episode. Default preserves the checkpoint manifest.")
    parser.add_argument("--n-decoys", type=int, default=None,
                        help="Override fixed decoy slot count. Default preserves the checkpoint manifest.")
    parser.add_argument("--active-decoys-min", type=int, default=None,
                        help="Minimum active decoys sampled per episode. Default preserves the checkpoint manifest.")
    parser.add_argument("--active-decoys-max", type=int, default=None,
                        help="Maximum active decoys sampled per episode. Default preserves the checkpoint manifest.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1020)))
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--fire-grid-size", type=int, default=None,
                        help="Override checkpoint/default fire, coverage, and confidence grid size.")
    parser.add_argument("--drone-perception-mode",
                        choices=("rgb", "rgb_thermal", "rgb+thermal", "rgb-thermal"),
                        default=None,
                        help="Override abstract UAV perception mode. rgb_thermal changes only smoke quality.")
    parser.add_argument("--uav-fire-block-threshold", type=float, default=None,
                        help="If set, mark UAV local blocked-observation cells as blocked when "
                             "fire intensity >= this threshold. Omitted preserves checkpoint/default.")
    parser.add_argument("--uav-fire-footprint-penalty", type=float, default=None,
                        help="Override per-UAV active-fire footprint penalty scale.")
    parser.add_argument("--uav-fire-penalty-threshold", type=float, default=None,
                        help="Override active-fire threshold for --uav-fire-footprint-penalty.")
    parser.add_argument("--drone-safety-clearance-by-land-cover-m", type=float, nargs="+", default=None,
                        help="Override variable UAV safety margins by land cover: "
                             "road open brush forest rock [water]. Omitted preserves checkpoint/default.")
    parser.add_argument("--drone-safety-clearance-by-object-m", type=float, nargs=3, default=None,
                        metavar=("NONE", "TREE", "HOUSE"),
                        help="Override variable UAV safety margins by object: none tree house.")
    parser.add_argument("--drone-fire-safety-clearance-m", type=float, default=None,
                        help="Override UAV active-fire safety margin in meters.")
    parser.add_argument("--drone-smoke-safety-clearance-m", type=float, default=None,
                        help="Override UAV smoke-plume safety margin in meters.")
    parser.add_argument("--drone-smoke-clearance-threshold", type=float, default=None,
                        help="Override smoke-grid threshold for applying UAV smoke clearance.")
    parser.add_argument("--no-variable-drone-clearance", action="store_true",
                        help="Disable variable UAV clearance and use scalar checkpoint/default clearance.")
    parser.add_argument("--enable-fire", dest="enable_fire", action="store_true",
                        help="Override checkpoint/default settings and enable fire/smoke dynamics.")
    parser.add_argument("--disable-fire", dest="enable_fire", action="store_false",
                        help="Override checkpoint/default settings and disable fire/smoke dynamics.")
    parser.set_defaults(enable_fire=None)
    parser.add_argument("--comms-dropout", type=float, default=None,
                        help="Override checkpoint communication dropout probability in [0, 1].")
    parser.add_argument("--comms-dropout-mode", choices=("iid", "bursty"), default=None,
                        help="Override checkpoint communication dropout mode.")
    parser.add_argument("--comms-map-mode", choices=("per_agent", "per-agent"), default=None,
                        help="Use communication-gated per-agent coverage/confidence maps.")
    parser.add_argument("--comms-dropout-min-steps", type=int, default=None,
                        help="Override minimum outage duration for bursty communication dropout.")
    parser.add_argument("--comms-dropout-max-steps", type=int, default=None,
                        help="Override maximum outage duration for bursty communication dropout.")
    parser.add_argument("--uav-decision-grid", type=int, default=None,
                        help="Override UAV internal decision-map grid size. Default preserves checkpoint settings.")
    parser.add_argument("--uav-confidence-reward-grid", type=int, default=None,
                        help="Override UAV confidence reward/penalty/opportunity grid size.")
    parser.add_argument("--uav-frontier-global-grid", type=int, default=None,
                        help="Override the global leg grid size for local-global UAV frontier scoring.")
    parser.add_argument("--uav-coverage-reward-grid", type=int, default=None,
                        help="Override UAV binary coverage reward/overlap grid size.")
    parser.add_argument(
        "--ugv-target-assignment-mode",
        choices=(
            "nearest",
            "greedy",
            "greedy_sticky",
            "greedy-sticky",
            "greedy_sequence_sticky",
            "greedy-sequence-sticky",
            "route_cost_greedy",
            "route-cost-greedy",
            "route_cost_sticky",
            "route-cost-sticky",
            "route_sequence_sticky",
            "route-sequence-sticky",
            "route_cost_global",
            "route-cost-global",
        ),
        default=None,
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument(
        "--diagnostic-level",
        choices=("full", "fast"),
        default="full",
        help="fast prints compact metrics and skips plots; full keeps the detailed plot output.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--plots-output", default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.time_bins <= 0:
        parser.error("--time-bins must be positive")
    if args.joint_diagnostic_ugvs < 1:
        parser.error("--joint-diagnostic-ugvs must be positive")
    if args.n_drones is not None and args.n_drones < 1:
        parser.error("--n-drones must be positive")
    if args.n_ugvs is not None and args.n_ugvs < 1:
        parser.error("--n-ugvs must be positive")
    if args.n_survivors is not None and args.n_survivors < 1:
        parser.error("--n-survivors must be positive")
    if args.active_survivors_min is not None and args.active_survivors_min < 0:
        parser.error("--active-survivors-min must be nonnegative")
    if args.active_survivors_max is not None and args.active_survivors_max < 0:
        parser.error("--active-survivors-max must be nonnegative")
    if (
        args.active_survivors_min is not None
        and args.active_survivors_max is not None
        and args.active_survivors_max < args.active_survivors_min
    ):
        parser.error("--active-survivors-max must be >= --active-survivors-min")
    if args.n_decoys is not None and args.n_decoys < 0:
        parser.error("--n-decoys must be nonnegative")
    if args.uav_fire_block_threshold is not None and args.uav_fire_block_threshold > 1.0:
        parser.error("--uav-fire-block-threshold must be <= 1; use a negative value to disable")
    if args.uav_fire_footprint_penalty is not None and args.uav_fire_footprint_penalty < 0.0:
        parser.error("--uav-fire-footprint-penalty must be nonnegative")
    if args.uav_fire_penalty_threshold is not None and args.uav_fire_penalty_threshold > 1.0:
        parser.error("--uav-fire-penalty-threshold must be <= 1; use a negative value to disable")
    if (
        args.comms_dropout is not None
        and (
            not math.isfinite(args.comms_dropout)
            or not 0.0 <= args.comms_dropout <= 1.0
        )
    ):
        parser.error("--comms-dropout must be finite and between 0 and 1")
    if args.comms_dropout_min_steps is not None and args.comms_dropout_min_steps < 1:
        parser.error("--comms-dropout-min-steps must be >= 1")
    if args.comms_dropout_max_steps is not None and args.comms_dropout_max_steps < 1:
        parser.error("--comms-dropout-max-steps must be >= 1")
    if (
        args.comms_dropout_min_steps is not None
        and args.comms_dropout_max_steps is not None
        and args.comms_dropout_max_steps < args.comms_dropout_min_steps
    ):
        parser.error("--comms-dropout-max-steps must be >= --comms-dropout-min-steps")
    if (
        args.drone_safety_clearance_by_land_cover_m is not None
        and len(args.drone_safety_clearance_by_land_cover_m) not in {5, 6}
    ):
        parser.error("--drone-safety-clearance-by-land-cover-m must contain 5 or 6 values")
    if (
        args.drone_safety_clearance_by_land_cover_m is not None
        and any(v < 0.0 for v in args.drone_safety_clearance_by_land_cover_m)
    ):
        parser.error("--drone-safety-clearance-by-land-cover-m values must be nonnegative")
    if (
        args.drone_safety_clearance_by_object_m is not None
        and any(v < 0.0 for v in args.drone_safety_clearance_by_object_m)
    ):
        parser.error("--drone-safety-clearance-by-object-m values must be nonnegative")
    if args.drone_fire_safety_clearance_m is not None and args.drone_fire_safety_clearance_m < 0.0:
        parser.error("--drone-fire-safety-clearance-m must be nonnegative")
    if args.drone_smoke_safety_clearance_m is not None and args.drone_smoke_safety_clearance_m < 0.0:
        parser.error("--drone-smoke-safety-clearance-m must be nonnegative")
    if (
        args.drone_smoke_clearance_threshold is not None
        and not (0.0 <= args.drone_smoke_clearance_threshold <= 1.0)
    ):
        parser.error("--drone-smoke-clearance-threshold must be in [0, 1]")
    if args.active_decoys_min is not None and args.active_decoys_min < 0:
        parser.error("--active-decoys-min must be nonnegative")
    if args.active_decoys_max is not None and args.active_decoys_max < 0:
        parser.error("--active-decoys-max must be nonnegative")
    if (
        args.active_decoys_min is not None
        and args.active_decoys_max is not None
        and args.active_decoys_max < args.active_decoys_min
    ):
        parser.error("--active-decoys-max must be >= --active-decoys-min")
    if args.joint_survivor_diagnostic and args.joint_schema_ugv_diagnostic:
        parser.error("--joint-survivor-diagnostic and --joint-schema-ugv-diagnostic are mutually exclusive")
    if args.terrain_cache_path is not None and not Path(args.terrain_cache_path).is_file():
        parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")
    if args.fire_grid_size is not None and args.fire_grid_size < 2:
        parser.error("--fire-grid-size must be at least 2")
    for arg_name in (
        "uav_decision_grid",
        "uav_confidence_reward_grid",
        "uav_frontier_global_grid",
        "uav_coverage_reward_grid",
    ):
        value = getattr(args, arg_name)
        if value is not None and (value < 0 or value == 1):
            parser.error(f"--{arg_name.replace('_', '-')} must be 0 or at least 2")

    checkpoint_dir = _checkpoint_path(args.checkpoint_dir)
    scenario_kwargs = _scenario_kwargs(checkpoint_dir, args)
    timing_max_survivors = int(
        scenario_kwargs.get("active_survivors_max", scenario_kwargs.get("n_survivors", 0))
    )
    try:
        actor_file_indices = actor_file_indices_for_scenario(checkpoint_dir, scenario_kwargs)
    except ValueError as exc:
        parser.error(str(exc))
    policy = HappoPolicy.from_checkpoint(
        checkpoint_dir,
        deterministic=not args.stochastic,
        scenario_kwargs=scenario_kwargs,
        actor_file_indices=actor_file_indices,
    )
    expected_agents = int(scenario_kwargs.get("n_drones", 0)) + int(scenario_kwargs.get("n_ground", 0))
    if len(policy.actors) != expected_agents:
        parser.error(
            f"checkpoint has {len(policy.actors)} actors, but diagnostics scenario has "
            f"{expected_agents} agents"
        )

    print(f"checkpoint: {checkpoint_dir}")
    print(
        "scenario: "
        f"{scenario_kwargs.get('n_drones')} UAVs, "
        f"{scenario_kwargs.get('n_ground')} UGVs, "
        f"{scenario_kwargs.get('n_survivors')} survivor slots "
        f"(active {scenario_kwargs.get('active_survivors_min', scenario_kwargs.get('n_survivors'))}"
        f"..{scenario_kwargs.get('active_survivors_max', scenario_kwargs.get('n_survivors'))}), "
        f"{scenario_kwargs.get('n_decoys', 0)} decoy slots "
        f"(active {scenario_kwargs.get('active_decoys_min', scenario_kwargs.get('n_decoys', 0))}"
        f"..{scenario_kwargs.get('active_decoys_max', scenario_kwargs.get('n_decoys', 0))}), "
        f"planner={scenario_kwargs.get('ugv_planner_hint')}, "
        f"assignment={scenario_kwargs.get('ugv_target_assignment_mode')}, "
        f"drone_perception={scenario_kwargs.get('drone_perception_mode', 'rgb')}, "
        f"uav_fire_block_threshold={scenario_kwargs.get('uav_fire_block_threshold', -1.0)}, "
        f"uav_fire_penalty={scenario_kwargs.get('r_uav_fire_footprint', 0.0)}, "
        f"uav_fire_penalty_threshold={scenario_kwargs.get('uav_fire_penalty_threshold', 0.6)}"
    )
    print(
        "communications: "
        f"dropout={scenario_kwargs.get('comms_dropout', 0.0)} "
        f"mode={scenario_kwargs.get('comms_dropout_mode', COMMS_DROPOUT_MODE)} "
        f"maps={scenario_kwargs.get('comms_map_mode', 'per_agent')} "
        f"burst_steps={scenario_kwargs.get('comms_dropout_min_steps', 5)}"
        f"..{scenario_kwargs.get('comms_dropout_max_steps', 15)}"
    )
    print(f"steps: {args.steps}")
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    total_rollout_steps = int(args.steps) * len(args.seeds)
    print(f"planned rollout: {len(args.seeds)} seeds x {args.steps} steps")
    print("-" * 88)

    json_path = Path(args.json_output) if args.json_output else None
    if json_path is not None:
        print(f"partial JSON checkpoint: {partial_json_path(json_path)}")

    rows = []
    rollout_started_at = time.perf_counter()
    for seed_index, seed in enumerate(args.seeds, start=1):
        rows.append(run_rollout(policy, scenario_kwargs, seed, time_bins=args.time_bins))
        if json_path is not None:
            partial_summary = summarize(
                rows,
                bins=args.time_bins,
                max_survivors=timing_max_survivors,
            )
            write_partial_json(
                json_path,
                {
                    "checkpoint": str(checkpoint_dir),
                    "deterministic": not args.stochastic,
                    "scenario": scenario_kwargs,
                    "summary": partial_summary,
                    "rows": rows,
                },
                completed_rollouts=seed_index,
                total_rollouts=len(args.seeds),
            )
        elapsed_rollout_s = time.perf_counter() - rollout_started_at
        completed_steps = int(args.steps) * seed_index
        rollout_steps_per_second = completed_steps / max(elapsed_rollout_s, 1e-9)
        remaining_steps = max(total_rollout_steps - completed_steps, 0)
        eta_s = remaining_steps / rollout_steps_per_second if rollout_steps_per_second > 0.0 else float("nan")
        if seed_index == 1:
            print(f"ETA {_format_duration(eta_s)}", flush=True)
        print(f"progress: {seed_index}/{len(args.seeds)} seeds", flush=True)
    summary = summarize(
        rows,
        bins=args.time_bins,
        max_survivors=timing_max_survivors,
    )
    if args.diagnostic_level != "fast":
        for row in rows:
            print(
                f"seed {row['seed']:>4}: "
                f"scout={row['scouted']}/{row['survivors']} "
                f"confirm={row['confirmed']}/{row['survivors']} "
                f"success={int(row['full_confirm_success'])} "
                f"cov={row['final_coverage_fraction']:.3f} "
                f"conf={row['final_confidence_mean']:.3f} "
                f"lat={row['avg_scout_to_confirm_latency_steps']:.1f} "
                f"uav_move={row['uav_movement_m_per_drone_step']:.2f}m/step "
                f"uav_path={row['uav_path_length_m']:.1f}m "
                f"ugv_path={row['ugv_path_length_m']:.1f}m "
                f"ugv_speed={row['ugv_speed_mps']:.2f}m/s "
                f"ugv_final={row['ugv_final_pending_distance_m']:.1f}m "
                f"pending={row['pending_target_time_fraction']:.2f}"
            )
        print("-" * 88)
        print(
            "means: "
            f"scout_recall={summary['mean_scout_recall']:.3f} "
            f"confirm_recall={summary['mean_confirm_recall']:.3f} "
            f"scout_auc={summary['mean_scout_auc']:.3f} "
            f"confirm_auc={summary['mean_confirm_auc']:.3f} "
            f"coverage_auc={summary['mean_coverage_auc']:.3f} "
            f"confidence_auc={summary['mean_confidence_auc']:.3f} "
            f"success={summary['full_confirm_success_rate']:.3f} "
            f"coverage={summary['mean_final_coverage_fraction']:.3f} "
            f"confidence={summary['mean_final_confidence']:.3f} "
            f"uav_move={summary['mean_uav_movement_m_per_drone_step']:.2f}m/step "
            f"ugv_speed={summary['mean_ugv_speed_mps']:.2f}m/s "
            f"ugv_final={summary['mean_ugv_final_pending_distance_m']:.1f}m "
            f"latency={summary['mean_scout_to_confirm_latency_steps']:.1f} steps"
        )

    payload = {
        "checkpoint": str(checkpoint_dir),
        "deterministic": not args.stochastic,
        "scenario": scenario_kwargs,
        "summary": summary,
        "rows": rows,
    }
    if json_path is not None:
        write_final_json(
            json_path,
            payload,
            completed_rollouts=len(rows),
            total_rollouts=len(args.seeds),
        )
        print(f"wrote JSON diagnostics: {json_path}")
    if args.plots_output and args.diagnostic_level == "fast":
        print("fast diagnostic level skips plot generation; ignoring --plots-output")
    elif args.plots_output:
        _plot(rows, summary, Path(args.plots_output))
    _print_core_joint_metrics(summary)
    if args.diagnostic_level == "fast":
        _print_fast_summary(summary)
    elapsed_s = time.perf_counter() - started_at
    print(f"Diagnostics complete in {elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
