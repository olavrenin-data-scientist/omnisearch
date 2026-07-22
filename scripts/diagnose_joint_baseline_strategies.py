#!/usr/bin/env python3
"""Diagnose one joint UAV search plus UGV confirmation strategy.

This is the heuristic counterpart to ``diagnose_joint_happo.py``.  It runs the
full joint task end to end: UAVs must scout unknown survivors, and UGVs can
confirm only survivors that have become known through scouting/communication.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import BASELINES, UGV_CONTROLLER_CHOICES, get_baseline
from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import (
    HappoPolicy,
    actor_file_indices_for_scenario,
    find_latest_happo_checkpoint,
)
from envs.wildfire_search import WildfireSearchScenario
from scripts.train_happo_smoke import (
    DEFAULT_JOINT_DIAG_DRONES,
    DEFAULT_JOINT_DIAG_UGVS,
    build_args,
)


AVAILABLE_STRATEGIES = tuple(BASELINES.keys())
DEFAULT_STRATEGY = "lawnmower"
STRATEGY_ALIASES = {
    **{name: name for name in AVAILABLE_STRATEGIES},
    # Backward-compatible spellings from the old joint diagnostic name.
    "lawnmower_astar": "lawnmower",
    "ant_colony_astar": "ant_colony",
}
RECALL_TIME_THRESHOLDS = (0.50, 0.80, 0.90, 1.00)


@dataclass(frozen=True)
class StrategySpec:
    label: str
    name: str
    checkpoint_dir: Path | None = None


def parse_strategy_specs(
    strategies: list[str] | None,
    *,
    happo_checkpoint: str | Path | None = None,
) -> list[StrategySpec]:
    raw = strategies or [DEFAULT_STRATEGY]
    if len(raw) != 1:
        raise ValueError("diagnose_joint_baseline_strategies.py now runs exactly one strategy at a time")
    expanded = list(raw)

    specs: list[StrategySpec] = []
    used: set[str] = set()
    for token in expanded:
        normalized = token.replace("-", "_")
        if normalized == "happo" or token.startswith("happo:"):
            checkpoint = token.split(":", 1)[1] if token.startswith("happo:") else happo_checkpoint
            specs.append(StrategySpec(
                label=_unique_label("happo", used),
                name="happo",
                checkpoint_dir=_resolve_happo_checkpoint(checkpoint),
            ))
            continue
        name = STRATEGY_ALIASES.get(normalized)
        if name is None:
            options = ", ".join((*AVAILABLE_STRATEGIES, "happo", "happo:/path/to/models"))
            raise ValueError(f"unknown strategy {token!r}; available: {options}")
        specs.append(StrategySpec(label=_unique_label(name, used), name=name))
    if not specs:
        raise ValueError("at least one strategy is required")
    return specs


def _unique_label(label: str, used: set[str]) -> str:
    if label not in used:
        used.add(label)
        return label
    idx = 2
    while f"{label}_{idx}" in used:
        idx += 1
    unique = f"{label}_{idx}"
    used.add(unique)
    return unique


def _resolve_happo_checkpoint(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    return find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _joint_defaults(joint_diagnostic_ugvs: int) -> dict[str, Any]:
    _, _algo_args, env_args = build_args(
        num_env_steps=100,
        episode_length=100,
        seed=1,
        comms_dropout=0.0,
        entropy_coef=0.01,
        exp_name="joint_baseline_strategy_defaults",
        joint_survivor_diagnostic=True,
        joint_diagnostic_ugvs=joint_diagnostic_ugvs,
    )
    return copy.deepcopy(env_args["scenario_kwargs"])


def _scenario_kwargs_from_checkpoint(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {}
    manifest = load_training_manifest(checkpoint_dir)
    if manifest is None:
        return {}
    scenario_kwargs = manifest.get("env_args", {}).get("scenario_kwargs", {})
    if not isinstance(scenario_kwargs, dict):
        return {}
    return copy.deepcopy(scenario_kwargs)


def _scenario_checkpoint_from_specs(
    args: argparse.Namespace,
    specs: list[StrategySpec],
) -> Path | None:
    if args.happo_checkpoint:
        return _resolve_happo_checkpoint(args.happo_checkpoint)
    for spec in specs:
        if spec.checkpoint_dir is not None:
            return spec.checkpoint_dir
    return None


def build_scenario_kwargs(
    args: argparse.Namespace,
    checkpoint_dir: Path | None = None,
    specs: list[StrategySpec] | None = None,
) -> dict[str, Any]:
    scenario_kwargs = _scenario_kwargs_from_checkpoint(checkpoint_dir)
    loaded_from_checkpoint = bool(scenario_kwargs)
    if not loaded_from_checkpoint:
        scenario_kwargs = _joint_defaults(int(args.joint_diagnostic_ugvs))
        if (
            args.ugv_target_assignment_mode is None
            and specs is not None
            and (
                any(spec.name == "matched_heuristic" for spec in specs)
                or args.baseline_ugv_controller.replace("-", "_") != "native"
            )
        ):
            scenario_kwargs["ugv_target_assignment_mode"] = "greedy_sticky"
    n_ugvs = getattr(args, "n_ugvs", None)
    default_steps = int(scenario_kwargs.get("max_steps", 300)) if loaded_from_checkpoint else 300
    scenario_kwargs["max_steps"] = int(
        args.steps if args.steps is not None else default_steps
    )
    scenario_kwargs.setdefault("n_drones", DEFAULT_JOINT_DIAG_DRONES)
    scenario_kwargs.setdefault("n_ground", int(args.joint_diagnostic_ugvs))
    scenario_kwargs.setdefault("n_survivors", 5)
    scenario_kwargs.setdefault("known_survivors_at_reset", False)
    scenario_kwargs.setdefault("delayed_survivor_knowledge", False)
    scenario_kwargs.setdefault("drone_can_confirm", False)
    scenario_kwargs.setdefault("comms_dropout", 0.0)
    scenario_kwargs.setdefault("comms_dropout_mode", "iid")
    scenario_kwargs.setdefault("comms_map_mode", "global")
    scenario_kwargs.setdefault("comms_dropout_min_steps", 5)
    scenario_kwargs.setdefault("comms_dropout_max_steps", 15)
    scenario_kwargs["uav_confidence_diagnostics"] = True
    if args.n_drones is not None:
        scenario_kwargs["n_drones"] = int(args.n_drones)
        scenario_kwargs["obs_schema_n_drones"] = int(args.n_drones)
    if n_ugvs is not None:
        scenario_kwargs["n_ground"] = int(n_ugvs)
        scenario_kwargs["obs_schema_n_ground"] = int(n_ugvs)
    if args.n_survivors is not None:
        scenario_kwargs["n_survivors"] = int(args.n_survivors)
        scenario_kwargs["obs_schema_n_survivors"] = int(args.n_survivors)
    if args.active_survivors_min is not None:
        scenario_kwargs["active_survivors_min"] = int(args.active_survivors_min)
    if args.active_survivors_max is not None:
        scenario_kwargs["active_survivors_max"] = int(args.active_survivors_max)
    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = str(args.terrain_cache_path)
    if args.enable_fire:
        scenario_kwargs["disable_fire"] = False
    if args.disable_fire:
        scenario_kwargs["disable_fire"] = True

    comms_overrides = {
        "comms_dropout": getattr(args, "comms_dropout", None),
        "comms_dropout_mode": getattr(args, "comms_dropout_mode", None),
        "comms_map_mode": getattr(args, "comms_map_mode", None),
        "comms_dropout_min_steps": getattr(args, "comms_dropout_min_steps", None),
        "comms_dropout_max_steps": getattr(args, "comms_dropout_max_steps", None),
    }
    for key, value in comms_overrides.items():
        if value is not None:
            scenario_kwargs[key] = value

    overrides = {
        "ugv_target_assignment_mode": (
            args.ugv_target_assignment_mode.replace("-", "_")
            if args.ugv_target_assignment_mode is not None
            else None
        ),
        "ugv_planner_fire_mode": args.ugv_planner_fire_mode,
        "ugv_planner_fire_replan_policy": args.ugv_planner_fire_replan_policy,
        "ugv_planner_fire_replan_interval_steps": args.ugv_planner_fire_replan_interval_steps,
        "ugv_planner_fire_cost": args.ugv_planner_fire_cost,
        "ugv_planner_fire_block_threshold": args.ugv_planner_fire_block_threshold,
        "ugv_planner_smoke_cost": args.ugv_planner_smoke_cost,
        "ugv_planner_smolder_cost": args.ugv_planner_smolder_cost,
        "ugv_planner_fire_buffer_m": args.ugv_planner_fire_buffer_m,
        "ugv_planner_fire_buffer_cost": args.ugv_planner_fire_buffer_cost,
        "ugv_planner_land_cover_costs": (
            tuple(args.ugv_planner_land_cover_costs)
            if args.ugv_planner_land_cover_costs is not None
            else None
        ),
    }
    for key, value in overrides.items():
        if value is not None:
            scenario_kwargs[key] = value
    return scenario_kwargs


def make_policy(
    spec: StrategySpec,
    env: Any,
    scenario_kwargs: dict[str, Any],
    *,
    baseline_ugv_controller: str = "native",
    happo_cache: dict[tuple[Path, bool, tuple[int, ...]], HappoPolicy] | None = None,
    deterministic_happo: bool = True,
) -> Callable[[Any], list[torch.Tensor]]:
    if spec.name in BASELINES:
        return get_baseline(
            spec.name,
            env,
            ugv_controller_mode=baseline_ugv_controller,
        )
    if spec.name == "happo":
        if spec.checkpoint_dir is None:
            raise ValueError("HAPPO strategy requires a checkpoint directory")
        actor_file_indices = actor_file_indices_for_scenario(spec.checkpoint_dir, scenario_kwargs)
        key = (
            spec.checkpoint_dir,
            deterministic_happo,
            tuple(actor_file_indices or ()),
        )
        if happo_cache is None:
            return HappoPolicy.from_checkpoint(
                spec.checkpoint_dir,
                deterministic=deterministic_happo,
                scenario_kwargs=scenario_kwargs,
                actor_file_indices=actor_file_indices,
            )
        if key not in happo_cache:
            happo_cache[key] = HappoPolicy.from_checkpoint(
                spec.checkpoint_dir,
                deterministic=deterministic_happo,
                scenario_kwargs=scenario_kwargs,
                actor_file_indices=actor_file_indices,
            )
        policy = happo_cache[key]
        expected_agents = len(env.agents)
        if len(policy.actors) != expected_agents:
            raise ValueError(
                f"HAPPO checkpoint has {len(policy.actors)} actors, "
                f"but the joint baseline scenario has {expected_agents} agents"
            )
        return policy
    raise ValueError(f"unsupported strategy {spec.name!r}")


def _positions(scenario: WildfireSearchScenario) -> torch.Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def _active_survivor_mask_for_env(
    scenario: WildfireSearchScenario,
    env_index: int = 0,
) -> np.ndarray:
    slots = int(getattr(scenario, "n_survivors", 0))
    active = getattr(scenario, "active_survivors", None)
    if active is None:
        return np.ones(slots, dtype=bool)
    return active[env_index].detach().cpu().numpy().astype(bool)


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


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else float("nan")


def _bin_index(step: int, steps: int, bins: int) -> int:
    return min(int(step / max(steps, 1) * max(bins, 1)), max(bins, 1) - 1)


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


def _recall_threshold_time_stats(
    rows: list[dict[str, Any]],
    *,
    key: str,
    threshold: float,
) -> dict[str, float]:
    times_s: list[float] = []
    for row in rows:
        survivors = int(row.get("survivors", 0))
        if survivors <= 0:
            times_s.append(0.0)
            continue
        required = max(1, int(math.ceil(float(threshold) * survivors - 1e-9)))
        event_steps = sorted(
            float(step)
            for step in row.get(key, [])
            if step is not None and math.isfinite(float(step))
        )
        if len(event_steps) < required:
            continue
        step_seconds = max(float(row.get("step_seconds", 1.0)), 1e-9)
        times_s.append(event_steps[required - 1] * step_seconds)
    total = max(len(rows), 1)
    return {
        "threshold": float(threshold),
        "reached_count": float(len(times_s)),
        "reached_fraction": float(len(times_s) / total),
        "mean_s": _mean(times_s),
        "std_s": _std(times_s),
    }


def _threshold_time_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    return {
        f"recall_{int(round(threshold * 100)):03d}": _recall_threshold_time_stats(
            rows,
            key=key,
            threshold=threshold,
        )
        for threshold in RECALL_TIME_THRESHOLDS
    }


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
            episode_steps = max(int(row.get("episode_steps", row.get("max_steps", 0))), 1)
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
        output.append({
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "mean_new_recall": _mean(new_recall),
            "mean_cumulative_recall": _mean(cumulative),
            "median_cumulative_recall": (
                float(np.median(cumulative)) if cumulative else float("nan")
            ),
        })
    return output


def _new_time_bins(count: int) -> list[dict[str, float]]:
    return [
        {
            "count": 0.0,
            "new_scouts": 0.0,
            "new_oracle_reveals": 0.0,
            "new_confirmations": 0.0,
            "pending": 0.0,
            "uav_new_cells": 0.0,
            "uav_confidence_gain": 0.0,
            "uav_displacement_m": 0.0,
            "uav_overlap": 0.0,
            "uav_excess_overlap": 0.0,
            "uav_edge_step": 0.0,
            "uav_moving_no_new": 0.0,
            "ugv_displacement_m": 0.0,
            "ugv_progress_m": 0.0,
            "ugv_route_active": 0.0,
            "duplicate_assignment": 0.0,
            "assignment_switches": 0.0,
        }
        for _ in range(max(int(count), 1))
    ]


def run_rollout(
    spec: StrategySpec,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    time_bins: int = 5,
    happo_cache: dict[tuple[Path, bool, tuple[int, ...]], HappoPolicy] | None = None,
    stochastic_happo: bool = False,
    baseline_ugv_controller: str = "native",
) -> dict[str, Any]:
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=int(seed),
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset(seed=int(seed))
    scenario = env.scenario
    policy = make_policy(
        spec,
        env,
        scenario_kwargs,
        baseline_ugv_controller=baseline_ugv_controller,
        happo_cache=happo_cache,
        deterministic_happo=not stochastic_happo,
    )
    if hasattr(policy, "reset"):
        policy.reset()

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
    pending_counts: list[float] = []
    duplicate_assignment: list[float] = []
    assignment_switches: list[float] = []
    uav_overlap_values: list[float] = []
    uav_excess_overlap_values: list[float] = []
    uav_edge_values: list[float] = []
    uav_moving_no_new_values: list[float] = []
    uav_fire_footprint_values: list[float] = []
    ugv_fire_exposure_values: list[float] = []
    hazard_exposure_values: list[float] = []
    ugv_travel_cost_values: list[float] = []
    scout_auc_sum = 0.0
    confirm_auc_sum = 0.0
    coverage_auc_sum = 0.0
    confidence_auc_sum = 0.0
    auc_steps = 0
    time_series = _new_time_bins(time_bins)

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
        ugv_travel_cost_values.append(abs(_to_float(info.get("cost/ugv_travel"))))

        bucket = time_series[_bin_index(step, max_steps, len(time_series))]
        bucket["count"] += 1.0
        bucket["new_scouts"] += _to_float(info.get("mission/new_scouts"))
        bucket["new_oracle_reveals"] += _to_float(info.get("mission/new_oracle_reveals"))
        bucket["new_confirmations"] += _to_float(info.get("mission/new_confirmations"))
        bucket["pending"] += pending
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

        bucket["uav_new_cells"] += mean_new_cells
        bucket["uav_confidence_gain"] += _to_float(info.get("diagnostic/uav_confidence_gain"))
        bucket["uav_displacement_m"] += mean_uav_displacement
        bucket["uav_overlap"] += mean_overlap
        bucket["uav_excess_overlap"] += mean_excess_overlap
        bucket["uav_edge_step"] += mean_edge
        bucket["uav_moving_no_new"] += mean_moving_no_new
        bucket["ugv_displacement_m"] += (
            float(displacement_m[n_drones:].mean()) if n_ground else 0.0
        )
        bucket["ugv_progress_m"] += _to_float(info.get("diagnostic/ugv_global_route_progress_m"))
        bucket["ugv_route_active"] += _to_float(info.get("diagnostic/ugv_global_route_active"))
        bucket["duplicate_assignment"] += duplicate
        bucket["assignment_switches"] += switches

    active_scout_steps = [first_scout_steps[idx] for idx in active_survivor_indices]
    active_confirm_steps = [first_confirm_steps[idx] for idx in active_survivor_indices]
    scout_count = sum(step is not None for step in active_scout_steps)
    confirm_count = sum(step is not None for step in active_confirm_steps)
    scout_to_confirm_latencies = [
        float(confirm_step - scout_step)
        for scout_step, confirm_step in zip(first_scout_steps, first_confirm_steps)
        if scout_step is not None and confirm_step is not None and confirm_step >= scout_step
    ]
    time_bin_rows = []
    for idx, bucket in enumerate(time_series):
        count = max(bucket["count"], 1.0)
        time_bin_rows.append({
            "episode_fraction": (idx + 0.5) / len(time_series),
            "new_scouts_per_step": bucket["new_scouts"] / count,
            "new_oracle_reveals_per_step": bucket["new_oracle_reveals"] / count,
            "new_confirmations_per_step": bucket["new_confirmations"] / count,
            "pending_known_survivors": bucket["pending"] / count,
            "uav_new_cells_per_step": bucket["uav_new_cells"] / count,
            "uav_confidence_gain_per_step": bucket["uav_confidence_gain"] / count,
            "uav_displacement_m_per_step": bucket["uav_displacement_m"] / count,
            "uav_displacement_m_per_uav_step": bucket["uav_displacement_m"] / count,
            "uav_overlap_fraction": bucket["uav_overlap"] / count,
            "uav_excess_overlap_fraction": bucket["uav_excess_overlap"] / count,
            "uav_edge_step_fraction": bucket["uav_edge_step"] / count,
            "uav_moving_no_new_fraction": bucket["uav_moving_no_new"] / count,
            "ugv_displacement_m_per_step": bucket["ugv_displacement_m"] / count,
            "ugv_displacement_m_per_ugv_step": bucket["ugv_displacement_m"] / count,
            "ugv_route_progress_m_per_step": bucket["ugv_progress_m"] / count,
            "ugv_route_active_fraction": bucket["ugv_route_active"] / max(count * n_ground, 1.0),
            "duplicate_assignment_fraction": bucket["duplicate_assignment"] / count,
            "assignment_switches_per_step": bucket["assignment_switches"] / count,
        })

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
        if float(scenario.coverage_grid[0].float().mean().detach().cpu().item()) < 0.70:
            uav_failure_labels.append("low_coverage")
        if float(scenario.uav_confidence_grid[0].float().mean().detach().cpu().item()) < 0.75:
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

    row = {
        "strategy": spec.label,
        "strategy_name": spec.name,
        "checkpoint_dir": None if spec.checkpoint_dir is None else str(spec.checkpoint_dir),
        "baseline_ugv_controller": baseline_ugv_controller,
        "seed": int(seed),
        "max_steps": max_steps,
        "episode_steps": max_steps,
        "step_seconds": step_seconds,
        "survivors": n_active_survivors,
        "active_survivors": n_active_survivors,
        "survivor_slots": survivor_slots,
        "scouted": int(scout_count),
        "confirmed": int(confirm_count),
        "scout_recall": float(scout_count / max(n_active_survivors, 1)),
        "confirm_recall": float(confirm_count / max(n_active_survivors, 1)),
        "scout_auc": float(scout_auc_sum / max(auc_steps, 1)),
        "confirm_auc": float(confirm_auc_sum / max(auc_steps, 1)),
        "coverage_auc": float(coverage_auc_sum / max(auc_steps, 1)),
        "confidence_auc": float(confidence_auc_sum / max(auc_steps, 1)),
        "full_confirm_success": bool(confirm_count == n_active_survivors),
        "overall_success": bool(confirm_count == n_active_survivors),
        "first_scout_steps": first_scout_steps,
        "first_confirm_steps": first_confirm_steps,
        "first_scout_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_scout_steps
        ],
        "first_confirm_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_confirm_steps
        ],
        "scout_to_confirm_latencies_steps": scout_to_confirm_latencies,
        "scout_to_confirm_latency_count": int(len(scout_to_confirm_latencies)),
        "scout_to_confirm_latencies_s": [
            float(value * step_seconds) for value in scout_to_confirm_latencies
        ],
        "avg_scout_to_confirm_latency_steps": _mean(scout_to_confirm_latencies),
        "median_scout_to_confirm_latency_steps": _median(scout_to_confirm_latencies),
        "avg_scout_to_confirm_latency_s": _mean(scout_to_confirm_latencies) * step_seconds,
        "final_coverage_fraction": float(scenario.coverage_grid[0].float().mean().detach().cpu().item()),
        "final_confidence_mean": float(scenario.uav_confidence_grid[0].float().mean().detach().cpu().item()),
        "uav_path_length_m": float(path_lengths_m[:n_drones].sum()) if n_drones else 0.0,
        "uav_path_length_by_agent_m": [float(value) for value in path_lengths_m[:n_drones]],
        "uav_path_length_by_drone_m": [float(value) for value in uav_path_lengths],
        "uav_movement_m_per_uav_step": (
            float(path_lengths_m[:n_drones].sum() / max(n_drones * max_steps, 1))
            if n_drones else 0.0
        ),
        "uav_movement_m_per_drone_step": (
            float(path_lengths_m[:n_drones].sum() / max(n_drones * max_steps, 1))
            if n_drones else 0.0
        ),
        "ugv_path_length_m": float(path_lengths_m[n_drones:].sum()) if n_ground else 0.0,
        "ugv_path_length_by_agent_m": [float(value) for value in path_lengths_m[n_drones:]],
        "ugv_path_length_by_ground_m": [float(value) for value in ugv_path_lengths],
        "ugv_movement_m_per_ugv_step": (
            float(path_lengths_m[n_drones:].sum() / max(n_ground * max_steps, 1))
            if n_ground else 0.0
        ),
        "ugv_movement_m_per_ground_step": (
            float(ugv_path_lengths.sum() / max(n_ground * max_steps, 1))
            if n_ground else 0.0
        ),
        "ugv_speed_mps": ugv_speed_mps,
        "ugv_travel_cost_per_step": _mean(ugv_travel_cost_values),
        "ugv_travel_cost_per_ground_step": (
            _mean([value / max(n_ground, 1) for value in ugv_travel_cost_values])
            if n_ground else 0.0
        ),
        "ugv_travel_cost_total": float(np.sum(ugv_travel_cost_values)),
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
        "path_length_by_agent_m": [float(value) for value in path_lengths_m],
        "pending_target_time_mean": _mean(pending_counts),
        "pending_target_time_fraction": float(
            np.count_nonzero(np.asarray(pending_counts) > 0) / max(len(pending_counts), 1)
        ),
        "duplicate_ugv_assignment_rate": _mean(duplicate_assignment),
        "ugv_assignment_switches_per_episode": float(np.sum(assignment_switches)),
        "time_bins": time_bin_rows,
    }
    close = getattr(env, "close", None)
    if close is not None:
        close()
    return row


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


def summarize(rows: list[dict[str, Any]], bins: int = 5) -> dict[str, Any]:
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
        "strategy": rows[0].get("strategy") if rows else None,
        "strategy_name": rows[0].get("strategy_name") if rows else None,
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
        "full_confirm_success_percent": 100.0 * success_count / max(len(rows), 1),
        "mean_final_coverage_fraction": _mean([row["final_coverage_fraction"] for row in rows]),
        "std_final_coverage_fraction": _std([row["final_coverage_fraction"] for row in rows]),
        "mean_final_confidence": _mean([row["final_confidence_mean"] for row in rows]),
        "std_final_confidence": _std([row["final_confidence_mean"] for row in rows]),
        "mean_scout_to_confirm_latency_steps": _mean([
            row["avg_scout_to_confirm_latency_steps"] for row in rows
        ]),
        "mean_scout_to_confirm_latency_s": _mean([
            row["avg_scout_to_confirm_latency_s"] for row in rows
        ]),
        "std_scout_to_confirm_latency_s": _std(latency_values_s),
        "mean_scout_to_confirm_latency_count": _mean([
            float(row["scout_to_confirm_latency_count"]) for row in rows
        ]),
        "total_scout_to_confirm_latency_count": float(
            sum(int(row["scout_to_confirm_latency_count"]) for row in rows)
        ),
        "median_scout_to_confirm_latency_steps": _median([
            latency
            for row in rows
            for latency in row["scout_to_confirm_latencies_steps"]
        ]),
        "mean_uav_path_length_m": _mean([row["uav_path_length_m"] for row in rows]),
        "std_uav_path_length_m": _std([row["uav_path_length_m"] for row in rows]),
        "mean_uav_movement_m_per_uav_step": _mean([
            row["uav_movement_m_per_uav_step"] for row in rows
        ]),
        "mean_uav_movement_m_per_drone_step": _mean([
            row["uav_movement_m_per_drone_step"] for row in rows
        ]),
        "mean_ugv_path_length_m": _mean([row["ugv_path_length_m"] for row in rows]),
        "std_ugv_path_length_m": _std([row["ugv_path_length_m"] for row in rows]),
        "mean_ugv_movement_m_per_ugv_step": _mean([
            row["ugv_movement_m_per_ugv_step"] for row in rows
        ]),
        "mean_ugv_movement_m_per_ground_step": _mean([
            row["ugv_movement_m_per_ground_step"] for row in rows
        ]),
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
        "time_to_scout_s": _threshold_time_summary(rows, "first_scout_steps"),
        "time_to_confirm_s": _threshold_time_summary(rows, "first_confirm_steps"),
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
        "time_bins": _summarize_time_bins(rows),
    }


def _summarize_time_bins(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not rows:
        return []
    bins = len(rows[0].get("time_bins", []))
    if not bins:
        return []
    keys = [key for key in rows[0]["time_bins"][0] if key != "episode_fraction"]
    out = []
    for idx in range(bins):
        item = {"episode_fraction": rows[0]["time_bins"][idx]["episode_fraction"]}
        for key in keys:
            item[key] = _mean([
                row["time_bins"][idx].get(key, math.nan)
                for row in rows
                if idx < len(row.get("time_bins", []))
            ])
        out.append(item)
    return out


def write_plots(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 4, figsize=(18, 26), constrained_layout=True)
    axes = axes.ravel()

    def hist(
        ax,
        title: str,
        values: list[float],
        xlabel: str,
        xlim: tuple[float, float] | None = None,
    ) -> None:
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
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
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
    axes[7].axis("off")

    uav_paths = summary.get("mean_uav_path_length_by_drone_m", [])
    axes[8].bar([f"d{i}" for i in range(len(uav_paths))], uav_paths, color="#36a269")
    axes[8].set_title("Per-Drone Path Length", fontsize=10)
    axes[8].set_ylabel("m")
    axes[8].grid(axis="y", alpha=0.25)
    heatmap(axes[9], "UAV Start Heatmap", "uav_start_positions_m", "Blues")

    if rows and rows[0].get("time_bins"):
        xs = [b["episode_fraction"] for b in rows[0]["time_bins"]]

        def mean_series(key: str) -> list[float]:
            return [
                _mean([row["time_bins"][i][key] for row in rows if i < len(row["time_bins"])])
                for i in range(len(xs))
            ]

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

        axes[17].axis("off")
    else:
        axes[10].axis("off")
        axes[17].axis("off")

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
        "Joint UAV+UGV Strategy Diagnostics "
        f"({summary.get('strategy')}, n={len(rows)}, success={summary['full_confirm_success_rate']:.2f})",
        fontsize=14,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['strategy']:>18s} seed {row['seed']:>4}: "
        f"scout={row['scouted']}/{row['survivors']} "
        f"confirm={row['confirmed']}/{row['survivors']} "
        f"success={int(row['full_confirm_success'])} "
        f"cov={row['final_coverage_fraction']:.3f} "
        f"conf={row['final_confidence_mean']:.3f} "
        f"lat={row['avg_scout_to_confirm_latency_steps']:.1f} steps "
        f"uav_move={row['uav_movement_m_per_uav_step']:.2f}m/uav-step "
        f"ugv_move={row['ugv_movement_m_per_ugv_step']:.2f}m/ugv-step"
    )


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


def _print_summary(summary: dict[str, Any]) -> None:
    print("-" * 104)
    print(
        f"{str(summary.get('strategy', 'strategy')):>18s}: "
        f"episodes={int(summary['episodes'])} "
        f"scout={summary['mean_scout_recall']:.3f}+/-{summary['std_scout_recall']:.3f} "
        f"confirm={summary['mean_confirm_recall']:.3f}+/-{summary['std_confirm_recall']:.3f} "
        f"scout_auc={summary['mean_scout_auc']:.3f} "
        f"confirm_auc={summary['mean_confirm_auc']:.3f} "
        f"coverage_auc={summary['mean_coverage_auc']:.3f} "
        f"confidence_auc={summary['mean_confidence_auc']:.3f} "
        f"success={summary['full_confirm_success_rate']:.3f} "
        f"cov={summary['mean_final_coverage_fraction']:.3f}+/-{summary['std_final_coverage_fraction']:.3f} "
        f"conf={summary['mean_final_confidence']:.3f}+/-{summary['std_final_confidence']:.3f} "
        f"lat={summary['mean_scout_to_confirm_latency_steps']:.1f} steps "
        f"uav_move={summary['mean_uav_movement_m_per_drone_step']:.2f}m "
        f"ugv_move={summary['mean_ugv_movement_m_per_ground_step']:.2f}m"
    )


def _spec_metadata(spec: StrategySpec, *, baseline_ugv_controller: str = "native") -> dict[str, Any]:
    return {
        "label": spec.label,
        "name": spec.name,
        "checkpoint_dir": None if spec.checkpoint_dir is None else str(spec.checkpoint_dir),
        "baseline_ugv_controller": baseline_ugv_controller,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        "--strategies",
        dest="strategies",
        nargs="+",
        default=[DEFAULT_STRATEGY],
        help=(
            "Run exactly one strategy: "
            + ", ".join(AVAILABLE_STRATEGIES)
            + ", or happo."
        ),
    )
    parser.add_argument("--happo-checkpoint", default=None,
                        help=(
                            "HAPPO models/ directory. If provided, its saved scenario settings "
                            "are used for all strategies unless explicitly overridden by CLI flags; "
                            "also used by the 'happo' strategy token."
                        ))
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample HAPPO actions instead of using deterministic actor means.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override episode length. Default uses checkpoint max_steps when available, else 300.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument(
        "--baseline-ugv-controller",
        choices=UGV_CONTROLLER_CHOICES,
        default="native",
        help=(
            "UGV controller for heuristic baselines. 'native' keeps each heuristic's "
            "current UGV target memory/routing; 'matched_heuristic' keeps the "
            "heuristic UAV actions but uses scenario assignment plus planner hints "
            "like HAPPO."
        ),
    )
    parser.add_argument("--joint-diagnostic-ugvs", type=int, default=DEFAULT_JOINT_DIAG_UGVS)
    parser.add_argument("--n-drones", "--n-uavs", dest="n_drones", type=int, default=None,
                        help="Override UAV count. Default uses checkpoint count when available, else joint default.")
    parser.add_argument("--n-ugvs", "--n-ground", dest="n_ugvs", type=int, default=None)
    parser.add_argument("--n-survivors", type=int, default=None,
                        help="Override survivor observation slots. Default uses checkpoint count when available.")
    parser.add_argument("--active-survivors-min", type=int, default=None,
                        help="Minimum active true survivors per episode. Default uses checkpoint setting.")
    parser.add_argument("--active-survivors-max", type=int, default=None,
                        help="Maximum active true survivors per episode. Default uses checkpoint setting.")
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--enable-fire", action="store_true")
    parser.add_argument("--disable-fire", action="store_true")
    parser.add_argument(
        "--comms-dropout",
        type=float,
        default=None,
        help="Override the checkpoint communication outage fraction.",
    )
    parser.add_argument(
        "--comms-dropout-mode",
        choices=("iid", "bursty"),
        default=None,
    )
    parser.add_argument(
        "--comms-map-mode",
        choices=("global", "per_agent", "per-agent"),
        default=None,
    )
    parser.add_argument("--comms-dropout-min-steps", type=int, default=None)
    parser.add_argument("--comms-dropout-max-steps", type=int, default=None)
    parser.add_argument("--ugv-target-assignment-mode",
                        choices=(
                            "nearest",
                            "greedy",
                            "greedy_sticky",
                            "greedy-sticky",
                            "route_cost_greedy",
                            "route-cost-greedy",
                            "route_cost_sticky",
                            "route-cost-sticky",
                            "route_sequence_sticky",
                            "route-sequence-sticky",
                            "route_cost_global",
                            "route-cost-global",
                        ),
                        default=None)
    parser.add_argument("--ugv-planner-fire-mode", choices=("off", "cost", "block"), default=None)
    parser.add_argument(
        "--ugv-planner-fire-replan-policy",
        choices=("always", "affected", "lazy", "threshold_lazy"),
        default=None,
    )
    parser.add_argument("--ugv-planner-fire-replan-interval-steps", type=int, default=None)
    parser.add_argument("--ugv-planner-fire-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-fire-block-threshold", type=float, default=None)
    parser.add_argument("--ugv-planner-smoke-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-smolder-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-fire-buffer-m", type=float, default=None)
    parser.add_argument("--ugv-planner-fire-buffer-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-land-cover-costs", type=float, nargs="+", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--plots-output", default=None)
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
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
    if args.n_survivors is not None:
        if args.active_survivors_min is not None and args.active_survivors_min > args.n_survivors:
            parser.error("--active-survivors-min must be <= --n-survivors")
        if args.active_survivors_max is not None and args.active_survivors_max > args.n_survivors:
            parser.error("--active-survivors-max must be <= --n-survivors")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    if args.enable_fire and args.disable_fire:
        parser.error("--enable-fire and --disable-fire are mutually exclusive")
    if args.comms_dropout is not None and not 0.0 <= args.comms_dropout <= 1.0:
        parser.error("--comms-dropout must be between 0 and 1")
    if args.comms_dropout_min_steps is not None and args.comms_dropout_min_steps < 1:
        parser.error("--comms-dropout-min-steps must be positive")
    if args.comms_dropout_max_steps is not None and args.comms_dropout_max_steps < 1:
        parser.error("--comms-dropout-max-steps must be positive")
    if (
        args.comms_dropout_min_steps is not None
        and args.comms_dropout_max_steps is not None
        and args.comms_dropout_max_steps < args.comms_dropout_min_steps
    ):
        parser.error("--comms-dropout-max-steps must be >= --comms-dropout-min-steps")
    if args.terrain_cache_path is not None:
        args.terrain_cache_path = Path(args.terrain_cache_path)
        if not args.terrain_cache_path.is_file():
            parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")
    for name in (
        "ugv_planner_fire_replan_interval_steps",
        "ugv_planner_fire_cost",
        "ugv_planner_fire_block_threshold",
        "ugv_planner_smoke_cost",
        "ugv_planner_smolder_cost",
        "ugv_planner_fire_buffer_m",
        "ugv_planner_fire_buffer_cost",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.ugv_planner_land_cover_costs is not None and len(args.ugv_planner_land_cover_costs) not in {5, 6}:
        parser.error("--ugv-planner-land-cover-costs must contain 5 or 6 values")
    return args


def main() -> None:
    args = _parse_args()
    try:
        specs = parse_strategy_specs(args.strategies, happo_checkpoint=args.happo_checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    scenario_checkpoint = _scenario_checkpoint_from_specs(args, specs)
    scenario_kwargs = build_scenario_kwargs(args, scenario_checkpoint, specs=specs)
    baseline_ugv_controller = args.baseline_ugv_controller.replace("-", "_")
    survivor_slots = int(scenario_kwargs["n_survivors"])
    active_survivors_min = int(
        scenario_kwargs.get("active_survivors_min", survivor_slots)
    )
    active_survivors_max = int(
        scenario_kwargs.get("active_survivors_max", survivor_slots)
    )
    print(
        "scenario: "
        f"{scenario_kwargs['n_drones']} UAVs, "
        f"{scenario_kwargs['n_ground']} UGVs, "
        f"{survivor_slots} survivors ({active_survivors_min}..{active_survivors_max} active), "
        f"steps={scenario_kwargs['max_steps']}, "
        f"fire={'on' if not scenario_kwargs.get('disable_fire', True) else 'off'}, "
        f"assignment={scenario_kwargs.get('ugv_target_assignment_mode')}"
    )
    print(
        "communication: "
        f"dropout={scenario_kwargs.get('comms_dropout', 0.0):.2f}, "
        f"mode={scenario_kwargs.get('comms_dropout_mode', 'iid')}, "
        f"maps={scenario_kwargs.get('comms_map_mode', 'global')}, "
        f"burst={scenario_kwargs.get('comms_dropout_min_steps', 5)}.."
        f"{scenario_kwargs.get('comms_dropout_max_steps', 15)} steps"
    )
    print(
        "scenario source: "
        + (
            f"checkpoint {scenario_checkpoint}"
            if scenario_checkpoint is not None
            else "joint defaults / CLI"
        )
    )
    print(f"terrain: {scenario_kwargs.get('terrain_cache_path')}")
    print("strategy: " + ", ".join(
        spec.label if spec.checkpoint_dir is None else f"{spec.label}:{spec.checkpoint_dir}"
        for spec in specs
    ))
    print(f"baseline UGV controller: {baseline_ugv_controller}")
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    print(f"planned rollout: {len(specs)} strategies x {len(args.seeds)} seeds x {args.steps} steps")
    print("-" * 104)

    rows: list[dict[str, Any]] = []
    happo_cache: dict[tuple[Path, bool, tuple[int, ...]], HappoPolicy] = {}
    total_rollouts = len(specs) * len(args.seeds)
    total_rollout_steps = total_rollouts * int(args.steps)
    rollout_started_at = time.perf_counter()
    completed_rollouts = 0
    for spec in specs:
        for seed in args.seeds:
            row = run_rollout(
                spec,
                scenario_kwargs,
                seed,
                time_bins=args.time_bins,
                happo_cache=happo_cache,
                stochastic_happo=args.stochastic,
                baseline_ugv_controller=baseline_ugv_controller,
            )
            rows.append(row)
            _print_row(row)
            completed_rollouts += 1
            elapsed_rollout_s = time.perf_counter() - rollout_started_at
            completed_steps = int(args.steps) * completed_rollouts
            rollout_steps_per_second = completed_steps / max(elapsed_rollout_s, 1e-9)
            remaining_steps = max(total_rollout_steps - completed_steps, 0)
            eta_s = (
                remaining_steps / rollout_steps_per_second
                if rollout_steps_per_second > 0.0
                else float("nan")
            )
            if completed_rollouts == 1:
                print(f"ETA {_format_duration(eta_s)}", flush=True)
            print(f"progress: {completed_rollouts}/{total_rollouts} rollouts", flush=True)

    summary = summarize(rows, bins=args.time_bins)
    _print_summary(summary)
    payload = {
        "scenario_kwargs": scenario_kwargs,
        "metadata": {
            "strategy": _spec_metadata(specs[0], baseline_ugv_controller=baseline_ugv_controller),
            "strategies": [
                _spec_metadata(spec, baseline_ugv_controller=baseline_ugv_controller)
                for spec in specs
            ],
            "happo_deterministic": not args.stochastic,
            "steps": int(scenario_kwargs["max_steps"]),
            "scenario_source_checkpoint": None if scenario_checkpoint is None else str(scenario_checkpoint),
            "seeds": [int(seed) for seed in args.seeds],
            "scenario_kwargs": scenario_kwargs,
        },
        "rows": rows,
        "summary": summary,
    }
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
        print(f"wrote json: {path}")
    if args.plots_output:
        write_plots(rows, summary, Path(args.plots_output))
        print(f"wrote plots: {args.plots_output}")


if __name__ == "__main__":
    main()
