"""
Compare UAV-only search strategies on observable mission metrics.

This script is intentionally strategy-agnostic: hand-coded baselines and a
trained HAPPO checkpoint are evaluated on the same UAV-only survivor-search
scenario and plotted in one combined figure. It avoids HAPPO-specific reward
and policy diagnostics so the comparison stays fair.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import BASELINES, get_baseline
from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario
from scripts.diagnose_uav_happo import (
    DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
    TIME_BIN_COUNT,
    _append_time_bin,
    _coverage_shape_metrics,
    _distances_to_edges_m,
    _finalize_time_bins,
    _finite_mean,
    _finite_median,
    _metric_array,
    _metric_scalar,
    _new_time_bins,
    _path_metrics,
    _start_metrics,
    _summarize_scout_time_bins,
    _summarize_time_bins,
)

DEFAULT_TERRAIN_CACHE_PATH = ROOT / "data" / "terrain_cache" / "malibu_creek_500m_128.npz"


@dataclass(frozen=True)
class StrategySpec:
    label: str
    kind: str
    baseline_name: str | None = None
    checkpoint_dir: Path | None = None


def parse_strategy_specs(
    strategies: list[str],
    *,
    happo_checkpoint: str | None = None,
) -> list[StrategySpec]:
    """Resolve CLI strategy tokens into unique strategy specs."""
    raw_tokens = strategies or ["lawnmower", "ant_colony"]
    expanded: list[str] = []
    for token in raw_tokens:
        if token == "all":
            expanded.extend(BASELINES.keys())
        else:
            expanded.append(token)

    specs: list[StrategySpec] = []
    used_labels: set[str] = set()
    for token in expanded:
        if token in BASELINES:
            label = _unique_label(token, used_labels)
            specs.append(StrategySpec(label=label, kind="baseline", baseline_name=token))
            continue

        if token == "happo" or token.startswith("happo:"):
            checkpoint = token.split(":", 1)[1] if token.startswith("happo:") else happo_checkpoint
            checkpoint_dir = _resolve_happo_checkpoint(checkpoint)
            label = _unique_label("happo", used_labels)
            specs.append(StrategySpec(label=label, kind="happo", checkpoint_dir=checkpoint_dir))
            continue

        available = ", ".join([*BASELINES.keys(), "happo", "happo:/path/to/models", "all"])
        raise ValueError(f"unknown strategy {token!r}; available: {available}")

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


def _resolve_happo_checkpoint(path: str | None) -> Path:
    if path:
        return Path(path)
    return find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _first_happo_checkpoint(specs: list[StrategySpec]) -> Path | None:
    for spec in specs:
        if spec.kind == "happo":
            return spec.checkpoint_dir
    return None


def _infer_terrain_cache_grid_size(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        if "land_cover" not in data:
            raise ValueError(f"terrain cache is missing required array 'land_cover': {path}")
        shape = tuple(int(v) for v in data["land_cover"].shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"terrain cache land_cover must be a square 2D grid, got {shape}: {path}")
    return int(shape[0])


def build_scenario_kwargs(args: argparse.Namespace, specs: list[StrategySpec]) -> dict[str, Any]:
    checkpoint_dir = _first_happo_checkpoint(specs)
    scenario_kwargs: dict[str, Any] = {}
    if checkpoint_dir is not None:
        manifest = load_training_manifest(checkpoint_dir)
        if manifest is not None:
            scenario_kwargs.update(copy.deepcopy(manifest.get("env_args", {}).get("scenario_kwargs", {})))

    for key in (
        "known_survivor_spawn_distance_m",
        "known_survivor_spawn_distance_min_m",
        "known_survivor_spawn_distance_max_m",
        "survivor_spawn_reference",
    ):
        scenario_kwargs.pop(key, None)

    scenario_kwargs.update({
        "max_steps": int(args.steps),
        "n_drones": int(args.n_drones),
        "n_ground": 0,
        "n_survivors": int(args.n_survivors),
        "fire_grid_size": int(args.grid_size),
        "known_survivors_at_reset": False,
        "drone_can_confirm": True,
        "disable_fire": True,
        "comms_dropout": 0.0,
        "uav_confidence_diagnostics": True,
        "terrain_source": "real",
        "terrain_cache_path": str(args.terrain_cache_path),
    })
    if args.drone_min_footprint_radius_m is not None:
        scenario_kwargs.pop("drone_min_footprint", None)
        scenario_kwargs["drone_min_footprint_m"] = max(float(args.drone_min_footprint_radius_m), 0.0)
    if args.uav_start_min_separation_m is not None:
        scenario_kwargs["uav_start_min_separation_m"] = max(float(args.uav_start_min_separation_m), 0.0)
    if args.uav_start_edge_margin_m is not None:
        scenario_kwargs["uav_start_edge_margin_m"] = max(float(args.uav_start_edge_margin_m), 0.0)
    return scenario_kwargs


def run_rollout(
    spec: StrategySpec,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    happo_cache: dict[Path, HappoPolicy],
    stochastic_happo: bool = False,
    moving_no_confidence_gain_threshold: float = DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
) -> dict[str, Any]:
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset()
    scenario = env.scenario
    policy = _make_policy(spec, env, happo_cache=happo_cache, stochastic_happo=stochastic_happo)
    if hasattr(policy, "reset"):
        policy.reset()

    start_metrics = _start_metrics(scenario)
    n_survivors = int(scenario.n_survivors)
    n_drones = int(scenario.n_drones)
    max_steps = int(scenario_kwargs["max_steps"])
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    meters_per_sim = 1.0 / max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)

    first_scout_steps: list[int | None] = [None] * n_survivors
    displacement_m_values: list[float] = []
    new_coverage_cells_values: list[float] = []
    confidence_mean_values: list[float] = []
    confidence_gain_values: list[float] = []
    confidence_gain_by_drone_values: list[float] = []
    confidence_weighted_gain_values: list[float] = []
    confidence_weighted_gain_by_drone_values: list[float] = []
    confidence_opportunity_fraction_values: list[float] = []
    confidence_opportunity_best_gain_values: list[float] = []
    confidence_low_fraction_values: list[float] = []
    confidence_high_fraction_values: list[float] = []
    confidence_overlap_fraction_values: list[float] = []
    outside_footprint_values: list[float] = []
    overlap_values: list[float] = []
    expected_overlap_values: list[float] = []
    excess_overlap_values: list[float] = []
    inter_uav_overlap_values: list[float] = []
    opportunity_cells_values: list[float] = []
    opportunity_fraction_values: list[float] = []
    opportunity_available_values: list[float] = []
    boundary_distance_m_values: list[float] = []
    footprint_radius_m_values: list[float] = []
    all_positions_sim: list[np.ndarray] = []
    moving_no_new_coverage = 0
    moving_no_confidence_gain = 0
    diagnostic_steps = 0
    time_bins = _new_time_bins(TIME_BIN_COUNT)
    per_drone_stats = [_new_drone_stats(drone_idx) for drone_idx in range(n_drones)]

    for step in range(max_steps):
        prev_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool).copy()
        pre_drone_pos = [
            agent.state.pos[0].detach().cpu().numpy().astype(float).copy()
            for agent in scenario.world.agents[:n_drones]
        ]
        for pos in pre_drone_pos:
            all_positions_sim.append(pos.copy())

        actions = policy(env)
        env.step(actions)

        coverage_cells = _metric_array(scenario, "metric_uav_new_coverage_cells_by_drone", n_drones)
        outside_footprint = _metric_array(scenario, "metric_uav_outside_footprint_fraction_by_drone", n_drones)
        overlap = _metric_array(scenario, "metric_uav_overlap_fraction_by_drone", n_drones)
        expected_overlap = _metric_array(scenario, "metric_uav_expected_overlap_fraction_by_drone", n_drones)
        excess_overlap = _metric_array(scenario, "metric_uav_excess_overlap_fraction_by_drone", n_drones)
        inter_uav_overlap = _metric_array(scenario, "metric_uav_inter_uav_overlap_fraction_by_drone", n_drones)
        opportunity_cells = _metric_array(scenario, "metric_uav_coverage_opportunity_cells_by_drone", n_drones)
        opportunity_fraction = _metric_array(scenario, "metric_uav_coverage_opportunity_fraction_by_drone", n_drones)
        opportunity_available = _metric_array(
            scenario,
            "metric_uav_coverage_opportunity_available_fraction_by_drone",
            n_drones,
        )
        confidence_gain_by_drone = _metric_array(
            scenario,
            "metric_uav_confidence_gain_by_drone",
            n_drones,
        )
        confidence_weighted_gain_by_drone = _metric_array(
            scenario,
            "metric_uav_weighted_confidence_gain_by_drone",
            n_drones,
        )
        confidence_opportunity_fraction = _metric_array(
            scenario,
            "metric_uav_confidence_opportunity_fraction_by_drone",
            n_drones,
        )
        confidence_opportunity_best_gain = _metric_array(
            scenario,
            "metric_uav_confidence_opportunity_best_gain_by_drone",
            n_drones,
        )
        confidence_overlap_fraction = _metric_array(
            scenario,
            "metric_uav_confidence_overlap_fraction_by_drone",
            n_drones,
        )
        boundary_distance_m = _metric_array(scenario, "metric_uav_boundary_distance_m_by_drone", n_drones)
        footprint_radius_m = _metric_array(scenario, "metric_uav_footprint_radius_m_by_drone", n_drones)
        confidence_mean = _metric_scalar(scenario, "metric_uav_confidence_mean")
        confidence_gain = _metric_scalar(scenario, "metric_uav_confidence_gain")
        confidence_weighted_gain = _metric_scalar(scenario, "metric_uav_weighted_confidence_gain")
        confidence_low_fraction = _metric_scalar(scenario, "metric_uav_confidence_low_fraction")
        confidence_high_fraction = _metric_scalar(scenario, "metric_uav_confidence_high_fraction")
        confidence_mean_values.append(confidence_mean)
        confidence_gain_values.append(confidence_gain)
        confidence_weighted_gain_values.append(confidence_weighted_gain)
        confidence_low_fraction_values.append(confidence_low_fraction)
        confidence_high_fraction_values.append(confidence_high_fraction)

        post_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        newly_scouted = post_scouted & ~prev_scouted
        drone_detections = (
            scenario.step_drone_detections[0].detach().cpu().numpy().astype(bool)
            if n_drones > 0 and n_survivors > 0
            else np.zeros((n_drones, n_survivors), dtype=bool)
        )
        scout_credit = drone_detections & newly_scouted.reshape(1, -1)

        coverage_fraction_now = float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
        for drone_idx in range(n_drones):
            post_pos = scenario.world.agents[drone_idx].state.pos[0].detach().cpu().numpy().astype(float)
            all_positions_sim.append(post_pos.copy())
            displacement_vec = post_pos - pre_drone_pos[drone_idx]
            displacement_m = float(np.linalg.norm(displacement_vec) * meters_per_sim)
            new_cells = float(coverage_cells[drone_idx])
            confidence_gain_drone = float(confidence_gain_by_drone[drone_idx])
            confidence_weighted_gain_drone = float(confidence_weighted_gain_by_drone[drone_idx])
            confidence_opportunity = float(confidence_opportunity_fraction[drone_idx])
            confidence_best_gain = float(confidence_opportunity_best_gain[drone_idx])
            confidence_overlap = float(confidence_overlap_fraction[drone_idx])
            footprint_radius = float(footprint_radius_m[drone_idx])
            distances_to_edges = _distances_to_edges_m(
                np.asarray([post_pos], dtype=float),
                scenario,
                meters_per_sim,
            )[0]
            edge_threshold = footprint_radius if math.isfinite(footprint_radius) and footprint_radius > 0.0 else 25.0
            is_edge_step = bool(float(boundary_distance_m[drone_idx]) <= edge_threshold)
            is_corner_step = bool(np.count_nonzero(distances_to_edges <= edge_threshold) >= 2)
            moving_no_new = bool(displacement_m > 1.0 and new_cells < 1.0)
            moving_no_conf = bool(
                displacement_m > 1.0
                and math.isfinite(confidence_weighted_gain_drone)
                and confidence_weighted_gain_drone <= moving_no_confidence_gain_threshold
            )

            displacement_m_values.append(displacement_m)
            new_coverage_cells_values.append(new_cells)
            confidence_gain_by_drone_values.append(confidence_gain_drone)
            confidence_weighted_gain_by_drone_values.append(confidence_weighted_gain_drone)
            confidence_opportunity_fraction_values.append(confidence_opportunity)
            confidence_opportunity_best_gain_values.append(confidence_best_gain)
            confidence_overlap_fraction_values.append(confidence_overlap)
            outside_footprint_values.append(float(outside_footprint[drone_idx]))
            overlap_values.append(float(overlap[drone_idx]))
            expected_overlap_values.append(float(expected_overlap[drone_idx]))
            excess_overlap_values.append(float(excess_overlap[drone_idx]))
            inter_uav_overlap_values.append(float(inter_uav_overlap[drone_idx]))
            opportunity_cells_values.append(float(opportunity_cells[drone_idx]))
            opportunity_fraction_values.append(float(opportunity_fraction[drone_idx]))
            opportunity_available_values.append(float(opportunity_available[drone_idx]))
            boundary_distance_m_values.append(float(boundary_distance_m[drone_idx]))
            footprint_radius_m_values.append(footprint_radius)
            moving_no_new_coverage += int(moving_no_new)
            moving_no_confidence_gain += int(moving_no_conf)
            diagnostic_steps += 1

            drone_stats = per_drone_stats[drone_idx]
            drone_stats["positions_sim"].extend([pre_drone_pos[drone_idx], post_pos.copy()])
            drone_stats["displacement_m"].append(displacement_m)
            drone_stats["new_coverage_cells"].append(new_cells)
            drone_stats["confidence_gain"].append(confidence_gain_drone)
            drone_stats["confidence_weighted_gain"].append(confidence_weighted_gain_drone)
            drone_stats["confidence_opportunity_fraction"].append(confidence_opportunity)
            drone_stats["confidence_opportunity_best_gain"].append(confidence_best_gain)
            drone_stats["confidence_overlap_fraction"].append(confidence_overlap)
            drone_stats["outside_footprint"].append(float(outside_footprint[drone_idx]))
            drone_stats["overlap"].append(float(overlap[drone_idx]))
            drone_stats["expected_overlap"].append(float(expected_overlap[drone_idx]))
            drone_stats["excess_overlap"].append(float(excess_overlap[drone_idx]))
            drone_stats["inter_uav_overlap"].append(float(inter_uav_overlap[drone_idx]))
            drone_stats["coverage_opportunity_cells"].append(float(opportunity_cells[drone_idx]))
            drone_stats["coverage_opportunity_fraction"].append(float(opportunity_fraction[drone_idx]))
            drone_stats["coverage_opportunity_available_fraction"].append(float(opportunity_available[drone_idx]))
            drone_stats["boundary_distance_m"].append(float(boundary_distance_m[drone_idx]))
            drone_stats["footprint_radius_m"].append(footprint_radius)
            drone_stats["is_edge_step"].append(float(is_edge_step))
            drone_stats["is_corner_step"].append(float(is_corner_step))
            drone_stats["moving_no_new_coverage"] += int(moving_no_new)
            drone_stats["moving_no_confidence_gain"] += int(moving_no_conf)
            drone_stats["diagnostic_steps"] += 1
            for survivor_idx in np.flatnonzero(scout_credit[drone_idx]):
                drone_stats["scout_credit_count"] += 1
                drone_stats["scouted_survivors"].add(int(survivor_idx))
                drone_stats["first_scout_steps"].append(step + 1)

            _append_time_bin(
                time_bins,
                step=step,
                max_steps=max_steps,
                values={
                    "coverage_fraction": coverage_fraction_now,
                    "confidence_mean": confidence_mean,
                    "confidence_gain": confidence_gain_drone,
                    "confidence_weighted_gain": confidence_weighted_gain_drone,
                    "confidence_opportunity_fraction": confidence_opportunity,
                    "confidence_overlap_fraction": confidence_overlap,
                    "confidence_low_fraction": confidence_low_fraction,
                    "confidence_high_fraction": confidence_high_fraction,
                    "displacement_m": displacement_m,
                    "new_coverage_cells": new_cells,
                    "overlap": float(overlap[drone_idx]),
                    "expected_overlap": float(expected_overlap[drone_idx]),
                    "excess_overlap": float(excess_overlap[drone_idx]),
                    "inter_uav_overlap": float(inter_uav_overlap[drone_idx]),
                    "coverage_opportunity_cells": float(opportunity_cells[drone_idx]),
                    "coverage_opportunity_fraction": float(opportunity_fraction[drone_idx]),
                    "coverage_opportunity_available_fraction": float(opportunity_available[drone_idx]),
                    "outside_footprint": float(outside_footprint[drone_idx]),
                    "edge_step": float(is_edge_step),
                    "corner_step": float(is_corner_step),
                    "moving_no_new_coverage": float(moving_no_new),
                    "moving_no_confidence_gain": float(moving_no_conf),
                },
            )

        scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        for survivor_idx, is_scouted in enumerate(scouted):
            if is_scouted and first_scout_steps[survivor_idx] is None:
                first_scout_steps[survivor_idx] = step + 1
        if all(value is not None for value in first_scout_steps):
            break

    scouted_count = sum(value is not None for value in first_scout_steps)
    missed_count = n_survivors - scouted_count
    scout_steps = [value for value in first_scout_steps if value is not None]
    all_scouted_step = max(scout_steps) if scouted_count == n_survivors and scout_steps else None
    final_coverage_fraction = float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
    final_confidence_mean = float(scenario.uav_confidence_grid[0].float().mean().detach().cpu().item())
    final_confidence_low_fraction = float(
        (scenario.uav_confidence_grid[0] < 0.50).float().mean().detach().cpu().item()
    )
    final_confidence_high_fraction = float(
        (scenario.uav_confidence_grid[0] >= 0.80).float().mean().detach().cpu().item()
    )
    path_metrics = _path_metrics(
        all_positions_sim,
        displacement_m_values,
        boundary_distance_m_values,
        footprint_radius_m_values,
        scenario,
    )
    coverage_shape_metrics = _coverage_shape_metrics(
        scenario.coverage_grid[0].detach().cpu().numpy().astype(bool),
        scenario,
        _finite_mean(footprint_radius_m_values),
    )
    per_drone = [_finalize_drone_stats(stats, scenario) for stats in per_drone_stats]
    total_new_coverage_cells = float(np.sum(new_coverage_cells_values)) if new_coverage_cells_values else 0.0
    total_confidence_gain_by_drone = (
        float(np.sum(confidence_gain_by_drone_values)) if confidence_gain_by_drone_values else 0.0
    )
    total_confidence_weighted_gain_by_drone = (
        float(np.sum(confidence_weighted_gain_by_drone_values))
        if confidence_weighted_gain_by_drone_values else 0.0
    )
    path_length_m = float(path_metrics.get("path_length_m", math.nan))

    row = {
        "strategy": spec.label,
        "strategy_kind": spec.kind,
        "checkpoint": None if spec.checkpoint_dir is None else str(spec.checkpoint_dir),
        "seed": int(seed),
        "max_steps": max_steps,
        "survivors": n_survivors,
        "scouted": scouted_count,
        "missed": missed_count,
        "recall": scouted_count / n_survivors if n_survivors else 0.0,
        "final_coverage_fraction": final_coverage_fraction,
        "final_confidence_mean": final_confidence_mean,
        "final_confidence_low_fraction": final_confidence_low_fraction,
        "final_confidence_high_fraction": final_confidence_high_fraction,
        "full_success": float(scouted_count == n_survivors),
        "avg_scout_step": float(np.mean(scout_steps)) if scout_steps else math.nan,
        "avg_scout_time_s": float(np.mean(scout_steps) * step_seconds) if scout_steps else math.nan,
        "all_scouted_step": all_scouted_step,
        "all_scouted_time_s": None if all_scouted_step is None else float(all_scouted_step * step_seconds),
        "first_scout_steps": first_scout_steps,
        "first_scout_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_scout_steps
        ],
        "avg_displacement_m": _finite_mean(displacement_m_values),
        "avg_new_coverage_cells": _finite_mean(new_coverage_cells_values),
        "total_new_coverage_cells": total_new_coverage_cells,
        "new_coverage_cells_per_meter": _safe_div(total_new_coverage_cells, path_length_m),
        "avg_confidence_mean": _finite_mean(confidence_mean_values),
        "avg_confidence_gain": _finite_mean(confidence_gain_values),
        "avg_confidence_gain_by_drone": _finite_mean(confidence_gain_by_drone_values),
        "total_confidence_gain_by_drone": total_confidence_gain_by_drone,
        "confidence_gain_per_meter": _safe_div(total_confidence_gain_by_drone, path_length_m),
        "avg_confidence_weighted_gain": _finite_mean(confidence_weighted_gain_values),
        "avg_confidence_weighted_gain_by_drone": _finite_mean(confidence_weighted_gain_by_drone_values),
        "total_confidence_weighted_gain_by_drone": total_confidence_weighted_gain_by_drone,
        "confidence_weighted_gain_per_meter": _safe_div(
            total_confidence_weighted_gain_by_drone,
            path_length_m,
        ),
        "avg_confidence_opportunity_fraction": _finite_mean(confidence_opportunity_fraction_values),
        "avg_confidence_opportunity_best_gain": _finite_mean(confidence_opportunity_best_gain_values),
        "avg_confidence_low_fraction": _finite_mean(confidence_low_fraction_values),
        "avg_confidence_high_fraction": _finite_mean(confidence_high_fraction_values),
        "avg_confidence_overlap_fraction": _finite_mean(confidence_overlap_fraction_values),
        "avg_outside_footprint_fraction": _finite_mean(outside_footprint_values),
        "max_outside_footprint_fraction": max(outside_footprint_values) if outside_footprint_values else 0.0,
        "outside_footprint_step_frac_10": (
            float(np.mean([value >= 0.10 for value in outside_footprint_values]))
            if outside_footprint_values else 0.0
        ),
        "avg_overlap_fraction": _finite_mean(overlap_values),
        "avg_expected_overlap_fraction": _finite_mean(expected_overlap_values),
        "avg_excess_overlap_fraction": _finite_mean(excess_overlap_values),
        "avg_inter_uav_overlap_fraction": _finite_mean(inter_uav_overlap_values),
        "avg_coverage_opportunity_cells": _finite_mean(opportunity_cells_values),
        "avg_coverage_opportunity_fraction": _finite_mean(opportunity_fraction_values),
        "avg_coverage_opportunity_available_fraction": _finite_mean(opportunity_available_values),
        "excess_overlap_step_frac_10": (
            float(np.mean([value >= 0.10 for value in excess_overlap_values]))
            if excess_overlap_values else 0.0
        ),
        "excess_overlap_step_frac_20": (
            float(np.mean([value >= 0.20 for value in excess_overlap_values]))
            if excess_overlap_values else 0.0
        ),
        "inter_uav_overlap_step_frac_20": (
            float(np.mean([value >= 0.20 for value in inter_uav_overlap_values]))
            if inter_uav_overlap_values else 0.0
        ),
        "overlap_step_frac_60": (
            float(np.mean([value >= 0.60 for value in overlap_values]))
            if overlap_values else 0.0
        ),
        "new_coverage_step_frac": (
            float(np.mean([value >= 1.0 for value in new_coverage_cells_values]))
            if new_coverage_cells_values else 0.0
        ),
        "moving_no_new_coverage_frac": moving_no_new_coverage / diagnostic_steps if diagnostic_steps else 0.0,
        "moving_no_confidence_gain_frac": (
            moving_no_confidence_gain / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "time_bins": _finalize_time_bins(time_bins),
        "per_drone": per_drone,
        **start_metrics,
        **path_metrics,
        **coverage_shape_metrics,
    }
    row["failure_label"] = _failure_label(row)
    close = getattr(env, "close", None)
    if close is not None:
        close()
    return row


def _safe_div(numerator: float, denominator: float, *, default: float = math.nan) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return default
    if abs(float(denominator)) <= 1e-12:
        return default
    return float(numerator) / float(denominator)


def _make_policy(
    spec: StrategySpec,
    env: Any,
    *,
    happo_cache: dict[Path, HappoPolicy],
    stochastic_happo: bool,
) -> Callable:
    if spec.kind == "baseline":
        assert spec.baseline_name is not None
        return get_baseline(spec.baseline_name, env)
    if spec.kind == "happo":
        assert spec.checkpoint_dir is not None
        checkpoint = spec.checkpoint_dir
        if checkpoint not in happo_cache:
            happo_cache[checkpoint] = HappoPolicy.from_checkpoint(
                checkpoint,
                deterministic=not stochastic_happo,
            )
        policy = happo_cache[checkpoint]
        expected_agents = int(env.scenario.n_drones) + int(env.scenario.n_ground)
        if len(policy.actors) != expected_agents:
            raise ValueError(
                f"{spec.label} checkpoint contains {len(policy.actors)} actors, "
                f"but scenario contains {expected_agents} agents"
            )
        return policy
    raise ValueError(f"unsupported strategy kind {spec.kind!r}")


def _new_drone_stats(drone_idx: int) -> dict[str, Any]:
    return {
        "drone": int(drone_idx),
        "positions_sim": [],
        "displacement_m": [],
        "new_coverage_cells": [],
        "confidence_gain": [],
        "confidence_weighted_gain": [],
        "confidence_opportunity_fraction": [],
        "confidence_opportunity_best_gain": [],
        "confidence_overlap_fraction": [],
        "outside_footprint": [],
        "overlap": [],
        "expected_overlap": [],
        "excess_overlap": [],
        "inter_uav_overlap": [],
        "coverage_opportunity_cells": [],
        "coverage_opportunity_fraction": [],
        "coverage_opportunity_available_fraction": [],
        "boundary_distance_m": [],
        "footprint_radius_m": [],
        "is_edge_step": [],
        "is_corner_step": [],
        "moving_no_new_coverage": 0,
        "moving_no_confidence_gain": 0,
        "diagnostic_steps": 0,
        "scout_credit_count": 0,
        "scouted_survivors": set(),
        "first_scout_steps": [],
    }


def _finalize_drone_stats(stats: dict[str, Any], scenario: WildfireSearchScenario) -> dict[str, Any]:
    steps = int(stats["diagnostic_steps"])
    path = _path_metrics(
        stats["positions_sim"],
        stats["displacement_m"],
        stats["boundary_distance_m"],
        stats["footprint_radius_m"],
        scenario,
    )
    total_new_coverage_cells = float(np.sum(stats["new_coverage_cells"])) if stats["new_coverage_cells"] else 0.0
    total_confidence_gain = float(np.sum(stats["confidence_gain"])) if stats["confidence_gain"] else 0.0
    total_confidence_weighted_gain = (
        float(np.sum(stats["confidence_weighted_gain"])) if stats["confidence_weighted_gain"] else 0.0
    )
    path_length_m = float(path.get("path_length_m", math.nan))
    return {
        "drone": int(stats["drone"]),
        "diagnostic_steps": steps,
        "scout_credit_count": int(stats["scout_credit_count"]),
        "scouted_survivors": sorted(int(v) for v in stats["scouted_survivors"]),
        "first_scout_steps": [int(v) for v in stats["first_scout_steps"]],
        "avg_displacement_m": _finite_mean(stats["displacement_m"]),
        "avg_new_coverage_cells": _finite_mean(stats["new_coverage_cells"]),
        "total_new_coverage_cells": total_new_coverage_cells,
        "new_coverage_cells_per_meter": _safe_div(total_new_coverage_cells, path_length_m),
        "avg_confidence_gain": _finite_mean(stats["confidence_gain"]),
        "total_confidence_gain": total_confidence_gain,
        "confidence_gain_per_meter": _safe_div(total_confidence_gain, path_length_m),
        "avg_confidence_weighted_gain": _finite_mean(stats["confidence_weighted_gain"]),
        "total_confidence_weighted_gain": total_confidence_weighted_gain,
        "confidence_weighted_gain_per_meter": _safe_div(total_confidence_weighted_gain, path_length_m),
        "avg_confidence_opportunity_fraction": _finite_mean(stats["confidence_opportunity_fraction"]),
        "avg_confidence_opportunity_best_gain": _finite_mean(stats["confidence_opportunity_best_gain"]),
        "avg_confidence_overlap_fraction": _finite_mean(stats["confidence_overlap_fraction"]),
        "new_coverage_step_frac": (
            float(np.mean([value >= 1.0 for value in stats["new_coverage_cells"]]))
            if stats["new_coverage_cells"] else 0.0
        ),
        "avg_outside_footprint_fraction": _finite_mean(stats["outside_footprint"]),
        "avg_overlap_fraction": _finite_mean(stats["overlap"]),
        "avg_expected_overlap_fraction": _finite_mean(stats["expected_overlap"]),
        "avg_excess_overlap_fraction": _finite_mean(stats["excess_overlap"]),
        "avg_inter_uav_overlap_fraction": _finite_mean(stats["inter_uav_overlap"]),
        "avg_coverage_opportunity_cells": _finite_mean(stats["coverage_opportunity_cells"]),
        "avg_coverage_opportunity_fraction": _finite_mean(stats["coverage_opportunity_fraction"]),
        "avg_coverage_opportunity_available_fraction": _finite_mean(
            stats["coverage_opportunity_available_fraction"]
        ),
        "edge_step_frac": _finite_mean(stats["is_edge_step"]),
        "corner_step_frac": _finite_mean(stats["is_corner_step"]),
        "moving_no_new_coverage_frac": (
            stats["moving_no_new_coverage"] / steps if steps else 0.0
        ),
        "moving_no_confidence_gain_frac": (
            stats["moving_no_confidence_gain"] / steps if steps else 0.0
        ),
        **path,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = sorted({str(row["strategy"]) for row in rows})
    by_strategy = {
        strategy: _summarize_rows([row for row in rows if row["strategy"] == strategy])
        for strategy in strategies
    }
    return {
        "episodes": float(len(rows)),
        "strategies": strategies,
        "by_strategy": by_strategy,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["all_scouted_step"] is not None]
    summary = {
        "episodes": float(len(rows)),
        "mean_scouted": _finite_mean([float(row["scouted"]) for row in rows]),
        "mean_missed": _finite_mean([float(row["missed"]) for row in rows]),
        "mean_recall": _finite_mean([float(row["recall"]) for row in rows]),
        "mean_final_coverage_fraction": _finite_mean([
            float(row["final_coverage_fraction"]) for row in rows
        ]),
        "mean_final_confidence_mean": _finite_mean([
            float(row["final_confidence_mean"]) for row in rows
        ]),
        "mean_final_confidence_low_fraction": _finite_mean([
            float(row["final_confidence_low_fraction"]) for row in rows
        ]),
        "mean_final_confidence_high_fraction": _finite_mean([
            float(row["final_confidence_high_fraction"]) for row in rows
        ]),
        "full_success_rate": _finite_mean([float(row["full_success"]) for row in rows]),
        "mean_avg_scout_step": _finite_mean([float(row["avg_scout_step"]) for row in rows]),
        "mean_avg_scout_time_s": _finite_mean([float(row["avg_scout_time_s"]) for row in rows]),
        "mean_all_scouted_step_successes": (
            _finite_mean([float(row["all_scouted_step"]) for row in successful])
            if successful else math.nan
        ),
        "mean_all_scouted_time_s_successes": (
            _finite_mean([float(row["all_scouted_time_s"]) for row in successful])
            if successful else math.nan
        ),
        "mean_displacement_m": _finite_mean([float(row["avg_displacement_m"]) for row in rows]),
        "mean_new_coverage_cells": _finite_mean([
            float(row["avg_new_coverage_cells"]) for row in rows
        ]),
        "mean_total_new_coverage_cells": _finite_mean([
            float(row["total_new_coverage_cells"]) for row in rows
        ]),
        "mean_new_coverage_cells_per_meter": _finite_mean([
            float(row["new_coverage_cells_per_meter"]) for row in rows
        ]),
        "mean_confidence_gain": _finite_mean([
            float(row["avg_confidence_gain"]) for row in rows
        ]),
        "mean_confidence_gain_by_drone": _finite_mean([
            float(row["avg_confidence_gain_by_drone"]) for row in rows
        ]),
        "mean_total_confidence_gain_by_drone": _finite_mean([
            float(row["total_confidence_gain_by_drone"]) for row in rows
        ]),
        "mean_confidence_gain_per_meter": _finite_mean([
            float(row["confidence_gain_per_meter"]) for row in rows
        ]),
        "mean_confidence_weighted_gain": _finite_mean([
            float(row["avg_confidence_weighted_gain"]) for row in rows
        ]),
        "mean_confidence_weighted_gain_by_drone": _finite_mean([
            float(row["avg_confidence_weighted_gain_by_drone"]) for row in rows
        ]),
        "mean_confidence_weighted_gain_per_meter": _finite_mean([
            float(row["confidence_weighted_gain_per_meter"]) for row in rows
        ]),
        "mean_confidence_opportunity_fraction": _finite_mean([
            float(row["avg_confidence_opportunity_fraction"]) for row in rows
        ]),
        "mean_confidence_opportunity_best_gain": _finite_mean([
            float(row["avg_confidence_opportunity_best_gain"]) for row in rows
        ]),
        "mean_confidence_overlap_fraction": _finite_mean([
            float(row["avg_confidence_overlap_fraction"]) for row in rows
        ]),
        "mean_new_coverage_step_frac": _finite_mean([
            float(row["new_coverage_step_frac"]) for row in rows
        ]),
        "mean_outside_footprint_fraction": _finite_mean([
            float(row["avg_outside_footprint_fraction"]) for row in rows
        ]),
        "mean_overlap_fraction": _finite_mean([float(row["avg_overlap_fraction"]) for row in rows]),
        "mean_expected_overlap_fraction": _finite_mean([
            float(row["avg_expected_overlap_fraction"]) for row in rows
        ]),
        "mean_excess_overlap_fraction": _finite_mean([
            float(row["avg_excess_overlap_fraction"]) for row in rows
        ]),
        "mean_inter_uav_overlap_fraction": _finite_mean([
            float(row["avg_inter_uav_overlap_fraction"]) for row in rows
        ]),
        "mean_coverage_opportunity_cells": _finite_mean([
            float(row["avg_coverage_opportunity_cells"]) for row in rows
        ]),
        "mean_coverage_opportunity_fraction": _finite_mean([
            float(row["avg_coverage_opportunity_fraction"]) for row in rows
        ]),
        "mean_coverage_opportunity_available_fraction": _finite_mean([
            float(row["avg_coverage_opportunity_available_fraction"]) for row in rows
        ]),
        "mean_moving_no_new_coverage_frac": _finite_mean([
            float(row["moving_no_new_coverage_frac"]) for row in rows
        ]),
        "mean_moving_no_confidence_gain_frac": _finite_mean([
            float(row["moving_no_confidence_gain_frac"]) for row in rows
        ]),
        "mean_path_length_m": _finite_mean([float(row["path_length_m"]) for row in rows]),
        "mean_edge_step_frac": _finite_mean([float(row["edge_step_frac"]) for row in rows]),
        "mean_corner_step_frac": _finite_mean([float(row["corner_step_frac"]) for row in rows]),
        "mean_stalled_step_frac": _finite_mean([float(row["stalled_step_frac"]) for row in rows]),
        "mean_longest_stall_steps": _finite_mean([float(row["longest_stall_steps"]) for row in rows]),
        "mean_coverage_center_fraction": _finite_mean([
            float(row["coverage_center_fraction"]) for row in rows
        ]),
        "mean_coverage_edge_bias": _finite_mean([float(row["coverage_edge_bias"]) for row in rows]),
        "mean_min_start_pair_distance_m": _finite_mean([
            float(row["min_start_pair_distance_m"]) for row in rows
        ]),
        "mean_min_start_edge_distance_m": _finite_mean([
            float(row["min_start_edge_distance_m"]) for row in rows
        ]),
        "time_bins": _summarize_time_bins(rows),
        "scout_time_bins": _summarize_scout_time_bins(rows),
        "per_drone": _summarize_per_drone(rows),
        "label_counts": _label_counts(rows),
        "distributions": _distribution_summary(rows),
    }
    return summary


def _summarize_per_drone(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drone_indices = sorted({
        int(drone["drone"])
        for row in rows
        for drone in row.get("per_drone", [])
    })
    metrics = (
        "scout_credit_count",
        "path_length_m",
        "avg_displacement_m",
        "avg_new_coverage_cells",
        "total_new_coverage_cells",
        "new_coverage_cells_per_meter",
        "new_coverage_step_frac",
        "avg_confidence_gain",
        "total_confidence_gain",
        "confidence_gain_per_meter",
        "avg_confidence_weighted_gain",
        "total_confidence_weighted_gain",
        "confidence_weighted_gain_per_meter",
        "avg_confidence_opportunity_fraction",
        "avg_confidence_opportunity_best_gain",
        "avg_confidence_overlap_fraction",
        "avg_outside_footprint_fraction",
        "avg_overlap_fraction",
        "avg_expected_overlap_fraction",
        "avg_excess_overlap_fraction",
        "avg_inter_uav_overlap_fraction",
        "avg_coverage_opportunity_fraction",
        "avg_coverage_opportunity_available_fraction",
        "edge_step_frac",
        "corner_step_frac",
        "moving_no_new_coverage_frac",
        "moving_no_confidence_gain_frac",
        "stalled_step_frac",
        "longest_stall_steps",
    )
    summaries = []
    for drone_idx in drone_indices:
        entries = [
            drone
            for row in rows
            for drone in row.get("per_drone", [])
            if int(drone.get("drone", -1)) == drone_idx
        ]
        row_summary: dict[str, Any] = {
            "drone": int(drone_idx),
            "episodes": float(len(entries)),
        }
        for metric in metrics:
            row_summary[f"mean_{metric}"] = _finite_mean([
                float(entry.get(metric, math.nan))
                for entry in entries
            ])
        summaries.append(row_summary)
    return summaries


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {
        "recall": "recall",
        "coverage": "final_coverage_fraction",
        "confidence": "final_confidence_mean",
        "move_m": "avg_displacement_m",
        "new_cells": "avg_new_coverage_cells",
        "new_cells_per_m": "new_coverage_cells_per_meter",
        "conf_gain_per_m": "confidence_gain_per_meter",
        "outside": "avg_outside_footprint_fraction",
        "overlap": "avg_overlap_fraction",
        "expected_overlap": "avg_expected_overlap_fraction",
        "excess_overlap": "avg_excess_overlap_fraction",
        "confidence_overlap": "avg_confidence_overlap_fraction",
        "edge_frac": "edge_step_frac",
        "corner_frac": "corner_step_frac",
        "moving_no_new": "moving_no_new_coverage_frac",
        "moving_no_conf": "moving_no_confidence_gain_frac",
    }
    out: dict[str, float] = {}
    for prefix, key in metrics.items():
        values = np.asarray(
            [float(row[key]) for row in rows if math.isfinite(float(row[key]))],
            dtype=float,
        )
        if values.size == 0:
            for q in (0, 25, 50, 75, 100):
                out[f"{prefix}_p{q}"] = math.nan
            continue
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        for q, value in zip((0, 25, 50, 75, 100), percentiles):
            out[f"{prefix}_p{q}"] = float(value)
    return out


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("failure_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _failure_label(row: dict[str, Any]) -> str:
    if row["full_success"]:
        return "success"
    if row["final_coverage_fraction"] >= 0.85 and row["recall"] < 0.80:
        return "survivor_miss_despite_coverage"
    if row["edge_step_frac"] > 0.35 or row["corner_step_frac"] > 0.10:
        return "edge_trap"
    if row["moving_no_new_coverage_frac"] > 0.30 or row["avg_excess_overlap_fraction"] > 0.25:
        return "wasteful_revisit"
    if row["final_coverage_fraction"] < 0.50:
        return "low_coverage"
    return "partial_search"


def write_distribution_plots(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
    })

    strategies = list(summary["strategies"])
    colors = _strategy_colors(strategies)
    fig, axes = plt.subplots(6, 3, figsize=(15, 26), constrained_layout=True)
    axes_flat = axes.ravel()

    hist_panels = [
        ("Survivor Recall", "recall", (0.0, 1.0), "survivors scouted / total survivors"),
        ("Final Coverage", "final_coverage_fraction", (0.0, 1.0), "fraction of map covered"),
        ("Final Confidence", "final_confidence_mean", (0.0, 1.0), "mean inspection confidence"),
        ("Movement per Drone-Step", "avg_displacement_m", None, "meters / drone-step"),
        ("New Cells per Drone-Step", "avg_new_coverage_cells", None, "new coverage cells / drone-step"),
        ("New Cells per Meter", "new_coverage_cells_per_meter", None, "new coverage cells / meter"),
        ("Confidence Gain per Meter", "confidence_gain_per_meter", None, "confidence gain / meter"),
        ("Excess Footprint Overlap", "avg_excess_overlap_fraction", (0.0, 1.0), "fraction above expected overlap"),
        ("Moving Without Confidence Gain", "moving_no_confidence_gain_frac", (0.0, 1.0), "fraction of drone-steps"),
    ]
    for ax, (title, key, xlim, xlabel) in zip(axes_flat[:9], hist_panels):
        _plot_overlay_hist(ax, rows, strategies, colors, key, title, xlabel, xlim)

    _plot_strategy_bars(
        axes_flat[9],
        summary,
        "mean_recall",
        "Mean Recall",
        "mean recall",
        colors,
        ylim=(0.0, 1.0),
    )
    _plot_strategy_bars(
        axes_flat[10],
        summary,
        "mean_final_coverage_fraction",
        "Mean Coverage",
        "mean final coverage",
        colors,
        ylim=(0.0, 1.0),
    )
    _plot_strategy_bars(
        axes_flat[11],
        summary,
        "mean_final_confidence_mean",
        "Mean Confidence",
        "mean final confidence",
        colors,
        ylim=(0.0, 1.0),
    )
    _plot_time_bin_lines(
        axes_flat[12],
        summary,
        "new_coverage_cells",
        "Time-Bin New Cells",
        "new cells / drone-step",
        colors,
    )
    _plot_time_bin_lines(
        axes_flat[13],
        summary,
        "confidence_mean",
        "Time-Bin Confidence",
        "mean confidence",
        colors,
        ylim=(0.0, 1.0),
    )
    _plot_time_bin_multi(
        axes_flat[14],
        summary,
        "Time-Bin Search Friction",
        {
            "excess": "excess_overlap",
            "moving no new": "moving_no_new_coverage",
            "moving no conf": "moving_no_confidence_gain",
        },
        colors,
    )
    _plot_per_drone_bars(axes_flat[15], summary, "mean_path_length_m", "Per-Drone Path Length", "m", colors)
    _plot_scout_time_bins(axes_flat[16], summary, colors)
    _plot_failure_labels(axes_flat[17], summary, colors)

    fig.suptitle(
        "UAV Strategy Diagnostics "
        f"(n={int(sum(summary['by_strategy'][s]['episodes'] for s in strategies))} episodes)",
        fontsize=15,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _strategy_colors(strategies: list[str]) -> dict[str, str]:
    palette = [
        "#4f7cff",
        "#36a269",
        "#d44a3a",
        "#8a5cf6",
        "#d4a72c",
        "#20242c",
        "#2aa6b8",
    ]
    return {strategy: palette[idx % len(palette)] for idx, strategy in enumerate(strategies)}


def _plot_overlay_hist(
    ax: Any,
    rows: list[dict[str, Any]],
    strategies: list[str],
    colors: dict[str, str],
    key: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float] | None,
) -> None:
    plotted = False
    values_by_strategy: dict[str, list[float]] = {}
    for strategy in strategies:
        values = [
            float(row[key])
            for row in rows
            if row["strategy"] == strategy and key in row and math.isfinite(float(row[key]))
        ]
        values_by_strategy[strategy] = values
    all_values = [
        value
        for values in values_by_strategy.values()
        for value in values
    ]
    bins = _shared_hist_bins(all_values, xlim)
    legend_lines = []
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    for strategy in strategies:
        values = values_by_strategy[strategy]
        if not values:
            continue
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        ax.hist(
            values,
            bins=bins,
            alpha=0.28,
            color=colors[strategy],
            edgecolor=colors[strategy],
            linewidth=0.6,
        )
        ax.axvline(mean_value, color=colors[strategy], linewidth=1.8)
        ax.axvline(median_value, color=colors[strategy], linewidth=1.4, linestyle="--")
        legend_lines.extend([
            Patch(
                facecolor=colors[strategy],
                edgecolor=colors[strategy],
                alpha=0.28,
                label=f"{strategy} dist (n={len(values)})",
            ),
            Line2D([0], [0], color=colors[strategy], linewidth=1.8, label=f"{strategy} mean {mean_value:.3f}"),
            Line2D([0], [0], color=colors[strategy], linewidth=1.4, linestyle="--", label=f"{strategy} med {median_value:.3f}"),
        ])
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("episodes")
    ax.grid(axis="y", alpha=0.22)
    if legend_lines:
        ax.legend(handles=legend_lines, fontsize=7, loc="best", ncol=1)


def _shared_hist_bins(values: list[float], xlim: tuple[float, float] | None) -> np.ndarray | int:
    if not values:
        return 10
    if xlim is not None:
        low, high = xlim
        return np.linspace(low, high, 21)
    arr = np.asarray(values, dtype=float)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if not math.isfinite(low) or not math.isfinite(high):
        return 10
    if abs(high - low) <= 1e-12:
        pad = max(abs(low) * 0.05, 1.0)
        low -= pad
        high += pad
    bins = min(max(len(values) // 8, 8), 28)
    return np.linspace(low, high, bins + 1)


def _plot_strategy_bars(
    ax: Any,
    summary: dict[str, Any],
    key: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    ylim: tuple[float, float] | None = None,
) -> None:
    strategies = list(summary["strategies"])
    values = [float(summary["by_strategy"][strategy].get(key, math.nan)) for strategy in strategies]
    x = np.arange(len(strategies))
    bars = ax.bar(x, values, color=[colors[strategy] for strategy in strategies], alpha=0.82)
    for bar, value in zip(bars, values):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, strategies, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.22)


def _plot_time_bin_lines(
    ax: Any,
    summary: dict[str, Any],
    key: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    ylim: tuple[float, float] | None = None,
) -> None:
    for strategy in summary["strategies"]:
        bins = summary["by_strategy"][strategy].get("time_bins", [])
        if not bins:
            continue
        centers = [
            0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
            for row in bins
        ]
        values = [float(row.get(key, math.nan)) for row in bins]
        ax.plot(centers, values, marker="o", linewidth=1.8, markersize=4, label=strategy, color=colors[strategy])
    ax.set_xlim(0.0, 1.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)


def _plot_time_bin_multi(
    ax: Any,
    summary: dict[str, Any],
    title: str,
    keys: dict[str, str],
    colors: dict[str, str],
) -> None:
    linestyles = ["-", "--", ":"]
    for strategy in summary["strategies"]:
        bins = summary["by_strategy"][strategy].get("time_bins", [])
        if not bins:
            continue
        centers = [
            0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
            for row in bins
        ]
        for idx, (label, key) in enumerate(keys.items()):
            values = [float(row.get(key, math.nan)) for row in bins]
            ax.plot(
                centers,
                values,
                marker="o",
                linewidth=1.4,
                linestyle=linestyles[idx % len(linestyles)],
                label=f"{strategy} {label}",
                color=colors[strategy],
                alpha=0.9,
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)


def _plot_scout_time_bins(ax: Any, summary: dict[str, Any], colors: dict[str, str]) -> None:
    for strategy in summary["strategies"]:
        bins = summary["by_strategy"][strategy].get("scout_time_bins", [])
        if not bins:
            continue
        centers = [
            0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
            for row in bins
        ]
        values = [float(row.get("mean_cumulative_recall", math.nan)) for row in bins]
        ax.plot(centers, values, marker="o", linewidth=1.8, markersize=4, label=strategy, color=colors[strategy])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("cumulative recall")
    ax.set_title("Survivor Discovery Over Time", fontsize=11)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)


def _plot_per_drone_bars(
    ax: Any,
    summary: dict[str, Any],
    key: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
) -> None:
    strategies = list(summary["strategies"])
    drone_indices = sorted({
        int(row["drone"])
        for strategy in strategies
        for row in summary["by_strategy"][strategy].get("per_drone", [])
    })
    if not drone_indices:
        ax.text(0.5, 0.5, "no drones", ha="center", va="center", transform=ax.transAxes)
        return
    width = 0.8 / max(len(strategies), 1)
    x = np.arange(len(drone_indices))
    for idx, strategy in enumerate(strategies):
        by_drone = {
            int(row["drone"]): row
            for row in summary["by_strategy"][strategy].get("per_drone", [])
        }
        values = [float(by_drone.get(drone, {}).get(key, math.nan)) for drone in drone_indices]
        offset = (idx - (len(strategies) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, color=colors[strategy], alpha=0.82, label=strategy)
    ax.set_xticks(x, [f"d{idx}" for idx in drone_indices])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)


def _plot_failure_labels(ax: Any, summary: dict[str, Any], colors: dict[str, str]) -> None:
    strategies = list(summary["strategies"])
    labels = sorted({
        label
        for strategy in strategies
        for label in summary["by_strategy"][strategy].get("label_counts", {})
    })
    if not labels:
        ax.text(0.5, 0.5, "no labels", ha="center", va="center", transform=ax.transAxes)
        return
    width = 0.8 / max(len(strategies), 1)
    x = np.arange(len(labels))
    for idx, strategy in enumerate(strategies):
        counts = summary["by_strategy"][strategy].get("label_counts", {})
        values = [float(counts.get(label, 0)) for label in labels]
        offset = (idx - (len(strategies) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, color=colors[strategy], alpha=0.82, label=strategy)
    ax.set_xticks(x, [_wrap_label(label) for label in labels], rotation=0, ha="center")
    ax.set_ylabel("episodes")
    ax.set_title("Failure Labels", fontsize=11)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)


def _wrap_label(label: str) -> str:
    replacements = {
        "survivor_miss_despite_coverage": "missed survivors\nwith high coverage",
        "wasteful_revisit": "wasteful\nrevisit",
        "partial_search": "partial\nsearch",
        "low_coverage": "low\ncoverage",
        "edge_trap": "edge\ntrap",
    }
    if label in replacements:
        return replacements[label]
    return label.replace("_", "\n") if len(label) > 12 else label


def _fmt_optional(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.1f}"


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['strategy']:>14s} seed {row['seed']:>4}: "
        f"scouted={row['scouted']}/{row['survivors']} "
        f"recall={row['recall']:.3f} "
        f"coverage={row['final_coverage_fraction']:.3f} "
        f"conf={row['final_confidence_mean']:.3f} "
        f"avg_scout={_fmt_optional(row['avg_scout_step'])} steps/"
        f"{_fmt_optional(row['avg_scout_time_s'])}s "
        f"all_scouted={_fmt_optional(row['all_scouted_step'])} steps/"
        f"{_fmt_optional(row['all_scouted_time_s'])}s "
        f"move={row['avg_displacement_m']:.2f}m "
        f"new={row['avg_new_coverage_cells']:.1f} "
        f"new/m={row['new_coverage_cells_per_meter']:.3f} "
        f"conf/m={row['confidence_gain_per_meter']:.6f} "
        f"overlap={row['avg_overlap_fraction']:.2f} "
        f"excess={row['avg_excess_overlap_fraction']:.2f} "
        f"edge={row['edge_step_frac']:.2f} "
        f"moving_nonew={row['moving_no_new_coverage_frac']:.2f} "
        f"moving_noconf={row['moving_no_confidence_gain_frac']:.2f} "
        f"label={row['failure_label']}"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print("-" * 100)
    for strategy in summary["strategies"]:
        item = summary["by_strategy"][strategy]
        print(
            f"{strategy:>14s}: "
            f"episodes={int(item['episodes'])} "
            f"recall={item['mean_recall']:.3f} "
            f"coverage={item['mean_final_coverage_fraction']:.3f} "
            f"confidence={item['mean_final_confidence_mean']:.3f} "
            f"success={item['full_success_rate']:.3f} "
            f"move={item['mean_displacement_m']:.2f}m "
            f"new={item['mean_new_coverage_cells']:.1f} "
            f"new/m={item['mean_new_coverage_cells_per_meter']:.3f} "
            f"conf/m={item['mean_confidence_gain_per_meter']:.6f} "
            f"excess={item['mean_excess_overlap_fraction']:.3f} "
            f"conf_ov={item['mean_confidence_overlap_fraction']:.3f} "
            f"edge={item['mean_edge_step_frac']:.3f} "
            f"moving_nonew={item['mean_moving_no_new_coverage_frac']:.3f} "
            f"moving_noconf={item['mean_moving_no_confidence_gain_frac']:.3f}"
        )
        labels = item.get("label_counts", {})
        if labels:
            print(
                " " * 16
                + "labels="
                + ", ".join(f"{name}:{count}" for name, count in labels.items())
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["lawnmower", "ant_colony"],
        help="Strategies to compare. Use any baseline name, all, happo, or happo:/path/to/models.",
    )
    parser.add_argument("--happo-checkpoint", default=None, help="Checkpoint used when --strategies includes happo.")
    parser.add_argument("--stochastic-happo", action="store_true", help="Sample HAPPO actions instead of actor means.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--n-drones", type=int, default=3)
    parser.add_argument("--n-survivors", type=int, default=5)
    parser.add_argument("--terrain-cache-path", default=str(DEFAULT_TERRAIN_CACHE_PATH))
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Scenario grid size. Defaults to the square land_cover size in --terrain-cache-path.",
    )
    parser.add_argument("--drone-min-footprint-radius-m", type=float, default=0.0)
    parser.add_argument("--uav-start-min-separation-m", type=float, default=150.0)
    parser.add_argument("--uav-start-edge-margin-m", type=float, default=50.0)
    parser.add_argument(
        "--moving-no-confidence-gain-threshold",
        type=float,
        default=DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
        help="Weighted confidence-gain threshold for the moving_no_confidence_gain metric.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--plots-output", default=None)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.n_drones < 1:
        parser.error("--n-drones must be positive")
    if args.n_survivors < 1:
        parser.error("--n-survivors must be positive")
    args.terrain_cache_path = Path(args.terrain_cache_path)
    if not args.terrain_cache_path.is_file():
        parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")
    if args.grid_size is None:
        try:
            args.grid_size = _infer_terrain_cache_grid_size(args.terrain_cache_path)
        except ValueError as exc:
            parser.error(str(exc))
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive")
    if args.uav_start_min_separation_m is not None and args.uav_start_min_separation_m < 0.0:
        parser.error("--uav-start-min-separation-m must be nonnegative")
    if args.uav_start_edge_margin_m is not None and args.uav_start_edge_margin_m < 0.0:
        parser.error("--uav-start-edge-margin-m must be nonnegative")
    if (
        not math.isfinite(float(args.moving_no_confidence_gain_threshold))
        or args.moving_no_confidence_gain_threshold < 0.0
    ):
        parser.error("--moving-no-confidence-gain-threshold must be finite and nonnegative")

    try:
        specs = parse_strategy_specs(args.strategies, happo_checkpoint=args.happo_checkpoint)
    except ValueError as exc:
        parser.error(str(exc))

    scenario_kwargs = build_scenario_kwargs(args, specs)
    print(
        "scenario: "
        f"{scenario_kwargs['n_drones']} UAVs, "
        f"{scenario_kwargs['n_ground']} UGVs, "
        f"{scenario_kwargs['n_survivors']} survivors, "
        f"grid={scenario_kwargs['fire_grid_size']}x{scenario_kwargs['fire_grid_size']}, "
        f"steps={scenario_kwargs['max_steps']}"
    )
    print(f"terrain: {scenario_kwargs.get('terrain_cache_path')}")
    print("strategies: " + ", ".join(spec.label for spec in specs))
    for spec in specs:
        if spec.kind == "happo":
            print(f"  {spec.label} checkpoint: {spec.checkpoint_dir}")
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    print("-" * 100)

    happo_cache: dict[Path, HappoPolicy] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in args.seeds:
            row = run_rollout(
                spec,
                scenario_kwargs,
                seed,
                happo_cache=happo_cache,
                stochastic_happo=args.stochastic_happo,
                moving_no_confidence_gain_threshold=args.moving_no_confidence_gain_threshold,
            )
            rows.append(row)
            _print_row(row)

    summary = summarize(rows)
    _print_summary(summary)

    payload = {
        "metadata": {
            "strategies": [spec.__dict__ | {"checkpoint_dir": None if spec.checkpoint_dir is None else str(spec.checkpoint_dir)} for spec in specs],
            "steps": int(args.steps),
            "seeds": [int(seed) for seed in args.seeds],
            "scenario_kwargs": scenario_kwargs,
            "moving_no_confidence_gain_threshold": float(args.moving_no_confidence_gain_threshold),
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
        write_distribution_plots(rows, summary, args.plots_output)
        print(f"wrote plots: {args.plots_output}")


if __name__ == "__main__":
    main()
