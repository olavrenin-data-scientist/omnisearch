"""
Diagnose HAPPO UAV-only survivor scouting checkpoints.

This is intentionally focused on the UAV survivor diagnostic task:
one or more drones, no UGVs, five survivors, no fire, and drone scouting
counts as mission success. The first metrics are recall-oriented: how many
survivors are scouted, how many are missed, and how long scouting takes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import vmas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario


def _checkpoint_path(path: str | None) -> Path:
    return Path(path) if path else find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _scenario_kwargs(checkpoint_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(checkpoint_dir)
    scenario_kwargs: dict[str, Any] = {}
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
        "max_steps": args.steps,
        "n_ground": 0,
        "n_survivors": 5,
        "known_survivors_at_reset": False,
        "drone_can_confirm": True,
        "disable_fire": True,
        "comms_dropout": 0.0,
    })
    if args.n_drones is not None:
        scenario_kwargs["n_drones"] = int(args.n_drones)
    else:
        scenario_kwargs.setdefault("n_drones", 1)

    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if args.local_map_patch_size is not None:
        scenario_kwargs["local_map_patch_size"] = int(args.local_map_patch_size)
    if args.drone_min_footprint_radius_m is not None:
        scenario_kwargs.pop("drone_min_footprint", None)
        scenario_kwargs["drone_min_footprint_m"] = max(float(args.drone_min_footprint_radius_m), 0.0)
    if getattr(args, "uav_start_min_separation_m", None) is not None:
        scenario_kwargs["uav_start_min_separation_m"] = max(float(args.uav_start_min_separation_m), 0.0)
    if getattr(args, "uav_start_edge_margin_m", None) is not None:
        scenario_kwargs["uav_start_edge_margin_m"] = max(float(args.uav_start_edge_margin_m), 0.0)
    return scenario_kwargs


def run_rollout(policy: HappoPolicy, scenario_kwargs: dict[str, Any], seed: int) -> dict[str, Any]:
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset()
    policy.reset()
    scenario = env.scenario
    start_metrics = _start_metrics(scenario)

    n_survivors = int(scenario.n_survivors)
    first_scout_steps: list[int | None] = [None] * n_survivors
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    action_norms: list[float] = []
    displacement_m_values: list[float] = []
    action_displacement_alignments: list[float] = []
    action_displacement_alignments_new_cov: list[float] = []
    action_displacement_alignments_no_new_cov: list[float] = []
    new_coverage_cells_values: list[float] = []
    outside_footprint_values: list[float] = []
    overlap_values: list[float] = []
    expected_overlap_values: list[float] = []
    excess_overlap_values: list[float] = []
    boundary_distance_m_values: list[float] = []
    footprint_radius_m_values: list[float] = []
    path_positions_sim: list[np.ndarray] = []
    low_action_high_motion = 0
    high_action_low_motion = 0
    moving_no_new_coverage = 0
    diagnostic_steps = 0

    for step in range(int(scenario_kwargs["max_steps"])):
        pre_drone_pos = [
            agent.state.pos[0].detach().cpu().numpy().astype(float).copy()
            for agent in scenario.world.agents[:scenario.n_drones]
        ]
        path_positions_sim.extend(pre_drone_pos)
        actions = policy(env)
        action_vectors = [
            actions[drone_idx][0].detach().cpu().numpy().astype(float).copy()
            for drone_idx in range(scenario.n_drones)
        ]
        env.step(actions)
        meters_per_sim = 1.0 / max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)
        coverage_cells = _metric_array(
            scenario,
            "metric_uav_new_coverage_cells",
            scenario.n_drones,
        )
        outside_footprint_fraction = _metric_array(
            scenario,
            "metric_uav_outside_footprint_fraction",
            scenario.n_drones,
        )
        overlap_fraction = _metric_array(
            scenario,
            "metric_uav_overlap_fraction_by_drone",
            scenario.n_drones,
        )
        expected_overlap_fraction = _metric_array(
            scenario,
            "metric_uav_expected_overlap_fraction_by_drone",
            scenario.n_drones,
        )
        excess_overlap_fraction = _metric_array(
            scenario,
            "metric_uav_excess_overlap_fraction_by_drone",
            scenario.n_drones,
        )
        boundary_distance_m = _metric_array(
            scenario,
            "metric_uav_boundary_distance_m",
            scenario.n_drones,
        )
        footprint_radius_m = _metric_array(
            scenario,
            "metric_uav_footprint_radius_m",
            scenario.n_drones,
        )
        for drone_idx, action_vec in enumerate(action_vectors):
            post_pos = scenario.world.agents[drone_idx].state.pos[0].detach().cpu().numpy().astype(float)
            path_positions_sim.append(post_pos.copy())
            displacement_vec = post_pos - pre_drone_pos[drone_idx]
            action_norm = float(np.linalg.norm(action_vec))
            displacement_m = float(np.linalg.norm(displacement_vec) * meters_per_sim)
            new_cells = float(coverage_cells[drone_idx])
            action_norms.append(action_norm)
            displacement_m_values.append(displacement_m)
            new_coverage_cells_values.append(new_cells)
            outside_footprint_values.append(float(outside_footprint_fraction[drone_idx]))
            overlap = float(overlap_fraction[drone_idx])
            footprint_radius = float(footprint_radius_m[drone_idx])
            expected_overlap = float(expected_overlap_fraction[drone_idx])
            excess_overlap = float(excess_overlap_fraction[drone_idx])
            overlap_values.append(overlap)
            expected_overlap_values.append(expected_overlap)
            excess_overlap_values.append(excess_overlap)
            boundary_distance_m_values.append(float(boundary_distance_m[drone_idx]))
            footprint_radius_m_values.append(footprint_radius)
            diagnostic_steps += 1

            if action_norm < 0.05 and displacement_m > 1.0:
                low_action_high_motion += 1
            if action_norm > 0.5 and displacement_m < 0.25:
                high_action_low_motion += 1
            if displacement_m > 1.0 and new_cells < 1.0:
                moving_no_new_coverage += 1

            displacement_norm_sim = float(np.linalg.norm(displacement_vec))
            if action_norm > 1e-6 and displacement_norm_sim > 1e-9:
                alignment = float(np.dot(action_vec[:2], displacement_vec[:2]) / (action_norm * displacement_norm_sim))
                alignment = max(min(alignment, 1.0), -1.0)
                action_displacement_alignments.append(alignment)
                if new_cells >= 1.0:
                    action_displacement_alignments_new_cov.append(alignment)
                else:
                    action_displacement_alignments_no_new_cov.append(alignment)

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
    path_metrics = _path_metrics(
        path_positions_sim,
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
    row = {
        "seed": int(seed),
        "survivors": n_survivors,
        "scouted": scouted_count,
        "missed": missed_count,
        "recall": scouted_count / n_survivors if n_survivors else 0.0,
        "final_coverage_fraction": final_coverage_fraction,
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
        "avg_action_norm": _finite_mean(action_norms),
        "avg_displacement_m": _finite_mean(displacement_m_values),
        "avg_action_displacement_alignment": _finite_mean(action_displacement_alignments),
        "avg_action_displacement_alignment_new_cov": _finite_mean(action_displacement_alignments_new_cov),
        "avg_action_displacement_alignment_no_new_cov": _finite_mean(action_displacement_alignments_no_new_cov),
        "avg_new_coverage_cells": _finite_mean(new_coverage_cells_values),
        "avg_outside_footprint_fraction": _finite_mean(outside_footprint_values),
        "max_outside_footprint_fraction": max(outside_footprint_values) if outside_footprint_values else 0.0,
        "outside_footprint_step_frac_10": (
            float(np.mean([value >= 0.10 for value in outside_footprint_values]))
            if outside_footprint_values else 0.0
        ),
        "avg_overlap_fraction": _finite_mean(overlap_values),
        "avg_expected_overlap_fraction": _finite_mean(expected_overlap_values),
        "avg_excess_overlap_fraction": _finite_mean(excess_overlap_values),
        "excess_overlap_step_frac_10": (
            float(np.mean([value >= 0.10 for value in excess_overlap_values]))
            if excess_overlap_values else 0.0
        ),
        "excess_overlap_step_frac_20": (
            float(np.mean([value >= 0.20 for value in excess_overlap_values]))
            if excess_overlap_values else 0.0
        ),
        "overlap_step_frac_60": (
            float(np.mean([value >= 0.60 for value in overlap_values]))
            if overlap_values else 0.0
        ),
        "new_coverage_step_frac": (
            float(np.mean([value >= 1.0 for value in new_coverage_cells_values]))
            if new_coverage_cells_values else 0.0
        ),
        "low_action_high_motion_frac": low_action_high_motion / diagnostic_steps if diagnostic_steps else 0.0,
        "high_action_low_motion_frac": high_action_low_motion / diagnostic_steps if diagnostic_steps else 0.0,
        "moving_no_new_coverage_frac": moving_no_new_coverage / diagnostic_steps if diagnostic_steps else 0.0,
        **start_metrics,
        **path_metrics,
        **coverage_shape_metrics,
    }
    row["failure_label"] = _failure_label(row)
    close = getattr(env, "close", None)
    if close is not None:
        close()
    return row


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _metric_array(scenario: WildfireSearchScenario, name: str, n_drones: int) -> np.ndarray:
    value = getattr(scenario, name, None)
    if value is None or n_drones <= 0:
        return np.zeros(max(n_drones, 1), dtype=float)
    arr = value.detach().cpu().numpy().astype(float).reshape(-1)
    if arr.size == n_drones:
        return arr
    if arr.size == 1:
        return np.repeat(arr, n_drones)
    return np.resize(arr, n_drones)


def _start_metrics(scenario: WildfireSearchScenario) -> dict[str, Any]:
    n_drones = int(getattr(scenario, "n_drones", 0))
    if n_drones <= 0:
        return {
            "start_positions_m": [],
            "min_start_pair_distance_m": math.nan,
            "mean_start_pair_distance_m": math.nan,
            "min_start_edge_distance_m": math.nan,
            "mean_start_edge_distance_m": math.nan,
            "start_map_width_m": math.nan,
            "start_map_height_m": math.nan,
        }

    positions_sim = np.asarray([
        agent.state.pos[0].detach().cpu().numpy().astype(float)
        for agent in scenario.world.agents[:n_drones]
    ])
    meters_per_sim = 1.0 / max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)
    positions_m = positions_sim * meters_per_sim
    if n_drones > 1:
        deltas = positions_m[:, None, :] - positions_m[None, :, :]
        pairwise = np.linalg.norm(deltas, axis=-1)
        pairwise = pairwise[np.triu_indices(n_drones, k=1)]
    else:
        pairwise = np.asarray([], dtype=float)

    x_half_m = float(scenario.x_semidim) * meters_per_sim
    y_half_m = float(scenario.y_semidim) * meters_per_sim
    edge_distances = np.stack(
        [
            positions_m[:, 0] + x_half_m,
            x_half_m - positions_m[:, 0],
            positions_m[:, 1] + y_half_m,
            y_half_m - positions_m[:, 1],
        ],
        axis=1,
    ).clip(min=0.0)
    nearest_edge = edge_distances.min(axis=1)

    return {
        "start_positions_m": positions_m.round(6).tolist(),
        "min_start_pair_distance_m": float(pairwise.min()) if pairwise.size else math.nan,
        "mean_start_pair_distance_m": float(pairwise.mean()) if pairwise.size else math.nan,
        "min_start_edge_distance_m": float(nearest_edge.min()) if nearest_edge.size else math.nan,
        "mean_start_edge_distance_m": float(nearest_edge.mean()) if nearest_edge.size else math.nan,
        "start_map_width_m": float(2.0 * x_half_m),
        "start_map_height_m": float(2.0 * y_half_m),
    }


def _path_metrics(
    positions_sim: list[np.ndarray],
    displacement_m_values: list[float],
    boundary_distance_m_values: list[float],
    footprint_radius_m_values: list[float],
    scenario: WildfireSearchScenario,
) -> dict[str, float]:
    if not positions_sim:
        return {
            "path_bbox_area_fraction": 0.0,
            "path_bbox_width_m": 0.0,
            "path_bbox_height_m": 0.0,
            "path_net_displacement_m": 0.0,
            "path_length_m": 0.0,
            "mean_boundary_distance_m": math.nan,
            "min_boundary_distance_m": math.nan,
            "edge_step_frac": 0.0,
            "corner_step_frac": 0.0,
            "end_boundary_distance_m": math.nan,
            "end_near_edge": 0.0,
            "end_near_corner": 0.0,
            "stalled_step_frac": 0.0,
            "longest_stall_steps": 0.0,
        }

    positions = np.asarray(positions_sim, dtype=float)
    meters_per_sim = 1.0 / max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)
    x_span_m = max(float(positions[:, 0].max() - positions[:, 0].min()) * meters_per_sim, 0.0)
    y_span_m = max(float(positions[:, 1].max() - positions[:, 1].min()) * meters_per_sim, 0.0)
    map_width_m = max(2.0 * float(scenario.x_semidim) * meters_per_sim, 1e-9)
    map_height_m = max(2.0 * float(scenario.y_semidim) * meters_per_sim, 1e-9)
    boundary = np.asarray([v for v in boundary_distance_m_values if math.isfinite(v)], dtype=float)
    footprint = _finite_mean(footprint_radius_m_values)
    edge_threshold_m = footprint if math.isfinite(footprint) and footprint > 0.0 else 25.0
    corner_threshold_m = edge_threshold_m
    boundary_array = boundary if boundary.size else np.asarray([], dtype=float)
    edge_step_frac = float(np.mean(boundary_array <= edge_threshold_m)) if boundary_array.size else 0.0

    distances_to_edges_m = _distances_to_edges_m(positions, scenario, meters_per_sim)
    two_near_edges = (distances_to_edges_m <= corner_threshold_m).sum(axis=1) >= 2
    corner_step_frac = float(np.mean(two_near_edges)) if two_near_edges.size else 0.0
    end_distances = distances_to_edges_m[-1]
    end_boundary_distance_m = float(end_distances.min())
    end_near_edge = float(end_boundary_distance_m <= edge_threshold_m)
    end_near_corner = float((end_distances <= corner_threshold_m).sum() >= 2)

    displacement = np.asarray(displacement_m_values, dtype=float)
    stalled = displacement < 0.5
    longest_stall = _longest_true_run(stalled)
    return {
        "path_bbox_area_fraction": float((x_span_m * y_span_m) / (map_width_m * map_height_m)),
        "path_bbox_width_m": x_span_m,
        "path_bbox_height_m": y_span_m,
        "path_net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0]) * meters_per_sim),
        "path_length_m": float(displacement.sum()) if displacement.size else 0.0,
        "mean_boundary_distance_m": float(boundary_array.mean()) if boundary_array.size else math.nan,
        "min_boundary_distance_m": float(boundary_array.min()) if boundary_array.size else math.nan,
        "edge_step_frac": edge_step_frac,
        "corner_step_frac": corner_step_frac,
        "end_boundary_distance_m": end_boundary_distance_m,
        "end_near_edge": end_near_edge,
        "end_near_corner": end_near_corner,
        "stalled_step_frac": float(np.mean(stalled)) if stalled.size else 0.0,
        "longest_stall_steps": float(longest_stall),
    }


def _distances_to_edges_m(
    positions_sim: np.ndarray,
    scenario: WildfireSearchScenario,
    meters_per_sim: float,
) -> np.ndarray:
    x_min = -float(scenario.x_semidim) + float(scenario.agent_radius)
    x_max = float(scenario.x_semidim) - float(scenario.agent_radius)
    y_min = -float(scenario.y_semidim) + float(scenario.agent_radius)
    y_max = float(scenario.y_semidim) - float(scenario.agent_radius)
    return np.stack(
        [
            (positions_sim[:, 0] - x_min) * meters_per_sim,
            (x_max - positions_sim[:, 0]) * meters_per_sim,
            (positions_sim[:, 1] - y_min) * meters_per_sim,
            (y_max - positions_sim[:, 1]) * meters_per_sim,
        ],
        axis=1,
    ).clip(min=0.0)


def _coverage_shape_metrics(
    coverage_grid: np.ndarray,
    scenario: WildfireSearchScenario,
    footprint_radius_m: float,
) -> dict[str, float]:
    covered = coverage_grid.astype(bool)
    size_y, size_x = covered.shape
    if not covered.any():
        return {
            "coverage_bbox_area_fraction": 0.0,
            "coverage_bbox_fill_fraction": 0.0,
            "coverage_center_fraction": 0.0,
            "coverage_border_band_fraction": 0.0,
            "coverage_interior_fraction": 0.0,
            "coverage_edge_bias": 0.0,
        }

    yy, xx = np.nonzero(covered)
    bbox_area = float((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1))
    total_cells = float(size_x * size_y)
    bbox_fill = float(covered.sum() / max(bbox_area, 1.0))

    center_margin_x = size_x // 4
    center_margin_y = size_y // 4
    center = covered[
        center_margin_y : size_y - center_margin_y,
        center_margin_x : size_x - center_margin_x,
    ]
    center_fraction = float(center.mean()) if center.size else 0.0

    meters_per_sim = 1.0 / max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)
    cell_width_m = (2.0 * float(scenario.x_semidim) * meters_per_sim) / max(size_x, 1)
    cell_height_m = (2.0 * float(scenario.y_semidim) * meters_per_sim) / max(size_y, 1)
    band_m = footprint_radius_m if math.isfinite(footprint_radius_m) and footprint_radius_m > 0.0 else 25.0
    band_x = max(int(math.ceil(band_m / max(cell_width_m, 1e-9))), 1)
    band_y = max(int(math.ceil(band_m / max(cell_height_m, 1e-9))), 1)
    border_mask = np.zeros_like(covered, dtype=bool)
    border_mask[:band_y, :] = True
    border_mask[-band_y:, :] = True
    border_mask[:, :band_x] = True
    border_mask[:, -band_x:] = True
    interior_mask = ~border_mask
    border_fraction = float(covered[border_mask].mean()) if border_mask.any() else 0.0
    interior_fraction = float(covered[interior_mask].mean()) if interior_mask.any() else 0.0

    return {
        "coverage_bbox_area_fraction": float(bbox_area / total_cells),
        "coverage_bbox_fill_fraction": bbox_fill,
        "coverage_center_fraction": center_fraction,
        "coverage_border_band_fraction": border_fraction,
        "coverage_interior_fraction": interior_fraction,
        "coverage_edge_bias": border_fraction - interior_fraction,
    }


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _failure_label(row: dict[str, Any]) -> str:
    if row["full_success"]:
        return "success"
    if row["end_near_corner"] and row["edge_step_frac"] > 0.35:
        return "corner_trap"
    if row["edge_step_frac"] > 0.55 and row["coverage_center_fraction"] < 0.25:
        return "edge_loop"
    if row["avg_outside_footprint_fraction"] > 0.15:
        return "outside_footprint_waste"
    if (
        row["moving_no_new_coverage_frac"] > 0.30
        or (
            row["avg_excess_overlap_fraction"] > 0.20
            and row["new_coverage_step_frac"] < 0.75
        )
        or (
            row["excess_overlap_step_frac_20"] > 0.30
            and row["moving_no_new_coverage_frac"] > 0.15
        )
    ):
        return "wasteful_revisit"
    if row["path_bbox_area_fraction"] < 0.10:
        return "small_search_area"
    if row["stalled_step_frac"] > 0.20 or row["longest_stall_steps"] >= 25:
        return "stalled"
    if row["new_coverage_step_frac"] >= 0.85 and row["avg_excess_overlap_fraction"] <= 0.15:
        return "productive_sweep"
    if row["final_coverage_fraction"] >= 0.45 and row["scouted"] <= 1:
        return "survivor_miss_despite_coverage"
    return "partial_search"


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    successful = [row for row in rows if row["all_scouted_step"] is not None]
    summary = {
        "episodes": float(len(rows)),
        "mean_scouted": float(np.mean([row["scouted"] for row in rows])) if rows else 0.0,
        "mean_missed": float(np.mean([row["missed"] for row in rows])) if rows else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
        "mean_final_coverage_fraction": (
            float(np.mean([row["final_coverage_fraction"] for row in rows])) if rows else 0.0
        ),
        "full_success_rate": float(np.mean([row["full_success"] for row in rows])) if rows else 0.0,
        "mean_avg_scout_step": _finite_mean([row["avg_scout_step"] for row in rows]),
        "mean_avg_scout_time_s": _finite_mean([row["avg_scout_time_s"] for row in rows]),
        "mean_all_scouted_step_successes": (
            float(np.mean([row["all_scouted_step"] for row in successful])) if successful else math.nan
        ),
        "mean_all_scouted_time_s_successes": (
            float(np.mean([row["all_scouted_time_s"] for row in successful])) if successful else math.nan
        ),
        "mean_action_norm": _finite_mean([row["avg_action_norm"] for row in rows]),
        "mean_displacement_m": _finite_mean([row["avg_displacement_m"] for row in rows]),
        "mean_action_displacement_alignment": _finite_mean([
            row["avg_action_displacement_alignment"] for row in rows
        ]),
        "mean_action_displacement_alignment_new_cov": _finite_mean([
            row["avg_action_displacement_alignment_new_cov"] for row in rows
        ]),
        "mean_action_displacement_alignment_no_new_cov": _finite_mean([
            row["avg_action_displacement_alignment_no_new_cov"] for row in rows
        ]),
        "mean_new_coverage_cells": _finite_mean([row["avg_new_coverage_cells"] for row in rows]),
        "mean_outside_footprint_fraction": _finite_mean([
            row["avg_outside_footprint_fraction"] for row in rows
        ]),
        "mean_outside_footprint_step_frac_10": _finite_mean([
            row["outside_footprint_step_frac_10"] for row in rows
        ]),
        "mean_overlap_fraction": _finite_mean([row["avg_overlap_fraction"] for row in rows]),
        "mean_expected_overlap_fraction": _finite_mean([
            row["avg_expected_overlap_fraction"] for row in rows
        ]),
        "mean_excess_overlap_fraction": _finite_mean([
            row["avg_excess_overlap_fraction"] for row in rows
        ]),
        "mean_excess_overlap_step_frac_10": _finite_mean([
            row["excess_overlap_step_frac_10"] for row in rows
        ]),
        "mean_excess_overlap_step_frac_20": _finite_mean([
            row["excess_overlap_step_frac_20"] for row in rows
        ]),
        "mean_overlap_step_frac_60": _finite_mean([row["overlap_step_frac_60"] for row in rows]),
        "mean_new_coverage_step_frac": _finite_mean([row["new_coverage_step_frac"] for row in rows]),
        "mean_low_action_high_motion_frac": _finite_mean([
            row["low_action_high_motion_frac"] for row in rows
        ]),
        "mean_high_action_low_motion_frac": _finite_mean([
            row["high_action_low_motion_frac"] for row in rows
        ]),
        "mean_moving_no_new_coverage_frac": _finite_mean([
            row["moving_no_new_coverage_frac"] for row in rows
        ]),
        "mean_min_start_pair_distance_m": _finite_mean([
            row["min_start_pair_distance_m"] for row in rows
        ]),
        "mean_min_start_edge_distance_m": _finite_mean([
            row["min_start_edge_distance_m"] for row in rows
        ]),
        "mean_path_bbox_area_fraction": _finite_mean([row["path_bbox_area_fraction"] for row in rows]),
        "mean_path_length_m": _finite_mean([row["path_length_m"] for row in rows]),
        "mean_boundary_distance_m": _finite_mean([row["mean_boundary_distance_m"] for row in rows]),
        "mean_edge_step_frac": _finite_mean([row["edge_step_frac"] for row in rows]),
        "mean_corner_step_frac": _finite_mean([row["corner_step_frac"] for row in rows]),
        "mean_stalled_step_frac": _finite_mean([row["stalled_step_frac"] for row in rows]),
        "mean_longest_stall_steps": _finite_mean([row["longest_stall_steps"] for row in rows]),
        "mean_coverage_bbox_fill_fraction": _finite_mean([
            row["coverage_bbox_fill_fraction"] for row in rows
        ]),
        "mean_coverage_center_fraction": _finite_mean([
            row["coverage_center_fraction"] for row in rows
        ]),
        "mean_coverage_border_band_fraction": _finite_mean([
            row["coverage_border_band_fraction"] for row in rows
        ]),
        "mean_coverage_interior_fraction": _finite_mean([
            row["coverage_interior_fraction"] for row in rows
        ]),
        "mean_coverage_edge_bias": _finite_mean([row["coverage_edge_bias"] for row in rows]),
    }
    summary.update(_distribution_summary(rows))
    return summary


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {
        "recall": "recall",
        "coverage": "final_coverage_fraction",
        "move_m": "avg_displacement_m",
        "new_cells": "avg_new_coverage_cells",
        "outside": "avg_outside_footprint_fraction",
        "overlap": "avg_overlap_fraction",
        "expected_overlap": "avg_expected_overlap_fraction",
        "excess_overlap": "avg_excess_overlap_fraction",
        "edge_frac": "edge_step_frac",
        "corner_frac": "corner_step_frac",
        "center_cov": "coverage_center_fraction",
        "moving_no_new": "moving_no_new_coverage_frac",
        "start_pair": "min_start_pair_distance_m",
        "start_edge": "min_start_edge_distance_m",
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


def _fmt_optional(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("failure_label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def write_distribution_plots(
    rows: list[dict[str, Any]],
    summary: dict[str, float],
    label_counts: dict[str, int],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Recall", "recall", (0.0, 1.0)),
        ("Final Coverage", "final_coverage_fraction", (0.0, 1.0)),
        ("Movement / Step (m)", "avg_displacement_m", None),
        ("New Cells / Step", "avg_new_coverage_cells", None),
        ("Outside Footprint", "avg_outside_footprint_fraction", (0.0, 1.0)),
        ("Overlap", "avg_overlap_fraction", (0.0, 1.0)),
        ("Excess Overlap", "avg_excess_overlap_fraction", (0.0, 1.0)),
        ("Edge Step Fraction", "edge_step_frac", (0.0, 1.0)),
        ("Corner Step Fraction", "corner_step_frac", (0.0, 1.0)),
        ("Center Coverage", "coverage_center_fraction", (0.0, 1.0)),
        ("Moving No New Coverage", "moving_no_new_coverage_frac", (0.0, 1.0)),
        ("Start Pair Min (m)", "min_start_pair_distance_m", None),
        ("Start Edge Min (m)", "min_start_edge_distance_m", None),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(14, 15), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (title, key, xlim) in zip(axes_flat, panels):
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            bins = min(max(len(values) // 2, 5), 20)
            ax.hist(values, bins=bins, color="#4f7cff", alpha=0.82, edgecolor="#1e2b4f")
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            ax.axvline(mean_value, color="#d44a3a", linewidth=1.4, label=f"mean {mean_value:.2f}")
            ax.axvline(median_value, color="#20242c", linewidth=1.1, linestyle="--", label=f"med {median_value:.2f}")
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("episodes")
        ax.grid(axis="y", alpha=0.25)

    heat_ax = axes_flat[-2]
    heat_ax.clear()
    starts = [
        position
        for row in rows
        for position in row.get("start_positions_m", [])
        if len(position) >= 2
    ]
    if starts:
        start_array = np.asarray(starts, dtype=float)
        map_width = _finite_mean([float(row.get("start_map_width_m", math.nan)) for row in rows])
        map_height = _finite_mean([float(row.get("start_map_height_m", math.nan)) for row in rows])
        if not math.isfinite(map_width):
            map_width = max(float(np.abs(start_array[:, 0]).max()) * 2.0, 1.0)
        if not math.isfinite(map_height):
            map_height = max(float(np.abs(start_array[:, 1]).max()) * 2.0, 1.0)
        heat = heat_ax.hist2d(
            start_array[:, 0],
            start_array[:, 1],
            bins=12,
            range=[
                [-0.5 * map_width, 0.5 * map_width],
                [-0.5 * map_height, 0.5 * map_height],
            ],
            cmap="Blues",
        )
        fig.colorbar(heat[3], ax=heat_ax, fraction=0.046, pad=0.04)
        heat_ax.set_aspect("equal", adjustable="box")
        heat_ax.set_xlabel("x start (m)")
        heat_ax.set_ylabel("y start (m)")
        heat_ax.set_title("UAV Start Heatmap", fontsize=10)
    else:
        heat_ax.text(0.5, 0.5, "no starts", ha="center", va="center", transform=heat_ax.transAxes)
        heat_ax.set_title("UAV Start Heatmap", fontsize=10)

    label_ax = axes_flat[-1]
    label_ax.clear()
    if label_counts:
        labels = list(label_counts.keys())
        counts = [label_counts[label] for label in labels]
        y = np.arange(len(labels))
        label_ax.barh(y, counts, color="#36a269", alpha=0.85)
        label_ax.set_yticks(y, labels)
        label_ax.invert_yaxis()
        label_ax.set_xlabel("episodes")
        label_ax.set_title("Failure Labels", fontsize=10)
        for idx, count in enumerate(counts):
            label_ax.text(count + 0.05, idx, str(count), va="center", fontsize=9)
    else:
        label_ax.text(0.5, 0.5, "no labels", ha="center", va="center", transform=label_ax.transAxes)
    label_ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "UAV HAPPO Diagnostics Distributions "
        f"(n={int(summary.get('episodes', len(rows)))})",
        fontsize=14,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=None, help="Path to a HARL models/ checkpoint directory.")
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--n-drones", type=int, default=None,
                        help="Override UAV count for legacy checkpoints. Default preserves the checkpoint manifest.")
    parser.add_argument("--local-map-patch-size", type=int, default=None)
    parser.add_argument("--drone-min-footprint-radius-m", type=float, default=None)
    parser.add_argument("--uav-start-min-separation-m", type=float, default=None,
                        help="Override checkpoint UAV start min separation in meters; pass 0 to disable.")
    parser.add_argument("--uav-start-edge-margin-m", type=float, default=None,
                        help="Override checkpoint UAV start edge margin in meters; pass 0 to disable.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actor means.")
    parser.add_argument("--json-output", default=None, help="Optional path to write per-seed rows and summary as JSON.")
    parser.add_argument("--plots-output", default=None, help="Optional path to write histogram diagnostics as a PNG.")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.n_drones is not None and args.n_drones < 1:
        parser.error("--n-drones must be positive")
    if args.local_map_patch_size is not None and (args.local_map_patch_size < 1 or args.local_map_patch_size % 2 != 1):
        parser.error("--local-map-patch-size must be a positive odd integer")
    if args.uav_start_min_separation_m is not None and args.uav_start_min_separation_m < 0.0:
        parser.error("--uav-start-min-separation-m must be nonnegative")
    if args.uav_start_edge_margin_m is not None and args.uav_start_edge_margin_m < 0.0:
        parser.error("--uav-start-edge-margin-m must be nonnegative")
    if args.terrain_cache_path is not None and not Path(args.terrain_cache_path).is_file():
        parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")

    checkpoint_dir = _checkpoint_path(args.checkpoint_dir)
    scenario_kwargs = _scenario_kwargs(checkpoint_dir, args)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"steps: {args.steps}")
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    print(
        "scenario: "
        f"{scenario_kwargs['n_drones']} UAVs, "
        f"{scenario_kwargs['n_ground']} UGVs, "
        f"{scenario_kwargs['n_survivors']} survivors, "
        f"dt={scenario_kwargs.get('sim_step_seconds', 'scenario-default')}s"
    )
    print(
        "uav starts: "
        f"min_sep={scenario_kwargs.get('uav_start_min_separation_m', 0.0)}m "
        f"edge_margin={scenario_kwargs.get('uav_start_edge_margin_m', 0.0)}m"
    )
    print("-" * 88)

    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=not args.stochastic)
    expected_agents = int(scenario_kwargs["n_drones"]) + int(scenario_kwargs["n_ground"])
    if len(policy.actors) != expected_agents:
        parser.error(
            f"checkpoint contains {len(policy.actors)} actors, but diagnostics scenario "
            f"contains {expected_agents} agents; use the checkpoint manifest settings or "
            "a matching --n-drones override for legacy checkpoints"
        )
    rows = [
        run_rollout(policy, scenario_kwargs, seed)
        for seed in args.seeds
    ]
    for row in rows:
        print(
            f"seed {row['seed']:>4}: "
            f"scouted={row['scouted']}/{row['survivors']} "
            f"missed={row['missed']} "
            f"recall={row['recall']:.3f} "
            f"coverage={row['final_coverage_fraction']:.3f} "
            f"avg_scout={_fmt_optional(row['avg_scout_step'])} steps/"
            f"{_fmt_optional(row['avg_scout_time_s'])}s "
            f"all_scouted={_fmt_optional(row['all_scouted_step'])} steps/"
            f"{_fmt_optional(row['all_scouted_time_s'])}s "
            f"act={row['avg_action_norm']:.3f} "
            f"move={row['avg_displacement_m']:.2f}m "
            f"align={row['avg_action_displacement_alignment']:.3f} "
            f"new_cells={row['avg_new_coverage_cells']:.1f} "
            f"new_frac={row['new_coverage_step_frac']:.2f} "
            f"edge={row['edge_step_frac']:.2f} "
            f"corner={row['corner_step_frac']:.2f} "
            f"outside={row['avg_outside_footprint_fraction']:.2f} "
            f"overlap={row['avg_overlap_fraction']:.2f} "
            f"exp_ov={row['avg_expected_overlap_fraction']:.2f} "
            f"excess_ov={row['avg_excess_overlap_fraction']:.2f} "
            f"center_cov={row['coverage_center_fraction']:.2f} "
            f"start_pair={_fmt_optional(row['min_start_pair_distance_m'])}m "
            f"start_edge={_fmt_optional(row['min_start_edge_distance_m'])}m "
            f"label={row['failure_label']} "
            f"first_steps={row['first_scout_steps']}"
        )

    summary = summarize(rows)
    label_counts = _label_counts(rows)
    print("-" * 88)
    print(
        "means: "
        f"scouted={summary['mean_scouted']:.3f} "
        f"missed={summary['mean_missed']:.3f} "
        f"recall={summary['mean_recall']:.3f} "
        f"coverage={summary['mean_final_coverage_fraction']:.3f} "
        f"success={summary['full_success_rate']:.3f} "
        f"avg_scout={summary['mean_avg_scout_step']:.1f} steps/"
        f"{summary['mean_avg_scout_time_s']:.1f}s "
        f"all_scouted_successes={summary['mean_all_scouted_step_successes']:.1f} steps/"
        f"{summary['mean_all_scouted_time_s_successes']:.1f}s"
    )
    print(
        "action/motion means: "
        f"act_norm={summary['mean_action_norm']:.3f} "
        f"move={summary['mean_displacement_m']:.2f}m "
        f"align={summary['mean_action_displacement_alignment']:.3f} "
        f"align_new={summary['mean_action_displacement_alignment_new_cov']:.3f} "
        f"align_nonew={summary['mean_action_displacement_alignment_no_new_cov']:.3f} "
        f"new_cells={summary['mean_new_coverage_cells']:.1f} "
        f"new_step_frac={summary['mean_new_coverage_step_frac']:.3f}"
    )
    print(
        "footprint/revisit means: "
        f"outside={summary['mean_outside_footprint_fraction']:.3f} "
        f"outside10={summary['mean_outside_footprint_step_frac_10']:.3f} "
        f"overlap={summary['mean_overlap_fraction']:.3f} "
        f"expected_overlap={summary['mean_expected_overlap_fraction']:.3f} "
        f"excess_overlap={summary['mean_excess_overlap_fraction']:.3f} "
        f"excess10={summary['mean_excess_overlap_step_frac_10']:.3f} "
        f"excess20={summary['mean_excess_overlap_step_frac_20']:.3f} "
        f"overlap60={summary['mean_overlap_step_frac_60']:.3f}"
    )
    print(
        "path/edge means: "
        f"path_len={summary['mean_path_length_m']:.1f}m "
        f"bbox_area={summary['mean_path_bbox_area_fraction']:.3f} "
        f"edge_frac={summary['mean_edge_step_frac']:.3f} "
        f"corner_frac={summary['mean_corner_step_frac']:.3f} "
        f"boundary_dist={summary['mean_boundary_distance_m']:.1f}m "
        f"stall_frac={summary['mean_stalled_step_frac']:.3f} "
        f"longest_stall={summary['mean_longest_stall_steps']:.1f} steps"
    )
    print(
        "coverage-shape means: "
        f"bbox_fill={summary['mean_coverage_bbox_fill_fraction']:.3f} "
        f"center={summary['mean_coverage_center_fraction']:.3f} "
        f"border_band={summary['mean_coverage_border_band_fraction']:.3f} "
        f"interior={summary['mean_coverage_interior_fraction']:.3f} "
        f"edge_bias={summary['mean_coverage_edge_bias']:.3f}"
    )
    print(
        "failure-mode fractions: "
        f"low_action_high_motion={summary['mean_low_action_high_motion_frac']:.3f} "
        f"high_action_low_motion={summary['mean_high_action_low_motion_frac']:.3f} "
        f"moving_no_new_coverage={summary['mean_moving_no_new_coverage_frac']:.3f}"
    )
    print(
        "start means: "
        f"min_pair={summary['mean_min_start_pair_distance_m']:.1f}m "
        f"min_edge={summary['mean_min_start_edge_distance_m']:.1f}m"
    )
    print(
        "failure labels: "
        + ", ".join(f"{label}={count}" for label, count in label_counts.items())
    )
    print(
        "distribution snapshots: "
        f"coverage p25/p50/p75="
        f"{summary['coverage_p25']:.3f}/{summary['coverage_p50']:.3f}/{summary['coverage_p75']:.3f} "
        f"overlap p25/p50/p75="
        f"{summary['overlap_p25']:.3f}/{summary['overlap_p50']:.3f}/{summary['overlap_p75']:.3f} "
        f"excess p25/p50/p75="
        f"{summary['excess_overlap_p25']:.3f}/{summary['excess_overlap_p50']:.3f}/{summary['excess_overlap_p75']:.3f} "
        f"edge p25/p50/p75="
        f"{summary['edge_frac_p25']:.3f}/{summary['edge_frac_p50']:.3f}/{summary['edge_frac_p75']:.3f}"
    )
    print("note: all_scouted_successes averages only episodes that scouted every survivor.")

    if args.json_output:
        output = {
            "checkpoint": str(checkpoint_dir),
            "scenario_kwargs": scenario_kwargs,
            "rows": rows,
            "summary": summary,
            "label_counts": label_counts,
        }
        Path(args.json_output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"wrote: {args.json_output}")

    if args.plots_output:
        write_distribution_plots(rows, summary, label_counts, args.plots_output)
        print(f"wrote plots: {args.plots_output}")


if __name__ == "__main__":
    main()
