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
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario

TIME_BIN_COUNT = 5
COUNTERFACTUAL_CANDIDATE_DIRECTIONS = np.asarray(
    [
        [1.0, 0.0],
        [math.sqrt(0.5), math.sqrt(0.5)],
        [0.0, 1.0],
        [-math.sqrt(0.5), math.sqrt(0.5)],
        [-1.0, 0.0],
        [-math.sqrt(0.5), -math.sqrt(0.5)],
        [0.0, -1.0],
        [math.sqrt(0.5), -math.sqrt(0.5)],
        [0.0, 0.0],
    ],
    dtype=float,
)
CONFIDENCE_REVISIT_THRESHOLD = 0.10
CONFIDENCE_REVISIT_USEFUL_OPPORTUNITY_THRESHOLD = 0.25
CONFIDENCE_REVISIT_WASTEFUL_OPPORTUNITY_THRESHOLD = 0.15
CONFIDENCE_REVISIT_MIN_GAIN = 1e-9
DEFAULT_DIAGNOSTIC_CONFIDENCE_FRONTIER_RADIUS_M = 60.0
DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD = 1e-6


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
        "n_survivors": int(
            args.n_survivors
            if getattr(args, "n_survivors", None) is not None
            else scenario_kwargs.get("n_survivors", 5)
        ),
        "known_survivors_at_reset": False,
        "drone_can_confirm": True,
        "disable_fire": not bool(getattr(args, "enable_fire", False)),
        "comms_dropout": float(
            0.0
            if getattr(args, "comms_dropout", None) is None
            else args.comms_dropout
        ),
        "uav_confidence_diagnostics": True,
        "uav_cleanup_target_diagnostics": (
            str(getattr(args, "diagnostic_level", "full")).replace("-", "_").lower() == "full"
            or bool(getattr(args, "include_cleanup_target_diagnostics", False))
        ),
    })
    if getattr(args, "comms_dropout_mode", None) is not None:
        scenario_kwargs["comms_dropout_mode"] = str(args.comms_dropout_mode).replace("-", "_")
    joint_observation_schema = bool(
        getattr(args, "joint_schema_uav_diagnostic", False)
        or getattr(args, "joint_observation_schema", False)
    )
    if args.n_drones is not None:
        scenario_kwargs["n_drones"] = int(args.n_drones)
    elif joint_observation_schema:
        scenario_kwargs.setdefault("n_drones", 3)
    else:
        scenario_kwargs.setdefault("n_drones", 1)
    if joint_observation_schema:
        scenario_kwargs.update({
            "obs_schema_n_drones": int(
                args.n_drones
                if getattr(args, "n_drones", None) is not None
                else scenario_kwargs.get("obs_schema_n_drones", 3)
            ),
            "obs_schema_n_ground": int(
                args.n_ugvs
                if getattr(args, "n_ugvs", None) is not None
                else scenario_kwargs.get("obs_schema_n_ground", 2)
            ),
            "obs_schema_n_survivors": int(
                args.n_survivors
                if getattr(args, "n_survivors", None) is not None
                else scenario_kwargs.get("obs_schema_n_survivors", scenario_kwargs.get("n_survivors", 5))
            ),
            "ugv_planner_hint": "global_astar",
            "ugv_assigned_target_obs_only": False,
            "survivor_assignment_obs": True,
        })

    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if args.local_map_patch_size is not None:
        scenario_kwargs["local_map_patch_size"] = int(args.local_map_patch_size)
    if getattr(args, "drone_perception_mode", None) is not None:
        scenario_kwargs["drone_perception_mode"] = (
            str(args.drone_perception_mode).replace("+", "_").replace("-", "_")
        )
    if args.drone_min_footprint_radius_m is not None:
        scenario_kwargs.pop("drone_min_footprint", None)
        scenario_kwargs["drone_min_footprint_m"] = max(float(args.drone_min_footprint_radius_m), 0.0)
    if getattr(args, "uav_start_min_separation_m", None) is not None:
        scenario_kwargs["uav_start_min_separation_m"] = max(float(args.uav_start_min_separation_m), 0.0)
    if getattr(args, "uav_start_edge_margin_m", None) is not None:
        scenario_kwargs["uav_start_edge_margin_m"] = max(float(args.uav_start_edge_margin_m), 0.0)
    if getattr(args, "uav_overlap_penalty_normalization", None) is not None:
        scenario_kwargs["uav_overlap_penalty_normalization"] = (
            str(args.uav_overlap_penalty_normalization).replace("-", "_").lower()
        )
    for attr in (
        "uav_decision_grid",
        "uav_confidence_reward_grid",
        "uav_frontier_global_grid",
        "uav_coverage_reward_grid",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            scenario_kwargs[attr] = int(value)
    if getattr(args, "n_decoys", None) is not None:
        scenario_kwargs["n_decoys"] = max(int(args.n_decoys), 0)
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
    return scenario_kwargs


def run_rollout(
    policy: HappoPolicy,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    diagnostic_confidence_frontier_radius_m: float = DEFAULT_DIAGNOSTIC_CONFIDENCE_FRONTIER_RADIUS_M,
    moving_no_confidence_gain_threshold: float = DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
    diagnostic_level: str = "full",
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
    policy.reset()
    scenario = env.scenario
    full_diagnostics = str(diagnostic_level).replace("-", "_").lower() == "full"
    start_metrics = _start_metrics(scenario)
    coverage_geometry = _coverage_grid_geometry(scenario) if full_diagnostics else None
    individual_coverage_history = (
        np.zeros(
            (
                int(scenario.n_drones),
                int(scenario.fire_grid_size),
                int(scenario.fire_grid_size),
            ),
            dtype=bool,
        )
        if full_diagnostics
        else None
    )

    survivor_slots = int(scenario.n_survivors)
    active_survivor_mask = _active_survivor_mask_for_env(scenario)
    active_survivor_indices = np.flatnonzero(active_survivor_mask)
    n_active_survivors = int(active_survivor_mask.sum())
    first_scout_steps: list[int | None] = [None] * survivor_slots
    first_confirm_steps: list[int | None] = [None] * survivor_slots
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    action_norms: list[float] = []
    displacement_m_values: list[float] = []
    action_displacement_alignments: list[float] = []
    action_displacement_alignments_new_cov: list[float] = []
    action_displacement_alignments_no_new_cov: list[float] = []
    action_frontier_alignment_values: list[float] = []
    action_frontier_alignment_new_cov_values: list[float] = []
    action_frontier_alignment_no_new_cov_values: list[float] = []
    action_frontier_intent_values: list[float] = []
    action_frontier_movement_gap_values: list[float] = []
    new_coverage_cells_values: list[float] = []
    raw_new_coverage_cells_values: list[float] = []
    outside_footprint_values: list[float] = []
    overlap_values: list[float] = []
    expected_overlap_values: list[float] = []
    excess_overlap_values: list[float] = []
    inter_uav_overlap_values: list[float] = []
    any_history_revisit_values: list[float] = []
    own_history_revisit_values: list[float] = []
    teammate_history_revisit_values: list[float] = []
    own_only_revisit_values: list[float] = []
    teammate_only_revisit_values: list[float] = []
    shared_history_revisit_values: list[float] = []
    unavoidable_revisit_values: list[float] = []
    avoidable_revisit_values: list[float] = []
    frontier_expected_new_cells_values: list[float] = []
    frontier_new_cell_capture_values: list[float] = []
    frontier_new_cell_gap_values: list[float] = []
    candidate_best_new_cells_values: list[float] = []
    candidate_capture_fraction_values: list[float] = []
    candidate_new_cell_regret_values: list[float] = []
    candidate_best_new_overlap_values: list[float] = []
    candidate_best_useful_overlap_values: list[float] = []
    candidate_avoidable_overlap_values: list[float] = []
    candidate_action_rank_values: list[float] = []
    candidate_movement_rank_values: list[float] = []
    candidate_action_capture_values: list[float] = []
    candidate_movement_capture_values: list[float] = []
    candidate_action_best_alignment_values: list[float] = []
    candidate_movement_best_alignment_values: list[float] = []
    candidate_no_opportunity_values: list[float] = []
    frontier_candidate_new_cells_values: list[float] = []
    frontier_candidate_capture_fraction_values: list[float] = []
    frontier_candidate_regret_values: list[float] = []
    frontier_candidate_best_alignment_values: list[float] = []
    frontier_candidate_rank_values: list[float] = []
    frontier_candidate_nearest_rank_values: list[float] = []
    frontier_candidate_is_best_values: list[float] = []
    frontier_candidate_bad_values: list[float] = []
    confidence_frontier_candidate_capture_fraction_values: list[float] = []
    confidence_frontier_candidate_best_alignment_values: list[float] = []
    confidence_frontier_candidate_rank_values: list[float] = []
    confidence_frontier_candidate_bad_values: list[float] = []
    confidence_lg_frontier_candidate_capture_fraction_values: list[float] = []
    confidence_lg_frontier_candidate_best_alignment_values: list[float] = []
    confidence_lg_frontier_candidate_rank_values: list[float] = []
    confidence_lg_frontier_candidate_bad_values: list[float] = []
    confidence_frontier_capture_advantage_values: list[float] = []
    confidence_lg_frontier_capture_advantage_values: list[float] = []
    frontier_alignment_values: list[float] = []
    frontier_progress_values: list[float] = []
    frontier_uncovered_ratio_values: list[float] = []
    frontier_obs_distance_values: list[float] = []
    frontier_obs_vector_norm_values: list[float] = []
    frontier_local_coverage_cos_values: list[float] = []
    frontier_global_coverage_cos_values: list[float] = []
    local_global_coverage_cos_values: list[float] = []
    frontier_sector_cos_values: list[float] = []
    frontier_sector_dominance_values: list[float] = []
    frontier_sector_entropy_values: list[float] = []
    frontier_cancellation_values: list[float] = []
    frontier_pairwise_cos_values: list[float] = []
    frontier_pairwise_same_dir_values: list[float] = []
    local_pairwise_same_dir_values: list[float] = []
    global_pairwise_same_dir_values: list[float] = []
    coverage_opportunity_cells_values: list[float] = []
    coverage_opportunity_fraction_values: list[float] = []
    coverage_opportunity_available_fraction_values: list[float] = []
    confidence_mean_values: list[float] = []
    confidence_gain_values: list[float] = []
    confidence_gain_by_drone_values: list[float] = []
    confidence_weighted_gain_values: list[float] = []
    confidence_weighted_gain_by_drone_values: list[float] = []
    confidence_opportunity_fraction_values: list[float] = []
    confidence_opportunity_best_gain_values: list[float] = []
    confidence_low_fraction_values: list[float] = []
    confidence_high_fraction_values: list[float] = []
    confidence_step_detection_probability_values: list[float] = []
    confidence_step_detection_probability_by_drone_values: list[float] = []
    confidence_overlap_fraction_values: list[float] = []
    confidence_overlap_regret_values: list[float] = []
    cleanup_target_valid_values: list[float] = []
    cleanup_target_distance_values: list[float] = []
    cleanup_target_value_values: list[float] = []
    cleanup_target_progress_values: list[float] = []
    cleanup_target_progress_fraction_values: list[float] = []
    cleanup_target_switch_values: list[float] = []
    cleanup_target_reached_values: list[float] = []
    cleanup_target_value_decay_values: list[float] = []
    cleanup_target_no_progress_values: list[float] = []
    cleanup_target_progress_with_new_cells_values: list[float] = []
    cleanup_target_progress_with_excess_overlap_values: list[float] = []
    cleanup_target_frontier_gate_values: list[float] = []
    reward_uav_coverage_values: list[float] = []
    reward_uav_move_coverage_values: list[float] = []
    reward_uav_frontier_values: list[float] = []
    reward_uav_confidence_values: list[float] = []
    reward_uav_team_confidence_values: list[float] = []
    penalty_uav_team_confidence_overlap_values: list[float] = []
    reward_uav_confidence_move_values: list[float] = []
    reward_uav_cleanup_target_progress_values: list[float] = []
    reward_uav_astar_progress_values: list[float] = []
    penalty_uav_inefficient_move_values: list[float] = []
    penalty_uav_confidence_overlap_values: list[float] = []
    penalty_uav_overlap_values: list[float] = []
    penalty_uav_inter_overlap_values: list[float] = []
    penalty_uav_outside_footprint_values: list[float] = []
    reward_uav_coverage_threshold_values: list[float] = []
    reward_uav_scout_values: list[float] = []
    reward_team_values: list[float] = []
    reward_all_survivors_found_values: list[float] = []
    reward_uav_aux_values: list[float] = []
    frontier_abs_reward_share_values: list[float] = []
    frontier_progress_edge_values: list[float] = []
    frontier_progress_interior_values: list[float] = []
    frontier_reward_edge_values: list[float] = []
    frontier_reward_interior_values: list[float] = []
    frontier_new_cells_edge_values: list[float] = []
    frontier_new_cells_interior_values: list[float] = []
    frontier_outside_edge_values: list[float] = []
    frontier_outside_interior_values: list[float] = []
    boundary_distance_m_values: list[float] = []
    footprint_radius_m_values: list[float] = []
    path_positions_sim: list[np.ndarray] = []
    per_drone_stats = [_new_drone_stats(drone_idx) for drone_idx in range(scenario.n_drones)]
    fast_drone_path_lengths = np.zeros(int(scenario.n_drones), dtype=float)
    fast_drone_displacements: list[list[float]] = [
        [] for _ in range(int(scenario.n_drones))
    ]
    fast_drone_action_norms: list[list[float]] = [
        [] for _ in range(int(scenario.n_drones))
    ]
    low_action_high_motion = 0
    high_action_low_motion = 0
    moving_no_new_coverage = 0
    moving_no_confidence_gain = 0
    frontier_high_progress_steps = 0
    frontier_high_progress_no_new_steps = 0
    frontier_high_progress_edge_steps = 0
    frontier_high_progress_corner_steps = 0
    action_frontier_aligned_steps = 0
    action_frontier_anti_aligned_steps = 0
    action_frontier_aligned_no_new_steps = 0
    action_frontier_aligned_edge_steps = 0
    frontier_obs_empty_steps = 0
    diagnostic_steps = 0
    time_bins = _new_time_bins(TIME_BIN_COUNT)
    perception_time_bins = _new_time_bins(TIME_BIN_COUNT)
    survivor_exposure_stats = _new_survivor_exposure_stats(survivor_slots)

    for step in range(int(scenario_kwargs["max_steps"])):
        prev_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool).copy()
        pre_drone_pos = [
            agent.state.pos[0].detach().cpu().numpy().astype(float).copy()
            for agent in scenario.world.agents[:scenario.n_drones]
        ]
        pre_team_coverage = None
        pre_individual_coverage = None
        if full_diagnostics:
            pre_team_coverage = scenario.coverage_grid[0].detach().cpu().numpy().astype(bool).copy()
            pre_individual_coverage = individual_coverage_history.copy()
        if scenario.n_drones > 0:
            pre_drone_pos_array = np.asarray(pre_drone_pos, dtype=float)
            if full_diagnostics:
                assert coverage_geometry is not None
                assert pre_team_coverage is not None
                pre_drone_pos_tensor = torch.stack(
                    [agent.state.pos for agent in scenario.world.agents[:scenario.n_drones]],
                    dim=1,
                )
                frontier_tensor = scenario._cached_uav_frontier_features_for_positions(
                    "diagnostic_current",
                    pre_drone_pos_tensor,
                )
                frontier_obs = frontier_tensor[0].detach().cpu().numpy().astype(float)
                coverage_signal = _coverage_signal_snapshot(scenario, pre_drone_pos_tensor, frontier_obs)
                pre_footprint_radius_sim = (
                    scenario._drone_camera_ranges()[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .reshape(-1)
                )
                max_step_sim = (
                    float(getattr(scenario, "drone_speed_mps", 0.0))
                    * float(getattr(scenario, "sim_step_seconds", 1.0))
                    * max(
                        float(
                            scenario.terrain_sim_units_per_meter[0]
                            .detach()
                            .cpu()
                            .item()
                        ),
                        1e-9,
                    )
                )
                frontier_expected_new_cells = _frontier_expected_new_cells(
                    scenario=scenario,
                    geometry=coverage_geometry,
                    positions=pre_drone_pos_array,
                    footprint_radii_sim=pre_footprint_radius_sim,
                    pre_team_coverage=pre_team_coverage,
                    frontier_obs=frontier_obs,
                    max_step_sim=max_step_sim,
                )
                counterfactual = _counterfactual_move_diagnostics(
                    scenario=scenario,
                    geometry=coverage_geometry,
                    positions=pre_drone_pos_array,
                    footprint_radii_sim=pre_footprint_radius_sim,
                    pre_team_coverage=pre_team_coverage,
                    max_step_sim=max_step_sim,
                )
                frontier_mode = str(getattr(scenario, "uav_frontier_mode", "centroid")).replace("-", "_")
                frontier_top_k = int(getattr(scenario, "uav_frontier_top_k", 1))
                current_frontier_config = _current_frontier_config(scenario)
                frontier_usefulness = _frontier_usefulness_diagnostics(
                    scenario=scenario,
                    geometry=coverage_geometry,
                    positions=pre_drone_pos_array,
                    footprint_radii_sim=pre_footprint_radius_sim,
                    pre_team_coverage=pre_team_coverage,
                    max_step_sim=max_step_sim,
                    frontier_obs=frontier_obs,
                    counterfactual=counterfactual,
                    frontier_mode=frontier_mode,
                    frontier_top_k=frontier_top_k,
                )
                confidence_frontier_config = _frontier_config_for(
                    scenario,
                    source="confidence",
                    mode=frontier_mode,
                )
                if _frontier_configs_match(confidence_frontier_config, current_frontier_config):
                    confidence_frontier_obs = frontier_obs
                    confidence_frontier_usefulness = frontier_usefulness
                else:
                    confidence_frontier_obs = _shadow_frontier_features(
                        scenario,
                        pre_drone_pos_tensor,
                        source=confidence_frontier_config["source"],
                        mode=confidence_frontier_config["mode"],
                        radius_m=confidence_frontier_config["radius_m"],
                    )
                    confidence_frontier_usefulness = _frontier_usefulness_diagnostics(
                        scenario=scenario,
                        geometry=coverage_geometry,
                        positions=pre_drone_pos_array,
                        footprint_radii_sim=pre_footprint_radius_sim,
                        pre_team_coverage=pre_team_coverage,
                        max_step_sim=max_step_sim,
                        frontier_obs=confidence_frontier_obs,
                        counterfactual=counterfactual,
                        frontier_mode=frontier_mode,
                        frontier_top_k=frontier_top_k,
                    )
                confidence_lg_frontier_config = _frontier_config_for(
                    scenario,
                    source="confidence",
                    mode="local_global",
                    radius_m=diagnostic_confidence_frontier_radius_m,
                )
                if _frontier_configs_match(confidence_lg_frontier_config, current_frontier_config):
                    confidence_lg_frontier_obs = frontier_obs
                    confidence_lg_frontier_usefulness = frontier_usefulness
                elif _frontier_configs_match(confidence_lg_frontier_config, confidence_frontier_config):
                    confidence_lg_frontier_obs = confidence_frontier_obs
                    confidence_lg_frontier_usefulness = confidence_frontier_usefulness
                else:
                    confidence_lg_frontier_obs = _shadow_frontier_features(
                        scenario,
                        pre_drone_pos_tensor,
                        source=confidence_lg_frontier_config["source"],
                        mode=confidence_lg_frontier_config["mode"],
                        radius_m=confidence_lg_frontier_config["radius_m"],
                    )
                    confidence_lg_frontier_usefulness = _frontier_usefulness_diagnostics(
                        scenario=scenario,
                        geometry=coverage_geometry,
                        positions=pre_drone_pos_array,
                        footprint_radii_sim=pre_footprint_radius_sim,
                        pre_team_coverage=pre_team_coverage,
                        max_step_sim=max_step_sim,
                        frontier_obs=confidence_lg_frontier_obs,
                        counterfactual=counterfactual,
                        frontier_mode="local_global",
                        frontier_top_k=2,
                    )
            else:
                frontier_obs = np.zeros((scenario.n_drones, 4), dtype=float)
                coverage_signal = _empty_coverage_signal_snapshot(scenario.n_drones)
                frontier_expected_new_cells = np.zeros(scenario.n_drones, dtype=float)
                counterfactual = _empty_counterfactual_move_diagnostics(scenario.n_drones)
                frontier_usefulness = _empty_frontier_usefulness_diagnostics(scenario.n_drones)
                confidence_frontier_usefulness = _empty_frontier_usefulness_diagnostics(scenario.n_drones)
                confidence_lg_frontier_usefulness = _empty_frontier_usefulness_diagnostics(scenario.n_drones)
        else:
            frontier_obs = np.zeros((0, 4), dtype=float)
            coverage_signal = _empty_coverage_signal_snapshot(0)
            frontier_expected_new_cells = np.zeros(0, dtype=float)
            counterfactual = _empty_counterfactual_move_diagnostics(0)
            frontier_usefulness = _empty_frontier_usefulness_diagnostics(0)
            confidence_frontier_usefulness = _empty_frontier_usefulness_diagnostics(0)
            confidence_lg_frontier_usefulness = _empty_frontier_usefulness_diagnostics(0)
        pre_survivor_confidence = _survivor_confidence_values(scenario) if full_diagnostics else []
        frontier_pairwise = _pairwise_direction_metrics(frontier_obs[:, :2])
        local_pairwise = _pairwise_direction_metrics(coverage_signal["local_vec"])
        global_pairwise = _pairwise_direction_metrics(coverage_signal["global_vec"])
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
            "metric_uav_new_coverage_cells_by_drone",
            scenario.n_drones,
        )
        outside_footprint_fraction = _metric_array(
            scenario,
            "metric_uav_outside_footprint_fraction_by_drone",
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
        inter_uav_overlap_fraction = _metric_array(
            scenario,
            "metric_uav_inter_uav_overlap_fraction_by_drone",
            scenario.n_drones,
        )
        coverage_opportunity_cells = _metric_array(
            scenario,
            "metric_uav_coverage_opportunity_cells_by_drone",
            scenario.n_drones,
        )
        coverage_opportunity_fraction = _metric_array(
            scenario,
            "metric_uav_coverage_opportunity_fraction_by_drone",
            scenario.n_drones,
        )
        coverage_opportunity_available_fraction = _metric_array(
            scenario,
            "metric_uav_coverage_opportunity_available_fraction_by_drone",
            scenario.n_drones,
        )
        confidence_gain_by_drone = _metric_array(
            scenario,
            "metric_uav_confidence_gain_by_drone",
            scenario.n_drones,
        )
        confidence_weighted_gain_by_drone = _metric_array(
            scenario,
            "metric_uav_weighted_confidence_gain_by_drone",
            scenario.n_drones,
        )
        confidence_reward_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_confidence_by_drone",
            scenario.n_drones,
        )
        team_confidence_reward_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_team_confidence_by_drone",
            scenario.n_drones,
        )
        team_confidence_overlap_penalty_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_team_confidence_overlap_by_drone",
            scenario.n_drones,
        )
        confidence_move_reward_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_confidence_move_by_drone",
            scenario.n_drones,
        )
        confidence_overlap_penalty_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_confidence_overlap_by_drone",
            scenario.n_drones,
        )
        cleanup_target_progress_reward_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_cleanup_target_progress_by_drone",
            scenario.n_drones,
        )
        astar_progress_reward_by_drone = _metric_array(
            scenario,
            "metric_reward_uav_astar_progress_by_drone",
            scenario.n_drones,
        )
        astar_progress_fraction_by_drone = _metric_array(
            scenario,
            "metric_uav_astar_progress_fraction_by_drone",
            scenario.n_drones,
        )
        astar_frontier_gate_by_drone = _metric_array(
            scenario,
            "metric_uav_astar_frontier_gate_by_drone",
            scenario.n_drones,
        )
        cleanup_target_frontier_gate_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_frontier_gate_by_drone",
            scenario.n_drones,
        )
        confidence_overlap_fraction_by_drone = _metric_array(
            scenario,
            "metric_uav_confidence_overlap_fraction_by_drone",
            scenario.n_drones,
        )
        confidence_overlap_regret_by_drone = _metric_array(
            scenario,
            "metric_uav_confidence_overlap_regret_by_drone",
            scenario.n_drones,
        )
        confidence_opportunity_fraction_by_drone = _metric_array(
            scenario,
            "metric_uav_confidence_opportunity_fraction_by_drone",
            scenario.n_drones,
        )
        confidence_opportunity_best_gain_by_drone = _metric_array(
            scenario,
            "metric_uav_confidence_opportunity_best_gain_by_drone",
            scenario.n_drones,
        )
        confidence_step_detection_probability_by_drone = _metric_array(
            scenario,
            "metric_uav_step_detection_probability_by_drone",
            scenario.n_drones,
        )
        cleanup_target_valid_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_valid_by_drone",
            scenario.n_drones,
        )
        cleanup_target_distance_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_distance_m_by_drone",
            scenario.n_drones,
        )
        cleanup_target_value_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_value_by_drone",
            scenario.n_drones,
        )
        cleanup_target_progress_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_progress_m_by_drone",
            scenario.n_drones,
        )
        cleanup_target_progress_fraction_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_progress_fraction_by_drone",
            scenario.n_drones,
        )
        cleanup_target_switch_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_switch_by_drone",
            scenario.n_drones,
        )
        cleanup_target_reached_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_reached_by_drone",
            scenario.n_drones,
        )
        cleanup_target_value_decay_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_value_decay_by_drone",
            scenario.n_drones,
        )
        cleanup_target_age_by_drone = _metric_array(
            scenario,
            "metric_uav_cleanup_target_age_by_drone",
            scenario.n_drones,
        )
        confidence_mean = _metric_scalar(scenario, "metric_uav_confidence_mean")
        confidence_gain = _metric_scalar(scenario, "metric_uav_confidence_gain")
        confidence_weighted_gain = _metric_scalar(scenario, "metric_uav_weighted_confidence_gain")
        confidence_low_fraction = _metric_scalar(scenario, "metric_uav_confidence_low_fraction")
        confidence_high_fraction = _metric_scalar(scenario, "metric_uav_confidence_high_fraction")
        confidence_step_detection_probability = _metric_scalar(
            scenario,
            "metric_uav_step_detection_probability",
        )
        confidence_mean_values.append(confidence_mean)
        confidence_gain_values.append(confidence_gain)
        confidence_weighted_gain_values.append(confidence_weighted_gain)
        confidence_low_fraction_values.append(confidence_low_fraction)
        confidence_high_fraction_values.append(confidence_high_fraction)
        confidence_step_detection_probability_values.append(confidence_step_detection_probability)
        frontier_alignment = _metric_array(
            scenario,
            "metric_uav_frontier_alignment_by_drone",
            scenario.n_drones,
        )
        frontier_progress = _metric_array(
            scenario,
            "metric_uav_frontier_progress_fraction_by_drone",
            scenario.n_drones,
        )
        frontier_uncovered_ratio = _metric_array(
            scenario,
            "metric_uav_frontier_uncovered_ratio_by_drone",
            scenario.n_drones,
        )
        boundary_distance_m = _metric_array(
            scenario,
            "metric_uav_boundary_distance_m_by_drone",
            scenario.n_drones,
        )
        footprint_radius_m = _metric_array(
            scenario,
            "metric_uav_footprint_radius_m_by_drone",
            scenario.n_drones,
        )
        team_reward = _metric_scalar(scenario, "metric_reward_team")
        all_survivors_found_reward = _metric_scalar(
            scenario,
            "metric_reward_all_survivors_found",
        )
        coverage_threshold_reward = _metric_scalar(
            scenario,
            "metric_reward_uav_coverage_threshold",
        )
        if full_diagnostics and scenario.n_drones > 0:
            assert coverage_geometry is not None
            assert pre_team_coverage is not None
            assert pre_individual_coverage is not None
            post_drone_pos_array = np.asarray(
                [
                    agent.state.pos[0].detach().cpu().numpy().astype(float)
                    for agent in scenario.world.agents[:scenario.n_drones]
                ],
                dtype=float,
            )
            post_footprint_radius_sim = (
                scenario._drone_camera_ranges()[0]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .reshape(-1)
            )
            current_claims = _footprint_claims(
                post_drone_pos_array,
                post_footprint_radius_sim,
                coverage_geometry,
            )
            revisit = _revisit_decomposition(
                current_claims,
                pre_individual_coverage,
                pre_team_coverage,
            )
            raw_new_coverage_cells = (
                current_claims & ~pre_team_coverage.reshape(1, *pre_team_coverage.shape)
            ).sum(axis=(1, 2)).astype(float)
        elif scenario.n_drones > 0:
            current_claims = np.zeros(
                (0, int(scenario.fire_grid_size), int(scenario.fire_grid_size)),
                dtype=bool,
            )
            revisit = _empty_revisit_decomposition(scenario.n_drones)
            raw_new_coverage_cells = coverage_cells.astype(float).copy()
        else:
            current_claims = np.zeros(
                (0, int(scenario.fire_grid_size), int(scenario.fire_grid_size)),
                dtype=bool,
            )
            revisit = _empty_revisit_decomposition(0)
            raw_new_coverage_cells = np.zeros(0, dtype=float)
        post_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        newly_scouted = post_scouted & ~prev_scouted
        drone_detections = (
            scenario.step_drone_detections[0].detach().cpu().numpy().astype(bool)
            if scenario.n_drones > 0 and survivor_slots > 0
            else np.zeros((scenario.n_drones, survivor_slots), dtype=bool)
        )
        scout_credit = drone_detections & newly_scouted.reshape(1, -1)
        if full_diagnostics:
            post_survivor_confidence = _survivor_confidence_values(scenario)
            perception = _drone_perception_snapshot(scenario, meters_per_sim)
            perception_step = _update_survivor_exposure_stats(
                survivor_exposure_stats,
                perception=perception,
                drone_detections=drone_detections,
                prev_scouted=prev_scouted,
                post_scouted=post_scouted,
                survivor_confidence_pre=pre_survivor_confidence,
                survivor_confidence_post=post_survivor_confidence,
                step=step + 1,
                n_survivors=survivor_slots,
                active_survivor_mask=active_survivor_mask,
            )
            _append_time_bin(
                perception_time_bins,
                step=step,
                max_steps=int(scenario_kwargs["max_steps"]),
                values=perception_step,
            )
        for drone_idx, action_vec in enumerate(action_vectors):
            post_pos = scenario.world.agents[drone_idx].state.pos[0].detach().cpu().numpy().astype(float)
            path_positions_sim.append(post_pos.copy())
            displacement_vec = post_pos - pre_drone_pos[drone_idx]
            action_norm = float(np.linalg.norm(action_vec))
            displacement_m = float(np.linalg.norm(displacement_vec) * meters_per_sim)
            new_cells = float(coverage_cells[drone_idx])
            raw_new_cells = float(raw_new_coverage_cells[drone_idx])
            action_norms.append(action_norm)
            displacement_m_values.append(displacement_m)
            new_coverage_cells_values.append(new_cells)
            raw_new_coverage_cells_values.append(raw_new_cells)
            outside_footprint_values.append(float(outside_footprint_fraction[drone_idx]))
            overlap = float(overlap_fraction[drone_idx])
            footprint_radius = float(footprint_radius_m[drone_idx])
            expected_overlap = float(expected_overlap_fraction[drone_idx])
            excess_overlap = float(excess_overlap_fraction[drone_idx])
            inter_uav_overlap = float(inter_uav_overlap_fraction[drone_idx])

            if not full_diagnostics:
                opportunity_fraction = float(coverage_opportunity_fraction[drone_idx])
                opportunity_available_fraction = float(
                    coverage_opportunity_available_fraction[drone_idx]
                )
                confidence_reward = float(confidence_reward_by_drone[drone_idx])
                team_confidence_reward = float(team_confidence_reward_by_drone[drone_idx])
                team_confidence_overlap_penalty = float(
                    team_confidence_overlap_penalty_by_drone[drone_idx]
                )
                confidence_move_reward = float(confidence_move_reward_by_drone[drone_idx])
                confidence_overlap_penalty = float(confidence_overlap_penalty_by_drone[drone_idx])
                confidence_weighted_gain_drone = float(confidence_weighted_gain_by_drone[drone_idx])
                confidence_overlap_fraction = float(confidence_overlap_fraction_by_drone[drone_idx])
                confidence_overlap_regret = float(confidence_overlap_regret_by_drone[drone_idx])
                cleanup_target_progress_reward = float(
                    cleanup_target_progress_reward_by_drone[drone_idx]
                )
                astar_progress_reward = float(astar_progress_reward_by_drone[drone_idx])
                frontier_progress_frac = float(frontier_progress[drone_idx])
                frontier_ratio = float(frontier_uncovered_ratio[drone_idx])
                scout_reward = float(np.count_nonzero(scout_credit[drone_idx])) * float(
                    getattr(scenario, "r_drone_scout", 0.0)
                )
                reward_terms = _uav_reward_terms(
                    scenario=scenario,
                    new_cells=new_cells,
                    displacement_m=displacement_m,
                    frontier_progress=frontier_progress_frac,
                    frontier_ratio=frontier_ratio,
                    overlap=overlap,
                    expected_overlap=expected_overlap,
                    inter_uav_overlap=inter_uav_overlap,
                    outside_footprint=float(outside_footprint_fraction[drone_idx]),
                    coverage_opportunity_fraction=opportunity_fraction,
                    coverage_opportunity_available_fraction=opportunity_available_fraction,
                    confidence_reward=confidence_reward,
                    team_confidence_reward=team_confidence_reward,
                    team_confidence_overlap_penalty=team_confidence_overlap_penalty,
                    confidence_move_reward=confidence_move_reward,
                    confidence_opportunity_fraction=float(
                        confidence_opportunity_fraction_by_drone[drone_idx]
                    ),
                    confidence_overlap_penalty=confidence_overlap_penalty,
                    cleanup_target_progress_reward=cleanup_target_progress_reward,
                    astar_progress_reward=astar_progress_reward,
                    scout_reward=scout_reward,
                )
                reward_terms["team"] = team_reward
                reward_terms["all_survivors_found"] = all_survivors_found_reward
                reward_terms["coverage_threshold"] = coverage_threshold_reward

                overlap_values.append(overlap)
                expected_overlap_values.append(expected_overlap)
                excess_overlap_values.append(excess_overlap)
                inter_uav_overlap_values.append(inter_uav_overlap)
                outside_footprint_values.append(float(outside_footprint_fraction[drone_idx]))
                boundary_distance_m_values.append(float(boundary_distance_m[drone_idx]))
                footprint_radius_m_values.append(footprint_radius)
                reward_uav_coverage_values.append(reward_terms["coverage"])
                reward_uav_move_coverage_values.append(reward_terms["move_coverage"])
                reward_uav_frontier_values.append(reward_terms["frontier"])
                reward_uav_confidence_values.append(reward_terms["confidence"])
                reward_uav_team_confidence_values.append(reward_terms["team_confidence"])
                penalty_uav_team_confidence_overlap_values.append(
                    reward_terms["team_confidence_overlap_penalty"]
                )
                reward_uav_confidence_move_values.append(reward_terms["confidence_move"])
                reward_uav_cleanup_target_progress_values.append(
                    reward_terms["cleanup_target_progress"]
                )
                reward_uav_astar_progress_values.append(reward_terms["astar_progress"])
                penalty_uav_inefficient_move_values.append(reward_terms["inefficient_move_penalty"])
                penalty_uav_confidence_overlap_values.append(reward_terms["confidence_overlap_penalty"])
                penalty_uav_overlap_values.append(reward_terms["overlap_penalty"])
                penalty_uav_inter_overlap_values.append(reward_terms["inter_uav_overlap_penalty"])
                penalty_uav_outside_footprint_values.append(reward_terms["outside_footprint_penalty"])
                reward_uav_coverage_threshold_values.append(reward_terms["coverage_threshold"])
                reward_uav_scout_values.append(reward_terms["scout"])
                reward_team_values.append(reward_terms["team"])
                reward_all_survivors_found_values.append(reward_terms["all_survivors_found"])
                reward_uav_aux_values.append(reward_terms["aux"])
                if displacement_m > 1.0 and new_cells < 1.0:
                    moving_no_new_coverage += 1
                if action_norm < 0.05 and displacement_m > 1.0:
                    low_action_high_motion += 1
                if action_norm > 0.5 and displacement_m < 0.25:
                    high_action_low_motion += 1
                moving_no_confidence_gain_step = bool(
                    displacement_m > 1.0
                    and math.isfinite(confidence_weighted_gain_drone)
                    and confidence_weighted_gain_drone <= moving_no_confidence_gain_threshold
                )
                if moving_no_confidence_gain_step:
                    moving_no_confidence_gain += 1

                edge_threshold = (
                    footprint_radius
                    if math.isfinite(footprint_radius) and footprint_radius > 0.0
                    else 25.0
                )
                is_edge_step = bool(float(boundary_distance_m[drone_idx]) <= edge_threshold)

                fast_drone_path_lengths[drone_idx] += displacement_m
                fast_drone_displacements[drone_idx].append(displacement_m)
                fast_drone_action_norms[drone_idx].append(action_norm)

                action_displacement_alignment = math.nan
                displacement_norm_sim = float(np.linalg.norm(displacement_vec))
                if action_norm > 1e-6 and displacement_norm_sim > 1e-9:
                    action_displacement_alignment = float(
                        np.dot(action_vec[:2], displacement_vec[:2]) / (action_norm * displacement_norm_sim)
                    )
                    action_displacement_alignment = max(min(action_displacement_alignment, 1.0), -1.0)
                    action_displacement_alignments.append(action_displacement_alignment)

                diagnostic_steps += 1
                _append_time_bin(
                    time_bins,
                    step=step,
                    max_steps=int(scenario_kwargs["max_steps"]),
                    values={
                        "action_norm": action_norm,
                        "displacement_m": displacement_m,
                        "new_coverage_cells": new_cells,
                        "action_displacement_alignment": action_displacement_alignment,
                        "overlap": overlap,
                        "excess_overlap": excess_overlap,
                        "edge_step": float(is_edge_step),
                        "moving_no_new_coverage": float(displacement_m > 1.0 and new_cells < 1.0),
                        "moving_no_confidence_gain": float(moving_no_confidence_gain_step),
                        "confidence_overlap_fraction": confidence_overlap_fraction,
                        "confidence_overlap_regret": confidence_overlap_regret,
                        "frontier_reward": reward_terms["frontier"],
                        "confidence_reward": reward_terms["confidence"],
                        "team_confidence_reward": reward_terms["team_confidence"],
                        "team_confidence_overlap_penalty": reward_terms[
                            "team_confidence_overlap_penalty"
                        ],
                        "confidence_move_reward": reward_terms["confidence_move"],
                        "cleanup_target_progress_reward": reward_terms["cleanup_target_progress"],
                        "astar_progress_reward": reward_terms["astar_progress"],
                        "inefficient_move_penalty": reward_terms["inefficient_move_penalty"],
                        "confidence_overlap_penalty": reward_terms["confidence_overlap_penalty"],
                        "coverage_reward": reward_terms["coverage"],
                        "move_coverage_reward": reward_terms["move_coverage"],
                        "overlap_penalty": reward_terms["overlap_penalty"],
                        "inter_uav_overlap_penalty": reward_terms["inter_uav_overlap_penalty"],
                        "outside_footprint_penalty": reward_terms["outside_footprint_penalty"],
                        "coverage_threshold_reward": reward_terms["coverage_threshold"],
                        "scout_reward": reward_terms["scout"],
                        "team_reward": reward_terms["team"],
                        "all_survivors_reward": reward_terms["all_survivors_found"],
                    },
                )
                continue

            any_history_revisit = float(revisit["any_history"][drone_idx])
            own_history_revisit = float(revisit["own_history"][drone_idx])
            teammate_history_revisit = float(revisit["teammate_history"][drone_idx])
            own_only_revisit = float(revisit["own_only"][drone_idx])
            teammate_only_revisit = float(revisit["teammate_only"][drone_idx])
            shared_history_revisit = float(revisit["shared_history"][drone_idx])
            unavoidable_revisit = min(any_history_revisit, expected_overlap)
            avoidable_revisit = max(any_history_revisit - expected_overlap, 0.0)
            frontier_expected_cells = float(frontier_expected_new_cells[drone_idx])
            frontier_new_cell_capture = (
                raw_new_cells / frontier_expected_cells
                if frontier_expected_cells > 1e-9
                else math.nan
            )
            frontier_new_cell_gap = (
                frontier_expected_cells - raw_new_cells
                if frontier_expected_cells > 1e-9
                else math.nan
            )
            candidate_new_cells = counterfactual["new_cells"][drone_idx]
            candidate_best_new_cells = float(counterfactual["best_new_cells"][drone_idx])
            candidate_best_new_overlap = float(counterfactual["best_new_overlap"][drone_idx])
            candidate_best_useful_overlap = float(counterfactual["best_useful_overlap"][drone_idx])
            candidate_capture_fraction = (
                raw_new_cells / candidate_best_new_cells
                if candidate_best_new_cells > 1e-9
                else math.nan
            )
            candidate_new_cell_regret = (
                max(candidate_best_new_cells - raw_new_cells, 0.0)
                if candidate_best_new_cells > 1e-9
                else math.nan
            )
            candidate_action = _counterfactual_choice_metrics(
                action_vec[:2],
                counterfactual["directions"],
                candidate_new_cells,
                counterfactual["best_new_direction"][drone_idx],
            )
            candidate_movement = _counterfactual_choice_metrics(
                displacement_vec[:2],
                counterfactual["directions"],
                candidate_new_cells,
                counterfactual["best_new_direction"][drone_idx],
            )
            frontier_use = _frontier_usefulness_for_drone(frontier_usefulness, drone_idx)
            confidence_frontier_use = _frontier_usefulness_for_drone(
                confidence_frontier_usefulness,
                drone_idx,
            )
            confidence_lg_frontier_use = _frontier_usefulness_for_drone(
                confidence_lg_frontier_usefulness,
                drone_idx,
            )
            frontier_candidate_capture = float(frontier_use["capture_fraction"])
            confidence_frontier_candidate_capture = float(
                confidence_frontier_use["capture_fraction"]
            )
            confidence_lg_frontier_candidate_capture = float(
                confidence_lg_frontier_use["capture_fraction"]
            )
            confidence_frontier_capture_advantage = (
                confidence_frontier_candidate_capture - frontier_candidate_capture
                if math.isfinite(confidence_frontier_candidate_capture)
                and math.isfinite(frontier_candidate_capture)
                else math.nan
            )
            confidence_lg_frontier_capture_advantage = (
                confidence_lg_frontier_candidate_capture - frontier_candidate_capture
                if math.isfinite(confidence_lg_frontier_candidate_capture)
                and math.isfinite(frontier_candidate_capture)
                else math.nan
            )
            candidate_avoidable_overlap = (
                max(any_history_revisit - candidate_best_useful_overlap, 0.0)
                if math.isfinite(candidate_best_useful_overlap)
                else math.nan
            )
            candidate_no_opportunity = float(candidate_best_new_cells < 1.0)
            opportunity_cells = float(coverage_opportunity_cells[drone_idx])
            opportunity_fraction = float(coverage_opportunity_fraction[drone_idx])
            opportunity_available_fraction = float(
                coverage_opportunity_available_fraction[drone_idx]
            )
            confidence_gain_drone = float(confidence_gain_by_drone[drone_idx])
            confidence_weighted_gain_drone = float(confidence_weighted_gain_by_drone[drone_idx])
            confidence_pass_probability = float(
                confidence_step_detection_probability_by_drone[drone_idx]
            )
            confidence_reward = float(confidence_reward_by_drone[drone_idx])
            team_confidence_reward = float(team_confidence_reward_by_drone[drone_idx])
            team_confidence_overlap_penalty = float(
                team_confidence_overlap_penalty_by_drone[drone_idx]
            )
            confidence_move_reward = float(confidence_move_reward_by_drone[drone_idx])
            confidence_overlap_penalty = float(confidence_overlap_penalty_by_drone[drone_idx])
            confidence_overlap_fraction = float(confidence_overlap_fraction_by_drone[drone_idx])
            confidence_overlap_regret = float(confidence_overlap_regret_by_drone[drone_idx])
            cleanup_target_progress_reward = float(
                cleanup_target_progress_reward_by_drone[drone_idx]
            )
            astar_progress_reward = float(astar_progress_reward_by_drone[drone_idx])
            astar_progress_fraction = float(astar_progress_fraction_by_drone[drone_idx])
            astar_frontier_gate = float(astar_frontier_gate_by_drone[drone_idx])
            cleanup_target_frontier_gate = float(cleanup_target_frontier_gate_by_drone[drone_idx])
            cleanup_target_valid = float(cleanup_target_valid_by_drone[drone_idx])
            cleanup_target_distance = float(cleanup_target_distance_by_drone[drone_idx])
            cleanup_target_value = float(cleanup_target_value_by_drone[drone_idx])
            cleanup_target_progress = float(cleanup_target_progress_by_drone[drone_idx])
            cleanup_target_progress_fraction = float(cleanup_target_progress_fraction_by_drone[drone_idx])
            cleanup_target_switch = float(cleanup_target_switch_by_drone[drone_idx])
            cleanup_target_reached = float(cleanup_target_reached_by_drone[drone_idx])
            cleanup_target_value_decay = float(cleanup_target_value_decay_by_drone[drone_idx])
            cleanup_target_age = float(cleanup_target_age_by_drone[drone_idx])
            confidence_opportunity_fraction = float(
                confidence_opportunity_fraction_by_drone[drone_idx]
            )
            confidence_opportunity_best_gain = float(
                confidence_opportunity_best_gain_by_drone[drone_idx]
            )
            confidence_revisit_step = bool(avoidable_revisit >= CONFIDENCE_REVISIT_THRESHOLD)
            confidence_revisit_has_opportunity = bool(
                math.isfinite(confidence_gain_drone)
                and math.isfinite(confidence_opportunity_fraction)
                and math.isfinite(confidence_opportunity_best_gain)
                and confidence_gain_drone > CONFIDENCE_REVISIT_MIN_GAIN
                and confidence_opportunity_best_gain > CONFIDENCE_REVISIT_MIN_GAIN
            )
            confidence_useful_revisit = bool(
                confidence_revisit_step
                and confidence_revisit_has_opportunity
                and confidence_opportunity_fraction >= CONFIDENCE_REVISIT_USEFUL_OPPORTUNITY_THRESHOLD
            )
            confidence_wasteful_revisit = bool(
                confidence_revisit_step
                and (
                    not confidence_revisit_has_opportunity
                    or confidence_opportunity_fraction < CONFIDENCE_REVISIT_WASTEFUL_OPPORTUNITY_THRESHOLD
                )
            )
            confidence_ambiguous_revisit = bool(
                confidence_revisit_step
                and not confidence_useful_revisit
                and not confidence_wasteful_revisit
            )
            frontier_align = float(frontier_alignment[drone_idx])
            frontier_progress_frac = float(frontier_progress[drone_idx])
            frontier_ratio = float(frontier_uncovered_ratio[drone_idx])
            frontier_candidates = _frontier_candidates_for_drone(scenario, frontier_obs, drone_idx)
            frontier_obs_vec = (
                frontier_candidates[0, :2]
                if len(frontier_candidates)
                else np.zeros(2)
            )
            frontier_obs_distance = (
                float(frontier_candidates[0, 2])
                if len(frontier_candidates)
                else 0.0
            )
            frontier_obs_ratio = (
                float(frontier_candidates[0, 3])
                if len(frontier_candidates)
                else 0.0
            )
            frontier_obs_norm = float(np.linalg.norm(frontier_obs_vec))
            local_coverage_vec = coverage_signal["local_vec"][drone_idx]
            global_coverage_vec = coverage_signal["global_vec"][drone_idx]
            dominant_sector_vec = coverage_signal["sector_vec"][drone_idx]
            frontier_local_cos = _cosine_or_nan(frontier_obs_vec, local_coverage_vec)
            frontier_global_cos = _cosine_or_nan(frontier_obs_vec, global_coverage_vec)
            local_global_cos = _cosine_or_nan(local_coverage_vec, global_coverage_vec)
            frontier_sector_cos = _cosine_or_nan(frontier_obs_vec, dominant_sector_vec)
            sector_dominance = float(coverage_signal["sector_dominance"][drone_idx])
            sector_entropy = float(coverage_signal["sector_entropy"][drone_idx])
            frontier_cancellation = float(coverage_signal["frontier_cancellation"][drone_idx])
            action_frontier_alignment = _best_frontier_candidate_cosine(
                action_vec[:2],
                frontier_candidates,
            )
            action_frontier_intent = _best_frontier_candidate_projection(
                action_vec[:2],
                frontier_candidates,
            )
            action_frontier_movement_gap = (
                action_frontier_alignment - frontier_align
                if math.isfinite(action_frontier_alignment) and math.isfinite(frontier_align)
                else math.nan
            )
            scout_reward = float(np.count_nonzero(scout_credit[drone_idx])) * float(
                getattr(scenario, "r_drone_scout", 0.0)
            )
            reward_terms = _uav_reward_terms(
                scenario=scenario,
                new_cells=new_cells,
                displacement_m=displacement_m,
                frontier_progress=frontier_progress_frac,
                frontier_ratio=frontier_ratio,
                overlap=overlap,
                expected_overlap=expected_overlap,
                inter_uav_overlap=inter_uav_overlap,
                outside_footprint=float(outside_footprint_fraction[drone_idx]),
                coverage_opportunity_fraction=opportunity_fraction,
                coverage_opportunity_available_fraction=opportunity_available_fraction,
                confidence_reward=confidence_reward,
                team_confidence_reward=team_confidence_reward,
                team_confidence_overlap_penalty=team_confidence_overlap_penalty,
                confidence_move_reward=confidence_move_reward,
                confidence_opportunity_fraction=confidence_opportunity_fraction,
                confidence_overlap_penalty=confidence_overlap_penalty,
                cleanup_target_progress_reward=cleanup_target_progress_reward,
                astar_progress_reward=astar_progress_reward,
                scout_reward=scout_reward,
            )
            reward_terms["team"] = team_reward
            reward_terms["all_survivors_found"] = all_survivors_found_reward
            reward_terms["coverage_threshold"] = coverage_threshold_reward
            distances_to_edges = _distances_to_edges_m(
                np.asarray([post_pos], dtype=float),
                scenario,
                meters_per_sim,
            )[0]
            edge_threshold = footprint_radius if math.isfinite(footprint_radius) and footprint_radius > 0.0 else 25.0
            is_edge_step = bool(float(boundary_distance_m[drone_idx]) <= edge_threshold)
            is_corner_step = bool(np.count_nonzero(distances_to_edges <= edge_threshold) >= 2)
            coverage_fraction_now = float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
            overlap_values.append(overlap)
            expected_overlap_values.append(expected_overlap)
            excess_overlap_values.append(excess_overlap)
            inter_uav_overlap_values.append(inter_uav_overlap)
            any_history_revisit_values.append(any_history_revisit)
            own_history_revisit_values.append(own_history_revisit)
            teammate_history_revisit_values.append(teammate_history_revisit)
            own_only_revisit_values.append(own_only_revisit)
            teammate_only_revisit_values.append(teammate_only_revisit)
            shared_history_revisit_values.append(shared_history_revisit)
            unavoidable_revisit_values.append(unavoidable_revisit)
            avoidable_revisit_values.append(avoidable_revisit)
            frontier_expected_new_cells_values.append(frontier_expected_cells)
            frontier_new_cell_capture_values.append(frontier_new_cell_capture)
            frontier_new_cell_gap_values.append(frontier_new_cell_gap)
            candidate_best_new_cells_values.append(candidate_best_new_cells)
            candidate_capture_fraction_values.append(candidate_capture_fraction)
            candidate_new_cell_regret_values.append(candidate_new_cell_regret)
            candidate_best_new_overlap_values.append(candidate_best_new_overlap)
            candidate_best_useful_overlap_values.append(candidate_best_useful_overlap)
            candidate_avoidable_overlap_values.append(candidate_avoidable_overlap)
            candidate_action_rank_values.append(candidate_action["rank"])
            candidate_movement_rank_values.append(candidate_movement["rank"])
            candidate_action_capture_values.append(candidate_action["capture_fraction"])
            candidate_movement_capture_values.append(candidate_movement["capture_fraction"])
            candidate_action_best_alignment_values.append(candidate_action["best_alignment"])
            candidate_movement_best_alignment_values.append(candidate_movement["best_alignment"])
            candidate_no_opportunity_values.append(candidate_no_opportunity)
            frontier_candidate_new_cells_values.append(float(frontier_use["new_cells"]))
            frontier_candidate_capture_fraction_values.append(frontier_candidate_capture)
            frontier_candidate_regret_values.append(float(frontier_use["regret"]))
            frontier_candidate_best_alignment_values.append(float(frontier_use["best_alignment"]))
            frontier_candidate_rank_values.append(float(frontier_use["rank"]))
            frontier_candidate_nearest_rank_values.append(float(frontier_use["nearest_rank"]))
            frontier_candidate_is_best_values.append(float(frontier_use["is_best"]))
            frontier_candidate_bad_values.append(float(frontier_use["bad"]))
            confidence_frontier_candidate_capture_fraction_values.append(
                confidence_frontier_candidate_capture
            )
            confidence_frontier_candidate_best_alignment_values.append(
                float(confidence_frontier_use["best_alignment"])
            )
            confidence_frontier_candidate_rank_values.append(float(confidence_frontier_use["rank"]))
            confidence_frontier_candidate_bad_values.append(float(confidence_frontier_use["bad"]))
            confidence_lg_frontier_candidate_capture_fraction_values.append(
                confidence_lg_frontier_candidate_capture
            )
            confidence_lg_frontier_candidate_best_alignment_values.append(
                float(confidence_lg_frontier_use["best_alignment"])
            )
            confidence_lg_frontier_candidate_rank_values.append(float(confidence_lg_frontier_use["rank"]))
            confidence_lg_frontier_candidate_bad_values.append(float(confidence_lg_frontier_use["bad"]))
            confidence_frontier_capture_advantage_values.append(confidence_frontier_capture_advantage)
            confidence_lg_frontier_capture_advantage_values.append(confidence_lg_frontier_capture_advantage)
            coverage_opportunity_cells_values.append(opportunity_cells)
            coverage_opportunity_fraction_values.append(opportunity_fraction)
            coverage_opportunity_available_fraction_values.append(opportunity_available_fraction)
            confidence_gain_by_drone_values.append(confidence_gain_drone)
            confidence_weighted_gain_by_drone_values.append(confidence_weighted_gain_drone)
            confidence_opportunity_fraction_values.append(confidence_opportunity_fraction)
            confidence_opportunity_best_gain_values.append(confidence_opportunity_best_gain)
            confidence_step_detection_probability_by_drone_values.append(confidence_pass_probability)
            confidence_overlap_fraction_values.append(confidence_overlap_fraction)
            confidence_overlap_regret_values.append(confidence_overlap_regret)
            cleanup_target_valid_values.append(cleanup_target_valid)
            cleanup_target_distance_values.append(cleanup_target_distance)
            cleanup_target_value_values.append(cleanup_target_value)
            cleanup_target_progress_values.append(cleanup_target_progress)
            cleanup_target_progress_fraction_values.append(cleanup_target_progress_fraction)
            cleanup_target_switch_values.append(cleanup_target_switch)
            cleanup_target_reached_values.append(cleanup_target_reached)
            cleanup_target_value_decay_values.append(cleanup_target_value_decay)
            cleanup_target_no_progress_values.append(
                float(cleanup_target_valid >= 0.5 and cleanup_target_progress <= 1e-6)
            )
            cleanup_target_progress_with_new_cells_values.append(
                float(cleanup_target_valid >= 0.5 and cleanup_target_progress > 1e-6 and new_cells >= 1.0)
            )
            cleanup_target_progress_with_excess_overlap_values.append(
                float(cleanup_target_valid >= 0.5 and cleanup_target_progress > 1e-6 and excess_overlap >= 0.10)
            )
            cleanup_target_frontier_gate_values.append(cleanup_target_frontier_gate)
            frontier_alignment_values.append(frontier_align)
            frontier_progress_values.append(frontier_progress_frac)
            frontier_uncovered_ratio_values.append(frontier_ratio)
            frontier_obs_distance_values.append(frontier_obs_distance)
            frontier_obs_vector_norm_values.append(frontier_obs_norm)
            frontier_local_coverage_cos_values.append(frontier_local_cos)
            frontier_global_coverage_cos_values.append(frontier_global_cos)
            local_global_coverage_cos_values.append(local_global_cos)
            frontier_sector_cos_values.append(frontier_sector_cos)
            frontier_sector_dominance_values.append(sector_dominance)
            frontier_sector_entropy_values.append(sector_entropy)
            frontier_cancellation_values.append(frontier_cancellation)
            frontier_pairwise_cos_values.append(frontier_pairwise["mean_cos"])
            frontier_pairwise_same_dir_values.append(frontier_pairwise["same_dir_frac"])
            local_pairwise_same_dir_values.append(local_pairwise["same_dir_frac"])
            global_pairwise_same_dir_values.append(global_pairwise["same_dir_frac"])
            action_frontier_alignment_values.append(action_frontier_alignment)
            action_frontier_intent_values.append(action_frontier_intent)
            action_frontier_movement_gap_values.append(action_frontier_movement_gap)
            if new_cells >= 1.0:
                action_frontier_alignment_new_cov_values.append(action_frontier_alignment)
            else:
                action_frontier_alignment_no_new_cov_values.append(action_frontier_alignment)
            reward_uav_coverage_values.append(reward_terms["coverage"])
            reward_uav_move_coverage_values.append(reward_terms["move_coverage"])
            reward_uav_frontier_values.append(reward_terms["frontier"])
            reward_uav_confidence_values.append(reward_terms["confidence"])
            reward_uav_team_confidence_values.append(reward_terms["team_confidence"])
            penalty_uav_team_confidence_overlap_values.append(
                reward_terms["team_confidence_overlap_penalty"]
            )
            reward_uav_confidence_move_values.append(reward_terms["confidence_move"])
            reward_uav_cleanup_target_progress_values.append(
                reward_terms["cleanup_target_progress"]
            )
            reward_uav_astar_progress_values.append(reward_terms["astar_progress"])
            penalty_uav_inefficient_move_values.append(reward_terms["inefficient_move_penalty"])
            penalty_uav_confidence_overlap_values.append(reward_terms["confidence_overlap_penalty"])
            penalty_uav_overlap_values.append(reward_terms["overlap_penalty"])
            penalty_uav_inter_overlap_values.append(reward_terms["inter_uav_overlap_penalty"])
            penalty_uav_outside_footprint_values.append(reward_terms["outside_footprint_penalty"])
            reward_uav_coverage_threshold_values.append(reward_terms["coverage_threshold"])
            reward_uav_scout_values.append(reward_terms["scout"])
            reward_team_values.append(reward_terms["team"])
            reward_all_survivors_found_values.append(reward_terms["all_survivors_found"])
            reward_uav_aux_values.append(reward_terms["aux"])
            frontier_abs_reward_share_values.append(reward_terms["frontier_abs_share"])
            if is_edge_step:
                frontier_progress_edge_values.append(frontier_progress_frac)
                frontier_reward_edge_values.append(reward_terms["frontier"])
                frontier_new_cells_edge_values.append(new_cells)
                frontier_outside_edge_values.append(float(outside_footprint_fraction[drone_idx]))
            else:
                frontier_progress_interior_values.append(frontier_progress_frac)
                frontier_reward_interior_values.append(reward_terms["frontier"])
                frontier_new_cells_interior_values.append(new_cells)
                frontier_outside_interior_values.append(float(outside_footprint_fraction[drone_idx]))
            boundary_distance_m_values.append(float(boundary_distance_m[drone_idx]))
            footprint_radius_m_values.append(footprint_radius)
            diagnostic_steps += 1
            drone_stats = per_drone_stats[drone_idx]
            drone_stats["positions_sim"].extend([pre_drone_pos[drone_idx], post_pos.copy()])
            drone_stats["action_norms"].append(action_norm)
            drone_stats["displacement_m"].append(displacement_m)
            drone_stats["new_coverage_cells"].append(new_cells)
            drone_stats["raw_new_coverage_cells"].append(raw_new_cells)
            drone_stats["outside_footprint"].append(float(outside_footprint_fraction[drone_idx]))
            drone_stats["overlap"].append(overlap)
            drone_stats["expected_overlap"].append(expected_overlap)
            drone_stats["excess_overlap"].append(excess_overlap)
            drone_stats["inter_uav_overlap"].append(inter_uav_overlap)
            drone_stats["any_history_revisit"].append(any_history_revisit)
            drone_stats["own_history_revisit"].append(own_history_revisit)
            drone_stats["teammate_history_revisit"].append(teammate_history_revisit)
            drone_stats["own_only_revisit"].append(own_only_revisit)
            drone_stats["teammate_only_revisit"].append(teammate_only_revisit)
            drone_stats["shared_history_revisit"].append(shared_history_revisit)
            drone_stats["unavoidable_revisit"].append(unavoidable_revisit)
            drone_stats["avoidable_revisit"].append(avoidable_revisit)
            drone_stats["frontier_expected_new_cells"].append(frontier_expected_cells)
            drone_stats["frontier_new_cell_capture"].append(frontier_new_cell_capture)
            drone_stats["frontier_new_cell_gap"].append(frontier_new_cell_gap)
            drone_stats["candidate_best_new_cells"].append(candidate_best_new_cells)
            drone_stats["candidate_capture_fraction"].append(candidate_capture_fraction)
            drone_stats["candidate_new_cell_regret"].append(candidate_new_cell_regret)
            drone_stats["candidate_best_new_overlap"].append(candidate_best_new_overlap)
            drone_stats["candidate_best_useful_overlap"].append(candidate_best_useful_overlap)
            drone_stats["candidate_avoidable_overlap"].append(candidate_avoidable_overlap)
            drone_stats["candidate_action_rank"].append(candidate_action["rank"])
            drone_stats["candidate_movement_rank"].append(candidate_movement["rank"])
            drone_stats["candidate_action_capture_fraction"].append(candidate_action["capture_fraction"])
            drone_stats["candidate_movement_capture_fraction"].append(candidate_movement["capture_fraction"])
            drone_stats["candidate_action_best_alignment"].append(candidate_action["best_alignment"])
            drone_stats["candidate_movement_best_alignment"].append(candidate_movement["best_alignment"])
            drone_stats["candidate_no_opportunity"].append(candidate_no_opportunity)
            drone_stats["frontier_candidate_new_cells"].append(float(frontier_use["new_cells"]))
            drone_stats["frontier_candidate_capture_fraction"].append(frontier_candidate_capture)
            drone_stats["frontier_candidate_regret"].append(float(frontier_use["regret"]))
            drone_stats["frontier_candidate_best_alignment"].append(float(frontier_use["best_alignment"]))
            drone_stats["frontier_candidate_rank"].append(float(frontier_use["rank"]))
            drone_stats["frontier_candidate_nearest_rank"].append(float(frontier_use["nearest_rank"]))
            drone_stats["frontier_candidate_is_best"].append(float(frontier_use["is_best"]))
            drone_stats["frontier_candidate_bad"].append(float(frontier_use["bad"]))
            drone_stats["confidence_frontier_candidate_capture_fraction"].append(
                confidence_frontier_candidate_capture
            )
            drone_stats["confidence_frontier_candidate_best_alignment"].append(
                float(confidence_frontier_use["best_alignment"])
            )
            drone_stats["confidence_frontier_candidate_rank"].append(
                float(confidence_frontier_use["rank"])
            )
            drone_stats["confidence_frontier_candidate_bad"].append(
                float(confidence_frontier_use["bad"])
            )
            drone_stats["confidence_lg_frontier_candidate_capture_fraction"].append(
                confidence_lg_frontier_candidate_capture
            )
            drone_stats["confidence_lg_frontier_candidate_best_alignment"].append(
                float(confidence_lg_frontier_use["best_alignment"])
            )
            drone_stats["confidence_lg_frontier_candidate_rank"].append(
                float(confidence_lg_frontier_use["rank"])
            )
            drone_stats["confidence_lg_frontier_candidate_bad"].append(
                float(confidence_lg_frontier_use["bad"])
            )
            drone_stats["confidence_frontier_capture_advantage"].append(
                confidence_frontier_capture_advantage
            )
            drone_stats["confidence_lg_frontier_capture_advantage"].append(
                confidence_lg_frontier_capture_advantage
            )
            drone_stats["coverage_opportunity_cells"].append(opportunity_cells)
            drone_stats["coverage_opportunity_fraction"].append(opportunity_fraction)
            drone_stats["coverage_opportunity_available_fraction"].append(opportunity_available_fraction)
            drone_stats["confidence_gain"].append(confidence_gain_drone)
            drone_stats["confidence_weighted_gain"].append(confidence_weighted_gain_drone)
            drone_stats["confidence_opportunity_fraction"].append(confidence_opportunity_fraction)
            drone_stats["confidence_opportunity_best_gain"].append(confidence_opportunity_best_gain)
            drone_stats["confidence_pass_probability"].append(confidence_pass_probability)
            drone_stats["confidence_overlap_fraction"].append(confidence_overlap_fraction)
            drone_stats["confidence_overlap_regret"].append(confidence_overlap_regret)
            drone_stats["cleanup_target_valid"].append(cleanup_target_valid)
            drone_stats["cleanup_target_distance_m"].append(cleanup_target_distance)
            drone_stats["cleanup_target_value"].append(cleanup_target_value)
            drone_stats["cleanup_target_progress_m"].append(cleanup_target_progress)
            drone_stats["cleanup_target_progress_fraction"].append(cleanup_target_progress_fraction)
            drone_stats["cleanup_target_switch"].append(cleanup_target_switch)
            drone_stats["cleanup_target_reached"].append(cleanup_target_reached)
            drone_stats["cleanup_target_value_decay"].append(cleanup_target_value_decay)
            drone_stats["cleanup_target_age"].append(cleanup_target_age)
            drone_stats["cleanup_target_frontier_gate"].append(cleanup_target_frontier_gate)
            drone_stats["frontier_alignment"].append(frontier_align)
            drone_stats["frontier_progress"].append(frontier_progress_frac)
            drone_stats["frontier_uncovered_ratio"].append(frontier_ratio)
            drone_stats["frontier_obs_distance"].append(frontier_obs_distance)
            drone_stats["frontier_obs_vector_norm"].append(frontier_obs_norm)
            drone_stats["frontier_obs_uncovered_ratio"].append(frontier_obs_ratio)
            drone_stats["frontier_local_coverage_cos"].append(frontier_local_cos)
            drone_stats["frontier_global_coverage_cos"].append(frontier_global_cos)
            drone_stats["local_global_coverage_cos"].append(local_global_cos)
            drone_stats["frontier_sector_cos"].append(frontier_sector_cos)
            drone_stats["frontier_sector_dominance"].append(sector_dominance)
            drone_stats["frontier_sector_entropy"].append(sector_entropy)
            drone_stats["frontier_cancellation"].append(frontier_cancellation)
            drone_stats["action_frontier_alignment"].append(action_frontier_alignment)
            drone_stats["action_frontier_intent"].append(action_frontier_intent)
            drone_stats["action_frontier_movement_gap"].append(action_frontier_movement_gap)
            for key, value in reward_terms.items():
                drone_stats["reward_terms"][key].append(value)
            drone_stats["is_edge_step"].append(float(is_edge_step))
            drone_stats["is_corner_step"].append(float(is_corner_step))
            drone_stats["boundary_distance_m"].append(float(boundary_distance_m[drone_idx]))
            drone_stats["footprint_radius_m"].append(footprint_radius)
            drone_stats["diagnostic_steps"] += 1
            for survivor_idx in np.flatnonzero(scout_credit[drone_idx]):
                drone_stats["scout_credit_count"] += 1
                drone_stats["scouted_survivors"].add(int(survivor_idx))
                drone_stats["first_scout_steps"].append(step + 1)

            if action_norm < 0.05 and displacement_m > 1.0:
                low_action_high_motion += 1
                drone_stats["low_action_high_motion"] += 1
            if action_norm > 0.5 and displacement_m < 0.25:
                high_action_low_motion += 1
                drone_stats["high_action_low_motion"] += 1
            if displacement_m > 1.0 and new_cells < 1.0:
                moving_no_new_coverage += 1
                drone_stats["moving_no_new_coverage"] += 1
            moving_no_confidence_gain_step = bool(
                displacement_m > 1.0
                and math.isfinite(confidence_weighted_gain_drone)
                and confidence_weighted_gain_drone <= moving_no_confidence_gain_threshold
            )
            if moving_no_confidence_gain_step:
                moving_no_confidence_gain += 1
                drone_stats["moving_no_confidence_gain"] += 1
            if frontier_obs_norm <= 1e-6:
                frontier_obs_empty_steps += 1
                drone_stats["frontier_obs_empty_steps"] += 1
            if math.isfinite(action_frontier_alignment) and action_frontier_alignment >= 0.50:
                action_frontier_aligned_steps += 1
                drone_stats["action_frontier_aligned_steps"] += 1
                if new_cells < 1.0:
                    action_frontier_aligned_no_new_steps += 1
                    drone_stats["action_frontier_aligned_no_new_steps"] += 1
                if is_edge_step:
                    action_frontier_aligned_edge_steps += 1
                    drone_stats["action_frontier_aligned_edge_steps"] += 1
            if math.isfinite(action_frontier_alignment) and action_frontier_alignment <= -0.50:
                action_frontier_anti_aligned_steps += 1
                drone_stats["action_frontier_anti_aligned_steps"] += 1
            if frontier_progress_frac >= 0.50:
                frontier_high_progress_steps += 1
                drone_stats["frontier_high_progress_steps"] += 1
                if new_cells < 1.0:
                    frontier_high_progress_no_new_steps += 1
                    drone_stats["frontier_high_progress_no_new_steps"] += 1
                if is_edge_step:
                    frontier_high_progress_edge_steps += 1
                    drone_stats["frontier_high_progress_edge_steps"] += 1
                if is_corner_step:
                    frontier_high_progress_corner_steps += 1
                    drone_stats["frontier_high_progress_corner_steps"] += 1

            action_displacement_alignment = math.nan
            displacement_norm_sim = float(np.linalg.norm(displacement_vec))
            if action_norm > 1e-6 and displacement_norm_sim > 1e-9:
                action_displacement_alignment = float(
                    np.dot(action_vec[:2], displacement_vec[:2]) / (action_norm * displacement_norm_sim)
                )
                action_displacement_alignment = max(min(action_displacement_alignment, 1.0), -1.0)
                action_displacement_alignments.append(action_displacement_alignment)
                drone_stats["alignments"].append(action_displacement_alignment)
                if new_cells >= 1.0:
                    action_displacement_alignments_new_cov.append(action_displacement_alignment)
                    drone_stats["alignments_new_cov"].append(action_displacement_alignment)
                else:
                    action_displacement_alignments_no_new_cov.append(action_displacement_alignment)
                    drone_stats["alignments_no_new_cov"].append(action_displacement_alignment)

            _append_time_bin(
                time_bins,
                step=step,
                max_steps=int(scenario_kwargs["max_steps"]),
                values={
                    "coverage_fraction": coverage_fraction_now,
                    "action_norm": action_norm,
                    "displacement_m": displacement_m,
                    "new_coverage_cells": new_cells,
                    "raw_new_coverage_cells": raw_new_cells,
                    "action_displacement_alignment": action_displacement_alignment,
                    "frontier_obs_distance": frontier_obs_distance,
                    "frontier_obs_vector_norm": frontier_obs_norm,
                    "frontier_obs_uncovered_ratio": frontier_obs_ratio,
                    "frontier_local_coverage_cos": frontier_local_cos,
                    "frontier_global_coverage_cos": frontier_global_cos,
                    "local_global_coverage_cos": local_global_cos,
                    "frontier_sector_cos": frontier_sector_cos,
                    "frontier_sector_dominance": sector_dominance,
                    "frontier_sector_entropy": sector_entropy,
                    "frontier_cancellation": frontier_cancellation,
                    "frontier_pairwise_cos": frontier_pairwise["mean_cos"],
                    "frontier_pairwise_same_dir": frontier_pairwise["same_dir_frac"],
                    "local_pairwise_same_dir": local_pairwise["same_dir_frac"],
                    "global_pairwise_same_dir": global_pairwise["same_dir_frac"],
                    "action_frontier_alignment": action_frontier_alignment,
                    "action_frontier_intent": action_frontier_intent,
                    "movement_frontier_alignment": frontier_align,
                    "frontier_progress": frontier_progress_frac,
                    "frontier_uncovered_ratio": frontier_ratio,
                    "frontier_reward": reward_terms["frontier"],
                    "confidence_reward": reward_terms["confidence"],
                    "team_confidence_reward": reward_terms["team_confidence"],
                    "team_confidence_overlap_penalty": reward_terms[
                        "team_confidence_overlap_penalty"
                    ],
                    "confidence_move_reward": reward_terms["confidence_move"],
                    "cleanup_target_progress_reward": reward_terms["cleanup_target_progress"],
                    "astar_progress_reward": reward_terms["astar_progress"],
                    "astar_progress_fraction": astar_progress_fraction,
                    "astar_frontier_gate": astar_frontier_gate,
                    "inefficient_move_penalty": reward_terms["inefficient_move_penalty"],
                    "confidence_overlap_penalty": reward_terms["confidence_overlap_penalty"],
                    "confidence_overlap_regret": confidence_overlap_regret,
                    "confidence_weighted_gain": confidence_weighted_gain_drone,
                    "coverage_reward": reward_terms["coverage"],
                    "move_coverage_reward": reward_terms["move_coverage"],
                    "overlap_penalty": reward_terms["overlap_penalty"],
                    "inter_uav_overlap_penalty": reward_terms["inter_uav_overlap_penalty"],
                    "outside_footprint_penalty": reward_terms["outside_footprint_penalty"],
                    "coverage_threshold_reward": reward_terms["coverage_threshold"],
                    "scout_reward": reward_terms["scout"],
                    "team_reward": reward_terms["team"],
                    "all_survivors_reward": reward_terms["all_survivors_found"],
                    "aux_reward": reward_terms["aux"],
                    "overlap": overlap,
                    "excess_overlap": excess_overlap,
                    "any_history_revisit": any_history_revisit,
                    "own_history_revisit": own_history_revisit,
                    "teammate_history_revisit": teammate_history_revisit,
                    "own_only_revisit": own_only_revisit,
                    "teammate_only_revisit": teammate_only_revisit,
                    "shared_history_revisit": shared_history_revisit,
                    "unavoidable_revisit": unavoidable_revisit,
                    "avoidable_revisit": avoidable_revisit,
                    "frontier_expected_new_cells": frontier_expected_cells,
                    "frontier_new_cell_capture": frontier_new_cell_capture,
                    "frontier_new_cell_gap": frontier_new_cell_gap,
                    "candidate_best_new_cells": candidate_best_new_cells,
                    "candidate_capture_fraction": candidate_capture_fraction,
                    "candidate_new_cell_regret": candidate_new_cell_regret,
                    "candidate_best_new_overlap": candidate_best_new_overlap,
                    "candidate_best_useful_overlap": candidate_best_useful_overlap,
                    "candidate_avoidable_overlap": candidate_avoidable_overlap,
                    "candidate_action_rank": candidate_action["rank"],
                    "candidate_movement_rank": candidate_movement["rank"],
                    "candidate_action_capture_fraction": candidate_action["capture_fraction"],
                    "candidate_movement_capture_fraction": candidate_movement["capture_fraction"],
                    "candidate_action_best_alignment": candidate_action["best_alignment"],
                    "candidate_movement_best_alignment": candidate_movement["best_alignment"],
                    "candidate_no_opportunity": candidate_no_opportunity,
                    "frontier_candidate_new_cells": float(frontier_use["new_cells"]),
                    "frontier_candidate_capture_fraction": frontier_candidate_capture,
                    "frontier_candidate_regret": float(frontier_use["regret"]),
                    "frontier_candidate_best_alignment": float(frontier_use["best_alignment"]),
                    "frontier_candidate_rank": float(frontier_use["rank"]),
                    "frontier_candidate_nearest_rank": float(frontier_use["nearest_rank"]),
                    "frontier_candidate_is_best": float(frontier_use["is_best"]),
                    "frontier_candidate_bad": float(frontier_use["bad"]),
                    "confidence_frontier_candidate_capture_fraction": (
                        confidence_frontier_candidate_capture
                    ),
                    "confidence_frontier_candidate_best_alignment": float(
                        confidence_frontier_use["best_alignment"]
                    ),
                    "confidence_frontier_candidate_rank": float(confidence_frontier_use["rank"]),
                    "confidence_frontier_candidate_bad": float(confidence_frontier_use["bad"]),
                    "confidence_lg_frontier_candidate_capture_fraction": (
                        confidence_lg_frontier_candidate_capture
                    ),
                    "confidence_lg_frontier_candidate_best_alignment": float(
                        confidence_lg_frontier_use["best_alignment"]
                    ),
                    "confidence_lg_frontier_candidate_rank": float(confidence_lg_frontier_use["rank"]),
                    "confidence_lg_frontier_candidate_bad": float(confidence_lg_frontier_use["bad"]),
                    "confidence_frontier_capture_advantage": confidence_frontier_capture_advantage,
                    "confidence_lg_frontier_capture_advantage": confidence_lg_frontier_capture_advantage,
                    "coverage_opportunity_cells": opportunity_cells,
                    "coverage_opportunity_fraction": opportunity_fraction,
                    "coverage_opportunity_available_fraction": opportunity_available_fraction,
                    "confidence_mean": confidence_mean,
                    "confidence_gain": confidence_gain_drone,
                    "confidence_opportunity_fraction": confidence_opportunity_fraction,
                    "confidence_opportunity_best_gain": confidence_opportunity_best_gain,
                    "confidence_revisit": float(confidence_revisit_step),
                    "confidence_useful_revisit": float(confidence_useful_revisit),
                    "confidence_wasteful_revisit": float(confidence_wasteful_revisit),
                    "confidence_ambiguous_revisit": float(confidence_ambiguous_revisit),
                    "confidence_team_gain": confidence_gain,
                    "confidence_low_fraction": confidence_low_fraction,
                    "confidence_high_fraction": confidence_high_fraction,
                    "confidence_pass_probability": confidence_pass_probability,
                    "confidence_overlap_fraction": confidence_overlap_fraction,
                    "confidence_overlap_regret": confidence_overlap_regret,
                    "confidence_step_detection_probability": confidence_step_detection_probability,
                    "cleanup_target_valid": cleanup_target_valid,
                    "cleanup_target_distance_m": cleanup_target_distance,
                    "cleanup_target_value": cleanup_target_value,
                    "cleanup_target_progress_m": cleanup_target_progress,
                    "cleanup_target_progress_fraction": cleanup_target_progress_fraction,
                    "cleanup_target_switch": cleanup_target_switch,
                    "cleanup_target_reached": cleanup_target_reached,
                    "cleanup_target_value_decay": cleanup_target_value_decay,
                    "cleanup_target_age": cleanup_target_age,
                    "cleanup_target_frontier_gate": cleanup_target_frontier_gate,
                    "outside_footprint": float(outside_footprint_fraction[drone_idx]),
                    "edge_step": float(is_edge_step),
                    "corner_step": float(is_corner_step),
                    "moving_no_new_coverage": float(displacement_m > 1.0 and new_cells < 1.0),
                    "moving_no_confidence_gain": float(moving_no_confidence_gain_step),
                    "frontier_obs_empty": float(frontier_obs_norm <= 1e-6),
                    "action_frontier_aligned": float(
                        math.isfinite(action_frontier_alignment)
                        and action_frontier_alignment >= 0.50
                    ),
                    "action_frontier_anti_aligned": float(
                        math.isfinite(action_frontier_alignment)
                        and action_frontier_alignment <= -0.50
                    ),
                    "frontier_high_progress": float(frontier_progress_frac >= 0.50),
                },
            )

        if full_diagnostics and scenario.n_drones > 0 and individual_coverage_history is not None:
            individual_coverage_history |= current_claims

        scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        for survivor_idx, is_scouted in enumerate(scouted):
            if (
                survivor_idx < len(active_survivor_mask)
                and active_survivor_mask[survivor_idx]
                and is_scouted
                and first_scout_steps[survivor_idx] is None
            ):
                first_scout_steps[survivor_idx] = step + 1
        confirmed = scenario.found_survivors[0].detach().cpu().numpy().astype(bool)
        for survivor_idx, is_confirmed in enumerate(confirmed):
            if (
                survivor_idx < len(active_survivor_mask)
                and active_survivor_mask[survivor_idx]
                and is_confirmed
                and first_confirm_steps[survivor_idx] is None
            ):
                first_confirm_steps[survivor_idx] = step + 1
        if (
            all(first_scout_steps[idx] is not None for idx in active_survivor_indices)
            and all(first_confirm_steps[idx] is not None for idx in active_survivor_indices)
        ):
            break

    active_scout_steps = [first_scout_steps[idx] for idx in active_survivor_indices]
    active_confirm_steps = [first_confirm_steps[idx] for idx in active_survivor_indices]
    scouted_count = sum(value is not None for value in active_scout_steps)
    missed_count = n_active_survivors - scouted_count
    scout_steps = [value for value in active_scout_steps if value is not None]
    all_scouted_step = (
        max(scout_steps)
        if n_active_survivors > 0 and scouted_count == n_active_survivors and scout_steps
        else (0 if n_active_survivors == 0 else None)
    )
    confirmed_count = sum(value is not None for value in active_confirm_steps)
    unconfirmed_count = n_active_survivors - confirmed_count
    confirm_steps = [value for value in active_confirm_steps if value is not None]
    all_confirmed_step = (
        max(confirm_steps)
        if n_active_survivors > 0 and confirmed_count == n_active_survivors and confirm_steps
        else (0 if n_active_survivors == 0 else None)
    )
    final_coverage_fraction = float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
    path_metrics = _path_metrics(
        path_positions_sim,
        displacement_m_values,
        boundary_distance_m_values,
        footprint_radius_m_values,
        scenario,
    )
    final_coverage_grid = scenario.coverage_grid[0].detach().cpu().numpy().astype(bool)
    coverage_shape_metrics = (
        _coverage_shape_metrics(
            final_coverage_grid,
            scenario,
            _finite_mean(footprint_radius_m_values),
        )
        if full_diagnostics
        else _fast_coverage_shape_metrics(final_coverage_grid)
    )
    per_drone = (
        [_finalize_drone_stats(stats, scenario) for stats in per_drone_stats]
        if full_diagnostics
        else [
            {
                "drone": int(drone_idx),
                "path_length_m": float(fast_drone_path_lengths[drone_idx]),
                "avg_displacement_m": _finite_mean(fast_drone_displacements[drone_idx]),
                "avg_action_norm": _finite_mean(fast_drone_action_norms[drone_idx]),
            }
            for drone_idx in range(int(scenario.n_drones))
        ]
    )
    final_survivor_confidence = _survivor_confidence_values(scenario)
    if full_diagnostics:
        survivor_exposures = _finalize_survivor_exposure_stats(
            survivor_exposure_stats,
            first_scout_steps,
        )
        survivor_exposures = [
            exposure
            for exposure in survivor_exposures
            if int(exposure.get("survivor", -1)) in set(int(idx) for idx in active_survivor_indices)
        ]
        for exposure in survivor_exposures:
            survivor_idx = int(exposure.get("survivor", -1))
            exposure["final_confidence"] = (
                float(final_survivor_confidence[survivor_idx])
                if 0 <= survivor_idx < len(final_survivor_confidence)
                else math.nan
            )
    else:
        survivor_exposures = []
    survivor_exposure_summary = _survivor_exposure_summary(survivor_exposures)
    row = {
        "seed": int(seed),
        "max_steps": int(scenario_kwargs["max_steps"]),
        "survivors": n_active_survivors,
        "active_survivors": n_active_survivors,
        "survivor_slots": survivor_slots,
        "scouted": scouted_count,
        "missed": missed_count,
        "recall": scouted_count / n_active_survivors if n_active_survivors else 1.0,
        "confirmed": confirmed_count,
        "unconfirmed": unconfirmed_count,
        "confirmation_recall": confirmed_count / n_active_survivors if n_active_survivors else 1.0,
        "final_coverage_fraction": final_coverage_fraction,
        "final_confidence_mean": float(
            scenario.uav_confidence_grid[0].float().mean().detach().cpu().item()
        ),
        "final_confidence_low_fraction": float(
            (scenario.uav_confidence_grid[0] < 0.50).float().mean().detach().cpu().item()
        ),
        "final_confidence_high_fraction": float(
            (scenario.uav_confidence_grid[0] >= 0.80).float().mean().detach().cpu().item()
        ),
        "final_survivor_confidence": final_survivor_confidence,
        "full_success": float(scouted_count == n_active_survivors),
        "full_confirmation_success": float(confirmed_count == n_active_survivors),
        "avg_scout_step": float(np.mean(scout_steps)) if scout_steps else math.nan,
        "avg_scout_time_s": float(np.mean(scout_steps) * step_seconds) if scout_steps else math.nan,
        "all_scouted_step": all_scouted_step,
        "all_scouted_time_s": None if all_scouted_step is None else float(all_scouted_step * step_seconds),
        "first_scout_steps": first_scout_steps,
        "first_scout_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_scout_steps
        ],
        "avg_confirm_step": float(np.mean(confirm_steps)) if confirm_steps else math.nan,
        "avg_confirm_time_s": float(np.mean(confirm_steps) * step_seconds) if confirm_steps else math.nan,
        "all_confirmed_step": all_confirmed_step,
        "all_confirmed_time_s": (
            None if all_confirmed_step is None else float(all_confirmed_step * step_seconds)
        ),
        "first_confirm_steps": first_confirm_steps,
        "first_confirm_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_confirm_steps
        ],
        "survivor_exposures": survivor_exposures,
        "avg_action_norm": _finite_mean(action_norms),
        "avg_displacement_m": _finite_mean(displacement_m_values),
        "avg_action_displacement_alignment": _finite_mean(action_displacement_alignments),
        "avg_action_displacement_alignment_new_cov": _finite_mean(action_displacement_alignments_new_cov),
        "avg_action_displacement_alignment_no_new_cov": _finite_mean(action_displacement_alignments_no_new_cov),
        "avg_action_frontier_alignment": _finite_mean(action_frontier_alignment_values),
        "avg_action_frontier_alignment_new_cov": _finite_mean(action_frontier_alignment_new_cov_values),
        "avg_action_frontier_alignment_no_new_cov": _finite_mean(action_frontier_alignment_no_new_cov_values),
        "avg_action_frontier_intent": _finite_mean(action_frontier_intent_values),
        "avg_action_frontier_movement_gap": _finite_mean(action_frontier_movement_gap_values),
        "avg_new_coverage_cells": _finite_mean(new_coverage_cells_values),
        "avg_raw_new_coverage_cells": _finite_mean(raw_new_coverage_cells_values),
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
        "avg_any_history_revisit_fraction": _finite_mean(any_history_revisit_values),
        "avg_own_history_revisit_fraction": _finite_mean(own_history_revisit_values),
        "avg_teammate_history_revisit_fraction": _finite_mean(teammate_history_revisit_values),
        "avg_own_only_revisit_fraction": _finite_mean(own_only_revisit_values),
        "avg_teammate_only_revisit_fraction": _finite_mean(teammate_only_revisit_values),
        "avg_shared_history_revisit_fraction": _finite_mean(shared_history_revisit_values),
        "avg_unavoidable_revisit_fraction": _finite_mean(unavoidable_revisit_values),
        "avg_avoidable_revisit_fraction": _finite_mean(avoidable_revisit_values),
        "avg_frontier_expected_new_cells": _finite_mean(frontier_expected_new_cells_values),
        "avg_frontier_new_cell_capture_fraction": _finite_mean(frontier_new_cell_capture_values),
        "avg_frontier_new_cell_gap": _finite_mean(frontier_new_cell_gap_values),
        "avg_candidate_best_new_cells": _finite_mean(candidate_best_new_cells_values),
        "avg_candidate_capture_fraction": _finite_mean(candidate_capture_fraction_values),
        "avg_candidate_new_cell_regret": _finite_mean(candidate_new_cell_regret_values),
        "avg_candidate_best_new_overlap": _finite_mean(candidate_best_new_overlap_values),
        "avg_candidate_best_useful_overlap": _finite_mean(candidate_best_useful_overlap_values),
        "avg_candidate_avoidable_overlap": _finite_mean(candidate_avoidable_overlap_values),
        "avg_candidate_action_rank": _finite_mean(candidate_action_rank_values),
        "avg_candidate_movement_rank": _finite_mean(candidate_movement_rank_values),
        "avg_candidate_action_capture_fraction": _finite_mean(candidate_action_capture_values),
        "avg_candidate_movement_capture_fraction": _finite_mean(candidate_movement_capture_values),
        "avg_candidate_action_best_alignment": _finite_mean(candidate_action_best_alignment_values),
        "avg_candidate_movement_best_alignment": _finite_mean(candidate_movement_best_alignment_values),
        "candidate_no_opportunity_frac": _finite_mean(candidate_no_opportunity_values),
        "avg_frontier_candidate_new_cells": _finite_mean(frontier_candidate_new_cells_values),
        "avg_frontier_candidate_capture_fraction": _finite_mean(
            frontier_candidate_capture_fraction_values
        ),
        "avg_frontier_candidate_regret": _finite_mean(frontier_candidate_regret_values),
        "avg_frontier_candidate_best_alignment": _finite_mean(
            frontier_candidate_best_alignment_values
        ),
        "avg_frontier_candidate_rank": _finite_mean(frontier_candidate_rank_values),
        "avg_frontier_candidate_nearest_rank": _finite_mean(
            frontier_candidate_nearest_rank_values
        ),
        "frontier_candidate_is_best_frac": _finite_mean(frontier_candidate_is_best_values),
        "frontier_candidate_bad_frac": _finite_mean(frontier_candidate_bad_values),
        "avg_confidence_frontier_candidate_capture_fraction": _finite_mean(
            confidence_frontier_candidate_capture_fraction_values
        ),
        "avg_confidence_frontier_candidate_best_alignment": _finite_mean(
            confidence_frontier_candidate_best_alignment_values
        ),
        "avg_confidence_frontier_candidate_rank": _finite_mean(
            confidence_frontier_candidate_rank_values
        ),
        "confidence_frontier_candidate_bad_frac": _finite_mean(
            confidence_frontier_candidate_bad_values
        ),
        "avg_confidence_lg_frontier_candidate_capture_fraction": _finite_mean(
            confidence_lg_frontier_candidate_capture_fraction_values
        ),
        "avg_confidence_lg_frontier_candidate_best_alignment": _finite_mean(
            confidence_lg_frontier_candidate_best_alignment_values
        ),
        "avg_confidence_lg_frontier_candidate_rank": _finite_mean(
            confidence_lg_frontier_candidate_rank_values
        ),
        "confidence_lg_frontier_candidate_bad_frac": _finite_mean(
            confidence_lg_frontier_candidate_bad_values
        ),
        "avg_confidence_frontier_capture_advantage": _finite_mean(
            confidence_frontier_capture_advantage_values
        ),
        "avg_confidence_lg_frontier_capture_advantage": _finite_mean(
            confidence_lg_frontier_capture_advantage_values
        ),
        "frontier_candidate_capture_new_cells_corr": _safe_corr(
            frontier_candidate_capture_fraction_values,
            new_coverage_cells_values,
        ),
        "frontier_candidate_alignment_new_cells_corr": _safe_corr(
            frontier_candidate_best_alignment_values,
            new_coverage_cells_values,
        ),
        "avg_coverage_opportunity_cells": _finite_mean(coverage_opportunity_cells_values),
        "avg_coverage_opportunity_fraction": _finite_mean(coverage_opportunity_fraction_values),
        "avg_coverage_opportunity_available_fraction": _finite_mean(
            coverage_opportunity_available_fraction_values
        ),
        "avg_confidence_mean": _finite_mean(confidence_mean_values),
        "avg_confidence_gain": _finite_mean(confidence_gain_values),
        "avg_confidence_gain_by_drone": _finite_mean(confidence_gain_by_drone_values),
        "avg_confidence_weighted_gain": _finite_mean(confidence_weighted_gain_values),
        "avg_confidence_weighted_gain_by_drone": _finite_mean(confidence_weighted_gain_by_drone_values),
        "avg_confidence_opportunity_fraction": _finite_mean(confidence_opportunity_fraction_values),
        "avg_confidence_opportunity_best_gain": _finite_mean(confidence_opportunity_best_gain_values),
        "avg_confidence_low_fraction": _finite_mean(confidence_low_fraction_values),
        "avg_confidence_high_fraction": _finite_mean(confidence_high_fraction_values),
        "avg_confidence_step_detection_probability": _finite_mean(
            confidence_step_detection_probability_values
        ),
        "avg_confidence_pass_probability": _finite_mean(
            confidence_step_detection_probability_by_drone_values
        ),
        "avg_confidence_overlap_fraction": _finite_mean(confidence_overlap_fraction_values),
        "avg_confidence_overlap_regret": _finite_mean(confidence_overlap_regret_values),
        "avg_cleanup_target_valid_fraction": _finite_mean(cleanup_target_valid_values),
        "avg_cleanup_target_distance_m": _finite_mean(cleanup_target_distance_values),
        "avg_cleanup_target_value": _finite_mean(cleanup_target_value_values),
        "avg_cleanup_target_progress_m": _finite_mean(cleanup_target_progress_values),
        "avg_cleanup_target_progress_fraction": _finite_mean(cleanup_target_progress_fraction_values),
        "cleanup_target_switch_rate": _finite_mean(cleanup_target_switch_values),
        "cleanup_target_reached_rate": _finite_mean(cleanup_target_reached_values),
        "avg_cleanup_target_value_decay": _finite_mean(cleanup_target_value_decay_values),
        "cleanup_target_no_progress_frac": _finite_mean(cleanup_target_no_progress_values),
        "cleanup_target_progress_with_new_cells_frac": _finite_mean(
            cleanup_target_progress_with_new_cells_values
        ),
        "cleanup_target_progress_with_excess_overlap_frac": _finite_mean(
            cleanup_target_progress_with_excess_overlap_values
        ),
        "avg_cleanup_target_frontier_gate": _finite_mean(cleanup_target_frontier_gate_values),
        "avg_frontier_alignment": _finite_mean(frontier_alignment_values),
        "avg_frontier_progress_fraction": _finite_mean(frontier_progress_values),
        "avg_frontier_uncovered_ratio": _finite_mean(frontier_uncovered_ratio_values),
        "avg_frontier_obs_distance": _finite_mean(frontier_obs_distance_values),
        "avg_frontier_obs_vector_norm": _finite_mean(frontier_obs_vector_norm_values),
        "avg_frontier_local_coverage_cos": _finite_mean(frontier_local_coverage_cos_values),
        "avg_frontier_global_coverage_cos": _finite_mean(frontier_global_coverage_cos_values),
        "avg_local_global_coverage_cos": _finite_mean(local_global_coverage_cos_values),
        "avg_frontier_sector_cos": _finite_mean(frontier_sector_cos_values),
        "avg_frontier_sector_dominance": _finite_mean(frontier_sector_dominance_values),
        "avg_frontier_sector_entropy": _finite_mean(frontier_sector_entropy_values),
        "avg_frontier_cancellation": _finite_mean(frontier_cancellation_values),
        "avg_frontier_pairwise_cos": _finite_mean(frontier_pairwise_cos_values),
        "avg_frontier_pairwise_same_dir": _finite_mean(frontier_pairwise_same_dir_values),
        "avg_local_pairwise_same_dir": _finite_mean(local_pairwise_same_dir_values),
        "avg_global_pairwise_same_dir": _finite_mean(global_pairwise_same_dir_values),
        "avg_reward_uav_coverage": _finite_mean(reward_uav_coverage_values),
        "avg_reward_uav_move_coverage": _finite_mean(reward_uav_move_coverage_values),
        "avg_reward_uav_frontier": _finite_mean(reward_uav_frontier_values),
        "avg_reward_uav_confidence": _finite_mean(reward_uav_confidence_values),
        "avg_reward_uav_team_confidence": _finite_mean(reward_uav_team_confidence_values),
        "avg_penalty_uav_team_confidence_overlap": _finite_mean(
            penalty_uav_team_confidence_overlap_values
        ),
        "avg_reward_uav_confidence_move": _finite_mean(reward_uav_confidence_move_values),
        "avg_reward_uav_cleanup_target_progress": _finite_mean(
            reward_uav_cleanup_target_progress_values
        ),
        "avg_reward_uav_astar_progress": _finite_mean(reward_uav_astar_progress_values),
        "avg_penalty_uav_inefficient_move": _finite_mean(penalty_uav_inefficient_move_values),
        "avg_penalty_uav_confidence_overlap": _finite_mean(penalty_uav_confidence_overlap_values),
        "avg_penalty_uav_overlap": _finite_mean(penalty_uav_overlap_values),
        "avg_penalty_uav_inter_overlap": _finite_mean(penalty_uav_inter_overlap_values),
        "avg_penalty_uav_outside_footprint": _finite_mean(penalty_uav_outside_footprint_values),
        "avg_reward_uav_coverage_threshold": _finite_mean(reward_uav_coverage_threshold_values),
        "avg_reward_uav_scout": _finite_mean(reward_uav_scout_values),
        "avg_reward_team": _finite_mean(reward_team_values),
        "avg_reward_all_survivors_found": _finite_mean(reward_all_survivors_found_values),
        "avg_reward_uav_aux": _finite_mean(reward_uav_aux_values),
        "avg_frontier_abs_reward_share": _finite_mean(frontier_abs_reward_share_values),
        "frontier_high_progress_step_frac": (
            frontier_high_progress_steps / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "frontier_high_progress_no_new_frac": (
            frontier_high_progress_no_new_steps / frontier_high_progress_steps
            if frontier_high_progress_steps else 0.0
        ),
        "frontier_high_progress_edge_frac": (
            frontier_high_progress_edge_steps / frontier_high_progress_steps
            if frontier_high_progress_steps else 0.0
        ),
        "frontier_high_progress_corner_frac": (
            frontier_high_progress_corner_steps / frontier_high_progress_steps
            if frontier_high_progress_steps else 0.0
        ),
        "frontier_edge_progress_mean": _finite_mean(frontier_progress_edge_values),
        "frontier_interior_progress_mean": _finite_mean(frontier_progress_interior_values),
        "frontier_edge_reward_mean": _finite_mean(frontier_reward_edge_values),
        "frontier_interior_reward_mean": _finite_mean(frontier_reward_interior_values),
        "frontier_edge_new_cells_mean": _finite_mean(frontier_new_cells_edge_values),
        "frontier_interior_new_cells_mean": _finite_mean(frontier_new_cells_interior_values),
        "frontier_edge_outside_mean": _finite_mean(frontier_outside_edge_values),
        "frontier_interior_outside_mean": _finite_mean(frontier_outside_interior_values),
        "frontier_progress_new_cells_corr": _safe_corr(frontier_progress_values, new_coverage_cells_values),
        "frontier_expected_raw_new_cells_corr": _safe_corr(
            frontier_expected_new_cells_values,
            raw_new_coverage_cells_values,
        ),
        "frontier_progress_boundary_distance_corr": _safe_corr(
            frontier_progress_values,
            boundary_distance_m_values,
        ),
        "frontier_obs_empty_step_frac": (
            frontier_obs_empty_steps / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "action_frontier_aligned_step_frac": (
            action_frontier_aligned_steps / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "action_frontier_anti_aligned_step_frac": (
            action_frontier_anti_aligned_steps / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "action_frontier_aligned_no_new_frac": (
            action_frontier_aligned_no_new_steps / action_frontier_aligned_steps
            if action_frontier_aligned_steps else 0.0
        ),
        "action_frontier_aligned_edge_frac": (
            action_frontier_aligned_edge_steps / action_frontier_aligned_steps
            if action_frontier_aligned_steps else 0.0
        ),
        "action_frontier_alignment_new_cells_corr": _safe_corr(
            action_frontier_alignment_values,
            new_coverage_cells_values,
        ),
        "action_frontier_alignment_boundary_distance_corr": _safe_corr(
            action_frontier_alignment_values,
            boundary_distance_m_values,
        ),
        "time_bins": _finalize_time_bins(time_bins),
        "perception_time_bins": _finalize_time_bins(perception_time_bins),
        "excess_overlap_step_frac_10": (
            float(np.mean([value >= 0.10 for value in excess_overlap_values]))
            if excess_overlap_values else 0.0
        ),
        "inter_uav_overlap_step_frac_20": (
            float(np.mean([value >= 0.20 for value in inter_uav_overlap_values]))
            if inter_uav_overlap_values else 0.0
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
        "raw_new_coverage_step_frac": (
            float(np.mean([value >= 1.0 for value in raw_new_coverage_cells_values]))
            if raw_new_coverage_cells_values else 0.0
        ),
        "low_action_high_motion_frac": low_action_high_motion / diagnostic_steps if diagnostic_steps else 0.0,
        "high_action_low_motion_frac": high_action_low_motion / diagnostic_steps if diagnostic_steps else 0.0,
        "moving_no_new_coverage_frac": moving_no_new_coverage / diagnostic_steps if diagnostic_steps else 0.0,
        "moving_no_confidence_gain_frac": (
            moving_no_confidence_gain / diagnostic_steps if diagnostic_steps else 0.0
        ),
        "per_drone": per_drone,
        **start_metrics,
        **path_metrics,
        **coverage_shape_metrics,
        **survivor_exposure_summary,
    }
    row.update(_confidence_revisit_metrics(
        avoidable_revisit_values,
        confidence_gain_by_drone_values,
        confidence_opportunity_fraction_values,
        confidence_opportunity_best_gain_values,
    ))
    row["failure_label"] = _failure_label(row)
    close = getattr(env, "close", None)
    if close is not None:
        close()
    return row


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def _confidence_revisit_metrics(
    avoidable_revisit: list[float],
    confidence_gain: list[float],
    confidence_opportunity_fraction: list[float],
    confidence_opportunity_best_gain: list[float],
) -> dict[str, float]:
    series = (
        avoidable_revisit,
        confidence_gain,
        confidence_opportunity_fraction,
        confidence_opportunity_best_gain,
    )
    n = min((len(values) for values in series), default=0)
    if n <= 0:
        return {
            "confidence_revisit_step_frac": 0.0,
            "confidence_useful_revisit_step_frac": 0.0,
            "confidence_wasteful_revisit_step_frac": 0.0,
            "confidence_ambiguous_revisit_step_frac": 0.0,
            "confidence_revisit_useful_share": math.nan,
            "confidence_revisit_wasteful_share": math.nan,
            "confidence_revisit_gain_share": math.nan,
            "confidence_gain_on_revisit": math.nan,
            "confidence_gain_off_revisit": math.nan,
            "confidence_opportunity_on_revisit": math.nan,
            "confidence_opportunity_off_revisit": math.nan,
            "confidence_best_gain_on_revisit": math.nan,
            "confidence_best_gain_off_revisit": math.nan,
        }

    revisit = np.asarray(avoidable_revisit[:n], dtype=float)
    gain = np.asarray(confidence_gain[:n], dtype=float)
    opportunity = np.asarray(confidence_opportunity_fraction[:n], dtype=float)
    best_gain = np.asarray(confidence_opportunity_best_gain[:n], dtype=float)

    revisit_mask = (
        np.isfinite(revisit)
        & (revisit >= CONFIDENCE_REVISIT_THRESHOLD)
    )
    has_confidence_opportunity = (
        np.isfinite(gain)
        & np.isfinite(opportunity)
        & np.isfinite(best_gain)
        & (gain > CONFIDENCE_REVISIT_MIN_GAIN)
        & (best_gain > CONFIDENCE_REVISIT_MIN_GAIN)
    )
    useful_mask = (
        revisit_mask
        & has_confidence_opportunity
        & (opportunity >= CONFIDENCE_REVISIT_USEFUL_OPPORTUNITY_THRESHOLD)
    )
    wasteful_mask = (
        revisit_mask
        & (
            ~has_confidence_opportunity
            | (opportunity < CONFIDENCE_REVISIT_WASTEFUL_OPPORTUNITY_THRESHOLD)
        )
    )
    ambiguous_mask = revisit_mask & ~useful_mask & ~wasteful_mask
    revisit_count = int(np.count_nonzero(revisit_mask))

    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        selected = values[mask & np.isfinite(values)]
        return float(np.mean(selected)) if selected.size else math.nan

    positive_gain = np.where(np.isfinite(gain) & (gain > 0.0), gain, 0.0)
    total_gain = float(np.sum(positive_gain))
    revisit_gain = float(np.sum(positive_gain[revisit_mask]))
    return {
        "confidence_revisit_step_frac": float(np.mean(revisit_mask)),
        "confidence_useful_revisit_step_frac": float(np.mean(useful_mask)),
        "confidence_wasteful_revisit_step_frac": float(np.mean(wasteful_mask)),
        "confidence_ambiguous_revisit_step_frac": float(np.mean(ambiguous_mask)),
        "confidence_revisit_useful_share": (
            float(np.count_nonzero(useful_mask) / revisit_count)
            if revisit_count else math.nan
        ),
        "confidence_revisit_wasteful_share": (
            float(np.count_nonzero(wasteful_mask) / revisit_count)
            if revisit_count else math.nan
        ),
        "confidence_revisit_gain_share": (
            float(revisit_gain / total_gain) if total_gain > 0.0 else math.nan
        ),
        "confidence_gain_on_revisit": _masked_mean(gain, revisit_mask),
        "confidence_gain_off_revisit": _masked_mean(gain, ~revisit_mask),
        "confidence_opportunity_on_revisit": _masked_mean(opportunity, revisit_mask),
        "confidence_opportunity_off_revisit": _masked_mean(opportunity, ~revisit_mask),
        "confidence_best_gain_on_revisit": _masked_mean(best_gain, revisit_mask),
        "confidence_best_gain_off_revisit": _masked_mean(best_gain, ~revisit_mask),
    }


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


def _metric_scalar(scenario: WildfireSearchScenario, name: str) -> float:
    value = getattr(scenario, name, None)
    if value is None:
        return 0.0
    arr = value.detach().cpu().numpy().astype(float).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _survivor_confidence_values(scenario: WildfireSearchScenario) -> list[float]:
    confidence_grid = getattr(scenario, "uav_confidence_grid", None)
    if confidence_grid is None or int(getattr(scenario, "n_survivors", 0)) <= 0:
        return []
    with torch.no_grad():
        surv_pos = torch.stack([survivor.state.pos for survivor in scenario._survivors], dim=1)
        gx, gy = scenario._positions_to_grid(surv_pos, grid_size=int(confidence_grid.shape[-1]))
        values = confidence_grid[0, gy[0], gx[0]].detach().cpu().numpy().astype(float)
    return [float(value) for value in values.reshape(-1)]


def _new_survivor_exposure_stats(n_survivors: int) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for survivor_idx in range(max(int(n_survivors), 0)):
        stats.append({
            "survivor": int(survivor_idx),
            "exposure_steps": 0,
            "pair_exposures": 0,
            "miss_attempt_steps": 0,
            "expected_detection_mass": 0.0,
            "missed_detection_probability_mass": 0.0,
            "cumulative_no_detection_probability": 1.0,
            "step_detection_probabilities": [],
            "step_best_norm_distances": [],
            "step_best_probability_norm_distances": [],
            "best_detection_probability": 0.0,
            "best_pair_detection_probability": 0.0,
            "best_norm_distance": math.inf,
            "best_probability_norm_distance": math.inf,
            "min_distance_m": math.inf,
            "best_margin_m": -math.inf,
            "near_edge_exposure_steps": 0,
            "very_edge_exposure_steps": 0,
            "central_exposure_steps": 0,
            "first_exposure_step": None,
            "last_exposure_step": None,
            "first_scout_step": None,
            "scout_drone": None,
            "scout_probability": math.nan,
            "scout_distance_m": math.nan,
            "scout_norm_distance": math.nan,
            "scout_margin_m": math.nan,
            "scout_distance_factor": math.nan,
            "scout_environment_factor": math.nan,
            "scout_cover_factor": math.nan,
            "scout_fire_smoke_factor": math.nan,
            "scout_altitude_quality": math.nan,
            "scout_confidence_pre": math.nan,
            "scout_confidence_post": math.nan,
            "land_cover": None,
        })
    return stats


def _drone_perception_snapshot(
    scenario: WildfireSearchScenario,
    meters_per_sim: float,
) -> dict[str, np.ndarray]:
    n_drones = int(getattr(scenario, "n_drones", 0))
    n_survivors = int(getattr(scenario, "n_survivors", 0))
    if n_drones <= 0 or n_survivors <= 0:
        shape = (max(n_drones, 0), max(n_survivors, 0))
        return {
            "probability": np.zeros(shape, dtype=float),
            "visible": np.zeros(shape, dtype=bool),
            "distance_m": np.zeros(shape, dtype=float),
            "norm_distance": np.full(shape, math.inf, dtype=float),
            "margin_m": np.full(shape, -math.inf, dtype=float),
            "distance_factor": np.zeros(shape, dtype=float),
            "environment_factor": np.zeros(shape, dtype=float),
            "cover_factor": np.zeros(shape, dtype=float),
            "fire_smoke_factor": np.zeros(shape, dtype=float),
            "altitude_quality": np.zeros(shape, dtype=float),
            "land_cover": np.zeros(max(n_survivors, 0), dtype=int),
        }

    with torch.no_grad():
        drone_pos = torch.stack(
            [agent.state.pos for agent in scenario.world.agents[:n_drones]],
            dim=1,
        )
        surv_pos = torch.stack([survivor.state.pos for survivor in scenario._survivors], dim=1)
        drone_dists = torch.cdist(drone_pos, surv_pos)
        components = scenario._drone_detection_components(drone_dists, drone_pos, surv_pos)

    probability = components["probability"][0].detach().cpu().numpy().astype(float)
    visible = components["visible"][0].detach().cpu().numpy().astype(bool)
    footprint_sim = components["footprint"][0].detach().cpu().numpy().astype(float).reshape(n_drones, 1)
    distance_sim = drone_dists[0].detach().cpu().numpy().astype(float)
    distance_m = distance_sim * float(meters_per_sim)
    footprint_m = footprint_sim * float(meters_per_sim)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_distance = distance_sim / np.maximum(footprint_sim, 1e-12)
    norm_distance = np.where(np.isfinite(norm_distance), norm_distance, math.inf)
    margin_m = footprint_m - distance_m

    return {
        "probability": probability,
        "visible": visible,
        "distance_m": distance_m,
        "norm_distance": norm_distance,
        "margin_m": margin_m,
        "distance_factor": components["distance_factor"][0].detach().cpu().numpy().astype(float),
        "environment_factor": components["environment_factor"][0].detach().cpu().numpy().astype(float),
        "cover_factor": components["environment_factor"][0].detach().cpu().numpy().astype(float),
        "fire_smoke_factor": components["fire_smoke_factor"][0].detach().cpu().numpy().astype(float),
        "altitude_quality": components["altitude_quality"][0].detach().cpu().numpy().astype(float),
        "land_cover": components["survivor_cover"][0].detach().cpu().numpy().astype(int),
    }


def _update_survivor_exposure_stats(
    stats: list[dict[str, Any]],
    *,
    perception: dict[str, np.ndarray],
    drone_detections: np.ndarray,
    prev_scouted: np.ndarray,
    post_scouted: np.ndarray,
    survivor_confidence_pre: list[float],
    survivor_confidence_post: list[float],
    step: int,
    n_survivors: int,
    active_survivor_mask: np.ndarray | None = None,
) -> dict[str, float]:
    probability = np.asarray(perception["probability"], dtype=float)
    visible = np.asarray(perception["visible"], dtype=bool)
    norm_distance = np.asarray(perception["norm_distance"], dtype=float)
    distance_m = np.asarray(perception["distance_m"], dtype=float)
    margin_m = np.asarray(perception["margin_m"], dtype=float)
    land_cover = np.asarray(perception["land_cover"], dtype=int).reshape(-1)
    detections = np.asarray(drone_detections, dtype=bool)
    n_survivors = max(int(n_survivors), 0)
    if active_survivor_mask is None:
        active_mask = np.ones(n_survivors, dtype=bool)
    else:
        active_mask = np.asarray(active_survivor_mask, dtype=bool).reshape(-1)[:n_survivors]
        if active_mask.size < n_survivors:
            active_mask = np.pad(active_mask, (0, n_survivors - active_mask.size), constant_values=False)
    n_active_survivors = int(np.count_nonzero(active_mask))

    exposed_count = 0
    expected_scouts = 0.0
    missed_probability_mass = 0.0
    actual_new_scouts = 0.0
    best_step_probabilities: list[float] = []
    best_step_norms: list[float] = []
    near_edge_exposed = 0
    central_exposed = 0
    unscouted_count = int(np.count_nonzero(active_mask & ~prev_scouted[:n_survivors]))

    for survivor_idx in range(n_survivors):
        if (
            survivor_idx >= len(stats)
            or not bool(active_mask[survivor_idx])
            or bool(prev_scouted[survivor_idx])
        ):
            continue
        stat = stats[survivor_idx]
        if survivor_idx < land_cover.size:
            stat["land_cover"] = int(land_cover[survivor_idx])

        survivor_visible = visible[:, survivor_idx] if visible.size else np.zeros(0, dtype=bool)
        survivor_prob = probability[:, survivor_idx] if probability.size else np.zeros(0, dtype=float)
        survivor_detected = (
            detections[:, survivor_idx]
            if detections.size
            else np.zeros_like(survivor_prob, dtype=bool)
        )
        if not survivor_visible.any():
            continue

        visible_probs = np.clip(survivor_prob[survivor_visible], 0.0, 1.0)
        step_detection_probability = float(1.0 - np.prod(1.0 - visible_probs))
        visible_drone_indices = np.flatnonzero(survivor_visible)
        visible_norms = norm_distance[visible_drone_indices, survivor_idx]
        visible_distances = distance_m[visible_drone_indices, survivor_idx]
        visible_margins = margin_m[visible_drone_indices, survivor_idx]

        best_norm_distance = float(np.nanmin(visible_norms)) if visible_norms.size else math.nan
        best_probability_local_idx = int(np.nanargmax(visible_probs)) if visible_probs.size else 0
        best_probability_drone = int(visible_drone_indices[best_probability_local_idx])
        best_pair_probability = float(visible_probs[best_probability_local_idx]) if visible_probs.size else 0.0
        best_probability_norm = float(norm_distance[best_probability_drone, survivor_idx])
        min_distance_m = float(np.nanmin(visible_distances)) if visible_distances.size else math.nan
        best_margin_m = float(np.nanmax(visible_margins)) if visible_margins.size else math.nan

        exposed_count += 1
        expected_scouts += step_detection_probability
        best_step_probabilities.append(step_detection_probability)
        best_step_norms.append(best_norm_distance)
        if math.isfinite(best_norm_distance) and best_norm_distance >= 0.75:
            near_edge_exposed += 1
        if math.isfinite(best_norm_distance) and best_norm_distance <= 0.50:
            central_exposed += 1

        stat["exposure_steps"] += 1
        stat["pair_exposures"] += int(np.count_nonzero(survivor_visible))
        stat["expected_detection_mass"] += step_detection_probability
        stat["cumulative_no_detection_probability"] *= max(1.0 - step_detection_probability, 0.0)
        stat["step_detection_probabilities"].append(step_detection_probability)
        stat["step_best_norm_distances"].append(best_norm_distance)
        stat["step_best_probability_norm_distances"].append(best_probability_norm)
        stat["best_detection_probability"] = max(
            float(stat["best_detection_probability"]),
            step_detection_probability,
        )
        stat["best_pair_detection_probability"] = max(
            float(stat["best_pair_detection_probability"]),
            best_pair_probability,
        )
        stat["best_norm_distance"] = min(float(stat["best_norm_distance"]), best_norm_distance)
        stat["best_probability_norm_distance"] = min(
            float(stat["best_probability_norm_distance"]),
            best_probability_norm,
        )
        stat["min_distance_m"] = min(float(stat["min_distance_m"]), min_distance_m)
        stat["best_margin_m"] = max(float(stat["best_margin_m"]), best_margin_m)
        stat["near_edge_exposure_steps"] += int(math.isfinite(best_norm_distance) and best_norm_distance >= 0.75)
        stat["very_edge_exposure_steps"] += int(math.isfinite(best_norm_distance) and best_norm_distance >= 0.90)
        stat["central_exposure_steps"] += int(math.isfinite(best_norm_distance) and best_norm_distance <= 0.50)
        if stat["first_exposure_step"] is None:
            stat["first_exposure_step"] = int(step)
        stat["last_exposure_step"] = int(step)

        detected_now = bool(survivor_detected.any())
        if not detected_now:
            stat["miss_attempt_steps"] += 1
            stat["missed_detection_probability_mass"] += step_detection_probability
            missed_probability_mass += step_detection_probability
        else:
            actual_new_scouts += 1.0
            detected_drone_indices = np.flatnonzero(survivor_detected)
            detected_probs = survivor_prob[detected_drone_indices]
            detected_choice = int(detected_drone_indices[int(np.nanargmax(detected_probs))])
            stat["first_scout_step"] = int(step)
            stat["scout_drone"] = int(detected_choice)
            stat["scout_probability"] = float(probability[detected_choice, survivor_idx])
            stat["scout_distance_m"] = float(distance_m[detected_choice, survivor_idx])
            stat["scout_norm_distance"] = float(norm_distance[detected_choice, survivor_idx])
            stat["scout_margin_m"] = float(margin_m[detected_choice, survivor_idx])
            stat["scout_distance_factor"] = float(perception["distance_factor"][detected_choice, survivor_idx])
            stat["scout_environment_factor"] = float(
                perception["environment_factor"][detected_choice, survivor_idx],
            )
            stat["scout_cover_factor"] = float(
                perception["environment_factor"][detected_choice, survivor_idx],
            )
            stat["scout_fire_smoke_factor"] = float(perception["fire_smoke_factor"][detected_choice, survivor_idx])
            stat["scout_altitude_quality"] = float(perception["altitude_quality"][detected_choice, survivor_idx])
            if survivor_idx < len(survivor_confidence_pre):
                stat["scout_confidence_pre"] = float(survivor_confidence_pre[survivor_idx])
            if survivor_idx < len(survivor_confidence_post):
                stat["scout_confidence_post"] = float(survivor_confidence_post[survivor_idx])

    return {
        "unscouted_survivors": float(unscouted_count),
        "exposed_unscouted_survivors": float(exposed_count),
        "exposed_unscouted_fraction": float(exposed_count / max(unscouted_count, 1)),
        "expected_scouts": expected_scouts,
        "expected_scout_recall": float(expected_scouts / max(n_active_survivors, 1)),
        "actual_new_scouts": actual_new_scouts,
        "actual_new_scout_recall": float(actual_new_scouts / max(n_active_survivors, 1)),
        "missed_detection_probability_mass": missed_probability_mass,
        "missed_detection_recall_mass": float(missed_probability_mass / max(n_active_survivors, 1)),
        "mean_exposed_detection_probability": _finite_mean(best_step_probabilities),
        "max_exposed_detection_probability": max(best_step_probabilities) if best_step_probabilities else math.nan,
        "mean_exposed_norm_distance": _finite_mean(best_step_norms),
        "near_edge_exposure_fraction": float(near_edge_exposed / max(exposed_count, 1)),
        "central_exposure_fraction": float(central_exposed / max(exposed_count, 1)),
    }


def _finalize_survivor_exposure_stats(
    stats: list[dict[str, Any]],
    first_scout_steps: list[int | None],
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for stat in stats:
        survivor_idx = int(stat["survivor"])
        first_scout_step = (
            first_scout_steps[survivor_idx]
            if survivor_idx < len(first_scout_steps)
            else stat.get("first_scout_step")
        )
        exposure_steps = int(stat["exposure_steps"])
        cumulative_detection_probability = 1.0 - float(stat["cumulative_no_detection_probability"])
        first_exposure_step = stat.get("first_exposure_step")
        latency = (
            int(first_scout_step) - int(first_exposure_step)
            if first_scout_step is not None and first_exposure_step is not None
            else None
        )
        finalized.append({
            "survivor": survivor_idx,
            "scouted": bool(first_scout_step is not None),
            "first_scout_step": None if first_scout_step is None else int(first_scout_step),
            "first_exposure_step": None if first_exposure_step is None else int(first_exposure_step),
            "last_exposure_step": (
                None if stat.get("last_exposure_step") is None else int(stat["last_exposure_step"])
            ),
            "exposure_to_scout_latency_steps": latency,
            "exposure_steps": exposure_steps,
            "pair_exposures": int(stat["pair_exposures"]),
            "miss_attempt_steps": int(stat["miss_attempt_steps"]),
            "expected_detection_mass": float(stat["expected_detection_mass"]),
            "missed_detection_probability_mass": float(stat["missed_detection_probability_mass"]),
            "cumulative_detection_probability": cumulative_detection_probability,
            "avg_step_detection_probability": _finite_mean(stat["step_detection_probabilities"]),
            "best_detection_probability": float(stat["best_detection_probability"]),
            "best_pair_detection_probability": float(stat["best_pair_detection_probability"]),
            "avg_best_norm_distance": _finite_mean(stat["step_best_norm_distances"]),
            "best_norm_distance": _finite_or_nan(stat["best_norm_distance"]),
            "best_probability_norm_distance": _finite_or_nan(stat["best_probability_norm_distance"]),
            "min_distance_m": _finite_or_nan(stat["min_distance_m"]),
            "best_margin_m": _finite_or_nan(stat["best_margin_m"]),
            "near_edge_exposure_fraction": (
                float(stat["near_edge_exposure_steps"] / exposure_steps) if exposure_steps else math.nan
            ),
            "very_edge_exposure_fraction": (
                float(stat["very_edge_exposure_steps"] / exposure_steps) if exposure_steps else math.nan
            ),
            "central_exposure_fraction": (
                float(stat["central_exposure_steps"] / exposure_steps) if exposure_steps else math.nan
            ),
            "scout_drone": stat.get("scout_drone"),
            "scout_probability": float(stat["scout_probability"]),
            "scout_distance_m": float(stat["scout_distance_m"]),
            "scout_norm_distance": float(stat["scout_norm_distance"]),
            "scout_margin_m": float(stat["scout_margin_m"]),
            "scout_distance_factor": float(stat["scout_distance_factor"]),
            "scout_environment_factor": float(stat["scout_environment_factor"]),
            "scout_cover_factor": float(stat["scout_cover_factor"]),
            "scout_fire_smoke_factor": float(stat["scout_fire_smoke_factor"]),
            "scout_altitude_quality": float(stat["scout_altitude_quality"]),
            "scout_confidence_pre": float(stat["scout_confidence_pre"]),
            "scout_confidence_post": float(stat["scout_confidence_post"]),
            "land_cover": stat.get("land_cover"),
        })
    return finalized


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else math.nan


def _survivor_exposure_summary(exposures: list[dict[str, Any]]) -> dict[str, float]:
    scouted = [entry for entry in exposures if bool(entry.get("scouted"))]
    missed = [entry for entry in exposures if not bool(entry.get("scouted"))]
    exposed = [entry for entry in exposures if int(entry.get("exposure_steps", 0)) > 0]
    never_exposed_missed = [entry for entry in missed if int(entry.get("exposure_steps", 0)) <= 0]
    low_cum_missed = [
        entry for entry in missed
        if float(entry.get("cumulative_detection_probability", 0.0)) < 0.50
    ]
    high_cum_missed = [
        entry for entry in missed
        if float(entry.get("cumulative_detection_probability", 0.0)) >= 0.80
    ]
    edge_limited_missed = [
        entry for entry in missed
        if math.isfinite(float(entry.get("best_norm_distance", math.nan)))
        and float(entry.get("best_norm_distance", math.nan)) >= 0.75
    ]
    return {
        "avg_survivor_exposure_steps": _finite_mean([
            float(entry.get("exposure_steps", math.nan)) for entry in exposures
        ]),
        "avg_scouted_survivor_exposure_steps": _finite_mean([
            float(entry.get("exposure_steps", math.nan)) for entry in scouted
        ]),
        "avg_missed_survivor_exposure_steps": _finite_mean([
            float(entry.get("exposure_steps", math.nan)) for entry in missed
        ]),
        "avg_survivor_pair_exposures": _finite_mean([
            float(entry.get("pair_exposures", math.nan)) for entry in exposures
        ]),
        "avg_survivor_cum_detection_probability": _finite_mean([
            float(entry.get("cumulative_detection_probability", math.nan)) for entry in exposures
        ]),
        "avg_scouted_cum_detection_probability": _finite_mean([
            float(entry.get("cumulative_detection_probability", math.nan)) for entry in scouted
        ]),
        "avg_missed_cum_detection_probability": _finite_mean([
            float(entry.get("cumulative_detection_probability", math.nan)) for entry in missed
        ]),
        "avg_survivor_final_confidence": _finite_mean([
            float(entry.get("final_confidence", math.nan)) for entry in exposures
        ]),
        "avg_scouted_survivor_final_confidence": _finite_mean([
            float(entry.get("final_confidence", math.nan)) for entry in scouted
        ]),
        "avg_missed_survivor_final_confidence": _finite_mean([
            float(entry.get("final_confidence", math.nan)) for entry in missed
        ]),
        "avg_survivor_best_detection_probability": _finite_mean([
            float(entry.get("best_detection_probability", math.nan)) for entry in exposures
        ]),
        "avg_missed_best_detection_probability": _finite_mean([
            float(entry.get("best_detection_probability", math.nan)) for entry in missed
        ]),
        "avg_scout_detection_probability": _finite_mean([
            float(entry.get("scout_probability", math.nan)) for entry in scouted
        ]),
        "avg_scout_detection_norm_distance": _finite_mean([
            float(entry.get("scout_norm_distance", math.nan)) for entry in scouted
        ]),
        "avg_scout_detection_margin_m": _finite_mean([
            float(entry.get("scout_margin_m", math.nan)) for entry in scouted
        ]),
        "avg_scout_confidence_pre": _finite_mean([
            float(entry.get("scout_confidence_pre", math.nan)) for entry in scouted
        ]),
        "avg_scout_confidence_post": _finite_mean([
            float(entry.get("scout_confidence_post", math.nan)) for entry in scouted
        ]),
        "avg_missed_best_norm_distance": _finite_mean([
            float(entry.get("best_norm_distance", math.nan)) for entry in missed
        ]),
        "avg_missed_min_distance_m": _finite_mean([
            float(entry.get("min_distance_m", math.nan)) for entry in missed
        ]),
        "avg_missed_best_margin_m": _finite_mean([
            float(entry.get("best_margin_m", math.nan)) for entry in missed
        ]),
        "avg_exposed_survivor_edge_fraction": _finite_mean([
            float(entry.get("near_edge_exposure_fraction", math.nan)) for entry in exposed
        ]),
        "avg_scout_exposure_to_scout_latency_steps": _finite_mean([
            float(entry.get("exposure_to_scout_latency_steps", math.nan)) for entry in scouted
            if entry.get("exposure_to_scout_latency_steps") is not None
        ]),
        "expected_recall_from_exposure": _finite_mean([
            float(entry.get("cumulative_detection_probability", math.nan)) for entry in exposures
        ]),
        "perception_recall_gap": _finite_mean([
            float(entry.get("cumulative_detection_probability", math.nan)) for entry in exposures
        ]) - _finite_mean([1.0 if bool(entry.get("scouted")) else 0.0 for entry in exposures]),
        "missed_never_exposed_fraction": (
            float(len(never_exposed_missed) / len(missed)) if missed else 0.0
        ),
        "missed_low_cum_probability_fraction": (
            float(len(low_cum_missed) / len(missed)) if missed else 0.0
        ),
        "missed_high_cum_probability_fraction": (
            float(len(high_cum_missed) / len(missed)) if missed else 0.0
        ),
        "missed_edge_limited_fraction": (
            float(len(edge_limited_missed) / len(missed)) if missed else 0.0
        ),
    }


def _coverage_grid_geometry(scenario: WildfireSearchScenario) -> dict[str, Any]:
    G = int(scenario.fire_grid_size)
    cell_width = 2.0 * float(scenario.x_semidim) / max(G, 1)
    cell_height = 2.0 * float(scenario.y_semidim) / max(G, 1)
    return {
        "G": G,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "xs": np.linspace(
            -float(scenario.x_semidim) + cell_width / 2.0,
            float(scenario.x_semidim) - cell_width / 2.0,
            G,
        ),
        "ys": np.linspace(
            -float(scenario.y_semidim) + cell_height / 2.0,
            float(scenario.y_semidim) - cell_height / 2.0,
            G,
        ),
    }


def _footprint_claim_mask(
    position: np.ndarray,
    radius_sim: float,
    geometry: dict[str, Any],
) -> np.ndarray:
    xs = np.asarray(geometry["xs"], dtype=float)
    ys = np.asarray(geometry["ys"], dtype=float)
    cell_width = float(geometry["cell_width"])
    cell_height = float(geometry["cell_height"])
    pos = np.asarray(position, dtype=float).reshape(-1)
    dx = np.maximum(np.abs(xs.reshape(1, -1) - float(pos[0])) - cell_width / 2.0, 0.0)
    dy = np.maximum(np.abs(ys.reshape(-1, 1) - float(pos[1])) - cell_height / 2.0, 0.0)
    radius = max(float(radius_sim), 0.0)
    return dx * dx + dy * dy <= radius * radius


def _footprint_claims(
    positions: np.ndarray,
    radii_sim: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 2)
    radii_arr = np.asarray(radii_sim, dtype=float).reshape(-1)
    G = int(geometry["G"])
    if positions_arr.shape[0] == 0:
        return np.zeros((0, G, G), dtype=bool)
    xs = np.asarray(geometry["xs"], dtype=float)
    ys = np.asarray(geometry["ys"], dtype=float)
    cell_width = float(geometry["cell_width"])
    cell_height = float(geometry["cell_height"])
    if len(radii_arr) == 0:
        radii = np.zeros(positions_arr.shape[0], dtype=float)
    elif len(radii_arr) == positions_arr.shape[0]:
        radii = radii_arr
    else:
        radii = np.resize(radii_arr, positions_arr.shape[0])
    dx = np.maximum(
        np.abs(xs.reshape(1, 1, G) - positions_arr[:, 0].reshape(-1, 1, 1))
        - cell_width / 2.0,
        0.0,
    )
    dy = np.maximum(
        np.abs(ys.reshape(1, G, 1) - positions_arr[:, 1].reshape(-1, 1, 1))
        - cell_height / 2.0,
        0.0,
    )
    radii = np.maximum(radii.reshape(-1, 1, 1), 0.0)
    return dx * dx + dy * dy <= radii * radii


def _empty_revisit_decomposition(n_drones: int) -> dict[str, np.ndarray]:
    n = max(int(n_drones), 0)
    return {
        "any_history": np.zeros(n, dtype=float),
        "own_history": np.zeros(n, dtype=float),
        "teammate_history": np.zeros(n, dtype=float),
        "own_only": np.zeros(n, dtype=float),
        "teammate_only": np.zeros(n, dtype=float),
        "shared_history": np.zeros(n, dtype=float),
    }


def _revisit_decomposition(
    current_claims: np.ndarray,
    individual_history: np.ndarray,
    team_history: np.ndarray,
) -> dict[str, np.ndarray]:
    claims = np.asarray(current_claims, dtype=bool)
    if claims.ndim != 3 or claims.shape[0] == 0:
        return _empty_revisit_decomposition(0)
    history = np.asarray(individual_history, dtype=bool)
    if history.shape != claims.shape:
        history = np.zeros_like(claims, dtype=bool)
    team = np.asarray(team_history, dtype=bool)
    if team.shape != claims.shape[1:]:
        team = history.any(axis=0)

    out = _empty_revisit_decomposition(claims.shape[0])
    for drone_idx, claim in enumerate(claims):
        denom = float(max(int(claim.sum()), 1))
        own = history[drone_idx]
        if claims.shape[0] > 1:
            teammate = np.delete(history, drone_idx, axis=0).any(axis=0)
        else:
            teammate = np.zeros_like(own, dtype=bool)
        shared = own & teammate
        out["any_history"][drone_idx] = float((claim & team).sum() / denom)
        out["own_history"][drone_idx] = float((claim & own).sum() / denom)
        out["teammate_history"][drone_idx] = float((claim & teammate).sum() / denom)
        out["own_only"][drone_idx] = float((claim & own & ~teammate).sum() / denom)
        out["teammate_only"][drone_idx] = float((claim & teammate & ~own).sum() / denom)
        out["shared_history"][drone_idx] = float((claim & shared).sum() / denom)
    return out


def _empty_counterfactual_move_diagnostics(n_drones: int) -> dict[str, np.ndarray]:
    n = max(int(n_drones), 0)
    c = COUNTERFACTUAL_CANDIDATE_DIRECTIONS.shape[0]
    return {
        "directions": COUNTERFACTUAL_CANDIDATE_DIRECTIONS.copy(),
        "new_cells": np.zeros((n, c), dtype=float),
        "overlap": np.zeros((n, c), dtype=float),
        "best_new_cells": np.zeros(n, dtype=float),
        "best_new_overlap": np.full(n, math.nan, dtype=float),
        "best_useful_overlap": np.full(n, math.nan, dtype=float),
        "best_new_direction": np.zeros((n, 2), dtype=float),
        "useful_candidate_count": np.zeros(n, dtype=float),
        "good_candidate_fraction": np.zeros(n, dtype=float),
    }


def _counterfactual_move_diagnostics(
    *,
    scenario: WildfireSearchScenario,
    geometry: dict[str, Any],
    positions: np.ndarray,
    footprint_radii_sim: np.ndarray,
    pre_team_coverage: np.ndarray,
    max_step_sim: float,
) -> dict[str, np.ndarray]:
    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 2)
    n_drones = positions_arr.shape[0]
    if n_drones == 0:
        return _empty_counterfactual_move_diagnostics(0)

    directions = COUNTERFACTUAL_CANDIDATE_DIRECTIONS.copy()
    radii_arr = np.asarray(footprint_radii_sim, dtype=float).reshape(-1)
    if radii_arr.size == 0:
        radii_arr = np.zeros(n_drones, dtype=float)
    elif radii_arr.size != n_drones:
        radii_arr = np.resize(radii_arr, n_drones)

    step = max(float(max_step_sim), 0.0)
    x_min = -float(scenario.x_semidim) + float(scenario.agent_radius)
    x_max = float(scenario.x_semidim) - float(scenario.agent_radius)
    y_min = -float(scenario.y_semidim) + float(scenario.agent_radius)
    y_max = float(scenario.y_semidim) - float(scenario.agent_radius)
    candidate_positions = positions_arr[:, None, :] + directions[None, :, :] * step
    candidate_positions[..., 0] = np.clip(candidate_positions[..., 0], x_min, x_max)
    candidate_positions[..., 1] = np.clip(candidate_positions[..., 1], y_min, y_max)

    flat_positions = candidate_positions.reshape(-1, 2)
    flat_radii = np.repeat(radii_arr, directions.shape[0])
    masks = _footprint_claims(flat_positions, flat_radii, geometry).reshape(
        n_drones,
        directions.shape[0],
        int(geometry["G"]),
        int(geometry["G"]),
    )
    coverage = np.asarray(pre_team_coverage, dtype=bool)
    new_cells = (masks & ~coverage.reshape(1, 1, *coverage.shape)).sum(axis=(2, 3)).astype(float)
    claim_sizes = masks.sum(axis=(2, 3)).astype(float)
    overlap_cells = (masks & coverage.reshape(1, 1, *coverage.shape)).sum(axis=(2, 3)).astype(float)
    overlap = np.divide(
        overlap_cells,
        np.maximum(claim_sizes, 1.0),
        out=np.zeros_like(overlap_cells, dtype=float),
        where=claim_sizes > 0,
    )

    out = _empty_counterfactual_move_diagnostics(n_drones)
    out["new_cells"] = new_cells
    out["overlap"] = overlap
    for drone_idx in range(n_drones):
        candidate_new = new_cells[drone_idx]
        candidate_overlap = overlap[drone_idx]
        best_new = float(np.max(candidate_new)) if candidate_new.size else 0.0
        out["best_new_cells"][drone_idx] = best_new
        if best_new > 0.0:
            # Prefer lower-overlap candidates when several directions uncover the
            # same number of cells, so the "best" direction is also search-efficient.
            best_mask = np.isclose(candidate_new, best_new, rtol=0.0, atol=1e-9)
            best_candidates = np.flatnonzero(best_mask)
            best_idx = int(best_candidates[np.argmin(candidate_overlap[best_candidates])])
            out["best_new_overlap"][drone_idx] = float(candidate_overlap[best_idx])
            out["best_new_direction"][drone_idx] = directions[best_idx]
            useful_threshold = max(1.0, 0.5 * best_new)
            useful_mask = candidate_new >= useful_threshold
            out["useful_candidate_count"][drone_idx] = float(np.count_nonzero(useful_mask))
            out["good_candidate_fraction"][drone_idx] = float(
                np.count_nonzero(useful_mask) / max(len(candidate_new), 1)
            )
            out["best_useful_overlap"][drone_idx] = float(np.min(candidate_overlap[useful_mask]))
        else:
            stay_idx = directions.shape[0] - 1
            out["best_new_overlap"][drone_idx] = float(candidate_overlap[stay_idx])
            out["best_useful_overlap"][drone_idx] = math.nan
            out["best_new_direction"][drone_idx] = np.zeros(2, dtype=float)
    return out


def _counterfactual_choice_metrics(
    vector: np.ndarray,
    candidate_directions: np.ndarray,
    candidate_new_cells: np.ndarray,
    best_direction: np.ndarray,
) -> dict[str, float]:
    directions = np.asarray(candidate_directions, dtype=float).reshape(-1, 2)
    new_cells = np.asarray(candidate_new_cells, dtype=float).reshape(-1)
    if directions.shape[0] == 0 or new_cells.shape[0] == 0:
        return {
            "new_cells": math.nan,
            "rank": math.nan,
            "capture_fraction": math.nan,
            "best_alignment": math.nan,
        }

    vec = np.asarray(vector, dtype=float).reshape(-1)[:2]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm <= 1e-9:
        chosen_idx = directions.shape[0] - 1
        best_alignment = math.nan
    else:
        move_dirs = directions[:-1]
        cosines = move_dirs @ (vec / vec_norm)
        chosen_idx = int(np.argmax(cosines))
        best_alignment = _cosine_or_nan(vec, np.asarray(best_direction, dtype=float))

    chosen_new = float(new_cells[chosen_idx])
    best_new = float(np.max(new_cells)) if new_cells.size else 0.0
    rank = 1.0 + float(np.count_nonzero(new_cells > chosen_new + 1e-9))
    capture = chosen_new / best_new if best_new > 1e-9 else math.nan
    return {
        "new_cells": chosen_new,
        "rank": rank,
        "capture_fraction": capture,
        "best_alignment": best_alignment,
    }


def _frontier_expected_new_cells(
    *,
    scenario: WildfireSearchScenario,
    geometry: dict[str, Any],
    positions: np.ndarray,
    footprint_radii_sim: np.ndarray,
    pre_team_coverage: np.ndarray,
    frontier_obs: np.ndarray,
    max_step_sim: float,
) -> np.ndarray:
    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 2)
    radii_arr = np.asarray(footprint_radii_sim, dtype=float).reshape(-1)
    expected = np.zeros(positions_arr.shape[0], dtype=float)
    if positions_arr.shape[0] == 0:
        return expected
    coverage = np.asarray(pre_team_coverage, dtype=bool)
    candidate_positions: list[np.ndarray] = []
    candidate_radii: list[float] = []
    candidate_drones: list[int] = []
    for drone_idx, pos in enumerate(positions_arr):
        candidates = _valid_frontier_candidates(
            _frontier_candidates_for_drone(scenario, frontier_obs, drone_idx)
        )
        for row in candidates:
            direction = np.asarray(row[:2], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            candidate_positions.append(pos + direction / norm * max(float(max_step_sim), 0.0))
            candidate_radii.append(
                float(radii_arr[min(drone_idx, len(radii_arr) - 1)]) if len(radii_arr) else 0.0
            )
            candidate_drones.append(drone_idx)
    if not candidate_positions:
        return expected
    masks = _footprint_claims(
        np.asarray(candidate_positions, dtype=float),
        np.asarray(candidate_radii, dtype=float),
        geometry,
    )
    values = (masks & ~coverage.reshape(1, *coverage.shape)).sum(axis=(1, 2)).astype(float)
    for drone_idx, value in zip(candidate_drones, values):
        expected[int(drone_idx)] = max(float(expected[int(drone_idx)]), float(value))
    return expected


def _empty_frontier_usefulness_diagnostics(n_drones: int) -> dict[str, np.ndarray]:
    n = max(int(n_drones), 0)
    return {
        "new_cells": np.full(n, math.nan, dtype=float),
        "capture_fraction": np.full(n, math.nan, dtype=float),
        "regret": np.full(n, math.nan, dtype=float),
        "best_alignment": np.full(n, math.nan, dtype=float),
        "rank": np.full(n, math.nan, dtype=float),
        "nearest_rank": np.full(n, math.nan, dtype=float),
        "nearest_capture_fraction": np.full(n, math.nan, dtype=float),
        "is_best": np.zeros(n, dtype=float),
        "bad": np.zeros(n, dtype=float),
        "valid": np.zeros(n, dtype=float),
    }


def _frontier_usefulness_for_drone(metrics: dict[str, np.ndarray], drone_idx: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        out[key] = float(arr[drone_idx]) if drone_idx < arr.size else math.nan
    return out


def _frontier_usefulness_diagnostics(
    *,
    scenario: WildfireSearchScenario,
    geometry: dict[str, Any],
    positions: np.ndarray,
    footprint_radii_sim: np.ndarray,
    pre_team_coverage: np.ndarray,
    max_step_sim: float,
    frontier_obs: np.ndarray,
    counterfactual: dict[str, np.ndarray],
    frontier_mode: str | None = None,
    frontier_top_k: int | None = None,
) -> dict[str, np.ndarray]:
    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 2)
    n_drones = positions_arr.shape[0]
    out = _empty_frontier_usefulness_diagnostics(n_drones)
    if n_drones == 0:
        return out

    radii_arr = np.asarray(footprint_radii_sim, dtype=float).reshape(-1)
    if radii_arr.size == 0:
        radii_arr = np.zeros(n_drones, dtype=float)
    elif radii_arr.size != n_drones:
        radii_arr = np.resize(radii_arr, n_drones)

    coverage = np.asarray(pre_team_coverage, dtype=bool)
    step = max(float(max_step_sim), 0.0)
    x_min = -float(scenario.x_semidim) + float(scenario.agent_radius)
    x_max = float(scenario.x_semidim) - float(scenario.agent_radius)
    y_min = -float(scenario.y_semidim) + float(scenario.agent_radius)
    y_max = float(scenario.y_semidim) - float(scenario.agent_radius)
    counter_new = np.asarray(counterfactual.get("new_cells", np.zeros((n_drones, 0))), dtype=float)
    counter_dirs = np.asarray(counterfactual.get("directions", np.zeros((0, 2))), dtype=float)
    best_new_cells = np.asarray(counterfactual.get("best_new_cells", np.zeros(n_drones)), dtype=float)
    best_dirs = np.asarray(counterfactual.get("best_new_direction", np.zeros((n_drones, 2))), dtype=float)

    candidate_positions: list[np.ndarray] = []
    candidate_radii: list[float] = []
    candidate_drone_indices: list[int] = []
    candidate_directions: list[np.ndarray] = []
    for drone_idx, pos in enumerate(positions_arr):
        candidates = _valid_frontier_candidates(
            _frontier_candidates_for_drone(
                scenario,
                frontier_obs,
                drone_idx,
                mode=frontier_mode,
                top_k=frontier_top_k,
            )
        )
        for candidate in candidates:
            direction = np.asarray(candidate[:2], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            unit = direction / norm
            next_pos = pos + unit * step
            next_pos = next_pos.copy()
            next_pos[0] = np.clip(next_pos[0], x_min, x_max)
            next_pos[1] = np.clip(next_pos[1], y_min, y_max)
            candidate_positions.append(next_pos)
            candidate_radii.append(float(radii_arr[min(drone_idx, len(radii_arr) - 1)]))
            candidate_drone_indices.append(drone_idx)
            candidate_directions.append(unit)

    if not candidate_positions:
        return out

    masks = _footprint_claims(
        np.asarray(candidate_positions, dtype=float),
        np.asarray(candidate_radii, dtype=float),
        geometry,
    )
    new_values = (masks & ~coverage.reshape(1, *coverage.shape)).sum(axis=(1, 2)).astype(float)

    by_drone: dict[int, list[tuple[float, np.ndarray]]] = {}
    for drone_idx, new_value, direction in zip(candidate_drone_indices, new_values, candidate_directions):
        by_drone.setdefault(int(drone_idx), []).append((float(new_value), np.asarray(direction, dtype=float)))

    for drone_idx in range(n_drones):
        entries = by_drone.get(drone_idx, [])
        if not entries:
            continue
        values = np.asarray([entry[0] for entry in entries], dtype=float)
        directions = np.asarray([entry[1] for entry in entries], dtype=float)
        best_frontier_idx = int(np.argmax(values))
        frontier_new = float(values[best_frontier_idx])
        frontier_direction = directions[best_frontier_idx]
        best_new = float(best_new_cells[drone_idx]) if drone_idx < best_new_cells.size else 0.0
        capture = frontier_new / best_new if best_new > 1e-9 else math.nan
        regret = max(best_new - frontier_new, 0.0) if best_new > 1e-9 else math.nan
        counter_row = counter_new[drone_idx] if drone_idx < counter_new.shape[0] else np.zeros(0)
        finite_counter = counter_row[np.isfinite(counter_row)]
        rank = (
            1.0 + float(np.count_nonzero(finite_counter > frontier_new + 1e-9))
            if finite_counter.size else math.nan
        )

        nearest_rank = math.nan
        nearest_capture = math.nan
        move_dirs = counter_dirs[:-1] if counter_dirs.shape[0] > 1 else counter_dirs
        if move_dirs.size and counter_row.size:
            cosines = move_dirs @ frontier_direction
            nearest_idx = int(np.argmax(cosines))
            nearest_new = float(counter_row[nearest_idx])
            nearest_rank = 1.0 + float(np.count_nonzero(counter_row > nearest_new + 1e-9))
            nearest_capture = nearest_new / best_new if best_new > 1e-9 else math.nan

        alignment = _cosine_or_nan(
            frontier_direction,
            best_dirs[drone_idx] if drone_idx < len(best_dirs) else np.zeros(2),
        )
        out["new_cells"][drone_idx] = frontier_new
        out["capture_fraction"][drone_idx] = capture
        out["regret"][drone_idx] = regret
        out["best_alignment"][drone_idx] = alignment
        out["rank"][drone_idx] = rank
        out["nearest_rank"][drone_idx] = nearest_rank
        out["nearest_capture_fraction"][drone_idx] = nearest_capture
        out["is_best"][drone_idx] = float(math.isfinite(capture) and capture >= 0.90)
        out["bad"][drone_idx] = float(math.isfinite(capture) and capture < 0.25)
        out["valid"][drone_idx] = 1.0
    return out


def _normalize_frontier_source(source: Any) -> str:
    return str(source).replace("-", "_").lower()


def _normalize_frontier_mode(mode: Any) -> str:
    return str(mode).replace("-", "_")


def _frontier_config_for(
    scenario: WildfireSearchScenario,
    *,
    source: Any,
    mode: Any,
    radius_m: float | None = None,
) -> dict[str, Any]:
    return {
        "source": _normalize_frontier_source(source),
        "mode": _normalize_frontier_mode(mode),
        "radius_m": (
            float(getattr(scenario, "uav_frontier_obs_radius_m", 0.0))
            if radius_m is None else float(radius_m)
        ),
    }


def _current_frontier_config(scenario: WildfireSearchScenario) -> dict[str, Any]:
    return _frontier_config_for(
        scenario,
        source=getattr(scenario, "uav_frontier_source", "coverage"),
        mode=getattr(scenario, "uav_frontier_mode", "centroid"),
        radius_m=getattr(scenario, "uav_frontier_obs_radius_m", 0.0),
    )


def _frontier_configs_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left.get("source")) == str(right.get("source"))
        and str(left.get("mode")) == str(right.get("mode"))
        and math.isclose(
            float(left.get("radius_m", 0.0)),
            float(right.get("radius_m", 0.0)),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )


def _frontier_cache_name(config: dict[str, Any]) -> str:
    radius = float(config.get("radius_m", 0.0))
    radius_label = f"{radius:.9g}".replace("-", "m").replace(".", "p")
    return f"diagnostic_{config.get('source')}_{config.get('mode')}_{radius_label}"


def _shadow_frontier_features(
    scenario: WildfireSearchScenario,
    positions_tensor: torch.Tensor,
    *,
    source: str,
    mode: str,
    radius_m: float | None = None,
) -> np.ndarray:
    old_source = getattr(scenario, "uav_frontier_source", "coverage")
    old_mode = getattr(scenario, "uav_frontier_mode", "centroid")
    old_radius = getattr(scenario, "uav_frontier_obs_radius_m", None)
    config = _frontier_config_for(
        scenario,
        source=source,
        mode=mode,
        radius_m=radius_m,
    )
    try:
        scenario.uav_frontier_source = config["source"]
        scenario.uav_frontier_mode = config["mode"]
        scenario.uav_frontier_obs_radius_m = float(config["radius_m"])
        return (
            scenario._cached_uav_frontier_features_for_positions(
                _frontier_cache_name(config),
                positions_tensor,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
    finally:
        scenario.uav_frontier_source = old_source
        scenario.uav_frontier_mode = old_mode
        if old_radius is not None:
            scenario.uav_frontier_obs_radius_m = old_radius


def _empty_coverage_signal_snapshot(n_drones: int) -> dict[str, np.ndarray]:
    n = max(int(n_drones), 0)
    return {
        "local_vec": np.zeros((n, 2), dtype=float),
        "global_vec": np.zeros((n, 2), dtype=float),
        "sector_vec": np.zeros((n, 2), dtype=float),
        "sector_dominance": np.full(n, math.nan, dtype=float),
        "sector_entropy": np.full(n, math.nan, dtype=float),
        "frontier_cancellation": np.full(n, math.nan, dtype=float),
    }


def _coverage_signal_snapshot(
    scenario: WildfireSearchScenario,
    positions_tensor: torch.Tensor,
    frontier_obs: np.ndarray,
) -> dict[str, np.ndarray]:
    positions = positions_tensor[0].detach().cpu().numpy().astype(float)
    n_drones = positions.shape[0]
    out = _empty_coverage_signal_snapshot(n_drones)
    if n_drones == 0:
        return out

    if int(getattr(scenario, "local_coverage_obs_grid", 0)) > 0:
        K_local = int(scenario.local_coverage_obs_grid)
        for drone_idx, agent in enumerate(scenario.world.agents[:n_drones]):
            patch = (
                scenario._local_coverage_observation(agent)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .reshape(K_local, K_local)
            )
            out["local_vec"][drone_idx] = _pooled_uncovered_patch_vector(patch)

    if int(getattr(scenario, "coverage_obs_grid", 0)) > 0:
        K_global = int(scenario.coverage_obs_grid)
        global_obs = scenario._coverage_observation()[0].detach().cpu().numpy().astype(float)
        pooled = global_obs[: K_global * K_global].reshape(K_global, K_global)
        out["global_vec"] = _global_pooled_uncovered_vectors(pooled, positions, scenario)

    sector = _frontier_sector_snapshot(scenario, positions, frontier_obs)
    out.update(sector)
    return out


def _pooled_uncovered_patch_vector(covered_patch: np.ndarray) -> np.ndarray:
    K = int(covered_patch.shape[0])
    if K <= 0:
        return np.zeros(2, dtype=float)
    weights = (1.0 - np.asarray(covered_patch, dtype=float)).clip(0.0, 1.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.zeros(2, dtype=float)
    centers = np.linspace(-1.0 + 1.0 / K, 1.0 - 1.0 / K, K)
    xx, yy = np.meshgrid(centers, centers)
    return np.asarray([
        float((weights * xx).sum() / total),
        float((weights * yy).sum() / total),
    ])


def _global_pooled_uncovered_vectors(
    covered_map: np.ndarray,
    positions: np.ndarray,
    scenario: WildfireSearchScenario,
) -> np.ndarray:
    K = int(covered_map.shape[0])
    weights = (1.0 - np.asarray(covered_map, dtype=float)).clip(0.0, 1.0)
    total = float(weights.sum())
    out = np.zeros((positions.shape[0], 2), dtype=float)
    if K <= 0 or total <= 1e-12:
        return out
    x_centers = np.linspace(
        -float(scenario.x_semidim) + float(scenario.x_semidim) / K,
        float(scenario.x_semidim) - float(scenario.x_semidim) / K,
        K,
    )
    y_centers = np.linspace(
        -float(scenario.y_semidim) + float(scenario.y_semidim) / K,
        float(scenario.y_semidim) - float(scenario.y_semidim) / K,
        K,
    )
    xx, yy = np.meshgrid(x_centers, y_centers)
    for idx, pos in enumerate(positions):
        dx = xx - float(pos[0])
        dy = yy - float(pos[1])
        out[idx] = np.asarray([
            float((weights * dx).sum() / total),
            float((weights * dy).sum() / total),
        ])
    return out


def _frontier_sector_snapshot(
    scenario: WildfireSearchScenario,
    positions: np.ndarray,
    frontier_obs: np.ndarray,
    n_sectors: int = 8,
) -> dict[str, np.ndarray]:
    n_drones = positions.shape[0]
    sector_vec = np.zeros((n_drones, 2), dtype=float)
    dominance = np.full(n_drones, math.nan, dtype=float)
    entropy = np.full(n_drones, math.nan, dtype=float)
    cancellation = np.full(n_drones, math.nan, dtype=float)
    if n_drones == 0:
        return {
            "sector_vec": sector_vec,
            "sector_dominance": dominance,
            "sector_entropy": entropy,
            "frontier_cancellation": cancellation,
        }

    coverage = scenario.coverage_grid[0].detach().cpu().numpy().astype(bool)
    uncovered = ~coverage
    G = int(scenario.fire_grid_size)
    cell_width = 2.0 * float(scenario.x_semidim) / G
    cell_height = 2.0 * float(scenario.y_semidim) / G
    xs = np.linspace(
        -float(scenario.x_semidim) + cell_width / 2.0,
        float(scenario.x_semidim) - cell_width / 2.0,
        G,
    )
    ys = np.linspace(
        -float(scenario.y_semidim) + cell_height / 2.0,
        float(scenario.y_semidim) - cell_height / 2.0,
        G,
    )
    xx, yy = np.meshgrid(xs, ys)
    sim_units_per_meter = max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)
    radius = max(float(getattr(scenario, "uav_frontier_obs_radius_m", 150.0)) * sim_units_per_meter, 1e-9)
    sector_width = 2.0 * math.pi / max(n_sectors, 1)
    for idx, pos in enumerate(positions):
        dx = xx - float(pos[0])
        dy = yy - float(pos[1])
        dist = np.sqrt(dx * dx + dy * dy)
        useful = (dist <= radius) & uncovered
        count = int(useful.sum())
        if count <= 0:
            continue
        angles = (np.arctan2(dy[useful], dx[useful]) + 2.0 * math.pi) % (2.0 * math.pi)
        sector_idx = np.floor(angles / sector_width).astype(int).clip(0, n_sectors - 1)
        counts = np.bincount(sector_idx, minlength=n_sectors).astype(float)
        probs = counts / max(float(counts.sum()), 1e-12)
        best = int(np.argmax(counts))
        angle = (best + 0.5) * sector_width
        sector_vec[idx] = np.asarray([math.cos(angle), math.sin(angle)])
        dominance[idx] = float(probs[best])
        nonzero = probs[probs > 0.0]
        entropy[idx] = float(-(nonzero * np.log(nonzero)).sum() / math.log(n_sectors))
        mean_dist_norm = float(dist[useful].mean() / radius)
        frontier_norm = float(np.linalg.norm(frontier_obs[idx, :2])) if idx < len(frontier_obs) else 0.0
        cancellation[idx] = (
            float(frontier_norm / mean_dist_norm)
            if mean_dist_norm > 1e-12
            else math.nan
        )
    return {
        "sector_vec": sector_vec,
        "sector_dominance": dominance,
        "sector_entropy": entropy,
        "frontier_cancellation": cancellation,
    }


def _pairwise_direction_metrics(vectors: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return {"mean_cos": math.nan, "same_dir_frac": math.nan}
    cosines = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            cosine = _cosine_or_nan(arr[i], arr[j])
            if math.isfinite(cosine):
                cosines.append(cosine)
    if not cosines:
        return {"mean_cos": math.nan, "same_dir_frac": math.nan}
    return {
        "mean_cos": float(np.mean(cosines)),
        "same_dir_frac": float(np.mean([value >= 0.75 for value in cosines])),
    }


def _uav_reward_terms(
    *,
    scenario: WildfireSearchScenario,
    new_cells: float,
    displacement_m: float,
    frontier_progress: float,
    frontier_ratio: float,
    overlap: float,
    expected_overlap: float,
    inter_uav_overlap: float,
    outside_footprint: float,
    coverage_opportunity_fraction: float,
    coverage_opportunity_available_fraction: float,
    confidence_reward: float,
    team_confidence_reward: float,
    team_confidence_overlap_penalty: float,
    confidence_move_reward: float,
    confidence_opportunity_fraction: float,
    confidence_overlap_penalty: float,
    cleanup_target_progress_reward: float,
    astar_progress_reward: float,
    scout_reward: float,
) -> dict[str, float]:
    grid_cells = float(max(int(getattr(scenario, "fire_grid_size", 1)) ** 2, 1))
    coverage_scale = float(getattr(scenario, "r_coverage", 0.0))
    coverage_normalization = str(
        getattr(scenario, "uav_coverage_normalization", "map")
    ).replace("-", "_").lower()
    opportunity_cap = max(float(getattr(scenario, "uav_coverage_opportunity_cap", 1.0)), 0.0)
    if coverage_normalization == "opportunity":
        coverage_ratio = min(max(coverage_opportunity_fraction, 0.0), opportunity_cap)
    else:
        coverage_ratio = new_cells / grid_cells
    coverage = coverage_scale * coverage_ratio

    move_scale = float(getattr(scenario, "r_uav_move_coverage", 0.0))
    move_cap = max(float(getattr(scenario, "r_uav_move_coverage_cap", 0.0)), 0.0)
    move_normalization = str(
        getattr(scenario, "uav_move_coverage_normalization", "raw")
    ).replace("-", "_").lower()
    if move_normalization == "opportunity":
        max_step_m = max(
            float(getattr(scenario, "drone_speed_mps", 0.0))
            * float(getattr(scenario, "sim_step_seconds", 1.0)),
            1e-6,
        )
        distance_fraction = min(max(displacement_m / max_step_m, 0.0), 1.0)
        move_base = distance_fraction * min(max(coverage_opportunity_fraction, 0.0), 1.0)
    else:
        move_base = max(displacement_m, 0.0) * max(new_cells, 0.0)
    move_coverage = move_base * move_scale
    if move_cap > 0.0:
        move_coverage = min(move_coverage, move_cap)

    frontier = (
        float(getattr(scenario, "r_uav_frontier_alignment", 0.0))
        * min(max(frontier_progress, 0.0), 1.0)
        * min(max(frontier_ratio, 0.0), 1.0)
    )
    inefficient_scale = float(getattr(scenario, "r_uav_inefficient_move", 0.0))
    inefficient_source = str(
        getattr(scenario, "uav_inefficient_move_source", "confidence")
    ).replace("-", "_").lower()
    if inefficient_source == "coverage":
        inefficient_opportunity = coverage_opportunity_fraction
    else:
        inefficient_opportunity = confidence_opportunity_fraction
    max_step_m = max(
        float(getattr(scenario, "drone_speed_mps", 0.0))
        * float(getattr(scenario, "sim_step_seconds", 1.0)),
        1e-6,
    )
    movement_fraction = min(max(displacement_m / max_step_m, 0.0), 1.0)
    inefficient_move_penalty = -inefficient_scale * movement_fraction * (
        1.0 - min(max(inefficient_opportunity, 0.0), 1.0)
    )

    overlap_penalty = _overlap_penalty_value(
        overlap=overlap,
        expected_overlap=expected_overlap,
        scale=float(getattr(scenario, "r_uav_overlap", 0.0)),
        allowed=float(getattr(scenario, "uav_overlap_allowed", 0.10)),
        opportunity_available_fraction=coverage_opportunity_available_fraction,
        normalization=str(getattr(scenario, "uav_overlap_penalty_normalization", "raw")),
    )
    inter_penalty = _fraction_penalty_value(
        value=inter_uav_overlap,
        scale=float(getattr(scenario, "r_uav_inter_uav_overlap", 0.0)),
        allowed=float(getattr(scenario, "uav_inter_uav_overlap_allowed", 0.20)),
    )
    outside_penalty = -float(getattr(scenario, "r_uav_outside_footprint", 0.0)) * min(
        max(outside_footprint, 0.0),
        1.0,
    )

    aux = (
        coverage
        + move_coverage
        + frontier
        + confidence_reward
        + team_confidence_reward
        + team_confidence_overlap_penalty
        + confidence_move_reward
        + cleanup_target_progress_reward
        + astar_progress_reward
        + inefficient_move_penalty
        + confidence_overlap_penalty
        + overlap_penalty
        + inter_penalty
        + outside_penalty
    )
    abs_denom = (
        abs(coverage)
        + abs(move_coverage)
        + abs(frontier)
        + abs(confidence_reward)
        + abs(team_confidence_reward)
        + abs(team_confidence_overlap_penalty)
        + abs(confidence_move_reward)
        + abs(cleanup_target_progress_reward)
        + abs(astar_progress_reward)
        + abs(inefficient_move_penalty)
        + abs(confidence_overlap_penalty)
        + abs(overlap_penalty)
        + abs(inter_penalty)
        + abs(outside_penalty)
        + abs(scout_reward)
    )
    return {
        "coverage": float(coverage),
        "move_coverage": float(move_coverage),
        "frontier": float(frontier),
        "confidence": float(confidence_reward),
        "team_confidence": float(team_confidence_reward),
        "team_confidence_overlap_penalty": float(team_confidence_overlap_penalty),
        "confidence_move": float(confidence_move_reward),
        "cleanup_target_progress": float(cleanup_target_progress_reward),
        "astar_progress": float(astar_progress_reward),
        "inefficient_move_penalty": float(inefficient_move_penalty),
        "confidence_overlap_penalty": float(confidence_overlap_penalty),
        "overlap_penalty": float(overlap_penalty),
        "inter_uav_overlap_penalty": float(inter_penalty),
        "outside_footprint_penalty": float(outside_penalty),
        "scout": float(scout_reward),
        "aux": float(aux),
        "frontier_abs_share": float(abs(frontier) / abs_denom) if abs_denom > 1e-12 else 0.0,
    }


def _overlap_penalty_value(
    *,
    overlap: float,
    expected_overlap: float,
    scale: float,
    allowed: float,
    opportunity_available_fraction: float = 1.0,
    normalization: str = "raw",
) -> float:
    if scale <= 0.0:
        return 0.0
    slack = min(max(allowed, 0.0), 0.999)
    threshold = min(max(expected_overlap, 0.0) + slack, 0.999)
    excess = max(overlap - threshold, 0.0)
    normalized = min(excess / max(1.0 - threshold, 1e-6), 1.0)
    if str(normalization).replace("-", "_").lower() == "opportunity":
        normalized *= min(max(opportunity_available_fraction, 0.0), 1.0)
    return float(-scale * normalized)


def _fraction_penalty_value(*, value: float, scale: float, allowed: float) -> float:
    if scale <= 0.0:
        return 0.0
    slack = min(max(allowed, 0.0), 0.999)
    excess = max(value - slack, 0.0)
    normalized = min(excess / max(1.0 - slack, 1e-6), 1.0)
    return float(-scale * normalized)


def _safe_corr(xs: list[float], ys: list[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return math.nan
    x_arr = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_arr = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _cosine_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if norm <= 1e-12:
        return math.nan
    return float(np.clip(np.dot(a_arr, b_arr) / norm, -1.0, 1.0))


def _project_onto_unit(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(b_arr))
    if norm <= 1e-12:
        return math.nan
    return float(np.dot(a_arr, b_arr / norm))


def _frontier_candidates_for_drone(
    scenario: WildfireSearchScenario,
    frontier_obs: np.ndarray,
    drone_idx: int,
    *,
    mode: str | None = None,
    top_k: int | None = None,
) -> np.ndarray:
    if drone_idx >= len(frontier_obs):
        return np.zeros((0, 4), dtype=float)
    row = np.asarray(frontier_obs[drone_idx], dtype=float).reshape(-1)
    mode = (
        str(getattr(scenario, "uav_frontier_mode", "centroid"))
        if mode is None
        else str(mode)
    ).replace("-", "_")
    if mode == "local_global":
        usable = min(8, len(row))
        if usable < 4:
            return np.zeros((0, 4), dtype=float)
        return row[:usable].reshape(-1, 4)
    if mode != "sector_topk":
        return row[:4].reshape(1, 4)
    top_k = max(
        int(getattr(scenario, "uav_frontier_top_k", max(len(row) // 4, 1)) if top_k is None else top_k),
        1,
    )
    usable = min(top_k * 4, len(row))
    if usable < 4:
        return np.zeros((0, 4), dtype=float)
    return row[:usable].reshape(-1, 4)


def _valid_frontier_candidates(candidates: np.ndarray) -> list[np.ndarray]:
    rows = np.asarray(candidates, dtype=float).reshape(-1, 4)
    return [
        row
        for row in rows
        if float(np.linalg.norm(row[:2])) > 1e-12 and float(row[3]) > 1e-12
    ]


def _best_frontier_candidate_cosine(vector: np.ndarray, candidates: np.ndarray) -> float:
    values = [
        _cosine_or_nan(vector, row[:2])
        for row in _valid_frontier_candidates(candidates)
    ]
    values = [value for value in values if math.isfinite(value)]
    return float(max(values)) if values else math.nan


def _best_frontier_candidate_projection(vector: np.ndarray, candidates: np.ndarray) -> float:
    values = [
        _project_onto_unit(vector, row[:2])
        for row in _valid_frontier_candidates(candidates)
    ]
    values = [value for value in values if math.isfinite(value)]
    return float(max(values)) if values else math.nan


def _new_time_bins(count: int) -> list[dict[str, Any]]:
    return [
        {
            "bin": int(idx),
            "start_fraction": idx / max(count, 1),
            "end_fraction": (idx + 1) / max(count, 1),
            "values": {},
        }
        for idx in range(max(int(count), 1))
    ]


def _append_time_bin(
    bins: list[dict[str, Any]],
    *,
    step: int,
    max_steps: int,
    values: dict[str, float],
) -> None:
    if not bins:
        return
    bin_idx = min(int(step * len(bins) / max(max_steps, 1)), len(bins) - 1)
    bucket = bins[bin_idx]["values"]
    for key, value in values.items():
        bucket.setdefault(key, []).append(float(value))


def _finalize_time_bins(bins: list[dict[str, Any]]) -> list[dict[str, float]]:
    finalized = []
    for item in bins:
        values = item["values"]
        row: dict[str, float] = {
            "bin": float(item["bin"]),
            "start_fraction": float(item["start_fraction"]),
            "end_fraction": float(item["end_fraction"]),
            "count": float(max((len(v) for v in values.values()), default=0)),
        }
        for key, series in values.items():
            row[key] = _finite_mean(series)
        finalized.append(row)
    return finalized


def _summarize_time_bins(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    max_bins = max((len(row.get("time_bins", [])) for row in rows), default=0)
    summary = []
    for bin_idx in range(max_bins):
        entries = [
            row["time_bins"][bin_idx]
            for row in rows
            if len(row.get("time_bins", [])) > bin_idx
        ]
        if not entries:
            continue
        keys = sorted({
            key
            for entry in entries
            for key in entry.keys()
            if key not in {"bin", "start_fraction", "end_fraction"}
        })
        row_summary: dict[str, float] = {
            "bin": float(bin_idx),
            "start_fraction": _finite_mean([
                float(entry.get("start_fraction", math.nan)) for entry in entries
            ]),
            "end_fraction": _finite_mean([
                float(entry.get("end_fraction", math.nan)) for entry in entries
            ]),
        }
        for key in keys:
            row_summary[key] = _finite_mean([
                float(entry.get(key, math.nan)) for entry in entries
            ])
        summary.append(row_summary)
    return summary


def _summarize_named_time_bins(rows: list[dict[str, Any]], row_key: str) -> list[dict[str, float]]:
    max_bins = max((len(row.get(row_key, [])) for row in rows), default=0)
    summary = []
    for bin_idx in range(max_bins):
        entries = [
            row[row_key][bin_idx]
            for row in rows
            if len(row.get(row_key, [])) > bin_idx
        ]
        if not entries:
            continue
        keys = sorted({
            key
            for entry in entries
            for key in entry.keys()
            if key not in {"bin", "start_fraction", "end_fraction"}
        })
        row_summary: dict[str, float] = {
            "bin": float(bin_idx),
            "start_fraction": _finite_mean([
                float(entry.get("start_fraction", math.nan)) for entry in entries
            ]),
            "end_fraction": _finite_mean([
                float(entry.get("end_fraction", math.nan)) for entry in entries
            ]),
        }
        for key in keys:
            row_summary[key] = _finite_mean([
                float(entry.get(key, math.nan)) for entry in entries
            ])
        summary.append(row_summary)
    return summary


def _scout_time_bin_series(row: dict[str, Any], bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    n_survivors = max(int(row.get("survivors", 0)), 1)
    max_steps = int(row.get("max_steps", 0))
    steps = [
        int(value)
        for value in row.get("first_scout_steps", [])
        if value is not None and int(value) > 0
    ]
    if max_steps <= 0:
        max_steps = max(steps, default=1)

    new_recall = np.zeros(max(int(bin_count), 1), dtype=float)
    for step in steps:
        bin_idx = min(
            int((max(step, 1) - 1) * len(new_recall) / max(max_steps, 1)),
            len(new_recall) - 1,
        )
        new_recall[bin_idx] += 1.0 / n_survivors
    return new_recall, np.cumsum(new_recall)


def _summarize_scout_time_bins(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    bin_count = max((len(row.get("time_bins", [])) for row in rows), default=TIME_BIN_COUNT)
    if bin_count <= 0:
        return []

    new_rows = []
    cumulative_rows = []
    for row in rows:
        new_recall, cumulative_recall = _scout_time_bin_series(row, bin_count)
        new_rows.append(new_recall)
        cumulative_rows.append(cumulative_recall)

    if not new_rows:
        return []

    new_matrix = np.vstack(new_rows)
    cumulative_matrix = np.vstack(cumulative_rows)
    summary = []
    for bin_idx in range(bin_count):
        new_values = new_matrix[:, bin_idx].astype(float).tolist()
        cumulative_values = cumulative_matrix[:, bin_idx].astype(float).tolist()
        summary.append({
            "bin": float(bin_idx),
            "start_fraction": float(bin_idx / bin_count),
            "end_fraction": float((bin_idx + 1) / bin_count),
            "mean_new_recall": _finite_mean(new_values),
            "median_new_recall": _finite_median(new_values),
            "mean_cumulative_recall": _finite_mean(cumulative_values),
            "median_cumulative_recall": _finite_median(cumulative_values),
        })
    return summary


def _summarize_survivor_exposure_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    groups = [
        (
            "scouted",
            [
                entry
                for row in rows
                for entry in row.get("survivor_exposures", [])
                if bool(entry.get("scouted"))
            ],
        ),
        (
            "missed",
            [
                entry
                for row in rows
                for entry in row.get("survivor_exposures", [])
                if not bool(entry.get("scouted"))
            ],
        ),
    ]
    summaries: list[dict[str, float | str]] = []
    for label, entries in groups:
        summaries.append({
            "group": label,
            "survivors": float(len(entries)),
            "mean_exposure_steps": _finite_mean([
                float(entry.get("exposure_steps", math.nan)) for entry in entries
            ]),
            "mean_pair_exposures": _finite_mean([
                float(entry.get("pair_exposures", math.nan)) for entry in entries
            ]),
            "mean_cumulative_detection_probability": _finite_mean([
                float(entry.get("cumulative_detection_probability", math.nan)) for entry in entries
            ]),
            "mean_final_confidence": _finite_mean([
                float(entry.get("final_confidence", math.nan)) for entry in entries
            ]),
            "mean_best_detection_probability": _finite_mean([
                float(entry.get("best_detection_probability", math.nan)) for entry in entries
            ]),
            "mean_best_pair_detection_probability": _finite_mean([
                float(entry.get("best_pair_detection_probability", math.nan)) for entry in entries
            ]),
            "mean_best_norm_distance": _finite_mean([
                float(entry.get("best_norm_distance", math.nan)) for entry in entries
            ]),
            "mean_avg_best_norm_distance": _finite_mean([
                float(entry.get("avg_best_norm_distance", math.nan)) for entry in entries
            ]),
            "mean_min_distance_m": _finite_mean([
                float(entry.get("min_distance_m", math.nan)) for entry in entries
            ]),
            "mean_best_margin_m": _finite_mean([
                float(entry.get("best_margin_m", math.nan)) for entry in entries
            ]),
            "mean_near_edge_exposure_fraction": _finite_mean([
                float(entry.get("near_edge_exposure_fraction", math.nan)) for entry in entries
            ]),
            "mean_central_exposure_fraction": _finite_mean([
                float(entry.get("central_exposure_fraction", math.nan)) for entry in entries
            ]),
            "mean_scout_probability": _finite_mean([
                float(entry.get("scout_probability", math.nan)) for entry in entries
            ]),
            "mean_scout_norm_distance": _finite_mean([
                float(entry.get("scout_norm_distance", math.nan)) for entry in entries
            ]),
            "mean_scout_margin_m": _finite_mean([
                float(entry.get("scout_margin_m", math.nan)) for entry in entries
            ]),
            "mean_scout_confidence_pre": _finite_mean([
                float(entry.get("scout_confidence_pre", math.nan)) for entry in entries
            ]),
            "mean_scout_confidence_post": _finite_mean([
                float(entry.get("scout_confidence_post", math.nan)) for entry in entries
            ]),
        })
    return summaries


def _new_drone_stats(drone_idx: int) -> dict[str, Any]:
    return {
        "drone": int(drone_idx),
        "positions_sim": [],
        "action_norms": [],
        "displacement_m": [],
        "alignments": [],
        "alignments_new_cov": [],
        "alignments_no_new_cov": [],
        "new_coverage_cells": [],
        "raw_new_coverage_cells": [],
        "outside_footprint": [],
        "overlap": [],
        "expected_overlap": [],
        "excess_overlap": [],
        "inter_uav_overlap": [],
        "any_history_revisit": [],
        "own_history_revisit": [],
        "teammate_history_revisit": [],
        "own_only_revisit": [],
        "teammate_only_revisit": [],
        "shared_history_revisit": [],
        "unavoidable_revisit": [],
        "avoidable_revisit": [],
        "frontier_expected_new_cells": [],
        "frontier_new_cell_capture": [],
        "frontier_new_cell_gap": [],
        "candidate_best_new_cells": [],
        "candidate_capture_fraction": [],
        "candidate_new_cell_regret": [],
        "candidate_best_new_overlap": [],
        "candidate_best_useful_overlap": [],
        "candidate_avoidable_overlap": [],
        "candidate_action_rank": [],
        "candidate_movement_rank": [],
        "candidate_action_capture_fraction": [],
        "candidate_movement_capture_fraction": [],
        "candidate_action_best_alignment": [],
        "candidate_movement_best_alignment": [],
        "candidate_no_opportunity": [],
        "frontier_candidate_new_cells": [],
        "frontier_candidate_capture_fraction": [],
        "frontier_candidate_regret": [],
        "frontier_candidate_best_alignment": [],
        "frontier_candidate_rank": [],
        "frontier_candidate_nearest_rank": [],
        "frontier_candidate_is_best": [],
        "frontier_candidate_bad": [],
        "confidence_frontier_candidate_capture_fraction": [],
        "confidence_frontier_candidate_best_alignment": [],
        "confidence_frontier_candidate_rank": [],
        "confidence_frontier_candidate_bad": [],
        "confidence_lg_frontier_candidate_capture_fraction": [],
        "confidence_lg_frontier_candidate_best_alignment": [],
        "confidence_lg_frontier_candidate_rank": [],
        "confidence_lg_frontier_candidate_bad": [],
        "confidence_frontier_capture_advantage": [],
        "confidence_lg_frontier_capture_advantage": [],
        "coverage_opportunity_cells": [],
        "coverage_opportunity_fraction": [],
        "coverage_opportunity_available_fraction": [],
        "confidence_gain": [],
        "confidence_weighted_gain": [],
        "confidence_opportunity_fraction": [],
        "confidence_opportunity_best_gain": [],
        "confidence_pass_probability": [],
        "confidence_overlap_fraction": [],
        "confidence_overlap_regret": [],
        "cleanup_target_valid": [],
        "cleanup_target_distance_m": [],
        "cleanup_target_value": [],
        "cleanup_target_progress_m": [],
        "cleanup_target_progress_fraction": [],
        "cleanup_target_switch": [],
        "cleanup_target_reached": [],
        "cleanup_target_value_decay": [],
        "cleanup_target_age": [],
        "cleanup_target_frontier_gate": [],
        "frontier_alignment": [],
        "frontier_progress": [],
        "frontier_uncovered_ratio": [],
        "frontier_obs_distance": [],
        "frontier_obs_vector_norm": [],
        "frontier_obs_uncovered_ratio": [],
        "frontier_local_coverage_cos": [],
        "frontier_global_coverage_cos": [],
        "local_global_coverage_cos": [],
        "frontier_sector_cos": [],
        "frontier_sector_dominance": [],
        "frontier_sector_entropy": [],
        "frontier_cancellation": [],
        "action_frontier_alignment": [],
        "action_frontier_intent": [],
        "action_frontier_movement_gap": [],
        "reward_terms": {
            "coverage": [],
            "move_coverage": [],
            "frontier": [],
            "confidence": [],
            "team_confidence": [],
            "team_confidence_overlap_penalty": [],
            "confidence_move": [],
            "cleanup_target_progress": [],
            "astar_progress": [],
            "inefficient_move_penalty": [],
            "confidence_overlap_penalty": [],
            "overlap_penalty": [],
            "inter_uav_overlap_penalty": [],
            "outside_footprint_penalty": [],
            "coverage_threshold": [],
            "scout": [],
            "team": [],
            "all_survivors_found": [],
            "aux": [],
            "frontier_abs_share": [],
        },
        "is_edge_step": [],
        "is_corner_step": [],
        "boundary_distance_m": [],
        "footprint_radius_m": [],
        "scout_credit_count": 0,
        "scouted_survivors": set(),
        "first_scout_steps": [],
        "diagnostic_steps": 0,
        "low_action_high_motion": 0,
        "high_action_low_motion": 0,
        "moving_no_new_coverage": 0,
        "moving_no_confidence_gain": 0,
        "frontier_high_progress_steps": 0,
        "frontier_high_progress_no_new_steps": 0,
        "frontier_high_progress_edge_steps": 0,
        "frontier_high_progress_corner_steps": 0,
        "frontier_obs_empty_steps": 0,
        "action_frontier_aligned_steps": 0,
        "action_frontier_anti_aligned_steps": 0,
        "action_frontier_aligned_no_new_steps": 0,
        "action_frontier_aligned_edge_steps": 0,
    }


def _finalize_drone_stats(stats: dict[str, Any], scenario: WildfireSearchScenario) -> dict[str, Any]:
    positions = stats["positions_sim"]
    displacement = stats["displacement_m"]
    boundary = stats["boundary_distance_m"]
    footprint = stats["footprint_radius_m"]
    path = _path_metrics(positions, displacement, boundary, footprint, scenario)
    steps = int(stats["diagnostic_steps"])
    final_position_m: list[float] | None = None
    if positions:
        meters_per_sim = 1.0 / max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-9)
        final_position_m = (np.asarray(positions[-1], dtype=float) * meters_per_sim).round(6).tolist()
    new_cells = stats["new_coverage_cells"]
    raw_new_cells = stats["raw_new_coverage_cells"]
    excess = stats["excess_overlap"]
    inter_uav = stats["inter_uav_overlap"]
    any_history_revisit = stats["any_history_revisit"]
    own_history_revisit = stats["own_history_revisit"]
    teammate_history_revisit = stats["teammate_history_revisit"]
    own_only_revisit = stats["own_only_revisit"]
    teammate_only_revisit = stats["teammate_only_revisit"]
    shared_history_revisit = stats["shared_history_revisit"]
    unavoidable_revisit = stats["unavoidable_revisit"]
    avoidable_revisit = stats["avoidable_revisit"]
    frontier_expected_new_cells = stats["frontier_expected_new_cells"]
    frontier_new_cell_capture = stats["frontier_new_cell_capture"]
    frontier_new_cell_gap = stats["frontier_new_cell_gap"]
    candidate_best_new_cells = stats["candidate_best_new_cells"]
    candidate_capture_fraction = stats["candidate_capture_fraction"]
    candidate_new_cell_regret = stats["candidate_new_cell_regret"]
    candidate_best_new_overlap = stats["candidate_best_new_overlap"]
    candidate_best_useful_overlap = stats["candidate_best_useful_overlap"]
    candidate_avoidable_overlap = stats["candidate_avoidable_overlap"]
    candidate_action_rank = stats["candidate_action_rank"]
    candidate_movement_rank = stats["candidate_movement_rank"]
    candidate_action_capture = stats["candidate_action_capture_fraction"]
    candidate_movement_capture = stats["candidate_movement_capture_fraction"]
    candidate_action_best_alignment = stats["candidate_action_best_alignment"]
    candidate_movement_best_alignment = stats["candidate_movement_best_alignment"]
    candidate_no_opportunity = stats["candidate_no_opportunity"]
    frontier_candidate_new_cells = stats["frontier_candidate_new_cells"]
    frontier_candidate_capture = stats["frontier_candidate_capture_fraction"]
    frontier_candidate_regret = stats["frontier_candidate_regret"]
    frontier_candidate_best_alignment = stats["frontier_candidate_best_alignment"]
    frontier_candidate_rank = stats["frontier_candidate_rank"]
    frontier_candidate_nearest_rank = stats["frontier_candidate_nearest_rank"]
    frontier_candidate_is_best = stats["frontier_candidate_is_best"]
    frontier_candidate_bad = stats["frontier_candidate_bad"]
    confidence_frontier_candidate_capture = stats["confidence_frontier_candidate_capture_fraction"]
    confidence_frontier_candidate_best_alignment = stats[
        "confidence_frontier_candidate_best_alignment"
    ]
    confidence_frontier_candidate_rank = stats["confidence_frontier_candidate_rank"]
    confidence_frontier_candidate_bad = stats["confidence_frontier_candidate_bad"]
    confidence_lg_frontier_candidate_capture = stats[
        "confidence_lg_frontier_candidate_capture_fraction"
    ]
    confidence_lg_frontier_candidate_best_alignment = stats[
        "confidence_lg_frontier_candidate_best_alignment"
    ]
    confidence_lg_frontier_candidate_rank = stats["confidence_lg_frontier_candidate_rank"]
    confidence_lg_frontier_candidate_bad = stats["confidence_lg_frontier_candidate_bad"]
    confidence_frontier_capture_advantage = stats["confidence_frontier_capture_advantage"]
    confidence_lg_frontier_capture_advantage = stats["confidence_lg_frontier_capture_advantage"]
    opportunity_cells = stats["coverage_opportunity_cells"]
    opportunity_fraction = stats["coverage_opportunity_fraction"]
    opportunity_available_fraction = stats["coverage_opportunity_available_fraction"]
    confidence_gain = stats["confidence_gain"]
    confidence_weighted_gain = stats["confidence_weighted_gain"]
    confidence_opportunity_fraction = stats["confidence_opportunity_fraction"]
    confidence_opportunity_best_gain = stats["confidence_opportunity_best_gain"]
    confidence_pass_probability = stats["confidence_pass_probability"]
    confidence_overlap_fraction = stats["confidence_overlap_fraction"]
    confidence_overlap_regret = stats["confidence_overlap_regret"]
    cleanup_target_valid = stats["cleanup_target_valid"]
    cleanup_target_distance = stats["cleanup_target_distance_m"]
    cleanup_target_value = stats["cleanup_target_value"]
    cleanup_target_progress = stats["cleanup_target_progress_m"]
    cleanup_target_progress_fraction = stats["cleanup_target_progress_fraction"]
    cleanup_target_switch = stats["cleanup_target_switch"]
    cleanup_target_reached = stats["cleanup_target_reached"]
    cleanup_target_value_decay = stats["cleanup_target_value_decay"]
    cleanup_target_age = stats["cleanup_target_age"]
    cleanup_target_frontier_gate = stats["cleanup_target_frontier_gate"]
    frontier_alignment = stats["frontier_alignment"]
    frontier_progress = stats["frontier_progress"]
    frontier_ratio = stats["frontier_uncovered_ratio"]
    frontier_obs_distance = stats["frontier_obs_distance"]
    frontier_obs_norm = stats["frontier_obs_vector_norm"]
    frontier_local_cos = stats["frontier_local_coverage_cos"]
    frontier_global_cos = stats["frontier_global_coverage_cos"]
    local_global_cos = stats["local_global_coverage_cos"]
    frontier_sector_cos = stats["frontier_sector_cos"]
    frontier_sector_dominance = stats["frontier_sector_dominance"]
    frontier_sector_entropy = stats["frontier_sector_entropy"]
    frontier_cancellation = stats["frontier_cancellation"]
    action_frontier_alignment = stats["action_frontier_alignment"]
    action_frontier_intent = stats["action_frontier_intent"]
    action_frontier_gap = stats["action_frontier_movement_gap"]
    outside = stats["outside_footprint"]
    edge_mask = [bool(value) for value in stats["is_edge_step"]]
    high_frontier = int(stats["frontier_high_progress_steps"])
    action_frontier_aligned = int(stats["action_frontier_aligned_steps"])
    reward_terms = stats["reward_terms"]
    confidence_revisit = _confidence_revisit_metrics(
        avoidable_revisit,
        confidence_gain,
        confidence_opportunity_fraction,
        confidence_opportunity_best_gain,
    )
    return {
        "drone": int(stats["drone"]),
        "scout_credit_count": int(stats["scout_credit_count"]),
        "scouted_survivors": sorted(int(idx) for idx in stats["scouted_survivors"]),
        "first_scout_steps": [int(step) for step in stats["first_scout_steps"]],
        "final_position_m": final_position_m,
        "avg_action_norm": _finite_mean(stats["action_norms"]),
        "avg_displacement_m": _finite_mean(displacement),
        "path_length_m": float(np.sum(displacement)) if displacement else 0.0,
        "avg_action_displacement_alignment": _finite_mean(stats["alignments"]),
        "avg_action_displacement_alignment_new_cov": _finite_mean(stats["alignments_new_cov"]),
        "avg_action_displacement_alignment_no_new_cov": _finite_mean(stats["alignments_no_new_cov"]),
        "avg_action_frontier_alignment": _finite_mean(action_frontier_alignment),
        "avg_action_frontier_intent": _finite_mean(action_frontier_intent),
        "avg_action_frontier_movement_gap": _finite_mean(action_frontier_gap),
        "avg_new_coverage_cells": _finite_mean(new_cells),
        "avg_raw_new_coverage_cells": _finite_mean(raw_new_cells),
        "total_new_coverage_cells": float(np.sum(new_cells)) if new_cells else 0.0,
        "total_raw_new_coverage_cells": float(np.sum(raw_new_cells)) if raw_new_cells else 0.0,
        "new_coverage_step_frac": (
            float(np.mean([value >= 1.0 for value in new_cells]))
            if new_cells else 0.0
        ),
        "raw_new_coverage_step_frac": (
            float(np.mean([value >= 1.0 for value in raw_new_cells]))
            if raw_new_cells else 0.0
        ),
        "avg_outside_footprint_fraction": _finite_mean(outside),
        "max_outside_footprint_fraction": max(outside) if outside else 0.0,
        "avg_overlap_fraction": _finite_mean(stats["overlap"]),
        "avg_expected_overlap_fraction": _finite_mean(stats["expected_overlap"]),
        "avg_excess_overlap_fraction": _finite_mean(excess),
        "avg_inter_uav_overlap_fraction": _finite_mean(inter_uav),
        "avg_any_history_revisit_fraction": _finite_mean(any_history_revisit),
        "avg_own_history_revisit_fraction": _finite_mean(own_history_revisit),
        "avg_teammate_history_revisit_fraction": _finite_mean(teammate_history_revisit),
        "avg_own_only_revisit_fraction": _finite_mean(own_only_revisit),
        "avg_teammate_only_revisit_fraction": _finite_mean(teammate_only_revisit),
        "avg_shared_history_revisit_fraction": _finite_mean(shared_history_revisit),
        "avg_unavoidable_revisit_fraction": _finite_mean(unavoidable_revisit),
        "avg_avoidable_revisit_fraction": _finite_mean(avoidable_revisit),
        "avg_frontier_expected_new_cells": _finite_mean(frontier_expected_new_cells),
        "avg_frontier_new_cell_capture_fraction": _finite_mean(frontier_new_cell_capture),
        "avg_frontier_new_cell_gap": _finite_mean(frontier_new_cell_gap),
        "avg_candidate_best_new_cells": _finite_mean(candidate_best_new_cells),
        "avg_candidate_capture_fraction": _finite_mean(candidate_capture_fraction),
        "avg_candidate_new_cell_regret": _finite_mean(candidate_new_cell_regret),
        "avg_candidate_best_new_overlap": _finite_mean(candidate_best_new_overlap),
        "avg_candidate_best_useful_overlap": _finite_mean(candidate_best_useful_overlap),
        "avg_candidate_avoidable_overlap": _finite_mean(candidate_avoidable_overlap),
        "avg_candidate_action_rank": _finite_mean(candidate_action_rank),
        "avg_candidate_movement_rank": _finite_mean(candidate_movement_rank),
        "avg_candidate_action_capture_fraction": _finite_mean(candidate_action_capture),
        "avg_candidate_movement_capture_fraction": _finite_mean(candidate_movement_capture),
        "avg_candidate_action_best_alignment": _finite_mean(candidate_action_best_alignment),
        "avg_candidate_movement_best_alignment": _finite_mean(candidate_movement_best_alignment),
        "candidate_no_opportunity_frac": _finite_mean(candidate_no_opportunity),
        "avg_frontier_candidate_new_cells": _finite_mean(frontier_candidate_new_cells),
        "avg_frontier_candidate_capture_fraction": _finite_mean(frontier_candidate_capture),
        "avg_frontier_candidate_regret": _finite_mean(frontier_candidate_regret),
        "avg_frontier_candidate_best_alignment": _finite_mean(frontier_candidate_best_alignment),
        "avg_frontier_candidate_rank": _finite_mean(frontier_candidate_rank),
        "avg_frontier_candidate_nearest_rank": _finite_mean(frontier_candidate_nearest_rank),
        "frontier_candidate_is_best_frac": _finite_mean(frontier_candidate_is_best),
        "frontier_candidate_bad_frac": _finite_mean(frontier_candidate_bad),
        "avg_confidence_frontier_candidate_capture_fraction": _finite_mean(
            confidence_frontier_candidate_capture
        ),
        "avg_confidence_frontier_candidate_best_alignment": _finite_mean(
            confidence_frontier_candidate_best_alignment
        ),
        "avg_confidence_frontier_candidate_rank": _finite_mean(confidence_frontier_candidate_rank),
        "confidence_frontier_candidate_bad_frac": _finite_mean(confidence_frontier_candidate_bad),
        "avg_confidence_lg_frontier_candidate_capture_fraction": _finite_mean(
            confidence_lg_frontier_candidate_capture
        ),
        "avg_confidence_lg_frontier_candidate_best_alignment": _finite_mean(
            confidence_lg_frontier_candidate_best_alignment
        ),
        "avg_confidence_lg_frontier_candidate_rank": _finite_mean(
            confidence_lg_frontier_candidate_rank
        ),
        "confidence_lg_frontier_candidate_bad_frac": _finite_mean(
            confidence_lg_frontier_candidate_bad
        ),
        "avg_confidence_frontier_capture_advantage": _finite_mean(
            confidence_frontier_capture_advantage
        ),
        "avg_confidence_lg_frontier_capture_advantage": _finite_mean(
            confidence_lg_frontier_capture_advantage
        ),
        "avg_coverage_opportunity_cells": _finite_mean(opportunity_cells),
        "avg_coverage_opportunity_fraction": _finite_mean(opportunity_fraction),
        "avg_coverage_opportunity_available_fraction": _finite_mean(opportunity_available_fraction),
        "avg_confidence_gain": _finite_mean(confidence_gain),
        "total_confidence_gain": float(np.sum(confidence_gain)) if confidence_gain else 0.0,
        "avg_confidence_weighted_gain": _finite_mean(confidence_weighted_gain),
        "avg_confidence_opportunity_fraction": _finite_mean(confidence_opportunity_fraction),
        "avg_confidence_opportunity_best_gain": _finite_mean(confidence_opportunity_best_gain),
        "avg_confidence_pass_probability": _finite_mean(confidence_pass_probability),
        "avg_confidence_overlap_fraction": _finite_mean(confidence_overlap_fraction),
        "avg_confidence_overlap_regret": _finite_mean(confidence_overlap_regret),
        "avg_cleanup_target_valid_fraction": _finite_mean(cleanup_target_valid),
        "avg_cleanup_target_distance_m": _finite_mean(cleanup_target_distance),
        "avg_cleanup_target_value": _finite_mean(cleanup_target_value),
        "avg_cleanup_target_progress_m": _finite_mean(cleanup_target_progress),
        "avg_cleanup_target_progress_fraction": _finite_mean(cleanup_target_progress_fraction),
        "cleanup_target_switch_rate": _finite_mean(cleanup_target_switch),
        "cleanup_target_reached_rate": _finite_mean(cleanup_target_reached),
        "avg_cleanup_target_value_decay": _finite_mean(cleanup_target_value_decay),
        "avg_cleanup_target_age": _finite_mean(cleanup_target_age),
        "avg_cleanup_target_frontier_gate": _finite_mean(cleanup_target_frontier_gate),
        "cleanup_target_no_progress_frac": (
            float(np.mean([
                valid >= 0.5 and progress <= 1e-6
                for valid, progress in zip(cleanup_target_valid, cleanup_target_progress)
            ]))
            if cleanup_target_valid else 0.0
        ),
        "avg_frontier_alignment": _finite_mean(frontier_alignment),
        "avg_frontier_progress_fraction": _finite_mean(frontier_progress),
        "avg_frontier_uncovered_ratio": _finite_mean(frontier_ratio),
        "avg_frontier_obs_distance": _finite_mean(frontier_obs_distance),
        "avg_frontier_obs_vector_norm": _finite_mean(frontier_obs_norm),
        "avg_frontier_local_coverage_cos": _finite_mean(frontier_local_cos),
        "avg_frontier_global_coverage_cos": _finite_mean(frontier_global_cos),
        "avg_local_global_coverage_cos": _finite_mean(local_global_cos),
        "avg_frontier_sector_cos": _finite_mean(frontier_sector_cos),
        "avg_frontier_sector_dominance": _finite_mean(frontier_sector_dominance),
        "avg_frontier_sector_entropy": _finite_mean(frontier_sector_entropy),
        "avg_frontier_cancellation": _finite_mean(frontier_cancellation),
        "avg_reward_uav_coverage": _finite_mean(reward_terms["coverage"]),
        "avg_reward_uav_move_coverage": _finite_mean(reward_terms["move_coverage"]),
        "avg_reward_uav_frontier": _finite_mean(reward_terms["frontier"]),
        "avg_reward_uav_confidence": _finite_mean(reward_terms["confidence"]),
        "avg_reward_uav_team_confidence": _finite_mean(reward_terms["team_confidence"]),
        "avg_penalty_uav_team_confidence_overlap": _finite_mean(
            reward_terms["team_confidence_overlap_penalty"]
        ),
        "avg_reward_uav_confidence_move": _finite_mean(reward_terms["confidence_move"]),
        "avg_reward_uav_cleanup_target_progress": _finite_mean(
            reward_terms["cleanup_target_progress"]
        ),
        "avg_reward_uav_astar_progress": _finite_mean(reward_terms["astar_progress"]),
        "avg_penalty_uav_inefficient_move": _finite_mean(
            reward_terms["inefficient_move_penalty"]
        ),
        "avg_penalty_uav_confidence_overlap": _finite_mean(
            reward_terms["confidence_overlap_penalty"]
        ),
        "avg_penalty_uav_overlap": _finite_mean(reward_terms["overlap_penalty"]),
        "avg_penalty_uav_inter_overlap": _finite_mean(reward_terms["inter_uav_overlap_penalty"]),
        "avg_penalty_uav_outside_footprint": _finite_mean(reward_terms["outside_footprint_penalty"]),
        "avg_reward_uav_coverage_threshold": _finite_mean(reward_terms["coverage_threshold"]),
        "avg_reward_uav_scout": _finite_mean(reward_terms["scout"]),
        "avg_reward_team": _finite_mean(reward_terms["team"]),
        "avg_reward_all_survivors_found": _finite_mean(reward_terms["all_survivors_found"]),
        "avg_reward_uav_aux": _finite_mean(reward_terms["aux"]),
        "avg_frontier_abs_reward_share": _finite_mean(reward_terms["frontier_abs_share"]),
        "frontier_high_progress_step_frac": high_frontier / steps if steps else 0.0,
        "frontier_high_progress_no_new_frac": (
            stats["frontier_high_progress_no_new_steps"] / high_frontier
            if high_frontier else 0.0
        ),
        "frontier_high_progress_edge_frac": (
            stats["frontier_high_progress_edge_steps"] / high_frontier
            if high_frontier else 0.0
        ),
        "frontier_high_progress_corner_frac": (
            stats["frontier_high_progress_corner_steps"] / high_frontier
            if high_frontier else 0.0
        ),
        "frontier_edge_progress_mean": _finite_mean([
            value for value, is_edge in zip(frontier_progress, edge_mask) if is_edge
        ]),
        "frontier_interior_progress_mean": _finite_mean([
            value for value, is_edge in zip(frontier_progress, edge_mask) if not is_edge
        ]),
        "frontier_edge_reward_mean": _finite_mean([
            value for value, is_edge in zip(reward_terms["frontier"], edge_mask) if is_edge
        ]),
        "frontier_interior_reward_mean": _finite_mean([
            value for value, is_edge in zip(reward_terms["frontier"], edge_mask) if not is_edge
        ]),
        "frontier_edge_new_cells_mean": _finite_mean([
            value for value, is_edge in zip(new_cells, edge_mask) if is_edge
        ]),
        "frontier_interior_new_cells_mean": _finite_mean([
            value for value, is_edge in zip(new_cells, edge_mask) if not is_edge
        ]),
        "frontier_edge_outside_mean": _finite_mean([
            value for value, is_edge in zip(outside, edge_mask) if is_edge
        ]),
        "frontier_interior_outside_mean": _finite_mean([
            value for value, is_edge in zip(outside, edge_mask) if not is_edge
        ]),
        "frontier_progress_new_cells_corr": _safe_corr(frontier_progress, new_cells),
        "frontier_progress_boundary_distance_corr": _safe_corr(frontier_progress, boundary),
        "frontier_obs_empty_step_frac": (
            stats["frontier_obs_empty_steps"] / steps if steps else 0.0
        ),
        "action_frontier_aligned_step_frac": (
            action_frontier_aligned / steps if steps else 0.0
        ),
        "action_frontier_anti_aligned_step_frac": (
            stats["action_frontier_anti_aligned_steps"] / steps if steps else 0.0
        ),
        "action_frontier_aligned_no_new_frac": (
            stats["action_frontier_aligned_no_new_steps"] / action_frontier_aligned
            if action_frontier_aligned else 0.0
        ),
        "action_frontier_aligned_edge_frac": (
            stats["action_frontier_aligned_edge_steps"] / action_frontier_aligned
            if action_frontier_aligned else 0.0
        ),
        "action_frontier_alignment_new_cells_corr": _safe_corr(action_frontier_alignment, new_cells),
        "action_frontier_alignment_boundary_distance_corr": _safe_corr(
            action_frontier_alignment,
            boundary,
        ),
        "excess_overlap_step_frac_10": (
            float(np.mean([value >= 0.10 for value in excess]))
            if excess else 0.0
        ),
        "inter_uav_overlap_step_frac_20": (
            float(np.mean([value >= 0.20 for value in inter_uav]))
            if inter_uav else 0.0
        ),
        "excess_overlap_step_frac_20": (
            float(np.mean([value >= 0.20 for value in excess]))
            if excess else 0.0
        ),
        "low_action_high_motion_frac": (
            stats["low_action_high_motion"] / steps if steps else 0.0
        ),
        "high_action_low_motion_frac": (
            stats["high_action_low_motion"] / steps if steps else 0.0
        ),
        "moving_no_new_coverage_frac": (
            stats["moving_no_new_coverage"] / steps if steps else 0.0
        ),
        "moving_no_confidence_gain_frac": (
            stats["moving_no_confidence_gain"] / steps if steps else 0.0
        ),
        **confidence_revisit,
        **path,
    }


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
            "coverage_bbox_hole_fraction": 0.0,
            "coverage_center_fraction": 0.0,
            "coverage_border_band_fraction": 0.0,
            "coverage_interior_fraction": 0.0,
            "coverage_edge_bias": 0.0,
            **_uncovered_component_metrics(~covered),
        }

    yy, xx = np.nonzero(covered)
    bbox_area = float((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1))
    total_cells = float(size_x * size_y)
    bbox_fill = float(covered.sum() / max(bbox_area, 1.0))
    bbox_hole_fraction = 1.0 - bbox_fill

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
        "coverage_bbox_hole_fraction": bbox_hole_fraction,
        "coverage_center_fraction": center_fraction,
        "coverage_border_band_fraction": border_fraction,
        "coverage_interior_fraction": interior_fraction,
        "coverage_edge_bias": border_fraction - interior_fraction,
        **_uncovered_component_metrics(~covered),
    }


def _fast_coverage_shape_metrics(coverage_grid: np.ndarray) -> dict[str, float]:
    covered = np.asarray(coverage_grid, dtype=bool)
    if covered.ndim != 2:
        return {
            "coverage_bbox_area_fraction": 0.0,
            "coverage_bbox_fill_fraction": 0.0,
            "coverage_bbox_hole_fraction": 0.0,
            "coverage_center_fraction": 0.0,
            "coverage_border_band_fraction": 0.0,
            "coverage_interior_fraction": 0.0,
            "coverage_edge_bias": 0.0,
            "coverage_uncovered_component_count": 0.0,
            "coverage_enclosed_uncovered_component_count": 0.0,
            "coverage_uncovered_fraction": 0.0,
            "coverage_enclosed_uncovered_fraction": 0.0,
            "coverage_largest_uncovered_component_fraction": 0.0,
            "coverage_largest_enclosed_hole_fraction": 0.0,
            "coverage_enclosed_hole_share": 0.0,
        }
    height, width = covered.shape
    total_cells = float(max(height * width, 1))
    if not covered.any():
        bbox_area_fraction = 0.0
        bbox_fill = 0.0
        bbox_hole_fraction = 0.0
    else:
        yy, xx = np.nonzero(covered)
        bbox_area = float((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1))
        bbox_area_fraction = float(bbox_area / total_cells)
        bbox_fill = float(covered.sum() / max(bbox_area, 1.0))
        bbox_hole_fraction = 1.0 - bbox_fill
    center_margin_x = width // 4
    center_margin_y = height // 4
    center = covered[
        center_margin_y : height - center_margin_y,
        center_margin_x : width - center_margin_x,
    ]
    center_fraction = float(center.mean()) if center.size else 0.0
    uncovered_fraction = float((~covered).mean()) if total_cells else 0.0
    return {
        "coverage_bbox_area_fraction": bbox_area_fraction,
        "coverage_bbox_fill_fraction": bbox_fill,
        "coverage_bbox_hole_fraction": bbox_hole_fraction,
        "coverage_center_fraction": center_fraction,
        "coverage_border_band_fraction": 0.0,
        "coverage_interior_fraction": 0.0,
        "coverage_edge_bias": 0.0,
        "coverage_uncovered_component_count": 0.0,
        "coverage_enclosed_uncovered_component_count": 0.0,
        "coverage_uncovered_fraction": uncovered_fraction,
        "coverage_enclosed_uncovered_fraction": 0.0,
        "coverage_largest_uncovered_component_fraction": 0.0,
        "coverage_largest_enclosed_hole_fraction": 0.0,
        "coverage_enclosed_hole_share": 0.0,
    }


def _uncovered_component_metrics(uncovered_grid: np.ndarray) -> dict[str, float]:
    uncovered = np.asarray(uncovered_grid, dtype=bool)
    if uncovered.ndim != 2:
        return {
            "coverage_uncovered_component_count": 0.0,
            "coverage_enclosed_uncovered_component_count": 0.0,
            "coverage_uncovered_fraction": 0.0,
            "coverage_enclosed_uncovered_fraction": 0.0,
            "coverage_largest_uncovered_component_fraction": 0.0,
            "coverage_largest_enclosed_hole_fraction": 0.0,
            "coverage_enclosed_hole_share": 0.0,
        }

    height, width = uncovered.shape
    total_cells = float(max(height * width, 1))
    uncovered_total = int(uncovered.sum())
    if uncovered_total == 0:
        return {
            "coverage_uncovered_component_count": 0.0,
            "coverage_enclosed_uncovered_component_count": 0.0,
            "coverage_uncovered_fraction": 0.0,
            "coverage_enclosed_uncovered_fraction": 0.0,
            "coverage_largest_uncovered_component_fraction": 0.0,
            "coverage_largest_enclosed_hole_fraction": 0.0,
            "coverage_enclosed_hole_share": 0.0,
        }

    visited = np.zeros_like(uncovered, dtype=bool)
    component_count = 0
    enclosed_count = 0
    enclosed_cells = 0
    largest_component = 0
    largest_enclosed = 0
    for y in range(height):
        for x in range(width):
            if not uncovered[y, x] or visited[y, x]:
                continue
            component_count += 1
            touches_border = False
            size = 0
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                size += 1
                if cy == 0 or cx == 0 or cy == height - 1 or cx == width - 1:
                    touches_border = True
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and uncovered[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            largest_component = max(largest_component, size)
            if not touches_border:
                enclosed_count += 1
                enclosed_cells += size
                largest_enclosed = max(largest_enclosed, size)

    return {
        "coverage_uncovered_component_count": float(component_count),
        "coverage_enclosed_uncovered_component_count": float(enclosed_count),
        "coverage_uncovered_fraction": float(uncovered_total / total_cells),
        "coverage_enclosed_uncovered_fraction": float(enclosed_cells / total_cells),
        "coverage_largest_uncovered_component_fraction": float(largest_component / total_cells),
        "coverage_largest_enclosed_hole_fraction": float(largest_enclosed / total_cells),
        "coverage_enclosed_hole_share": float(enclosed_cells / max(uncovered_total, 1)),
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
    binary_revisit_bad = (
        row["moving_no_new_coverage_frac"] > 0.30
        or (
            row["avg_excess_overlap_fraction"] > 0.20
            and row["new_coverage_step_frac"] < 0.75
        )
        or (
            row["excess_overlap_step_frac_20"] > 0.30
            and row["moving_no_new_coverage_frac"] > 0.15
        )
    )
    if binary_revisit_bad:
        confidence_reward_active = (
            abs(float(row.get("avg_reward_uav_confidence", 0.0))) > 1e-9
            or abs(float(row.get("avg_reward_uav_confidence_move", 0.0))) > 1e-9
        )
        if not confidence_reward_active:
            return "wasteful_revisit"
        useful_share = float(row.get("confidence_revisit_useful_share", math.nan))
        revisit_frac = float(row.get("confidence_revisit_step_frac", 0.0))
        wasteful_frac = float(row.get("confidence_wasteful_revisit_step_frac", 0.0))
        useful_frac = float(row.get("confidence_useful_revisit_step_frac", 0.0))
        if (
            not math.isfinite(useful_share)
            and revisit_frac <= 0.0
            and wasteful_frac <= 0.0
            and useful_frac <= 0.0
        ):
            return "wasteful_revisit"
        if (
            wasteful_frac > 0.20
            or (
                revisit_frac > 0.20
                and (not math.isfinite(useful_share) or useful_share < 0.35)
            )
        ):
            return "wasteful_revisit"
        if useful_frac > 0.10 and math.isfinite(useful_share) and useful_share >= 0.45:
            return "confidence_reinspection"
    if row["path_bbox_area_fraction"] < 0.10:
        return "small_search_area"
    if row["stalled_step_frac"] > 0.20 or row["longest_stall_steps"] >= 25:
        return "stalled"
    if row["new_coverage_step_frac"] >= 0.85 and row["avg_excess_overlap_fraction"] <= 0.15:
        return "productive_sweep"
    if row["final_coverage_fraction"] >= 0.45 and row["scouted"] <= 1:
        return "survivor_miss_despite_coverage"
    return "partial_search"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["all_scouted_step"] is not None]
    confirmation_successful = [
        row for row in rows if row.get("all_confirmed_step") is not None
    ]
    summary = {
        "episodes": float(len(rows)),
        "mean_scouted": float(np.mean([row["scouted"] for row in rows])) if rows else 0.0,
        "mean_missed": float(np.mean([row["missed"] for row in rows])) if rows else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
        "mean_confirmed": float(np.mean([row.get("confirmed", 0.0) for row in rows])) if rows else 0.0,
        "mean_unconfirmed": float(np.mean([row.get("unconfirmed", 0.0) for row in rows])) if rows else 0.0,
        "mean_confirmation_recall": (
            float(np.mean([row.get("confirmation_recall", 0.0) for row in rows])) if rows else 0.0
        ),
        "mean_final_coverage_fraction": (
            float(np.mean([row["final_coverage_fraction"] for row in rows])) if rows else 0.0
        ),
        "mean_final_confidence_mean": _finite_mean([
            row["final_confidence_mean"] for row in rows
        ]),
        "mean_final_confidence_low_fraction": _finite_mean([
            row["final_confidence_low_fraction"] for row in rows
        ]),
        "mean_final_confidence_high_fraction": _finite_mean([
            row["final_confidence_high_fraction"] for row in rows
        ]),
        "full_success_rate": float(np.mean([row["full_success"] for row in rows])) if rows else 0.0,
        "full_confirmation_success_rate": (
            float(np.mean([row.get("full_confirmation_success", 0.0) for row in rows])) if rows else 0.0
        ),
        "mean_avg_scout_step": _finite_mean([row["avg_scout_step"] for row in rows]),
        "mean_avg_scout_time_s": _finite_mean([row["avg_scout_time_s"] for row in rows]),
        "mean_avg_confirm_step": _finite_mean([
            row.get("avg_confirm_step", math.nan) for row in rows
        ]),
        "mean_avg_confirm_time_s": _finite_mean([
            row.get("avg_confirm_time_s", math.nan) for row in rows
        ]),
        "mean_all_scouted_step_successes": (
            float(np.mean([row["all_scouted_step"] for row in successful])) if successful else math.nan
        ),
        "mean_all_scouted_time_s_successes": (
            float(np.mean([row["all_scouted_time_s"] for row in successful])) if successful else math.nan
        ),
        "mean_all_confirmed_step_successes": (
            float(np.mean([row["all_confirmed_step"] for row in confirmation_successful]))
            if confirmation_successful else math.nan
        ),
        "mean_all_confirmed_time_s_successes": (
            float(np.mean([row["all_confirmed_time_s"] for row in confirmation_successful]))
            if confirmation_successful else math.nan
        ),
        "mean_survivor_exposure_steps": _finite_mean([
            row["avg_survivor_exposure_steps"] for row in rows
        ]),
        "mean_scouted_survivor_exposure_steps": _finite_mean([
            row["avg_scouted_survivor_exposure_steps"] for row in rows
        ]),
        "mean_missed_survivor_exposure_steps": _finite_mean([
            row["avg_missed_survivor_exposure_steps"] for row in rows
        ]),
        "mean_survivor_pair_exposures": _finite_mean([
            row["avg_survivor_pair_exposures"] for row in rows
        ]),
        "mean_survivor_cum_detection_probability": _finite_mean([
            row["avg_survivor_cum_detection_probability"] for row in rows
        ]),
        "mean_scouted_cum_detection_probability": _finite_mean([
            row["avg_scouted_cum_detection_probability"] for row in rows
        ]),
        "mean_missed_cum_detection_probability": _finite_mean([
            row["avg_missed_cum_detection_probability"] for row in rows
        ]),
        "mean_survivor_final_confidence": _finite_mean([
            row["avg_survivor_final_confidence"] for row in rows
        ]),
        "mean_scouted_survivor_final_confidence": _finite_mean([
            row["avg_scouted_survivor_final_confidence"] for row in rows
        ]),
        "mean_missed_survivor_final_confidence": _finite_mean([
            row["avg_missed_survivor_final_confidence"] for row in rows
        ]),
        "mean_survivor_best_detection_probability": _finite_mean([
            row["avg_survivor_best_detection_probability"] for row in rows
        ]),
        "mean_missed_best_detection_probability": _finite_mean([
            row["avg_missed_best_detection_probability"] for row in rows
        ]),
        "mean_scout_detection_probability": _finite_mean([
            row["avg_scout_detection_probability"] for row in rows
        ]),
        "mean_scout_detection_norm_distance": _finite_mean([
            row["avg_scout_detection_norm_distance"] for row in rows
        ]),
        "mean_scout_detection_margin_m": _finite_mean([
            row["avg_scout_detection_margin_m"] for row in rows
        ]),
        "mean_scout_confidence_pre": _finite_mean([
            row["avg_scout_confidence_pre"] for row in rows
        ]),
        "mean_scout_confidence_post": _finite_mean([
            row["avg_scout_confidence_post"] for row in rows
        ]),
        "mean_missed_best_norm_distance": _finite_mean([
            row["avg_missed_best_norm_distance"] for row in rows
        ]),
        "mean_missed_min_distance_m": _finite_mean([
            row["avg_missed_min_distance_m"] for row in rows
        ]),
        "mean_missed_best_margin_m": _finite_mean([
            row["avg_missed_best_margin_m"] for row in rows
        ]),
        "mean_exposed_survivor_edge_fraction": _finite_mean([
            row["avg_exposed_survivor_edge_fraction"] for row in rows
        ]),
        "mean_scout_exposure_to_scout_latency_steps": _finite_mean([
            row["avg_scout_exposure_to_scout_latency_steps"] for row in rows
        ]),
        "mean_expected_recall_from_exposure": _finite_mean([
            row["expected_recall_from_exposure"] for row in rows
        ]),
        "mean_perception_recall_gap": _finite_mean([
            row["perception_recall_gap"] for row in rows
        ]),
        "mean_missed_never_exposed_fraction": _finite_mean([
            row["missed_never_exposed_fraction"] for row in rows
        ]),
        "mean_missed_low_cum_probability_fraction": _finite_mean([
            row["missed_low_cum_probability_fraction"] for row in rows
        ]),
        "mean_missed_high_cum_probability_fraction": _finite_mean([
            row["missed_high_cum_probability_fraction"] for row in rows
        ]),
        "mean_missed_edge_limited_fraction": _finite_mean([
            row["missed_edge_limited_fraction"] for row in rows
        ]),
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
        "mean_action_frontier_alignment": _finite_mean([
            row["avg_action_frontier_alignment"] for row in rows
        ]),
        "mean_action_frontier_alignment_new_cov": _finite_mean([
            row["avg_action_frontier_alignment_new_cov"] for row in rows
        ]),
        "mean_action_frontier_alignment_no_new_cov": _finite_mean([
            row["avg_action_frontier_alignment_no_new_cov"] for row in rows
        ]),
        "mean_action_frontier_intent": _finite_mean([
            row["avg_action_frontier_intent"] for row in rows
        ]),
        "mean_action_frontier_movement_gap": _finite_mean([
            row["avg_action_frontier_movement_gap"] for row in rows
        ]),
        "mean_new_coverage_cells": _finite_mean([row["avg_new_coverage_cells"] for row in rows]),
        "mean_raw_new_coverage_cells": _finite_mean([
            row["avg_raw_new_coverage_cells"] for row in rows
        ]),
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
        "mean_inter_uav_overlap_fraction": _finite_mean([
            row["avg_inter_uav_overlap_fraction"] for row in rows
        ]),
        "mean_any_history_revisit_fraction": _finite_mean([
            row["avg_any_history_revisit_fraction"] for row in rows
        ]),
        "mean_own_history_revisit_fraction": _finite_mean([
            row["avg_own_history_revisit_fraction"] for row in rows
        ]),
        "mean_teammate_history_revisit_fraction": _finite_mean([
            row["avg_teammate_history_revisit_fraction"] for row in rows
        ]),
        "mean_own_only_revisit_fraction": _finite_mean([
            row["avg_own_only_revisit_fraction"] for row in rows
        ]),
        "mean_teammate_only_revisit_fraction": _finite_mean([
            row["avg_teammate_only_revisit_fraction"] for row in rows
        ]),
        "mean_shared_history_revisit_fraction": _finite_mean([
            row["avg_shared_history_revisit_fraction"] for row in rows
        ]),
        "mean_unavoidable_revisit_fraction": _finite_mean([
            row["avg_unavoidable_revisit_fraction"] for row in rows
        ]),
        "mean_avoidable_revisit_fraction": _finite_mean([
            row["avg_avoidable_revisit_fraction"] for row in rows
        ]),
        "mean_frontier_expected_new_cells": _finite_mean([
            row["avg_frontier_expected_new_cells"] for row in rows
        ]),
        "mean_frontier_new_cell_capture_fraction": _finite_mean([
            row["avg_frontier_new_cell_capture_fraction"] for row in rows
        ]),
        "mean_frontier_new_cell_gap": _finite_mean([
            row["avg_frontier_new_cell_gap"] for row in rows
        ]),
        "mean_candidate_best_new_cells": _finite_mean([
            row["avg_candidate_best_new_cells"] for row in rows
        ]),
        "mean_candidate_capture_fraction": _finite_mean([
            row["avg_candidate_capture_fraction"] for row in rows
        ]),
        "mean_candidate_new_cell_regret": _finite_mean([
            row["avg_candidate_new_cell_regret"] for row in rows
        ]),
        "mean_candidate_best_new_overlap": _finite_mean([
            row["avg_candidate_best_new_overlap"] for row in rows
        ]),
        "mean_candidate_best_useful_overlap": _finite_mean([
            row["avg_candidate_best_useful_overlap"] for row in rows
        ]),
        "mean_candidate_avoidable_overlap": _finite_mean([
            row["avg_candidate_avoidable_overlap"] for row in rows
        ]),
        "mean_candidate_action_rank": _finite_mean([
            row["avg_candidate_action_rank"] for row in rows
        ]),
        "mean_candidate_movement_rank": _finite_mean([
            row["avg_candidate_movement_rank"] for row in rows
        ]),
        "mean_candidate_action_capture_fraction": _finite_mean([
            row["avg_candidate_action_capture_fraction"] for row in rows
        ]),
        "mean_candidate_movement_capture_fraction": _finite_mean([
            row["avg_candidate_movement_capture_fraction"] for row in rows
        ]),
        "mean_candidate_action_best_alignment": _finite_mean([
            row["avg_candidate_action_best_alignment"] for row in rows
        ]),
        "mean_candidate_movement_best_alignment": _finite_mean([
            row["avg_candidate_movement_best_alignment"] for row in rows
        ]),
        "mean_candidate_no_opportunity_frac": _finite_mean([
            row["candidate_no_opportunity_frac"] for row in rows
        ]),
        "mean_frontier_candidate_new_cells": _finite_mean([
            row["avg_frontier_candidate_new_cells"] for row in rows
        ]),
        "mean_frontier_candidate_capture_fraction": _finite_mean([
            row["avg_frontier_candidate_capture_fraction"] for row in rows
        ]),
        "mean_frontier_candidate_regret": _finite_mean([
            row["avg_frontier_candidate_regret"] for row in rows
        ]),
        "mean_frontier_candidate_best_alignment": _finite_mean([
            row["avg_frontier_candidate_best_alignment"] for row in rows
        ]),
        "mean_frontier_candidate_rank": _finite_mean([
            row["avg_frontier_candidate_rank"] for row in rows
        ]),
        "mean_frontier_candidate_nearest_rank": _finite_mean([
            row["avg_frontier_candidate_nearest_rank"] for row in rows
        ]),
        "mean_frontier_candidate_is_best_frac": _finite_mean([
            row["frontier_candidate_is_best_frac"] for row in rows
        ]),
        "mean_frontier_candidate_bad_frac": _finite_mean([
            row["frontier_candidate_bad_frac"] for row in rows
        ]),
        "mean_confidence_frontier_candidate_capture_fraction": _finite_mean([
            row["avg_confidence_frontier_candidate_capture_fraction"] for row in rows
        ]),
        "mean_confidence_frontier_candidate_best_alignment": _finite_mean([
            row["avg_confidence_frontier_candidate_best_alignment"] for row in rows
        ]),
        "mean_confidence_frontier_candidate_rank": _finite_mean([
            row["avg_confidence_frontier_candidate_rank"] for row in rows
        ]),
        "mean_confidence_frontier_candidate_bad_frac": _finite_mean([
            row["confidence_frontier_candidate_bad_frac"] for row in rows
        ]),
        "mean_confidence_lg_frontier_candidate_capture_fraction": _finite_mean([
            row["avg_confidence_lg_frontier_candidate_capture_fraction"] for row in rows
        ]),
        "mean_confidence_lg_frontier_candidate_best_alignment": _finite_mean([
            row["avg_confidence_lg_frontier_candidate_best_alignment"] for row in rows
        ]),
        "mean_confidence_lg_frontier_candidate_rank": _finite_mean([
            row["avg_confidence_lg_frontier_candidate_rank"] for row in rows
        ]),
        "mean_confidence_lg_frontier_candidate_bad_frac": _finite_mean([
            row["confidence_lg_frontier_candidate_bad_frac"] for row in rows
        ]),
        "mean_confidence_frontier_capture_advantage": _finite_mean([
            row["avg_confidence_frontier_capture_advantage"] for row in rows
        ]),
        "mean_confidence_lg_frontier_capture_advantage": _finite_mean([
            row["avg_confidence_lg_frontier_capture_advantage"] for row in rows
        ]),
        "mean_frontier_candidate_capture_new_cells_corr": _finite_mean([
            row["frontier_candidate_capture_new_cells_corr"] for row in rows
        ]),
        "mean_frontier_candidate_alignment_new_cells_corr": _finite_mean([
            row["frontier_candidate_alignment_new_cells_corr"] for row in rows
        ]),
        "mean_coverage_opportunity_cells": _finite_mean([
            row["avg_coverage_opportunity_cells"] for row in rows
        ]),
        "mean_coverage_opportunity_fraction": _finite_mean([
            row["avg_coverage_opportunity_fraction"] for row in rows
        ]),
        "mean_coverage_opportunity_available_fraction": _finite_mean([
            row["avg_coverage_opportunity_available_fraction"] for row in rows
        ]),
        "mean_confidence_mean": _finite_mean([
            row["avg_confidence_mean"] for row in rows
        ]),
        "mean_confidence_gain": _finite_mean([
            row["avg_confidence_gain"] for row in rows
        ]),
        "mean_confidence_gain_by_drone": _finite_mean([
            row["avg_confidence_gain_by_drone"] for row in rows
        ]),
        "mean_confidence_weighted_gain": _finite_mean([
            row["avg_confidence_weighted_gain"] for row in rows
        ]),
        "mean_confidence_weighted_gain_by_drone": _finite_mean([
            row["avg_confidence_weighted_gain_by_drone"] for row in rows
        ]),
        "mean_confidence_opportunity_fraction": _finite_mean([
            row["avg_confidence_opportunity_fraction"] for row in rows
        ]),
        "mean_confidence_opportunity_best_gain": _finite_mean([
            row["avg_confidence_opportunity_best_gain"] for row in rows
        ]),
        "mean_confidence_revisit_step_frac": _finite_mean([
            float(row.get("confidence_revisit_step_frac", math.nan)) for row in rows
        ]),
        "mean_confidence_useful_revisit_step_frac": _finite_mean([
            float(row.get("confidence_useful_revisit_step_frac", math.nan)) for row in rows
        ]),
        "mean_confidence_wasteful_revisit_step_frac": _finite_mean([
            float(row.get("confidence_wasteful_revisit_step_frac", math.nan)) for row in rows
        ]),
        "mean_confidence_ambiguous_revisit_step_frac": _finite_mean([
            float(row.get("confidence_ambiguous_revisit_step_frac", math.nan)) for row in rows
        ]),
        "mean_confidence_revisit_useful_share": _finite_mean([
            float(row.get("confidence_revisit_useful_share", math.nan)) for row in rows
        ]),
        "mean_confidence_revisit_wasteful_share": _finite_mean([
            float(row.get("confidence_revisit_wasteful_share", math.nan)) for row in rows
        ]),
        "mean_confidence_revisit_gain_share": _finite_mean([
            float(row.get("confidence_revisit_gain_share", math.nan)) for row in rows
        ]),
        "mean_confidence_gain_on_revisit": _finite_mean([
            float(row.get("confidence_gain_on_revisit", math.nan)) for row in rows
        ]),
        "mean_confidence_gain_off_revisit": _finite_mean([
            float(row.get("confidence_gain_off_revisit", math.nan)) for row in rows
        ]),
        "mean_confidence_opportunity_on_revisit": _finite_mean([
            float(row.get("confidence_opportunity_on_revisit", math.nan)) for row in rows
        ]),
        "mean_confidence_opportunity_off_revisit": _finite_mean([
            float(row.get("confidence_opportunity_off_revisit", math.nan)) for row in rows
        ]),
        "mean_confidence_best_gain_on_revisit": _finite_mean([
            float(row.get("confidence_best_gain_on_revisit", math.nan)) for row in rows
        ]),
        "mean_confidence_best_gain_off_revisit": _finite_mean([
            float(row.get("confidence_best_gain_off_revisit", math.nan)) for row in rows
        ]),
        "mean_confidence_low_fraction": _finite_mean([
            row["avg_confidence_low_fraction"] for row in rows
        ]),
        "mean_confidence_high_fraction": _finite_mean([
            row["avg_confidence_high_fraction"] for row in rows
        ]),
        "mean_confidence_step_detection_probability": _finite_mean([
            row["avg_confidence_step_detection_probability"] for row in rows
        ]),
        "mean_confidence_pass_probability": _finite_mean([
            row["avg_confidence_pass_probability"] for row in rows
        ]),
        "mean_confidence_overlap_fraction": _finite_mean([
            row["avg_confidence_overlap_fraction"] for row in rows
        ]),
        "mean_confidence_overlap_regret": _finite_mean([
            row.get("avg_confidence_overlap_regret", math.nan) for row in rows
        ]),
        "mean_cleanup_target_valid_fraction": _finite_mean([
            row.get("avg_cleanup_target_valid_fraction", math.nan) for row in rows
        ]),
        "mean_cleanup_target_distance_m": _finite_mean([
            row.get("avg_cleanup_target_distance_m", math.nan) for row in rows
        ]),
        "mean_cleanup_target_value": _finite_mean([
            row.get("avg_cleanup_target_value", math.nan) for row in rows
        ]),
        "mean_cleanup_target_progress_m": _finite_mean([
            row.get("avg_cleanup_target_progress_m", math.nan) for row in rows
        ]),
        "mean_cleanup_target_progress_fraction": _finite_mean([
            row.get("avg_cleanup_target_progress_fraction", math.nan) for row in rows
        ]),
        "mean_cleanup_target_switch_rate": _finite_mean([
            row.get("cleanup_target_switch_rate", math.nan) for row in rows
        ]),
        "mean_cleanup_target_reached_rate": _finite_mean([
            row.get("cleanup_target_reached_rate", math.nan) for row in rows
        ]),
        "mean_cleanup_target_value_decay": _finite_mean([
            row.get("avg_cleanup_target_value_decay", math.nan) for row in rows
        ]),
        "mean_cleanup_target_no_progress_frac": _finite_mean([
            row.get("cleanup_target_no_progress_frac", math.nan) for row in rows
        ]),
        "mean_cleanup_target_progress_with_new_cells_frac": _finite_mean([
            row.get("cleanup_target_progress_with_new_cells_frac", math.nan) for row in rows
        ]),
        "mean_cleanup_target_progress_with_excess_overlap_frac": _finite_mean([
            row.get("cleanup_target_progress_with_excess_overlap_frac", math.nan) for row in rows
        ]),
        "mean_cleanup_target_frontier_gate": _finite_mean([
            row.get("avg_cleanup_target_frontier_gate", math.nan) for row in rows
        ]),
        "mean_frontier_alignment": _finite_mean([
            row["avg_frontier_alignment"] for row in rows
        ]),
        "mean_frontier_progress_fraction": _finite_mean([
            row["avg_frontier_progress_fraction"] for row in rows
        ]),
        "mean_frontier_uncovered_ratio": _finite_mean([
            row["avg_frontier_uncovered_ratio"] for row in rows
        ]),
        "mean_frontier_obs_distance": _finite_mean([
            row["avg_frontier_obs_distance"] for row in rows
        ]),
        "mean_frontier_obs_vector_norm": _finite_mean([
            row["avg_frontier_obs_vector_norm"] for row in rows
        ]),
        "mean_frontier_local_coverage_cos": _finite_mean([
            row["avg_frontier_local_coverage_cos"] for row in rows
        ]),
        "mean_frontier_global_coverage_cos": _finite_mean([
            row["avg_frontier_global_coverage_cos"] for row in rows
        ]),
        "mean_local_global_coverage_cos": _finite_mean([
            row["avg_local_global_coverage_cos"] for row in rows
        ]),
        "mean_frontier_sector_cos": _finite_mean([
            row["avg_frontier_sector_cos"] for row in rows
        ]),
        "mean_frontier_sector_dominance": _finite_mean([
            row["avg_frontier_sector_dominance"] for row in rows
        ]),
        "mean_frontier_sector_entropy": _finite_mean([
            row["avg_frontier_sector_entropy"] for row in rows
        ]),
        "mean_frontier_cancellation": _finite_mean([
            row["avg_frontier_cancellation"] for row in rows
        ]),
        "mean_frontier_pairwise_cos": _finite_mean([
            row["avg_frontier_pairwise_cos"] for row in rows
        ]),
        "mean_frontier_pairwise_same_dir": _finite_mean([
            row["avg_frontier_pairwise_same_dir"] for row in rows
        ]),
        "mean_local_pairwise_same_dir": _finite_mean([
            row["avg_local_pairwise_same_dir"] for row in rows
        ]),
        "mean_global_pairwise_same_dir": _finite_mean([
            row["avg_global_pairwise_same_dir"] for row in rows
        ]),
        "mean_reward_uav_coverage": _finite_mean([
            row["avg_reward_uav_coverage"] for row in rows
        ]),
        "mean_reward_uav_move_coverage": _finite_mean([
            row["avg_reward_uav_move_coverage"] for row in rows
        ]),
        "mean_reward_uav_frontier": _finite_mean([
            row["avg_reward_uav_frontier"] for row in rows
        ]),
        "mean_reward_uav_confidence": _finite_mean([
            row["avg_reward_uav_confidence"] for row in rows
        ]),
        "mean_reward_uav_team_confidence": _finite_mean([
            row.get("avg_reward_uav_team_confidence", math.nan) for row in rows
        ]),
        "mean_penalty_uav_team_confidence_overlap": _finite_mean([
            row.get("avg_penalty_uav_team_confidence_overlap", math.nan) for row in rows
        ]),
        "mean_reward_uav_confidence_move": _finite_mean([
            row["avg_reward_uav_confidence_move"] for row in rows
        ]),
        "mean_reward_uav_cleanup_target_progress": _finite_mean([
            row.get("avg_reward_uav_cleanup_target_progress", math.nan) for row in rows
        ]),
        "mean_reward_uav_astar_progress": _finite_mean([
            row.get("avg_reward_uav_astar_progress", math.nan) for row in rows
        ]),
        "mean_penalty_uav_inefficient_move": _finite_mean([
            row["avg_penalty_uav_inefficient_move"] for row in rows
        ]),
        "mean_penalty_uav_confidence_overlap": _finite_mean([
            row["avg_penalty_uav_confidence_overlap"] for row in rows
        ]),
        "mean_penalty_uav_overlap": _finite_mean([
            row["avg_penalty_uav_overlap"] for row in rows
        ]),
        "mean_penalty_uav_inter_overlap": _finite_mean([
            row["avg_penalty_uav_inter_overlap"] for row in rows
        ]),
        "mean_penalty_uav_outside_footprint": _finite_mean([
            row["avg_penalty_uav_outside_footprint"] for row in rows
        ]),
        "mean_reward_uav_coverage_threshold": _finite_mean([
            row["avg_reward_uav_coverage_threshold"] for row in rows
        ]),
        "mean_reward_uav_scout": _finite_mean([
            row["avg_reward_uav_scout"] for row in rows
        ]),
        "mean_reward_team": _finite_mean([
            row["avg_reward_team"] for row in rows
        ]),
        "mean_reward_all_survivors_found": _finite_mean([
            row["avg_reward_all_survivors_found"] for row in rows
        ]),
        "mean_reward_uav_aux": _finite_mean([
            row["avg_reward_uav_aux"] for row in rows
        ]),
        "mean_frontier_abs_reward_share": _finite_mean([
            row["avg_frontier_abs_reward_share"] for row in rows
        ]),
        "mean_frontier_high_progress_step_frac": _finite_mean([
            row["frontier_high_progress_step_frac"] for row in rows
        ]),
        "mean_frontier_high_progress_no_new_frac": _finite_mean([
            row["frontier_high_progress_no_new_frac"] for row in rows
        ]),
        "mean_frontier_high_progress_edge_frac": _finite_mean([
            row["frontier_high_progress_edge_frac"] for row in rows
        ]),
        "mean_frontier_high_progress_corner_frac": _finite_mean([
            row["frontier_high_progress_corner_frac"] for row in rows
        ]),
        "mean_frontier_edge_progress": _finite_mean([
            row["frontier_edge_progress_mean"] for row in rows
        ]),
        "mean_frontier_interior_progress": _finite_mean([
            row["frontier_interior_progress_mean"] for row in rows
        ]),
        "mean_frontier_edge_reward": _finite_mean([
            row["frontier_edge_reward_mean"] for row in rows
        ]),
        "mean_frontier_interior_reward": _finite_mean([
            row["frontier_interior_reward_mean"] for row in rows
        ]),
        "mean_frontier_edge_new_cells": _finite_mean([
            row["frontier_edge_new_cells_mean"] for row in rows
        ]),
        "mean_frontier_interior_new_cells": _finite_mean([
            row["frontier_interior_new_cells_mean"] for row in rows
        ]),
        "mean_frontier_edge_outside": _finite_mean([
            row["frontier_edge_outside_mean"] for row in rows
        ]),
        "mean_frontier_interior_outside": _finite_mean([
            row["frontier_interior_outside_mean"] for row in rows
        ]),
        "mean_frontier_progress_new_cells_corr": _finite_mean([
            row["frontier_progress_new_cells_corr"] for row in rows
        ]),
        "mean_frontier_expected_raw_new_cells_corr": _finite_mean([
            row["frontier_expected_raw_new_cells_corr"] for row in rows
        ]),
        "mean_frontier_progress_boundary_distance_corr": _finite_mean([
            row["frontier_progress_boundary_distance_corr"] for row in rows
        ]),
        "mean_frontier_obs_empty_step_frac": _finite_mean([
            row["frontier_obs_empty_step_frac"] for row in rows
        ]),
        "mean_action_frontier_aligned_step_frac": _finite_mean([
            row["action_frontier_aligned_step_frac"] for row in rows
        ]),
        "mean_action_frontier_anti_aligned_step_frac": _finite_mean([
            row["action_frontier_anti_aligned_step_frac"] for row in rows
        ]),
        "mean_action_frontier_aligned_no_new_frac": _finite_mean([
            row["action_frontier_aligned_no_new_frac"] for row in rows
        ]),
        "mean_action_frontier_aligned_edge_frac": _finite_mean([
            row["action_frontier_aligned_edge_frac"] for row in rows
        ]),
        "mean_action_frontier_alignment_new_cells_corr": _finite_mean([
            row["action_frontier_alignment_new_cells_corr"] for row in rows
        ]),
        "mean_action_frontier_alignment_boundary_distance_corr": _finite_mean([
            row["action_frontier_alignment_boundary_distance_corr"] for row in rows
        ]),
        "mean_excess_overlap_step_frac_10": _finite_mean([
            row["excess_overlap_step_frac_10"] for row in rows
        ]),
        "mean_inter_uav_overlap_step_frac_20": _finite_mean([
            row["inter_uav_overlap_step_frac_20"] for row in rows
        ]),
        "mean_excess_overlap_step_frac_20": _finite_mean([
            row["excess_overlap_step_frac_20"] for row in rows
        ]),
        "mean_overlap_step_frac_60": _finite_mean([row["overlap_step_frac_60"] for row in rows]),
        "mean_new_coverage_step_frac": _finite_mean([row["new_coverage_step_frac"] for row in rows]),
        "mean_raw_new_coverage_step_frac": _finite_mean([
            row["raw_new_coverage_step_frac"] for row in rows
        ]),
        "mean_low_action_high_motion_frac": _finite_mean([
            row["low_action_high_motion_frac"] for row in rows
        ]),
        "mean_high_action_low_motion_frac": _finite_mean([
            row["high_action_low_motion_frac"] for row in rows
        ]),
        "mean_moving_no_new_coverage_frac": _finite_mean([
            row["moving_no_new_coverage_frac"] for row in rows
        ]),
        "mean_moving_no_confidence_gain_frac": _finite_mean([
            float(row.get("moving_no_confidence_gain_frac", math.nan)) for row in rows
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
        "mean_coverage_bbox_hole_fraction": _finite_mean([
            row["coverage_bbox_hole_fraction"] for row in rows
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
        "mean_coverage_uncovered_component_count": _finite_mean([
            row["coverage_uncovered_component_count"] for row in rows
        ]),
        "mean_coverage_enclosed_uncovered_component_count": _finite_mean([
            row["coverage_enclosed_uncovered_component_count"] for row in rows
        ]),
        "mean_coverage_uncovered_fraction": _finite_mean([
            row["coverage_uncovered_fraction"] for row in rows
        ]),
        "mean_coverage_enclosed_uncovered_fraction": _finite_mean([
            row["coverage_enclosed_uncovered_fraction"] for row in rows
        ]),
        "mean_coverage_largest_uncovered_component_fraction": _finite_mean([
            row["coverage_largest_uncovered_component_fraction"] for row in rows
        ]),
        "mean_coverage_largest_enclosed_hole_fraction": _finite_mean([
            row["coverage_largest_enclosed_hole_fraction"] for row in rows
        ]),
        "mean_coverage_enclosed_hole_share": _finite_mean([
            row["coverage_enclosed_hole_share"] for row in rows
        ]),
    }
    summary.update(_distribution_summary(rows))
    summary["per_drone"] = _summarize_per_drone(rows)
    summary["time_bins"] = _summarize_time_bins(rows)
    summary["perception_time_bins"] = _summarize_named_time_bins(rows, "perception_time_bins")
    summary["scout_time_bins"] = _summarize_scout_time_bins(rows)
    summary["outcome_splits"] = _summarize_outcome_splits(rows)
    summary["survivor_exposure_outcomes"] = _summarize_survivor_exposure_outcomes(rows)
    return summary


def _time_bin_value(row: dict[str, Any], bin_idx: int, key: str) -> float:
    bins = row.get("time_bins", [])
    if not bins:
        return math.nan
    idx = bin_idx if bin_idx >= 0 else len(bins) + bin_idx
    if idx < 0 or idx >= len(bins):
        return math.nan
    return float(bins[idx].get(key, math.nan))


def _summarize_outcome_splits(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    groups = [
        ("success", [row for row in rows if bool(row.get("full_success", 0.0))]),
        ("failure", [row for row in rows if not bool(row.get("full_success", 0.0))]),
    ]
    summaries: list[dict[str, float | str]] = []
    for label, entries in groups:
        summaries.append({
            "group": label,
            "episodes": float(len(entries)),
            "mean_recall": _finite_mean([float(row.get("recall", math.nan)) for row in entries]),
            "mean_confirmation_recall": _finite_mean([
                float(row.get("confirmation_recall", math.nan)) for row in entries
            ]),
            "mean_final_coverage_fraction": _finite_mean([
                float(row.get("final_coverage_fraction", math.nan)) for row in entries
            ]),
            "early_candidate_capture": _finite_mean([
                _time_bin_value(row, 0, "candidate_capture_fraction") for row in entries
            ]),
            "early_candidate_best_new_cells": _finite_mean([
                _time_bin_value(row, 0, "candidate_best_new_cells") for row in entries
            ]),
            "early_candidate_regret": _finite_mean([
                _time_bin_value(row, 0, "candidate_new_cell_regret") for row in entries
            ]),
            "early_candidate_action_rank": _finite_mean([
                _time_bin_value(row, 0, "candidate_action_rank") for row in entries
            ]),
            "early_candidate_avoidable_overlap": _finite_mean([
                _time_bin_value(row, 0, "candidate_avoidable_overlap") for row in entries
            ]),
            "mid_candidate_capture": _finite_mean([
                _time_bin_value(row, 2, "candidate_capture_fraction") for row in entries
            ]),
            "late_candidate_capture": _finite_mean([
                _time_bin_value(row, -1, "candidate_capture_fraction") for row in entries
            ]),
            "late_candidate_best_new_cells": _finite_mean([
                _time_bin_value(row, -1, "candidate_best_new_cells") for row in entries
            ]),
            "late_candidate_regret": _finite_mean([
                _time_bin_value(row, -1, "candidate_new_cell_regret") for row in entries
            ]),
            "late_candidate_action_rank": _finite_mean([
                _time_bin_value(row, -1, "candidate_action_rank") for row in entries
            ]),
            "late_candidate_avoidable_overlap": _finite_mean([
                _time_bin_value(row, -1, "candidate_avoidable_overlap") for row in entries
            ]),
            "avg_candidate_capture": _finite_mean([
                float(row.get("avg_candidate_capture_fraction", math.nan)) for row in entries
            ]),
            "avg_candidate_regret": _finite_mean([
                float(row.get("avg_candidate_new_cell_regret", math.nan)) for row in entries
            ]),
            "avg_candidate_avoidable_overlap": _finite_mean([
                float(row.get("avg_candidate_avoidable_overlap", math.nan)) for row in entries
            ]),
            "moving_no_new_coverage_frac": _finite_mean([
                float(row.get("moving_no_new_coverage_frac", math.nan)) for row in entries
            ]),
            "moving_no_confidence_gain_frac": _finite_mean([
                float(row.get("moving_no_confidence_gain_frac", math.nan)) for row in entries
            ]),
            "coverage_bbox_fill_fraction": _finite_mean([
                float(row.get("coverage_bbox_fill_fraction", math.nan)) for row in entries
            ]),
            "coverage_bbox_hole_fraction": _finite_mean([
                float(row.get("coverage_bbox_hole_fraction", math.nan)) for row in entries
            ]),
            "coverage_enclosed_uncovered_fraction": _finite_mean([
                float(row.get("coverage_enclosed_uncovered_fraction", math.nan)) for row in entries
            ]),
            "coverage_enclosed_hole_share": _finite_mean([
                float(row.get("coverage_enclosed_hole_share", math.nan)) for row in entries
            ]),
            "coverage_largest_enclosed_hole_fraction": _finite_mean([
                float(row.get("coverage_largest_enclosed_hole_fraction", math.nan)) for row in entries
            ]),
            "coverage_largest_uncovered_component_fraction": _finite_mean([
                float(row.get("coverage_largest_uncovered_component_fraction", math.nan)) for row in entries
            ]),
            "coverage_edge_bias": _finite_mean([
                float(row.get("coverage_edge_bias", math.nan)) for row in entries
            ]),
            "path_bbox_area_fraction": _finite_mean([
                float(row.get("path_bbox_area_fraction", math.nan)) for row in entries
            ]),
            "edge_step_frac": _finite_mean([
                float(row.get("edge_step_frac", math.nan)) for row in entries
            ]),
        })
    return summaries


def _summarize_per_drone(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drone_indices = sorted({
        int(drone["drone"])
        for row in rows
        for drone in row.get("per_drone", [])
    })
    metrics = (
        "scout_credit_count",
        "avg_action_norm",
        "avg_displacement_m",
        "path_length_m",
        "avg_action_displacement_alignment",
        "avg_action_frontier_alignment",
        "avg_action_frontier_intent",
        "avg_action_frontier_movement_gap",
        "avg_new_coverage_cells",
        "avg_raw_new_coverage_cells",
        "total_new_coverage_cells",
        "total_raw_new_coverage_cells",
        "new_coverage_step_frac",
        "raw_new_coverage_step_frac",
        "avg_outside_footprint_fraction",
        "avg_overlap_fraction",
        "avg_expected_overlap_fraction",
        "avg_excess_overlap_fraction",
        "avg_inter_uav_overlap_fraction",
        "avg_any_history_revisit_fraction",
        "avg_own_history_revisit_fraction",
        "avg_teammate_history_revisit_fraction",
        "avg_own_only_revisit_fraction",
        "avg_teammate_only_revisit_fraction",
        "avg_shared_history_revisit_fraction",
        "avg_unavoidable_revisit_fraction",
        "avg_avoidable_revisit_fraction",
        "avg_frontier_expected_new_cells",
        "avg_frontier_new_cell_capture_fraction",
        "avg_frontier_new_cell_gap",
        "avg_candidate_best_new_cells",
        "avg_candidate_capture_fraction",
        "avg_candidate_new_cell_regret",
        "avg_candidate_best_new_overlap",
        "avg_candidate_best_useful_overlap",
        "avg_candidate_avoidable_overlap",
        "avg_candidate_action_rank",
        "avg_candidate_movement_rank",
        "avg_candidate_action_capture_fraction",
        "avg_candidate_movement_capture_fraction",
        "avg_candidate_action_best_alignment",
        "avg_candidate_movement_best_alignment",
        "candidate_no_opportunity_frac",
        "avg_frontier_candidate_new_cells",
        "avg_frontier_candidate_capture_fraction",
        "avg_frontier_candidate_regret",
        "avg_frontier_candidate_best_alignment",
        "avg_frontier_candidate_rank",
        "avg_frontier_candidate_nearest_rank",
        "frontier_candidate_is_best_frac",
        "frontier_candidate_bad_frac",
        "avg_confidence_frontier_candidate_capture_fraction",
        "avg_confidence_frontier_candidate_best_alignment",
        "avg_confidence_frontier_candidate_rank",
        "confidence_frontier_candidate_bad_frac",
        "avg_confidence_lg_frontier_candidate_capture_fraction",
        "avg_confidence_lg_frontier_candidate_best_alignment",
        "avg_confidence_lg_frontier_candidate_rank",
        "confidence_lg_frontier_candidate_bad_frac",
        "avg_confidence_frontier_capture_advantage",
        "avg_confidence_lg_frontier_capture_advantage",
        "avg_coverage_opportunity_cells",
        "avg_coverage_opportunity_fraction",
        "avg_coverage_opportunity_available_fraction",
        "avg_confidence_gain",
        "total_confidence_gain",
        "avg_confidence_weighted_gain",
        "avg_confidence_opportunity_fraction",
        "avg_confidence_opportunity_best_gain",
        "confidence_revisit_step_frac",
        "confidence_useful_revisit_step_frac",
        "confidence_wasteful_revisit_step_frac",
        "confidence_ambiguous_revisit_step_frac",
        "confidence_revisit_useful_share",
        "confidence_revisit_wasteful_share",
        "confidence_revisit_gain_share",
        "confidence_gain_on_revisit",
        "confidence_gain_off_revisit",
        "confidence_opportunity_on_revisit",
        "confidence_opportunity_off_revisit",
        "confidence_best_gain_on_revisit",
        "confidence_best_gain_off_revisit",
        "avg_confidence_pass_probability",
        "avg_confidence_overlap_fraction",
        "avg_confidence_overlap_regret",
        "avg_cleanup_target_valid_fraction",
        "avg_cleanup_target_distance_m",
        "avg_cleanup_target_value",
        "avg_cleanup_target_progress_m",
        "avg_cleanup_target_progress_fraction",
        "cleanup_target_switch_rate",
        "cleanup_target_reached_rate",
        "avg_cleanup_target_value_decay",
        "avg_cleanup_target_age",
        "cleanup_target_no_progress_frac",
        "avg_cleanup_target_frontier_gate",
        "avg_frontier_alignment",
        "avg_frontier_progress_fraction",
        "avg_frontier_uncovered_ratio",
        "avg_frontier_obs_distance",
        "avg_frontier_obs_vector_norm",
        "avg_frontier_local_coverage_cos",
        "avg_frontier_global_coverage_cos",
        "avg_local_global_coverage_cos",
        "avg_frontier_sector_cos",
        "avg_frontier_sector_dominance",
        "avg_frontier_sector_entropy",
        "avg_frontier_cancellation",
        "avg_reward_uav_coverage",
        "avg_reward_uav_move_coverage",
        "avg_reward_uav_frontier",
        "avg_reward_uav_confidence",
        "avg_reward_uav_team_confidence",
        "avg_penalty_uav_team_confidence_overlap",
        "avg_reward_uav_confidence_move",
        "avg_reward_uav_cleanup_target_progress",
        "avg_reward_uav_astar_progress",
        "avg_penalty_uav_inefficient_move",
        "avg_penalty_uav_confidence_overlap",
        "avg_penalty_uav_overlap",
        "avg_penalty_uav_inter_overlap",
        "avg_penalty_uav_outside_footprint",
        "avg_reward_uav_coverage_threshold",
        "avg_reward_uav_scout",
        "avg_reward_team",
        "avg_reward_all_survivors_found",
        "avg_reward_uav_aux",
        "avg_frontier_abs_reward_share",
        "frontier_high_progress_step_frac",
        "frontier_high_progress_no_new_frac",
        "frontier_high_progress_edge_frac",
        "frontier_high_progress_corner_frac",
        "frontier_edge_progress_mean",
        "frontier_interior_progress_mean",
        "frontier_edge_reward_mean",
        "frontier_interior_reward_mean",
        "frontier_edge_new_cells_mean",
        "frontier_interior_new_cells_mean",
        "frontier_edge_outside_mean",
        "frontier_interior_outside_mean",
        "frontier_progress_new_cells_corr",
        "frontier_progress_boundary_distance_corr",
        "frontier_obs_empty_step_frac",
        "action_frontier_aligned_step_frac",
        "action_frontier_anti_aligned_step_frac",
        "action_frontier_aligned_no_new_frac",
        "action_frontier_aligned_edge_frac",
        "action_frontier_alignment_new_cells_corr",
        "action_frontier_alignment_boundary_distance_corr",
        "excess_overlap_step_frac_10",
        "inter_uav_overlap_step_frac_20",
        "edge_step_frac",
        "corner_step_frac",
        "stalled_step_frac",
        "longest_stall_steps",
        "moving_no_new_coverage_frac",
        "moving_no_confidence_gain_frac",
        "mean_boundary_distance_m",
        "min_boundary_distance_m",
    )
    summaries: list[dict[str, float]] = []
    for drone_idx in drone_indices:
        entries = [
            drone
            for row in rows
            for drone in row.get("per_drone", [])
            if int(drone.get("drone", -1)) == drone_idx
        ]
        summary: dict[str, Any] = {
            "drone": int(drone_idx),
            "episodes": float(len(entries)),
        }
        for metric in metrics:
            summary[f"mean_{metric}"] = _finite_mean([
                float(entry.get(metric, math.nan))
                for entry in entries
            ])
        summaries.append(summary)
    return summaries


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {
        "recall": "recall",
        "confirmation_recall": "confirmation_recall",
        "coverage": "final_coverage_fraction",
        "expected_recall": "expected_recall_from_exposure",
        "perception_gap": "perception_recall_gap",
        "scout_detect_prob": "avg_scout_detection_probability",
        "scout_norm": "avg_scout_detection_norm_distance",
        "scout_conf_pre": "avg_scout_confidence_pre",
        "scout_conf_post": "avg_scout_confidence_post",
        "missed_cum_prob": "avg_missed_cum_detection_probability",
        "missed_best_prob": "avg_missed_best_detection_probability",
        "missed_best_norm": "avg_missed_best_norm_distance",
        "missed_never_exposed": "missed_never_exposed_fraction",
        "missed_high_prob": "missed_high_cum_probability_fraction",
        "missed_edge_limited": "missed_edge_limited_fraction",
        "move_m": "avg_displacement_m",
        "new_cells": "avg_new_coverage_cells",
        "raw_new_cells": "avg_raw_new_coverage_cells",
        "outside": "avg_outside_footprint_fraction",
        "overlap": "avg_overlap_fraction",
        "expected_overlap": "avg_expected_overlap_fraction",
        "excess_overlap": "avg_excess_overlap_fraction",
        "inter_uav_overlap": "avg_inter_uav_overlap_fraction",
        "any_revisit": "avg_any_history_revisit_fraction",
        "own_revisit": "avg_own_history_revisit_fraction",
        "team_revisit": "avg_teammate_history_revisit_fraction",
        "own_only_revisit": "avg_own_only_revisit_fraction",
        "team_only_revisit": "avg_teammate_only_revisit_fraction",
        "shared_revisit": "avg_shared_history_revisit_fraction",
        "unavoidable_revisit": "avg_unavoidable_revisit_fraction",
        "avoidable_revisit": "avg_avoidable_revisit_fraction",
        "frontier_expected_new": "avg_frontier_expected_new_cells",
        "frontier_new_capture": "avg_frontier_new_cell_capture_fraction",
        "frontier_new_gap": "avg_frontier_new_cell_gap",
        "candidate_best_new": "avg_candidate_best_new_cells",
        "candidate_capture": "avg_candidate_capture_fraction",
        "candidate_regret": "avg_candidate_new_cell_regret",
        "candidate_best_overlap": "avg_candidate_best_new_overlap",
        "candidate_useful_overlap": "avg_candidate_best_useful_overlap",
        "candidate_avoidable": "avg_candidate_avoidable_overlap",
        "candidate_action_rank": "avg_candidate_action_rank",
        "candidate_movement_rank": "avg_candidate_movement_rank",
        "candidate_action_capture": "avg_candidate_action_capture_fraction",
        "candidate_movement_capture": "avg_candidate_movement_capture_fraction",
        "candidate_no_opportunity": "candidate_no_opportunity_frac",
        "frontier_candidate_capture": "avg_frontier_candidate_capture_fraction",
        "frontier_candidate_rank": "avg_frontier_candidate_rank",
        "frontier_candidate_align": "avg_frontier_candidate_best_alignment",
        "frontier_candidate_bad": "frontier_candidate_bad_frac",
        "confidence_frontier_capture": "avg_confidence_frontier_candidate_capture_fraction",
        "confidence_frontier_rank": "avg_confidence_frontier_candidate_rank",
        "confidence_frontier_bad": "confidence_frontier_candidate_bad_frac",
        "confidence_lg_frontier_capture": "avg_confidence_lg_frontier_candidate_capture_fraction",
        "confidence_lg_frontier_rank": "avg_confidence_lg_frontier_candidate_rank",
        "confidence_lg_frontier_bad": "confidence_lg_frontier_candidate_bad_frac",
        "confidence_frontier_advantage": "avg_confidence_frontier_capture_advantage",
        "confidence_lg_frontier_advantage": "avg_confidence_lg_frontier_capture_advantage",
        "coverage_opportunity_cells": "avg_coverage_opportunity_cells",
        "coverage_opportunity": "avg_coverage_opportunity_fraction",
        "coverage_opportunity_available": "avg_coverage_opportunity_available_fraction",
        "confidence_final": "final_confidence_mean",
        "confidence_gain": "avg_confidence_gain",
        "confidence_drone_gain": "avg_confidence_gain_by_drone",
        "confidence_weighted_gain": "avg_confidence_weighted_gain",
        "confidence_opportunity": "avg_confidence_opportunity_fraction",
        "confidence_best_gain": "avg_confidence_opportunity_best_gain",
        "confidence_revisit": "confidence_revisit_step_frac",
        "confidence_useful_revisit": "confidence_useful_revisit_step_frac",
        "confidence_wasteful_revisit": "confidence_wasteful_revisit_step_frac",
        "confidence_revisit_useful_share": "confidence_revisit_useful_share",
        "confidence_revisit_gain_share": "confidence_revisit_gain_share",
        "confidence_pass": "avg_confidence_pass_probability",
        "confidence_overlap": "avg_confidence_overlap_fraction",
        "confidence_overlap_regret": "avg_confidence_overlap_regret",
        "frontier_align": "avg_frontier_alignment",
        "frontier_progress": "avg_frontier_progress_fraction",
        "frontier_ratio": "avg_frontier_uncovered_ratio",
        "frontier_obs_dist": "avg_frontier_obs_distance",
        "frontier_local": "avg_frontier_local_coverage_cos",
        "frontier_global": "avg_frontier_global_coverage_cos",
        "frontier_sector": "avg_frontier_sector_cos",
        "sector_dom": "avg_frontier_sector_dominance",
        "sector_entropy": "avg_frontier_sector_entropy",
        "frontier_cancel": "avg_frontier_cancellation",
        "frontier_pair": "avg_frontier_pairwise_cos",
        "frontier_pair_same": "avg_frontier_pairwise_same_dir",
        "action_frontier": "avg_action_frontier_alignment",
        "action_frontier_intent": "avg_action_frontier_intent",
        "action_frontier_gap": "avg_action_frontier_movement_gap",
        "action_frontier_aligned": "action_frontier_aligned_step_frac",
        "action_frontier_anti": "action_frontier_anti_aligned_step_frac",
        "action_frontier_no_new": "action_frontier_aligned_no_new_frac",
        "frontier_reward": "avg_reward_uav_frontier",
        "confidence_move_reward": "avg_reward_uav_confidence_move",
        "team_confidence_overlap_penalty": "avg_penalty_uav_team_confidence_overlap",
        "cleanup_target_reward": "avg_reward_uav_cleanup_target_progress",
        "astar_progress_reward": "avg_reward_uav_astar_progress",
        "cleanup_target_gate": "avg_cleanup_target_frontier_gate",
        "confidence_overlap_penalty": "avg_penalty_uav_confidence_overlap",
        "frontier_share": "avg_frontier_abs_reward_share",
        "frontier_high": "frontier_high_progress_step_frac",
        "frontier_high_no_new": "frontier_high_progress_no_new_frac",
        "frontier_high_edge": "frontier_high_progress_edge_frac",
        "frontier_new_corr": "frontier_progress_new_cells_corr",
        "edge_frac": "edge_step_frac",
        "corner_frac": "corner_step_frac",
        "bbox_fill": "coverage_bbox_fill_fraction",
        "bbox_hole": "coverage_bbox_hole_fraction",
        "enclosed_hole": "coverage_enclosed_uncovered_fraction",
        "enclosed_hole_share": "coverage_enclosed_hole_share",
        "largest_hole": "coverage_largest_enclosed_hole_fraction",
        "largest_uncovered": "coverage_largest_uncovered_component_fraction",
        "center_cov": "coverage_center_fraction",
        "moving_no_new": "moving_no_new_coverage_frac",
        "moving_no_conf_gain": "moving_no_confidence_gain_frac",
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


def _format_per_drone_row(drones: list[dict[str, Any]]) -> str:
    parts = []
    for drone in drones:
        parts.append(
            f"d{int(drone['drone'])}:"
            f"scout={int(drone['scout_credit_count'])} "
            f"path={drone['path_length_m']:.0f}m "
            f"move={drone['avg_displacement_m']:.1f}m "
            f"new={drone['avg_new_coverage_cells']:.1f} "
            f"raw={drone['avg_raw_new_coverage_cells']:.1f} "
            f"opp={drone['avg_coverage_opportunity_fraction']:.2f} "
            f"opp_avail={drone['avg_coverage_opportunity_available_fraction']:.2f} "
            f"conf_gain={drone['avg_confidence_gain']:.5f} "
            f"conf_rev={drone['confidence_revisit_step_frac']:.2f}/"
            f"{drone['confidence_useful_revisit_step_frac']:.2f}/"
            f"{drone['confidence_wasteful_revisit_step_frac']:.2f} "
            f"conf_ov={drone['avg_confidence_overlap_fraction']:.2f} "
            f"pass_p={drone['avg_confidence_pass_probability']:.2f} "
            f"edge={drone['edge_step_frac']:.2f} "
            f"corner={drone['corner_step_frac']:.2f} "
            f"excess={drone['avg_excess_overlap_fraction']:.2f} "
            f"own_rev={drone['avg_own_history_revisit_fraction']:.2f} "
            f"team_rev={drone['avg_teammate_history_revisit_fraction']:.2f} "
            f"avoid_rev={drone['avg_avoidable_revisit_fraction']:.2f} "
            f"inter={drone['avg_inter_uav_overlap_fraction']:.2f} "
            f"front_exp={drone['avg_frontier_expected_new_cells']:.1f} "
            f"front_cap={drone['avg_frontier_new_cell_capture_fraction']:.2f} "
            f"cand_best={drone['avg_candidate_best_new_cells']:.1f} "
            f"cand_cap={drone['avg_candidate_capture_fraction']:.2f} "
            f"cand_rank={drone['avg_candidate_action_rank']:.1f}/"
            f"{drone['avg_candidate_movement_rank']:.1f} "
            f"front_use={drone['avg_frontier_candidate_capture_fraction']:.2f}/"
            f"{drone['avg_frontier_candidate_rank']:.1f}/"
            f"{drone['avg_frontier_candidate_best_alignment']:.2f} "
            f"conf_front={drone['avg_confidence_frontier_candidate_capture_fraction']:.2f} "
            f"conf_lg={drone['avg_confidence_lg_frontier_candidate_capture_fraction']:.2f} "
            f"cand_avoid={drone['avg_candidate_avoidable_overlap']:.2f} "
            f"front={drone['avg_frontier_alignment']:.2f}/"
            f"{drone['avg_frontier_progress_fraction']:.2f}/"
            f"{drone['avg_frontier_uncovered_ratio']:.2f} "
            f"act_front={drone['avg_action_frontier_alignment']:.2f}/"
            f"{drone['action_frontier_aligned_step_frac']:.2f} "
            f"fhi={drone['frontier_high_progress_step_frac']:.2f}/"
            f"{drone['frontier_high_progress_no_new_frac']:.2f}/"
            f"{drone['frontier_high_progress_edge_frac']:.2f} "
            f"move_no_conf={drone.get('moving_no_confidence_gain_frac', math.nan):.2f} "
            f"stall={drone['stalled_step_frac']:.2f}"
        )
    return "; ".join(parts)


def _format_per_drone_summary(drones: list[dict[str, Any]]) -> list[str]:
    lines = []
    for drone in drones:
        lines.append(
            f"  d{int(drone['drone'])}: "
            f"scouts={drone['mean_scout_credit_count']:.2f} "
            f"path={drone['mean_path_length_m']:.1f}m "
            f"move={drone['mean_avg_displacement_m']:.2f}m "
            f"new_cells={drone['mean_avg_new_coverage_cells']:.1f} "
            f"raw_new={drone['mean_avg_raw_new_coverage_cells']:.1f} "
            f"opp={drone['mean_avg_coverage_opportunity_fraction']:.3f} "
            f"opp_avail={drone['mean_avg_coverage_opportunity_available_fraction']:.3f} "
            f"conf_gain={drone['mean_avg_confidence_gain']:.5f} "
            f"conf_opp={drone['mean_avg_confidence_opportunity_fraction']:.3f} "
            f"conf_rev={drone['mean_confidence_revisit_step_frac']:.3f}/"
            f"{drone['mean_confidence_useful_revisit_step_frac']:.3f}/"
            f"{drone['mean_confidence_wasteful_revisit_step_frac']:.3f} "
            f"conf_rev_share={drone['mean_confidence_revisit_useful_share']:.3f} "
            f"conf_ov={drone['mean_avg_confidence_overlap_fraction']:.3f} "
            f"pass_p={drone['mean_avg_confidence_pass_probability']:.3f} "
            f"edge={drone['mean_edge_step_frac']:.3f} "
            f"corner={drone['mean_corner_step_frac']:.3f} "
            f"outside={drone['mean_avg_outside_footprint_fraction']:.3f} "
            f"excess={drone['mean_avg_excess_overlap_fraction']:.3f} "
            f"own_rev={drone['mean_avg_own_history_revisit_fraction']:.3f} "
            f"team_rev={drone['mean_avg_teammate_history_revisit_fraction']:.3f} "
            f"avoid_rev={drone['mean_avg_avoidable_revisit_fraction']:.3f} "
            f"inter={drone['mean_avg_inter_uav_overlap_fraction']:.3f} "
            f"front_exp={drone['mean_avg_frontier_expected_new_cells']:.1f} "
            f"front_cap={drone['mean_avg_frontier_new_cell_capture_fraction']:.3f} "
            f"cand_best={drone['mean_avg_candidate_best_new_cells']:.1f} "
            f"cand_cap={drone['mean_avg_candidate_capture_fraction']:.3f} "
            f"cand_reg={drone['mean_avg_candidate_new_cell_regret']:.1f} "
            f"cand_rank={drone['mean_avg_candidate_action_rank']:.2f}/"
            f"{drone['mean_avg_candidate_movement_rank']:.2f} "
            f"front_use={drone['mean_avg_frontier_candidate_capture_fraction']:.3f}/"
            f"{drone['mean_avg_frontier_candidate_rank']:.2f}/"
            f"{drone['mean_avg_frontier_candidate_best_alignment']:.3f} "
            f"conf_front={drone['mean_avg_confidence_frontier_candidate_capture_fraction']:.3f} "
            f"conf_lg={drone['mean_avg_confidence_lg_frontier_candidate_capture_fraction']:.3f} "
            f"cand_avoid={drone['mean_avg_candidate_avoidable_overlap']:.3f} "
            f"front={drone['mean_avg_frontier_alignment']:.3f}/"
            f"{drone['mean_avg_frontier_progress_fraction']:.3f}/"
            f"{drone['mean_avg_frontier_uncovered_ratio']:.3f} "
            f"act_front={drone['mean_avg_action_frontier_alignment']:.3f}/"
            f"{drone['mean_action_frontier_aligned_step_frac']:.3f} "
            f"front_rew={drone['mean_avg_reward_uav_frontier']:.4f} "
            f"conf_move={drone['mean_avg_reward_uav_confidence_move']:.4f} "
            f"cleanup_rew={drone['mean_avg_reward_uav_cleanup_target_progress']:.4f} "
            f"astar_rew={drone['mean_avg_reward_uav_astar_progress']:.4f} "
            f"cleanup_gate={drone['mean_avg_cleanup_target_frontier_gate']:.3f} "
            f"front_hi={drone['mean_frontier_high_progress_step_frac']:.3f}/"
            f"{drone['mean_frontier_high_progress_no_new_frac']:.3f}/"
            f"{drone['mean_frontier_high_progress_edge_frac']:.3f} "
            f"move_no_conf={drone['mean_moving_no_confidence_gain_frac']:.3f} "
            f"stall={drone['mean_stalled_step_frac']:.3f}"
        )
    return lines


def _format_time_bin_summary(time_bins: list[dict[str, float]]) -> list[str]:
    lines = []
    for item in time_bins:
        start = 100.0 * float(item.get("start_fraction", 0.0))
        end = 100.0 * float(item.get("end_fraction", 0.0))
        lines.append(
            f"  {start:>3.0f}-{end:<3.0f}%: "
            f"cov={item.get('coverage_fraction', math.nan):.3f} "
            f"move={item.get('displacement_m', math.nan):.2f}m "
            f"new={item.get('new_coverage_cells', math.nan):.1f} "
            f"raw={item.get('raw_new_coverage_cells', math.nan):.1f} "
            f"opp={item.get('coverage_opportunity_fraction', math.nan):.3f} "
            f"opp_avail={item.get('coverage_opportunity_available_fraction', math.nan):.3f} "
            f"conf_opp={item.get('confidence_opportunity_fraction', math.nan):.3f} "
            f"conf_rev={item.get('confidence_revisit', math.nan):.3f}/"
            f"{item.get('confidence_useful_revisit', math.nan):.3f}/"
            f"{item.get('confidence_wasteful_revisit', math.nan):.3f} "
            f"edge={item.get('edge_step', math.nan):.3f} "
            f"overlap={item.get('overlap', math.nan):.3f} "
            f"excess={item.get('excess_overlap', math.nan):.3f} "
            f"own_rev={item.get('own_history_revisit', math.nan):.3f} "
            f"team_rev={item.get('teammate_history_revisit', math.nan):.3f} "
            f"avoid_rev={item.get('avoidable_revisit', math.nan):.3f} "
            f"outside={item.get('outside_footprint', math.nan):.3f} "
            f"front_exp={item.get('frontier_expected_new_cells', math.nan):.1f} "
            f"front_cap={item.get('frontier_new_cell_capture', math.nan):.3f} "
            f"cand_best={item.get('candidate_best_new_cells', math.nan):.1f} "
            f"cand_cap={item.get('candidate_capture_fraction', math.nan):.3f} "
            f"cand_reg={item.get('candidate_new_cell_regret', math.nan):.1f} "
            f"cand_rank={item.get('candidate_action_rank', math.nan):.2f}/"
            f"{item.get('candidate_movement_rank', math.nan):.2f} "
            f"front_use={item.get('frontier_candidate_capture_fraction', math.nan):.3f}/"
            f"{item.get('frontier_candidate_rank', math.nan):.2f}/"
            f"{item.get('frontier_candidate_best_alignment', math.nan):.3f} "
            f"conf_front={item.get('confidence_frontier_candidate_capture_fraction', math.nan):.3f} "
            f"conf_lg={item.get('confidence_lg_frontier_candidate_capture_fraction', math.nan):.3f} "
            f"cand_avoid={item.get('candidate_avoidable_overlap', math.nan):.3f} "
            f"moving_nonew={item.get('moving_no_new_coverage', math.nan):.3f} "
            f"moving_noconf={item.get('moving_no_confidence_gain', math.nan):.3f} "
            f"obs_dist={item.get('frontier_obs_distance', math.nan):.3f} "
            f"f_loc={item.get('frontier_local_coverage_cos', math.nan):.3f} "
            f"f_glob={item.get('frontier_global_coverage_cos', math.nan):.3f} "
            f"sect_dom={item.get('frontier_sector_dominance', math.nan):.3f} "
            f"centroid={item.get('frontier_cancellation', math.nan):.3f} "
            f"pair_same={item.get('frontier_pairwise_same_dir', math.nan):.3f} "
            f"act_front={item.get('action_frontier_alignment', math.nan):.3f} "
            f"move_front={item.get('movement_frontier_alignment', math.nan):.3f} "
            f"front_prog={item.get('frontier_progress', math.nan):.3f} "
            f"front_score={item.get('frontier_uncovered_ratio', math.nan):.3f} "
            f"front_rew={item.get('frontier_reward', math.nan):.4f} "
            f"conf_move={item.get('confidence_move_reward', math.nan):.4f} "
            f"move_pen={item.get('inefficient_move_penalty', math.nan):.4f} "
            f"conf_ov={item.get('confidence_overlap_fraction', math.nan):.3f} "
            f"team_conf_ov_pen={item.get('team_confidence_overlap_penalty', math.nan):.4f} "
            f"conf_ov_pen={item.get('confidence_overlap_penalty', math.nan):.4f}"
        )
    return lines


def _format_perception_time_bin_summary(time_bins: list[dict[str, float]]) -> list[str]:
    lines = []
    for item in time_bins:
        start = 100.0 * float(item.get("start_fraction", 0.0))
        end = 100.0 * float(item.get("end_fraction", 0.0))
        lines.append(
            f"  {start:>3.0f}-{end:<3.0f}%: "
            f"unscouted={item.get('unscouted_survivors', math.nan):.2f} "
            f"exposed={item.get('exposed_unscouted_survivors', math.nan):.2f} "
            f"exposed_frac={item.get('exposed_unscouted_fraction', math.nan):.3f} "
            f"exp_scout={item.get('expected_scouts', math.nan):.3f} "
            f"actual_scout={item.get('actual_new_scouts', math.nan):.3f} "
            f"miss_p={item.get('missed_detection_probability_mass', math.nan):.3f} "
            f"mean_p={item.get('mean_exposed_detection_probability', math.nan):.3f} "
            f"max_p={item.get('max_exposed_detection_probability', math.nan):.3f} "
            f"norm={item.get('mean_exposed_norm_distance', math.nan):.3f} "
            f"edge={item.get('near_edge_exposure_fraction', math.nan):.3f} "
            f"central={item.get('central_exposure_fraction', math.nan):.3f}"
        )
    return lines


def _format_scout_time_bin_summary(time_bins: list[dict[str, float]]) -> list[str]:
    lines = []
    for item in time_bins:
        start = 100.0 * float(item.get("start_fraction", 0.0))
        end = 100.0 * float(item.get("end_fraction", 0.0))
        lines.append(
            f"  {start:>3.0f}-{end:<3.0f}%: "
            f"new_mean={item.get('mean_new_recall', math.nan):.3f} "
            f"new_med={item.get('median_new_recall', math.nan):.3f} "
            f"cum_mean={item.get('mean_cumulative_recall', math.nan):.3f} "
            f"cum_med={item.get('median_cumulative_recall', math.nan):.3f}"
        )
    return lines


def _format_outcome_split_summary(splits: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in splits:
        lines.append(
            f"  {item.get('group', '?')}: "
            f"n={item.get('episodes', math.nan):.0f} "
            f"recall={item.get('mean_recall', math.nan):.3f} "
            f"cov={item.get('mean_final_coverage_fraction', math.nan):.3f} "
            f"cand_cap early/mid/late="
            f"{item.get('early_candidate_capture', math.nan):.3f}/"
            f"{item.get('mid_candidate_capture', math.nan):.3f}/"
            f"{item.get('late_candidate_capture', math.nan):.3f} "
            f"cand_reg early/late="
            f"{item.get('early_candidate_regret', math.nan):.1f}/"
            f"{item.get('late_candidate_regret', math.nan):.1f} "
            f"rank early/late="
            f"{item.get('early_candidate_action_rank', math.nan):.2f}/"
            f"{item.get('late_candidate_action_rank', math.nan):.2f} "
            f"bbox_fill={item.get('coverage_bbox_fill_fraction', math.nan):.3f} "
            f"bbox_hole={item.get('coverage_bbox_hole_fraction', math.nan):.3f} "
            f"enclosed={item.get('coverage_enclosed_uncovered_fraction', math.nan):.3f} "
            f"hole_share={item.get('coverage_enclosed_hole_share', math.nan):.3f} "
            f"largest_hole={item.get('coverage_largest_enclosed_hole_fraction', math.nan):.3f} "
            f"edge_bias={item.get('coverage_edge_bias', math.nan):.3f}"
        )
    return lines


def _format_survivor_exposure_outcomes(outcomes: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in outcomes:
        lines.append(
            f"  {item.get('group', '?')}: "
            f"n={item.get('survivors', math.nan):.0f} "
            f"exp_steps={item.get('mean_exposure_steps', math.nan):.1f} "
            f"pair_exp={item.get('mean_pair_exposures', math.nan):.1f} "
            f"cum_p={item.get('mean_cumulative_detection_probability', math.nan):.3f} "
            f"final_conf={item.get('mean_final_confidence', math.nan):.3f} "
            f"best_p={item.get('mean_best_detection_probability', math.nan):.3f} "
            f"best_pair_p={item.get('mean_best_pair_detection_probability', math.nan):.3f} "
            f"best_norm={item.get('mean_best_norm_distance', math.nan):.3f} "
            f"min_dist={item.get('mean_min_distance_m', math.nan):.1f}m "
            f"margin={item.get('mean_best_margin_m', math.nan):.1f}m "
            f"edge_frac={item.get('mean_near_edge_exposure_fraction', math.nan):.3f} "
            f"central_frac={item.get('mean_central_exposure_fraction', math.nan):.3f} "
            f"scout_p={item.get('mean_scout_probability', math.nan):.3f} "
            f"scout_C={item.get('mean_scout_confidence_pre', math.nan):.3f}/"
            f"{item.get('mean_scout_confidence_post', math.nan):.3f} "
            f"scout_norm={item.get('mean_scout_norm_distance', math.nan):.3f}"
        )
    return lines


def _plot_per_drone_bars(
    ax: Any,
    summary: dict[str, Any],
    key: str,
    title: str,
    ylabel: str,
) -> None:
    per_drone = summary.get("per_drone", [])
    if not per_drone:
        ax.text(0.5, 0.5, "no drones", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    labels = [f"d{int(row['drone'])}" for row in per_drone]
    values = [float(row.get(key, math.nan)) for row in per_drone]
    x = np.arange(len(labels))
    ax.bar(x, values, color="#36a269", alpha=0.85)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def _plot_time_bins(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Frontier (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("action", "action_frontier_alignment", "#4f7cff"),
        ("movement", "movement_frontier_alignment", "#36a269"),
        ("progress", "frontier_progress", "#d44a3a"),
        ("edge", "edge_step", "#20242c"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(centers, values, marker="o", linewidth=1.5, label=label, color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_title("Time-Bin Frontier (mean)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _plot_time_bins_coverage_signals(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Coverage Signals (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("front/local", "frontier_local_coverage_cos", "#4f7cff"),
        ("front/global", "frontier_global_coverage_cos", "#36a269"),
        ("sector dom", "frontier_sector_dominance", "#d44a3a"),
        ("same dir", "frontier_pairwise_same_dir", "#20242c"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(centers, values, marker="o", linewidth=1.5, label=label, color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_title("Time-Bin Coverage Signals (mean)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _plot_time_bins_search_efficiency(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Search Efficiency (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    new_cells = [float(row.get("new_coverage_cells", math.nan)) for row in time_bins]
    line_new = ax.plot(
        centers,
        new_cells,
        marker="o",
        linewidth=1.7,
        label="new cells",
        color="#4f7cff",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("new cells / drone-step")
    ax.grid(alpha=0.25)

    ax_frac = ax.twinx()
    frac_series = [
        ("overlap", "overlap", "#36a269"),
        ("excess", "excess_overlap", "#d44a3a"),
        ("edge", "edge_step", "#20242c"),
        ("moving no new", "moving_no_new_coverage", "#8a5cf6"),
        ("move no conf", "moving_no_confidence_gain", "#a855f7"),
        ("conf sat", "confidence_overlap_fraction", "#be185d"),
        ("conf regret", "confidence_overlap_regret", "#f43f5e"),
    ]
    frac_lines = []
    for label, key, color in frac_series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        frac_lines.extend(
            ax_frac.plot(
                centers,
                values,
                marker="o",
                linewidth=1.2,
                label=label,
                color=color,
                alpha=0.9,
            )
        )
    ax_frac.set_ylim(0.0, 1.0)
    ax_frac.set_ylabel("fraction")

    lines = line_new + frac_lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, frameon=False)
    ax.set_title("Time-Bin Search Efficiency (mean)", fontsize=10)


def _plot_time_bins_reward_scale(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Reward Scale (mean abs)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("coverage", "coverage_reward", "#4f7cff", False),
        ("move cov", "move_coverage_reward", "#36a269", False),
        ("frontier", "frontier_reward", "#8a5cf6", False),
        ("conf", "confidence_reward", "#0f766e", False),
        ("team conf", "team_confidence_reward", "#0891b2", False),
        ("team conf ov", "team_confidence_overlap_penalty", "#db2777", True),
        ("conf move", "confidence_move_reward", "#14b8a6", False),
        ("cleanup", "cleanup_target_progress_reward", "#f97316", False),
        ("astar", "astar_progress_reward", "#fb7185", False),
        ("move pen", "inefficient_move_penalty", "#7c3aed", True),
        ("conf ov pen", "confidence_overlap_penalty", "#be185d", True),
        ("coverage95", "coverage_threshold_reward", "#0f9d58", False),
        ("scout", "scout_reward", "#d4a72c", False),
        ("team", "team_reward", "#2d8cff", False),
        ("all found", "all_survivors_reward", "#18a999", False),
        ("overlap pen", "overlap_penalty", "#d44a3a", True),
        ("inter pen", "inter_uav_overlap_penalty", "#e07b39", True),
        ("outside pen", "outside_footprint_penalty", "#20242c", True),
    ]
    max_value = 0.0
    for label, key, color, is_penalty in series:
        values = [
            abs(float(row.get(key, math.nan)))
            if is_penalty
            else float(row.get(key, math.nan))
            for row in time_bins
        ]
        finite_values = [value for value in values if math.isfinite(value)]
        if finite_values:
            max_value = max(max_value, max(finite_values))
        linestyle = "--" if is_penalty else "-"
        ax.plot(
            centers,
            values,
            marker="o",
            linewidth=1.3,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    ax.set_xlim(0.0, 1.0)
    if max_value > 0.0:
        ax.set_ylim(0.0, max_value * 1.15)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("abs reward/penalty / drone-step")
    ax.set_title("Time-Bin Reward Scale (mean abs)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)


def _plot_time_bins_coverage_opportunity(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Coverage Opportunity (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    cells = [float(row.get("coverage_opportunity_cells", math.nan)) for row in time_bins]
    line_cells = ax.plot(
        centers,
        cells,
        marker="o",
        linewidth=1.6,
        label="opportunity cells",
        color="#4f7cff",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("reachable uncovered cells")
    ax.grid(alpha=0.25)

    ax_frac = ax.twinx()
    fraction = [
        float(row.get("coverage_opportunity_fraction", math.nan))
        for row in time_bins
    ]
    line_fraction = ax_frac.plot(
        centers,
        fraction,
        marker="o",
        linewidth=1.3,
        label="capture fraction",
        color="#d44a3a",
    )
    available = [
        float(row.get("coverage_opportunity_available_fraction", math.nan))
        for row in time_bins
    ]
    line_available = ax_frac.plot(
        centers,
        available,
        marker="o",
        linewidth=1.3,
        label="available fraction",
        color="#36a269",
    )
    ax_frac.set_ylim(0.0, 1.0)
    ax_frac.set_ylabel("fraction")

    lines = line_cells + line_fraction + line_available
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, frameon=False)
    ax.set_title("Time-Bin Coverage Opportunity (mean)", fontsize=10)


def _plot_time_bins_counterfactual(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Counterfactual Moves (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    best_new = [float(row.get("candidate_best_new_cells", math.nan)) for row in time_bins]
    actual_new = [float(row.get("raw_new_coverage_cells", math.nan)) for row in time_bins]
    line_best = ax.plot(
        centers,
        best_new,
        marker="o",
        linewidth=1.6,
        label="best new",
        color="#4f7cff",
    )
    line_actual = ax.plot(
        centers,
        actual_new,
        marker="o",
        linewidth=1.4,
        label="actual new",
        color="#36a269",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("cells / drone-step")
    ax.grid(alpha=0.25)

    ax_frac = ax.twinx()
    frac_series = [
        ("capture", "candidate_capture_fraction", "#d44a3a"),
        ("action rank / 9", "candidate_action_rank", "#8a5cf6"),
        ("cand avoid", "candidate_avoidable_overlap", "#20242c"),
    ]
    frac_lines = []
    for label, key, color in frac_series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        if key == "candidate_action_rank":
            values = [value / 9.0 if math.isfinite(value) else math.nan for value in values]
        frac_lines.extend(
            ax_frac.plot(
                centers,
                values,
                marker="o",
                linewidth=1.2,
                label=label,
                color=color,
            )
        )
    ax_frac.set_ylim(0.0, 1.0)
    ax_frac.set_ylabel("fraction")

    lines = line_best + line_actual + frac_lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, frameon=False)
    ax.set_title("Time-Bin Counterfactual Moves (mean)", fontsize=10)


def _plot_time_bins_frontier_usefulness(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Frontier Usefulness (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("current cap", "frontier_candidate_capture_fraction", "#4f7cff"),
        ("conf same cap", "confidence_frontier_candidate_capture_fraction", "#14b8a6"),
        ("conf lg cap", "confidence_lg_frontier_candidate_capture_fraction", "#36a269"),
        ("current align", "frontier_candidate_best_alignment", "#8a5cf6"),
        ("current bad", "frontier_candidate_bad", "#d44a3a"),
        ("rank / 9", "frontier_candidate_rank", "#20242c"),
    ]
    lines = []
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        if key == "frontier_candidate_rank":
            values = [value / 9.0 if math.isfinite(value) else math.nan for value in values]
        lines.extend(
            ax.plot(
                centers,
                values,
                marker="o",
                linewidth=1.25,
                label=label,
                color=color,
            )
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.55)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction / cosine / scaled rank")
    ax.grid(alpha=0.25)

    ax_cells = ax.twinx()
    new_cells = [float(row.get("new_coverage_cells", math.nan)) for row in time_bins]
    line_new = ax_cells.plot(
        centers,
        new_cells,
        marker="o",
        linewidth=1.2,
        linestyle="--",
        label="actual new",
        color="#f97316",
    )
    finite_new = [value for value in new_cells if math.isfinite(value)]
    if finite_new:
        ax_cells.set_ylim(0.0, max(max(finite_new), 1.0) * 1.2)
    ax_cells.set_ylabel("new cells / drone-step")
    lines += line_new
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, frameon=False, ncol=2)
    ax.set_title("Time-Bin Frontier Usefulness (mean)", fontsize=10)


def _plot_time_bins_revisit_sources(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Revisit Sources (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("own only", "own_only_revisit", "#4f7cff"),
        ("teammate only", "teammate_only_revisit", "#36a269"),
        ("shared old", "shared_history_revisit", "#8a5cf6"),
        ("avoidable", "avoidable_revisit", "#d44a3a"),
        ("conf useful", "confidence_useful_revisit", "#14b8a6"),
        ("conf wasteful", "confidence_wasteful_revisit", "#f97316"),
        ("capture", "frontier_new_cell_capture", "#20242c"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(
            centers,
            values,
            marker="o",
            linewidth=1.3,
            label=label,
            color=color,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction")
    ax.set_title("Time-Bin Revisit Sources (mean)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)


def _plot_outcome_candidate_splits(ax: Any, summary: dict[str, Any]) -> None:
    splits = summary.get("outcome_splits", [])
    if not splits:
        ax.text(0.5, 0.5, "no outcome splits", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Outcome Candidate Capture", fontsize=10)
        return
    labels = [str(row.get("group", "?")) for row in splits]
    x = np.arange(len(labels))
    width = 0.24
    series = [
        ("early", "early_candidate_capture", "#4f7cff"),
        ("mid", "mid_candidate_capture", "#36a269"),
        ("late", "late_candidate_capture", "#d44a3a"),
    ]
    for idx, (label, key, color) in enumerate(series):
        values = [float(row.get(key, math.nan)) for row in splits]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("actual / best candidate")
    ax.set_title("Outcome Candidate Capture", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _plot_outcome_structure_splits(ax: Any, summary: dict[str, Any]) -> None:
    splits = summary.get("outcome_splits", [])
    if not splits:
        ax.text(0.5, 0.5, "no outcome splits", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Outcome Coverage Structure", fontsize=10)
        return
    labels = [str(row.get("group", "?")) for row in splits]
    x = np.arange(len(labels))
    width = 0.20
    series = [
        ("bbox fill", "coverage_bbox_fill_fraction", "#4f7cff"),
        ("bbox holes", "coverage_bbox_hole_fraction", "#d44a3a"),
        ("enclosed", "coverage_enclosed_uncovered_fraction", "#8a5cf6"),
        ("edge bias", "coverage_edge_bias", "#20242c"),
    ]
    for idx, (label, key, color) in enumerate(series):
        values = [float(row.get(key, math.nan)) for row in splits]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            label=label,
            color=color,
            alpha=0.85,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.25, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Outcome Coverage Structure", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)


def _plot_perception_time_bins(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("perception_time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no perception bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Survivor Exposure", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("exposed", "exposed_unscouted_fraction", "#4f7cff"),
        ("expected recall", "expected_scout_recall", "#36a269"),
        ("actual recall", "actual_new_scout_recall", "#d44a3a"),
        ("miss prob mass", "missed_detection_recall_mass", "#20242c"),
        ("edge exposed", "near_edge_exposure_fraction", "#8a5cf6"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(centers, values, marker="o", linewidth=1.3, label=label, color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction / recall mass")
    ax.set_title("Time-Bin Survivor Exposure", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)


def _plot_perception_probability_bins(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("perception_time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no perception bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Detection Probability", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("mean p exposed", "mean_exposed_detection_probability", "#4f7cff"),
        ("max p exposed", "max_exposed_detection_probability", "#36a269"),
        ("mean norm", "mean_exposed_norm_distance", "#d44a3a"),
        ("central exposed", "central_exposure_fraction", "#20242c"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(centers, values, marker="o", linewidth=1.3, label=label, color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction")
    ax.set_title("Time-Bin Detection Probability", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)


def _plot_survivor_exposure_outcomes(ax: Any, summary: dict[str, Any]) -> None:
    outcomes = summary.get("survivor_exposure_outcomes", [])
    if not outcomes:
        ax.text(0.5, 0.5, "no survivor outcomes", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Survivor Exposure Outcomes", fontsize=10)
        return
    labels = [str(row.get("group", "?")) for row in outcomes]
    x = np.arange(len(labels))
    series = [
        ("cum p", "mean_cumulative_detection_probability", "#4f7cff"),
        ("final C", "mean_final_confidence", "#20242c"),
        ("scout C pre", "mean_scout_confidence_pre", "#0f766e"),
        ("scout C post", "mean_scout_confidence_post", "#14b8a6"),
        ("best p", "mean_best_detection_probability", "#36a269"),
        ("best norm", "mean_best_norm_distance", "#d44a3a"),
        ("edge frac", "mean_near_edge_exposure_fraction", "#8a5cf6"),
    ]
    width = min(0.15, 0.8 / max(len(series), 1))
    center_offset = 0.5 * (len(series) - 1)
    for idx, (label, key, color) in enumerate(series):
        values = [float(row.get(key, math.nan)) for row in outcomes]
        ax.bar(
            x + (idx - center_offset) * width,
            values,
            width=width,
            label=label,
            color=color,
            alpha=0.85,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Survivor Exposure Outcomes", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)


def _plot_time_bins_confidence(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Confidence", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    gain = [float(row.get("confidence_gain", math.nan)) for row in time_bins]
    best_gain = [
        float(row.get("confidence_opportunity_best_gain", math.nan))
        for row in time_bins
    ]
    pass_probability = [
        float(row.get("confidence_pass_probability", math.nan))
        for row in time_bins
    ]
    opportunity_fraction = [
        float(row.get("confidence_opportunity_fraction", math.nan))
        for row in time_bins
    ]
    mean_confidence = [float(row.get("confidence_mean", math.nan)) for row in time_bins]
    low_confidence = [float(row.get("confidence_low_fraction", math.nan)) for row in time_bins]
    high_confidence = [float(row.get("confidence_high_fraction", math.nan)) for row in time_bins]

    line_gain = ax.plot(
        centers,
        gain,
        marker="o",
        linewidth=1.6,
        label="gain",
        color="#4f7cff",
    )
    line_best = ax.plot(
        centers,
        best_gain,
        marker="o",
        linewidth=1.2,
        label="best gain",
        color="#14b8a6",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("confidence gain / drone-step")
    ax.grid(alpha=0.25)

    ax_frac = ax.twinx()
    lines = line_gain + line_best
    for label, values, color in (
        ("opp frac", opportunity_fraction, "#0f766e"),
        ("pass p", pass_probability, "#36a269"),
        ("mean C", mean_confidence, "#20242c"),
        ("low C", low_confidence, "#d44a3a"),
        ("high C", high_confidence, "#8a5cf6"),
    ):
        lines += ax_frac.plot(
            centers,
            values,
            marker="o",
            linewidth=1.2,
            label=label,
            color=color,
            alpha=0.9,
        )
    ax_frac.set_ylim(0.0, 1.0)
    ax_frac.set_ylabel("fraction / probability")
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, frameon=False, ncol=2)
    ax.set_title("Time-Bin Confidence", fontsize=10)


def _plot_time_bins_scouts(ax: Any, summary: dict[str, Any]) -> None:
    scout_bins = summary.get("scout_time_bins", [])
    if not scout_bins:
        ax.text(0.5, 0.5, "no scout bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Survivor Discovery Over Time", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in scout_bins
    ]
    widths = [
        0.72 * (
            float(row.get("end_fraction", 0.0))
            - float(row.get("start_fraction", 0.0))
        )
        for row in scout_bins
    ]
    mean_new = [float(row.get("mean_new_recall", math.nan)) for row in scout_bins]
    mean_cumulative = [
        float(row.get("mean_cumulative_recall", math.nan))
        for row in scout_bins
    ]
    median_cumulative = [
        float(row.get("median_cumulative_recall", math.nan))
        for row in scout_bins
    ]

    ax.bar(
        centers,
        mean_new,
        width=widths,
        color="#80bfff",
        alpha=0.35,
        label="new recall/bin mean",
    )
    ax.plot(
        centers,
        mean_cumulative,
        marker="o",
        linewidth=1.7,
        color="#d44a3a",
        label="cumulative mean",
    )
    ax.plot(
        centers,
        median_cumulative,
        marker="o",
        linewidth=1.3,
        linestyle="--",
        color="#20242c",
        label="cumulative median",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("recall fraction")
    ax.set_title("Survivor Discovery Over Time", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _plot_time_bins_frontier_inputs(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Frontier Score/Progress (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("score/uncovered", "frontier_uncovered_ratio", "#4f7cff"),
        ("progress", "frontier_progress", "#d44a3a"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(centers, values, marker="o", linewidth=1.6, label=label, color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction")
    ax.set_title("Frontier Score/Progress (mean)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def _plot_time_bins_frontier_reward_yield(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Frontier Reward vs New Cells (mean)", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    new_cells = [float(row.get("new_coverage_cells", math.nan)) for row in time_bins]
    reward = [float(row.get("frontier_reward", math.nan)) for row in time_bins]
    line_new = ax.plot(
        centers,
        new_cells,
        marker="o",
        linewidth=1.7,
        label="new cells",
        color="#36a269",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("new cells / drone-step")
    ax.grid(alpha=0.25)

    ax_reward = ax.twinx()
    line_reward = ax_reward.plot(
        centers,
        reward,
        marker="o",
        linewidth=1.4,
        label="frontier reward",
        color="#8a5cf6",
    )
    finite_reward = [value for value in reward if math.isfinite(value)]
    if finite_reward:
        reward_max = max(max(finite_reward), 1e-6)
        ax_reward.set_ylim(0.0, reward_max * 1.2)
    ax_reward.set_ylabel("reward / drone-step")

    lines = line_new + line_reward
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, frameon=False)
    ax.set_title("Frontier Reward vs New Cells (mean)", fontsize=10)


def _plot_time_bins_cleanup_targets(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cleanup Target Over Time", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    series = [
        ("valid", "cleanup_target_valid", "#4f7cff"),
        ("value", "cleanup_target_value", "#36a269"),
        ("progress", "cleanup_target_progress_fraction", "#d44a3a"),
        ("gate", "cleanup_target_frontier_gate", "#0f766e"),
        ("switch", "cleanup_target_switch", "#8a5cf6"),
        ("reached", "cleanup_target_reached", "#d6a21d"),
    ]
    for label, key, color in series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        ax.plot(centers, values, marker="o", linewidth=1.4, label=label, color=color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction / norm")
    ax.grid(alpha=0.25)

    distance = [float(row.get("cleanup_target_distance_m", math.nan)) for row in time_bins]
    ax_distance = ax.twinx()
    line_distance = ax_distance.plot(
        centers,
        distance,
        marker="o",
        linewidth=1.2,
        linestyle="--",
        color="#20242c",
        label="distance",
    )
    finite_distance = [value for value in distance if math.isfinite(value)]
    if finite_distance:
        ax_distance.set_ylim(0.0, max(max(finite_distance) * 1.2, 1.0))
    ax_distance.set_ylabel("distance (m)")
    lines = ax.lines + line_distance
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, frameon=False)
    ax.set_title("Cleanup Target Over Time", fontsize=10)


def _plot_hist_panel(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    title: str,
    key: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)

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


def _plot_failure_labels_panel(
    ax: Any,
    label_counts: dict[str, int],
) -> None:
    ax.clear()
    if label_counts:
        labels = list(label_counts.keys())
        counts = [label_counts[label] for label in labels]
        y = np.arange(len(labels))
        ax.barh(y, counts, color="#36a269", alpha=0.85)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("episodes")
        for idx, count in enumerate(counts):
            ax.text(count + 0.05, idx, str(count), va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no labels", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Failure Labels", fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def _plot_uav_start_heatmap_panel(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    fig: Any | None = None,
) -> None:
    ax.clear()
    starts = [
        position
        for row in rows
        for position in row.get("start_positions_m", [])
        if len(position) >= 2
    ]
    if starts:
        start_array = np.asarray(starts, dtype=float)
        map_width = _finite_mean([
            float(row.get("start_map_width_m", math.nan)) for row in rows
        ])
        map_height = _finite_mean([
            float(row.get("start_map_height_m", math.nan)) for row in rows
        ])
        if not math.isfinite(map_width):
            map_width = max(float(np.abs(start_array[:, 0]).max()) * 2.0, 1.0)
        if not math.isfinite(map_height):
            map_height = max(float(np.abs(start_array[:, 1]).max()) * 2.0, 1.0)
        heat = ax.hist2d(
            start_array[:, 0],
            start_array[:, 1],
            bins=12,
            range=[
                [-0.5 * map_width, 0.5 * map_width],
                [-0.5 * map_height, 0.5 * map_height],
            ],
            cmap="Blues",
        )
        if fig is not None:
            fig.colorbar(heat[3], ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x start (m)")
        ax.set_ylabel("y start (m)")
    else:
        ax.text(0.5, 0.5, "no starts", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("UAV Start Heatmap", fontsize=10)


def _write_fast_distribution_plots(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    label_counts: dict[str, int],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
    axes_flat = axes.ravel()
    hist_panels = [
        ("Recall", "recall", (0.0, 1.0)),
        ("Final Coverage", "final_coverage_fraction", (0.0, 1.0)),
        ("Final Confidence", "final_confidence_mean", (0.0, 1.0)),
        ("Movement / Step (m)", "avg_displacement_m", None),
    ]
    for ax, (title, key, xlim) in zip(axes_flat, hist_panels):
        _plot_hist_panel(ax, rows, title=title, key=key, xlim=xlim)

    custom_start = len(hist_panels)
    _plot_failure_labels_panel(axes_flat[custom_start], label_counts)
    _plot_per_drone_bars(
        axes_flat[custom_start + 1],
        summary,
        "mean_path_length_m",
        "Per-Drone Path Length",
        "m",
    )
    _plot_uav_start_heatmap_panel(axes_flat[custom_start + 2], rows, fig=fig)
    _plot_time_bins_scouts(axes_flat[custom_start + 3], summary)
    _plot_time_bins_search_efficiency(axes_flat[custom_start + 4], summary)
    _plot_time_bins_reward_scale(axes_flat[custom_start + 5], summary)

    for ax in axes_flat[custom_start + 6:]:
        ax.axis("off")

    fig.suptitle(
        "UAV HAPPO Fast Diagnostics "
        f"(n={int(summary.get('episodes', len(rows)))})",
        fontsize=14,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_distribution_plots(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    label_counts: dict[str, int],
    output_path: str,
    *,
    diagnostic_level: str = "full",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if str(diagnostic_level).replace("-", "_").lower() == "fast":
        _write_fast_distribution_plots(rows, summary, label_counts, output_path)
        return

    panels = [
        ("Recall", "recall", (0.0, 1.0)),
        ("Confirmation Recall", "confirmation_recall", (0.0, 1.0)),
        ("Final Coverage", "final_coverage_fraction", (0.0, 1.0)),
        ("Final Confidence", "final_confidence_mean", (0.0, 1.0)),
        ("Confidence Gain / Step", "avg_confidence_gain", None),
        ("Confidence Pass P", "avg_confidence_pass_probability", (0.0, 1.0)),
        ("Expected Recall From Exposure", "expected_recall_from_exposure", (0.0, 1.0)),
        ("Perception Recall Gap", "perception_recall_gap", (-1.0, 1.0)),
        ("Scout Detection Probability", "avg_scout_detection_probability", (0.0, 1.0)),
        ("Scout Norm Distance", "avg_scout_detection_norm_distance", (0.0, 1.0)),
        ("Scout Confidence Pre", "avg_scout_confidence_pre", (0.0, 1.0)),
        ("Scout Confidence Post", "avg_scout_confidence_post", (0.0, 1.0)),
        ("Missed Cum Detection P", "avg_missed_cum_detection_probability", (0.0, 1.0)),
        ("Missed Best Detection P", "avg_missed_best_detection_probability", (0.0, 1.0)),
        ("Missed Best Norm Distance", "avg_missed_best_norm_distance", (0.0, 1.0)),
        ("Missed Never Exposed", "missed_never_exposed_fraction", (0.0, 1.0)),
        ("Missed High Cum P", "missed_high_cum_probability_fraction", (0.0, 1.0)),
        ("Missed Edge Limited", "missed_edge_limited_fraction", (0.0, 1.0)),
        ("Movement / Step (m)", "avg_displacement_m", None),
        ("New Cells / Step", "avg_new_coverage_cells", None),
        ("Candidate Capture", "avg_candidate_capture_fraction", (0.0, 1.0)),
        ("Candidate Regret Cells", "avg_candidate_new_cell_regret", None),
        ("Candidate Action Rank", "avg_candidate_action_rank", (1.0, 9.0)),
        ("Frontier Candidate Capture", "avg_frontier_candidate_capture_fraction", (0.0, 1.5)),
        ("Frontier Candidate Rank", "avg_frontier_candidate_rank", (1.0, 9.0)),
        ("Frontier Align Best Move", "avg_frontier_candidate_best_alignment", (-1.0, 1.0)),
        ("Frontier Bad Candidate", "frontier_candidate_bad_frac", (0.0, 1.0)),
        ("Conf Frontier Capture", "avg_confidence_frontier_candidate_capture_fraction", (0.0, 1.5)),
        ("Conf LG Frontier Capture", "avg_confidence_lg_frontier_candidate_capture_fraction", (0.0, 1.5)),
        ("Conf Frontier Advantage", "avg_confidence_frontier_capture_advantage", (-1.0, 1.0)),
        ("Conf LG Frontier Advantage", "avg_confidence_lg_frontier_capture_advantage", (-1.0, 1.0)),
        ("Outside Footprint", "avg_outside_footprint_fraction", (0.0, 1.0)),
        ("Overlap", "avg_overlap_fraction", (0.0, 1.0)),
        ("Excess Overlap", "avg_excess_overlap_fraction", (0.0, 1.0)),
        ("Edge Step Fraction", "edge_step_frac", (0.0, 1.0)),
        ("Corner Step Fraction", "corner_step_frac", (0.0, 1.0)),
        ("Center Coverage", "coverage_center_fraction", (0.0, 1.0)),
        ("Moving No New Coverage", "moving_no_new_coverage_frac", (0.0, 1.0)),
        ("Moving No Confidence Gain", "moving_no_confidence_gain_frac", (0.0, 1.0)),
        ("Confidence Revisit", "confidence_revisit_step_frac", (0.0, 1.0)),
        ("Confidence Useful Revisit", "confidence_useful_revisit_step_frac", (0.0, 1.0)),
        ("Confidence Wasteful Revisit", "confidence_wasteful_revisit_step_frac", (0.0, 1.0)),
        ("Confidence Saturated Footprint", "avg_confidence_overlap_fraction", (0.0, 1.0)),
        ("Confidence Overlap Regret", "avg_confidence_overlap_regret", (0.0, 1.0)),
        ("Confidence Revisit Useful Share", "confidence_revisit_useful_share", (0.0, 1.0)),
        ("Confidence Revisit Gain Share", "confidence_revisit_gain_share", (0.0, 1.0)),
        ("Cleanup Target Progress", "avg_cleanup_target_progress_fraction", (-1.0, 1.0)),
        ("Cleanup Target Switch", "cleanup_target_switch_rate", (0.0, 1.0)),
        ("Cleanup Target Reached", "cleanup_target_reached_rate", (0.0, 1.0)),
        ("Cleanup Target No Progress", "cleanup_target_no_progress_frac", (0.0, 1.0)),
        ("Start Pair Min (m)", "min_start_pair_distance_m", None),
        ("Start Edge Min (m)", "min_start_edge_distance_m", None),
        ("Action Frontier Align", "avg_action_frontier_alignment", (-1.0, 1.0)),
        ("Action Frontier Aligned", "action_frontier_aligned_step_frac", (0.0, 1.0)),
        ("Action Frontier No New", "action_frontier_aligned_no_new_frac", (0.0, 1.0)),
        ("Frontier Reward", "avg_reward_uav_frontier", None),
        ("Confidence Overlap Penalty", "avg_penalty_uav_confidence_overlap", None),
        ("Frontier Reward Share", "avg_frontier_abs_reward_share", (0.0, 1.0)),
        ("High Frontier No New", "frontier_high_progress_no_new_frac", (0.0, 1.0)),
        ("High Frontier Edge", "frontier_high_progress_edge_frac", (0.0, 1.0)),
        ("Frontier/New Corr", "frontier_progress_new_cells_corr", (-1.0, 1.0)),
    ]

    custom_panel_count = 24
    ncols = 3
    nrows = math.ceil((len(panels) + custom_panel_count) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows), constrained_layout=True)
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

    custom_start = len(panels)
    _plot_per_drone_bars(
        axes_flat[custom_start],
        summary,
        "mean_path_length_m",
        "Per-Drone Path Length",
        "m",
    )
    _plot_per_drone_bars(
        axes_flat[custom_start + 1],
        summary,
        "mean_edge_step_frac",
        "Per-Drone Edge Fraction",
        "fraction",
    )
    _plot_per_drone_bars(
        axes_flat[custom_start + 2],
        summary,
        "mean_avg_reward_uav_frontier",
        "Per-Drone Frontier Reward",
        "reward",
    )
    _plot_time_bins(axes_flat[custom_start + 3], summary)

    _plot_per_drone_bars(
        axes_flat[custom_start + 4],
        summary,
        "mean_avg_excess_overlap_fraction",
        "Per-Drone Excess Overlap",
        "fraction",
    )

    heat_ax = axes_flat[custom_start + 5]
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

    label_ax = axes_flat[custom_start + 6]
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

    _plot_time_bins_coverage_signals(axes_flat[custom_start + 7], summary)
    _plot_time_bins_search_efficiency(axes_flat[custom_start + 8], summary)
    _plot_time_bins_reward_scale(axes_flat[custom_start + 9], summary)
    _plot_time_bins_scouts(axes_flat[custom_start + 10], summary)
    _plot_time_bins_frontier_inputs(axes_flat[custom_start + 11], summary)
    _plot_time_bins_frontier_reward_yield(axes_flat[custom_start + 12], summary)
    _plot_time_bins_coverage_opportunity(axes_flat[custom_start + 13], summary)
    _plot_time_bins_counterfactual(axes_flat[custom_start + 14], summary)
    _plot_time_bins_revisit_sources(axes_flat[custom_start + 15], summary)
    _plot_outcome_candidate_splits(axes_flat[custom_start + 16], summary)
    _plot_outcome_structure_splits(axes_flat[custom_start + 17], summary)
    _plot_perception_time_bins(axes_flat[custom_start + 18], summary)
    _plot_perception_probability_bins(axes_flat[custom_start + 19], summary)
    _plot_survivor_exposure_outcomes(axes_flat[custom_start + 20], summary)
    _plot_time_bins_confidence(axes_flat[custom_start + 21], summary)
    _plot_time_bins_frontier_usefulness(axes_flat[custom_start + 22], summary)
    _plot_time_bins_cleanup_targets(axes_flat[custom_start + 23], summary)

    for ax in axes_flat[custom_start + 24:]:
        ax.axis("off")

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
    parser.add_argument("--joint-schema-uav-diagnostic", action="store_true",
                        help="Evaluate UAV-only checkpoints trained with final joint-schema observation padding.")
    parser.add_argument("--joint-observation-schema", action="store_true",
                        help="Alias for --joint-schema-uav-diagnostic.")
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--n-drones", "--n-uavs", dest="n_drones", type=int, default=None,
                        help="Override UAV count for legacy checkpoints. Default preserves the checkpoint manifest.")
    parser.add_argument("--n-ugvs", "--n-ground", dest="n_ugvs", type=int, default=None,
                        help="Override UGV schema count for joint-schema UAV diagnostics.")
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
    parser.add_argument("--local-map-patch-size", type=int, default=None)
    parser.add_argument("--enable-fire", dest="enable_fire", action="store_true",
                        help="Enable fire/smoke dynamics during UAV diagnostics. Defaults to disabled.")
    parser.add_argument("--disable-fire", dest="enable_fire", action="store_false",
                        help="Disable fire/smoke dynamics during UAV diagnostics.")
    parser.set_defaults(enable_fire=False)
    parser.add_argument("--comms-dropout", type=float, default=None,
                        help="Communication dropout probability during evaluation. "
                             "Omitted or 0 preserves the full-communication diagnostic default.")
    parser.add_argument("--comms-dropout-mode", choices=("iid", "bursty"), default=None,
                        help="Communication dropout process. Omitted preserves the checkpoint mode, "
                             "falling back to iid for legacy checkpoints.")
    parser.add_argument("--drone-min-footprint-radius-m", type=float, default=None)
    parser.add_argument("--drone-perception-mode",
                        choices=("rgb", "rgb_thermal", "rgb+thermal", "rgb-thermal"),
                        default=None,
                        help="Override abstract UAV perception mode. rgb_thermal currently aliases rgb.")
    parser.add_argument("--uav-start-min-separation-m", type=float, default=None,
                        help="Override checkpoint UAV start min separation in meters; pass 0 to disable.")
    parser.add_argument("--uav-start-edge-margin-m", type=float, default=None,
                        help="Override checkpoint UAV start edge margin in meters; pass 0 to disable.")
    parser.add_argument("--uav-overlap-penalty-normalization",
                        choices=("raw", "opportunity"),
                        default=None,
                        help="Override overlap penalty normalization for diagnostic reward terms. "
                             "Default uses the checkpoint manifest, falling back to raw.")
    parser.add_argument("--uav-decision-grid", type=int, default=None,
                        help="Override UAV internal decision-map grid size for diagnostics. "
                             "Default preserves checkpoint settings.")
    parser.add_argument("--uav-confidence-reward-grid", type=int, default=None,
                        help="Override UAV confidence reward/penalty/opportunity grid size for diagnostics.")
    parser.add_argument("--uav-frontier-global-grid", type=int, default=None,
                        help="Override the global leg grid size for local-global UAV frontier diagnostics.")
    parser.add_argument("--uav-coverage-reward-grid", type=int, default=None,
                        help="Override UAV binary coverage reward/overlap grid size for diagnostics.")
    parser.add_argument("--diagnostic-confidence-frontier-radius-m", type=float,
                        default=DEFAULT_DIAGNOSTIC_CONFIDENCE_FRONTIER_RADIUS_M,
                        help="Local radius for the shadow confidence local-global frontier usefulness "
                             "diagnostic. This does not change the evaluated policy.")
    parser.add_argument("--moving-no-confidence-gain-threshold", type=float,
                        default=DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
                        help="Weighted confidence-gain threshold used by the diagnostics-only "
                             "moving_no_confidence_gain metric. A drone-step counts when the UAV "
                             "moves more than 1m and weighted confidence gain is at or below this value.")
    parser.add_argument("--diagnostic-level", choices=("full", "fast"), default="full",
                        help="Use 'fast' to keep core 100-seed diagnostics but skip expensive "
                             "counterfactual, shadow-frontier, survivor-perception, and cleanup-target probes.")
    parser.add_argument("--include-cleanup-target-diagnostics", action="store_true",
                        help="Compute cleanup-target diagnostics even when --diagnostic-level fast is used.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actor means.")
    parser.add_argument("--json-output", default=None, help="Optional path to write per-seed rows and summary as JSON.")
    parser.add_argument("--plots-output", default=None, help="Optional path to write histogram diagnostics as a PNG.")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
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
    if args.local_map_patch_size is not None and (args.local_map_patch_size < 1 or args.local_map_patch_size % 2 != 1):
        parser.error("--local-map-patch-size must be a positive odd integer")
    if (
        args.comms_dropout is not None
        and (
            not math.isfinite(args.comms_dropout)
            or not 0.0 <= args.comms_dropout <= 1.0
        )
    ):
        parser.error("--comms-dropout must be finite and between 0 and 1")
    if args.uav_start_min_separation_m is not None and args.uav_start_min_separation_m < 0.0:
        parser.error("--uav-start-min-separation-m must be nonnegative")
    if args.uav_start_edge_margin_m is not None and args.uav_start_edge_margin_m < 0.0:
        parser.error("--uav-start-edge-margin-m must be nonnegative")
    for arg_name in (
        "uav_decision_grid",
        "uav_confidence_reward_grid",
        "uav_frontier_global_grid",
        "uav_coverage_reward_grid",
    ):
        value = getattr(args, arg_name)
        if value is not None and (value < 0 or value == 1):
            parser.error(f"--{arg_name.replace('_', '-')} must be 0 or at least 2")
    if args.diagnostic_confidence_frontier_radius_m <= 0.0:
        parser.error("--diagnostic-confidence-frontier-radius-m must be positive")
    if (
        not math.isfinite(args.moving_no_confidence_gain_threshold)
        or args.moving_no_confidence_gain_threshold < 0.0
    ):
        parser.error("--moving-no-confidence-gain-threshold must be finite and nonnegative")
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
        f"{scenario_kwargs['n_survivors']} survivor slots "
        f"(active {scenario_kwargs.get('active_survivors_min', scenario_kwargs['n_survivors'])}"
        f"..{scenario_kwargs.get('active_survivors_max', scenario_kwargs['n_survivors'])}), "
        f"{scenario_kwargs.get('n_decoys', 0)} decoy slots "
        f"(active {scenario_kwargs.get('active_decoys_min', scenario_kwargs.get('n_decoys', 0))}"
        f"..{scenario_kwargs.get('active_decoys_max', scenario_kwargs.get('n_decoys', 0))}), "
        f"dt={scenario_kwargs.get('sim_step_seconds', 'scenario-default')}s"
    )
    if args.joint_schema_uav_diagnostic or args.joint_observation_schema:
        print(
            "joint_observation_schema: "
            f"obs_schema=({scenario_kwargs.get('obs_schema_n_drones')}, "
            f"{scenario_kwargs.get('obs_schema_n_ground')}, "
            f"{scenario_kwargs.get('obs_schema_n_survivors')})"
        )
    print(
        "uav starts: "
        f"min_sep={scenario_kwargs.get('uav_start_min_separation_m', 0.0)}m "
        f"edge_margin={scenario_kwargs.get('uav_start_edge_margin_m', 0.0)}m"
    )
    print(
        "communications: "
        f"dropout={scenario_kwargs.get('comms_dropout', 0.0)} "
        f"mode={scenario_kwargs.get('comms_dropout_mode', 'iid')} "
        f"burst_steps={scenario_kwargs.get('comms_dropout_min_steps', 5)}"
        f"..{scenario_kwargs.get('comms_dropout_max_steps', 15)}"
    )
    print(f"drone_perception_mode: {scenario_kwargs.get('drone_perception_mode', 'rgb')}")
    print(
        "uav overlap penalty: "
        f"normalization={scenario_kwargs.get('uav_overlap_penalty_normalization', 'raw')}"
    )
    print(
        "shadow confidence frontier diagnostic: "
        f"local_global_radius={args.diagnostic_confidence_frontier_radius_m}m"
    )
    print(
        "diagnostics: "
        f"level={args.diagnostic_level} "
        f"cleanup_target={bool(scenario_kwargs.get('uav_cleanup_target_diagnostics', False))} "
        f"move_no_conf_thr={args.moving_no_confidence_gain_threshold:g}"
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
        run_rollout(
            policy,
            scenario_kwargs,
            seed,
            diagnostic_confidence_frontier_radius_m=args.diagnostic_confidence_frontier_radius_m,
            moving_no_confidence_gain_threshold=args.moving_no_confidence_gain_threshold,
            diagnostic_level=args.diagnostic_level,
        )
        for seed in args.seeds
    ]
    for row in rows:
        if args.diagnostic_level == "fast":
            print(
                f"seed {row['seed']:>4}: "
                f"scouted={row['scouted']}/{row['survivors']} "
                f"missed={row['missed']} "
                f"recall={row['recall']:.3f} "
                f"confirmed={row['confirmed']}/{row['survivors']} "
                f"confirm_recall={row['confirmation_recall']:.3f} "
                f"coverage={row['final_coverage_fraction']:.3f} "
                f"conf={row['final_confidence_mean']:.3f} "
                f"move={row['avg_displacement_m']:.2f}m "
                f"avg_scout={_fmt_optional(row['avg_scout_step'])} steps/"
                f"{_fmt_optional(row['avg_scout_time_s'])}s "
                f"all_scouted={_fmt_optional(row['all_scouted_step'])} steps/"
                f"{_fmt_optional(row['all_scouted_time_s'])}s "
                f"label={row['failure_label']} "
                f"first_steps={row['first_scout_steps']}"
            )
            continue
        print(
            f"seed {row['seed']:>4}: "
            f"scouted={row['scouted']}/{row['survivors']} "
            f"missed={row['missed']} "
            f"recall={row['recall']:.3f} "
            f"confirmed={row['confirmed']}/{row['survivors']} "
            f"confirm_recall={row['confirmation_recall']:.3f} "
            f"coverage={row['final_coverage_fraction']:.3f} "
            f"conf={row['final_confidence_mean']:.3f} "
            f"conf_gain={row['avg_confidence_gain']:.5f} "
            f"conf_pass={row['avg_confidence_pass_probability']:.2f} "
            f"conf_rev={row['confidence_revisit_step_frac']:.2f}/"
            f"{row['confidence_useful_revisit_step_frac']:.2f}/"
            f"{row['confidence_wasteful_revisit_step_frac']:.2f} "
            f"conf_ov={row['avg_confidence_overlap_fraction']:.2f} "
            f"move_no_conf={row['moving_no_confidence_gain_frac']:.2f} "
            f"avg_scout={_fmt_optional(row['avg_scout_step'])} steps/"
            f"{_fmt_optional(row['avg_scout_time_s'])}s "
            f"all_scouted={_fmt_optional(row['all_scouted_step'])} steps/"
            f"{_fmt_optional(row['all_scouted_time_s'])}s "
            f"exp_recall={row['expected_recall_from_exposure']:.2f} "
            f"scout_p={row['avg_scout_detection_probability']:.2f} "
            f"scout_C={row['avg_scout_confidence_pre']:.2f}/"
            f"{row['avg_scout_confidence_post']:.2f} "
            f"scout_norm={row['avg_scout_detection_norm_distance']:.2f} "
            f"miss_cum_p={row['avg_missed_cum_detection_probability']:.2f} "
            f"miss_never/high/edge="
            f"{row['missed_never_exposed_fraction']:.2f}/"
            f"{row['missed_high_cum_probability_fraction']:.2f}/"
            f"{row['missed_edge_limited_fraction']:.2f} "
            f"act={row['avg_action_norm']:.3f} "
            f"move={row['avg_displacement_m']:.2f}m "
            f"align={row['avg_action_displacement_alignment']:.3f} "
            f"new_cells={row['avg_new_coverage_cells']:.1f} "
            f"raw_new={row['avg_raw_new_coverage_cells']:.1f} "
            f"new_frac={row['new_coverage_step_frac']:.2f} "
            f"raw_frac={row['raw_new_coverage_step_frac']:.2f} "
            f"opp={row['avg_coverage_opportunity_fraction']:.2f} "
            f"opp_avail={row['avg_coverage_opportunity_available_fraction']:.2f} "
            f"edge={row['edge_step_frac']:.2f} "
            f"corner={row['corner_step_frac']:.2f} "
            f"outside={row['avg_outside_footprint_fraction']:.2f} "
            f"overlap={row['avg_overlap_fraction']:.2f} "
            f"exp_ov={row['avg_expected_overlap_fraction']:.2f} "
            f"excess_ov={row['avg_excess_overlap_fraction']:.2f} "
            f"inter_ov={row['avg_inter_uav_overlap_fraction']:.2f} "
            f"own_rev={row['avg_own_history_revisit_fraction']:.2f} "
            f"team_rev={row['avg_teammate_history_revisit_fraction']:.2f} "
            f"avoid_rev={row['avg_avoidable_revisit_fraction']:.2f} "
            f"front_exp={row['avg_frontier_expected_new_cells']:.1f} "
            f"front_cap={row['avg_frontier_new_cell_capture_fraction']:.2f} "
            f"cand_best={row['avg_candidate_best_new_cells']:.1f} "
            f"cand_cap={row['avg_candidate_capture_fraction']:.2f} "
            f"cand_reg={row['avg_candidate_new_cell_regret']:.1f} "
            f"cand_rank={row['avg_candidate_action_rank']:.1f}/"
            f"{row['avg_candidate_movement_rank']:.1f} "
            f"front_use={row['avg_frontier_candidate_capture_fraction']:.2f}/"
            f"{row['avg_frontier_candidate_rank']:.1f}/"
            f"{row['avg_frontier_candidate_best_alignment']:.2f} "
            f"conf_front={row['avg_confidence_frontier_candidate_capture_fraction']:.2f} "
            f"conf_lg={row['avg_confidence_lg_frontier_candidate_capture_fraction']:.2f} "
            f"cand_avoid={row['avg_candidate_avoidable_overlap']:.2f} "
            f"front={row['avg_frontier_alignment']:.2f}/"
            f"{row['avg_frontier_progress_fraction']:.2f}/"
            f"{row['avg_frontier_uncovered_ratio']:.2f} "
            f"act_front={row['avg_action_frontier_alignment']:.2f}/"
            f"{row['action_frontier_aligned_step_frac']:.2f} "
            f"front_rew={row['avg_reward_uav_frontier']:.4f} "
            f"front_hi={row['frontier_high_progress_step_frac']:.2f}/"
            f"{row['frontier_high_progress_no_new_frac']:.2f}/"
            f"{row['frontier_high_progress_edge_frac']:.2f} "
            f"center_cov={row['coverage_center_fraction']:.2f} "
            f"start_pair={_fmt_optional(row['min_start_pair_distance_m'])}m "
            f"start_edge={_fmt_optional(row['min_start_edge_distance_m'])}m "
            f"label={row['failure_label']} "
            f"first_steps={row['first_scout_steps']}"
        )
        if row.get("per_drone"):
            print(f"  drones: {_format_per_drone_row(row['per_drone'])}")

    summary = summarize(rows)
    label_counts = _label_counts(rows)
    print("-" * 88)
    print(
        "means: "
        f"scouted={summary['mean_scouted']:.3f} "
        f"missed={summary['mean_missed']:.3f} "
        f"recall={summary['mean_recall']:.3f} "
        f"confirmed={summary['mean_confirmed']:.3f} "
        f"confirm_recall={summary['mean_confirmation_recall']:.3f} "
        f"coverage={summary['mean_final_coverage_fraction']:.3f} "
        f"confidence={summary['mean_final_confidence_mean']:.3f} "
        f"success={summary['full_success_rate']:.3f} "
        f"confirm_success={summary['full_confirmation_success_rate']:.3f} "
        f"avg_scout={summary['mean_avg_scout_step']:.1f} steps/"
        f"{summary['mean_avg_scout_time_s']:.1f}s "
        f"all_scouted_successes={summary['mean_all_scouted_step_successes']:.1f} steps/"
        f"{summary['mean_all_scouted_time_s_successes']:.1f}s "
        f"avg_confirm={summary['mean_avg_confirm_step']:.1f} steps/"
        f"{summary['mean_avg_confirm_time_s']:.1f}s "
        f"all_confirmed_successes={summary['mean_all_confirmed_step_successes']:.1f} steps/"
        f"{summary['mean_all_confirmed_time_s_successes']:.1f}s"
    )
    if args.diagnostic_level == "fast":
        print(
            "core search means: "
            f"move={summary['mean_displacement_m']:.2f}m "
            f"coverage={summary['mean_final_coverage_fraction']:.3f} "
            f"confidence={summary['mean_final_confidence_mean']:.3f}"
        )
        print(
            "uav reward-scale means: "
            f"coverage={summary['mean_reward_uav_coverage']:.4f} "
            f"move_cov={summary['mean_reward_uav_move_coverage']:.4f} "
            f"frontier={summary['mean_reward_uav_frontier']:.4f} "
            f"confidence={summary['mean_reward_uav_confidence']:.4f} "
            f"conf_move={summary['mean_reward_uav_confidence_move']:.4f} "
            f"cleanup={summary['mean_reward_uav_cleanup_target_progress']:.4f} "
            f"astar={summary['mean_reward_uav_astar_progress']:.4f} "
            f"move_pen={summary['mean_penalty_uav_inefficient_move']:.4f} "
            f"conf_ov_pen={summary['mean_penalty_uav_confidence_overlap']:.4f} "
            f"overlap_pen={summary['mean_penalty_uav_overlap']:.4f} "
            f"inter_pen={summary['mean_penalty_uav_inter_overlap']:.4f} "
            f"outside_pen={summary['mean_penalty_uav_outside_footprint']:.4f} "
            f"scout={summary['mean_reward_uav_scout']:.4f} "
            f"team={summary['mean_reward_team']:.4f}"
        )
        print(
            "failure labels: "
            + ", ".join(f"{label}={count}" for label, count in label_counts.items())
        )
        print(
            "distribution snapshots: "
            f"coverage p25/p50/p75="
            f"{summary['coverage_p25']:.3f}/{summary['coverage_p50']:.3f}/{summary['coverage_p75']:.3f} "
            f"confidence p25/p50/p75="
            f"{summary['confidence_final_p25']:.3f}/{summary['confidence_final_p50']:.3f}/{summary['confidence_final_p75']:.3f} "
            f"movement p25/p50/p75="
            f"{summary['move_m_p25']:.2f}/{summary['move_m_p50']:.2f}/{summary['move_m_p75']:.2f}m"
        )
        print("note: fast diagnostics include only recall, final coverage/confidence, movement, failure labels, and reward-scale time bins.")
        print("note: all_scouted_successes averages only episodes that scouted every survivor.")

        if args.json_output:
            output = {
                "checkpoint": str(checkpoint_dir),
                "diagnostic_level": args.diagnostic_level,
                "scenario_kwargs": scenario_kwargs,
                "rows": rows,
                "summary": summary,
                "label_counts": label_counts,
            }
            Path(args.json_output).write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"wrote: {args.json_output}")

        if args.plots_output:
            write_distribution_plots(
                rows,
                summary,
                label_counts,
                args.plots_output,
                diagnostic_level=args.diagnostic_level,
            )
            print(f"wrote plots: {args.plots_output}")
        return
    print(
        "survivor perception means: "
        f"exp_steps={summary['mean_survivor_exposure_steps']:.1f} "
        f"scouted_exp={summary['mean_scouted_survivor_exposure_steps']:.1f} "
        f"missed_exp={summary['mean_missed_survivor_exposure_steps']:.1f} "
        f"cum_p={summary['mean_survivor_cum_detection_probability']:.3f} "
        f"scouted_cum_p={summary['mean_scouted_cum_detection_probability']:.3f} "
        f"missed_cum_p={summary['mean_missed_cum_detection_probability']:.3f} "
        f"cell_conf={summary['mean_survivor_final_confidence']:.3f} "
        f"scouted_cell_conf={summary['mean_scouted_survivor_final_confidence']:.3f} "
        f"missed_cell_conf={summary['mean_missed_survivor_final_confidence']:.3f} "
        f"scout_p={summary['mean_scout_detection_probability']:.3f} "
        f"scout_C={summary['mean_scout_confidence_pre']:.3f}/"
        f"{summary['mean_scout_confidence_post']:.3f} "
        f"scout_norm={summary['mean_scout_detection_norm_distance']:.3f} "
        f"scout_margin={summary['mean_scout_detection_margin_m']:.1f}m "
        f"miss_best_p={summary['mean_missed_best_detection_probability']:.3f} "
        f"miss_best_norm={summary['mean_missed_best_norm_distance']:.3f} "
        f"miss_min_dist={summary['mean_missed_min_distance_m']:.1f}m "
        f"miss_margin={summary['mean_missed_best_margin_m']:.1f}m "
        f"expected_recall={summary['mean_expected_recall_from_exposure']:.3f} "
        f"recall_gap={summary['mean_perception_recall_gap']:.3f} "
        f"miss_never={summary['mean_missed_never_exposed_fraction']:.3f} "
        f"miss_low_p={summary['mean_missed_low_cum_probability_fraction']:.3f} "
        f"miss_high_p={summary['mean_missed_high_cum_probability_fraction']:.3f} "
        f"miss_edge={summary['mean_missed_edge_limited_fraction']:.3f}"
    )
    print(
        "action/motion means: "
        f"act_norm={summary['mean_action_norm']:.3f} "
        f"move={summary['mean_displacement_m']:.2f}m "
        f"align={summary['mean_action_displacement_alignment']:.3f} "
        f"align_new={summary['mean_action_displacement_alignment_new_cov']:.3f} "
        f"align_nonew={summary['mean_action_displacement_alignment_no_new_cov']:.3f} "
        f"new_cells={summary['mean_new_coverage_cells']:.1f} "
        f"raw_new={summary['mean_raw_new_coverage_cells']:.1f} "
        f"new_step_frac={summary['mean_new_coverage_step_frac']:.3f} "
        f"raw_new_step_frac={summary['mean_raw_new_coverage_step_frac']:.3f}"
    )
    print(
        "confidence map means: "
        f"final={summary['mean_final_confidence_mean']:.3f} "
        f"low<0.5={summary['mean_final_confidence_low_fraction']:.3f} "
        f"high>=0.8={summary['mean_final_confidence_high_fraction']:.3f} "
        f"step_mean={summary['mean_confidence_mean']:.3f} "
        f"team_gain={summary['mean_confidence_gain']:.5f} "
        f"drone_gain={summary['mean_confidence_gain_by_drone']:.5f} "
        f"weighted_gain={summary['mean_confidence_weighted_gain_by_drone']:.5f} "
        f"move_no_conf={summary['mean_moving_no_confidence_gain_frac']:.3f} "
        f"opp={summary['mean_confidence_opportunity_fraction']:.3f} "
        f"best_gain={summary['mean_confidence_opportunity_best_gain']:.5f} "
        f"pass_p={summary['mean_confidence_pass_probability']:.3f} "
        f"revisit/useful/wasteful="
        f"{summary['mean_confidence_revisit_step_frac']:.3f}/"
        f"{summary['mean_confidence_useful_revisit_step_frac']:.3f}/"
        f"{summary['mean_confidence_wasteful_revisit_step_frac']:.3f} "
        f"sat_footprint={summary['mean_confidence_overlap_fraction']:.3f} "
        f"useful_share={summary['mean_confidence_revisit_useful_share']:.3f} "
        f"gain_share={summary['mean_confidence_revisit_gain_share']:.3f}"
    )
    print(
        "cleanup target means: "
        f"valid={summary['mean_cleanup_target_valid_fraction']:.3f} "
        f"dist={summary['mean_cleanup_target_distance_m']:.1f}m "
        f"value={summary['mean_cleanup_target_value']:.3f} "
        f"progress={summary['mean_cleanup_target_progress_m']:.2f}m/"
        f"{summary['mean_cleanup_target_progress_fraction']:.3f} "
        f"gate={summary['mean_cleanup_target_frontier_gate']:.3f} "
        f"switch={summary['mean_cleanup_target_switch_rate']:.3f} "
        f"reached={summary['mean_cleanup_target_reached_rate']:.3f} "
        f"no_prog={summary['mean_cleanup_target_no_progress_frac']:.3f} "
        f"prog_new={summary['mean_cleanup_target_progress_with_new_cells_frac']:.3f} "
        f"prog_excess={summary['mean_cleanup_target_progress_with_excess_overlap_frac']:.3f}"
    )
    print(
        "frontier observation/action means: "
        f"obs_empty={summary['mean_frontier_obs_empty_step_frac']:.3f} "
        f"obs_dist={summary['mean_frontier_obs_distance']:.3f} "
        f"obs_norm={summary['mean_frontier_obs_vector_norm']:.3f} "
        f"act_front={summary['mean_action_frontier_alignment']:.3f} "
        f"act_front_new={summary['mean_action_frontier_alignment_new_cov']:.3f} "
        f"act_front_nonew={summary['mean_action_frontier_alignment_no_new_cov']:.3f} "
        f"act_intent={summary['mean_action_frontier_intent']:.3f} "
        f"act_move_gap={summary['mean_action_frontier_movement_gap']:.3f} "
        f"act_aligned={summary['mean_action_frontier_aligned_step_frac']:.3f} "
        f"act_anti={summary['mean_action_frontier_anti_aligned_step_frac']:.3f} "
        f"act_aligned_no_new={summary['mean_action_frontier_aligned_no_new_frac']:.3f} "
        f"act_aligned_edge={summary['mean_action_frontier_aligned_edge_frac']:.3f} "
        f"corr_act_new={summary['mean_action_frontier_alignment_new_cells_corr']:.3f} "
        f"corr_act_boundary={summary['mean_action_frontier_alignment_boundary_distance_corr']:.3f}"
    )
    print(
        "coverage-observation comparison means: "
        f"front_local={summary['mean_frontier_local_coverage_cos']:.3f} "
        f"front_global={summary['mean_frontier_global_coverage_cos']:.3f} "
        f"local_global={summary['mean_local_global_coverage_cos']:.3f} "
        f"front_sector={summary['mean_frontier_sector_cos']:.3f} "
        f"sector_dom={summary['mean_frontier_sector_dominance']:.3f} "
        f"sector_entropy={summary['mean_frontier_sector_entropy']:.3f} "
        f"centroid_strength={summary['mean_frontier_cancellation']:.3f} "
        f"front_pair_cos={summary['mean_frontier_pairwise_cos']:.3f} "
        f"front_pair_same={summary['mean_frontier_pairwise_same_dir']:.3f} "
        f"local_pair_same={summary['mean_local_pairwise_same_dir']:.3f} "
        f"global_pair_same={summary['mean_global_pairwise_same_dir']:.3f}"
    )
    print(
        "counterfactual move means: "
        f"best_new={summary['mean_candidate_best_new_cells']:.1f} "
        f"capture={summary['mean_candidate_capture_fraction']:.3f} "
        f"regret={summary['mean_candidate_new_cell_regret']:.1f} "
        f"best_overlap={summary['mean_candidate_best_new_overlap']:.3f} "
        f"useful_overlap={summary['mean_candidate_best_useful_overlap']:.3f} "
        f"cand_avoidable={summary['mean_candidate_avoidable_overlap']:.3f} "
        f"action_rank={summary['mean_candidate_action_rank']:.2f} "
        f"movement_rank={summary['mean_candidate_movement_rank']:.2f} "
        f"action_cap={summary['mean_candidate_action_capture_fraction']:.3f} "
        f"move_cap={summary['mean_candidate_movement_capture_fraction']:.3f} "
        f"action_best_align={summary['mean_candidate_action_best_alignment']:.3f} "
        f"move_best_align={summary['mean_candidate_movement_best_alignment']:.3f} "
        f"no_opp={summary['mean_candidate_no_opportunity_frac']:.3f}"
    )
    print(
        "frontier usefulness means: "
        f"current_cap={summary['mean_frontier_candidate_capture_fraction']:.3f} "
        f"current_rank={summary['mean_frontier_candidate_rank']:.2f} "
        f"current_nearest_rank={summary['mean_frontier_candidate_nearest_rank']:.2f} "
        f"current_align_best={summary['mean_frontier_candidate_best_alignment']:.3f} "
        f"current_bad={summary['mean_frontier_candidate_bad_frac']:.3f} "
        f"conf_same_cap={summary['mean_confidence_frontier_candidate_capture_fraction']:.3f} "
        f"conf_same_rank={summary['mean_confidence_frontier_candidate_rank']:.2f} "
        f"conf_same_bad={summary['mean_confidence_frontier_candidate_bad_frac']:.3f} "
        f"conf_lg_cap={summary['mean_confidence_lg_frontier_candidate_capture_fraction']:.3f} "
        f"conf_lg_rank={summary['mean_confidence_lg_frontier_candidate_rank']:.2f} "
        f"conf_lg_bad={summary['mean_confidence_lg_frontier_candidate_bad_frac']:.3f} "
        f"conf_same_adv={summary['mean_confidence_frontier_capture_advantage']:.3f} "
        f"conf_lg_adv={summary['mean_confidence_lg_frontier_capture_advantage']:.3f} "
        f"corr_cap_new={summary['mean_frontier_candidate_capture_new_cells_corr']:.3f}"
    )
    print(
        "footprint/revisit means: "
        f"outside={summary['mean_outside_footprint_fraction']:.3f} "
        f"outside10={summary['mean_outside_footprint_step_frac_10']:.3f} "
        f"overlap={summary['mean_overlap_fraction']:.3f} "
        f"expected_overlap={summary['mean_expected_overlap_fraction']:.3f} "
        f"excess_overlap={summary['mean_excess_overlap_fraction']:.3f} "
        f"inter_uav_overlap={summary['mean_inter_uav_overlap_fraction']:.3f} "
        f"own_revisit={summary['mean_own_history_revisit_fraction']:.3f} "
        f"teammate_revisit={summary['mean_teammate_history_revisit_fraction']:.3f} "
        f"own_only={summary['mean_own_only_revisit_fraction']:.3f} "
        f"teammate_only={summary['mean_teammate_only_revisit_fraction']:.3f} "
        f"shared_old={summary['mean_shared_history_revisit_fraction']:.3f} "
        f"unavoidable={summary['mean_unavoidable_revisit_fraction']:.3f} "
        f"avoidable={summary['mean_avoidable_revisit_fraction']:.3f} "
        f"frontier_expected_new={summary['mean_frontier_expected_new_cells']:.1f} "
        f"frontier_capture={summary['mean_frontier_new_cell_capture_fraction']:.3f} "
        f"frontier_gap={summary['mean_frontier_new_cell_gap']:.1f} "
        f"opp_cells={summary['mean_coverage_opportunity_cells']:.1f} "
        f"opp_frac={summary['mean_coverage_opportunity_fraction']:.3f} "
        f"opp_avail={summary['mean_coverage_opportunity_available_fraction']:.3f} "
        f"frontier={summary['mean_frontier_alignment']:.3f}/"
        f"{summary['mean_frontier_progress_fraction']:.3f}/"
        f"{summary['mean_frontier_uncovered_ratio']:.3f} "
        f"excess10={summary['mean_excess_overlap_step_frac_10']:.3f} "
        f"inter20={summary['mean_inter_uav_overlap_step_frac_20']:.3f} "
        f"excess20={summary['mean_excess_overlap_step_frac_20']:.3f} "
        f"overlap60={summary['mean_overlap_step_frac_60']:.3f}"
    )
    print(
        "uav reward-scale means: "
        f"coverage={summary['mean_reward_uav_coverage']:.4f} "
        f"move_cov={summary['mean_reward_uav_move_coverage']:.4f} "
        f"frontier={summary['mean_reward_uav_frontier']:.4f} "
        f"confidence={summary['mean_reward_uav_confidence']:.4f} "
        f"team_conf={summary['mean_reward_uav_team_confidence']:.4f} "
        f"team_conf_ov_pen={summary['mean_penalty_uav_team_confidence_overlap']:.4f} "
        f"conf_move={summary['mean_reward_uav_confidence_move']:.4f} "
        f"cleanup={summary['mean_reward_uav_cleanup_target_progress']:.4f} "
        f"astar={summary['mean_reward_uav_astar_progress']:.4f} "
        f"move_pen={summary['mean_penalty_uav_inefficient_move']:.4f} "
        f"conf_ov_pen={summary['mean_penalty_uav_confidence_overlap']:.4f} "
        f"coverage95={summary['mean_reward_uav_coverage_threshold']:.4f} "
        f"overlap_pen={summary['mean_penalty_uav_overlap']:.4f} "
        f"inter_pen={summary['mean_penalty_uav_inter_overlap']:.4f} "
        f"outside_pen={summary['mean_penalty_uav_outside_footprint']:.4f} "
        f"scout={summary['mean_reward_uav_scout']:.4f} "
        f"team={summary['mean_reward_team']:.4f} "
        f"all_found={summary['mean_reward_all_survivors_found']:.4f} "
        f"aux_net={summary['mean_reward_uav_aux']:.4f} "
        f"frontier_abs_share={summary['mean_frontier_abs_reward_share']:.3f}"
    )
    print(
        "frontier diagnostic means: "
        f"high_progress={summary['mean_frontier_high_progress_step_frac']:.3f} "
        f"high_no_new={summary['mean_frontier_high_progress_no_new_frac']:.3f} "
        f"high_edge={summary['mean_frontier_high_progress_edge_frac']:.3f} "
        f"high_corner={summary['mean_frontier_high_progress_corner_frac']:.3f} "
        f"edge_prog={summary['mean_frontier_edge_progress']:.3f} "
        f"interior_prog={summary['mean_frontier_interior_progress']:.3f} "
        f"edge_new={summary['mean_frontier_edge_new_cells']:.1f} "
        f"interior_new={summary['mean_frontier_interior_new_cells']:.1f} "
        f"edge_outside={summary['mean_frontier_edge_outside']:.3f} "
        f"interior_outside={summary['mean_frontier_interior_outside']:.3f} "
        f"corr_prog_new={summary['mean_frontier_progress_new_cells_corr']:.3f} "
        f"corr_prog_boundary={summary['mean_frontier_progress_boundary_distance_corr']:.3f}"
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
        f"bbox_hole={summary['mean_coverage_bbox_hole_fraction']:.3f} "
        f"center={summary['mean_coverage_center_fraction']:.3f} "
        f"border_band={summary['mean_coverage_border_band_fraction']:.3f} "
        f"interior={summary['mean_coverage_interior_fraction']:.3f} "
        f"edge_bias={summary['mean_coverage_edge_bias']:.3f} "
        f"uncovered_comp={summary['mean_coverage_uncovered_component_count']:.1f} "
        f"enclosed_comp={summary['mean_coverage_enclosed_uncovered_component_count']:.1f} "
        f"enclosed_frac={summary['mean_coverage_enclosed_uncovered_fraction']:.3f} "
        f"hole_share={summary['mean_coverage_enclosed_hole_share']:.3f} "
        f"largest_hole={summary['mean_coverage_largest_enclosed_hole_fraction']:.3f} "
        f"largest_uncovered={summary['mean_coverage_largest_uncovered_component_fraction']:.3f}"
    )
    print(
        "failure-mode fractions: "
        f"low_action_high_motion={summary['mean_low_action_high_motion_frac']:.3f} "
        f"high_action_low_motion={summary['mean_high_action_low_motion_frac']:.3f} "
        f"moving_no_new_coverage={summary['mean_moving_no_new_coverage_frac']:.3f} "
        f"moving_no_confidence_gain={summary['mean_moving_no_confidence_gain_frac']:.3f}"
    )
    print(
        "start means: "
        f"min_pair={summary['mean_min_start_pair_distance_m']:.1f}m "
        f"min_edge={summary['mean_min_start_edge_distance_m']:.1f}m"
    )
    if summary.get("per_drone"):
        print("per-drone means:")
        for line in _format_per_drone_summary(summary["per_drone"]):
            print(line)
    if summary.get("time_bins"):
        print("time-bin means:")
        for line in _format_time_bin_summary(summary["time_bins"]):
            print(line)
    if summary.get("perception_time_bins"):
        print("survivor perception time-bins:")
        for line in _format_perception_time_bin_summary(summary["perception_time_bins"]):
            print(line)
    if summary.get("scout_time_bins"):
        print("survivor discovery time-bins:")
        for line in _format_scout_time_bin_summary(summary["scout_time_bins"]):
            print(line)
    if summary.get("survivor_exposure_outcomes"):
        print("scouted/missed survivor exposure splits:")
        for line in _format_survivor_exposure_outcomes(summary["survivor_exposure_outcomes"]):
            print(line)
    if summary.get("outcome_splits"):
        print("success/failure diagnostic splits:")
        for line in _format_outcome_split_summary(summary["outcome_splits"]):
            print(line)
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
        f"inter p25/p50/p75="
        f"{summary['inter_uav_overlap_p25']:.3f}/{summary['inter_uav_overlap_p50']:.3f}/{summary['inter_uav_overlap_p75']:.3f} "
        f"frontier p25/p50/p75="
        f"{summary['frontier_progress_p25']:.3f}/{summary['frontier_progress_p50']:.3f}/{summary['frontier_progress_p75']:.3f} "
        f"edge p25/p50/p75="
        f"{summary['edge_frac_p25']:.3f}/{summary['edge_frac_p50']:.3f}/{summary['edge_frac_p75']:.3f}"
    )
    print("note: all_scouted_successes averages only episodes that scouted every survivor.")

    if args.json_output:
        output = {
            "checkpoint": str(checkpoint_dir),
            "diagnostic_level": args.diagnostic_level,
            "scenario_kwargs": scenario_kwargs,
            "rows": rows,
            "summary": summary,
            "label_counts": label_counts,
        }
        Path(args.json_output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"wrote: {args.json_output}")

    if args.plots_output:
        write_distribution_plots(
            rows,
            summary,
            label_counts,
            args.plots_output,
            diagnostic_level=args.diagnostic_level,
        )
        print(f"wrote plots: {args.plots_output}")


if __name__ == "__main__":
    main()
