"""Fast HAPPO UAV survivor-scouting diagnostics."""

from __future__ import annotations

import argparse
import copy
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

TIME_BIN_COUNT = 5
CONFIDENCE_REVISIT_THRESHOLD = 0.10
CONFIDENCE_REVISIT_USEFUL_OPPORTUNITY_THRESHOLD = 0.25
CONFIDENCE_REVISIT_WASTEFUL_OPPORTUNITY_THRESHOLD = 0.15
CONFIDENCE_REVISIT_MIN_GAIN = 1e-9
DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD = 1e-6
FIRE_DIAGNOSTIC_FOOTPRINT_THRESHOLD = 0.01
FIRE_DIAGNOSTIC_MATERIAL_FOOTPRINT_THRESHOLD = 0.10


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


def _schema_count(
    scenario_kwargs: dict[str, Any],
    schema_key: str,
    physical_key: str,
    fallback: int,
) -> int:
    return max(int(scenario_kwargs.get(schema_key, scenario_kwargs.get(physical_key, fallback))), 0)


def _scenario_kwargs(checkpoint_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(checkpoint_dir)
    scenario_kwargs: dict[str, Any] = {}
    if manifest is not None:
        scenario_kwargs.update(copy.deepcopy(manifest.get("env_args", {}).get("scenario_kwargs", {})))
    has_manifest = manifest is not None
    checkpoint_n_drones = max(int(scenario_kwargs.get("n_drones", 1 if not has_manifest else 0)), 0)
    checkpoint_schema_n_drones = _schema_count(
        scenario_kwargs,
        "obs_schema_n_drones",
        "n_drones",
        checkpoint_n_drones if checkpoint_n_drones > 0 else 1,
    )
    checkpoint_schema_n_ground = _schema_count(
        scenario_kwargs,
        "obs_schema_n_ground",
        "n_ground",
        0,
    )
    checkpoint_schema_n_survivors = _schema_count(
        scenario_kwargs,
        "obs_schema_n_survivors",
        "n_survivors",
        5,
    )

    for key in (
        "known_survivor_spawn_distance_m",
        "known_survivor_spawn_distance_min_m",
        "known_survivor_spawn_distance_max_m",
        "survivor_spawn_reference",
    ):
        scenario_kwargs.pop(key, None)

    scenario_kwargs.update({
        "max_steps": args.steps,
        "n_drones": int(
            args.n_drones
            if getattr(args, "n_drones", None) is not None
            else checkpoint_n_drones
        ),
        "n_ground": int(0 if getattr(args, "n_ugvs", None) is None else args.n_ugvs),
        "n_survivors": int(
            args.n_survivors
            if getattr(args, "n_survivors", None) is not None
            else scenario_kwargs.get("n_survivors", checkpoint_schema_n_survivors)
        ),
        "known_survivors_at_reset": False,
        "drone_can_confirm": True,
        "comms_dropout": float(
            0.0
            if getattr(args, "comms_dropout", None) is None
            else args.comms_dropout
        ),
        "uav_confidence_diagnostics": True,
        "uav_cleanup_target_diagnostics": False,
    })
    scenario_kwargs["obs_schema_n_drones"] = int(
        args.n_drones
        if getattr(args, "n_drones", None) is not None
        else checkpoint_schema_n_drones
    )
    scenario_kwargs["obs_schema_n_ground"] = int(
        args.n_ugvs
        if getattr(args, "n_ugvs", None) is not None
        else checkpoint_schema_n_ground
    )
    scenario_kwargs["obs_schema_n_survivors"] = int(
        args.n_survivors
        if getattr(args, "n_survivors", None) is not None
        else checkpoint_schema_n_survivors
    )
    scenario_kwargs.setdefault("disable_fire", True)
    fire_override = getattr(args, "enable_fire", None)
    if fire_override is not None:
        scenario_kwargs["disable_fire"] = not bool(fire_override)
    if getattr(args, "comms_dropout_mode", None) is not None:
        scenario_kwargs["comms_dropout_mode"] = str(args.comms_dropout_mode).replace("-", "_")
    joint_observation_schema = bool(
        getattr(args, "joint_schema_uav_diagnostic", False)
        or getattr(args, "joint_observation_schema", False)
        or int(scenario_kwargs.get("obs_schema_n_ground", 0)) > 0
    )
    if joint_observation_schema:
        scenario_kwargs.setdefault("ugv_planner_hint", "global_astar")
        scenario_kwargs.setdefault("ugv_assigned_target_obs_only", False)
        scenario_kwargs.setdefault("survivor_assignment_obs", True)

    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if args.local_map_patch_size is not None:
        scenario_kwargs["local_map_patch_size"] = int(args.local_map_patch_size)
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
    if getattr(args, "drone_safety_clearance_by_object_m", None) is not None:
        scenario_kwargs["drone_safety_clearance_by_object_m"] = tuple(
            float(v) for v in args.drone_safety_clearance_by_object_m
        )
    if getattr(args, "drone_flight_levels_m", None):
        levels = tuple(
            float(value)
            for value in str(args.drone_flight_levels_m).split(",")
            if value.strip()
        )
        if levels:
            scenario_kwargs["drone_flight_levels_m"] = levels
    for attr in (
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
    scenario_kwargs["comms_map_mode"] = "per_agent"
    return scenario_kwargs


def run_rollout(
    policy: HappoPolicy,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    moving_no_confidence_gain_threshold: float = DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
    diagnostic_level: str = "fast",
) -> dict[str, Any]:
    if str(diagnostic_level).replace("-", "_").lower() != "fast":
        raise ValueError("diagnose_uav_happo.py only supports fast diagnostics")

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
    start_metrics = _start_metrics(scenario)

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
    new_coverage_cells_values: list[float] = []
    raw_new_coverage_cells_values: list[float] = []
    outside_footprint_values: list[float] = []
    fire_footprint_values: list[float] = []
    overlap_values: list[float] = []
    expected_overlap_values: list[float] = []
    excess_overlap_values: list[float] = []
    inter_uav_overlap_values: list[float] = []
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
    fire_confidence_gain_values: list[float] = []
    nonfire_confidence_gain_values: list[float] = []
    fire_confidence_weighted_gain_values: list[float] = []
    nonfire_confidence_weighted_gain_values: list[float] = []
    fire_confidence_reward_values: list[float] = []
    nonfire_confidence_reward_values: list[float] = []
    fire_confidence_positive_values: list[float] = []
    fire_step_values: list[float] = []
    fire_repeat_step_values: list[float] = []
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
    penalty_uav_fire_footprint_values: list[float] = []
    reward_uav_coverage_threshold_values: list[float] = []
    reward_uav_scout_values: list[float] = []
    reward_team_values: list[float] = []
    reward_all_survivors_found_values: list[float] = []
    reward_uav_aux_values: list[float] = []
    boundary_distance_m_values: list[float] = []
    footprint_radius_m_values: list[float] = []
    path_positions_sim: list[np.ndarray] = []
    fast_drone_path_lengths = np.zeros(int(scenario.n_drones), dtype=float)
    fast_drone_displacements: list[list[float]] = [[] for _ in range(int(scenario.n_drones))]
    fast_drone_action_norms: list[list[float]] = [[] for _ in range(int(scenario.n_drones))]
    fire_streak_lengths = np.zeros(int(scenario.n_drones), dtype=int)
    max_fire_streak_by_drone = np.zeros(int(scenario.n_drones), dtype=int)

    low_action_high_motion = 0
    high_action_low_motion = 0
    moving_no_new_coverage = 0
    moving_no_confidence_gain = 0
    diagnostic_steps = 0
    time_bins = _new_time_bins(TIME_BIN_COUNT)
    scout_auc_sum = 0.0
    confirm_auc_sum = 0.0
    coverage_auc_sum = 0.0
    confidence_auc_sum = 0.0
    auc_steps = 0

    def _record_fire_confidence_diagnostic(
        drone_idx: int,
        *,
        fire_fraction: float,
        confidence_gain: float,
        confidence_weighted_gain: float,
        confidence_reward: float,
    ) -> tuple[float, float]:
        fire_step = bool(fire_fraction >= FIRE_DIAGNOSTIC_FOOTPRINT_THRESHOLD)
        repeat_step = bool(fire_step and fire_streak_lengths[drone_idx] > 0)
        if fire_step:
            fire_streak_lengths[drone_idx] += 1
            max_fire_streak_by_drone[drone_idx] = max(
                int(max_fire_streak_by_drone[drone_idx]),
                int(fire_streak_lengths[drone_idx]),
            )
        else:
            fire_streak_lengths[drone_idx] = 0

        fire_step_values.append(float(fire_step))
        fire_repeat_step_values.append(float(repeat_step))
        if fire_step:
            fire_confidence_gain_values.append(float(confidence_gain))
            fire_confidence_weighted_gain_values.append(float(confidence_weighted_gain))
            fire_confidence_reward_values.append(float(confidence_reward))
            fire_confidence_positive_values.append(
                float(
                    math.isfinite(confidence_weighted_gain)
                    and confidence_weighted_gain > CONFIDENCE_REVISIT_MIN_GAIN
                )
            )
        else:
            nonfire_confidence_gain_values.append(float(confidence_gain))
            nonfire_confidence_weighted_gain_values.append(float(confidence_weighted_gain))
            nonfire_confidence_reward_values.append(float(confidence_reward))
        return float(fire_step), float(repeat_step)

    for step in range(int(scenario_kwargs["max_steps"])):
        prev_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool).copy()
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

        meters_per_sim = 1.0 / max(
            float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()),
            1e-9,
        )
        coverage_cells = _metric_array(scenario, "metric_uav_new_coverage_cells_by_drone", scenario.n_drones)
        outside_footprint_fraction = _metric_array(
            scenario,
            "metric_uav_outside_footprint_fraction_by_drone",
            scenario.n_drones,
        )
        fire_footprint_fraction = _metric_array(
            scenario,
            "metric_uav_fire_footprint_fraction_by_drone",
            scenario.n_drones,
        )
        overlap_fraction = _metric_array(scenario, "metric_uav_overlap_fraction_by_drone", scenario.n_drones)
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
        confidence_mean = _metric_scalar(scenario, "metric_uav_confidence_mean")
        confidence_gain = _metric_scalar(scenario, "metric_uav_confidence_gain")
        confidence_weighted_gain = _metric_scalar(scenario, "metric_uav_weighted_confidence_gain")
        confidence_low_fraction = _metric_scalar(scenario, "metric_uav_confidence_low_fraction")
        confidence_high_fraction = _metric_scalar(scenario, "metric_uav_confidence_high_fraction")
        confidence_step_detection_probability = _metric_scalar(
            scenario,
            "metric_uav_step_detection_probability",
        )
        coverage_fraction_now = float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
        team_reward = _metric_scalar(scenario, "metric_reward_team")
        all_survivors_found_reward = _metric_scalar(scenario, "metric_reward_all_survivors_found")
        coverage_threshold_reward = _metric_scalar(scenario, "metric_reward_uav_coverage_threshold")

        confidence_mean_values.append(confidence_mean)
        confidence_gain_values.append(confidence_gain)
        confidence_weighted_gain_values.append(confidence_weighted_gain)
        confidence_low_fraction_values.append(confidence_low_fraction)
        confidence_high_fraction_values.append(confidence_high_fraction)
        confidence_step_detection_probability_values.append(confidence_step_detection_probability)

        post_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        newly_scouted = post_scouted & ~prev_scouted
        drone_detections = (
            scenario.step_drone_detections[0].detach().cpu().numpy().astype(bool)
            if scenario.n_drones > 0 and survivor_slots > 0
            else np.zeros((scenario.n_drones, survivor_slots), dtype=bool)
        )
        scout_credit = drone_detections & newly_scouted.reshape(1, -1)

        for drone_idx, action_vec in enumerate(action_vectors):
            post_pos = scenario.world.agents[drone_idx].state.pos[0].detach().cpu().numpy().astype(float)
            path_positions_sim.append(post_pos.copy())
            displacement_vec = post_pos - pre_drone_pos[drone_idx]
            action_norm = float(np.linalg.norm(action_vec))
            displacement_m = float(np.linalg.norm(displacement_vec) * meters_per_sim)
            new_cells = float(coverage_cells[drone_idx])
            raw_new_cells = new_cells
            overlap = float(overlap_fraction[drone_idx])
            expected_overlap = float(expected_overlap_fraction[drone_idx])
            excess_overlap = float(excess_overlap_fraction[drone_idx])
            inter_uav_overlap = float(inter_uav_overlap_fraction[drone_idx])
            opportunity_fraction = float(coverage_opportunity_fraction[drone_idx])
            opportunity_available_fraction = float(coverage_opportunity_available_fraction[drone_idx])
            confidence_gain_drone = float(confidence_gain_by_drone[drone_idx])
            confidence_weighted_gain_drone = float(confidence_weighted_gain_by_drone[drone_idx])
            confidence_reward = float(confidence_reward_by_drone[drone_idx])
            team_confidence_reward = float(team_confidence_reward_by_drone[drone_idx])
            team_confidence_overlap_penalty = float(team_confidence_overlap_penalty_by_drone[drone_idx])
            confidence_move_reward = float(confidence_move_reward_by_drone[drone_idx])
            confidence_overlap_penalty = float(confidence_overlap_penalty_by_drone[drone_idx])
            confidence_overlap_fraction = float(confidence_overlap_fraction_by_drone[drone_idx])
            confidence_overlap_regret = float(confidence_overlap_regret_by_drone[drone_idx])
            confidence_opportunity_fraction = float(confidence_opportunity_fraction_by_drone[drone_idx])
            confidence_opportunity_best_gain = float(confidence_opportunity_best_gain_by_drone[drone_idx])
            fire_fraction = float(fire_footprint_fraction[drone_idx])
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
                fire_footprint=fire_fraction,
                coverage_opportunity_fraction=opportunity_fraction,
                coverage_opportunity_available_fraction=opportunity_available_fraction,
                confidence_reward=confidence_reward,
                team_confidence_reward=team_confidence_reward,
                team_confidence_overlap_penalty=team_confidence_overlap_penalty,
                confidence_move_reward=confidence_move_reward,
                confidence_opportunity_fraction=confidence_opportunity_fraction,
                confidence_overlap_penalty=confidence_overlap_penalty,
                cleanup_target_progress_reward=float(cleanup_target_progress_reward_by_drone[drone_idx]),
                astar_progress_reward=float(astar_progress_reward_by_drone[drone_idx]),
                scout_reward=scout_reward,
            )
            reward_terms["team"] = team_reward
            reward_terms["all_survivors_found"] = all_survivors_found_reward
            reward_terms["coverage_threshold"] = coverage_threshold_reward
            fire_step, fire_repeat_step = _record_fire_confidence_diagnostic(
                drone_idx,
                fire_fraction=fire_fraction,
                confidence_gain=confidence_gain_drone,
                confidence_weighted_gain=confidence_weighted_gain_drone,
                confidence_reward=reward_terms["confidence"],
            )

            action_norms.append(action_norm)
            displacement_m_values.append(displacement_m)
            new_coverage_cells_values.append(new_cells)
            raw_new_coverage_cells_values.append(raw_new_cells)
            outside_footprint_values.append(float(outside_footprint_fraction[drone_idx]))
            fire_footprint_values.append(fire_fraction)
            overlap_values.append(overlap)
            expected_overlap_values.append(expected_overlap)
            excess_overlap_values.append(excess_overlap)
            inter_uav_overlap_values.append(inter_uav_overlap)
            coverage_opportunity_fraction_values.append(opportunity_fraction)
            coverage_opportunity_available_fraction_values.append(opportunity_available_fraction)
            confidence_gain_by_drone_values.append(confidence_gain_drone)
            confidence_weighted_gain_by_drone_values.append(confidence_weighted_gain_drone)
            confidence_opportunity_fraction_values.append(confidence_opportunity_fraction)
            confidence_opportunity_best_gain_values.append(confidence_opportunity_best_gain)
            confidence_step_detection_probability_by_drone_values.append(
                float(confidence_step_detection_probability_by_drone[drone_idx])
            )
            confidence_overlap_fraction_values.append(confidence_overlap_fraction)
            confidence_overlap_regret_values.append(confidence_overlap_regret)
            boundary_distance_m_values.append(float(boundary_distance_m[drone_idx]))
            footprint_radius = float(footprint_radius_m[drone_idx])
            footprint_radius_m_values.append(footprint_radius)
            reward_uav_coverage_values.append(reward_terms["coverage"])
            reward_uav_move_coverage_values.append(reward_terms["move_coverage"])
            reward_uav_frontier_values.append(reward_terms["frontier"])
            reward_uav_confidence_values.append(reward_terms["confidence"])
            reward_uav_team_confidence_values.append(reward_terms["team_confidence"])
            penalty_uav_team_confidence_overlap_values.append(reward_terms["team_confidence_overlap_penalty"])
            reward_uav_confidence_move_values.append(reward_terms["confidence_move"])
            reward_uav_cleanup_target_progress_values.append(reward_terms["cleanup_target_progress"])
            reward_uav_astar_progress_values.append(reward_terms["astar_progress"])
            penalty_uav_inefficient_move_values.append(reward_terms["inefficient_move_penalty"])
            penalty_uav_confidence_overlap_values.append(reward_terms["confidence_overlap_penalty"])
            penalty_uav_overlap_values.append(reward_terms["overlap_penalty"])
            penalty_uav_inter_overlap_values.append(reward_terms["inter_uav_overlap_penalty"])
            penalty_uav_outside_footprint_values.append(reward_terms["outside_footprint_penalty"])
            penalty_uav_fire_footprint_values.append(reward_terms["fire_footprint_penalty"])
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

            fast_drone_path_lengths[drone_idx] += displacement_m
            fast_drone_displacements[drone_idx].append(displacement_m)
            fast_drone_action_norms[drone_idx].append(action_norm)

            action_displacement_alignment = math.nan
            displacement_norm_sim = float(np.linalg.norm(displacement_vec))
            if action_norm > 1e-6 and displacement_norm_sim > 1e-9:
                action_displacement_alignment = float(
                    np.dot(action_vec[:2], displacement_vec[:2])
                    / (action_norm * displacement_norm_sim)
                )
                action_displacement_alignment = max(min(action_displacement_alignment, 1.0), -1.0)
                action_displacement_alignments.append(action_displacement_alignment)

            edge_threshold = footprint_radius if math.isfinite(footprint_radius) and footprint_radius > 0.0 else 25.0
            is_edge_step = bool(float(boundary_distance_m[drone_idx]) <= edge_threshold)
            diagnostic_steps += 1
            _append_time_bin(
                time_bins,
                step=step,
                max_steps=int(scenario_kwargs["max_steps"]),
                values={
                    "action_norm": action_norm,
                    "displacement_m": displacement_m,
                    "new_coverage_cells": new_cells,
                    "raw_new_coverage_cells": raw_new_cells,
                    "action_displacement_alignment": action_displacement_alignment,
                    "overlap": overlap,
                    "excess_overlap": excess_overlap,
                    "fire_footprint": fire_fraction,
                    "fire_step": fire_step,
                    "fire_repeat_step": fire_repeat_step,
                    "fire_confidence_weighted_gain": confidence_weighted_gain_drone if fire_step >= 0.5 else 0.0,
                    "nonfire_confidence_weighted_gain": confidence_weighted_gain_drone if fire_step < 0.5 else 0.0,
                    "fire_confidence_reward": reward_terms["confidence"] if fire_step >= 0.5 else 0.0,
                    "fire_confidence_positive": float(
                        fire_step >= 0.5
                        and math.isfinite(confidence_weighted_gain_drone)
                        and confidence_weighted_gain_drone > CONFIDENCE_REVISIT_MIN_GAIN
                    ),
                    "edge_step": float(is_edge_step),
                    "moving_no_new_coverage": float(displacement_m > 1.0 and new_cells < 1.0),
                    "moving_no_confidence_gain": float(moving_no_confidence_gain_step),
                    "confidence_overlap_fraction": confidence_overlap_fraction,
                    "confidence_overlap_regret": confidence_overlap_regret,
                    "frontier_reward": reward_terms["frontier"],
                    "confidence_reward": reward_terms["confidence"],
                    "team_confidence_reward": reward_terms["team_confidence"],
                    "team_confidence_overlap_penalty": reward_terms["team_confidence_overlap_penalty"],
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
                    "fire_footprint_penalty": reward_terms["fire_footprint_penalty"],
                    "coverage_threshold_reward": reward_terms["coverage_threshold"],
                    "scout_reward": reward_terms["scout"],
                    "team_reward": reward_terms["team"],
                    "all_survivors_reward": reward_terms["all_survivors_found"],
                },
            )

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
        if n_active_survivors > 0:
            scout_auc_sum += float(np.logical_and(active_survivor_mask, scouted).sum() / n_active_survivors)
            confirm_auc_sum += float(np.logical_and(active_survivor_mask, confirmed).sum() / n_active_survivors)
        else:
            scout_auc_sum += 1.0
            confirm_auc_sum += 1.0
        coverage_auc_sum += coverage_fraction_now
        confidence_auc_sum += confidence_mean
        auc_steps += 1
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
    final_confidence_mean = float(scenario.uav_confidence_grid[0].float().mean().detach().cpu().item())
    max_steps = int(scenario_kwargs["max_steps"])
    if auc_steps < max_steps:
        final_scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        final_confirmed = scenario.found_survivors[0].detach().cpu().numpy().astype(bool)
        if n_active_survivors > 0:
            final_scout_recall = float(np.logical_and(active_survivor_mask, final_scouted).sum() / n_active_survivors)
            final_confirm_recall = float(np.logical_and(active_survivor_mask, final_confirmed).sum() / n_active_survivors)
        else:
            final_scout_recall = 1.0
            final_confirm_recall = 1.0
        remaining_steps = max_steps - auc_steps
        scout_auc_sum += final_scout_recall * remaining_steps
        confirm_auc_sum += final_confirm_recall * remaining_steps
        coverage_auc_sum += final_coverage_fraction * remaining_steps
        confidence_auc_sum += final_confidence_mean * remaining_steps
    final_coverage_grid = scenario.coverage_grid[0].detach().cpu().numpy().astype(bool)
    path_metrics = _path_metrics(
        path_positions_sim,
        displacement_m_values,
        boundary_distance_m_values,
        footprint_radius_m_values,
        scenario,
    )
    coverage_shape_metrics = _fast_coverage_shape_metrics(final_coverage_grid)
    per_drone = [
        {
            "drone": int(drone_idx),
            "path_length_m": float(fast_drone_path_lengths[drone_idx]),
            "avg_displacement_m": _finite_mean(fast_drone_displacements[drone_idx]),
            "avg_action_norm": _finite_mean(fast_drone_action_norms[drone_idx]),
        }
        for drone_idx in range(int(scenario.n_drones))
    ]
    final_survivor_confidence = _survivor_confidence_values(scenario)
    fire_weighted_gain_sum = float(
        sum(max(float(value), 0.0) for value in fire_confidence_weighted_gain_values if math.isfinite(float(value)))
    )
    nonfire_weighted_gain_sum = float(
        sum(max(float(value), 0.0) for value in nonfire_confidence_weighted_gain_values if math.isfinite(float(value)))
    )
    total_weighted_gain_sum = fire_weighted_gain_sum + nonfire_weighted_gain_sum
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
        "scout_auc": float(scout_auc_sum / max(max_steps, 1)),
        "confirmation_auc": float(confirm_auc_sum / max(max_steps, 1)),
        "coverage_auc": float(coverage_auc_sum / max(max_steps, 1)),
        "confidence_auc": float(confidence_auc_sum / max(max_steps, 1)),
        "final_coverage_fraction": final_coverage_fraction,
        "final_confidence_mean": final_confidence_mean,
        "final_confidence_low_fraction": float((scenario.uav_confidence_grid[0] < 0.50).float().mean().detach().cpu().item()),
        "final_confidence_high_fraction": float((scenario.uav_confidence_grid[0] >= 0.80).float().mean().detach().cpu().item()),
        "final_survivor_confidence": final_survivor_confidence,
        "full_success": float(scouted_count == n_active_survivors),
        "full_confirmation_success": float(confirmed_count == n_active_survivors),
        "avg_scout_step": float(np.mean(scout_steps)) if scout_steps else math.nan,
        "avg_scout_time_s": float(np.mean(scout_steps) * step_seconds) if scout_steps else math.nan,
        "all_scouted_step": all_scouted_step,
        "all_scouted_time_s": None if all_scouted_step is None else float(all_scouted_step * step_seconds),
        "first_scout_steps": first_scout_steps,
        "first_scout_times_s": [None if value is None else float(value * step_seconds) for value in first_scout_steps],
        "avg_confirm_step": float(np.mean(confirm_steps)) if confirm_steps else math.nan,
        "avg_confirm_time_s": float(np.mean(confirm_steps) * step_seconds) if confirm_steps else math.nan,
        "all_confirmed_step": all_confirmed_step,
        "all_confirmed_time_s": None if all_confirmed_step is None else float(all_confirmed_step * step_seconds),
        "first_confirm_steps": first_confirm_steps,
        "first_confirm_times_s": [None if value is None else float(value * step_seconds) for value in first_confirm_steps],
        "avg_action_norm": _finite_mean(action_norms),
        "avg_displacement_m": _finite_mean(displacement_m_values),
        "avg_action_displacement_alignment": _finite_mean(action_displacement_alignments),
        "avg_new_coverage_cells": _finite_mean(new_coverage_cells_values),
        "avg_raw_new_coverage_cells": _finite_mean(raw_new_coverage_cells_values),
        "avg_outside_footprint_fraction": _finite_mean(outside_footprint_values),
        "max_outside_footprint_fraction": max(outside_footprint_values) if outside_footprint_values else 0.0,
        "outside_footprint_step_frac_10": float(np.mean([value >= 0.10 for value in outside_footprint_values])) if outside_footprint_values else 0.0,
        "avg_fire_footprint_fraction": _finite_mean(fire_footprint_values),
        "max_fire_footprint_fraction": max(fire_footprint_values) if fire_footprint_values else 0.0,
        "fire_footprint_step_frac_10": float(np.mean([value >= FIRE_DIAGNOSTIC_MATERIAL_FOOTPRINT_THRESHOLD for value in fire_footprint_values])) if fire_footprint_values else 0.0,
        "fire_footprint_step_frac_01": _finite_mean(fire_step_values),
        "fire_repeat_step_frac": _finite_mean(fire_repeat_step_values),
        "fire_confidence_positive_frac": _finite_mean(fire_confidence_positive_values),
        "fire_confidence_gain_share": fire_weighted_gain_sum / total_weighted_gain_sum if total_weighted_gain_sum > 1e-12 else 0.0,
        "avg_confidence_gain_on_fire": _finite_mean(fire_confidence_gain_values),
        "avg_confidence_gain_off_fire": _finite_mean(nonfire_confidence_gain_values),
        "avg_confidence_weighted_gain_on_fire": _finite_mean(fire_confidence_weighted_gain_values),
        "avg_confidence_weighted_gain_off_fire": _finite_mean(nonfire_confidence_weighted_gain_values),
        "avg_reward_uav_confidence_on_fire": _finite_mean(fire_confidence_reward_values),
        "avg_reward_uav_confidence_off_fire": _finite_mean(nonfire_confidence_reward_values),
        "max_fire_streak_steps": int(max_fire_streak_by_drone.max()) if max_fire_streak_by_drone.size else 0,
        "mean_max_fire_streak_steps_by_drone": float(max_fire_streak_by_drone.mean()) if max_fire_streak_by_drone.size else 0.0,
        "avg_overlap_fraction": _finite_mean(overlap_values),
        "avg_expected_overlap_fraction": _finite_mean(expected_overlap_values),
        "avg_excess_overlap_fraction": _finite_mean(excess_overlap_values),
        "avg_inter_uav_overlap_fraction": _finite_mean(inter_uav_overlap_values),
        "avg_coverage_opportunity_fraction": _finite_mean(coverage_opportunity_fraction_values),
        "avg_coverage_opportunity_available_fraction": _finite_mean(coverage_opportunity_available_fraction_values),
        "avg_confidence_mean": _finite_mean(confidence_mean_values),
        "avg_confidence_gain": _finite_mean(confidence_gain_values),
        "avg_confidence_gain_by_drone": _finite_mean(confidence_gain_by_drone_values),
        "avg_confidence_weighted_gain": _finite_mean(confidence_weighted_gain_values),
        "avg_confidence_weighted_gain_by_drone": _finite_mean(confidence_weighted_gain_by_drone_values),
        "avg_confidence_opportunity_fraction": _finite_mean(confidence_opportunity_fraction_values),
        "avg_confidence_opportunity_best_gain": _finite_mean(confidence_opportunity_best_gain_values),
        "avg_confidence_low_fraction": _finite_mean(confidence_low_fraction_values),
        "avg_confidence_high_fraction": _finite_mean(confidence_high_fraction_values),
        "avg_confidence_step_detection_probability": _finite_mean(confidence_step_detection_probability_values),
        "avg_confidence_pass_probability": _finite_mean(confidence_step_detection_probability_by_drone_values),
        "avg_confidence_overlap_fraction": _finite_mean(confidence_overlap_fraction_values),
        "avg_confidence_overlap_regret": _finite_mean(confidence_overlap_regret_values),
        "avg_reward_uav_coverage": _finite_mean(reward_uav_coverage_values),
        "avg_reward_uav_move_coverage": _finite_mean(reward_uav_move_coverage_values),
        "avg_reward_uav_frontier": _finite_mean(reward_uav_frontier_values),
        "avg_reward_uav_confidence": _finite_mean(reward_uav_confidence_values),
        "avg_reward_uav_team_confidence": _finite_mean(reward_uav_team_confidence_values),
        "avg_penalty_uav_team_confidence_overlap": _finite_mean(penalty_uav_team_confidence_overlap_values),
        "avg_reward_uav_confidence_move": _finite_mean(reward_uav_confidence_move_values),
        "avg_reward_uav_cleanup_target_progress": _finite_mean(reward_uav_cleanup_target_progress_values),
        "avg_reward_uav_astar_progress": _finite_mean(reward_uav_astar_progress_values),
        "avg_penalty_uav_inefficient_move": _finite_mean(penalty_uav_inefficient_move_values),
        "avg_penalty_uav_confidence_overlap": _finite_mean(penalty_uav_confidence_overlap_values),
        "avg_penalty_uav_overlap": _finite_mean(penalty_uav_overlap_values),
        "avg_penalty_uav_inter_overlap": _finite_mean(penalty_uav_inter_overlap_values),
        "avg_penalty_uav_outside_footprint": _finite_mean(penalty_uav_outside_footprint_values),
        "avg_penalty_uav_fire_footprint": _finite_mean(penalty_uav_fire_footprint_values),
        "avg_reward_uav_coverage_threshold": _finite_mean(reward_uav_coverage_threshold_values),
        "avg_reward_uav_scout": _finite_mean(reward_uav_scout_values),
        "avg_reward_team": _finite_mean(reward_team_values),
        "avg_reward_all_survivors_found": _finite_mean(reward_all_survivors_found_values),
        "avg_reward_uav_aux": _finite_mean(reward_uav_aux_values),
        "excess_overlap_step_frac_10": float(np.mean([value >= 0.10 for value in excess_overlap_values])) if excess_overlap_values else 0.0,
        "inter_uav_overlap_step_frac_20": float(np.mean([value >= 0.20 for value in inter_uav_overlap_values])) if inter_uav_overlap_values else 0.0,
        "excess_overlap_step_frac_20": float(np.mean([value >= 0.20 for value in excess_overlap_values])) if excess_overlap_values else 0.0,
        "overlap_step_frac_60": float(np.mean([value >= 0.60 for value in overlap_values])) if overlap_values else 0.0,
        "new_coverage_step_frac": float(np.mean([value >= 1.0 for value in new_coverage_cells_values])) if new_coverage_cells_values else 0.0,
        "raw_new_coverage_step_frac": float(np.mean([value >= 1.0 for value in raw_new_coverage_cells_values])) if raw_new_coverage_cells_values else 0.0,
        "low_action_high_motion_frac": low_action_high_motion / diagnostic_steps if diagnostic_steps else 0.0,
        "high_action_low_motion_frac": high_action_low_motion / diagnostic_steps if diagnostic_steps else 0.0,
        "moving_no_new_coverage_frac": moving_no_new_coverage / diagnostic_steps if diagnostic_steps else 0.0,
        "moving_no_confidence_gain_frac": moving_no_confidence_gain / diagnostic_steps if diagnostic_steps else 0.0,
        "time_bins": _finalize_time_bins(time_bins),
        "perception_time_bins": [],
        "per_drone": per_drone,
        **start_metrics,
        **path_metrics,
        **coverage_shape_metrics,
    }
    row.update(_confidence_revisit_metrics(
        excess_overlap_values,
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
    fire_footprint: float,
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
    fire_scale = float(getattr(scenario, "r_uav_fire_footprint", 0.0))
    fire_threshold = float(getattr(scenario, "uav_fire_penalty_threshold", 0.6))
    fire_penalty = (
        -fire_scale * min(max(fire_footprint, 0.0), 1.0)
        if fire_scale > 0.0 and fire_threshold >= 0.0
        else 0.0
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
        + fire_penalty
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
        + abs(fire_penalty)
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
        "fire_footprint_penalty": float(fire_penalty),
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
    confirmation_successful = [row for row in rows if row.get("all_confirmed_step") is not None]

    def _percentiles(prefix: str, key: str) -> dict[str, float]:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))],
            dtype=float,
        )
        if values.size == 0:
            return {f"{prefix}_p{q}": math.nan for q in (0, 25, 50, 75, 100)}
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        return {f"{prefix}_p{q}": float(value) for q, value in zip((0, 25, 50, 75, 100), percentiles)}

    summary: dict[str, Any] = {
        "episodes": float(len(rows)),
        "mean_scouted": float(np.mean([row["scouted"] for row in rows])) if rows else 0.0,
        "mean_missed": float(np.mean([row["missed"] for row in rows])) if rows else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
        "mean_confirmed": float(np.mean([row.get("confirmed", 0.0) for row in rows])) if rows else 0.0,
        "mean_unconfirmed": float(np.mean([row.get("unconfirmed", 0.0) for row in rows])) if rows else 0.0,
        "mean_confirmation_recall": float(np.mean([row.get("confirmation_recall", 0.0) for row in rows])) if rows else 0.0,
        "mean_scout_auc": _finite_mean([row.get("scout_auc", math.nan) for row in rows]),
        "mean_confirmation_auc": _finite_mean([row.get("confirmation_auc", math.nan) for row in rows]),
        "mean_coverage_auc": _finite_mean([row.get("coverage_auc", math.nan) for row in rows]),
        "mean_confidence_auc": _finite_mean([row.get("confidence_auc", math.nan) for row in rows]),
        "mean_final_coverage_fraction": _finite_mean([row["final_coverage_fraction"] for row in rows]),
        "mean_final_confidence_mean": _finite_mean([row["final_confidence_mean"] for row in rows]),
        "mean_final_confidence_low_fraction": _finite_mean([row["final_confidence_low_fraction"] for row in rows]),
        "mean_final_confidence_high_fraction": _finite_mean([row["final_confidence_high_fraction"] for row in rows]),
        "full_success_rate": float(np.mean([row["full_success"] for row in rows])) if rows else 0.0,
        "full_confirmation_success_rate": float(np.mean([row.get("full_confirmation_success", 0.0) for row in rows])) if rows else 0.0,
        "mean_avg_scout_step": _finite_mean([row["avg_scout_step"] for row in rows]),
        "mean_avg_scout_time_s": _finite_mean([row["avg_scout_time_s"] for row in rows]),
        "mean_avg_confirm_step": _finite_mean([row.get("avg_confirm_step", math.nan) for row in rows]),
        "mean_avg_confirm_time_s": _finite_mean([row.get("avg_confirm_time_s", math.nan) for row in rows]),
        "mean_all_scouted_step_successes": float(np.mean([row["all_scouted_step"] for row in successful])) if successful else math.nan,
        "mean_all_scouted_time_s_successes": float(np.mean([row["all_scouted_time_s"] for row in successful])) if successful else math.nan,
        "mean_all_confirmed_step_successes": float(np.mean([row["all_confirmed_step"] for row in confirmation_successful])) if confirmation_successful else math.nan,
        "mean_all_confirmed_time_s_successes": float(np.mean([row["all_confirmed_time_s"] for row in confirmation_successful])) if confirmation_successful else math.nan,
        "mean_action_norm": _finite_mean([row["avg_action_norm"] for row in rows]),
        "mean_displacement_m": _finite_mean([row["avg_displacement_m"] for row in rows]),
        "mean_action_displacement_alignment": _finite_mean([row["avg_action_displacement_alignment"] for row in rows]),
        "mean_new_coverage_cells": _finite_mean([row["avg_new_coverage_cells"] for row in rows]),
        "mean_raw_new_coverage_cells": _finite_mean([row["avg_raw_new_coverage_cells"] for row in rows]),
        "mean_outside_footprint_fraction": _finite_mean([row["avg_outside_footprint_fraction"] for row in rows]),
        "mean_outside_footprint_step_frac_10": _finite_mean([row["outside_footprint_step_frac_10"] for row in rows]),
        "mean_fire_footprint_fraction": _finite_mean([row["avg_fire_footprint_fraction"] for row in rows]),
        "mean_fire_footprint_step_frac_10": _finite_mean([row["fire_footprint_step_frac_10"] for row in rows]),
        "mean_fire_footprint_step_frac_01": _finite_mean([row.get("fire_footprint_step_frac_01", math.nan) for row in rows]),
        "mean_fire_repeat_step_frac": _finite_mean([row.get("fire_repeat_step_frac", math.nan) for row in rows]),
        "mean_fire_confidence_positive_frac": _finite_mean([row.get("fire_confidence_positive_frac", math.nan) for row in rows]),
        "mean_fire_confidence_gain_share": _finite_mean([row.get("fire_confidence_gain_share", math.nan) for row in rows]),
        "mean_confidence_weighted_gain_on_fire": _finite_mean([row.get("avg_confidence_weighted_gain_on_fire", math.nan) for row in rows]),
        "mean_confidence_weighted_gain_off_fire": _finite_mean([row.get("avg_confidence_weighted_gain_off_fire", math.nan) for row in rows]),
        "mean_reward_uav_confidence_on_fire": _finite_mean([row.get("avg_reward_uav_confidence_on_fire", math.nan) for row in rows]),
        "mean_reward_uav_confidence_off_fire": _finite_mean([row.get("avg_reward_uav_confidence_off_fire", math.nan) for row in rows]),
        "mean_max_fire_streak_steps": _finite_mean([row.get("max_fire_streak_steps", math.nan) for row in rows]),
        "mean_overlap_fraction": _finite_mean([row["avg_overlap_fraction"] for row in rows]),
        "mean_expected_overlap_fraction": _finite_mean([row["avg_expected_overlap_fraction"] for row in rows]),
        "mean_excess_overlap_fraction": _finite_mean([row["avg_excess_overlap_fraction"] for row in rows]),
        "mean_inter_uav_overlap_fraction": _finite_mean([row["avg_inter_uav_overlap_fraction"] for row in rows]),
        "mean_coverage_opportunity_fraction": _finite_mean([row["avg_coverage_opportunity_fraction"] for row in rows]),
        "mean_coverage_opportunity_available_fraction": _finite_mean([row["avg_coverage_opportunity_available_fraction"] for row in rows]),
        "mean_confidence_mean": _finite_mean([row["avg_confidence_mean"] for row in rows]),
        "mean_confidence_gain": _finite_mean([row["avg_confidence_gain"] for row in rows]),
        "mean_confidence_gain_by_drone": _finite_mean([row["avg_confidence_gain_by_drone"] for row in rows]),
        "mean_confidence_weighted_gain": _finite_mean([row["avg_confidence_weighted_gain"] for row in rows]),
        "mean_confidence_weighted_gain_by_drone": _finite_mean([row["avg_confidence_weighted_gain_by_drone"] for row in rows]),
        "mean_confidence_opportunity_fraction": _finite_mean([row["avg_confidence_opportunity_fraction"] for row in rows]),
        "mean_confidence_opportunity_best_gain": _finite_mean([row["avg_confidence_opportunity_best_gain"] for row in rows]),
        "mean_confidence_low_fraction": _finite_mean([row["avg_confidence_low_fraction"] for row in rows]),
        "mean_confidence_high_fraction": _finite_mean([row["avg_confidence_high_fraction"] for row in rows]),
        "mean_confidence_pass_probability": _finite_mean([row["avg_confidence_pass_probability"] for row in rows]),
        "mean_confidence_overlap_fraction": _finite_mean([row["avg_confidence_overlap_fraction"] for row in rows]),
        "mean_confidence_overlap_regret": _finite_mean([row.get("avg_confidence_overlap_regret", math.nan) for row in rows]),
        "mean_reward_uav_coverage": _finite_mean([row["avg_reward_uav_coverage"] for row in rows]),
        "mean_reward_uav_move_coverage": _finite_mean([row["avg_reward_uav_move_coverage"] for row in rows]),
        "mean_reward_uav_frontier": _finite_mean([row["avg_reward_uav_frontier"] for row in rows]),
        "mean_reward_uav_confidence": _finite_mean([row["avg_reward_uav_confidence"] for row in rows]),
        "mean_reward_uav_team_confidence": _finite_mean([row.get("avg_reward_uav_team_confidence", math.nan) for row in rows]),
        "mean_penalty_uav_team_confidence_overlap": _finite_mean([row.get("avg_penalty_uav_team_confidence_overlap", math.nan) for row in rows]),
        "mean_reward_uav_confidence_move": _finite_mean([row["avg_reward_uav_confidence_move"] for row in rows]),
        "mean_reward_uav_cleanup_target_progress": _finite_mean([row.get("avg_reward_uav_cleanup_target_progress", math.nan) for row in rows]),
        "mean_reward_uav_astar_progress": _finite_mean([row.get("avg_reward_uav_astar_progress", math.nan) for row in rows]),
        "mean_penalty_uav_inefficient_move": _finite_mean([row["avg_penalty_uav_inefficient_move"] for row in rows]),
        "mean_penalty_uav_confidence_overlap": _finite_mean([row["avg_penalty_uav_confidence_overlap"] for row in rows]),
        "mean_penalty_uav_overlap": _finite_mean([row["avg_penalty_uav_overlap"] for row in rows]),
        "mean_penalty_uav_inter_overlap": _finite_mean([row["avg_penalty_uav_inter_overlap"] for row in rows]),
        "mean_penalty_uav_outside_footprint": _finite_mean([row["avg_penalty_uav_outside_footprint"] for row in rows]),
        "mean_penalty_uav_fire_footprint": _finite_mean([row["avg_penalty_uav_fire_footprint"] for row in rows]),
        "mean_reward_uav_coverage_threshold": _finite_mean([row["avg_reward_uav_coverage_threshold"] for row in rows]),
        "mean_reward_uav_scout": _finite_mean([row["avg_reward_uav_scout"] for row in rows]),
        "mean_reward_team": _finite_mean([row["avg_reward_team"] for row in rows]),
        "mean_reward_all_survivors_found": _finite_mean([row["avg_reward_all_survivors_found"] for row in rows]),
        "mean_reward_uav_aux": _finite_mean([row["avg_reward_uav_aux"] for row in rows]),
        "mean_excess_overlap_step_frac_10": _finite_mean([row["excess_overlap_step_frac_10"] for row in rows]),
        "mean_inter_uav_overlap_step_frac_20": _finite_mean([row["inter_uav_overlap_step_frac_20"] for row in rows]),
        "mean_excess_overlap_step_frac_20": _finite_mean([row["excess_overlap_step_frac_20"] for row in rows]),
        "mean_overlap_step_frac_60": _finite_mean([row["overlap_step_frac_60"] for row in rows]),
        "mean_new_coverage_step_frac": _finite_mean([row["new_coverage_step_frac"] for row in rows]),
        "mean_raw_new_coverage_step_frac": _finite_mean([row["raw_new_coverage_step_frac"] for row in rows]),
        "mean_low_action_high_motion_frac": _finite_mean([row["low_action_high_motion_frac"] for row in rows]),
        "mean_high_action_low_motion_frac": _finite_mean([row["high_action_low_motion_frac"] for row in rows]),
        "mean_moving_no_new_coverage_frac": _finite_mean([row["moving_no_new_coverage_frac"] for row in rows]),
        "mean_moving_no_confidence_gain_frac": _finite_mean([row["moving_no_confidence_gain_frac"] for row in rows]),
        "mean_min_start_pair_distance_m": _finite_mean([row["min_start_pair_distance_m"] for row in rows]),
        "mean_min_start_edge_distance_m": _finite_mean([row["min_start_edge_distance_m"] for row in rows]),
        "mean_path_bbox_area_fraction": _finite_mean([row["path_bbox_area_fraction"] for row in rows]),
        "mean_path_length_m": _finite_mean([row["path_length_m"] for row in rows]),
        "mean_boundary_distance_m": _finite_mean([row["mean_boundary_distance_m"] for row in rows]),
        "mean_edge_step_frac": _finite_mean([row["edge_step_frac"] for row in rows]),
        "mean_corner_step_frac": _finite_mean([row["corner_step_frac"] for row in rows]),
        "mean_stalled_step_frac": _finite_mean([row["stalled_step_frac"] for row in rows]),
        "mean_longest_stall_steps": _finite_mean([row["longest_stall_steps"] for row in rows]),
        "mean_coverage_bbox_fill_fraction": _finite_mean([row["coverage_bbox_fill_fraction"] for row in rows]),
        "mean_coverage_bbox_hole_fraction": _finite_mean([row["coverage_bbox_hole_fraction"] for row in rows]),
        "mean_coverage_center_fraction": _finite_mean([row["coverage_center_fraction"] for row in rows]),
    }
    for prefix, key in (
        ("recall", "recall"),
        ("confirmation_recall", "confirmation_recall"),
        ("coverage", "final_coverage_fraction"),
        ("confidence_final", "final_confidence_mean"),
        ("move_m", "avg_displacement_m"),
        ("overlap", "avg_overlap_fraction"),
        ("excess_overlap", "avg_excess_overlap_fraction"),
        ("inter_uav_overlap", "avg_inter_uav_overlap_fraction"),
        ("edge_frac", "edge_step_frac"),
        ("moving_no_new", "moving_no_new_coverage_frac"),
        ("moving_no_conf_gain", "moving_no_confidence_gain_frac"),
    ):
        summary.update(_percentiles(prefix, key))
    summary["per_drone"] = _summarize_per_drone(rows)
    summary["time_bins"] = _summarize_time_bins(rows)
    summary["scout_time_bins"] = _summarize_scout_time_bins(rows)
    return summary

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
        "avg_fire_footprint_fraction",
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
        "avg_penalty_uav_fire_footprint",
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
        ("fire", "fire_step", "#dc2626"),
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
        ("fire pen", "fire_footprint_penalty", "#dc2626", True),
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


def _plot_time_bins_fire_diagnostics(ax: Any, summary: dict[str, Any]) -> None:
    time_bins = summary.get("time_bins", [])
    if not time_bins:
        ax.text(0.5, 0.5, "no time bins", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time-Bin Fire/Confidence", fontsize=10)
        return
    centers = [
        0.5 * (float(row.get("start_fraction", 0.0)) + float(row.get("end_fraction", 0.0)))
        for row in time_bins
    ]
    frac_series = [
        ("fire footprint", "fire_footprint", "#ef4444"),
        ("fire step", "fire_step", "#dc2626"),
        ("repeat fire", "fire_repeat_step", "#991b1b"),
        ("positive fire gain", "fire_confidence_positive", "#f97316"),
    ]
    frac_lines = []
    for label, key, color in frac_series:
        values = [float(row.get(key, math.nan)) for row in time_bins]
        frac_lines.extend(
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
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("fraction")
    ax.grid(alpha=0.25)

    ax_gain = ax.twinx()
    gain_series = [
        ("gain on fire", "fire_confidence_weighted_gain", "#0f766e", "-"),
        ("gain off fire", "nonfire_confidence_weighted_gain", "#14b8a6", "-"),
        ("conf rew on fire", "fire_confidence_reward", "#2563eb", "--"),
        ("fire pen", "fire_footprint_penalty", "#20242c", "--"),
    ]
    gain_lines = []
    max_gain = 0.0
    for label, key, color, linestyle in gain_series:
        values = [abs(float(row.get(key, math.nan))) for row in time_bins]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            max_gain = max(max_gain, max(finite))
        gain_lines.extend(
            ax_gain.plot(
                centers,
                values,
                marker="s",
                linewidth=1.1,
                linestyle=linestyle,
                label=label,
                color=color,
                alpha=0.9,
            )
        )
    if max_gain > 0.0:
        ax_gain.set_ylim(0.0, max_gain * 1.15)
    ax_gain.set_ylabel("abs gain/reward")
    lines = frac_lines + gain_lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, frameon=False)
    ax.set_title("Time-Bin Fire/Confidence", fontsize=10)


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
    _plot_time_bins_fire_diagnostics(axes_flat[custom_start + 6], summary)

    for ax in axes_flat[custom_start + 7:]:
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
    diagnostic_level: str = "fast",
) -> None:
    _write_fast_distribution_plots(rows, summary, label_counts, output_path)


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
                        help="Override physical UGV count for joint checkpoints and UGV schema count for joint-schema UAV diagnostics.")
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
                        help="Override checkpoint/default settings and enable fire/smoke dynamics.")
    parser.add_argument("--disable-fire", dest="enable_fire", action="store_false",
                        help="Override checkpoint/default settings and disable fire/smoke dynamics.")
    parser.set_defaults(enable_fire=None)
    parser.add_argument("--comms-dropout", type=float, default=None,
                        help="Communication dropout probability during evaluation. "
                             "Omitted or 0 preserves the full-communication diagnostic default.")
    parser.add_argument("--comms-dropout-mode", choices=("iid", "bursty"), default=None,
                        help="Communication dropout process. Omitted preserves the checkpoint mode, "
                             "falling back to bursty for legacy checkpoints.")
    parser.add_argument("--drone-perception-mode",
                        choices=("rgb", "rgb_thermal", "rgb+thermal", "rgb-thermal"),
                        default=None,
                        help="Override abstract UAV perception mode. rgb_thermal changes only smoke quality.")
    parser.add_argument("--uav-fire-block-threshold", type=float, default=None,
                        help="If set, mark UAV local blocked-observation cells as blocked when "
                             "fire intensity >= this threshold. Omitted preserves checkpoint/default.")
    parser.add_argument("--uav-fire-footprint-penalty", type=float, default=None,
                        help="Override per-UAV active-fire footprint penalty scale. "
                             "Omitted preserves checkpoint/default.")
    parser.add_argument("--uav-fire-penalty-threshold", type=float, default=None,
                        help="Override active-fire threshold for --uav-fire-footprint-penalty. "
                             "Use a negative value to disable.")
    parser.add_argument("--drone-safety-clearance-by-object-m", type=float, nargs=3, default=None,
                        metavar=("NONE", "TREE", "HOUSE"),
                        help="Override variable UAV safety margins by object: none tree house.")
    parser.add_argument("--drone-flight-levels-m", default=None,
                        help="Comma-separated UAV flight altitudes in meters, e.g. 30,50,75.")
    parser.add_argument("--uav-confidence-reward-grid", type=int, default=None,
                        help="Override UAV confidence reward/penalty/opportunity grid size for diagnostics.")
    parser.add_argument("--uav-frontier-global-grid", type=int, default=None,
                        help="Override the global leg grid size for local-global UAV frontier diagnostics.")
    parser.add_argument("--uav-coverage-reward-grid", type=int, default=None,
                        help="Override UAV binary coverage reward/overlap grid size for diagnostics.")
    parser.add_argument("--diagnostic-level", choices=("fast",), default="fast",
                        help="Fast core diagnostics only.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actor means.")
    parser.add_argument("--json-output", default=None, help="Optional path to write per-seed rows and summary as JSON.")
    parser.add_argument("--plots-output", default=None, help="Optional path to write histogram diagnostics as a PNG.")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.n_drones is not None and args.n_drones < 1:
        parser.error("--n-drones must be positive")
    if args.n_ugvs is not None and args.n_ugvs < 0:
        parser.error("--n-ugvs must be nonnegative")
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
    if args.uav_fire_block_threshold is not None and args.uav_fire_block_threshold > 1.0:
        parser.error("--uav-fire-block-threshold must be <= 1; use a negative value to disable")
    if args.uav_fire_footprint_penalty is not None and args.uav_fire_footprint_penalty < 0.0:
        parser.error("--uav-fire-footprint-penalty must be nonnegative")
    if args.uav_fire_penalty_threshold is not None and args.uav_fire_penalty_threshold > 1.0:
        parser.error("--uav-fire-penalty-threshold must be <= 1; use a negative value to disable")
    if (
        args.drone_safety_clearance_by_object_m is not None
        and any(v < 0.0 for v in args.drone_safety_clearance_by_object_m)
    ):
        parser.error("--drone-safety-clearance-by-object-m values must be nonnegative")
    if (
        args.comms_dropout is not None
        and (
            not math.isfinite(args.comms_dropout)
            or not 0.0 <= args.comms_dropout <= 1.0
        )
    ):
        parser.error("--comms-dropout must be finite and between 0 and 1")
    for arg_name in (
        "uav_confidence_reward_grid",
        "uav_frontier_global_grid",
        "uav_coverage_reward_grid",
    ):
        value = getattr(args, arg_name)
        if value is not None and (value < 0 or value == 1):
            parser.error(f"--{arg_name.replace('_', '-')} must be 0 or at least 2")
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
    if (
        args.joint_schema_uav_diagnostic
        or args.joint_observation_schema
        or int(scenario_kwargs.get("obs_schema_n_drones", 0)) != int(scenario_kwargs.get("n_drones", 0))
        or int(scenario_kwargs.get("obs_schema_n_ground", 0)) != int(scenario_kwargs.get("n_ground", 0))
    ):
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
        f"mode={scenario_kwargs.get('comms_dropout_mode', COMMS_DROPOUT_MODE)} "
        f"burst_steps={scenario_kwargs.get('comms_dropout_min_steps', 5)}"
        f"..{scenario_kwargs.get('comms_dropout_max_steps', 15)}"
    )
    print(f"drone_perception_mode: {scenario_kwargs.get('drone_perception_mode', 'rgb')}")
    print(f"uav_fire_block_threshold: {scenario_kwargs.get('uav_fire_block_threshold', -1.0)}")
    print(f"uav_fire_footprint_penalty: {scenario_kwargs.get('r_uav_fire_footprint', 0.0)}")
    print(f"uav_fire_penalty_threshold: {scenario_kwargs.get('uav_fire_penalty_threshold', 0.6)}")
    print(
        "uav overlap penalty: "
        f"normalization={scenario_kwargs.get('uav_overlap_penalty_normalization', 'raw')}"
    )
    print(
        "diagnostics: "
        f"level={args.diagnostic_level} "
        f"cleanup_target={bool(scenario_kwargs.get('uav_cleanup_target_diagnostics', False))} "
        f"move_no_conf_thr={DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD:g}"
    )
    print("-" * 88)

    if int(scenario_kwargs.get("n_drones", 0)) < 1:
        parser.error("diagnose_uav_happo.py needs at least one physical UAV actor in the checkpoint")
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
    expected_agents = int(scenario_kwargs["n_drones"]) + int(scenario_kwargs["n_ground"])
    if len(policy.actors) != expected_agents:
        parser.error(
            f"checkpoint contains {len(policy.actors)} actors, but diagnostics scenario "
            f"contains {expected_agents} agents; use the checkpoint manifest settings or "
            "matching --n-drones/--n-ugvs overrides for legacy checkpoints"
        )
    json_path = Path(args.json_output) if args.json_output else None
    if json_path is not None:
        print(f"partial JSON checkpoint: {partial_json_path(json_path)}")

    rows = []
    for seed_index, seed in enumerate(args.seeds, start=1):
        rows.append(
            run_rollout(
                policy,
                scenario_kwargs,
                seed,
                moving_no_confidence_gain_threshold=DEFAULT_MOVING_NO_CONFIDENCE_GAIN_THRESHOLD,
                diagnostic_level=args.diagnostic_level,
            )
        )
        if json_path is not None:
            partial_summary = summarize(rows)
            partial_label_counts = _label_counts(rows)
            write_partial_json(
                json_path,
                {
                    "checkpoint": str(checkpoint_dir),
                    "diagnostic_level": args.diagnostic_level,
                    "scenario_kwargs": scenario_kwargs,
                    "rows": rows,
                    "summary": partial_summary,
                    "label_counts": partial_label_counts,
                },
                completed_rollouts=seed_index,
                total_rollouts=len(args.seeds),
            )
    for row in rows:
        print(
            f"seed {row['seed']:>4}: "
            f"scouted={row['scouted']}/{row['survivors']} "
            f"missed={row['missed']} "
            f"recall={row['recall']:.3f} "
            f"confirmed={row['confirmed']}/{row['survivors']} "
            f"confirm_recall={row['confirmation_recall']:.3f} "
            f"coverage={row['final_coverage_fraction']:.3f} "
            f"conf={row['final_confidence_mean']:.3f} "
            f"avg_scout={_fmt_optional(row['avg_scout_step'])} steps/"
            f"{_fmt_optional(row['avg_scout_time_s'])}s "
            f"all_scouted={_fmt_optional(row['all_scouted_step'])} steps/"
            f"{_fmt_optional(row['all_scouted_time_s'])}s "
            f"move={row['avg_displacement_m']:.2f}m "
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
        f"confirmed={summary['mean_confirmed']:.3f} "
        f"confirm_recall={summary['mean_confirmation_recall']:.3f} "
        f"scout_auc={summary['mean_scout_auc']:.3f} "
        f"confirm_auc={summary['mean_confirmation_auc']:.3f} "
        f"coverage_auc={summary['mean_coverage_auc']:.3f} "
        f"confidence_auc={summary['mean_confidence_auc']:.3f} "
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
        "fire diagnostic means: "
        f"fire01={summary['mean_fire_footprint_step_frac_01']:.3f} "
        f"fire10={summary['mean_fire_footprint_step_frac_10']:.3f} "
        f"repeat={summary['mean_fire_repeat_step_frac']:.3f} "
        f"positive_gain={summary['mean_fire_confidence_positive_frac']:.3f} "
        f"gain_share={summary['mean_fire_confidence_gain_share']:.3f} "
        f"gain_on/off="
        f"{summary['mean_confidence_weighted_gain_on_fire']:.6f}/"
        f"{summary['mean_confidence_weighted_gain_off_fire']:.6f} "
        f"conf_rew_on/off="
        f"{summary['mean_reward_uav_confidence_on_fire']:.4f}/"
        f"{summary['mean_reward_uav_confidence_off_fire']:.4f} "
        f"max_streak={summary['mean_max_fire_streak_steps']:.1f}"
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

    if json_path is not None:
        output = {
            "checkpoint": str(checkpoint_dir),
            "diagnostic_level": args.diagnostic_level,
            "scenario_kwargs": scenario_kwargs,
            "rows": rows,
            "summary": summary,
            "label_counts": label_counts,
        }
        write_final_json(
            json_path,
            output,
            completed_rollouts=len(rows),
            total_rollouts=len(args.seeds),
        )
        print(f"wrote: {json_path}")

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
