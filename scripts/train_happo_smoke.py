"""
HAPPO training on the WildfireSearchScenario via HARL.

HARL doesn't ship a VMAS interface, so this script:

  1. Monkey-patches HARL's env registry to recognise env_name="wildfire"
  2. Registers a minimal logger so HARL's runner doesn't fall over
  3. Builds algo_args + env_args from CLI flags
  4. Runs HARL's OnPolicyHARunner

Budgets:
  smoke    (default)    ~2 000 steps, episode_length=150   ≈ 5-10 s on CPU
  research (--research)  400 000 steps, episode_length=500 ≈ tens of minutes on CPU

Run from repo root:

    python scripts/train_happo_smoke.py
    python scripts/train_happo_smoke.py --research
    python scripts/train_happo_smoke.py --research --preset tuned
    python scripts/train_happo_smoke.py --num-env-steps 20000 --comms-dropout 0.3
    python scripts/train_happo_smoke.py --entropy-coef 0.05 --seed 42

Prerequisite: HARL must be installed in the active venv:

    pip install -e ".[happo]"

or, for a local editable HARL checkout:

    git clone https://github.com/PKU-MARL/HARL ../HARL
    pip install -e ../HARL
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.harl_terrain_cnn import (
    TERRAIN_CNN_CHANNELS,
    TERRAIN_CNN_OBS_OFFSET,
    wildfire_single_observation_dim,
)

DEFAULT_UGV_APPROACH_REWARD = 0.05
DEFAULT_UGV_APPROACH_MILESTONE_RADII_M = (75.0, 50.0, 40.0, 30.0, 20.0)
DEFAULT_UAV_DIAG_COVERAGE_OBS_GRID = 0
DEFAULT_UAV_DIAG_LOCAL_COVERAGE_OBS_GRID = 0
DEFAULT_UAV_DIAG_LOCAL_COVERAGE_OBS_RADIUS_M = 150.0
DEFAULT_UAV_DIAG_CONFIDENCE_OBS_GRID = 32
DEFAULT_UAV_DIAG_LOCAL_CONFIDENCE_OBS_GRID = 9
DEFAULT_UAV_DIAG_LOCAL_CONFIDENCE_OBS_RADIUS_M = 60.0
DEFAULT_UAV_FRONTIER_OBS_RADIUS_M = 150.0
DEFAULT_UAV_DIAG_FRONTIER_OBS_RADIUS_M = 60.0
DEFAULT_UAV_DIAG_DRONES = 3
DEFAULT_UAV_DIAG_COVERAGE_REWARD = 0.0
DEFAULT_UAV_DIAG_MOVE_COVERAGE_REWARD = 0.0
DEFAULT_UAV_DIAG_OVERLAP_PENALTY = 0.0
DEFAULT_UAV_DIAG_OVERLAP_ALLOWED = 0.10
DEFAULT_UAV_DIAG_OUTSIDE_FOOTPRINT_PENALTY = 0.10
DEFAULT_UAV_DIAG_FRONTIER_ALIGNMENT_REWARD = 0.05
DEFAULT_UAV_DIAG_CONFIDENCE_REWARD = 30.0
DEFAULT_UAV_DIAG_CONFIDENCE_MOVE_REWARD = 0.10
DEFAULT_UAV_DIAG_INEFFICIENT_MOVE_PENALTY = 0.005
DEFAULT_UAV_DIAG_INEFFICIENT_MOVE_SOURCE = "confidence"
DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_PENALTY = 0.06
DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_MODE = "opportunity_regret"
DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_THRESHOLD = 0.80
DEFAULT_UAV_DIAG_CLEANUP_TARGET_PROGRESS_REWARD = 0.0
DEFAULT_UAV_DIAG_ENTROPY_COEF = 0.05
DEFAULT_UAV_DIAG_EPISODE_LENGTH = 300
DEFAULT_UAV_DIAG_N_ROLLOUT_THREADS = 8
DEFAULT_UAV_DIAG_LOCAL_MAP_PATCH_SIZE = 7
DEFAULT_UAV_DIAG_START_MIN_SEPARATION_M = 150.0
DEFAULT_UAV_DIAG_START_EDGE_MARGIN_M = 50.0
DEFAULT_UAV_DIAG_TERRAIN_CACHE_PATH = ROOT / "data" / "terrain_cache" / "malibu_creek_500m_128.npz"
DEFAULT_UGV_DIAG_LOCAL_MAP_PATCH_SIZE = 7
DEFAULT_UGV_DIAG_TARGET_DISTANCE_MIN_M = 30.0
DEFAULT_UGV_DIAG_LR = 2.5e-4
DEFAULT_UGV_DIAG_CRITIC_LR = 5.0e-4
DEFAULT_UGV_DIAG_TERRAIN_CACHE_PATH = ROOT / "data" / "terrain_cache" / "malibu_creek_500m_128.npz"
DEFAULT_UGV_DIAG_PLANNER_HINT = "global_astar"
DEFAULT_UGV_DIAG_DENSE_REWARD_MODE = "planner_follow"
DEFAULT_UGV_DIAG_GLOBAL_PLANNER_HEURISTIC = "euclidean"
DEFAULT_UGV_DIAG_GLOBAL_PLANNER_LOOKAHEAD_M = 20.0
DEFAULT_UGV_DIAG_PLANNER_PROGRESS_REWARD = 0.0
DEFAULT_UGV_DIAG_ACTION_TRANSFORM = "radial_tanh"
DEFAULT_UGV_DIAG_PLANNER_FIRE_MODE = "block"
DEFAULT_UGV_DIAG_PLANNER_FIRE_COST = 25.0
DEFAULT_UGV_DIAG_PLANNER_SMOKE_COST = 5.0
DEFAULT_UGV_DIAG_PLANNER_SMOLDER_COST = 3.0
DEFAULT_UGV_DIAG_PLANNER_FIRE_BUFFER_M = 10.0
DEFAULT_UGV_DIAG_PLANNER_FIRE_BUFFER_COST = 8.0
DEFAULT_UGV_DIAG_PLANNER_FIRE_REPLAN_POLICY = "lazy"
DEFAULT_UGV_DIAG_PLANNER_FIRE_REPLAN_INTERVAL_STEPS = 15
DEFAULT_UGV_DIAG_PLANNER_FIRE_BLOCK_THRESHOLD = 0.6
DEFAULT_UGV_DIAG_PLANNER_LAND_COVER_COSTS = (0.85, 1.0, 1.15, 1.35, 4.0, 8.0)
DEFAULT_JOINT_DIAG_DRONES = 3
DEFAULT_JOINT_DIAG_UGVS = 2
DEFAULT_JOINT_DIAG_SURVIVORS = 5
DEFAULT_JOINT_DIAG_TEAM_SCOUT_REWARD = 1.0
DEFAULT_JOINT_DIAG_TEAM_CONFIRM_REWARD = 4.0
DEFAULT_JOINT_DIAG_GROUND_CONFIRM_REWARD = 10.0
DEFAULT_JOINT_DIAG_PENDING_PENALTY = -0.02
DEFAULT_JOINT_DIAG_ROUTE_PROGRESS_SHORTFALL_PENALTY = 0.10
DEFAULT_JOINT_DIAG_UGV_TARGET_ASSIGNMENT_MODE = "route_cost_sticky"
DEFAULT_UAV_FRONTIER_MODE = "sector_topk"
DEFAULT_UAV_DIAG_FRONTIER_MODE = "local_global"
DEFAULT_UAV_DIAG_FRONTIER_SOURCE = "confidence"
DEFAULT_UAV_FRONTIER_SECTORS = 8
DEFAULT_UAV_FRONTIER_TOP_K = 2
DEFAULT_UAV_FRONTIER_OWNERSHIP = True
DEFAULT_UAV_DIAG_CLEANUP_TARGET_REFRESH_MODE = "fixed_hold"


def _resolve_uav_reward_defaults(
    *,
    uav_survivor_diagnostic: bool,
    uav_coverage_reward: float | None,
    uav_move_coverage_reward: float | None,
    uav_overlap_penalty: float | None,
    uav_overlap_allowed: float | None,
    uav_outside_footprint_penalty: float | None,
) -> tuple[float, float, float, float, float]:
    """Resolve omitted UAV reward flags while preserving explicit zeroes."""

    if uav_survivor_diagnostic:
        coverage_reward = (
            DEFAULT_UAV_DIAG_COVERAGE_REWARD
            if uav_coverage_reward is None
            else uav_coverage_reward
        )
        move_coverage_reward = (
            DEFAULT_UAV_DIAG_MOVE_COVERAGE_REWARD
            if uav_move_coverage_reward is None
            else uav_move_coverage_reward
        )
        overlap_penalty = (
            DEFAULT_UAV_DIAG_OVERLAP_PENALTY
            if uav_overlap_penalty is None
            else uav_overlap_penalty
        )
        overlap_allowed = (
            DEFAULT_UAV_DIAG_OVERLAP_ALLOWED
            if uav_overlap_allowed is None
            else uav_overlap_allowed
        )
        outside_footprint_penalty = (
            DEFAULT_UAV_DIAG_OUTSIDE_FOOTPRINT_PENALTY
            if uav_outside_footprint_penalty is None
            else uav_outside_footprint_penalty
        )
    else:
        coverage_reward = 5.0 if uav_coverage_reward is None else uav_coverage_reward
        move_coverage_reward = 0.0 if uav_move_coverage_reward is None else uav_move_coverage_reward
        overlap_penalty = 0.0 if uav_overlap_penalty is None else uav_overlap_penalty
        overlap_allowed = 0.10 if uav_overlap_allowed is None else uav_overlap_allowed
        outside_footprint_penalty = (
            0.0
            if uav_outside_footprint_penalty is None
            else uav_outside_footprint_penalty
        )

    return (
        float(coverage_reward),
        float(move_coverage_reward),
        float(overlap_penalty),
        float(overlap_allowed),
        float(outside_footprint_penalty),
    )


# ----------------------------------------------------------------------
# Step 1 — monkey-patch HARL's env registry to recognise "wildfire"
# ----------------------------------------------------------------------
def _register_wildfire_with_harl():
    from agents.harl_runner import register_wildfire_with_harl

    register_wildfire_with_harl()


# ----------------------------------------------------------------------
# Step 2 — build HARL args from CLI parameters
# ----------------------------------------------------------------------
def build_args(
    num_env_steps:  int,
    episode_length: int,
    seed:           int,
    comms_dropout:  float,
    entropy_coef:   float,
    exp_name:       str,
    lr: float = 5e-4,
    critic_lr: float = 5e-4,
    linear_lr_decay: bool | None = None,
    share_param: bool | None = None,
    share_param_by_agent_class: bool | None = None,
    n_rollout_threads: int = 1,
    terrain_cache_path: str | None = None,
    drone_min_footprint_m: float = 0.0,
    ground_confirm_min_m: float = 0.0,
    fire_grid_size: int = 128,
    reward_search: bool = False,
    reward_confirm: bool = False,
    recurrent: bool = False,
    model_dir: str | None = None,
    warmstart_uav_model_dir: str | None = None,
    warmstart_ugv_model_dir: str | None = None,
    drone_camera_fov_deg: float | None = None,
    drone_flight_levels_m: tuple[float, ...] | None = None,
    ground_confirmation_range_m: float | None = None,
    coverage_obs_grid: int | None = None,
    confirm_requires_los: bool = False,
    drone_can_confirm: bool = False,
    r_drone_confirm: float = 0.0,
    local_coverage_obs_grid: int | None = None,
    local_coverage_obs_radius_m: float = 150.0,
    uav_confidence_obs_grid: int | None = None,
    uav_local_confidence_obs_grid: int | None = None,
    uav_local_confidence_obs_radius_m: float | None = None,
    uav_frontier_obs: bool | None = None,
    uav_frontier_obs_radius_m: float | None = None,
    uav_frontier_mode: str | None = None,
    uav_frontier_source: str | None = None,
    uav_frontier_sectors: int = DEFAULT_UAV_FRONTIER_SECTORS,
    uav_frontier_top_k: int = DEFAULT_UAV_FRONTIER_TOP_K,
    uav_frontier_ownership: bool = DEFAULT_UAV_FRONTIER_OWNERSHIP,
    uav_cleanup_target_obs: bool | None = None,
    uav_cleanup_target_grid: int = 16,
    uav_cleanup_target_hold_steps: int = 15,
    uav_cleanup_target_confidence_threshold: float = 0.80,
    uav_cleanup_target_min_value: float = 0.05,
    uav_cleanup_target_assignment_distance_scale_m: float = 250.0,
    uav_cleanup_target_refresh_mode: str | None = None,
    uav_astar_route_obs: bool = False,
    uav_astar_grid: int = 32,
    uav_astar_confidence_cost_alpha: float = 3.0,
    uav_astar_confidence_cost_gamma: float = 2.0,
    uav_astar_waypoint_lookahead_m: float = 50.0,
    uav_astar_route_replan_steps: int = 5,
    uav_astar_waypoint_reached_m: float = 20.0,
    ugv_known_survivor_diagnostic: bool = False,
    uav_survivor_diagnostic: bool = False,
    joint_schema_uav_diagnostic: bool = False,
    joint_survivor_diagnostic: bool = False,
    joint_schema_ugv_diagnostic: bool = False,
    uav_diagnostic_drones: int = DEFAULT_UAV_DIAG_DRONES,
    joint_diagnostic_ugvs: int = DEFAULT_JOINT_DIAG_UGVS,
    n_drones: int | None = None,
    n_ugvs: int | None = None,
    n_survivors: int | None = None,
    delayed_survivor_knowledge: bool = False,
    survivor_reveal_schedule: str = "stratified_uniform",
    survivor_reveal_initial_count: int = 1,
    survivor_reveal_start_step: int = 10,
    survivor_reveal_end_step: int = 180,
    survivor_assignment_obs: bool | None = None,
    ugv_diagnostic_target_distance_min_m: float | None = None,
    ugv_diagnostic_target_distance_max_m: float | None = None,
    uav_no_global_coverage_obs: bool = False,
    uav_coverage_only: bool = False,
    uav_found_survivor_reward: float | None = None,
    uav_all_survivors_reward: float | None = None,
    team_scout_reward: float | None = None,
    uav_time_penalty: float | None = None,
    uav_coverage_reward: float | None = None,
    uav_coverage_normalization: str = "map",
    uav_move_coverage_reward: float | None = None,
    uav_move_coverage_normalization: str = "raw",
    uav_move_coverage_cap: float = 0.1,
    uav_coverage_threshold_reward: float | None = None,
    uav_coverage_threshold_fraction: float = 0.95,
    uav_coverage_opportunity_reward: float | None = None,
    uav_coverage_opportunity_cap: float = 1.0,
    uav_frontier_alignment_reward: float | None = None,
    uav_confidence_reward: float | None = None,
    uav_team_confidence_reward: float = 0.0,
    uav_team_confidence_overlap_penalty: float = 0.0,
    uav_confidence_move_reward: float | None = None,
    uav_inefficient_move_penalty: float | None = None,
    uav_inefficient_move_source: str | None = None,
    uav_confidence_overlap_penalty: float | None = None,
    uav_confidence_overlap_mode: str | None = None,
    uav_confidence_overlap_allowed_regret: float = 0.10,
    uav_cleanup_target_progress_reward: float | None = None,
    uav_astar_progress_reward: float = 0.0,
    uav_confidence_overlap_threshold: float | None = None,
    uav_confidence_gamma: float = 2.0,
    uav_confidence_eps: float = 0.05,
    uav_confidence_opportunity_eps: float = 1e-6,
    uav_overlap_penalty: float | None = None,
    uav_overlap_allowed: float | None = None,
    uav_overlap_penalty_normalization: str = "raw",
    uav_inter_uav_overlap_penalty: float = 0.0,
    uav_inter_uav_overlap_allowed: float = 0.20,
    uav_outside_footprint_penalty: float | None = None,
    uav_boundary_soft_margin_m: float = 25.0,
    uav_start_min_separation_m: float | None = None,
    uav_start_edge_margin_m: float | None = None,
    ugv_ground_shaping_reward: float | None = None,
    ugv_movement_alignment_reward: float = 0.20,
    ugv_pending_penalty: float | None = None,
    ugv_planner_progress_reward: float = 0.0,
    ugv_route_aware_reward: bool = False,
    ugv_dense_reward_mode: str = "target",
    ugv_planner_blend_weight: float = 0.70,
    ugv_escape_stall_steps: int = 5,
    ugv_escape_progress_threshold_m: float = 0.10,
    ugv_escape_movement_threshold_m: float = 0.25,
    ugv_escape_waypoint_reached_m: float = 4.0,
    ugv_escape_max_steps: int = 15,
    ugv_approach_reward: float = DEFAULT_UGV_APPROACH_REWARD,
    ugv_approach_milestone_radii_m: tuple[float, ...] = DEFAULT_UGV_APPROACH_MILESTONE_RADII_M,
    ugv_stall_penalty: float = 0.0,
    ugv_stall_displacement_threshold_m: float = 0.05,
    ugv_route_progress_floor_penalty: float = 0.0,
    ugv_route_progress_floor_m: float = 0.0,
    ugv_route_progress_shortfall_penalty: float | None = None,
    local_map_patch_size: int = 3,
    slope_speed_weight: float | None = None,
    land_cover_speeds: tuple[float, ...] | None = None,
    action_transform: str = "clip",
    terrain_cnn_encoder: bool = False,
    terrain_cnn_embed_dim: int = 16,
    ugv_planner_hint: str = "none",
    ugv_planner_detour_obs: bool = False,
    ugv_planner_patch_size: int = 11,
    ugv_planner_lookahead_cells: int = 10,
    ugv_global_planner_lookahead_m: float = 20.0,
    ugv_global_planner_heuristic: str = "euclidean",
    ugv_planner_fire_mode: str = "off",
    ugv_planner_fire_replan_policy: str = "always",
    ugv_planner_fire_replan_interval_steps: int = 15,
    ugv_planner_fire_cost: float = 25.0,
    ugv_planner_fire_block_threshold: float = 0.0,
    ugv_planner_smoke_cost: float = 5.0,
    ugv_planner_smolder_cost: float = 3.0,
    ugv_planner_fire_buffer_m: float = 10.0,
    ugv_planner_fire_buffer_cost: float = 8.0,
    ugv_planner_land_cover_costs: tuple[float, ...] | None = None,
    ugv_target_assignment_mode: str | None = None,
    ugv_assigned_target_obs_only: bool | None = None,
    ugv_sticky_switch_margin_m: float = 20.0,
    ugv_sticky_switch_ratio: float = 0.80,
    ugv_sticky_min_age_steps: int = 10,
    enable_fire: bool | None = None,
) -> tuple[dict, dict, dict]:
    survivor_count = (
        DEFAULT_JOINT_DIAG_SURVIVORS
        if n_survivors is None
        else max(int(n_survivors), 1)
    )
    joint_drone_count = DEFAULT_JOINT_DIAG_DRONES if n_drones is None else max(int(n_drones), 0)
    joint_ugv_count = DEFAULT_JOINT_DIAG_UGVS if n_ugvs is None else max(int(n_ugvs), 0)
    if n_drones is not None:
        uav_diagnostic_drones = max(int(n_drones), 1)
    if n_ugvs is not None:
        joint_diagnostic_ugvs = max(int(n_ugvs), 1)
    ugv_planner_hint = str(ugv_planner_hint).replace("-", "_")
    uav_search_diagnostic = (
        uav_survivor_diagnostic
        or joint_schema_uav_diagnostic
        or joint_survivor_diagnostic
        or joint_schema_ugv_diagnostic
    )
    ugv_global_diagnostic = (
        ugv_known_survivor_diagnostic
        or joint_survivor_diagnostic
        or joint_schema_ugv_diagnostic
    )
    if ugv_global_diagnostic:
        defaulted_ugv_planner_hint = ugv_planner_hint == "none"
        if terrain_cache_path is None:
            terrain_cache_path = str(DEFAULT_UGV_DIAG_TERRAIN_CACHE_PATH)
        if local_map_patch_size == 3:
            local_map_patch_size = DEFAULT_UGV_DIAG_LOCAL_MAP_PATCH_SIZE
        if ugv_known_survivor_diagnostic and ugv_diagnostic_target_distance_min_m is None:
            ugv_diagnostic_target_distance_min_m = DEFAULT_UGV_DIAG_TARGET_DISTANCE_MIN_M
        if lr == 5e-4:
            lr = DEFAULT_UGV_DIAG_LR
        if critic_lr == 5e-4:
            critic_lr = DEFAULT_UGV_DIAG_CRITIC_LR
        if linear_lr_decay is None:
            linear_lr_decay = True
        if defaulted_ugv_planner_hint:
            ugv_planner_hint = DEFAULT_UGV_DIAG_PLANNER_HINT
        if defaulted_ugv_planner_hint and str(ugv_dense_reward_mode).replace("-", "_") == "target":
            ugv_dense_reward_mode = DEFAULT_UGV_DIAG_DENSE_REWARD_MODE
        if ugv_global_planner_heuristic == "euclidean":
            ugv_global_planner_heuristic = DEFAULT_UGV_DIAG_GLOBAL_PLANNER_HEURISTIC
        if ugv_global_planner_lookahead_m == 20.0:
            ugv_global_planner_lookahead_m = DEFAULT_UGV_DIAG_GLOBAL_PLANNER_LOOKAHEAD_M
        if ugv_planner_progress_reward == 0.0:
            ugv_planner_progress_reward = DEFAULT_UGV_DIAG_PLANNER_PROGRESS_REWARD
        if action_transform == "clip":
            action_transform = DEFAULT_UGV_DIAG_ACTION_TRANSFORM
        if enable_fire is None:
            enable_fire = bool(ugv_known_survivor_diagnostic)
        if joint_survivor_diagnostic and enable_fire is False:
            ugv_planner_fire_mode = "off"
        if ugv_planner_fire_mode == "off":
            if enable_fire:
                ugv_planner_fire_mode = DEFAULT_UGV_DIAG_PLANNER_FIRE_MODE
        if ugv_planner_fire_replan_policy == "always":
            ugv_planner_fire_replan_policy = DEFAULT_UGV_DIAG_PLANNER_FIRE_REPLAN_POLICY
        if ugv_planner_fire_replan_interval_steps == 15:
            ugv_planner_fire_replan_interval_steps = DEFAULT_UGV_DIAG_PLANNER_FIRE_REPLAN_INTERVAL_STEPS
        if ugv_planner_fire_cost == 25.0:
            ugv_planner_fire_cost = DEFAULT_UGV_DIAG_PLANNER_FIRE_COST
        if ugv_planner_fire_block_threshold == 0.0:
            ugv_planner_fire_block_threshold = DEFAULT_UGV_DIAG_PLANNER_FIRE_BLOCK_THRESHOLD
        if ugv_planner_smoke_cost == 5.0:
            ugv_planner_smoke_cost = DEFAULT_UGV_DIAG_PLANNER_SMOKE_COST
        if ugv_planner_smolder_cost == 3.0:
            ugv_planner_smolder_cost = DEFAULT_UGV_DIAG_PLANNER_SMOLDER_COST
        if ugv_planner_fire_buffer_m == 10.0:
            ugv_planner_fire_buffer_m = DEFAULT_UGV_DIAG_PLANNER_FIRE_BUFFER_M
        if ugv_planner_fire_buffer_cost == 8.0:
            ugv_planner_fire_buffer_cost = DEFAULT_UGV_DIAG_PLANNER_FIRE_BUFFER_COST
        if ugv_planner_land_cover_costs is None:
            ugv_planner_land_cover_costs = DEFAULT_UGV_DIAG_PLANNER_LAND_COVER_COSTS
    if linear_lr_decay is None:
        linear_lr_decay = False
    if enable_fire is None:
        enable_fire = False
    ugv_local_planners = {"local_astar", "local_escape_astar"}
    ugv_planners = ugv_local_planners | {"global_astar"}
    if ugv_planner_hint not in {"none"} | ugv_planners:
        raise ValueError("ugv_planner_hint must be one of: none, local_astar, local_escape_astar, global_astar")
    if ugv_planner_detour_obs and ugv_planner_hint not in ugv_planners:
        raise ValueError("ugv_planner_detour_obs requires a UGV planner hint")
    if ugv_route_aware_reward and ugv_planner_hint not in ugv_local_planners:
        raise ValueError("ugv_route_aware_reward requires a local UGV planner hint")
    ugv_dense_reward_mode = str(ugv_dense_reward_mode).replace("-", "_")
    if ugv_dense_reward_mode not in {
        "target",
        "positive_target",
            "planner_blend",
            "escape_blend",
            "escape_route_switch",
            "planner_follow",
        }:
        raise ValueError(
            "ugv_dense_reward_mode must be one of: target, positive_target, "
            "planner_blend, escape_blend, escape_route_switch, planner_follow"
        )
    if ugv_dense_reward_mode == "planner_blend" and ugv_planner_hint not in ugv_local_planners:
        raise ValueError("ugv_dense_reward_mode='planner_blend' requires a local UGV planner hint")
    if ugv_dense_reward_mode == "escape_blend" and ugv_planner_hint != "local_escape_astar":
        raise ValueError("ugv_dense_reward_mode='escape_blend' requires ugv_planner_hint='local_escape_astar'")
    if ugv_dense_reward_mode == "escape_route_switch" and ugv_planner_hint != "local_astar":
        raise ValueError("ugv_dense_reward_mode='escape_route_switch' requires ugv_planner_hint='local_astar'")
    if ugv_dense_reward_mode == "planner_follow" and ugv_planner_hint != "global_astar":
        raise ValueError("ugv_dense_reward_mode='planner_follow' requires ugv_planner_hint='global_astar'")
    if ugv_route_aware_reward and ugv_dense_reward_mode != "target":
        raise ValueError("ugv_route_aware_reward can only be combined with ugv_dense_reward_mode='target'")
    ugv_global_planner_heuristic = str(ugv_global_planner_heuristic).replace("-", "_")
    if ugv_global_planner_heuristic not in {"euclidean", "terrain"}:
        raise ValueError("ugv_global_planner_heuristic must be one of: euclidean, terrain")
    ugv_planner_fire_mode = str(ugv_planner_fire_mode).replace("-", "_")
    if ugv_planner_fire_mode not in {"off", "cost", "block"}:
        raise ValueError("ugv_planner_fire_mode must be one of: off, cost, block")
    ugv_planner_fire_replan_policy = str(ugv_planner_fire_replan_policy).replace("-", "_")
    if ugv_planner_fire_replan_policy not in {"always", "affected", "lazy"}:
        raise ValueError("ugv_planner_fire_replan_policy must be one of: always, affected, lazy")
    ugv_planner_fire_replan_interval_steps = max(int(ugv_planner_fire_replan_interval_steps), 1)
    if ugv_target_assignment_mode is None:
        ugv_target_assignment_mode = (
            DEFAULT_JOINT_DIAG_UGV_TARGET_ASSIGNMENT_MODE
            if joint_survivor_diagnostic or joint_schema_ugv_diagnostic
            else "nearest"
        )
    ugv_target_assignment_mode = str(ugv_target_assignment_mode).replace("-", "_").lower()
    valid_assignment_modes = {
        "nearest",
        "greedy",
        "greedy_sticky",
        "route_cost_greedy",
        "route_cost_sticky",
        "route_cost_global",
    }
    if ugv_target_assignment_mode not in valid_assignment_modes:
        raise ValueError(
            "ugv_target_assignment_mode must be one of: nearest, greedy, "
            "greedy_sticky, route_cost_greedy, route_cost_sticky, route_cost_global"
        )
    if ugv_assigned_target_obs_only is None:
        ugv_assigned_target_obs_only = False
    ugv_assigned_target_obs_only = bool(ugv_assigned_target_obs_only)
    if ugv_route_progress_shortfall_penalty is None:
        ugv_route_progress_shortfall_penalty = (
            DEFAULT_JOINT_DIAG_ROUTE_PROGRESS_SHORTFALL_PENALTY
            if joint_survivor_diagnostic or joint_schema_ugv_diagnostic
            else 0.0
        )
    if float(ugv_route_progress_shortfall_penalty) < 0.0:
        raise ValueError("ugv_route_progress_shortfall_penalty must be nonnegative")
    ugv_route_progress_shortfall_penalty = float(ugv_route_progress_shortfall_penalty)
    if survivor_assignment_obs is None:
        survivor_assignment_obs = bool(
            joint_survivor_diagnostic
            or joint_schema_uav_diagnostic
            or joint_schema_ugv_diagnostic
        )
    survivor_assignment_obs = bool(survivor_assignment_obs)
    survivor_reveal_schedule = str(survivor_reveal_schedule).replace("-", "_").lower()
    if survivor_reveal_schedule not in {"stratified_uniform"}:
        raise ValueError("survivor_reveal_schedule must be stratified_uniform")
    if int(survivor_reveal_initial_count) < 0:
        raise ValueError("survivor_reveal_initial_count must be nonnegative")
    if int(survivor_reveal_start_step) < 0 or int(survivor_reveal_end_step) < 0:
        raise ValueError("survivor reveal steps must be nonnegative")
    if int(survivor_reveal_end_step) < int(survivor_reveal_start_step):
        raise ValueError("survivor_reveal_end_step must be >= survivor_reveal_start_step")
    if float(ugv_sticky_switch_margin_m) < 0.0:
        raise ValueError("ugv_sticky_switch_margin_m must be nonnegative")
    if float(ugv_sticky_switch_ratio) < 0.0:
        raise ValueError("ugv_sticky_switch_ratio must be nonnegative")
    if int(ugv_sticky_min_age_steps) < 0:
        raise ValueError("ugv_sticky_min_age_steps must be nonnegative")
    if not 0.0 <= float(ugv_planner_fire_block_threshold) <= 1.0:
        raise ValueError("ugv_planner_fire_block_threshold must be in [0, 1]")
    if uav_search_diagnostic:
        uav_diagnostic_drones = int(uav_diagnostic_drones)
        if uav_diagnostic_drones < 1:
            raise ValueError("uav_diagnostic_drones must be positive")
        if terrain_cache_path is None:
            terrain_cache_path = str(DEFAULT_UAV_DIAG_TERRAIN_CACHE_PATH)
        if local_map_patch_size == 3:
            local_map_patch_size = DEFAULT_UAV_DIAG_LOCAL_MAP_PATCH_SIZE
        if uav_no_global_coverage_obs:
            coverage_obs_grid = 0
        elif coverage_obs_grid is None:
            coverage_obs_grid = DEFAULT_UAV_DIAG_COVERAGE_OBS_GRID
        if local_coverage_obs_grid is None:
            local_coverage_obs_grid = DEFAULT_UAV_DIAG_LOCAL_COVERAGE_OBS_GRID
            local_coverage_obs_radius_m = DEFAULT_UAV_DIAG_LOCAL_COVERAGE_OBS_RADIUS_M
        if uav_confidence_obs_grid is None:
            uav_confidence_obs_grid = DEFAULT_UAV_DIAG_CONFIDENCE_OBS_GRID
        if uav_local_confidence_obs_grid is None:
            uav_local_confidence_obs_grid = DEFAULT_UAV_DIAG_LOCAL_CONFIDENCE_OBS_GRID
            uav_local_confidence_obs_radius_m = DEFAULT_UAV_DIAG_LOCAL_CONFIDENCE_OBS_RADIUS_M
        if uav_frontier_obs is None:
            uav_frontier_obs = True
        if uav_frontier_obs_radius_m is None:
            uav_frontier_obs_radius_m = DEFAULT_UAV_DIAG_FRONTIER_OBS_RADIUS_M
        if uav_frontier_mode is None:
            uav_frontier_mode = DEFAULT_UAV_DIAG_FRONTIER_MODE
        if uav_frontier_source is None:
            uav_frontier_source = DEFAULT_UAV_DIAG_FRONTIER_SOURCE
        if uav_frontier_alignment_reward is None:
            uav_frontier_alignment_reward = DEFAULT_UAV_DIAG_FRONTIER_ALIGNMENT_REWARD
        if uav_cleanup_target_obs is None:
            uav_cleanup_target_obs = False
        if uav_cleanup_target_refresh_mode is None:
            uav_cleanup_target_refresh_mode = DEFAULT_UAV_DIAG_CLEANUP_TARGET_REFRESH_MODE
        if uav_confidence_reward is None:
            uav_confidence_reward = DEFAULT_UAV_DIAG_CONFIDENCE_REWARD
        if uav_confidence_move_reward is None:
            uav_confidence_move_reward = DEFAULT_UAV_DIAG_CONFIDENCE_MOVE_REWARD
        if uav_inefficient_move_penalty is None:
            uav_inefficient_move_penalty = DEFAULT_UAV_DIAG_INEFFICIENT_MOVE_PENALTY
        if uav_inefficient_move_source is None:
            uav_inefficient_move_source = DEFAULT_UAV_DIAG_INEFFICIENT_MOVE_SOURCE
        if uav_confidence_overlap_penalty is None:
            uav_confidence_overlap_penalty = DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_PENALTY
        if uav_confidence_overlap_mode is None:
            uav_confidence_overlap_mode = DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_MODE
        if uav_confidence_overlap_threshold is None:
            uav_confidence_overlap_threshold = DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_THRESHOLD
        if uav_cleanup_target_progress_reward is None:
            uav_cleanup_target_progress_reward = DEFAULT_UAV_DIAG_CLEANUP_TARGET_PROGRESS_REWARD
        if share_param is None and (uav_survivor_diagnostic or joint_schema_uav_diagnostic) and not bool(share_param_by_agent_class):
            share_param = True
        if uav_start_min_separation_m is None:
            uav_start_min_separation_m = DEFAULT_UAV_DIAG_START_MIN_SEPARATION_M
        if uav_start_edge_margin_m is None:
            uav_start_edge_margin_m = DEFAULT_UAV_DIAG_START_EDGE_MARGIN_M
        if action_transform == "clip":
            action_transform = "radial_tanh"
        if joint_schema_uav_diagnostic and ugv_planner_hint == "none":
            ugv_planner_hint = DEFAULT_UGV_DIAG_PLANNER_HINT
    if uav_frontier_obs is None:
        uav_frontier_obs = False
    if uav_confidence_obs_grid is None:
        uav_confidence_obs_grid = 0
    if uav_local_confidence_obs_grid is None:
        uav_local_confidence_obs_grid = 0
    if uav_local_confidence_obs_radius_m is None:
        uav_local_confidence_obs_radius_m = 150.0
    if uav_frontier_obs_radius_m is None:
        uav_frontier_obs_radius_m = DEFAULT_UAV_FRONTIER_OBS_RADIUS_M
    if uav_frontier_mode is None:
        uav_frontier_mode = DEFAULT_UAV_FRONTIER_MODE
    if uav_frontier_source is None:
        uav_frontier_source = "coverage"
    if uav_frontier_alignment_reward is None:
        uav_frontier_alignment_reward = 0.0
    if uav_cleanup_target_obs is None:
        uav_cleanup_target_obs = False
    if uav_cleanup_target_refresh_mode is None:
        uav_cleanup_target_refresh_mode = "exact"
    if uav_confidence_reward is None:
        uav_confidence_reward = 0.0
    if uav_confidence_move_reward is None:
        uav_confidence_move_reward = 0.0
    if uav_inefficient_move_penalty is None:
        uav_inefficient_move_penalty = 0.0
    if uav_inefficient_move_source is None:
        uav_inefficient_move_source = "confidence"
    if uav_confidence_overlap_penalty is None:
        uav_confidence_overlap_penalty = 0.0
    if uav_confidence_overlap_mode is None:
        uav_confidence_overlap_mode = "raw"
    if uav_confidence_overlap_threshold is None:
        uav_confidence_overlap_threshold = 0.65
    if uav_cleanup_target_progress_reward is None:
        uav_cleanup_target_progress_reward = 0.0
    if share_param_by_agent_class is None:
        share_param_by_agent_class = bool(
            (joint_survivor_diagnostic or joint_schema_ugv_diagnostic)
            and not bool(share_param)
        )
    if share_param is None:
        share_param = False
    if bool(share_param) and bool(share_param_by_agent_class):
        raise ValueError("share_param and share_param_by_agent_class are mutually exclusive")
    if model_dir and (warmstart_uav_model_dir or warmstart_ugv_model_dir):
        raise ValueError("model_dir cannot be combined with class warm-start model dirs")
    if (warmstart_uav_model_dir or warmstart_ugv_model_dir) and not bool(share_param_by_agent_class):
        raise ValueError("class warm-start model dirs require share_param_by_agent_class=True")
    uav_confidence_reward = float(uav_confidence_reward)
    if uav_confidence_reward < 0.0:
        raise ValueError("uav_confidence_reward must be nonnegative")
    uav_team_confidence_reward = float(uav_team_confidence_reward)
    if uav_team_confidence_reward < 0.0:
        raise ValueError("uav_team_confidence_reward must be nonnegative")
    uav_team_confidence_overlap_penalty = float(uav_team_confidence_overlap_penalty)
    if uav_team_confidence_overlap_penalty < 0.0:
        raise ValueError("uav_team_confidence_overlap_penalty must be nonnegative")
    uav_confidence_move_reward = float(uav_confidence_move_reward)
    if uav_confidence_move_reward < 0.0:
        raise ValueError("uav_confidence_move_reward must be nonnegative")
    uav_inefficient_move_penalty = float(uav_inefficient_move_penalty)
    if uav_inefficient_move_penalty < 0.0:
        raise ValueError("uav_inefficient_move_penalty must be nonnegative")
    uav_inefficient_move_source = str(uav_inefficient_move_source).replace("-", "_").lower()
    if uav_inefficient_move_source not in {"coverage", "confidence"}:
        raise ValueError("uav_inefficient_move_source must be one of: coverage, confidence")
    uav_confidence_overlap_penalty = float(uav_confidence_overlap_penalty)
    if uav_confidence_overlap_penalty < 0.0:
        raise ValueError("uav_confidence_overlap_penalty must be nonnegative")
    uav_confidence_overlap_mode = str(uav_confidence_overlap_mode).replace("-", "_").lower()
    if uav_confidence_overlap_mode not in {"raw", "opportunity_regret"}:
        raise ValueError("uav_confidence_overlap_mode must be one of: raw, opportunity_regret")
    uav_confidence_overlap_allowed_regret = float(uav_confidence_overlap_allowed_regret)
    if not 0.0 <= uav_confidence_overlap_allowed_regret <= 1.0:
        raise ValueError("uav_confidence_overlap_allowed_regret must be in [0, 1]")
    uav_cleanup_target_progress_reward = float(uav_cleanup_target_progress_reward)
    if uav_cleanup_target_progress_reward < 0.0:
        raise ValueError("uav_cleanup_target_progress_reward must be nonnegative")
    uav_astar_progress_reward = float(uav_astar_progress_reward)
    if uav_astar_progress_reward < 0.0:
        raise ValueError("uav_astar_progress_reward must be nonnegative")
    uav_confidence_overlap_threshold = float(uav_confidence_overlap_threshold)
    if not 0.0 <= uav_confidence_overlap_threshold < 1.0:
        raise ValueError("uav_confidence_overlap_threshold must be in [0, 1)")
    uav_confidence_gamma = max(float(uav_confidence_gamma), 0.0)
    uav_confidence_eps = max(float(uav_confidence_eps), 0.0)
    uav_confidence_opportunity_eps = max(float(uav_confidence_opportunity_eps), 0.0)
    uav_coverage_normalization = str(uav_coverage_normalization).replace("-", "_").lower()
    if uav_coverage_normalization not in {"map", "opportunity"}:
        raise ValueError("uav_coverage_normalization must be one of: map, opportunity")
    uav_move_coverage_normalization = str(uav_move_coverage_normalization).replace("-", "_").lower()
    if uav_move_coverage_normalization not in {"raw", "opportunity"}:
        raise ValueError("uav_move_coverage_normalization must be one of: raw, opportunity")
    uav_overlap_penalty_normalization = str(uav_overlap_penalty_normalization).replace("-", "_").lower()
    if uav_overlap_penalty_normalization not in {"raw", "opportunity"}:
        raise ValueError("uav_overlap_penalty_normalization must be one of: raw, opportunity")
    if uav_coverage_opportunity_reward is not None:
        legacy_opportunity_reward = float(uav_coverage_opportunity_reward)
        if legacy_opportunity_reward < 0.0:
            raise ValueError("uav_coverage_opportunity_reward must be nonnegative")
        if (
            uav_coverage_reward is not None
            and float(uav_coverage_reward) > 0.0
            and legacy_opportunity_reward > 0.0
        ):
            raise ValueError(
                "Use either uav_coverage_reward with uav_coverage_normalization='opportunity' "
                "or legacy uav_coverage_opportunity_reward, not both"
            )
        if legacy_opportunity_reward > 0.0:
            uav_coverage_reward = legacy_opportunity_reward
            uav_coverage_normalization = "opportunity"
    (
        uav_coverage_reward,
        uav_move_coverage_reward,
        uav_overlap_penalty,
        uav_overlap_allowed,
        uav_outside_footprint_penalty,
    ) = _resolve_uav_reward_defaults(
        uav_survivor_diagnostic=uav_search_diagnostic,
        uav_coverage_reward=uav_coverage_reward,
        uav_move_coverage_reward=uav_move_coverage_reward,
        uav_overlap_penalty=uav_overlap_penalty,
        uav_overlap_allowed=uav_overlap_allowed,
        uav_outside_footprint_penalty=uav_outside_footprint_penalty,
    )
    ugv_planner_patch_size = int(ugv_planner_patch_size)
    if ugv_planner_patch_size < 1 or ugv_planner_patch_size % 2 != 1:
        raise ValueError("ugv_planner_patch_size must be a positive odd integer")
    ugv_planner_lookahead_cells = int(ugv_planner_lookahead_cells)
    if ugv_planner_lookahead_cells < 1:
        raise ValueError("ugv_planner_lookahead_cells must be positive")
    ugv_planner_lookahead_cells = min(
        ugv_planner_lookahead_cells,
        max(ugv_planner_patch_size // 2, 1),
    )
    ugv_planner_progress_reward = float(ugv_planner_progress_reward)
    if ugv_planner_progress_reward < 0.0:
        raise ValueError("ugv_planner_progress_reward must be nonnegative")
    if ugv_planner_progress_reward > 0.0 and ugv_planner_hint not in ugv_planners:
        raise ValueError("ugv_planner_progress_reward requires a UGV planner hint")
    if ugv_route_aware_reward and ugv_planner_hint not in ugv_local_planners:
        raise ValueError("ugv_route_aware_reward requires a local UGV planner hint")
    ugv_planner_blend_weight = min(max(float(ugv_planner_blend_weight), 0.0), 1.0)
    ugv_escape_stall_steps = max(int(ugv_escape_stall_steps), 1)
    ugv_escape_progress_threshold_m = max(float(ugv_escape_progress_threshold_m), 0.0)
    ugv_escape_movement_threshold_m = max(float(ugv_escape_movement_threshold_m), 0.0)
    ugv_escape_waypoint_reached_m = max(float(ugv_escape_waypoint_reached_m), 1e-6)
    ugv_escape_max_steps = max(int(ugv_escape_max_steps), 1)
    ugv_global_planner_lookahead_m = max(float(ugv_global_planner_lookahead_m), 1e-6)
    uav_coverage_reward = float(uav_coverage_reward)
    if uav_coverage_reward < 0.0:
        raise ValueError("uav_coverage_reward must be nonnegative")
    uav_move_coverage_reward = float(uav_move_coverage_reward)
    if uav_move_coverage_reward < 0.0:
        raise ValueError("uav_move_coverage_reward must be nonnegative")
    uav_move_coverage_cap = max(float(uav_move_coverage_cap), 0.0)
    if uav_coverage_threshold_reward is not None and float(uav_coverage_threshold_reward) < 0.0:
        raise ValueError("uav_coverage_threshold_reward must be nonnegative")
    uav_coverage_threshold_fraction = float(uav_coverage_threshold_fraction)
    if not 0.0 <= uav_coverage_threshold_fraction <= 1.0:
        raise ValueError("uav_coverage_threshold_fraction must be in [0, 1]")
    uav_coverage_opportunity_cap = max(float(uav_coverage_opportunity_cap), 0.0)
    uav_overlap_penalty = float(uav_overlap_penalty)
    if uav_overlap_penalty < 0.0:
        raise ValueError("uav_overlap_penalty must be nonnegative")
    uav_overlap_allowed = float(uav_overlap_allowed)
    if not 0.0 <= uav_overlap_allowed < 1.0:
        raise ValueError("uav_overlap_allowed must be in [0, 1)")
    uav_inter_uav_overlap_penalty = float(uav_inter_uav_overlap_penalty)
    if uav_inter_uav_overlap_penalty < 0.0:
        raise ValueError("uav_inter_uav_overlap_penalty must be nonnegative")
    uav_inter_uav_overlap_allowed = float(uav_inter_uav_overlap_allowed)
    if not 0.0 <= uav_inter_uav_overlap_allowed < 1.0:
        raise ValueError("uav_inter_uav_overlap_allowed must be in [0, 1)")
    uav_outside_footprint_penalty = float(uav_outside_footprint_penalty)
    if uav_outside_footprint_penalty < 0.0:
        raise ValueError("uav_outside_footprint_penalty must be nonnegative")
    uav_boundary_soft_margin_m = max(float(uav_boundary_soft_margin_m), 1e-6)
    if uav_start_min_separation_m is not None:
        uav_start_min_separation_m = max(float(uav_start_min_separation_m), 0.0)
    if uav_start_edge_margin_m is not None:
        uav_start_edge_margin_m = max(float(uav_start_edge_margin_m), 0.0)

    args = {
        "algo":        "happo",
        "env":         "wildfire",
        "exp_name":    exp_name,
        "load_config": "",
    }

    algo_args = {
        "seed":   {"seed_specify": True, "seed": seed},
        "device": {
            "cuda": False, "cuda_deterministic": True, "torch_threads": 4,
        },
        "train": {
            "n_rollout_threads":      n_rollout_threads,
            "num_env_steps":          num_env_steps,
            "episode_length":         episode_length,
            "log_interval":           1,
            "eval_interval":          1,
            "use_valuenorm":          True,
            "use_linear_lr_decay":    linear_lr_decay,
            "use_proper_time_limits": True,
            "model_dir":              model_dir,
            "warmstart_uav_model_dir": warmstart_uav_model_dir,
            "warmstart_ugv_model_dir": warmstart_ugv_model_dir,
        },
        "eval": {
            "use_eval":               False,
            "n_eval_rollout_threads": 1,
            "eval_episodes":          2,
        },
        "render": {
            "use_render":      False,
            "render_episodes": 1,
        },
        "model": {
            "hidden_sizes":               [128, 128],
            "activation_func":            "relu",
            "use_feature_normalization":  True,
            "initialization_method":      "orthogonal_",
            "gain":                       0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy":       recurrent,
            "recurrent_n":                1,
            "data_chunk_length":          10,
            "lr":                         lr,
            "critic_lr":                  critic_lr,
            "opti_eps":                   1e-5,
            "weight_decay":               0,
            "std_x_coef":                 1,
            "std_y_coef":                 1.0,  # was 0.5; higher keeps action std from
                                                # collapsing into saturated corner-camping
            "use_terrain_cnn_encoder":     terrain_cnn_encoder,
            "terrain_cnn_patch_size":      local_map_patch_size,
            "terrain_cnn_channels":        TERRAIN_CNN_CHANNELS,
            "terrain_cnn_obs_offset":      TERRAIN_CNN_OBS_OFFSET,
            "terrain_cnn_embed_dim":       terrain_cnn_embed_dim,
            "terrain_cnn_hidden_channels": 8,
        },
        "algo": {
            "ppo_epoch":               2,
            "critic_epoch":            2,
            "use_clipped_value_loss":  True,
            "clip_param":              0.2,
            "actor_num_mini_batch":    1,
            "critic_num_mini_batch":   1,
            "entropy_coef":            entropy_coef,
            "value_loss_coef":         1,
            "use_max_grad_norm":       True,
            "max_grad_norm":           10.0,
            "use_gae":                 True,
            "gamma":                   0.99,
            "gae_lambda":              0.95,
            "use_huber_loss":          True,
            "use_policy_active_masks": True,
            "huber_delta":             10.0,
            "action_aggregation":      "prod",
            "share_param":             bool(share_param),
            "share_param_by_agent_class": bool(share_param_by_agent_class),
            "share_param_groups":      [],
            "share_param_group_names": [],
            "fixed_order":             False,
        },
        "logger": {
            "log_dir": str(ROOT / "results" / "harl_runs"),
        },
    }

    scenario_kwargs = {
        "max_steps":     episode_length,
        "n_drones":      joint_drone_count,
        "n_ground":      joint_ugv_count,
        "n_survivors":   survivor_count,
        "comms_dropout": comms_dropout,
        "fire_grid_size": fire_grid_size,
        "local_map_patch_size": local_map_patch_size,
        "ugv_planner_hint": ugv_planner_hint,
        "ugv_planner_detour_obs": bool(ugv_planner_detour_obs),
        "ugv_route_aware_reward": bool(ugv_route_aware_reward),
        "ugv_dense_reward_mode": ugv_dense_reward_mode,
        "ugv_planner_blend_weight": ugv_planner_blend_weight,
        "ugv_escape_stall_steps": ugv_escape_stall_steps,
        "ugv_escape_progress_threshold_m": ugv_escape_progress_threshold_m,
        "ugv_escape_movement_threshold_m": ugv_escape_movement_threshold_m,
        "ugv_escape_waypoint_reached_m": ugv_escape_waypoint_reached_m,
        "ugv_escape_max_steps": ugv_escape_max_steps,
        "ugv_planner_patch_size": ugv_planner_patch_size,
        "ugv_planner_lookahead_cells": ugv_planner_lookahead_cells,
        "ugv_global_planner_lookahead_m": ugv_global_planner_lookahead_m,
        "ugv_global_planner_heuristic": ugv_global_planner_heuristic,
        "ugv_planner_fire_mode": ugv_planner_fire_mode,
        "ugv_planner_fire_replan_policy": ugv_planner_fire_replan_policy,
        "ugv_planner_fire_replan_interval_steps": ugv_planner_fire_replan_interval_steps,
        "ugv_planner_fire_cost": ugv_planner_fire_cost,
        "ugv_planner_fire_block_threshold": ugv_planner_fire_block_threshold,
        "ugv_planner_smoke_cost": ugv_planner_smoke_cost,
        "ugv_planner_smolder_cost": ugv_planner_smolder_cost,
        "ugv_planner_fire_buffer_m": ugv_planner_fire_buffer_m,
        "ugv_planner_fire_buffer_cost": ugv_planner_fire_buffer_cost,
        "ugv_target_assignment_mode": ugv_target_assignment_mode,
        "ugv_assigned_target_obs_only": ugv_assigned_target_obs_only,
        "survivor_assignment_obs": survivor_assignment_obs,
        "ugv_sticky_switch_margin_m": ugv_sticky_switch_margin_m,
        "ugv_sticky_switch_ratio": ugv_sticky_switch_ratio,
        "ugv_sticky_min_age_steps": ugv_sticky_min_age_steps,
        "delayed_survivor_knowledge": bool(delayed_survivor_knowledge),
        "survivor_reveal_schedule": survivor_reveal_schedule,
        "survivor_reveal_initial_count": survivor_reveal_initial_count,
        "survivor_reveal_start_step": survivor_reveal_start_step,
        "survivor_reveal_end_step": survivor_reveal_end_step,
        "drone_min_footprint_m": drone_min_footprint_m,
        "ground_confirm_min_m": ground_confirm_min_m,
        "r_found_survivor": 10.0,
        "r_team_scout": 0.0 if team_scout_reward is None else float(team_scout_reward),
        "r_drone_scout": 2.0,
        "r_ground_confirm": 4.0,
        "r_drone_shaping": 0.30,
        "r_ground_shaping": 0.50,
        "r_ground_approach": ugv_approach_reward,
        "ground_approach_milestone_radii_m": tuple(float(v) for v in ugv_approach_milestone_radii_m),
        "r_ugv_movement_alignment": ugv_movement_alignment_reward,
        "r_ugv_planner_progress": ugv_planner_progress_reward,
        "r_ugv_stall_penalty": ugv_stall_penalty,
        "ugv_stall_displacement_threshold_m": ugv_stall_displacement_threshold_m,
        "r_ugv_route_progress_floor_penalty": ugv_route_progress_floor_penalty,
        "ugv_route_progress_floor_m": ugv_route_progress_floor_m,
        "r_ugv_route_progress_shortfall_penalty": ugv_route_progress_shortfall_penalty,
        "r_fire_penalty": -0.20,
        "r_ground_travel_cost": -0.01,
        "r_drone_climb_cost": -0.005,
        "r_time_penalty": -0.0005,
        "r_coverage": 5.0,
    }
    if slope_speed_weight is not None:
        scenario_kwargs["slope_speed_weight"] = float(slope_speed_weight)
    if land_cover_speeds is not None:
        scenario_kwargs["land_cover_speeds"] = tuple(float(v) for v in land_cover_speeds)
    if ugv_planner_land_cover_costs is not None:
        scenario_kwargs["ugv_planner_land_cover_costs"] = tuple(
            float(v) for v in ugv_planner_land_cover_costs
        )
    if terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = terrain_cache_path
    # Physical (non-floor) sensor levers. On a small terrain these give a real
    # detection footprint at floor 0: footprint ~ flight_altitude * tan(fov/2),
    # both converted to sim units via sim_units_per_meter.
    if drone_camera_fov_deg is not None:
        scenario_kwargs["drone_camera_fov_deg"] = float(drone_camera_fov_deg)
    if drone_flight_levels_m is not None:
        scenario_kwargs["drone_flight_levels_m"] = tuple(float(v) for v in drone_flight_levels_m)
    if ground_confirmation_range_m is not None:
        scenario_kwargs["ground_confirmation_range_m"] = float(ground_confirmation_range_m)
    coverage_obs_grid = 0 if coverage_obs_grid is None else int(coverage_obs_grid)
    if coverage_obs_grid < 0:
        raise ValueError("coverage_obs_grid must be nonnegative")
    if coverage_obs_grid > 0:
        scenario_kwargs["coverage_obs_grid"] = coverage_obs_grid
    if confirm_requires_los:
        scenario_kwargs["confirm_requires_los"] = True
    if drone_can_confirm:
        scenario_kwargs["drone_can_confirm"] = True
        scenario_kwargs["r_drone_confirm"] = float(r_drone_confirm)
    local_coverage_obs_grid = 0 if local_coverage_obs_grid is None else int(local_coverage_obs_grid)
    if local_coverage_obs_grid < 0 or (local_coverage_obs_grid > 0 and local_coverage_obs_grid % 2 != 1):
        raise ValueError("local_coverage_obs_grid must be 0 or a positive odd integer")
    local_coverage_obs_radius_m = float(local_coverage_obs_radius_m)
    if local_coverage_obs_radius_m <= 0.0:
        raise ValueError("local_coverage_obs_radius_m must be positive")
    if local_coverage_obs_grid > 0:
        scenario_kwargs["local_coverage_obs_grid"] = local_coverage_obs_grid
        scenario_kwargs["local_coverage_obs_radius_m"] = local_coverage_obs_radius_m
    uav_confidence_obs_grid = int(uav_confidence_obs_grid)
    if uav_confidence_obs_grid < 0:
        raise ValueError("uav_confidence_obs_grid must be nonnegative")
    if uav_confidence_obs_grid > 0:
        scenario_kwargs["uav_confidence_obs_grid"] = uav_confidence_obs_grid
    uav_local_confidence_obs_grid = int(uav_local_confidence_obs_grid)
    if uav_local_confidence_obs_grid < 0 or (
        uav_local_confidence_obs_grid > 0 and uav_local_confidence_obs_grid % 2 != 1
    ):
        raise ValueError("uav_local_confidence_obs_grid must be 0 or a positive odd integer")
    uav_local_confidence_obs_radius_m = float(uav_local_confidence_obs_radius_m)
    if uav_local_confidence_obs_radius_m <= 0.0:
        raise ValueError("uav_local_confidence_obs_radius_m must be positive")
    if uav_local_confidence_obs_grid > 0:
        scenario_kwargs["local_confidence_obs_grid"] = uav_local_confidence_obs_grid
        scenario_kwargs["local_confidence_obs_radius_m"] = uav_local_confidence_obs_radius_m
    uav_frontier_obs_radius_m = float(uav_frontier_obs_radius_m)
    if uav_frontier_obs_radius_m <= 0.0:
        raise ValueError("uav_frontier_obs_radius_m must be positive")
    uav_frontier_mode = str(uav_frontier_mode).replace("-", "_")
    if uav_frontier_mode not in {"centroid", "sector_topk", "local_global"}:
        raise ValueError("uav_frontier_mode must be one of: centroid, sector_topk, local_global")
    uav_frontier_source = str(uav_frontier_source).replace("-", "_").lower()
    if uav_frontier_source not in {"coverage", "confidence"}:
        raise ValueError("uav_frontier_source must be one of: coverage, confidence")
    uav_frontier_sectors = int(uav_frontier_sectors)
    if uav_frontier_sectors < 2:
        raise ValueError("uav_frontier_sectors must be at least 2")
    uav_frontier_top_k = int(uav_frontier_top_k)
    if uav_frontier_top_k < 1 or uav_frontier_top_k > uav_frontier_sectors:
        raise ValueError("uav_frontier_top_k must be in [1, uav_frontier_sectors]")
    uav_frontier_alignment_reward = float(uav_frontier_alignment_reward)
    if uav_frontier_alignment_reward < 0.0:
        raise ValueError("uav_frontier_alignment_reward must be nonnegative")
    uav_cleanup_target_grid = int(uav_cleanup_target_grid)
    if uav_cleanup_target_grid < 2:
        raise ValueError("uav_cleanup_target_grid must be at least 2")
    uav_cleanup_target_hold_steps = int(uav_cleanup_target_hold_steps)
    if uav_cleanup_target_hold_steps < 1:
        raise ValueError("uav_cleanup_target_hold_steps must be positive")
    if not 0.0 <= float(uav_cleanup_target_confidence_threshold) <= 1.0:
        raise ValueError("uav_cleanup_target_confidence_threshold must be in [0, 1]")
    if float(uav_cleanup_target_min_value) < 0.0:
        raise ValueError("uav_cleanup_target_min_value must be nonnegative")
    if float(uav_cleanup_target_assignment_distance_scale_m) <= 0.0:
        raise ValueError("uav_cleanup_target_assignment_distance_scale_m must be positive")
    uav_cleanup_target_refresh_mode = str(uav_cleanup_target_refresh_mode).replace("-", "_").lower()
    if uav_cleanup_target_refresh_mode not in {"exact", "fixed_hold"}:
        raise ValueError("uav_cleanup_target_refresh_mode must be one of: exact, fixed_hold")
    uav_astar_grid = int(uav_astar_grid)
    if uav_astar_grid < 2:
        raise ValueError("uav_astar_grid must be at least 2")
    uav_astar_confidence_cost_alpha = float(uav_astar_confidence_cost_alpha)
    if uav_astar_confidence_cost_alpha < 0.0:
        raise ValueError("uav_astar_confidence_cost_alpha must be nonnegative")
    uav_astar_confidence_cost_gamma = float(uav_astar_confidence_cost_gamma)
    if uav_astar_confidence_cost_gamma < 0.0:
        raise ValueError("uav_astar_confidence_cost_gamma must be nonnegative")
    if float(uav_astar_waypoint_lookahead_m) <= 0.0:
        raise ValueError("uav_astar_waypoint_lookahead_m must be positive")
    uav_astar_route_replan_steps = int(uav_astar_route_replan_steps)
    if uav_astar_route_replan_steps < 1:
        raise ValueError("uav_astar_route_replan_steps must be positive")
    if float(uav_astar_waypoint_reached_m) <= 0.0:
        raise ValueError("uav_astar_waypoint_reached_m must be positive")
    if uav_frontier_obs or uav_frontier_alignment_reward > 0.0:
        scenario_kwargs["uav_frontier_obs"] = bool(uav_frontier_obs)
        scenario_kwargs["uav_frontier_obs_radius_m"] = uav_frontier_obs_radius_m
        scenario_kwargs["uav_frontier_mode"] = uav_frontier_mode
        scenario_kwargs["uav_frontier_source"] = uav_frontier_source
        scenario_kwargs["uav_frontier_sectors"] = uav_frontier_sectors
        scenario_kwargs["uav_frontier_top_k"] = uav_frontier_top_k
        scenario_kwargs["uav_frontier_ownership"] = bool(uav_frontier_ownership)
        scenario_kwargs["r_uav_frontier_alignment"] = uav_frontier_alignment_reward
    if (
        uav_cleanup_target_obs
        or uav_cleanup_target_progress_reward > 0.0
        or uav_astar_route_obs
        or uav_astar_progress_reward > 0.0
    ):
        if uav_cleanup_target_obs:
            scenario_kwargs["uav_cleanup_target_obs"] = True
        scenario_kwargs["uav_cleanup_target_grid"] = uav_cleanup_target_grid
        scenario_kwargs["uav_cleanup_target_hold_steps"] = uav_cleanup_target_hold_steps
        scenario_kwargs["uav_cleanup_target_confidence_threshold"] = float(
            uav_cleanup_target_confidence_threshold
        )
        scenario_kwargs["uav_cleanup_target_min_value"] = float(uav_cleanup_target_min_value)
        scenario_kwargs["uav_cleanup_target_assignment_distance_scale_m"] = float(
            uav_cleanup_target_assignment_distance_scale_m
        )
        scenario_kwargs["uav_cleanup_target_refresh_mode"] = uav_cleanup_target_refresh_mode
        scenario_kwargs["r_uav_cleanup_target_progress"] = uav_cleanup_target_progress_reward
        scenario_kwargs["r_uav_astar_progress"] = uav_astar_progress_reward
        if uav_astar_route_obs or uav_astar_progress_reward > 0.0:
            scenario_kwargs["uav_astar_route_obs"] = bool(uav_astar_route_obs)
            scenario_kwargs["uav_astar_grid"] = uav_astar_grid
            scenario_kwargs["uav_astar_confidence_cost_alpha"] = uav_astar_confidence_cost_alpha
            scenario_kwargs["uav_astar_confidence_cost_gamma"] = uav_astar_confidence_cost_gamma
            scenario_kwargs["uav_astar_waypoint_lookahead_m"] = float(uav_astar_waypoint_lookahead_m)
            scenario_kwargs["uav_astar_route_replan_steps"] = uav_astar_route_replan_steps
            scenario_kwargs["uav_astar_waypoint_reached_m"] = float(uav_astar_waypoint_reached_m)
    if (
        uav_confidence_reward > 0.0
        or uav_team_confidence_reward > 0.0
        or uav_team_confidence_overlap_penalty > 0.0
        or uav_confidence_move_reward > 0.0
        or uav_confidence_overlap_penalty > 0.0
        or uav_cleanup_target_progress_reward > 0.0
        or uav_astar_progress_reward > 0.0
        or uav_astar_route_obs
        or uav_frontier_source == "confidence"
    ):
        scenario_kwargs["uav_frontier_source"] = uav_frontier_source
        scenario_kwargs["r_uav_confidence"] = uav_confidence_reward
        scenario_kwargs["r_uav_team_confidence"] = uav_team_confidence_reward
        scenario_kwargs["r_uav_team_confidence_overlap"] = uav_team_confidence_overlap_penalty
        scenario_kwargs["r_uav_confidence_move"] = uav_confidence_move_reward
        scenario_kwargs["r_uav_confidence_overlap"] = uav_confidence_overlap_penalty
        scenario_kwargs["uav_confidence_overlap_mode"] = uav_confidence_overlap_mode
        scenario_kwargs["uav_confidence_overlap_allowed_regret"] = uav_confidence_overlap_allowed_regret
        scenario_kwargs["uav_confidence_overlap_threshold"] = uav_confidence_overlap_threshold
        scenario_kwargs["uav_confidence_gamma"] = uav_confidence_gamma
        scenario_kwargs["uav_confidence_eps"] = uav_confidence_eps
        scenario_kwargs["uav_confidence_opportunity_eps"] = uav_confidence_opportunity_eps
    if uav_start_min_separation_m is not None:
        scenario_kwargs["uav_start_min_separation_m"] = uav_start_min_separation_m
    if uav_start_edge_margin_m is not None:
        scenario_kwargs["uav_start_edge_margin_m"] = uav_start_edge_margin_m
    if reward_search:
        # Kept explicit for the legacy flag; these now match the default reward
        # profile used by normal smoke training.
        scenario_kwargs.update({
            "r_found_survivor": 10.0,
            "r_drone_scout": 2.0,
            "r_ground_confirm": 4.0,
            "r_drone_shaping": 0.30,
            "r_ground_shaping": 0.50,
            "r_ugv_movement_alignment": ugv_movement_alignment_reward,
            "r_ugv_planner_progress": ugv_planner_progress_reward,
            "r_fire_penalty": -0.20,
            "r_ground_travel_cost": -0.01,
            "r_drone_climb_cost": -0.005,
            "r_time_penalty": -0.0005,
            "r_coverage": 5.0,
        })
    if reward_confirm:
        # Confirmation-dominant reward. Diagnosis: under reward_search the ground
        # robots learn to stand still (UGV travel ~2.86 vs expert ~4.8) because
        # confirming is rare/costly and idling is a safe zero. This makes ground
        # confirmation the dominant signal, removes the costs that punish moving,
        # strengthens the dense pull toward scouted survivors, and adds a per-step
        # penalty for every scouted-but-unconfirmed survivor still waiting.
        scenario_kwargs.update({
            "r_found_survivor":      15.0,
            "r_ground_confirm":      12.0,   # dominant term
            "r_drone_scout":          1.0,   # was 2.0 (don't over-reward scouting alone)
            "r_drone_shaping":        0.15,
            "r_ground_shaping":       1.5,    # strong potential-based pull
            "r_ground_approach":      1.0,    # strong dense bonus peaking ON a survivor
            "ground_approach_radius": 0.6,
            "r_fire_penalty":         0.0,    # cut: was making robots avoid moving
            "r_ground_travel_cost":   0.0,    # cut: was making standing still optimal
            "r_drone_climb_cost":     0.0,
            "r_time_penalty":        -0.001,
            "r_coverage":             0.5,    # small: encourage spread, don't farm it
            "r_pending_penalty":     -0.005,  # per scouted-unconfirmed survivor per step
            "r_ground_coverage":      3.0,    # ground robots sweep instead of waiting
            "ground_coverage_radius": 0.08,
        })

    diagnostic_modes = [
        bool(ugv_known_survivor_diagnostic),
        bool(uav_survivor_diagnostic),
        bool(joint_schema_uav_diagnostic),
        bool(joint_survivor_diagnostic),
        bool(joint_schema_ugv_diagnostic),
    ]
    if sum(diagnostic_modes) > 1:
        raise ValueError("Choose only one diagnostic mode: UGV, UAV, joint-schema UAV, joint, or joint-schema UGV")

    if ugv_known_survivor_diagnostic:
        distance_kwargs = {}
        if ugv_diagnostic_target_distance_min_m is None and ugv_diagnostic_target_distance_max_m is None:
            pass
        else:
            target_distance_min_m = max(
                float(0.0 if ugv_diagnostic_target_distance_min_m is None else ugv_diagnostic_target_distance_min_m),
                0.0,
            )
            distance_kwargs["known_survivor_spawn_distance_min_m"] = target_distance_min_m
            if ugv_diagnostic_target_distance_max_m is not None:
                target_distance_max_m = max(
                    float(ugv_diagnostic_target_distance_max_m),
                    0.0,
                )
                if target_distance_max_m < target_distance_min_m:
                    raise ValueError(
                        "ugv_diagnostic_target_distance_max_m must be >= "
                        "ugv_diagnostic_target_distance_min_m"
                    )
                target_distance_m = 0.5 * (target_distance_min_m + target_distance_max_m)
                distance_kwargs.update({
                    "known_survivor_spawn_distance_m": target_distance_m,
                    "known_survivor_spawn_distance_max_m": target_distance_max_m,
                })
        scenario_kwargs.update({
            "n_drones": 0,
            "n_ground": 1 if n_ugvs is None else max(int(n_ugvs), 1),
            "n_survivors": 1 if n_survivors is None else survivor_count,
            "known_survivors_at_reset": True,
            "disable_fire": not bool(enable_fire),
            "comms_dropout": 0.0,
            "r_found_survivor": 10.0,
            "r_drone_scout": 0.0,
            "r_ground_confirm": 4.0,
            "r_drone_shaping": 0.0,
            "r_ground_shaping": 0.50,
            "r_ground_approach": ugv_approach_reward,
            "ground_approach_milestone_radii_m": tuple(float(v) for v in ugv_approach_milestone_radii_m),
            "r_ugv_movement_alignment": ugv_movement_alignment_reward,
            "r_ugv_planner_progress": ugv_planner_progress_reward,
            "r_ugv_stall_penalty": ugv_stall_penalty,
            "ugv_stall_displacement_threshold_m": ugv_stall_displacement_threshold_m,
            "r_fire_penalty": 0.0,
            "r_ground_travel_cost": 0.0,
            "r_drone_climb_cost": 0.0,
            "r_time_penalty": -0.0005,
            "r_coverage": 0.0,
        })
        scenario_kwargs.update(distance_kwargs)
    if uav_survivor_diagnostic or joint_schema_uav_diagnostic:
        found_reward = 0.0
        all_survivors_reward = 0.0
        scout_reward = 0.0 if uav_coverage_only else 2.0
        time_penalty = 0.0
        if uav_found_survivor_reward is not None:
            found_reward = float(uav_found_survivor_reward)
        if uav_all_survivors_reward is not None:
            all_survivors_reward = float(uav_all_survivors_reward)
        if uav_time_penalty is not None:
            time_penalty = float(uav_time_penalty)
        coverage_threshold_reward = (
            0.0
            if uav_coverage_threshold_reward is None
            else float(uav_coverage_threshold_reward)
        )
        scenario_kwargs.update({
            "n_drones": uav_diagnostic_drones,
            "n_ground": 0,
            "n_survivors": survivor_count,
            "known_survivors_at_reset": False,
            "drone_can_confirm": True,
            "disable_fire": not bool(enable_fire),
            "comms_dropout": 0.0,
            "r_found_survivor": found_reward,
            "r_all_survivors_found": all_survivors_reward,
            "r_team_scout": 0.0 if team_scout_reward is None else float(team_scout_reward),
            "r_drone_scout": scout_reward,
            "r_ground_confirm": 0.0,
            "r_drone_shaping": 0.0,
            "r_ground_shaping": 0.0,
            "r_ground_approach": 0.0,
            "r_ugv_movement_alignment": 0.0,
            "r_ugv_planner_progress": 0.0,
            "r_ugv_stall_penalty": 0.0,
            "r_fire_penalty": 0.0,
            "r_ground_travel_cost": 0.0,
            "r_drone_climb_cost": 0.0,
            "r_time_penalty": time_penalty,
            "r_coverage": uav_coverage_reward,
            "uav_coverage_normalization": uav_coverage_normalization,
            "r_uav_move_coverage": uav_move_coverage_reward,
            "uav_move_coverage_normalization": uav_move_coverage_normalization,
            "r_uav_move_coverage_cap": uav_move_coverage_cap,
            "r_uav_coverage_threshold": coverage_threshold_reward,
            "uav_coverage_threshold_fraction": uav_coverage_threshold_fraction,
            "uav_coverage_opportunity_cap": uav_coverage_opportunity_cap,
            "r_uav_confidence": uav_confidence_reward,
            "r_uav_team_confidence": uav_team_confidence_reward,
            "r_uav_team_confidence_overlap": uav_team_confidence_overlap_penalty,
            "r_uav_confidence_move": uav_confidence_move_reward,
            "r_uav_inefficient_move": uav_inefficient_move_penalty,
            "uav_inefficient_move_source": uav_inefficient_move_source,
            "r_uav_confidence_overlap": uav_confidence_overlap_penalty,
            "uav_confidence_overlap_mode": uav_confidence_overlap_mode,
            "uav_confidence_overlap_allowed_regret": uav_confidence_overlap_allowed_regret,
            "r_uav_cleanup_target_progress": uav_cleanup_target_progress_reward,
            "r_uav_astar_progress": uav_astar_progress_reward,
            "uav_cleanup_target_refresh_mode": uav_cleanup_target_refresh_mode,
            "uav_astar_route_obs": bool(uav_astar_route_obs),
            "uav_astar_grid": uav_astar_grid,
            "uav_astar_confidence_cost_alpha": uav_astar_confidence_cost_alpha,
            "uav_astar_confidence_cost_gamma": uav_astar_confidence_cost_gamma,
            "uav_astar_waypoint_lookahead_m": float(uav_astar_waypoint_lookahead_m),
            "uav_astar_route_replan_steps": uav_astar_route_replan_steps,
            "uav_astar_waypoint_reached_m": float(uav_astar_waypoint_reached_m),
            "uav_confidence_overlap_threshold": uav_confidence_overlap_threshold,
            "uav_confidence_gamma": uav_confidence_gamma,
            "uav_confidence_eps": uav_confidence_eps,
            "uav_confidence_opportunity_eps": uav_confidence_opportunity_eps,
            "uav_frontier_source": uav_frontier_source,
            "r_uav_overlap": uav_overlap_penalty,
            "uav_overlap_allowed": uav_overlap_allowed,
            "uav_overlap_penalty_normalization": uav_overlap_penalty_normalization,
            "r_uav_inter_uav_overlap": uav_inter_uav_overlap_penalty,
            "uav_inter_uav_overlap_allowed": uav_inter_uav_overlap_allowed,
            "r_uav_outside_footprint": uav_outside_footprint_penalty,
            "uav_boundary_soft_margin_m": uav_boundary_soft_margin_m,
        })
        if joint_schema_uav_diagnostic:
            scenario_kwargs.update({
                "obs_schema_n_drones": joint_drone_count,
                "obs_schema_n_ground": joint_ugv_count,
                "obs_schema_n_survivors": survivor_count,
                "ugv_assigned_target_obs_only": False,
                "survivor_assignment_obs": True,
            })
    if joint_survivor_diagnostic:
        joint_diagnostic_ugvs = max(int(joint_diagnostic_ugvs), 1)
        joint_drone_count = max(int(joint_drone_count), 1)
        coverage_threshold_reward = (
            0.0
            if uav_coverage_threshold_reward is None
            else float(uav_coverage_threshold_reward)
        )
        scenario_kwargs.update({
            "n_drones": joint_drone_count,
            "n_ground": joint_diagnostic_ugvs,
            "n_survivors": survivor_count,
            "known_survivors_at_reset": False,
            "drone_can_confirm": False,
            "disable_fire": not bool(enable_fire),
            "comms_dropout": 0.0,
            "ugv_target_assignment_mode": ugv_target_assignment_mode,
            "ugv_assigned_target_obs_only": ugv_assigned_target_obs_only,
            "survivor_assignment_obs": True,
            "r_found_survivor": DEFAULT_JOINT_DIAG_TEAM_CONFIRM_REWARD,
            "r_team_scout": (
                DEFAULT_JOINT_DIAG_TEAM_SCOUT_REWARD
                if team_scout_reward is None
                else float(team_scout_reward)
            ),
            "r_all_survivors_found": 0.0,
            "r_drone_scout": 2.0,
            "r_ground_confirm": DEFAULT_JOINT_DIAG_GROUND_CONFIRM_REWARD,
            "r_drone_shaping": 0.0,
            "r_ground_shaping": 0.50,
            "r_ground_approach": 0.0,
            "r_ugv_movement_alignment": ugv_movement_alignment_reward,
            "r_ugv_planner_progress": 0.0,
            "r_ugv_stall_penalty": ugv_stall_penalty,
            "r_pending_penalty": DEFAULT_JOINT_DIAG_PENDING_PENALTY,
            "r_fire_penalty": 0.0,
            "r_ground_travel_cost": 0.0,
            "r_drone_climb_cost": 0.0,
            "r_time_penalty": 0.0,
            "r_coverage": uav_coverage_reward,
            "uav_coverage_normalization": uav_coverage_normalization,
            "r_uav_move_coverage": uav_move_coverage_reward,
            "uav_move_coverage_normalization": uav_move_coverage_normalization,
            "r_uav_move_coverage_cap": uav_move_coverage_cap,
            "r_uav_coverage_threshold": coverage_threshold_reward,
            "uav_coverage_threshold_fraction": uav_coverage_threshold_fraction,
            "uav_coverage_opportunity_cap": uav_coverage_opportunity_cap,
            "r_uav_confidence": uav_confidence_reward,
            "r_uav_team_confidence": uav_team_confidence_reward,
            "r_uav_team_confidence_overlap": uav_team_confidence_overlap_penalty,
            "r_uav_confidence_move": uav_confidence_move_reward,
            "r_uav_inefficient_move": uav_inefficient_move_penalty,
            "uav_inefficient_move_source": uav_inefficient_move_source,
            "r_uav_confidence_overlap": uav_confidence_overlap_penalty,
            "uav_confidence_overlap_mode": uav_confidence_overlap_mode,
            "uav_confidence_overlap_allowed_regret": uav_confidence_overlap_allowed_regret,
            "r_uav_cleanup_target_progress": uav_cleanup_target_progress_reward,
            "r_uav_astar_progress": uav_astar_progress_reward,
            "uav_cleanup_target_refresh_mode": uav_cleanup_target_refresh_mode,
            "uav_astar_route_obs": bool(uav_astar_route_obs),
            "uav_confidence_overlap_threshold": uav_confidence_overlap_threshold,
            "uav_confidence_gamma": uav_confidence_gamma,
            "uav_confidence_eps": uav_confidence_eps,
            "uav_confidence_opportunity_eps": uav_confidence_opportunity_eps,
            "r_uav_overlap": uav_overlap_penalty,
            "uav_overlap_allowed": uav_overlap_allowed,
            "uav_overlap_penalty_normalization": uav_overlap_penalty_normalization,
            "r_uav_inter_uav_overlap": uav_inter_uav_overlap_penalty,
            "uav_inter_uav_overlap_allowed": uav_inter_uav_overlap_allowed,
            "r_uav_outside_footprint": uav_outside_footprint_penalty,
            "uav_boundary_soft_margin_m": uav_boundary_soft_margin_m,
        })
    if joint_schema_ugv_diagnostic:
        scenario_kwargs.update({
            "n_drones": 0,
            "n_ground": joint_ugv_count,
            "n_survivors": survivor_count,
            "obs_schema_n_drones": joint_drone_count,
            "obs_schema_n_ground": joint_ugv_count,
            "obs_schema_n_survivors": survivor_count,
            "known_survivors_at_reset": False,
            "delayed_survivor_knowledge": True,
            "survivor_reveal_schedule": survivor_reveal_schedule,
            "survivor_reveal_initial_count": int(survivor_reveal_initial_count),
            "survivor_reveal_start_step": int(survivor_reveal_start_step),
            "survivor_reveal_end_step": int(survivor_reveal_end_step),
            "drone_can_confirm": False,
            "disable_fire": not bool(enable_fire),
            "comms_dropout": 0.0,
            "ugv_target_assignment_mode": ugv_target_assignment_mode,
            "ugv_zero_uav_search_observations": True,
            "ugv_assigned_target_obs_only": ugv_assigned_target_obs_only,
            "survivor_assignment_obs": True,
            "r_found_survivor": DEFAULT_JOINT_DIAG_TEAM_CONFIRM_REWARD,
            "r_team_scout": 0.0 if team_scout_reward is None else float(team_scout_reward),
            "r_all_survivors_found": 0.0,
            "r_drone_scout": 0.0,
            "r_ground_confirm": DEFAULT_JOINT_DIAG_GROUND_CONFIRM_REWARD,
            "r_drone_shaping": 0.0,
            "r_ground_shaping": 0.50,
            "r_ground_approach": 0.0,
            "r_ugv_movement_alignment": ugv_movement_alignment_reward,
            "r_ugv_planner_progress": 0.0,
            "r_ugv_stall_penalty": ugv_stall_penalty,
            "r_pending_penalty": DEFAULT_JOINT_DIAG_PENDING_PENALTY,
            "r_fire_penalty": 0.0,
            "r_ground_travel_cost": 0.0,
            "r_drone_climb_cost": 0.0,
            "r_time_penalty": 0.0,
            "r_coverage": 0.0,
            "r_uav_move_coverage": 0.0,
            "r_uav_coverage_threshold": 0.0,
            "r_uav_confidence": 0.0,
            "r_uav_team_confidence": 0.0,
            "r_uav_team_confidence_overlap": 0.0,
            "r_uav_confidence_move": 0.0,
            "r_uav_inefficient_move": 0.0,
            "r_uav_frontier_alignment": 0.0,
            "r_uav_confidence_overlap": 0.0,
            "r_uav_cleanup_target_progress": 0.0,
            "r_uav_astar_progress": 0.0,
            "r_uav_overlap": 0.0,
            "r_uav_inter_uav_overlap": 0.0,
            "r_uav_outside_footprint": 0.0,
        })
    n_agents = int(scenario_kwargs["n_drones"]) + int(scenario_kwargs["n_ground"])
    if share_param_by_agent_class:
        share_param_groups: list[int] = []
        share_param_group_names: list[str] = []
        n_drones = int(scenario_kwargs["n_drones"])
        n_ground = int(scenario_kwargs["n_ground"])
        if n_drones > 0:
            share_param_group_names.append("uav")
            share_param_groups.extend([len(share_param_group_names) - 1] * n_drones)
        if n_ground > 0:
            share_param_group_names.append("ugv")
            share_param_groups.extend([len(share_param_group_names) - 1] * n_ground)
        if len(share_param_groups) != n_agents:
            raise ValueError("share_param_by_agent_class group mapping does not match agent count")
        algo_args["algo"]["share_param_groups"] = share_param_groups
        algo_args["algo"]["share_param_group_names"] = share_param_group_names
    if ugv_ground_shaping_reward is not None:
        scenario_kwargs["r_ground_shaping"] = float(ugv_ground_shaping_reward)
    if ugv_pending_penalty is not None:
        scenario_kwargs["r_pending_penalty"] = float(ugv_pending_penalty)
    obs_dim_n_agents = int(scenario_kwargs.get("obs_schema_n_drones", scenario_kwargs["n_drones"])) + int(
        scenario_kwargs.get("obs_schema_n_ground", scenario_kwargs["n_ground"])
    )
    obs_dim_n_survivors = int(
        scenario_kwargs.get("obs_schema_n_survivors", scenario_kwargs["n_survivors"])
    )
    algo_args["model"]["terrain_cnn_single_obs_dim"] = wildfire_single_observation_dim(
        local_map_patch_size=int(local_map_patch_size),
        n_agents=obs_dim_n_agents,
        n_survivors=obs_dim_n_survivors,
        ugv_planner_hint=ugv_planner_hint,
        ugv_planner_detour_obs=bool(ugv_planner_detour_obs),
        coverage_obs_grid=int(coverage_obs_grid),
        local_coverage_obs_grid=int(local_coverage_obs_grid),
        uav_confidence_obs_grid=int(uav_confidence_obs_grid),
        local_confidence_obs_grid=int(uav_local_confidence_obs_grid),
        uav_frontier_obs=bool(uav_frontier_obs),
        uav_frontier_mode=uav_frontier_mode,
        uav_frontier_top_k=uav_frontier_top_k,
        uav_cleanup_target_obs=bool(uav_cleanup_target_obs),
        uav_astar_route_obs=bool(uav_astar_route_obs),
        survivor_assignment_obs=bool(survivor_assignment_obs),
    )
    env_args = {
        "max_cycles":      episode_length,
        "scenario_kwargs": scenario_kwargs,
        "action_transform": action_transform,
    }

    return args, algo_args, env_args


# ----------------------------------------------------------------------
# Step 3 — CLI + main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research",       action="store_true",
                   help="Larger budget: 400k steps, episode_length=500 (tens of minutes on CPU).")
    p.add_argument("--num-env-steps",  type=int,   default=None,
                   help="Total env steps override (default: 2000 smoke / 400000 research).")
    p.add_argument("--episode-length", type=int,   default=None,
                   help="Episode length override (default: 150 smoke / 500 research).")
    p.add_argument("--seed",           type=int,   default=1)
    p.add_argument("--comms-dropout",  type=float, default=0.0,
                   help="Per-step prob each agent's comms are dropped (default: 0.0).")
    p.add_argument("--entropy-coef",   type=float, default=0.01,
                   help="Higher (0.05+) encourages exploration — helps break "
                        "the drones-at-corners action-saturation pathology.")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Actor learning rate.")
    p.add_argument("--critic-lr", type=float, default=5e-4,
                   help="Critic learning rate.")
    p.set_defaults(linear_lr_decay=None)
    p.add_argument("--linear-lr-decay", dest="linear_lr_decay", action="store_true",
                   help="Linearly decay actor/critic learning rates over training.")
    p.add_argument("--no-linear-lr-decay", dest="linear_lr_decay", action="store_false",
                   help="Disable linear LR decay, overriding diagnostic preset defaults.")
    p.set_defaults(share_param=None)
    p.add_argument("--share-param", dest="share_param", action="store_true",
                   help="Enable HARL global actor parameter sharing. Use only for homogeneous-agent "
                        "runs such as UAV-only diagnostics; mixed UAV/UGV sharing needs class-wise sharing.")
    p.add_argument("--no-share-param", dest="share_param", action="store_false",
                   help="Disable HARL global actor parameter sharing, overriding UAV diagnostic defaults.")
    p.set_defaults(share_param_by_agent_class=None)
    p.add_argument("--share-param-by-agent-class", dest="share_param_by_agent_class", action="store_true",
                   help="Share actor parameters within each agent class, e.g. one UAV policy and one UGV policy.")
    p.add_argument("--no-share-param-by-agent-class", dest="share_param_by_agent_class", action="store_false",
                   help="Disable class-wise actor parameter sharing, overriding joint diagnostic defaults.")
    p.add_argument("--terrain-cnn-encoder", action="store_true",
                   help="Encode the mobility/blocked local map patch with a tiny CNN before the HAPPO MLP.")
    p.add_argument("--terrain-cnn-embed-dim", type=int, default=16,
                   help="Embedding size per local terrain patch when --terrain-cnn-encoder is enabled.")
    p.add_argument("--exp-name",       default="happo_smoke")
    p.add_argument("--n-rollout-threads", type=int, default=1,
                   help="Parallel rollout envs. More threads => more diverse data per "
                        "update and faster wall-clock (e.g. 8).")
    p.add_argument("--terrain-cache-path", default=None,
                   help="Train on this cached real terrain (recommended: match what you evaluate on, "
                        "e.g. data/terrain_cache/malibu_creek_1km_128.npz). Default uses the scenario default.")
    p.add_argument(
                   "--drone-min-footprint-radius-m",
                   dest="drone_min_footprint_radius_m",
                   type=float,
                   default=0.0,
                   help="Minimum drone scout-footprint radius in meters. "
                        "The terrain scale converts it internally; 0 disables the floor.")
    p.add_argument("--drone-min-footprint",
                   dest="drone_min_footprint_radius_m",
                   type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument(
                   "--ground-min-confirm-radius-m",
                   dest="ground_min_confirm_radius_m",
                   type=float,
                   default=0.0,
                   help="Minimum ground confirmation radius in meters. "
                        "The terrain scale converts it internally; 0 disables the floor.")
    p.add_argument("--ground-confirm-min",
                   dest="ground_min_confirm_radius_m",
                   type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--fire-grid-size", type=int, default=128)
    p.add_argument("--local-map-patch-size", type=int, default=3,
                   help="Odd square patch size for local mobility and blocked-cell observations. "
                        "All agents receive this patch plus a fixed 3x3 aerial-clearance patch.")
    p.add_argument("--ugv-planner-hint",
                   choices=(
                       "none",
                       "local_astar",
                       "local-astar",
                       "local_escape_astar",
                       "local-escape-astar",
                       "global_astar",
                       "global-astar",
                   ),
                   default="none",
                   help="Optional UGV observation hint. local_astar exposes a local A* waypoint vector; "
                        "local_escape_astar uses the same shape with escape-aware local exits; "
                        "global_astar follows a cached full-map static route.")
    p.add_argument("--ugv-planner-detour-obs", action="store_true",
                   help="Append a detour-needed bit to the local A* UGV planner hint. "
                        "Changes observation size, so use only for new A* trainings.")
    p.add_argument("--ugv-route-aware-reward", action="store_true",
                   help="When local A* detects a detour, suppress direct target-distance and "
                        "movement-alignment rewards for that UGV step. Requires a local UGV planner hint.")
    p.add_argument("--ugv-dense-reward-mode",
                   choices=(
                       "target",
                       "positive_target",
                       "positive-target",
                       "planner_blend",
                       "planner-blend",
                       "escape_blend",
                       "escape-blend",
                       "escape_route_switch",
                       "escape-route-switch",
                       "planner_follow",
                       "planner-follow",
                   ),
                   default="target",
                   help="How UGV dense progress/alignment rewards are shaped. target keeps legacy "
                        "signed survivor homing; positive_target clips survivor progress/alignment "
                        "to nonnegative; planner_blend blends nonnegative survivor and local-A* "
                        "signals during local detours; escape_blend only blends during "
                        "local_escape_astar escape steps; escape_route_switch keeps survivor "
                        "shaping until local_astar detects sustained stalled blocked motion, then "
                        "temporarily follows a stored escape route; planner_follow replaces dense "
                        "survivor shaping with the global_astar waypoint.")
    p.add_argument("--ugv-planner-blend-weight", type=float, default=0.70,
                   help="Planner weight used by --ugv-dense-reward-mode planner_blend during local detours.")
    p.add_argument("--ugv-escape-stall-steps", type=int, default=5,
                   help="Consecutive blocked/low-progress UGV steps before escape_route_switch enters escape mode.")
    p.add_argument("--ugv-escape-progress-threshold-m", type=float, default=0.10,
                   help="Maximum target progress per step treated as stalled for escape_route_switch.")
    p.add_argument("--ugv-escape-movement-threshold-m", type=float, default=0.25,
                   help="Movement below this per step can contribute to escape_route_switch stall detection.")
    p.add_argument("--ugv-escape-waypoint-reached-m", type=float, default=4.0,
                   help="Distance threshold for reaching escape route waypoints or final exits.")
    p.add_argument("--ugv-escape-max-steps", type=int, default=15,
                   help="Maximum steps to stay in one escape_route_switch episode before returning to normal mode.")
    p.add_argument("--ugv-planner-patch-size", type=int, default=11,
                   help="Odd local grid size used by local UGV planner hints.")
    p.add_argument("--ugv-planner-lookahead-cells", type=int, default=10,
                   help="Maximum number of A* route cells to skip ahead when forming the waypoint hint.")
    p.add_argument("--ugv-global-planner-lookahead-m", type=float, default=20.0,
                   help="Physical lookahead distance for global_astar waypoint hints.")
    p.add_argument("--ugv-global-planner-heuristic",
                   choices=("euclidean", "terrain"),
                   default="euclidean",
                   help="A* heuristic for global_astar. euclidean preserves legacy behavior; "
                        "terrain uses a cached terrain-only reverse-Dijkstra lower bound.")
    p.add_argument("--ugv-planner-fire-mode",
                   choices=("off", "cost", "block"),
                   default="off",
                   help="Fire treatment for UGV A* planners. off ignores fire; cost adds fire/smoke costs; "
                        "block treats active fire as non-traversable and uses soft smoke/buffer costs.")
    p.add_argument("--ugv-planner-fire-replan-policy",
                   choices=("always", "affected", "lazy"),
                   default="always",
                   help="When fire-aware planning is enabled, always replan on fire spread or only when "
                        "active fire/buffer touches the cached global route. lazy only replans when "
                        "the near route segment is risky or the interval expires.")
    p.add_argument("--ugv-planner-fire-replan-interval-steps", type=int, default=15,
                   help="For --ugv-planner-fire-replan-policy lazy, maximum fire-change steps "
                        "between full global replans.")
    p.add_argument("--ugv-planner-fire-cost", type=float, default=25.0,
                   help="Additional movement cost for active fire cells in cost mode.")
    p.add_argument("--ugv-planner-fire-block-threshold", type=float, default=0.0,
                   help="In block mode, only active fire cells with intensity >= this threshold "
                        "are non-traversable. Default 0.0 preserves binary fire blocking; "
                        "sub-threshold active fire receives soft planner fire cost.")
    p.add_argument("--ugv-planner-smoke-cost", type=float, default=5.0,
                   help="Additional movement cost multiplier for smoke intensity in fire-aware planner modes.")
    p.add_argument("--ugv-planner-smolder-cost", type=float, default=3.0,
                   help="Additional movement cost multiplier for smolder intensity in fire-aware planner modes.")
    p.add_argument("--ugv-planner-fire-buffer-m", type=float, default=10.0,
                   help="Physical radius around active fire that receives a soft planner cost.")
    p.add_argument("--ugv-planner-fire-buffer-cost", type=float, default=8.0,
                   help="Additional movement cost for non-burning cells inside the active-fire buffer.")
    p.add_argument("--ugv-planner-land-cover-costs", type=float, nargs="+", default=None,
                   help="Planner-only land-cover costs for road/open/brush/forest/rock[/water]. "
                        "Physical UGV speeds and terrain observations are unchanged. "
                        "Example: --ugv-planner-land-cover-costs 0.85 1.0 1.15 1.35 4.0 8.0")
    p.add_argument("--ugv-target-assignment-mode",
                   choices=(
                       "nearest",
                       "greedy",
                       "greedy_sticky",
                       "greedy-sticky",
                       "route_cost_greedy",
                       "route-cost-greedy",
                       "route_cost_sticky",
                       "route-cost-sticky",
                       "route_cost_global",
                       "route-cost-global",
                   ),
                   default=None,
                   help="How UGV planner targets are selected from known, unconfirmed survivors. "
                        "Omit to use the diagnostic-mode default.")
    p.add_argument("--ugv-assigned-target-obs-only", action=argparse.BooleanOptionalAction, default=None,
                   help="Legacy compatibility flag. UGV survivor-message observations now keep all "
                        "known survivor slots visible; assignment still controls planner hints/rewards.")
    p.add_argument("--survivor-assignment-obs", action=argparse.BooleanOptionalAction, default=None,
                   help="Append assigned_to_me and assigned_to_other_ugv flags to each survivor-message "
                        "slot. Defaults on for joint UAV/UGV schema diagnostics.")
    p.add_argument("--ugv-sticky-switch-margin-m", type=float, default=20.0,
                   help="Sticky assignment switches only if the new target beats this absolute margin.")
    p.add_argument("--ugv-sticky-switch-ratio", type=float, default=0.80,
                   help="Sticky assignment switches only if new_distance < current_distance * ratio after margin.")
    p.add_argument("--ugv-sticky-min-age-steps", type=int, default=10,
                   help="Minimum target age before sticky assignment can switch to a better target.")
    p.set_defaults(enable_fire=None)
    p.add_argument("--enable-fire", dest="enable_fire", action="store_true",
                   help="Allow fire to run in diagnostic modes that otherwise disable it.")
    p.add_argument("--disable-fire", dest="enable_fire", action="store_false",
                   help="Disable fire, overriding diagnostic preset defaults.")
    p.add_argument("--model-dir", default=None,
                   help="Warm-start actors from a checkpoint dir (e.g. a behaviour-cloned results/bc_happo) and RL-fine-tune.")
    p.add_argument("--warmstart-uav-model-dir", default=None,
                   help="Warm-start the class-shared UAV actor from actor_agent0.pt in this models/ directory.")
    p.add_argument("--warmstart-ugv-model-dir", default=None,
                   help="Warm-start the class-shared UGV actor from actor_agent0.pt in this models/ directory.")
    p.add_argument("--recurrent", action="store_true",
                   help="Use a recurrent (GRU) policy so agents remember where they have searched.")
    p.add_argument("--reward-search", action="store_true",
                   help="Use a search-dominant reward (survivor find/scout >> movement/hazard cost) "
                        "to avoid the do-nothing degenerate policy.")
    p.add_argument("--reward-confirm", action="store_true",
                   help="Use a confirmation-dominant reward: ground confirmation is the dominant "
                        "term, movement/fire costs are cut, and idling while survivors are scouted-"
                        "but-unconfirmed is penalized. Fixes the 'ground robots don't move' optimum.")
    p.add_argument("--drone-camera-fov-deg", type=float, default=None,
                   help="Drone camera field of view (deg). Wider FOV => larger scout footprint. "
                        "Lets RL get detection signal at floor 0 on small terrains (e.g. 140).")
    p.add_argument("--drone-flight-levels-m", default=None,
                   help="Comma-separated flight altitudes in meters (>=2 values), e.g. '50,80,100'. "
                        "Higher altitude => larger scout footprint.")
    p.add_argument("--ground-confirmation-range-m", type=float, default=None,
                   help="Ground confirmation range in meters (physical, not a floor), e.g. 30.")
    p.add_argument("--coverage-obs-grid", type=int, default=None,
                   help="Add a KxK team-coverage map + global fraction to the observation so the "
                        "policy can learn systematic sweeping (e.g. 6). 0 = off; omitted uses the "
                        "UAV diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--confirm-requires-los", action="store_true",
                   help="Require unobstructed terrain line-of-sight (not just range) for confirmation.")
    p.add_argument("--drone-can-confirm", action="store_true",
                   help="Let drones (EO/IR) confirm survivors from altitude with top-down line-of-sight "
                        "(realistic aerial SAR; the honest route to >=0.9 recall).")
    p.add_argument("--r-drone-confirm", type=float, default=0.0,
                   help="Per-drone reward for a confirmation it makes (training signal for --drone-can-confirm).")
    p.add_argument("--local-coverage-obs-grid", type=int, default=None,
                   help="Add a pooled KxK ego-centric coverage map around each agent. "
                        "Use an odd value such as 9. 0 = off; omitted uses the UAV diagnostic "
                        "default in --uav-survivor-diagnostic mode.")
    p.add_argument("--local-coverage-obs-radius-m", type=float, default=150.0,
                   help="Physical half-width/radius in meters for --local-coverage-obs-grid. "
                        "Example: 150 with K=9 gives bins about 33m wide on a 500m map.")
    p.add_argument("--uav-confidence-obs-grid", type=int, default=None,
                   help="Add a KxK UAV inspection-confidence map + global mean to the observation. "
                        "0 = off; omitted uses the UAV diagnostic default in "
                        "--uav-survivor-diagnostic mode.")
    p.add_argument("--uav-local-confidence-obs-grid", type=int, default=None,
                   help="Add a pooled KxK ego-centric UAV inspection-confidence map. "
                        "Use an odd value such as 9. 0 = off; omitted uses the UAV "
                        "diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--uav-local-confidence-obs-radius-m", type=float, default=None,
                   help="Physical half-width/radius in meters for --uav-local-confidence-obs-grid. "
                        "Omitted uses the UAV diagnostic default when the local confidence map "
                        "is enabled by a diagnostic mode.")
    p.set_defaults(uav_frontier_obs=None)
    p.add_argument("--uav-frontier-obs", dest="uav_frontier_obs", action="store_true",
                   help="Add UAV frontier features: direction, distance, and local uncovered ratio "
                        "toward nearby unsearched team-coverage cells.")
    p.add_argument("--uav-no-frontier-obs", dest="uav_frontier_obs", action="store_false",
                   help="Disable UAV frontier observation, overriding UAV diagnostic defaults.")
    p.add_argument("--uav-frontier-obs-radius-m", type=float, default=None,
                   help="Physical radius in meters used for --uav-frontier-obs and frontier-alignment reward.")
    p.add_argument("--uav-frontier-mode", choices=("centroid", "sector_topk", "sector-topk", "local_global", "local-global"),
                   default=None,
                   help="Frontier feature/reward mode. centroid is the legacy averaged direction; "
                        "sector_topk uses top-k uncovered sector candidates; local_global exposes "
                        "one local candidate plus one team-diversified global candidate. Omitted uses "
                        "the UAV diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--uav-frontier-source", choices=("coverage", "confidence"),
                   default=None,
                   help="Map used to score UAV frontier candidates. coverage uses binary uncovered cells; "
                        "confidence uses the probabilistic inspection-confidence map. Omitted uses the "
                        "UAV diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--uav-frontier-sectors", type=int, default=DEFAULT_UAV_FRONTIER_SECTORS,
                   help="Number of angular sectors for --uav-frontier-mode sector_topk.")
    p.add_argument("--uav-frontier-top-k", type=int, default=DEFAULT_UAV_FRONTIER_TOP_K,
                   help="How many sector candidates to expose in --uav-frontier-mode sector_topk.")
    p.set_defaults(uav_frontier_ownership=DEFAULT_UAV_FRONTIER_OWNERSHIP)
    p.add_argument("--uav-frontier-ownership", dest="uav_frontier_ownership", action="store_true",
                   help="Ownership-weight sector scores by which UAV is closest to uncovered cells.")
    p.add_argument("--uav-frontier-no-ownership", dest="uav_frontier_ownership", action="store_false",
                   help="Disable ownership weighting for sector-top-k frontier scoring.")
    p.set_defaults(uav_cleanup_target_obs=None)
    p.add_argument("--uav-cleanup-target-obs", dest="uav_cleanup_target_obs", action="store_true",
                   help="Append a persistent low-confidence cleanup target observation to each UAV.")
    p.add_argument("--uav-no-cleanup-target-obs", dest="uav_cleanup_target_obs", action="store_false",
                   help="Disable persistent cleanup target observation, overriding UAV diagnostic defaults.")
    p.add_argument("--uav-cleanup-target-grid", type=int, default=16,
                   help="Coarse grid size used for persistent UAV cleanup targets.")
    p.add_argument("--uav-cleanup-target-hold-steps", type=int, default=15,
                   help="Number of steps to persist a valid UAV cleanup target before refresh.")
    p.add_argument("--uav-cleanup-target-confidence-threshold", type=float, default=0.80,
                   help="Pooled confidence below which a cleanup cell is targetable.")
    p.add_argument("--uav-cleanup-target-min-value", type=float, default=0.05,
                   help="Minimum pooled cleanup mass for targetable cleanup cells.")
    p.add_argument("--uav-cleanup-target-assignment-distance-scale-m", type=float, default=250.0,
                   help="Distance scale for greedy cleanup target assignment in meters.")
    p.add_argument("--uav-cleanup-target-refresh-mode", default=None,
                   choices=("exact", "fixed-hold", "fixed_hold"),
                   help="How held cleanup targets are refreshed. 'exact' recomputes the assigned "
                        "component each step; 'fixed-hold' keeps centroid/value fixed until expiry, "
                        "reach, or reassignment. Omitted uses the UAV diagnostic default in "
                        "--uav-survivor-diagnostic mode.")
    p.add_argument("--uav-astar-route-obs", action="store_true",
                   help="Append a 4-scalar A* route observation for UAVs: waypoint direction, "
                        "waypoint distance, and normalized path cost to the cleanup target. "
                        "This adds no reward by itself.")
    p.add_argument("--uav-astar-grid", type=int, default=32,
                   help="Coarse global grid size used by --uav-astar-route-obs.")
    p.add_argument("--uav-astar-confidence-cost-alpha", type=float, default=3.0,
                   help="A* cell cost scale in 1 + alpha * confidence^gamma.")
    p.add_argument("--uav-astar-confidence-cost-gamma", type=float, default=2.0,
                   help="A* cell cost exponent in 1 + alpha * confidence^gamma.")
    p.add_argument("--uav-astar-waypoint-lookahead-m", type=float, default=50.0,
                   help="Physical lookahead distance for selecting the exposed A* waypoint.")
    p.add_argument("--uav-astar-route-replan-steps", type=int, default=5,
                   help="Hold an A* waypoint for this many steps unless reached or target changes.")
    p.add_argument("--uav-astar-waypoint-reached-m", type=float, default=20.0,
                   help="Distance threshold in meters for replacing the current A* waypoint.")
    p.add_argument("--preset", choices=("smoke", "tuned", "floor0-1km"), default="smoke",
                   help="Preset for defaults. 'floor0-1km' (recommended) trains on the 1km terrain "
                        "with wide-FOV/high-altitude sensors so detection works at floor 0.")
    p.add_argument("--ugv-known-survivor-diagnostic", action="store_true",
                   help="Train a minimal diagnostic task: 0 drones, 1 UGV, 1 survivor known at reset, no fire.")
    p.add_argument("--uav-survivor-diagnostic", action="store_true",
                   help="Train a UAV-only diagnostic task: UAVs only, 0 UGVs, no fire; drone scouting counts as success.")
    p.add_argument("--joint-schema-uav-diagnostic", action="store_true",
                   help="Train UAV-only search with final joint-schema observations, padding absent UGV slots.")
    p.add_argument("--joint-survivor-diagnostic", action="store_true",
                   help="Train a joint task: UAVs scout unknown survivors, UGVs confirm known targets.")
    p.add_argument("--joint-schema-ugv-diagnostic", action="store_true",
                   help="Train 2 UGVs with delayed survivor knowledge and final joint-schema observations.")
    p.add_argument("--uav-diagnostic-drones", type=int, default=DEFAULT_UAV_DIAG_DRONES,
                   help="Number of UAVs in --uav-survivor-diagnostic mode.")
    p.add_argument("--joint-diagnostic-ugvs", type=int, default=DEFAULT_JOINT_DIAG_UGVS,
                   help="Number of UGVs in --joint-survivor-diagnostic mode.")
    p.add_argument("--n-drones", "--n-uavs", dest="n_drones", type=int, default=None,
                   help="Override the number of UAVs/drones for training scenarios. "
                        "Joint-schema modes use the same value for the UAV observation schema.")
    p.add_argument("--n-ugvs", "--n-ground", dest="n_ugvs", type=int, default=None,
                   help="Override the number of UGVs/ground agents for training scenarios. "
                        "Joint-schema modes use the same value for the UGV observation schema.")
    p.add_argument("--n-survivors", type=int, default=None,
                   help="Override the number of survivors for training/diagnostic scenarios. "
                        "Joint-schema modes use the same value for the survivor observation schema.")
    p.add_argument("--delayed-survivor-knowledge", action="store_true",
                   help="Reveal survivors over time as oracle scout events for curriculum scenarios.")
    p.add_argument("--survivor-reveal-schedule", choices=("stratified_uniform", "stratified-uniform"),
                   default="stratified_uniform",
                   help="Sampling scheme for delayed survivor knowledge reveal times.")
    p.add_argument("--survivor-reveal-initial-count", type=int, default=1,
                   help="Number of survivors revealed at reset under delayed survivor knowledge.")
    p.add_argument("--survivor-reveal-start-step", type=int, default=10,
                   help="Earliest delayed reveal step after the initial survivors.")
    p.add_argument("--survivor-reveal-end-step", type=int, default=180,
                   help="Latest delayed reveal step after the initial survivors.")
    p.add_argument("--ugv-diagnostic-target-distance-min-m", type=float, default=None,
                   help="Minimum known-survivor start distance sampled at reset for the UGV diagnostic task.")
    p.add_argument("--ugv-diagnostic-target-distance-max-m", type=float, default=None,
                   help="Maximum known-survivor start distance sampled at reset for the UGV diagnostic task. "
                        "Omit for no upper bound; use min=max for an exact target distance.")
    p.add_argument("--uav-coverage-only", action="store_true",
                   help="In UAV diagnostic mode, disable survivor and time rewards; "
                        "leave only coverage rewards and coverage-quality penalties.")
    p.add_argument("--uav-no-global-coverage-obs", action="store_true",
                   help="In UAV diagnostic mode, keep local coverage observation but disable the default global coverage map.")
    p.add_argument("--uav-found-survivor-reward", type=float, default=None,
                   help="Override r_found_survivor in UAV diagnostic mode.")
    p.add_argument("--uav-all-survivors-reward", type=float, default=None,
                   help="One-time team reward when the final survivor is found in UAV diagnostic mode. "
                        "Omit for 0; pass a positive value to add an explicit mission-completion bonus.")
    p.add_argument("--team-scout-reward", "--uav-team-scout-reward", dest="team_scout_reward",
                   type=float, default=None,
                   help="Override r_team_scout, the team reward paid when a survivor is newly scouted.")
    p.add_argument("--uav-time-penalty", type=float, default=None,
                   help="Override r_time_penalty in UAV diagnostic mode.")
    p.add_argument("--uav-coverage-reward", type=float, default=None,
                   help="Total reward scale for team-new UAV camera footprint coverage. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-coverage-normalization", choices=("map", "opportunity"),
                   default="map",
                   help="Normalization for --uav-coverage-reward. map uses new_cells / total_map_cells; "
                        "opportunity uses new_cells / one-step reachable uncovered cells.")
    p.add_argument("--uav-move-coverage-reward", type=float, default=None,
                   help="Reward scale for UAV actual displacement in meters multiplied by newly covered cells. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-move-coverage-normalization", choices=("raw", "opportunity"),
                   default="raw",
                   help="Normalization for --uav-move-coverage-reward. raw uses distance_m * new_cells; "
                        "opportunity uses (distance_m / max_step_m) * "
                        "(new_cells / one-step reachable uncovered cells).")
    p.add_argument("--uav-move-coverage-cap", type=float, default=0.1,
                   help="Per-drone, per-step cap for the UAV movement-coverage reward.")
    p.add_argument("--uav-coverage-threshold-reward", type=float, default=None,
                   help="One-time team reward when UAV coverage crosses the threshold fraction. "
                        "Omit for 0 in UAV diagnostic mode.")
    p.add_argument("--uav-coverage-threshold-fraction", type=float, default=0.95,
                   help="Coverage fraction that triggers --uav-coverage-threshold-reward. Default 0.95.")
    p.add_argument("--uav-coverage-opportunity-reward", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--uav-coverage-opportunity-cap", type=float, default=1.0,
                   help="Cap applied to the opportunity capture fraction when "
                        "--uav-coverage-normalization opportunity is used.")
    p.add_argument("--uav-frontier-alignment-reward", type=float, default=None,
                   help="Reward scale for clamped progress toward the local uncovered frontier. "
                        "Use with --uav-frontier-obs for the cleanest learning signal.")
    p.add_argument("--uav-confidence-reward", type=float, default=None,
                   help="Optional per-UAV reward scale for marginal probabilistic inspection-confidence gain. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-team-confidence-reward", "--team-confidence-reward",
                   dest="uav_team_confidence_reward", type=float, default=0.0,
                   help="Scale for adding the mean per-UAV confidence reward to every UAV reward.")
    p.add_argument("--uav-team-confidence-overlap-penalty", "--team-confidence-overlap-penalty",
                   dest="uav_team_confidence_overlap_penalty", type=float, default=0.0,
                   help="Scale for adding the mean confidence-overlap penalty term to every UAV reward. "
                        "Uses --uav-confidence-overlap-mode and threshold, but is independent of "
                        "--uav-confidence-overlap-penalty.")
    p.add_argument("--uav-confidence-move-reward", type=float, default=None,
                   help="Optional per-UAV reward scale for movement that captures a high fraction of the "
                        "best one-step confidence-gain opportunity. Omit in UAV diagnostic mode for its "
                        "default; pass 0 to disable.")
    p.add_argument("--uav-inefficient-move-penalty", type=float, default=None,
                   help="Optional per-UAV penalty scale for movement that captures little search opportunity. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-inefficient-move-source", choices=("coverage", "confidence"),
                   default=None,
                   help="Opportunity signal used by --uav-inefficient-move-penalty. "
                        "confidence uses actual/best confidence gain; coverage uses actual/reachable new cells. "
                        "Omitted uses the UAV diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--uav-confidence-overlap-penalty", type=float, default=None,
                   help="Optional per-UAV penalty scale for flying over medium/high confidence cells. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-confidence-overlap-mode",
                   choices=("raw", "opportunity_regret", "opportunity-regret"),
                   default=None,
                   help="raw penalizes high-confidence footprint fraction directly; "
                        "opportunity-regret scales it by missed one-step confidence-gain opportunity. "
                        "Omitted uses the UAV diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--uav-confidence-overlap-allowed-regret", type=float, default=0.10,
                   help="Regret slack for --uav-confidence-overlap-mode opportunity-regret. "
                        "0.10 means no penalty for being within 10%% of the best one-step candidate.")
    p.add_argument("--uav-cleanup-target-progress-reward", type=float, default=None,
                   help="Optional per-UAV reward scale for positive progress toward the persistent "
                        "cleanup target, gated off when local frontier opportunity is strong. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-astar-progress-reward", type=float, default=0.0,
                   help="Optional per-UAV reward scale for reducing A* route cost toward the "
                        "cleanup target, gated off when local frontier opportunity is strong. "
                        "Default 0 disables this reward.")
    p.add_argument("--uav-confidence-overlap-threshold", type=float, default=None,
                   help="Confidence value where --uav-confidence-overlap-penalty starts. "
                        "Cells below this threshold are free; cells at 1.0 receive full penalty. "
                        "Omitted uses the UAV diagnostic default in --uav-survivor-diagnostic mode.")
    p.add_argument("--uav-confidence-gamma", type=float, default=2.0,
                   help="Exponent for weighting confidence reward toward low-confidence cells. "
                        "Higher values focus more strongly on poorly inspected cells.")
    p.add_argument("--uav-confidence-eps", type=float, default=0.05,
                   help="Minimum confidence reward weight for already high-confidence cells.")
    p.add_argument("--uav-confidence-opportunity-eps", type=float, default=1e-6,
                   help="Small denominator floor for --uav-confidence-move-reward opportunity normalization.")
    p.add_argument("--uav-overlap-penalty", type=float, default=None,
                   help="Maximum per-UAV per-step penalty at maximum excess footprint overlap. "
                        "The expected overlap from actual movement is not penalized. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-overlap-allowed", type=float, default=None,
                   help="Excess footprint-overlap slack above the physics-expected overlap "
                        "before the UAV overlap penalty starts.")
    p.add_argument("--uav-overlap-penalty-normalization", choices=("raw", "opportunity"),
                   default="raw",
                   help="Normalization for --uav-overlap-penalty. raw preserves the current excess-overlap "
                        "penalty; opportunity multiplies it by reachable_uncovered / reachable_total.")
    p.add_argument("--uav-inter-uav-overlap-penalty", type=float, default=0.0,
                   help="Maximum per-UAV per-step penalty for same-step footprint overlap with "
                        "other UAVs. Default 0 disables this optional coordination penalty.")
    p.add_argument("--uav-inter-uav-overlap-allowed", type=float, default=0.20,
                   help="Same-step inter-UAV footprint-overlap slack before "
                        "--uav-inter-uav-overlap-penalty starts.")
    p.add_argument("--uav-outside-footprint-penalty", type=float, default=None,
                   help="Maximum per-UAV per-step penalty when the camera footprint is fully outside the map. "
                        "Penalty scales linearly with the estimated outside-footprint fraction. "
                        "Omit in UAV diagnostic mode for its default; pass 0 to disable.")
    p.add_argument("--uav-boundary-soft-margin-m", type=float, default=25.0,
                   help="Physical margin from map edge used for UAV boundary risk diagnostics.")
    p.add_argument("--uav-start-min-separation-m", type=float, default=None,
                   help="Minimum pairwise UAV start distance in meters. In UAV diagnostic mode, "
                        "default is 150m; pass 0 to disable.")
    p.add_argument("--uav-start-edge-margin-m", type=float, default=None,
                   help="Minimum UAV start-center distance from each map edge in meters. In UAV "
                        "diagnostic mode, default is 50m; pass 0 to disable.")
    p.add_argument("--ugv-ground-shaping-reward", type=float, default=None,
                   help="Override r_ground_shaping for UGV route/target progress. "
                        "Omit to use the diagnostic mode default.")
    p.add_argument("--ugv-movement-alignment-reward", type=float, default=0.20,
                   help="Reward scale for UGV actual-movement alignment toward a known survivor in the "
                        "diagnostic task.")
    p.add_argument("--ugv-pending-penalty", type=float, default=None,
                   help="Override per-UGV, per-step penalty per known unconfirmed survivor. "
                        "Use a negative value, e.g. -0.02. Omit to use the diagnostic mode default.")
    p.add_argument("--ugv-planner-progress-reward", type=float, default=0.0,
                   help="Reward scale for actual UGV progress toward the local A* waypoint when "
                        "the planner detects a detour. Requires a local UGV planner hint.")
    p.add_argument("--ugv-approach-reward", type=float, default=DEFAULT_UGV_APPROACH_REWARD,
                   help="Inner UGV approach milestone reward. "
                        "Default fractions make this 0.05 produce 75/50/40/30/20m "
                        "one-time rewards of 0.02/0.025/0.03/0.04/0.05.")
    p.add_argument("--ugv-approach-milestone-radii-m", type=float, nargs="+",
                   default=list(DEFAULT_UGV_APPROACH_MILESTONE_RADII_M),
                   help="One-time UGV approach milestone radii in meters.")
    p.add_argument("--ugv-approach-radius-m", type=float, default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--ugv-stall-penalty", type=float, default=0.0,
                   help="Penalty magnitude subtracted when a UGV barely moves while seeking a known target.")
    p.add_argument("--ugv-stall-displacement-threshold-m", type=float, default=0.05,
                   help="Actual per-step movement below this distance is treated as stalled.")
    p.add_argument("--ugv-route-progress-floor-penalty", type=float, default=0.0,
                   help="Penalty coefficient per meter of missing planner-route progress. "
                        "Applies only while a UGV has an active assigned route and is outside confirmation range.")
    p.add_argument("--ugv-route-progress-floor-m", type=float, default=0.0,
                   help="Minimum expected planner-route progress per step before "
                        "--ugv-route-progress-floor-penalty starts.")
    p.add_argument("--ugv-route-progress-shortfall-penalty", type=float, default=None,
                   help="Penalty coefficient per meter that planner-route progress falls short of "
                        "remaining_route_distance / remaining_episode_steps. Omit to use the "
                        "diagnostic mode default; pass 0 to disable.")
    p.add_argument("--slope-speed-weight", type=float, default=None,
                   help="Override slope penalty in UGV speed multiplier. "
                        "Default scenario value is 0.5; larger values make slopes slower.")
    p.add_argument("--land-cover-speeds", type=float, nargs="+", default=None,
                   help="Override UGV speed multipliers for road/open/brush/forest/rock[/water]. "
                        "Example: --land-cover-speeds 1.0 0.95 0.8 0.7 0.0 0.0")
    p.add_argument("--action-transform", choices=("clip", "tanh", "radial_tanh", "radial-tanh"), default="clip",
                   help="How to bound raw continuous HAPPO actions before VMAS. "
                        "'clip' is the HARL-compatible default; 'tanh' is an experimental "
                        "plain tanh post-transform; 'radial_tanh' preserves raw action "
                        "direction while smoothly bounding vector magnitude.")
    args = p.parse_args()
    args.action_transform = args.action_transform.replace("-", "_")
    args.ugv_planner_hint = args.ugv_planner_hint.replace("-", "_")
    if args.ugv_target_assignment_mode is not None:
        args.ugv_target_assignment_mode = args.ugv_target_assignment_mode.replace("-", "_")
    args.survivor_reveal_schedule = args.survivor_reveal_schedule.replace("-", "_")

    if (
        args.ugv_known_survivor_diagnostic
        or args.joint_survivor_diagnostic
        or args.joint_schema_ugv_diagnostic
    ):
        defaulted_ugv_planner_hint = args.ugv_planner_hint == "none"
        if args.terrain_cache_path is None:
            args.terrain_cache_path = str(DEFAULT_UGV_DIAG_TERRAIN_CACHE_PATH)
        if args.local_map_patch_size == 3:
            args.local_map_patch_size = DEFAULT_UGV_DIAG_LOCAL_MAP_PATCH_SIZE
        if args.ugv_known_survivor_diagnostic and args.ugv_diagnostic_target_distance_min_m is None:
            args.ugv_diagnostic_target_distance_min_m = DEFAULT_UGV_DIAG_TARGET_DISTANCE_MIN_M
        if args.lr == 5e-4:
            args.lr = DEFAULT_UGV_DIAG_LR
        if args.critic_lr == 5e-4:
            args.critic_lr = DEFAULT_UGV_DIAG_CRITIC_LR
        if args.linear_lr_decay is None:
            args.linear_lr_decay = True
        if defaulted_ugv_planner_hint:
            args.ugv_planner_hint = DEFAULT_UGV_DIAG_PLANNER_HINT
        if defaulted_ugv_planner_hint and args.ugv_dense_reward_mode.replace("-", "_") == "target":
            args.ugv_dense_reward_mode = DEFAULT_UGV_DIAG_DENSE_REWARD_MODE
        if args.ugv_global_planner_heuristic == "euclidean":
            args.ugv_global_planner_heuristic = DEFAULT_UGV_DIAG_GLOBAL_PLANNER_HEURISTIC
        if args.ugv_global_planner_lookahead_m == 20.0:
            args.ugv_global_planner_lookahead_m = DEFAULT_UGV_DIAG_GLOBAL_PLANNER_LOOKAHEAD_M
        if args.ugv_planner_progress_reward == 0.0:
            args.ugv_planner_progress_reward = DEFAULT_UGV_DIAG_PLANNER_PROGRESS_REWARD
        if args.action_transform == "clip":
            args.action_transform = DEFAULT_UGV_DIAG_ACTION_TRANSFORM
        if args.enable_fire is None:
            args.enable_fire = bool(args.ugv_known_survivor_diagnostic)
        if (args.joint_survivor_diagnostic or args.joint_schema_ugv_diagnostic) and args.enable_fire is False:
            args.ugv_planner_fire_mode = "off"
        if args.ugv_planner_fire_mode == "off":
            if args.enable_fire:
                args.ugv_planner_fire_mode = DEFAULT_UGV_DIAG_PLANNER_FIRE_MODE
        if args.ugv_planner_fire_replan_policy == "always":
            args.ugv_planner_fire_replan_policy = DEFAULT_UGV_DIAG_PLANNER_FIRE_REPLAN_POLICY
        if args.ugv_planner_fire_replan_interval_steps == 15:
            args.ugv_planner_fire_replan_interval_steps = (
                DEFAULT_UGV_DIAG_PLANNER_FIRE_REPLAN_INTERVAL_STEPS
            )
        if args.ugv_planner_fire_cost == 25.0:
            args.ugv_planner_fire_cost = DEFAULT_UGV_DIAG_PLANNER_FIRE_COST
        if args.ugv_planner_fire_block_threshold == 0.0:
            args.ugv_planner_fire_block_threshold = DEFAULT_UGV_DIAG_PLANNER_FIRE_BLOCK_THRESHOLD
        if args.ugv_planner_smoke_cost == 5.0:
            args.ugv_planner_smoke_cost = DEFAULT_UGV_DIAG_PLANNER_SMOKE_COST
        if args.ugv_planner_smolder_cost == 3.0:
            args.ugv_planner_smolder_cost = DEFAULT_UGV_DIAG_PLANNER_SMOLDER_COST
        if args.ugv_planner_fire_buffer_m == 10.0:
            args.ugv_planner_fire_buffer_m = DEFAULT_UGV_DIAG_PLANNER_FIRE_BUFFER_M
        if args.ugv_planner_fire_buffer_cost == 8.0:
            args.ugv_planner_fire_buffer_cost = DEFAULT_UGV_DIAG_PLANNER_FIRE_BUFFER_COST
        if args.ugv_planner_land_cover_costs is None:
            args.ugv_planner_land_cover_costs = list(DEFAULT_UGV_DIAG_PLANNER_LAND_COVER_COSTS)

    if args.linear_lr_decay is None:
        args.linear_lr_decay = False
    if args.enable_fire is None:
        args.enable_fire = False

    if args.land_cover_speeds is not None and len(args.land_cover_speeds) not in (5, 6):
        p.error("--land-cover-speeds must provide 5 or 6 values: road open brush forest rock [water]")
    if (
        args.ugv_planner_land_cover_costs is not None
        and len(args.ugv_planner_land_cover_costs) not in (5, 6)
    ):
        p.error("--ugv-planner-land-cover-costs must provide 5 or 6 values: road open brush forest rock [water]")
    if (
        args.ugv_planner_land_cover_costs is not None
        and any(value < 0.0 for value in args.ugv_planner_land_cover_costs)
    ):
        p.error("--ugv-planner-land-cover-costs values must be nonnegative")
    if args.local_map_patch_size < 1 or args.local_map_patch_size % 2 != 1:
        p.error("--local-map-patch-size must be a positive odd integer")
    if args.ugv_planner_patch_size < 1 or args.ugv_planner_patch_size % 2 != 1:
        p.error("--ugv-planner-patch-size must be a positive odd integer")
    if args.ugv_planner_lookahead_cells < 1:
        p.error("--ugv-planner-lookahead-cells must be positive")
    args.ugv_planner_lookahead_cells = min(
        args.ugv_planner_lookahead_cells,
        max(args.ugv_planner_patch_size // 2, 1),
    )
    if not 0.0 <= args.comms_dropout <= 1.0:
        p.error("--comms-dropout must be in [0, 1]")
    if args.entropy_coef < 0.0:
        p.error("--entropy-coef must be nonnegative")
    if args.lr <= 0.0:
        p.error("--lr must be positive")
    if args.critic_lr <= 0.0:
        p.error("--critic-lr must be positive")
    if args.terrain_cnn_embed_dim <= 0:
        p.error("--terrain-cnn-embed-dim must be positive")
    if args.joint_diagnostic_ugvs < 1:
        p.error("--joint-diagnostic-ugvs must be positive")
    if args.coverage_obs_grid is not None and args.coverage_obs_grid < 0:
        p.error("--coverage-obs-grid must be nonnegative")
    if args.local_coverage_obs_grid is not None and (
        args.local_coverage_obs_grid < 0
        or (args.local_coverage_obs_grid > 0 and args.local_coverage_obs_grid % 2 != 1)
    ):
        p.error("--local-coverage-obs-grid must be 0 or a positive odd integer")
    if args.local_coverage_obs_radius_m <= 0.0:
        p.error("--local-coverage-obs-radius-m must be positive")
    if args.uav_confidence_obs_grid is not None and args.uav_confidence_obs_grid < 0:
        p.error("--uav-confidence-obs-grid must be nonnegative")
    if args.uav_local_confidence_obs_grid is not None and (
        args.uav_local_confidence_obs_grid < 0 or (
        args.uav_local_confidence_obs_grid > 0
        and args.uav_local_confidence_obs_grid % 2 != 1
        )
    ):
        p.error("--uav-local-confidence-obs-grid must be 0 or a positive odd integer")
    if args.uav_local_confidence_obs_radius_m is not None and args.uav_local_confidence_obs_radius_m <= 0.0:
        p.error("--uav-local-confidence-obs-radius-m must be positive")
    if args.uav_frontier_obs_radius_m is not None and args.uav_frontier_obs_radius_m <= 0.0:
        p.error("--uav-frontier-obs-radius-m must be positive")
    if args.uav_frontier_mode is not None:
        args.uav_frontier_mode = str(args.uav_frontier_mode).replace("-", "_")
        if args.uav_frontier_mode not in {"centroid", "sector_topk", "local_global"}:
            p.error("--uav-frontier-mode must be one of: centroid, sector_topk, local_global")
    if args.uav_frontier_source is not None:
        args.uav_frontier_source = str(args.uav_frontier_source).replace("-", "_").lower()
        if args.uav_frontier_source not in {"coverage", "confidence"}:
            p.error("--uav-frontier-source must be one of: coverage, confidence")
    if args.uav_frontier_sectors < 2:
        p.error("--uav-frontier-sectors must be at least 2")
    if args.uav_frontier_top_k < 1 or args.uav_frontier_top_k > args.uav_frontier_sectors:
        p.error("--uav-frontier-top-k must be in [1, --uav-frontier-sectors]")
    if args.uav_cleanup_target_grid < 2:
        p.error("--uav-cleanup-target-grid must be at least 2")
    if args.uav_cleanup_target_hold_steps < 1:
        p.error("--uav-cleanup-target-hold-steps must be positive")
    if not 0.0 <= args.uav_cleanup_target_confidence_threshold <= 1.0:
        p.error("--uav-cleanup-target-confidence-threshold must be in [0, 1]")
    if args.uav_cleanup_target_min_value < 0.0:
        p.error("--uav-cleanup-target-min-value must be nonnegative")
    if args.uav_cleanup_target_assignment_distance_scale_m <= 0.0:
        p.error("--uav-cleanup-target-assignment-distance-scale-m must be positive")
    if args.uav_cleanup_target_refresh_mode is not None:
        args.uav_cleanup_target_refresh_mode = str(args.uav_cleanup_target_refresh_mode).replace("-", "_").lower()
        if args.uav_cleanup_target_refresh_mode not in {"exact", "fixed_hold"}:
            p.error("--uav-cleanup-target-refresh-mode must be one of: exact, fixed-hold")
    if args.uav_astar_grid < 2:
        p.error("--uav-astar-grid must be at least 2")
    if args.uav_astar_confidence_cost_alpha < 0.0:
        p.error("--uav-astar-confidence-cost-alpha must be nonnegative")
    if args.uav_astar_confidence_cost_gamma < 0.0:
        p.error("--uav-astar-confidence-cost-gamma must be nonnegative")
    if args.uav_astar_waypoint_lookahead_m <= 0.0:
        p.error("--uav-astar-waypoint-lookahead-m must be positive")
    if args.uav_astar_route_replan_steps < 1:
        p.error("--uav-astar-route-replan-steps must be positive")
    if args.uav_astar_waypoint_reached_m <= 0.0:
        p.error("--uav-astar-waypoint-reached-m must be positive")
    if args.ugv_ground_shaping_reward is not None and args.ugv_ground_shaping_reward < 0.0:
        p.error("--ugv-ground-shaping-reward must be nonnegative")
    if args.ugv_movement_alignment_reward < 0.0:
        p.error("--ugv-movement-alignment-reward must be nonnegative")
    if args.ugv_pending_penalty is not None and args.ugv_pending_penalty > 0.0:
        p.error("--ugv-pending-penalty must be zero or negative")
    if args.uav_coverage_reward is not None and args.uav_coverage_reward < 0.0:
        p.error("--uav-coverage-reward must be nonnegative")
    if args.uav_found_survivor_reward is not None and args.uav_found_survivor_reward < 0.0:
        p.error("--uav-found-survivor-reward must be nonnegative")
    if args.uav_all_survivors_reward is not None and args.uav_all_survivors_reward < 0.0:
        p.error("--uav-all-survivors-reward must be nonnegative")
    if args.team_scout_reward is not None and args.team_scout_reward < 0.0:
        p.error("--team-scout-reward must be nonnegative")
    if args.uav_move_coverage_reward is not None and args.uav_move_coverage_reward < 0.0:
        p.error("--uav-move-coverage-reward must be nonnegative")
    if args.uav_move_coverage_cap < 0.0:
        p.error("--uav-move-coverage-cap must be nonnegative")
    if (
        args.uav_coverage_threshold_reward is not None
        and args.uav_coverage_threshold_reward < 0.0
    ):
        p.error("--uav-coverage-threshold-reward must be nonnegative")
    if not 0.0 <= args.uav_coverage_threshold_fraction <= 1.0:
        p.error("--uav-coverage-threshold-fraction must be in [0, 1]")
    if args.uav_coverage_opportunity_reward is not None and args.uav_coverage_opportunity_reward < 0.0:
        p.error("--uav-coverage-opportunity-reward must be nonnegative")
    if (
        args.uav_coverage_opportunity_reward is not None
        and args.uav_coverage_opportunity_reward > 0.0
        and args.uav_coverage_reward is not None
        and args.uav_coverage_reward > 0.0
    ):
        p.error(
            "Use --uav-coverage-reward with --uav-coverage-normalization opportunity, "
            "or the legacy --uav-coverage-opportunity-reward, not both"
        )
    if args.uav_coverage_opportunity_reward is not None and args.uav_coverage_opportunity_reward > 0.0:
        args.uav_coverage_reward = args.uav_coverage_opportunity_reward
        args.uav_coverage_normalization = "opportunity"
        args.uav_coverage_opportunity_reward = None
    if args.uav_coverage_opportunity_cap < 0.0:
        p.error("--uav-coverage-opportunity-cap must be nonnegative")
    if args.uav_frontier_alignment_reward is not None and args.uav_frontier_alignment_reward < 0.0:
        p.error("--uav-frontier-alignment-reward must be nonnegative")
    if args.uav_confidence_reward is not None and args.uav_confidence_reward < 0.0:
        p.error("--uav-confidence-reward must be nonnegative")
    if args.uav_team_confidence_reward < 0.0:
        p.error("--uav-team-confidence-reward must be nonnegative")
    if args.uav_team_confidence_overlap_penalty < 0.0:
        p.error("--uav-team-confidence-overlap-penalty must be nonnegative")
    if args.uav_confidence_move_reward is not None and args.uav_confidence_move_reward < 0.0:
        p.error("--uav-confidence-move-reward must be nonnegative")
    if args.uav_astar_progress_reward < 0.0:
        p.error("--uav-astar-progress-reward must be nonnegative")
    if args.uav_inefficient_move_penalty is not None and args.uav_inefficient_move_penalty < 0.0:
        p.error("--uav-inefficient-move-penalty must be nonnegative")
    if args.uav_confidence_overlap_penalty is not None and args.uav_confidence_overlap_penalty < 0.0:
        p.error("--uav-confidence-overlap-penalty must be nonnegative")
    args.uav_confidence_overlap_mode = str(args.uav_confidence_overlap_mode).replace("-", "_").lower()
    if args.uav_confidence_overlap_mode not in {"raw", "opportunity_regret"}:
        p.error("--uav-confidence-overlap-mode must be one of: raw, opportunity-regret")
    if not 0.0 <= args.uav_confidence_overlap_allowed_regret <= 1.0:
        p.error("--uav-confidence-overlap-allowed-regret must be in [0, 1]")
    if (
        args.uav_cleanup_target_progress_reward is not None
        and args.uav_cleanup_target_progress_reward < 0.0
    ):
        p.error("--uav-cleanup-target-progress-reward must be nonnegative")
    if (
        args.uav_confidence_overlap_threshold is not None
        and not 0.0 <= args.uav_confidence_overlap_threshold < 1.0
    ):
        p.error("--uav-confidence-overlap-threshold must be in [0, 1)")
    if args.uav_confidence_gamma < 0.0:
        p.error("--uav-confidence-gamma must be nonnegative")
    if args.uav_confidence_eps < 0.0:
        p.error("--uav-confidence-eps must be nonnegative")
    if args.uav_confidence_opportunity_eps < 0.0:
        p.error("--uav-confidence-opportunity-eps must be nonnegative")
    if args.uav_overlap_penalty is not None and args.uav_overlap_penalty < 0.0:
        p.error("--uav-overlap-penalty must be nonnegative")
    if args.uav_overlap_allowed is not None and not 0.0 <= args.uav_overlap_allowed < 1.0:
        p.error("--uav-overlap-allowed must be in [0, 1)")
    if args.uav_inter_uav_overlap_penalty < 0.0:
        p.error("--uav-inter-uav-overlap-penalty must be nonnegative")
    if not 0.0 <= args.uav_inter_uav_overlap_allowed < 1.0:
        p.error("--uav-inter-uav-overlap-allowed must be in [0, 1)")
    if (
        args.uav_outside_footprint_penalty is not None
        and args.uav_outside_footprint_penalty < 0.0
    ):
        p.error("--uav-outside-footprint-penalty must be nonnegative")
    if args.uav_boundary_soft_margin_m <= 0.0:
        p.error("--uav-boundary-soft-margin-m must be positive")
    if args.uav_start_min_separation_m is not None and args.uav_start_min_separation_m < 0.0:
        p.error("--uav-start-min-separation-m must be nonnegative")
    if args.uav_start_edge_margin_m is not None and args.uav_start_edge_margin_m < 0.0:
        p.error("--uav-start-edge-margin-m must be nonnegative")
    if args.uav_diagnostic_drones < 1:
        p.error("--uav-diagnostic-drones must be positive")
    if args.n_drones is not None and args.n_drones < 1:
        p.error("--n-drones must be positive")
    if args.n_ugvs is not None and args.n_ugvs < 1:
        p.error("--n-ugvs must be positive")
    if args.ugv_planner_progress_reward < 0.0:
        p.error("--ugv-planner-progress-reward must be nonnegative")
    ugv_local_planners = {"local_astar", "local_escape_astar"}
    ugv_planners = ugv_local_planners | {"global_astar"}
    if args.ugv_planner_progress_reward > 0.0 and args.ugv_planner_hint not in ugv_planners:
        p.error("--ugv-planner-progress-reward requires a UGV planner hint")
    if args.ugv_planner_detour_obs and args.ugv_planner_hint not in ugv_planners:
        p.error("--ugv-planner-detour-obs requires a UGV planner hint")
    if args.ugv_route_aware_reward and args.ugv_planner_hint not in ugv_local_planners:
        p.error("--ugv-route-aware-reward requires a local UGV planner hint")
    args.ugv_dense_reward_mode = args.ugv_dense_reward_mode.replace("-", "_")
    if args.ugv_dense_reward_mode == "planner_blend" and args.ugv_planner_hint not in ugv_local_planners:
        p.error("--ugv-dense-reward-mode planner_blend requires a local UGV planner hint")
    if args.ugv_dense_reward_mode == "escape_blend" and args.ugv_planner_hint != "local_escape_astar":
        p.error("--ugv-dense-reward-mode escape_blend requires --ugv-planner-hint local_escape_astar")
    if args.ugv_dense_reward_mode == "escape_route_switch" and args.ugv_planner_hint != "local_astar":
        p.error("--ugv-dense-reward-mode escape_route_switch requires --ugv-planner-hint local_astar")
    if args.ugv_dense_reward_mode == "planner_follow" and args.ugv_planner_hint != "global_astar":
        p.error("--ugv-dense-reward-mode planner_follow requires --ugv-planner-hint global_astar")
    if args.ugv_route_aware_reward and args.ugv_dense_reward_mode != "target":
        p.error("--ugv-route-aware-reward can only be combined with --ugv-dense-reward-mode target")
    args.ugv_planner_blend_weight = min(max(float(args.ugv_planner_blend_weight), 0.0), 1.0)
    if args.ugv_escape_stall_steps < 1:
        p.error("--ugv-escape-stall-steps must be positive")
    if args.ugv_escape_progress_threshold_m < 0.0:
        p.error("--ugv-escape-progress-threshold-m must be nonnegative")
    if args.ugv_escape_movement_threshold_m < 0.0:
        p.error("--ugv-escape-movement-threshold-m must be nonnegative")
    if args.ugv_escape_waypoint_reached_m <= 0.0:
        p.error("--ugv-escape-waypoint-reached-m must be positive")
    if args.ugv_escape_max_steps < 1:
        p.error("--ugv-escape-max-steps must be positive")
    if args.ugv_global_planner_lookahead_m <= 0.0:
        p.error("--ugv-global-planner-lookahead-m must be positive")
    args.ugv_global_planner_heuristic = args.ugv_global_planner_heuristic.replace("-", "_")
    if args.ugv_planner_fire_replan_interval_steps < 1:
        p.error("--ugv-planner-fire-replan-interval-steps must be positive")
    if not 0.0 <= args.ugv_planner_fire_block_threshold <= 1.0:
        p.error("--ugv-planner-fire-block-threshold must be in [0, 1]")
    for flag_name in (
        "ugv_planner_fire_cost",
        "ugv_planner_smoke_cost",
        "ugv_planner_smolder_cost",
        "ugv_planner_fire_buffer_m",
        "ugv_planner_fire_buffer_cost",
    ):
        if getattr(args, flag_name) < 0.0:
            p.error(f"--{flag_name.replace('_', '-')} must be nonnegative")
    if args.ugv_approach_reward < 0.0:
        p.error("--ugv-approach-reward must be nonnegative")
    if hasattr(args, "ugv_approach_radius_m"):
        args.ugv_approach_milestone_radii_m = [args.ugv_approach_radius_m]
    if not args.ugv_approach_milestone_radii_m or any(
        value <= 0.0 for value in args.ugv_approach_milestone_radii_m
    ):
        p.error("--ugv-approach-milestone-radii-m must contain positive distances")
    if args.ugv_stall_penalty < 0.0:
        p.error("--ugv-stall-penalty must be nonnegative")
    if args.ugv_stall_displacement_threshold_m < 0.0:
        p.error("--ugv-stall-displacement-threshold-m must be nonnegative")
    if args.ugv_route_progress_floor_penalty < 0.0:
        p.error("--ugv-route-progress-floor-penalty must be nonnegative")
    if args.ugv_route_progress_floor_m < 0.0:
        p.error("--ugv-route-progress-floor-m must be nonnegative")
    if (
        args.ugv_route_progress_shortfall_penalty is not None
        and args.ugv_route_progress_shortfall_penalty < 0.0
    ):
        p.error("--ugv-route-progress-shortfall-penalty must be nonnegative")
    if args.survivor_reveal_initial_count < 0:
        p.error("--survivor-reveal-initial-count must be nonnegative")
    if args.survivor_reveal_start_step < 0 or args.survivor_reveal_end_step < 0:
        p.error("--survivor-reveal-start-step and --survivor-reveal-end-step must be nonnegative")
    if args.survivor_reveal_end_step < args.survivor_reveal_start_step:
        p.error("--survivor-reveal-end-step must be >= --survivor-reveal-start-step")
    if args.ugv_sticky_switch_margin_m < 0.0:
        p.error("--ugv-sticky-switch-margin-m must be nonnegative")
    if args.ugv_sticky_switch_ratio < 0.0:
        p.error("--ugv-sticky-switch-ratio must be nonnegative")
    if args.ugv_sticky_min_age_steps < 0:
        p.error("--ugv-sticky-min-age-steps must be nonnegative")
    if args.n_survivors is not None and args.n_survivors < 1:
        p.error("--n-survivors must be positive")
    if args.fire_grid_size < 2:
        p.error("--fire-grid-size must be at least 2")
    if sum(
        (
            bool(args.ugv_known_survivor_diagnostic),
            bool(args.uav_survivor_diagnostic),
            bool(args.joint_schema_uav_diagnostic),
            bool(args.joint_survivor_diagnostic),
            bool(args.joint_schema_ugv_diagnostic),
        )
    ) > 1:
        p.error(
            "Choose only one diagnostic mode: --ugv-known-survivor-diagnostic, "
            "--uav-survivor-diagnostic, --joint-schema-uav-diagnostic, "
            "--joint-survivor-diagnostic, or --joint-schema-ugv-diagnostic"
        )
    if args.ugv_target_assignment_mode is None and (
        args.joint_survivor_diagnostic or args.joint_schema_ugv_diagnostic
    ):
        args.ugv_target_assignment_mode = DEFAULT_JOINT_DIAG_UGV_TARGET_ASSIGNMENT_MODE
    if args.ugv_target_assignment_mode is None:
        args.ugv_target_assignment_mode = "nearest"
    if args.ugv_route_progress_shortfall_penalty is None:
        args.ugv_route_progress_shortfall_penalty = (
            DEFAULT_JOINT_DIAG_ROUTE_PROGRESS_SHORTFALL_PENALTY
            if args.joint_survivor_diagnostic or args.joint_schema_ugv_diagnostic
            else 0.0
        )
    if args.ugv_assigned_target_obs_only is None:
        args.ugv_assigned_target_obs_only = False
    if args.survivor_assignment_obs is None:
        args.survivor_assignment_obs = bool(
            args.joint_survivor_diagnostic
            or args.joint_schema_uav_diagnostic
            or args.joint_schema_ugv_diagnostic
        )
    if args.terrain_cache_path is not None and not Path(args.terrain_cache_path).is_file():
        p.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")

    drone_flight_levels_m = None
    if args.drone_flight_levels_m:
        drone_flight_levels_m = tuple(
            float(v) for v in str(args.drone_flight_levels_m).split(",") if v.strip()
        )

    uav_search_diagnostic = (
        args.uav_survivor_diagnostic
        or args.joint_schema_uav_diagnostic
        or args.joint_survivor_diagnostic
        or args.joint_schema_ugv_diagnostic
    )
    if uav_search_diagnostic:
        if args.terrain_cache_path is None:
            args.terrain_cache_path = str(DEFAULT_UAV_DIAG_TERRAIN_CACHE_PATH)
        if args.local_map_patch_size == 3:
            args.local_map_patch_size = DEFAULT_UAV_DIAG_LOCAL_MAP_PATCH_SIZE
        if args.entropy_coef == 0.01:
            args.entropy_coef = DEFAULT_UAV_DIAG_ENTROPY_COEF
        if args.uav_no_global_coverage_obs:
            args.coverage_obs_grid = 0
        elif args.coverage_obs_grid is None:
            args.coverage_obs_grid = DEFAULT_UAV_DIAG_COVERAGE_OBS_GRID
        if args.local_coverage_obs_grid is None:
            args.local_coverage_obs_grid = DEFAULT_UAV_DIAG_LOCAL_COVERAGE_OBS_GRID
            args.local_coverage_obs_radius_m = DEFAULT_UAV_DIAG_LOCAL_COVERAGE_OBS_RADIUS_M
        if args.uav_confidence_obs_grid is None:
            args.uav_confidence_obs_grid = DEFAULT_UAV_DIAG_CONFIDENCE_OBS_GRID
        if args.uav_local_confidence_obs_grid is None:
            args.uav_local_confidence_obs_grid = DEFAULT_UAV_DIAG_LOCAL_CONFIDENCE_OBS_GRID
            args.uav_local_confidence_obs_radius_m = DEFAULT_UAV_DIAG_LOCAL_CONFIDENCE_OBS_RADIUS_M
        if args.uav_frontier_obs is None:
            args.uav_frontier_obs = True
        if args.uav_frontier_obs_radius_m is None:
            args.uav_frontier_obs_radius_m = DEFAULT_UAV_DIAG_FRONTIER_OBS_RADIUS_M
        if args.uav_frontier_mode is None:
            args.uav_frontier_mode = DEFAULT_UAV_DIAG_FRONTIER_MODE
        if args.uav_frontier_source is None:
            args.uav_frontier_source = DEFAULT_UAV_DIAG_FRONTIER_SOURCE
        if args.uav_frontier_alignment_reward is None:
            args.uav_frontier_alignment_reward = DEFAULT_UAV_DIAG_FRONTIER_ALIGNMENT_REWARD
        if args.uav_cleanup_target_obs is None:
            args.uav_cleanup_target_obs = False
        if args.uav_cleanup_target_refresh_mode is None:
            args.uav_cleanup_target_refresh_mode = DEFAULT_UAV_DIAG_CLEANUP_TARGET_REFRESH_MODE
        if args.uav_confidence_reward is None:
            args.uav_confidence_reward = DEFAULT_UAV_DIAG_CONFIDENCE_REWARD
        if args.uav_confidence_move_reward is None:
            args.uav_confidence_move_reward = DEFAULT_UAV_DIAG_CONFIDENCE_MOVE_REWARD
        if args.uav_inefficient_move_penalty is None:
            args.uav_inefficient_move_penalty = DEFAULT_UAV_DIAG_INEFFICIENT_MOVE_PENALTY
        if args.uav_inefficient_move_source is None:
            args.uav_inefficient_move_source = DEFAULT_UAV_DIAG_INEFFICIENT_MOVE_SOURCE
        if args.uav_confidence_overlap_penalty is None:
            args.uav_confidence_overlap_penalty = DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_PENALTY
        if args.uav_confidence_overlap_mode is None:
            args.uav_confidence_overlap_mode = DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_MODE
        if args.uav_confidence_overlap_threshold is None:
            args.uav_confidence_overlap_threshold = DEFAULT_UAV_DIAG_CONFIDENCE_OVERLAP_THRESHOLD
        if args.uav_cleanup_target_progress_reward is None:
            args.uav_cleanup_target_progress_reward = DEFAULT_UAV_DIAG_CLEANUP_TARGET_PROGRESS_REWARD
        if (
            args.share_param is None
            and (args.uav_survivor_diagnostic or args.joint_schema_uav_diagnostic)
            and not bool(args.share_param_by_agent_class)
        ):
            args.share_param = True
        if args.uav_start_min_separation_m is None:
            args.uav_start_min_separation_m = DEFAULT_UAV_DIAG_START_MIN_SEPARATION_M
        if args.uav_start_edge_margin_m is None:
            args.uav_start_edge_margin_m = DEFAULT_UAV_DIAG_START_EDGE_MARGIN_M
        if args.action_transform == "clip":
            args.action_transform = "radial_tanh"
        if args.joint_schema_uav_diagnostic and args.ugv_planner_hint == "none":
            args.ugv_planner_hint = DEFAULT_UGV_DIAG_PLANNER_HINT
        if args.n_rollout_threads == 1:
            args.n_rollout_threads = DEFAULT_UAV_DIAG_N_ROLLOUT_THREADS
    if args.joint_schema_ugv_diagnostic:
        args.uav_coverage_reward = 0.0
        args.uav_move_coverage_reward = 0.0
        args.uav_coverage_threshold_reward = 0.0
        args.uav_frontier_alignment_reward = 0.0
        args.uav_confidence_reward = 0.0
        args.uav_confidence_move_reward = 0.0
        args.uav_inefficient_move_penalty = 0.0
        args.uav_confidence_overlap_penalty = 0.0
        args.uav_cleanup_target_progress_reward = 0.0
        args.uav_astar_progress_reward = 0.0
        args.uav_overlap_penalty = 0.0
        args.uav_inter_uav_overlap_penalty = 0.0
        args.uav_outside_footprint_penalty = 0.0
    if args.uav_frontier_obs is None:
        args.uav_frontier_obs = False
    if args.uav_confidence_obs_grid is None:
        args.uav_confidence_obs_grid = 0
    if args.uav_local_confidence_obs_grid is None:
        args.uav_local_confidence_obs_grid = 0
    if args.uav_local_confidence_obs_radius_m is None:
        args.uav_local_confidence_obs_radius_m = 150.0
    if args.uav_frontier_obs_radius_m is None:
        args.uav_frontier_obs_radius_m = DEFAULT_UAV_FRONTIER_OBS_RADIUS_M
    if args.uav_frontier_mode is None:
        args.uav_frontier_mode = DEFAULT_UAV_FRONTIER_MODE
    if args.uav_frontier_source is None:
        args.uav_frontier_source = "coverage"
    if args.uav_frontier_alignment_reward is None:
        args.uav_frontier_alignment_reward = 0.0
    if args.uav_cleanup_target_obs is None:
        args.uav_cleanup_target_obs = False
    if args.uav_cleanup_target_refresh_mode is None:
        args.uav_cleanup_target_refresh_mode = "exact"
    if args.uav_confidence_reward is None:
        args.uav_confidence_reward = 0.0
    if args.uav_confidence_move_reward is None:
        args.uav_confidence_move_reward = 0.0
    if args.uav_inefficient_move_penalty is None:
        args.uav_inefficient_move_penalty = 0.0
    if args.uav_inefficient_move_source is None:
        args.uav_inefficient_move_source = "confidence"
    if args.uav_confidence_overlap_penalty is None:
        args.uav_confidence_overlap_penalty = 0.0
    if args.uav_confidence_overlap_mode is None:
        args.uav_confidence_overlap_mode = "raw"
    if args.uav_confidence_overlap_threshold is None:
        args.uav_confidence_overlap_threshold = 0.65
    if args.uav_cleanup_target_progress_reward is None:
        args.uav_cleanup_target_progress_reward = 0.0
    if args.share_param_by_agent_class is None:
        args.share_param_by_agent_class = bool(
            (args.joint_survivor_diagnostic or args.joint_schema_ugv_diagnostic)
            and not bool(args.share_param)
        )
    if args.share_param is None:
        args.share_param = False
    if bool(args.share_param) and bool(args.share_param_by_agent_class):
        p.error("--share-param and --share-param-by-agent-class are mutually exclusive")
    class_warmstart_dirs = [
        value for value in (args.warmstart_uav_model_dir, args.warmstart_ugv_model_dir)
        if value
    ]
    if class_warmstart_dirs:
        if args.model_dir:
            p.error("--model-dir cannot be combined with --warmstart-uav-model-dir/--warmstart-ugv-model-dir")
        if not bool(args.share_param_by_agent_class):
            p.error("--warmstart-uav-model-dir/--warmstart-ugv-model-dir require --share-param-by-agent-class")
        for value in class_warmstart_dirs:
            actor_path = Path(value) / "actor_agent0.pt"
            if not actor_path.is_file():
                p.error(f"class warm-start directory is missing actor_agent0.pt: {value}")
    args.uav_frontier_mode = str(args.uav_frontier_mode).replace("-", "_")
    args.uav_frontier_source = str(args.uav_frontier_source).replace("-", "_").lower()
    args.uav_cleanup_target_refresh_mode = str(args.uav_cleanup_target_refresh_mode).replace("-", "_").lower()

    (
        args.uav_coverage_reward,
        args.uav_move_coverage_reward,
        args.uav_overlap_penalty,
        args.uav_overlap_allowed,
        args.uav_outside_footprint_penalty,
    ) = _resolve_uav_reward_defaults(
        uav_survivor_diagnostic=uav_search_diagnostic,
        uav_coverage_reward=args.uav_coverage_reward,
        uav_move_coverage_reward=args.uav_move_coverage_reward,
        uav_overlap_penalty=args.uav_overlap_penalty,
        uav_overlap_allowed=args.uav_overlap_allowed,
        uav_outside_footprint_penalty=args.uav_outside_footprint_penalty,
    )

    if args.preset == "floor0-1km":
        # Validated floor-0 config: small (1km) terrain restores real detection
        # geometry, wide FOV + high altitude enlarge the scout footprint, and
        # long episodes give time to confirm. Expert recall ceiling ~0.47 here
        # at floor 0 (scripts/diag_floor0_ceiling.py), so RL has signal to learn.
        if args.terrain_cache_path is None:
            args.terrain_cache_path = str(
                ROOT / "data" / "terrain_cache" / "malibu_creek_1km_128.npz"
            )
        if args.drone_camera_fov_deg is None:
            args.drone_camera_fov_deg = 140.0
        if drone_flight_levels_m is None:
            drone_flight_levels_m = (50.0, 80.0, 100.0)
        if args.ground_confirmation_range_m is None:
            args.ground_confirmation_range_m = 30.0
        args.recurrent = True
        # Confirmation-dominant reward is the default for this preset (search
        # reward let the ground robots stand still). --reward-search still
        # overrides if explicitly requested.
        if not args.reward_search:
            args.reward_confirm = True
        # Floor stays 0 by design.
        args.drone_min_footprint = 0.0
        args.ground_confirm_min = 0.0
        if args.entropy_coef == 0.01:
            args.entropy_coef = 0.02
        # Team-coverage observation: lets the policy learn systematic sweeping.
        if args.coverage_obs_grid is None:
            args.coverage_obs_grid = 6

    if args.research:
        num_env_steps  = args.num_env_steps  or 400_000
        episode_length = args.episode_length or (1_000 if args.preset == "floor0-1km" else 500)
    elif args.uav_survivor_diagnostic or args.joint_schema_uav_diagnostic or args.joint_survivor_diagnostic:
        num_env_steps  = args.num_env_steps  or 2_000
        episode_length = args.episode_length or DEFAULT_UAV_DIAG_EPISODE_LENGTH
    elif args.preset == "floor0-1km":
        num_env_steps  = args.num_env_steps  or 240_000
        episode_length = args.episode_length or 1_000
    else:
        num_env_steps  = args.num_env_steps  or 2_000
        episode_length = args.episode_length or 150
    if num_env_steps <= 0:
        p.error("--num-env-steps must be positive")
    if episode_length <= 0:
        p.error("--episode-length must be positive")

    if args.preset == "tuned":
        # Keep explicit user values, but upgrade defaults to convergence-oriented
        # settings that consistently beat the smoke profile.
        if args.entropy_coef == 0.01:
            args.entropy_coef = 0.02
        if args.drone_min_footprint <= 0.0:
            args.drone_min_footprint = 0.15
        if args.ground_confirm_min <= 0.0:
            args.ground_confirm_min = 0.12
        args.recurrent = True
        args.reward_search = True

    print("=" * 60)
    print(f" OmniSearch — HAPPO ({'RESEARCH' if args.research else 'SMOKE'})")
    print(f" num_env_steps:  {num_env_steps}")
    print(f" episode_length: {episode_length}")
    print(f" seed:           {args.seed}")
    print(f" comms_dropout:  {args.comms_dropout}")
    print(f" entropy_coef:   {args.entropy_coef}")
    print(f" lr:             {args.lr}")
    print(f" critic_lr:      {args.critic_lr}")
    print(f" linear_lr_decay: {args.linear_lr_decay}")
    print(f" share_param:    {args.share_param}")
    print(f" share_param_by_agent_class: {args.share_param_by_agent_class}")
    print(f" terrain_cnn_encoder: {args.terrain_cnn_encoder}")
    print(f" local_map_patch_size: {args.local_map_patch_size}")
    print(f" local_coverage_obs_grid: {args.local_coverage_obs_grid}")
    print(f" local_coverage_obs_radius_m: {args.local_coverage_obs_radius_m}")
    print(f" uav_confidence_obs_grid: {args.uav_confidence_obs_grid}")
    print(f" uav_local_confidence_obs_grid: {args.uav_local_confidence_obs_grid}")
    print(f" uav_local_confidence_obs_radius_m: {args.uav_local_confidence_obs_radius_m}")
    print(f" uav_frontier_obs: {args.uav_frontier_obs}")
    print(f" uav_frontier_obs_radius_m: {args.uav_frontier_obs_radius_m}")
    print(f" uav_frontier_mode: {args.uav_frontier_mode}")
    print(f" uav_frontier_source: {args.uav_frontier_source}")
    print(f" uav_frontier_sectors: {args.uav_frontier_sectors}")
    print(f" uav_frontier_top_k: {args.uav_frontier_top_k}")
    print(f" uav_frontier_ownership: {args.uav_frontier_ownership}")
    print(f" uav_cleanup_target_obs: {args.uav_cleanup_target_obs}")
    print(f" uav_cleanup_target_grid: {args.uav_cleanup_target_grid}")
    print(f" uav_cleanup_target_hold_steps: {args.uav_cleanup_target_hold_steps}")
    print(f" uav_cleanup_target_refresh_mode: {args.uav_cleanup_target_refresh_mode}")
    print(f" uav_astar_route_obs: {args.uav_astar_route_obs}")
    print(f" uav_astar_grid: {args.uav_astar_grid}")
    print(f" uav_astar_confidence_cost_alpha: {args.uav_astar_confidence_cost_alpha}")
    print(f" uav_astar_confidence_cost_gamma: {args.uav_astar_confidence_cost_gamma}")
    print(f" uav_astar_waypoint_lookahead_m: {args.uav_astar_waypoint_lookahead_m}")
    print(f" uav_astar_route_replan_steps: {args.uav_astar_route_replan_steps}")
    print(f" uav_astar_waypoint_reached_m: {args.uav_astar_waypoint_reached_m}")
    print(f" uav_diagnostic_drones: {args.uav_diagnostic_drones}")
    print(f" n_drones: {args.n_drones if args.n_drones is not None else 'default'}")
    print(f" n_ugvs: {args.n_ugvs if args.n_ugvs is not None else 'default'}")
    print(f" joint_schema_uav_diagnostic: {args.joint_schema_uav_diagnostic}")
    print(f" joint_survivor_diagnostic: {args.joint_survivor_diagnostic}")
    print(f" joint_schema_ugv_diagnostic: {args.joint_schema_ugv_diagnostic}")
    print(f" n_survivors: {args.n_survivors if args.n_survivors is not None else 'default'}")
    print(f" joint_diagnostic_ugvs: {args.joint_diagnostic_ugvs}")
    print(
        " survivor_reveal: "
        f"delayed={args.delayed_survivor_knowledge or args.joint_schema_ugv_diagnostic} "
        f"schedule={args.survivor_reveal_schedule} "
        f"initial={args.survivor_reveal_initial_count} "
        f"range={args.survivor_reveal_start_step}-{args.survivor_reveal_end_step}"
    )
    print(f" ugv_planner_hint: {args.ugv_planner_hint}")
    print(f" ugv_planner_detour_obs: {bool(args.ugv_planner_detour_obs)}")
    print(f" ugv_route_aware_reward: {bool(args.ugv_route_aware_reward)}")
    print(f" ugv_dense_reward_mode: {args.ugv_dense_reward_mode}")
    print(f" ugv_planner_blend_weight: {args.ugv_planner_blend_weight}")
    print(
        " ugv_escape_route_switch: "
        f"stall_steps={args.ugv_escape_stall_steps} "
        f"progress_thr={args.ugv_escape_progress_threshold_m}m "
        f"move_thr={args.ugv_escape_movement_threshold_m}m "
        f"reached={args.ugv_escape_waypoint_reached_m}m "
        f"max_steps={args.ugv_escape_max_steps}"
    )
    print(f" ugv_planner_patch_size: {args.ugv_planner_patch_size}")
    print(f" ugv_global_planner_lookahead_m: {args.ugv_global_planner_lookahead_m}")
    print(f" ugv_global_planner_heuristic: {args.ugv_global_planner_heuristic}")
    print(f" ugv_planner_fire_mode: {args.ugv_planner_fire_mode}")
    print(f" ugv_planner_fire_replan_policy: {args.ugv_planner_fire_replan_policy}")
    print(f" ugv_planner_fire_replan_interval_steps: {args.ugv_planner_fire_replan_interval_steps}")
    print(f" ugv_planner_fire_block_threshold: {args.ugv_planner_fire_block_threshold}")
    print(f" ugv_planner_land_cover_costs: {args.ugv_planner_land_cover_costs}")
    print(f" ugv_target_assignment_mode: {args.ugv_target_assignment_mode}")
    print(f" ugv_assigned_target_obs_only: {args.ugv_assigned_target_obs_only}")
    print(f" survivor_assignment_obs: {args.survivor_assignment_obs}")
    print(
        " ugv_sticky_assignment: "
        f"margin={args.ugv_sticky_switch_margin_m}m "
        f"ratio={args.ugv_sticky_switch_ratio} "
        f"min_age={args.ugv_sticky_min_age_steps}"
    )
    print(f" ugv_planner_progress_reward: {args.ugv_planner_progress_reward}")
    print(f" ugv_ground_shaping_reward: {args.ugv_ground_shaping_reward}")
    print(f" ugv_movement_alignment_reward: {args.ugv_movement_alignment_reward}")
    print(f" ugv_pending_penalty: {args.ugv_pending_penalty}")
    print(f" ugv_route_progress_floor_penalty: {args.ugv_route_progress_floor_penalty}")
    print(f" ugv_route_progress_floor_m: {args.ugv_route_progress_floor_m}")
    print(f" ugv_route_progress_shortfall_penalty: {args.ugv_route_progress_shortfall_penalty}")
    print(f" uav_coverage_only: {args.uav_coverage_only}")
    print(f" uav_all_survivors_reward: {args.uav_all_survivors_reward}")
    print(f" team_scout_reward: {args.team_scout_reward}")
    print(f" uav_coverage_reward: {args.uav_coverage_reward}")
    print(f" uav_coverage_normalization: {args.uav_coverage_normalization}")
    print(f" uav_move_coverage_reward: {args.uav_move_coverage_reward}")
    print(f" uav_move_coverage_normalization: {args.uav_move_coverage_normalization}")
    print(f" uav_move_coverage_cap: {args.uav_move_coverage_cap}")
    print(f" uav_coverage_threshold_reward: {args.uav_coverage_threshold_reward}")
    print(f" uav_coverage_threshold_fraction: {args.uav_coverage_threshold_fraction}")
    print(f" uav_coverage_opportunity_cap: {args.uav_coverage_opportunity_cap}")
    print(f" uav_frontier_alignment_reward: {args.uav_frontier_alignment_reward}")
    print(f" uav_confidence_reward: {args.uav_confidence_reward}")
    print(f" uav_team_confidence_reward: {args.uav_team_confidence_reward}")
    print(f" uav_team_confidence_overlap_penalty: {args.uav_team_confidence_overlap_penalty}")
    print(f" uav_confidence_move_reward: {args.uav_confidence_move_reward}")
    print(f" uav_inefficient_move_penalty: {args.uav_inefficient_move_penalty}")
    print(f" uav_inefficient_move_source: {args.uav_inefficient_move_source}")
    print(f" uav_confidence_overlap_penalty: {args.uav_confidence_overlap_penalty}")
    print(f" uav_confidence_overlap_mode: {args.uav_confidence_overlap_mode}")
    print(f" uav_confidence_overlap_allowed_regret: {args.uav_confidence_overlap_allowed_regret}")
    print(f" uav_cleanup_target_progress_reward: {args.uav_cleanup_target_progress_reward}")
    print(f" uav_astar_progress_reward: {args.uav_astar_progress_reward}")
    print(f" uav_confidence_overlap_threshold: {args.uav_confidence_overlap_threshold}")
    print(f" uav_confidence_gamma: {args.uav_confidence_gamma}")
    print(f" uav_confidence_eps: {args.uav_confidence_eps}")
    print(f" uav_confidence_opportunity_eps: {args.uav_confidence_opportunity_eps}")
    print(f" uav_overlap_penalty: {args.uav_overlap_penalty}")
    print(f" uav_overlap_allowed: {args.uav_overlap_allowed}")
    print(f" uav_overlap_penalty_normalization: {args.uav_overlap_penalty_normalization}")
    print(f" uav_inter_uav_overlap_penalty: {args.uav_inter_uav_overlap_penalty}")
    print(f" uav_inter_uav_overlap_allowed: {args.uav_inter_uav_overlap_allowed}")
    print(f" uav_outside_footprint_penalty: {args.uav_outside_footprint_penalty}")
    print(f" uav_boundary_soft_margin_m: {args.uav_boundary_soft_margin_m}")
    print(f" uav_start_min_separation_m: {args.uav_start_min_separation_m}")
    print(f" uav_start_edge_margin_m: {args.uav_start_edge_margin_m}")
    print(f" action_transform: {args.action_transform}")
    print(f" warmstart_uav_model_dir: {args.warmstart_uav_model_dir}")
    print(f" warmstart_ugv_model_dir: {args.warmstart_ugv_model_dir}")
    print(f" exp_name:       {args.exp_name}")
    print("=" * 60)

    _register_wildfire_with_harl()
    from agents.harl_runner import _build_diagnostic_happo_runner_class
    from agents.happo_checkpoint import save_training_manifest
    OnPolicyHARunner = _build_diagnostic_happo_runner_class()

    harl_args, algo_args, env_args = build_args(
        num_env_steps  = num_env_steps,
        episode_length = episode_length,
        seed           = args.seed,
        comms_dropout  = args.comms_dropout,
        entropy_coef   = args.entropy_coef,
        lr             = args.lr,
        critic_lr      = args.critic_lr,
        linear_lr_decay = args.linear_lr_decay,
        share_param    = args.share_param,
        share_param_by_agent_class = args.share_param_by_agent_class,
        exp_name       = args.exp_name,
        n_rollout_threads = args.n_rollout_threads,
        terrain_cache_path = args.terrain_cache_path,
        drone_min_footprint_m = args.drone_min_footprint_radius_m,
        ground_confirm_min_m = args.ground_min_confirm_radius_m,
        fire_grid_size = args.fire_grid_size,
        local_map_patch_size = args.local_map_patch_size,
        reward_search = args.reward_search,
        reward_confirm = args.reward_confirm,
        recurrent = args.recurrent,
        model_dir = args.model_dir,
        warmstart_uav_model_dir = args.warmstart_uav_model_dir,
        warmstart_ugv_model_dir = args.warmstart_ugv_model_dir,
        drone_camera_fov_deg = args.drone_camera_fov_deg,
        drone_flight_levels_m = drone_flight_levels_m,
        ground_confirmation_range_m = args.ground_confirmation_range_m,
        coverage_obs_grid = args.coverage_obs_grid,
        confirm_requires_los = args.confirm_requires_los,
        drone_can_confirm = args.drone_can_confirm,
        r_drone_confirm = args.r_drone_confirm,
        local_coverage_obs_grid = args.local_coverage_obs_grid,
        local_coverage_obs_radius_m = args.local_coverage_obs_radius_m,
        uav_confidence_obs_grid = args.uav_confidence_obs_grid,
        uav_local_confidence_obs_grid = args.uav_local_confidence_obs_grid,
        uav_local_confidence_obs_radius_m = args.uav_local_confidence_obs_radius_m,
        uav_frontier_obs = args.uav_frontier_obs,
        uav_frontier_obs_radius_m = args.uav_frontier_obs_radius_m,
        uav_frontier_mode = args.uav_frontier_mode,
        uav_frontier_source = args.uav_frontier_source,
        uav_frontier_sectors = args.uav_frontier_sectors,
        uav_frontier_top_k = args.uav_frontier_top_k,
        uav_frontier_ownership = args.uav_frontier_ownership,
        uav_cleanup_target_obs = args.uav_cleanup_target_obs,
        uav_cleanup_target_grid = args.uav_cleanup_target_grid,
        uav_cleanup_target_hold_steps = args.uav_cleanup_target_hold_steps,
        uav_cleanup_target_confidence_threshold = args.uav_cleanup_target_confidence_threshold,
        uav_cleanup_target_min_value = args.uav_cleanup_target_min_value,
        uav_cleanup_target_assignment_distance_scale_m = args.uav_cleanup_target_assignment_distance_scale_m,
        uav_cleanup_target_refresh_mode = args.uav_cleanup_target_refresh_mode,
        uav_astar_route_obs = args.uav_astar_route_obs,
        uav_astar_grid = args.uav_astar_grid,
        uav_astar_confidence_cost_alpha = args.uav_astar_confidence_cost_alpha,
        uav_astar_confidence_cost_gamma = args.uav_astar_confidence_cost_gamma,
        uav_astar_waypoint_lookahead_m = args.uav_astar_waypoint_lookahead_m,
        uav_astar_route_replan_steps = args.uav_astar_route_replan_steps,
        uav_astar_waypoint_reached_m = args.uav_astar_waypoint_reached_m,
        ugv_known_survivor_diagnostic = args.ugv_known_survivor_diagnostic,
        uav_survivor_diagnostic = args.uav_survivor_diagnostic,
        joint_schema_uav_diagnostic = args.joint_schema_uav_diagnostic,
        joint_survivor_diagnostic = args.joint_survivor_diagnostic,
        joint_schema_ugv_diagnostic = args.joint_schema_ugv_diagnostic,
        uav_diagnostic_drones = args.uav_diagnostic_drones,
        joint_diagnostic_ugvs = args.joint_diagnostic_ugvs,
        n_drones = args.n_drones,
        n_ugvs = args.n_ugvs,
        n_survivors = args.n_survivors,
        delayed_survivor_knowledge = bool(
            args.delayed_survivor_knowledge or args.joint_schema_ugv_diagnostic
        ),
        survivor_reveal_schedule = args.survivor_reveal_schedule,
        survivor_reveal_initial_count = args.survivor_reveal_initial_count,
        survivor_reveal_start_step = args.survivor_reveal_start_step,
        survivor_reveal_end_step = args.survivor_reveal_end_step,
        survivor_assignment_obs = args.survivor_assignment_obs,
        ugv_diagnostic_target_distance_min_m = args.ugv_diagnostic_target_distance_min_m,
        ugv_diagnostic_target_distance_max_m = args.ugv_diagnostic_target_distance_max_m,
        uav_no_global_coverage_obs = args.uav_no_global_coverage_obs,
        uav_coverage_only = args.uav_coverage_only,
        uav_found_survivor_reward = args.uav_found_survivor_reward,
        uav_all_survivors_reward = args.uav_all_survivors_reward,
        team_scout_reward = args.team_scout_reward,
        uav_time_penalty = args.uav_time_penalty,
        uav_coverage_reward = args.uav_coverage_reward,
        uav_coverage_normalization = args.uav_coverage_normalization,
        uav_move_coverage_reward = args.uav_move_coverage_reward,
        uav_move_coverage_normalization = args.uav_move_coverage_normalization,
        uav_move_coverage_cap = args.uav_move_coverage_cap,
        uav_coverage_threshold_reward = args.uav_coverage_threshold_reward,
        uav_coverage_threshold_fraction = args.uav_coverage_threshold_fraction,
        uav_coverage_opportunity_reward = args.uav_coverage_opportunity_reward,
        uav_coverage_opportunity_cap = args.uav_coverage_opportunity_cap,
        uav_frontier_alignment_reward = args.uav_frontier_alignment_reward,
        uav_confidence_reward = args.uav_confidence_reward,
        uav_team_confidence_reward = args.uav_team_confidence_reward,
        uav_team_confidence_overlap_penalty = args.uav_team_confidence_overlap_penalty,
        uav_confidence_move_reward = args.uav_confidence_move_reward,
        uav_inefficient_move_penalty = args.uav_inefficient_move_penalty,
        uav_inefficient_move_source = args.uav_inefficient_move_source,
        uav_confidence_overlap_penalty = args.uav_confidence_overlap_penalty,
        uav_confidence_overlap_mode = args.uav_confidence_overlap_mode,
        uav_confidence_overlap_allowed_regret = args.uav_confidence_overlap_allowed_regret,
        uav_cleanup_target_progress_reward = args.uav_cleanup_target_progress_reward,
        uav_astar_progress_reward = args.uav_astar_progress_reward,
        uav_confidence_overlap_threshold = args.uav_confidence_overlap_threshold,
        uav_confidence_gamma = args.uav_confidence_gamma,
        uav_confidence_eps = args.uav_confidence_eps,
        uav_confidence_opportunity_eps = args.uav_confidence_opportunity_eps,
        uav_overlap_penalty = args.uav_overlap_penalty,
        uav_overlap_allowed = args.uav_overlap_allowed,
        uav_overlap_penalty_normalization = args.uav_overlap_penalty_normalization,
        uav_inter_uav_overlap_penalty = args.uav_inter_uav_overlap_penalty,
        uav_inter_uav_overlap_allowed = args.uav_inter_uav_overlap_allowed,
        uav_outside_footprint_penalty = args.uav_outside_footprint_penalty,
        uav_boundary_soft_margin_m = args.uav_boundary_soft_margin_m,
        uav_start_min_separation_m = args.uav_start_min_separation_m,
        uav_start_edge_margin_m = args.uav_start_edge_margin_m,
        ugv_ground_shaping_reward = args.ugv_ground_shaping_reward,
        ugv_movement_alignment_reward = args.ugv_movement_alignment_reward,
        ugv_pending_penalty = args.ugv_pending_penalty,
        ugv_planner_progress_reward = args.ugv_planner_progress_reward,
        ugv_route_aware_reward = bool(args.ugv_route_aware_reward),
        ugv_dense_reward_mode = args.ugv_dense_reward_mode,
        ugv_planner_blend_weight = args.ugv_planner_blend_weight,
        ugv_escape_stall_steps = args.ugv_escape_stall_steps,
        ugv_escape_progress_threshold_m = args.ugv_escape_progress_threshold_m,
        ugv_escape_movement_threshold_m = args.ugv_escape_movement_threshold_m,
        ugv_escape_waypoint_reached_m = args.ugv_escape_waypoint_reached_m,
        ugv_escape_max_steps = args.ugv_escape_max_steps,
        ugv_approach_reward = args.ugv_approach_reward,
        ugv_approach_milestone_radii_m = tuple(args.ugv_approach_milestone_radii_m),
        ugv_stall_penalty = args.ugv_stall_penalty,
        ugv_stall_displacement_threshold_m = args.ugv_stall_displacement_threshold_m,
        ugv_route_progress_floor_penalty = args.ugv_route_progress_floor_penalty,
        ugv_route_progress_floor_m = args.ugv_route_progress_floor_m,
        ugv_route_progress_shortfall_penalty = args.ugv_route_progress_shortfall_penalty,
        slope_speed_weight = args.slope_speed_weight,
        land_cover_speeds = tuple(args.land_cover_speeds) if args.land_cover_speeds is not None else None,
        action_transform = args.action_transform,
        terrain_cnn_encoder = args.terrain_cnn_encoder,
        terrain_cnn_embed_dim = args.terrain_cnn_embed_dim,
        ugv_planner_hint = args.ugv_planner_hint,
        ugv_planner_detour_obs = bool(args.ugv_planner_detour_obs),
        ugv_planner_patch_size = args.ugv_planner_patch_size,
        ugv_planner_lookahead_cells = args.ugv_planner_lookahead_cells,
        ugv_global_planner_lookahead_m = args.ugv_global_planner_lookahead_m,
        ugv_global_planner_heuristic = args.ugv_global_planner_heuristic,
        ugv_planner_fire_mode = args.ugv_planner_fire_mode,
        ugv_planner_fire_replan_policy = args.ugv_planner_fire_replan_policy,
        ugv_planner_fire_replan_interval_steps = args.ugv_planner_fire_replan_interval_steps,
        ugv_planner_fire_cost = args.ugv_planner_fire_cost,
        ugv_planner_fire_block_threshold = args.ugv_planner_fire_block_threshold,
        ugv_planner_smoke_cost = args.ugv_planner_smoke_cost,
        ugv_planner_smolder_cost = args.ugv_planner_smolder_cost,
        ugv_planner_fire_buffer_m = args.ugv_planner_fire_buffer_m,
        ugv_planner_fire_buffer_cost = args.ugv_planner_fire_buffer_cost,
        ugv_planner_land_cover_costs = (
            tuple(args.ugv_planner_land_cover_costs)
            if args.ugv_planner_land_cover_costs is not None else None
        ),
        ugv_target_assignment_mode = args.ugv_target_assignment_mode,
        ugv_assigned_target_obs_only = args.ugv_assigned_target_obs_only,
        ugv_sticky_switch_margin_m = args.ugv_sticky_switch_margin_m,
        ugv_sticky_switch_ratio = args.ugv_sticky_switch_ratio,
        ugv_sticky_min_age_steps = args.ugv_sticky_min_age_steps,
        enable_fire = bool(args.enable_fire),
    )
    print(f" log dir: {algo_args['logger']['log_dir']}")
    print("-" * 60)

    t0 = time.time()
    runner = OnPolicyHARunner(harl_args, algo_args, env_args)
    manifest_path = save_training_manifest(
        runner,
        harl_args=harl_args,
        algo_args=algo_args,
        env_args=env_args,
    )
    print(f" training config: {manifest_path}")
    print(f" tensorboard log dir: {runner.log_dir}")
    print(f" tensorboard: tensorboard --logdir \"{runner.log_dir}\" --port 6006")
    runner.run()
    runner.close()

    print("-" * 60)
    print(f" HAPPO training complete in {time.time() - t0:.1f}s")
    print(f" Checkpoints saved to: {algo_args['logger']['log_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
