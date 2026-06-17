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
    linear_lr_decay: bool = False,
    n_rollout_threads: int = 1,
    terrain_cache_path: str | None = None,
    drone_min_footprint_m: float = 0.0,
    ground_confirm_min_m: float = 0.0,
    fire_grid_size: int = 128,
    reward_search: bool = False,
    reward_confirm: bool = False,
    recurrent: bool = False,
    model_dir: str | None = None,
    drone_camera_fov_deg: float | None = None,
    drone_flight_levels_m: tuple[float, ...] | None = None,
    ground_confirmation_range_m: float | None = None,
    coverage_obs_grid: int = 0,
    ugv_known_survivor_diagnostic: bool = False,
    uav_survivor_diagnostic: bool = False,
    ugv_diagnostic_target_distance_min_m: float | None = None,
    ugv_diagnostic_target_distance_max_m: float | None = None,
    uav_diagnostic_target_distance_min_m: float | None = None,
    uav_diagnostic_target_distance_max_m: float | None = None,
    uav_coverage_reward: float = 5.0,
    ugv_movement_alignment_reward: float = 0.20,
    ugv_planner_progress_reward: float = 0.0,
    ugv_approach_reward: float = DEFAULT_UGV_APPROACH_REWARD,
    ugv_approach_milestone_radii_m: tuple[float, ...] = DEFAULT_UGV_APPROACH_MILESTONE_RADII_M,
    ugv_stall_penalty: float = 0.0,
    ugv_stall_displacement_threshold_m: float = 0.05,
    local_map_patch_size: int = 3,
    slope_speed_weight: float | None = None,
    land_cover_speeds: tuple[float, ...] | None = None,
    action_transform: str = "clip",
    terrain_cnn_encoder: bool = False,
    terrain_cnn_embed_dim: int = 16,
    ugv_planner_hint: str = "none",
    ugv_planner_patch_size: int = 11,
    ugv_planner_lookahead_cells: int = 10,
) -> tuple[dict, dict, dict]:
    ugv_planner_hint = str(ugv_planner_hint).replace("-", "_")
    if ugv_planner_hint not in {"none", "local_astar"}:
        raise ValueError("ugv_planner_hint must be one of: none, local_astar")
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
    if ugv_planner_progress_reward > 0.0 and ugv_planner_hint != "local_astar":
        raise ValueError("ugv_planner_progress_reward requires ugv_planner_hint='local_astar'")
    uav_coverage_reward = float(uav_coverage_reward)
    if uav_coverage_reward < 0.0:
        raise ValueError("uav_coverage_reward must be nonnegative")

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
            "share_param":             False,
            "fixed_order":             False,
        },
        "logger": {
            "log_dir": str(ROOT / "results" / "harl_runs"),
        },
    }

    scenario_kwargs = {
        "max_steps":     episode_length,
        "n_drones":      3,
        "n_ground":      2,
        "n_survivors":   5,
        "comms_dropout": comms_dropout,
        "fire_grid_size": fire_grid_size,
        "local_map_patch_size": local_map_patch_size,
        "ugv_planner_hint": ugv_planner_hint,
        "ugv_planner_patch_size": ugv_planner_patch_size,
        "ugv_planner_lookahead_cells": ugv_planner_lookahead_cells,
        "drone_min_footprint_m": drone_min_footprint_m,
        "ground_confirm_min_m": ground_confirm_min_m,
        "r_found_survivor": 10.0,
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
    if coverage_obs_grid and coverage_obs_grid > 0:
        scenario_kwargs["coverage_obs_grid"] = int(coverage_obs_grid)
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

    if ugv_known_survivor_diagnostic and uav_survivor_diagnostic:
        raise ValueError("Choose only one diagnostic mode: UGV or UAV")

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
            "n_ground": 1,
            "n_survivors": 1,
            "known_survivors_at_reset": True,
            "disable_fire": True,
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
    if uav_survivor_diagnostic:
        distance_kwargs = {}
        if uav_diagnostic_target_distance_min_m is None and uav_diagnostic_target_distance_max_m is None:
            pass
        else:
            target_distance_min_m = max(
                float(0.0 if uav_diagnostic_target_distance_min_m is None else uav_diagnostic_target_distance_min_m),
                0.0,
            )
            distance_kwargs["known_survivor_spawn_distance_min_m"] = target_distance_min_m
            if uav_diagnostic_target_distance_max_m is not None:
                target_distance_max_m = max(
                    float(uav_diagnostic_target_distance_max_m),
                    0.0,
                )
                if target_distance_max_m < target_distance_min_m:
                    raise ValueError(
                        "uav_diagnostic_target_distance_max_m must be >= "
                        "uav_diagnostic_target_distance_min_m"
                    )
                target_distance_m = 0.5 * (target_distance_min_m + target_distance_max_m)
                distance_kwargs.update({
                    "known_survivor_spawn_distance_m": target_distance_m,
                    "known_survivor_spawn_distance_max_m": target_distance_max_m,
                })
        scenario_kwargs.update({
            "n_drones": 3,
            "n_ground": 0,
            "n_survivors": 5,
            "known_survivors_at_reset": False,
            "survivor_spawn_reference": "drone",
            "drone_scouts_confirm_survivors": True,
            "disable_fire": True,
            "comms_dropout": 0.0,
            "r_found_survivor": 10.0,
            "r_drone_scout": 2.0,
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
            "r_time_penalty": -0.0005,
            "r_coverage": uav_coverage_reward,
        })
        scenario_kwargs.update(distance_kwargs)
    n_agents = int(scenario_kwargs["n_drones"]) + int(scenario_kwargs["n_ground"])
    algo_args["model"]["terrain_cnn_single_obs_dim"] = wildfire_single_observation_dim(
        local_map_patch_size=int(local_map_patch_size),
        n_agents=n_agents,
        n_survivors=int(scenario_kwargs["n_survivors"]),
        ugv_planner_hint=ugv_planner_hint,
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
    p.add_argument("--linear-lr-decay", action="store_true",
                   help="Linearly decay actor/critic learning rates over training.")
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
    p.add_argument("--ugv-planner-hint", choices=("none", "local_astar", "local-astar"), default="none",
                   help="Optional UGV observation hint. local_astar exposes a local A* waypoint vector.")
    p.add_argument("--ugv-planner-patch-size", type=int, default=11,
                   help="Odd local grid size used by --ugv-planner-hint local_astar.")
    p.add_argument("--ugv-planner-lookahead-cells", type=int, default=10,
                   help="Maximum number of A* route cells to skip ahead when forming the waypoint hint.")
    p.add_argument("--model-dir", default=None,
                   help="Warm-start actors from a checkpoint dir (e.g. a behaviour-cloned results/bc_happo) and RL-fine-tune.")
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
    p.add_argument("--coverage-obs-grid", type=int, default=0,
                   help="Add a KxK team-coverage map + global fraction to the observation so the "
                        "policy can learn systematic sweeping (e.g. 6). 0 = off.")
    p.add_argument("--preset", choices=("smoke", "tuned", "floor0-1km"), default="smoke",
                   help="Preset for defaults. 'floor0-1km' (recommended) trains on the 1km terrain "
                        "with wide-FOV/high-altitude sensors so detection works at floor 0.")
    p.add_argument("--ugv-known-survivor-diagnostic", action="store_true",
                   help="Train a minimal diagnostic task: 0 drones, 1 UGV, 1 survivor known at reset, no fire.")
    p.add_argument("--uav-survivor-diagnostic", action="store_true",
                   help="Train a UAV-only diagnostic task: 3 UAVs, 0 UGVs, 5 survivors, no fire; drone scouting counts as success.")
    p.add_argument("--ugv-diagnostic-target-distance-min-m", type=float, default=None,
                   help="Minimum known-survivor start distance sampled at reset for the UGV diagnostic task.")
    p.add_argument("--ugv-diagnostic-target-distance-max-m", type=float, default=None,
                   help="Maximum known-survivor start distance sampled at reset for the UGV diagnostic task. "
                        "Omit for no upper bound; use min=max for an exact target distance.")
    p.add_argument("--uav-diagnostic-target-distance-min-m", type=float, default=None,
                   help="Minimum survivor start distance sampled from the UAV at reset for the UAV diagnostic task.")
    p.add_argument("--uav-diagnostic-target-distance-max-m", type=float, default=None,
                   help="Maximum survivor start distance sampled from the UAV at reset for the UAV diagnostic task. "
                        "Omit for no upper bound; use min=max for an exact target distance.")
    p.add_argument("--uav-coverage-reward", type=float, default=5.0,
                   help="Total reward scale for team-new UAV camera footprint coverage in the UAV diagnostic task.")
    p.add_argument("--ugv-movement-alignment-reward", type=float, default=0.20,
                   help="Reward scale for UGV actual-movement alignment toward a known survivor in the "
                        "diagnostic task.")
    p.add_argument("--ugv-planner-progress-reward", type=float, default=0.0,
                   help="Reward scale for actual UGV progress toward the local A* waypoint when "
                        "the planner detects a detour. Requires --ugv-planner-hint local_astar.")
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

    if args.land_cover_speeds is not None and len(args.land_cover_speeds) not in (5, 6):
        p.error("--land-cover-speeds must provide 5 or 6 values: road open brush forest rock [water]")
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
    if args.ugv_movement_alignment_reward < 0.0:
        p.error("--ugv-movement-alignment-reward must be nonnegative")
    if args.uav_coverage_reward < 0.0:
        p.error("--uav-coverage-reward must be nonnegative")
    if args.ugv_planner_progress_reward < 0.0:
        p.error("--ugv-planner-progress-reward must be nonnegative")
    if args.ugv_planner_progress_reward > 0.0 and args.ugv_planner_hint != "local_astar":
        p.error("--ugv-planner-progress-reward requires --ugv-planner-hint local_astar")
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
    if args.fire_grid_size < 2:
        p.error("--fire-grid-size must be at least 2")
    if args.ugv_known_survivor_diagnostic and args.uav_survivor_diagnostic:
        p.error("Choose only one diagnostic mode: --ugv-known-survivor-diagnostic or --uav-survivor-diagnostic")
    if args.terrain_cache_path is not None and not Path(args.terrain_cache_path).is_file():
        p.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")

    drone_flight_levels_m = None
    if args.drone_flight_levels_m:
        drone_flight_levels_m = tuple(
            float(v) for v in str(args.drone_flight_levels_m).split(",") if v.strip()
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
        if args.coverage_obs_grid <= 0:
            args.coverage_obs_grid = 6

    if args.research:
        num_env_steps  = args.num_env_steps  or 400_000
        episode_length = args.episode_length or (1_000 if args.preset == "floor0-1km" else 500)
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
    print(f" terrain_cnn_encoder: {args.terrain_cnn_encoder}")
    print(f" local_map_patch_size: {args.local_map_patch_size}")
    print(f" ugv_planner_hint: {args.ugv_planner_hint}")
    print(f" ugv_planner_patch_size: {args.ugv_planner_patch_size}")
    print(f" ugv_planner_progress_reward: {args.ugv_planner_progress_reward}")
    print(f" uav_coverage_reward: {args.uav_coverage_reward}")
    print(f" action_transform: {args.action_transform}")
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
        drone_camera_fov_deg = args.drone_camera_fov_deg,
        drone_flight_levels_m = drone_flight_levels_m,
        ground_confirmation_range_m = args.ground_confirmation_range_m,
        coverage_obs_grid = args.coverage_obs_grid,
        ugv_known_survivor_diagnostic = args.ugv_known_survivor_diagnostic,
        uav_survivor_diagnostic = args.uav_survivor_diagnostic,
        ugv_diagnostic_target_distance_min_m = args.ugv_diagnostic_target_distance_min_m,
        ugv_diagnostic_target_distance_max_m = args.ugv_diagnostic_target_distance_max_m,
        uav_diagnostic_target_distance_min_m = args.uav_diagnostic_target_distance_min_m,
        uav_diagnostic_target_distance_max_m = args.uav_diagnostic_target_distance_max_m,
        uav_coverage_reward = args.uav_coverage_reward,
        ugv_movement_alignment_reward = args.ugv_movement_alignment_reward,
        ugv_planner_progress_reward = args.ugv_planner_progress_reward,
        ugv_approach_reward = args.ugv_approach_reward,
        ugv_approach_milestone_radii_m = tuple(args.ugv_approach_milestone_radii_m),
        ugv_stall_penalty = args.ugv_stall_penalty,
        ugv_stall_displacement_threshold_m = args.ugv_stall_displacement_threshold_m,
        slope_speed_weight = args.slope_speed_weight,
        land_cover_speeds = tuple(args.land_cover_speeds) if args.land_cover_speeds is not None else None,
        action_transform = args.action_transform,
        terrain_cnn_encoder = args.terrain_cnn_encoder,
        terrain_cnn_embed_dim = args.terrain_cnn_embed_dim,
        ugv_planner_hint = args.ugv_planner_hint,
        ugv_planner_patch_size = args.ugv_planner_patch_size,
        ugv_planner_lookahead_cells = args.ugv_planner_lookahead_cells,
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
