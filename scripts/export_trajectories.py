"""
Export one trajectory per baseline strategy into web/trajectories/ as JSON.

The web viewer (`web/index.html`) auto-discovers any *.json in that folder
and lets the user switch between them. Run this whenever you want fresh
trajectories shown in the viewer:

    python scripts/export_trajectories.py
    python scripts/export_trajectories.py --seed 7 --steps 500
    python scripts/export_trajectories.py --seed 7 --steps 500 --grid-size 128
    python scripts/export_trajectories.py --approach happo
    python scripts/export_trajectories.py --comms-dropout 0.5  # show dropout effect
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmas

from envs.wildfire_defaults import (
    DRONE_CAMERA_FOV_DEG,
    DRONE_FLIGHT_LEVELS_M,
    DRONE_SAFETY_CLEARANCE_M,
    DRONE_SPEED_MPS,
    DRONE_U_MULTIPLIER,
    GROUND_ACCEL_MPS2,
    GROUND_SPEED_MPS,
    SIM_STEP_SECONDS,
)
from envs.wildfire_search import WildfireSearchScenario
from agents.baselines import BASELINES, RandomActionPolicy
from evaluation.trajectory_export import export_trajectory


def _selected_baselines(approach: str) -> list[str]:
    if approach == "all":
        return list(BASELINES)
    if approach == "happo":
        return []
    return [approach]


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def _resolve_happo_checkpoint(path: str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.name != "models" and (candidate / "models").is_dir():
        candidate = candidate / "models"
    if not candidate.is_dir():
        raise SystemExit(f"--happo-checkpoint does not exist or is not a directory: {path}")
    return candidate.resolve()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--approach",
        choices=("all", "happo", *BASELINES),
        default="all",
        help="Export all approaches (default), only HAPPO, or one named baseline.",
    )
    p.add_argument(
        "--ignore-happo-env",
        action="store_true",
        help="Use the CLI scenario settings exactly instead of restoring the "
             "latest HAPPO checkpoint environment. Intended for baseline-only "
             "terrain experiments.",
    )
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--frame-stride", type=int, default=1, help="Record every Nth frame (keeps long-run JSON loadable; physics still runs every step).")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default=str(ROOT / "web" / "trajectories"))
    p.add_argument("--x-semidim", type=float, default=1.0)
    p.add_argument("--y-semidim", type=float, default=1.0)
    p.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="Fire/terrain grid resolution (default: 128).",
    )
    p.add_argument(
        "--comms-dropout",
        type=float,
        default=0.0,
        help="Per-step prob each agent's teammate-obs is zeroed. "
             "0.0 = perfect radio, 0.3 = visible dropouts in viewer, "
             "0.8 = mostly broken.",
    )
    p.add_argument("--comms-dropout-mode", choices=("iid", "bursty"), default="iid",
                   help="Communication dropout process for trajectory export.")
    p.add_argument("--comms-map-mode", choices=("global", "per_agent", "per-agent"), default="global",
                   help="Coverage/confidence map observation memory for HAPPO export.")
    p.add_argument("--comms-dropout-min-steps", type=int, default=5,
                   help="Minimum outage duration for --comms-dropout-mode bursty.")
    p.add_argument("--comms-dropout-max-steps", type=int, default=15,
                   help="Maximum outage duration for --comms-dropout-mode bursty.")
    p.add_argument("--enable-fire", action="store_true",
                   help="Enable fire in exported trajectories when the merged scenario disables it.")
    p.add_argument("--joint-survivor-diagnostic", action="store_true",
                   help="Export the joint UAV+UGV diagnostic scenario when not relying on a checkpoint manifest.")
    p.add_argument("--joint-schema-ugv-diagnostic", action="store_true",
                   help="Export the 2-UGV delayed-knowledge joint-schema curriculum scenario.")
    p.add_argument("--joint-diagnostic-ugvs", type=int, default=1,
                   help="Number of UGVs for --joint-survivor-diagnostic manual exports.")
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
                   help="Override UGV assignment for known, unconfirmed survivor targets.")
    p.add_argument("--ugv-planner-fire-mode",
                   choices=("off", "cost", "block"),
                   default=None,
                   help="Override the UGV planner fire mode for trajectory export.")
    p.add_argument("--ugv-planner-fire-replan-policy",
                   choices=("always", "affected", "lazy"),
                   default=None,
                   help="Override when fire-aware UGV global routes are replanned after fire spread.")
    p.add_argument("--ugv-planner-fire-replan-interval-steps", type=int, default=None,
                   help="Override lazy fire-aware global route replan interval.")
    p.add_argument("--ugv-global-planner-heuristic",
                   choices=("euclidean", "terrain"),
                   default=None,
                   help="Override the UGV global_astar heuristic.")
    p.add_argument("--ugv-planner-fire-cost", type=float, default=None)
    p.add_argument("--ugv-planner-fire-block-threshold", type=float, default=None,
                   help="In block mode, only active fire cells with intensity >= threshold are blocked.")
    p.add_argument("--ugv-planner-smoke-cost", type=float, default=None)
    p.add_argument("--ugv-planner-smolder-cost", type=float, default=None)
    p.add_argument("--ugv-planner-fire-buffer-m", type=float, default=None)
    p.add_argument("--ugv-planner-fire-buffer-cost", type=float, default=None)
    p.add_argument("--ugv-planner-land-cover-costs", type=float, nargs="+", default=None,
                   help="Override planner-only land-cover costs for road/open/brush/forest/rock[/water].")
    p.add_argument(
        "--terrain-source",
        choices=("real",),
        default="real",
        help="Terrain backend. Only cached real terrain is supported.",
    )
    p.add_argument("--terrain-place", default="Malibu Creek State Park, California")
    p.add_argument("--terrain-cache-dir", default=str(ROOT / "data" / "terrain_cache"))
    p.add_argument("--terrain-cache-path", default=None)
    p.add_argument(
        "--drone-min-footprint-radius-m",
        dest="drone_min_footprint_radius_m",
        type=float,
        default=None,
        help="Override the HAPPO checkpoint's drone footprint floor in meters. "
             "Use 0 to disable it.",
    )
    p.add_argument(
        "--drone-min-footprint",
        dest="drone_min_footprint_radius_m",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ground-min-confirm-radius-m",
        dest="ground_min_confirm_radius_m",
        type=float,
        default=None,
        help="Override the HAPPO checkpoint's ground confirmation floor in meters. "
             "Use 0 to disable it.",
    )
    p.add_argument(
        "--ground-confirm-min",
        dest="ground_min_confirm_radius_m",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--drone-flight-levels-m",
        nargs="+",
        type=float,
        default=DRONE_FLIGHT_LEVELS_M,
        help="Drone AGL flight levels in meters.",
    )
    p.add_argument(
        "--drone-camera-fov-deg",
        type=float,
        default=DRONE_CAMERA_FOV_DEG,
        help="Downward drone camera field of view in degrees.",
    )
    p.add_argument(
        "--ground-confirmation-range-m",
        type=float,
        default=None,
        help="Physical ground confirmation range in meters (not a floor). "
             "Overrides the checkpoint manifest when set.",
    )
    p.add_argument(
        "--skip-happo-manifest",
        action="store_true",
        help="Do NOT merge the latest HAPPO checkpoint's training scenario config. "
             "Use this to render baselines under explicit (e.g. realistic) sensor flags.",
    )
    p.add_argument(
        "--happo-checkpoint",
        default=None,
        help="Path to a HAPPO checkpoint models/ directory, or to its parent run directory. "
             "Defaults to the newest checkpoint under results/harl_runs.",
    )
    p.add_argument(
        "--drone-safety-clearance-m",
        type=float,
        default=DRONE_SAFETY_CLEARANCE_M,
        help="Minimum aerial clearance above terrain obstacles in meters.",
    )
    p.add_argument(
        "--sim-step-seconds",
        type=float,
        default=SIM_STEP_SECONDS,
        help="Physical duration represented by one simulation step.",
    )
    p.add_argument(
        "--drone-speed-mps",
        type=float,
        default=DRONE_SPEED_MPS,
        help="Drone horizontal speed cap in meters per second.",
    )
    p.add_argument(
        "--drone-u-multiplier",
        type=float,
        default=DRONE_U_MULTIPLIER,
        help="VMAS drone action-to-acceleration multiplier.",
    )
    p.add_argument(
        "--ground-speed-mps",
        type=float,
        default=GROUND_SPEED_MPS,
        help="Nominal UGV horizontal speed cap on easy terrain in meters per second.",
    )
    p.add_argument(
        "--ground-accel-mps2",
        type=float,
        default=GROUND_ACCEL_MPS2,
        help="Nominal UGV operational acceleration in meters per second squared.",
    )
    p.add_argument(
        "--ground-u-multiplier",
        type=float,
        default=None,
        help="Optional VMAS UGV action-to-acceleration override. Defaults to a value derived from --ground-accel-mps2.",
    )
    p.add_argument("--enable-cv", action="store_true", help="Add NAIP/SARD preliminary CV detections to exported frames.")
    p.add_argument("--cv-out-dir", default=None, help="Directory for rendered CV images when --enable-cv is set.")
    p.add_argument("--cv-save-images-every", type=int, default=0, help="Save rendered drone images every N steps; 0 disables image writes.")
    p.add_argument("--cv-naip-image-path", default=None, help="Optional cached NAIP PNG matching the terrain bbox.")
    p.add_argument(
        "--cv-target-gsd-m",
        type=float,
        default=0.5,
        help="Target NAIP ground resolution in meters per pixel. Use 0.5 for high-quality simulation, 2.0 for tests.",
    )
    p.add_argument(
        "--cv-naip-size",
        type=int,
        default=8192,
        help="Fallback square NAIP image size when --cv-target-gsd-m is disabled.",
    )
    p.add_argument("--cv-tile-size", type=int, default=1024, help="Tile size for high-resolution NAIP export requests.")
    p.add_argument(
        "--cv-single-naip-export",
        action="store_true",
        help="Use one NAIP export request instead of tiled export. Faster, but usually lower quality/reliability.",
    )
    p.add_argument(
        "--cv-build-full-naip",
        action="store_true",
        help="Build one stitched full-bbox NAIP image instead of lazy tile caching.",
    )
    p.add_argument("--cv-image-size", type=int, default=512, help="Rendered detector crop size.")
    p.add_argument(
        "--cv-disable-wildfire-effects",
        action="store_true",
        help="Disable burn scar, flame, and smoke rendering in saved CV drone crops.",
    )
    p.add_argument("--cv-human-asset", default=str(ROOT / "data/cv_assets/sard_grabcut/sard_survivor_0280.png"))
    p.add_argument(
        "--cv-human-assets-dir",
        default=str(ROOT / "data/cv_assets/sard_grabcut"),
        help="Directory of transparent SARD survivor PNGs. Overrides --cv-human-asset when present.",
    )
    p.add_argument(
        "--cv-human-asset-list",
        default=str(ROOT / "configs/cv/sard_grabcut_asset_review.json"),
        help="Optional JSON review list. When set, only accepted_assets are used.",
    )
    p.add_argument(
        "--cv-preview-altitude-m",
        type=float,
        default=20.0,
        help="Altitude in meters for centered per-survivor preview images.",
    )
    p.add_argument("--cv-detection-probability", type=float, default=1.0)
    p.add_argument("--cv-pixel-noise-std", type=float, default=0.0)
    p.add_argument(
        "--cv-detector",
        choices=("preliminary", "yolo"),
        default="preliminary",
        help="CV detection backend. 'preliminary' echoes ground-truth boxes (fast, default); "
             "'yolo' runs a real YOLOv8 person detector over the rendered crop.",
    )
    p.add_argument("--cv-person-model", default="yolov8n.pt",
                   help="Ultralytics weights for the yolo backend (yolov8n/s/m/l/x.pt).")
    p.add_argument("--cv-person-conf", type=float, default=0.35,
                   help="YOLO confidence threshold for survivor detection.")
    p.add_argument("--cv-person-imgsz", type=int, default=None,
                   help="YOLO inference image size; defaults to max(cv-image-size, 1280) for small survivors.")
    p.add_argument("--cv-person-no-tiling", action="store_true",
                   help="Disable tiled small-object inference for the yolo backend.")
    p.add_argument("--cv-person-tile-grid", type=int, default=2,
                   help="Tile grid (NxN) for tiled yolo inference.")
    p.add_argument("--cv-augment", action="store_true",
                   help="Enable test-time augmentation (TTA) for YOLO inference. "
                        "Improves recall +2-5%% at ~3x inference cost.")
    p.add_argument("--cv-adaptive-conf", action="store_true",
                   help="Enable per-altitude adaptive confidence thresholds. "
                        "Lower threshold at high altitude (small targets), higher at low altitude.")
    p.add_argument("--cv-tracking", action="store_true",
                   help="Enable ByteTrack multi-object tracking for temporal FP suppression.")
    p.add_argument("--cv-tracking-min-hits", type=int, default=2,
                   help="Minimum consecutive detections before a track is confirmed.")
    p.add_argument("--detection-mode", type=str, default="cv",
                   choices=["cv", "thermal", "cv+thermal", "motion", "cv+motion"],
                   help="Detection modality: 'cv' (pure computer vision, default), "
                        "'thermal' (simulated thermal IR only), "
                        "'cv+thermal' (sensor fusion - both run, results merged), "
                        "'motion' (frame differencing only), "
                        "'cv+motion' (CV with motion confirmation boost).")
    p.add_argument("--thermal-detector", type=str, default="physics",
                   choices=["physics", "yolo"],
                   help="Thermal backend for thermal/cv+thermal modes: 'physics' "
                        "(closed-form probability model, default) or 'yolo' (render "
                        "a simulated TIR frame and run models/thermal_yolov8n.pt).")
    p.add_argument("--cv-camera-tilt", type=float, default=0.0,
                   help="Drone camera tilt in degrees from nadir (0 = straight down, "
                        "default). Nonzero values simulate an oblique/side-angle view: "
                        "survivors render taller, matching --oblique-frac training data.")
    args = p.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.grid_size < 8:
        raise SystemExit("--grid-size must be at least 8; try 16, 32, or 64")
    if args.joint_diagnostic_ugvs < 1:
        raise SystemExit("--joint-diagnostic-ugvs must be positive")
    if args.joint_survivor_diagnostic and args.joint_schema_ugv_diagnostic:
        raise SystemExit("--joint-survivor-diagnostic and --joint-schema-ugv-diagnostic are mutually exclusive")
    if args.comms_dropout_min_steps < 1:
        raise SystemExit("--comms-dropout-min-steps must be >= 1")
    if args.comms_dropout_max_steps < args.comms_dropout_min_steps:
        raise SystemExit("--comms-dropout-max-steps must be >= --comms-dropout-min-steps")
    args.comms_map_mode = str(args.comms_map_mode).replace("-", "_")
    if args.comms_map_mode not in {"global", "per_agent"}:
        raise SystemExit("--comms-map-mode must be one of: global, per_agent")

    out_dir = Path(args.out)
    print(f" Output:        {_display_path(out_dir)}")
    print(f" Approach:      {args.approach}")
    print(f" Steps:         {args.steps}")
    print(f" Grid:          {args.grid_size}x{args.grid_size}")
    print(f" Comms dropout: {args.comms_dropout}")
    print(f" Comms mode:    {args.comms_dropout_mode}")
    print(f" Comms maps:    {args.comms_map_mode}")
    print(f" Terrain:       {args.terrain_source}")
    print("-" * 60)

    scenario_kwargs = {
        "max_steps":        args.steps,
        "x_semidim":        args.x_semidim,
        "y_semidim":        args.y_semidim,
        "fire_grid_size":   args.grid_size,
        "comms_dropout":    args.comms_dropout,
        "comms_dropout_mode": args.comms_dropout_mode,
        "comms_map_mode": args.comms_map_mode,
        "comms_dropout_min_steps": args.comms_dropout_min_steps,
        "comms_dropout_max_steps": args.comms_dropout_max_steps,
        "terrain_source":   args.terrain_source,
        "terrain_place":    args.terrain_place,
        "terrain_cache_dir": args.terrain_cache_dir,
        "terrain_cache_path": args.terrain_cache_path,
        "drone_flight_levels_m": tuple(args.drone_flight_levels_m),
        "drone_camera_fov_deg": args.drone_camera_fov_deg,
        "drone_safety_clearance_m": args.drone_safety_clearance_m,
        "sim_step_seconds": args.sim_step_seconds,
        "drone_speed_mps": args.drone_speed_mps,
        "drone_u_multiplier": args.drone_u_multiplier,
        "ground_speed_mps": args.ground_speed_mps,
        "ground_accel_mps2": args.ground_accel_mps2,
    }
    if args.ground_u_multiplier is not None:
        scenario_kwargs["ground_u_multiplier"] = args.ground_u_multiplier

    # A trained policy must be compared in the environment it learned in.
    # New checkpoints carry a project-owned manifest beside models/. Apply its
    # scenario configuration to HAPPO and all baselines so comparisons remain
    # apples-to-apples. Episode length and dropout stay evaluation controls.
    happo_checkpoint = None
    training_manifest = None
    try:
        from agents.happo_checkpoint import load_training_manifest, merge_training_scenario
        from agents.happo_policy import find_latest_happo_checkpoint

        happo_checkpoint = _resolve_happo_checkpoint(args.happo_checkpoint)
        if happo_checkpoint is None:
            happo_checkpoint = find_latest_happo_checkpoint().resolve()
        training_manifest = load_training_manifest(happo_checkpoint)
        if args.skip_happo_manifest or args.ignore_happo_env:
            print(" HAPPO env:     manifest merge skipped (using explicit CLI sensors)")
            training_manifest = None
        elif training_manifest is not None:
            scenario_kwargs = merge_training_scenario(
                scenario_kwargs,
                training_manifest,
                max_steps=args.steps,
                comms_dropout=args.comms_dropout,
            )
            scenario_kwargs.update({
                "comms_dropout_mode": args.comms_dropout_mode,
                "comms_map_mode": args.comms_map_mode,
                "comms_dropout_min_steps": args.comms_dropout_min_steps,
                "comms_dropout_max_steps": args.comms_dropout_max_steps,
            })
            print(f" HAPPO env:     restored from {happo_checkpoint.parent.name}")
        else:
            print(" HAPPO env:     legacy checkpoint (no saved training config)")
    except (ImportError, FileNotFoundError):
        training_manifest = None

    if args.ground_confirmation_range_m is not None:
        scenario_kwargs["ground_confirmation_range_m"] = max(args.ground_confirmation_range_m, 0.0)
    if args.drone_min_footprint_radius_m is not None:
        scenario_kwargs.pop("drone_min_footprint", None)
        scenario_kwargs["drone_min_footprint_m"] = max(
            args.drone_min_footprint_radius_m, 0.0,
        )
    if args.ground_min_confirm_radius_m is not None:
        scenario_kwargs.pop("ground_confirm_min", None)
        scenario_kwargs["ground_confirm_min_m"] = max(
            args.ground_min_confirm_radius_m, 0.0,
        )
    if args.joint_survivor_diagnostic:
        scenario_kwargs.update({
            "n_drones": 3,
            "n_ground": max(int(args.joint_diagnostic_ugvs), 1),
            "n_survivors": 5,
            "known_survivors_at_reset": False,
            "drone_can_confirm": False,
            "comms_dropout": args.comms_dropout,
            "comms_dropout_mode": args.comms_dropout_mode,
            "comms_map_mode": args.comms_map_mode,
            "comms_dropout_min_steps": args.comms_dropout_min_steps,
            "comms_dropout_max_steps": args.comms_dropout_max_steps,
            "ugv_target_assignment_mode": "greedy_sticky",
            "ugv_assigned_target_obs_only": False,
            "survivor_assignment_obs": True,
            "ugv_planner_hint": "global_astar",
            "ugv_dense_reward_mode": "planner_follow",
            "ugv_global_planner_lookahead_m": 20.0,
            "disable_fire": True,
        })
    if args.joint_schema_ugv_diagnostic:
        scenario_kwargs.update({
            "n_drones": 0,
            "n_ground": 2,
            "n_survivors": 5,
            "obs_schema_n_drones": 3,
            "obs_schema_n_ground": 2,
            "obs_schema_n_survivors": 5,
            "known_survivors_at_reset": False,
            "delayed_survivor_knowledge": True,
            "survivor_reveal_schedule": "stratified_uniform",
            "survivor_reveal_initial_count": 1,
            "survivor_reveal_start_step": 10,
            "survivor_reveal_end_step": 180,
            "drone_can_confirm": False,
            "comms_dropout": args.comms_dropout,
            "comms_dropout_mode": args.comms_dropout_mode,
            "comms_map_mode": args.comms_map_mode,
            "comms_dropout_min_steps": args.comms_dropout_min_steps,
            "comms_dropout_max_steps": args.comms_dropout_max_steps,
            "ugv_target_assignment_mode": "greedy_sticky",
            "ugv_assigned_target_obs_only": False,
            "survivor_assignment_obs": True,
            "ugv_planner_hint": "global_astar",
            "ugv_dense_reward_mode": "planner_follow",
            "ugv_global_planner_lookahead_m": 20.0,
            "ugv_zero_uav_search_observations": True,
            "coverage_obs_grid": 6,
            "local_coverage_obs_grid": 9,
            "local_coverage_obs_radius_m": 150.0,
            "uav_confidence_obs_grid": 6,
            "uav_frontier_obs": True,
            "uav_frontier_obs_radius_m": 60.0,
            "uav_frontier_mode": "local_global",
            "uav_frontier_source": "confidence",
            "uav_cleanup_target_obs": True,
            "disable_fire": True,
        })
    if args.enable_fire:
        scenario_kwargs["disable_fire"] = False
    if args.ugv_target_assignment_mode is not None:
        scenario_kwargs["ugv_target_assignment_mode"] = args.ugv_target_assignment_mode.replace("-", "_")
    if args.ugv_planner_fire_mode is not None:
        scenario_kwargs["ugv_planner_fire_mode"] = args.ugv_planner_fire_mode.replace("-", "_")
    if args.ugv_planner_fire_replan_policy is not None:
        scenario_kwargs["ugv_planner_fire_replan_policy"] = (
            args.ugv_planner_fire_replan_policy.replace("-", "_")
        )
    if args.ugv_planner_fire_replan_interval_steps is not None:
        if args.ugv_planner_fire_replan_interval_steps < 1:
            raise SystemExit("--ugv-planner-fire-replan-interval-steps must be positive")
        scenario_kwargs["ugv_planner_fire_replan_interval_steps"] = int(
            args.ugv_planner_fire_replan_interval_steps
        )
    if args.ugv_global_planner_heuristic is not None:
        scenario_kwargs["ugv_global_planner_heuristic"] = (
            args.ugv_global_planner_heuristic.replace("-", "_")
        )
    for arg_name in (
        "ugv_planner_fire_cost",
        "ugv_planner_smoke_cost",
        "ugv_planner_smolder_cost",
        "ugv_planner_fire_buffer_m",
        "ugv_planner_fire_buffer_cost",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            if value < 0.0:
                raise SystemExit(f"--{arg_name.replace('_', '-')} must be nonnegative")
            scenario_kwargs[arg_name] = float(value)
    if args.ugv_planner_fire_block_threshold is not None:
        if not 0.0 <= args.ugv_planner_fire_block_threshold <= 1.0:
            raise SystemExit("--ugv-planner-fire-block-threshold must be in [0, 1]")
        scenario_kwargs["ugv_planner_fire_block_threshold"] = float(
            args.ugv_planner_fire_block_threshold
        )
    if args.ugv_planner_land_cover_costs is not None:
        if len(args.ugv_planner_land_cover_costs) not in (5, 6):
            raise SystemExit(
                "--ugv-planner-land-cover-costs must provide 5 or 6 values: "
                "road open brush forest rock [water]"
            )
        if any(value < 0.0 for value in args.ugv_planner_land_cover_costs):
            raise SystemExit("--ugv-planner-land-cover-costs values must be nonnegative")
        scenario_kwargs["ugv_planner_land_cover_costs"] = tuple(
            float(v) for v in args.ugv_planner_land_cover_costs
        )
    cv_options = None
    if args.enable_cv:
        if scenario_kwargs.get("terrain_cache_path") is None:
            raise SystemExit("--enable-cv currently requires --terrain-cache-path so the NAIP bbox is unambiguous.")
        cv_options = {
            "enabled": True,
            "output_dir": args.cv_out_dir or str(out_dir / "cv_frames"),
            "save_images_every": args.cv_save_images_every,
            "naip_image_path": args.cv_naip_image_path,
            "target_gsd_m": args.cv_target_gsd_m if args.cv_target_gsd_m > 0 else None,
            "lazy_tile_cache": not args.cv_build_full_naip,
            "naip_size": args.cv_naip_size,
            "tiled_naip": not args.cv_single_naip_export,
            "tile_size": args.cv_tile_size,
            "image_size": args.cv_image_size,
            "render_wildfire_effects": not args.cv_disable_wildfire_effects,
            "human_asset_path": args.cv_human_asset,
            "human_assets_dir": args.cv_human_assets_dir or None,
            "human_asset_list_path": args.cv_human_asset_list or None,
            "survivor_preview_altitude_m": args.cv_preview_altitude_m,
            "detection_probability": args.cv_detection_probability,
            "pixel_noise_std": args.cv_pixel_noise_std,
            "detector_backend": args.cv_detector,
            "person_model": args.cv_person_model,
            "person_conf": args.cv_person_conf,
            "person_imgsz": args.cv_person_imgsz,
            "person_tiled": not args.cv_person_no_tiling,
            "person_tile_grid": args.cv_person_tile_grid,
            "person_augment": args.cv_augment,
            "adaptive_conf": args.cv_adaptive_conf,
            "enable_tracking": args.cv_tracking,
            "tracking_min_hits": args.cv_tracking_min_hits,
            "detection_mode": args.detection_mode,
            "thermal_detector": args.thermal_detector,
            "camera_tilt_deg": args.cv_camera_tilt,
        }

    for name in _selected_baselines(args.approach):
        cls = BASELINES[name]
        def make_policy(env, _cls=cls):
            return _cls() if _cls is RandomActionPolicy else _cls(env)
        run_cv_options = None
        if cv_options is not None:
            run_cv_options = dict(cv_options)
            if args.cv_out_dir is None:
                run_cv_options["output_dir"] = str(out_dir / f"{name}_cv")
        t0 = time.time()
        path = export_trajectory(
            strategy_name=name,
            make_policy=make_policy,
            output_path=out_dir / f"{name}.json",
            n_steps=args.steps,
            seed=args.seed,
            scenario_kwargs=scenario_kwargs,
            cv_options=run_cv_options,
            frame_stride=args.frame_stride,
        )
        print(f"  ✓ {name:22s} → {_display_path(path)}  ({time.time() - t0:.1f}s)")

    if args.approach in ("all", "happo"):
        # Trained HAPPO policy pulls the most recent checkpoint from results/harl_runs/.
        try:
            from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
            ckpt = happo_checkpoint or find_latest_happo_checkpoint().resolve()
            try:
                ckpt_disp = ckpt.relative_to(ROOT.resolve())
            except ValueError:
                ckpt_disp = ckpt
            print(f"  · HAPPO checkpoint: {ckpt_disp}")
            # Build the policy before export_trajectory creates and resets the
            # eval env. Policy construction creates a temporary VMAS env to
            # infer spaces; doing that after reset shifts torch RNG and changes
            # probabilistic survivor detections for the same seed.
            happo_policy = HappoPolicy.from_checkpoint(ckpt)
            def make_happo(env, _policy=happo_policy):
                _policy.reset()
                return _policy
            run_cv_options = None
            if cv_options is not None:
                run_cv_options = dict(cv_options)
                if args.cv_out_dir is None:
                    run_cv_options["output_dir"] = str(out_dir / "happo_trained_cv")
            t0 = time.time()
            path = export_trajectory(
                strategy_name="happo_trained",
                make_policy=make_happo,
                output_path=out_dir / "happo_trained.json",
                n_steps=args.steps,
                seed=args.seed,
                scenario_kwargs=scenario_kwargs,
                cv_options=run_cv_options,
                frame_stride=args.frame_stride,
            )
            print(f"  ✓ {'happo_trained':22s} → {_display_path(path)}  ({time.time() - t0:.1f}s)")
        except ImportError as e:
            print(f"  ⚠ HAPPO export skipped — missing dependency ({e})")
            print(f"    Install HARL deps or activate the correct venv.")
        except FileNotFoundError as e:
            print(f"  ⚠ HAPPO export skipped — no checkpoint found ({e})")
            print(f"    Run `python scripts/train_happo_smoke.py` first to produce a checkpoint.")
        except RuntimeError as e:
            print(f"  ⚠ HAPPO export skipped — checkpoint incompatible with current scenario ({e})")
            print(f"    The saved policy's observation/action shapes no longer match the env "
                  f"(it predates a scenario change). Retrain with `python scripts/train_happo_smoke.py`.")

    print("-" * 60)
    print(f" Done. Serve with: python -m http.server -d web")


if __name__ == "__main__":
    main()
