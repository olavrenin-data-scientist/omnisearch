"""
Export one trajectory per baseline strategy into web/trajectories/ as JSON.

The web viewer (`web/index.html`) auto-discovers any *.json in that folder
and lets the user switch between them. Run this whenever you want fresh
trajectories shown in the viewer:

    python scripts/export_trajectories.py
    python scripts/export_trajectories.py --seed 7 --steps 500
    python scripts/export_trajectories.py --seed 7 --steps 500 --grid-size 128
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

from envs.wildfire_search import WildfireSearchScenario
from agents.baselines import BASELINES, RandomPolicy
from evaluation.trajectory_export import export_trajectory


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default=str(ROOT / "web" / "trajectories"))
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
        "--drone-flight-levels-m",
        nargs="+",
        type=float,
        default=(20.0, 35.0, 50.0),
        help="Drone AGL flight levels in meters.",
    )
    p.add_argument(
        "--drone-camera-fov-deg",
        type=float,
        default=90.0,
        help="Downward drone camera field of view in degrees.",
    )
    p.add_argument(
        "--drone-safety-clearance-m",
        type=float,
        default=3.0,
        help="Minimum aerial clearance above terrain obstacles in meters.",
    )
    p.add_argument(
        "--sim-step-seconds",
        type=float,
        default=2.0,
        help="Physical duration represented by one simulation step.",
    )
    p.add_argument(
        "--drone-speed-mps",
        type=float,
        default=10.0,
        help="Drone horizontal speed cap in meters per second.",
    )
    p.add_argument(
        "--drone-u-multiplier",
        type=float,
        default=2.25,
        help="VMAS drone action-to-acceleration multiplier.",
    )
    p.add_argument(
        "--ground-speed-mps",
        type=float,
        default=1.6,
        help="Nominal UGV horizontal speed cap on easy terrain in meters per second.",
    )
    p.add_argument(
        "--ground-accel-mps2",
        type=float,
        default=2.0,
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
    args = p.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.grid_size < 8:
        raise SystemExit("--grid-size must be at least 8; try 16, 32, or 64")

    out_dir = Path(args.out)
    print(f" Output:        {_display_path(out_dir)}")
    print(f" Steps:         {args.steps}")
    print(f" Grid:          {args.grid_size}x{args.grid_size}")
    print(f" Comms dropout: {args.comms_dropout}")
    print(f" Terrain:       {args.terrain_source}")
    print("-" * 60)

    scenario_kwargs = {
        "max_steps":        args.steps,
        "fire_grid_size":   args.grid_size,
        "comms_dropout":    args.comms_dropout,
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
    cv_options = None
    if args.enable_cv:
        if args.terrain_cache_path is None:
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
            "human_asset_path": args.cv_human_asset,
            "human_assets_dir": args.cv_human_assets_dir or None,
            "human_asset_list_path": args.cv_human_asset_list or None,
            "survivor_preview_altitude_m": args.cv_preview_altitude_m,
            "detection_probability": args.cv_detection_probability,
            "pixel_noise_std": args.cv_pixel_noise_std,
        }

    for name, cls in BASELINES.items():
        def make_policy(env, _cls=cls):
            return _cls() if _cls is RandomPolicy else _cls(env)
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
        )
        print(f"  ✓ {name:22s} → {_display_path(path)}  ({time.time() - t0:.1f}s)")

    # Trained HAPPO policy — pulls the most recent checkpoint from results/harl_runs/
    try:
        from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
        ckpt = find_latest_happo_checkpoint().resolve()
        try:
            ckpt_disp = ckpt.relative_to(ROOT.resolve())
        except ValueError:
            ckpt_disp = ckpt
        print(f"  · HAPPO checkpoint: {ckpt_disp}")
        def make_happo(env, _ckpt=ckpt):
            return HappoPolicy.from_checkpoint(_ckpt)
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
        )
        print(f"  ✓ {'happo_trained':22s} → {_display_path(path)}  ({time.time() - t0:.1f}s)")
    except ImportError as e:
        print(f"  ⚠ HAPPO export skipped — missing dependency ({e})")
        print(f"    Install HARL deps or activate the correct venv.")
    except FileNotFoundError as e:
        print(f"  ⚠ HAPPO export skipped — no checkpoint found ({e})")
        print(f"    Run `python scripts/train_happo_smoke.py` first to produce a checkpoint.")

    print("-" * 60)
    print(f" Done. Serve with: python -m http.server -d web")


if __name__ == "__main__":
    main()
