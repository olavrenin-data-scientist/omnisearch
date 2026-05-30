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
    }

    for name, cls in BASELINES.items():
        def make_policy(env, _cls=cls):
            return _cls() if _cls is RandomPolicy else _cls(env)
        t0 = time.time()
        path = export_trajectory(
            strategy_name=name,
            make_policy=make_policy,
            output_path=out_dir / f"{name}.json",
            n_steps=args.steps,
            seed=args.seed,
            scenario_kwargs=scenario_kwargs,
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
        t0 = time.time()
        path = export_trajectory(
            strategy_name="happo_trained",
            make_policy=make_happo,
            output_path=out_dir / "happo_trained.json",
            n_steps=args.steps,
            seed=args.seed,
            scenario_kwargs=scenario_kwargs,
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
