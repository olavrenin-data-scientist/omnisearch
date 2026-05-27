"""
Export one trajectory per baseline strategy into web/trajectories/ as JSON.

The web viewer (`web/index.html`) auto-discovers any *.json in that folder
and lets the user switch between them. Run this whenever you want fresh
trajectories shown in the viewer:

    python scripts/export_trajectories.py
    python scripts/export_trajectories.py --seed 7 --steps 300
    python scripts/export_trajectories.py --seed 7 --steps 500 --grid-size 32
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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default=str(ROOT / "web" / "trajectories"))
    p.add_argument(
        "--grid-size",
        type=int,
        default=16,
        help="Fire/terrain grid resolution. Try 32 for a finer map.",
    )
    args = p.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.grid_size < 8:
        raise SystemExit("--grid-size must be at least 8; try 16, 32, or 64")

    out_dir = Path(args.out)
    print(f" Output: {_display_path(out_dir)}")
    print(f" Steps:  {args.steps}")
    print(f" Grid:   {args.grid_size}x{args.grid_size}")
    print("-" * 60)

    scenario_kwargs = {
        "max_steps": args.steps,
        "fire_grid_size": args.grid_size,
    }

    for name, cls in BASELINES.items():
        # export_trajectory builds the env, then calls make_policy(env) so
        # baselines see the env they'll actually be stepped on.
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

    print("-" * 60)
    print(f" Done. Serve with: python -m http.server -d web")


if __name__ == "__main__":
    main()
