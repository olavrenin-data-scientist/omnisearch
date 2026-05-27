"""
Export one trajectory per baseline strategy into web/trajectories/ as JSON.

The web viewer (`web/index.html`) auto-discovers any *.json in that folder
and lets the user switch between them. Run this whenever you want fresh
trajectories shown in the viewer:

    python scripts/export_trajectories.py
    python scripts/export_trajectories.py --seed 7 --steps 300
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default=str(ROOT / "web" / "trajectories"))
    args = p.parse_args()

    out_dir = Path(args.out)
    print(f" Output: {out_dir.relative_to(ROOT)}")
    print("-" * 60)

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
        )
        print(f"  ✓ {name:22s} → {path.relative_to(ROOT)}  ({time.time() - t0:.1f}s)")

    print("-" * 60)
    print(f" Done. Serve with: python -m http.server -d web")


if __name__ == "__main__":
    main()
