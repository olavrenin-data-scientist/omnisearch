"""
Comms-dropout ablation sweep — the capstone's headline experiment.

Sweeps ``comms_dropout`` ∈ {0.0, 0.2, 0.5, 0.8} for two algorithms:
  - MAPPO (centralized critic, decentralized actors)
  - IPPO  (fully decentralized — no centralized critic)

Hypothesis: as comms_dropout grows, IPPO degrades faster than MAPPO because
the centralized critic in MAPPO can still see the global state, while IPPO
loses both the neighbor observations *and* has no global signal to fall
back on.

Default budget is the smoke budget (~6k frames per cell, ~50s wall total).
Pass --research for a credible budget (~400k frames per cell — hours on CPU).

Run from repo root:
    python scripts/comms_dropout_sweep.py            # smoke
    python scripts/comms_dropout_sweep.py --research # real
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig

from agents.wildfire_task import make_wildfire_task
from agents.train_helpers import smoke_config, research_config


ALGORITHMS = {
    "mappo": MappoConfig,
    "ippo":  IppoConfig,
}
DROPOUTS = [0.0, 0.2, 0.5, 0.8]


def run_one(
    algo_name: str,
    dropout: float,
    experiment_cfg,
    seed: int = 0,
) -> dict:
    task = make_wildfire_task(comms_dropout=dropout, max_steps=150)
    experiment = Experiment(
        task=task,
        algorithm_config=ALGORITHMS[algo_name].get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        seed=seed,
        config=experiment_cfg,
    )
    t0 = time.time()
    experiment.run()
    dt = time.time() - t0

    return {
        "algo":          algo_name,
        "comms_dropout": dropout,
        "seed":          seed,
        "iters":         experiment_cfg.max_n_iters,
        "frames":        experiment_cfg.max_n_iters * experiment_cfg.on_policy_collected_frames_per_batch,
        "mean_return":   float(experiment.mean_return),
        "wall_sec":      round(dt, 2),
        "folder":        str(experiment.folder_name),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", action="store_true", help="Use research budget (slow)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_fn = research_config if args.research else smoke_config
    print("=" * 72)
    print(f" OmniSearch — comms-dropout sweep ({'RESEARCH' if args.research else 'SMOKE'} budget)")
    print(f" algos:    {list(ALGORITHMS.keys())}")
    print(f" dropouts: {DROPOUTS}")
    print(f" seed:     {args.seed}")
    print("=" * 72)

    results = []
    for algo in ALGORITHMS:
        for dropout in DROPOUTS:
            print(f"\n>>> {algo.upper()}  comms_dropout={dropout}")
            r = run_one(algo, dropout, cfg_fn(), seed=args.seed)
            print(f"    mean_return={r['mean_return']:+.3f}  ({r['wall_sec']}s)")
            results.append(r)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(" SUMMARY  (final iteration mean return per cell)")
    print("=" * 72)
    print(f" {'algo':6}  " + "  ".join(f"d={d:.1f}" for d in DROPOUTS))
    for algo in ALGORITHMS:
        cells = [r for r in results if r["algo"] == algo]
        row = "  ".join(
            f"{next(c['mean_return'] for c in cells if c['comms_dropout'] == d):+5.2f} "
            for d in DROPOUTS
        )
        print(f" {algo:6}  {row}")
    print("=" * 72)

    # Persist sweep summary to results/
    out = ROOT / "results" / f"comms_dropout_sweep_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n Sweep summary written to: {out}")


if __name__ == "__main__":
    main()
