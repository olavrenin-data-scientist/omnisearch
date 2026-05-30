"""
Comms-dropout ablation sweep across three algorithms × multiple seeds.

The capstone's central experiment: as comms_dropout grows from 0.0 → 0.8,
how does each algorithm's training reward degrade? Compares:

    MAPPO  — centralized critic + per-group actors          (BenchMARL)
    IPPO   — fully decentralized (per-group actor + critic)  (BenchMARL)
    HAPPO  — sequential update + monotonic improvement       (HARL)

Each (algo, dropout) cell is repeated across N seeds. Aggregate output:

  - results/comms_dropout_sweep_<ts>.json — per-cell flat list of every
    seed's result, plus summary (mean ± std) and Mann-Whitney U tests.
  - Console summary table and significance table.

Note on metric comparability: BenchMARL reports `experiment.mean_return`
(team reward); HARL reports per-episode mean reward. These are NOT
directly comparable across libraries in absolute terms — but WITHIN one
algorithm they're consistent across dropout levels, so the per-algorithm
degradation curves are meaningful. Cross-algo comparison should use the
``compare_baselines.py`` harness which evaluates trained policies on
shared mission metrics (TODO once checkpoint loading lands).

Run from repo root:
    python scripts/comms_dropout_sweep.py                    # default smoke
    python scripts/comms_dropout_sweep.py --seeds 5
    python scripts/comms_dropout_sweep.py --algos mappo happo --seeds 5
    python scripts/comms_dropout_sweep.py --research         # bigger budget
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DROPOUTS = [0.0, 0.2, 0.5, 0.8]


# ----------------------------------------------------------------------
# Per-algorithm training functions
# ----------------------------------------------------------------------
def _run_benchmarl(algo_name: str, seed: int, comms_dropout: float,
                   frames_per_iter: int, iters: int) -> dict:
    from benchmarl.algorithms import IppoConfig, MappoConfig
    from benchmarl.experiment import Experiment
    from benchmarl.models.mlp import MlpConfig
    from agents.wildfire_task import make_wildfire_task
    from agents.train_helpers import smoke_config

    cfg = smoke_config(iters=iters, frames_per_batch=frames_per_iter)
    algo_cfg = {"mappo": MappoConfig, "ippo": IppoConfig}[algo_name].get_from_yaml()
    task = make_wildfire_task(comms_dropout=comms_dropout, max_steps=150)

    t0 = time.time()
    exp = Experiment(
        task             = task,
        algorithm_config = algo_cfg,
        model_config     = MlpConfig.get_from_yaml(),
        seed             = seed,
        config           = cfg,
    )
    exp.run()
    return {"metric": float(exp.mean_return), "wall_sec": round(time.time() - t0, 2)}


def _run_happo(seed: int, comms_dropout: float, num_env_steps: int) -> dict:
    from agents.harl_runner import train_happo
    r = train_happo(
        seed              = seed,
        num_env_steps     = num_env_steps,
        comms_dropout     = comms_dropout,
        n_rollout_threads = 8,
        exp_name          = f"happo_d{int(comms_dropout*100)}_s{seed}",
    )
    return {"metric": r["mean_episode_reward"], "wall_sec": r["wall_sec"]}


def run_cell(algo: str, seed: int, dropout: float, budgets: dict) -> dict:
    if algo in ("mappo", "ippo"):
        return _run_benchmarl(algo, seed, dropout,
                              frames_per_iter=budgets["frames_per_iter"],
                              iters=budgets["iters"])
    if algo == "happo":
        return _run_happo(seed, dropout, num_env_steps=budgets["num_env_steps"])
    raise ValueError(f"Unknown algo: {algo}")


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------
def mann_whitney(x: list, y: list) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test. Returns (U, p). NaN if too few samples."""
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(u), float(p)


def sig_marker(p: float) -> str:
    if p != p:    return ""            # NaN
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    type=int, default=3)
    parser.add_argument("--algos",    nargs="+", default=["mappo", "ippo", "happo"])
    parser.add_argument("--dropouts", nargs="+", type=float, default=DROPOUTS)
    parser.add_argument("--research", action="store_true",
                        help="Bigger budget per cell (slower)")
    args = parser.parse_args()

    if args.research:
        budgets = {"frames_per_iter": 6_000, "iters": 50, "num_env_steps": 80_000}
    else:
        budgets = {"frames_per_iter": 2_000, "iters": 3,  "num_env_steps": 8_000}

    n_cells = len(args.algos) * len(args.dropouts) * args.seeds
    print("=" * 78)
    print(f" OmniSearch — comms-dropout sweep ({'RESEARCH' if args.research else 'SMOKE'})")
    print(f" algos:    {args.algos}")
    print(f" dropouts: {args.dropouts}")
    print(f" seeds:    {args.seeds}   (total cells: {n_cells})")
    print("=" * 78)

    cells: list[dict] = []
    t_total = time.time()

    for algo in args.algos:
        for dropout in args.dropouts:
            print(f"\n>>> {algo.upper():6s} comms_dropout={dropout}")
            for s in range(args.seeds):
                r = run_cell(algo, s, dropout, budgets)
                r.update(algo=algo, comms_dropout=dropout, seed=s, **budgets)
                cells.append(r)
                print(f"  seed {s}: metric={r['metric']:+6.2f}  ({r['wall_sec']}s)")

    total_wall = time.time() - t_total

    # Aggregate: mean ± std per (algo, dropout)
    summary = []
    for algo in args.algos:
        for dropout in args.dropouts:
            metrics = [c["metric"] for c in cells
                       if c["algo"] == algo and c["comms_dropout"] == dropout]
            summary.append({
                "algo":    algo,
                "dropout": dropout,
                "n":       len(metrics),
                "mean":    float(mean(metrics)) if metrics else float("nan"),
                "std":     float(stdev(metrics)) if len(metrics) > 1 else 0.0,
                "values":  metrics,
            })

    # Mann-Whitney U: within each algo, dropout=base vs each higher dropout
    base_dropout = args.dropouts[0]
    sig_results: list[dict] = []
    for algo in args.algos:
        base_vals = next(s["values"] for s in summary
                         if s["algo"] == algo and s["dropout"] == base_dropout)
        for dropout in args.dropouts[1:]:
            other_vals = next(s["values"] for s in summary
                              if s["algo"] == algo and s["dropout"] == dropout)
            u, p = mann_whitney(base_vals, other_vals)
            sig_results.append({
                "algo":           algo,
                "base_dropout":   base_dropout,
                "vs_dropout":     dropout,
                "U":              u,
                "p":              p,
                "significant_05": (p == p) and (p < 0.05),
            })

    # Print summary table
    print(f"\n{'=' * 78}")
    print(f" SUMMARY  (mean ± std across {args.seeds} seeds)")
    print(f"{'=' * 78}")
    print(f" {'algo':6s} " + " ".join(f"{'d='+str(d):>14s}" for d in args.dropouts))
    for algo in args.algos:
        row = f" {algo:6s} "
        for dropout in args.dropouts:
            s = next(x for x in summary if x["algo"] == algo and x["dropout"] == dropout)
            row += f"   {s['mean']:+5.2f} ± {s['std']:.2f}"
        print(row)

    # Print significance table
    print(f"\n{'=' * 78}")
    print(f" MANN-WHITNEY U  (within algo: d={base_dropout} vs higher dropouts)")
    print(f" * p<0.05    ** p<0.01    *** p<0.001")
    print(f"{'=' * 78}")
    print(f" {'algo':6s}  {'pair':>14s}    {'U':>6s}    {'p':>6s}   sig")
    for r in sig_results:
        pair = f"d={r['base_dropout']} vs d={r['vs_dropout']}"
        p_str = f"{r['p']:.3f}" if r["p"] == r["p"] else "  nan"
        u_str = f"{r['U']:.1f}"  if r["U"] == r["U"] else "  nan"
        print(f" {r['algo']:6s}  {pair:>14s}    {u_str:>6s}    {p_str:>6s}   {sig_marker(r['p'])}")

    # Persist
    out = ROOT / "results" / f"comms_dropout_sweep_{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {
            "params": {
                "seeds":    args.seeds,
                "algos":    args.algos,
                "dropouts": args.dropouts,
                "budgets":  budgets,
                "research": args.research,
            },
            "cells":              cells,
            "summary":            summary,
            "significance_tests": sig_results,
            "total_wall_sec":     round(total_wall, 1),
        },
        indent=2,
    ))
    print(f"\n Total wall: {total_wall:.1f}s")
    print(f" Wrote:      {out.relative_to(ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
