"""
Compare baseline coordination strategies on mission-level metrics.

Runs each strategy in `agents.baselines.BASELINES` across multiple seeds,
collects MissionMetrics per episode, aggregates (mean ± std), and prints a
comparison table. Optionally writes JSON to ``results/`` for later plotting.

This is the harness the capstone uses to answer the plan's central
question: does HAPPO actually beat hand-coded heuristics? Plug a trained
policy in via `--strategy trained:path/to/checkpoint.pt` (TODO — checkpoint
loader to be added once a research-budget training run is committed).

Run from repo root:
    python scripts/compare_baselines.py                       # all baselines, 3 seeds
    python scripts/compare_baselines.py --seeds 5 --steps 250
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

import vmas

from envs.wildfire_search import WildfireSearchScenario
from agents.baselines import BASELINES, RandomPolicy
from evaluation.mission_metrics import EpisodeRecorder, MissionMetrics


def run_one(strategy_name: str, seed: int, steps: int) -> MissionMetrics:
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=2, device="cpu",
        continuous_actions=True, seed=seed,
    )
    env.reset()
    cls = BASELINES[strategy_name]
    policy = cls() if cls is RandomPolicy else cls(env)

    rec = EpisodeRecorder(env.scenario, env_index=0)
    for _ in range(steps):
        env.step(policy(env))
        rec.step()
        if env.scenario.done()[0].item():
            break
    return rec.finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--strategies", nargs="+", default=list(BASELINES.keys()))
    args = p.parse_args()

    t0 = time.time()
    rows = []
    print(f"\n{'='*78}")
    print(f" OmniSearch — Baseline Comparison  "
          f"({args.seeds} seeds × {args.steps} steps)")
    print(f"{'='*78}")
    print(f"\n{'strategy':22s} {'recall':>7} {'ttv':>8} {'haz':>5} {'ugv_dist':>9}")
    print("-" * 78)

    for name in args.strategies:
        per_seed: list[MissionMetrics] = []
        for s in range(args.seeds):
            per_seed.append(run_one(name, seed=s, steps=args.steps))

        recalls   = [m.survivor_recall for m in per_seed]
        ttvs      = [m.time_to_verification for m in per_seed
                     if m.time_to_verification == m.time_to_verification]
        hazards   = [m.hazard_exposure for m in per_seed]
        travels   = [m.ugv_travel_cost for m in per_seed]

        recall_m = mean(recalls)
        ttv_m    = mean(ttvs) if ttvs else float("nan")
        haz_m    = mean(hazards)
        trav_m   = mean(travels)
        ttv_s    = f"{ttv_m:>8.1f}" if ttv_m == ttv_m else "     nan"
        print(f"{name:22s} {recall_m:>7.2f} {ttv_s} {haz_m:>5.0f} {trav_m:>9.2f}")

        rows.append({
            "strategy":             name,
            "seeds":                args.seeds,
            "steps":                args.steps,
            "survivor_recall_mean": recall_m,
            "survivor_recall_std":  stdev(recalls) if len(recalls) > 1 else 0.0,
            "time_to_verification": ttv_m,
            "hazard_exposure_mean": haz_m,
            "ugv_travel_cost_mean": trav_m,
            "per_seed": [m.as_dict() for m in per_seed],
        })

    out_path = ROOT / "results" / f"baseline_comparison_{int(time.time())}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\n{'-' * 78}")
    print(f" Wall time: {time.time() - t0:.1f}s")
    print(f" Wrote:     {out_path.relative_to(ROOT)}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
