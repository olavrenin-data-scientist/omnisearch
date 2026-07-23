"""Diagnostic: does survivor detection work at floor 0 across terrain scales?

The detection geometry (drone footprint, ground confirm radius) is expressed in
meters and converted to sim units via ``sim_units_per_meter``. On the huge
default park terrain that conversion is ~9e-05, so at floor 0 a survivor is a
sub-0.1%-of-map pinpoint and recall collapses to 0. Smaller terrains have a much
larger conversion, so the *real* sensor footprint becomes usable WITHOUT any
detection floor.

This runs the hand-coded experts (which represent an upper bound on what a
trained policy could plausibly achieve) at floor 0 on each cached terrain.

    python -u scripts/diag_terrain_floor0.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmas

from envs.wildfire_search import WildfireSearchScenario
from agents.baselines import BASELINES
from evaluation.mission_metrics import EpisodeRecorder

CACHE = ROOT / "data" / "terrain_cache"

TERRAINS = {
    "park (default, ~22km)": CACHE / "malibu_creek_state_park_california_128.npz",
    "small (~3km)":          CACHE / "malibu_creek_small_128.npz",
    "1km":                   CACHE / "malibu_creek_1km_128.npz",
}

STRATEGIES = ["lawnmower", "nearest_candidate"]


def run_one(cache_path: Path, strategy: str, seed: int, steps: int,
            confirm_m: float | None) -> float:
    kwargs = dict(
        terrain_cache_path=str(cache_path),
        drone_min_footprint=0.0,
        ground_confirm_min=0.0,
    )
    if confirm_m is not None:
        kwargs["ground_confirmation_range_m"] = confirm_m
    env = WildfireSearchScenario.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=2, device="cpu",
        continuous_actions=True, seed=seed,
        **kwargs,
    )
    env.reset()
    policy = BASELINES[strategy](env)
    rec = EpisodeRecorder(env.scenario, env_index=0)
    for _ in range(steps):
        env.step(policy(env))
        rec.step()
        if env.scenario.done()[0].item():
            break
    return rec.finalize().survivor_recall


def main() -> None:
    seeds = 3
    steps = 400
    print(f"\nFloor-0 detection diagnostic ({seeds} seeds x {steps} steps)\n")
    header = f"{'terrain':24s} {'confirm_m':>9s}  " + "  ".join(
        f"{s:>17s}" for s in STRATEGIES
    )
    print(header)
    print("-" * len(header))
    t0 = time.time()
    for confirm_m in (None, 40.0):
        label_cm = "default(10)" if confirm_m is None else f"{confirm_m:.0f}"
        for name, path in TERRAINS.items():
            if not path.exists():
                print(f"{name:24s} {label_cm:>9s}  MISSING {path.name}")
                continue
            cells = []
            for strat in STRATEGIES:
                recalls = [run_one(path, strat, s, steps, confirm_m)
                           for s in range(seeds)]
                cells.append(f"{mean(recalls):>17.2f}")
            print(f"{name:24s} {label_cm:>9s}  " + "  ".join(cells))
        print("-" * len(header))
    print(f"\nWall time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
