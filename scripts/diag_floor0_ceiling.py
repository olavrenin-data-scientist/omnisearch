"""Diagnostic: achievable floor-0 recall ceiling on the 1km terrain.

Pushes the *physical* (non-floor) sensor levers to see how high expert recall
can go at floor 0: wider drone FOV + higher flight = bigger scout footprint,
longer episodes = more confirm time. This sets the signal ceiling a trained
HAPPO policy could chase.

    python -u scripts/diag_floor0_ceiling.py
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
TERRAIN = CACHE / "malibu_creek_1km_128.npz"
STRATEGIES = ["lawnmower", "nearest_candidate"]

CONFIGS = {
    "baseline(fov90,50m)":  dict(),
    "fov120,fly80":         dict(drone_camera_fov_deg=120.0,
                                 drone_flight_levels_m=(40.0, 60.0, 80.0)),
    "fov140,fly100,c30":    dict(drone_camera_fov_deg=140.0,
                                 drone_flight_levels_m=(50.0, 80.0, 100.0),
                                 ground_confirmation_range_m=30.0),
}


def run_one(strategy: str, seed: int, steps: int, extra: dict) -> float:
    kwargs = dict(
        terrain_cache_path=str(TERRAIN),
        drone_min_footprint=0.0,
        ground_confirm_min=0.0,
    )
    kwargs.update(extra)
    env = vmas.make_env(
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
    steps = 600
    print(f"\nFloor-0 ceiling on 1km terrain ({seeds} seeds x {steps} steps)\n")
    header = f"{'config':22s}  " + "  ".join(f"{s:>17s}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    t0 = time.time()
    for label, extra in CONFIGS.items():
        cells = []
        for strat in STRATEGIES:
            recalls = [run_one(strat, s, steps, extra) for s in range(seeds)]
            cells.append(f"{mean(recalls):>17.2f}")
        print(f"{label:22s}  " + "  ".join(cells))
    print("-" * len(header))
    print(f"\nWall time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
