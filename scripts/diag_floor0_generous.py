"""Find a floor-0 sensor config where experts approach ~0.9 recall.

Floor stays 0. We push only physical levers (confirm range, FOV, flight
altitude, episode length, #ground robots) to make the mission generous enough
that 90% recall becomes reachable — and thus a learnable target for HAPPO.

    python -u scripts/diag_floor0_generous.py
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

TERRAIN = str(ROOT / "data" / "terrain_cache" / "malibu_creek_1km_128.npz")
STRATEGIES = ["lawnmower", "nearest_candidate"]

# n_ground fixed at 2 (per requirement). With only 2 confirmers, the confirm
# range is the dominant lever for reaching high recall at floor 0.
CONFIGS = {
    "fov160,fly160,c120":  dict(drone_camera_fov_deg=160.0,
                                drone_flight_levels_m=(80.0, 120.0, 160.0),
                                ground_confirmation_range_m=120.0, n_ground=2),
    "fov160,fly180,c200":  dict(drone_camera_fov_deg=160.0,
                                drone_flight_levels_m=(100.0, 150.0, 180.0),
                                ground_confirmation_range_m=200.0, n_ground=2),
    "fov170,fly200,c300":  dict(drone_camera_fov_deg=170.0,
                                drone_flight_levels_m=(120.0, 180.0, 200.0),
                                ground_confirmation_range_m=300.0, n_ground=2),
    "fov170,fly200,c600":  dict(drone_camera_fov_deg=170.0,
                                drone_flight_levels_m=(120.0, 180.0, 200.0),
                                ground_confirmation_range_m=600.0, n_ground=2),
}


def run_one(strategy: str, seed: int, steps: int, extra: dict) -> float:
    kwargs = dict(
        terrain_cache_path=TERRAIN, terrain_source="real",
        drone_min_footprint=0.0, ground_confirm_min=0.0,  # floor stays 0
        max_steps=steps,
    )
    kwargs.update(extra)
    env = WildfireSearchScenario.make_env(scenario=WildfireSearchScenario(), num_envs=2, device="cpu",
                        continuous_actions=True, seed=seed, **kwargs)
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
    seeds, steps = 2, 1500
    print(f"\nFloor-0 generous-sensor ceiling on 1km ({seeds} seeds x {steps} steps)\n")
    header = f"{'config':24s}  " + "  ".join(f"{s:>17s}" for s in STRATEGIES)
    print(header); print("-" * len(header))
    t0 = time.time()
    for label, extra in CONFIGS.items():
        cells = []
        for strat in STRATEGIES:
            recalls = [run_one(strat, s, steps, extra) for s in range(seeds)]
            cells.append(f"{mean(recalls):>17.2f}")
        print(f"{label:24s}  " + "  ".join(cells))
    print("-" * len(header))
    print(f"\nWall time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
