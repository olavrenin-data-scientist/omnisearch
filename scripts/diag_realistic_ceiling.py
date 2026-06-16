"""Find the realistic recall ceiling under line-of-sight confirmation.

Realistic = no detection floor, realistic camera FOV/altitude, terrain line-of-sight
required for confirmation (no confirming through ridges), n_ground fixed at 2.

The only physically-defensible lever for higher recall is the confirmation range
(a thermal/EO sensor detects a human at ~150-300 m with clear line of sight). This
script sweeps that range and reports the best hand-coded expert recall, so we can
state honestly what sensor capability a given recall (e.g. 0.9) actually requires.

Seeds are batched across parallel VMAS envs for speed.

    python -u scripts/diag_realistic_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import vmas

from agents.baselines import BASELINES
from envs.wildfire_search import WildfireSearchScenario

TERRAIN = str(ROOT / "data" / "terrain_cache" / "malibu_creek_1km_128.npz")


def recall_for(name: str, confirm_m: float, steps: int, n_envs: int, fov: float,
               flight: tuple[float, ...], seed: int, drone_confirm: bool = False) -> tuple[float, float]:
    """Return (mean recall, mean UGV travel) over n_envs parallel episodes."""
    env = vmas.make_env(
        scenario=WildfireSearchScenario(), num_envs=n_envs, device="cpu",
        continuous_actions=True, seed=seed,
        terrain_source="real", terrain_cache_path=TERRAIN,
        drone_camera_fov_deg=fov, drone_flight_levels_m=flight,
        ground_confirmation_range_m=float(confirm_m), confirm_requires_los=True,
        drone_can_confirm=drone_confirm,
        max_steps=steps,
    )
    env.reset()
    pol = BASELINES[name](env)
    sc = env.scenario
    for _ in range(steps):
        env.step(pol(env))
        if sc.done().all():
            break
    found = sc.found_survivors.float()                  # [B, S]
    recall = (found.sum(dim=1) / found.shape[1]).mean().item()
    ugv = sc.metric_ugv_travel.mean().item() if hasattr(sc, "metric_ugv_travel") else float("nan")
    return recall, ugv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n-envs", type=int, default=8, help="Parallel episodes = effective seeds.")
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--flight", default="50,80,100")
    p.add_argument("--ranges", default="60,100,150,200,250,300")
    p.add_argument("--approaches", default="lawnmower,nearest_candidate")
    p.add_argument("--drone-confirm", action="store_true",
                   help="Let drones (EO/IR) confirm survivors from altitude with top-down line-of-sight.")
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    flight = tuple(float(v) for v in args.flight.split(",") if v.strip())
    ranges = [float(v) for v in args.ranges.split(",") if v.strip()]
    approaches = [a for a in args.approaches.split(",") if a.strip()]

    print(f"Realistic ceiling: LOS on, FOV {args.fov}, flight {flight} m, "
          f"n_ground=2, drone_confirm={args.drone_confirm}, "
          f"{args.n_envs} envs x {args.steps} steps", flush=True)
    header = f"{'confirm_m':>9s} " + " ".join(f"{a:>18s}" for a in approaches)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for rng in ranges:
        cells = []
        for a in approaches:
            rec, ugv = recall_for(a, rng, args.steps, args.n_envs, args.fov, flight, args.seed,
                                  drone_confirm=args.drone_confirm)
            cells.append(f"{rec:.2f} (ugv {ugv:.1f})")
        print(f"{rng:>9.0f} " + " ".join(f"{c:>18s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
