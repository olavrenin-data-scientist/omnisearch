"""Evaluate a trained HAPPO checkpoint on the floor-0 / 1km config.

Must evaluate on the SAME terrain + sensor settings used for training, otherwise
the detection geometry collapses and recall reads 0 (see the State Park terrain).

    python -u scripts/eval_floor0_1km.py \
        --checkpoint-dir results/harl_runs/.../seed-.../models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_policy import HappoPolicy
from agents.happo_checkpoint import load_training_manifest
from agents.baselines import BASELINES
from evaluation.mission_metrics import evaluate_policy

TERRAIN = str(ROOT / "data" / "terrain_cache" / "malibu_creek_1km_128.npz")


def base_scenario_kwargs(checkpoint_dir: str) -> dict:
    """Recover the exact scenario kwargs the policy was trained with so eval uses
    the same terrain, sensors, and observation features. Falls back to the
    floor0-1km defaults if no manifest is found."""
    manifest = load_training_manifest(checkpoint_dir)
    if manifest is not None:
        kw = dict(manifest.get("env_args", {}).get("scenario_kwargs", {}) or {})
        if kw:
            return kw
    return {
        "drone_min_footprint": 0.0,
        "ground_confirm_min": 0.0,
        "terrain_source": "real",
        "terrain_cache_path": TERRAIN,
        "drone_camera_fov_deg": 140.0,
        "drone_flight_levels_m": (50.0, 80.0, 100.0),
        "ground_confirmation_range_m": 30.0,
    }


def scenario_kwargs(base: dict, dropout: float, episode_length: int,
                    confirm_requires_los: bool = False) -> dict:
    kw = dict(base)
    kw["max_steps"] = episode_length
    kw["comms_dropout"] = float(dropout)
    kw["drone_min_footprint"] = 0.0   # floor stays 0
    kw["ground_confirm_min"] = 0.0    # floor stays 0
    if confirm_requires_los:
        kw["confirm_requires_los"] = True
    return kw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--dropouts", type=str, default="0.0,0.3")
    p.add_argument("--with-baselines", action="store_true",
                   help="Also report lawnmower/nearest_candidate on the same config.")
    p.add_argument("--confirm-requires-los", action="store_true",
                   help="Require unobstructed terrain line-of-sight (not just range) "
                        "for a ground robot to confirm a survivor.")
    args = p.parse_args()

    dropouts = [float(x) for x in args.dropouts.split(",") if x.strip()]
    base = base_scenario_kwargs(args.checkpoint_dir)

    print(f"\nEval HAPPO @ floor 0 on 1km terrain "
          f"({args.eval_seeds} seeds x {args.episode_length} steps)\n")
    print(f"checkpoint: {args.checkpoint_dir}\n")
    print(f"{'agent':22s} {'dropout':>7s} {'recall':>7s} {'ttv':>8s} {'haz':>6s} {'ugv':>8s}")
    print("-" * 64)

    policy = HappoPolicy.from_checkpoint(
        checkpoint_dir=args.checkpoint_dir, deterministic=True,
    )
    for d in dropouts:
        runs = []
        for k in range(args.eval_seeds):
            policy.reset()
            runs.append(evaluate_policy(
                n_steps=args.episode_length,
                seed=4242 + 100 * k,
                num_envs=2, env_index=0,
                action_fn=policy,
                scenario_kwargs=scenario_kwargs(base, d, args.episode_length,
                                                args.confirm_requires_los),
                device="cpu",
            ))
        _row("happo(trained)", d, runs)

    if args.with_baselines:
        import vmas
        from envs.wildfire_search import WildfireSearchScenario
        from evaluation.mission_metrics import EpisodeRecorder

        for name in ("lawnmower", "nearest_candidate"):
            for d in dropouts:
                runs = []
                for k in range(args.eval_seeds):
                    env = vmas.make_env(
                        scenario=WildfireSearchScenario(),
                        num_envs=2, device="cpu", continuous_actions=True,
                        seed=4242 + 100 * k,
                        **scenario_kwargs(base, d, args.episode_length,
                                          args.confirm_requires_los),
                    )
                    env.reset()
                    pol = BASELINES[name](env)
                    rec = EpisodeRecorder(env.scenario, env_index=0)
                    for _ in range(args.episode_length):
                        env.step(pol(env))
                        rec.step()
                        if env.scenario.done()[0].item():
                            break
                    runs.append(rec.finalize())
                _row(name, d, runs)

    print("-" * 64)


def _row(label: str, dropout: float, runs: list) -> None:
    recall = mean(m.survivor_recall for m in runs)
    ttvs = [m.time_to_verification for m in runs
            if m.time_to_verification == m.time_to_verification]
    ttv = mean(ttvs) if ttvs else float("nan")
    haz = mean(m.hazard_exposure for m in runs)
    ugv = mean(m.ugv_travel_cost for m in runs)
    ttv_s = f"{ttv:>8.1f}" if ttv == ttv else "     nan"
    print(f"{label:22s} {dropout:>7.2f} {recall:>7.2f} {ttv_s} {haz:>6.0f} {ugv:>8.2f}")


if __name__ == "__main__":
    main()
