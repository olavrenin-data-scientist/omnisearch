"""
Diagnose HAPPO UAV-only survivor scouting checkpoints.

This is intentionally focused on the UAV survivor diagnostic task:
three drones, no UGVs, five survivors, no fire, and drone scouting counts as
mission success. The first metrics are recall-oriented: how many survivors are
scouted, how many are missed, and how long scouting takes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import vmas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario


def _checkpoint_path(path: str | None) -> Path:
    return Path(path) if path else find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _distance_kwargs(args: argparse.Namespace) -> dict[str, float]:
    if args.uav_diagnostic_target_distance_min_m is None and args.uav_diagnostic_target_distance_max_m is None:
        return {}

    target_distance_min_m = max(
        float(0.0 if args.uav_diagnostic_target_distance_min_m is None else args.uav_diagnostic_target_distance_min_m),
        0.0,
    )
    out = {"known_survivor_spawn_distance_min_m": target_distance_min_m}
    if args.uav_diagnostic_target_distance_max_m is not None:
        target_distance_max_m = max(float(args.uav_diagnostic_target_distance_max_m), 0.0)
        if target_distance_max_m < target_distance_min_m:
            raise ValueError(
                "uav_diagnostic_target_distance_max_m must be >= "
                "uav_diagnostic_target_distance_min_m"
            )
        out.update({
            "known_survivor_spawn_distance_m": 0.5 * (target_distance_min_m + target_distance_max_m),
            "known_survivor_spawn_distance_max_m": target_distance_max_m,
        })
    return out


def _scenario_kwargs(checkpoint_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(checkpoint_dir)
    scenario_kwargs: dict[str, Any] = {}
    if manifest is not None:
        scenario_kwargs.update(copy.deepcopy(manifest.get("env_args", {}).get("scenario_kwargs", {})))

    for key in (
        "known_survivor_spawn_distance_m",
        "known_survivor_spawn_distance_min_m",
        "known_survivor_spawn_distance_max_m",
    ):
        if (
            args.uav_diagnostic_target_distance_min_m is not None
            or args.uav_diagnostic_target_distance_max_m is not None
        ):
            scenario_kwargs.pop(key, None)

    scenario_kwargs.update({
        "max_steps": args.steps,
        "n_drones": 3,
        "n_ground": 0,
        "n_survivors": 5,
        "known_survivors_at_reset": False,
        "survivor_spawn_reference": "drone",
        "drone_can_confirm": True,
        "disable_fire": True,
        "comms_dropout": 0.0,
    })
    scenario_kwargs.update(_distance_kwargs(args))

    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if args.local_map_patch_size is not None:
        scenario_kwargs["local_map_patch_size"] = int(args.local_map_patch_size)
    if args.drone_min_footprint_radius_m is not None:
        scenario_kwargs.pop("drone_min_footprint", None)
        scenario_kwargs["drone_min_footprint_m"] = max(float(args.drone_min_footprint_radius_m), 0.0)
    return scenario_kwargs


def run_rollout(policy: HappoPolicy, scenario_kwargs: dict[str, Any], seed: int) -> dict[str, Any]:
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset()
    policy.reset()
    scenario = env.scenario

    n_survivors = int(scenario.n_survivors)
    first_scout_steps: list[int | None] = [None] * n_survivors
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)

    for step in range(int(scenario_kwargs["max_steps"])):
        actions = policy(env)
        env.step(actions)
        scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        for survivor_idx, is_scouted in enumerate(scouted):
            if is_scouted and first_scout_steps[survivor_idx] is None:
                first_scout_steps[survivor_idx] = step + 1
        if all(value is not None for value in first_scout_steps):
            break

    scouted_count = sum(value is not None for value in first_scout_steps)
    missed_count = n_survivors - scouted_count
    scout_steps = [value for value in first_scout_steps if value is not None]
    all_scouted_step = max(scout_steps) if scouted_count == n_survivors and scout_steps else None
    final_coverage_fraction = float(scenario.coverage_grid[0].float().mean().detach().cpu().item())
    row = {
        "seed": int(seed),
        "survivors": n_survivors,
        "scouted": scouted_count,
        "missed": missed_count,
        "recall": scouted_count / n_survivors if n_survivors else 0.0,
        "final_coverage_fraction": final_coverage_fraction,
        "full_success": float(scouted_count == n_survivors),
        "avg_scout_step": float(np.mean(scout_steps)) if scout_steps else math.nan,
        "avg_scout_time_s": float(np.mean(scout_steps) * step_seconds) if scout_steps else math.nan,
        "all_scouted_step": all_scouted_step,
        "all_scouted_time_s": None if all_scouted_step is None else float(all_scouted_step * step_seconds),
        "first_scout_steps": first_scout_steps,
        "first_scout_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_scout_steps
        ],
    }
    close = getattr(env, "close", None)
    if close is not None:
        close()
    return row


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    successful = [row for row in rows if row["all_scouted_step"] is not None]
    return {
        "episodes": float(len(rows)),
        "mean_scouted": float(np.mean([row["scouted"] for row in rows])) if rows else 0.0,
        "mean_missed": float(np.mean([row["missed"] for row in rows])) if rows else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
        "mean_final_coverage_fraction": (
            float(np.mean([row["final_coverage_fraction"] for row in rows])) if rows else 0.0
        ),
        "full_success_rate": float(np.mean([row["full_success"] for row in rows])) if rows else 0.0,
        "mean_avg_scout_step": _finite_mean([row["avg_scout_step"] for row in rows]),
        "mean_avg_scout_time_s": _finite_mean([row["avg_scout_time_s"] for row in rows]),
        "mean_all_scouted_step_successes": (
            float(np.mean([row["all_scouted_step"] for row in successful])) if successful else math.nan
        ),
        "mean_all_scouted_time_s_successes": (
            float(np.mean([row["all_scouted_time_s"] for row in successful])) if successful else math.nan
        ),
    }


def _fmt_optional(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=None, help="Path to a HARL models/ checkpoint directory.")
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--uav-diagnostic-target-distance-min-m", type=float, default=None)
    parser.add_argument("--uav-diagnostic-target-distance-max-m", type=float, default=None,
                        help="Omit for no upper bound when a min distance is provided.")
    parser.add_argument("--local-map-patch-size", type=int, default=None)
    parser.add_argument("--drone-min-footprint-radius-m", type=float, default=None)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actor means.")
    parser.add_argument("--json-output", default=None, help="Optional path to write per-seed rows and summary as JSON.")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.local_map_patch_size is not None and (args.local_map_patch_size < 1 or args.local_map_patch_size % 2 != 1):
        parser.error("--local-map-patch-size must be a positive odd integer")
    if args.terrain_cache_path is not None and not Path(args.terrain_cache_path).is_file():
        parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")

    checkpoint_dir = _checkpoint_path(args.checkpoint_dir)
    scenario_kwargs = _scenario_kwargs(checkpoint_dir, args)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"steps: {args.steps}")
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    print(
        "scenario: "
        f"{scenario_kwargs['n_drones']} UAVs, "
        f"{scenario_kwargs['n_ground']} UGVs, "
        f"{scenario_kwargs['n_survivors']} survivors, "
        f"dt={scenario_kwargs.get('sim_step_seconds', 'scenario-default')}s"
    )
    print("-" * 88)

    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=not args.stochastic)
    rows = [
        run_rollout(policy, scenario_kwargs, seed)
        for seed in args.seeds
    ]
    for row in rows:
        print(
            f"seed {row['seed']:>4}: "
            f"scouted={row['scouted']}/{row['survivors']} "
            f"missed={row['missed']} "
            f"recall={row['recall']:.3f} "
            f"coverage={row['final_coverage_fraction']:.3f} "
            f"avg_scout={_fmt_optional(row['avg_scout_step'])} steps/"
            f"{_fmt_optional(row['avg_scout_time_s'])}s "
            f"all_scouted={_fmt_optional(row['all_scouted_step'])} steps/"
            f"{_fmt_optional(row['all_scouted_time_s'])}s "
            f"first_steps={row['first_scout_steps']}"
        )

    summary = summarize(rows)
    print("-" * 88)
    print(
        "means: "
        f"scouted={summary['mean_scouted']:.3f} "
        f"missed={summary['mean_missed']:.3f} "
        f"recall={summary['mean_recall']:.3f} "
        f"coverage={summary['mean_final_coverage_fraction']:.3f} "
        f"success={summary['full_success_rate']:.3f} "
        f"avg_scout={summary['mean_avg_scout_step']:.1f} steps/"
        f"{summary['mean_avg_scout_time_s']:.1f}s "
        f"all_scouted_successes={summary['mean_all_scouted_step_successes']:.1f} steps/"
        f"{summary['mean_all_scouted_time_s_successes']:.1f}s"
    )
    print("note: all_scouted_successes averages only episodes that scouted every survivor.")

    if args.json_output:
        output = {
            "checkpoint": str(checkpoint_dir),
            "scenario_kwargs": scenario_kwargs,
            "rows": rows,
            "summary": summary,
        }
        Path(args.json_output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"wrote: {args.json_output}")


if __name__ == "__main__":
    main()
