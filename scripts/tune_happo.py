"""
Automated HAPPO hyperparameter tuning on mission metrics.

This script trains HAPPO across a small hyperparameter grid and ranks
configurations using mission-level outcomes (survivor recall + DRR), not just
training reward.

User-request constraints are enforced by default:
  - drone_min_footprint = 0.0
  - ground_confirm_min  = 0.0
  - episode_length      >= 1000

Example:
    python scripts/tune_happo.py --research --seeds 3 --eval-seeds 2
    python scripts/tune_happo.py --research --episode-length 1200 --num-env-steps 500000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_policy import HappoPolicy
from agents.harl_runner import train_happo
from evaluation.mission_metrics import MissionMetrics, degradation_resilience_ratio, evaluate_policy


@dataclass
class TrialConfig:
    entropy_coef: float
    recurrent: bool
    reward_search: bool
    n_rollout_threads: int


@dataclass
class TrialSummary:
    trial_index: int
    train_seed: int
    train_result: dict
    config: dict
    by_dropout: dict
    recall_at_base: float
    drr_recall: float
    ttv_at_base: float
    score: float


def _parse_float_list(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        v = chunk.strip()
        if not v:
            continue
        values.append(float(v))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        v = chunk.strip()
        if not v:
            continue
        values.append(int(v))
    if not values:
        raise ValueError("Expected at least one int value.")
    return values


def _parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for chunk in raw.split(","):
        v = chunk.strip().lower()
        if not v:
            continue
        if v in {"1", "true", "t", "yes", "y"}:
            values.append(True)
        elif v in {"0", "false", "f", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid bool token: {chunk!r}")
    if not values:
        raise ValueError("Expected at least one bool value.")
    return values


def _mean_metrics(rows: list[MissionMetrics]) -> MissionMetrics:
    return MissionMetrics(
        survivor_recall=mean(r.survivor_recall for r in rows),
        time_to_verification=mean(
            r.time_to_verification for r in rows
            if r.time_to_verification == r.time_to_verification
        ) if any(r.time_to_verification == r.time_to_verification for r in rows) else float("nan"),
        false_positive_trips=int(round(mean(r.false_positive_trips for r in rows))),
        hazard_exposure=int(round(mean(r.hazard_exposure for r in rows))),
        ugv_travel_cost=mean(r.ugv_travel_cost for r in rows),
        n_steps=int(round(mean(r.n_steps for r in rows))),
    )


def _score_trial(recall_at_base: float, drr_recall: float, ttv_at_base: float) -> float:
    # Primary objective: recall at no dropout. Secondary: robustness (DRR).
    # Small tie-break preference for faster verification.
    if ttv_at_base == ttv_at_base:
        ttv_bonus = 1.0 / (1.0 + max(ttv_at_base, 0.0))
    else:
        ttv_bonus = 0.0
    return 3.0 * recall_at_base + 2.0 * drr_recall + 0.1 * ttv_bonus


def _evaluate_checkpoint(
    checkpoint_dir: str,
    *,
    eval_dropouts: list[float],
    eval_seeds: int,
    episode_length: int,
    train_seed: int,
) -> tuple[dict[float, MissionMetrics], dict[float, dict]]:
    policy = HappoPolicy.from_checkpoint(checkpoint_dir=checkpoint_dir, deterministic=True)

    metrics_by_dropout: dict[float, MissionMetrics] = {}
    json_by_dropout: dict[float, dict] = {}

    for d in eval_dropouts:
        runs: list[MissionMetrics] = []
        for k in range(eval_seeds):
            policy.reset()
            m = evaluate_policy(
                n_steps=episode_length,
                seed=train_seed * 10_000 + 100 + k,
                num_envs=2,
                env_index=0,
                action_fn=policy,
                scenario_kwargs={
                    "max_steps": episode_length,
                    "comms_dropout": float(d),
                    "drone_min_footprint": 0.0,
                    "ground_confirm_min": 0.0,
                },
                device="cpu",
            )
            runs.append(m)

        avg_m = _mean_metrics(runs)
        metrics_by_dropout[d] = avg_m
        json_by_dropout[d] = {
            "mean": avg_m.as_dict(),
            "per_seed": [r.as_dict() for r in runs],
        }

    return metrics_by_dropout, json_by_dropout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--research", action="store_true",
                   help="Use research profile defaults.")
    p.add_argument("--num-env-steps", type=int, default=None,
                   help="Total training env steps per trial.")
    p.add_argument("--episode-length", type=int, default=1000,
                   help="Episode length for training/eval (minimum enforced at 1000).")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of train seeds per hyperparameter config.")
    p.add_argument("--eval-seeds", type=int, default=2,
                   help="Number of eval seeds per dropout per trained checkpoint.")
    p.add_argument("--train-dropout", type=float, default=0.0,
                   help="Comms dropout used during training.")
    p.add_argument("--eval-dropouts", type=str, default="0.0,0.2,0.5,0.8",
                   help="Comma-separated dropouts used for post-train evaluation.")
    p.add_argument("--entropy-grid", type=str, default="0.01,0.02,0.03",
                   help="Comma-separated entropy coefficients.")
    p.add_argument("--recurrent-grid", type=str, default="true,false",
                   help="Comma-separated booleans for recurrent policy.")
    p.add_argument("--reward-search-grid", type=str, default="true,false",
                   help="Comma-separated booleans for search-dominant reward shaping.")
    p.add_argument("--thread-grid", type=str, default="16",
                   help="Comma-separated rollout thread counts.")
    p.add_argument("--exp-prefix", default="happo_tune")
    args = p.parse_args()

    eval_dropouts = _parse_float_list(args.eval_dropouts)
    entropy_grid = _parse_float_list(args.entropy_grid)
    recurrent_grid = _parse_bool_list(args.recurrent_grid)
    reward_search_grid = _parse_bool_list(args.reward_search_grid)
    thread_grid = _parse_int_list(args.thread_grid)

    episode_length = max(int(args.episode_length), 1000)
    profile = "research" if args.research else "smoke"
    if args.num_env_steps is None:
        num_env_steps = 400_000 if args.research else 80_000
    else:
        num_env_steps = int(args.num_env_steps)

    grid = [
        TrialConfig(*cfg)
        for cfg in itertools.product(
            entropy_grid,
            recurrent_grid,
            reward_search_grid,
            thread_grid,
        )
    ]

    print("=" * 96)
    print(" OmniSearch — HAPPO Tuner")
    print(f" profile:         {profile}")
    print(f" num_env_steps:   {num_env_steps}")
    print(f" episode_length:  {episode_length}")
    print(f" train_dropout:   {args.train_dropout}")
    print(f" eval_dropouts:   {eval_dropouts}")
    print(f" train_seeds:     {args.seeds}")
    print(f" eval_seeds:      {args.eval_seeds}")
    print(" floors:          drone_min_footprint=0.0, ground_confirm_min=0.0")
    print(f" grid size:       {len(grid)} configs")
    print(f" total trials:    {len(grid) * args.seeds}")
    print("=" * 96)

    all_trials: list[TrialSummary] = []
    t0 = time.time()
    trial_idx = 0
    ts = int(time.time())

    for cfg_idx, cfg in enumerate(grid):
        print(
            f"\n--- Config {cfg_idx + 1}/{len(grid)} | "
            f"entropy={cfg.entropy_coef:.4f} recurrent={cfg.recurrent} "
            f"reward_search={cfg.reward_search} threads={cfg.n_rollout_threads}"
        )
        for seed in range(args.seeds):
            trial_idx += 1
            exp_name = f"{args.exp_prefix}_cfg{cfg_idx:02d}_s{seed}_{ts}"
            print(f"  [{trial_idx}] train seed={seed} exp={exp_name}")

            train_result = train_happo(
                seed=seed,
                num_env_steps=num_env_steps,
                comms_dropout=float(args.train_dropout),
                max_steps=episode_length,
                n_rollout_threads=cfg.n_rollout_threads,
                exp_name=exp_name,
                entropy_coef=cfg.entropy_coef,
                profile=profile,
                recurrent=cfg.recurrent,
                reward_search=cfg.reward_search,
                drone_min_footprint=0.0,
                ground_confirm_min=0.0,
            )

            checkpoint_dir = train_result["checkpoint_dir"]
            metrics_by_dropout, by_dropout_json = _evaluate_checkpoint(
                checkpoint_dir,
                eval_dropouts=eval_dropouts,
                eval_seeds=args.eval_seeds,
                episode_length=episode_length,
                train_seed=seed,
            )

            base_dropout = eval_dropouts[0]
            recall_at_base = metrics_by_dropout[base_dropout].survivor_recall
            ttv_at_base = metrics_by_dropout[base_dropout].time_to_verification
            drr_recall = degradation_resilience_ratio(
                metrics_by_dropout,
                metric="survivor_recall",
                baseline_dropout=base_dropout,
            )
            score = _score_trial(recall_at_base, drr_recall, ttv_at_base)
            print(
                f"      recall@d{base_dropout:.1f}={recall_at_base:.3f} "
                f"drr={drr_recall:.3f} score={score:.3f}"
            )

            all_trials.append(
                TrialSummary(
                    trial_index=trial_idx,
                    train_seed=seed,
                    train_result=train_result,
                    config=asdict(cfg),
                    by_dropout=by_dropout_json,
                    recall_at_base=recall_at_base,
                    drr_recall=drr_recall,
                    ttv_at_base=ttv_at_base,
                    score=score,
                )
            )

    all_trials.sort(
        key=lambda t: (
            t.recall_at_base,
            t.drr_recall,
            -t.ttv_at_base if t.ttv_at_base == t.ttv_at_base else -math.inf,
            t.score,
        ),
        reverse=True,
    )
    best = all_trials[0]

    print("\n" + "=" * 96)
    print(" Top 5 trials")
    print("=" * 96)
    for row in all_trials[:5]:
        print(
            f" trial={row.trial_index:3d} seed={row.train_seed} "
            f"recall@0={row.recall_at_base:.3f} drr={row.drr_recall:.3f} "
            f"ttv={row.ttv_at_base if row.ttv_at_base == row.ttv_at_base else float('nan'):.1f} "
            f"score={row.score:.3f} cfg={row.config}"
        )

    print("\nBest config:")
    print(best.config)
    print(f"Best checkpoint: {best.train_result['checkpoint_dir']}")
    print(f"Best recall@0:   {best.recall_at_base:.3f}")
    print(f"Best DRR recall: {best.drr_recall:.3f}")

    payload = {
        "params": {
            "research": args.research,
            "profile": profile,
            "num_env_steps": num_env_steps,
            "episode_length": episode_length,
            "train_dropout": args.train_dropout,
            "eval_dropouts": eval_dropouts,
            "train_seeds": args.seeds,
            "eval_seeds": args.eval_seeds,
            "floors": {
                "drone_min_footprint": 0.0,
                "ground_confirm_min": 0.0,
            },
            "entropy_grid": entropy_grid,
            "recurrent_grid": recurrent_grid,
            "reward_search_grid": reward_search_grid,
            "thread_grid": thread_grid,
        },
        "best_trial": asdict(best),
        "all_trials": [asdict(t) for t in all_trials],
        "total_wall_sec": round(time.time() - t0, 2),
    }

    out = ROOT / "results" / f"happo_tuning_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote: {out.relative_to(ROOT)}")
    print("=" * 96)


if __name__ == "__main__":
    main()
