#!/usr/bin/env python3
"""Diagnostics for joint UAV scouting plus UGV confirmation HAPPO checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario
from scripts.train_happo_smoke import build_args


REWARD_COMPONENTS = (
    ("team", "reward/team"),
    ("team_scout", "reward/team_scout"),
    ("drone_scout", "reward/drone_scout"),
    ("ground_confirm", "reward/ground_confirm"),
    ("uav_conf", "reward/uav_confidence"),
    ("uav_conf_move", "reward/uav_confidence_move"),
    ("uav_conf_overlap", "reward/uav_confidence_overlap"),
    ("uav_frontier", "reward/uav_frontier_alignment"),
    ("uav_cleanup", "reward/uav_cleanup_target_progress"),
    ("ugv_progress", "reward/ugv_progress"),
    ("ugv_align", "reward/ugv_movement_alignment"),
    ("ugv_route_floor", "reward/ugv_route_progress_floor_penalty"),
    ("pending", "reward/pending_penalty"),
)


def _checkpoint_path(path: str | None) -> Path:
    return Path(path) if path else find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _joint_defaults() -> dict[str, Any]:
    _, _algo_args, env_args = build_args(
        num_env_steps=100,
        episode_length=100,
        seed=1,
        comms_dropout=0.0,
        entropy_coef=0.01,
        exp_name="joint_diag_eval_defaults",
        joint_survivor_diagnostic=True,
    )
    return copy.deepcopy(env_args["scenario_kwargs"])


def _joint_schema_ugv_defaults() -> dict[str, Any]:
    _, _algo_args, env_args = build_args(
        num_env_steps=100,
        episode_length=100,
        seed=1,
        comms_dropout=0.0,
        entropy_coef=0.01,
        exp_name="joint_schema_ugv_diag_eval_defaults",
        joint_schema_ugv_diagnostic=True,
    )
    return copy.deepcopy(env_args["scenario_kwargs"])


def _scenario_kwargs(checkpoint_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_training_manifest(checkpoint_dir)
    scenario_kwargs: dict[str, Any] = {}
    if manifest is not None:
        scenario_kwargs.update(copy.deepcopy(manifest.get("env_args", {}).get("scenario_kwargs", {})))
    if args.joint_survivor_diagnostic or args.joint_schema_ugv_diagnostic or not scenario_kwargs:
        defaults = _joint_schema_ugv_defaults() if args.joint_schema_ugv_diagnostic else _joint_defaults()
        defaults.update(scenario_kwargs)
        scenario_kwargs = defaults

    scenario_kwargs["max_steps"] = int(args.steps)
    scenario_kwargs.setdefault("n_drones", 3)
    scenario_kwargs.setdefault("n_ground", int(args.joint_diagnostic_ugvs))
    scenario_kwargs.setdefault("n_survivors", 5)
    if args.n_survivors is not None:
        scenario_kwargs["n_survivors"] = int(args.n_survivors)
        if args.joint_schema_ugv_diagnostic:
            scenario_kwargs["obs_schema_n_survivors"] = int(args.n_survivors)
    scenario_kwargs.setdefault("known_survivors_at_reset", False)
    scenario_kwargs.setdefault("drone_can_confirm", False)
    scenario_kwargs.setdefault("comms_dropout", 0.0)
    scenario_kwargs.setdefault("ugv_target_assignment_mode", "greedy")
    scenario_kwargs.setdefault("ugv_planner_hint", "global_astar")
    scenario_kwargs.setdefault("ugv_dense_reward_mode", "planner_follow")
    scenario_kwargs["uav_confidence_diagnostics"] = True

    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if args.enable_fire:
        scenario_kwargs["disable_fire"] = False
    elif args.disable_fire:
        scenario_kwargs["disable_fire"] = True
    if args.ugv_target_assignment_mode is not None:
        scenario_kwargs["ugv_target_assignment_mode"] = args.ugv_target_assignment_mode.replace("-", "_")
    return scenario_kwargs


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _bin_index(step: int, steps: int, bins: int) -> int:
    if bins <= 1:
        return 0
    return min(int(step / max(steps, 1) * bins), bins - 1)


def _new_time_bins(bins: int) -> list[dict[str, float]]:
    return [
        {
            "count": 0.0,
            "uav_new_cells": 0.0,
            "uav_conf_gain": 0.0,
            "uav_displacement_m": 0.0,
            "ugv_progress_m": 0.0,
            "ugv_route_active": 0.0,
            "pending": 0.0,
            "new_scouts": 0.0,
            "new_oracle_reveals": 0.0,
            "new_confirmations": 0.0,
            "duplicate_assignment": 0.0,
            "assignment_switches": 0.0,
            **{f"reward_{name}": 0.0 for name, _key in REWARD_COMPONENTS},
        }
        for _ in range(bins)
    ]


def _positions(scenario: WildfireSearchScenario) -> torch.Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def run_rollout(
    policy: HappoPolicy,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    time_bins: int,
) -> dict[str, Any]:
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
    n_drones = int(scenario.n_drones)
    n_ground = int(scenario.n_ground)
    n_agents = int(scenario.n_agents)
    n_survivors = int(scenario.n_survivors)
    max_steps = int(scenario_kwargs["max_steps"])
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)

    first_scout_steps: list[int | None] = [None] * n_survivors
    first_confirm_steps: list[int | None] = [None] * n_survivors
    path_lengths_m = np.zeros(n_agents, dtype=float)
    time_series = _new_time_bins(max(int(time_bins), 1))
    pending_counts: list[float] = []
    duplicate_assignment: list[float] = []
    assignment_switches: list[float] = []
    reward_terms = {name: [] for name, _key in REWARD_COMPONENTS}

    prev_pos = _positions(scenario).clone()
    for step in range(max_steps):
        actions = policy(env)
        env.step(actions)

        pos = _positions(scenario).clone()
        scale = max(float(scenario.terrain_sim_units_per_meter[0]), 1e-12)
        displacement_m = (
            torch.linalg.norm(pos[0] - prev_pos[0], dim=-1).detach().cpu().numpy() / scale
        )
        path_lengths_m += displacement_m
        prev_pos = pos

        scouted = scenario.scouted_survivors[0].detach().cpu().numpy().astype(bool)
        confirmed = scenario.found_survivors[0].detach().cpu().numpy().astype(bool)
        for survivor_idx in range(n_survivors):
            if scouted[survivor_idx] and first_scout_steps[survivor_idx] is None:
                first_scout_steps[survivor_idx] = step + 1
            if confirmed[survivor_idx] and first_confirm_steps[survivor_idx] is None:
                first_confirm_steps[survivor_idx] = step + 1

        info = scenario.info(env.agents[0])
        pending = float(np.logical_and(scouted, ~confirmed).sum())
        pending_counts.append(pending)
        duplicate = _to_float(info.get("diagnostic/ugv_duplicate_assignment_fraction"))
        switches = _to_float(info.get("diagnostic/ugv_assignment_switches"))
        duplicate_assignment.append(duplicate)
        assignment_switches.append(switches)

        bin_row = time_series[_bin_index(step, max_steps, len(time_series))]
        bin_row["count"] += 1.0
        bin_row["uav_new_cells"] += _to_float(info.get("diagnostic/uav_new_coverage_cells"))
        bin_row["uav_conf_gain"] += _to_float(info.get("diagnostic/uav_confidence_gain"))
        bin_row["uav_displacement_m"] += _to_float(info.get("diagnostic/uav_displacement_m"))
        bin_row["ugv_progress_m"] += _to_float(info.get("diagnostic/ugv_global_route_progress_m"))
        bin_row["ugv_route_active"] += _to_float(info.get("diagnostic/ugv_global_route_active"))
        bin_row["pending"] += pending
        bin_row["new_scouts"] += _to_float(info.get("mission/new_scouts"))
        bin_row["new_oracle_reveals"] += _to_float(info.get("mission/new_oracle_reveals"))
        bin_row["new_confirmations"] += _to_float(info.get("mission/new_confirmations"))
        bin_row["duplicate_assignment"] += duplicate
        bin_row["assignment_switches"] += switches
        for name, key in REWARD_COMPONENTS:
            value = abs(_to_float(info.get(key)))
            reward_terms[name].append(value)
            bin_row[f"reward_{name}"] += value

    scout_count = sum(step is not None for step in first_scout_steps)
    confirm_count = sum(step is not None for step in first_confirm_steps)
    latencies = [
        float(confirm_step - scout_step)
        for scout_step, confirm_step in zip(first_scout_steps, first_confirm_steps)
        if scout_step is not None and confirm_step is not None and confirm_step >= scout_step
    ]
    time_bin_rows = []
    for idx, bucket in enumerate(time_series):
        count = max(bucket["count"], 1.0)
        row = {
            "episode_fraction": (idx + 0.5) / len(time_series),
            "uav_new_cells_per_step": bucket["uav_new_cells"] / count,
            "uav_confidence_gain_per_step": bucket["uav_conf_gain"] / count,
            "uav_displacement_m_per_step": bucket["uav_displacement_m"] / count,
            "ugv_route_progress_m_per_step": bucket["ugv_progress_m"] / count,
            "ugv_route_active_fraction": bucket["ugv_route_active"] / count,
            "pending_known_survivors": bucket["pending"] / count,
            "new_scouts_per_step": bucket["new_scouts"] / count,
            "new_oracle_reveals_per_step": bucket["new_oracle_reveals"] / count,
            "new_confirmations_per_step": bucket["new_confirmations"] / count,
            "duplicate_assignment_fraction": bucket["duplicate_assignment"] / count,
            "assignment_switches_per_step": bucket["assignment_switches"] / count,
        }
        for name, _key in REWARD_COMPONENTS:
            row[f"reward_{name}"] = bucket[f"reward_{name}"] / count
        time_bin_rows.append(row)

    return {
        "seed": int(seed),
        "survivors": n_survivors,
        "scouted": int(scout_count),
        "confirmed": int(confirm_count),
        "scout_recall": float(scout_count / max(n_survivors, 1)),
        "confirm_recall": float(confirm_count / max(n_survivors, 1)),
        "overall_success": bool(confirm_count == n_survivors),
        "full_confirm_success": bool(confirm_count == n_survivors),
        "first_scout_steps": first_scout_steps,
        "first_confirm_steps": first_confirm_steps,
        "scout_to_confirm_latencies_steps": latencies,
        "scout_to_confirm_latency_count": int(len(latencies)),
        "avg_scout_to_confirm_latency_steps": _mean(latencies),
        "avg_scout_to_confirm_latency_s": _mean(latencies) * step_seconds,
        "final_coverage_fraction": float(scenario.coverage_grid[0].float().mean().item()),
        "final_confidence_mean": float(scenario.uav_confidence_grid[0].float().mean().item()),
        "uav_path_length_m": float(path_lengths_m[:n_drones].sum()) if n_drones > 0 else 0.0,
        "uav_movement_m_per_drone_step": (
            float(path_lengths_m[:n_drones].sum() / max(n_drones * max_steps, 1))
            if n_drones > 0 else 0.0
        ),
        "ugv_path_length_m": float(path_lengths_m[n_drones:].sum()) if n_ground > 0 else 0.0,
        "path_length_by_agent_m": [float(v) for v in path_lengths_m],
        "pending_target_time_mean": _mean(pending_counts),
        "pending_target_time_fraction": float(np.count_nonzero(np.asarray(pending_counts) > 0) / max(len(pending_counts), 1)),
        "duplicate_ugv_assignment_rate": _mean(duplicate_assignment),
        "ugv_assignment_switches_per_episode": float(np.sum(assignment_switches)),
        "avg_reward_components_abs": {name: _mean(values) for name, values in reward_terms.items()},
        "time_bins": time_bin_rows,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "episodes": float(len(rows)),
        "mean_scout_recall": _mean([row["scout_recall"] for row in rows]),
        "mean_confirm_recall": _mean([row["confirm_recall"] for row in rows]),
        "full_confirm_success_rate": _mean([float(row["full_confirm_success"]) for row in rows]),
        "mean_final_coverage_fraction": _mean([row["final_coverage_fraction"] for row in rows]),
        "mean_final_confidence": _mean([row["final_confidence_mean"] for row in rows]),
        "mean_uav_path_length_m": _mean([row["uav_path_length_m"] for row in rows]),
        "mean_uav_movement_m_per_drone_step": _mean([
            row["uav_movement_m_per_drone_step"] for row in rows
        ]),
        "mean_ugv_path_length_m": _mean([row["ugv_path_length_m"] for row in rows]),
        "mean_scout_to_confirm_latency_count": _mean([
            float(row["scout_to_confirm_latency_count"]) for row in rows
        ]),
        "mean_scout_to_confirm_latency_steps": _mean([
            row["avg_scout_to_confirm_latency_steps"] for row in rows
        ]),
        "mean_scout_to_confirm_latency_s": _mean([
            row["avg_scout_to_confirm_latency_s"] for row in rows
        ]),
        "mean_pending_target_time_fraction": _mean([
            row["pending_target_time_fraction"] for row in rows
        ]),
        "mean_duplicate_ugv_assignment_rate": _mean([
            row["duplicate_ugv_assignment_rate"] for row in rows
        ]),
        "mean_ugv_assignment_switches_per_episode": _mean([
            row["ugv_assignment_switches_per_episode"] for row in rows
        ]),
    }


def _plot(rows: list[dict[str, Any]], summary: dict[str, float], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    def hist(ax, title: str, values: list[float], xlabel: str, xlim: tuple[float, float] | None = None):
        values = [float(v) for v in values if math.isfinite(float(v))]
        if values:
            ax.hist(values, bins=12, color="#4f7df3", alpha=0.75)
        else:
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        mean = _mean(values)
        median = float(np.nanmedian(values)) if values else float("nan")
        if math.isfinite(mean):
            ax.axvline(mean, color="#ef4444", label=f"mean {mean:.2f}")
        if math.isfinite(median):
            ax.axvline(median, color="#111827", linestyle="--", label=f"med {median:.2f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("episodes")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    hist(axes[0], "Scout Recall", [row["scout_recall"] for row in rows], "fraction", (0, 1))
    hist(axes[1], "Confirm Recall", [row["confirm_recall"] for row in rows], "fraction", (0, 1))
    hist(axes[2], "Final Confidence", [row["final_confidence_mean"] for row in rows], "mean confidence", (0, 1))
    hist(axes[3], "Final Coverage", [row["final_coverage_fraction"] for row in rows], "covered fraction", (0, 1))
    hist(axes[4], "Scout-to-Confirm Latency", [
        row["avg_scout_to_confirm_latency_steps"] for row in rows
    ], "steps")

    hist(
        axes[5],
        "UAV Movement",
        [row["uav_movement_m_per_drone_step"] for row in rows],
        "m / UAV-step",
    )
    axes[5].set_title("UAV Movement")
    axes[5].grid(axis="y", alpha=0.25)

    if rows and rows[0].get("time_bins"):
        xs = [b["episode_fraction"] for b in rows[0]["time_bins"]]

        def mean_series(key: str) -> list[float]:
            return [
                _mean([row["time_bins"][i][key] for row in rows if i < len(row["time_bins"])])
                for i in range(len(xs))
            ]

        ax = axes[6]
        ax.plot(xs, mean_series("uav_new_cells_per_step"), marker="o", label="new cells")
        ax.plot(xs, mean_series("uav_confidence_gain_per_step"), marker="o", label="conf gain")
        ax.plot(xs, mean_series("uav_displacement_m_per_step"), marker="o", label="uav move")
        ax.set_title("UAV Search Over Time")
        ax.set_xlabel("episode fraction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[7]
        ax.plot(xs, mean_series("ugv_route_progress_m_per_step"), marker="o", label="route progress")
        ax.plot(xs, mean_series("ugv_route_active_fraction"), marker="o", label="route active")
        ax.plot(xs, mean_series("pending_known_survivors"), marker="o", label="pending")
        ax.plot(xs, mean_series("new_oracle_reveals_per_step"), marker="o", label="oracle reveal")
        ax.plot(xs, mean_series("new_confirmations_per_step"), marker="o", label="confirm events")
        ax.plot(xs, mean_series("assignment_switches_per_step"), marker="o", label="switches")
        ax.set_title("UGV Confirmation Over Time")
        ax.set_xlabel("episode fraction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[8]
        for name in (
            "team_scout",
            "drone_scout",
            "ground_confirm",
            "uav_conf",
            "uav_conf_overlap",
            "ugv_progress",
            "ugv_route_floor",
        ):
            ax.plot(xs, mean_series(f"reward_{name}"), marker="o", label=name)
        ax.set_title("Reward Scale (mean abs)")
        ax.set_xlabel("episode fraction")
        ax.set_ylabel("abs reward / step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Joint UAV+UGV HAPPO Diagnostics "
        f"(n={len(rows)}, success={summary['full_confirm_success_rate']:.2f})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", "--checkpoint", dest="checkpoint_dir", default=None)
    parser.add_argument("--joint-survivor-diagnostic", action="store_true",
                        help="Use joint diagnostic defaults when the checkpoint has no manifest.")
    parser.add_argument("--joint-schema-ugv-diagnostic", action="store_true",
                        help="Use 2-UGV delayed-knowledge joint-schema curriculum defaults.")
    parser.add_argument("--joint-diagnostic-ugvs", type=int, default=1)
    parser.add_argument("--n-survivors", type=int, default=None,
                        help="Override survivor count. Default preserves the checkpoint manifest or uses 5.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1020)))
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--enable-fire", action="store_true")
    parser.add_argument("--disable-fire", action="store_true")
    parser.add_argument(
        "--ugv-target-assignment-mode",
        choices=(
            "nearest",
            "greedy",
            "greedy_sticky",
            "greedy-sticky",
            "route_cost_sticky",
            "route-cost-sticky",
        ),
        default=None,
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--plots-output", default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.time_bins <= 0:
        parser.error("--time-bins must be positive")
    if args.joint_diagnostic_ugvs < 1:
        parser.error("--joint-diagnostic-ugvs must be positive")
    if args.n_survivors is not None and args.n_survivors < 1:
        parser.error("--n-survivors must be positive")
    if args.joint_survivor_diagnostic and args.joint_schema_ugv_diagnostic:
        parser.error("--joint-survivor-diagnostic and --joint-schema-ugv-diagnostic are mutually exclusive")
    if args.terrain_cache_path is not None and not Path(args.terrain_cache_path).is_file():
        parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")

    checkpoint_dir = _checkpoint_path(args.checkpoint_dir)
    scenario_kwargs = _scenario_kwargs(checkpoint_dir, args)
    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=not args.stochastic)
    expected_agents = int(scenario_kwargs.get("n_drones", 0)) + int(scenario_kwargs.get("n_ground", 0))
    if len(policy.actors) != expected_agents:
        parser.error(
            f"checkpoint has {len(policy.actors)} actors, but diagnostics scenario has "
            f"{expected_agents} agents"
        )

    print(f"checkpoint: {checkpoint_dir}")
    print(
        "scenario: "
        f"{scenario_kwargs.get('n_drones')} UAVs, "
        f"{scenario_kwargs.get('n_ground')} UGVs, "
        f"{scenario_kwargs.get('n_survivors')} survivors, "
        f"planner={scenario_kwargs.get('ugv_planner_hint')}, "
        f"assignment={scenario_kwargs.get('ugv_target_assignment_mode')}"
    )
    print(f"steps: {args.steps}")
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    print("-" * 88)

    rows = [
        run_rollout(policy, scenario_kwargs, seed, time_bins=args.time_bins)
        for seed in args.seeds
    ]
    summary = summarize(rows)
    for row in rows:
        print(
            f"seed {row['seed']:>4}: "
            f"scout={row['scouted']}/{row['survivors']} "
            f"confirm={row['confirmed']}/{row['survivors']} "
            f"success={int(row['full_confirm_success'])} "
            f"cov={row['final_coverage_fraction']:.3f} "
            f"conf={row['final_confidence_mean']:.3f} "
            f"lat={row['avg_scout_to_confirm_latency_steps']:.1f} "
            f"uav_move={row['uav_movement_m_per_drone_step']:.2f}m/step "
            f"uav_path={row['uav_path_length_m']:.1f}m "
            f"ugv_path={row['ugv_path_length_m']:.1f}m "
            f"pending={row['pending_target_time_fraction']:.2f}"
        )
    print("-" * 88)
    print(
        "means: "
        f"scout_recall={summary['mean_scout_recall']:.3f} "
        f"confirm_recall={summary['mean_confirm_recall']:.3f} "
        f"success={summary['full_confirm_success_rate']:.3f} "
        f"coverage={summary['mean_final_coverage_fraction']:.3f} "
        f"confidence={summary['mean_final_confidence']:.3f} "
        f"uav_move={summary['mean_uav_movement_m_per_drone_step']:.2f}m/step "
        f"latency={summary['mean_scout_to_confirm_latency_steps']:.1f} steps"
    )

    payload = {
        "checkpoint": str(checkpoint_dir),
        "deterministic": not args.stochastic,
        "scenario": scenario_kwargs,
        "summary": summary,
        "rows": rows,
    }
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.plots_output:
        _plot(rows, summary, Path(args.plots_output))


if __name__ == "__main__":
    main()
