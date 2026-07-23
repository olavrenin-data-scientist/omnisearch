#!/usr/bin/env python3
"""Compare heuristic UGV A* confirmation baselines with matched reveal timing."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import AntColonyPolicy, LawnmowerPolicy
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario
from scripts.diagnostic_json import (
    partial_json_path,
    write_final_json,
    write_partial_json,
)
from scripts.train_happo_smoke import build_args


DEFAULT_STRATEGIES = ("lawnmower_astar", "ant_colony_astar")


@dataclass(frozen=True)
class StrategySpec:
    label: str
    name: str
    checkpoint_dir: Path | None = None


def parse_strategy_specs(
    strategies: list[str] | None,
    *,
    happo_checkpoint: str | Path | None = None,
) -> list[StrategySpec]:
    raw = strategies or list(DEFAULT_STRATEGIES)
    expanded: list[str] = []
    for token in raw:
        expanded.extend(DEFAULT_STRATEGIES if token == "all" else (token,))

    specs: list[StrategySpec] = []
    used: set[str] = set()
    valid = set(DEFAULT_STRATEGIES)
    for token in expanded:
        name = token.replace("-", "_")
        if token == "happo" or token.startswith("happo:"):
            checkpoint = token.split(":", 1)[1] if token.startswith("happo:") else happo_checkpoint
            label = _unique_label("happo", used)
            specs.append(StrategySpec(
                label=label,
                name="happo",
                checkpoint_dir=_resolve_happo_checkpoint(checkpoint),
            ))
            continue
        if name not in valid:
            raise ValueError(
                f"unknown strategy {token!r}; available: "
                f"{', '.join((*DEFAULT_STRATEGIES, 'happo', 'happo:/path/to/models', 'all'))}"
            )
        label = _unique_label(name, used)
        specs.append(StrategySpec(label=label, name=name))
    if not specs:
        raise ValueError("at least one strategy is required")
    return specs


def _unique_label(label: str, used: set[str]) -> str:
    if label not in used:
        used.add(label)
        return label
    idx = 2
    while f"{label}_{idx}" in used:
        idx += 1
    unique = f"{label}_{idx}"
    used.add(unique)
    return unique


def _resolve_happo_checkpoint(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    return find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _joint_schema_ugv_defaults() -> dict[str, Any]:
    _, _algo_args, env_args = build_args(
        num_env_steps=100,
        episode_length=100,
        seed=1,
        comms_dropout=0.0,
        entropy_coef=0.01,
        exp_name="ugv_baseline_strategy_defaults",
        joint_schema_ugv_diagnostic=True,
    )
    return copy.deepcopy(env_args["scenario_kwargs"])


def build_scenario_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    scenario_kwargs = _joint_schema_ugv_defaults()
    scenario_kwargs.update({
        "max_steps": int(args.steps),
        "n_drones": 0,
        "n_ground": int(args.n_ugvs),
        "n_survivors": 5,
        "obs_schema_n_ground": int(args.n_ugvs),
        "known_survivors_at_reset": False,
        "delayed_survivor_knowledge": True,
        "survivor_reveal_initial_count": int(args.survivor_reveal_initial_count),
        "survivor_reveal_start_step": int(args.survivor_reveal_start_step),
        "survivor_reveal_end_step": int(args.survivor_reveal_end_step),
        "drone_can_confirm": False,
        "comms_dropout": 0.0,
    })
    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = str(args.terrain_cache_path)
    if args.enable_fire:
        scenario_kwargs["disable_fire"] = False
    if args.disable_fire:
        scenario_kwargs["disable_fire"] = True

    overrides = {
        "ugv_planner_fire_mode": args.ugv_planner_fire_mode,
        "ugv_planner_fire_replan_policy": args.ugv_planner_fire_replan_policy,
        "ugv_planner_fire_replan_interval_steps": args.ugv_planner_fire_replan_interval_steps,
        "ugv_planner_fire_cost": args.ugv_planner_fire_cost,
        "ugv_planner_fire_block_threshold": args.ugv_planner_fire_block_threshold,
        "ugv_planner_smoke_cost": args.ugv_planner_smoke_cost,
        "ugv_planner_smolder_cost": args.ugv_planner_smolder_cost,
        "ugv_planner_fire_buffer_m": args.ugv_planner_fire_buffer_m,
        "ugv_planner_fire_buffer_cost": args.ugv_planner_fire_buffer_cost,
        "ugv_planner_land_cover_costs": (
            tuple(args.ugv_planner_land_cover_costs)
            if args.ugv_planner_land_cover_costs is not None
            else None
        ),
    }
    for key, value in overrides.items():
        if value is not None:
            scenario_kwargs[key] = value
    return scenario_kwargs


class MatchedRevealAntColonyPolicy:
    """Ant-colony wrapper that receives oracle-delayed UGV survivor reveals."""

    def __init__(self, env: Any):
        self.policy = AntColonyPolicy(env)
        self.scenario: WildfireSearchScenario = env.scenario

    def __call__(self, env: Any) -> list[torch.Tensor]:
        sync_ant_colony_oracle_knowledge(self.policy, self.scenario)
        return self.policy(env)


def sync_ant_colony_oracle_knowledge(policy: AntColonyPolicy, scenario: WildfireSearchScenario) -> None:
    if scenario.n_survivors <= 0 or scenario.n_ground <= 0:
        return
    revealed = scenario.scouted_survivors
    confirmed = scenario.found_survivors
    if revealed.numel() == 0:
        return
    policy.known_survivors[:, scenario.n_drones:, :] |= revealed.unsqueeze(1)
    policy.known_survivors |= confirmed.unsqueeze(1)
    policy.known_confirmed |= confirmed.unsqueeze(1)


def make_policy(
    spec: StrategySpec,
    env: Any,
    *,
    happo_cache: dict[tuple[Path, bool], HappoPolicy] | None = None,
    deterministic_happo: bool = True,
) -> Callable[[Any], list[torch.Tensor]]:
    if spec.name == "lawnmower_astar":
        return LawnmowerPolicy(env)
    if spec.name == "ant_colony_astar":
        return MatchedRevealAntColonyPolicy(env)
    if spec.name == "happo":
        if spec.checkpoint_dir is None:
            raise ValueError("HAPPO strategy requires a checkpoint directory")
        key = (spec.checkpoint_dir, deterministic_happo)
        if happo_cache is None:
            return HappoPolicy.from_checkpoint(spec.checkpoint_dir, deterministic=deterministic_happo)
        if key not in happo_cache:
            happo_cache[key] = HappoPolicy.from_checkpoint(
                spec.checkpoint_dir,
                deterministic=deterministic_happo,
            )
        return happo_cache[key]
    raise ValueError(f"unsupported strategy {spec.name!r}")


def _positions(scenario: WildfireSearchScenario) -> torch.Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


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


def _active_survivor_mask_for_env(scenario: WildfireSearchScenario, env_index: int = 0) -> np.ndarray:
    slots = int(getattr(scenario, "n_survivors", 0))
    active = getattr(scenario, "active_survivors", None)
    if active is None:
        return np.ones(slots, dtype=bool)
    return active[env_index].detach().cpu().numpy().astype(bool)


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else float("nan")


def _bin_index(step: int, steps: int, bins: int) -> int:
    return min(int(step / max(steps, 1) * max(bins, 1)), max(bins, 1) - 1)


def _new_time_bins(count: int) -> list[dict[str, float]]:
    return [
        {
            "count": 0.0,
            "pending": 0.0,
            "new_reveals": 0.0,
            "new_confirmations": 0.0,
            "ugv_displacement_m": 0.0,
            "ugv_progress_m": 0.0,
            "ugv_route_active": 0.0,
            "duplicate_assignment": 0.0,
            "assignment_switches": 0.0,
        }
        for _ in range(max(int(count), 1))
    ]


def _pending_min_distance_m(scenario: WildfireSearchScenario) -> float:
    if scenario.n_ground <= 0 or scenario.n_survivors <= 0:
        return math.nan
    pending = scenario.scouted_survivors[0] & ~scenario.found_survivors[0]
    if not bool(pending.any().item()):
        return math.nan
    ground_pos = torch.stack(
        [agent.state.pos[0] for agent in scenario.world.agents[scenario.n_drones:]],
        dim=0,
    )
    survivor_pos = torch.stack([survivor.state.pos[0] for survivor in scenario._survivors], dim=0)
    distances = torch.linalg.norm(ground_pos.unsqueeze(1) - survivor_pos.unsqueeze(0), dim=-1)
    distances = distances[:, pending]
    min_sim = float(distances.min().detach().cpu().item())
    scale = max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-12)
    return min_sim / scale


def run_rollout(
    spec: StrategySpec,
    scenario_kwargs: dict[str, Any],
    seed: int,
    *,
    time_bins: int = 5,
    happo_cache: dict[tuple[Path, bool], HappoPolicy] | None = None,
    stochastic_happo: bool = False,
) -> dict[str, Any]:
    env = WildfireSearchScenario.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=int(seed),
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset(seed=int(seed))
    scenario = env.scenario
    policy = make_policy(
        spec,
        env,
        happo_cache=happo_cache,
        deterministic_happo=not stochastic_happo,
    )
    if hasattr(policy, "reset"):
        policy.reset()

    n_ground = int(scenario.n_ground)
    n_agents = int(scenario.n_agents)
    n_survivors = int(scenario.n_survivors)
    active_survivor_mask = _active_survivor_mask_for_env(scenario)
    n_active_survivors = int(active_survivor_mask.sum())
    max_steps = int(scenario_kwargs["max_steps"])
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    scale = max(float(scenario.terrain_sim_units_per_meter[0].detach().cpu().item()), 1e-12)

    reveal_steps = [
        int(value)
        for value in scenario.survivor_reveal_steps[0].detach().cpu().tolist()
    ]
    first_confirm_steps: list[int | None] = [None] * n_survivors
    path_lengths_m = np.zeros(n_agents, dtype=float)
    pending_counts: list[float] = []
    pending_min_distances: list[float] = []
    duplicate_assignment: list[float] = []
    assignment_switches: list[float] = []
    confirm_auc_sum = 0.0
    auc_steps = 0
    time_series = _new_time_bins(time_bins)

    prev_pos = _positions(scenario).clone()
    for step in range(max_steps):
        actions = policy(env)
        env.step(actions)

        pos = _positions(scenario).clone()
        displacement_m = (
            torch.linalg.norm(pos[0] - prev_pos[0], dim=-1).detach().cpu().numpy()
            / scale
        )
        path_lengths_m += displacement_m
        prev_pos = pos

        confirmed = scenario.found_survivors[0].detach().cpu().numpy().astype(bool)
        confirm_auc_sum += (
            float(np.logical_and(confirmed, active_survivor_mask).sum() / n_active_survivors)
            if n_active_survivors > 0 else 1.0
        )
        auc_steps += 1
        for survivor_idx, is_confirmed in enumerate(confirmed):
            if is_confirmed and first_confirm_steps[survivor_idx] is None:
                first_confirm_steps[survivor_idx] = step + 1

        info = scenario.info(env.agents[0])
        pending = float((scenario.scouted_survivors[0] & ~scenario.found_survivors[0]).float().sum().item())
        pending_counts.append(pending)
        pending_min_distances.append(_pending_min_distance_m(scenario))
        duplicate = _to_float(info.get("diagnostic/ugv_duplicate_assignment_fraction"))
        switches = _to_float(info.get("diagnostic/ugv_assignment_switches"))
        duplicate_assignment.append(duplicate)
        assignment_switches.append(switches)

        bin_row = time_series[_bin_index(step, max_steps, len(time_series))]
        bin_row["count"] += 1.0
        bin_row["pending"] += pending
        bin_row["new_reveals"] += _to_float(info.get("mission/new_oracle_reveals"))
        bin_row["new_confirmations"] += _to_float(info.get("mission/new_confirmations"))
        bin_row["ugv_displacement_m"] += float(displacement_m.sum())
        bin_row["ugv_progress_m"] += _to_float(info.get("diagnostic/ugv_global_route_progress_m"))
        bin_row["ugv_route_active"] += _to_float(info.get("diagnostic/ugv_global_route_active"))
        bin_row["duplicate_assignment"] += duplicate
        bin_row["assignment_switches"] += switches

    confirmed_count = sum(
        step is not None
        for survivor_idx, step in enumerate(first_confirm_steps)
        if survivor_idx < len(active_survivor_mask) and active_survivor_mask[survivor_idx]
    )
    reveal_to_confirm_latencies = [
        float(confirm_step - reveal_step)
        for reveal_step, confirm_step in zip(reveal_steps, first_confirm_steps)
        if confirm_step is not None and confirm_step >= reveal_step
    ]
    time_bin_rows = []
    for idx, bucket in enumerate(time_series):
        count = max(bucket["count"], 1.0)
        time_bin_rows.append({
            "episode_fraction": (idx + 0.5) / len(time_series),
            "pending_known_survivors": bucket["pending"] / count,
            "new_reveals_per_step": bucket["new_reveals"] / count,
            "new_confirmations_per_step": bucket["new_confirmations"] / count,
            "ugv_displacement_m_per_step": bucket["ugv_displacement_m"] / count,
            "ugv_displacement_m_per_ugv_step": (
                bucket["ugv_displacement_m"] / max(count * n_ground, 1.0)
            ),
            "ugv_route_progress_m_per_step": bucket["ugv_progress_m"] / count,
            "ugv_route_active_fraction": bucket["ugv_route_active"] / max(count * n_ground, 1.0),
            "duplicate_assignment_fraction": bucket["duplicate_assignment"] / count,
            "assignment_switches_per_step": bucket["assignment_switches"] / count,
        })

    final_pending_distance_m = _pending_min_distance_m(scenario)
    row = {
        "strategy": spec.label,
        "strategy_name": spec.name,
        "checkpoint_dir": None if spec.checkpoint_dir is None else str(spec.checkpoint_dir),
        "seed": int(seed),
        "max_steps": max_steps,
        "survivors": n_active_survivors,
        "active_survivors": n_active_survivors,
        "survivor_slots": n_survivors,
        "survivor_reveal_steps": reveal_steps,
        "confirmed": int(confirmed_count),
        "confirmation_recall": float(confirmed_count / max(n_active_survivors, 1)),
        "confirmation_auc": float(confirm_auc_sum / max(auc_steps, 1)),
        "full_confirm_success": bool(confirmed_count == n_active_survivors),
        "first_confirm_steps": first_confirm_steps,
        "first_confirm_times_s": [
            None if value is None else float(value * step_seconds)
            for value in first_confirm_steps
        ],
        "reveal_to_confirm_latencies_steps": reveal_to_confirm_latencies,
        "reveal_to_confirm_latencies_s": [
            float(value * step_seconds)
            for value in reveal_to_confirm_latencies
        ],
        "avg_reveal_to_confirm_latency_steps": _mean(reveal_to_confirm_latencies),
        "avg_reveal_to_confirm_latency_s": _mean(reveal_to_confirm_latencies) * step_seconds,
        "ugv_path_length_m": float(path_lengths_m.sum()),
        "ugv_path_length_by_agent_m": [float(value) for value in path_lengths_m],
        "ugv_movement_m_per_ugv_step": float(path_lengths_m.sum() / max(n_ground * max_steps, 1)),
        "pending_target_time_mean": _mean(pending_counts),
        "pending_target_time_fraction": float(
            np.count_nonzero(np.asarray(pending_counts) > 0) / max(len(pending_counts), 1)
        ),
        "duplicate_ugv_assignment_rate": _mean(duplicate_assignment),
        "ugv_assignment_switches_per_episode": float(np.sum(assignment_switches)),
        "final_pending_min_distance_m": final_pending_distance_m,
        "min_pending_distance_m": (
            min(value for value in pending_min_distances if math.isfinite(value))
            if any(math.isfinite(value) for value in pending_min_distances)
            else math.nan
        ),
        "time_bins": time_bin_rows,
    }
    close = getattr(env, "close", None)
    if close is not None:
        close()
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = sorted({str(row["strategy"]) for row in rows})
    return {
        "episodes": float(len(rows)),
        "strategies": strategies,
        "by_strategy": {
            strategy: _summarize_strategy([row for row in rows if row["strategy"] == strategy])
            for strategy in strategies
        },
    }


def _summarize_strategy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": float(len(rows)),
        "mean_confirmation_recall": _mean([row["confirmation_recall"] for row in rows]),
        "mean_confirmation_auc": _mean([row["confirmation_auc"] for row in rows]),
        "full_confirm_success_rate": _mean([float(row["full_confirm_success"]) for row in rows]),
        "mean_reveal_to_confirm_latency_steps": _mean([
            row["avg_reveal_to_confirm_latency_steps"] for row in rows
        ]),
        "median_reveal_to_confirm_latency_steps": _median([
            latency
            for row in rows
            for latency in row["reveal_to_confirm_latencies_steps"]
        ]),
        "mean_ugv_path_length_m": _mean([row["ugv_path_length_m"] for row in rows]),
        "mean_ugv_movement_m_per_ugv_step": _mean([
            row["ugv_movement_m_per_ugv_step"] for row in rows
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
        "mean_final_pending_min_distance_m": _mean([
            row["final_pending_min_distance_m"] for row in rows
        ]),
        "mean_min_pending_distance_m": _mean([
            row["min_pending_distance_m"] for row in rows
        ]),
        "time_bins": _summarize_time_bins(rows),
    }


def _summarize_time_bins(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not rows:
        return []
    bins = len(rows[0].get("time_bins", []))
    out = []
    keys = [
        key
        for key in rows[0]["time_bins"][0]
        if key != "episode_fraction"
    ] if bins else []
    for idx in range(bins):
        item = {"episode_fraction": rows[0]["time_bins"][idx]["episode_fraction"]}
        for key in keys:
            item[key] = _mean([
                row["time_bins"][idx].get(key, math.nan)
                for row in rows
                if idx < len(row.get("time_bins", []))
            ])
        out.append(item)
    return out


def write_plots(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    strategies = list(summary["strategies"])
    colors = {
        "lawnmower_astar": "#2563eb",
        "ant_colony_astar": "#10b981",
        "happo": "#f97316",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    def values(strategy: str, key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if row["strategy"] == strategy and math.isfinite(float(row[key]))
        ]

    def hist(ax, title: str, key: str, xlabel: str, xlim: tuple[float, float] | None = None) -> None:
        for strategy in strategies:
            vals = values(strategy, key)
            if vals:
                ax.hist(
                    vals,
                    bins=12,
                    alpha=0.45,
                    label=f"{strategy} mean={_mean(vals):.2f}",
                    color=colors.get(strategy),
                )
                ax.axvline(_mean(vals), color=colors.get(strategy), linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("episodes")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    hist(axes[0], "Confirmation Recall", "confirmation_recall", "fraction", (0, 1))
    success_rates = [
        summary["by_strategy"][strategy]["full_confirm_success_rate"]
        for strategy in strategies
    ]
    axes[1].bar(strategies, success_rates, color=[colors.get(s, "#64748b") for s in strategies], alpha=0.75)
    axes[1].set_title("Full-Confirm Success")
    axes[1].set_ylabel("rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)

    latency_rows: list[tuple[str, float]] = []
    for row in rows:
        for latency in row["reveal_to_confirm_latencies_steps"]:
            latency_rows.append((row["strategy"], float(latency)))
    for strategy in strategies:
        vals = [value for label, value in latency_rows if label == strategy and math.isfinite(value)]
        if vals:
            axes[2].hist(
                vals,
                bins=12,
                alpha=0.45,
                label=f"{strategy} mean={_mean(vals):.1f}",
                color=colors.get(strategy),
            )
            axes[2].axvline(_mean(vals), color=colors.get(strategy), linewidth=1.5)
    axes[2].set_title("Reveal-to-Confirm Latency")
    axes[2].set_xlabel("steps")
    axes[2].set_ylabel("survivors")
    axes[2].grid(alpha=0.25)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles and labels:
        axes[2].legend(fontsize=8)

    hist(axes[3], "UGV Path Length", "ugv_path_length_m", "m")
    hist(axes[4], "UGV Movement", "ugv_movement_m_per_ugv_step", "m / UGV-step")

    ax = axes[5]
    for strategy in strategies:
        bins = summary["by_strategy"][strategy].get("time_bins", [])
        if not bins:
            continue
        xs = [row["episode_fraction"] for row in bins]
        ax.plot(
            xs,
            [row["pending_known_survivors"] for row in bins],
            marker="o",
            label=f"{strategy} pending",
            color=colors.get(strategy),
        )
        ax.plot(
            xs,
            [row["new_confirmations_per_step"] for row in bins],
            marker="x",
            linestyle="--",
            label=f"{strategy} confirms",
            color=colors.get(strategy),
        )
    ax.set_title("Pending Targets / Confirm Events")
    ax.set_xlabel("episode fraction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("UGV Heuristic A* Baselines With Matched Reveal Timing", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['strategy']:>18s} seed {row['seed']:>4}: "
        f"confirmed={row['confirmed']}/{row['survivors']} "
        f"recall={row['confirmation_recall']:.3f} "
        f"success={int(row['full_confirm_success'])} "
        f"lat={row['avg_reveal_to_confirm_latency_steps']:.1f} steps "
        f"move={row['ugv_movement_m_per_ugv_step']:.2f}m/ugv-step "
        f"path={row['ugv_path_length_m']:.1f}m "
        f"pending={row['pending_target_time_fraction']:.2f}"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    print("-" * 96)
    for strategy in summary["strategies"]:
        item = summary["by_strategy"][strategy]
        print(
            f"{strategy:>18s}: "
            f"episodes={int(item['episodes'])} "
            f"confirm_recall={item['mean_confirmation_recall']:.3f} "
            f"confirm_auc={item['mean_confirmation_auc']:.3f} "
            f"success={item['full_confirm_success_rate']:.3f} "
            f"lat={item['mean_reveal_to_confirm_latency_steps']:.1f} steps "
            f"path={item['mean_ugv_path_length_m']:.1f}m "
            f"move={item['mean_ugv_movement_m_per_ugv_step']:.2f}m/ugv-step "
            f"pending={item['mean_pending_target_time_fraction']:.3f}"
        )


def _spec_metadata(spec: StrategySpec) -> dict[str, Any]:
    return {
        "label": spec.label,
        "name": spec.name,
        "checkpoint_dir": None if spec.checkpoint_dir is None else str(spec.checkpoint_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--happo-checkpoint", default=None,
                        help="Checkpoint models/ directory used by the 'happo' strategy token.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample HAPPO actions instead of using deterministic actor means.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1100)))
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--enable-fire", action="store_true")
    parser.add_argument("--disable-fire", action="store_true")
    parser.add_argument("--n-ugvs", "--n-ground", dest="n_ugvs", type=int, default=2)
    parser.add_argument("--survivor-reveal-initial-count", type=int, default=1)
    parser.add_argument("--survivor-reveal-start-step", type=int, default=10)
    parser.add_argument("--survivor-reveal-end-step", type=int, default=180)
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument("--ugv-planner-fire-mode", choices=("off", "cost", "block"), default=None)
    parser.add_argument(
        "--ugv-planner-fire-replan-policy",
        choices=("always", "affected", "lazy", "threshold_lazy"),
        default=None,
    )
    parser.add_argument("--ugv-planner-fire-replan-interval-steps", type=int, default=None)
    parser.add_argument("--ugv-planner-fire-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-fire-block-threshold", type=float, default=None)
    parser.add_argument("--ugv-planner-smoke-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-smolder-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-fire-buffer-m", type=float, default=None)
    parser.add_argument("--ugv-planner-fire-buffer-cost", type=float, default=None)
    parser.add_argument("--ugv-planner-land-cover-costs", type=float, nargs="+", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--plots-output", default=None)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.time_bins <= 0:
        parser.error("--time-bins must be positive")
    if args.n_ugvs < 1:
        parser.error("--n-ugvs must be positive")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    if args.enable_fire and args.disable_fire:
        parser.error("--enable-fire and --disable-fire are mutually exclusive")
    if args.survivor_reveal_initial_count < 0:
        parser.error("--survivor-reveal-initial-count must be nonnegative")
    if args.survivor_reveal_start_step < 0 or args.survivor_reveal_end_step < 0:
        parser.error("--survivor-reveal-start-step and --survivor-reveal-end-step must be nonnegative")
    if args.survivor_reveal_end_step < args.survivor_reveal_start_step:
        parser.error("--survivor-reveal-end-step must be >= --survivor-reveal-start-step")
    if args.terrain_cache_path is not None:
        args.terrain_cache_path = Path(args.terrain_cache_path)
        if not args.terrain_cache_path.is_file():
            parser.error(f"--terrain-cache-path does not exist: {args.terrain_cache_path}")
    for name in (
        "ugv_planner_fire_replan_interval_steps",
        "ugv_planner_fire_cost",
        "ugv_planner_fire_block_threshold",
        "ugv_planner_smoke_cost",
        "ugv_planner_smolder_cost",
        "ugv_planner_fire_buffer_m",
        "ugv_planner_fire_buffer_cost",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.ugv_planner_land_cover_costs is not None and len(args.ugv_planner_land_cover_costs) not in {5, 6}:
        parser.error("--ugv-planner-land-cover-costs must contain 5 or 6 values")
    return args


def main() -> None:
    args = _parse_args()
    try:
        specs = parse_strategy_specs(args.strategies, happo_checkpoint=args.happo_checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    scenario_kwargs = build_scenario_kwargs(args)
    print(
        "scenario: "
        f"{scenario_kwargs['n_drones']} UAVs, "
        f"{scenario_kwargs['n_ground']} UGVs, "
        f"{scenario_kwargs['n_survivors']} survivors, "
        f"steps={scenario_kwargs['max_steps']}, "
        f"delayed_reveal={scenario_kwargs['survivor_reveal_initial_count']} initial "
        f"{scenario_kwargs['survivor_reveal_start_step']}-{scenario_kwargs['survivor_reveal_end_step']}"
    )
    print(f"terrain: {scenario_kwargs.get('terrain_cache_path')}")
    print("strategies: " + ", ".join(
        spec.label if spec.checkpoint_dir is None else f"{spec.label}:{spec.checkpoint_dir}"
        for spec in specs
    ))
    print(f"seeds: {len(args.seeds)} ({args.seeds[0]}..{args.seeds[-1]})")
    print("-" * 96)

    json_path = Path(args.json_output) if args.json_output else None
    if json_path is not None:
        print(f"partial JSON checkpoint: {partial_json_path(json_path)}")

    rows: list[dict[str, Any]] = []
    happo_cache: dict[tuple[Path, bool], HappoPolicy] = {}
    total_rollouts = len(specs) * len(args.seeds)
    for spec in specs:
        for seed in args.seeds:
            row = run_rollout(
                spec,
                scenario_kwargs,
                seed,
                time_bins=args.time_bins,
                happo_cache=happo_cache,
                stochastic_happo=args.stochastic,
            )
            rows.append(row)
            _print_row(row)
            if json_path is not None:
                partial_summary = summarize(rows)
                write_partial_json(
                    json_path,
                    {
                        "scenario_kwargs": scenario_kwargs,
                        "metadata": {
                            "strategies": [_spec_metadata(item) for item in specs],
                            "happo_deterministic": not args.stochastic,
                            "steps": int(args.steps),
                            "seeds": [int(item) for item in args.seeds],
                            "scenario_kwargs": scenario_kwargs,
                        },
                        "rows": rows,
                        "summary": partial_summary,
                    },
                    completed_rollouts=len(rows),
                    total_rollouts=total_rollouts,
                )

    summary = summarize(rows)
    _print_summary(summary)
    payload = {
        "scenario_kwargs": scenario_kwargs,
        "metadata": {
            "strategies": [_spec_metadata(spec) for spec in specs],
            "happo_deterministic": not args.stochastic,
            "steps": int(args.steps),
            "seeds": [int(seed) for seed in args.seeds],
            "scenario_kwargs": scenario_kwargs,
        },
        "rows": rows,
        "summary": summary,
    }
    if json_path is not None:
        write_final_json(
            json_path,
            payload,
            completed_rollouts=len(rows),
            total_rollouts=total_rollouts,
        )
        print(f"wrote json: {json_path}")
    if args.plots_output:
        write_plots(rows, summary, Path(args.plots_output))
        print(f"wrote plots: {args.plots_output}")


if __name__ == "__main__":
    main()
