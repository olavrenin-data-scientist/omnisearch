"""
Diagnose whether a HAPPO UGV actor moves toward a known survivor.

This is intentionally narrower than full mission evaluation. It is meant for
the UGV-known-survivor diagnostic task: one ground robot, one survivor already
scouted/known at reset, and no fire.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import vmas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.happo_checkpoint import load_training_manifest
from agents.happo_policy import HappoPolicy, find_latest_happo_checkpoint
from envs.wildfire_search import WildfireSearchScenario


REWARD_COMPONENTS = (
    "team",
    "ground_confirm",
    "progress",
    "movement_align",
    "approach",
    "planner_progress",
    "stall_penalty",
    "travel_penalty",
)


def _checkpoint_path(path: str | None) -> Path:
    return Path(path) if path else find_latest_happo_checkpoint(ROOT / "results" / "harl_runs")


def _scenario_kwargs(checkpoint_dir: Path, args: argparse.Namespace) -> dict:
    manifest = load_training_manifest(checkpoint_dir)
    scenario_kwargs = {}
    if manifest is not None:
        scenario_kwargs.update(copy.deepcopy(manifest.get("env_args", {}).get("scenario_kwargs", {})))

    distance_kwargs = {}
    if args.ugv_diagnostic_target_distance_min_m is None and args.ugv_diagnostic_target_distance_max_m is None:
        pass
    else:
        target_distance_min_m = max(
            float(0.0 if args.ugv_diagnostic_target_distance_min_m is None else args.ugv_diagnostic_target_distance_min_m),
            0.0,
        )
        distance_kwargs["known_survivor_spawn_distance_min_m"] = target_distance_min_m
        if args.ugv_diagnostic_target_distance_max_m is not None:
            target_distance_max_m = max(
                float(args.ugv_diagnostic_target_distance_max_m),
                0.0,
            )
            if target_distance_max_m < target_distance_min_m:
                raise ValueError(
                    "ugv_diagnostic_target_distance_max_m must be >= "
                    "ugv_diagnostic_target_distance_min_m"
                )
            target_distance_m = 0.5 * (target_distance_min_m + target_distance_max_m)
            distance_kwargs.update({
                "known_survivor_spawn_distance_m": target_distance_m,
                "known_survivor_spawn_distance_max_m": target_distance_max_m,
            })
        for key in (
            "known_survivor_spawn_distance_m",
            "known_survivor_spawn_distance_min_m",
            "known_survivor_spawn_distance_max_m",
        ):
            scenario_kwargs.pop(key, None)

    scenario_kwargs.update({
        "max_steps": args.steps,
        "n_drones": 0,
        "n_ground": 1,
        "n_survivors": 1,
        "known_survivors_at_reset": True,
        "disable_fire": True,
        "comms_dropout": 0.0,
    })
    scenario_kwargs.update(distance_kwargs)
    if args.terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = args.terrain_cache_path
    if args.ground_min_confirm_radius_m is not None:
        scenario_kwargs.pop("ground_confirm_min", None)
        scenario_kwargs["ground_confirm_min_m"] = max(float(args.ground_min_confirm_radius_m), 0.0)
    if args.local_map_patch_size is not None:
        scenario_kwargs["local_map_patch_size"] = int(args.local_map_patch_size)
    if args.ugv_planner_hint is not None:
        scenario_kwargs["ugv_planner_hint"] = args.ugv_planner_hint.replace("-", "_")
    if args.ugv_planner_detour_obs is not None:
        scenario_kwargs["ugv_planner_detour_obs"] = bool(args.ugv_planner_detour_obs)
    if args.ugv_route_aware_reward is not None:
        scenario_kwargs["ugv_route_aware_reward"] = bool(args.ugv_route_aware_reward)
    if args.ugv_planner_patch_size is not None:
        scenario_kwargs["ugv_planner_patch_size"] = int(args.ugv_planner_patch_size)
    if args.ugv_planner_lookahead_cells is not None:
        scenario_kwargs["ugv_planner_lookahead_cells"] = int(args.ugv_planner_lookahead_cells)
    if "ugv_planner_lookahead_cells" in scenario_kwargs:
        patch_size = int(scenario_kwargs.get("ugv_planner_patch_size", 11))
        scenario_kwargs["ugv_planner_lookahead_cells"] = min(
            max(int(scenario_kwargs["ugv_planner_lookahead_cells"]), 1),
            max(patch_size // 2, 1),
        )
    return scenario_kwargs


def _distance_m(scenario: WildfireSearchScenario, ground_pos: torch.Tensor, survivor_pos: torch.Tensor) -> float:
    dist_sim = torch.linalg.norm(ground_pos - survivor_pos, dim=-1)
    scale = float(scenario.terrain_sim_units_per_meter[0])
    return float(dist_sim[0] / scale) if scale > 1e-9 else float(dist_sim[0])


def _distance_sim_to_m(scenario: WildfireSearchScenario, dist_sim: torch.Tensor) -> float:
    scale = float(scenario.terrain_sim_units_per_meter[0])
    return float(dist_sim[0] / scale) if scale > 1e-9 else float(dist_sim[0])


def _cosine_alignment(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True)
    if float(a_norm[0]) <= 1e-9 or float(b_norm[0]) <= 1e-9:
        return None
    value = (a / a_norm.clamp_min(1e-9) * b / b_norm.clamp_min(1e-9)).sum(dim=-1)
    return float(value[0])


def _action_alignment(action: torch.Tensor, ground_pos: torch.Tensor, survivor_pos: torch.Tensor) -> float | None:
    return _cosine_alignment(action, survivor_pos - ground_pos)


def _planner_hint_from_observation(
    scenario: WildfireSearchScenario,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, bool] | None:
    """Extract [unit_dx, unit_dy, ..., valid, direct_blocked] from one UGV obs."""
    if getattr(scenario, "ugv_planner_hint", "none") != "local_astar":
        return None
    patch_size = int(getattr(scenario, "local_map_patch_size", 3))
    offset = 4 + 12 + 1 + 2 * patch_size * patch_size + 9
    if obs.shape[-1] < offset + 5:
        return None
    hint = obs[:, offset : offset + 5]
    valid = bool((hint[0, 3] > 0.5).detach().cpu().item())
    if not valid:
        return None
    planner_vec = hint[:, :2]
    direct_blocked = bool((hint[0, 4] > 0.5).detach().cpu().item())
    return planner_vec, direct_blocked


def _shadow_astar_hint(
    scenario: WildfireSearchScenario,
    pos: torch.Tensor,
    target_pos: torch.Tensor,
) -> dict:
    """Compute the local A* hint without appending it to the actor observation."""
    out = {
        "valid": False,
        "unit": None,
        "waypoint_pos": None,
        "waypoint_distance_m": None,
        "direct_blocked": False,
        "detour_needed": False,
    }
    route = scenario._local_astar_route_for_env(0, pos[0], target_pos[0])
    if route is None:
        return out
    waypoint, direct_blocked, detour_needed = route
    waypoint_pos = scenario._grid_cell_center_to_world(
        waypoint,
        device=pos.device,
        dtype=pos.dtype,
    ).view(1, 2)
    delta = waypoint_pos - pos
    dist = torch.linalg.norm(delta, dim=-1)
    if float(dist[0].detach().cpu().item()) <= 1e-9:
        return out
    scale = float(scenario.terrain_sim_units_per_meter[0])
    out.update({
        "valid": True,
        "unit": delta / dist.clamp_min(1e-9).unsqueeze(-1),
        "waypoint_pos": waypoint_pos,
        "waypoint_distance_m": float(dist[0].detach().cpu().item()) / max(scale, 1e-9),
        "direct_blocked": bool(direct_blocked),
        "detour_needed": bool(detour_needed),
    })
    return out


def _empty_planner_bucket() -> dict:
    return {
        "steps": 0,
        "speed_sum": 0.0,
        "action_target_sum": 0.0,
        "action_target_count": 0,
        "movement_target_sum": 0.0,
        "movement_target_count": 0,
        "action_planner_sum": 0.0,
        "action_planner_count": 0,
        "movement_planner_sum": 0.0,
        "movement_planner_count": 0,
    }


def _add_optional_metric(bucket: dict, name: str, value: float | None) -> None:
    if value is None:
        return
    bucket[f"{name}_sum"] += float(value)
    bucket[f"{name}_count"] += 1


def _record_planner_bucket(
    bucket: dict,
    *,
    action_target: float | None,
    movement_target: float | None,
    action_planner: float | None,
    movement_planner: float | None,
    speed_mps: float,
) -> None:
    bucket["steps"] += 1
    bucket["speed_sum"] += float(speed_mps)
    _add_optional_metric(bucket, "action_target", action_target)
    _add_optional_metric(bucket, "movement_target", movement_target)
    _add_optional_metric(bucket, "action_planner", action_planner)
    _add_optional_metric(bucket, "movement_planner", movement_planner)


def _merge_planner_bucket(dst: dict, src: dict) -> None:
    for key, value in src.items():
        dst[key] += value


def _planner_metric(bucket: dict, name: str) -> float:
    count = bucket[f"{name}_count"]
    return float(bucket[f"{name}_sum"] / count) if count else 0.0


def _planner_speed(bucket: dict) -> float:
    return float(bucket["speed_sum"] / bucket["steps"]) if bucket["steps"] else 0.0


def _ensure_policy_rnn(policy: HappoPolicy, env) -> None:
    B = env.scenario.world.batch_dim
    if getattr(policy, "_rnn_states", None) is None or policy._rnn_states[0].shape[0] != B:
        policy._rnn_states = [
            np.zeros((B, policy._recurrent_n, policy._rnn_hidden_size), dtype=np.float32)
            for _ in env.agents
        ]


def _actor_distribution(policy: HappoPolicy, env, agent_idx: int = 0, *, return_obs: bool = False):
    """Return the current raw Gaussian action distribution for one actor."""
    _ensure_policy_rnn(policy, env)
    agent = env.agents[agent_idx]
    obs_tensor = env.scenario.observation(agent)
    obs = obs_tensor.cpu().numpy()
    action_dim = env.get_agent_action_size(agent)
    zeros = np.zeros((env.scenario.world.batch_dim, action_dim), dtype=np.float32)
    masks = np.ones((env.scenario.world.batch_dim, 1), dtype=np.float32)
    with torch.no_grad():
        _, _, dist = policy.actors[agent_idx].actor.evaluate_actions(
            obs,
            policy._rnn_states[agent_idx],
            zeros,
            masks,
            available_actions=None,
            active_masks=None,
        )
    if return_obs:
        return dist, obs_tensor
    return dist


def _angle_deg(vec: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def _angle_error_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    a_deg = _angle_deg(a)
    b_deg = _angle_deg(b)
    if a_deg is None or b_deg is None:
        return None
    return float(((a_deg - b_deg + 180.0) % 360.0) - 180.0)


def _ground_cell_diagnostics(scenario: WildfireSearchScenario, pos: torch.Tensor) -> dict:
    gx, gy = scenario._positions_to_grid(pos)
    speed = scenario._grid_values_at_positions(
        scenario.speed_multiplier_grid,
        pos.unsqueeze(1),
    ).squeeze(1)
    mobility = scenario._grid_values_at_positions(
        scenario.mobility_cost_grid,
        pos.unsqueeze(1),
    ).squeeze(1)
    traversable = scenario._grid_values_at_positions(
        scenario.traversable_grid,
        pos.unsqueeze(1),
    ).squeeze(1)
    blocked_patch = (~scenario._local_grid_patch(
        scenario.traversable_grid,
        pos,
        scenario.local_map_patch_size,
    )).float()
    mobility_patch = scenario._local_grid_patch(
        scenario.mobility_cost_grid,
        pos,
        scenario.local_map_patch_size,
    )
    return {
        "grid_x": int(gx[0].detach().cpu().item()),
        "grid_y": int(gy[0].detach().cpu().item()),
        "cell_speed": float(speed[0].detach().cpu().item()),
        "cell_mobility": float(mobility[0].detach().cpu().item()),
        "cell_traversable": bool(traversable[0].detach().cpu().item()),
        "blocked_frac": float(blocked_patch[0].mean().detach().cpu().item()),
        "blocked_count": int(blocked_patch[0].sum().detach().cpu().item()),
        "mobility_patch_mean": float(mobility_patch[0].mean().detach().cpu().item()),
        "mobility_patch_max": float(mobility_patch[0].max().detach().cpu().item()),
    }


def _tensor_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _metric_scalar(scenario: WildfireSearchScenario, name: str) -> float:
    value = getattr(scenario, name, None)
    if value is None:
        return 0.0
    return _tensor_scalar(value)


def _reward_components(scenario: WildfireSearchScenario) -> dict[str, float]:
    travel_penalty = 0.0
    if getattr(scenario, "n_ground", 0) > 0:
        travel_penalty = (
            _tensor_scalar(scenario.step_ugv_travel_cost[0, 0])
            * float(getattr(scenario, "r_ground_travel_cost", 0.0))
        )
    return {
        "team": _metric_scalar(scenario, "metric_reward_team"),
        "ground_confirm": _metric_scalar(scenario, "metric_reward_ground_confirm"),
        "progress": _metric_scalar(scenario, "metric_reward_ugv_progress"),
        "movement_align": _metric_scalar(scenario, "metric_reward_ugv_movement_alignment"),
        "approach": _metric_scalar(scenario, "metric_reward_ugv_approach"),
        "planner_progress": _metric_scalar(scenario, "metric_reward_ugv_planner_progress"),
        "stall_penalty": _metric_scalar(scenario, "metric_reward_ugv_stall_penalty"),
        "travel_penalty": travel_penalty,
    }


def _new_time_series() -> dict:
    return {
        "step": [],
        "episode_fraction": [],
        "distance_m": [],
        "progress_m": [],
        "movement_m": [],
        "speed_mps": [],
        "action_norm": [],
        "raw_mean_norm": [],
        "raw_oob": [],
        "action_saturated": [],
        "action_target_alignment": [],
        "movement_target_alignment": [],
        "action_movement_alignment": [],
        "blocked": [],
        "speed_limited": [],
        "near_zero": [],
        "path_speed": [],
        "speed_limit_scale": [],
        "motion_correction_m": [],
        "within_confirm_range": [],
        "shadow_astar_valid": [],
        "shadow_astar_direct_blocked": [],
        "shadow_astar_detour_needed": [],
        "shadow_astar_waypoint_distance_m": [],
        "shadow_astar_action_alignment": [],
        "shadow_astar_movement_alignment": [],
        "shadow_astar_progress_m": [],
        "reward": {name: [] for name in REWARD_COMPONENTS},
    }


def _append_optional(series: list, value: float | None) -> None:
    series.append(None if value is None else float(value))


def _finite(values) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _finite_mean(values) -> float:
    xs = _finite(values)
    return float(np.mean(xs)) if xs else float("nan")


def _finite_median(values) -> float:
    xs = _finite(values)
    return float(np.median(xs)) if xs else float("nan")


def _finite_sum(values) -> float:
    xs = _finite(values)
    return float(np.sum(xs)) if xs else 0.0


def _json_sanitize(value):
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, torch.Tensor):
        return _json_sanitize(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _time_bin_summary(rows: list[dict], bins: int) -> list[dict]:
    bins = max(int(bins), 1)
    bucket_rows: list[list[dict]] = [[] for _ in range(bins)]
    for row in rows:
        ts = row.get("time_series", {})
        fractions = ts.get("episode_fraction", [])
        rewards = ts.get("reward", {})
        for i, frac in enumerate(fractions):
            if frac is None:
                continue
            bin_idx = min(max(int(float(frac) * bins), 0), bins - 1)
            step = {
                "distance_m": _series_at(ts, "distance_m", i),
                "progress_m": _series_at(ts, "progress_m", i),
                "movement_m": _series_at(ts, "movement_m", i),
                "speed_mps": _series_at(ts, "speed_mps", i),
                "action_norm": _series_at(ts, "action_norm", i),
                "raw_mean_norm": _series_at(ts, "raw_mean_norm", i),
                "raw_oob": _series_at(ts, "raw_oob", i),
                "action_saturated": _series_at(ts, "action_saturated", i),
                "action_target_alignment": _series_at(ts, "action_target_alignment", i),
                "movement_target_alignment": _series_at(ts, "movement_target_alignment", i),
                "action_movement_alignment": _series_at(ts, "action_movement_alignment", i),
                "blocked": _series_at(ts, "blocked", i),
                "speed_limited": _series_at(ts, "speed_limited", i),
                "near_zero": _series_at(ts, "near_zero", i),
                "path_speed": _series_at(ts, "path_speed", i),
                "speed_limit_scale": _series_at(ts, "speed_limit_scale", i),
                "motion_correction_m": _series_at(ts, "motion_correction_m", i),
                "within_confirm_range": _series_at(ts, "within_confirm_range", i),
                "shadow_astar_valid": _series_at(ts, "shadow_astar_valid", i),
                "shadow_astar_direct_blocked": _series_at(ts, "shadow_astar_direct_blocked", i),
                "shadow_astar_detour_needed": _series_at(ts, "shadow_astar_detour_needed", i),
                "shadow_astar_waypoint_distance_m": _series_at(ts, "shadow_astar_waypoint_distance_m", i),
                "shadow_astar_action_alignment": _series_at(ts, "shadow_astar_action_alignment", i),
                "shadow_astar_movement_alignment": _series_at(ts, "shadow_astar_movement_alignment", i),
                "shadow_astar_progress_m": _series_at(ts, "shadow_astar_progress_m", i),
            }
            for name in REWARD_COMPONENTS:
                step[f"reward_abs_{name}"] = _series_at(rewards, name, i, absolute=True)
                step[f"reward_{name}"] = _series_at(rewards, name, i)
            bucket_rows[bin_idx].append(step)

    out = []
    for idx, bucket in enumerate(bucket_rows):
        row = {
            "bin": idx,
            "episode_fraction_mid": (idx + 0.5) / bins,
            "n_steps": len(bucket),
        }
        keys = (
            "distance_m",
            "progress_m",
            "movement_m",
            "speed_mps",
            "action_norm",
            "raw_mean_norm",
            "raw_oob",
            "action_saturated",
            "action_target_alignment",
            "movement_target_alignment",
            "action_movement_alignment",
            "blocked",
            "speed_limited",
            "near_zero",
            "path_speed",
            "speed_limit_scale",
            "motion_correction_m",
            "within_confirm_range",
            "shadow_astar_valid",
            "shadow_astar_direct_blocked",
            "shadow_astar_detour_needed",
            "shadow_astar_waypoint_distance_m",
            "shadow_astar_action_alignment",
            "shadow_astar_movement_alignment",
            "shadow_astar_progress_m",
        )
        for key in keys:
            row[key] = _finite_mean(step.get(key) for step in bucket)
        for name in REWARD_COMPONENTS:
            row[f"reward_abs_{name}"] = _finite_mean(
                step.get(f"reward_abs_{name}") for step in bucket
            )
            row[f"reward_{name}"] = _finite_mean(
                step.get(f"reward_{name}") for step in bucket
            )
        out.append(row)
    return out


def _series_at(container: dict, key: str, index: int, *, absolute: bool = False):
    values = container.get(key, [])
    if index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    value = float(value)
    return abs(value) if absolute else value


def _summarize_rows(rows: list[dict], bins: int) -> dict:
    confirmation_steps = [
        row["confirmation_step"]
        for row in rows
        if row.get("confirmation_step") is not None
    ]
    path_efficiencies = [
        row["path_efficiency"]
        for row in rows
        if row.get("path_efficiency") is not None
    ]
    return {
        "episodes": len(rows),
        "success_rate": _finite_mean(row["full_success"] for row in rows),
        "mean_confirmed": _finite_mean(row["confirmed"] for row in rows),
        "mean_initial_distance_m": _finite_mean(row["initial_distance_m"] for row in rows),
        "mean_final_distance_m": _finite_mean(row["final_distance_m"] for row in rows),
        "mean_min_distance_m": _finite_mean(row["min_distance_m"] for row in rows),
        "mean_confirmation_step_successes": _finite_mean(confirmation_steps),
        "median_confirmation_step_successes": _finite_median(confirmation_steps),
        "mean_confirmation_time_s_successes": _finite_mean(
            row["confirmation_time_s"]
            for row in rows
            if row.get("confirmation_time_s") is not None
        ),
        "mean_path_length_m": _finite_mean(row["path_length_m"] for row in rows),
        "mean_path_efficiency_successes_or_progress": _finite_mean(path_efficiencies),
        "mean_path_to_initial_ratio": _finite_mean(row["path_to_initial_ratio"] for row in rows),
        "mean_action_target_alignment": _finite_mean(
            row["mean_action_target_alignment"] for row in rows
        ),
        "mean_displacement_target_alignment": _finite_mean(
            row["mean_displacement_target_alignment"] for row in rows
        ),
        "mean_action_displacement_alignment": _finite_mean(
            row["mean_action_displacement_alignment"] for row in rows
        ),
        "mean_speed_mps": _finite_mean(row["mean_speed_mps"] for row in rows),
        "mean_frac_raw_mean_oob": _finite_mean(row["frac_raw_mean_oob"] for row in rows),
        "mean_frac_action_saturated": _finite_mean(row["frac_action_saturated"] for row in rows),
        "mean_frac_proposed_path_blocked": _finite_mean(
            row["frac_proposed_path_blocked"] for row in rows
        ),
        "mean_frac_speed_limited": _finite_mean(row["frac_speed_limited"] for row in rows),
        "mean_motion_correction_m": _finite_mean(row["mean_motion_correction_m"] for row in rows),
        "mean_shadow_astar_valid_fraction": _finite_mean(
            row["shadow_astar_valid_fraction"] for row in rows
        ),
        "mean_shadow_astar_direct_blocked_fraction": _finite_mean(
            row["shadow_astar_direct_blocked_fraction"] for row in rows
        ),
        "mean_shadow_astar_detour_needed_fraction": _finite_mean(
            row["shadow_astar_detour_needed_fraction"] for row in rows
        ),
        "mean_shadow_astar_action_alignment": _finite_mean(
            row["mean_shadow_astar_action_alignment"] for row in rows
        ),
        "mean_shadow_astar_movement_alignment": _finite_mean(
            row["mean_shadow_astar_movement_alignment"] for row in rows
        ),
        "mean_shadow_astar_progress_m": _finite_mean(
            row["mean_shadow_astar_progress_m"] for row in rows
        ),
        "time_bins": _time_bin_summary(rows, bins),
    }


def _plot_hist_by_success(ax, rows: list[dict], key: str, title: str, xlabel: str) -> None:
    success = _finite(row.get(key) for row in rows if row.get("full_success", 0.0) > 0.0)
    failure = _finite(row.get(key) for row in rows if row.get("full_success", 0.0) <= 0.0)
    data = []
    labels = []
    colors = []
    if failure:
        data.append(failure)
        labels.append("failure")
        colors.append("#ef4444")
    if success:
        data.append(success)
        labels.append("success")
        colors.append("#22c55e")
    if data:
        ax.hist(data, bins=16, alpha=0.65, label=labels, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("episodes")
    if len(data) > 1:
        ax.legend(fontsize=8)
    all_values = failure + success
    if all_values:
        ax.axvline(float(np.mean(all_values)), color="black", lw=1.2, label="mean")


def _plot_ugv_diagnostics(
    rows: list[dict],
    summary: dict,
    output_path: Path,
    *,
    deterministic: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 3, figsize=(15, 21), constrained_layout=True)
    axes = axes.reshape(-1)

    ax = axes[0]
    successes = int(sum(1 for row in rows if row.get("full_success", 0.0) > 0.0))
    failures = len(rows) - successes
    ax.bar(["success", "failure"], [successes, failures], color=["#22c55e", "#ef4444"], alpha=0.8)
    ax.set_title("Confirmation Outcome")
    ax.set_ylabel("episodes")
    ax.text(
        0.02,
        0.95,
        f"success={summary['success_rate']:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    _plot_hist_by_success(axes[1], rows, "final_distance_m", "Final Distance", "m")
    _plot_hist_by_success(axes[2], rows, "min_distance_m", "Min Distance", "m")
    _plot_hist_by_success(axes[3], rows, "confirmation_step", "Time To Confirm", "step")
    _plot_hist_by_success(axes[4], rows, "path_length_m", "Path Length", "m")
    _plot_hist_by_success(axes[5], rows, "path_to_initial_ratio", "Path / Initial Distance", "ratio")
    _plot_hist_by_success(axes[6], rows, "mean_speed_mps", "Speed", "m/s")
    _plot_hist_by_success(axes[7], rows, "mean_action_target_alignment", "Action Target Alignment", "cosine")

    ax = axes[8]
    metrics = [
        ("blocked", "frac_proposed_path_blocked"),
        ("speedlim", "frac_speed_limited"),
        ("near0", "frac_near_zero_displacement"),
        ("raw_oob", "frac_raw_mean_oob"),
        ("sat", "frac_action_saturated"),
    ]
    ax.bar(
        [label for label, _ in metrics],
        [_finite_mean(row.get(key) for row in rows) for _, key in metrics],
        color="#64748b",
        alpha=0.75,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Constraint / Action Health")
    ax.set_ylabel("fraction")

    bins = summary.get("time_bins", [])
    x = [b["episode_fraction_mid"] for b in bins]

    ax = axes[9]
    ax.plot(x, [b["distance_m"] for b in bins], marker="o", label="distance", color="#2563eb")
    ax.set_title("Time-Bin Distance / Progress")
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("distance (m)")
    ax2 = ax.twinx()
    ax2.plot(x, [b["progress_m"] for b in bins], marker="o", label="progress", color="#16a34a")
    ax2.set_ylabel("progress (m/step)")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[10]
    ax.plot(x, [b["speed_mps"] for b in bins], marker="o", label="speed", color="#2563eb")
    ax.plot(x, [b["movement_m"] for b in bins], marker="o", label="move/step", color="#0f766e")
    ax.set_title("Time-Bin Movement")
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("m/s or m/step")
    ax2 = ax.twinx()
    ax2.plot(x, [b["blocked"] for b in bins], marker="o", label="blocked", color="#ef4444")
    ax2.plot(x, [b["speed_limited"] for b in bins], marker="o", label="speedlim", color="#f97316")
    ax2.plot(x, [b["near_zero"] for b in bins], marker="o", label="near0", color="#a855f7")
    ax2.set_ylabel("fraction")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[11]
    reward_colors = {
        "team": "#2563eb",
        "ground_confirm": "#22c55e",
        "progress": "#0f766e",
        "movement_align": "#84cc16",
        "approach": "#eab308",
        "planner_progress": "#8b5cf6",
        "stall_penalty": "#ef4444",
        "travel_penalty": "#111827",
    }
    for name in REWARD_COMPONENTS:
        y = [b[f"reward_abs_{name}"] for b in bins]
        if any(math.isfinite(float(v)) and abs(float(v)) > 1e-12 for v in y if v is not None):
            ax.plot(
                x,
                y,
                marker="o",
                label=name.replace("_", " "),
                color=reward_colors.get(name),
            )
    ax.set_title("Time-Bin Reward Scale (mean abs)")
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("abs reward/penalty per step")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[12]
    shadow_metrics = [
        ("valid", "shadow_astar_valid_fraction"),
        ("direct blocked", "shadow_astar_direct_blocked_fraction"),
        ("detour", "shadow_astar_detour_needed_fraction"),
    ]
    ax.bar(
        [label for label, _ in shadow_metrics],
        [_finite_mean(row.get(key) for row in rows) for _, key in shadow_metrics],
        color="#6366f1",
        alpha=0.75,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Shadow A* Route State")
    ax.set_ylabel("fraction")

    _plot_hist_by_success(
        axes[13],
        rows,
        "mean_shadow_astar_movement_alignment",
        "Movement vs Shadow A*",
        "cosine",
    )

    ax = axes[14]
    ax.plot(x, [b["shadow_astar_action_alignment"] for b in bins], marker="o", label="action align", color="#2563eb")
    ax.plot(x, [b["shadow_astar_movement_alignment"] for b in bins], marker="o", label="move align", color="#16a34a")
    ax.plot(x, [b["shadow_astar_progress_m"] for b in bins], marker="o", label="progress m", color="#0f766e")
    ax.set_title("Time-Bin Shadow A*")
    ax.set_xlabel("episode fraction")
    ax.set_ylabel("cosine or m/step")
    ax2 = ax.twinx()
    ax2.plot(x, [b["shadow_astar_valid"] for b in bins], marker="o", label="valid", color="#64748b")
    ax2.plot(x, [b["shadow_astar_direct_blocked"] for b in bins], marker="o", label="direct blocked", color="#ef4444")
    ax2.plot(x, [b["shadow_astar_detour_needed"] for b in bins], marker="o", label="detour", color="#f97316")
    ax2.set_ylabel("fraction")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7, ncol=2)

    fig.suptitle(
        f"UGV HAPPO Diagnostics ({'deterministic' if deterministic else 'stochastic'}, n={len(rows)})",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_rollout(
    checkpoint_dir: Path,
    scenario_kwargs: dict,
    seed: int,
    deterministic: bool,
    *,
    shadow_planner: bool = True,
) -> dict:
    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=deterministic)
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
    ground = env.agents[0]
    survivor = scenario._survivors[0]

    initial_distance = _distance_m(scenario, ground.state.pos, survivor.state.pos)
    min_distance = initial_distance
    alignments: list[float] = []
    displacement_target_alignments: list[float] = []
    action_displacement_alignments: list[float] = []
    displacement_meters: list[float] = []
    action_norms: list[float] = []
    saturated_actions: list[bool] = []
    raw_mean_norms: list[float] = []
    raw_mean_abs_max: list[float] = []
    raw_mean_oob: list[bool] = []
    raw_stds: list[float] = []
    proposed_path_blocked: list[bool] = []
    speed_limited: list[bool] = []
    path_speeds: list[float] = []
    speed_limit_scales: list[float] = []
    proposed_displacements_m: list[float] = []
    corrected_displacements_m: list[float] = []
    actual_displacements_m: list[float] = []
    motion_corrections_m: list[float] = []
    planner_buckets = {
        "all": _empty_planner_bucket(),
        "clear": _empty_planner_bucket(),
        "blocked": _empty_planner_bucket(),
    }
    shadow_planner_buckets = {
        "all": _empty_planner_bucket(),
        "clear": _empty_planner_bucket(),
        "blocked": _empty_planner_bucket(),
        "detour": _empty_planner_bucket(),
    }
    planner_total_steps = 0
    shadow_planner_total_steps = 0
    shadow_astar_valid: list[bool] = []
    shadow_astar_direct_blocked: list[bool] = []
    shadow_astar_detour_needed: list[bool] = []
    shadow_astar_waypoint_distances_m: list[float] = []
    shadow_astar_action_alignments: list[float] = []
    shadow_astar_movement_alignments: list[float] = []
    shadow_astar_progress_meters: list[float] = []
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    max_steps = int(scenario_kwargs["max_steps"])
    time_series = _new_time_series()
    confirmation_step: int | None = None

    for step in range(max_steps):
        pos_before = ground.state.pos.clone()
        survivor_before = survivor.state.pos.clone()
        dist_before_m = _distance_m(scenario, pos_before, survivor_before)
        shadow_hint = _shadow_astar_hint(scenario, pos_before, survivor_before) if shadow_planner else None
        dist, obs_before = _actor_distribution(policy, env, agent_idx=0, return_obs=True)
        planner_hint = _planner_hint_from_observation(scenario, obs_before)
        raw_mean = dist.mean.detach().cpu().numpy()
        raw_std = dist.stddev.detach().cpu().numpy()
        raw_mean_norm = float(np.linalg.norm(raw_mean[0]))
        raw_mean_norms.append(raw_mean_norm)
        raw_mean_abs_max.append(float(np.max(np.abs(raw_mean[0]))))
        raw_oob = bool(np.any(np.abs(raw_mean[0]) > 1.0))
        raw_mean_oob.append(raw_oob)
        raw_stds.append(float(np.mean(raw_std[0])))
        actions = policy(env)
        action_target_alignment = _action_alignment(actions[0], ground.state.pos, survivor.state.pos)
        if action_target_alignment is not None:
            alignments.append(action_target_alignment)
        action_planner_alignment = None
        if planner_hint is not None:
            planner_vec, direct_blocked = planner_hint
            action_planner_alignment = _cosine_alignment(actions[0], planner_vec)
        action_shadow_alignment = None
        if shadow_hint and shadow_hint["valid"]:
            action_shadow_alignment = _cosine_alignment(actions[0], shadow_hint["unit"])
        action_norm = float(torch.linalg.norm(actions[0], dim=-1)[0])
        action_norms.append(action_norm)
        saturated = bool((actions[0].abs() >= 0.98).any().item())
        saturated_actions.append(saturated)
        env.step(actions)
        blocked = bool(scenario.step_ugv_proposed_path_blocked[0, 0].item())
        proposed_path_blocked.append(blocked)
        limited = bool(scenario.step_ugv_speed_limited[0, 0].item())
        speed_limited.append(limited)
        path_speed = float(scenario.step_ugv_path_speed[0, 0].detach().cpu().item())
        path_speeds.append(path_speed)
        speed_limit_scale = float(scenario.step_ugv_speed_limit_scale[0, 0].detach().cpu().item())
        speed_limit_scales.append(speed_limit_scale)
        proposed_displacement_m = float(
            scenario.step_ugv_proposed_displacement_m[0, 0].detach().cpu().item()
        )
        proposed_displacements_m.append(proposed_displacement_m)
        corrected_displacement_m = float(
            scenario.step_ugv_corrected_displacement_m[0, 0].detach().cpu().item()
        )
        corrected_displacements_m.append(corrected_displacement_m)
        actual_displacement_m = float(
            scenario.step_ugv_actual_displacement_m[0, 0].detach().cpu().item()
        )
        actual_displacements_m.append(actual_displacement_m)
        motion_correction_m = float(
            scenario.step_ugv_motion_correction_m[0, 0].detach().cpu().item()
        )
        motion_corrections_m.append(motion_correction_m)
        displacement = ground.state.pos - pos_before
        target_before = survivor_before - pos_before
        disp_alignment = _cosine_alignment(displacement, target_before)
        if disp_alignment is not None:
            displacement_target_alignments.append(disp_alignment)
        action_disp_alignment = _cosine_alignment(actions[0], displacement)
        if action_disp_alignment is not None:
            action_displacement_alignments.append(action_disp_alignment)
        displacement_m = _distance_sim_to_m(scenario, torch.linalg.norm(displacement, dim=-1))
        displacement_meters.append(displacement_m)
        dist_after_m = _distance_m(scenario, ground.state.pos, survivor.state.pos)
        shadow_movement_alignment = None
        shadow_progress_m = None
        shadow_valid = bool(shadow_hint and shadow_hint["valid"])
        shadow_direct_blocked = bool(shadow_hint["direct_blocked"]) if shadow_valid else False
        shadow_detour_needed = bool(shadow_hint["detour_needed"]) if shadow_valid else False
        shadow_waypoint_distance_m = (
            float(shadow_hint["waypoint_distance_m"])
            if shadow_valid and shadow_hint["waypoint_distance_m"] is not None
            else None
        )
        if shadow_valid:
            shadow_movement_alignment = _cosine_alignment(displacement, shadow_hint["unit"])
            scale = float(scenario.terrain_sim_units_per_meter[0])
            waypoint_pos = shadow_hint["waypoint_pos"]
            before_wp_m = float(torch.linalg.norm(waypoint_pos - pos_before, dim=-1)[0].detach().cpu().item()) / max(scale, 1e-9)
            after_wp_m = float(torch.linalg.norm(waypoint_pos - ground.state.pos, dim=-1)[0].detach().cpu().item()) / max(scale, 1e-9)
            shadow_progress_m = before_wp_m - after_wp_m
            shadow_astar_valid.append(True)
            shadow_astar_direct_blocked.append(shadow_direct_blocked)
            shadow_astar_detour_needed.append(shadow_detour_needed)
            if shadow_waypoint_distance_m is not None:
                shadow_astar_waypoint_distances_m.append(shadow_waypoint_distance_m)
            if action_shadow_alignment is not None:
                shadow_astar_action_alignments.append(action_shadow_alignment)
            if shadow_movement_alignment is not None:
                shadow_astar_movement_alignments.append(shadow_movement_alignment)
            shadow_astar_progress_meters.append(shadow_progress_m)
            shadow_planner_total_steps += 1
            speed_mps = displacement_m / step_seconds
            for bucket_name in ("all", "blocked" if shadow_direct_blocked else "clear"):
                _record_planner_bucket(
                    shadow_planner_buckets[bucket_name],
                    action_target=action_target_alignment,
                    movement_target=disp_alignment,
                    action_planner=action_shadow_alignment,
                    movement_planner=shadow_movement_alignment,
                    speed_mps=speed_mps,
                )
            if shadow_detour_needed:
                _record_planner_bucket(
                    shadow_planner_buckets["detour"],
                    action_target=action_target_alignment,
                    movement_target=disp_alignment,
                    action_planner=action_shadow_alignment,
                    movement_planner=shadow_movement_alignment,
                    speed_mps=speed_mps,
                )
        else:
            shadow_astar_valid.append(False)
            shadow_astar_direct_blocked.append(False)
            shadow_astar_detour_needed.append(False)
        reward_components = _reward_components(scenario)
        time_series["step"].append(step)
        time_series["episode_fraction"].append((step + 1) / max_steps)
        time_series["distance_m"].append(dist_after_m)
        time_series["progress_m"].append(dist_before_m - dist_after_m)
        time_series["movement_m"].append(displacement_m)
        time_series["speed_mps"].append(displacement_m / step_seconds)
        time_series["action_norm"].append(action_norm)
        time_series["raw_mean_norm"].append(raw_mean_norm)
        time_series["raw_oob"].append(float(raw_oob))
        time_series["action_saturated"].append(float(saturated))
        _append_optional(time_series["action_target_alignment"], action_target_alignment)
        _append_optional(time_series["movement_target_alignment"], disp_alignment)
        _append_optional(time_series["action_movement_alignment"], action_disp_alignment)
        time_series["blocked"].append(float(blocked))
        time_series["speed_limited"].append(float(limited))
        time_series["near_zero"].append(float(displacement_m < 0.05))
        time_series["path_speed"].append(path_speed)
        time_series["speed_limit_scale"].append(speed_limit_scale)
        time_series["motion_correction_m"].append(motion_correction_m)
        time_series["within_confirm_range"].append(
            _metric_scalar(scenario, "metric_ugv_within_confirm_range")
        )
        time_series["shadow_astar_valid"].append(float(shadow_valid))
        time_series["shadow_astar_direct_blocked"].append(float(shadow_direct_blocked))
        time_series["shadow_astar_detour_needed"].append(float(shadow_detour_needed))
        _append_optional(time_series["shadow_astar_waypoint_distance_m"], shadow_waypoint_distance_m)
        _append_optional(time_series["shadow_astar_action_alignment"], action_shadow_alignment)
        _append_optional(time_series["shadow_astar_movement_alignment"], shadow_movement_alignment)
        _append_optional(time_series["shadow_astar_progress_m"], shadow_progress_m)
        for name in REWARD_COMPONENTS:
            time_series["reward"][name].append(reward_components.get(name, 0.0))
        if planner_hint is not None:
            movement_planner_alignment = _cosine_alignment(displacement, planner_vec)
            speed_mps = displacement_m / step_seconds
            planner_total_steps += 1
            for bucket_name in ("all", "blocked" if direct_blocked else "clear"):
                _record_planner_bucket(
                    planner_buckets[bucket_name],
                    action_target=action_target_alignment,
                    movement_target=disp_alignment,
                    action_planner=action_planner_alignment,
                    movement_planner=movement_planner_alignment,
                    speed_mps=speed_mps,
                )
        min_distance = min(min_distance, dist_after_m)
        if bool(scenario.found_survivors[0, 0]):
            confirmation_step = step
            break

    final_distance = _distance_m(scenario, ground.state.pos, survivor.state.pos)
    path_length_m = _finite_sum(displacement_meters)
    direct_progress_m = initial_distance - final_distance
    path_efficiency = (
        path_length_m / direct_progress_m
        if direct_progress_m > 1e-6
        else None
    )
    confirmation_time_s = (
        (confirmation_step + 1) * step_seconds
        if confirmation_step is not None
        else None
    )
    return {
        "seed": seed,
        "confirmed": float(scenario.found_survivors[0].sum().item()),
        "full_success": float(bool(scenario.found_survivors[0].all())),
        "initial_distance_m": initial_distance,
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
        "confirmation_step": confirmation_step,
        "confirmation_time_s": confirmation_time_s,
        "episode_steps": len(time_series["step"]),
        "path_length_m": path_length_m,
        "direct_progress_m": direct_progress_m,
        "path_efficiency": path_efficiency,
        "path_to_initial_ratio": path_length_m / max(initial_distance, 1e-6),
        "mean_action_target_alignment": float(np.mean(alignments)) if alignments else 0.0,
        "frac_action_toward_target": float(np.mean([a > 0.0 for a in alignments])) if alignments else 0.0,
        "mean_displacement_target_alignment": (
            float(np.mean(displacement_target_alignments)) if displacement_target_alignments else 0.0
        ),
        "frac_displacement_toward_target": (
            float(np.mean([a > 0.0 for a in displacement_target_alignments]))
            if displacement_target_alignments else 0.0
        ),
        "mean_action_displacement_alignment": (
            float(np.mean(action_displacement_alignments)) if action_displacement_alignments else 0.0
        ),
        "frac_action_displacement_disagree": (
            float(np.mean([a < 0.5 for a in action_displacement_alignments]))
            if action_displacement_alignments else 0.0
        ),
        "mean_displacement_m": float(np.mean(displacement_meters)) if displacement_meters else 0.0,
        "mean_speed_mps": (
            float(np.mean(displacement_meters) / step_seconds) if displacement_meters else 0.0
        ),
        "frac_near_zero_displacement": (
            float(np.mean([d < 0.05 for d in displacement_meters])) if displacement_meters else 0.0
        ),
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "frac_action_saturated": float(np.mean(saturated_actions)) if saturated_actions else 0.0,
        "mean_raw_mean_norm": float(np.mean(raw_mean_norms)) if raw_mean_norms else 0.0,
        "mean_raw_mean_abs_max": float(np.mean(raw_mean_abs_max)) if raw_mean_abs_max else 0.0,
        "frac_raw_mean_oob": float(np.mean(raw_mean_oob)) if raw_mean_oob else 0.0,
        "mean_raw_std": float(np.mean(raw_stds)) if raw_stds else 0.0,
        "frac_proposed_path_blocked": float(np.mean(proposed_path_blocked)) if proposed_path_blocked else 0.0,
        "frac_speed_limited": float(np.mean(speed_limited)) if speed_limited else 0.0,
        "mean_path_speed": float(np.mean(path_speeds)) if path_speeds else 0.0,
        "mean_speed_limit_scale": float(np.mean(speed_limit_scales)) if speed_limit_scales else 0.0,
        "mean_proposed_displacement_m": (
            float(np.mean(proposed_displacements_m)) if proposed_displacements_m else 0.0
        ),
        "mean_corrected_displacement_m": (
            float(np.mean(corrected_displacements_m)) if corrected_displacements_m else 0.0
        ),
        "mean_actual_displacement_m": (
            float(np.mean(actual_displacements_m)) if actual_displacements_m else 0.0
        ),
        "mean_motion_correction_m": float(np.mean(motion_corrections_m)) if motion_corrections_m else 0.0,
        "planner_total_steps": planner_total_steps,
        "planner_buckets": planner_buckets,
        "shadow_planner_total_steps": shadow_planner_total_steps,
        "shadow_planner_buckets": shadow_planner_buckets,
        "shadow_astar_valid_fraction": (
            float(np.mean(shadow_astar_valid)) if shadow_astar_valid else 0.0
        ),
        "shadow_astar_direct_blocked_fraction": (
            float(np.mean(shadow_astar_direct_blocked)) if shadow_astar_direct_blocked else 0.0
        ),
        "shadow_astar_detour_needed_fraction": (
            float(np.mean(shadow_astar_detour_needed)) if shadow_astar_detour_needed else 0.0
        ),
        "mean_shadow_astar_waypoint_distance_m": (
            float(np.mean(shadow_astar_waypoint_distances_m)) if shadow_astar_waypoint_distances_m else 0.0
        ),
        "mean_shadow_astar_action_alignment": (
            float(np.mean(shadow_astar_action_alignments)) if shadow_astar_action_alignments else 0.0
        ),
        "mean_shadow_astar_movement_alignment": (
            float(np.mean(shadow_astar_movement_alignments)) if shadow_astar_movement_alignments else 0.0
        ),
        "mean_shadow_astar_progress_m": (
            float(np.mean(shadow_astar_progress_meters)) if shadow_astar_progress_meters else 0.0
        ),
        "time_series": time_series,
    }


def run_failure_trace(
    checkpoint_dir: Path,
    scenario_kwargs: dict,
    seed: int,
    deterministic: bool,
) -> dict:
    """Run one rollout and keep per-step diagnostics for debugging failures."""
    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=deterministic)
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
    ground = env.agents[0]
    survivor = scenario._survivors[0]
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)

    trace = []
    initial_distance = _distance_m(scenario, ground.state.pos, survivor.state.pos)
    min_distance = initial_distance
    for step in range(scenario_kwargs["max_steps"]):
        pos_before = ground.state.pos.clone()
        survivor_before = survivor.state.pos.clone()
        dist_before = _distance_m(scenario, pos_before, survivor_before)
        cell_before = _ground_cell_diagnostics(scenario, pos_before)

        dist = _actor_distribution(policy, env, agent_idx=0)
        raw_mean = dist.mean.detach().cpu().numpy()[0]
        raw_std = dist.stddev.detach().cpu().numpy()[0]
        actions = policy(env)
        action = actions[0]
        action_np = action[0].detach().cpu().numpy()
        action_align = _action_alignment(action, pos_before, survivor_before)

        env.step(actions)

        pos_after = ground.state.pos.clone()
        dist_after = _distance_m(scenario, pos_after, survivor.state.pos)
        min_distance = min(min_distance, dist_after)
        displacement = pos_after - pos_before
        displacement_sim = torch.linalg.norm(displacement, dim=-1)
        displacement_m = _distance_sim_to_m(scenario, displacement_sim)
        target_before = survivor_before - pos_before
        disp_align = _cosine_alignment(displacement, target_before)
        action_disp_align = _cosine_alignment(action, displacement)
        path_speed = float(
            scenario._terrain_path_speed_multiplier(pos_before, pos_after)[0].detach().cpu().item()
        )
        actual_path_traversable = bool(
            scenario._path_is_traversable(
                pos_before.unsqueeze(1),
                pos_after.unsqueeze(1),
            )[0, 0].detach().cpu().item()
        )
        cell_after = _ground_cell_diagnostics(scenario, pos_after)

        trace.append({
            "step": step,
            "dist_before_m": dist_before,
            "dist_after_m": dist_after,
            "progress_m": dist_before - dist_after,
            "min_distance_m": min_distance,
            "action": action_np,
            "action_norm": float(np.linalg.norm(action_np)),
            "action_align": action_align,
            "saturated": bool(np.any(np.abs(action_np) >= 0.98)),
            "raw_mean": raw_mean,
            "raw_std_mean": float(np.mean(raw_std)),
            "raw_oob": bool(np.any(np.abs(raw_mean) > 1.0)),
            "displacement_m": displacement_m,
            "speed_mps": displacement_m / step_seconds,
            "disp_align": disp_align,
            "action_disp_align": action_disp_align,
            "path_speed": path_speed,
            "proposed_path_blocked": bool(scenario.step_ugv_proposed_path_blocked[0, 0].item()),
            "speed_limited": bool(scenario.step_ugv_speed_limited[0, 0].item()),
            "speed_limit_scale": float(scenario.step_ugv_speed_limit_scale[0, 0].detach().cpu().item()),
            "proposed_displacement_m": float(
                scenario.step_ugv_proposed_displacement_m[0, 0].detach().cpu().item()
            ),
            "corrected_displacement_m": float(
                scenario.step_ugv_corrected_displacement_m[0, 0].detach().cpu().item()
            ),
            "actual_displacement_m": float(
                scenario.step_ugv_actual_displacement_m[0, 0].detach().cpu().item()
            ),
            "motion_correction_m": float(
                scenario.step_ugv_motion_correction_m[0, 0].detach().cpu().item()
            ),
            "actual_path_traversable": actual_path_traversable,
            **{f"before_{k}": v for k, v in cell_before.items()},
            **{f"after_{k}": v for k, v in cell_after.items()},
        })

        if bool(scenario.found_survivors[0, 0]):
            break

    return {
        "seed": seed,
        "confirmed": float(scenario.found_survivors[0].sum().item()),
        "full_success": float(bool(scenario.found_survivors[0].all())),
        "initial_distance_m": initial_distance,
        "final_distance_m": _distance_m(scenario, ground.state.pos, survivor.state.pos),
        "min_distance_m": min_distance,
        "trace": trace,
    }


def run_action_magnitude_probe(checkpoint_dir: Path, scenario_kwargs: dict, seeds: list[int]) -> list[tuple[str, float, float]]:
    """Measure actual one-step displacement from fixed cardinal/diagonal commands."""
    del checkpoint_dir
    commands = {
        "east": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "north": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "northeast": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "half_ne": torch.tensor([[0.70710677, 0.70710677]], dtype=torch.float32),
    }
    out = []
    for name, action in commands.items():
        distances = []
        for seed in seeds:
            env = vmas.make_env(
                scenario=WildfireSearchScenario(),
                num_envs=1,
                device="cpu",
                continuous_actions=True,
                seed=seed,
                **copy.deepcopy(scenario_kwargs),
            )
            env.reset()
            scenario = env.scenario
            ground = env.agents[0]
            before = ground.state.pos.clone()
            env.step([action.clone()])
            displacement = torch.linalg.norm(ground.state.pos - before, dim=-1)
            distances.append(_distance_sim_to_m(scenario, displacement))
        out.append((name, float(np.mean(distances)), float(np.std(distances))))
    return out


def run_direction_probe(checkpoint_dir: Path, scenario_kwargs: dict) -> list[tuple[str, np.ndarray, float | None]]:
    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=True)
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=123,
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset()
    scenario = env.scenario
    ground = env.agents[0]
    survivor = scenario._survivors[0]

    probes = {
        "east": (0.5, 0.0),
        "west": (-0.5, 0.0),
        "north": (0.0, 0.5),
        "south": (0.0, -0.5),
        "northeast": (0.35, 0.35),
        "southwest": (-0.35, -0.35),
    }
    out = []
    for name, offset in probes.items():
        policy.reset()
        ground.state.pos[:] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        survivor.state.pos[:] = torch.tensor([offset], dtype=torch.float32)
        scenario.scouted_survivors[0, 0] = True
        scenario.known_survivors_by_agent[0, 0, 0] = True
        scenario.found_survivors[0, 0] = False
        action = policy(env)[0]
        alignment = _action_alignment(action, ground.state.pos, survivor.state.pos)
        out.append((name, action[0].cpu().numpy(), alignment))
    return out


def run_angle_bucket_probe(
    checkpoint_dir: Path,
    scenario_kwargs: dict,
    radius_m: float = 80.0,
    n_angles: int = 16,
) -> list[dict]:
    policy = HappoPolicy.from_checkpoint(checkpoint_dir, deterministic=True)
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        seed=123,
        **copy.deepcopy(scenario_kwargs),
    )
    env.reset()
    scenario = env.scenario
    ground = env.agents[0]
    survivor = scenario._survivors[0]
    scale = float(scenario.terrain_sim_units_per_meter[0])
    radius_sim = radius_m * scale if scale > 1e-9 else radius_m

    rows = []
    for angle in np.linspace(0.0, 360.0, n_angles, endpoint=False):
        policy.reset()
        target = np.array([
            np.cos(np.radians(angle)) * radius_sim,
            np.sin(np.radians(angle)) * radius_sim,
        ], dtype=np.float32)
        ground.state.pos[:] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        ground.state.vel[:] = torch.zeros_like(ground.state.vel)
        survivor.state.pos[:] = torch.from_numpy(target).view(1, 2)
        scenario.scouted_survivors[0, 0] = True
        scenario.known_survivors_by_agent[0, 0, 0] = True
        scenario.found_survivors[0, 0] = False

        dist = _actor_distribution(policy, env, agent_idx=0)
        raw_mean = dist.mean.detach().cpu().numpy()[0]
        raw_std = dist.stddev.detach().cpu().numpy()[0]
        action = policy(env)[0][0].cpu().numpy()
        error = _angle_error_deg(action, target)
        rows.append({
            "target_angle": float(angle),
            "action": action,
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "action_angle": _angle_deg(action),
            "angle_error": error,
            "saturated": bool(np.any(np.abs(action) >= 0.98)),
            "raw_oob": bool(np.any(np.abs(raw_mean) > 1.0)),
        })
    return rows


def _fmt_optional(value: float | None, width: int = 6, precision: int = 3) -> str:
    if value is None:
        return " " * (width - 4) + "none"
    return f"{value:{width}.{precision}f}"


def _aggregate_planner_buckets(
    rows: list[dict],
    *,
    buckets_key: str = "planner_buckets",
    total_steps_key: str = "planner_total_steps",
    bucket_names: tuple[str, ...] = ("all", "clear", "blocked"),
) -> tuple[int, dict]:
    buckets = {name: _empty_planner_bucket() for name in bucket_names}
    total_steps = 0
    for row in rows:
        total_steps += int(row.get(total_steps_key, 0))
        row_buckets = row.get(buckets_key, {})
        for name in buckets:
            if name in row_buckets:
                _merge_planner_bucket(buckets[name], row_buckets[name])
    return total_steps, buckets


def _print_planner_alignment_table(
    title: str,
    rows: list[dict],
    *,
    buckets_key: str = "planner_buckets",
    total_steps_key: str = "planner_total_steps",
    labels: tuple[tuple[str, str], ...] = (("all", "all"), ("clear", "clear"), ("blocked", "blocked")),
) -> None:
    total_steps, buckets = _aggregate_planner_buckets(
        rows,
        buckets_key=buckets_key,
        total_steps_key=total_steps_key,
        bucket_names=tuple(name for name, _ in labels),
    )
    valid_steps = int(buckets["all"]["steps"])
    if total_steps <= 0 or valid_steps <= 0:
        return

    print("-" * 72)
    print(title)
    print(
        "condition  frac_steps steps act_target move_target act_astar "
        "move_astar speed"
    )
    for name, label in labels:
        bucket = buckets[name]
        steps = int(bucket["steps"])
        frac = steps / valid_steps if valid_steps else 0.0
        print(
            f"{label:9s} "
            f"{frac:10.3f} "
            f"{steps:5d} "
            f"{_planner_metric(bucket, 'action_target'):10.3f} "
            f"{_planner_metric(bucket, 'movement_target'):11.3f} "
            f"{_planner_metric(bucket, 'action_planner'):9.3f} "
            f"{_planner_metric(bucket, 'movement_planner'):10.3f} "
            f"{_planner_speed(bucket):5.2f}m/s"
        )


def _print_failure_trace(result: dict, tail: int, stride: int) -> None:
    trace = result["trace"]
    print("-" * 72)
    print(
        f"failure trace seed {result['seed']}: "
        f"confirmed={result['confirmed']:.0f} "
        f"initial={result['initial_distance_m']:.1f}m "
        f"final={result['final_distance_m']:.1f}m "
        f"min={result['min_distance_m']:.1f}m"
    )
    if not trace:
        print("  no steps recorded")
        return

    selected: dict[int, dict] = {}
    for row in trace[: min(10, len(trace))]:
        selected[row["step"]] = row
    for row in trace[-max(tail, 0):]:
        selected[row["step"]] = row
    if stride > 0:
        for row in trace:
            if row["step"] % stride == 0:
                selected[row["step"]] = row
    for row in trace:
        suspicious = (
            row["progress_m"] < -0.5
            or row["displacement_m"] < 0.05
            or (row["action_disp_align"] is not None and row["action_disp_align"] < 0.5)
            or row["proposed_path_blocked"]
            or row["motion_correction_m"] > 0.5
            or row["before_blocked_frac"] > 0.25
            or row["path_speed"] < 0.55
        )
        if suspicious:
            selected[row["step"]] = row

    print(
        " step dist->after progress act_align disp_align act_disp "
        "move_m prop corr speed pathspd lim blocked cellspd action raw sat"
    )
    for step in sorted(selected):
        row = selected[step]
        action = row["action"]
        raw = row["raw_mean"]
        print(
            f"{row['step']:5d} "
            f"{row['dist_before_m']:6.1f}->{row['dist_after_m']:6.1f} "
            f"{row['progress_m']:8.2f} "
            f"{_fmt_optional(row['action_align'])} "
            f"{_fmt_optional(row['disp_align'])} "
            f"{_fmt_optional(row['action_disp_align'])} "
            f"{row['displacement_m']:6.2f} "
            f"{row['proposed_displacement_m']:5.2f} "
            f"{row['motion_correction_m']:5.2f} "
            f"{row['speed_mps']:5.2f} "
            f"{row['path_speed']:7.3f} "
            f"{int(row['speed_limited'])}/{row['speed_limit_scale']:.2f} "
            f"{int(row['proposed_path_blocked'])} "
            f"{row['before_blocked_count']:2d}/{row['before_blocked_frac']:.2f} "
            f"{row['before_cell_speed']:6.3f} "
            f"[{action[0]: .2f},{action[1]: .2f}] "
            f"[{raw[0]: .2f},{raw[1]: .2f}] "
            f"{int(row['saturated'])}/{int(row['raw_oob'])}"
        )
    print(
        "  sat/raw columns are action_saturated/raw_mean_out_of_bounds; "
        "prop is VMAS-proposed move, corr is final-vs-proposed correction; "
        "lim is speed_limited/speed_limit_scale; blocked is proposed path blocked, "
        "then blocked cells in the local map patch."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=None, help="Path to a HARL models/ checkpoint directory.")
    parser.add_argument("--terrain-cache-path", default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--ground-min-confirm-radius-m", type=float, default=None)
    parser.add_argument("--ugv-diagnostic-target-distance-min-m", type=float, default=None)
    parser.add_argument("--ugv-diagnostic-target-distance-max-m", type=float, default=None,
                        help="Omit for no upper bound when a min distance is provided.")
    parser.add_argument("--local-map-patch-size", type=int, default=None)
    parser.add_argument("--ugv-planner-hint", choices=("none", "local_astar", "local-astar"), default=None,
                        help="Override the checkpoint's UGV planner hint setting.")
    parser.add_argument("--ugv-planner-detour-obs", action=argparse.BooleanOptionalAction, default=None,
                        help="Override whether local A* planner observations include detour_needed.")
    parser.add_argument("--ugv-route-aware-reward", action=argparse.BooleanOptionalAction, default=None,
                        help="Override route-aware reward switching for local A* detours.")
    parser.add_argument("--ugv-planner-patch-size", type=int, default=None,
                        help="Override the checkpoint's local A* planner patch size.")
    parser.add_argument("--ugv-planner-lookahead-cells", type=int, default=None,
                        help="Override the checkpoint's planner waypoint lookahead.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actor means.")
    parser.add_argument("--trace-failures", action="store_true",
                        help="Print per-step diagnostics for seeds that fail to confirm.")
    parser.add_argument("--trace-all", action="store_true",
                        help="Print per-step diagnostics for every seed, not only failures.")
    parser.add_argument("--trace-tail", type=int, default=25,
                        help="How many final steps to print in each trace.")
    parser.add_argument("--trace-stride", type=int, default=25,
                        help="Also print every Nth trace step; 0 disables periodic trace rows.")
    parser.add_argument("--time-bins", type=int, default=5,
                        help="Number of episode-fraction bins for time-series diagnostics.")
    parser.add_argument("--json-output", default=None,
                        help="Optional path to write structured diagnostic rows and summary.")
    parser.add_argument("--plots-output", default=None,
                        help="Optional path to write diagnostic distribution/time-bin plots.")
    parser.add_argument("--shadow-ugv-planner", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute local A* diagnostics without changing the actor observation.")
    args = parser.parse_args()
    if args.local_map_patch_size is not None and (args.local_map_patch_size < 1 or args.local_map_patch_size % 2 != 1):
        parser.error("--local-map-patch-size must be a positive odd integer")
    if args.ugv_planner_patch_size is not None and (
        args.ugv_planner_patch_size < 1 or args.ugv_planner_patch_size % 2 != 1
    ):
        parser.error("--ugv-planner-patch-size must be a positive odd integer")
    if args.ugv_planner_lookahead_cells is not None and args.ugv_planner_lookahead_cells < 1:
        parser.error("--ugv-planner-lookahead-cells must be positive")
    if args.ugv_planner_lookahead_cells is not None:
        patch_size = args.ugv_planner_patch_size if args.ugv_planner_patch_size is not None else 11
        args.ugv_planner_lookahead_cells = min(
            args.ugv_planner_lookahead_cells,
            max(patch_size // 2, 1),
        )

    checkpoint_dir = _checkpoint_path(args.checkpoint_dir)
    scenario_kwargs = _scenario_kwargs(checkpoint_dir, args)
    if (
        bool(scenario_kwargs.get("ugv_route_aware_reward", False))
        and scenario_kwargs.get("ugv_planner_hint", "none") != "local_astar"
    ):
        parser.error("--ugv-route-aware-reward requires --ugv-planner-hint local_astar")
    print(f"checkpoint: {checkpoint_dir}")
    print(f"steps: {args.steps}")
    print(
        "planner_hint: "
        f"{scenario_kwargs.get('ugv_planner_hint', 'none')} "
        f"detour_obs={bool(scenario_kwargs.get('ugv_planner_detour_obs', False))} "
        f"route_aware={bool(scenario_kwargs.get('ugv_route_aware_reward', False))} "
        f"patch={scenario_kwargs.get('ugv_planner_patch_size', 11)} "
        f"lookahead={scenario_kwargs.get('ugv_planner_lookahead_cells', 10)}"
    )
    print(f"shadow_ugv_planner: {bool(args.shadow_ugv_planner)}")
    print("-" * 72)

    rows = [
        run_rollout(
            checkpoint_dir,
            scenario_kwargs,
            seed,
            deterministic=not args.stochastic,
            shadow_planner=bool(args.shadow_ugv_planner),
        )
        for seed in args.seeds
    ]
    summary = _summarize_rows(rows, args.time_bins)
    for row in rows:
        print(
            f"seed {row['seed']:>4}: confirmed={row['confirmed']:.0f} "
            f"success={row['full_success']:.0f} "
            f"initial={row['initial_distance_m']:.1f}m "
            f"final={row['final_distance_m']:.1f}m "
            f"min={row['min_distance_m']:.1f}m "
            f"ttc={row['confirmation_step'] if row['confirmation_step'] is not None else '-'} "
            f"path={row['path_length_m']:.1f}m "
            f"align={row['mean_action_target_alignment']:.3f} "
            f"toward={row['frac_action_toward_target']:.3f} "
            f"disp_align={row['mean_displacement_target_alignment']:.3f} "
            f"disp_toward={row['frac_displacement_toward_target']:.3f} "
            f"act_disp={row['mean_action_displacement_alignment']:.3f} "
            f"speed={row['mean_speed_mps']:.2f}m/s "
            f"sat={row['frac_action_saturated']:.3f} "
            f"raw_oob={row['frac_raw_mean_oob']:.3f} "
            f"blocked={row['frac_proposed_path_blocked']:.3f} "
            f"speedlim={row['frac_speed_limited']:.3f} "
            f"corr={row['mean_motion_correction_m']:.2f}m "
            f"astar_align={row['mean_shadow_astar_movement_alignment']:.3f} "
            f"astar_blocked={row['shadow_astar_direct_blocked_fraction']:.3f} "
            f"astar_detour={row['shadow_astar_detour_needed_fraction']:.3f}"
        )

    print("-" * 72)
    print(
        "means: "
        f"confirmed={summary['mean_confirmed']:.3f} "
        f"success={summary['success_rate']:.3f} "
        f"final={summary['mean_final_distance_m']:.1f}m "
        f"min={summary['mean_min_distance_m']:.1f}m "
        f"ttc={summary['mean_confirmation_step_successes']:.1f} steps "
        f"path={summary['mean_path_length_m']:.1f}m "
        f"path_eff={summary['mean_path_efficiency_successes_or_progress']:.2f} "
        f"align={np.mean([r['mean_action_target_alignment'] for r in rows]):.3f} "
        f"toward={np.mean([r['frac_action_toward_target'] for r in rows]):.3f} "
        f"disp_align={np.mean([r['mean_displacement_target_alignment'] for r in rows]):.3f} "
        f"disp_toward={np.mean([r['frac_displacement_toward_target'] for r in rows]):.3f} "
        f"act_disp={np.mean([r['mean_action_displacement_alignment'] for r in rows]):.3f} "
        f"disagree={np.mean([r['frac_action_displacement_disagree'] for r in rows]):.3f} "
        f"speed={np.mean([r['mean_speed_mps'] for r in rows]):.2f}m/s "
        f"near_zero={np.mean([r['frac_near_zero_displacement'] for r in rows]):.3f} "
        f"action_norm={np.mean([r['mean_action_norm'] for r in rows]):.3f} "
        f"sat={np.mean([r['frac_action_saturated'] for r in rows]):.3f} "
        f"raw_mean_norm={np.mean([r['mean_raw_mean_norm'] for r in rows]):.3f} "
        f"raw_absmax={np.mean([r['mean_raw_mean_abs_max'] for r in rows]):.3f} "
        f"raw_oob={np.mean([r['frac_raw_mean_oob'] for r in rows]):.3f} "
        f"raw_std={np.mean([r['mean_raw_std'] for r in rows]):.3f} "
        f"blocked={np.mean([r['frac_proposed_path_blocked'] for r in rows]):.3f} "
        f"speedlim={np.mean([r['frac_speed_limited'] for r in rows]):.3f} "
        f"path_speed={np.mean([r['mean_path_speed'] for r in rows]):.3f} "
        f"limit_scale={np.mean([r['mean_speed_limit_scale'] for r in rows]):.3f} "
        f"proposed_move={np.mean([r['mean_proposed_displacement_m'] for r in rows]):.2f}m "
        f"corrected_move={np.mean([r['mean_corrected_displacement_m'] for r in rows]):.2f}m "
        f"actual_move={np.mean([r['mean_actual_displacement_m'] for r in rows]):.2f}m "
        f"correction={np.mean([r['mean_motion_correction_m'] for r in rows]):.2f}m "
        f"shadow_astar_valid={summary['mean_shadow_astar_valid_fraction']:.3f} "
        f"shadow_astar_blocked={summary['mean_shadow_astar_direct_blocked_fraction']:.3f} "
        f"shadow_astar_detour={summary['mean_shadow_astar_detour_needed_fraction']:.3f} "
        f"shadow_astar_align={summary['mean_shadow_astar_movement_alignment']:.3f} "
        f"shadow_astar_progress={summary['mean_shadow_astar_progress_m']:.2f}m"
    )
    _print_planner_alignment_table("planner alignment by A* direct-path state:", rows)
    failed_rows = [row for row in rows if row["full_success"] <= 0.0]
    if failed_rows:
        _print_planner_alignment_table("planner alignment by A* direct-path state, failures only:", failed_rows)
    _print_planner_alignment_table(
        "shadow A* alignment by local route state:",
        rows,
        buckets_key="shadow_planner_buckets",
        total_steps_key="shadow_planner_total_steps",
        labels=(("all", "all"), ("clear", "clear"), ("blocked", "blocked"), ("detour", "detour")),
    )
    if failed_rows:
        _print_planner_alignment_table(
            "shadow A* alignment by local route state, failures only:",
            failed_rows,
            buckets_key="shadow_planner_buckets",
            total_steps_key="shadow_planner_total_steps",
            labels=(("all", "all"), ("clear", "clear"), ("blocked", "blocked"), ("detour", "detour")),
        )

    print("-" * 72)
    print("fixed-command one-step motion probe:")
    for name, mean_m, std_m in run_action_magnitude_probe(checkpoint_dir, scenario_kwargs, args.seeds):
        print(f"command {name:>9}: displacement={mean_m:.3f}m +/- {std_m:.3f}m")

    print("-" * 72)
    for name, action, alignment in run_direction_probe(checkpoint_dir, scenario_kwargs):
        align_text = "none" if alignment is None else f"{alignment: .3f}"
        print(f"probe {name:>9}: action=[{action[0]: .3f}, {action[1]: .3f}] align={align_text}")

    print("-" * 72)
    print("angle-bucket steering probe:")
    for row in run_angle_bucket_probe(
        checkpoint_dir,
        scenario_kwargs,
        radius_m=80.0,
    ):
        action = row["action"]
        raw_mean = row["raw_mean"]
        raw_std = row["raw_std"]
        action_angle = "none" if row["action_angle"] is None else f"{row['action_angle']:6.1f}"
        angle_error = "none" if row["angle_error"] is None else f"{row['angle_error']:6.1f}"
        print(
            f"target={row['target_angle']:6.1f}deg "
            f"action=[{action[0]: .3f}, {action[1]: .3f}] "
            f"raw=[{raw_mean[0]: .3f}, {raw_mean[1]: .3f}] "
            f"std=[{raw_std[0]: .3f}, {raw_std[1]: .3f}] "
            f"act_angle={action_angle}deg "
            f"err={angle_error}deg "
            f"sat={int(row['saturated'])} raw_oob={int(row['raw_oob'])}"
        )

    if args.trace_failures or args.trace_all:
        trace_rows = [
            run_failure_trace(checkpoint_dir, scenario_kwargs, seed, deterministic=not args.stochastic)
            for seed in args.seeds
        ]
        for result in trace_rows:
            if args.trace_all or result["full_success"] <= 0.0:
                _print_failure_trace(result, tail=args.trace_tail, stride=args.trace_stride)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_dir),
            "deterministic": not args.stochastic,
            "steps": int(args.steps),
            "seeds": list(args.seeds),
            "shadow_ugv_planner": bool(args.shadow_ugv_planner),
            "scenario_kwargs": scenario_kwargs,
            "summary": summary,
            "rows": rows,
        }
        output.write_text(
            json.dumps(_json_sanitize(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote JSON diagnostics: {output}")

    if args.plots_output:
        output = Path(args.plots_output)
        _plot_ugv_diagnostics(
            rows,
            summary,
            output,
            deterministic=not args.stochastic,
        )
        print(f"wrote diagnostic plots: {output}")


if __name__ == "__main__":
    main()
