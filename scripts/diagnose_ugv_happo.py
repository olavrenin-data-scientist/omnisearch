"""
Diagnose whether a HAPPO UGV actor moves toward a known survivor.

This is intentionally narrower than full mission evaluation. It is meant for
the UGV-known-survivor diagnostic task: one ground robot, one survivor already
scouted/known at reset, and no fire.
"""

from __future__ import annotations

import argparse
import copy
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
        if args.ugv_diagnostic_target_distance_min_m is None or args.ugv_diagnostic_target_distance_max_m is None:
            raise ValueError(
                "ugv_diagnostic_target_distance_min_m and "
                "ugv_diagnostic_target_distance_max_m must be provided together"
            )
        target_distance_min_m = max(
            float(args.ugv_diagnostic_target_distance_min_m),
            0.0,
        )
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
            "known_survivor_spawn_distance_min_m": target_distance_min_m,
            "known_survivor_spawn_distance_max_m": target_distance_max_m,
        })

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


def _ensure_policy_rnn(policy: HappoPolicy, env) -> None:
    B = env.scenario.world.batch_dim
    if getattr(policy, "_rnn_states", None) is None or policy._rnn_states[0].shape[0] != B:
        policy._rnn_states = [
            np.zeros((B, policy._recurrent_n, policy._rnn_hidden_size), dtype=np.float32)
            for _ in env.agents
        ]


def _actor_distribution(policy: HappoPolicy, env, agent_idx: int = 0):
    """Return the current raw Gaussian action distribution for one actor."""
    _ensure_policy_rnn(policy, env)
    agent = env.agents[agent_idx]
    obs = env.scenario.observation(agent).cpu().numpy()
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


def run_rollout(checkpoint_dir: Path, scenario_kwargs: dict, seed: int, deterministic: bool) -> dict:
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

    for _ in range(scenario_kwargs["max_steps"]):
        pos_before = ground.state.pos.clone()
        survivor_before = survivor.state.pos.clone()
        dist = _actor_distribution(policy, env, agent_idx=0)
        raw_mean = dist.mean.detach().cpu().numpy()
        raw_std = dist.stddev.detach().cpu().numpy()
        raw_mean_norms.append(float(np.linalg.norm(raw_mean[0])))
        raw_mean_abs_max.append(float(np.max(np.abs(raw_mean[0]))))
        raw_mean_oob.append(bool(np.any(np.abs(raw_mean[0]) > 1.0)))
        raw_stds.append(float(np.mean(raw_std[0])))
        actions = policy(env)
        alignment = _action_alignment(actions[0], ground.state.pos, survivor.state.pos)
        if alignment is not None:
            alignments.append(alignment)
        action_norms.append(float(torch.linalg.norm(actions[0], dim=-1)[0]))
        saturated_actions.append(bool((actions[0].abs() >= 0.98).any().item()))
        env.step(actions)
        proposed_path_blocked.append(bool(scenario.step_ugv_proposed_path_blocked[0, 0].item()))
        speed_limited.append(bool(scenario.step_ugv_speed_limited[0, 0].item()))
        path_speeds.append(float(scenario.step_ugv_path_speed[0, 0].detach().cpu().item()))
        speed_limit_scales.append(float(scenario.step_ugv_speed_limit_scale[0, 0].detach().cpu().item()))
        proposed_displacements_m.append(
            float(scenario.step_ugv_proposed_displacement_m[0, 0].detach().cpu().item())
        )
        corrected_displacements_m.append(
            float(scenario.step_ugv_corrected_displacement_m[0, 0].detach().cpu().item())
        )
        actual_displacements_m.append(
            float(scenario.step_ugv_actual_displacement_m[0, 0].detach().cpu().item())
        )
        motion_corrections_m.append(
            float(scenario.step_ugv_motion_correction_m[0, 0].detach().cpu().item())
        )
        displacement = ground.state.pos - pos_before
        target_before = survivor_before - pos_before
        disp_alignment = _cosine_alignment(displacement, target_before)
        if disp_alignment is not None:
            displacement_target_alignments.append(disp_alignment)
        action_disp_alignment = _cosine_alignment(actions[0], displacement)
        if action_disp_alignment is not None:
            action_displacement_alignments.append(action_disp_alignment)
        displacement_meters.append(_distance_sim_to_m(scenario, torch.linalg.norm(displacement, dim=-1)))
        min_distance = min(min_distance, _distance_m(scenario, ground.state.pos, survivor.state.pos))
        if bool(scenario.found_survivors[0, 0]):
            break

    final_distance = _distance_m(scenario, ground.state.pos, survivor.state.pos)
    step_seconds = max(float(getattr(scenario, "sim_step_seconds", 1.0)), 1e-9)
    return {
        "seed": seed,
        "confirmed": float(scenario.found_survivors[0].sum().item()),
        "full_success": float(bool(scenario.found_survivors[0].all())),
        "initial_distance_m": initial_distance,
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
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
    parser.add_argument("--ugv-diagnostic-target-distance-max-m", type=float, default=None)
    parser.add_argument("--local-map-patch-size", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actor means.")
    parser.add_argument("--trace-failures", action="store_true",
                        help="Print per-step diagnostics for seeds that fail to confirm.")
    parser.add_argument("--trace-all", action="store_true",
                        help="Print per-step diagnostics for every seed, not only failures.")
    parser.add_argument("--trace-tail", type=int, default=25,
                        help="How many final steps to print in each trace.")
    parser.add_argument("--trace-stride", type=int, default=25,
                        help="Also print every Nth trace step; 0 disables periodic trace rows.")
    args = parser.parse_args()

    checkpoint_dir = _checkpoint_path(args.checkpoint_dir)
    scenario_kwargs = _scenario_kwargs(checkpoint_dir, args)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"steps: {args.steps}")
    print("-" * 72)

    rows = [
        run_rollout(checkpoint_dir, scenario_kwargs, seed, deterministic=not args.stochastic)
        for seed in args.seeds
    ]
    for row in rows:
        print(
            f"seed {row['seed']:>4}: confirmed={row['confirmed']:.0f} "
            f"success={row['full_success']:.0f} "
            f"initial={row['initial_distance_m']:.1f}m "
            f"final={row['final_distance_m']:.1f}m "
            f"min={row['min_distance_m']:.1f}m "
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
            f"corr={row['mean_motion_correction_m']:.2f}m"
        )

    print("-" * 72)
    print(
        "means: "
        f"confirmed={np.mean([r['confirmed'] for r in rows]):.3f} "
        f"success={np.mean([r['full_success'] for r in rows]):.3f} "
        f"final={np.mean([r['final_distance_m'] for r in rows]):.1f}m "
        f"min={np.mean([r['min_distance_m'] for r in rows]):.1f}m "
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
        f"correction={np.mean([r['mean_motion_correction_m'] for r in rows]):.2f}m"
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


if __name__ == "__main__":
    main()
