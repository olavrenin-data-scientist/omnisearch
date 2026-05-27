"""
Export a scenario rollout to JSON for the web viewer.

Each frame captures the minimum needed to render a top-down replay:
agent positions + types, survivor positions + scout/found state, and
the set of burning fire cells. Mission metrics are computed and bundled
alongside so the viewer can show a metrics panel without re-running
anything.

JSON schema (one object per file):

    {
      "metadata": {
        "strategy":        "nearest_candidate",
        "seed":            0,
        "n_steps":         200,
        "world":           { "x_semidim": 1.0, "y_semidim": 1.0 },
        "fire_grid_size":  16,
        "terrain": {
          "land_cover":    [[1, 1, 0, ...], ...],
          "elevation":     [[0.1, ...], ...],
          "slope":         [[0.2, ...], ...],
          "traversable":   [[true, ...], ...],
          "movement_cost": [[1.0, ...], ...],
          "obstacle_type": [[0, 1, 0, ...], ...],
          "obstacle_height": [[0.0, 0.2, ...], ...]
        },
        "n_drones":        3,
        "n_ground":        2,
        "n_survivors":     5,
        "metrics":         { ... MissionMetrics.as_dict() ... }
      },
      "frames": [
        {
          "step":       0,
          "agents":     [{ "name": "drone_0", "type": "drone",  "x": 0.1, "y": -0.5 }, ...],
          "survivors":  [{ "x": 0.3, "y": 0.2, "scouted": false, "found": false }, ...],
          "fire_cells": [[gx, gy], ...]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import vmas

from envs.wildfire_search import WildfireSearchScenario, X, Y
from evaluation.mission_metrics import EpisodeRecorder


def _agent_record(agent, scenario, env_index: int) -> dict:
    pos = agent.state.pos[env_index]
    record = {
        "name": agent.name,
        "type": "drone" if getattr(agent, "is_drone", False) else "ground",
        "x":    float(pos[X]),
        "y":    float(pos[Y]),
    }
    if getattr(agent, "is_drone", False):
        drone_index = scenario.world.agents.index(agent)
        record["altitude"] = float(scenario.drone_altitude[env_index, drone_index])
        record["altitude_level"] = int(scenario.drone_altitude_level[env_index, drone_index])
    return record


def _survivor_records(scenario, env_index: int) -> List[dict]:
    scouted = scenario.scouted_survivors[env_index].cpu().tolist()
    found   = scenario.found_survivors[env_index].cpu().tolist()
    out = []
    for i, s in enumerate(scenario._survivors):
        pos = s.state.pos[env_index]
        out.append({
            "x":       float(pos[X]),
            "y":       float(pos[Y]),
            "scouted": bool(scouted[i]),
            "found":   bool(found[i]),
        })
    return out


def _fire_cells(scenario, env_index: int) -> List[List[int]]:
    grid = scenario.fire_grid[env_index].cpu().numpy()
    ys, xs = (grid != 0).nonzero()
    return [[int(x), int(y)] for x, y in zip(xs, ys)]


def _smoke_cells(scenario, env_index: int) -> List[List[float]]:
    grid = scenario.smoke_grid[env_index].cpu().numpy()
    ys, xs = (grid > 0.02).nonzero()
    return [[int(x), int(y), round(float(grid[y, x]), 4)] for x, y in zip(xs, ys)]


def _terrain_record(scenario, env_index: int) -> dict:
    """Static map layers used by the replay viewer and later route analysis."""
    def rounded_rows(tensor) -> List[List[float]]:
        return [[round(float(v), 4) for v in row] for row in tensor.cpu().tolist()]

    return {
        "land_cover": scenario.land_cover_grid[env_index].cpu().tolist(),
        "elevation": rounded_rows(scenario.elevation_grid[env_index]),
        "slope": rounded_rows(scenario.slope_grid[env_index]),
        "traversable": scenario.traversable_grid[env_index].cpu().tolist(),
        "movement_cost": rounded_rows(scenario.mobility_cost_grid[env_index]),
        "cover_names": ["road", "open", "brush", "forest", "rock"],
        "obstacle_type": scenario.obstacle_type_grid[env_index].cpu().tolist(),
        "obstacle_height": rounded_rows(scenario.obstacle_height_grid[env_index]),
        "required_clearance": rounded_rows(scenario.required_clearance_grid[env_index]),
        "obstacle_names": ["none", "tree", "house"],
        "drone_flight_levels": [round(float(v), 4) for v in scenario.drone_flight_levels.cpu().tolist()],
        "drone_camera_fov_deg": round(float(scenario.drone_camera_fov_deg), 4),
        "drone_sensor_max_range": round(float(scenario.drone_sensor_max_range), 4),
        "drone_detection_quality": [
            round(float(v), 4) for v in scenario.drone_detection_quality.cpu().tolist()
        ],
        "drone_cover_detection_factors": [
            round(float(v), 4) for v in scenario.drone_cover_detection_factors.cpu().tolist()
        ],
    }


def export_trajectory(
    strategy_name: str,
    make_policy:   Callable,
    output_path:   Path,
    n_steps:       int = 200,
    seed:          int = 0,
    num_envs:      int = 2,
    env_index:     int = 0,
    scenario_kwargs: Optional[dict] = None,
) -> Path:
    """
    Run a rollout, capture every frame, write to JSON, return the path.

    ``make_policy`` must be a ``callable(env) -> action_fn`` — we need to
    build the policy AFTER the env is created so baselines that hold a
    scenario reference get the right one. If you have a pre-built
    action_fn, wrap it as ``make_policy=lambda env: existing_fn``.
    """
    scenario_kwargs = dict(scenario_kwargs or {})
    max_steps = scenario_kwargs.pop("max_steps", n_steps)

    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=num_envs,
        device="cpu",
        continuous_actions=True,
        seed=seed,
        max_steps=max_steps,
        **scenario_kwargs,
    )
    env.reset()
    sc = env.scenario
    if max_steps is not None:
        sc.max_steps = max_steps
    action_fn = make_policy(env)

    recorder = EpisodeRecorder(sc, env_index=env_index)

    metadata: Dict = {
        "strategy":       strategy_name,
        "seed":           seed,
        "n_steps":        n_steps,
        "world":          {"x_semidim": sc.x_semidim, "y_semidim": sc.y_semidim},
        "fire_grid_size": sc.fire_grid_size,
        "terrain":        _terrain_record(sc, env_index),
        "n_drones":       sc.n_drones,
        "n_ground":       sc.n_ground,
        "n_survivors":    sc.n_survivors,
    }

    frames: List[Dict] = []

    # Initial frame (post-reset, before any step)
    frames.append({
        "step":       0,
        "agents":     [_agent_record(a, sc, env_index) for a in sc.world.agents],
        "survivors":  _survivor_records(sc, env_index),
        "fire_cells": _fire_cells(sc, env_index),
        "smoke_cells": _smoke_cells(sc, env_index),
    })

    for step in range(1, n_steps + 1):
        env.step(action_fn(env))
        recorder.step()
        frames.append({
            "step":       step,
            "agents":     [_agent_record(a, sc, env_index) for a in sc.world.agents],
            "survivors":  _survivor_records(sc, env_index),
            "fire_cells": _fire_cells(sc, env_index),
            "smoke_cells": _smoke_cells(sc, env_index),
        })
        if sc.done()[env_index].item():
            break

    metadata["metrics"] = recorder.finalize().as_dict()
    metadata["actual_n_steps"] = len(frames) - 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False forces NaN → error; we sanitize first so JS's JSON.parse
    # (which doesn't accept the JS literal `NaN`) can load the file cleanly.
    payload = _sanitize_for_json({"metadata": metadata, "frames": frames})
    output_path.write_text(json.dumps(payload, allow_nan=False))
    return output_path


def _sanitize_for_json(obj):
    """Recursively replace NaN / +-inf with None so JSON.parse can load it."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj
