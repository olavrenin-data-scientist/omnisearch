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
          "speed_multiplier": [[1.0, ...], ...],
          "obstacle_type": [[0, 1, 0, ...], ...],
          "obstacle_height": [[0.0, 0.002, ...], ...],
          "sim_units_per_meter": 0.0004
        },
        "n_drones":        3,
        "n_ground":        2,
        "n_survivors":     5,
        "metrics":         { ... MissionMetrics.as_dict() ... }
      },
      "frames": [
        {
          "step":       0,
          "agents":     [{ "name": "drone_0", "type": "drone",  "x": 0.1, "y": -0.5,
                           "altitude_agl": 0.2, "altitude_msl": 0.4,
                           "target_altitude_agl": 0.24 }, ...],
          "survivors":  [{ "x": 0.3, "y": 0.2, "scouted": false, "found": false }, ...],
          "fire_cells": [[gx, gy, intensity], ...],
          "drone_perception": [
            {
              "name": "drone_0",
              "footprint": 0.11,
              "survivors": [{ "index": 0, "visible": true, "probability": 0.72, ... }]
            }
          ]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import vmas

from detection.simulation_adapter import SimDrone, SimEntity, SimulationCvAdapter
from envs.wildfire_search import WildfireSearchScenario, X, Y
from evaluation.mission_metrics import EpisodeRecorder


def _agent_record(agent, scenario, env_index: int) -> dict:
    pos = agent.state.pos[env_index]
    comms_up_t = getattr(agent, "comms_up", None)
    comms_up = bool(comms_up_t[env_index].item()) if comms_up_t is not None else True
    record = {
        "name":     agent.name,
        "type":     "drone" if getattr(agent, "is_drone", False) else "ground",
        "x":        float(pos[X]),
        "y":        float(pos[Y]),
        "comms_up": comms_up,
    }
    if getattr(agent, "is_drone", False):
        drone_index = scenario.world.agents.index(agent)
        record["altitude"] = float(scenario.drone_altitude[env_index, drone_index])
        record["altitude_agl"] = float(scenario.drone_altitude[env_index, drone_index])
        record["altitude_msl"] = float(scenario.drone_altitude_msl[env_index, drone_index])
        record["target_altitude_agl"] = float(scenario.drone_target_altitude[env_index, drone_index])
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


def _fire_cells(scenario, env_index: int) -> List[List[float]]:
    grid = scenario.fire_grid[env_index].cpu().numpy()
    intensity = scenario.fire_intensity_grid[env_index].cpu().numpy()
    ys, xs = (grid != 0).nonzero()
    return [[int(x), int(y), round(float(intensity[y, x]), 4)] for x, y in zip(xs, ys)]


def _burned_cells_added(scenario, env_index: int, previous_grid) -> List[List[int]]:
    grid = scenario.burned_grid[env_index].cpu().numpy()
    ys, xs = (grid & ~previous_grid).nonzero()
    previous_grid[...] = grid
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
        "source": getattr(scenario, "terrain_source", "real"),
        "source_description": getattr(scenario, "terrain_source_description", ["real"])[env_index],
        "source_metadata": getattr(scenario, "terrain_source_metadata", [{}])[env_index],
        "land_cover": scenario.land_cover_grid[env_index].cpu().tolist(),
        "elevation": rounded_rows(scenario.elevation_grid[env_index]),
        "slope": rounded_rows(scenario.slope_grid[env_index]),
        "moisture": rounded_rows(scenario.moisture_grid[env_index]),
        "fuel_density": rounded_rows(scenario.fuel_density_grid[env_index]),
        "rockiness": rounded_rows(scenario.rockiness_grid[env_index]),
        "traversable": scenario.traversable_grid[env_index].cpu().tolist(),
        "movement_cost": rounded_rows(scenario.mobility_cost_grid[env_index]),
        "speed_multiplier": rounded_rows(scenario.speed_multiplier_grid[env_index]),
        "cover_names": ["road", "open", "brush", "forest", "rock", "water"],
        "obstacle_type": scenario.obstacle_type_grid[env_index].cpu().tolist(),
        "obstacle_height": rounded_rows(scenario.obstacle_height_grid[env_index]),
        "required_clearance": rounded_rows(scenario.required_clearance_grid[env_index]),
        "required_clearance_agl": rounded_rows(scenario.required_clearance_grid[env_index]),
        "required_clearance_msl": rounded_rows(scenario.required_clearance_msl_grid[env_index]),
        "obstacle_names": ["none", "tree", "house"],
        "drone_flight_levels": [
            round(float(v), 6) for v in scenario.drone_flight_levels_by_env[env_index].cpu().tolist()
        ],
        "drone_flight_levels_m": [
            round(float(v) / max(float(scenario.terrain_sim_units_per_meter[env_index]), 1e-12), 2)
            for v in scenario.drone_flight_levels_by_env[env_index].cpu().tolist()
        ],
        "drone_altitude_model": "continuous_agl_rate_limited",
        "drone_flight_level_reference": "AGL",
        "drone_climb_rate": round(float(scenario.drone_climb_rate), 4),
        "drone_descent_rate": round(float(scenario.drone_descent_rate), 4),
        "drone_altitude_release_margin": round(float(scenario.drone_altitude_release_margin), 4),
        "drone_safety_clearance": round(float(scenario.drone_safety_clearance_by_env[env_index]), 6),
        "drone_safety_clearance_m": round(float(scenario.drone_safety_clearance_m), 4),
        "sim_units_per_meter": round(float(scenario.terrain_sim_units_per_meter[env_index]), 8),
        "sim_step_seconds": round(float(getattr(scenario, "sim_step_seconds", 1.0)), 4),
        "drone_speed_mps": round(float(getattr(scenario, "drone_speed_mps", 0.0)), 4),
        "drone_distance_per_step_m": round(
            float(getattr(scenario, "drone_speed_mps", 0.0))
            * float(getattr(scenario, "sim_step_seconds", 1.0)),
            4,
        ),
        "drone_max_speed_sim": round(float(getattr(scenario, "drone_max_speed_sim", 0.0)), 8),
        "drone_u_multiplier": round(float(getattr(scenario, "drone_u_multiplier", 0.0)), 4),
        "ground_speed_mps": round(float(getattr(scenario, "ground_speed_mps", 0.0)), 4),
        "ground_accel_mps2": round(float(getattr(scenario, "ground_accel_mps2", 0.0)), 4),
        "ground_distance_per_step_m": round(
            float(getattr(scenario, "ground_speed_mps", 0.0))
            * float(getattr(scenario, "sim_step_seconds", 1.0)),
            4,
        ),
        "ground_max_speed_sim": round(float(getattr(scenario, "ground_max_speed_sim", 0.0)), 8),
        "ground_u_multiplier": round(float(getattr(scenario, "ground_u_multiplier", 0.0)), 4),
        "drone_camera_fov_deg": round(float(scenario.drone_camera_fov_deg), 4),
        "drone_sensor_max_range": round(float(scenario.drone_sensor_max_range_by_env[env_index]), 6),
        "drone_detection_quality": [
            round(float(v), 4) for v in scenario.drone_detection_quality.cpu().tolist()
        ],
        "drone_perception_path_samples": int(scenario.drone_perception_path_samples),
        "drone_smoke_extinction": round(float(scenario.drone_smoke_extinction), 4),
        "drone_fire_glare_penalty": round(float(scenario.drone_fire_glare_penalty), 4),
        "drone_heat_distortion_penalty": round(float(scenario.drone_heat_distortion_penalty), 4),
        "drone_cover_detection_factors": [
            round(float(v), 4) for v in scenario.drone_cover_detection_factors.cpu().tolist()
        ],
        "wind_direction": [round(float(v), 4) for v in scenario.wind_direction],
        "wind_strength": round(float(scenario.wind_strength), 4),
        "land_cover_fire_fuel": [
            round(float(v), 4) for v in scenario.land_cover_fire_fuel.cpu().tolist()
        ],
        "object_fire_fuel": [
            round(float(v), 4) for v in scenario.object_fire_fuel.cpu().tolist()
        ],
    }


def _cv_perception_records(
    scenario,
    env_index: int,
    *,
    adapter: SimulationCvAdapter | None,
    image_dir: Path | None,
    step: int,
    save_images_every: int,
) -> List[dict]:
    if adapter is None or scenario.n_drones == 0:
        return []

    drone_agents = scenario.world.agents[:scenario.n_drones]
    survivors = [
        SimEntity(
            index=i,
            world_xy=(
                float(survivor.state.pos[env_index, X]),
                float(survivor.state.pos[env_index, Y]),
            ),
        )
        for i, survivor in enumerate(scenario._survivors)
        if not bool(scenario.found_survivors[env_index, i].item())
    ]
    records = []
    for drone_idx, agent in enumerate(drone_agents):
        pos = agent.state.pos[env_index]
        drone = SimDrone(
            index=drone_idx,
            name=agent.name,
            world_xy=(float(pos[X]), float(pos[Y])),
            altitude_agl=float(scenario.drone_altitude[env_index, drone_idx]),
        )
        image_path = None
        if image_dir is not None and save_images_every > 0 and step % save_images_every == 0:
            image_path = image_dir / f"step_{step:04d}_{agent.name}.png"
        records.append(adapter.render_and_detect(drone=drone, survivors=survivors, image_path=image_path))
    return records


def export_trajectory(
    strategy_name: str,
    make_policy:   Callable,
    output_path:   Path,
    n_steps:       int = 200,
    seed:          int = 0,
    num_envs:      int = 2,
    env_index:     int = 0,
    scenario_kwargs: Optional[dict] = None,
    cv_options: Optional[dict] = None,
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
    cv_options = dict(cv_options or {})
    cv_adapter = None
    cv_image_dir = None
    cv_save_images_every = int(cv_options.pop("save_images_every", 0)) if cv_options else 0
    if cv_options.pop("enabled", False):
        cv_output_dir = Path(cv_options.pop("output_dir", output_path.parent / f"{output_path.stem}_cv"))
        if not cv_output_dir.is_absolute():
            cv_output_dir = Path.cwd() / cv_output_dir
        cv_image_dir = cv_output_dir / "images"
        if cv_save_images_every > 0:
            cv_image_dir.mkdir(parents=True, exist_ok=True)
        cv_adapter = SimulationCvAdapter(
            terrain_cache_path=sc.terrain_cache_path or scenario_kwargs.get("terrain_cache_path"),
            fov_deg=float(sc.drone_camera_fov_deg),
            **cv_options,
        )

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
        "agent_radius":   round(float(sc.agent_radius), 4),
        "survivor_radius": round(float(sc.survivor_radius), 4),
        "fire_model": {
            "spread_prob": round(float(sc.fire_spread_prob), 4),
            "spread_variability": round(float(sc.fire_spread_variability), 4),
            "wind_spread_weight": round(float(sc.fire_wind_spread_weight), 4),
            "slope_spread_weight": round(float(sc.fire_slope_spread_weight), 4),
            "moisture_damping": round(float(sc.fire_moisture_damping), 4),
            "intensity_decay": round(float(sc.fire_intensity_decay), 4),
            "smoke_emission": round(float(sc.smoke_emission), 4),
            "smoke_decay": round(float(sc.smoke_decay), 4),
            "smoke_diffusion": round(float(sc.smoke_diffusion), 4),
            "smolder_smoke_emission": round(float(sc.smolder_smoke_emission), 4),
            "smolder_decay": round(float(sc.smolder_decay), 4),
            "smolder_start_fraction": round(float(sc.smolder_start_fraction), 4),
            "land_cover_burnout_min_updates": list(sc.land_cover_fire_burnout_min_updates),
            "land_cover_burnout_max_updates": list(sc.land_cover_fire_burnout_max_updates),
            "land_cover_burnout_min_minutes": [
                round(float(v) * float(sc.fire_step_interval) * float(getattr(sc, "sim_step_seconds", 1.0)) / 60.0, 2)
                for v in sc.land_cover_fire_burnout_min_updates
            ],
            "land_cover_burnout_max_minutes": [
                round(float(v) * float(sc.fire_step_interval) * float(getattr(sc, "sim_step_seconds", 1.0)) / 60.0, 2)
                for v in sc.land_cover_fire_burnout_max_updates
            ],
        },
    }
    if cv_adapter is not None:
        metadata["cv_perception"] = {
            "mode": "naip_sard_preliminary_detector",
            "naip_image_path": str(cv_adapter.naip_image_path) if cv_adapter.naip_image_path else None,
            "naip_tile_manifest_path": (
                str(cv_adapter.tile_cache.manifest_path)
                if cv_adapter.tile_cache is not None
                else None
            ),
            "human_asset_path": str(cv_adapter.human_asset_path) if cv_adapter.human_asset_path else None,
            "image_size_px": int(cv_adapter.image_size),
            "background_size_px": list(cv_adapter.background_size_px),
            "background_gsd_m_per_px": [
                round(float(cv_adapter.background_gsd_m_per_px[0]), 4),
                round(float(cv_adapter.background_gsd_m_per_px[1]), 4),
            ],
            "sim_units_per_meter": round(float(cv_adapter.sim_units_per_meter), 10),
            "save_images_every": cv_save_images_every,
        }

    frames: List[Dict] = []
    previous_burned_grid = np.zeros_like(sc.burned_grid[env_index].cpu().numpy(), dtype=bool)

    # Initial frame (post-reset, before any step)
    frames.append({
        "step":       0,
        "agents":     [_agent_record(a, sc, env_index) for a in sc.world.agents],
        "survivors":  _survivor_records(sc, env_index),
        "fire_cells": _fire_cells(sc, env_index),
        "burned_cells_added": _burned_cells_added(sc, env_index, previous_burned_grid),
        "smoke_cells": _smoke_cells(sc, env_index),
        "drone_perception": sc.drone_perception_debug(env_index),
        "cv_perception": _cv_perception_records(
            sc,
            env_index,
            adapter=cv_adapter,
            image_dir=cv_image_dir,
            step=0,
            save_images_every=cv_save_images_every,
        ),
    })

    for step in range(1, n_steps + 1):
        env.step(action_fn(env))
        recorder.step()
        frames.append({
            "step":       step,
            "agents":     [_agent_record(a, sc, env_index) for a in sc.world.agents],
            "survivors":  _survivor_records(sc, env_index),
            "fire_cells": _fire_cells(sc, env_index),
            "burned_cells_added": _burned_cells_added(sc, env_index, previous_burned_grid),
            "smoke_cells": _smoke_cells(sc, env_index),
            "drone_perception": sc.drone_perception_debug(env_index),
            "cv_perception": _cv_perception_records(
                sc,
                env_index,
                adapter=cv_adapter,
                image_dir=cv_image_dir,
                step=step,
                save_images_every=cv_save_images_every,
            ),
        })
        if sc.done()[env_index].item():
            break

    metadata["metrics"] = recorder.finalize().as_dict()
    metadata["actual_n_steps"] = len(frames) - 1
    metadata["sim_step_seconds"] = round(float(getattr(sc, "sim_step_seconds", 1.0)), 4)
    metadata["actual_duration_seconds"] = round(
        metadata["actual_n_steps"] * metadata["sim_step_seconds"],
        4,
    )

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
