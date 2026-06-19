"""
OmniSearch Custom VMAS Scenario: Wildfire Survivor Search
==========================================================

Heterogeneous drones (fast, wide lidar) + ground robots (slow, fire-sensitive)
search for survivor landmarks in a 2D world while a cellular-automata fire
spreads over a discrete grid overlaid on the continuous world.

Detection in this scenario is **abstract** (lidar / distance-based) — it's the
MARL training proxy for what the deployed system does with the YOLOv8 person
detector (see `detection/`). Drones scout fast and broad; ground robots
confirm precisely, pay a penalty for entering burning cells, and expend more
travel effort while crossing difficult terrain.

References:
  - VMAS scenarios: https://vmas.readthedocs.io/en/stable/usage/scenarios.html
  - Based on the structure of vmas.scenarios.discovery
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils

from envs.wildfire_defaults import (
    DRONE_CAMERA_FOV_DEG,
    DRONE_FLIGHT_LEVELS_M,
    DRONE_SAFETY_CLEARANCE_M,
    DRONE_SPEED_MPS,
    DRONE_U_MULTIPLIER,
    GROUND_ACCEL_MPS2,
    GROUND_ARRIVAL_DAMPING,
    GROUND_ARRIVAL_SLOWDOWN_M,
    GROUND_CONFIRMATION_RANGE_M,
    GROUND_LIDAR_RANGE_M,
    GROUND_SPEED_MPS,
    SIM_STEP_SECONDS,
)


# Indices into agent position tensors
X, Y = 0, 1

# Land-cover types stored in land_cover_grid. Terrain affects ground robots;
# drones fly above it but observe the map to coordinate ground routes.
LAND_ROAD, LAND_OPEN, LAND_BRUSH, LAND_FOREST, LAND_ROCK, LAND_WATER = range(6)
OBJECT_NONE, OBJECT_TREE, OBJECT_HOUSE = range(3)
DEFAULT_GROUND_APPROACH_REWARD = 0.05
DEFAULT_GROUND_APPROACH_MILESTONE_RADII_M = (75.0, 50.0, 40.0, 30.0, 20.0)
DEFAULT_GROUND_APPROACH_MILESTONE_REWARD_FRACTIONS = (0.4, 0.5, 0.6, 0.8, 1.0)
UGV_PLANNER_HINT_DIM = 5


def _land_cover_values(values, *, water_value: float, name: str) -> tuple[float, ...]:
    """Accept legacy 5-class configs and append the water class default."""
    values = tuple(float(v) for v in values)
    if len(values) == 5:
        return values + (float(water_value),)
    if len(values) == 6:
        return values
    raise ValueError(f"{name} must cover road/open/brush/forest/rock or road/open/brush/forest/rock/water")


class WildfireSearchScenario(BaseScenario):
    """Heterogeneous air-ground survivor search in a spreading wildfire."""

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        # Team composition
        self.n_drones    = kwargs.pop("n_drones", 3)
        self.n_ground    = kwargs.pop("n_ground", 2)
        self.n_survivors = kwargs.pop("n_survivors", 5)
        self.n_agents    = self.n_drones + self.n_ground

        # World geometry
        self.x_semidim = float(kwargs.pop("x_semidim", 1.0))
        self.y_semidim = float(kwargs.pop("y_semidim", 1.0))
        if self.x_semidim <= 0.0 or self.y_semidim <= 0.0:
            raise ValueError("x_semidim and y_semidim must be positive")
        if not math.isclose(self.x_semidim, self.y_semidim, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                "x_semidim and y_semidim must be equal so one simulation-unit "
                "scale preserves circular distances and physical dimensions"
            )
        self.world_scale = self.x_semidim

        # Detection / sensing
        # Drone search uses a downward camera footprint, not a fixed magic
        # radius: altitude * tan(FOV / 2) gives the visible ground radius.
        kwargs.pop("drone_lidar_range", None)  # legacy name; replaced by camera FOV.
        self.drone_camera_fov_deg = kwargs.pop("drone_camera_fov_deg", DRONE_CAMERA_FOV_DEG)
        if not 0.0 < self.drone_camera_fov_deg < 180.0:
            raise ValueError("drone_camera_fov_deg must be between 0 and 180")
        self.drone_camera_half_angle_tan = math.tan(math.radians(self.drone_camera_fov_deg) / 2.0)
        self.ground_lidar_range_sim_override = kwargs.pop("ground_lidar_range", None)
        self.ground_lidar_range_m = max(
            float(kwargs.pop("ground_lidar_range_m", GROUND_LIDAR_RANGE_M)), 0.0,
        )
        self.ground_lidar_range = (
            max(float(self.ground_lidar_range_sim_override), 0.0)
            if self.ground_lidar_range_sim_override is not None
            else 0.20
        )
        self.n_lidar_rays       = kwargs.pop("n_lidar_rays", 12)
        # Physical dimensions are converted after terrain metadata is loaded.
        # The legacy simulation-unit kwargs remain available as explicit
        # overrides for old experiments.
        self.agent_radius_sim_override = kwargs.pop("agent_radius", None)
        self.survivor_radius_sim_override = kwargs.pop("survivor_radius", None)
        self.detection_range_sim_override = kwargs.pop("detection_range", None)
        # Public floor settings use meters and are converted per terrain.
        # The old names remain simulation-unit overrides for legacy manifests.
        legacy_drone_floor = kwargs.pop("drone_min_footprint", None)
        self.drone_min_footprint_sim_override = (
            max(float(legacy_drone_floor), 0.0) if legacy_drone_floor is not None else None
        )
        self.drone_min_footprint_m = max(
            float(kwargs.pop("drone_min_footprint_m", 0.0)), 0.0,
        )
        legacy_ground_floor = kwargs.pop("ground_confirm_min", None)
        self.ground_confirm_min_sim_override = (
            max(float(legacy_ground_floor), 0.0) if legacy_ground_floor is not None else None
        )
        self.ground_confirm_min_m = max(
            float(kwargs.pop("ground_confirm_min_m", 0.0)), 0.0,
        )
        self.agent_radius_m = max(float(kwargs.pop("agent_radius_m", 0.50)), 0.01)
        self.survivor_radius_m = max(float(kwargs.pop("survivor_radius_m", 0.35)), 0.01)
        self.ground_confirmation_range_m = max(
            float(kwargs.pop("ground_confirmation_range_m", GROUND_CONFIRMATION_RANGE_M)), 0.0,
        )
        # Require an unobstructed terrain line-of-sight (not just proximity) for a
        # ground robot to confirm a survivor. When True, confirmation needs both
        # range AND that no intervening terrain ridge rises above the eye->target
        # sight line. This removes the "confirm through a mountain" loophole that a
        # pure-distance check allows, especially at large confirmation ranges.
        self.confirm_requires_los = bool(kwargs.pop("confirm_requires_los", False))
        # Observer (UGV sensor) eye height and survivor target height in meters,
        # added to local terrain elevation when tracing the sight line.
        self.confirm_observer_height_m = max(float(kwargs.pop("confirm_observer_height_m", 1.5)), 0.0)
        self.confirm_target_height_m = max(float(kwargs.pop("confirm_target_height_m", 1.0)), 0.0)
        # Number of samples along the sight line (more = finer occlusion detection).
        self.confirm_los_samples = max(int(kwargs.pop("confirm_los_samples", 24)), 3)
        # Allow drones (EO/IR camera) to CONFIRM survivors from altitude, not just
        # scout them. A drone confirms a survivor inside its camera footprint with a
        # clear top-down sight line. This mirrors real aerial SAR (drones detect and
        # confirm people from above) and is the realistic route to high recall with
        # a fixed, small ground team. Default False keeps ground-only confirmation.
        self.drone_can_confirm = bool(kwargs.pop("drone_can_confirm", False))
        # Per-drone reward for a confirmation it makes (training signal). 0 = off.
        self.r_drone_confirm = float(kwargs.pop("r_drone_confirm", 0.0))
        kwargs.pop("ground_approach_radius", None)  # obsolete triangular approach reward option
        kwargs.pop("ground_approach_radius_m", None)  # obsolete triangular approach reward option
        self.spawn_padding_m = max(float(kwargs.pop("spawn_padding_m", 1.0)), 0.0)
        self.agent_radius = (
            max(float(self.agent_radius_sim_override), 1e-6)
            if self.agent_radius_sim_override is not None
            else 0.04
        )
        self.survivor_radius = (
            max(float(self.survivor_radius_sim_override), 1e-6)
            if self.survivor_radius_sim_override is not None
            else 0.03
        )
        self.detection_range = (
            max(float(self.detection_range_sim_override), 0.0)
            if self.detection_range_sim_override is not None
            else 0.10
        )

        # Fire spread (cellular automata on a discrete grid overlay)
        self.disable_fire = bool(kwargs.pop("disable_fire", False))
        self.fire_grid_size      = kwargs.pop("fire_grid_size", 128)
        self.terrain_reference_grid_size = kwargs.pop("terrain_reference_grid_size", 16)
        self.fire_spread_prob    = kwargs.pop("fire_spread_prob", 0.065)
        self.fire_spread_variability = kwargs.pop("fire_spread_variability", 0.55)
        self.fire_spotting_prob = kwargs.pop("fire_spotting_prob", 0.00008)
        self.fire_wind_spread_weight = max(float(kwargs.pop("fire_wind_spread_weight", 1.25)), 0.0)
        self.fire_slope_spread_weight = max(float(kwargs.pop("fire_slope_spread_weight", 1.65)), 0.0)
        self.fire_moisture_damping = max(float(kwargs.pop("fire_moisture_damping", 1.15)), 0.0)
        self.fire_intensity_decay = min(max(float(kwargs.pop("fire_intensity_decay", 0.82)), 0.0), 1.0)
        self.initial_fire_cells  = kwargs.pop("initial_fire_cells", 1)
        self.initial_fire_area_fraction = kwargs.pop("initial_fire_area_fraction", 0.025)
        global_burnout_min = kwargs.pop("fire_burnout_min_updates", None)
        global_burnout_max = kwargs.pop("fire_burnout_max_updates", None)
        if global_burnout_min is not None or global_burnout_max is not None:
            min_updates = max(int(global_burnout_min if global_burnout_min is not None else 5), 1)
            max_updates = max(int(global_burnout_max if global_burnout_max is not None else 14), min_updates)
            default_burnout_min = (min_updates,) * 6
            default_burnout_max = (max_updates,) * 6
        else:
            # Active-fire residence time by land cover. With the default
            # 6-second fire update period: road/open 0.5-2 min, brush 2-6 min,
            # forest 6-20 min. Rock/water cannot ignite.
            default_burnout_min = (5, 5, 20, 60, 0, 0)
            default_burnout_max = (20, 20, 60, 200, 0, 0)
        land_cover_fire_burnout_min = _land_cover_values(
            kwargs.pop("land_cover_fire_burnout_min_updates", default_burnout_min),
            water_value=0.0,
            name="land_cover_fire_burnout_min_updates",
        )
        land_cover_fire_burnout_max = _land_cover_values(
            kwargs.pop("land_cover_fire_burnout_max_updates", default_burnout_max),
            water_value=0.0,
            name="land_cover_fire_burnout_max_updates",
        )
        self.land_cover_fire_burnout_min_updates = tuple(max(int(round(v)), 0) for v in land_cover_fire_burnout_min)
        self.land_cover_fire_burnout_max_updates = tuple(
            max(int(round(vmax)), self.land_cover_fire_burnout_min_updates[i])
            for i, vmax in enumerate(land_cover_fire_burnout_max)
        )
        positive_min_updates = [v for v in self.land_cover_fire_burnout_min_updates if v > 0]
        self.fire_burnout_min_updates = min(positive_min_updates) if positive_min_updates else 1
        self.fire_burnout_max_updates = max(self.land_cover_fire_burnout_max_updates)
        self.fire_step_interval  = kwargs.pop("fire_step_interval", 3)  # spread every N env steps
        self.smoke_emission = max(float(kwargs.pop("smoke_emission", 0.18)), 0.0)
        self.smoke_decay = min(max(float(kwargs.pop("smoke_decay", 0.985)), 0.0), 1.0)
        self.smoke_diffusion = min(max(float(kwargs.pop("smoke_diffusion", 0.16)), 0.0), 1.0)
        self.smolder_smoke_emission = max(float(kwargs.pop("smolder_smoke_emission", 0.04)), 0.0)
        self.smolder_decay = min(max(float(kwargs.pop("smolder_decay", 0.995)), 0.0), 1.0)
        self.smolder_start_fraction = min(max(float(kwargs.pop("smolder_start_fraction", 0.65)), 0.0), 1.0)
        land_cover_fire_fuel = _land_cover_values(
            kwargs.pop("land_cover_fire_fuel", (0.05, 0.40, 1.10, 1.35, 0.0, 0.0)),
            water_value=0.0,
            name="land_cover_fire_fuel",
        )
        object_fire_fuel = kwargs.pop("object_fire_fuel", (0.0, 0.25, 1.00))
        if len(object_fire_fuel) != 3:
            raise ValueError("object_fire_fuel must cover none/tree/house")
        wind_direction = kwargs.pop("wind_direction", kwargs.pop("smoke_wind", (1, 0)))
        if len(wind_direction) != 2:
            raise ValueError("wind_direction must be a 2D vector")
        self.wind_direction = (float(wind_direction[0]), float(wind_direction[1]))
        self.wind_strength = min(
            max(float(kwargs.pop("wind_strength", kwargs.pop("smoke_wind_strength", 0.06))), 0.0),
            0.95,
        )

        # Ground terrain is loaded from a real terrain cache produced from
        # USGS 3DEP, OpenStreetMap, and optionally LANDFIRE.
        self.terrain_source = kwargs.pop("terrain_source", "real")
        if self.terrain_source != "real":
            raise ValueError("procedural terrain has been removed; use terrain_source='real'")
        self.terrain_place = kwargs.pop("terrain_place", "Malibu Creek State Park, California")
        self.terrain_bbox = kwargs.pop("terrain_bbox", None)
        if self.terrain_bbox is not None and len(self.terrain_bbox) != 4:
            raise ValueError("terrain_bbox must be (west, south, east, north)")
        self.terrain_cache_dir = kwargs.pop("terrain_cache_dir", "data/terrain_cache")
        self.terrain_cache_path = kwargs.pop("terrain_cache_path", None)
        self.max_ground_slope = kwargs.pop("max_ground_slope", 0.70)
        self.slope_cost_weight = kwargs.pop("slope_cost_weight", 2.0)
        self.slope_speed_weight = kwargs.pop("slope_speed_weight", 0.5)
        self.terrain_path_samples = kwargs.pop("terrain_path_samples", 6)
        self.local_map_patch_size = int(kwargs.pop("local_map_patch_size", 3))
        if self.local_map_patch_size < 1 or self.local_map_patch_size % 2 != 1:
            raise ValueError("local_map_patch_size must be a positive odd integer")
        kwargs.pop("ugv_local_map_patch_shift", None)  # obsolete target-lookahead patch option
        self.ugv_planner_hint = str(kwargs.pop("ugv_planner_hint", "none")).replace("-", "_")
        if self.ugv_planner_hint not in {"none", "local_astar"}:
            raise ValueError("ugv_planner_hint must be one of: none, local_astar")
        self.ugv_planner_patch_size = int(kwargs.pop("ugv_planner_patch_size", 11))
        if self.ugv_planner_patch_size < 1 or self.ugv_planner_patch_size % 2 != 1:
            raise ValueError("ugv_planner_patch_size must be a positive odd integer")
        self.ugv_planner_lookahead_cells = min(
            max(int(kwargs.pop("ugv_planner_lookahead_cells", 10)), 1),
            max(self.ugv_planner_patch_size // 2, 1),
        )
        land_cover_costs = _land_cover_values(
            kwargs.pop("land_cover_costs", (0.65, 1.0, 1.5, 2.2, 4.0, 8.0)),
            water_value=8.0,
            name="land_cover_costs",
        )
        land_cover_speeds = _land_cover_values(
            kwargs.pop("land_cover_speeds", (1.0, 0.95, 0.8, 0.7, 0.0, 0.0)),
            water_value=0.0,
            name="land_cover_speeds",
        )

        # 2.5D drone flight: horizontal VMAS motion plus an automatic safe
        # continuous AGL altitude. MSL altitude is derived from local terrain elevation.
        # Meter-based anchors are converted to sim units after loading terrain metadata.
        self.sim_step_seconds = max(
            float(kwargs.pop("sim_step_seconds", SIM_STEP_SECONDS)), 1e-6,
        )
        self.drone_speed_mps = max(
            float(kwargs.pop("drone_speed_mps", DRONE_SPEED_MPS)), 0.0,
        )
        self.uav_boundary_soft_margin_m = max(float(kwargs.pop("uav_boundary_soft_margin_m", 25.0)), 1e-6)
        self.uav_boundary_escape_m = max(float(kwargs.pop("uav_boundary_escape_m", 0.0)), 0.0)
        self.uav_boundary_escape_raw_threshold = max(
            float(kwargs.pop("uav_boundary_escape_raw_threshold", 0.2)), 0.0,
        )
        self.uav_boundary_escape_projected_threshold = max(
            float(kwargs.pop("uav_boundary_escape_projected_threshold", 0.05)), 0.0,
        )
        # Calibrated so a 10 m/s drone reaches cruise speed in roughly one
        # environment step, matching ~8-10 m/s^2 Crazyflie acceleration.
        self.drone_u_multiplier = max(
            float(kwargs.pop("drone_u_multiplier", DRONE_U_MULTIPLIER)), 0.0,
        )
        self.drone_max_speed_sim_override = kwargs.pop("drone_max_speed", None)
        self.drone_max_speed_sim = (
            max(float(self.drone_max_speed_sim_override), 0.0)
            if self.drone_max_speed_sim_override is not None
            else 0.5
        )
        # Ground robots are calibrated in physical units, then converted to
        # VMAS units once the terrain cache gives us sim_units_per_meter.
        self.ground_speed_mps = max(
            float(kwargs.pop("ground_speed_mps", GROUND_SPEED_MPS)), 0.0,
        )
        self.ground_accel_mps2 = max(
            float(kwargs.pop("ground_accel_mps2", GROUND_ACCEL_MPS2)), 0.0,
        )
        self.ground_arrival_slowdown_m = max(
            float(kwargs.pop("ground_arrival_slowdown_m", GROUND_ARRIVAL_SLOWDOWN_M)), 1e-6,
        )
        self.ground_arrival_damping = max(
            float(kwargs.pop("ground_arrival_damping", GROUND_ARRIVAL_DAMPING)), 0.0,
        )
        # Optional minimum physical step. It defaults to zero; the legacy
        # simulation-unit override remains available for old experiments.
        ground_min_step_sim_override = kwargs.pop("ground_min_step_sim", None)
        self.ground_min_step_sim_override = ground_min_step_sim_override
        self.ground_min_step_m = max(float(kwargs.pop("ground_min_step_m", 0.0)), 0.0)
        self.ground_min_step_sim = (
            max(float(ground_min_step_sim_override), 0.0)
            if ground_min_step_sim_override is not None
            else 0.0
        )
        self.ground_max_speed_sim_override = kwargs.pop("ground_max_speed", None)
        self.ground_max_speed_sim = (
            max(float(self.ground_max_speed_sim_override), 0.0)
            if self.ground_max_speed_sim_override is not None
            else 0.2
        )
        ground_u_multiplier_override = kwargs.pop("ground_u_multiplier", None)
        self.ground_u_multiplier = (
            max(float(ground_u_multiplier_override), 0.0)
            if ground_u_multiplier_override is not None
            else max(0.25 * self.ground_accel_mps2, 0.0)
        )
        drone_flight_levels_m = kwargs.pop("drone_flight_levels_m", DRONE_FLIGHT_LEVELS_M)
        drone_flight_levels_override = kwargs.pop("drone_flight_levels", None)
        drone_flight_levels = (
            tuple(float(v) for v in drone_flight_levels_override)
            if drone_flight_levels_override is not None
            else tuple(float(v) for v in drone_flight_levels_m)
        )
        drone_detection_quality = kwargs.pop(
            "drone_detection_quality",
            kwargs.pop("drone_detection_factors", (0.95, 0.75, 0.55)),
        )
        drone_energy_costs = kwargs.pop("drone_energy_costs", (0.0, 0.002, 0.006))
        if not (len(drone_flight_levels) == len(drone_detection_quality) == len(drone_energy_costs)):
            raise ValueError("drone flight levels, detection quality, and energy costs must align")
        if len(drone_flight_levels) < 2:
            raise ValueError("drone_flight_levels must contain at least two values for continuous interpolation")
        self.drone_flight_levels_sim_override = drone_flight_levels_override is not None
        self.drone_flight_levels_m = tuple(max(float(v), 0.0) for v in drone_flight_levels_m)
        drone_cover_detection_factors = kwargs.pop(
            "drone_cover_detection_factors", (1.0, 1.0, 0.72, 0.45, 0.35, 0.95),
        )
        drone_cover_detection_factors = _land_cover_values(
            drone_cover_detection_factors,
            water_value=0.95,
            name="drone_cover_detection_factors",
        )
        self.drone_smoke_detection_factor = kwargs.pop("drone_smoke_detection_factor", 0.55)
        self.drone_perception_path_samples = max(int(kwargs.pop("drone_perception_path_samples", 8)), 2)
        self.drone_smoke_extinction = max(float(kwargs.pop("drone_smoke_extinction", 1.4)), 0.0)
        self.drone_fire_glare_penalty = min(
            max(float(kwargs.pop("drone_fire_glare_penalty", 0.35)), 0.0),
            1.0,
        )
        self.drone_heat_distortion_penalty = min(
            max(float(kwargs.pop("drone_heat_distortion_penalty", 0.20)), 0.0),
            1.0,
        )
        self.drone_edge_detection_floor = kwargs.pop("drone_edge_detection_floor", 0.20)
        self.drone_safety_clearance_sim_override = kwargs.pop("drone_safety_clearance", None)
        self.drone_safety_clearance_m = max(
            float(kwargs.pop("drone_safety_clearance_m", DRONE_SAFETY_CLEARANCE_M)), 0.0,
        )
        self.drone_climb_rate = max(float(kwargs.pop("drone_climb_rate", 0.035)), 0.0)
        self.drone_descent_rate = max(float(kwargs.pop("drone_descent_rate", 0.020)), 0.0)
        self.drone_climb_rate_m = max(float(kwargs.pop("drone_climb_rate_m", 10.0)), 0.0)
        self.drone_descent_rate_m = max(float(kwargs.pop("drone_descent_rate_m", 8.0)), 0.0)
        self.drone_altitude_release_margin_m = max(float(kwargs.pop("drone_altitude_release_margin_m", 10.0)), 0.0)
        self.drone_altitude_release_margin = max(float(kwargs.pop("drone_altitude_release_margin", 0.04)), 0.0)
        self.r_drone_climb_cost = kwargs.pop("r_drone_climb_cost", -0.005)
        self.drone_sensor_max_range = float(max(drone_flight_levels) * self.drone_camera_half_angle_tan)

        # Communication
        self.comms_dropout = kwargs.pop("comms_dropout", 0.0)
        self.survivor_message_distance_scale_m = max(
            float(kwargs.pop("survivor_message_distance_scale_m", 100.0)),
            1e-6,
        )

        # Episode
        self.max_steps = kwargs.pop("max_steps", 500)
        self.known_survivors_at_reset = bool(kwargs.pop("known_survivors_at_reset", False))
        self.survivor_spawn_reference = str(kwargs.pop("survivor_spawn_reference", "auto")).lower()
        if self.survivor_spawn_reference not in {"auto", "ground", "drone"}:
            raise ValueError("survivor_spawn_reference must be one of: auto, ground, drone")
        self.drone_can_confirm = self.drone_can_confirm or bool(
            kwargs.pop("drone_scouts_confirm_survivors", False)
        )
        self.known_survivor_spawn_distance_m = max(
            float(kwargs.pop("known_survivor_spawn_distance_m", 0.0)), 0.0,
        )
        known_spawn_min_m = kwargs.pop("known_survivor_spawn_distance_min_m", None)
        known_spawn_max_m = kwargs.pop("known_survivor_spawn_distance_max_m", None)
        if known_spawn_min_m is None and known_spawn_max_m is None:
            self.known_survivor_spawn_distance_min_m = self.known_survivor_spawn_distance_m
            self.known_survivor_spawn_distance_max_m = self.known_survivor_spawn_distance_m
        else:
            self.known_survivor_spawn_distance_min_m = max(
                float(self.known_survivor_spawn_distance_m if known_spawn_min_m is None else known_spawn_min_m),
                0.0,
            )
            self.known_survivor_spawn_distance_max_m = (
                math.inf
                if known_spawn_max_m is None
                else max(float(known_spawn_max_m), 0.0)
            )
            if (
                math.isfinite(self.known_survivor_spawn_distance_max_m)
                and self.known_survivor_spawn_distance_max_m < self.known_survivor_spawn_distance_min_m
            ):
                raise ValueError("known_survivor_spawn_distance_max_m must be >= known_survivor_spawn_distance_min_m")
        kwargs.pop("action_transform", None)  # obsolete diagnostic option, ignored for old manifests

        # Reward weights
        self.r_found_survivor = kwargs.pop("r_found_survivor", 10.0)
        self.r_drone_scout    = kwargs.pop("r_drone_scout", 2.0)
        self.r_ground_confirm = kwargs.pop("r_ground_confirm", 4.0)
        self.r_time_penalty   = kwargs.pop("r_time_penalty", -0.0005)
        self.r_fire_penalty   = kwargs.pop("r_fire_penalty", -0.20)
        self.r_ground_travel_cost = kwargs.pop("r_ground_travel_cost", -0.01)
        self.r_drone_shaping  = kwargs.pop("r_drone_shaping",  0.30)
        self.r_ground_shaping = kwargs.pop("r_ground_shaping", 0.50)
        self.r_ugv_movement_alignment = kwargs.pop("r_ugv_movement_alignment", 0.20)
        self.r_ugv_planner_progress = max(float(kwargs.pop("r_ugv_planner_progress", 0.0)), 0.0)
        self.r_ugv_stall_penalty = max(float(kwargs.pop("r_ugv_stall_penalty", 0.0)), 0.0)
        self.ugv_stall_displacement_threshold_m = max(
            float(kwargs.pop("ugv_stall_displacement_threshold_m", 0.05)),
            0.0,
        )
        self.ground_progress_scale_m = max(
            float(kwargs.pop(
                "ground_progress_scale_m",
                self.ground_speed_mps * self.sim_step_seconds,
            )),
            1e-6,
        )
        self.ugv_planner_progress_scale_m = max(
            float(kwargs.pop("ugv_planner_progress_scale_m", self.ground_progress_scale_m)),
            1e-6,
        )
        # Coverage reward: maximum team bonus for covering the full map once.
        # Credit uses the physical camera footprint and is split when drones
        # simultaneously claim the same new cell. Default 0.0 keeps it off.
        self.r_coverage = kwargs.pop("r_coverage", 5.0)
        # UAV movement-coverage reward: actual drone displacement in meters
        # times newly covered grid cells, capped per drone per step. This
        # nudges sweeping motion without paying for movement over old coverage.
        self.r_uav_move_coverage = max(float(kwargs.pop("r_uav_move_coverage", 0.0)), 0.0)
        self.r_uav_move_coverage_cap = max(float(kwargs.pop("r_uav_move_coverage_cap", 0.1)), 0.0)
        self.r_uav_overlap = max(float(kwargs.pop("r_uav_overlap", 0.0)), 0.0)
        self.uav_overlap_allowed = min(
            max(float(kwargs.pop("uav_overlap_allowed", 0.60)), 0.0),
            0.999,
        )
        self.r_uav_outside_footprint = max(float(kwargs.pop("r_uav_outside_footprint", 0.0)), 0.0)
        kwargs.pop("coverage_radius_cells", None)  # obsolete fixed-grid footprint
        # Per-step penalty for each survivor that is scouted but not yet
        # confirmed. This makes standing still cost something while survivors
        # wait, breaking the degenerate "ground robots don't move" optimum.
        # Default 0.0 keeps it off (applied to ground robots).
        self.r_pending_penalty = kwargs.pop("r_pending_penalty", 0.0)
        # Ground exploration reward: per-step fraction of NEW map area a ground
        # robot visits. Gives ground robots a reason to sweep the map even before
        # any survivor has been scouted (mirrors the expert's continuous sweep),
        # instead of waiting near spawn. Default 0.0 keeps it off.
        self.r_ground_coverage = kwargs.pop("r_ground_coverage", 0.0)
        self.ground_coverage_radius = float(kwargs.pop("ground_coverage_radius", 0.08))
        # Optional team-coverage observation: a KxK downsampled map of which
        # cells drones have already scouted, plus a global covered-fraction
        # scalar. This gives the policy the "where have I already searched"
        # memory the hand-coded sweep expert has but a reactive policy lacks.
        # 0 = off (observation dim unchanged); 6 gives a 6x6 map (+37 dims).
        self.coverage_obs_grid = int(kwargs.pop("coverage_obs_grid", 0))
        self.local_coverage_obs_grid = int(kwargs.pop("local_coverage_obs_grid", 0))
        if self.local_coverage_obs_grid < 0:
            raise ValueError("local_coverage_obs_grid must be nonnegative")
        if self.local_coverage_obs_grid > 0 and self.local_coverage_obs_grid % 2 != 1:
            raise ValueError("local_coverage_obs_grid must be 0 or a positive odd integer")
        self.local_coverage_obs_radius_m = max(
            float(kwargs.pop("local_coverage_obs_radius_m", 150.0)),
            1e-6,
        )
        # One-time directed-approach milestones for ground robots. The scalar
        # is the final/inner milestone reward; default fractions make 0.05 map
        # to rewards [0.02, 0.025, 0.03, 0.04, 0.05] for 75/50/40/30/20m.
        self.r_ground_approach = max(
            float(kwargs.pop("r_ground_approach", DEFAULT_GROUND_APPROACH_REWARD)),
            0.0,
        )
        milestone_radii_m = kwargs.pop(
            "ground_approach_milestone_radii_m",
            DEFAULT_GROUND_APPROACH_MILESTONE_RADII_M,
        )
        if isinstance(milestone_radii_m, (int, float)):
            milestone_radii_m = (float(milestone_radii_m),)
        self.ground_approach_milestone_radii_m = tuple(float(v) for v in milestone_radii_m)
        if any(v <= 0.0 for v in self.ground_approach_milestone_radii_m):
            raise ValueError("ground_approach_milestone_radii_m must contain positive distances")

        milestone_rewards = kwargs.pop("ground_approach_milestone_rewards", None)
        if milestone_rewards is None:
            if len(self.ground_approach_milestone_radii_m) == len(DEFAULT_GROUND_APPROACH_MILESTONE_REWARD_FRACTIONS):
                reward_fractions = DEFAULT_GROUND_APPROACH_MILESTONE_REWARD_FRACTIONS
            elif len(self.ground_approach_milestone_radii_m) == 1:
                reward_fractions = (1.0,)
            else:
                span = len(self.ground_approach_milestone_radii_m) - 1
                reward_fractions = tuple(0.4 + 0.6 * i / span for i in range(len(self.ground_approach_milestone_radii_m)))
            self.ground_approach_milestone_rewards = tuple(
                self.r_ground_approach * float(fraction)
                for fraction in reward_fractions
            )
        else:
            if isinstance(milestone_rewards, (int, float)):
                milestone_rewards = (float(milestone_rewards),)
            self.ground_approach_milestone_rewards = tuple(float(v) for v in milestone_rewards)
            if len(self.ground_approach_milestone_rewards) != len(self.ground_approach_milestone_radii_m):
                raise ValueError(
                    "ground_approach_milestone_rewards must match "
                    "ground_approach_milestone_radii_m length"
                )
            if any(v < 0.0 for v in self.ground_approach_milestone_rewards):
                raise ValueError("ground_approach_milestone_rewards must be nonnegative")

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # ---- Build world ----
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=300,
            substeps=2,
            drag=0.25,
        )

        survivor_filter: Callable = lambda e: e.name.startswith("survivor")
        drone_collision_filter: Callable = lambda e: getattr(e, "is_drone", False) is True
        survivor_collision_filter: Callable = lambda e: getattr(e, "is_drone", False) is False

        # Drones: fast aerial searchers. Survivor detection is handled by the
        # camera model, not VMAS lidar, so drones do not need to physically
        # collide with survivor landmarks.
        for i in range(self.n_drones):
            agent = Agent(
                name=f"drone_{i}",
                collide=True,
                collision_filter=drone_collision_filter,
                shape=Sphere(radius=self.agent_radius),
                max_speed=self.drone_max_speed_sim,
                u_range=1.0,
                u_multiplier=self.drone_u_multiplier,
                color=Color.BLUE,
                sensors=[],
            )
            agent.is_drone = True
            world.add_agent(agent)

        # Ground robots: slow, narrow lidar, fire-sensitive
        for i in range(self.n_ground):
            agent = Agent(
                name=f"ground_{i}",
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                max_speed=self.ground_max_speed_sim,
                u_range=1.0,
                u_multiplier=self.ground_u_multiplier,
                color=Color.GREEN,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.ground_lidar_range,
                        entity_filter=survivor_filter,
                        render_color=Color.RED,
                    ),
                ],
            )
            agent.is_drone = False
            world.add_agent(agent)

        # Survivor landmarks. They collide with ground robots but not drones;
        # drone detection is handled by the camera model above the terrain.
        self._survivors: List[Landmark] = []
        for i in range(self.n_survivors):
            survivor = Landmark(
                name=f"survivor_{i}",
                collide=True,
                collision_filter=survivor_collision_filter,
                movable=False,
                shape=Sphere(radius=self.survivor_radius),
                color=Color.RED,
            )
            world.add_landmark(survivor)
            self._survivors.append(survivor)

        # ---- Per-batch scenario state ----
        self.found_survivors = torch.zeros(
            batch_dim, self.n_survivors, dtype=torch.bool, device=device,
        )
        self.scouted_survivors = torch.zeros_like(self.found_survivors)
        self.step_drone_detections = torch.zeros(
            batch_dim, self.n_drones, self.n_survivors, dtype=torch.bool, device=device,
        )
        self.step_ground_confirmations = torch.zeros(
            batch_dim, self.n_ground, self.n_survivors, dtype=torch.bool, device=device,
        )
        # Agent-local mission knowledge. A UAV learns from its own detections,
        # a UGV learns from its own confirmations, and connected receivers
        # merge the team's accumulated knowledge into their local memory.
        self.known_survivors_by_agent = torch.zeros(
            batch_dim, self.n_agents, self.n_survivors, dtype=torch.bool, device=device,
        )
        self.confirmed_survivors_by_agent = torch.zeros_like(self.known_survivors_by_agent)
        self.coverage_grid = torch.zeros(
            batch_dim, self.fire_grid_size, self.fire_grid_size, dtype=torch.bool, device=device,
        )
        # Ground-robot visitation map (drives the ground exploration reward).
        self.ground_coverage_grid = torch.zeros(
            batch_dim, self.fire_grid_size, self.fire_grid_size, dtype=torch.bool, device=device,
        )
        self.fire_grid = torch.zeros(
            batch_dim, self.fire_grid_size, self.fire_grid_size,
            dtype=torch.bool, device=device,
        )
        self.burned_grid = torch.zeros_like(self.fire_grid)
        self.fire_age_grid = torch.zeros(
            batch_dim, self.fire_grid_size, self.fire_grid_size,
            dtype=torch.long, device=device,
        )
        self.fire_lifetime_grid = torch.zeros_like(self.fire_age_grid)
        self.fire_intensity_grid = torch.zeros_like(self.fire_grid, dtype=torch.float)
        self.smoke_grid = torch.zeros(
            batch_dim, self.fire_grid_size, self.fire_grid_size,
            dtype=torch.float, device=device,
        )
        self.smolder_grid = torch.zeros_like(self.smoke_grid)
        self.land_cover_grid = torch.full(
            (batch_dim, self.fire_grid_size, self.fire_grid_size),
            LAND_OPEN, dtype=torch.long, device=device,
        )
        self.elevation_grid = torch.zeros_like(self.fire_grid, dtype=torch.float)
        self.slope_grid = torch.zeros_like(self.elevation_grid)
        self.moisture_grid = torch.zeros_like(self.elevation_grid)
        self.fuel_density_grid = torch.zeros_like(self.elevation_grid)
        self.rockiness_grid = torch.zeros_like(self.elevation_grid)
        self.obstacle_type_grid = torch.zeros_like(self.land_cover_grid)
        self.obstacle_height_grid = torch.zeros_like(self.elevation_grid)
        self.required_clearance_grid = torch.zeros_like(self.elevation_grid)
        self.required_clearance_msl_grid = torch.zeros_like(self.elevation_grid)
        initial_clearance = (
            0.03 if self.drone_safety_clearance_sim_override is None
            else max(float(self.drone_safety_clearance_sim_override), 0.0)
        )
        self.drone_safety_clearance_by_env = torch.full(
            (batch_dim,), initial_clearance, dtype=torch.float, device=device,
        )
        self.terrain_sim_units_per_meter = torch.zeros(batch_dim, dtype=torch.float, device=device)
        self.agent_radius_by_env = torch.full(
            (batch_dim,), self.agent_radius, dtype=torch.float, device=device,
        )
        self.survivor_radius_by_env = torch.full(
            (batch_dim,), self.survivor_radius, dtype=torch.float, device=device,
        )
        self.detection_range_by_env = torch.full(
            (batch_dim,), self.detection_range, dtype=torch.float, device=device,
        )
        initial_drone_footprint_floor = (
            self.drone_min_footprint_sim_override
            if self.drone_min_footprint_sim_override is not None
            else 0.0
        )
        self.drone_min_footprint_by_env = torch.full(
            (batch_dim,), initial_drone_footprint_floor, dtype=torch.float, device=device,
        )
        self.ground_approach_milestone_radii_m_tensor = torch.tensor(
            self.ground_approach_milestone_radii_m,
            dtype=torch.float,
            device=device,
        )
        self.ground_approach_milestone_rewards_tensor = torch.tensor(
            self.ground_approach_milestone_rewards,
            dtype=torch.float,
            device=device,
        )
        self.ground_approach_milestones_reached = torch.zeros(
            (
                batch_dim,
                self.n_ground,
                self.n_survivors,
                len(self.ground_approach_milestone_radii_m),
            ),
            dtype=torch.bool,
            device=device,
        )
        self.spawn_padding_by_env = torch.zeros(batch_dim, dtype=torch.float, device=device)
        self.drone_safety_clearance = initial_clearance
        self.drone_flight_levels_by_env = torch.tensor(
            drone_flight_levels, dtype=torch.float, device=device,
        ).view(1, -1).repeat(batch_dim, 1)
        self.drone_min_altitude_by_env = self.drone_flight_levels_by_env[:, 0].clone()
        self.drone_max_altitude_by_env = self.drone_flight_levels_by_env[:, -1].clone()
        self.drone_sensor_max_range_by_env = self.drone_max_altitude_by_env * self.drone_camera_half_angle_tan
        self.traversable_grid = torch.ones_like(self.fire_grid)
        self.mobility_cost_grid = torch.ones_like(self.elevation_grid)
        self.speed_multiplier_grid = torch.ones_like(self.elevation_grid)
        self.land_cover_cost_values = torch.tensor(land_cover_costs, dtype=torch.float, device=device)
        self.land_cover_speed_values = torch.tensor(land_cover_speeds, dtype=torch.float, device=device)
        self.land_cover_fire_fuel = torch.tensor(land_cover_fire_fuel, dtype=torch.float, device=device)
        self.land_cover_fire_burnout_min = torch.tensor(
            self.land_cover_fire_burnout_min_updates, dtype=torch.long, device=device,
        )
        self.land_cover_fire_burnout_max = torch.tensor(
            self.land_cover_fire_burnout_max_updates, dtype=torch.long, device=device,
        )
        self.object_fire_fuel = torch.tensor(object_fire_fuel, dtype=torch.float, device=device)
        self.drone_flight_levels = torch.tensor(drone_flight_levels, dtype=torch.float, device=device)
        self.drone_detection_quality = torch.tensor(drone_detection_quality, dtype=torch.float, device=device)
        self.drone_cover_detection_factors = torch.tensor(
            drone_cover_detection_factors, dtype=torch.float, device=device,
        )
        self.drone_energy_costs = torch.tensor(drone_energy_costs, dtype=torch.float, device=device)
        self.drone_min_altitude = float(self.drone_flight_levels.min().item())
        self.drone_max_altitude = float(self.drone_flight_levels.max().item())
        # Compatibility note: drone_altitude is altitude above ground level
        # (AGL). Absolute MSL altitude is tracked separately.
        self.drone_altitude = torch.zeros(batch_dim, self.n_drones, device=device)
        self.drone_target_altitude = torch.zeros(batch_dim, self.n_drones, device=device)
        self.drone_altitude_msl = torch.zeros(batch_dim, self.n_drones, device=device)
        self.drone_altitude_level = torch.zeros(batch_dim, self.n_drones, dtype=torch.long, device=device)
        self.drone_altitude_quality = torch.ones(batch_dim, self.n_drones, device=device)
        self.drone_energy_cost = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_drone_climb = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_uav_boundary_projection_norm = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_uav_boundary_projection_count = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_uav_boundary_hit = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self._prev_ground_pos = torch.zeros(batch_dim, self.n_ground, 2, device=device)
        self._pre_step_ground_pos = torch.zeros_like(self._prev_ground_pos)
        self._pre_step_drone_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.step_ugv_travel_cost = torch.zeros(batch_dim, self.n_ground, device=device)
        self.step_ugv_proposed_path_blocked = torch.zeros(
            batch_dim, self.n_ground, dtype=torch.bool, device=device,
        )
        self.step_ugv_speed_limited = torch.zeros(
            batch_dim, self.n_ground, dtype=torch.bool, device=device,
        )
        self.step_ugv_path_speed = torch.ones(batch_dim, self.n_ground, device=device)
        self.step_ugv_speed_limit_scale = torch.ones(batch_dim, self.n_ground, device=device)
        self.step_ugv_proposed_displacement_m = torch.zeros(batch_dim, self.n_ground, device=device)
        self.step_ugv_corrected_displacement_m = torch.zeros(batch_dim, self.n_ground, device=device)
        self.step_ugv_actual_displacement_m = torch.zeros(batch_dim, self.n_ground, device=device)
        self.step_ugv_motion_correction_m = torch.zeros(batch_dim, self.n_ground, device=device)
        self.prev_drone_dist  = torch.full((batch_dim, self.n_drones),  float("inf"), device=device)
        self.prev_ground_dist = torch.full((batch_dim, self.n_ground), float("inf"), device=device)
        self.prev_ground_target_idx = torch.full(
            (batch_dim, self.n_ground), -1, dtype=torch.long, device=device,
        )
        self._zero_step_metric_buffers(batch_dim, device)
        self.terrain_source_description = ["real"] * batch_dim
        self.terrain_source_metadata = [{} for _ in range(batch_dim)]

        # Per-agent reward buffers (filled in _compute_step_rewards)
        for agent in world.agents:
            agent.scenario_reward = torch.zeros(batch_dim, device=device)

        return world

    def _zero_step_metric_buffers(self, batch_dim: int, device: torch.device) -> None:
        self.metric_new_scouts = torch.zeros(batch_dim, device=device)
        self.metric_new_confirmations = torch.zeros(batch_dim, device=device)
        self.metric_full_success = torch.zeros(batch_dim, device=device)
        self.metric_reward_team = torch.zeros(batch_dim, device=device)
        self.metric_reward_drone_scout = torch.zeros(batch_dim, device=device)
        self.metric_reward_drone_progress = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_move_coverage = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_overlap = torch.zeros(batch_dim, device=device)
        self.metric_uav_overlap_fraction = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_outside_footprint = torch.zeros(batch_dim, device=device)
        self.metric_uav_outside_footprint_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_boundary_soft_risk = torch.zeros(batch_dim, device=device)
        self.metric_uav_boundary_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_displacement_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_new_coverage_cells = torch.zeros(batch_dim, device=device)
        self.metric_uav_target_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_footprint_radius_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_target_within_footprint = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_progress = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_approach = torch.zeros(batch_dim, device=device)
        self.metric_reward_ground_confirm = torch.zeros(batch_dim, device=device)
        self.metric_reward_coverage = torch.zeros(batch_dim, device=device)
        self.metric_cost_ugv_fire_exposure = torch.zeros(batch_dim, device=device)
        self.metric_cost_ugv_travel = torch.zeros(batch_dim, device=device)
        self.metric_cost_drone_energy = torch.zeros(batch_dim, device=device)
        self.metric_cost_drone_climb = torch.zeros(batch_dim, device=device)
        self.metric_ugv_known_target_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_confirm_range_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_within_confirm_range = torch.zeros(batch_dim, device=device)
        self.metric_ugv_within_12m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_within_15m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_known_target_valid = torch.zeros(batch_dim, device=device)
        self.metric_ugv_same_target = torch.zeros(batch_dim, device=device)
        self.metric_ugv_prev_distance_valid = torch.zeros(batch_dim, device=device)
        self.metric_ugv_progress_gate_active = torch.zeros(batch_dim, device=device)
        self.metric_ugv_target_index = torch.full((batch_dim,), -1.0, device=device)
        self.metric_ugv_ground_progress_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_ground_progress_scaled = torch.zeros(batch_dim, device=device)
        self.metric_ugv_action_alignment = torch.zeros(batch_dim, device=device)
        self.metric_ugv_movement_alignment = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_movement_alignment = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_planner_progress = torch.zeros(batch_dim, device=device)
        self.metric_ugv_planner_progress_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_planner_progress_scaled = torch.zeros(batch_dim, device=device)
        self.metric_ugv_planner_active = torch.zeros(batch_dim, device=device)
        self.metric_ugv_planner_direct_blocked = torch.zeros(batch_dim, device=device)
        self.metric_ugv_planner_detour_needed = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_stall_penalty = torch.zeros(batch_dim, device=device)

    def _reset_step_metric_buffers(self, env_index: int | None = None) -> None:
        buffers = [
            self.metric_new_scouts,
            self.metric_new_confirmations,
            self.metric_full_success,
            self.metric_reward_team,
            self.metric_reward_drone_scout,
            self.metric_reward_drone_progress,
            self.metric_reward_uav_move_coverage,
            self.metric_reward_uav_overlap,
            self.metric_uav_overlap_fraction,
            self.metric_reward_uav_outside_footprint,
            self.metric_uav_outside_footprint_fraction,
            self.metric_uav_boundary_soft_risk,
            self.metric_uav_boundary_distance_m,
            self.metric_uav_displacement_m,
            self.metric_uav_new_coverage_cells,
            self.metric_uav_target_distance_m,
            self.metric_uav_footprint_radius_m,
            self.metric_uav_target_within_footprint,
            self.metric_reward_ugv_progress,
            self.metric_reward_ugv_approach,
            self.metric_reward_ground_confirm,
            self.metric_reward_coverage,
            self.metric_cost_ugv_fire_exposure,
            self.metric_cost_ugv_travel,
            self.metric_cost_drone_energy,
            self.metric_cost_drone_climb,
            self.metric_ugv_known_target_distance_m,
            self.metric_ugv_confirm_range_m,
            self.metric_ugv_within_confirm_range,
            self.metric_ugv_within_12m,
            self.metric_ugv_within_15m,
            self.metric_ugv_known_target_valid,
            self.metric_ugv_same_target,
            self.metric_ugv_prev_distance_valid,
            self.metric_ugv_progress_gate_active,
            self.metric_ugv_target_index,
            self.metric_ugv_ground_progress_m,
            self.metric_ugv_ground_progress_scaled,
            self.metric_ugv_action_alignment,
            self.metric_ugv_movement_alignment,
            self.metric_reward_ugv_movement_alignment,
            self.metric_reward_ugv_planner_progress,
            self.metric_ugv_planner_progress_m,
            self.metric_ugv_planner_progress_scaled,
            self.metric_ugv_planner_active,
            self.metric_ugv_planner_direct_blocked,
            self.metric_ugv_planner_detour_needed,
            self.metric_reward_ugv_stall_penalty,
        ]
        for buffer in buffers:
            if env_index is None:
                buffer.zero_()
            else:
                buffer[env_index] = 0.0

    def _reset_ground_motion_diagnostics(self, env_index: int | None = None) -> None:
        zero_buffers = [
            self.step_ugv_proposed_path_blocked,
            self.step_ugv_speed_limited,
            self.step_ugv_proposed_displacement_m,
            self.step_ugv_corrected_displacement_m,
            self.step_ugv_actual_displacement_m,
            self.step_ugv_motion_correction_m,
        ]
        one_buffers = [
            self.step_ugv_path_speed,
            self.step_ugv_speed_limit_scale,
        ]
        for buffer in zero_buffers:
            if env_index is None:
                buffer.zero_()
            else:
                buffer[env_index] = 0
        for buffer in one_buffers:
            if env_index is None:
                buffer.fill_(1.0)
            else:
                buffer[env_index] = 1.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            entities=self._survivors + self.world.agents,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=(
                2 * self.agent_radius
                + float(self.spawn_padding_by_env.max().item())
            ),
            x_bounds=(-self.x_semidim, self.x_semidim),
            y_bounds=(-self.y_semidim, self.y_semidim),
        )

        if env_index is None:
            self.found_survivors.zero_()
            self.scouted_survivors.zero_()
            self.step_drone_detections.zero_()
            self.step_ground_confirmations.zero_()
            self.known_survivors_by_agent.zero_()
            self.confirmed_survivors_by_agent.zero_()
            self.coverage_grid.zero_()
            self.ground_coverage_grid.zero_()
            self.fire_grid.zero_()
            self.burned_grid.zero_()
            self.fire_age_grid.zero_()
            self.fire_lifetime_grid.zero_()
            self.fire_intensity_grid.zero_()
            self.smoke_grid.zero_()
            self.smolder_grid.zero_()
            self.step_count.zero_()
            self.prev_drone_dist.fill_(float("inf"))
            self.prev_ground_dist.fill_(float("inf"))
            self.prev_ground_target_idx.fill_(-1)
            self.ground_approach_milestones_reached.zero_()
            self._reset_step_metric_buffers()
            self._reset_ground_motion_diagnostics()
            envs_to_seed = range(self.world.batch_dim)
        else:
            self.found_survivors[env_index] = False
            self.scouted_survivors[env_index] = False
            self.step_drone_detections[env_index] = False
            self.step_ground_confirmations[env_index] = False
            self.known_survivors_by_agent[env_index] = False
            self.confirmed_survivors_by_agent[env_index] = False
            self.coverage_grid[env_index] = False
            self.ground_coverage_grid[env_index] = False
            self.fire_grid[env_index] = False
            self.burned_grid[env_index] = False
            self.fire_age_grid[env_index] = 0
            self.fire_lifetime_grid[env_index] = 0
            self.fire_intensity_grid[env_index] = 0.0
            self.smoke_grid[env_index] = 0.0
            self.smolder_grid[env_index] = 0.0
            self.step_count[env_index] = 0
            self.prev_drone_dist[env_index]  = float("inf")
            self.prev_ground_dist[env_index] = float("inf")
            self.prev_ground_target_idx[env_index] = -1
            self.ground_approach_milestones_reached[env_index] = False
            self._reset_step_metric_buffers(env_index)
            self._reset_ground_motion_diagnostics(env_index)
            envs_to_seed = [env_index]

        H = W = self.fire_grid_size
        for b in envs_to_seed:
            self._generate_terrain(b)
            self._place_diagnostic_survivors_near_reference_agents(b)
            if not self.disable_fire:
                self._seed_initial_fire(b, H, W)
            self._initialize_known_survivors_at_reset(b)

        drone_agents = self.world.agents[:self.n_drones]
        if drone_agents:
            drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1)
            if env_index is None:
                self._pre_step_drone_pos = drone_pos.clone()
                self._initialize_drone_altitudes(drone_pos)
                self.step_drone_climb.zero_()
            else:
                self._pre_step_drone_pos[env_index] = drone_pos[env_index]
                one = torch.tensor([env_index], device=drone_pos.device)
                self._initialize_drone_altitudes(drone_pos[env_index:env_index + 1], one)
                self.step_drone_climb[env_index] = 0

        ground_agents = self.world.agents[self.n_drones:]
        if ground_agents:
            ground_pos = torch.stack([a.state.pos for a in ground_agents], dim=1)
            if env_index is None:
                self._prev_ground_pos = ground_pos.clone()
                self._pre_step_ground_pos = ground_pos.clone()
            else:
                self._prev_ground_pos[env_index] = ground_pos[env_index]
                self._pre_step_ground_pos[env_index] = ground_pos[env_index]
        if env_index is None:
            self.step_ugv_travel_cost.zero_()
        else:
            self.step_ugv_travel_cost[env_index] = 0

    def _initialize_known_survivors_at_reset(self, env_index: int) -> None:
        """Optionally start the episode with survivors known to ground agents."""
        if not self.known_survivors_at_reset or self.n_survivors <= 0:
            return
        self.scouted_survivors[env_index] = True
        if self.n_ground > 0:
            self.known_survivors_by_agent[env_index, self.n_drones:, :] = True

    def _place_diagnostic_survivors_near_reference_agents(self, env_index: int) -> None:
        """For diagnostic episodes, place survivors near UGV or UAV starts."""
        if (
            self.known_survivor_spawn_distance_min_m <= 0.0
            and self.known_survivor_spawn_distance_max_m <= 0.0
        ) or self.n_survivors <= 0:
            return

        reference_kind = self.survivor_spawn_reference
        if reference_kind == "auto":
            if not self.known_survivors_at_reset:
                return
            reference_kind = "ground"
        if (
            (reference_kind == "ground" and self.n_ground <= 0)
            or (reference_kind == "drone" and self.n_drones <= 0)
        ):
            return

        candidates = self._land_entity_spawn_candidate_mask(env_index)
        if not bool(candidates.any().item()):
            return

        size = self.fire_grid_size
        device = self.fire_grid.device
        available = candidates.clone()

        reference_agents = (
            self.world.agents[self.n_drones:]
            if reference_kind == "ground"
            else self.world.agents[:self.n_drones]
        )
        for survivor_idx, survivor in enumerate(self._survivors):
            reference_agent = reference_agents[survivor_idx % len(reference_agents)]
            gx, gy = self._positions_to_grid(reference_agent.state.pos[env_index].view(1, 1, 2))
            x, y = int(gx.item()), int(gy.item())
            if not math.isfinite(self.known_survivor_spawn_distance_max_m):
                new_x, new_y = self._sample_known_survivor_cell_beyond_min_distance(
                    env_index=env_index,
                    ground_grid_x=x,
                    ground_grid_y=y,
                    available=available,
                    fallback_candidates=candidates,
                    min_distance_m=self.known_survivor_spawn_distance_min_m,
                )
            else:
                if self.known_survivor_spawn_distance_max_m > self.known_survivor_spawn_distance_min_m:
                    sample = torch.rand((), device=device)
                    target_m = (
                        self.known_survivor_spawn_distance_min_m
                        + sample * (
                            self.known_survivor_spawn_distance_max_m
                            - self.known_survivor_spawn_distance_min_m
                        )
                    )
                else:
                    target_m = torch.tensor(
                        self.known_survivor_spawn_distance_min_m,
                        device=device,
                        dtype=torch.float32,
                    )
                new_x, new_y = self._sample_known_survivor_cell_near_ground(
                    env_index=env_index,
                    ground_grid_x=x,
                    ground_grid_y=y,
                    available=available,
                    fallback_candidates=candidates,
                    target_distance_m=target_m,
                )
            new_pos = self._grid_cell_center_to_world(new_x, new_y, device=device)
            all_pos = survivor.state.pos.clone()
            all_pos[env_index] = new_pos
            survivor.set_pos(all_pos, batch_index=None)
            available[new_y, new_x] = False

    def _sample_known_survivor_cell_near_ground(
        self,
        *,
        env_index: int,
        ground_grid_x: int,
        ground_grid_y: int,
        available: Tensor,
        fallback_candidates: Tensor,
        target_distance_m: Tensor,
    ) -> tuple[int, int]:
        """Sample a survivor cell near a UGV without imposing row-major angle bias."""
        size = self.fire_grid_size
        device = self.fire_grid.device
        yy, xx = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        cell_size_sim = (2.0 * self.x_semidim) / float(size)
        scale = float(self.terrain_sim_units_per_meter[env_index])
        cell_size_m = cell_size_sim / max(scale, 1e-9)
        cell_diag_m = cell_size_m * math.sqrt(2.0)
        target_m = float(target_distance_m)

        dx = (xx - int(ground_grid_x)).float()
        dy = (yy - int(ground_grid_y)).float()
        dist_m = torch.sqrt(dx.square() + dy.square()) * cell_size_m
        angle = torch.atan2(dy, dx)
        target_angle = torch.rand((), device=device) * (2.0 * math.pi) - math.pi
        angular_error = torch.atan2(
            torch.sin(angle - target_angle),
            torch.cos(angle - target_angle),
        ).abs()

        confirm_m = float(self.detection_range_by_env[env_index]) / max(scale, 1e-9)
        min_m = max(confirm_m * 1.5, target_m * 0.25)
        distance_tolerance_m = max(1.5 * cell_diag_m, 0.10 * target_m)
        base_mask = available & (dist_m >= min_m)
        radial_mask = base_mask & ((dist_m - target_m).abs() <= distance_tolerance_m)

        for tolerance_deg in (22.5, 45.0, 90.0, 180.0):
            sector_mask = radial_mask & (angular_error <= math.radians(tolerance_deg))
            if bool(sector_mask.any().item()):
                return self._sample_random_cell_from_mask(sector_mask)

        candidate_mask = radial_mask if bool(radial_mask.any().item()) else base_mask
        if bool(candidate_mask.any().item()):
            radial_error = torch.where(
                candidate_mask,
                (dist_m - target_m).abs(),
                torch.full_like(dist_m, float("inf")),
            ).flatten()
            k = min(128, int(torch.isfinite(radial_error).sum().item()))
            top_indices = torch.topk(radial_error, k=k, largest=False).indices
            choice = top_indices[torch.randint(k, (1,), device=device)].item()
            return int(choice % size), int(choice // size)

        final_mask = available if bool(available.any().item()) else fallback_candidates
        return self._sample_random_cell_from_mask(final_mask)

    def _sample_known_survivor_cell_beyond_min_distance(
        self,
        *,
        env_index: int,
        ground_grid_x: int,
        ground_grid_y: int,
        available: Tensor,
        fallback_candidates: Tensor,
        min_distance_m: float,
    ) -> tuple[int, int]:
        """Sample a known survivor cell with a minimum distance and no max cap."""
        size = self.fire_grid_size
        device = self.fire_grid.device
        yy, xx = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        cell_size_sim = (2.0 * self.x_semidim) / float(size)
        scale = float(self.terrain_sim_units_per_meter[env_index])
        cell_size_m = cell_size_sim / max(scale, 1e-9)

        dx = (xx - int(ground_grid_x)).float()
        dy = (yy - int(ground_grid_y)).float()
        dist_m = torch.sqrt(dx.square() + dy.square()) * cell_size_m
        angle = torch.atan2(dy, dx)
        target_angle = torch.rand((), device=device) * (2.0 * math.pi) - math.pi
        angular_error = torch.atan2(
            torch.sin(angle - target_angle),
            torch.cos(angle - target_angle),
        ).abs()

        confirm_m = float(self.detection_range_by_env[env_index]) / max(scale, 1e-9)
        min_m = max(confirm_m * 1.5, float(min_distance_m))
        base_mask = available & (dist_m >= min_m)

        for tolerance_deg in (22.5, 45.0, 90.0, 180.0):
            sector_mask = base_mask & (angular_error <= math.radians(tolerance_deg))
            if bool(sector_mask.any().item()):
                return self._sample_random_cell_from_mask(sector_mask)

        final_mask = available if bool(available.any().item()) else fallback_candidates
        return self._sample_random_cell_from_mask(final_mask)

    def _sample_random_cell_from_mask(self, mask: Tensor) -> tuple[int, int]:
        flat = mask.flatten().nonzero(as_tuple=False).flatten()
        if flat.numel() == 0:
            raise ValueError("Cannot sample from an empty cell mask")
        choice = int(flat[torch.randint(flat.numel(), (1,), device=flat.device)].item())
        return choice % self.fire_grid_size, choice // self.fire_grid_size

    def _seed_initial_fire(self, env_index: int, height: int, width: int) -> None:
        """Start an irregular compact fire patch with resolution-independent area."""
        fuel = self._fire_fuel_grid(env_index)
        candidates = (fuel > 0.20).flatten().nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(height * width, device=self.fire_grid.device)
        candidate_mask = torch.zeros(height * width, dtype=torch.bool, device=self.fire_grid.device)
        candidate_mask[candidates] = True
        fractional_cells = int(round(height * width * float(self.initial_fire_area_fraction)))
        n_cells = min(
            max(self._scaled_area_cell_count(self.initial_fire_cells, min_cells=1), fractional_cells),
            int(candidates.numel()),
        )
        if float(self.initial_fire_cells) > 0.0 and n_cells > 0:
            seed = candidates[
                torch.randint(candidates.numel(), (1,), device=self.fire_grid.device)
            ].squeeze(0)
            seed_y = torch.div(seed, width, rounding_mode="floor")
            seed_x = seed % width
            scores = self._initial_fire_scores(seed_x, seed_y, fuel, height, width)
            scores = torch.where(candidate_mask, scores, torch.full_like(scores, float("inf")))
            choice = torch.topk(scores, k=n_cells, largest=False).indices
            self._ignite_fire_cells(self._cell_choice_mask(env_index, choice, height, width))

    def _initial_fire_scores(
        self,
        seed_x: Tensor,
        seed_y: Tensor,
        fuel: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Rank cells for a natural-looking ignition patch instead of a circle."""
        device = self.fire_grid.device
        ys = torch.arange(height, device=device).view(height, 1)
        xs = torch.arange(width, device=device).view(1, width)
        dx = (xs - seed_x).float()
        dy = (ys - seed_y).float()

        angle = torch.rand((), device=device) * (2.0 * math.pi)
        stretch = 0.65 + torch.rand((), device=device) * 1.1
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotated_x = cos_a * dx + sin_a * dy
        rotated_y = -sin_a * dx + cos_a * dy
        ellipse = (rotated_x / stretch).square() + (rotated_y * stretch).square()

        texture = torch.rand(height, width, device=device)
        for _ in range(3):
            texture = 0.55 * texture + 0.45 * (self._neighbor_sum(texture.unsqueeze(0)).squeeze(0) / 4.0)
        texture = (texture - texture.min()) / (texture.max() - texture.min()).clamp_min(1e-6)
        fuel_bias = fuel.clamp(0.0, 1.5) / 1.5
        ragged = ellipse * (0.72 + 0.55 * texture) - 0.40 * fuel_bias
        return ragged.flatten()

    def _cell_choice_mask(
        self,
        env_index: int,
        choice: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        mask = torch.zeros_like(self.fire_grid)
        mask[env_index].view(height * width)[choice] = True
        return mask

    def _ignite_fire_cells(self, new_burns: Tensor) -> None:
        """Mark cells as actively burning and assign each a random burn lifetime."""
        if not bool(new_burns.any().item()):
            return
        min_lifetime = self.land_cover_fire_burnout_min[self.land_cover_grid]
        max_lifetime = self.land_cover_fire_burnout_max[self.land_cover_grid]
        lifetime_span = (max_lifetime - min_lifetime + 1).clamp_min(1)
        random_lifetime = min_lifetime + torch.floor(
            torch.rand_like(self.fire_intensity_grid) * lifetime_span.float()
        ).long()
        self.fire_grid = self.fire_grid | new_burns
        self.burned_grid = self.burned_grid | new_burns
        self.fire_age_grid = torch.where(new_burns, torch.zeros_like(self.fire_age_grid), self.fire_age_grid)
        self.fire_lifetime_grid = torch.where(new_burns, random_lifetime, self.fire_lifetime_grid)
        ignition_intensity = self._fire_intensity_potential() * (0.80 + 0.40 * torch.rand_like(self.fire_intensity_grid))
        self.fire_intensity_grid = torch.where(
            new_burns,
            ignition_intensity.clamp(0.25, 1.0),
            self.fire_intensity_grid,
        )

    def _generate_terrain(self, env_index: int) -> None:
        """Load real terrain and place land entities on feasible cells."""
        self._load_real_terrain(env_index)
        self._place_land_entities_uniformly_on_valid_cells(env_index)
        self._refresh_mobility_layers(env_index)

    def _load_real_terrain(self, env_index: int) -> None:
        """Fill terrain tensors from a preprocessed USGS/OSM terrain cache."""
        try:
            from terrain.real_terrain import load_real_terrain
        except ImportError as exc:
            try:
                from omnisearch.terrain.real_terrain import load_real_terrain
            except ImportError:
                raise ImportError(
                    "Could not import the real terrain loader. Run from the "
                    "omnisearch project root or ensure the repo is on PYTHONPATH."
                ) from exc

        terrain = load_real_terrain(
            grid_size=self.fire_grid_size,
            place=self.terrain_place,
            bbox=self.terrain_bbox,
            cache_dir=self.terrain_cache_dir,
            cache_path=self.terrain_cache_path,
        )
        device = self.land_cover_grid.device
        self.land_cover_grid[env_index] = torch.as_tensor(
            terrain.land_cover, dtype=torch.long, device=device,
        )
        self.elevation_grid[env_index] = torch.as_tensor(
            terrain.elevation, dtype=torch.float, device=device,
        ) * self.world_scale
        self.slope_grid[env_index] = torch.as_tensor(
            terrain.slope, dtype=torch.float, device=device,
        )
        self.moisture_grid[env_index] = torch.as_tensor(
            terrain.moisture, dtype=torch.float, device=device,
        )
        self.fuel_density_grid[env_index] = torch.as_tensor(
            terrain.fuel_density, dtype=torch.float, device=device,
        )
        self.rockiness_grid[env_index] = torch.as_tensor(
            terrain.rockiness, dtype=torch.float, device=device,
        )
        self.obstacle_type_grid[env_index] = torch.as_tensor(
            terrain.obstacle_type, dtype=torch.long, device=device,
        )
        self.obstacle_height_grid[env_index] = torch.as_tensor(
            terrain.obstacle_height, dtype=torch.float, device=device,
        ) * self.world_scale
        self.terrain_source_description[env_index] = terrain.source
        self.terrain_source_metadata[env_index] = dict(terrain.metadata)
        sim_units_per_meter = self._terrain_sim_units_per_meter(terrain.metadata)
        self.terrain_sim_units_per_meter[env_index] = sim_units_per_meter
        self._refresh_physical_size_conversions(env_index, sim_units_per_meter)
        self._refresh_ground_sensor_conversions(env_index, sim_units_per_meter)
        self._refresh_drone_unit_conversions(env_index, sim_units_per_meter)
        self._refresh_drone_speed_conversion(sim_units_per_meter)
        self._refresh_ground_speed_conversion(sim_units_per_meter)
        if self.drone_safety_clearance_sim_override is None and sim_units_per_meter > 0.0:
            clearance = self.drone_safety_clearance_m * sim_units_per_meter
        else:
            clearance = self.drone_safety_clearance_by_env[env_index].item()
        self.drone_safety_clearance_by_env[env_index] = max(float(clearance), 0.0)
        self.drone_safety_clearance = float(self.drone_safety_clearance_by_env.mean().item())

    def _refresh_physical_size_conversions(self, env_index: int, sim_units_per_meter: float) -> None:
        """Convert robot, survivor, confirmation, and shaping dimensions from meters."""
        scale = max(float(sim_units_per_meter), 0.0)
        agent_radius = (
            max(float(self.agent_radius_sim_override), 1e-6)
            if self.agent_radius_sim_override is not None
            else max(self.agent_radius_m * scale, 1e-6)
        )
        survivor_radius = (
            max(float(self.survivor_radius_sim_override), 1e-6)
            if self.survivor_radius_sim_override is not None
            else max(self.survivor_radius_m * scale, 1e-6)
        )
        detection_range = (
            max(float(self.detection_range_sim_override), 0.0)
            if self.detection_range_sim_override is not None
            else self.ground_confirmation_range_m * scale
        )
        ground_confirm_floor = (
            self.ground_confirm_min_sim_override
            if self.ground_confirm_min_sim_override is not None
            else self.ground_confirm_min_m * scale
        )
        if ground_confirm_floor > 0.0:
            detection_range = max(detection_range, ground_confirm_floor)
        drone_footprint_floor = (
            self.drone_min_footprint_sim_override
            if self.drone_min_footprint_sim_override is not None
            else self.drone_min_footprint_m * scale
        )
        self.agent_radius_by_env[env_index] = agent_radius
        self.survivor_radius_by_env[env_index] = survivor_radius
        self.detection_range_by_env[env_index] = detection_range
        self.drone_min_footprint_by_env[env_index] = drone_footprint_floor

        # VMAS shape dimensions are shared across a vectorized batch. All
        # environments in this scenario load the same terrain configuration,
        # so their conversion scale is identical.
        self.agent_radius = agent_radius
        self.survivor_radius = survivor_radius
        self.detection_range = detection_range
        for agent in self.world.agents:
            agent.shape._radius = agent_radius
        for survivor in self._survivors:
            survivor.shape._radius = survivor_radius

    def _refresh_ground_sensor_conversions(self, env_index: int, sim_units_per_meter: float) -> None:
        scale = max(float(sim_units_per_meter), 0.0)
        self.ground_lidar_range = (
            max(float(self.ground_lidar_range_sim_override), 0.0)
            if self.ground_lidar_range_sim_override is not None
            else self.ground_lidar_range_m * scale
        )
        self.spawn_padding_by_env[env_index] = self.spawn_padding_m * scale
        self.ground_min_step_sim = (
            max(float(self.ground_min_step_sim_override), 0.0)
            if self.ground_min_step_sim_override is not None
            else self.ground_min_step_m * scale
        )
        for agent in self.world.agents[self.n_drones:]:
            agent.sensors[0]._max_range = self.ground_lidar_range

    def _refresh_drone_unit_conversions(self, env_index: int, sim_units_per_meter: float) -> None:
        if not self.drone_flight_levels_sim_override and sim_units_per_meter > 0.0:
            levels = torch.tensor(
                self.drone_flight_levels_m,
                dtype=torch.float,
                device=self.drone_flight_levels_by_env.device,
            ) * float(sim_units_per_meter)
            self.drone_flight_levels_by_env[env_index] = levels
            self.drone_flight_levels = self.drone_flight_levels_by_env[env_index]
            self.drone_min_altitude_by_env[env_index] = levels.min()
            self.drone_max_altitude_by_env[env_index] = levels.max()
            self.drone_sensor_max_range_by_env[env_index] = levels.max() * self.drone_camera_half_angle_tan
            self.drone_climb_rate = self.drone_climb_rate_m * float(sim_units_per_meter)
            self.drone_descent_rate = self.drone_descent_rate_m * float(sim_units_per_meter)
            self.drone_altitude_release_margin = self.drone_altitude_release_margin_m * float(sim_units_per_meter)
        else:
            levels = self.drone_flight_levels_by_env[env_index]
            self.drone_flight_levels = levels
            self.drone_min_altitude_by_env[env_index] = levels.min()
            self.drone_max_altitude_by_env[env_index] = levels.max()
            self.drone_sensor_max_range_by_env[env_index] = levels.max() * self.drone_camera_half_angle_tan

        self.drone_min_altitude = float(self.drone_min_altitude_by_env[env_index].item())
        self.drone_max_altitude = float(self.drone_max_altitude_by_env[env_index].item())
        self.drone_sensor_max_range = float(self.drone_sensor_max_range_by_env[env_index].item())

    def _refresh_drone_speed_conversion(self, sim_units_per_meter: float) -> None:
        if self.drone_max_speed_sim_override is None and sim_units_per_meter > 0.0:
            world_dt = max(float(getattr(self.world, "dt", 1.0)), 1e-6)
            self.drone_max_speed_sim = (
                self.drone_speed_mps
                * self.sim_step_seconds
                * float(sim_units_per_meter)
                / world_dt
            )
        for agent in self.world.agents[:self.n_drones]:
            agent._max_speed = self.drone_max_speed_sim

    def _refresh_ground_speed_conversion(self, sim_units_per_meter: float) -> None:
        if self.ground_max_speed_sim_override is None and sim_units_per_meter > 0.0:
            world_dt = max(float(getattr(self.world, "dt", 1.0)), 1e-6)
            physical = (
                self.ground_speed_mps
                * self.sim_step_seconds
                * float(sim_units_per_meter)
                / world_dt
            )
            # Floor the velocity cap so robots stay mobile on large terrains.
            self.ground_max_speed_sim = max(physical, self.ground_min_step_sim / world_dt)
        for agent in self.world.agents[self.n_drones:]:
            agent._max_speed = self.ground_max_speed_sim

    def _terrain_sim_units_per_meter(self, metadata: dict) -> float:
        """Return the meter conversion for the configured simulation bounds."""
        candidates = (
            metadata.get("units", {}).get("sim_units_per_meter"),
            metadata.get("inputs", {}).get("usgs_3dep", {}).get("sim_units_per_meter"),
        )
        for value in candidates:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if value_f > 0.0:
                return value_f * self.world_scale

        bounds = metadata.get("inputs", {}).get("usgs_3dep", {}).get("projected_bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            west_m, south_m, east_m, north_m = (float(v) for v in bounds)
            extent_m = max(abs(east_m - west_m), abs(north_m - south_m), 1e-6)
            return 2.0 * self.world_scale / extent_m

        cell_size_m = metadata.get("inputs", {}).get("usgs_3dep", {}).get("cell_size_m")
        try:
            return 2.0 * self.world_scale / max(
                float(cell_size_m) * float(self.fire_grid_size), 1e-6,
            )
        except (TypeError, ValueError):
            return 0.0

    def _place_land_entities_uniformly_on_valid_cells(self, env_index: int) -> None:
        """Place survivors and UGV starts by uniformly sampling feasible terrain cells."""
        candidates = self._land_entity_spawn_candidate_mask(env_index)
        if not bool(candidates.any().item()):
            return

        device = self.fire_grid.device
        available = candidates.clone()
        entities = self._survivors + self.world.agents[self.n_drones:]
        for entity in entities:
            search_mask = available if bool(available.any().item()) else candidates
            new_x, new_y = self._sample_random_cell_from_mask(search_mask)
            new_pos = self._grid_cell_center_to_world(new_x, new_y, device=device)
            all_pos = entity.state.pos.clone()
            all_pos[env_index] = new_pos
            entity.set_pos(all_pos, batch_index=None)
            available[new_y, new_x] = False

    def _land_entity_candidate_mask(self, env_index: int) -> Tensor:
        cover = self.land_cover_grid[env_index]
        objects = self.obstacle_type_grid[env_index]
        slope = self.slope_grid[env_index]
        road = cover == LAND_ROAD
        return (
            (cover != LAND_WATER)
            & (cover != LAND_ROCK)
            & (objects == OBJECT_NONE)
            & ((slope <= self.max_ground_slope) | road)
        )

    def _land_entity_spawn_candidate_mask(self, env_index: int) -> Tensor:
        """Prefer locally navigable land cells while preserving the real terrain grid."""
        candidates = self._land_entity_candidate_mask(env_index)
        if not bool(candidates.any().item()):
            return candidates

        radius = max(
            self._world_length_to_cells(max(self.agent_radius, self.survivor_radius) * 1.4),
            1,
        )
        local_count = self._local_true_count(candidates, radius)
        window_area = float((2 * radius + 1) ** 2)
        min_count = max(3.0, window_area * 0.35)
        preferred = candidates & (local_count >= min_count)
        return preferred if bool(preferred.any().item()) else candidates

    @staticmethod
    def _local_true_count(mask: Tensor, radius: int) -> Tensor:
        """Count true cells in a square neighborhood without wrapping at edges."""
        if radius <= 0:
            return mask.float()
        height, width = mask.shape
        count = torch.zeros(mask.shape, dtype=torch.float, device=mask.device)
        values = mask.float()
        for dy in range(-radius, radius + 1):
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            for dx in range(-radius, radius + 1):
                src_x0 = max(0, -dx)
                src_x1 = min(width, width - dx)
                dst_x0 = max(0, dx)
                dst_x1 = min(width, width + dx)
                count[dst_y0:dst_y1, dst_x0:dst_x1] += values[src_y0:src_y1, src_x0:src_x1]
        return count

    def _grid_cell_center_to_world(self, gx: int, gy: int, *, device: torch.device) -> Tensor:
        x = ((float(gx) + 0.5) / self.fire_grid_size) * (2.0 * self.x_semidim) - self.x_semidim
        y = ((float(gy) + 0.5) / self.fire_grid_size) * (2.0 * self.y_semidim) - self.y_semidim
        return torch.tensor([x, y], dtype=torch.float, device=device)

    def _grid_scale(self) -> float:
        return self.fire_grid_size / max(float(self.terrain_reference_grid_size), 1.0)

    def _scaled_area_cell_count(self, cells_at_reference: float, min_cells: int = 1) -> int:
        return max(int(round(float(cells_at_reference) * self._grid_scale() ** 2)), min_cells)

    def _world_length_to_cells(self, length: float, min_cells: int = 1) -> int:
        return max(int(round(float(length) / (2.0 * self.x_semidim) * self.fire_grid_size)), min_cells)

    def _refresh_mobility_layers(self, env_index: int) -> None:
        cover = self.land_cover_grid[env_index]
        slope = self.slope_grid[env_index]
        objects = self.obstacle_type_grid[env_index]
        road = cover == LAND_ROAD
        traversable = (
            (cover != LAND_WATER) & (cover != LAND_ROCK) & (objects == OBJECT_NONE)
            & ((slope <= self.max_ground_slope) | road)
        )
        cost = self.land_cover_cost_values[cover] * (1.0 + self.slope_cost_weight * slope)
        speed = self.land_cover_speed_values[cover] / (1.0 + self.slope_speed_weight * slope)
        clearance = self.drone_safety_clearance_by_env[env_index]
        self.required_clearance_grid[env_index] = self.obstacle_height_grid[env_index] + clearance
        self.required_clearance_msl_grid[env_index] = (
            self.elevation_grid[env_index] + self.required_clearance_grid[env_index]
        )
        if self.required_clearance_grid[env_index].max() > self.drone_max_altitude_by_env[env_index]:
            raise ValueError("highest drone_flight_levels entry must clear generated obstacles plus safety margin")
        self.traversable_grid[env_index] = traversable
        self.mobility_cost_grid[env_index] = cost
        self.speed_multiplier_grid[env_index] = torch.where(
            traversable, speed.clamp(0.0, 1.0), torch.zeros_like(speed),
        )

    # ------------------------------------------------------------------
    # Per-step hooks
    # ------------------------------------------------------------------
    def pre_step(self):
        """Spread fire, evolve smoke, and bump step counter."""
        ground_agents = self.world.agents[self.n_drones:]
        if ground_agents:
            self._pre_step_ground_pos = torch.stack([a.state.pos for a in ground_agents], dim=1).clone()
            self._reset_ground_motion_diagnostics()
        drone_agents = self.world.agents[:self.n_drones]
        if drone_agents:
            self._pre_step_drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1).clone()
            self.step_uav_boundary_projection_norm.zero_()
            self.step_uav_boundary_projection_count.zero_()
            self.step_uav_boundary_hit.zero_()
        self.step_count += 1

        if int(self.step_count.max().item()) % self.fire_step_interval == 0:
            self._spread_fire()
        self._update_smoke()

    def _spread_fire(self) -> None:
        exposure = self._directional_fire_exposure()
        fuel = self._fire_fuel_grid()
        moisture_factor = torch.exp(-self.fire_moisture_damping * self.moisture_grid.clamp(0.0, 1.0))
        burnable = (fuel > 0.0) & ~self.burned_grid
        random_rate = torch.exp(
            torch.randn(
                self.world.batch_dim, 1, 1,
                device=self.fire_grid.device,
            ) * self.fire_spread_variability
        ).clamp(0.35, 2.25)
        rate = (
            exposure
            * fuel
            * moisture_factor
            * self._grid_scale()
            * random_rate
        )
        p_ignite = 1.0 - (1.0 - self.fire_spread_prob) ** rate.clamp_min(0.0)

        smoke_spotting = self.smoke_grid > 0.08
        p_spot = (
            self.fire_spotting_prob
            * self._grid_scale()
            * random_rate
            * fuel.clamp(0.0, 1.0)
            * moisture_factor
        )
        new_burns = (
            ((torch.rand_like(p_ignite) < p_ignite) | ((torch.rand_like(p_spot) < p_spot) & smoke_spotting))
            & burnable
        )
        self._ignite_fire_cells(new_burns)

        self.fire_age_grid = torch.where(
            self.fire_grid,
            self.fire_age_grid + 1,
            self.fire_age_grid,
        )
        self._update_fire_intensity()
        burned_out = self.fire_grid & (self.fire_age_grid >= self.fire_lifetime_grid.clamp_min(1))
        new_smolder = self.fire_intensity_grid * float(self.smolder_start_fraction)
        self.smolder_grid = torch.where(
            burned_out,
            torch.maximum(self.smolder_grid, new_smolder),
            self.smolder_grid,
        )
        self.fire_grid = self.fire_grid & ~burned_out
        self.fire_intensity_grid = torch.where(
            self.fire_grid,
            self.fire_intensity_grid,
            torch.zeros_like(self.fire_intensity_grid),
        )

    def _directional_fire_exposure(self) -> Tensor:
        """Directional source exposure from burning cells into neighboring cells."""
        source = self.fire_grid.float() * self.fire_intensity_grid.clamp(0.0, 1.0)
        exposure = torch.zeros_like(self.fire_intensity_grid)
        wind_x, wind_y = self._normalized_wind()
        offsets = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        for dx, dy in offsets:
            distance = math.hypot(dx, dy)
            direction_x = dx / distance
            direction_y = dy / distance
            wind_alignment = wind_x * direction_x + wind_y * direction_y
            wind_factor = math.exp(self.fire_wind_spread_weight * self.wind_strength * wind_alignment)

            shifted_source = self._shift_grid_no_wrap(source, dx, dy)
            shifted_elevation = self._shift_grid_no_wrap(self.elevation_grid, dx, dy)
            uphill_rise = (self.elevation_grid - shifted_elevation).clamp(-0.25, 0.25)
            slope_factor = torch.exp(self.fire_slope_spread_weight * uphill_rise).clamp(0.45, 2.25)
            exposure = exposure + shifted_source * wind_factor * slope_factor / distance
        return exposure

    def _update_fire_intensity(self) -> None:
        """Evolve flame intensity from fuel, moisture, slope, and burn age."""
        potential = self._fire_intensity_potential()
        lifetime = self.fire_lifetime_grid.clamp_min(1).float()
        age_fraction = (self.fire_age_grid.float() / lifetime).clamp(0.0, 1.0)
        lifecycle = (1.0 - 0.65 * age_fraction).clamp(0.25, 1.0)
        target_intensity = (potential * lifecycle).clamp(0.0, 1.0)
        active_intensity = 0.58 * self.fire_intensity_grid + 0.42 * target_intensity
        decayed_intensity = self.fire_intensity_grid * self.fire_intensity_decay
        self.fire_intensity_grid = torch.where(self.fire_grid, active_intensity, decayed_intensity)

    def _fire_intensity_potential(self) -> Tensor:
        fuel = (self._fire_fuel_grid() / 1.5).clamp(0.0, 1.0)
        moisture_factor = torch.exp(-0.75 * self.fire_moisture_damping * self.moisture_grid.clamp(0.0, 1.0))
        slope_factor = (1.0 + 0.25 * self.slope_grid.clamp(0.0, 1.0)).clamp(1.0, 1.25)
        return (0.15 + 0.85 * fuel * moisture_factor * slope_factor).clamp(0.0, 1.0)

    def _update_smoke(self) -> None:
        """Emit smoke from active fire and smoldering cells, then diffuse, drift, and decay."""
        smoke = self.smoke_grid * self.smoke_decay
        self.smolder_grid = torch.where(
            self.fire_grid,
            torch.zeros_like(self.smolder_grid),
            self.smolder_grid * self.smolder_decay,
        )
        self.smolder_grid = torch.where(
            self.smolder_grid < 0.005,
            torch.zeros_like(self.smolder_grid),
            self.smolder_grid,
        )
        fuel = self._fire_fuel_grid()
        flame_output = self.fire_grid.float() * self.fire_intensity_grid.clamp(0.15, 1.0)
        smoke = smoke + flame_output * self.smoke_emission * fuel.clamp(0.0, 1.0)
        smoke = smoke + self.smolder_grid * self.smolder_smoke_emission * fuel.clamp(0.0, 1.0)
        neighbor_mean = self._neighbor_sum(smoke) / 4.0
        smoke = smoke + self.smoke_diffusion * (neighbor_mean - smoke)

        shifted = self._wind_advected_grid(smoke)
        if shifted is not None:
            smoke = smoke + self.wind_strength * (shifted - smoke)

        self.smoke_grid = smoke.clamp(0.0, 1.0)

    def _normalized_wind(self) -> tuple[float, float]:
        wind_x, wind_y = self.wind_direction
        magnitude = math.hypot(wind_x, wind_y)
        if magnitude <= 1e-6 or self.wind_strength <= 0:
            return 0.0, 0.0
        return wind_x / magnitude, wind_y / magnitude

    def _wind_cell_offset(self) -> tuple[int, int]:
        wind_x, wind_y = self._normalized_wind()
        return int(round(wind_x)), int(round(wind_y))

    def _wind_advected_grid(self, grid: Tensor) -> Tensor | None:
        wind_x, wind_y = self._normalized_wind()
        if self.wind_strength <= 0 or (wind_x == 0.0 and wind_y == 0.0):
            return None
        abs_x, abs_y = abs(wind_x), abs(wind_y)
        total = abs_x + abs_y
        if total <= 1e-6:
            return None
        advected = torch.zeros_like(grid)
        if abs_x > 1e-6:
            advected = advected + (abs_x / total) * self._shift_grid_no_wrap(grid, 1 if wind_x > 0 else -1, 0)
        if abs_y > 1e-6:
            advected = advected + (abs_y / total) * self._shift_grid_no_wrap(grid, 0, 1 if wind_y > 0 else -1)
        return advected

    def _wind_weighted_neighbor_sum(self, grid: Tensor) -> Tensor:
        padded = torch.zeros(
            grid.shape[0], grid.shape[1] + 2, grid.shape[2] + 2,
            device=grid.device, dtype=grid.dtype,
        )
        padded[:, 1:-1, 1:-1] = grid
        wind_x, wind_y = self._normalized_wind()
        strength = self.wind_strength
        return (
            padded[:, :-2, 1:-1] * (1.0 + strength * wind_y)   # source north, spread south
            + padded[:, 2:, 1:-1] * (1.0 - strength * wind_y)  # source south, spread north
            + padded[:, 1:-1, :-2] * (1.0 + strength * wind_x) # source west, spread east
            + padded[:, 1:-1, 2:] * (1.0 - strength * wind_x)  # source east, spread west
        )

    def _neighbor_sum(self, grid: Tensor) -> Tensor:
        padded = torch.zeros(
            grid.shape[0], grid.shape[1] + 2, grid.shape[2] + 2,
            device=grid.device, dtype=grid.dtype,
        )
        padded[:, 1:-1, 1:-1] = grid
        return (
            padded[:, :-2, 1:-1]    # up
            + padded[:, 2:, 1:-1]   # down
            + padded[:, 1:-1, :-2]  # left
            + padded[:, 1:-1, 2:]   # right
        )

    def _fire_fuel_grid(self, env_index: int | None = None) -> Tensor:
        if env_index is None:
            cover = self.land_cover_grid
            objects = self.obstacle_type_grid
            density = self.fuel_density_grid
        else:
            cover = self.land_cover_grid[env_index]
            objects = self.obstacle_type_grid[env_index]
            density = self.fuel_density_grid[env_index]
        base_fuel = self.land_cover_fire_fuel[cover] * (0.65 + 0.55 * density.clamp(0.0, 1.0))
        fuel = base_fuel + self.object_fire_fuel[objects]
        return fuel.clamp(0.0, 1.5)

    def _shift_grid_no_wrap(self, grid: Tensor, dx: int, dy: int) -> Tensor:
        shifted = torch.zeros_like(grid)
        h, w = grid.shape[-2:]
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)
        if src_x0 < src_x1 and src_y0 < src_y1:
            shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = grid[:, src_y0:src_y1, src_x0:src_x1]
        return shifted

    def process_action(self, agent: Agent):
        """Normalize UGV command magnitude, then apply local terrain traction."""
        if agent.is_drone:
            self._project_drone_action_at_boundary(agent)
            return
        max_action_norm = float(agent.u_range) * float(agent.u_multiplier)
        action_norm = agent.action.u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        action_scale = torch.minimum(
            torch.ones_like(action_norm),
            torch.full_like(action_norm, max_action_norm) / action_norm,
        )
        normalized_action = agent.action.u * action_scale
        speed = self._grid_values_at_positions(
            self.speed_multiplier_grid, agent.state.pos.unsqueeze(1),
        ).squeeze(1)
        agent.action.u = normalized_action * speed.unsqueeze(-1)

    def _project_drone_action_at_boundary(self, agent: Agent) -> None:
        """Remove outward drone command components near the search boundary."""
        action = agent.action.u
        if action is None:
            return

        try:
            drone_idx = int(agent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return
        if drone_idx < 0 or drone_idx >= self.n_drones:
            return

        margin = self.agent_radius
        x_min = -self.x_semidim + self.agent_radius + margin
        x_max = self.x_semidim - self.agent_radius - margin
        y_min = -self.y_semidim + self.agent_radius + margin
        y_max = self.y_semidim - self.agent_radius - margin

        pos = agent.state.pos
        projected = action.clone()
        zeros_x = torch.zeros_like(projected[:, X])
        zeros_y = torch.zeros_like(projected[:, Y])

        near_left = pos[:, X] <= x_min
        near_right = pos[:, X] >= x_max
        near_bottom = pos[:, Y] <= y_min
        near_top = pos[:, Y] >= y_max

        projected[:, X] = torch.where(
            near_left & (projected[:, X] < 0.0),
            zeros_x,
            projected[:, X],
        )
        projected[:, X] = torch.where(
            near_right & (projected[:, X] > 0.0),
            zeros_x,
            projected[:, X],
        )
        projected[:, Y] = torch.where(
            near_bottom & (projected[:, Y] < 0.0),
            zeros_y,
            projected[:, Y],
        )
        projected[:, Y] = torch.where(
            near_top & (projected[:, Y] > 0.0),
            zeros_y,
            projected[:, Y],
        )

        pure_projected = projected
        raw_norm = action.norm(dim=-1)
        projected_norm = pure_projected.norm(dim=-1)
        inward = torch.zeros_like(projected)
        inward[:, X] = torch.where(near_left, torch.ones_like(inward[:, X]), inward[:, X])
        inward[:, X] = torch.where(near_right, -torch.ones_like(inward[:, X]), inward[:, X])
        inward[:, Y] = torch.where(near_bottom, torch.ones_like(inward[:, Y]), inward[:, Y])
        inward[:, Y] = torch.where(near_top, -torch.ones_like(inward[:, Y]), inward[:, Y])
        inward_norm = inward.norm(dim=-1, keepdim=True)
        inward_unit = inward / inward_norm.clamp_min(1e-12)

        max_step_m = max(self.drone_speed_mps * self.sim_step_seconds, 1e-6)
        escape_strength = min(self.uav_boundary_escape_m / max_step_m, 0.25)
        escape_strength = max(escape_strength, 0.0)
        needs_escape = (
            (raw_norm > self.uav_boundary_escape_raw_threshold)
            & (projected_norm < self.uav_boundary_escape_projected_threshold)
            & (inward_norm.squeeze(-1) > 0.0)
            & (escape_strength > 0.0)
        )
        escaped = pure_projected + inward_unit * escape_strength
        projected = torch.where(needs_escape.unsqueeze(-1), escaped, pure_projected)

        removed_norm = (action - pure_projected).norm(dim=-1)
        self.step_uav_boundary_projection_norm[:, drone_idx] = removed_norm
        self.step_uav_boundary_projection_count[:, drone_idx] = (removed_norm > 1e-12).float()
        agent.action.u = projected

    def post_step(self):
        """Apply blocked ground routes and auto-select safe drone altitude."""
        ground_agents = self.world.agents[self.n_drones:]
        if ground_agents:
            ground_pos = torch.stack([a.state.pos for a in ground_agents], dim=1)
            traversable = self._path_is_traversable(self._pre_step_ground_pos, ground_pos)
            meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(ground_pos.device).clamp_min(1e-9)
            for i, agent in enumerate(ground_agents):
                proposed_pos = agent.state.pos.clone()
                start_pos = self._pre_step_ground_pos[:, i]
                blocked = ~traversable[:, i]
                soft_pos = self._soft_blocked_ground_position(
                    start_pos, proposed_pos,
                )
                corrected_pos = torch.where(
                    blocked.unsqueeze(-1), soft_pos, proposed_pos,
                )
                speed = self._terrain_path_speed_multiplier(
                    start_pos, corrected_pos,
                ).clamp(0.0, 1.0)
                base_step = (
                    self.ground_speed_mps
                    * self.sim_step_seconds
                    * self.terrain_sim_units_per_meter.to(corrected_pos.device)
                ).clamp_min(self.ground_min_step_sim)
                # traction (speed) still scales the floored base, so robots
                # remain slower on brush/forest/slopes but never crawl to a halt.
                max_step = base_step * speed
                delta = corrected_pos - start_pos
                dist = delta.norm(dim=-1).clamp_min(1e-12)
                scale = torch.minimum(torch.ones_like(dist), max_step / dist)
                speed_limited_pos = start_pos + delta * scale.unsqueeze(-1)
                soft_vel = (speed_limited_pos - start_pos) / self.world.dt
                corrected_vel = torch.where(
                    (blocked | (scale < 1.0)).unsqueeze(-1), soft_vel, agent.state.vel,
                )

                self.step_ugv_proposed_path_blocked[:, i] = blocked
                self.step_ugv_speed_limited[:, i] = scale < 0.999
                self.step_ugv_path_speed[:, i] = speed
                self.step_ugv_speed_limit_scale[:, i] = scale
                self.step_ugv_proposed_displacement_m[:, i] = (
                    (proposed_pos - start_pos).norm(dim=-1) * meters_per_sim
                )
                self.step_ugv_corrected_displacement_m[:, i] = (
                    (corrected_pos - start_pos).norm(dim=-1) * meters_per_sim
                )
                self.step_ugv_actual_displacement_m[:, i] = (
                    (speed_limited_pos - start_pos).norm(dim=-1) * meters_per_sim
                )
                self.step_ugv_motion_correction_m[:, i] = (
                    (speed_limited_pos - proposed_pos).norm(dim=-1) * meters_per_sim
                )
                agent.set_pos(speed_limited_pos, batch_index=None)
                agent.set_vel(corrected_vel, batch_index=None)

        drone_agents = self.world.agents[:self.n_drones]
        if drone_agents:
            self._clamp_agents_to_world()
            drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1)
            self._update_drone_altitudes(self._pre_step_drone_pos, drone_pos)
        else:
            self._clamp_agents_to_world()

    def _clamp_agents_to_world(self) -> None:
        """Keep agent bodies inside the visible world bounds."""
        x_min, x_max = -self.x_semidim + self.agent_radius, self.x_semidim - self.agent_radius
        y_min, y_max = -self.y_semidim + self.agent_radius, self.y_semidim - self.agent_radius
        for agent_idx, agent in enumerate(self.world.agents):
            pos = agent.state.pos
            clamped = pos.clone()
            clamped[:, X] = clamped[:, X].clamp(x_min, x_max)
            clamped[:, Y] = clamped[:, Y].clamp(y_min, y_max)
            hit_boundary = (clamped != pos).any(dim=-1, keepdim=True)
            if getattr(agent, "is_drone", False) and agent_idx < self.n_drones:
                self.step_uav_boundary_hit[:, agent_idx] = hit_boundary.squeeze(-1).float()
            vel = agent.state.vel.clone()
            outward_x = (
                ((clamped[:, X] <= x_min) & (vel[:, X] < 0.0))
                | ((clamped[:, X] >= x_max) & (vel[:, X] > 0.0))
            )
            outward_y = (
                ((clamped[:, Y] <= y_min) & (vel[:, Y] < 0.0))
                | ((clamped[:, Y] >= y_max) & (vel[:, Y] > 0.0))
            )
            vel[:, X] = torch.where(outward_x, torch.zeros_like(vel[:, X]), vel[:, X])
            vel[:, Y] = torch.where(outward_y, torch.zeros_like(vel[:, Y]), vel[:, Y])
            agent.set_pos(clamped, batch_index=None)
            agent.set_vel(vel, batch_index=None)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, agent: Agent) -> Tensor:
        if agent is self.world.agents[0]:
            self._compute_step_rewards()
        return agent.scenario_reward

    def _confirm_los_mask(self, ground_pos: Tensor, surv_pos: Tensor) -> Tensor:
        """Boolean [B, G, S]: True when terrain does NOT occlude the sight line
        from each ground robot (at eye height) to each survivor (at target
        height). A segment is occluded if any interior sample's terrain elevation
        rises above the straight eye->target sight line (all heights in meters)."""
        # elevation_grid is in vertical SIM units; convert meter heights with the
        # same per-env meters->sim scale used for the horizontal confirm range.
        spm = self.terrain_sim_units_per_meter.view(-1, 1)                         # [B, 1]
        g_elev = self._grid_values_at_positions(self.elevation_grid, ground_pos)  # [B, O]
        eye_msl = g_elev + self.confirm_observer_height_m * spm                    # [B, O]
        return self._los_mask(ground_pos, eye_msl, surv_pos)

    def _drone_confirm_los_mask(self, drone_pos: Tensor, surv_pos: Tensor) -> Tensor:
        """Boolean [B, D, S]: clear top-down sight line from each drone (at its MSL
        flight altitude) to each survivor. Terrain rarely occludes a steep top-down
        view, but a near ridge can — so this is still a real (if mild) constraint."""
        return self._los_mask(drone_pos, self.drone_altitude_msl, surv_pos)

    def _los_mask(self, obs_pos: Tensor, eye_msl: Tensor, surv_pos: Tensor) -> Tensor:
        """Boolean [B, O, S]: True when terrain does NOT occlude the straight sight
        line from each observer (xy at ``eye_msl`` height, sim units) to each
        survivor (ground elevation + target height). Heights are in sim units."""
        samples = self.confirm_los_samples
        spm = self.terrain_sim_units_per_meter.view(-1, 1)                         # [B, 1]
        s_elev = self._grid_values_at_positions(self.elevation_grid, surv_pos)    # [B, S]
        tgt = s_elev + self.confirm_target_height_m * spm                          # [B, S]

        path = self._sample_pair_paths(obs_pos, surv_pos, samples)               # [B, O, S, K, 2]
        terrain = self._grid_values_at_positions(self.elevation_grid, path)       # [B, O, S, K]

        alpha = torch.linspace(0.0, 1.0, samples, device=obs_pos.device).view(1, 1, 1, -1)
        sight = (
            eye_msl.unsqueeze(2).unsqueeze(3) * (1.0 - alpha)
            + tgt.unsqueeze(1).unsqueeze(3) * alpha
        )                                                                          # [B, O, S, K]

        interior = torch.ones(samples, dtype=torch.bool, device=obs_pos.device)
        interior[0] = False
        interior[-1] = False
        eps = 0.5 * self.terrain_sim_units_per_meter.view(-1, 1, 1, 1)  # ~0.5 m tolerance
        occluded = ((terrain > sight + eps) & interior.view(1, 1, 1, -1)).any(dim=-1)
        return ~occluded

    def _compute_step_rewards(self):
        device = self.fire_grid.device

        agent_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B, A, 2]
        surv_pos  = torch.stack([s.state.pos for s in self._survivors], dim=1)    # [B, S, 2]
        dists = torch.cdist(agent_pos, surv_pos)                                  # [B, A, S]

        drone_pos = agent_pos[:, :self.n_drones, :]
        drone_dists = dists[:, :self.n_drones, :]
        drone_seen = self._drone_survivor_detections(drone_dists, drone_pos, surv_pos)
        seen_by_drone       = drone_seen.any(dim=1)
        confirm_range = self.detection_range_by_env.view(-1, 1, 1)
        within_confirm      = dists < confirm_range
        self.step_drone_detections = drone_seen
        sim_units_per_meter_env = self.terrain_sim_units_per_meter.to(device).clamp_min(1e-9)
        meters_per_sim_env = 1.0 / sim_units_per_meter_env

        newly_scouted = seen_by_drone & ~self.scouted_survivors & ~self.found_survivors
        eligible_ground_confirmations = (
            within_confirm[:, self.n_drones:, :]
            & self.scouted_survivors.unsqueeze(1)
        )
        if self.confirm_requires_los:
            # Confirmation also requires an unobstructed terrain sight line, not
            # just proximity — removes the "confirm through a ridge" loophole.
            eligible_ground_confirmations = eligible_ground_confirmations & self._confirm_los_mask(
                agent_pos[:, self.n_drones:, :], surv_pos,
        )
        self.step_ground_confirmations = eligible_ground_confirmations
        confirmed_by_ground = eligible_ground_confirmations.any(dim=1)

        if self.drone_can_confirm:
            # Drones confirm a survivor inside their camera footprint with a clear
            # top-down sight line (realistic EO/IR aerial detection).
            drone_conf = drone_seen
            if self.confirm_requires_los:
                drone_conf = drone_conf & self._drone_confirm_los_mask(drone_pos, surv_pos)
            self.step_drone_confirmations = drone_conf
            confirmed_by_drone = drone_conf.any(dim=1)
        else:
            self.step_drone_confirmations = torch.zeros_like(drone_seen)
            confirmed_by_drone = torch.zeros_like(confirmed_by_ground)

        newly_found = (confirmed_by_ground | confirmed_by_drone) & ~self.found_survivors

        self.scouted_survivors = self.scouted_survivors | newly_scouted
        self.found_survivors   = self.found_survivors   | newly_found
        self._record_local_survivor_knowledge(drone_seen, eligible_ground_confirmations)

        # Dense potential-based shaping (Ng et al. 1999): α · (prev_dist − curr_dist)
        # Drones: target = unscouted survivors
        INF = float("inf")
        unscouted = ~self.scouted_survivors
        drone_d = torch.where(
            unscouted.unsqueeze(1),
            dists[:, :self.n_drones, :],
            torch.full_like(dists[:, :self.n_drones, :], INF),
        )
        curr_drone_dist, _ = drone_d.min(dim=2)
        if self.n_drones > 0 and self.n_survivors > 0:
            nearest_drone_dist_sim = drone_dists.flatten(1).min(dim=1).values
            drone_footprint_m = self._drone_camera_ranges() * meters_per_sim_env.view(-1, 1)
            target_within_footprint = (
                drone_dists <= self._drone_camera_ranges().unsqueeze(-1)
            ).any(dim=(1, 2))
            self.metric_uav_target_distance_m = nearest_drone_dist_sim * meters_per_sim_env
            self.metric_uav_footprint_radius_m = drone_footprint_m.mean(dim=1)
            self.metric_uav_target_within_footprint = target_within_footprint.float()
        else:
            self.metric_uav_target_distance_m = torch.zeros(self.world.batch_dim, device=device)
            self.metric_uav_footprint_radius_m = torch.zeros(self.world.batch_dim, device=device)
            self.metric_uav_target_within_footprint = torch.zeros(self.world.batch_dim, device=device)
        all_scouted = ~unscouted.any(dim=1, keepdim=True)
        prev_known = ~torch.isinf(self.prev_drone_dist) & ~all_scouted
        drone_shaping = torch.where(
            prev_known,
            (self.prev_drone_dist - curr_drone_dist) * self.r_drone_shaping,
            torch.zeros_like(curr_drone_dist),
        )
        self.prev_drone_dist = curr_drone_dist

        # Ground robots: target = locally known, scouted, unconfirmed survivors.
        # Progress is measured in meters and clipped to one nominal UGV step so
        # the shaping coefficient has a terrain-independent reward scale.
        unconfirmed_scouted = self.scouted_survivors & ~self.found_survivors
        ground_known = self.known_survivors_by_agent[:, self.n_drones:, :]
        known_ground_targets = ground_known & unconfirmed_scouted.unsqueeze(1)
        ground_d_sim = torch.where(
            known_ground_targets,
            dists[:, self.n_drones:, :],
            torch.full_like(dists[:, self.n_drones:, :], INF),
        )
        curr_ground_dist_sim, curr_ground_target_idx = ground_d_sim.min(dim=2)
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(curr_ground_dist_sim.device)
        meters_per_sim = torch.where(
            sim_units_per_meter > 1e-9,
            1.0 / sim_units_per_meter,
            torch.ones_like(sim_units_per_meter),
        ).view(-1, 1)
        curr_ground_dist_m = curr_ground_dist_sim * meters_per_sim
        valid_ground_target = ~torch.isinf(curr_ground_dist_m)
        nearest_ground_dist_m = curr_ground_dist_m.min(dim=1).values if self.n_ground > 0 else torch.full(
            (self.world.batch_dim,), INF, device=device,
        )
        valid_any_ground_target = valid_ground_target.any(dim=1) if self.n_ground > 0 else torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=device,
        )
        confirm_range_m = self.detection_range_by_env.to(device) * meters_per_sim.squeeze(1)
        nearest_ground_dist_for_metrics = torch.where(
            valid_any_ground_target,
            nearest_ground_dist_m,
            torch.zeros_like(nearest_ground_dist_m),
        )
        same_ground_target = self.prev_ground_target_idx == curr_ground_target_idx
        prev_distance_valid = ~torch.isinf(self.prev_ground_dist)
        prev_known = (
            prev_distance_valid
            & valid_ground_target
            & same_ground_target
        )
        ground_progress_m = self.prev_ground_dist - curr_ground_dist_m
        ground_progress_scaled = (
            ground_progress_m / self.ground_progress_scale_m
        ).clamp(min=-1.0, max=1.0)
        ground_shaping = torch.where(
            prev_known,
            ground_progress_scaled * self.r_ground_shaping,
            torch.zeros_like(curr_ground_dist_m),
        )
        if self.n_ground > 0:
            ground_pos = agent_pos[:, self.n_drones:, :]
            target_idx_safe = curr_ground_target_idx.clamp(min=0)
            target_pos = torch.gather(
                surv_pos,
                dim=1,
                index=target_idx_safe.unsqueeze(-1).expand(-1, -1, 2),
            )
            target_vec = target_pos - ground_pos
            target_unit = target_vec / target_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)

            action_tensors = [
                agent.action.u
                for agent in self.world.agents[self.n_drones:]
                if agent.action.u is not None
            ]
            if len(action_tensors) == self.n_ground:
                ground_actions = torch.stack(action_tensors, dim=1)
            else:
                ground_actions = torch.zeros_like(ground_pos)
            action_norm = ground_actions.norm(dim=-1, keepdim=True)
            action_alignment = (ground_actions / action_norm.clamp_min(1e-9) * target_unit).sum(dim=-1)
            action_alignment = torch.where(
                prev_known & (action_norm.squeeze(-1) > 1e-6),
                action_alignment.clamp(-1.0, 1.0),
                torch.zeros_like(action_alignment),
            )

            movement_vec = ground_pos - self._pre_step_ground_pos
            movement_norm = movement_vec.norm(dim=-1, keepdim=True)
            movement_alignment = (movement_vec / movement_norm.clamp_min(1e-9) * target_unit).sum(dim=-1)
            movement_alignment = torch.where(
                prev_known & (movement_norm.squeeze(-1) > 1e-9),
                movement_alignment.clamp(-1.0, 1.0),
                torch.zeros_like(movement_alignment),
            )
        else:
            action_alignment = torch.zeros_like(curr_ground_dist_m)
            movement_alignment = torch.zeros_like(curr_ground_dist_m)
        movement_alignment_reward = movement_alignment * self.r_ugv_movement_alignment
        outside_confirm_range = valid_ground_target & (
            curr_ground_dist_m >= confirm_range_m.view(-1, 1)
        )
        if self.n_ground > 0:
            (
                planner_progress_reward,
                planner_progress_m,
                planner_progress_scaled,
                planner_active,
                planner_direct_blocked,
                planner_detour_needed,
            ) = self._ugv_planner_progress_rewards(
                self._pre_step_ground_pos,
                ground_pos,
                target_pos,
                prev_known & outside_confirm_range,
            )
        else:
            planner_progress_reward = torch.zeros_like(curr_ground_dist_m)
            planner_progress_m = torch.zeros_like(curr_ground_dist_m)
            planner_progress_scaled = torch.zeros_like(curr_ground_dist_m)
            planner_active = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            planner_direct_blocked = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            planner_detour_needed = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
        stalled_while_seeking = (
            prev_known
            & outside_confirm_range
            & (self.step_ugv_actual_displacement_m < self.ugv_stall_displacement_threshold_m)
        )
        ugv_stall_penalty = -self.r_ugv_stall_penalty * stalled_while_seeking.float()
        if (
            self.n_ground > 0
            and self.n_survivors > 0
            and self.ground_approach_milestone_rewards_tensor.numel() > 0
        ):
            milestone_radii = self.ground_approach_milestone_radii_m_tensor.view(1, 1, -1)
            milestone_rewards = self.ground_approach_milestone_rewards_tensor.view(1, 1, -1)
            target_idx_safe = curr_ground_target_idx.clamp(min=0)
            milestone_index = target_idx_safe.unsqueeze(-1).unsqueeze(-1).expand(
                -1,
                -1,
                1,
                milestone_radii.shape[-1],
            )
            target_milestones_reached = self.ground_approach_milestones_reached.gather(
                dim=2,
                index=milestone_index,
            ).squeeze(2)
            milestone_gate = (
                (ground_progress_m > 0.0)
                & (movement_alignment > 0.5)
            )
            milestone_crossed = (
                milestone_gate.unsqueeze(-1)
                & prev_known.unsqueeze(-1)
                & (self.prev_ground_dist.unsqueeze(-1) >= milestone_radii)
                & (curr_ground_dist_m.unsqueeze(-1) < milestone_radii)
                & ~target_milestones_reached
            )
            ground_approach = (milestone_crossed.float() * milestone_rewards).sum(dim=-1)
            updated_milestones = target_milestones_reached | milestone_crossed
            self.ground_approach_milestones_reached.scatter_(
                dim=2,
                index=milestone_index,
                src=updated_milestones.unsqueeze(2),
            )
        else:
            ground_approach = torch.zeros_like(curr_ground_dist_m)
        self.prev_ground_dist = torch.where(
            valid_ground_target,
            curr_ground_dist_m,
            torch.full_like(curr_ground_dist_m, INF),
        )
        self.prev_ground_target_idx = torch.where(
            valid_ground_target,
            curr_ground_target_idx,
            torch.full_like(curr_ground_target_idx, -1),
        )

        team_reward = (
            newly_found.float().sum(dim=1) * self.r_found_survivor
            + self.r_time_penalty
        )

        scout_credit_mask    = drone_seen & newly_scouted.unsqueeze(1)
        scout_per_drone      = scout_credit_mask.float().sum(dim=2)         # [B, D]

        drone_confirm_credit = self.step_drone_confirmations & newly_found.unsqueeze(1)
        confirm_per_drone    = drone_confirm_credit.float().sum(dim=2)      # [B, D]

        ground_within        = eligible_ground_confirmations
        confirm_credit_mask  = ground_within & newly_found.unsqueeze(1)
        confirm_per_ground   = confirm_credit_mask.float().sum(dim=2)       # [B, G]

        # Per-step pressure: number of scouted-but-unconfirmed survivors still
        # waiting. Applied to ground robots so idling while survivors are pending
        # is no longer a safe zero-reward option.
        n_pending = unconfirmed_scouted.float().sum(dim=1)  # [B]

        ground_agents = self.world.agents[self.n_drones:]
        ground_in_fire = self._agents_in_fire(ground_agents)  # [B, G]
        ground_cov_new = torch.zeros(
            self.world.batch_dim, max(len(ground_agents), 1), device=device,
        )
        if ground_agents:
            ground_pos = torch.stack([a.state.pos for a in ground_agents], dim=1)
            self.step_ugv_travel_cost = self._terrain_path_cost(self._prev_ground_pos, ground_pos)
            self._prev_ground_pos = ground_pos.clone()
            ground_cov_new = self._ground_coverage_reward(ground_pos)  # [B, G]
        else:
            self.step_ugv_travel_cost.zero_()

        coverage_new, uav_overlap_fraction, uav_outside_footprint_fraction = self._coverage_reward(drone_pos)  # [B, D]
        (
            uav_move_coverage_reward,
            drone_displacement_m,
            coverage_new_cells,
        ) = self._uav_move_coverage_reward(drone_pos, coverage_new)
        uav_overlap_penalty = self._uav_overlap_penalty(uav_overlap_fraction)
        uav_outside_footprint_penalty = self._uav_outside_footprint_penalty(
            uav_outside_footprint_fraction,
        )
        boundary_soft_risk, boundary_distance_m = self._uav_boundary_risk_metrics(drone_pos)

        self.metric_new_scouts = newly_scouted.float().sum(dim=1)
        self.metric_new_confirmations = newly_found.float().sum(dim=1)
        self.metric_full_success = self.found_survivors.all(dim=1).float()
        self.metric_reward_team = team_reward
        self.metric_reward_drone_scout = (scout_per_drone * self.r_drone_scout).sum(dim=1)
        self.metric_reward_drone_progress = drone_shaping.sum(dim=1)
        self.metric_reward_uav_move_coverage = uav_move_coverage_reward.sum(dim=1)
        self.metric_reward_uav_overlap = uav_overlap_penalty.sum(dim=1)
        self.metric_uav_overlap_fraction = (
            uav_overlap_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_reward_uav_outside_footprint = uav_outside_footprint_penalty.sum(dim=1)
        self.metric_uav_outside_footprint_fraction = (
            uav_outside_footprint_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_boundary_soft_risk = boundary_soft_risk.sum(dim=1)
        self.metric_uav_boundary_distance_m = (
            boundary_distance_m.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_displacement_m = drone_displacement_m.sum(dim=1)
        self.metric_uav_new_coverage_cells = coverage_new_cells.sum(dim=1)
        self.metric_reward_ugv_progress = ground_shaping.sum(dim=1)
        self.metric_reward_ugv_approach = ground_approach.sum(dim=1)
        self.metric_reward_ugv_movement_alignment = movement_alignment_reward.sum(dim=1)
        self.metric_reward_ugv_planner_progress = planner_progress_reward.sum(dim=1)
        self.metric_reward_ugv_stall_penalty = ugv_stall_penalty.sum(dim=1)
        self.metric_reward_ground_confirm = (
            confirm_per_ground * self.r_ground_confirm
        ).sum(dim=1)
        self.metric_reward_coverage = (coverage_new * self.r_coverage).sum(dim=1)
        self.metric_cost_ugv_fire_exposure = ground_in_fire.float().sum(dim=1)
        self.metric_cost_ugv_travel = self.step_ugv_travel_cost.sum(dim=1)
        self.metric_cost_drone_energy = self.drone_energy_cost.sum(dim=1)
        self.metric_cost_drone_climb = self.step_drone_climb.sum(dim=1)
        self.metric_ugv_known_target_distance_m = nearest_ground_dist_for_metrics
        self.metric_ugv_confirm_range_m = confirm_range_m
        self.metric_ugv_known_target_valid = valid_ground_target.float().sum(dim=1)
        self.metric_ugv_same_target = (
            same_ground_target & valid_ground_target & prev_distance_valid
        ).float().sum(dim=1)
        self.metric_ugv_prev_distance_valid = (
            prev_distance_valid & valid_ground_target
        ).float().sum(dim=1)
        self.metric_ugv_progress_gate_active = prev_known.float().sum(dim=1)
        if self.n_ground > 0:
            self.metric_ugv_target_index = torch.where(
                valid_any_ground_target,
                curr_ground_target_idx.float().min(dim=1).values,
                torch.full((self.world.batch_dim,), -1.0, device=device),
            )
        else:
            self.metric_ugv_target_index = torch.full((self.world.batch_dim,), -1.0, device=device)
        self.metric_ugv_ground_progress_m = torch.where(
            prev_known,
            ground_progress_m,
            torch.zeros_like(ground_progress_m),
        ).sum(dim=1)
        self.metric_ugv_ground_progress_scaled = torch.where(
            prev_known,
            ground_progress_scaled,
            torch.zeros_like(ground_progress_scaled),
        ).sum(dim=1)
        self.metric_ugv_planner_progress_m = torch.where(
            planner_active,
            planner_progress_m,
            torch.zeros_like(planner_progress_m),
        ).sum(dim=1)
        self.metric_ugv_planner_progress_scaled = torch.where(
            planner_active,
            planner_progress_scaled,
            torch.zeros_like(planner_progress_scaled),
        ).sum(dim=1)
        self.metric_ugv_planner_active = planner_active.float().sum(dim=1)
        self.metric_ugv_planner_direct_blocked = planner_direct_blocked.float().sum(dim=1)
        self.metric_ugv_planner_detour_needed = planner_detour_needed.float().sum(dim=1)
        self.metric_ugv_action_alignment = action_alignment.sum(dim=1)
        self.metric_ugv_movement_alignment = movement_alignment.sum(dim=1)
        self.metric_ugv_within_confirm_range = (
            valid_any_ground_target & (nearest_ground_dist_m < confirm_range_m)
        ).float()
        self.metric_ugv_within_12m = (
            valid_any_ground_target & (nearest_ground_dist_m < 12.0)
        ).float()
        self.metric_ugv_within_15m = (
            valid_any_ground_target & (nearest_ground_dist_m < 15.0)
        ).float()

        for i, agent in enumerate(self.world.agents):
            r = team_reward.clone()
            if agent.is_drone:
                r = r + scout_per_drone[:, i] * self.r_drone_scout
                r = r + confirm_per_drone[:, i] * self.r_drone_confirm
                r = r - self.drone_energy_cost[:, i]
                r = r + self.step_drone_climb[:, i] * self.r_drone_climb_cost
                r = r + drone_shaping[:, i]
                r = r + coverage_new[:, i] * self.r_coverage
                r = r + uav_move_coverage_reward[:, i]
                r = r + uav_overlap_penalty[:, i]
                r = r + uav_outside_footprint_penalty[:, i]
            else:
                g = i - self.n_drones
                r = r + confirm_per_ground[:, g] * self.r_ground_confirm
                r = r + ground_in_fire[:, g].float() * self.r_fire_penalty
                r = r + self.step_ugv_travel_cost[:, g] * self.r_ground_travel_cost
                r = r + ground_shaping[:, g]
                r = r + ground_approach[:, g]
                r = r + n_pending * self.r_pending_penalty
                r = r + ground_cov_new[:, g] * self.r_ground_coverage
                r = r + movement_alignment_reward[:, g]
                r = r + planner_progress_reward[:, g]
                r = r + ugv_stall_penalty[:, g]
            agent.scenario_reward = r

    def _drone_survivor_detections(
        self,
        drone_dists: Tensor,
        drone_pos: Tensor,
        surv_pos: Tensor,
    ) -> Tensor:
        """Stochastic drone scouting from camera footprint and scene quality."""
        components = self._drone_detection_components(drone_dists, drone_pos, surv_pos)
        probability = components["probability"]
        return torch.rand_like(probability) < probability

    def _drone_detection_components(
        self,
        drone_dists: Tensor,
        drone_pos: Tensor,
        surv_pos: Tensor,
    ) -> Dict[str, Tensor]:
        """Deterministic factors behind stochastic drone survivor detection."""
        if self.n_drones == 0:
            shape = (self.world.batch_dim, 0, self.n_survivors)
            probability = torch.zeros(shape, dtype=torch.float, device=self.fire_grid.device)
            return {
                "probability": probability,
                "visible": torch.zeros(shape, dtype=torch.bool, device=self.fire_grid.device),
                "footprint": torch.zeros(self.world.batch_dim, 0, dtype=torch.float, device=self.fire_grid.device),
                "distance_factor": probability,
                "cover_factor": probability,
                "fire_smoke_factor": probability,
                "altitude_quality": probability,
                "survivor_cover": torch.zeros(
                    self.world.batch_dim, self.n_survivors, dtype=torch.long, device=self.fire_grid.device,
                ),
            }
        footprint = self._drone_camera_ranges().unsqueeze(-1)
        visible = drone_dists <= footprint
        normalized_distance = (drone_dists / footprint.clamp_min(1e-6)).clamp(0.0, 1.0)
        distance_factor = 1.0 - (1.0 - self.drone_edge_detection_floor) * normalized_distance.square()

        gx, gy = self._positions_to_grid(surv_pos)
        b_idx = torch.arange(self.world.batch_dim, device=surv_pos.device).view(-1, 1).expand_as(gx)
        survivor_cover = self.land_cover_grid[b_idx, gy, gx]
        cover_factor = self.drone_cover_detection_factors[survivor_cover].unsqueeze(1)
        fire_smoke_factor = self._drone_fire_smoke_visibility_factor(drone_pos, surv_pos)
        altitude_quality = self.drone_altitude_quality.unsqueeze(-1)

        probability = (altitude_quality * distance_factor * cover_factor * fire_smoke_factor).clamp(0.0, 1.0)
        probability = torch.where(visible, probability, torch.zeros_like(probability))
        return {
            "probability": probability,
            "visible": visible,
            "footprint": footprint.squeeze(-1),
            "distance_factor": distance_factor,
            "cover_factor": cover_factor.expand(-1, self.n_drones, -1),
            "fire_smoke_factor": fire_smoke_factor,
            "altitude_quality": altitude_quality.expand(-1, -1, self.n_survivors),
            "survivor_cover": survivor_cover,
        }

    def drone_perception_debug(self, env_index: int = 0) -> List[Dict]:
        """Frame-level drone camera footprint and detection probabilities for visualization."""
        if self.n_drones == 0 or self.n_survivors == 0:
            return []

        def scalar(value: Tensor) -> float:
            return round(float(value.detach().cpu().item()), 4)

        with torch.no_grad():
            drone_agents = self.world.agents[:self.n_drones]
            drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1)
            surv_pos = torch.stack([s.state.pos for s in self._survivors], dim=1)
            drone_dists = torch.cdist(drone_pos, surv_pos)
            components = self._drone_detection_components(drone_dists, drone_pos, surv_pos)

            records: List[Dict] = []
            for drone_idx, agent in enumerate(drone_agents):
                pos = drone_pos[env_index, drone_idx]
                record = {
                    "name": agent.name,
                    "x": scalar(pos[X]),
                    "y": scalar(pos[Y]),
                    "footprint": scalar(components["footprint"][env_index, drone_idx]),
                    "altitude_agl": scalar(self.drone_altitude[env_index, drone_idx]),
                    "altitude_msl": scalar(self.drone_altitude_msl[env_index, drone_idx]),
                    "altitude_level": int(self.drone_altitude_level[env_index, drone_idx].detach().cpu().item()),
                    "survivors": [],
                }
                for survivor_idx, survivor in enumerate(self._survivors):
                    survivor_pos = survivor.state.pos[env_index]
                    record["survivors"].append({
                        "index": survivor_idx,
                        "x": scalar(survivor_pos[X]),
                        "y": scalar(survivor_pos[Y]),
                        "distance": scalar(drone_dists[env_index, drone_idx, survivor_idx]),
                        "visible": bool(components["visible"][env_index, drone_idx, survivor_idx].detach().cpu().item()),
                        "probability": scalar(components["probability"][env_index, drone_idx, survivor_idx]),
                        "distance_factor": scalar(components["distance_factor"][env_index, drone_idx, survivor_idx]),
                        "cover_factor": scalar(components["cover_factor"][env_index, drone_idx, survivor_idx]),
                        "fire_smoke_factor": scalar(
                            components["fire_smoke_factor"][env_index, drone_idx, survivor_idx],
                        ),
                        "altitude_quality": scalar(
                            components["altitude_quality"][env_index, drone_idx, survivor_idx],
                        ),
                        "land_cover": int(components["survivor_cover"][env_index, survivor_idx].detach().cpu().item()),
                    })
                records.append(record)
            return records

    def _drone_fire_smoke_visibility_factor(self, drone_pos: Tensor, surv_pos: Tensor) -> Tensor:
        """Attenuate camera detections by smoke, flame glare, and heat shimmer."""
        path = self._sample_pair_paths(drone_pos, surv_pos, self.drone_perception_path_samples)
        smoke_path = self._grid_values_at_positions(self.smoke_grid, path)
        fire_path = self._grid_values_at_positions(self.fire_intensity_grid, path)

        smoke_mean = smoke_path.mean(dim=-1)
        target_smoke = smoke_path[..., -1]
        smoke_load = 0.65 * smoke_mean + 0.35 * target_smoke
        smoke_factor = torch.exp(-self.drone_smoke_extinction * smoke_load)
        smoke_floor = torch.full_like(smoke_factor, float(self.drone_smoke_detection_factor))
        smoke_factor = torch.maximum(smoke_factor, smoke_floor)

        target_fire_density = self._local_fire_density_at_positions(surv_pos).unsqueeze(1)
        fire_path_mean = fire_path.mean(dim=-1)
        fire_path_max = fire_path.amax(dim=-1)
        glare_load = torch.maximum(fire_path_max, target_fire_density)
        glare_factor = 1.0 - self.drone_fire_glare_penalty * glare_load
        heat_factor = 1.0 - self.drone_heat_distortion_penalty * fire_path_mean

        return (smoke_factor * glare_factor.clamp(0.0, 1.0) * heat_factor.clamp(0.0, 1.0)).clamp(0.0, 1.0)

    def _agents_in_fire(self, agents: List[Agent]) -> Tensor:
        if len(agents) == 0:
            return torch.zeros(self.world.batch_dim, 0, device=self.fire_grid.device)
        pos = torch.stack([a.state.pos for a in agents], dim=1)  # [B, G, 2]
        gx, gy = self._positions_to_grid(pos)
        b_idx = torch.arange(self.world.batch_dim, device=pos.device).view(-1, 1).expand_as(gx)
        return self.fire_grid[b_idx, gy, gx]

    def _terrain_movement_multiplier(self, agents: List[Agent]) -> Tensor:
        """Return terrain travel multipliers underneath the provided agents."""
        if len(agents) == 0:
            return torch.zeros(self.world.batch_dim, 0, device=self.land_cover_grid.device)
        pos = torch.stack([a.state.pos for a in agents], dim=1)
        return self._grid_values_at_positions(self.mobility_cost_grid, pos)

    def _terrain_path_cost(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        env_indices: Tensor | None = None,
    ) -> Tensor:
        """Terrain-weighted path length, sampled between old and new positions."""
        samples = max(int(self.terrain_path_samples), 2)
        alpha = torch.linspace(0.0, 1.0, samples, device=start_pos.device).view(1, 1, -1, 1)
        path = start_pos.unsqueeze(2) + (end_pos - start_pos).unsqueeze(2) * alpha
        multipliers = self._grid_values_at_positions(self.mobility_cost_grid, path, env_indices)
        distance = (end_pos - start_pos).norm(dim=-1)
        return distance * multipliers.mean(dim=-1)

    def _terrain_path_speed_multiplier(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        env_indices: Tensor | None = None,
    ) -> Tensor:
        """Average terrain speed multiplier along a UGV movement segment."""
        single_agent = start_pos.ndim == 2
        if single_agent:
            start_pos = start_pos.unsqueeze(1)
            end_pos = end_pos.unsqueeze(1)
        path = self._sample_path(start_pos, end_pos)
        speed = self._grid_values_at_positions(self.speed_multiplier_grid, path, env_indices).mean(dim=-1)
        return speed.squeeze(1) if single_agent else speed

    def _sample_path(self, start_pos: Tensor, end_pos: Tensor) -> Tensor:
        samples = max(int(self.terrain_path_samples), 2)
        alpha = torch.linspace(0.0, 1.0, samples, device=start_pos.device).view(1, 1, -1, 1)
        return start_pos.unsqueeze(2) + (end_pos - start_pos).unsqueeze(2) * alpha

    def _sample_pair_paths(self, start_pos: Tensor, end_pos: Tensor, samples: int) -> Tensor:
        alpha = torch.linspace(0.0, 1.0, max(int(samples), 2), device=start_pos.device)
        start = start_pos.unsqueeze(2).unsqueeze(3)
        end = end_pos.unsqueeze(1).unsqueeze(3)
        return start + (end - start) * alpha.view(1, 1, 1, -1, 1)

    def _path_is_traversable(self, start_pos: Tensor, end_pos: Tensor) -> Tensor:
        delta = (end_pos - start_pos).abs()
        cells_x = delta[..., X] * self.fire_grid_size / (2.0 * self.x_semidim)
        cells_y = delta[..., Y] * self.fire_grid_size / (2.0 * self.y_semidim)
        crossed_cells = torch.maximum(cells_x, cells_y)
        samples = max(
            int(self.terrain_path_samples),
            int(torch.ceil(crossed_cells.max()).item()) * 2 + 1,
        )
        alpha = torch.linspace(0.0, 1.0, samples, device=start_pos.device).view(1, 1, -1, 1)
        path = start_pos.unsqueeze(2) + (end_pos - start_pos).unsqueeze(2) * alpha
        return self._grid_values_at_positions(self.traversable_grid, path).all(dim=-1)

    def _update_drone_altitudes(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        env_indices: Tensor | None = None,
    ) -> None:
        """Move continuous AGL altitude toward safe path clearance."""
        path = self._sample_path(start_pos, end_pos)
        required = self._grid_values_at_positions(
            self.required_clearance_grid, path, env_indices,
        ).amax(dim=-1)
        min_altitude, max_altitude = self._drone_altitude_bounds(required.shape, required.device, env_indices)
        target = required.clamp(
            min=min_altitude,
            max=max_altitude,
        )
        end_ground_msl = self._grid_values_at_positions(self.elevation_grid, end_pos, env_indices)

        if env_indices is None:
            previous_msl = self.drone_altitude_msl.clone()
            self.drone_target_altitude = target
            self.drone_altitude = self._smooth_altitude_step(
                current=self.drone_altitude,
                target=target,
                env_indices=env_indices,
            )
            self._refresh_drone_altitude_bins()
            self.drone_altitude_msl = end_ground_msl + self.drone_altitude
            self.step_drone_climb = (self.drone_altitude_msl - previous_msl).abs()
        else:
            previous_msl = self.drone_altitude_msl[env_indices].clone()
            self.drone_target_altitude[env_indices] = target
            self.drone_altitude[env_indices] = self._smooth_altitude_step(
                current=self.drone_altitude[env_indices],
                target=target,
                env_indices=env_indices,
            )
            self._refresh_drone_altitude_bins(env_indices)
            self.drone_altitude_msl[env_indices] = end_ground_msl + self.drone_altitude[env_indices]
            self.step_drone_climb[env_indices] = (self.drone_altitude_msl[env_indices] - previous_msl).abs()

    def _initialize_drone_altitudes(
        self,
        drone_pos: Tensor,
        env_indices: Tensor | None = None,
    ) -> None:
        """Place drones at the continuous safe altitude for reset without a climb spike."""
        required = self._grid_values_at_positions(
            self.required_clearance_grid, drone_pos.unsqueeze(2), env_indices,
        ).squeeze(-1)
        min_altitude, max_altitude = self._drone_altitude_bounds(required.shape, required.device, env_indices)
        altitude = required.clamp(
            min=min_altitude,
            max=max_altitude,
        )
        ground_msl = self._grid_values_at_positions(self.elevation_grid, drone_pos, env_indices)
        if env_indices is None:
            self.drone_target_altitude = altitude
            self.drone_altitude = altitude
            self.drone_altitude_msl = ground_msl + altitude
            self._refresh_drone_altitude_bins()
        else:
            self.drone_target_altitude[env_indices] = altitude
            self.drone_altitude[env_indices] = altitude
            self.drone_altitude_msl[env_indices] = ground_msl + altitude
            self._refresh_drone_altitude_bins(env_indices)

    def _smooth_altitude_step(
        self,
        current: Tensor,
        target: Tensor,
        env_indices: Tensor | None = None,
    ) -> Tensor:
        """Rate-limit vertical motion and avoid low/medium chatter."""
        min_altitude, max_altitude = self._drone_altitude_bounds(current.shape, current.device, env_indices)
        hold_height = torch.where(
            current > target,
            torch.maximum(current - self.drone_altitude_release_margin, min_altitude),
            target,
        )
        desired = torch.maximum(target, hold_height)
        delta = desired - current
        climb = torch.full_like(delta, self.drone_climb_rate)
        descent = torch.full_like(delta, self.drone_descent_rate)
        step = torch.where(delta >= 0.0, torch.minimum(delta, climb), torch.maximum(delta, -descent))
        return torch.minimum(torch.maximum(current + step, min_altitude), max_altitude)

    def _refresh_drone_altitude_bins(self, env_indices: Tensor | None = None) -> None:
        altitude = self.drone_altitude if env_indices is None else self.drone_altitude[env_indices]
        levels = self.drone_flight_levels_by_env if env_indices is None else self.drone_flight_levels_by_env[env_indices]
        level, quality, energy = self._continuous_altitude_properties(altitude, levels)
        if env_indices is None:
            self.drone_altitude_level = level
            self.drone_altitude_quality = quality
            self.drone_energy_cost = energy
        else:
            self.drone_altitude_level[env_indices] = level
            self.drone_altitude_quality[env_indices] = quality
            self.drone_energy_cost[env_indices] = energy

    def _drone_altitude_bounds(
        self,
        shape: torch.Size,
        device: torch.device,
        env_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if env_indices is None:
            min_altitude = self.drone_min_altitude_by_env.to(device=device).view(-1, 1)
            max_altitude = self.drone_max_altitude_by_env.to(device=device).view(-1, 1)
        else:
            min_altitude = self.drone_min_altitude_by_env[env_indices].to(device=device).view(-1, 1)
            max_altitude = self.drone_max_altitude_by_env[env_indices].to(device=device).view(-1, 1)
        return min_altitude.expand(shape), max_altitude.expand(shape)

    def _continuous_altitude_properties(self, altitude: Tensor, levels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if levels.ndim == 1:
            levels = levels.view(*((1,) * altitude.ndim), -1).expand(*altitude.shape, -1)
        elif levels.ndim == 2:
            levels = levels.unsqueeze(1).expand(-1, altitude.shape[1], -1)
        quality_values = self.drone_detection_quality
        energy_values = self.drone_energy_costs
        level_idx = (altitude.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        hi = (altitude.unsqueeze(-1) > levels).sum(dim=-1).clamp(max=levels.shape[-1] - 1)
        lo = (hi - 1).clamp(min=0)
        upper = levels.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
        lower = levels.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
        span = (upper - lower).clamp_min(1e-6)
        weight = torch.where(hi == lo, torch.zeros_like(altitude), (altitude - lower) / span)
        quality = quality_values[lo] + weight * (quality_values[hi] - quality_values[lo])
        energy = energy_values[lo] + weight * (energy_values[hi] - energy_values[lo])
        return level_idx.long(), quality, energy

    def _soft_blocked_ground_position(self, start_pos: Tensor, end_pos: Tensor) -> Tensor:
        """Slide or shorten blocked UGV moves instead of freezing in place."""
        delta = end_pos - start_pos
        perp = torch.stack([-delta[..., Y], delta[..., X]], dim=-1)
        zero = torch.zeros_like(delta)
        candidates = torch.stack(
            (
                delta * 0.85,
                delta * 0.60,
                delta * 0.35,
                torch.stack([delta[..., X], zero[..., Y]], dim=-1),
                torch.stack([zero[..., X], delta[..., Y]], dim=-1),
                delta * 0.45 + perp * 0.35,
                delta * 0.45 - perp * 0.35,
                perp * 0.35,
                -perp * 0.35,
                zero,
            ),
            dim=1,
        )
        endpoints = start_pos.unsqueeze(1) + candidates
        endpoints[..., X] = endpoints[..., X].clamp(-self.x_semidim, self.x_semidim)
        endpoints[..., Y] = endpoints[..., Y].clamp(-self.y_semidim, self.y_semidim)
        starts = start_pos.unsqueeze(1).expand_as(endpoints)
        traversable = self._path_is_traversable(starts, endpoints)
        displacement = (endpoints - starts).norm(dim=-1)
        alignment = ((endpoints - starts) * delta.unsqueeze(1)).sum(dim=-1)
        score = 0.7 * displacement + 0.3 * alignment.clamp_min(0.0)
        score = torch.where(traversable, score, torch.full_like(score, float("-inf")))
        best = score.argmax(dim=-1)
        chosen = endpoints.gather(1, best.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        any_safe = traversable.any(dim=-1)
        return torch.where(any_safe.unsqueeze(-1), chosen, start_pos)

    def _coverage_reward(self, drone_pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-drone fraction of the map newly covered by camera footprints.

        All drone claims are calculated before updating ``coverage_grid``.
        Simultaneous overlap is split equally, so total team credit across an
        episode cannot exceed 1.0 regardless of map resolution.
        """
        B = drone_pos.shape[0]
        new_per_drone = torch.zeros(B, self.n_drones, device=drone_pos.device)
        overlap_fraction = torch.zeros_like(new_per_drone)
        outside_footprint_fraction = torch.zeros_like(new_per_drone)
        if self.n_drones == 0:
            return new_per_drone, overlap_fraction, outside_footprint_fraction
        # The coverage grid is updated below even when the coverage reward is
        # off, because it can also feed the team-coverage observation.
        if (
            self.r_coverage <= 0.0
            and self.coverage_obs_grid <= 0
            and self.r_uav_overlap <= 0.0
            and self.r_uav_outside_footprint <= 0.0
        ):
            return new_per_drone, overlap_fraction, outside_footprint_fraction

        G = self.fire_grid_size
        cell_width = 2.0 * self.x_semidim / G
        cell_height = 2.0 * self.y_semidim / G
        xs = torch.linspace(
            -self.x_semidim + cell_width / 2.0,
            self.x_semidim - cell_width / 2.0,
            G,
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        )
        ys = torch.linspace(
            -self.y_semidim + cell_height / 2.0,
            self.y_semidim - cell_height / 2.0,
            G,
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        )

        # Circle-cell intersection uses the nearest point in each cell, so even
        # footprints smaller than one cell remain physically meaningful.
        dx = (
            (xs.view(1, 1, 1, G) - drone_pos[..., X].view(B, self.n_drones, 1, 1)).abs()
            - cell_width / 2.0
        ).clamp_min(0.0)
        dy = (
            (ys.view(1, 1, G, 1) - drone_pos[..., Y].view(B, self.n_drones, 1, 1)).abs()
            - cell_height / 2.0
        ).clamp_min(0.0)
        footprint = self._drone_camera_ranges().view(B, self.n_drones, 1, 1)
        claims = dx.square() + dy.square() <= footprint.square()

        footprint_cells = claims.sum(dim=(-1, -2)).clamp_min(1)
        ideal_footprint_cells = (
            math.pi * footprint.squeeze(-1).squeeze(-1).square() / max(cell_width * cell_height, 1e-12)
        ).clamp_min(1e-6)
        outside_footprint_fraction = (
            (ideal_footprint_cells - footprint_cells.float()) / ideal_footprint_cells
        ).clamp(min=0.0, max=1.0)
        already_covered = claims & self.coverage_grid.unsqueeze(1)
        overlap_fraction = already_covered.float().sum(dim=(-1, -2)) / footprint_cells.float()
        team_claims = claims.any(dim=1)
        newly_covered = team_claims & ~self.coverage_grid
        claim_count = claims.sum(dim=1).clamp_min(1)
        split_credit = (
            claims.float()
            * newly_covered.unsqueeze(1).float()
            / claim_count.unsqueeze(1)
        )
        self.coverage_grid |= team_claims
        return (
            split_credit.sum(dim=(-1, -2)) / float(G * G),
            overlap_fraction,
            outside_footprint_fraction,
        )

    def _uav_move_coverage_reward(self, drone_pos: Tensor, coverage_new: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reward actual UAV displacement only when it produces new coverage."""
        if self.n_drones == 0:
            empty = torch.zeros(self.world.batch_dim, 0, device=drone_pos.device)
            return empty, empty, empty
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(drone_pos.device).clamp_min(1e-9)
        meters_per_sim = 1.0 / sim_units_per_meter
        displacement_sim = (drone_pos - self._pre_step_drone_pos).norm(dim=-1)
        displacement_m = displacement_sim * meters_per_sim.view(-1, 1)
        coverage_new_cells = coverage_new * float(self.fire_grid_size * self.fire_grid_size)
        reward = (
            displacement_m
            * coverage_new_cells
            * self.r_uav_move_coverage
        ).clamp(max=self.r_uav_move_coverage_cap)
        return reward, displacement_m, coverage_new_cells

    def _uav_overlap_penalty(self, overlap_fraction: Tensor) -> Tensor:
        """Penalize only the excess footprint overlap above an allowed fraction."""
        if self.n_drones == 0:
            return torch.zeros(self.world.batch_dim, 0, device=overlap_fraction.device)
        if self.r_uav_overlap <= 0.0:
            return torch.zeros_like(overlap_fraction)
        allowed = min(max(float(self.uav_overlap_allowed), 0.0), 0.999)
        excess = (overlap_fraction - allowed).clamp(min=0.0)
        normalized = (excess / max(1.0 - allowed, 1e-6)).clamp(max=1.0)
        return -self.r_uav_overlap * normalized

    def _uav_outside_footprint_penalty(self, outside_fraction: Tensor) -> Tensor:
        """Penalize camera footprint area that falls outside the searchable map."""
        if self.n_drones == 0:
            return torch.zeros(self.world.batch_dim, 0, device=outside_fraction.device)
        if self.r_uav_outside_footprint <= 0.0:
            return torch.zeros_like(outside_fraction)
        return -self.r_uav_outside_footprint * outside_fraction.clamp(min=0.0, max=1.0)

    def _uav_boundary_risk_metrics(self, drone_pos: Tensor) -> tuple[Tensor, Tensor]:
        """Diagnostic UAV distance-to-boundary risk before the hard world clamp is hit."""
        if self.n_drones == 0:
            empty = torch.zeros(self.world.batch_dim, 0, device=drone_pos.device)
            return empty, empty

        x_min = -self.x_semidim + self.agent_radius
        x_max = self.x_semidim - self.agent_radius
        y_min = -self.y_semidim + self.agent_radius
        y_max = self.y_semidim - self.agent_radius
        distance_sim = torch.stack(
            (
                drone_pos[..., X] - x_min,
                x_max - drone_pos[..., X],
                drone_pos[..., Y] - y_min,
                y_max - drone_pos[..., Y],
            ),
            dim=-1,
        ).amin(dim=-1)
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(drone_pos.device).clamp_min(1e-9)
        distance_m = distance_sim.clamp_min(0.0) / sim_units_per_meter.view(-1, 1)
        margin_m = max(float(self.uav_boundary_soft_margin_m), 1e-6)
        risk = ((margin_m - distance_m) / margin_m).clamp(0.0, 1.0)
        return risk, distance_m

    def _ground_coverage_reward(self, ground_pos: Tensor) -> Tensor:
        """Per-ground-robot fraction of NEW map area visited this step.

        Mirrors the drone coverage reward but uses a fixed visitation radius, so
        ground robots are rewarded for spreading out and sweeping the map rather
        than waiting near spawn for a survivor to be scouted.
        """
        B = ground_pos.shape[0]
        n_ground = ground_pos.shape[1]
        new_per_ground = torch.zeros(B, n_ground, device=ground_pos.device)
        if self.r_ground_coverage <= 0.0 or n_ground == 0:
            return new_per_ground

        G = self.fire_grid_size
        cell_width = 2.0 * self.x_semidim / G
        cell_height = 2.0 * self.y_semidim / G
        xs = torch.linspace(
            -self.x_semidim + cell_width / 2.0, self.x_semidim - cell_width / 2.0,
            G, device=ground_pos.device, dtype=ground_pos.dtype,
        )
        ys = torch.linspace(
            -self.y_semidim + cell_height / 2.0, self.y_semidim - cell_height / 2.0,
            G, device=ground_pos.device, dtype=ground_pos.dtype,
        )
        dx = (
            (xs.view(1, 1, 1, G) - ground_pos[..., X].view(B, n_ground, 1, 1)).abs()
            - cell_width / 2.0
        ).clamp_min(0.0)
        dy = (
            (ys.view(1, 1, G, 1) - ground_pos[..., Y].view(B, n_ground, 1, 1)).abs()
            - cell_height / 2.0
        ).clamp_min(0.0)
        radius = max(self.ground_coverage_radius, 1e-6)
        claims = dx.square() + dy.square() <= radius * radius

        team_claims = claims.any(dim=1)
        newly_covered = team_claims & ~self.ground_coverage_grid
        claim_count = claims.sum(dim=1).clamp_min(1)
        split_credit = (
            claims.float()
            * newly_covered.unsqueeze(1).float()
            / claim_count.unsqueeze(1)
        )
        self.ground_coverage_grid |= team_claims
        return split_credit.sum(dim=(-1, -2)) / float(G * G)

    def _drone_camera_ranges(self) -> Tensor:
        """Ground footprint radius for each drone's current flight altitude.

        Both altitude and horizontal positions use simulation units, so this
        remains physically consistent across terrain scales. An optional
        meter-based floor guarantees a usable scout radius for training while
        remaining consistent across terrains.
        """
        physical = self.drone_altitude * self.drone_camera_half_angle_tan
        floor = self.drone_min_footprint_by_env.view(-1, 1)
        return torch.maximum(physical, floor)

    def _grid_values_at_positions(
        self,
        grid: Tensor,
        pos: Tensor,
        env_indices: Tensor | None = None,
    ) -> Tensor:
        gx, gy = self._positions_to_grid(pos)
        if env_indices is None:
            env_indices = torch.arange(pos.shape[0], device=pos.device)
        expand_shape = (pos.shape[0],) + (1,) * (gx.ndim - 1)
        b_idx = env_indices.view(expand_shape).expand_as(gx)
        return grid[b_idx, gy, gx]

    def _positions_to_grid(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        gx = ((pos[..., X] + self.x_semidim) / (2 * self.x_semidim) * self.fire_grid_size).clamp(
            0, self.fire_grid_size - 1
        ).long()
        gy = ((pos[..., Y] + self.y_semidim) / (2 * self.y_semidim) * self.fire_grid_size).clamp(
            0, self.fire_grid_size - 1
        ).long()
        return gx, gy

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observation(self, agent: Agent) -> Tensor:
        own_pos    = agent.state.pos                # [B, 2]
        own_vel    = agent.state.vel                # [B, 2]
        if agent.is_drone:
            lidar_obs = self.drone_sensor_max_range_by_env.view(-1, 1).expand(
                self.world.batch_dim, self.n_lidar_rays,
            )
        else:
            lidar_obs = agent.sensors[0].measure()     # [B, n_rays]
        fire_local = self._local_fire_density(agent)        # [B, 1]
        flight_state = self._flight_state(agent)             # [B, 2]
        comms_keep = self._communication_keep(agent)
        survivor_messages = self._survivor_message_observations(agent, comms_keep)
        terrain_local = self._local_terrain_features(agent)
        planner_hint = self._ugv_planner_hint_observations(agent)
        neighbor = self._neighbor_observations(agent, comms_keep)  # [B, (A-1)*2]
        parts = [
            own_pos,
            own_vel,
            lidar_obs,
            fire_local,
            terrain_local,
            planner_hint,
            flight_state,
            self._boundary_observation(agent),
            neighbor,
            survivor_messages,
        ]
        if self.coverage_obs_grid > 0:
            parts.append(self._coverage_observation())       # [B, K*K + 1]
        if self.local_coverage_obs_grid > 0:
            parts.append(self._local_coverage_observation(agent))  # [B, K*K]
        return torch.cat(parts, dim=-1)

    def _coverage_observation(self) -> Tensor:
        """Team-coverage situational awareness: a downsampled absolute map of
        already-scouted cells plus the global covered fraction.

        Same for every agent (shared team memory). Lets the policy steer toward
        not-yet-covered regions instead of re-sweeping covered ground.
        """
        import torch.nn.functional as F

        K = self.coverage_obs_grid
        cov = self.coverage_grid.float().unsqueeze(1)            # [B, 1, G, G]
        pooled = F.adaptive_avg_pool2d(cov, (K, K)).flatten(1)   # [B, K*K]
        global_frac = self.coverage_grid.float().mean(dim=(1, 2), keepdim=True).squeeze(1)  # [B, 1]
        return torch.cat([pooled, global_frac], dim=-1)

    def _local_coverage_observation(self, agent: Agent) -> Tensor:
        """Pooled ego-centric coverage patch extracted from the coverage grid.

        ``local_coverage_obs_radius_m`` is converted to coverage-grid cells at
        runtime, so the physical window stays stable across map sizes. Outside
        map cells are filled as covered, then the raw patch is adaptively
        average-pooled to KxK.
        """
        K = self.local_coverage_obs_grid
        if K <= 0:
            return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)

        import torch.nn.functional as F

        pos = agent.state.pos
        device = pos.device
        dtype = pos.dtype
        G = int(self.fire_grid_size)
        cell_width_m = 1.0 / (
            self.terrain_sim_units_per_meter.to(device=device, dtype=dtype).clamp_min(1e-9)
            * (float(G) / (2.0 * float(self.x_semidim)))
        )
        radius_cells = torch.round(float(self.local_coverage_obs_radius_m) / cell_width_m).long().clamp_min(1)
        max_radius = int(radius_cells.max().detach().cpu().item())
        patch_size = 2 * max_radius + 1

        coverage = self.coverage_grid.to(device=device, dtype=dtype)
        gx, gy = self._positions_to_grid(pos)
        out = torch.empty(self.world.batch_dim, K * K, device=device, dtype=dtype)
        for env_idx in range(self.world.batch_dim):
            radius = int(radius_cells[env_idx].detach().cpu().item())
            raw_patch_size = 2 * radius + 1
            patch = torch.ones(raw_patch_size, raw_patch_size, device=device, dtype=dtype)
            x0 = int(gx[env_idx].item()) - radius
            x1 = int(gx[env_idx].item()) + radius + 1
            y0 = int(gy[env_idx].item()) - radius
            y1 = int(gy[env_idx].item()) + radius + 1
            sx0 = max(x0, 0)
            sx1 = min(x1, G)
            sy0 = max(y0, 0)
            sy1 = min(y1, G)
            if sx1 > sx0 and sy1 > sy0:
                px0 = sx0 - x0
                px1 = px0 + (sx1 - sx0)
                py0 = sy0 - y0
                py1 = py0 + (sy1 - sy0)
                patch[py0:py1, px0:px1] = coverage[env_idx, sy0:sy1, sx0:sx1]
            if raw_patch_size != patch_size:
                padded = torch.ones(patch_size, patch_size, device=device, dtype=dtype)
                offset = max_radius - radius
                padded[offset : offset + raw_patch_size, offset : offset + raw_patch_size] = patch
                patch = padded
            pooled = F.adaptive_avg_pool2d(patch.view(1, 1, patch_size, patch_size), (K, K))
            out[env_idx] = pooled.flatten()
        return out

    def _boundary_observation(self, agent: Agent) -> Tensor:
        """Distances to left, right, bottom, top bounds, normalized by footprint."""
        pos = agent.state.pos
        if agent.is_drone and self.n_drones > 0:
            try:
                drone_idx = int(agent.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                drone_idx = 0
            drone_idx = min(max(drone_idx, 0), self.n_drones - 1)
            radius = self._drone_camera_ranges()[:, drone_idx]
        elif self.n_drones > 0:
            radius = self._drone_camera_ranges().mean(dim=1)
        else:
            radius = torch.full((self.world.batch_dim,), self.world_scale, device=pos.device, dtype=pos.dtype)

        radius = radius.to(device=pos.device, dtype=pos.dtype).clamp_min(1e-6).unsqueeze(-1)
        x_min = -self.x_semidim + self.agent_radius
        x_max = self.x_semidim - self.agent_radius
        y_min = -self.y_semidim + self.agent_radius
        y_max = self.y_semidim - self.agent_radius
        distances = torch.stack(
            (
                pos[:, X] - x_min,
                x_max - pos[:, X],
                pos[:, Y] - y_min,
                y_max - pos[:, Y],
            ),
            dim=-1,
        )
        return (distances / radius).clamp(0.0, 1.0)

    def _record_local_survivor_knowledge(
        self,
        drone_detections: Tensor,
        ground_confirmations: Tensor,
    ) -> None:
        """Persist mission events at the agent that directly observed them."""
        if self.n_drones:
            self.known_survivors_by_agent[:, :self.n_drones] |= drone_detections
        if self.n_ground:
            ground_slice = slice(self.n_drones, self.n_agents)
            self.known_survivors_by_agent[:, ground_slice] |= ground_confirmations
            self.confirmed_survivors_by_agent[:, ground_slice] |= ground_confirmations

    def _communication_keep(self, agent: Agent) -> Tensor:
        """Sample one receiver-level communication state for this observation."""
        if self.comms_dropout > 0:
            keep = (
                torch.rand(self.world.batch_dim, 1, device=agent.state.pos.device)
                > self.comms_dropout
            )
        else:
            keep = torch.ones(
                self.world.batch_dim, 1, dtype=torch.bool, device=agent.state.pos.device,
            )
        agent.comms_up = keep[:, 0]
        return keep

    def _survivor_message_observations(self, agent: Agent, comms_keep: Tensor) -> Tensor:
        """Encode known candidates as [known, dx, dy, ux, uy, distance_norm, confirmed]."""
        agent_idx = self.world.agents.index(agent)
        local_known = self.known_survivors_by_agent[:, agent_idx]
        local_confirmed = self.confirmed_survivors_by_agent[:, agent_idx]

        team_known = self.known_survivors_by_agent.any(dim=1)
        team_confirmed = self.confirmed_survivors_by_agent.any(dim=1)
        connected = comms_keep.expand_as(local_known)
        local_known = torch.where(connected, team_known, local_known)
        local_confirmed = torch.where(connected, team_confirmed, local_confirmed)

        self.known_survivors_by_agent[:, agent_idx] = local_known
        self.confirmed_survivors_by_agent[:, agent_idx] = local_confirmed

        survivor_pos = torch.stack([s.state.pos for s in self._survivors], dim=1)
        relative_pos = survivor_pos - agent.state.pos.unsqueeze(1)
        relative_pos = relative_pos * local_known.unsqueeze(-1).float()
        dist_sim = torch.linalg.norm(relative_pos, dim=-1, keepdim=True)
        unit_direction = relative_pos / dist_sim.clamp_min(1e-9)
        distance_m = dist_sim / self.terrain_sim_units_per_meter.view(-1, 1, 1).clamp_min(1e-9)
        distance_norm = distance_m / self.survivor_message_distance_scale_m
        features = torch.cat(
            [
                local_known.unsqueeze(-1).float(),
                relative_pos,
                unit_direction,
                distance_norm,
                local_confirmed.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        return features.flatten(start_dim=1)

    def _local_fire_density(self, agent: Agent) -> Tensor:
        pos = agent.state.pos
        return self._local_fire_density_at_positions(pos).unsqueeze(-1)

    def _local_fire_density_at_positions(self, pos: Tensor) -> Tensor:
        gx = ((pos[..., X] + self.x_semidim) / (2 * self.x_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2
        ).long()
        gy = ((pos[..., Y] + self.y_semidim) / (2 * self.y_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2
        ).long()
        b_idx = torch.arange(self.world.batch_dim, device=pos.device).view(
            (pos.shape[0],) + (1,) * (gx.ndim - 1)
        ).expand_as(gx)
        density = torch.zeros_like(gx, dtype=torch.float)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                density = density + self.fire_grid[b_idx, gy + dy, gx + dx].float()
        return density / 9.0

    def _local_terrain_features(self, agent: Agent) -> Tensor:
        """Expose mobility cost, blocked masks, and AGL air-clearance requirements."""
        pos = agent.state.pos
        clearance = self._local_grid_patch(self.required_clearance_grid, pos, 3)
        normalized_clearance = clearance / self.drone_max_altitude_by_env.unsqueeze(-1).clamp_min(1e-6)
        patch_cells = self.local_map_patch_size * self.local_map_patch_size
        if agent.is_drone:
            normalized_costs = torch.zeros(pos.shape[0], patch_cells, device=pos.device, dtype=pos.dtype)
            blocked = self._local_outside_map_patch(pos, self.local_map_patch_size)
        else:
            costs = self._local_grid_patch(self.mobility_cost_grid, pos, self.local_map_patch_size)
            blocked = (~self._local_grid_patch(self.traversable_grid, pos, self.local_map_patch_size)).float()
            cost_max = self.mobility_cost_grid.amax(dim=(1, 2), keepdim=False).unsqueeze(-1).clamp_min(1e-12)
            normalized_costs = (costs / cost_max).clamp(0.0, 1.0)
            normalized_clearance = torch.zeros_like(normalized_clearance)
        return torch.cat([normalized_costs, blocked, normalized_clearance], dim=-1)

    def _ugv_planner_hint_observations(self, agent: Agent) -> Tensor:
        """Optional local A* waypoint hint for ground robots.

        Feature order is [unit_dx, unit_dy, distance_norm, valid, direct_blocked].
        The planner is constrained to ``ugv_planner_patch_size`` cells around the
        UGV and does not expose the full route.
        """
        if self.ugv_planner_hint == "none":
            return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)
        if self.ugv_planner_hint != "local_astar":
            raise RuntimeError(f"unsupported ugv_planner_hint: {self.ugv_planner_hint!r}")
        if agent.is_drone or self.n_survivors == 0:
            return torch.zeros(self.world.batch_dim, UGV_PLANNER_HINT_DIM, device=agent.state.pos.device)

        agent_idx = self.world.agents.index(agent)
        local_known = self.known_survivors_by_agent[:, agent_idx]
        local_confirmed = self.confirmed_survivors_by_agent[:, agent_idx]
        targetable = local_known & ~local_confirmed
        survivor_pos = torch.stack([s.state.pos for s in self._survivors], dim=1)
        distances = torch.linalg.norm(survivor_pos - agent.state.pos.unsqueeze(1), dim=-1)
        masked_distances = torch.where(
            targetable,
            distances,
            torch.full_like(distances, float("inf")),
        )
        target_idx = masked_distances.argmin(dim=-1)
        has_target = targetable.any(dim=-1)
        target_pos = survivor_pos.gather(1, target_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

        out = torch.zeros(self.world.batch_dim, UGV_PLANNER_HINT_DIM, device=agent.state.pos.device)
        for env_index in range(self.world.batch_dim):
            if not bool(has_target[env_index].item()):
                continue
            out[env_index] = self._local_astar_hint_for_env(
                env_index,
                agent.state.pos[env_index],
                target_pos[env_index],
            )
        return out

    def _local_astar_hint_for_env(self, env_index: int, pos: Tensor, target_pos: Tensor) -> Tensor:
        device = pos.device
        hint = torch.zeros(UGV_PLANNER_HINT_DIM, device=device)
        route = self._local_astar_route_for_env(env_index, pos, target_pos)
        if route is None:
            return hint
        waypoint, direct_blocked, _detour_needed = route
        waypoint_pos = self._grid_cell_center_to_world(waypoint, device=device, dtype=pos.dtype)
        delta = waypoint_pos - pos
        dist = torch.linalg.norm(delta)
        if float(dist.item()) <= 1e-9:
            return hint

        patch_size = self.ugv_planner_patch_size
        radius = patch_size // 2
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        cell_w_sim = 2.0 * float(self.x_semidim) / float(self.fire_grid_size)
        cell_h_sim = 2.0 * float(self.y_semidim) / float(self.fire_grid_size)
        planner_range_m = max(radius * max(cell_w_sim, cell_h_sim) / max(scale, 1e-9), 1e-6)
        dist_m = float(dist.detach().cpu().item()) / max(scale, 1e-9)
        unit = delta / dist.clamp_min(1e-9)
        hint[0] = unit[X]
        hint[1] = unit[Y]
        hint[2] = min(max(dist_m / planner_range_m, 0.0), 1.0)
        hint[3] = 1.0
        hint[4] = 1.0 if direct_blocked else 0.0
        return hint

    def _local_astar_route_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
    ) -> tuple[tuple[int, int], bool, bool] | None:
        patch_size = self.ugv_planner_patch_size
        radius = patch_size // 2
        pos_cell = self._single_position_to_grid_cell(pos)
        target_cell = self._single_position_to_grid_cell(target_pos)
        sx, sy = pos_cell
        x0 = max(0, sx - radius)
        x1 = min(self.fire_grid_size - 1, sx + radius)
        y0 = max(0, sy - radius)
        y1 = min(self.fire_grid_size - 1, sy + radius)
        bounds = (x0, x1, y0, y1)

        start = self._nearest_traversable_cell_in_bounds(env_index, sx, sy, bounds)
        if start is None:
            return None

        goal_candidates = self._local_planner_goal_candidates(
            env_index,
            start,
            target_cell,
            bounds,
        )
        if not goal_candidates:
            return None

        path: list[tuple[int, int]] = []
        goal = goal_candidates[0]
        for candidate in goal_candidates:
            path = self._local_astar_grid_path(env_index, start, candidate, bounds)
            if len(path) >= 2:
                goal = candidate
                break
        if len(path) < 2:
            return None

        traversable = self.traversable_grid[env_index]
        direct_blocked = not self._grid_segment_is_traversable(traversable, start, goal)
        waypoint = self._route_lookahead_cell(traversable, path, 0)
        detour_needed = self._local_astar_detour_needed(env_index, start, goal, waypoint, path, direct_blocked)
        return waypoint, direct_blocked, detour_needed

    def _ugv_planner_progress_rewards(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        target_pos: Tensor,
        gate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        reward = torch.zeros_like(gate, dtype=start_pos.dtype)
        progress_m = torch.zeros_like(reward)
        progress_scaled = torch.zeros_like(reward)
        active = torch.zeros_like(gate, dtype=torch.bool)
        direct_blocked_out = torch.zeros_like(gate, dtype=torch.bool)
        detour_needed_out = torch.zeros_like(gate, dtype=torch.bool)
        if (
            self.r_ugv_planner_progress <= 0.0
            or self.ugv_planner_hint != "local_astar"
            or start_pos.shape[1] == 0
        ):
            return reward, progress_m, progress_scaled, active, direct_blocked_out, detour_needed_out

        sim_units_per_meter = self.terrain_sim_units_per_meter.to(start_pos.device).clamp_min(1e-9)
        batch_dim, n_ground, _ = start_pos.shape
        for env_index in range(batch_dim):
            scale = sim_units_per_meter[env_index]
            for ground_index in range(n_ground):
                if not bool(gate[env_index, ground_index].item()):
                    continue
                route = self._local_astar_route_for_env(
                    env_index,
                    start_pos[env_index, ground_index],
                    target_pos[env_index, ground_index],
                )
                if route is None:
                    continue
                waypoint, direct_blocked, detour_needed = route
                direct_blocked_out[env_index, ground_index] = direct_blocked
                detour_needed_out[env_index, ground_index] = detour_needed
                if not detour_needed:
                    continue
                waypoint_pos = self._grid_cell_center_to_world(
                    waypoint,
                    device=start_pos.device,
                    dtype=start_pos.dtype,
                )
                before_m = torch.linalg.norm(waypoint_pos - start_pos[env_index, ground_index]) / scale
                after_m = torch.linalg.norm(waypoint_pos - end_pos[env_index, ground_index]) / scale
                step_progress_m = before_m - after_m
                step_progress_scaled = (step_progress_m / self.ugv_planner_progress_scale_m).clamp(-1.0, 1.0)
                progress_m[env_index, ground_index] = step_progress_m
                progress_scaled[env_index, ground_index] = step_progress_scaled
                reward[env_index, ground_index] = step_progress_scaled * self.r_ugv_planner_progress
                active[env_index, ground_index] = True
        return reward, progress_m, progress_scaled, active, direct_blocked_out, detour_needed_out

    def _local_astar_detour_needed(
        self,
        env_index: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        waypoint: tuple[int, int],
        path: list[tuple[int, int]],
        direct_blocked: bool,
    ) -> bool:
        if direct_blocked:
            return True

        sx, sy = start
        gx, gy = goal
        wx, wy = waypoint
        direct_vec = (gx - sx, gy - sy)
        waypoint_vec = (wx - sx, wy - sy)
        direct_norm = math.hypot(*direct_vec)
        waypoint_norm = math.hypot(*waypoint_vec)
        direction_detour = False
        if direct_norm > 1e-9 and waypoint_norm > 1e-9:
            cos_to_goal = (
                direct_vec[0] * waypoint_vec[0] + direct_vec[1] * waypoint_vec[1]
            ) / (direct_norm * waypoint_norm)
            direction_detour = cos_to_goal < math.cos(math.radians(30.0))

        direct_cells = self._grid_segment_cells(start, goal)
        direct_cost = self._grid_path_cost(env_index, direct_cells)
        astar_cost = self._grid_path_cost(env_index, path)
        cost_detour = (
            math.isfinite(direct_cost)
            and math.isfinite(astar_cost)
            and astar_cost > 1e-9
            and direct_cost > astar_cost * 1.25
        )
        return direction_detour or cost_detour

    def _single_position_to_grid_cell(self, pos: Tensor) -> tuple[int, int]:
        gx, gy = self._positions_to_grid(pos.view(1, 1, 2))
        return int(gx[0, 0].item()), int(gy[0, 0].item())

    def _nearest_traversable_cell_in_bounds(
        self,
        env_index: int,
        gx: int,
        gy: int,
        bounds: tuple[int, int, int, int],
    ) -> tuple[int, int] | None:
        x0, x1, y0, y1 = bounds
        gx = max(x0, min(x1, gx))
        gy = max(y0, min(y1, gy))
        traversable = self.traversable_grid[env_index]
        if bool(traversable[gy, gx].item()):
            return gx, gy
        max_radius = max(x1 - x0, y1 - y0)
        for radius in range(1, max_radius + 1):
            candidates = []
            cy0, cy1 = max(y0, gy - radius), min(y1, gy + radius)
            cx0, cx1 = max(x0, gx - radius), min(x1, gx + radius)
            for y in range(cy0, cy1 + 1):
                candidates.extend(((cx0, y), (cx1, y)))
            for x in range(cx0 + 1, cx1):
                candidates.extend(((x, cy0), (x, cy1)))
            valid = [(x, y) for x, y in candidates if bool(traversable[y, x].item())]
            if valid:
                return min(valid, key=lambda cell: (cell[0] - gx) ** 2 + (cell[1] - gy) ** 2)
        return None

    def _local_planner_goal_candidates(
        self,
        env_index: int,
        start: tuple[int, int],
        target: tuple[int, int],
        bounds: tuple[int, int, int, int],
    ) -> list[tuple[int, int]]:
        x0, x1, y0, y1 = bounds
        tx, ty = target
        if x0 <= tx <= x1 and y0 <= ty <= y1:
            nearest = self._nearest_traversable_cell_in_bounds(env_index, tx, ty, bounds)
            return [] if nearest is None else [nearest]

        sx, sy = start
        dir_x = float(tx - sx)
        dir_y = float(ty - sy)
        dir_norm = max(math.hypot(dir_x, dir_y), 1e-9)
        dir_x /= dir_norm
        dir_y /= dir_norm
        traversable = self.traversable_grid[env_index]
        boundary = []
        interior = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if not bool(traversable[y, x].item()) or (x, y) == start:
                    continue
                dx = x - sx
                dy = y - sy
                projection = dx * dir_x + dy * dir_y
                lateral = abs(dx * dir_y - dy * dir_x)
                score = projection - 0.05 * lateral
                item = (score, (x, y))
                if x in (x0, x1) or y in (y0, y1):
                    boundary.append(item)
                else:
                    interior.append(item)
        ordered = sorted(boundary or interior, key=lambda item: item[0], reverse=True)
        return [cell for _, cell in ordered]

    def _local_astar_grid_path(
        self,
        env_index: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        bounds: tuple[int, int, int, int],
    ) -> list[tuple[int, int]]:
        if start == goal:
            return [start]
        x0, x1, y0, y1 = bounds
        traversable = self.traversable_grid[env_index]
        movement_cost = self.mobility_cost_grid[env_index] + self.fire_grid[env_index].float() * 25.0

        def open_cell(cell: tuple[int, int]) -> bool:
            x, y = cell
            if x < x0 or x > x1 or y < y0 or y > y1:
                return False
            return bool(traversable[y, x].item()) and math.isfinite(float(movement_cost[y, x].item()))

        if not open_cell(start) or not open_cell(goal):
            return []

        local_traversable = traversable[y0 : y1 + 1, x0 : x1 + 1]
        local_cost = movement_cost[y0 : y1 + 1, x0 : x1 + 1]
        finite_open_cost = local_cost[local_traversable & torch.isfinite(local_cost)]
        if finite_open_cost.numel() == 0:
            return []
        min_cost = float(finite_open_cost.min().item())
        min_cost = max(min_cost, 1e-6)

        def heuristic(cell: tuple[int, int]) -> float:
            return math.hypot(goal[0] - cell[0], goal[1] - cell[1]) * min_cost

        open_heap = [(heuristic(start), 0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best_cost = {start: 0.0}
        neighbor_offsets = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        while open_heap:
            _, cost_so_far, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_grid_path(came_from, current)
            if cost_so_far > best_cost.get(current, float("inf")) + 1e-9:
                continue
            cx, cy = current
            for ox, oy, step_len in neighbor_offsets:
                nxt = (cx + ox, cy + oy)
                if not open_cell(nxt):
                    continue
                if ox != 0 and oy != 0 and (not open_cell((cx + ox, cy)) or not open_cell((cx, cy + oy))):
                    continue
                nx, ny = nxt
                edge_cost = step_len * (
                    float(movement_cost[cy, cx].item()) + float(movement_cost[ny, nx].item())
                ) * 0.5
                new_cost = cost_so_far + edge_cost
                if new_cost < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (new_cost + heuristic(nxt), new_cost, nxt))
        return []

    def _reconstruct_grid_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _route_lookahead_cell(
        self,
        traversable: Tensor,
        path: list[tuple[int, int]],
        nearest_idx: int,
    ) -> tuple[int, int]:
        start = path[nearest_idx]
        best = path[min(nearest_idx + 1, len(path) - 1)]
        stop = min(nearest_idx + self.ugv_planner_lookahead_cells, len(path) - 1)
        for idx in range(nearest_idx + 2, stop + 1):
            candidate = path[idx]
            if not self._grid_segment_is_traversable(traversable, start, candidate):
                break
            best = candidate
        return best

    def _grid_segment_cells(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return [start]
        cells = []
        previous: tuple[int, int] | None = None
        for i in range(steps + 1):
            x = int(round(x0 + (x1 - x0) * i / steps))
            y = int(round(y0 + (y1 - y0) * i / steps))
            cell = (x, y)
            if cell != previous:
                cells.append(cell)
                previous = cell
        return cells

    def _grid_path_cost(self, env_index: int, path: list[tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        traversable = self.traversable_grid[env_index]
        movement_cost = self.mobility_cost_grid[env_index] + self.fire_grid[env_index].float() * 25.0
        total = 0.0
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            if (
                not bool(traversable[y0, x0].item())
                or not bool(traversable[y1, x1].item())
            ):
                return float("inf")
            if x0 != x1 and y0 != y1:
                if (
                    not bool(traversable[y0, x1].item())
                    or not bool(traversable[y1, x0].item())
                ):
                    return float("inf")
                step_len = math.sqrt(2.0)
            else:
                step_len = 1.0
            c0 = float(movement_cost[y0, x0].item())
            c1 = float(movement_cost[y1, x1].item())
            if not math.isfinite(c0) or not math.isfinite(c1):
                return float("inf")
            total += step_len * (c0 + c1) * 0.5
        return total

    def _grid_segment_is_traversable(
        self,
        traversable: Tensor,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> bool:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return bool(traversable[y0, x0].item())
        previous = start
        for i in range(steps + 1):
            x = int(round(x0 + (x1 - x0) * i / steps))
            y = int(round(y0 + (y1 - y0) * i / steps))
            if not bool(traversable[y, x].item()):
                return False
            px, py = previous
            if x != px and y != py:
                if not bool(traversable[py, x].item()) or not bool(traversable[y, px].item()):
                    return False
            previous = (x, y)
        return True

    def _grid_cell_center_to_world(
        self,
        cell: tuple[int, int] | int,
        gy: int | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if gy is None:
            gx, gy = cell
        else:
            gx = int(cell)
        cell_w = 2.0 * self.x_semidim / self.fire_grid_size
        cell_h = 2.0 * self.y_semidim / self.fire_grid_size
        return torch.tensor(
            [
                -self.x_semidim + (gx + 0.5) * cell_w,
                -self.y_semidim + (gy + 0.5) * cell_h,
            ],
            dtype=dtype,
            device=device,
        )

    def _local_grid_patch(
        self,
        grid: Tensor,
        pos: Tensor,
        patch_size: int,
    ) -> Tensor:
        """Return a flattened square patch around each position, clamped at edges."""
        radius = patch_size // 2
        gx, gy = self._positions_to_grid(pos)
        b_idx = torch.arange(self.world.batch_dim, device=pos.device)
        values = []
        for dy in range(-radius, radius + 1):
            py = (gy + dy).clamp(0, self.fire_grid_size - 1)
            for dx in range(-radius, radius + 1):
                px = (gx + dx).clamp(0, self.fire_grid_size - 1)
                values.append(grid[b_idx, py, px])
        return torch.stack(values, dim=-1)

    def _local_outside_map_patch(self, pos: Tensor, patch_size: int) -> Tensor:
        """Flattened local mask where 1 marks samples outside the search grid."""
        radius = patch_size // 2
        gx, gy = self._positions_to_grid(pos)
        values = []
        for dy in range(-radius, radius + 1):
            py = gy + dy
            for dx in range(-radius, radius + 1):
                px = gx + dx
                values.append(
                    ((px < 0) | (px >= self.fire_grid_size) | (py < 0) | (py >= self.fire_grid_size)).float()
                )
        return torch.stack(values, dim=-1).to(device=pos.device, dtype=pos.dtype)

    def _flight_state(self, agent: Agent) -> Tensor:
        state = torch.zeros(self.world.batch_dim, 2, device=self.fire_grid.device)
        if agent.is_drone:
            drone_idx = self.world.agents.index(agent)
            state[:, 0] = self.drone_altitude[:, drone_idx] / self.drone_max_altitude_by_env.clamp_min(1e-6)
            state[:, 1] = self.drone_altitude_quality[:, drone_idx]
        return state

    def _neighbor_observations(self, agent: Agent, comms_keep: Tensor) -> Tensor:
        deltas = []
        for other in self.world.agents:
            if other is agent:
                continue
            deltas.append(other.state.pos - agent.state.pos)
        if not deltas:
            return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)
        rel = torch.cat(deltas, dim=-1)
        return rel * comms_keep.float()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    def done(self) -> Tensor:
        all_found = self.found_survivors.all(dim=1)
        timed_out = self.step_count >= self.max_steps
        return all_found | timed_out

    # ------------------------------------------------------------------
    # Info (for evaluation / debugging)
    # ------------------------------------------------------------------
    def info(self, agent: Agent) -> Dict[str, Tensor]:
        mean_drone_altitude = (
            self.drone_altitude.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
        )
        mean_drone_altitude_msl = (
            self.drone_altitude_msl.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
        )
        return {
            "n_found":   self.found_survivors.sum(dim=1).float(),
            "n_scouted": self.scouted_survivors.sum(dim=1).float(),
            "mission/new_scouts": self.metric_new_scouts,
            "mission/new_confirmations": self.metric_new_confirmations,
            "mission/n_scouted": self.scouted_survivors.sum(dim=1).float(),
            "mission/n_confirmed": self.found_survivors.sum(dim=1).float(),
            "mission/full_success": self.metric_full_success,
            "reward/team": self.metric_reward_team,
            "reward/drone_scout": self.metric_reward_drone_scout,
            "reward/drone_progress": self.metric_reward_drone_progress,
            "reward/uav_move_coverage": self.metric_reward_uav_move_coverage,
            "reward/uav_overlap": self.metric_reward_uav_overlap,
            "reward/uav_outside_footprint": self.metric_reward_uav_outside_footprint,
            "reward/ugv_progress": self.metric_reward_ugv_progress,
            "reward/ugv_approach": self.metric_reward_ugv_approach,
            "reward/ugv_movement_alignment": self.metric_reward_ugv_movement_alignment,
            "reward/ugv_planner_progress": self.metric_reward_ugv_planner_progress,
            "reward/ugv_stall_penalty": self.metric_reward_ugv_stall_penalty,
            "reward/ground_confirm": self.metric_reward_ground_confirm,
            "reward/coverage": self.metric_reward_coverage,
            "cost/ugv_fire_exposure": self.metric_cost_ugv_fire_exposure,
            "cost/ugv_travel": self.metric_cost_ugv_travel,
            "cost/drone_energy": self.metric_cost_drone_energy,
            "cost/drone_climb": self.metric_cost_drone_climb,
            "diagnostic/ugv_proposed_path_blocked": self.step_ugv_proposed_path_blocked.float().sum(dim=1),
            "diagnostic/uav_overlap_fraction": self.metric_uav_overlap_fraction,
            "diagnostic/uav_outside_footprint_fraction": self.metric_uav_outside_footprint_fraction,
            "diagnostic/ugv_speed_limited": self.step_ugv_speed_limited.float().sum(dim=1),
            "diagnostic/ugv_path_speed": (
                self.step_ugv_path_speed.mean(dim=1)
                if self.n_ground > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/ugv_speed_limit_scale": (
                self.step_ugv_speed_limit_scale.mean(dim=1)
                if self.n_ground > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/ugv_proposed_displacement_m": self.step_ugv_proposed_displacement_m.sum(dim=1),
            "diagnostic/ugv_corrected_displacement_m": self.step_ugv_corrected_displacement_m.sum(dim=1),
            "diagnostic/ugv_actual_displacement_m": self.step_ugv_actual_displacement_m.sum(dim=1),
            "diagnostic/ugv_motion_correction_m": self.step_ugv_motion_correction_m.sum(dim=1),
            "diagnostic/ugv_final_known_target_distance_m": self.metric_ugv_known_target_distance_m,
            "diagnostic/ugv_min_known_target_distance_m": self.metric_ugv_known_target_distance_m,
            "diagnostic/ugv_confirm_range_m": self.metric_ugv_confirm_range_m,
            "diagnostic/ugv_steps_within_confirm_range": self.metric_ugv_within_confirm_range,
            "diagnostic/ugv_steps_within_12m": self.metric_ugv_within_12m,
            "diagnostic/ugv_steps_within_15m": self.metric_ugv_within_15m,
            "diagnostic/ugv_known_target_valid": self.metric_ugv_known_target_valid,
            "diagnostic/ugv_same_target": self.metric_ugv_same_target,
            "diagnostic/ugv_prev_distance_valid": self.metric_ugv_prev_distance_valid,
            "diagnostic/ugv_progress_gate_active": self.metric_ugv_progress_gate_active,
            "diagnostic/ugv_target_index": self.metric_ugv_target_index,
            "diagnostic/ugv_ground_progress_m": self.metric_ugv_ground_progress_m,
            "diagnostic/ugv_ground_progress_scaled": self.metric_ugv_ground_progress_scaled,
            "diagnostic/ugv_planner_progress_m": self.metric_ugv_planner_progress_m,
            "diagnostic/ugv_planner_progress_scaled": self.metric_ugv_planner_progress_scaled,
            "diagnostic/ugv_planner_active": self.metric_ugv_planner_active,
            "diagnostic/ugv_planner_direct_blocked": self.metric_ugv_planner_direct_blocked,
            "diagnostic/ugv_planner_detour_needed": self.metric_ugv_planner_detour_needed,
            "diagnostic/ugv_action_alignment": self.metric_ugv_action_alignment,
            "diagnostic/ugv_movement_alignment": self.metric_ugv_movement_alignment,
            "diagnostic/uav_final_target_distance_m": self.metric_uav_target_distance_m,
            "diagnostic/uav_min_target_distance_m": self.metric_uav_target_distance_m,
            "diagnostic/uav_footprint_radius_m": self.metric_uav_footprint_radius_m,
            "diagnostic/uav_steps_with_target_in_footprint": self.metric_uav_target_within_footprint,
            "diagnostic/uav_displacement_m": self.metric_uav_displacement_m,
            "diagnostic/uav_new_coverage_cells": self.metric_uav_new_coverage_cells,
            "diagnostic/uav_boundary_projection_count": self.step_uav_boundary_projection_count.sum(dim=1),
            "diagnostic/uav_boundary_projection_norm": self.step_uav_boundary_projection_norm.sum(dim=1),
            "diagnostic/uav_boundary_hit_count": self.step_uav_boundary_hit.sum(dim=1),
            "diagnostic/uav_boundary_soft_risk": self.metric_uav_boundary_soft_risk,
            "diagnostic/uav_boundary_distance_m": self.metric_uav_boundary_distance_m,
            "n_burning": self.fire_grid.flatten(1).sum(dim=1).float(),
            "n_burned":  self.burned_grid.flatten(1).sum(dim=1).float(),
            "affected_fraction": self.burned_grid.float().flatten(1).mean(dim=1),
            "mean_fire_intensity": self.fire_intensity_grid.flatten(1).mean(dim=1),
            "ugv_step_travel_cost": self.step_ugv_travel_cost.sum(dim=1),
            "mean_drone_altitude": mean_drone_altitude,
            "mean_drone_altitude_msl": mean_drone_altitude_msl,
        }


if __name__ == "__main__":
    import argparse
    from vmas import render_interactively
    p = argparse.ArgumentParser()
    p.add_argument("--terrain-cache-path", default="data/terrain_cache/malibu_128.npz")
    p.add_argument("--terrain-place", default="Malibu Creek State Park, California")
    p.add_argument("--n-drones",    type=int, default=3)
    p.add_argument("--n-ground",    type=int, default=2)
    p.add_argument("--n-survivors", type=int, default=5)
    p.add_argument("--grid-size",   type=int, default=128)
    p.add_argument("--comms-dropout", type=float, default=0.0)
    args = p.parse_args()
    render_interactively(
        WildfireSearchScenario(),
        control_two_agents=True,
        n_drones=args.n_drones,
        n_ground=args.n_ground,
        n_survivors=args.n_survivors,
        fire_grid_size=args.grid_size,
        comms_dropout=args.comms_dropout,
        terrain_cache_path=args.terrain_cache_path,
        terrain_place=args.terrain_place,
    )
