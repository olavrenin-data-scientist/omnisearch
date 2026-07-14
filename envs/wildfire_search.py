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

import numpy as np
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
UGV_LOCAL_PLANNER_HINT_MODES = {"local_astar", "local_escape_astar"}
UGV_PLANNER_HINT_MODES = UGV_LOCAL_PLANNER_HINT_MODES | {"global_astar"}
UGV_GLOBAL_PLANNER_HEURISTICS = {"euclidean", "terrain"}
_SCIPY_SPARSE_TOOLS = None
_SCIPY_SPARSE_UNAVAILABLE = False


def _scipy_sparse_tools():
    global _SCIPY_SPARSE_TOOLS, _SCIPY_SPARSE_UNAVAILABLE
    if _SCIPY_SPARSE_TOOLS is not None:
        return _SCIPY_SPARSE_TOOLS
    if _SCIPY_SPARSE_UNAVAILABLE:
        return None
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
    except Exception:
        _SCIPY_SPARSE_UNAVAILABLE = True
        return None
    _SCIPY_SPARSE_TOOLS = (csr_matrix, dijkstra)
    return _SCIPY_SPARSE_TOOLS


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
        self.active_survivors_min = int(kwargs.pop("active_survivors_min", self.n_survivors))
        self.active_survivors_max = int(kwargs.pop("active_survivors_max", self.n_survivors))
        if self.active_survivors_min < 0:
            raise ValueError("active_survivors_min must be nonnegative")
        if self.active_survivors_max < self.active_survivors_min:
            raise ValueError("active_survivors_max must be >= active_survivors_min")
        if self.active_survivors_max > self.n_survivors:
            raise ValueError("active_survivors_max must be <= n_survivors")
        self.n_agents    = self.n_drones + self.n_ground
        self.obs_schema_n_drones = int(kwargs.pop("obs_schema_n_drones", self.n_drones))
        self.obs_schema_n_ground = int(kwargs.pop("obs_schema_n_ground", self.n_ground))
        self.obs_schema_n_survivors = int(kwargs.pop("obs_schema_n_survivors", self.n_survivors))
        if self.obs_schema_n_drones < self.n_drones:
            raise ValueError("obs_schema_n_drones must be >= n_drones")
        if self.obs_schema_n_ground < self.n_ground:
            raise ValueError("obs_schema_n_ground must be >= n_ground")
        if self.obs_schema_n_survivors < self.n_survivors:
            raise ValueError("obs_schema_n_survivors must be >= n_survivors")
        self.obs_schema_n_agents = self.obs_schema_n_drones + self.obs_schema_n_ground

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
        # detection_backend selects how drone scouting is computed:
        #   "abstract" (default) uses the existing probabilistic perception model.
        #   "cv" runs YOLOv8 on rendered NAIP+survivor imagery.
        self.detection_backend = str(kwargs.pop("detection_backend", "abstract")).lower()
        if self.detection_backend not in ("abstract", "cv"):
            raise ValueError(f"detection_backend must be 'abstract' or 'cv', got {self.detection_backend!r}")
        self.cv_image_size = int(kwargs.pop("cv_image_size", 512))
        self.cv_person_model = kwargs.pop("cv_person_model", None)
        self.cv_conf_threshold = float(kwargs.pop("cv_conf_threshold", 0.35))
        self._cv_adapter = None  # lazily initialized on first CV detection step.

        # False-positive perception model (decoy landmarks).
        # Disabled when no decoys are configured because there are no decoy
        # landmarks to detect. When decoys are enabled, drones can falsely scout
        # them and UGVs can waste trips dismissing them.
        self.n_decoys = max(int(kwargs.pop("n_decoys", 0)), 0)
        self.active_decoys_min = int(kwargs.pop("active_decoys_min", self.n_decoys))
        self.active_decoys_max = int(kwargs.pop("active_decoys_max", self.n_decoys))
        if self.active_decoys_min < 0:
            raise ValueError("active_decoys_min must be nonnegative")
        if self.active_decoys_max < self.active_decoys_min:
            raise ValueError("active_decoys_max must be >= active_decoys_min")
        if self.active_decoys_max > self.n_decoys:
            raise ValueError("active_decoys_max must be <= n_decoys")
        self.drone_false_positive_rate = min(
            max(float(kwargs.pop("drone_false_positive_rate", 0.05)), 0.0),
            1.0,
        )
        self.r_decoy_pursuit_penalty = float(kwargs.pop("r_decoy_pursuit_penalty", 0.0))

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
        self.uav_start_min_separation_m = max(
            float(kwargs.pop("uav_start_min_separation_m", 0.0)), 0.0,
        )
        self.uav_start_edge_margin_m = max(
            float(kwargs.pop("uav_start_edge_margin_m", 0.0)), 0.0,
        )
        self.uav_start_max_attempts = max(int(kwargs.pop("uav_start_max_attempts", 512)), 1)
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
        if self.ugv_planner_hint not in {"none"} | UGV_PLANNER_HINT_MODES:
            raise ValueError(
                "ugv_planner_hint must be one of: none, local_astar, local_escape_astar, global_astar"
            )
        self.ugv_planner_detour_obs = bool(kwargs.pop("ugv_planner_detour_obs", False))
        self.ugv_route_aware_reward = bool(kwargs.pop("ugv_route_aware_reward", False))
        if self.ugv_route_aware_reward and self.ugv_planner_hint not in UGV_LOCAL_PLANNER_HINT_MODES:
            raise ValueError("ugv_route_aware_reward requires a local UGV planner hint")
        self.ugv_planner_patch_size = int(kwargs.pop("ugv_planner_patch_size", 11))
        if self.ugv_planner_patch_size < 1 or self.ugv_planner_patch_size % 2 != 1:
            raise ValueError("ugv_planner_patch_size must be a positive odd integer")
        self.ugv_planner_lookahead_cells = min(
            max(int(kwargs.pop("ugv_planner_lookahead_cells", 10)), 1),
            max(self.ugv_planner_patch_size // 2, 1),
        )
        self.ugv_global_planner_lookahead_m = max(
            float(kwargs.pop("ugv_global_planner_lookahead_m", 20.0)),
            1e-6,
        )
        self.ugv_global_planner_heuristic = str(
            kwargs.pop("ugv_global_planner_heuristic", "euclidean")
        ).replace("-", "_")
        if self.ugv_global_planner_heuristic not in UGV_GLOBAL_PLANNER_HEURISTICS:
            raise ValueError("ugv_global_planner_heuristic must be one of: euclidean, terrain")
        self.ugv_planner_fire_mode = str(kwargs.pop("ugv_planner_fire_mode", "off")).replace("-", "_")
        if self.ugv_planner_fire_mode not in {"off", "cost", "block"}:
            raise ValueError("ugv_planner_fire_mode must be one of: off, cost, block")
        self.ugv_planner_fire_replan_policy = str(
            kwargs.pop("ugv_planner_fire_replan_policy", "always"),
        ).replace("-", "_")
        if self.ugv_planner_fire_replan_policy not in {"always", "affected", "lazy"}:
            raise ValueError("ugv_planner_fire_replan_policy must be one of: always, affected, lazy")
        self.ugv_planner_fire_replan_interval_steps = max(
            int(kwargs.pop("ugv_planner_fire_replan_interval_steps", 15)),
            1,
        )
        self.ugv_planner_fire_cost = max(float(kwargs.pop("ugv_planner_fire_cost", 25.0)), 0.0)
        self.ugv_planner_fire_block_threshold = float(
            kwargs.pop("ugv_planner_fire_block_threshold", 0.0)
        )
        if not 0.0 <= self.ugv_planner_fire_block_threshold <= 1.0:
            raise ValueError("ugv_planner_fire_block_threshold must be in [0, 1]")
        self.ugv_planner_smoke_cost = max(float(kwargs.pop("ugv_planner_smoke_cost", 5.0)), 0.0)
        self.ugv_planner_smolder_cost = max(float(kwargs.pop("ugv_planner_smolder_cost", 3.0)), 0.0)
        self.ugv_planner_fire_buffer_m = max(float(kwargs.pop("ugv_planner_fire_buffer_m", 10.0)), 0.0)
        self.ugv_planner_fire_buffer_cost = max(float(kwargs.pop("ugv_planner_fire_buffer_cost", 8.0)), 0.0)
        planner_land_cover_costs_arg = kwargs.pop("ugv_planner_land_cover_costs", None)
        planner_land_cover_costs = None
        if planner_land_cover_costs_arg is not None:
            planner_land_cover_costs = _land_cover_values(
                planner_land_cover_costs_arg,
                water_value=8.0,
                name="ugv_planner_land_cover_costs",
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
            "drone_cover_detection_factors", (1.0, 0.95, 0.75, 0.55, 0.45, 0.90),
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
        self.drone_edge_detection_floor = kwargs.pop("drone_edge_detection_floor", 0.40)
        self.uav_confidence_diagnostics = bool(kwargs.pop("uav_confidence_diagnostics", False))
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
        self.comms_dropout = min(max(float(kwargs.pop("comms_dropout", 0.0)), 0.0), 1.0)
        self.comms_dropout_mode = str(kwargs.pop("comms_dropout_mode", "iid")).replace("-", "_").lower()
        if self.comms_dropout_mode not in {"iid", "bursty"}:
            raise ValueError("comms_dropout_mode must be one of: iid, bursty")
        self.comms_map_mode = str(kwargs.pop("comms_map_mode", "global")).replace("-", "_").lower()
        if self.comms_map_mode not in {"global", "per_agent"}:
            raise ValueError("comms_map_mode must be one of: global, per_agent")
        self.comms_dropout_min_steps = max(int(kwargs.pop("comms_dropout_min_steps", 5)), 1)
        self.comms_dropout_max_steps = max(
            int(kwargs.pop("comms_dropout_max_steps", 15)),
            self.comms_dropout_min_steps,
        )
        self.survivor_message_distance_scale_m = max(
            float(kwargs.pop("survivor_message_distance_scale_m", 100.0)),
            1e-6,
        )

        # Episode
        self.max_steps = kwargs.pop("max_steps", 500)
        self.known_survivors_at_reset = bool(kwargs.pop("known_survivors_at_reset", False))
        self.delayed_survivor_knowledge = bool(kwargs.pop("delayed_survivor_knowledge", False))
        self.survivor_reveal_schedule = str(
            kwargs.pop("survivor_reveal_schedule", "stratified_uniform")
        ).replace("-", "_").lower()
        if self.survivor_reveal_schedule not in {"stratified_uniform"}:
            raise ValueError("survivor_reveal_schedule must be stratified_uniform")
        self.survivor_reveal_initial_count = max(
            int(kwargs.pop("survivor_reveal_initial_count", 1)),
            0,
        )
        self.survivor_reveal_start_step = max(
            int(kwargs.pop("survivor_reveal_start_step", 10)),
            0,
        )
        self.survivor_reveal_end_step = max(
            int(kwargs.pop("survivor_reveal_end_step", 180)),
            self.survivor_reveal_start_step,
        )
        self.delayed_decoy_knowledge = bool(kwargs.pop("delayed_decoy_knowledge", False))
        self.decoy_reveal_schedule = str(
            kwargs.pop("decoy_reveal_schedule", self.survivor_reveal_schedule)
        ).replace("-", "_").lower()
        if self.decoy_reveal_schedule not in {"stratified_uniform"}:
            raise ValueError("decoy_reveal_schedule must be stratified_uniform")
        self.decoy_reveal_initial_count = max(
            int(kwargs.pop("decoy_reveal_initial_count", 0)),
            0,
        )
        self.decoy_reveal_start_step = max(
            int(kwargs.pop("decoy_reveal_start_step", self.survivor_reveal_start_step)),
            0,
        )
        self.decoy_reveal_end_step = max(
            int(kwargs.pop("decoy_reveal_end_step", self.survivor_reveal_end_step)),
            self.decoy_reveal_start_step,
        )
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
        self.r_all_survivors_found = kwargs.pop("r_all_survivors_found", 0.0)
        self.r_team_scout = kwargs.pop("r_team_scout", 0.0)
        self.r_drone_scout    = kwargs.pop("r_drone_scout", 2.0)
        self.r_ground_confirm = kwargs.pop("r_ground_confirm", 4.0)
        self.r_time_penalty   = kwargs.pop("r_time_penalty", -0.0005)
        self.r_fire_penalty   = kwargs.pop("r_fire_penalty", -0.20)
        self.r_ground_travel_cost = kwargs.pop("r_ground_travel_cost", -0.01)
        self.r_drone_shaping  = kwargs.pop("r_drone_shaping",  0.30)
        self.r_ground_shaping = kwargs.pop("r_ground_shaping", 0.50)
        self.r_ugv_movement_alignment = kwargs.pop("r_ugv_movement_alignment", 0.20)
        self.ugv_target_assignment_mode = str(
            kwargs.pop("ugv_target_assignment_mode", "nearest")
        ).replace("-", "_").lower()
        valid_assignment_modes = {
            "nearest",
            "greedy",
            "greedy_sticky",
            "route_cost_greedy",
            "route_cost_sticky",
            "route_cost_global",
        }
        if self.ugv_target_assignment_mode not in valid_assignment_modes:
            raise ValueError(
                "ugv_target_assignment_mode must be one of: nearest, greedy, "
                "greedy_sticky, route_cost_greedy, route_cost_sticky, route_cost_global"
            )
        self.ugv_sticky_switch_margin_m = max(
            float(kwargs.pop("ugv_sticky_switch_margin_m", 20.0)),
            0.0,
        )
        self.ugv_sticky_switch_ratio = max(
            float(kwargs.pop("ugv_sticky_switch_ratio", 0.80)),
            0.0,
        )
        self.ugv_sticky_min_age_steps = max(
            int(kwargs.pop("ugv_sticky_min_age_steps", 10)),
            0,
        )
        self.r_ugv_planner_progress = max(float(kwargs.pop("r_ugv_planner_progress", 0.0)), 0.0)
        self.ugv_dense_reward_mode = str(kwargs.pop("ugv_dense_reward_mode", "target")).replace("-", "_")
        if self.ugv_dense_reward_mode not in {
            "target",
            "positive_target",
            "planner_blend",
            "escape_blend",
            "escape_route_switch",
            "planner_follow",
        }:
            raise ValueError(
                "ugv_dense_reward_mode must be one of: target, positive_target, "
                "planner_blend, escape_blend, escape_route_switch, planner_follow"
            )
        if self.ugv_dense_reward_mode == "planner_blend" and self.ugv_planner_hint not in UGV_LOCAL_PLANNER_HINT_MODES:
            raise ValueError("ugv_dense_reward_mode='planner_blend' requires a local UGV planner hint")
        if self.ugv_dense_reward_mode == "escape_blend" and self.ugv_planner_hint != "local_escape_astar":
            raise ValueError("ugv_dense_reward_mode='escape_blend' requires ugv_planner_hint='local_escape_astar'")
        if self.ugv_dense_reward_mode == "escape_route_switch" and self.ugv_planner_hint != "local_astar":
            raise ValueError("ugv_dense_reward_mode='escape_route_switch' requires ugv_planner_hint='local_astar'")
        if self.ugv_dense_reward_mode == "planner_follow" and self.ugv_planner_hint != "global_astar":
            raise ValueError("ugv_dense_reward_mode='planner_follow' requires ugv_planner_hint='global_astar'")
        if self.ugv_route_aware_reward and self.ugv_dense_reward_mode != "target":
            raise ValueError("ugv_route_aware_reward can only be combined with ugv_dense_reward_mode='target'")
        self.ugv_planner_blend_weight = min(
            max(float(kwargs.pop("ugv_planner_blend_weight", 0.70)), 0.0),
            1.0,
        )
        self.ugv_escape_stall_steps = max(int(kwargs.pop("ugv_escape_stall_steps", 5)), 1)
        self.ugv_escape_progress_threshold_m = max(
            float(kwargs.pop("ugv_escape_progress_threshold_m", 0.10)),
            0.0,
        )
        self.ugv_escape_movement_threshold_m = max(
            float(kwargs.pop("ugv_escape_movement_threshold_m", 0.25)),
            0.0,
        )
        self.ugv_escape_waypoint_reached_m = max(
            float(kwargs.pop("ugv_escape_waypoint_reached_m", 4.0)),
            1e-6,
        )
        self.ugv_escape_max_steps = max(int(kwargs.pop("ugv_escape_max_steps", 15)), 1)
        self.r_ugv_stall_penalty = max(float(kwargs.pop("r_ugv_stall_penalty", 0.0)), 0.0)
        self.ugv_stall_displacement_threshold_m = max(
            float(kwargs.pop("ugv_stall_displacement_threshold_m", 0.05)),
            0.0,
        )
        self.r_ugv_route_progress_floor_penalty = max(
            float(kwargs.pop("r_ugv_route_progress_floor_penalty", 0.0)),
            0.0,
        )
        self.ugv_route_progress_floor_m = max(
            float(kwargs.pop("ugv_route_progress_floor_m", 0.0)),
            0.0,
        )
        self.r_ugv_route_progress_shortfall_penalty = max(
            float(kwargs.pop("r_ugv_route_progress_shortfall_penalty", 0.0)),
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
        self.r_coverage = float(kwargs.pop("r_coverage", 5.0))
        # UAV movement-coverage reward: actual drone displacement in meters
        # times newly covered grid cells, capped per drone per step. This
        # nudges sweeping motion without paying for movement over old coverage.
        self.r_uav_move_coverage = max(float(kwargs.pop("r_uav_move_coverage", 0.0)), 0.0)
        self.r_uav_move_coverage_cap = max(float(kwargs.pop("r_uav_move_coverage_cap", 0.1)), 0.0)
        self.uav_move_coverage_normalization = str(
            kwargs.pop("uav_move_coverage_normalization", "raw")
        ).replace("-", "_").lower()
        if self.uav_move_coverage_normalization not in {"raw", "opportunity"}:
            raise ValueError("uav_move_coverage_normalization must be one of: raw, opportunity")
        legacy_uav_coverage_opportunity = kwargs.pop("r_uav_coverage_opportunity", None)
        uav_coverage_normalization = kwargs.pop("uav_coverage_normalization", None)
        if uav_coverage_normalization is None:
            if legacy_uav_coverage_opportunity is not None and float(legacy_uav_coverage_opportunity) > 0.0:
                self.r_coverage = max(float(legacy_uav_coverage_opportunity), 0.0)
                uav_coverage_normalization = "opportunity"
            else:
                uav_coverage_normalization = "map"
        self.uav_coverage_normalization = str(uav_coverage_normalization).replace("-", "_").lower()
        if self.uav_coverage_normalization not in {"map", "opportunity"}:
            raise ValueError("uav_coverage_normalization must be one of: map, opportunity")
        self.uav_coverage_opportunity_cap = max(float(kwargs.pop("uav_coverage_opportunity_cap", 1.0)), 0.0)
        self.r_uav_coverage_threshold = max(float(kwargs.pop("r_uav_coverage_threshold", 0.0)), 0.0)
        self.uav_coverage_threshold_fraction = min(
            max(float(kwargs.pop("uav_coverage_threshold_fraction", 0.95)), 0.0),
            1.0,
        )
        self.r_uav_frontier_alignment = max(float(kwargs.pop("r_uav_frontier_alignment", 0.0)), 0.0)
        self.r_uav_overlap = max(float(kwargs.pop("r_uav_overlap", 0.0)), 0.0)
        self.uav_overlap_allowed = min(
            max(float(kwargs.pop("uav_overlap_allowed", 0.10)), 0.0),
            0.999,
        )
        self.uav_overlap_penalty_normalization = str(
            kwargs.pop("uav_overlap_penalty_normalization", "raw")
        ).replace("-", "_").lower()
        if self.uav_overlap_penalty_normalization not in {"raw", "opportunity"}:
            raise ValueError("uav_overlap_penalty_normalization must be one of: raw, opportunity")
        self.r_uav_confidence = max(float(kwargs.pop("r_uav_confidence", 0.0)), 0.0)
        self.r_uav_team_confidence = max(float(kwargs.pop("r_uav_team_confidence", 0.0)), 0.0)
        self.r_uav_team_confidence_overlap = max(
            float(kwargs.pop("r_uav_team_confidence_overlap", 0.0)),
            0.0,
        )
        self.r_uav_confidence_move = max(float(kwargs.pop("r_uav_confidence_move", 0.0)), 0.0)
        self.r_uav_inefficient_move = max(
            float(kwargs.pop("r_uav_inefficient_move", 0.0)),
            0.0,
        )
        self.uav_inefficient_move_source = str(
            kwargs.pop("uav_inefficient_move_source", "confidence")
        ).replace("-", "_").lower()
        if self.uav_inefficient_move_source not in {"coverage", "confidence"}:
            raise ValueError("uav_inefficient_move_source must be one of: coverage, confidence")
        self.r_uav_confidence_overlap = max(
            float(kwargs.pop("r_uav_confidence_overlap", 0.0)),
            0.0,
        )
        self.uav_confidence_overlap_mode = str(
            kwargs.pop("uav_confidence_overlap_mode", "raw")
        ).replace("-", "_").lower()
        if self.uav_confidence_overlap_mode not in {"raw", "opportunity_regret"}:
            raise ValueError(
                "uav_confidence_overlap_mode must be one of: raw, opportunity_regret"
            )
        self.uav_confidence_overlap_allowed_regret = min(
            max(float(kwargs.pop("uav_confidence_overlap_allowed_regret", 0.10)), 0.0),
            1.0,
        )
        self.r_uav_cleanup_target_progress = max(
            float(kwargs.pop("r_uav_cleanup_target_progress", 0.0)),
            0.0,
        )
        self.r_uav_astar_progress = max(
            float(kwargs.pop("r_uav_astar_progress", 0.0)),
            0.0,
        )
        self.uav_confidence_overlap_threshold = min(
            max(float(kwargs.pop("uav_confidence_overlap_threshold", 0.65)), 0.0),
            0.999,
        )
        self.uav_confidence_opportunity_eps = max(
            float(kwargs.pop("uav_confidence_opportunity_eps", 1e-6)),
            0.0,
        )
        self.uav_confidence_gamma = max(float(kwargs.pop("uav_confidence_gamma", 2.0)), 0.0)
        self.uav_confidence_eps = max(float(kwargs.pop("uav_confidence_eps", 0.05)), 0.0)
        self.r_uav_inter_uav_overlap = max(float(kwargs.pop("r_uav_inter_uav_overlap", 0.0)), 0.0)
        self.uav_inter_uav_overlap_allowed = min(
            max(float(kwargs.pop("uav_inter_uav_overlap_allowed", 0.20)), 0.0),
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
        self.ugv_zero_uav_search_observations = bool(
            kwargs.pop("ugv_zero_uav_search_observations", False)
        )
        self.ugv_assigned_target_obs_only = bool(
            kwargs.pop("ugv_assigned_target_obs_only", False)
        )
        self.survivor_assignment_obs = bool(
            kwargs.pop("survivor_assignment_obs", False)
        )
        if self.local_coverage_obs_grid < 0:
            raise ValueError("local_coverage_obs_grid must be nonnegative")
        if self.local_coverage_obs_grid > 0 and self.local_coverage_obs_grid % 2 != 1:
            raise ValueError("local_coverage_obs_grid must be 0 or a positive odd integer")
        self.local_coverage_obs_radius_m = max(
            float(kwargs.pop("local_coverage_obs_radius_m", 150.0)),
            1e-6,
        )
        self.uav_confidence_obs_grid = int(kwargs.pop("uav_confidence_obs_grid", 0))
        if self.uav_confidence_obs_grid < 0:
            raise ValueError("uav_confidence_obs_grid must be nonnegative")
        self.local_confidence_obs_grid = int(kwargs.pop("local_confidence_obs_grid", 0))
        if self.local_confidence_obs_grid < 0:
            raise ValueError("local_confidence_obs_grid must be nonnegative")
        if self.local_confidence_obs_grid > 0 and self.local_confidence_obs_grid % 2 != 1:
            raise ValueError("local_confidence_obs_grid must be 0 or a positive odd integer")
        self.local_confidence_obs_radius_m = max(
            float(kwargs.pop("local_confidence_obs_radius_m", self.local_coverage_obs_radius_m)),
            1e-6,
        )
        self.uav_decision_grid = int(kwargs.pop("uav_decision_grid", 0))
        if self.uav_decision_grid < 0:
            raise ValueError("uav_decision_grid must be nonnegative")
        if self.uav_decision_grid == 1:
            raise ValueError("uav_decision_grid must be 0 or at least 2")
        self.uav_confidence_reward_grid = int(
            kwargs.pop("uav_confidence_reward_grid", self.uav_decision_grid)
        )
        if self.uav_confidence_reward_grid < 0:
            raise ValueError("uav_confidence_reward_grid must be nonnegative")
        if self.uav_confidence_reward_grid == 1:
            raise ValueError("uav_confidence_reward_grid must be 0 or at least 2")
        self.uav_coverage_reward_grid = int(
            kwargs.pop("uav_coverage_reward_grid", self.uav_decision_grid)
        )
        if self.uav_coverage_reward_grid < 0:
            raise ValueError("uav_coverage_reward_grid must be nonnegative")
        if self.uav_coverage_reward_grid == 1:
            raise ValueError("uav_coverage_reward_grid must be 0 or at least 2")
        self.uav_frontier_global_grid = int(
            kwargs.pop("uav_frontier_global_grid", self.uav_decision_grid)
        )
        if self.uav_frontier_global_grid < 0:
            raise ValueError("uav_frontier_global_grid must be nonnegative")
        if self.uav_frontier_global_grid == 1:
            raise ValueError("uav_frontier_global_grid must be 0 or at least 2")
        self.uav_confidence_map_grid_size = (
            self.fire_grid_size
            if self.uav_confidence_reward_grid <= 0
            else self.uav_confidence_reward_grid
        )
        self.uav_coverage_map_grid_size = (
            self.fire_grid_size
            if self.uav_coverage_reward_grid <= 0
            else self.uav_coverage_reward_grid
        )
        self.uav_frontier_obs = bool(kwargs.pop("uav_frontier_obs", False))
        self.uav_frontier_obs_radius_m = max(
            float(kwargs.pop("uav_frontier_obs_radius_m", self.local_coverage_obs_radius_m)),
            1e-6,
        )
        self.uav_frontier_mode = str(kwargs.pop("uav_frontier_mode", "centroid")).replace("-", "_")
        if self.uav_frontier_mode not in {"centroid", "sector_topk", "local_global"}:
            raise ValueError("uav_frontier_mode must be one of: centroid, sector_topk, local_global")
        self.uav_frontier_source = str(kwargs.pop("uav_frontier_source", "coverage")).replace("-", "_").lower()
        if self.uav_frontier_source not in {"coverage", "confidence"}:
            raise ValueError("uav_frontier_source must be one of: coverage, confidence")
        self.uav_frontier_sectors = int(kwargs.pop("uav_frontier_sectors", 8))
        if self.uav_frontier_sectors < 2:
            raise ValueError("uav_frontier_sectors must be at least 2")
        self.uav_frontier_top_k = int(kwargs.pop("uav_frontier_top_k", 2))
        if self.uav_frontier_top_k < 1:
            raise ValueError("uav_frontier_top_k must be positive")
        if self.uav_frontier_top_k > self.uav_frontier_sectors:
            raise ValueError("uav_frontier_top_k must be <= uav_frontier_sectors")
        self.uav_frontier_ownership = bool(kwargs.pop("uav_frontier_ownership", False))
        self.uav_cleanup_target_obs = bool(kwargs.pop("uav_cleanup_target_obs", False))
        self.uav_cleanup_target_diagnostics = bool(kwargs.pop("uav_cleanup_target_diagnostics", False))
        self.uav_cleanup_target_grid = int(kwargs.pop("uav_cleanup_target_grid", 16))
        if self.uav_cleanup_target_grid < 2:
            raise ValueError("uav_cleanup_target_grid must be at least 2")
        self.uav_cleanup_target_hold_steps = max(
            int(kwargs.pop("uav_cleanup_target_hold_steps", 15)),
            1,
        )
        self.uav_cleanup_target_confidence_threshold = min(
            max(float(kwargs.pop("uav_cleanup_target_confidence_threshold", 0.80)), 0.0),
            1.0,
        )
        self.uav_cleanup_target_min_value = max(
            float(kwargs.pop("uav_cleanup_target_min_value", 0.05)),
            0.0,
        )
        self.uav_cleanup_target_assignment_distance_scale_m = max(
            float(kwargs.pop("uav_cleanup_target_assignment_distance_scale_m", 250.0)),
            1e-6,
        )
        self.uav_cleanup_target_refresh_mode = str(
            kwargs.pop("uav_cleanup_target_refresh_mode", "exact")
        ).replace("-", "_").lower()
        if self.uav_cleanup_target_refresh_mode not in {"exact", "fixed_hold"}:
            raise ValueError("uav_cleanup_target_refresh_mode must be one of: exact, fixed_hold")
        self.uav_astar_route_obs = bool(kwargs.pop("uav_astar_route_obs", False))
        self.uav_astar_grid = int(kwargs.pop("uav_astar_grid", 32))
        if self.uav_astar_grid < 2:
            raise ValueError("uav_astar_grid must be at least 2")
        self.uav_astar_confidence_cost_alpha = max(
            float(kwargs.pop("uav_astar_confidence_cost_alpha", 3.0)),
            0.0,
        )
        self.uav_astar_confidence_cost_gamma = max(
            float(kwargs.pop("uav_astar_confidence_cost_gamma", 2.0)),
            0.0,
        )
        self.uav_astar_waypoint_lookahead_m = max(
            float(kwargs.pop("uav_astar_waypoint_lookahead_m", 50.0)),
            1e-6,
        )
        self.uav_astar_route_replan_steps = max(
            int(kwargs.pop("uav_astar_route_replan_steps", 5)),
            1,
        )
        self.uav_astar_waypoint_reached_m = max(
            float(kwargs.pop("uav_astar_waypoint_reached_m", 20.0)),
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

        # Decoy landmarks: non-survivor objects a drone may misclassify as a
        # survivor. They collide like survivors so UGVs can physically
        # investigate them, but they can never be confirmed.
        self._decoys: List[Landmark] = []
        for i in range(self.n_decoys):
            decoy = Landmark(
                name=f"decoy_{i}",
                collide=True,
                collision_filter=survivor_collision_filter,
                movable=False,
                shape=Sphere(radius=self.survivor_radius),
                color=Color.ORANGE,
            )
            world.add_landmark(decoy)
            self._decoys.append(decoy)

        # ---- Per-batch scenario state ----
        self.found_survivors = torch.zeros(
            batch_dim, self.n_survivors, dtype=torch.bool, device=device,
        )
        self.active_survivors = torch.ones_like(self.found_survivors)
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
        self.scouted_decoys = torch.zeros(
            batch_dim, self.n_decoys, dtype=torch.bool, device=device,
        )
        self.active_decoys = torch.ones_like(self.scouted_decoys)
        self.dismissed_decoys = torch.zeros_like(self.scouted_decoys)
        self.known_decoys_by_agent = torch.zeros(
            batch_dim, self.n_agents, self.n_decoys, dtype=torch.bool, device=device,
        )
        self.step_decoy_false_detections = torch.zeros(
            batch_dim, self.n_drones, self.n_decoys, dtype=torch.bool, device=device,
        )
        self.comms_dropout_remaining_steps = torch.zeros(
            batch_dim, self.n_agents, dtype=torch.long, device=device,
        )
        self.comms_dropout_last_update_step = torch.full(
            (batch_dim,), -1, dtype=torch.long, device=device,
        )
        self.survivor_reveal_steps = torch.full(
            (batch_dim, self.n_survivors), -1, dtype=torch.long, device=device,
        )
        self.survivor_oracle_revealed = torch.zeros_like(self.found_survivors)
        self.decoy_reveal_steps = torch.full(
            (batch_dim, self.n_decoys), -1, dtype=torch.long, device=device,
        )
        self.decoy_oracle_revealed = torch.zeros_like(self.scouted_decoys)
        self.coverage_grid = torch.zeros(
            batch_dim,
            self.uav_coverage_map_grid_size,
            self.uav_coverage_map_grid_size,
            dtype=torch.bool,
            device=device,
        )
        self.uav_confidence_grid = torch.zeros(
            batch_dim,
            self.uav_confidence_map_grid_size,
            self.uav_confidence_map_grid_size,
            dtype=torch.float,
            device=device,
        )
        if self._comms_maps_enabled():
            self.comm_agent_coverage_grid = torch.zeros(
                batch_dim,
                self.n_agents,
                self.uav_coverage_map_grid_size,
                self.uav_coverage_map_grid_size,
                dtype=torch.bool,
                device=device,
            )
            self.comm_agent_confidence_grid = torch.zeros(
                batch_dim,
                self.n_agents,
                self.uav_confidence_map_grid_size,
                self.uav_confidence_map_grid_size,
                dtype=torch.float,
                device=device,
            )
            self.comm_team_coverage_grid = torch.zeros_like(self.coverage_grid)
            self.comm_team_confidence_grid = torch.zeros_like(self.uav_confidence_grid)
            self.comm_map_last_sync_step = torch.full(
                (batch_dim,), -1, dtype=torch.long, device=device,
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
        self.ugv_planner_land_cover_cost_values = (
            torch.tensor(planner_land_cover_costs, dtype=torch.float, device=device)
            if planner_land_cover_costs is not None else None
        )
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
        self.uav_cleanup_target_valid = torch.zeros(
            batch_dim,
            self.n_drones,
            dtype=torch.bool,
            device=device,
        )
        self.uav_cleanup_target_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.uav_cleanup_target_value = torch.zeros(batch_dim, self.n_drones, device=device)
        self.uav_cleanup_target_initial_value = torch.zeros(batch_dim, self.n_drones, device=device)
        self.uav_cleanup_target_age = torch.zeros(
            batch_dim,
            self.n_drones,
            dtype=torch.long,
            device=device,
        )
        self.uav_cleanup_target_id = torch.full(
            (batch_dim, self.n_drones),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.uav_cleanup_target_prev_distance_m = torch.full(
            (batch_dim, self.n_drones),
            float("inf"),
            device=device,
        )
        self._uav_cleanup_target_last_assignment_step = torch.full(
            (batch_dim,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.uav_astar_waypoint_valid = torch.zeros(
            batch_dim,
            self.n_drones,
            dtype=torch.bool,
            device=device,
        )
        self.uav_astar_waypoint_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.uav_astar_waypoint_target_id = torch.full(
            (batch_dim, self.n_drones),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.uav_astar_waypoint_age = torch.zeros(
            batch_dim,
            self.n_drones,
            dtype=torch.long,
            device=device,
        )
        self.uav_astar_path_cost_norm = torch.zeros(batch_dim, self.n_drones, device=device)
        self._uav_astar_last_plan_step = torch.full(
            (batch_dim,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self._prev_ground_pos = torch.zeros(batch_dim, self.n_ground, 2, device=device)
        self._pre_step_ground_pos = torch.zeros_like(self._prev_ground_pos)
        self._pre_step_drone_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.ugv_sticky_target_idx = torch.full(
            (batch_dim, self.n_ground), -1, dtype=torch.long, device=device,
        )
        self.ugv_sticky_target_age = torch.zeros(
            batch_dim, self.n_ground, dtype=torch.long, device=device,
        )
        self.ugv_assignment_cache_step = torch.full(
            (batch_dim,), -1, dtype=torch.long, device=device,
        )
        self.ugv_assignment_cache_idx = torch.full(
            (batch_dim, self.n_ground), -1, dtype=torch.long, device=device,
        )
        self.ugv_assignment_cache_dist = torch.full(
            (batch_dim, self.n_ground), float("inf"), device=device,
        )
        self._ugv_assignment_result_cache = None
        self._uav_grid_geometry_cache = {}
        self._uav_sector_geometry_cache = {}
        self._uav_stencil_direction_cache = {}
        self._uav_land_cover_factor_cache = {}
        self._uav_frontier_feature_cache = {}
        self._uav_local_confidence_obs_cache = {}
        self._uav_cleanup_target_geometry_cache = {}
        self._uav_terrain_cache_version = 0
        self._ugv_planner_route_cache: dict[tuple, tuple[tuple[int, int], bool, bool] | None] = {}
        self._ugv_planner_terrain_cache_version = 0
        self._ugv_planner_layer_cache_version = 0
        self._ugv_planner_fire_mask_cache_version = 0
        self._ugv_static_planner_cache_version = 0
        self._ugv_planner_fire_buffer_mask_cache = {}
        self._ugv_planner_blocked_fire_mask_cache = {}
        self._ugv_planner_layer_tensor_cache = {}
        self._ugv_planner_layer_array_cache = {}
        self._ugv_static_planner_layer_array_cache = {}
        self._ugv_static_planner_graph_cache = {}
        self._ugv_global_heuristic_cache = {}
        self._ugv_global_raw_cost_to_go_cache = {}
        self._ugv_route_assignment_cost_grid_cache = {}
        self.ugv_escape_route_active = torch.zeros(
            batch_dim,
            self.n_ground,
            dtype=torch.bool,
            device=device,
        )
        self.ugv_escape_route_age = torch.zeros(
            batch_dim,
            self.n_ground,
            dtype=torch.long,
            device=device,
        )
        self.ugv_escape_route_stall_counter = torch.zeros(
            batch_dim,
            self.n_ground,
            dtype=torch.long,
            device=device,
        )
        self.ugv_escape_route_target_idx = torch.full(
            (batch_dim, self.n_ground),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_escape_route_path_index = torch.zeros(
            batch_dim,
            self.n_ground,
            dtype=torch.long,
            device=device,
        )
        self.ugv_escape_route_goal_cell = torch.full(
            (batch_dim, self.n_ground, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_escape_route_waypoint_cell = torch.full(
            (batch_dim, self.n_ground, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_escape_route_paths: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(self.n_ground)]
            for _ in range(batch_dim)
        ]
        self.ugv_global_route_target_idx = torch.full(
            (batch_dim, self.n_ground),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_global_route_path_index = torch.zeros(
            batch_dim,
            self.n_ground,
            dtype=torch.long,
            device=device,
        )
        self.ugv_global_route_goal_cell = torch.full(
            (batch_dim, self.n_ground, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_global_route_waypoint_cell = torch.full(
            (batch_dim, self.n_ground, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_global_route_last_replan_step = torch.full(
            (batch_dim, self.n_ground),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.ugv_global_route_paths: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(self.n_ground)]
            for _ in range(batch_dim)
        ]
        self.ugv_global_route_fire_replan_pending = torch.zeros(
            batch_dim,
            self.n_ground,
            dtype=torch.bool,
            device=device,
        )
        self.ugv_global_route_replanned_after_fire_flag = torch.zeros_like(
            self.ugv_global_route_fire_replan_pending
        )
        self.ugv_global_route_fire_blocked_no_path_flag = torch.zeros_like(
            self.ugv_global_route_fire_replan_pending
        )
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
        self.metric_false_positive_detections = torch.zeros(batch_dim, device=device)
        self.metric_false_positive_trips = torch.zeros(batch_dim, device=device)
        self.metric_reward_team = torch.zeros(batch_dim, device=device)
        self.metric_reward_all_survivors_found = torch.zeros(batch_dim, device=device)
        self.metric_reward_team_scout = torch.zeros(batch_dim, device=device)
        self.metric_reward_pending_penalty = torch.zeros(batch_dim, device=device)
        self.metric_survivor_oracle_reveals = torch.zeros(batch_dim, device=device)
        self.metric_decoy_oracle_reveals = torch.zeros(batch_dim, device=device)
        self.metric_ugv_assignment_switches = torch.zeros(batch_dim, device=device)
        self.metric_reward_drone_scout = torch.zeros(batch_dim, device=device)
        self.metric_reward_drone_progress = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_move_coverage = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_inefficient_move = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_inefficient_move_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_reward_uav_coverage_threshold = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_frontier_alignment = torch.zeros(batch_dim, device=device)
        self.metric_uav_frontier_alignment = torch.zeros(batch_dim, device=device)
        self.metric_uav_frontier_alignment_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_frontier_progress_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_frontier_progress_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_frontier_uncovered_ratio = torch.zeros(batch_dim, device=device)
        self.metric_uav_frontier_uncovered_ratio_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_overlap = torch.zeros(batch_dim, device=device)
        self.metric_uav_overlap_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_overlap_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_expected_overlap_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_excess_overlap_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_inter_uav_overlap = torch.zeros(batch_dim, device=device)
        self.metric_uav_inter_uav_overlap_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_inter_uav_overlap_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_outside_footprint = torch.zeros(batch_dim, device=device)
        self.metric_uav_outside_footprint_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_outside_footprint_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_boundary_soft_risk = torch.zeros(batch_dim, device=device)
        self.metric_uav_boundary_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_boundary_distance_m_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_displacement_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_new_coverage_cells = torch.zeros(batch_dim, device=device)
        self.metric_uav_displacement_m_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_new_coverage_cells_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_coverage_opportunity_cells = torch.zeros(batch_dim, device=device)
        self.metric_uav_coverage_opportunity_cells_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_coverage_opportunity_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_coverage_opportunity_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_coverage_opportunity_available_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_coverage_opportunity_available_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_confidence = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_confidence_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_team_confidence = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_team_confidence_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_team_confidence_overlap = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_team_confidence_overlap_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_reward_uav_confidence_move = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_confidence_move_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_reward_uav_confidence_overlap = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_confidence_overlap_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_reward_uav_cleanup_target_progress = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_cleanup_target_progress_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_reward_uav_astar_progress = torch.zeros(batch_dim, device=device)
        self.metric_reward_uav_astar_progress_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_astar_progress_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_astar_progress_fraction_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_astar_frontier_gate = torch.zeros(batch_dim, device=device)
        self.metric_uav_astar_frontier_gate_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_astar_path_cost_before_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_astar_path_cost_after_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_cleanup_target_frontier_gate = torch.zeros(batch_dim, device=device)
        self.metric_uav_cleanup_target_frontier_gate_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_confidence_overlap_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_overlap_fraction_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_confidence_overlap_regret = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_overlap_regret_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_confidence_mean = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_gain = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_gain_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_weighted_confidence_gain = torch.zeros(batch_dim, device=device)
        self.metric_uav_weighted_confidence_gain_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_confidence_opportunity_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_opportunity_fraction_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_confidence_opportunity_best_gain = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_opportunity_best_gain_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_confidence_low_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_confidence_high_fraction = torch.zeros(batch_dim, device=device)
        self.metric_uav_step_detection_probability = torch.zeros(batch_dim, device=device)
        self.metric_uav_step_detection_probability_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_cleanup_target_valid_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_cleanup_target_distance_m_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_cleanup_target_value_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_cleanup_target_progress_m_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_cleanup_target_progress_fraction_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_cleanup_target_switch_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_cleanup_target_reached_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_cleanup_target_value_decay_by_drone = torch.zeros(
            batch_dim,
            self.n_drones,
            device=device,
        )
        self.metric_uav_cleanup_target_age_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
        self.metric_uav_target_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_footprint_radius_m = torch.zeros(batch_dim, device=device)
        self.metric_uav_footprint_radius_m_by_drone = torch.zeros(batch_dim, self.n_drones, device=device)
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
        self.metric_ugv_duplicate_assignment_fraction = torch.zeros(batch_dim, device=device)
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
        self.metric_ugv_planner_escape_mode = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_active = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_enter = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_exit = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_stall_counter = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_age = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_waypoint_progress_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_waypoint_progress_scaled = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_waypoint_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_path_index = torch.zeros(batch_dim, device=device)
        self.metric_ugv_escape_route_path_length = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_valid = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_active = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_waypoint_distance_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_progress_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_progress_scaled = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_path_index = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_path_length = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_direct_blocked = torch.zeros(batch_dim, device=device)
        self.metric_ugv_global_route_detour_needed = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_fire_cells = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_smoke_mean = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_smolder_mean = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_fire_buffer_cells = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_replanned_after_fire = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_fire_blocked_no_path = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_aware_active = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_stall_penalty = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_route_progress_floor_penalty = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_progress_floor_shortfall_m = torch.zeros(batch_dim, device=device)
        self.metric_reward_ugv_route_progress_shortfall_penalty = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_progress_required_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_progress_shortfall_m = torch.zeros(batch_dim, device=device)
        self.metric_ugv_route_remaining_distance_m = torch.zeros(batch_dim, device=device)

    def _reset_step_metric_buffers(self, env_index: int | None = None) -> None:
        buffers = [
            self.metric_new_scouts,
            self.metric_new_confirmations,
            self.metric_full_success,
            self.metric_false_positive_detections,
            self.metric_false_positive_trips,
            self.metric_reward_team,
            self.metric_reward_all_survivors_found,
            self.metric_reward_team_scout,
            self.metric_reward_pending_penalty,
            self.metric_survivor_oracle_reveals,
            self.metric_decoy_oracle_reveals,
            self.metric_ugv_assignment_switches,
            self.metric_reward_drone_scout,
            self.metric_reward_drone_progress,
            self.metric_reward_uav_move_coverage,
            self.metric_reward_uav_inefficient_move,
            self.metric_reward_uav_inefficient_move_by_drone,
            self.metric_reward_uav_coverage_threshold,
            self.metric_reward_uav_frontier_alignment,
            self.metric_uav_frontier_alignment,
            self.metric_uav_frontier_alignment_by_drone,
            self.metric_uav_frontier_progress_fraction,
            self.metric_uav_frontier_progress_fraction_by_drone,
            self.metric_uav_frontier_uncovered_ratio,
            self.metric_uav_frontier_uncovered_ratio_by_drone,
            self.metric_reward_uav_overlap,
            self.metric_uav_overlap_fraction,
            self.metric_uav_overlap_fraction_by_drone,
            self.metric_uav_expected_overlap_fraction_by_drone,
            self.metric_uav_excess_overlap_fraction_by_drone,
            self.metric_reward_uav_inter_uav_overlap,
            self.metric_uav_inter_uav_overlap_fraction,
            self.metric_uav_inter_uav_overlap_fraction_by_drone,
            self.metric_reward_uav_outside_footprint,
            self.metric_uav_outside_footprint_fraction,
            self.metric_uav_outside_footprint_fraction_by_drone,
            self.metric_uav_boundary_soft_risk,
            self.metric_uav_boundary_distance_m,
            self.metric_uav_boundary_distance_m_by_drone,
            self.metric_uav_displacement_m,
            self.metric_uav_new_coverage_cells,
            self.metric_uav_displacement_m_by_drone,
            self.metric_uav_new_coverage_cells_by_drone,
            self.metric_uav_coverage_opportunity_cells,
            self.metric_uav_coverage_opportunity_cells_by_drone,
            self.metric_uav_coverage_opportunity_fraction,
            self.metric_uav_coverage_opportunity_fraction_by_drone,
            self.metric_uav_coverage_opportunity_available_fraction,
            self.metric_uav_coverage_opportunity_available_fraction_by_drone,
            self.metric_reward_uav_confidence,
            self.metric_reward_uav_confidence_by_drone,
            self.metric_reward_uav_team_confidence,
            self.metric_reward_uav_team_confidence_by_drone,
            self.metric_reward_uav_team_confidence_overlap,
            self.metric_reward_uav_team_confidence_overlap_by_drone,
            self.metric_reward_uav_confidence_move,
            self.metric_reward_uav_confidence_move_by_drone,
            self.metric_reward_uav_confidence_overlap,
            self.metric_reward_uav_confidence_overlap_by_drone,
            self.metric_reward_uav_cleanup_target_progress,
            self.metric_reward_uav_cleanup_target_progress_by_drone,
            self.metric_reward_uav_astar_progress,
            self.metric_reward_uav_astar_progress_by_drone,
            self.metric_uav_astar_progress_fraction,
            self.metric_uav_astar_progress_fraction_by_drone,
            self.metric_uav_astar_frontier_gate,
            self.metric_uav_astar_frontier_gate_by_drone,
            self.metric_uav_astar_path_cost_before_by_drone,
            self.metric_uav_astar_path_cost_after_by_drone,
            self.metric_uav_cleanup_target_frontier_gate,
            self.metric_uav_cleanup_target_frontier_gate_by_drone,
            self.metric_uav_confidence_overlap_fraction,
            self.metric_uav_confidence_overlap_fraction_by_drone,
            self.metric_uav_confidence_overlap_regret,
            self.metric_uav_confidence_overlap_regret_by_drone,
            self.metric_uav_confidence_mean,
            self.metric_uav_confidence_gain,
            self.metric_uav_confidence_gain_by_drone,
            self.metric_uav_weighted_confidence_gain,
            self.metric_uav_weighted_confidence_gain_by_drone,
            self.metric_uav_confidence_opportunity_fraction,
            self.metric_uav_confidence_opportunity_fraction_by_drone,
            self.metric_uav_confidence_opportunity_best_gain,
            self.metric_uav_confidence_opportunity_best_gain_by_drone,
            self.metric_uav_confidence_low_fraction,
            self.metric_uav_confidence_high_fraction,
            self.metric_uav_step_detection_probability,
            self.metric_uav_step_detection_probability_by_drone,
            self.metric_uav_cleanup_target_valid_by_drone,
            self.metric_uav_cleanup_target_distance_m_by_drone,
            self.metric_uav_cleanup_target_value_by_drone,
            self.metric_uav_cleanup_target_progress_m_by_drone,
            self.metric_uav_cleanup_target_progress_fraction_by_drone,
            self.metric_uav_cleanup_target_switch_by_drone,
            self.metric_uav_cleanup_target_reached_by_drone,
            self.metric_uav_cleanup_target_value_decay_by_drone,
            self.metric_uav_cleanup_target_age_by_drone,
            self.metric_uav_target_distance_m,
            self.metric_uav_footprint_radius_m,
            self.metric_uav_footprint_radius_m_by_drone,
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
            self.metric_ugv_duplicate_assignment_fraction,
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
            self.metric_ugv_planner_escape_mode,
            self.metric_ugv_escape_route_active,
            self.metric_ugv_escape_route_enter,
            self.metric_ugv_escape_route_exit,
            self.metric_ugv_escape_route_stall_counter,
            self.metric_ugv_escape_route_age,
            self.metric_ugv_escape_route_waypoint_progress_m,
            self.metric_ugv_escape_route_waypoint_progress_scaled,
            self.metric_ugv_escape_route_waypoint_distance_m,
            self.metric_ugv_escape_route_path_index,
            self.metric_ugv_escape_route_path_length,
            self.metric_ugv_global_route_valid,
            self.metric_ugv_global_route_active,
            self.metric_ugv_global_route_waypoint_distance_m,
            self.metric_ugv_global_route_progress_m,
            self.metric_ugv_global_route_progress_scaled,
            self.metric_ugv_global_route_path_index,
            self.metric_ugv_global_route_path_length,
            self.metric_ugv_global_route_direct_blocked,
            self.metric_ugv_global_route_detour_needed,
            self.metric_ugv_route_fire_cells,
            self.metric_ugv_route_smoke_mean,
            self.metric_ugv_route_smolder_mean,
            self.metric_ugv_route_fire_buffer_cells,
            self.metric_ugv_route_replanned_after_fire,
            self.metric_ugv_route_fire_blocked_no_path,
            self.metric_ugv_route_aware_active,
            self.metric_reward_ugv_stall_penalty,
            self.metric_reward_ugv_route_progress_floor_penalty,
            self.metric_ugv_route_progress_floor_shortfall_m,
            self.metric_reward_ugv_route_progress_shortfall_penalty,
            self.metric_ugv_route_progress_required_m,
            self.metric_ugv_route_progress_shortfall_m,
            self.metric_ugv_route_remaining_distance_m,
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

    def _reset_ugv_escape_routes(self, env_index: int | None = None) -> None:
        if not hasattr(self, "ugv_escape_route_active"):
            return
        tensors = (
            self.ugv_escape_route_active,
            self.ugv_escape_route_age,
            self.ugv_escape_route_stall_counter,
            self.ugv_escape_route_path_index,
        )
        if env_index is None:
            for tensor in tensors:
                tensor.zero_()
            self.ugv_escape_route_target_idx.fill_(-1)
            self.ugv_escape_route_goal_cell.fill_(-1)
            self.ugv_escape_route_waypoint_cell.fill_(-1)
            self.ugv_escape_route_paths = [
                [[] for _ in range(self.n_ground)]
                for _ in range(self.world.batch_dim)
            ]
            return
        env_index = int(env_index)
        for tensor in tensors:
            tensor[env_index] = 0
        self.ugv_escape_route_target_idx[env_index] = -1
        self.ugv_escape_route_goal_cell[env_index] = -1
        self.ugv_escape_route_waypoint_cell[env_index] = -1
        if hasattr(self, "ugv_escape_route_paths"):
            self.ugv_escape_route_paths[env_index] = [
                [] for _ in range(self.n_ground)
            ]

    def _reset_ugv_global_routes(self, env_index: int | None = None) -> None:
        if not hasattr(self, "ugv_global_route_target_idx"):
            return
        tensors = (
            self.ugv_global_route_path_index,
        )
        if env_index is None:
            for tensor in tensors:
                tensor.zero_()
            self.ugv_global_route_target_idx.fill_(-1)
            self.ugv_global_route_goal_cell.fill_(-1)
            self.ugv_global_route_waypoint_cell.fill_(-1)
            self.ugv_global_route_last_replan_step.fill_(-1)
            self.ugv_global_route_fire_replan_pending.zero_()
            self.ugv_global_route_replanned_after_fire_flag.zero_()
            self.ugv_global_route_fire_blocked_no_path_flag.zero_()
            self.ugv_global_route_paths = [
                [[] for _ in range(self.n_ground)]
                for _ in range(self.world.batch_dim)
            ]
            return
        env_index = int(env_index)
        for tensor in tensors:
            tensor[env_index] = 0
        self.ugv_global_route_target_idx[env_index] = -1
        self.ugv_global_route_goal_cell[env_index] = -1
        self.ugv_global_route_waypoint_cell[env_index] = -1
        self.ugv_global_route_last_replan_step[env_index] = -1
        self.ugv_global_route_fire_replan_pending[env_index] = False
        self.ugv_global_route_replanned_after_fire_flag[env_index] = False
        self.ugv_global_route_fire_blocked_no_path_flag[env_index] = False
        if hasattr(self, "ugv_global_route_paths"):
            self.ugv_global_route_paths[env_index] = [
                [] for _ in range(self.n_ground)
            ]

    def _reset_uav_cleanup_targets(self, env_index: int | None = None) -> None:
        if not hasattr(self, "uav_cleanup_target_valid"):
            return
        if env_index is None:
            self.uav_cleanup_target_valid.zero_()
            self.uav_cleanup_target_pos.zero_()
            self.uav_cleanup_target_value.zero_()
            self.uav_cleanup_target_initial_value.zero_()
            self.uav_cleanup_target_age.zero_()
            self.uav_cleanup_target_id.fill_(-1)
            self.uav_cleanup_target_prev_distance_m.fill_(float("inf"))
            self._uav_cleanup_target_last_assignment_step.fill_(-1)
        else:
            self.uav_cleanup_target_valid[env_index] = False
            self.uav_cleanup_target_pos[env_index] = 0.0
            self.uav_cleanup_target_value[env_index] = 0.0
            self.uav_cleanup_target_initial_value[env_index] = 0.0
            self.uav_cleanup_target_age[env_index] = 0
            self.uav_cleanup_target_id[env_index] = -1
            self.uav_cleanup_target_prev_distance_m[env_index] = float("inf")
            self._uav_cleanup_target_last_assignment_step[env_index] = -1
        self._reset_uav_astar_routes(env_index)

    def _reset_uav_astar_routes(self, env_index: int | None = None) -> None:
        if not hasattr(self, "uav_astar_waypoint_valid"):
            return
        if env_index is None:
            self.uav_astar_waypoint_valid.zero_()
            self.uav_astar_waypoint_pos.zero_()
            self.uav_astar_waypoint_target_id.fill_(-1)
            self.uav_astar_waypoint_age.zero_()
            self.uav_astar_path_cost_norm.zero_()
            self._uav_astar_last_plan_step.fill_(-1)
        else:
            self.uav_astar_waypoint_valid[env_index] = False
            self.uav_astar_waypoint_pos[env_index] = 0.0
            self.uav_astar_waypoint_target_id[env_index] = -1
            self.uav_astar_waypoint_age[env_index] = 0
            self.uav_astar_path_cost_norm[env_index] = 0.0
            self._uav_astar_last_plan_step[env_index] = -1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            entities=self._survivors + self._decoys + self.world.agents,
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
            self.active_survivors.fill_(True)
            self.active_decoys.fill_(True)
            self.found_survivors.zero_()
            self.scouted_survivors.zero_()
            self.step_drone_detections.zero_()
            self.step_ground_confirmations.zero_()
            self.known_survivors_by_agent.zero_()
            self.confirmed_survivors_by_agent.zero_()
            self.scouted_decoys.zero_()
            self.dismissed_decoys.zero_()
            self.known_decoys_by_agent.zero_()
            self.step_decoy_false_detections.zero_()
            self.comms_dropout_remaining_steps.zero_()
            self.comms_dropout_last_update_step.fill_(-1)
            self.survivor_reveal_steps.fill_(-1)
            self.survivor_oracle_revealed.zero_()
            self.decoy_reveal_steps.fill_(-1)
            self.decoy_oracle_revealed.zero_()
            self.coverage_grid.zero_()
            self.uav_confidence_grid.zero_()
            if self._comms_maps_enabled():
                self.comm_agent_coverage_grid.zero_()
                self.comm_agent_confidence_grid.zero_()
                self.comm_team_coverage_grid.zero_()
                self.comm_team_confidence_grid.zero_()
                self.comm_map_last_sync_step.fill_(-1)
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
            self.ugv_sticky_target_idx.fill_(-1)
            self.ugv_sticky_target_age.zero_()
            self._invalidate_ugv_assignment_cache()
            self._invalidate_ugv_route_assignment_cost_grid_cache()
            self.ground_approach_milestones_reached.zero_()
            self._reset_step_metric_buffers()
            self._reset_ground_motion_diagnostics()
            self._reset_uav_cleanup_targets()
            self._reset_ugv_escape_routes()
            self._reset_ugv_global_routes()
            envs_to_seed = range(self.world.batch_dim)
        else:
            self.active_survivors[env_index] = True
            self.active_decoys[env_index] = True
            self.found_survivors[env_index] = False
            self.scouted_survivors[env_index] = False
            self.step_drone_detections[env_index] = False
            self.step_ground_confirmations[env_index] = False
            self.known_survivors_by_agent[env_index] = False
            self.confirmed_survivors_by_agent[env_index] = False
            self.scouted_decoys[env_index] = False
            self.dismissed_decoys[env_index] = False
            self.known_decoys_by_agent[env_index] = False
            self.step_decoy_false_detections[env_index] = False
            self.comms_dropout_remaining_steps[env_index] = 0
            self.comms_dropout_last_update_step[env_index] = -1
            self.survivor_reveal_steps[env_index] = -1
            self.survivor_oracle_revealed[env_index] = False
            self.decoy_reveal_steps[env_index] = -1
            self.decoy_oracle_revealed[env_index] = False
            self.coverage_grid[env_index] = False
            self.uav_confidence_grid[env_index] = 0.0
            if self._comms_maps_enabled():
                self.comm_agent_coverage_grid[env_index] = False
                self.comm_agent_confidence_grid[env_index] = 0.0
                self.comm_team_coverage_grid[env_index] = False
                self.comm_team_confidence_grid[env_index] = 0.0
                self.comm_map_last_sync_step[env_index] = -1
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
            self.ugv_sticky_target_idx[env_index] = -1
            self.ugv_sticky_target_age[env_index] = 0
            self._invalidate_ugv_assignment_cache(env_index)
            self._invalidate_ugv_route_assignment_cost_grid_cache(env_index)
            self.ground_approach_milestones_reached[env_index] = False
            self._reset_step_metric_buffers(env_index)
            self._reset_ground_motion_diagnostics(env_index)
            self._reset_uav_cleanup_targets(env_index)
            self._reset_ugv_escape_routes(env_index)
            self._reset_ugv_global_routes(env_index)
            envs_to_seed = [env_index]

        H = W = self.fire_grid_size
        for b in envs_to_seed:
            self._generate_terrain(b)
            self._sample_active_survivors(b)
            self._sample_active_decoys(b)
            ScenarioUtils.spawn_entities_randomly(
                entities=self._active_survivor_entities(b) + self._active_decoy_entities(b) + self.world.agents,
                world=self.world,
                env_index=b,
                min_dist_between_entities=(
                    2 * self.agent_radius
                    + float(self.spawn_padding_by_env[b].item())
                ),
                x_bounds=(-self.x_semidim, self.x_semidim),
                y_bounds=(-self.y_semidim, self.y_semidim),
            )
            self._place_drones_jointly_uniform_interior(b)
            self._place_diagnostic_survivors_near_reference_agents(b)
            self._move_inactive_survivors_outside_map(b)
            self._move_inactive_decoys_outside_map(b)
            if not self.disable_fire:
                self._seed_initial_fire(b, H, W)
            if self.delayed_survivor_knowledge:
                self._sample_delayed_survivor_reveals(b)
            else:
                self._initialize_known_survivors_at_reset(b)
            if self.delayed_decoy_knowledge:
                self._sample_delayed_decoy_reveals(b)
        if self.delayed_survivor_knowledge:
            self._apply_delayed_survivor_reveals(env_index)
        if self.delayed_decoy_knowledge:
            self._apply_delayed_decoy_reveals(env_index)
        self._invalidate_uav_terrain_caches()
        self._invalidate_ugv_planner_route_cache(env_index, terrain_changed=True)

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

    def _active_survivor_mask(self) -> Tensor:
        if hasattr(self, "active_survivors"):
            return self.active_survivors
        return torch.ones_like(self.found_survivors)

    def _active_survivor_count(self) -> Tensor:
        return self._active_survivor_mask().float().sum(dim=1)

    def _active_decoy_mask(self) -> Tensor:
        if hasattr(self, "active_decoys"):
            return self.active_decoys
        return torch.ones_like(self.scouted_decoys)

    def _active_decoy_count(self) -> Tensor:
        return self._active_decoy_mask().float().sum(dim=1)

    def _all_active_survivors_found(self) -> Tensor:
        active = self._active_survivor_mask()
        return (self.found_survivors | ~active).all(dim=1)

    def _all_active_survivors_scouted(self) -> Tensor:
        active = self._active_survivor_mask()
        return (self.scouted_survivors | ~active).all(dim=1)

    def _sample_active_survivors(self, env_index: int) -> None:
        if self.n_survivors <= 0:
            return
        device = self.active_survivors.device
        min_count = min(max(int(self.active_survivors_min), 0), self.n_survivors)
        max_count = min(max(int(self.active_survivors_max), min_count), self.n_survivors)
        if min_count == max_count:
            count = min_count
        else:
            count = int(torch.randint(min_count, max_count + 1, (1,), device=device).item())
        mask = torch.zeros(self.n_survivors, dtype=torch.bool, device=device)
        if count > 0:
            mask[torch.randperm(self.n_survivors, device=device)[:count]] = True
        self.active_survivors[env_index] = mask

    def _active_survivor_entities(self, env_index: int) -> list[Landmark]:
        if self.n_survivors <= 0:
            return []
        active = self._active_survivor_mask()[env_index]
        return [
            survivor
            for survivor_idx, survivor in enumerate(self._survivors)
            if bool(active[survivor_idx].item())
        ]

    def _move_inactive_survivors_outside_map(self, env_index: int) -> None:
        if self.n_survivors <= 0:
            return
        inactive = ~self._active_survivor_mask()[env_index]
        if not bool(inactive.any().item()):
            return
        offmap = torch.tensor(
            [self.x_semidim + 10.0 * self.agent_radius, self.y_semidim + 10.0 * self.agent_radius],
            device=self.fire_grid.device,
            dtype=torch.float32,
        )
        for survivor_idx, survivor in enumerate(self._survivors):
            if bool(inactive[survivor_idx].item()):
                all_pos = survivor.state.pos.clone()
                all_pos[env_index] = offmap.to(device=all_pos.device, dtype=all_pos.dtype)
                survivor.set_pos(all_pos, batch_index=None)

    def _sample_active_decoys(self, env_index: int) -> None:
        if self.n_decoys <= 0:
            return
        device = self.active_decoys.device
        min_count = min(max(int(self.active_decoys_min), 0), self.n_decoys)
        max_count = min(max(int(self.active_decoys_max), min_count), self.n_decoys)
        if min_count == max_count:
            count = min_count
        else:
            count = int(torch.randint(min_count, max_count + 1, (1,), device=device).item())
        mask = torch.zeros(self.n_decoys, dtype=torch.bool, device=device)
        if count > 0:
            mask[torch.randperm(self.n_decoys, device=device)[:count]] = True
        self.active_decoys[env_index] = mask

    def _active_decoy_entities(self, env_index: int) -> list[Landmark]:
        if self.n_decoys <= 0:
            return []
        active = self._active_decoy_mask()[env_index]
        return [
            decoy
            for decoy_idx, decoy in enumerate(self._decoys)
            if bool(active[decoy_idx].item())
        ]

    def _move_inactive_decoys_outside_map(self, env_index: int) -> None:
        if self.n_decoys <= 0:
            return
        inactive = ~self._active_decoy_mask()[env_index]
        if not bool(inactive.any().item()):
            return
        offmap = torch.tensor(
            [self.x_semidim + 12.0 * self.agent_radius, self.y_semidim + 12.0 * self.agent_radius],
            device=self.fire_grid.device,
            dtype=torch.float32,
        )
        for decoy_idx, decoy in enumerate(self._decoys):
            if bool(inactive[decoy_idx].item()):
                all_pos = decoy.state.pos.clone()
                all_pos[env_index] = offmap.to(device=all_pos.device, dtype=all_pos.dtype)
                decoy.set_pos(all_pos, batch_index=None)

    def _initialize_known_survivors_at_reset(self, env_index: int) -> None:
        """Optionally start the episode with survivors known to ground agents."""
        if not self.known_survivors_at_reset or self.n_survivors <= 0:
            return
        active = self._active_survivor_mask()[env_index]
        self.scouted_survivors[env_index] = active
        if self.n_ground > 0:
            self.known_survivors_by_agent[env_index, self.n_drones:, :] = active.unsqueeze(0)

    def _sample_delayed_survivor_reveals(self, env_index: int) -> None:
        if not self.delayed_survivor_knowledge or self.n_survivors <= 0:
            return
        device = self.survivor_reveal_steps.device
        active_idx = torch.nonzero(
            self._active_survivor_mask()[env_index],
            as_tuple=False,
        ).flatten()
        count = int(active_idx.numel())
        reveal_steps = torch.full((self.n_survivors,), -1, dtype=torch.long, device=device)
        if count <= 0:
            self.survivor_reveal_steps[env_index] = reveal_steps
            return
        initial_count = min(int(self.survivor_reveal_initial_count), count)
        active_reveal_steps = torch.full((count,), self.max_steps + 1, dtype=torch.long, device=device)
        order = torch.randperm(count, device=device)
        if initial_count > 0:
            active_reveal_steps[order[:initial_count]] = 0
        remaining = count - initial_count
        if remaining > 0:
            start = int(min(max(self.survivor_reveal_start_step, 0), self.max_steps))
            end = int(min(max(self.survivor_reveal_end_step, start), self.max_steps))
            edges = torch.linspace(float(start), float(end), remaining + 1, device=device)
            for k, active_order_idx in enumerate(order[initial_count:]):
                lo = int(torch.floor(edges[k]).item())
                hi = int(torch.floor(edges[k + 1]).item())
                if hi <= lo:
                    step = lo
                else:
                    step = int(torch.randint(lo, hi + 1, (1,), device=device).item())
                active_reveal_steps[active_order_idx] = step
        reveal_steps[active_idx] = active_reveal_steps
        self.survivor_reveal_steps[env_index] = reveal_steps

    def _sample_delayed_decoy_reveals(self, env_index: int) -> None:
        if not self.delayed_decoy_knowledge or self.n_decoys <= 0:
            return
        device = self.decoy_reveal_steps.device
        active_idx = torch.nonzero(
            self._active_decoy_mask()[env_index],
            as_tuple=False,
        ).flatten()
        count = int(active_idx.numel())
        reveal_steps = torch.full((self.n_decoys,), -1, dtype=torch.long, device=device)
        if count <= 0:
            self.decoy_reveal_steps[env_index] = reveal_steps
            return
        initial_count = min(int(self.decoy_reveal_initial_count), count)
        active_reveal_steps = torch.full((count,), self.max_steps + 1, dtype=torch.long, device=device)
        order = torch.randperm(count, device=device)
        if initial_count > 0:
            active_reveal_steps[order[:initial_count]] = 0
        remaining = count - initial_count
        if remaining > 0:
            start = int(min(max(self.decoy_reveal_start_step, 0), self.max_steps))
            end = int(min(max(self.decoy_reveal_end_step, start), self.max_steps))
            edges = torch.linspace(float(start), float(end), remaining + 1, device=device)
            for k, active_order_idx in enumerate(order[initial_count:]):
                lo = int(torch.floor(edges[k]).item())
                hi = int(torch.floor(edges[k + 1]).item())
                if hi <= lo:
                    step = lo
                else:
                    step = int(torch.randint(lo, hi + 1, (1,), device=device).item())
                active_reveal_steps[active_order_idx] = step
        reveal_steps[active_idx] = active_reveal_steps
        self.decoy_reveal_steps[env_index] = reveal_steps

    def _apply_delayed_survivor_reveals(self, env_index: int | None = None) -> None:
        if not self.delayed_survivor_knowledge or self.n_survivors <= 0:
            return
        if env_index is None:
            env_slice = slice(None)
            steps = self.step_count.view(-1, 1)
        else:
            env_slice = slice(env_index, env_index + 1)
            steps = self.step_count[env_index : env_index + 1].view(1, 1)
        due = (
            (self.survivor_reveal_steps[env_slice] >= 0)
            & (self.survivor_reveal_steps[env_slice] <= steps)
            & ~self.survivor_oracle_revealed[env_slice]
            & self._active_survivor_mask()[env_slice]
        )
        if due.numel() == 0 or not bool(due.any().item()):
            return
        self.survivor_oracle_revealed[env_slice] |= due
        self.scouted_survivors[env_slice] |= due
        if self.n_ground > 0:
            self.known_survivors_by_agent[env_slice, self.n_drones:, :] |= due.unsqueeze(1)
        reveal_counts = due.float().sum(dim=1)
        if env_index is None:
            self.metric_survivor_oracle_reveals.copy_(reveal_counts)
        else:
            self.metric_survivor_oracle_reveals[env_index] = reveal_counts[0]
        self._invalidate_ugv_assignment_cache(env_index)

    def _apply_delayed_decoy_reveals(self, env_index: int | None = None) -> None:
        if not self.delayed_decoy_knowledge or self.n_decoys <= 0:
            return
        if env_index is None:
            env_slice = slice(None)
            steps = self.step_count.view(-1, 1)
        else:
            env_slice = slice(env_index, env_index + 1)
            steps = self.step_count[env_index : env_index + 1].view(1, 1)
        due = (
            (self.decoy_reveal_steps[env_slice] >= 0)
            & (self.decoy_reveal_steps[env_slice] <= steps)
            & ~self.decoy_oracle_revealed[env_slice]
            & ~self.dismissed_decoys[env_slice]
            & self._active_decoy_mask()[env_slice]
        )
        if due.numel() == 0 or not bool(due.any().item()):
            return
        self.decoy_oracle_revealed[env_slice] |= due
        self.scouted_decoys[env_slice] |= due
        if self.n_ground > 0:
            self.known_decoys_by_agent[env_slice, self.n_drones:, :] |= due.unsqueeze(1)
        reveal_counts = due.float().sum(dim=1)
        if env_index is None:
            self.metric_decoy_oracle_reveals.copy_(reveal_counts)
        else:
            self.metric_decoy_oracle_reveals[env_index] = reveal_counts[0]
        self._invalidate_ugv_assignment_cache(env_index)

    def _ugv_ground_target_candidates(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return combined UGV targets: true survivors first, then decoys."""
        device = self.fire_grid.device
        ground_slice = slice(self.n_drones, self.n_agents)
        target_pos_parts: list[Tensor] = []
        targetable_parts: list[Tensor] = []
        is_decoy_parts: list[Tensor] = []

        if self.n_survivors > 0:
            survivor_pos = torch.stack([s.state.pos for s in self._survivors], dim=1)
            active = self._active_survivor_mask()
            unconfirmed_scouted = active & self.scouted_survivors & ~self.found_survivors
            ground_known = self.known_survivors_by_agent[:, ground_slice]
            survivor_targetable = ground_known & unconfirmed_scouted.unsqueeze(1)
            target_pos_parts.append(survivor_pos)
            targetable_parts.append(survivor_targetable)
            is_decoy_parts.append(torch.zeros(
                self.world.batch_dim,
                self.n_survivors,
                dtype=torch.bool,
                device=device,
            ))

        if self.n_decoys > 0:
            decoy_pos = torch.stack([d.state.pos for d in self._decoys], dim=1)
            ground_decoy_known = self.known_decoys_by_agent[:, ground_slice]
            decoy_targetable = (
                ground_decoy_known
                & self.scouted_decoys.unsqueeze(1)
                & ~self.dismissed_decoys.unsqueeze(1)
                & self._active_decoy_mask().unsqueeze(1)
            )
            target_pos_parts.append(decoy_pos)
            targetable_parts.append(decoy_targetable)
            is_decoy_parts.append(torch.ones(
                self.world.batch_dim,
                self.n_decoys,
                dtype=torch.bool,
                device=device,
            ))

        if not target_pos_parts:
            return (
                torch.zeros(self.world.batch_dim, 0, 2, device=device),
                torch.zeros(self.world.batch_dim, self.n_ground, 0, dtype=torch.bool, device=device),
                torch.zeros(self.world.batch_dim, 0, dtype=torch.bool, device=device),
            )
        return (
            torch.cat(target_pos_parts, dim=1),
            torch.cat(targetable_parts, dim=2),
            torch.cat(is_decoy_parts, dim=1),
        )

    def _invalidate_ugv_assignment_cache(self, env_index: int | None = None) -> None:
        if not hasattr(self, "ugv_assignment_cache_step"):
            return
        if env_index is None:
            self.ugv_assignment_cache_step.fill_(-1)
        else:
            self.ugv_assignment_cache_step[env_index] = -1
        self._ugv_assignment_result_cache = None

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
            if not bool(self._active_survivor_mask()[env_index, survivor_idx].item()):
                continue
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
        entities = self._survivors + self._decoys + self.world.agents[self.n_drones:]
        for entity in entities:
            search_mask = available if bool(available.any().item()) else candidates
            new_x, new_y = self._sample_random_cell_from_mask(search_mask)
            new_pos = self._grid_cell_center_to_world(new_x, new_y, device=device)
            all_pos = entity.state.pos.clone()
            all_pos[env_index] = new_pos
            entity.set_pos(all_pos, batch_index=None)
            available[new_y, new_x] = False

    def _place_drones_jointly_uniform_interior(self, env_index: int) -> None:
        """Jointly sample UAV starts, rejecting the whole team if spacing fails."""
        if self.n_drones <= 0:
            return
        if self.uav_start_min_separation_m <= 0.0 and self.uav_start_edge_margin_m <= 0.0:
            return

        drone_agents = self.world.agents[:self.n_drones]
        if not drone_agents:
            return

        device = drone_agents[0].state.pos.device
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        if scale <= 0.0:
            return

        margin_sim = self.agent_radius + self.uav_start_edge_margin_m * scale
        x_min = -self.x_semidim + margin_sim
        x_max = self.x_semidim - margin_sim
        y_min = -self.y_semidim + margin_sim
        y_max = self.y_semidim - margin_sim
        if x_min > x_max or y_min > y_max:
            raise ValueError(
                "uav_start_edge_margin_m leaves no valid interior start area "
                f"({self.uav_start_edge_margin_m:.1f}m on this terrain)"
            )

        min_sep_sim = self.uav_start_min_separation_m * scale
        best_min_distance = -1.0
        for _ in range(self.uav_start_max_attempts):
            xs = x_min + torch.rand(self.n_drones, device=device) * max(x_max - x_min, 0.0)
            ys = y_min + torch.rand(self.n_drones, device=device) * max(y_max - y_min, 0.0)
            positions = torch.stack([xs, ys], dim=-1)
            pairwise = torch.pdist(positions) if self.n_drones > 1 else torch.empty(0, device=device)
            min_distance = float(pairwise.min().detach().cpu().item()) if pairwise.numel() else math.inf
            if min_distance > best_min_distance:
                best_min_distance = min_distance
            if min_distance + 1e-9 >= min_sep_sim:
                perm = torch.randperm(self.n_drones, device=device)
                self._set_drone_start_positions(env_index, positions[perm])
                return

        best_m = best_min_distance / max(scale, 1e-9)
        raise ValueError(
            "Could not sample UAV starts satisfying "
            f"uav_start_min_separation_m={self.uav_start_min_separation_m:.1f} "
            f"after {self.uav_start_max_attempts} attempts; best was {best_m:.1f}m. "
            "Reduce the separation/margin or increase uav_start_max_attempts."
        )

    def _set_drone_start_positions(self, env_index: int, positions: Tensor) -> None:
        for drone_idx, agent in enumerate(self.world.agents[:self.n_drones]):
            all_pos = agent.state.pos.clone()
            all_pos[env_index] = positions[drone_idx]
            agent.set_pos(all_pos, batch_index=None)
            if hasattr(agent.state, "vel"):
                all_vel = agent.state.vel.clone()
                all_vel[env_index] = 0.0
                agent.set_vel(all_vel, batch_index=None)

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
        if hasattr(self, "_ugv_planner_layer_tensor_cache"):
            self._invalidate_ugv_planner_layer_cache(env_index)
        if hasattr(self, "_ugv_global_heuristic_cache"):
            self._invalidate_ugv_global_heuristic_cache(env_index)

    def _invalidate_ugv_planner_fire_mask_cache(self, env_index: int | None = None) -> None:
        del env_index
        self._ugv_planner_fire_mask_cache_version = (
            getattr(self, "_ugv_planner_fire_mask_cache_version", 0) + 1
        )
        caches = (
            "_ugv_planner_fire_buffer_mask_cache",
            "_ugv_planner_blocked_fire_mask_cache",
        )
        for name in caches:
            if hasattr(self, name):
                getattr(self, name).clear()

    def _invalidate_ugv_planner_layer_cache(
        self,
        env_index: int | None = None,
        *,
        fire_masks_changed: bool = True,
    ) -> None:
        del env_index
        if fire_masks_changed:
            self._invalidate_ugv_planner_fire_mask_cache()
        self._ugv_planner_layer_cache_version = (
            getattr(self, "_ugv_planner_layer_cache_version", 0) + 1
        )
        caches = (
            "_ugv_planner_layer_tensor_cache",
            "_ugv_planner_layer_array_cache",
        )
        for name in caches:
            if hasattr(self, name):
                getattr(self, name).clear()

    def _invalidate_ugv_global_heuristic_cache(self, env_index: int | None = None) -> None:
        self._invalidate_ugv_assignment_cache(env_index)
        self._invalidate_ugv_route_assignment_cost_grid_cache(env_index)
        caches = (
            "_ugv_static_planner_layer_array_cache",
            "_ugv_static_planner_graph_cache",
            "_ugv_global_heuristic_cache",
            "_ugv_global_raw_cost_to_go_cache",
        )
        if env_index is None:
            self._ugv_static_planner_cache_version = (
                getattr(self, "_ugv_static_planner_cache_version", 0) + 1
            )
            for name in caches:
                if hasattr(self, name):
                    getattr(self, name).clear()
            return
        env_index = int(env_index)
        for name in caches:
            if not hasattr(self, name):
                continue
            cache = getattr(self, name)
            for key in list(cache.keys()):
                if key and int(key[0]) == env_index:
                    del cache[key]

    def _invalidate_ugv_route_assignment_cost_grid_cache(self, env_index: int | None = None) -> None:
        if not hasattr(self, "_ugv_route_assignment_cost_grid_cache"):
            self._ugv_route_assignment_cost_grid_cache = {}
        if env_index is None:
            self._ugv_route_assignment_cost_grid_cache.clear()
            return
        env_index = int(env_index)
        self._ugv_route_assignment_cost_grid_cache = {
            key: value
            for key, value in self._ugv_route_assignment_cost_grid_cache.items()
            if key[0] != env_index
        }

    def _ugv_planner_fire_buffer_mask(self, env_index: int) -> Tensor:
        version = int(getattr(self, "_ugv_planner_fire_mask_cache_version", 0))
        key = (int(env_index), version)
        cache = getattr(self, "_ugv_planner_fire_buffer_mask_cache", None)
        if cache is None:
            self._ugv_planner_fire_buffer_mask_cache = {}
            cache = self._ugv_planner_fire_buffer_mask_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        fire = self.fire_grid[env_index].bool()
        if (
            self.ugv_planner_fire_buffer_m <= 0.0
            or self.ugv_planner_fire_buffer_cost <= 0.0
            or not bool(fire.any().item())
        ):
            mask = torch.zeros_like(fire)
            cache[key] = mask
            return mask
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        radius_sim = float(self.ugv_planner_fire_buffer_m) * max(scale, 1e-9)
        radius_cells = self._world_length_to_cells(radius_sim, min_cells=0)
        if radius_cells <= 0:
            mask = torch.zeros_like(fire)
            cache[key] = mask
            return mask
        mask = (self._local_true_count(fire, radius_cells) > 0.0) & ~fire
        cache[key] = mask
        return mask

    def _ugv_planner_blocked_fire_mask(self, env_index: int) -> Tensor:
        version = int(getattr(self, "_ugv_planner_fire_mask_cache_version", 0))
        key = (int(env_index), version)
        cache = getattr(self, "_ugv_planner_blocked_fire_mask_cache", None)
        if cache is None:
            self._ugv_planner_blocked_fire_mask_cache = {}
            cache = self._ugv_planner_blocked_fire_mask_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        fire = self.fire_grid[env_index].bool()
        if self.ugv_planner_fire_block_threshold <= 0.0:
            cache[key] = fire
            return fire
        mask = fire & (
            self.fire_intensity_grid[env_index].clamp(0.0, 1.0)
            >= float(self.ugv_planner_fire_block_threshold)
        )
        cache[key] = mask
        return mask

    def _ugv_planner_layer_tensors_for_env(self, env_index: int) -> tuple[Tensor, Tensor]:
        version = int(getattr(self, "_ugv_planner_layer_cache_version", 0))
        key = (int(env_index), version)
        cache = getattr(self, "_ugv_planner_layer_tensor_cache", None)
        if cache is None:
            self._ugv_planner_layer_tensor_cache = {}
            cache = self._ugv_planner_layer_tensor_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        traversable = self.traversable_grid[env_index].clone()
        if self.ugv_planner_land_cover_cost_values is None:
            movement_cost = self.mobility_cost_grid[env_index].clone()
        else:
            cover = self.land_cover_grid[env_index]
            slope = self.slope_grid[env_index]
            movement_cost = (
                self.ugv_planner_land_cover_cost_values[cover]
                * (1.0 + self.slope_cost_weight * slope)
            )
        if self.ugv_planner_fire_mode == "off":
            return traversable, movement_cost

        fire = self.fire_grid[env_index].bool()
        if self.ugv_planner_fire_mode == "block":
            blocked_fire = self._ugv_planner_blocked_fire_mask(env_index)
            traversable = traversable & ~blocked_fire
            if self.ugv_planner_fire_cost > 0.0:
                soft_fire = fire & ~blocked_fire
                movement_cost = movement_cost + (
                    soft_fire.float()
                    * self.fire_intensity_grid[env_index].clamp(0.0, 1.0)
                    * float(self.ugv_planner_fire_cost)
                )
        elif self.ugv_planner_fire_cost > 0.0:
            movement_cost = movement_cost + fire.float() * float(self.ugv_planner_fire_cost)

        if self.ugv_planner_smoke_cost > 0.0:
            movement_cost = movement_cost + (
                self.smoke_grid[env_index].clamp(0.0, 1.0) * float(self.ugv_planner_smoke_cost)
            )
        if self.ugv_planner_smolder_cost > 0.0:
            movement_cost = movement_cost + (
                self.smolder_grid[env_index].clamp(0.0, 1.0) * float(self.ugv_planner_smolder_cost)
            )
        if self.ugv_planner_fire_buffer_cost > 0.0 and self.ugv_planner_fire_buffer_m > 0.0:
            movement_cost = movement_cost + (
                self._ugv_planner_fire_buffer_mask(env_index).float()
                * float(self.ugv_planner_fire_buffer_cost)
            )
        cache[key] = (traversable, movement_cost)
        return traversable, movement_cost

    def _ugv_planner_layer_arrays_for_env(self, env_index: int) -> tuple[np.ndarray, np.ndarray]:
        version = int(getattr(self, "_ugv_planner_layer_cache_version", 0))
        key = (int(env_index), version)
        cache = getattr(self, "_ugv_planner_layer_array_cache", None)
        if cache is None:
            self._ugv_planner_layer_array_cache = {}
            cache = self._ugv_planner_layer_array_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        traversable, movement_cost = self._ugv_planner_layer_tensors_for_env(env_index)
        arrays = (
            traversable.detach().cpu().numpy().astype(bool, copy=False),
            movement_cost.detach().cpu().numpy(),
        )
        cache[key] = arrays
        return arrays

    def _ugv_static_planner_layer_arrays_for_env(self, env_index: int) -> tuple[np.ndarray, np.ndarray]:
        version = int(getattr(self, "_ugv_static_planner_cache_version", 0))
        key = (int(env_index), version)
        cache = getattr(self, "_ugv_static_planner_layer_array_cache", None)
        if cache is None:
            self._ugv_static_planner_layer_array_cache = {}
            cache = self._ugv_static_planner_layer_array_cache
        cached = cache.get(key)
        if cached is not None:
            return cached

        traversable = self.traversable_grid[env_index].detach().cpu().numpy().astype(bool, copy=True)
        if self.ugv_planner_land_cover_cost_values is None:
            movement_cost = self.mobility_cost_grid[env_index].detach().cpu().numpy().astype(
                np.float64,
                copy=True,
            )
        else:
            cover = self.land_cover_grid[env_index]
            slope = self.slope_grid[env_index]
            planner_cost = (
                self.ugv_planner_land_cover_cost_values[cover]
                * (1.0 + self.slope_cost_weight * slope)
            )
            movement_cost = planner_cost.detach().cpu().numpy().astype(np.float64, copy=True)
        arrays = (traversable, movement_cost)
        cache[key] = arrays
        return arrays

    def _ugv_static_planner_graph_for_env(self, env_index: int):
        tools = _scipy_sparse_tools()
        if tools is None:
            return None
        csr_matrix, _dijkstra = tools
        version = int(getattr(self, "_ugv_static_planner_cache_version", 0))
        key = (int(env_index), version)
        cache = getattr(self, "_ugv_static_planner_graph_cache", None)
        if cache is None:
            self._ugv_static_planner_graph_cache = {}
            cache = self._ugv_static_planner_graph_cache
        cached = cache.get(key)
        if cached is not None:
            return cached

        traversable, movement_cost = self._ugv_static_planner_layer_arrays_for_env(env_index)
        valid = traversable & np.isfinite(movement_cost)
        G = int(self.fire_grid_size)
        indices = np.arange(G * G, dtype=np.int32).reshape(G, G)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []

        def add_edges(mask: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray, weight: np.ndarray) -> None:
            if not bool(np.any(mask)):
                return
            a = a_idx[mask].astype(np.int32, copy=False)
            b = b_idx[mask].astype(np.int32, copy=False)
            w = weight[mask].astype(np.float64, copy=False)
            rows.extend((a, b))
            cols.extend((b, a))
            data.extend((w, w))

        right = valid[:, :-1] & valid[:, 1:]
        add_edges(
            right,
            indices[:, :-1],
            indices[:, 1:],
            0.5 * (movement_cost[:, :-1] + movement_cost[:, 1:]),
        )
        down = valid[:-1, :] & valid[1:, :]
        add_edges(
            down,
            indices[:-1, :],
            indices[1:, :],
            0.5 * (movement_cost[:-1, :] + movement_cost[1:, :]),
        )
        diag_scale = math.sqrt(2.0) * 0.5
        down_right = (
            valid[:-1, :-1]
            & valid[1:, 1:]
            & valid[:-1, 1:]
            & valid[1:, :-1]
        )
        add_edges(
            down_right,
            indices[:-1, :-1],
            indices[1:, 1:],
            diag_scale * (movement_cost[:-1, :-1] + movement_cost[1:, 1:]),
        )
        down_left = (
            valid[:-1, 1:]
            & valid[1:, :-1]
            & valid[:-1, :-1]
            & valid[1:, 1:]
        )
        add_edges(
            down_left,
            indices[:-1, 1:],
            indices[1:, :-1],
            diag_scale * (movement_cost[:-1, 1:] + movement_cost[1:, :-1]),
        )

        if rows:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            values = np.concatenate(data)
        else:
            row = np.empty((0,), dtype=np.int32)
            col = np.empty((0,), dtype=np.int32)
            values = np.empty((0,), dtype=np.float64)
        graph = csr_matrix((values, (row, col)), shape=(G * G, G * G))
        cache[key] = graph
        return graph

    def _ugv_route_fire_stats_for_env(self, env_index: int, path: list[tuple[int, int]]) -> dict[str, float]:
        if not path:
            return {
                "fire_cells": 0.0,
                "smoke_mean": 0.0,
                "smolder_mean": 0.0,
                "fire_buffer_cells": 0.0,
            }
        xs = torch.tensor([int(x) for x, _y in path], dtype=torch.long, device=self.fire_grid.device)
        ys = torch.tensor([int(y) for _x, y in path], dtype=torch.long, device=self.fire_grid.device)
        fire = self.fire_grid[env_index, ys, xs].float()
        smoke = self.smoke_grid[env_index, ys, xs].float()
        smolder = self.smolder_grid[env_index, ys, xs].float()
        buffer = self._ugv_planner_fire_buffer_mask(env_index)[ys, xs].float()
        return {
            "fire_cells": float(fire.sum().detach().cpu().item()),
            "smoke_mean": float(smoke.mean().detach().cpu().item()),
            "smolder_mean": float(smolder.mean().detach().cpu().item()),
            "fire_buffer_cells": float(buffer.sum().detach().cpu().item()),
        }

    def _ugv_fire_blocked_no_path_active(self, env_index: int) -> bool:
        return (
            self.ugv_planner_fire_mode == "block"
            and bool(self._ugv_planner_blocked_fire_mask(env_index).any().item())
        )

    def _clear_ugv_global_route(
        self,
        env_index: int,
        ground_index: int,
        *,
        fire_changed: bool = False,
    ) -> None:
        if not hasattr(self, "ugv_global_route_paths"):
            return
        self.ugv_global_route_target_idx[env_index, ground_index] = -1
        self.ugv_global_route_path_index[env_index, ground_index] = 0
        self.ugv_global_route_goal_cell[env_index, ground_index] = -1
        self.ugv_global_route_waypoint_cell[env_index, ground_index] = -1
        self.ugv_global_route_last_replan_step[env_index, ground_index] = -1
        self.ugv_global_route_paths[env_index][ground_index] = []
        if fire_changed:
            self.ugv_global_route_fire_replan_pending[env_index, ground_index] = True
        else:
            self.ugv_global_route_fire_replan_pending[env_index, ground_index] = False
        self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index] = False
        self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index] = False

    def _ugv_global_route_near_risk(
        self,
        env_index: int,
        ground_index: int,
        risk: Tensor,
    ) -> bool:
        path = self.ugv_global_route_paths[env_index][ground_index]
        if not path:
            return False
        start_idx = int(self.ugv_global_route_path_index[env_index, ground_index].item())
        start_idx = max(0, min(start_idx, len(path) - 1))
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        cell_w_m = (2.0 * float(self.x_semidim) / float(self.fire_grid_size)) / max(scale, 1e-9)
        cell_h_m = (2.0 * float(self.y_semidim) / float(self.fire_grid_size)) / max(scale, 1e-9)
        accumulated_m = 0.0
        previous = path[start_idx]
        end_idx = start_idx
        for idx in range(start_idx + 1, len(path)):
            candidate = path[idx]
            step_m = math.hypot(
                abs(candidate[0] - previous[0]) * cell_w_m,
                abs(candidate[1] - previous[1]) * cell_h_m,
            )
            next_accumulated = accumulated_m + step_m
            if next_accumulated > self.ugv_global_planner_lookahead_m and idx > start_idx + 1:
                break
            end_idx = idx
            accumulated_m = next_accumulated
            previous = candidate

        cells = path[start_idx : end_idx + 1]
        if not cells:
            return False
        xs = torch.tensor(
            [int(x) for x, _y in cells],
            dtype=torch.long,
            device=self.fire_grid.device,
        )
        ys = torch.tensor(
            [int(y) for _x, y in cells],
            dtype=torch.long,
            device=self.fire_grid.device,
        )
        return bool(risk[ys, xs].any().item())

    def _ugv_global_route_lazy_replan_due(self, env_index: int, ground_index: int) -> bool:
        last_step = int(self.ugv_global_route_last_replan_step[env_index, ground_index].item())
        if last_step < 0:
            return True
        current_step = int(self.step_count[env_index].item())
        return (current_step - last_step) >= int(self.ugv_planner_fire_replan_interval_steps)

    def _invalidate_ugv_planner_routes_for_fire_change(self) -> None:
        if self.ugv_planner_fire_mode == "off":
            return
        self._invalidate_ugv_planner_layer_cache()
        if self.ugv_planner_fire_replan_policy == "always":
            self._invalidate_ugv_planner_route_cache(terrain_changed=True, fire_changed=True)
            return
        if not hasattr(self, "_ugv_planner_route_cache"):
            self._ugv_planner_route_cache = {}
        self._ugv_planner_route_cache.clear()
        self._ugv_planner_terrain_cache_version = (
            getattr(self, "_ugv_planner_terrain_cache_version", 0) + 1
        )
        if hasattr(self, "ugv_escape_route_active"):
            self._reset_ugv_escape_routes()
        if not hasattr(self, "ugv_global_route_paths"):
            return

        for env_index in range(self.world.batch_dim):
            risk = self.fire_grid[env_index].bool()
            if self.ugv_planner_fire_buffer_m > 0.0 and self.ugv_planner_fire_buffer_cost > 0.0:
                risk = risk | self._ugv_planner_fire_buffer_mask(env_index)
            if not bool(risk.any().item()):
                continue
            for ground_index in range(self.n_ground):
                path = self.ugv_global_route_paths[env_index][ground_index]
                if not path:
                    continue
                if self.ugv_planner_fire_replan_policy == "lazy":
                    should_replan = self._ugv_global_route_near_risk(
                        env_index,
                        ground_index,
                        risk,
                    ) or self._ugv_global_route_lazy_replan_due(env_index, ground_index)
                else:
                    xs = torch.tensor(
                        [int(x) for x, _y in path],
                        dtype=torch.long,
                        device=self.fire_grid.device,
                    )
                    ys = torch.tensor(
                        [int(y) for _x, y in path],
                        dtype=torch.long,
                        device=self.fire_grid.device,
                    )
                    should_replan = bool(risk[ys, xs].any().item())
                if should_replan:
                    self._clear_ugv_global_route(
                        env_index,
                        ground_index,
                        fire_changed=True,
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
        self.metric_survivor_oracle_reveals.zero_()
        self.metric_decoy_oracle_reveals.zero_()
        self.metric_ugv_assignment_switches.zero_()
        self._apply_delayed_survivor_reveals()
        self._apply_delayed_decoy_reveals()

        if not self.disable_fire:
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
        if self.ugv_planner_fire_mode != "off":
            self._invalidate_ugv_planner_routes_for_fire_change()

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
        if (
            self.ugv_planner_fire_mode != "off"
            and (self.ugv_planner_smoke_cost > 0.0 or self.ugv_planner_smolder_cost > 0.0)
        ):
            self._invalidate_ugv_planner_layer_cache(fire_masks_changed=False)

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
            self._invalidate_uav_local_confidence_obs_cache()
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

    def _ugv_assigned_target_indices(
        self,
        ground_pos: Tensor,
        survivor_pos: Tensor,
        targetable: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cached = getattr(self, "_ugv_assignment_result_cache", None)
        step_key = tuple(int(v) for v in self.step_count.detach().cpu().reshape(-1).tolist())
        cache_key = (
            step_key,
            self.ugv_target_assignment_mode,
            int(self.n_ground),
            int(self.n_survivors),
            int(getattr(self, "_ugv_static_planner_cache_version", 0)),
            int(getattr(self, "_ugv_planner_terrain_cache_version", 0)),
            str(ground_pos.device),
            str(ground_pos.dtype),
        )
        if cached is not None and cached.get("key") == cache_key:
            if (
                torch.equal(cached["ground_pos"], ground_pos)
                and torch.equal(cached["survivor_pos"], survivor_pos)
                and torch.equal(cached["targetable"], targetable)
            ):
                return cached["assigned_idx"], cached["assigned_dist"]

        assigned_idx, assigned_dist = self._ugv_assigned_target_indices_uncached(
            ground_pos,
            survivor_pos,
            targetable,
        )
        self._ugv_assignment_result_cache = {
            "key": cache_key,
            "ground_pos": ground_pos.detach().clone(),
            "survivor_pos": survivor_pos.detach().clone(),
            "targetable": targetable.detach().clone(),
            "assigned_idx": assigned_idx,
            "assigned_dist": assigned_dist,
        }
        return assigned_idx, assigned_dist

    def _ugv_assigned_target_indices_uncached(
        self,
        ground_pos: Tensor,
        survivor_pos: Tensor,
        targetable: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Assign known, unconfirmed survivor targets to UGVs."""
        batch_dim = ground_pos.shape[0]
        device = ground_pos.device
        n_targets = int(targetable.shape[2]) if targetable.ndim == 3 else 0
        if self.n_ground == 0 or n_targets == 0:
            return (
                torch.full((batch_dim, self.n_ground), -1, dtype=torch.long, device=device),
                torch.full((batch_dim, self.n_ground), float("inf"), device=device),
            )

        distances = torch.linalg.norm(survivor_pos.unsqueeze(1) - ground_pos.unsqueeze(2), dim=-1)
        masked = torch.where(targetable, distances, torch.full_like(distances, float("inf")))
        if self.ugv_target_assignment_mode == "nearest" or (
            self.n_ground <= 1
            and self.ugv_target_assignment_mode
            not in {"route_cost_greedy", "route_cost_sticky", "route_cost_global"}
        ):
            assigned_dist, assigned_idx = masked.min(dim=2)
            assigned_idx = torch.where(
                torch.isfinite(assigned_dist),
                assigned_idx,
                torch.full_like(assigned_idx, -1),
            )
            return assigned_idx, assigned_dist

        if self.ugv_target_assignment_mode == "greedy_sticky":
            assigned_idx = self._ugv_sticky_target_indices(distances, targetable)
            assigned_idx_safe = assigned_idx.clamp(min=0)
            assigned_dist = torch.gather(distances, dim=2, index=assigned_idx_safe.unsqueeze(-1)).squeeze(-1)
            assigned_dist = torch.where(
                assigned_idx >= 0,
                assigned_dist,
                torch.full_like(assigned_dist, float("inf")),
            )
            return assigned_idx, assigned_dist

        if self.ugv_target_assignment_mode == "route_cost_greedy":
            route_costs_m = self._ugv_route_assignment_costs_m(
                ground_pos,
                survivor_pos,
                targetable,
            )
            scoreable = targetable & torch.isfinite(route_costs_m)
            assigned_idx, _assigned_score = self._ugv_greedy_assigned_target_indices(
                route_costs_m,
                scoreable,
            )
            assigned_idx_safe = assigned_idx.clamp(min=0)
            assigned_dist = torch.gather(distances, dim=2, index=assigned_idx_safe.unsqueeze(-1)).squeeze(-1)
            assigned_dist = torch.where(
                assigned_idx >= 0,
                assigned_dist,
                torch.full_like(assigned_dist, float("inf")),
            )
            return assigned_idx, assigned_dist

        if self.ugv_target_assignment_mode == "route_cost_sticky":
            route_costs_m = self._ugv_route_assignment_costs_m(
                ground_pos,
                survivor_pos,
                targetable,
            )
            assigned_idx = self._ugv_route_cost_sticky_target_indices(route_costs_m, targetable)
            assigned_idx_safe = assigned_idx.clamp(min=0)
            assigned_dist = torch.gather(distances, dim=2, index=assigned_idx_safe.unsqueeze(-1)).squeeze(-1)
            assigned_dist = torch.where(
                assigned_idx >= 0,
                assigned_dist,
                torch.full_like(assigned_dist, float("inf")),
            )
            return assigned_idx, assigned_dist

        if self.ugv_target_assignment_mode == "route_cost_global":
            route_costs_m = self._ugv_route_assignment_costs_m(
                ground_pos,
                survivor_pos,
                targetable,
            )
            assigned_idx = self._ugv_global_score_assignment_indices(route_costs_m, targetable)
            assigned_idx_safe = assigned_idx.clamp(min=0)
            assigned_dist = torch.gather(distances, dim=2, index=assigned_idx_safe.unsqueeze(-1)).squeeze(-1)
            assigned_dist = torch.where(
                assigned_idx >= 0,
                assigned_dist,
                torch.full_like(assigned_dist, float("inf")),
            )
            return assigned_idx, assigned_dist

        return self._ugv_greedy_assigned_target_indices(distances, targetable)

    def _ugv_greedy_assigned_target_indices(
        self,
        distances: Tensor,
        targetable: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_dim = distances.shape[0]
        device = distances.device
        assigned_idx = torch.full(
            (batch_dim, self.n_ground), -1, dtype=torch.long, device=device,
        )
        assigned_dist = torch.full((batch_dim, self.n_ground), float("inf"), device=device)
        masked = torch.where(targetable, distances, torch.full_like(distances, float("inf")))
        for env_index in range(batch_dim):
            best_per_ground = masked[env_index].min(dim=1).values
            order = torch.argsort(best_per_ground)
            used: set[int] = set()
            for ground_index_tensor in order:
                ground_index = int(ground_index_tensor.item())
                valid_targets = torch.nonzero(targetable[env_index, ground_index], as_tuple=False).flatten()
                if valid_targets.numel() == 0:
                    continue
                unused_targets = [
                    int(target.item())
                    for target in valid_targets
                    if int(target.item()) not in used
                ]
                candidate_targets = (
                    torch.tensor(unused_targets, dtype=torch.long, device=device)
                    if unused_targets
                    else valid_targets
                )
                candidate_distances = distances[env_index, ground_index, candidate_targets]
                selected_offset = int(torch.argmin(candidate_distances).item())
                selected = int(candidate_targets[selected_offset].item())
                assigned_idx[env_index, ground_index] = selected
                assigned_dist[env_index, ground_index] = distances[env_index, ground_index, selected]
                if unused_targets:
                    used.add(selected)
        return assigned_idx, assigned_dist

    def _ugv_sticky_target_indices(self, distances: Tensor, targetable: Tensor) -> Tensor:
        margin_by_env = (
            float(self.ugv_sticky_switch_margin_m)
            * self.terrain_sim_units_per_meter.to(device=distances.device)
        )
        return self._ugv_sticky_target_indices_from_scores(
            distances,
            targetable,
            margin_by_env=margin_by_env,
        )

    def _ugv_route_cost_sticky_target_indices(self, route_costs_m: Tensor, targetable: Tensor) -> Tensor:
        margin_by_env = torch.full(
            (route_costs_m.shape[0],),
            float(self.ugv_sticky_switch_margin_m),
            device=route_costs_m.device,
            dtype=route_costs_m.dtype,
        )
        return self._ugv_sticky_target_indices_from_scores(
            route_costs_m,
            targetable,
            margin_by_env=margin_by_env,
        )

    def _ugv_global_score_assignment_indices(self, scores: Tensor, targetable: Tensor) -> Tensor:
        """Exact min-cost one-to-one UGV-target assignment for each environment.

        The solver maximizes the number of assigned UGVs first, then minimizes
        total score. Extra UGVs remain unassigned when there are fewer useful
        targets than UGVs.
        """
        batch_dim = scores.shape[0]
        device = scores.device
        assigned = torch.full((batch_dim, self.n_ground), -1, dtype=torch.long, device=device)
        n_targets = int(targetable.shape[2]) if targetable.ndim == 3 else 0
        if self.n_ground == 0 or n_targets == 0:
            return assigned

        for env_index in range(batch_dim):
            states: dict[int, tuple[int, float, tuple[int, ...]]] = {0: (0, 0.0, tuple())}
            for ground_index in range(self.n_ground):
                next_states: dict[int, tuple[int, float, tuple[int, ...]]] = {}
                for mask, (count, cost, prefix) in states.items():
                    skip = (count, cost, prefix + (-1,))
                    self._ugv_assignment_keep_best(next_states, mask, skip)

                    scoreable = (
                        targetable[env_index, ground_index]
                        & torch.isfinite(scores[env_index, ground_index])
                    )
                    valid_targets = torch.nonzero(scoreable, as_tuple=False).flatten()
                    for target_tensor in valid_targets:
                        target_index = int(target_tensor.item())
                        target_bit = 1 << target_index
                        if mask & target_bit:
                            continue
                        candidate = (
                            count + 1,
                            cost + float(scores[env_index, ground_index, target_index].item()),
                            prefix + (target_index,),
                        )
                        self._ugv_assignment_keep_best(next_states, mask | target_bit, candidate)
                states = next_states

            best = max(
                states.values(),
                key=lambda state: (
                    state[0],
                    -state[1],
                    tuple(-value for value in state[2]),
                ),
            )
            assigned[env_index] = torch.tensor(best[2], dtype=torch.long, device=device)
        return assigned

    @staticmethod
    def _ugv_assignment_keep_best(
        states: dict[int, tuple[int, float, tuple[int, ...]]],
        mask: int,
        candidate: tuple[int, float, tuple[int, ...]],
    ) -> None:
        current = states.get(mask)
        if current is None:
            states[mask] = candidate
            return
        if candidate[0] > current[0]:
            states[mask] = candidate
            return
        if candidate[0] < current[0]:
            return
        if candidate[1] < current[1] - 1e-9:
            states[mask] = candidate
            return
        if abs(candidate[1] - current[1]) <= 1e-9 and candidate[2] < current[2]:
            states[mask] = candidate

    def _ugv_sticky_target_indices_from_scores(
        self,
        scores: Tensor,
        targetable: Tensor,
        *,
        margin_by_env: Tensor,
    ) -> Tensor:
        batch_dim = scores.shape[0]
        device = scores.device
        assigned = torch.full((batch_dim, self.n_ground), -1, dtype=torch.long, device=device)
        n_targets = int(targetable.shape[2]) if targetable.ndim == 3 else 0
        if self.n_ground == 0 or n_targets == 0:
            return assigned

        current_step = self.step_count.to(device=device)
        for env_index in range(batch_dim):
            if int(self.ugv_assignment_cache_step[env_index].item()) == int(current_step[env_index].item()):
                assigned[env_index] = self.ugv_assignment_cache_idx[env_index]
                continue

            previous = self.ugv_sticky_target_idx[env_index].clone()
            reserved_current: set[int] = set()
            for ground_index in range(self.n_ground):
                current = int(previous[ground_index].item())
                if (
                    current >= 0
                    and bool(targetable[env_index, ground_index, current].item())
                    and bool(torch.isfinite(scores[env_index, ground_index, current]).item())
                    and current not in reserved_current
                ):
                    reserved_current.add(current)

            used: set[int] = set()
            deferred: list[int] = []

            for ground_index in range(self.n_ground):
                current = int(previous[ground_index].item())
                current_valid = (
                    current >= 0
                    and bool(targetable[env_index, ground_index, current].item())
                    and bool(torch.isfinite(scores[env_index, ground_index, current]).item())
                    and current not in used
                )
                scoreable = targetable[env_index, ground_index] & torch.isfinite(scores[env_index, ground_index])
                valid_targets = torch.nonzero(scoreable, as_tuple=False).flatten()
                if valid_targets.numel() == 0:
                    continue
                unused_targets = [
                    int(target.item())
                    for target in valid_targets
                    if int(target.item()) not in used
                    and (int(target.item()) == current or int(target.item()) not in reserved_current)
                ]
                candidate_targets = (
                    torch.tensor(unused_targets, dtype=torch.long, device=device)
                    if unused_targets
                    else valid_targets
                )
                candidate_scores = scores[env_index, ground_index, candidate_targets]
                candidate = int(candidate_targets[int(torch.argmin(candidate_scores).item())].item())
                switch = not current_valid
                if current_valid and candidate != current:
                    current_score = float(scores[env_index, ground_index, current].item())
                    candidate_score = float(scores[env_index, ground_index, candidate].item())
                    margin = float(margin_by_env[env_index].item())
                    age = int(self.ugv_sticky_target_age[env_index, ground_index].item())
                    switch = (
                        age >= int(self.ugv_sticky_min_age_steps)
                        and candidate_score + margin < current_score * float(self.ugv_sticky_switch_ratio)
                    )
                if current_valid and not switch:
                    assigned[env_index, ground_index] = current
                    used.add(current)
                else:
                    deferred.append(ground_index)

            for ground_index in deferred:
                scoreable = targetable[env_index, ground_index] & torch.isfinite(scores[env_index, ground_index])
                valid_targets = torch.nonzero(scoreable, as_tuple=False).flatten()
                if valid_targets.numel() == 0:
                    continue
                unused_targets = [
                    int(target.item())
                    for target in valid_targets
                    if int(target.item()) not in used
                ]
                candidate_targets = (
                    torch.tensor(unused_targets, dtype=torch.long, device=device)
                    if unused_targets
                    else valid_targets
                )
                candidate_scores = scores[env_index, ground_index, candidate_targets]
                selected = int(candidate_targets[int(torch.argmin(candidate_scores).item())].item())
                assigned[env_index, ground_index] = selected
                if unused_targets:
                    used.add(selected)

            switched = (previous >= 0) & (assigned[env_index] >= 0) & (previous != assigned[env_index])
            self.metric_ugv_assignment_switches[env_index] += switched.float().sum()
            same = assigned[env_index] == previous
            self.ugv_sticky_target_age[env_index] = torch.where(
                (assigned[env_index] >= 0) & same,
                self.ugv_sticky_target_age[env_index] + 1,
                torch.zeros_like(self.ugv_sticky_target_age[env_index]),
            )
            self.ugv_sticky_target_idx[env_index] = assigned[env_index]
            self.ugv_assignment_cache_idx[env_index] = assigned[env_index]
            self.ugv_assignment_cache_step[env_index] = current_step[env_index]
        return assigned

    def _ugv_route_assignment_costs_m(
        self,
        ground_pos: Tensor,
        survivor_pos: Tensor,
        targetable: Tensor,
    ) -> Tensor:
        """Cost UGV-target assignments by cached terrain route cost instead of straight-line distance."""
        batch_dim, n_ground, _ = ground_pos.shape
        n_targets = int(targetable.shape[2]) if targetable.ndim == 3 else 0
        costs_m = torch.full(
            (batch_dim, n_ground, n_targets),
            float("inf"),
            device=ground_pos.device,
            dtype=ground_pos.dtype,
        )
        if n_ground == 0 or n_targets == 0:
            return costs_m
        if self.ugv_planner_hint != "global_astar":
            scale = self.terrain_sim_units_per_meter.to(device=ground_pos.device).view(-1, 1, 1).clamp_min(1e-9)
            distances = torch.linalg.norm(survivor_pos.unsqueeze(1) - ground_pos.unsqueeze(2), dim=-1)
            return torch.where(targetable, distances / scale, costs_m)

        G = int(self.fire_grid_size)
        cell_w_sim = 2.0 * float(self.x_semidim) / float(G)
        cell_h_sim = 2.0 * float(self.y_semidim) / float(G)
        bounds = (0, G - 1, 0, G - 1)
        for env_index in range(batch_dim):
            scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
            cell_m = max(cell_w_sim, cell_h_sim) / max(scale, 1e-9)
            traversable, movement_cost = self._ugv_static_planner_layer_arrays_for_env(env_index)
            traversable_tensor = self.traversable_grid[env_index]
            for target_index in range(n_targets):
                if not bool(targetable[env_index, :, target_index].any().item()):
                    continue
                target_pos = survivor_pos[env_index, target_index]
                cost_to_go = self._ugv_route_assignment_cost_grid_for_survivor(
                    env_index,
                    target_index,
                    target_pos,
                    traversable,
                    movement_cost,
                    traversable_tensor,
                    bounds,
                )
                if cost_to_go is None:
                    continue
                for ground_index in range(n_ground):
                    if not bool(targetable[env_index, ground_index, target_index].item()):
                        continue
                    start_cell = self._single_position_to_grid_cell(ground_pos[env_index, ground_index])
                    sx, sy = start_cell
                    if not (0 <= sx < G and 0 <= sy < G and bool(traversable[sy, sx])):
                        nearest_start = self._nearest_traversable_cell_in_bounds(
                            env_index,
                            sx,
                            sy,
                            bounds,
                            traversable=traversable_tensor,
                        )
                        if nearest_start is None:
                            continue
                        sx, sy = nearest_start
                    route_cost = float(cost_to_go[sy, sx])
                    if math.isfinite(route_cost):
                        costs_m[env_index, ground_index, target_index] = route_cost * cell_m
        return costs_m

    def _ugv_route_assignment_cost_grid_for_survivor(
        self,
        env_index: int,
        survivor_index: int,
        target_pos: Tensor,
        traversable: np.ndarray,
        movement_cost: np.ndarray,
        traversable_tensor: Tensor,
        bounds: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """Static route-cost grid for assigning UGVs to one survivor.

        This is the same cost-to-go grid previously requested inside every
        route-assignment call. Survivors and terrain are static within an
        episode, so caching the grid avoids repeatedly rebuilding goal cells and
        asking the reverse-Dijkstra cache for the same target.
        """
        cache = getattr(self, "_ugv_route_assignment_cost_grid_cache", None)
        if cache is None:
            self._ugv_route_assignment_cost_grid_cache = {}
            cache = self._ugv_route_assignment_cost_grid_cache
        version = int(getattr(self, "_ugv_static_planner_cache_version", 0))
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        confirm_radius_sim = float(self.detection_range_by_env[env_index].detach().cpu().item())
        if confirm_radius_sim <= 1e-9 and scale > 1e-9:
            confirm_radius_sim = float(self.ground_confirmation_range_m) * scale
        target_x = float(target_pos[X].detach().cpu().item())
        target_y = float(target_pos[Y].detach().cpu().item())
        key = (
            int(env_index),
            int(survivor_index),
            version,
            target_x,
            target_y,
            confirm_radius_sim,
        )
        if key in cache:
            return cache[key]

        goals = self._global_astar_goal_cells_for_env(
            env_index,
            target_pos,
            traversable,
            movement_cost,
        )
        if not goals:
            target_cell = self._single_position_to_grid_cell(target_pos)
            nearest_goal = self._nearest_traversable_cell_in_bounds(
                env_index,
                target_cell[0],
                target_cell[1],
                bounds,
                traversable=traversable_tensor,
            )
            if nearest_goal is None:
                cache[key] = None
                return None
            goals = [nearest_goal]
        cost_to_go = self._global_astar_static_raw_cost_to_go_for_env(env_index, goals)
        cache[key] = cost_to_go
        return cost_to_go

    def _ugv_duplicate_assignment_fraction(self, target_idx: Tensor, valid_target: Tensor) -> Tensor:
        if self.n_ground <= 1:
            return torch.zeros(target_idx.shape[0], device=target_idx.device)
        out = torch.zeros(target_idx.shape[0], device=target_idx.device)
        for env_index in range(target_idx.shape[0]):
            assigned = [
                int(idx.item())
                for idx, valid in zip(target_idx[env_index], valid_target[env_index])
                if bool(valid.item()) and int(idx.item()) >= 0
            ]
            if assigned:
                out[env_index] = float(len(assigned) - len(set(assigned))) / float(len(assigned))
        return out

    def _compute_step_rewards(self):
        device = self.fire_grid.device

        agent_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B, A, 2]
        surv_pos  = torch.stack([s.state.pos for s in self._survivors], dim=1)    # [B, S, 2]
        dists = torch.cdist(agent_pos, surv_pos)                                  # [B, A, S]

        drone_pos = agent_pos[:, :self.n_drones, :]
        drone_dists = dists[:, :self.n_drones, :]
        drone_seen = self._drone_survivor_detections(drone_dists, drone_pos, surv_pos)
        active_survivors = self._active_survivor_mask()
        drone_seen = drone_seen & active_survivors.unsqueeze(1)
        seen_by_drone       = drone_seen.any(dim=1)
        confirm_range = self.detection_range_by_env.view(-1, 1, 1)
        within_confirm      = dists < confirm_range
        self.step_drone_detections = drone_seen
        sim_units_per_meter_env = self.terrain_sim_units_per_meter.to(device).clamp_min(1e-9)
        meters_per_sim_env = 1.0 / sim_units_per_meter_env

        newly_scouted = (
            active_survivors
            & seen_by_drone
            & ~self.scouted_survivors
            & ~self.found_survivors
        )
        eligible_ground_confirmations = (
            within_confirm[:, self.n_drones:, :]
            & active_survivors.unsqueeze(1)
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

        previously_all_found = self._all_active_survivors_found()
        newly_found = (confirmed_by_ground | confirmed_by_drone) & ~self.found_survivors

        self.scouted_survivors = self.scouted_survivors | (newly_scouted & active_survivors)
        self.found_survivors   = self.found_survivors   | (newly_found & active_survivors)
        if bool((newly_scouted | newly_found).any().item()):
            self._invalidate_ugv_assignment_cache()
        all_survivors_found_now = (
            self._all_active_survivors_found()
            & ~previously_all_found
        )
        self._record_local_survivor_knowledge(drone_seen, eligible_ground_confirmations)

        # False-positive perception: drones may misclassify decoys as survivors;
        # UGVs can waste trips investigating them. This is a zero tensor when the
        # decoy scaffold is disabled.
        decoy_pursuit_penalty = self._process_decoy_false_positives(
            agent_pos,
            confirm_range,
            device,
        )

        # Dense potential-based shaping (Ng et al. 1999): α · (prev_dist − curr_dist)
        # Drones: target = unscouted survivors
        INF = float("inf")
        unscouted = active_survivors & ~self.scouted_survivors
        drone_d = torch.where(
            unscouted.unsqueeze(1),
            dists[:, :self.n_drones, :],
            torch.full_like(dists[:, :self.n_drones, :], INF),
        )
        curr_drone_dist, _ = drone_d.min(dim=2)
        if self.n_drones > 0 and self.n_survivors > 0:
            active_drone_dists = torch.where(
                active_survivors.unsqueeze(1),
                drone_dists,
                torch.full_like(drone_dists, INF),
            )
            nearest_drone_dist_sim = active_drone_dists.flatten(1).min(dim=1).values
            nearest_drone_dist_sim = torch.where(
                torch.isfinite(nearest_drone_dist_sim),
                nearest_drone_dist_sim,
                torch.zeros_like(nearest_drone_dist_sim),
            )
            drone_footprint_m = self._drone_camera_ranges() * meters_per_sim_env.view(-1, 1)
            target_within_footprint = (
                active_drone_dists <= self._drone_camera_ranges().unsqueeze(-1)
            ).any(dim=(1, 2))
            self.metric_uav_target_distance_m = nearest_drone_dist_sim * meters_per_sim_env
            self.metric_uav_footprint_radius_m = drone_footprint_m.mean(dim=1)
            self.metric_uav_footprint_radius_m_by_drone = drone_footprint_m
            self.metric_uav_target_within_footprint = target_within_footprint.float()
        else:
            self.metric_uav_target_distance_m = torch.zeros(self.world.batch_dim, device=device)
            self.metric_uav_footprint_radius_m = torch.zeros(self.world.batch_dim, device=device)
            self.metric_uav_footprint_radius_m_by_drone = torch.zeros(
                self.world.batch_dim, self.n_drones, device=device,
            )
            self.metric_uav_target_within_footprint = torch.zeros(self.world.batch_dim, device=device)
        all_scouted = self._all_active_survivors_scouted().view(-1, 1)
        prev_known = ~torch.isinf(self.prev_drone_dist) & ~all_scouted
        drone_shaping = torch.where(
            prev_known,
            (self.prev_drone_dist - curr_drone_dist) * self.r_drone_shaping,
            torch.zeros_like(curr_drone_dist),
        )
        self.prev_drone_dist = curr_drone_dist

        # Ground robots: target = locally known pending candidates. True survivors
        # are first in the target tensor; false-positive decoys follow them.
        # Progress is measured in meters and clipped to one nominal UGV step so
        # the shaping coefficient has a terrain-independent reward scale.
        unconfirmed_scouted = active_survivors & self.scouted_survivors & ~self.found_survivors
        ground_target_pos, known_ground_targets, ground_target_is_decoy = self._ugv_ground_target_candidates()
        ground_pos_for_assignment = agent_pos[:, self.n_drones:, :]
        curr_ground_target_idx, curr_ground_dist_sim = self._ugv_assigned_target_indices(
            ground_pos_for_assignment,
            ground_target_pos,
            known_ground_targets,
        )
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
        if self.n_ground > 0:
            ground_pos = agent_pos[:, self.n_drones:, :]
            target_idx_safe = curr_ground_target_idx.clamp(min=0)
            target_pos = torch.gather(
                ground_target_pos,
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
                planner_escape_mode,
                planner_required_progress_m,
                planner_shortfall_m,
                planner_remaining_distance_m,
            ) = self._ugv_planner_progress_rewards(
                self._pre_step_ground_pos,
                ground_pos,
                target_pos,
                curr_ground_target_idx,
                prev_known & outside_confirm_range,
            )
        else:
            planner_progress_reward = torch.zeros_like(curr_ground_dist_m)
            planner_progress_m = torch.zeros_like(curr_ground_dist_m)
            planner_progress_scaled = torch.zeros_like(curr_ground_dist_m)
            planner_active = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            planner_direct_blocked = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            planner_detour_needed = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            planner_escape_mode = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            planner_required_progress_m = torch.zeros_like(curr_ground_dist_m)
            planner_shortfall_m = torch.zeros_like(curr_ground_dist_m)
            planner_remaining_distance_m = torch.zeros_like(curr_ground_dist_m)
        if self.n_ground > 0:
            (
                escape_route_progress_m,
                escape_route_progress_scaled,
                escape_route_movement_alignment,
                escape_route_reward_active,
                escape_route_enter,
                escape_route_exit,
                escape_route_waypoint_distance_m,
                escape_route_path_index,
                escape_route_path_length,
            ) = self._ugv_escape_route_switch_rewards(
                self._pre_step_ground_pos,
                ground_pos,
                target_pos,
                curr_ground_target_idx,
                prev_known & outside_confirm_range,
                ground_progress_m,
                movement_alignment,
            )
        else:
            escape_route_progress_m = torch.zeros_like(curr_ground_dist_m)
            escape_route_progress_scaled = torch.zeros_like(curr_ground_dist_m)
            escape_route_movement_alignment = torch.zeros_like(curr_ground_dist_m)
            escape_route_reward_active = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            escape_route_enter = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            escape_route_exit = torch.zeros_like(curr_ground_dist_m, dtype=torch.bool)
            escape_route_waypoint_distance_m = torch.zeros_like(curr_ground_dist_m)
            escape_route_path_index = torch.zeros_like(curr_ground_dist_m)
            escape_route_path_length = torch.zeros_like(curr_ground_dist_m)
        progress_reward_basis = ground_progress_scaled
        movement_reward_basis = movement_alignment
        if self.ugv_dense_reward_mode == "positive_target":
            progress_reward_basis = ground_progress_scaled.clamp(min=0.0)
            movement_reward_basis = movement_alignment.clamp(min=0.0)
        elif self.ugv_dense_reward_mode in {"planner_blend", "escape_blend"}:
            if self.ugv_dense_reward_mode == "escape_blend":
                detour_blend_active = planner_active & planner_escape_mode
            else:
                detour_blend_active = planner_active & planner_detour_needed
            w = float(self.ugv_planner_blend_weight)
            target_progress = ground_progress_scaled.clamp(min=0.0)
            target_movement = movement_alignment.clamp(min=0.0)
            planner_progress = planner_progress_scaled.clamp(min=0.0)
            planner_movement = (
                planner_progress_m
                / self.step_ugv_actual_displacement_m.to(planner_progress_m.device).clamp_min(1e-6)
            ).clamp(min=0.0, max=1.0)
            progress_reward_basis = torch.where(
                detour_blend_active,
                (1.0 - w) * target_progress + w * planner_progress,
                target_progress,
            )
            movement_reward_basis = torch.where(
                detour_blend_active,
                (1.0 - w) * target_movement + w * planner_movement,
                target_movement,
            )
        elif self.ugv_dense_reward_mode == "escape_route_switch":
            progress_reward_basis = torch.where(
                escape_route_reward_active,
                escape_route_progress_scaled,
                ground_progress_scaled,
            )
            movement_reward_basis = torch.where(
                escape_route_reward_active,
                escape_route_movement_alignment,
                movement_alignment,
            )
            planner_progress_reward = torch.where(
                escape_route_reward_active,
                torch.zeros_like(planner_progress_reward),
                planner_progress_reward,
            )
        elif self.ugv_dense_reward_mode == "planner_follow":
            planner_movement = (
                planner_progress_m
                / self.step_ugv_actual_displacement_m.to(planner_progress_m.device).clamp_min(1e-6)
            ).clamp(min=-1.0, max=1.0)
            progress_reward_basis = torch.where(
                planner_active,
                planner_progress_scaled,
                torch.zeros_like(ground_progress_scaled),
            )
            movement_reward_basis = torch.where(
                planner_active,
                planner_movement,
                torch.zeros_like(movement_alignment),
            )
            planner_progress_reward = torch.zeros_like(planner_progress_reward)
        ground_shaping = torch.where(
            prev_known,
            progress_reward_basis * self.r_ground_shaping,
            torch.zeros_like(curr_ground_dist_m),
        )
        movement_alignment_reward = torch.where(
            prev_known,
            movement_reward_basis * self.r_ugv_movement_alignment,
            torch.zeros_like(curr_ground_dist_m),
        )
        route_aware_active = (
            planner_active & planner_detour_needed
            if self.ugv_route_aware_reward
            else torch.zeros_like(planner_active)
        )
        if self.ugv_route_aware_reward:
            ground_shaping = torch.where(
                route_aware_active,
                torch.zeros_like(ground_shaping),
                ground_shaping,
            )
            movement_alignment_reward = torch.where(
                route_aware_active,
                torch.zeros_like(movement_alignment_reward),
                movement_alignment_reward,
            )
        stalled_while_seeking = (
            prev_known
            & outside_confirm_range
            & (self.step_ugv_actual_displacement_m < self.ugv_stall_displacement_threshold_m)
        )
        ugv_stall_penalty = -self.r_ugv_stall_penalty * stalled_while_seeking.float()
        route_progress_floor_shortfall_m = torch.where(
            planner_active & prev_known & outside_confirm_range,
            (self.ugv_route_progress_floor_m - planner_progress_m).clamp(min=0.0),
            torch.zeros_like(planner_progress_m),
        )
        ugv_route_progress_floor_penalty = (
            -self.r_ugv_route_progress_floor_penalty
            * route_progress_floor_shortfall_m
        )
        ugv_route_progress_shortfall_penalty = (
            -self.r_ugv_route_progress_shortfall_penalty
            * planner_shortfall_m
        )
        if (
            self.n_ground > 0
            and self.n_survivors > 0
            and self.ground_approach_milestone_rewards_tensor.numel() > 0
        ):
            milestone_radii = self.ground_approach_milestone_radii_m_tensor.view(1, 1, -1)
            milestone_rewards = self.ground_approach_milestone_rewards_tensor.view(1, 1, -1)
            real_ground_target = valid_ground_target & (curr_ground_target_idx < self.n_survivors)
            target_idx_safe = curr_ground_target_idx.clamp(min=0, max=max(self.n_survivors - 1, 0))
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
                & real_ground_target.unsqueeze(-1)
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

        scout_credit_mask    = drone_seen & newly_scouted.unsqueeze(1)
        scout_per_drone      = scout_credit_mask.float().sum(dim=2)         # [B, D]

        drone_confirm_credit = self.step_drone_confirmations & newly_found.unsqueeze(1)
        confirm_per_drone    = drone_confirm_credit.float().sum(dim=2)      # [B, D]

        ground_within        = eligible_ground_confirmations
        confirm_credit_mask  = ground_within & newly_found.unsqueeze(1)
        confirm_per_ground   = confirm_credit_mask.float().sum(dim=2)       # [B, G]

        comms_up = self._team_reward_comms_up_mask(device)
        scout_actor_events = torch.zeros(
            self.world.batch_dim,
            self.n_agents,
            self.n_survivors,
            dtype=torch.bool,
            device=device,
        )
        if self.n_drones > 0:
            scout_actor_events[:, : self.n_drones, :] = scout_credit_mask
        confirm_actor_events = torch.zeros_like(scout_actor_events)
        if self.n_drones > 0:
            confirm_actor_events[:, : self.n_drones, :] = drone_confirm_credit
        if self.n_ground > 0:
            confirm_actor_events[:, self.n_drones :, :] = confirm_credit_mask

        team_scout_reward_by_agent = self._communication_gated_team_event_reward(
            newly_scouted,
            scout_actor_events,
            float(self.r_team_scout),
            comms_up,
        )
        team_confirm_reward_by_agent = self._communication_gated_team_event_reward(
            newly_found,
            confirm_actor_events,
            float(self.r_found_survivor),
            comms_up,
        )
        all_survivors_found_reward_by_agent = torch.zeros_like(team_confirm_reward_by_agent)
        if self.r_all_survivors_found != 0.0:
            all_found_recipients = (
                comms_up | confirm_actor_events.any(dim=2)
            ) & all_survivors_found_now.unsqueeze(1)
            all_survivors_found_reward_by_agent = (
                all_found_recipients.float() * float(self.r_all_survivors_found)
            )
        all_survivors_found_reward = all_survivors_found_reward_by_agent.mean(dim=1)
        team_scout_reward = team_scout_reward_by_agent.mean(dim=1)
        team_reward_by_agent = (
            team_scout_reward_by_agent
            + team_confirm_reward_by_agent
            + all_survivors_found_reward_by_agent
            + float(self.r_time_penalty)
        )

        # Per-step pressure from each UGV's local mission memory. A disconnected
        # UGV is only penalized for candidates it currently knows and has not
        # locally marked as resolved; reconnection updates that local memory via
        # the survivor-message synchronization path.
        if self.n_ground > 0:
            ground_slice = slice(self.n_drones, self.n_agents)
            local_pending_survivors = (
                self.known_survivors_by_agent[:, ground_slice, :]
                & ~self.confirmed_survivors_by_agent[:, ground_slice, :]
                & self._active_survivor_mask().unsqueeze(1)
            )
            if self.n_decoys > 0:
                local_pending_decoys = (
                    self.known_decoys_by_agent[:, ground_slice, :]
                    & ~self.dismissed_decoys.unsqueeze(1)
                    & self._active_decoy_mask().unsqueeze(1)
                )
                local_pending_targets = torch.cat(
                    (local_pending_survivors, local_pending_decoys),
                    dim=2,
                )
            else:
                local_pending_targets = local_pending_survivors
            n_pending_by_ground = local_pending_targets.float().sum(dim=2)
        else:
            n_pending_by_ground = torch.zeros(
                self.world.batch_dim,
                0,
                device=device,
            )
        pending_penalty_by_ground = n_pending_by_ground * float(self.r_pending_penalty)
        pending_penalty_reward = pending_penalty_by_ground.sum(dim=1)

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

        (
            uav_frontier_alignment_reward,
            uav_frontier_alignment,
            uav_frontier_progress_fraction,
            uav_frontier_uncovered_ratio,
        ) = self._uav_frontier_alignment_reward(drone_pos)
        previous_coverage_fraction = self.coverage_grid.float().mean(dim=(1, 2))
        uav_reward_coverage_map = self._uav_coverage_maps_for_reward()
        (
            coverage_new,
            uav_overlap_fraction,
            uav_outside_footprint_fraction,
            uav_inter_uav_overlap_fraction,
            uav_coverage_opportunity_fraction,
            uav_coverage_opportunity_cells,
            uav_coverage_opportunity_available_fraction,
        ) = self._coverage_reward(
            drone_pos,
            known_coverage=uav_reward_coverage_map,
        )  # [B, D]
        uav_confidence_probability = None
        uav_confidence_visible = None
        if self._uav_confidence_active():
            with torch.no_grad():
                uav_confidence_probability, uav_confidence_visible = self._uav_cell_detection_probability(drone_pos)
        uav_pre_confidence_by_drone = (
            self._uav_confidence_maps_for_reward().clone()
            if self._comms_maps_enabled() and self.n_drones > 0
            else None
        )
        uav_pre_confidence_grid = (
            uav_pre_confidence_by_drone
            if uav_pre_confidence_by_drone is not None
            else (
                self.uav_confidence_grid.clone()
                if (
                    self.r_uav_astar_progress > 0.0
                    or (
                        (
                            self.r_uav_confidence_overlap > 0.0
                            or self.r_uav_team_confidence_overlap > 0.0
                        )
                        and self.uav_confidence_overlap_mode != "raw"
                    )
                )
                else self.uav_confidence_grid
            )
        )
        if self.uav_confidence_overlap_mode == "raw":
            uav_confidence_overlap_penalty = self._uav_confidence_overlap_penalty(
                drone_pos,
                visible=uav_confidence_visible,
                previous=(
                    uav_pre_confidence_by_drone
                    if uav_pre_confidence_by_drone is not None
                    else self.uav_confidence_grid
                ),
            )
        else:
            uav_confidence_overlap_penalty = torch.zeros(
                self.world.batch_dim,
                self.n_drones,
                device=device,
            )
        uav_confidence_reward, uav_confidence_move_reward = self._update_uav_confidence(
            drone_pos,
            probability=uav_confidence_probability,
            visible=uav_confidence_visible,
            previous_by_drone=uav_pre_confidence_by_drone,
        )
        uav_team_confidence_reward = torch.zeros_like(uav_confidence_reward)
        if self.n_drones > 0 and self.r_uav_team_confidence > 0.0:
            team_confidence = uav_confidence_reward.mean(dim=1, keepdim=True)
            uav_team_confidence_reward = (
                team_confidence * float(self.r_uav_team_confidence)
            ).expand_as(uav_confidence_reward).clone()
        if self.uav_confidence_overlap_mode != "raw":
            uav_confidence_overlap_penalty = self._uav_confidence_overlap_penalty(
                drone_pos,
                visible=uav_confidence_visible,
                previous=uav_pre_confidence_grid,
            )
        uav_team_confidence_overlap_penalty = torch.zeros_like(uav_confidence_overlap_penalty)
        if self.n_drones > 0 and self.r_uav_team_confidence_overlap > 0.0:
            overlap_terms = (
                self.metric_uav_confidence_overlap_fraction_by_drone.to(
                    device=device,
                    dtype=uav_confidence_overlap_penalty.dtype,
                ).clamp(0.0, 1.0)
                * self.metric_uav_confidence_overlap_regret_by_drone.to(
                    device=device,
                    dtype=uav_confidence_overlap_penalty.dtype,
                ).clamp(0.0, 1.0)
            )
            team_overlap = -float(self.r_uav_team_confidence_overlap) * overlap_terms.mean(
                dim=1,
                keepdim=True,
            )
            uav_team_confidence_overlap_penalty = (
                team_overlap.expand_as(uav_confidence_overlap_penalty).clone()
            )
        self._update_uav_cleanup_target_step_metrics(drone_pos)
        (
            uav_cleanup_target_progress_reward,
            uav_cleanup_target_frontier_gate,
        ) = self._uav_cleanup_target_progress_reward(drone_pos)
        (
            uav_astar_progress_reward,
            uav_astar_frontier_gate,
            uav_astar_progress_fraction,
            uav_astar_path_cost_before,
            uav_astar_path_cost_after,
        ) = self._uav_astar_progress_reward(
            drone_pos,
            confidence_grid=uav_pre_confidence_grid,
        )
        current_coverage_fraction = self.coverage_grid.float().mean(dim=(1, 2))
        coverage_threshold_crossed = (
            (previous_coverage_fraction < self.uav_coverage_threshold_fraction)
            & (current_coverage_fraction >= self.uav_coverage_threshold_fraction)
        )
        uav_coverage_threshold_reward = (
            coverage_threshold_crossed.float() * self.r_uav_coverage_threshold
        )
        team_reward_by_agent = team_reward_by_agent + uav_coverage_threshold_reward.unsqueeze(1)
        (
            uav_move_coverage_reward,
            drone_displacement_m,
            coverage_new_cells,
        ) = self._uav_move_coverage_reward(
            drone_pos,
            coverage_new,
            uav_coverage_opportunity_fraction,
        )
        uav_inefficient_move_penalty = self._uav_inefficient_move_penalty(
            drone_displacement_m,
            uav_coverage_opportunity_fraction,
            self.metric_uav_confidence_opportunity_fraction_by_drone,
        )
        uav_coverage_reward = self._uav_coverage_reward(
            coverage_new,
            uav_coverage_opportunity_fraction,
        )
        uav_expected_overlap_fraction = self._uav_expected_overlap_fraction(drone_displacement_m)
        uav_excess_overlap_fraction = (
            uav_overlap_fraction - uav_expected_overlap_fraction
        ).clamp(min=0.0)
        uav_overlap_penalty = self._uav_overlap_penalty(
            uav_overlap_fraction,
            uav_expected_overlap_fraction,
            uav_coverage_opportunity_available_fraction,
        )
        uav_inter_uav_overlap_penalty = self._uav_inter_uav_overlap_penalty(
            uav_inter_uav_overlap_fraction,
        )
        uav_outside_footprint_penalty = self._uav_outside_footprint_penalty(
            uav_outside_footprint_fraction,
        )
        boundary_soft_risk, boundary_distance_m = self._uav_boundary_risk_metrics(drone_pos)

        self.metric_new_scouts = (newly_scouted & active_survivors).float().sum(dim=1)
        self.metric_new_confirmations = (newly_found & active_survivors).float().sum(dim=1)
        self.metric_full_success = self._all_active_survivors_found().float()
        self.metric_reward_team = team_reward_by_agent.mean(dim=1)
        self.metric_reward_all_survivors_found = all_survivors_found_reward
        self.metric_reward_team_scout = team_scout_reward
        self.metric_reward_pending_penalty = pending_penalty_reward
        self.metric_reward_drone_scout = (scout_per_drone * self.r_drone_scout).sum(dim=1)
        self.metric_reward_drone_progress = drone_shaping.sum(dim=1)
        self.metric_reward_uav_move_coverage = uav_move_coverage_reward.sum(dim=1)
        self.metric_reward_uav_inefficient_move = uav_inefficient_move_penalty.sum(dim=1)
        self.metric_reward_uav_inefficient_move_by_drone = uav_inefficient_move_penalty
        self.metric_reward_uav_coverage_threshold = uav_coverage_threshold_reward
        self.metric_reward_uav_frontier_alignment = uav_frontier_alignment_reward.sum(dim=1)
        self.metric_reward_uav_confidence = uav_confidence_reward.sum(dim=1)
        self.metric_reward_uav_confidence_by_drone = uav_confidence_reward
        self.metric_reward_uav_team_confidence = uav_team_confidence_reward.sum(dim=1)
        self.metric_reward_uav_team_confidence_by_drone = uav_team_confidence_reward
        self.metric_reward_uav_team_confidence_overlap = uav_team_confidence_overlap_penalty.sum(dim=1)
        self.metric_reward_uav_team_confidence_overlap_by_drone = uav_team_confidence_overlap_penalty
        self.metric_reward_uav_confidence_move = uav_confidence_move_reward.sum(dim=1)
        self.metric_reward_uav_confidence_move_by_drone = uav_confidence_move_reward
        self.metric_reward_uav_confidence_overlap = uav_confidence_overlap_penalty.sum(dim=1)
        self.metric_reward_uav_confidence_overlap_by_drone = uav_confidence_overlap_penalty
        self.metric_reward_uav_cleanup_target_progress = uav_cleanup_target_progress_reward.sum(dim=1)
        self.metric_reward_uav_cleanup_target_progress_by_drone = uav_cleanup_target_progress_reward
        self.metric_reward_uav_astar_progress = uav_astar_progress_reward.sum(dim=1)
        self.metric_reward_uav_astar_progress_by_drone = uav_astar_progress_reward
        self.metric_uav_astar_progress_fraction_by_drone = uav_astar_progress_fraction
        self.metric_uav_astar_progress_fraction = (
            uav_astar_progress_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_astar_frontier_gate_by_drone = uav_astar_frontier_gate
        self.metric_uav_astar_frontier_gate = (
            uav_astar_frontier_gate.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_astar_path_cost_before_by_drone = uav_astar_path_cost_before
        self.metric_uav_astar_path_cost_after_by_drone = uav_astar_path_cost_after
        self.metric_uav_cleanup_target_frontier_gate_by_drone = uav_cleanup_target_frontier_gate
        self.metric_uav_cleanup_target_frontier_gate = (
            uav_cleanup_target_frontier_gate.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_frontier_alignment_by_drone = uav_frontier_alignment
        self.metric_uav_frontier_progress_fraction_by_drone = uav_frontier_progress_fraction
        self.metric_uav_frontier_uncovered_ratio_by_drone = uav_frontier_uncovered_ratio
        self.metric_uav_frontier_alignment = (
            uav_frontier_alignment.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_frontier_progress_fraction = (
            uav_frontier_progress_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_frontier_uncovered_ratio = (
            uav_frontier_uncovered_ratio.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_reward_uav_overlap = uav_overlap_penalty.sum(dim=1)
        self.metric_uav_overlap_fraction_by_drone = uav_overlap_fraction
        self.metric_uav_expected_overlap_fraction_by_drone = uav_expected_overlap_fraction
        self.metric_uav_excess_overlap_fraction_by_drone = uav_excess_overlap_fraction
        self.metric_uav_overlap_fraction = (
            uav_overlap_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_reward_uav_inter_uav_overlap = uav_inter_uav_overlap_penalty.sum(dim=1)
        self.metric_uav_inter_uav_overlap_fraction_by_drone = uav_inter_uav_overlap_fraction
        self.metric_uav_inter_uav_overlap_fraction = (
            uav_inter_uav_overlap_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_reward_uav_outside_footprint = uav_outside_footprint_penalty.sum(dim=1)
        self.metric_uav_outside_footprint_fraction_by_drone = uav_outside_footprint_fraction
        self.metric_uav_outside_footprint_fraction = (
            uav_outside_footprint_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_boundary_soft_risk = boundary_soft_risk.sum(dim=1)
        self.metric_uav_boundary_distance_m_by_drone = boundary_distance_m
        self.metric_uav_boundary_distance_m = (
            boundary_distance_m.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_displacement_m_by_drone = drone_displacement_m
        self.metric_uav_new_coverage_cells_by_drone = coverage_new_cells
        self.metric_uav_coverage_opportunity_cells_by_drone = uav_coverage_opportunity_cells
        self.metric_uav_coverage_opportunity_fraction_by_drone = uav_coverage_opportunity_fraction
        self.metric_uav_coverage_opportunity_available_fraction_by_drone = (
            uav_coverage_opportunity_available_fraction
        )
        self.metric_uav_displacement_m = drone_displacement_m.sum(dim=1)
        self.metric_uav_new_coverage_cells = coverage_new_cells.sum(dim=1)
        self.metric_uav_coverage_opportunity_cells = uav_coverage_opportunity_cells.sum(dim=1)
        self.metric_uav_coverage_opportunity_fraction = (
            uav_coverage_opportunity_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_uav_coverage_opportunity_available_fraction = (
            uav_coverage_opportunity_available_fraction.mean(dim=1)
            if self.n_drones > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_reward_ugv_progress = ground_shaping.sum(dim=1)
        self.metric_reward_ugv_approach = ground_approach.sum(dim=1)
        self.metric_reward_ugv_movement_alignment = movement_alignment_reward.sum(dim=1)
        self.metric_reward_ugv_planner_progress = planner_progress_reward.sum(dim=1)
        self.metric_reward_ugv_stall_penalty = ugv_stall_penalty.sum(dim=1)
        self.metric_reward_ugv_route_progress_floor_penalty = (
            ugv_route_progress_floor_penalty.sum(dim=1)
        )
        self.metric_ugv_route_progress_floor_shortfall_m = (
            route_progress_floor_shortfall_m.sum(dim=1)
        )
        self.metric_reward_ugv_route_progress_shortfall_penalty = (
            ugv_route_progress_shortfall_penalty.sum(dim=1)
        )
        self.metric_ugv_route_progress_required_m = planner_required_progress_m.sum(dim=1)
        self.metric_ugv_route_progress_shortfall_m = planner_shortfall_m.sum(dim=1)
        self.metric_ugv_route_remaining_distance_m = planner_remaining_distance_m.sum(dim=1)
        self.metric_reward_ground_confirm = (
            confirm_per_ground * self.r_ground_confirm
        ).sum(dim=1)
        self.metric_reward_coverage = uav_coverage_reward.sum(dim=1)
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
            self.metric_ugv_duplicate_assignment_fraction = self._ugv_duplicate_assignment_fraction(
                curr_ground_target_idx,
                valid_ground_target,
            )
        else:
            self.metric_ugv_target_index = torch.full((self.world.batch_dim,), -1.0, device=device)
            self.metric_ugv_duplicate_assignment_fraction = torch.zeros(
                self.world.batch_dim, device=device,
            )
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
        self.metric_ugv_planner_escape_mode = planner_escape_mode.float().sum(dim=1)
        self.metric_ugv_escape_route_active = (
            self.ugv_escape_route_active.float().sum(dim=1)
            if self.n_ground > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_ugv_escape_route_enter = escape_route_enter.float().sum(dim=1)
        self.metric_ugv_escape_route_exit = escape_route_exit.float().sum(dim=1)
        self.metric_ugv_escape_route_stall_counter = (
            self.ugv_escape_route_stall_counter.float().sum(dim=1)
            if self.n_ground > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_ugv_escape_route_age = (
            self.ugv_escape_route_age.float().sum(dim=1)
            if self.n_ground > 0
            else torch.zeros(self.world.batch_dim, device=device)
        )
        self.metric_ugv_escape_route_waypoint_progress_m = torch.where(
            escape_route_reward_active,
            escape_route_progress_m,
            torch.zeros_like(escape_route_progress_m),
        ).sum(dim=1)
        self.metric_ugv_escape_route_waypoint_progress_scaled = torch.where(
            escape_route_reward_active,
            escape_route_progress_scaled,
            torch.zeros_like(escape_route_progress_scaled),
        ).sum(dim=1)
        self.metric_ugv_escape_route_waypoint_distance_m = torch.where(
            self.ugv_escape_route_active,
            escape_route_waypoint_distance_m,
            torch.zeros_like(escape_route_waypoint_distance_m),
        ).sum(dim=1)
        self.metric_ugv_escape_route_path_index = torch.where(
            self.ugv_escape_route_active,
            escape_route_path_index,
            torch.zeros_like(escape_route_path_index),
        ).sum(dim=1)
        self.metric_ugv_escape_route_path_length = torch.where(
            self.ugv_escape_route_active,
            escape_route_path_length,
            torch.zeros_like(escape_route_path_length),
        ).sum(dim=1)
        self.metric_ugv_route_aware_active = route_aware_active.float().sum(dim=1)
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
            r = team_reward_by_agent[:, i].clone()
            if agent.is_drone:
                r = r + scout_per_drone[:, i] * self.r_drone_scout
                r = r + confirm_per_drone[:, i] * self.r_drone_confirm
                r = r - self.drone_energy_cost[:, i]
                r = r + self.step_drone_climb[:, i] * self.r_drone_climb_cost
                r = r + drone_shaping[:, i]
                r = r + uav_coverage_reward[:, i]
                r = r + uav_move_coverage_reward[:, i]
                r = r + uav_inefficient_move_penalty[:, i]
                r = r + uav_frontier_alignment_reward[:, i]
                r = r + uav_confidence_reward[:, i]
                r = r + uav_team_confidence_reward[:, i]
                r = r + uav_team_confidence_overlap_penalty[:, i]
                r = r + uav_confidence_move_reward[:, i]
                r = r + uav_confidence_overlap_penalty[:, i]
                r = r + uav_cleanup_target_progress_reward[:, i]
                r = r + uav_astar_progress_reward[:, i]
                r = r + uav_overlap_penalty[:, i]
                r = r + uav_inter_uav_overlap_penalty[:, i]
                r = r + uav_outside_footprint_penalty[:, i]
            else:
                g = i - self.n_drones
                r = r + confirm_per_ground[:, g] * self.r_ground_confirm
                r = r + ground_in_fire[:, g].float() * self.r_fire_penalty
                r = r + self.step_ugv_travel_cost[:, g] * self.r_ground_travel_cost
                r = r + ground_shaping[:, g]
                r = r + ground_approach[:, g]
                r = r + pending_penalty_by_ground[:, g]
                r = r + ground_cov_new[:, g] * self.r_ground_coverage
                r = r + movement_alignment_reward[:, g]
                r = r + planner_progress_reward[:, g]
                r = r + ugv_stall_penalty[:, g]
                r = r + ugv_route_progress_floor_penalty[:, g]
                r = r + ugv_route_progress_shortfall_penalty[:, g]
                r = r + decoy_pursuit_penalty[:, g]
            agent.scenario_reward = r

    def _drone_survivor_detections(
        self,
        drone_dists: Tensor,
        drone_pos: Tensor,
        surv_pos: Tensor,
    ) -> Tensor:
        """Drone scouting, using either abstract or opt-in CV perception."""
        if self.detection_backend == "cv":
            return self._drone_survivor_detections_cv(drone_pos, surv_pos)
        components = self._drone_detection_components(drone_dists, drone_pos, surv_pos)
        probability = components["probability"]
        return torch.rand_like(probability) < probability

    def _process_decoy_false_positives(
        self,
        agent_pos: Tensor,
        confirm_range: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Drive the optional false-positive perception model over decoys."""
        n_ground = self.n_ground
        if self.n_decoys == 0:
            if hasattr(self, "metric_false_positive_detections"):
                self.metric_false_positive_detections.zero_()
            if hasattr(self, "metric_false_positive_trips"):
                self.metric_false_positive_trips.zero_()
            return torch.zeros(self.world.batch_dim, max(n_ground, 0), device=device)

        decoy_pos = torch.stack([d.state.pos for d in self._decoys], dim=1)
        agent_decoy_dists = torch.cdist(agent_pos, decoy_pos)
        active_decoys = self._active_decoy_mask()

        if self.n_drones > 0 and self.drone_false_positive_rate > 0.0:
            drone_decoy_dists = agent_decoy_dists[:, :self.n_drones, :]
            footprint = self._drone_camera_ranges().unsqueeze(-1)
            in_view = drone_decoy_dists <= footprint
            draw = torch.rand_like(drone_decoy_dists)
            false_det = in_view & (draw < self.drone_false_positive_rate)
            false_det = false_det & active_decoys.unsqueeze(1) & ~self.dismissed_decoys.unsqueeze(1)
            self.step_decoy_false_detections = false_det
            newly_scouted_decoys = false_det.any(dim=1) & active_decoys & ~self.dismissed_decoys
            self.scouted_decoys = self.scouted_decoys | newly_scouted_decoys
            self.known_decoys_by_agent[:, :self.n_drones] |= false_det
        else:
            self.step_decoy_false_detections = torch.zeros(
                self.world.batch_dim,
                self.n_drones,
                self.n_decoys,
                dtype=torch.bool,
                device=device,
            )

        decoy_penalty = torch.zeros(self.world.batch_dim, max(n_ground, 0), device=device)
        newly_dismissed = torch.zeros_like(self.dismissed_decoys)
        if n_ground > 0:
            ground_decoy_dists = agent_decoy_dists[:, self.n_drones:, :]
            within = ground_decoy_dists < confirm_range
            investigatable = (
                within
                & active_decoys.unsqueeze(1)
                & self.scouted_decoys.unsqueeze(1)
                & ~self.dismissed_decoys.unsqueeze(1)
            )
            self.known_decoys_by_agent[:, self.n_drones:] |= investigatable
            newly_dismissed = investigatable.any(dim=1) & active_decoys & ~self.dismissed_decoys
            self.dismissed_decoys = self.dismissed_decoys | newly_dismissed
            if bool(newly_dismissed.any().item()):
                self._invalidate_ugv_assignment_cache()
            trips_per_ground = investigatable.float().sum(dim=2)
            decoy_penalty = trips_per_ground * self.r_decoy_pursuit_penalty

        self.metric_false_positive_detections = (self.scouted_decoys & active_decoys).float().sum(dim=1)
        self.metric_false_positive_trips = newly_dismissed.float().sum(dim=1)
        return decoy_penalty

    def _init_cv_adapter(self) -> None:
        """Lazily build the SimulationCvAdapter for real CV-based detection."""
        from detection.simulation_adapter import SimulationCvAdapter

        kwargs = {
            "terrain_cache_path": self.terrain_cache_path or "data/terrain_cache/malibu_128.npz",
            "image_size": self.cv_image_size,
            "detector_backend": "yolo",
            "person_conf": self.cv_conf_threshold,
        }
        if self.cv_person_model:
            kwargs["person_model"] = self.cv_person_model
        self._cv_adapter = SimulationCvAdapter(**kwargs)

    def _drone_survivor_detections_cv(
        self,
        drone_pos: Tensor,
        surv_pos: Tensor,
    ) -> Tensor:
        """Run YOLOv8 detection per drone and return a [B, D, S] boolean tensor."""
        if self._cv_adapter is None:
            self._init_cv_adapter()

        from detection.simulation_adapter import SimDrone, SimEntity, SimWildfireState

        device = drone_pos.device
        batch_dim = self.world.batch_dim
        n_drones = self.n_drones
        n_survivors = self.n_survivors
        result = torch.zeros(batch_dim, n_drones, n_survivors, dtype=torch.bool, device=device)
        self.step_cv_false_positives = 0

        for env_i in range(batch_dim):
            survivors = [
                SimEntity(
                    index=s,
                    world_xy=(float(surv_pos[env_i, s, X]), float(surv_pos[env_i, s, Y])),
                )
                for s in range(n_survivors)
            ]
            wildfire_state = None
            if hasattr(self, "fire_grid") and self.fire_grid is not None:
                wildfire_state = SimWildfireState(
                    fire_grid=self.fire_grid[env_i].cpu().numpy(),
                    fire_intensity_grid=self.fire_intensity_grid[env_i].cpu().numpy(),
                    burned_grid=self.burned_grid[env_i].cpu().numpy(),
                    smoke_grid=self.smoke_grid[env_i].cpu().numpy() if self.smoke_grid is not None else None,
                )
            for drone_i in range(n_drones):
                drone_agent = self.world.agents[drone_i]
                drone = SimDrone(
                    index=drone_i,
                    name=drone_agent.name,
                    world_xy=(float(drone_pos[env_i, drone_i, X]), float(drone_pos[env_i, drone_i, Y])),
                    altitude_agl=float(self.drone_altitude[env_i, drone_i]),
                )
                det_result = self._cv_adapter.render_and_detect(
                    drone=drone,
                    survivors=survivors,
                    wildfire_state=wildfire_state,
                )
                for det in det_result.get("detections", []):
                    matched_idx = det.get("matched_survivor_index")
                    if matched_idx is not None and 0 <= matched_idx < n_survivors:
                        result[env_i, drone_i, matched_idx] = True
                    else:
                        self.step_cv_false_positives += 1

        return result

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

    def _device_cache_key(self, device: torch.device, dtype: torch.dtype) -> tuple:
        return (device.type, device.index, str(dtype))

    def _invalidate_uav_frontier_feature_cache(self) -> None:
        if hasattr(self, "_uav_frontier_feature_cache"):
            self._uav_frontier_feature_cache.clear()

    def _invalidate_uav_local_confidence_obs_cache(self) -> None:
        if hasattr(self, "_uav_local_confidence_obs_cache"):
            self._uav_local_confidence_obs_cache.clear()

    def _invalidate_uav_runtime_caches(self) -> None:
        self._invalidate_uav_frontier_feature_cache()
        self._invalidate_uav_local_confidence_obs_cache()

    def _comms_maps_enabled(self) -> bool:
        return getattr(self, "comms_map_mode", "global") == "per_agent"

    def _agent_index(self, agent: Agent) -> int:
        try:
            return self.world.agents.index(agent)
        except ValueError:
            return 0

    def _sync_comm_agent_maps_for_observation(self, agent: Agent, comms_keep: Tensor) -> None:
        """Synchronize private map memories according to the active comm state.

        Rewards and diagnostics still use the global maps. This only controls
        which coverage/confidence memories are visible inside observations.
        """
        if not self._comms_maps_enabled() or self.n_agents <= 0:
            return
        if self.comms_dropout <= 0.0 or self.comms_dropout_mode == "bursty":
            current_step = self.step_count.to(
                device=self.comm_map_last_sync_step.device,
                dtype=self.comm_map_last_sync_step.dtype,
            )
            due = current_step != self.comm_map_last_sync_step
            if not bool(due.any().item()):
                return
            if self.comms_dropout <= 0.0:
                connected = due.view(self.world.batch_dim, 1).expand(-1, self.n_agents)
            else:
                self._advance_bursty_comms_dropout()
                connected = (self.comms_dropout_remaining_steps <= 0) & due.view(
                    self.world.batch_dim,
                    1,
                )
            if not bool(connected.any().item()):
                self.comm_map_last_sync_step[due] = current_step[due]
                return
            connected_4d = connected.view(self.world.batch_dim, self.n_agents, 1, 1)
            connected_coverage = self.comm_agent_coverage_grid & connected_4d
            merged_coverage = self.comm_team_coverage_grid | connected_coverage.any(dim=1)
            connected_confidence = torch.where(
                connected_4d,
                self.comm_agent_confidence_grid,
                torch.zeros_like(self.comm_agent_confidence_grid),
            )
            merged_confidence = torch.maximum(
                self.comm_team_confidence_grid,
                connected_confidence.amax(dim=1),
            )
            self.comm_team_coverage_grid.copy_(merged_coverage)
            self.comm_team_confidence_grid.copy_(merged_confidence)
            self.comm_agent_coverage_grid.copy_(torch.where(
                connected_4d,
                self.comm_team_coverage_grid.unsqueeze(1),
                self.comm_agent_coverage_grid,
            ))
            self.comm_agent_confidence_grid.copy_(torch.where(
                connected_4d,
                self.comm_team_confidence_grid.unsqueeze(1),
                self.comm_agent_confidence_grid,
            ))
            self.comm_map_last_sync_step[due] = current_step[due]
            return

        agent_idx = self._agent_index(agent)
        keep = comms_keep[:, 0].to(device=self.comm_agent_coverage_grid.device, dtype=torch.bool)
        if not bool(keep.any().item()):
            return
        self.comm_team_coverage_grid[keep] |= self.comm_agent_coverage_grid[keep, agent_idx]
        self.comm_team_confidence_grid[keep] = torch.maximum(
            self.comm_team_confidence_grid[keep],
            self.comm_agent_confidence_grid[keep, agent_idx],
        )
        self.comm_agent_coverage_grid[keep, agent_idx] = self.comm_team_coverage_grid[keep]
        self.comm_agent_confidence_grid[keep, agent_idx] = self.comm_team_confidence_grid[keep]

    def _coverage_grid_for_observation(self, agent: Agent | None = None) -> Tensor:
        if agent is None or not self._comms_maps_enabled():
            return self.coverage_grid.float()
        agent_idx = min(max(self._agent_index(agent), 0), self.n_agents - 1)
        return self.comm_agent_coverage_grid[:, agent_idx].float()

    def _confidence_grid_for_observation(self, agent: Agent | None = None) -> Tensor:
        if agent is None or not self._comms_maps_enabled():
            return self.uav_confidence_grid.float()
        agent_idx = min(max(self._agent_index(agent), 0), self.n_agents - 1)
        return self.comm_agent_confidence_grid[:, agent_idx].float()

    def _drone_coverage_grid_for_observation(self, drone_idx: int) -> Tensor:
        if not self._comms_maps_enabled():
            return self.coverage_grid.float()
        agent_idx = min(max(int(drone_idx), 0), self.n_agents - 1)
        return self.comm_agent_coverage_grid[:, agent_idx].float()

    def _drone_confidence_grid_for_observation(self, drone_idx: int) -> Tensor:
        if not self._comms_maps_enabled():
            return self.uav_confidence_grid.float()
        agent_idx = min(max(int(drone_idx), 0), self.n_agents - 1)
        return self.comm_agent_confidence_grid[:, agent_idx].float()

    def _uav_coverage_maps_for_reward(self) -> Tensor:
        if self.n_drones <= 0:
            return self.coverage_grid.new_zeros(
                self.world.batch_dim,
                0,
                self.coverage_grid.shape[-2],
                self.coverage_grid.shape[-1],
            )
        if self._comms_maps_enabled():
            return self.comm_agent_coverage_grid[:, : self.n_drones]
        return self.coverage_grid.unsqueeze(1).expand(-1, self.n_drones, -1, -1)

    def _uav_confidence_maps_for_reward(self) -> Tensor:
        if self.n_drones <= 0:
            return self.uav_confidence_grid.new_zeros(
                self.world.batch_dim,
                0,
                self.uav_confidence_grid.shape[-2],
                self.uav_confidence_grid.shape[-1],
            )
        if self._comms_maps_enabled():
            return self.comm_agent_confidence_grid[:, : self.n_drones]
        return self.uav_confidence_grid.unsqueeze(1).expand(-1, self.n_drones, -1, -1)

    def _update_comm_agent_coverage_from_claims(self, claims: Tensor) -> None:
        if not self._comms_maps_enabled() or self.n_drones <= 0:
            return
        count = min(self.n_drones, self.n_agents, claims.shape[1])
        if count <= 0:
            return
        self.comm_agent_coverage_grid[:, :count] |= claims[:, :count].to(
            device=self.comm_agent_coverage_grid.device,
            dtype=torch.bool,
        )
        if hasattr(self, "comm_map_last_sync_step"):
            self.comm_map_last_sync_step.fill_(-1)

    def _update_comm_agent_confidence_from_probability(self, probability: Tensor) -> None:
        if not self._comms_maps_enabled() or self.n_drones <= 0:
            return
        count = min(self.n_drones, self.n_agents, probability.shape[1])
        if count <= 0:
            return
        current = self.comm_agent_confidence_grid[:, :count]
        probability = probability[:, :count].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        updated = 1.0 - (1.0 - current.clamp(0.0, 1.0)) * (1.0 - probability)
        current.copy_(updated)
        if hasattr(self, "comm_map_last_sync_step"):
            self.comm_map_last_sync_step.fill_(-1)

    def _invalidate_uav_terrain_caches(self) -> None:
        if hasattr(self, "_uav_land_cover_factor_cache"):
            self._uav_land_cover_factor_cache.clear()
        self._uav_terrain_cache_version = getattr(self, "_uav_terrain_cache_version", 0) + 1
        self._invalidate_uav_runtime_caches()

    def _invalidate_ugv_planner_route_cache(
        self,
        env_index: int | None = None,
        *,
        terrain_changed: bool = False,
        fire_changed: bool = False,
    ) -> None:
        if not hasattr(self, "_ugv_planner_route_cache"):
            self._ugv_planner_route_cache = {}
        if terrain_changed:
            self._invalidate_ugv_planner_layer_cache(env_index)
            if not fire_changed:
                self._invalidate_ugv_global_heuristic_cache(env_index)
            self._ugv_planner_terrain_cache_version = (
                getattr(self, "_ugv_planner_terrain_cache_version", 0) + 1
            )
            if hasattr(self, "ugv_escape_route_active"):
                self._reset_ugv_escape_routes(env_index)
            if hasattr(self, "ugv_global_route_target_idx"):
                self._reset_ugv_global_routes(env_index)
            if fire_changed and hasattr(self, "ugv_global_route_fire_replan_pending"):
                if env_index is None:
                    self.ugv_global_route_fire_replan_pending.fill_(True)
                else:
                    self.ugv_global_route_fire_replan_pending[int(env_index)] = True
        if env_index is None:
            self._ugv_planner_route_cache.clear()
            return
        env_index = int(env_index)
        self._ugv_planner_route_cache = {
            key: value
            for key, value in self._ugv_planner_route_cache.items()
            if key[0] != env_index
        }

    def _uav_grid_geometry(
        self,
        device: torch.device,
        dtype: torch.dtype,
        grid_size: int | None = None,
    ) -> tuple:
        if not hasattr(self, "_uav_grid_geometry_cache"):
            self._uav_grid_geometry_cache = {}
        G = int(self.fire_grid_size if grid_size is None else grid_size)
        if G < 2:
            raise ValueError("UAV grid size must be at least 2")
        key = self._device_cache_key(device, dtype) + (G, float(self.x_semidim), float(self.y_semidim))
        cached = self._uav_grid_geometry_cache.get(key)
        if cached is not None:
            return cached

        cell_width = 2.0 * self.x_semidim / G
        cell_height = 2.0 * self.y_semidim / G
        xs = torch.linspace(
            -self.x_semidim + cell_width / 2.0,
            self.x_semidim - cell_width / 2.0,
            G,
            device=device,
            dtype=dtype,
        )
        ys = torch.linspace(
            -self.y_semidim + cell_height / 2.0,
            self.y_semidim - cell_height / 2.0,
            G,
            device=device,
            dtype=dtype,
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cell_pos = torch.stack((grid_x, grid_y), dim=-1).reshape(1, G * G, 2)
        cached = (
            xs,
            ys,
            xs.view(1, G),
            ys.view(G, 1),
            cell_pos,
            float(cell_width),
            float(cell_height),
            max(float(cell_width * cell_height), 1e-12),
        )
        self._uav_grid_geometry_cache[key] = cached
        return cached

    def _uav_sector_geometry(self, sectors: int, device: torch.device, dtype: torch.dtype) -> tuple:
        if not hasattr(self, "_uav_sector_geometry_cache"):
            self._uav_sector_geometry_cache = {}
        key = self._device_cache_key(device, dtype) + (int(sectors),)
        cached = self._uav_sector_geometry_cache.get(key)
        if cached is not None:
            return cached

        sector_width = 2.0 * math.pi / float(sectors)
        sector_angles = (
            -math.pi
            + (torch.arange(sectors, device=device, dtype=dtype) + 0.5)
            * sector_width
        )
        sector_unit = torch.stack((torch.cos(sector_angles), torch.sin(sector_angles)), dim=-1)
        cached = (sector_width, sector_unit)
        self._uav_sector_geometry_cache[key] = cached
        return cached

    def _uav_stencil_directions(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if not hasattr(self, "_uav_stencil_direction_cache"):
            self._uav_stencil_direction_cache = {}
        key = self._device_cache_key(device, dtype)
        cached = self._uav_stencil_direction_cache.get(key)
        if cached is not None:
            return cached

        root_half = math.sqrt(0.5)
        directions = torch.tensor(
            [
                [1.0, 0.0],
                [root_half, root_half],
                [0.0, 1.0],
                [-root_half, root_half],
                [-1.0, 0.0],
                [-root_half, -root_half],
                [0.0, -1.0],
                [root_half, -root_half],
            ],
            device=device,
            dtype=dtype,
        )
        self._uav_stencil_direction_cache[key] = directions
        return directions

    def _uav_land_cover_detection_factor(
        self,
        device: torch.device,
        dtype: torch.dtype,
        grid_size: int | None = None,
    ) -> Tensor:
        if not hasattr(self, "_uav_land_cover_factor_cache"):
            self._uav_land_cover_factor_cache = {}
        G = int(self.fire_grid_size if grid_size is None else grid_size)
        key = (
            getattr(self, "_uav_terrain_cache_version", 0),
            G,
            *self._device_cache_key(device, dtype),
        )
        cached = self._uav_land_cover_factor_cache.get(key)
        if cached is not None:
            return cached

        cover_factors = self.drone_cover_detection_factors.to(device=device, dtype=dtype)
        if G == int(self.fire_grid_size):
            cover_index = self.land_cover_grid.to(device=device)
        else:
            _, _, _, _, cell_pos, _, _, _ = self._uav_grid_geometry(device, dtype, grid_size=G)
            pos = cell_pos.expand(self.world.batch_dim, -1, -1)
            cover_index = self._grid_values_at_positions(
                self.land_cover_grid.to(device=device),
                pos,
            ).long().view(self.world.batch_dim, G, G)
        cover_factor = cover_factors[cover_index].unsqueeze(1)
        self._uav_land_cover_factor_cache[key] = cover_factor
        return cover_factor

    def _uav_cell_detection_probability(
        self,
        drone_pos: Tensor,
        *,
        altitude_quality: Tensor | None = None,
        footprint: Tensor | None = None,
        grid_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Detection probability for every coverage-grid cell and UAV position."""
        if drone_pos.ndim != 3:
            raise ValueError("drone_pos must have shape [B, N, 2]")
        B, N, _ = drone_pos.shape
        G = int(self.uav_confidence_map_grid_size if grid_size is None else grid_size)
        if B == 0 or N == 0:
            empty_probability = drone_pos.new_zeros(B, N, G, G)
            empty_visible = torch.zeros(B, N, G, G, device=drone_pos.device, dtype=torch.bool)
            return empty_probability, empty_visible
        device = drone_pos.device
        dtype = drone_pos.dtype
        if footprint is None:
            if N != self.n_drones:
                raise ValueError("footprint is required when N != n_drones")
            footprint = self._drone_camera_ranges()
        if altitude_quality is None:
            if N != self.n_drones:
                raise ValueError("altitude_quality is required when N != n_drones")
            altitude_quality = self.drone_altitude_quality
        footprint = footprint.to(device=device, dtype=dtype).view(B, N)
        altitude_quality = altitude_quality.to(device=device, dtype=dtype).view(B, N)

        if not self.disable_fire:
            return self._uav_cell_detection_probability_fire_patch(
                drone_pos,
                altitude_quality=altitude_quality,
                footprint=footprint,
                grid_size=G,
            )

        xs, ys, _, _, _, _, _, _ = self._uav_grid_geometry(device, dtype, grid_size=G)
        center_dx = xs.view(1, 1, 1, G) - drone_pos[..., X].view(B, N, 1, 1)
        center_dy = ys.view(1, 1, G, 1) - drone_pos[..., Y].view(B, N, 1, 1)
        center_dist = torch.sqrt(center_dx.square() + center_dy.square())
        footprint_grid = footprint.view(B, N, 1, 1)
        visible = center_dist <= footprint_grid
        normalized_distance = (center_dist / footprint_grid.clamp_min(1e-6)).clamp(0.0, 1.0)
        distance_factor = 1.0 - (1.0 - float(self.drone_edge_detection_floor)) * normalized_distance.square()
        cover_factor = self._uav_land_cover_detection_factor(device, dtype, grid_size=G)
        altitude_quality_grid = altitude_quality.view(B, N, 1, 1)
        probability = (
            altitude_quality_grid
            * distance_factor
            * cover_factor
        ).clamp(0.0, 1.0)
        return torch.where(visible, probability, torch.zeros_like(probability)), visible

    def _uav_cell_detection_probability_fire_patch(
        self,
        drone_pos: Tensor,
        *,
        altitude_quality: Tensor,
        footprint: Tensor,
        grid_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Fire-aware UAV detection probability, computed only over footprint patches."""
        B, N, _ = drone_pos.shape
        G = int(grid_size)
        device = drone_pos.device
        dtype = drone_pos.dtype
        probability = torch.zeros(B, N, G, G, device=device, dtype=dtype)
        visible = torch.zeros(B, N, G, G, device=device, dtype=torch.bool)

        xs, ys, _, _, _, cell_width, cell_height, _ = self._uav_grid_geometry(
            device,
            dtype,
            grid_size=G,
        )
        max_footprint = float(footprint.detach().max().cpu().item()) if footprint.numel() > 0 else 0.0
        rx = min(G - 1, int(math.ceil(max_footprint / max(float(cell_width), 1e-12))) + 1)
        ry = min(G - 1, int(math.ceil(max_footprint / max(float(cell_height), 1e-12))) + 1)
        offset_x = torch.arange(-rx, rx + 1, device=device).view(1, 1, 1, -1)
        offset_y = torch.arange(-ry, ry + 1, device=device).view(1, 1, -1, 1)
        center_gx, center_gy = self._positions_to_grid(drone_pos, grid_size=G)
        gx_raw = center_gx.view(B, N, 1, 1) + offset_x
        gy_raw = center_gy.view(B, N, 1, 1) + offset_y
        valid = (gx_raw >= 0) & (gx_raw < G) & (gy_raw >= 0) & (gy_raw < G)
        gx = gx_raw.clamp(0, G - 1).expand_as(valid)
        gy = gy_raw.clamp(0, G - 1).expand_as(valid)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(valid)
        drone_idx = torch.arange(N, device=device).view(1, N, 1, 1).expand_as(valid)

        center_dx = xs[gx] - drone_pos[..., X].view(B, N, 1, 1)
        center_dy = ys[gy] - drone_pos[..., Y].view(B, N, 1, 1)
        center_dist = torch.sqrt(center_dx.square() + center_dy.square())
        footprint_patch = footprint.view(B, N, 1, 1)
        visible_patch = valid & (center_dist <= footprint_patch)
        normalized_distance = (center_dist / footprint_patch.clamp_min(1e-6)).clamp(0.0, 1.0)
        distance_factor = 1.0 - (1.0 - float(self.drone_edge_detection_floor)) * normalized_distance.square()

        cover_factor = self._uav_land_cover_detection_factor(device, dtype, grid_size=G)[:, 0]
        cover_patch = cover_factor[batch_idx, gy, gx]
        patch_cell_pos = torch.stack(
            (
                xs[gx].expand_as(center_dist),
                ys[gy].expand_as(center_dist),
            ),
            dim=-1,
        )
        fire_smoke_factor = self._uav_patch_fire_smoke_visibility_factor(
            drone_pos.to(device=device, dtype=dtype),
            patch_cell_pos,
        )
        probability_patch = (
            altitude_quality.view(B, N, 1, 1)
            * distance_factor
            * cover_patch
            * fire_smoke_factor
        ).clamp(0.0, 1.0)
        probability_patch = torch.where(
            visible_patch,
            probability_patch,
            torch.zeros_like(probability_patch),
        )
        probability[
            batch_idx[visible_patch],
            drone_idx[visible_patch],
            gy[visible_patch],
            gx[visible_patch],
        ] = probability_patch[visible_patch]
        visible[
            batch_idx[visible_patch],
            drone_idx[visible_patch],
            gy[visible_patch],
            gx[visible_patch],
        ] = True
        return probability, visible

    def _uav_confidence_stencil_candidates(self, previous: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = previous.shape[0]
        D = self.n_drones
        directions = self._uav_stencil_directions(previous.device, previous.dtype)
        max_step_sim = (
            float(self.drone_speed_mps)
            * float(self.sim_step_seconds)
            * self.terrain_sim_units_per_meter.to(
                device=previous.device,
                dtype=previous.dtype,
            ).clamp_min(1e-9)
        ).view(B, 1, 1, 1)
        pre_pos = self._pre_step_drone_pos.to(device=previous.device, dtype=previous.dtype)
        candidate_pos = pre_pos.unsqueeze(2) + directions.view(1, 1, 8, 2) * max_step_sim
        x_min = -float(self.x_semidim) + float(self.agent_radius)
        x_max = float(self.x_semidim) - float(self.agent_radius)
        y_min = -float(self.y_semidim) + float(self.agent_radius)
        y_max = float(self.y_semidim) - float(self.agent_radius)
        candidate_pos[..., X] = candidate_pos[..., X].clamp(x_min, x_max)
        candidate_pos[..., Y] = candidate_pos[..., Y].clamp(y_min, y_max)
        flat_pos = candidate_pos.reshape(B, D * 8, 2)
        altitude_quality = (
            self.drone_altitude_quality.to(device=previous.device, dtype=previous.dtype)
            .unsqueeze(2)
            .expand(B, D, 8)
            .reshape(B, D * 8)
        )
        footprint = (
            self._drone_camera_ranges().to(device=previous.device, dtype=previous.dtype)
            .unsqueeze(2)
            .expand(B, D, 8)
            .reshape(B, D * 8)
        )
        return flat_pos, altitude_quality, footprint

    def _uav_confidence_full_grid_stencil_gain(
        self,
        previous: Tensor,
        confidence_weight: Tensor,
        flat_pos: Tensor,
        altitude_quality: Tensor,
        footprint: Tensor,
    ) -> Tensor:
        probability, _ = self._uav_cell_detection_probability(
            flat_pos,
            altitude_quality=altitude_quality,
            footprint=footprint,
            grid_size=previous.shape[-1],
        )
        candidate_gain = (1.0 - previous).clamp(0.0, 1.0).unsqueeze(1) * probability
        return (confidence_weight.unsqueeze(1) * candidate_gain).mean(dim=(-1, -2))

    def _uav_patch_fire_smoke_visibility_factor(
        self,
        drone_pos: Tensor,
        cell_pos: Tensor,
    ) -> Tensor:
        """Fire/smoke visibility for candidate-specific footprint patch cells."""
        B, C = drone_pos.shape[:2]
        device = drone_pos.device
        dtype = drone_pos.dtype
        samples = max(int(self.drone_perception_path_samples), 2)
        alpha = torch.linspace(0.0, 1.0, samples, device=device, dtype=dtype)
        start = drone_pos.view(B, C, 1, 1, 1, 2)
        end = cell_pos.unsqueeze(-2)
        path = start + (end - start) * alpha.view(1, 1, 1, 1, samples, 1)
        env_indices = torch.arange(B, device=device)
        smoke_path = self._grid_values_at_positions(
            self.smoke_grid,
            path,
            env_indices,
        ).to(dtype=dtype)
        fire_path = self._grid_values_at_positions(
            self.fire_intensity_grid,
            path,
            env_indices,
        ).to(dtype=dtype)

        smoke_mean = smoke_path.mean(dim=-1)
        target_smoke = smoke_path[..., -1]
        smoke_load = 0.65 * smoke_mean + 0.35 * target_smoke
        smoke_factor = torch.exp(-self.drone_smoke_extinction * smoke_load)
        smoke_floor = torch.full_like(smoke_factor, float(self.drone_smoke_detection_factor))
        smoke_factor = torch.maximum(smoke_factor, smoke_floor)

        gx = ((cell_pos[..., X] + self.x_semidim) / (2 * self.x_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2,
        ).long()
        gy = ((cell_pos[..., Y] + self.y_semidim) / (2 * self.y_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2,
        ).long()
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(gx)
        target_fire_density = torch.zeros_like(gx, dtype=dtype)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                target_fire_density = target_fire_density + self.fire_grid[
                    b_idx,
                    gy + dy,
                    gx + dx,
                ].to(dtype=dtype)
        target_fire_density = target_fire_density / 9.0

        fire_path_mean = fire_path.mean(dim=-1)
        fire_path_max = fire_path.amax(dim=-1)
        glare_load = torch.maximum(fire_path_max, target_fire_density)
        glare_factor = 1.0 - self.drone_fire_glare_penalty * glare_load
        heat_factor = 1.0 - self.drone_heat_distortion_penalty * fire_path_mean
        return (
            smoke_factor
            * glare_factor.clamp(0.0, 1.0)
            * heat_factor.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

    def _uav_confidence_patch_stencil_gain(
        self,
        previous: Tensor,
        confidence_weight: Tensor,
        flat_pos: Tensor,
        altitude_quality: Tensor,
        footprint: Tensor,
    ) -> Tensor:
        B, C, _ = flat_pos.shape
        G = int(previous.shape[-1])
        device = previous.device
        dtype = previous.dtype
        xs, ys, _, _, _, cell_width, cell_height, _ = self._uav_grid_geometry(
            device,
            dtype,
            grid_size=G,
        )
        max_footprint = float(footprint.detach().max().cpu().item()) if footprint.numel() > 0 else 0.0
        rx = min(G - 1, int(math.ceil(max_footprint / max(float(cell_width), 1e-12))) + 1)
        ry = min(G - 1, int(math.ceil(max_footprint / max(float(cell_height), 1e-12))) + 1)

        offset_x = torch.arange(-rx, rx + 1, device=device).view(1, 1, 1, -1)
        offset_y = torch.arange(-ry, ry + 1, device=device).view(1, 1, -1, 1)
        center_gx, center_gy = self._positions_to_grid(flat_pos, grid_size=G)
        gx_raw = center_gx.view(B, C, 1, 1) + offset_x
        gy_raw = center_gy.view(B, C, 1, 1) + offset_y
        valid = (gx_raw >= 0) & (gx_raw < G) & (gy_raw >= 0) & (gy_raw < G)
        gx = gx_raw.clamp(0, G - 1)
        gy = gy_raw.clamp(0, G - 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(gx)

        previous_patch = previous[batch_idx, gy, gx]
        confidence_weight_patch = confidence_weight[batch_idx, gy, gx]
        cover_factor = self._uav_land_cover_detection_factor(device, dtype, grid_size=G)[:, 0]
        cover_patch = cover_factor[batch_idx, gy, gx]

        center_dx = xs[gx] - flat_pos[..., X].view(B, C, 1, 1)
        center_dy = ys[gy] - flat_pos[..., Y].view(B, C, 1, 1)
        center_dist = torch.sqrt(center_dx.square() + center_dy.square())
        footprint_patch = footprint.to(device=device, dtype=dtype).view(B, C, 1, 1)
        visible = valid & (center_dist <= footprint_patch)
        normalized_distance = (center_dist / footprint_patch.clamp_min(1e-6)).clamp(0.0, 1.0)
        distance_factor = 1.0 - (1.0 - float(self.drone_edge_detection_floor)) * normalized_distance.square()
        if self.disable_fire:
            fire_smoke_factor = torch.ones_like(distance_factor)
        else:
            patch_cell_pos = torch.stack(
                (
                    xs[gx].expand_as(center_dist),
                    ys[gy].expand_as(center_dist),
                ),
                dim=-1,
            )
            fire_smoke_factor = self._uav_patch_fire_smoke_visibility_factor(
                flat_pos.to(device=device, dtype=dtype),
                patch_cell_pos,
            )
        probability = (
            altitude_quality.to(device=device, dtype=dtype).view(B, C, 1, 1)
            * distance_factor
            * cover_patch
            * fire_smoke_factor
        ).clamp(0.0, 1.0)
        probability = torch.where(visible, probability, torch.zeros_like(probability))
        weighted_gain = (
            confidence_weight_patch
            * (1.0 - previous_patch).clamp(0.0, 1.0)
            * probability
        )
        return weighted_gain.sum(dim=(-1, -2)) / float(G * G)

    def _uav_confidence_best_stencil_gain(
        self,
        previous: Tensor,
        confidence_weight: Tensor,
    ) -> Tensor:
        """Best weighted confidence gain from an 8-way max-speed move stencil."""
        if self.n_drones == 0:
            return previous.new_zeros(previous.shape[0], 0)
        B = previous.shape[0]
        D = self.n_drones
        flat_pos, altitude_quality, footprint = self._uav_confidence_stencil_candidates(previous)
        G = int(previous.shape[-1])
        _, _, _, _, _, cell_width, cell_height, _ = self._uav_grid_geometry(
            previous.device,
            previous.dtype,
            grid_size=G,
        )
        max_footprint = float(footprint.detach().max().cpu().item()) if footprint.numel() > 0 else 0.0
        rx = min(G - 1, int(math.ceil(max_footprint / max(float(cell_width), 1e-12))) + 1)
        ry = min(G - 1, int(math.ceil(max_footprint / max(float(cell_height), 1e-12))) + 1)
        patch_area = (2 * rx + 1) * (2 * ry + 1)
        if patch_area < G * G:
            weighted_gain = self._uav_confidence_patch_stencil_gain(
                previous,
                confidence_weight,
                flat_pos,
                altitude_quality,
                footprint,
            )
        else:
            weighted_gain = self._uav_confidence_full_grid_stencil_gain(
                previous,
                confidence_weight,
                flat_pos,
                altitude_quality,
                footprint,
            )
        return weighted_gain.view(B, D, 8).max(dim=2).values

    def _uav_confidence_best_stencil_gain_by_drone(
        self,
        previous_by_drone: Tensor,
        confidence_weight_by_drone: Tensor,
    ) -> Tensor:
        """Per-UAV opportunity gain using each UAV's own confidence memory."""
        if self.n_drones == 0:
            return previous_by_drone.new_zeros(previous_by_drone.shape[0], 0)
        gains = []
        for drone_idx in range(self.n_drones):
            gains.append(self._uav_confidence_best_stencil_gain(
                previous_by_drone[:, drone_idx],
                confidence_weight_by_drone[:, drone_idx],
            )[:, drone_idx])
        return torch.stack(gains, dim=1)

    def _uav_confidence_active(self) -> bool:
        return self.n_drones > 0 and not (
            not self.uav_confidence_diagnostics
            and self.r_uav_confidence <= 0.0
            and self.r_uav_confidence_move <= 0.0
            and not (
                self.r_uav_inefficient_move > 0.0
                and self.uav_inefficient_move_source == "confidence"
            )
            and self.r_uav_confidence_overlap <= 0.0
            and self.r_uav_team_confidence_overlap <= 0.0
            and self.uav_frontier_source != "confidence"
            and self.uav_confidence_obs_grid <= 0
            and self.local_confidence_obs_grid <= 0
            and not self.uav_cleanup_target_obs
            and not self.uav_cleanup_target_diagnostics
            and not self.uav_astar_route_obs
            and self.r_uav_cleanup_target_progress <= 0.0
            and self.r_uav_astar_progress <= 0.0
        )

    def _uav_cleanup_target_active(self) -> bool:
        return self.n_drones > 0 and (
            bool(self.uav_cleanup_target_obs)
            or bool(self.uav_cleanup_target_diagnostics)
            or bool(self.uav_astar_route_obs)
            or self.r_uav_cleanup_target_progress > 0.0
            or self.r_uav_astar_progress > 0.0
        )

    def _uav_cleanup_target_obs_dim(self) -> int:
        return 4

    def _uav_astar_route_obs_dim(self) -> int:
        return 4

    def _uav_cleanup_target_mass_map(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        confidence = self.uav_confidence_grid.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
        confidence_weight = (
            float(self.uav_confidence_eps)
            + uncertainty.pow(float(self.uav_confidence_gamma))
        )
        return (confidence_weight * uncertainty).clamp(min=0.0)

    def _uav_cleanup_target_pooled_maps(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        import torch.nn.functional as F

        K = int(self.uav_cleanup_target_grid)
        confidence = self.uav_confidence_grid.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        mass = self._uav_cleanup_target_mass_map(device=device, dtype=dtype)
        pooled_confidence = F.adaptive_avg_pool2d(confidence.unsqueeze(1), (K, K)).squeeze(1)
        pooled_mass = F.adaptive_avg_pool2d(mass.unsqueeze(1), (K, K)).squeeze(1)
        return pooled_confidence, pooled_mass

    def _uav_cleanup_target_coarse_geometry(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, float, float]:
        if not hasattr(self, "_uav_cleanup_target_geometry_cache"):
            self._uav_cleanup_target_geometry_cache = {}
        K = int(self.uav_cleanup_target_grid)
        key = self._device_cache_key(device, dtype) + (
            K,
            float(self.x_semidim),
            float(self.y_semidim),
        )
        cached = self._uav_cleanup_target_geometry_cache.get(key)
        if cached is not None:
            return cached

        cell_width = 2.0 * float(self.x_semidim) / float(K)
        cell_height = 2.0 * float(self.y_semidim) / float(K)
        coarse_x = torch.linspace(
            -float(self.x_semidim) + cell_width / 2.0,
            float(self.x_semidim) - cell_width / 2.0,
            K,
            device=device,
            dtype=dtype,
        )
        coarse_y = torch.linspace(
            -float(self.y_semidim) + cell_height / 2.0,
            float(self.y_semidim) - cell_height / 2.0,
            K,
            device=device,
            dtype=dtype,
        )
        cached = (coarse_x, coarse_y, cell_width, cell_height)
        self._uav_cleanup_target_geometry_cache[key] = cached
        return cached

    def _uav_cleanup_target_component_from_cells(
        self,
        *,
        env_idx: int,
        cells: list[tuple[int, int]],
        pooled_confidence: Tensor,
        pooled_mass: Tensor,
        coarse_x: Tensor,
        coarse_y: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor | int] | None:
        if not cells:
            return None
        K = int(self.uav_cleanup_target_grid)
        ys = torch.tensor([cell[0] for cell in cells], device=device, dtype=torch.long)
        xs = torch.tensor([cell[1] for cell in cells], device=device, dtype=torch.long)
        weights = pooled_mass[env_idx, ys, xs].clamp_min(0.0)
        total = weights.sum()
        if float(total.detach().cpu().item()) <= 1e-12:
            return None
        centroid_x = (coarse_x[xs] * weights).sum() / total
        centroid_y = (coarse_y[ys] * weights).sum() / total
        mean_value = weights.mean().clamp(0.0, 1.0)
        current_confidence = (pooled_confidence[env_idx, ys, xs] * weights).sum() / total
        component_id = min(int(cell[0]) * K + int(cell[1]) for cell in cells)
        return {
            "id": component_id,
            "centroid": torch.stack((centroid_x, centroid_y)).to(device=device, dtype=dtype),
            "value": mean_value.to(device=device, dtype=dtype),
            "confidence": current_confidence.clamp(0.0, 1.0).to(device=device, dtype=dtype),
        }

    def _uav_cleanup_target_component_for_id(
        self,
        *,
        env_idx: int,
        component_id: int,
        pooled_confidence: Tensor,
        pooled_mass: Tensor,
        coarse_x: Tensor,
        coarse_y: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor | int] | None:
        K = int(self.uav_cleanup_target_grid)
        if component_id < 0 or component_id >= K * K:
            return None
        start_y = int(component_id) // K
        start_x = int(component_id) % K
        targetable = (
            (pooled_confidence[env_idx] < float(self.uav_cleanup_target_confidence_threshold))
            & (pooled_mass[env_idx] > float(self.uav_cleanup_target_min_value))
        )
        targetable_rows = targetable.detach().cpu().tolist()
        if not bool(targetable_rows[start_y][start_x]):
            return None

        visited = [[False for _ in range(K)] for _ in range(K)]
        stack = [(start_y, start_x)]
        visited[start_y][start_x] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < K
                    and 0 <= nx < K
                    and not visited[ny][nx]
                    and bool(targetable_rows[ny][nx])
                ):
                    visited[ny][nx] = True
                    stack.append((ny, nx))
        component = self._uav_cleanup_target_component_from_cells(
            env_idx=env_idx,
            cells=cells,
            pooled_confidence=pooled_confidence,
            pooled_mass=pooled_mass,
            coarse_x=coarse_x,
            coarse_y=coarse_y,
            device=device,
            dtype=dtype,
        )
        if component is None or int(component["id"]) != int(component_id):
            return None
        return component

    def _uav_cleanup_target_components(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        pooled_confidence: Tensor | None = None,
        pooled_mass: Tensor | None = None,
        env_indices: Tensor | list[int] | None = None,
    ) -> list[list[dict[str, Tensor | int]]]:
        K = int(self.uav_cleanup_target_grid)
        if pooled_confidence is None or pooled_mass is None:
            pooled_confidence, pooled_mass = self._uav_cleanup_target_pooled_maps(
                device=device,
                dtype=dtype,
            )
        coarse_x, coarse_y, _, _ = self._uav_cleanup_target_coarse_geometry(
            device=device,
            dtype=dtype,
        )

        batch_dim = int(self.world.batch_dim)
        all_components: list[list[dict[str, Tensor | int]]] = [[] for _ in range(batch_dim)]
        if env_indices is None:
            env_iter = range(batch_dim)
        elif isinstance(env_indices, Tensor):
            env_iter = [int(value) for value in env_indices.detach().cpu().tolist()]
        else:
            env_iter = [int(value) for value in env_indices]
        threshold = float(self.uav_cleanup_target_confidence_threshold)
        min_value = float(self.uav_cleanup_target_min_value)
        for env_idx in env_iter:
            targetable = (
                (pooled_confidence[env_idx] < threshold)
                & (pooled_mass[env_idx] > min_value)
            )
            targetable_rows = targetable.detach().cpu().tolist()
            visited = [[False for _ in range(K)] for _ in range(K)]
            env_components: list[dict[str, Tensor | int]] = []
            for y in range(K):
                for x in range(K):
                    if visited[y][x] or not bool(targetable_rows[y][x]):
                        continue
                    stack = [(y, x)]
                    visited[y][x] = True
                    cells: list[tuple[int, int]] = []
                    while stack:
                        cy, cx = stack.pop()
                        cells.append((cy, cx))
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if (
                                0 <= ny < K
                                and 0 <= nx < K
                                and not visited[ny][nx]
                                and bool(targetable_rows[ny][nx])
                            ):
                                visited[ny][nx] = True
                                stack.append((ny, nx))
                    component = self._uav_cleanup_target_component_from_cells(
                        env_idx=env_idx,
                        cells=cells,
                        pooled_confidence=pooled_confidence,
                        pooled_mass=pooled_mass,
                        coarse_x=coarse_x,
                        coarse_y=coarse_y,
                        device=device,
                        dtype=dtype,
                    )
                    if component is not None:
                        env_components.append(component)
            all_components[env_idx] = env_components
        return all_components

    def _refresh_uav_cleanup_target_assignments(self, drone_pos: Tensor) -> None:
        if not self._uav_cleanup_target_active() or self.n_drones <= 0:
            return
        if not hasattr(self, "uav_cleanup_target_valid"):
            return
        with torch.no_grad():
            device = drone_pos.device
            dtype = drone_pos.dtype
            current_step = self.step_count.to(device=device)
            stale = self._uav_cleanup_target_last_assignment_step.to(device=device) != current_step
            if not bool(stale.any().detach().cpu().item()):
                return

            hold_steps = int(self.uav_cleanup_target_hold_steps)

            self.metric_uav_cleanup_target_switch_by_drone[stale] = 0.0
            previous_reached = self.metric_uav_cleanup_target_reached_by_drone.to(
                device=device,
            ) > 0.5
            fixed_hold_mode = self.uav_cleanup_target_refresh_mode == "fixed_hold"
            fixed_keep_mask = torch.zeros_like(self.uav_cleanup_target_valid, dtype=torch.bool)
            refresh_env_mask = stale.clone()
            if fixed_hold_mode:
                valid = self.uav_cleanup_target_valid.to(device=device)
                target_id = self.uav_cleanup_target_id.to(device=device)
                age = self.uav_cleanup_target_age.to(device=device)
                fixed_keep_mask = (
                    stale.view(-1, 1)
                    & valid
                    & (target_id >= 0)
                    & (age < hold_steps)
                    & ~previous_reached
                )
                needs_refresh = stale.view(-1, 1) & ~fixed_keep_mask
                refresh_env_mask = stale & needs_refresh.any(dim=1)
                if bool(fixed_keep_mask.any().detach().cpu().item()):
                    self.uav_cleanup_target_age = torch.where(
                        fixed_keep_mask,
                        self.uav_cleanup_target_age + 1,
                        self.uav_cleanup_target_age,
                    )
                no_refresh_env_mask = stale & ~refresh_env_mask
                if bool(no_refresh_env_mask.any().detach().cpu().item()):
                    self._uav_cleanup_target_last_assignment_step[no_refresh_env_mask] = current_step[
                        no_refresh_env_mask
                    ].to(device=self._uav_cleanup_target_last_assignment_step.device)
                if not bool(refresh_env_mask.any().detach().cpu().item()):
                    return

            pooled_confidence, pooled_mass = self._uav_cleanup_target_pooled_maps(
                device=device,
                dtype=dtype,
            )
            coarse_x, coarse_y, _, _ = self._uav_cleanup_target_coarse_geometry(
                device=device,
                dtype=dtype,
            )
            meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(
                device=device,
                dtype=dtype,
            ).clamp_min(1e-9)
            distance_scale_m = float(self.uav_cleanup_target_assignment_distance_scale_m)
            stale_env_indices = [
                int(value)
                for value in refresh_env_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            ]
            for env_idx in stale_env_indices:
                self.metric_uav_cleanup_target_switch_by_drone[env_idx] = 0.0
                used_ids: set[int] = set()
                needing_assignment: list[int] = []
                for drone_idx in range(self.n_drones):
                    old_valid = bool(self.uav_cleanup_target_valid[env_idx, drone_idx].detach().cpu().item())
                    old_id = int(self.uav_cleanup_target_id[env_idx, drone_idx].detach().cpu().item())
                    age = int(self.uav_cleanup_target_age[env_idx, drone_idx].detach().cpu().item())
                    reached = bool(previous_reached[env_idx, drone_idx].detach().cpu().item())
                    comp = None
                    if fixed_hold_mode and bool(fixed_keep_mask[env_idx, drone_idx].detach().cpu().item()):
                        used_ids.add(old_id)
                        continue
                    elif (
                        old_valid
                        and age < hold_steps
                        and not reached
                    ):
                        comp = self._uav_cleanup_target_component_for_id(
                            env_idx=env_idx,
                            component_id=old_id,
                            pooled_confidence=pooled_confidence,
                            pooled_mass=pooled_mass,
                            coarse_x=coarse_x,
                            coarse_y=coarse_y,
                            device=device,
                            dtype=dtype,
                        )
                    if comp is not None:
                        self.uav_cleanup_target_pos[env_idx, drone_idx] = comp["centroid"]
                        self.uav_cleanup_target_value[env_idx, drone_idx] = comp["value"]
                        self.uav_cleanup_target_age[env_idx, drone_idx] += 1
                        used_ids.add(old_id)
                    else:
                        needing_assignment.append(drone_idx)

                components: list[dict[str, Tensor | int]] | None = None
                if needing_assignment:
                    components_by_env = self._uav_cleanup_target_components(
                        device=device,
                        dtype=dtype,
                        pooled_confidence=pooled_confidence,
                        pooled_mass=pooled_mass,
                        env_indices=[env_idx],
                    )
                    components = components_by_env[env_idx]

                if components:
                    comp_ids = [int(comp["id"]) for comp in components]
                    comp_pos = torch.stack([comp["centroid"].to(device=device, dtype=dtype) for comp in components])
                    comp_value = torch.stack([comp["value"].to(device=device, dtype=dtype) for comp in components])
                    drone_indices = needing_assignment
                    if drone_indices:
                        drone_positions = drone_pos[env_idx, drone_indices]
                        dists_m = (drone_positions.unsqueeze(1) - comp_pos.unsqueeze(0)).norm(dim=-1) * meters_per_sim[env_idx]
                        scores = comp_value.unsqueeze(0) / (1.0 + dists_m / distance_scale_m)
                        order = sorted(
                            range(len(drone_indices)),
                            key=lambda idx: float(scores[idx].max().detach().cpu().item()),
                            reverse=True,
                        )
                        for local_idx in order:
                            drone_idx = drone_indices[local_idx]
                            available = [
                                comp_i
                                for comp_i, comp_id in enumerate(comp_ids)
                                if comp_id not in used_ids
                            ]
                            if not available:
                                available = list(range(len(components)))
                            best_comp_i = max(
                                available,
                                key=lambda comp_i: float(scores[local_idx, comp_i].detach().cpu().item()),
                            )
                            comp = components[best_comp_i]
                            new_id = int(comp["id"])
                            old_valid = bool(self.uav_cleanup_target_valid[env_idx, drone_idx].detach().cpu().item())
                            old_id = int(self.uav_cleanup_target_id[env_idx, drone_idx].detach().cpu().item())
                            self.uav_cleanup_target_valid[env_idx, drone_idx] = True
                            self.uav_cleanup_target_pos[env_idx, drone_idx] = comp["centroid"]
                            self.uav_cleanup_target_value[env_idx, drone_idx] = comp["value"]
                            self.uav_cleanup_target_initial_value[env_idx, drone_idx] = comp["value"]
                            self.uav_cleanup_target_age[env_idx, drone_idx] = 0
                            self.uav_cleanup_target_id[env_idx, drone_idx] = new_id
                            current_distance_m = (
                                drone_pos[env_idx, drone_idx] - comp["centroid"].to(device=device, dtype=dtype)
                            ).norm() * meters_per_sim[env_idx]
                            self.uav_cleanup_target_prev_distance_m[env_idx, drone_idx] = current_distance_m
                            if old_valid and old_id != new_id:
                                self.metric_uav_cleanup_target_switch_by_drone[env_idx, drone_idx] = 1.0
                            used_ids.add(new_id)

                for drone_idx in needing_assignment:
                    if bool(self.uav_cleanup_target_valid[env_idx, drone_idx].detach().cpu().item()):
                        continue
                    self.uav_cleanup_target_pos[env_idx, drone_idx] = 0.0
                    self.uav_cleanup_target_value[env_idx, drone_idx] = 0.0
                    self.uav_cleanup_target_initial_value[env_idx, drone_idx] = 0.0
                    self.uav_cleanup_target_age[env_idx, drone_idx] = 0
                    self.uav_cleanup_target_id[env_idx, drone_idx] = -1
                    self.uav_cleanup_target_prev_distance_m[env_idx, drone_idx] = float("inf")

                if needing_assignment and not components:
                    previous_valid = self.uav_cleanup_target_valid[env_idx].clone()
                    self.uav_cleanup_target_valid[env_idx] = False
                    self.uav_cleanup_target_pos[env_idx] = 0.0
                    self.uav_cleanup_target_value[env_idx] = 0.0
                    self.uav_cleanup_target_initial_value[env_idx] = 0.0
                    self.uav_cleanup_target_age[env_idx] = 0
                    self.uav_cleanup_target_id[env_idx] = -1
                    self.uav_cleanup_target_prev_distance_m[env_idx] = float("inf")
                    self.metric_uav_cleanup_target_switch_by_drone[env_idx] = previous_valid.to(
                        dtype=self.metric_uav_cleanup_target_switch_by_drone.dtype
                    )
                self._uav_cleanup_target_last_assignment_step[env_idx] = current_step[env_idx]

    def _update_uav_cleanup_target_step_metrics(self, drone_pos: Tensor) -> None:
        if not self._uav_cleanup_target_active() or self.n_drones <= 0:
            return
        with torch.no_grad():
            self._refresh_uav_cleanup_target_assignments(drone_pos)
            valid = self.uav_cleanup_target_valid.to(device=drone_pos.device)
            meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(
                device=drone_pos.device,
                dtype=drone_pos.dtype,
            ).clamp_min(1e-9)
            target_vec = self.uav_cleanup_target_pos.to(device=drone_pos.device, dtype=drone_pos.dtype) - drone_pos
            distance_m = target_vec.norm(dim=-1) * meters_per_sim.view(-1, 1)
            previous_distance = self.uav_cleanup_target_prev_distance_m.to(
                device=drone_pos.device,
                dtype=drone_pos.dtype,
            )
            progress_m = torch.where(
                valid & torch.isfinite(previous_distance),
                previous_distance - distance_m,
                torch.zeros_like(distance_m),
            )
            max_step_m = max(float(self.drone_speed_mps) * float(self.sim_step_seconds), 1e-6)
            progress_fraction = (progress_m / max_step_m).clamp(-1.0, 1.0)
            footprint_m = self._drone_camera_ranges().to(
                device=drone_pos.device,
                dtype=drone_pos.dtype,
            ) * meters_per_sim.view(-1, 1)
            reached = valid & (distance_m <= footprint_m)
            value = self.uav_cleanup_target_value.to(device=drone_pos.device, dtype=drone_pos.dtype)
            initial_value = self.uav_cleanup_target_initial_value.to(device=drone_pos.device, dtype=drone_pos.dtype)
            value_decay = (initial_value - value).clamp_min(0.0)
            valid_f = valid.to(dtype=drone_pos.dtype)
            self.metric_uav_cleanup_target_valid_by_drone = valid_f
            self.metric_uav_cleanup_target_distance_m_by_drone = torch.where(
                valid,
                distance_m,
                torch.zeros_like(distance_m),
            )
            self.metric_uav_cleanup_target_value_by_drone = torch.where(
                valid,
                value,
                torch.zeros_like(value),
            )
            self.metric_uav_cleanup_target_progress_m_by_drone = torch.where(
                valid,
                progress_m,
                torch.zeros_like(progress_m),
            )
            self.metric_uav_cleanup_target_progress_fraction_by_drone = torch.where(
                valid,
                progress_fraction,
                torch.zeros_like(progress_fraction),
            )
            self.metric_uav_cleanup_target_reached_by_drone = reached.to(dtype=drone_pos.dtype)
            self.metric_uav_cleanup_target_value_decay_by_drone = torch.where(
                valid,
                value_decay,
                torch.zeros_like(value_decay),
            )
            self.metric_uav_cleanup_target_age_by_drone = torch.where(
                valid,
                self.uav_cleanup_target_age.to(device=drone_pos.device, dtype=drone_pos.dtype),
                torch.zeros_like(value),
            )
            self.uav_cleanup_target_prev_distance_m = torch.where(
                valid,
                distance_m,
                torch.full_like(distance_m, float("inf")),
            )

    def _uav_local_frontier_score(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Configured local frontier strength per UAV, used to gate cleanup progress."""
        if self.n_drones == 0:
            return torch.zeros(self.world.batch_dim, 0, device=device, dtype=dtype)
        features = self._pre_step_uav_frontier_features().to(device=device, dtype=dtype)
        if self.uav_frontier_mode == "sector_topk":
            B, D, _ = features.shape
            top_k = max(int(self.uav_frontier_top_k), 1)
            return features.view(B, D, top_k, 4)[..., 3].clamp(0.0, 1.0).max(dim=-1).values
        if self.uav_frontier_mode == "local_global":
            B, D, _ = features.shape
            return features.view(B, D, 2, 4)[..., 0, 3].clamp(0.0, 1.0)
        return features[..., 3].clamp(0.0, 1.0)

    def _uav_cleanup_target_progress_reward(self, drone_pos: Tensor) -> tuple[Tensor, Tensor]:
        """Reward progress toward cleanup targets when the local frontier is weak."""
        reward = drone_pos.new_zeros(drone_pos.shape[0], self.n_drones)
        gate = torch.zeros_like(reward)
        if self.n_drones == 0 or self.r_uav_cleanup_target_progress <= 0.0:
            return reward, gate

        valid = self.metric_uav_cleanup_target_valid_by_drone.to(
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        ).clamp(0.0, 1.0)
        progress = self.metric_uav_cleanup_target_progress_fraction_by_drone.to(
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        ).clamp(0.0, 1.0)
        target_value = self.metric_uav_cleanup_target_value_by_drone.to(
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        ).clamp(0.0, 1.0)
        local_frontier_score = self._uav_local_frontier_score(
            drone_pos.device,
            drone_pos.dtype,
        )
        gate = (1.0 - local_frontier_score).clamp(0.0, 1.0) * valid
        reward = (
            float(self.r_uav_cleanup_target_progress)
            * progress
            * target_value
            * gate
        )
        return reward, gate

    def _uav_astar_position_to_cell(self, pos: Tensor, K: int) -> tuple[int, int]:
        gx = int(
            torch.clamp(
                (pos[X] + float(self.x_semidim)) / (2.0 * float(self.x_semidim)) * float(K),
                0,
                K - 1,
            )
            .long()
            .detach()
            .cpu()
            .item()
        )
        gy = int(
            torch.clamp(
                (pos[Y] + float(self.y_semidim)) / (2.0 * float(self.y_semidim)) * float(K),
                0,
                K - 1,
            )
            .long()
            .detach()
            .cpu()
            .item()
        )
        return gx, gy

    def _uav_astar_cell_center(
        self,
        cell: tuple[int, int],
        *,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        gx, gy = cell
        cell_w = 2.0 * float(self.x_semidim) / float(K)
        cell_h = 2.0 * float(self.y_semidim) / float(K)
        return torch.tensor(
            [
                -float(self.x_semidim) + (float(gx) + 0.5) * cell_w,
                -float(self.y_semidim) + (float(gy) + 0.5) * cell_h,
            ],
            device=device,
            dtype=dtype,
        )

    def _uav_astar_pooled_confidence(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        confidence_grid: Tensor | None = None,
    ) -> Tensor:
        import torch.nn.functional as F

        K = int(self.uav_astar_grid)
        source = self.uav_confidence_grid if confidence_grid is None else confidence_grid
        confidence = source.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if confidence.ndim == 4:
            B, D, G, _ = confidence.shape
            pooled = F.adaptive_avg_pool2d(
                confidence.reshape(B * D, 1, G, G),
                (K, K),
            ).reshape(B, D, K, K)
            return pooled
        return F.adaptive_avg_pool2d(confidence.unsqueeze(1), (K, K)).squeeze(1)

    def _uav_astar_plan(
        self,
        cell_cost: Tensor,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[list[tuple[int, int]], float]:
        K = int(cell_cost.shape[-1])
        costs = cell_cost.detach().cpu().tolist()
        sx, sy = start
        gx, gy = goal
        if not (0 <= sx < K and 0 <= sy < K and 0 <= gx < K and 0 <= gy < K):
            return [], float("inf")
        if start == goal:
            return [start], 0.0

        def heuristic(cell: tuple[int, int]) -> float:
            x, y = cell
            return math.hypot(float(gx - x), float(gy - y))

        neighbors = (
            (-1, -1, math.sqrt(2.0)),
            (0, -1, 1.0),
            (1, -1, math.sqrt(2.0)),
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (-1, 1, math.sqrt(2.0)),
            (0, 1, 1.0),
            (1, 1, math.sqrt(2.0)),
        )
        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start), 0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current_cost > g_score.get(current, float("inf")) + 1e-12:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, current_cost

            x, y = current
            current_cell_cost = float(costs[y][x])
            if not math.isfinite(current_cell_cost):
                continue
            for dx, dy, step_len in neighbors:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= K or ny < 0 or ny >= K:
                    continue
                next_cell_cost = float(costs[ny][nx])
                if not math.isfinite(next_cell_cost):
                    continue
                edge_cost = step_len * 0.5 * (current_cell_cost + next_cell_cost)
                tentative = current_cost + edge_cost
                neighbor = (nx, ny)
                if tentative + 1e-12 < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    heapq.heappush(open_heap, (tentative + heuristic(neighbor), tentative, neighbor))
        return [], float("inf")

    def _uav_astar_max_step_cost(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        K = int(self.uav_astar_grid)
        meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(
            device=device,
            dtype=dtype,
        ).clamp_min(1e-9)
        map_width_m = 2.0 * float(self.x_semidim) * meters_per_sim
        map_height_m = 2.0 * float(self.y_semidim) * meters_per_sim
        cell_size_m = torch.minimum(
            map_width_m / float(K),
            map_height_m / float(K),
        ).clamp_min(1e-6)
        max_step_m = max(float(self.drone_speed_mps) * float(self.sim_step_seconds), 1e-6)
        max_cell_steps = torch.as_tensor(max_step_m, device=device, dtype=dtype) / cell_size_m
        return (max_cell_steps * (1.0 + float(self.uav_astar_confidence_cost_alpha))).clamp_min(1e-6)

    def _uav_astar_path_costs_for_positions(
        self,
        positions: Tensor,
        *,
        confidence_grid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        costs = positions.new_zeros(positions.shape[0], self.n_drones)
        finite = torch.zeros(
            positions.shape[0],
            self.n_drones,
            dtype=torch.bool,
            device=positions.device,
        )
        if self.n_drones <= 0:
            return costs, finite

        with torch.no_grad():
            device = positions.device
            dtype = positions.dtype
            K = int(self.uav_astar_grid)
            pooled_confidence = self._uav_astar_pooled_confidence(
                device=device,
                dtype=dtype,
                confidence_grid=confidence_grid,
            )
            alpha = float(self.uav_astar_confidence_cost_alpha)
            gamma = float(self.uav_astar_confidence_cost_gamma)
            cell_cost = 1.0 + alpha * pooled_confidence.clamp(0.0, 1.0).pow(gamma)
            valid = self.uav_cleanup_target_valid.to(device=device)
            target_pos = self.uav_cleanup_target_pos.to(device=device, dtype=dtype)

            for env_idx in range(int(positions.shape[0])):
                for drone_idx in range(self.n_drones):
                    if not bool(valid[env_idx, drone_idx].detach().cpu().item()):
                        continue
                    start = self._uav_astar_position_to_cell(positions[env_idx, drone_idx], K)
                    goal = self._uav_astar_position_to_cell(target_pos[env_idx, drone_idx], K)
                    if cell_cost.ndim == 4:
                        route_cost = cell_cost[env_idx, drone_idx]
                    else:
                        route_cost = cell_cost[env_idx]
                    _, path_cost = self._uav_astar_plan(route_cost, start, goal)
                    if math.isfinite(path_cost):
                        costs[env_idx, drone_idx] = float(path_cost)
                        finite[env_idx, drone_idx] = True
        return costs, finite

    def _uav_astar_progress_reward(
        self,
        drone_pos: Tensor,
        *,
        confidence_grid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        reward = drone_pos.new_zeros(drone_pos.shape[0], self.n_drones)
        gate = torch.zeros_like(reward)
        progress_fraction = torch.zeros_like(reward)
        cost_before = torch.zeros_like(reward)
        cost_after = torch.zeros_like(reward)
        if self.n_drones == 0 or self.r_uav_astar_progress <= 0.0:
            return reward, gate, progress_fraction, cost_before, cost_after

        valid = self.metric_uav_cleanup_target_valid_by_drone.to(
            device=drone_pos.device,
            dtype=torch.bool,
        )
        target_value = self.metric_uav_cleanup_target_value_by_drone.to(
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        ).clamp(0.0, 1.0)
        pre_pos = self._pre_step_drone_pos.to(device=drone_pos.device, dtype=drone_pos.dtype)
        cost_before, before_finite = self._uav_astar_path_costs_for_positions(
            pre_pos,
            confidence_grid=confidence_grid,
        )
        cost_after, after_finite = self._uav_astar_path_costs_for_positions(
            drone_pos,
            confidence_grid=confidence_grid,
        )
        valid = valid & before_finite & after_finite
        raw_progress = cost_before - cost_after
        max_step_cost = self._uav_astar_max_step_cost(
            device=drone_pos.device,
            dtype=drone_pos.dtype,
        ).view(-1, 1)
        progress_fraction = torch.where(
            valid,
            (raw_progress / max_step_cost).clamp(0.0, 1.0),
            torch.zeros_like(raw_progress),
        )
        local_frontier_score = self._uav_local_frontier_score(
            drone_pos.device,
            drone_pos.dtype,
        )
        gate = (1.0 - local_frontier_score).clamp(0.0, 1.0) * valid.to(dtype=drone_pos.dtype)
        reward = (
            float(self.r_uav_astar_progress)
            * progress_fraction
            * target_value
            * gate
        )
        cost_before = torch.where(valid, cost_before, torch.zeros_like(cost_before))
        cost_after = torch.where(valid, cost_after, torch.zeros_like(cost_after))
        return reward, gate, progress_fraction, cost_before, cost_after

    def _refresh_uav_astar_routes(self, drone_pos: Tensor) -> None:
        if not bool(self.uav_astar_route_obs) or self.n_drones <= 0:
            return
        if not hasattr(self, "uav_astar_waypoint_valid"):
            return
        with torch.no_grad():
            self._refresh_uav_cleanup_target_assignments(drone_pos)
            device = drone_pos.device
            dtype = drone_pos.dtype
            current_step = self.step_count.to(device=device)
            stale = self._uav_astar_last_plan_step.to(device=device) != current_step
            if not bool(stale.any().detach().cpu().item()):
                return

            K = int(self.uav_astar_grid)
            pooled_confidence = self._uav_astar_pooled_confidence(device=device, dtype=dtype)
            alpha = float(self.uav_astar_confidence_cost_alpha)
            gamma = float(self.uav_astar_confidence_cost_gamma)
            cell_cost = 1.0 + alpha * pooled_confidence.clamp(0.0, 1.0).pow(gamma)
            meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(
                device=device,
                dtype=dtype,
            ).clamp_min(1e-9)
            replan_steps = int(self.uav_astar_route_replan_steps)
            reached_m = float(self.uav_astar_waypoint_reached_m)
            lookahead_m = float(self.uav_astar_waypoint_lookahead_m)
            normalizer = max(math.hypot(K - 1, K - 1) * (1.0 + alpha), 1e-6)
            env_indices = [
                int(value)
                for value in stale.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            ]
            for env_idx in env_indices:
                for drone_idx in range(self.n_drones):
                    target_valid = bool(
                        self.uav_cleanup_target_valid[env_idx, drone_idx].detach().cpu().item()
                    )
                    if not target_valid:
                        self.uav_astar_waypoint_valid[env_idx, drone_idx] = False
                        self.uav_astar_waypoint_pos[env_idx, drone_idx] = 0.0
                        self.uav_astar_waypoint_target_id[env_idx, drone_idx] = -1
                        self.uav_astar_waypoint_age[env_idx, drone_idx] = 0
                        self.uav_astar_path_cost_norm[env_idx, drone_idx] = 0.0
                        continue

                    target_id = int(self.uav_cleanup_target_id[env_idx, drone_idx].detach().cpu().item())
                    old_target_id = int(
                        self.uav_astar_waypoint_target_id[env_idx, drone_idx].detach().cpu().item()
                    )
                    waypoint_valid = bool(
                        self.uav_astar_waypoint_valid[env_idx, drone_idx].detach().cpu().item()
                    )
                    waypoint_age = int(
                        self.uav_astar_waypoint_age[env_idx, drone_idx].detach().cpu().item()
                    )
                    waypoint_distance_m = float("inf")
                    if waypoint_valid:
                        waypoint_distance_m = float(
                            (
                                self.uav_astar_waypoint_pos[env_idx, drone_idx].to(
                                    device=device,
                                    dtype=dtype,
                                )
                                - drone_pos[env_idx, drone_idx]
                            )
                            .norm()
                            .mul(meters_per_sim[env_idx])
                            .detach()
                            .cpu()
                            .item()
                        )
                    keep_existing = (
                        waypoint_valid
                        and old_target_id == target_id
                        and waypoint_age < replan_steps
                        and waypoint_distance_m > reached_m
                    )
                    if keep_existing:
                        self.uav_astar_waypoint_age[env_idx, drone_idx] += 1
                        continue

                    start = self._uav_astar_position_to_cell(drone_pos[env_idx, drone_idx], K)
                    goal = self._uav_astar_position_to_cell(
                        self.uav_cleanup_target_pos[env_idx, drone_idx].to(device=device, dtype=dtype),
                        K,
                    )
                    path, path_cost = self._uav_astar_plan(cell_cost[env_idx], start, goal)
                    if not path or not math.isfinite(path_cost):
                        self.uav_astar_waypoint_valid[env_idx, drone_idx] = False
                        self.uav_astar_waypoint_pos[env_idx, drone_idx] = 0.0
                        self.uav_astar_waypoint_target_id[env_idx, drone_idx] = -1
                        self.uav_astar_waypoint_age[env_idx, drone_idx] = 0
                        self.uav_astar_path_cost_norm[env_idx, drone_idx] = 0.0
                        continue

                    waypoint_cell = path[-1]
                    for cell in path[1:]:
                        candidate_pos = self._uav_astar_cell_center(
                            cell,
                            K=K,
                            device=device,
                            dtype=dtype,
                        )
                        candidate_distance_m = float(
                            (candidate_pos - drone_pos[env_idx, drone_idx])
                            .norm()
                            .mul(meters_per_sim[env_idx])
                            .detach()
                            .cpu()
                            .item()
                        )
                        if candidate_distance_m >= lookahead_m:
                            waypoint_cell = cell
                            break
                    waypoint_pos = self._uav_astar_cell_center(
                        waypoint_cell,
                        K=K,
                        device=device,
                        dtype=dtype,
                    )
                    self.uav_astar_waypoint_valid[env_idx, drone_idx] = True
                    self.uav_astar_waypoint_pos[env_idx, drone_idx] = waypoint_pos
                    self.uav_astar_waypoint_target_id[env_idx, drone_idx] = target_id
                    self.uav_astar_waypoint_age[env_idx, drone_idx] = 0
                    self.uav_astar_path_cost_norm[env_idx, drone_idx] = min(
                        max(path_cost / normalizer, 0.0),
                        1.0,
                    )
                self._uav_astar_last_plan_step[env_idx] = current_step[env_idx].to(
                    device=self._uav_astar_last_plan_step.device
                )

    def _uav_confidence_overlap_penalty(
        self,
        drone_pos: Tensor,
        *,
        visible: Tensor | None = None,
        previous: Tensor | None = None,
    ) -> Tensor:
        """Penalize spending footprint area over already high-confidence cells."""
        penalty = drone_pos.new_zeros(drone_pos.shape[0], self.n_drones)
        if (
            self.n_drones == 0
            or (
                self.r_uav_confidence_overlap <= 0.0
                and self.r_uav_team_confidence_overlap <= 0.0
            )
        ):
            if self.n_drones > 0:
                self.metric_uav_confidence_overlap_fraction_by_drone = torch.zeros_like(penalty)
                self.metric_uav_confidence_overlap_fraction = penalty.mean(dim=1)
                self.metric_uav_confidence_overlap_regret_by_drone = torch.zeros_like(penalty)
                self.metric_uav_confidence_overlap_regret = penalty.mean(dim=1)
            return penalty

        with torch.no_grad():
            if visible is None:
                _, visible = self._uav_cell_detection_probability(drone_pos)
            if previous is None:
                previous = self.uav_confidence_grid
            previous = previous.to(device=drone_pos.device, dtype=drone_pos.dtype).clamp(0.0, 1.0)
            threshold = float(self.uav_confidence_overlap_threshold)
            saturated = ((previous - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0)
            visible_f = visible.to(dtype=drone_pos.dtype)
            visible_cells = visible_f.sum(dim=(-1, -2)).clamp_min(1.0)
            if saturated.ndim == 3:
                saturated = saturated.unsqueeze(1)
            saturated_fraction = (
                visible_f * saturated
            ).sum(dim=(-1, -2)) / visible_cells
            regret = torch.zeros_like(saturated_fraction)
            if self.uav_confidence_overlap_mode == "opportunity_regret":
                opportunity_fraction = self.metric_uav_confidence_opportunity_fraction_by_drone.to(
                    device=drone_pos.device,
                    dtype=drone_pos.dtype,
                ).clamp(0.0, 1.0)
                best_gain = self.metric_uav_confidence_opportunity_best_gain_by_drone.to(
                    device=drone_pos.device,
                    dtype=drone_pos.dtype,
                )
                has_opportunity = best_gain > float(self.uav_confidence_opportunity_eps)
                allowed_regret = float(self.uav_confidence_overlap_allowed_regret)
                regret = torch.where(
                    has_opportunity,
                    (1.0 - opportunity_fraction - allowed_regret).clamp(0.0, 1.0),
                    torch.zeros_like(opportunity_fraction),
                )
            else:
                regret = torch.ones_like(saturated_fraction)
            penalty = -float(self.r_uav_confidence_overlap) * saturated_fraction * regret
            self.metric_uav_confidence_overlap_fraction_by_drone = saturated_fraction
            self.metric_uav_confidence_overlap_fraction = saturated_fraction.mean(dim=1)
            self.metric_uav_confidence_overlap_regret_by_drone = regret
            self.metric_uav_confidence_overlap_regret = regret.mean(dim=1)
            return penalty

    def _update_uav_confidence(
        self,
        drone_pos: Tensor,
        *,
        probability: Tensor | None = None,
        visible: Tensor | None = None,
        previous_by_drone: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update the optional probabilistic inspection-confidence map.

        When ``r_uav_confidence`` is positive, each UAV receives a reward for
        marginal confidence gain weighted toward cells whose current inspection
        confidence is still low. With the default zero scale this remains a
        diagnostics-only map.
        """
        reward = drone_pos.new_zeros(drone_pos.shape[0], self.n_drones)
        move_reward = torch.zeros_like(reward)
        if self.n_drones == 0:
            return reward, move_reward
        if not self._uav_confidence_active():
            return reward, move_reward

        with torch.no_grad():
            if probability is None or visible is None:
                probability, visible = self._uav_cell_detection_probability(drone_pos)
            previous = self.uav_confidence_grid.to(device=drone_pos.device, dtype=drone_pos.dtype).clamp(0.0, 1.0)
            miss_probability = (1.0 - probability).clamp(0.0, 1.0)
            miss_probability_all = miss_probability.prod(dim=1)
            updated = 1.0 - (1.0 - previous) * miss_probability_all
            updated = updated.clamp(0.0, 1.0)
            team_gain = (updated - previous).clamp(min=0.0)
            confidence_weight = (
                float(self.uav_confidence_eps)
                + (1.0 - previous).clamp(0.0, 1.0).pow(float(self.uav_confidence_gamma))
            )
            weighted_team_gain = confidence_weight * team_gain

            if previous_by_drone is not None:
                previous_individual = previous_by_drone.to(
                    device=drone_pos.device,
                    dtype=drone_pos.dtype,
                ).clamp(0.0, 1.0)
                confidence_weight_individual = (
                    float(self.uav_confidence_eps)
                    + (1.0 - previous_individual).clamp(0.0, 1.0).pow(
                        float(self.uav_confidence_gamma)
                    )
                )
                updated_individual = (
                    1.0
                    - (1.0 - previous_individual)
                    * (1.0 - probability).clamp(0.0, 1.0)
                ).clamp(0.0, 1.0)
                marginal = (updated_individual - previous_individual).clamp(min=0.0)
                confidence_weight_for_reward = confidence_weight_individual
                best_gain = self._uav_confidence_best_stencil_gain_by_drone(
                    previous_individual,
                    confidence_weight_individual,
                )
            else:
                if self.n_drones == 1:
                    miss_without = torch.ones_like(miss_probability)
                else:
                    ones = torch.ones_like(miss_probability[:, :1])
                    prefix = torch.cumprod(miss_probability, dim=1)
                    suffix = torch.flip(
                        torch.cumprod(torch.flip(miss_probability, dims=[1]), dim=1),
                        dims=[1],
                    )
                    miss_before = torch.cat((ones, prefix[:, :-1]), dim=1)
                    miss_after = torch.cat((suffix[:, 1:], ones), dim=1)
                    miss_without = miss_before * miss_after
                confidence_without = 1.0 - (1.0 - previous).unsqueeze(1) * miss_without
                marginal = (updated.unsqueeze(1) - confidence_without).clamp(min=0.0)
                confidence_weight_for_reward = confidence_weight.unsqueeze(1)
                best_gain = self._uav_confidence_best_stencil_gain(previous, confidence_weight)
            marginal_gain_by_drone = marginal.mean(dim=(-1, -2))
            weighted_marginal_gain_by_drone = (
                confidence_weight_for_reward * marginal
            ).mean(dim=(-1, -2))
            reward = weighted_marginal_gain_by_drone * float(self.r_uav_confidence)
            opportunity_fraction = torch.where(
                best_gain > float(self.uav_confidence_opportunity_eps),
                weighted_marginal_gain_by_drone / best_gain.clamp_min(float(self.uav_confidence_opportunity_eps)),
                torch.zeros_like(weighted_marginal_gain_by_drone),
            ).clamp(0.0, 1.0)
            sim_units_per_meter = self.terrain_sim_units_per_meter.to(
                device=drone_pos.device,
                dtype=drone_pos.dtype,
            ).clamp_min(1e-9)
            meters_per_sim = 1.0 / sim_units_per_meter
            displacement_m = (drone_pos - self._pre_step_drone_pos).norm(dim=-1) * meters_per_sim.view(-1, 1)
            max_step_m = max(float(self.drone_speed_mps) * float(self.sim_step_seconds), 1e-6)
            movement_fraction = (displacement_m / max_step_m).clamp(0.0, 1.0)
            move_reward = (
                float(self.r_uav_confidence_move)
                * movement_fraction
                * opportunity_fraction
            )

            visible_cells = visible.float().sum(dim=(-1, -2)).clamp_min(1.0)
            step_detection_probability_by_drone = probability.sum(dim=(-1, -2)) / visible_cells

            self.uav_confidence_grid.copy_(updated.to(dtype=self.uav_confidence_grid.dtype))
            self._update_comm_agent_confidence_from_probability(probability)
            self._invalidate_uav_runtime_caches()
            self.metric_reward_uav_confidence_by_drone = reward
            self.metric_reward_uav_confidence = reward.sum(dim=1)
            self.metric_reward_uav_confidence_move_by_drone = move_reward
            self.metric_reward_uav_confidence_move = move_reward.sum(dim=1)
            self.metric_uav_confidence_mean = updated.mean(dim=(1, 2))
            self.metric_uav_confidence_gain = team_gain.mean(dim=(1, 2))
            self.metric_uav_confidence_gain_by_drone = marginal_gain_by_drone
            self.metric_uav_weighted_confidence_gain = weighted_team_gain.mean(dim=(1, 2))
            self.metric_uav_weighted_confidence_gain_by_drone = weighted_marginal_gain_by_drone
            self.metric_uav_confidence_opportunity_fraction_by_drone = opportunity_fraction
            self.metric_uav_confidence_opportunity_fraction = opportunity_fraction.mean(dim=1)
            self.metric_uav_confidence_opportunity_best_gain_by_drone = best_gain
            self.metric_uav_confidence_opportunity_best_gain = best_gain.mean(dim=1)
            self.metric_uav_confidence_low_fraction = (updated < 0.50).float().mean(dim=(1, 2))
            self.metric_uav_confidence_high_fraction = (updated >= 0.80).float().mean(dim=(1, 2))
            self.metric_uav_step_detection_probability_by_drone = step_detection_probability_by_drone
            self.metric_uav_step_detection_probability = step_detection_probability_by_drone.mean(dim=1)
        return reward, move_reward

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

    def _coverage_reward(
        self,
        drone_pos: Tensor,
        *,
        known_coverage: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Per-drone fraction of the map newly covered by camera footprints.

        All drone claims are calculated before updating ``coverage_grid``.
        Simultaneous overlap is split equally, so total team credit across an
        episode cannot exceed 1.0 regardless of map resolution.
        """
        B = drone_pos.shape[0]
        new_per_drone = torch.zeros(B, self.n_drones, device=drone_pos.device)
        overlap_fraction = torch.zeros_like(new_per_drone)
        outside_footprint_fraction = torch.zeros_like(new_per_drone)
        inter_uav_overlap_fraction = torch.zeros_like(new_per_drone)
        opportunity_fraction = torch.zeros_like(new_per_drone)
        opportunity_cells = torch.zeros_like(new_per_drone)
        opportunity_available_fraction = torch.zeros_like(new_per_drone)
        if self.n_drones == 0:
            return (
                new_per_drone,
                overlap_fraction,
                outside_footprint_fraction,
                inter_uav_overlap_fraction,
                opportunity_fraction,
                opportunity_cells,
                opportunity_available_fraction,
            )
        if known_coverage is None:
            known_coverage = self.coverage_grid.unsqueeze(1).expand(-1, self.n_drones, -1, -1)
        elif known_coverage.ndim == 3:
            known_coverage = known_coverage.unsqueeze(1).expand(-1, self.n_drones, -1, -1)
        known_coverage = known_coverage.to(device=drone_pos.device, dtype=torch.bool)
        # The coverage grid is updated below even when the coverage reward is
        # off, because it can also feed the team-coverage observation.
        if (
            self.r_coverage <= 0.0
            and self.r_uav_move_coverage <= 0.0
            and not (
                self.r_uav_inefficient_move > 0.0
                and self.uav_inefficient_move_source == "coverage"
            )
            and self.r_uav_coverage_threshold <= 0.0
            and self.coverage_obs_grid <= 0
            and self.local_coverage_obs_grid <= 0
            and not self.uav_frontier_obs
            and self.r_uav_frontier_alignment <= 0.0
            and self.r_uav_overlap <= 0.0
            and self.r_uav_inter_uav_overlap <= 0.0
            and self.r_uav_outside_footprint <= 0.0
        ):
            return (
                new_per_drone,
                overlap_fraction,
                outside_footprint_fraction,
                inter_uav_overlap_fraction,
                opportunity_fraction,
                opportunity_cells,
                opportunity_available_fraction,
            )

        G = int(known_coverage.shape[-1])
        xs, ys, _, _, _, cell_width, cell_height, _ = self._uav_grid_geometry(
            drone_pos.device,
            drone_pos.dtype,
            grid_size=G,
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
        already_covered = claims & known_coverage
        overlap_fraction = already_covered.float().sum(dim=(-1, -2)) / footprint_cells.float()
        team_claims = claims.any(dim=1)
        newly_known_by_drone = claims & ~known_coverage
        claim_count = claims.sum(dim=1).clamp_min(1)
        if self.n_drones > 1:
            inter_uav_overlap_fraction = (
                (claims & (claim_count.unsqueeze(1) > 1)).float().sum(dim=(-1, -2))
                / footprint_cells.float()
            )
        split_credit = (
            claims.float()
            * newly_known_by_drone.float()
            / claim_count.unsqueeze(1)
        )
        new_credit_cells = split_credit.sum(dim=(-1, -2))

        max_step_sim = (
            float(self.drone_speed_mps)
            * float(self.sim_step_seconds)
            * self.terrain_sim_units_per_meter.to(
                device=drone_pos.device,
                dtype=drone_pos.dtype,
            ).clamp_min(1e-9)
        ).view(B, 1, 1, 1)
        pre_pos = self._pre_step_drone_pos.to(device=drone_pos.device, dtype=drone_pos.dtype)
        reach_dx = (
            (xs.view(1, 1, 1, G) - pre_pos[..., X].view(B, self.n_drones, 1, 1)).abs()
            - cell_width / 2.0
        ).clamp_min(0.0)
        reach_dy = (
            (ys.view(1, 1, G, 1) - pre_pos[..., Y].view(B, self.n_drones, 1, 1)).abs()
            - cell_height / 2.0
        ).clamp_min(0.0)
        reachable_footprint = footprint + max_step_sim
        reachable_claims = reach_dx.square() + reach_dy.square() <= reachable_footprint.square()
        reachable_total_cells = reachable_claims.float().sum(dim=(-1, -2))
        opportunity_cells = (
            reachable_claims & ~known_coverage
        ).float().sum(dim=(-1, -2))
        opportunity_fraction = torch.where(
            opportunity_cells > 0.0,
            new_credit_cells / opportunity_cells.clamp_min(1e-6),
            torch.zeros_like(new_credit_cells),
        ).clamp(0.0, 1.0)
        opportunity_available_fraction = torch.where(
            reachable_total_cells > 0.0,
            opportunity_cells / reachable_total_cells.clamp_min(1e-6),
            torch.zeros_like(opportunity_cells),
        ).clamp(0.0, 1.0)

        self.coverage_grid |= team_claims
        self._update_comm_agent_coverage_from_claims(claims)
        self._invalidate_uav_runtime_caches()
        return (
            new_credit_cells / float(G * G),
            overlap_fraction,
            outside_footprint_fraction,
            inter_uav_overlap_fraction,
            opportunity_fraction,
            opportunity_cells,
            opportunity_available_fraction,
        )

    def _uav_coverage_reward(self, map_fraction: Tensor, opportunity_fraction: Tensor) -> Tensor:
        """Per-UAV coverage reward using the selected normalization mode."""
        if self.n_drones == 0 or self.r_coverage <= 0.0:
            return torch.zeros_like(map_fraction)
        if self.uav_coverage_normalization == "opportunity":
            ratio = opportunity_fraction.clamp(
                min=0.0,
                max=self.uav_coverage_opportunity_cap,
            )
        else:
            ratio = map_fraction
        return ratio * self.r_coverage

    def _uav_move_coverage_reward(
        self,
        drone_pos: Tensor,
        coverage_new: Tensor,
        opportunity_fraction: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reward actual UAV displacement only when it produces new coverage."""
        if self.n_drones == 0:
            empty = torch.zeros(self.world.batch_dim, 0, device=drone_pos.device)
            return empty, empty, empty
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(drone_pos.device).clamp_min(1e-9)
        meters_per_sim = 1.0 / sim_units_per_meter
        displacement_sim = (drone_pos - self._pre_step_drone_pos).norm(dim=-1)
        displacement_m = displacement_sim * meters_per_sim.view(-1, 1)
        coverage_grid_size = int(self.coverage_grid.shape[-1])
        coverage_new_cells = coverage_new * float(coverage_grid_size * coverage_grid_size)
        if self.uav_move_coverage_normalization == "opportunity":
            if opportunity_fraction is None:
                opportunity_fraction = torch.zeros_like(coverage_new)
            max_step_m = max(float(self.drone_speed_mps) * float(self.sim_step_seconds), 1e-6)
            distance_fraction = (displacement_m / max_step_m).clamp(min=0.0, max=1.0)
            reward_base = distance_fraction * opportunity_fraction.clamp(min=0.0, max=1.0)
        else:
            reward_base = displacement_m * coverage_new_cells
        reward = (reward_base * self.r_uav_move_coverage).clamp(max=self.r_uav_move_coverage_cap)
        return reward, displacement_m, coverage_new_cells

    def _uav_inefficient_move_penalty(
        self,
        displacement_m: Tensor,
        coverage_opportunity_fraction: Tensor,
        confidence_opportunity_fraction: Tensor,
    ) -> Tensor:
        """Penalize motion that captures little available search opportunity."""
        if self.n_drones == 0:
            return torch.zeros(self.world.batch_dim, 0, device=displacement_m.device)
        if self.r_uav_inefficient_move <= 0.0:
            return torch.zeros_like(displacement_m)

        if self.uav_inefficient_move_source == "confidence":
            opportunity = confidence_opportunity_fraction
        else:
            opportunity = coverage_opportunity_fraction
        opportunity = opportunity.to(
            device=displacement_m.device,
            dtype=displacement_m.dtype,
        ).clamp(0.0, 1.0)
        max_step_m = max(float(self.drone_speed_mps) * float(self.sim_step_seconds), 1e-6)
        movement_fraction = (displacement_m / max_step_m).clamp(0.0, 1.0)
        inefficiency = movement_fraction * (1.0 - opportunity)
        return -float(self.r_uav_inefficient_move) * inefficiency

    def _uav_frontier_obs_dim(self) -> int:
        if self.uav_frontier_mode == "local_global":
            return 8
        if self.uav_frontier_mode == "sector_topk":
            return 4 * int(self.uav_frontier_top_k)
        return 4

    def _uav_frontier_cell_scores(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        coverage_grid: Tensor | None = None,
        confidence_grid: Tensor | None = None,
    ) -> Tensor:
        """Cell mass used by the UAV frontier observation/reward."""
        if self.uav_frontier_source == "confidence":
            source = self.uav_confidence_grid if confidence_grid is None else confidence_grid
            confidence = source.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
            confidence_weight = (
                float(self.uav_confidence_eps)
                + uncertainty.pow(float(self.uav_confidence_gamma))
            )
            return (confidence_weight * uncertainty).clamp(min=0.0)
        source = self.coverage_grid if coverage_grid is None else coverage_grid
        covered = source.to(device=device)
        if covered.dtype == torch.bool:
            return (~covered).to(dtype=dtype)
        return (1.0 - covered.to(dtype=dtype).clamp(0.0, 1.0)).clamp(min=0.0)

    def _resize_uav_score_grid(self, scores: Tensor, grid_size: int) -> Tensor:
        grid_size = int(grid_size)
        if grid_size <= 0 or grid_size == int(scores.shape[-1]):
            return scores
        import torch.nn.functional as F

        return F.adaptive_avg_pool2d(
            scores.unsqueeze(1),
            (grid_size, grid_size),
        ).squeeze(1)

    def _uav_frontier_features_for_positions(
        self,
        positions: Tensor,
        *,
        coverage_grid: Tensor | None = None,
        confidence_grid: Tensor | None = None,
    ) -> Tensor:
        """UAV frontier features for the configured frontier mode."""
        if self.uav_frontier_mode == "local_global":
            return self._uav_frontier_local_global_features_for_positions(
                positions,
                coverage_grid=coverage_grid,
                confidence_grid=confidence_grid,
            )
        if self.uav_frontier_mode == "sector_topk":
            return self._uav_frontier_sector_topk_features_for_positions(
                positions,
                coverage_grid=coverage_grid,
                confidence_grid=confidence_grid,
            )
        return self._uav_frontier_centroid_features_for_positions(
            positions,
            coverage_grid=coverage_grid,
            confidence_grid=confidence_grid,
        )

    def _uav_frontier_cache_signature(self) -> tuple:
        coverage_version = getattr(self.coverage_grid, "_version", 0)
        confidence_version = getattr(self.uav_confidence_grid, "_version", 0)
        return (
            self.uav_frontier_mode,
            self.uav_frontier_source,
            float(self.uav_frontier_obs_radius_m),
            int(self.uav_frontier_sectors),
            int(self.uav_frontier_top_k),
            int(self.uav_frontier_global_grid),
            bool(self.uav_frontier_ownership),
            coverage_version,
            confidence_version,
        )

    def _cached_uav_frontier_features_for_positions(
        self,
        cache_name: str,
        positions: Tensor,
    ) -> Tensor:
        if not hasattr(self, "_uav_frontier_feature_cache"):
            self._uav_frontier_feature_cache = {}
        key = (
            cache_name,
            self._uav_frontier_cache_signature(),
            positions.device.type,
            positions.device.index,
            str(positions.dtype),
            tuple(positions.shape),
        )
        cached = self._uav_frontier_feature_cache.get(key)
        if cached is not None:
            return cached
        features = self._uav_frontier_features_for_positions(positions)
        self._uav_frontier_feature_cache[key] = features
        return features

    def _pre_step_uav_frontier_features(self) -> Tensor:
        if self._comms_maps_enabled() and self.n_drones > 0:
            features = []
            for drone_idx in range(self.n_drones):
                features.append(self._uav_frontier_features_for_positions(
                    self._pre_step_drone_pos[:, drone_idx : drone_idx + 1],
                    coverage_grid=self._drone_coverage_grid_for_observation(drone_idx),
                    confidence_grid=self._drone_confidence_grid_for_observation(drone_idx),
                )[:, 0])
            return torch.stack(features, dim=1)
        return self._cached_uav_frontier_features_for_positions(
            "decision",
            self._pre_step_drone_pos,
        )

    def _current_uav_frontier_features(self) -> Tensor:
        drone_pos = torch.stack(
            [drone.state.pos for drone in self.world.agents[: self.n_drones]],
            dim=1,
        )
        if self._comms_maps_enabled():
            features = []
            for drone_idx in range(self.n_drones):
                features.append(self._uav_frontier_features_for_positions(
                    drone_pos[:, drone_idx : drone_idx + 1],
                    coverage_grid=self._drone_coverage_grid_for_observation(drone_idx),
                    confidence_grid=self._drone_confidence_grid_for_observation(drone_idx),
                )[:, 0])
            return torch.stack(features, dim=1)
        return self._cached_uav_frontier_features_for_positions("decision", drone_pos)

    def _uav_frontier_centroid_features_for_positions(
        self,
        positions: Tensor,
        *,
        coverage_grid: Tensor | None = None,
        confidence_grid: Tensor | None = None,
    ) -> Tensor:
        """Direction, distance, and strength of nearby uncovered coverage mass."""
        if positions.ndim != 3:
            raise ValueError("positions must have shape [B, N, 2]")
        B, N, _ = positions.shape
        out = torch.zeros(B, N, 4, device=positions.device, dtype=positions.dtype)
        if B == 0 or N == 0:
            return out

        frontier_scores = self._uav_frontier_cell_scores(
            device=positions.device,
            dtype=positions.dtype,
            coverage_grid=coverage_grid,
            confidence_grid=confidence_grid,
        )
        G = int(frontier_scores.shape[-1])
        xs, ys, _, _, _, _, _, cell_area = self._uav_grid_geometry(
            positions.device,
            positions.dtype,
            grid_size=G,
        )
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(
            device=positions.device,
            dtype=positions.dtype,
        ).clamp_min(1e-9)
        radius_sim = (float(self.uav_frontier_obs_radius_m) * sim_units_per_meter).clamp_min(1e-9)
        for env_idx in range(B):
            radius = radius_sim[env_idx]
            ideal_cells = (
                math.pi * float(radius.detach().cpu().item()) ** 2 / cell_area
            )
            ideal_cells = max(ideal_cells, 1.0)
            env_scores = frontier_scores[env_idx]
            for item_idx in range(N):
                pos = positions[env_idx, item_idx]
                dx = xs.view(1, G) - pos[X]
                dy = ys.view(G, 1) - pos[Y]
                in_radius = dx.square() + dy.square() <= radius.square()
                useful_weight = torch.where(
                    in_radius,
                    env_scores,
                    torch.zeros_like(env_scores),
                )
                count = useful_weight.sum()
                if float(count.detach().cpu().item()) <= 0.0:
                    continue
                vec_x = (dx * useful_weight).sum() / count
                vec_y = (dy * useful_weight).sum() / count
                distance = torch.sqrt(vec_x.square() + vec_y.square())
                out[env_idx, item_idx, 0] = (vec_x / radius).clamp(-1.0, 1.0)
                out[env_idx, item_idx, 1] = (vec_y / radius).clamp(-1.0, 1.0)
                out[env_idx, item_idx, 2] = (distance / radius).clamp(0.0, 1.0)
                out[env_idx, item_idx, 3] = (count / ideal_cells).clamp(0.0, 1.0)
        return out

    def _uav_frontier_sector_topk_features_for_positions(
        self,
        positions: Tensor,
        *,
        coverage_grid: Tensor | None = None,
        confidence_grid: Tensor | None = None,
    ) -> Tensor:
        """Top-k uncovered sector candidates with optional team ownership weighting.

        Each candidate is encoded as [unit_dx, unit_dy, distance_norm, score].
        The score is ownership-weighted uncovered cell mass divided by the ideal
        sector area; distance is the weighted mean distance of those cells.
        """
        if positions.ndim != 3:
            raise ValueError("positions must have shape [B, N, 2]")
        B, N, _ = positions.shape
        top_k = int(self.uav_frontier_top_k)
        sectors = int(self.uav_frontier_sectors)
        out = torch.zeros(B, N, top_k * 4, device=positions.device, dtype=positions.dtype)
        if B == 0 or N == 0:
            return out

        frontier_scores = self._uav_frontier_cell_scores(
            device=positions.device,
            dtype=positions.dtype,
            coverage_grid=coverage_grid,
            confidence_grid=confidence_grid,
        )
        G = int(frontier_scores.shape[-1])
        _, _, x_grid, y_grid, _, _, _, cell_area = self._uav_grid_geometry(
            positions.device,
            positions.dtype,
            grid_size=G,
        )
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(
            device=positions.device,
            dtype=positions.dtype,
        ).clamp_min(1e-9)
        radius_sim = (float(self.uav_frontier_obs_radius_m) * sim_units_per_meter).clamp_min(1e-9)
        sector_width, sector_unit = self._uav_sector_geometry(sectors, positions.device, positions.dtype)

        dx = x_grid.view(1, 1, 1, G) - positions[..., X].view(B, N, 1, 1)
        dy = y_grid.view(1, 1, G, 1) - positions[..., Y].view(B, N, 1, 1)
        dist_sq = dx.square() + dy.square()
        dist = dist_sq.sqrt()
        radius = radius_sim.view(B, 1, 1, 1)
        useful_any = (dist_sq <= radius.square()) & (frontier_scores.unsqueeze(1) > 1e-9)

        if self.uav_frontier_ownership and N > 1:
            own_mask = torch.eye(N, device=positions.device, dtype=torch.bool).view(1, N, N, 1, 1)
            inf = torch.full((), float("inf"), device=positions.device, dtype=positions.dtype)
            other_dist = dist.unsqueeze(1).masked_fill(own_mask, inf).amin(dim=2)
            ownership = other_dist / (dist + other_dist + 1e-9)
        else:
            ownership = torch.ones_like(dist)

        angles = torch.atan2(dy, dx)
        sector_index = torch.floor((angles + math.pi) / sector_width).long().clamp(0, sectors - 1)
        weighted = useful_any.to(dtype=positions.dtype) * frontier_scores.unsqueeze(1) * ownership

        sector_ids = torch.arange(sectors, device=positions.device).view(1, 1, sectors, 1, 1)
        sector_mask = sector_index.unsqueeze(2) == sector_ids
        sector_weight = (weighted.unsqueeze(2) * sector_mask).sum(dim=(-2, -1))
        sector_dist_sum = ((dist * weighted).unsqueeze(2) * sector_mask).sum(dim=(-2, -1))
        nonzero = sector_weight > 0.0
        safe_weight = sector_weight.clamp_min(1e-9)
        sector_distances = torch.where(
            nonzero,
            (sector_dist_sum / safe_weight / radius_sim.view(B, 1, 1)).clamp(0.0, 1.0),
            torch.zeros_like(sector_weight),
        )
        ideal_sector_cells = (
            math.pi * radius_sim.square() / cell_area / float(sectors)
        ).clamp_min(1.0).view(B, 1, 1)
        sector_scores = torch.where(
            nonzero,
            (sector_weight / ideal_sector_cells).clamp(0.0, 1.0),
            torch.zeros_like(sector_weight),
        )

        top_scores, top_indices = torch.topk(sector_scores, k=top_k, largest=True, sorted=True)
        top_distances = torch.gather(sector_distances, 2, top_indices)
        top_dirs = sector_unit[top_indices]
        features = torch.cat(
            (
                top_dirs,
                top_distances.unsqueeze(-1),
                top_scores.unsqueeze(-1),
            ),
            dim=-1,
        )
        features = torch.where(top_scores.unsqueeze(-1) > 1e-9, features, torch.zeros_like(features))
        return features.reshape(B, N, top_k * 4)

    def _uav_frontier_best_sector_features(
        self,
        positions: Tensor,
        scores: Tensor,
        *,
        x_grid: Tensor,
        y_grid: Tensor,
        sector_width: float,
        sectors: int,
        radius: Tensor | None,
        distance_scale: Tensor | float,
        ideal_cells: Tensor | float,
        ownership: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Batched best-sector aggregation for confidence frontier candidates."""
        B, N, _ = positions.shape
        G = int(scores.shape[-1])
        device = positions.device
        dtype = positions.dtype

        def broadcast_candidate_value(value: Tensor | float) -> Tensor:
            if torch.is_tensor(value):
                value_tensor = value.to(device=device, dtype=dtype)
            else:
                value_tensor = torch.as_tensor(value, device=device, dtype=dtype)
            if value_tensor.ndim == 0:
                return value_tensor.view(1, 1, 1)
            if value_tensor.ndim == 1:
                return value_tensor.view(B, 1, 1)
            if value_tensor.ndim == 2:
                return value_tensor.view(B, N, 1)
            return value_tensor.view(B, N, 1)

        dx = x_grid.view(1, 1, 1, G) - positions[..., X].view(B, N, 1, 1)
        dy = y_grid.view(1, 1, G, 1) - positions[..., Y].view(B, N, 1, 1)
        dist_sq = dx.square() + dy.square()

        useful_scores = scores.unsqueeze(1).to(device=device, dtype=dtype)
        if ownership is not None:
            useful_scores = useful_scores * ownership.to(device=device, dtype=dtype)
        if radius is None:
            useful_any = useful_scores > 1e-9
        else:
            if torch.is_tensor(radius):
                radius_tensor = radius.to(device=device, dtype=dtype)
            else:
                radius_tensor = torch.as_tensor(radius, device=device, dtype=dtype)
            if radius_tensor.ndim == 0:
                radius_sq = radius_tensor.square().view(1, 1, 1, 1)
            elif radius_tensor.ndim == 1:
                radius_sq = radius_tensor.square().view(B, 1, 1, 1)
            else:
                radius_sq = radius_tensor.square().view(B, N, 1, 1)
            useful_any = (dist_sq <= radius_sq) & (useful_scores > 1e-9)

        angles = torch.atan2(dy, dx)
        sector_index = torch.floor((angles + math.pi) / sector_width).long().clamp(0, sectors - 1)
        weighted = useful_any.to(dtype=dtype) * useful_scores

        flat_index = sector_index.reshape(B * N, G * G)
        flat_weight = weighted.reshape(B * N, G * G)
        sector_weight = torch.zeros(B * N, sectors, device=device, dtype=dtype)
        sector_vec_x = torch.zeros_like(sector_weight)
        sector_vec_y = torch.zeros_like(sector_weight)
        sector_weight.scatter_add_(1, flat_index, flat_weight)
        sector_vec_x.scatter_add_(1, flat_index, (dx * weighted).reshape(B * N, G * G))
        sector_vec_y.scatter_add_(1, flat_index, (dy * weighted).reshape(B * N, G * G))

        sector_weight = sector_weight.view(B, N, sectors)
        sector_vec_x = sector_vec_x.view(B, N, sectors)
        sector_vec_y = sector_vec_y.view(B, N, sectors)
        nonzero = sector_weight > 0.0
        safe_weight = sector_weight.clamp_min(1e-9)
        vec_x = sector_vec_x / safe_weight
        vec_y = sector_vec_y / safe_weight
        vec_norm = torch.sqrt(vec_x.square() + vec_y.square()).clamp_min(1e-9)
        sector_dirs = torch.stack((vec_x / vec_norm, vec_y / vec_norm), dim=-1)
        sector_dirs = torch.where(nonzero.unsqueeze(-1), sector_dirs, torch.zeros_like(sector_dirs))
        sector_distances = torch.where(
            nonzero,
            (vec_norm / broadcast_candidate_value(distance_scale).clamp_min(1e-9)).clamp(0.0, 1.0),
            torch.zeros_like(sector_weight),
        )
        sector_scores = torch.where(
            nonzero,
            (sector_weight / broadcast_candidate_value(ideal_cells).clamp_min(1e-9)).clamp(0.0, 1.0),
            torch.zeros_like(sector_weight),
        )

        best_score, best_sector = sector_scores.max(dim=-1)
        has_score = best_score > 1e-9
        gather_idx = best_sector.unsqueeze(-1)
        best_dirs = torch.gather(
            sector_dirs,
            2,
            best_sector.view(B, N, 1, 1).expand(B, N, 1, 2),
        ).squeeze(2)
        best_dist = torch.gather(sector_distances, 2, gather_idx).squeeze(-1)
        features = torch.cat(
            (best_dirs, best_dist.unsqueeze(-1), best_score.unsqueeze(-1)),
            dim=-1,
        )
        features = torch.where(has_score.unsqueeze(-1), features, torch.zeros_like(features))
        selected_mask = (
            useful_any
            & (sector_index == best_sector.view(B, N, 1, 1))
            & has_score.view(B, N, 1, 1)
        )
        return features, selected_mask

    def _uav_frontier_local_best_sector_features(
        self,
        positions: Tensor,
        scores: Tensor,
        *,
        xs: Tensor,
        ys: Tensor,
        sector_width: float,
        sectors: int,
        radius: Tensor,
        ideal_cells: Tensor | float,
    ) -> Tensor:
        """Best local frontier sector using only cells in a radius-sized patch."""
        B, N, _ = positions.shape
        G = int(scores.shape[-1])
        device = positions.device
        dtype = positions.dtype
        if B == 0 or N == 0:
            return torch.zeros(B, N, 4, device=device, dtype=dtype)

        _, _, _, _, _, cell_width, cell_height, _ = self._uav_grid_geometry(
            device,
            dtype,
            grid_size=G,
        )
        radius_tensor = radius.to(device=device, dtype=dtype)
        max_radius = float(radius_tensor.detach().max().cpu().item()) if radius_tensor.numel() > 0 else 0.0
        # Add two cells to cover positions near cell boundaries. Extra cells are
        # masked by the exact radius check, so this preserves full-grid semantics.
        rx = min(G - 1, int(math.ceil(max_radius / max(float(cell_width), 1e-12))) + 2)
        ry = min(G - 1, int(math.ceil(max_radius / max(float(cell_height), 1e-12))) + 2)

        offset_x = torch.arange(-rx, rx + 1, device=device).view(1, 1, 1, -1)
        offset_y = torch.arange(-ry, ry + 1, device=device).view(1, 1, -1, 1)
        center_gx, center_gy = self._positions_to_grid(positions, grid_size=G)
        gx_raw = center_gx.view(B, N, 1, 1) + offset_x
        gy_raw = center_gy.view(B, N, 1, 1) + offset_y
        valid = (gx_raw >= 0) & (gx_raw < G) & (gy_raw >= 0) & (gy_raw < G)
        gx = gx_raw.clamp(0, G - 1)
        gy = gy_raw.clamp(0, G - 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1)

        score_patch = scores.to(device=device, dtype=dtype)[batch_idx, gy, gx]
        x_patch = xs[gx].expand_as(score_patch)
        y_patch = ys[gy].expand_as(score_patch)
        dx = x_patch - positions[..., X].view(B, N, 1, 1)
        dy = y_patch - positions[..., Y].view(B, N, 1, 1)
        dist_sq = dx.square() + dy.square()
        radius_sq = radius_tensor.square().view(B, 1, 1, 1)
        useful_any = valid & (dist_sq <= radius_sq) & (score_patch > 1e-9)

        angles = torch.atan2(dy, dx)
        sector_index = torch.floor((angles + math.pi) / sector_width).long().clamp(0, sectors - 1)
        weighted = useful_any.to(dtype=dtype) * score_patch

        flat_index = sector_index.reshape(B * N, -1)
        flat_weight = weighted.reshape(B * N, -1)
        sector_weight = torch.zeros(B * N, sectors, device=device, dtype=dtype)
        sector_vec_x = torch.zeros_like(sector_weight)
        sector_vec_y = torch.zeros_like(sector_weight)
        sector_weight.scatter_add_(1, flat_index, flat_weight)
        sector_vec_x.scatter_add_(1, flat_index, (dx * weighted).reshape(B * N, -1))
        sector_vec_y.scatter_add_(1, flat_index, (dy * weighted).reshape(B * N, -1))

        sector_weight = sector_weight.view(B, N, sectors)
        sector_vec_x = sector_vec_x.view(B, N, sectors)
        sector_vec_y = sector_vec_y.view(B, N, sectors)
        nonzero = sector_weight > 0.0
        safe_weight = sector_weight.clamp_min(1e-9)
        vec_x = sector_vec_x / safe_weight
        vec_y = sector_vec_y / safe_weight
        vec_norm = torch.sqrt(vec_x.square() + vec_y.square()).clamp_min(1e-9)
        sector_dirs = torch.stack((vec_x / vec_norm, vec_y / vec_norm), dim=-1)
        sector_dirs = torch.where(nonzero.unsqueeze(-1), sector_dirs, torch.zeros_like(sector_dirs))
        sector_distances = torch.where(
            nonzero,
            (vec_norm / radius_tensor.view(B, 1, 1).clamp_min(1e-9)).clamp(0.0, 1.0),
            torch.zeros_like(sector_weight),
        )
        if torch.is_tensor(ideal_cells):
            ideal = ideal_cells.to(device=device, dtype=dtype)
        else:
            ideal = torch.as_tensor(ideal_cells, device=device, dtype=dtype)
        if ideal.ndim == 0:
            ideal = ideal.view(1, 1, 1)
        else:
            ideal = ideal.view(B, 1, 1)
        sector_scores = torch.where(
            nonzero,
            (sector_weight / ideal.clamp_min(1e-9)).clamp(0.0, 1.0),
            torch.zeros_like(sector_weight),
        )

        best_score, best_sector = sector_scores.max(dim=-1)
        has_score = best_score > 1e-9
        gather_idx = best_sector.unsqueeze(-1)
        best_dirs = torch.gather(
            sector_dirs,
            2,
            best_sector.view(B, N, 1, 1).expand(B, N, 1, 2),
        ).squeeze(2)
        best_dist = torch.gather(sector_distances, 2, gather_idx).squeeze(-1)
        features = torch.cat(
            (best_dirs, best_dist.unsqueeze(-1), best_score.unsqueeze(-1)),
            dim=-1,
        )
        return torch.where(has_score.unsqueeze(-1), features, torch.zeros_like(features))

    def _uav_frontier_nearest_other_ownership(
        self,
        positions: Tensor,
        selected_indices: Tensor,
        *,
        x_grid: Tensor,
        y_grid: Tensor,
    ) -> Tensor:
        """Ownership weight for selected UAVs against their nearest teammate."""
        B, N, _ = positions.shape
        G = int(x_grid.shape[-1])
        device = positions.device
        dtype = positions.dtype
        batch_idx = torch.arange(B, device=device)
        selected_pos = positions[batch_idx, selected_indices]

        self_dx = x_grid.view(1, 1, G) - selected_pos[:, X].view(B, 1, 1)
        self_dy = y_grid.view(1, G, 1) - selected_pos[:, Y].view(B, 1, 1)
        self_dist = (self_dx.square() + self_dy.square()).sqrt()

        other_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        other_mask[batch_idx, selected_indices] = False
        other_positions = positions[other_mask].view(B, N - 1, 2)
        other_dx = x_grid.view(1, 1, 1, G) - other_positions[..., X].view(B, N - 1, 1, 1)
        other_dy = y_grid.view(1, 1, G, 1) - other_positions[..., Y].view(B, N - 1, 1, 1)
        other_dist = (other_dx.square() + other_dy.square()).sqrt().amin(dim=1)
        return (other_dist / (self_dist + other_dist + 1e-9)).unsqueeze(1).to(dtype=dtype)

    def _uav_frontier_local_global_features_for_positions(
        self,
        positions: Tensor,
        *,
        coverage_grid: Tensor | None = None,
        confidence_grid: Tensor | None = None,
    ) -> Tensor:
        """Frontier with one tactical local and one diversified global candidate.

        Encodes ``[local_dx, local_dy, local_dist, local_score,
        global_dx, global_dy, global_dist, global_score]`` per UAV. The local
        candidate uses ``uav_frontier_obs_radius_m``. The global candidate uses
        the full configured frontier map and greedily suppresses already
        assigned sectors so drones receive different strategic directions when
        alternatives exist.
        """
        if positions.ndim != 3:
            raise ValueError("positions must have shape [B, N, 2]")
        B, N, _ = positions.shape
        out = torch.zeros(B, N, 8, device=positions.device, dtype=positions.dtype)
        if B == 0 or N == 0:
            return out

        sectors = int(self.uav_frontier_sectors)
        sim_units_per_meter = self.terrain_sim_units_per_meter.to(
            device=positions.device,
            dtype=positions.dtype,
        ).clamp_min(1e-9)
        local_radius_sim = (
            float(self.uav_frontier_obs_radius_m) * sim_units_per_meter
        ).clamp_min(1e-9)
        global_distance_scale = max(
            float(math.hypot(2.0 * self.x_semidim, 2.0 * self.y_semidim)),
            1e-9,
        )
        frontier_scores = self._uav_frontier_cell_scores(
            device=positions.device,
            dtype=positions.dtype,
            coverage_grid=coverage_grid,
            confidence_grid=confidence_grid,
        )
        local_G = int(frontier_scores.shape[-1])
        _, _, x_grid, y_grid, _, _, _, cell_area = self._uav_grid_geometry(
            positions.device,
            positions.dtype,
            grid_size=local_G,
        )
        sector_width, _ = self._uav_sector_geometry(sectors, positions.device, positions.dtype)
        local_ideal_cells = (
            math.pi * local_radius_sim.square() / cell_area / float(sectors)
        ).clamp_min(1.0)

        local_features = self._uav_frontier_local_best_sector_features(
            positions,
            frontier_scores,
            xs=x_grid.reshape(-1),
            ys=y_grid.reshape(-1),
            sector_width=sector_width,
            sectors=sectors,
            radius=local_radius_sim,
            ideal_cells=local_ideal_cells,
        )
        out[:, :, :4] = local_features

        global_scores = self._resize_uav_score_grid(frontier_scores, self.uav_frontier_global_grid)
        global_G = int(global_scores.shape[-1])
        _, _, global_x_grid, global_y_grid, _, _, _, _ = self._uav_grid_geometry(
            positions.device,
            positions.dtype,
            grid_size=global_G,
        )
        global_ideal_cells = max(float(global_G * global_G) / float(sectors), 1.0)
        remaining_scores = global_scores.clone()
        assignment_order = torch.argsort(out[:, :, 3], dim=1, stable=True)
        batch_idx = torch.arange(B, device=positions.device)
        for rank_idx in range(N):
            drone_idx = assignment_order[:, rank_idx]
            selected_positions = positions[batch_idx, drone_idx].unsqueeze(1)
            ownership = None
            if self.uav_frontier_ownership and N > 1:
                ownership = self._uav_frontier_nearest_other_ownership(
                    positions,
                    drone_idx,
                    x_grid=global_x_grid,
                    y_grid=global_y_grid,
                )

            global_feature, selected_mask = self._uav_frontier_best_sector_features(
                selected_positions,
                remaining_scores,
                x_grid=global_x_grid,
                y_grid=global_y_grid,
                sector_width=sector_width,
                sectors=sectors,
                radius=None,
                distance_scale=global_distance_scale,
                ideal_cells=global_ideal_cells,
                ownership=ownership,
            )
            out[batch_idx, drone_idx, 4:8] = global_feature[:, 0]
            remaining_scores = torch.where(
                selected_mask[:, 0],
                torch.zeros_like(remaining_scores),
                remaining_scores,
            )
        return out

    def _uav_frontier_alignment_reward(self, drone_pos: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reward clamped progress toward nearby uncovered coverage mass."""
        if self.n_drones == 0:
            empty = torch.zeros(self.world.batch_dim, 0, device=drone_pos.device)
            return empty, empty, empty, empty
        if not self.uav_frontier_obs and self.r_uav_frontier_alignment <= 0.0:
            empty = torch.zeros(self.world.batch_dim, self.n_drones, device=drone_pos.device)
            return empty, empty, empty, empty

        if self.uav_frontier_mode == "sector_topk":
            return self._uav_frontier_sector_topk_alignment_reward(drone_pos)
        if self.uav_frontier_mode == "local_global":
            return self._uav_frontier_local_global_alignment_reward(drone_pos)

        features = self._pre_step_uav_frontier_features()
        frontier_vec = features[..., :2]
        uncovered_ratio = features[..., 3]
        displacement = drone_pos - self._pre_step_drone_pos
        frontier_norm = frontier_vec.norm(dim=-1)
        displacement_norm = displacement.norm(dim=-1)
        denom = (frontier_norm * displacement_norm).clamp_min(1e-9)
        alignment = (frontier_vec * displacement).sum(dim=-1) / denom
        alignment = torch.where(
            (frontier_norm > 1e-6) & (displacement_norm > 1e-9),
            alignment.clamp(-1.0, 1.0),
            torch.zeros_like(alignment),
        )
        unit_frontier = frontier_vec / frontier_norm.clamp_min(1e-9).unsqueeze(-1)
        progress_sim = (displacement * unit_frontier).sum(dim=-1)
        max_step_sim = (
            float(self.drone_speed_mps)
            * float(self.sim_step_seconds)
            * self.terrain_sim_units_per_meter.to(device=drone_pos.device, dtype=drone_pos.dtype).clamp_min(1e-9)
        ).view(-1, 1)
        progress_fraction = torch.where(
            frontier_norm > 1e-6,
            (progress_sim / max_step_sim).clamp(0.0, 1.0),
            torch.zeros_like(progress_sim),
        )
        reward = (
            self.r_uav_frontier_alignment
            * progress_fraction
            * uncovered_ratio.clamp(0.0, 1.0)
        )
        return reward, alignment, progress_fraction, uncovered_ratio

    def _uav_frontier_sector_topk_alignment_reward(
        self,
        drone_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reward progress toward whichever top-k sector candidate was followed."""
        features = self._pre_step_uav_frontier_features()
        B, D, _ = features.shape
        top_k = int(self.uav_frontier_top_k)
        candidates = features.view(B, D, top_k, 4)
        candidate_vec = candidates[..., :2]
        candidate_score = candidates[..., 3].clamp(0.0, 1.0)
        displacement = (drone_pos - self._pre_step_drone_pos).unsqueeze(2)
        displacement_norm = displacement.norm(dim=-1)
        frontier_norm = candidate_vec.norm(dim=-1)
        unit_frontier = candidate_vec / frontier_norm.clamp_min(1e-9).unsqueeze(-1)
        alignment = (unit_frontier * displacement).sum(dim=-1) / displacement_norm.clamp_min(1e-9)
        alignment = torch.where(
            (frontier_norm > 1e-6) & (displacement_norm > 1e-9),
            alignment.clamp(-1.0, 1.0),
            torch.zeros_like(alignment),
        )
        progress_sim = (displacement * unit_frontier).sum(dim=-1)
        max_step_sim = (
            float(self.drone_speed_mps)
            * float(self.sim_step_seconds)
            * self.terrain_sim_units_per_meter.to(device=drone_pos.device, dtype=drone_pos.dtype).clamp_min(1e-9)
        ).view(-1, 1, 1)
        progress_fraction = torch.where(
            frontier_norm > 1e-6,
            (progress_sim / max_step_sim).clamp(0.0, 1.0),
            torch.zeros_like(progress_sim),
        )
        candidate_value = progress_fraction * candidate_score
        best_value, best_idx = candidate_value.max(dim=2)
        gather_idx = best_idx.unsqueeze(-1)
        best_alignment = torch.gather(alignment, 2, gather_idx).squeeze(-1)
        best_progress = torch.gather(progress_fraction, 2, gather_idx).squeeze(-1)
        best_score = torch.gather(candidate_score, 2, gather_idx).squeeze(-1)
        reward = self.r_uav_frontier_alignment * best_value
        return reward, best_alignment, best_progress, best_score

    def _uav_frontier_local_global_alignment_reward(
        self,
        drone_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reward local confidence search, or gated global relocation if local opportunity is weak."""
        features = self._pre_step_uav_frontier_features()
        B, D, _ = features.shape
        candidates = features.view(B, D, 2, 4)
        candidate_vec = candidates[..., :2]
        raw_score = candidates[..., 3].clamp(0.0, 1.0)
        local_score = raw_score[..., 0].unsqueeze(-1)
        candidate_score = torch.stack(
            (
                raw_score[..., 0],
                raw_score[..., 1] * (1.0 - local_score.squeeze(-1)).clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        displacement = (drone_pos - self._pre_step_drone_pos).unsqueeze(2)
        displacement_norm = displacement.norm(dim=-1)
        frontier_norm = candidate_vec.norm(dim=-1)
        unit_frontier = candidate_vec / frontier_norm.clamp_min(1e-9).unsqueeze(-1)
        alignment = (unit_frontier * displacement).sum(dim=-1) / displacement_norm.clamp_min(1e-9)
        alignment = torch.where(
            (frontier_norm > 1e-6) & (displacement_norm > 1e-9),
            alignment.clamp(-1.0, 1.0),
            torch.zeros_like(alignment),
        )
        progress_sim = (displacement * unit_frontier).sum(dim=-1)
        max_step_sim = (
            float(self.drone_speed_mps)
            * float(self.sim_step_seconds)
            * self.terrain_sim_units_per_meter.to(device=drone_pos.device, dtype=drone_pos.dtype).clamp_min(1e-9)
        ).view(-1, 1, 1)
        progress_fraction = torch.where(
            frontier_norm > 1e-6,
            (progress_sim / max_step_sim).clamp(0.0, 1.0),
            torch.zeros_like(progress_sim),
        )
        candidate_value = progress_fraction * candidate_score
        best_value, best_idx = candidate_value.max(dim=2)
        gather_idx = best_idx.unsqueeze(-1)
        best_alignment = torch.gather(alignment, 2, gather_idx).squeeze(-1)
        best_progress = torch.gather(progress_fraction, 2, gather_idx).squeeze(-1)
        best_score = torch.gather(candidate_score, 2, gather_idx).squeeze(-1)
        reward = self.r_uav_frontier_alignment * best_value
        return reward, best_alignment, best_progress, best_score

    def _uav_expected_overlap_fraction(self, displacement_m: Tensor) -> Tensor:
        """Expected overlap of consecutive circular footprints from actual motion."""
        if self.n_drones == 0:
            return torch.zeros(self.world.batch_dim, 0, device=displacement_m.device)
        footprint_sim = self._drone_camera_ranges().to(device=displacement_m.device)
        meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(displacement_m.device).clamp_min(1e-9)
        radius_m = footprint_sim * meters_per_sim.view(-1, 1)
        radius_safe = radius_m.clamp_min(1e-9)
        distance = displacement_m.clamp(min=0.0)
        ratio = (distance / (2.0 * radius_safe)).clamp(min=0.0, max=1.0)
        overlap_area = (
            2.0 * radius_safe.square() * torch.acos(ratio)
            - 0.5
            * distance
            * (4.0 * radius_safe.square() - distance.square()).clamp_min(0.0).sqrt()
        )
        fraction = (overlap_area / (math.pi * radius_safe.square())).clamp(min=0.0, max=1.0)
        return torch.where(
            (radius_m > 0.0) & (distance < 2.0 * radius_safe),
            fraction,
            torch.zeros_like(fraction),
        )

    def _uav_overlap_penalty(
        self,
        overlap_fraction: Tensor,
        expected_overlap_fraction: Tensor,
        opportunity_available_fraction: Tensor | None = None,
    ) -> Tensor:
        """Penalize only overlap above physics-expected footprint overlap plus slack."""
        if self.n_drones == 0:
            return torch.zeros(self.world.batch_dim, 0, device=overlap_fraction.device)
        if self.r_uav_overlap <= 0.0:
            return torch.zeros_like(overlap_fraction)
        allowed = min(max(float(self.uav_overlap_allowed), 0.0), 0.999)
        threshold = (expected_overlap_fraction + allowed).clamp(max=0.999)
        excess = (overlap_fraction - threshold).clamp(min=0.0)
        normalized = (excess / (1.0 - threshold).clamp_min(1e-6)).clamp(max=1.0)
        if self.uav_overlap_penalty_normalization == "opportunity":
            if opportunity_available_fraction is None:
                opportunity_available_fraction = torch.ones_like(normalized)
            normalized = normalized * opportunity_available_fraction.clamp(min=0.0, max=1.0)
        return -self.r_uav_overlap * normalized

    def _uav_inter_uav_overlap_penalty(self, inter_uav_overlap_fraction: Tensor) -> Tensor:
        """Penalize same-step footprint overlap between different UAVs."""
        if self.n_drones <= 1:
            return torch.zeros(self.world.batch_dim, self.n_drones, device=inter_uav_overlap_fraction.device)
        if self.r_uav_inter_uav_overlap <= 0.0:
            return torch.zeros_like(inter_uav_overlap_fraction)
        allowed = min(max(float(self.uav_inter_uav_overlap_allowed), 0.0), 0.999)
        excess = (inter_uav_overlap_fraction - allowed).clamp(min=0.0)
        normalized = (excess / (1.0 - allowed)).clamp(max=1.0)
        return -self.r_uav_inter_uav_overlap * normalized

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
        gx, gy = self._positions_to_grid(pos, grid_size=int(grid.shape[-1]))
        if env_indices is None:
            env_indices = torch.arange(pos.shape[0], device=pos.device)
        expand_shape = (pos.shape[0],) + (1,) * (gx.ndim - 1)
        b_idx = env_indices.view(expand_shape).expand_as(gx)
        return grid[b_idx, gy, gx]

    def _positions_to_grid(self, pos: Tensor, grid_size: int | None = None) -> tuple[Tensor, Tensor]:
        G = int(self.fire_grid_size if grid_size is None else grid_size)
        gx = ((pos[..., X] + self.x_semidim) / (2 * self.x_semidim) * G).clamp(
            0, G - 1
        ).long()
        gy = ((pos[..., Y] + self.y_semidim) / (2 * self.y_semidim) * G).clamp(
            0, G - 1
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
        self._sync_comm_agent_maps_for_observation(agent, comms_keep)
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
            if self.ugv_zero_uav_search_observations and not agent.is_drone:
                parts.append(torch.zeros(
                    self.world.batch_dim,
                    self.coverage_obs_grid * self.coverage_obs_grid + 1,
                    device=agent.state.pos.device,
                    dtype=agent.state.pos.dtype,
                ))
            else:
                parts.append(self._coverage_observation(agent))       # [B, K*K + 1]
        if self.local_coverage_obs_grid > 0:
            if self.ugv_zero_uav_search_observations and not agent.is_drone:
                parts.append(torch.zeros(
                    self.world.batch_dim,
                    self.local_coverage_obs_grid * self.local_coverage_obs_grid,
                    device=agent.state.pos.device,
                    dtype=agent.state.pos.dtype,
                ))
            else:
                parts.append(self._local_coverage_observation(agent))  # [B, K*K]
        if self.uav_confidence_obs_grid > 0:
            if self.ugv_zero_uav_search_observations and not agent.is_drone:
                parts.append(torch.zeros(
                    self.world.batch_dim,
                    self.uav_confidence_obs_grid * self.uav_confidence_obs_grid + 1,
                    device=agent.state.pos.device,
                    dtype=agent.state.pos.dtype,
                ))
            else:
                parts.append(self._uav_confidence_observation(agent))       # [B, K*K + 1]
        if self.local_confidence_obs_grid > 0:
            if self.ugv_zero_uav_search_observations and not agent.is_drone:
                parts.append(torch.zeros(
                    self.world.batch_dim,
                    self.local_confidence_obs_grid * self.local_confidence_obs_grid,
                    device=agent.state.pos.device,
                    dtype=agent.state.pos.dtype,
                ))
            else:
                parts.append(self._local_confidence_observation(agent))  # [B, K*K]
        if self.uav_frontier_obs:
            parts.append(self._uav_frontier_observation(agent))
        if self.uav_cleanup_target_obs:
            parts.append(self._uav_cleanup_target_observation(agent))
        if self.uav_astar_route_obs:
            parts.append(self._uav_astar_route_observation(agent))
        return torch.cat(parts, dim=-1)

    def _coverage_observation(self, agent: Agent | None = None) -> Tensor:
        """Team-coverage situational awareness: a downsampled absolute map of
        already-scouted cells plus the global covered fraction.

        Same for every agent (shared team memory). Lets the policy steer toward
        not-yet-covered regions instead of re-sweeping covered ground.
        """
        K = self.coverage_obs_grid
        return self._global_grid_observation(self._coverage_grid_for_observation(agent), K)

    def _local_coverage_observation(self, agent: Agent) -> Tensor:
        """Pooled ego-centric coverage patch extracted from the coverage grid.

        ``local_coverage_obs_radius_m`` is converted to coverage-grid cells at
        runtime, so the physical window stays stable across map sizes. Outside
        map cells are filled as covered, then the raw patch is adaptively
        average-pooled to KxK.
        """
        return self._local_grid_observation(
            agent,
            self._coverage_grid_for_observation(agent),
            K=self.local_coverage_obs_grid,
            radius_m=self.local_coverage_obs_radius_m,
            outside_value=1.0,
        )

    def _uav_confidence_observation(self, agent: Agent | None = None) -> Tensor:
        """Team inspection confidence: downsampled map plus global mean."""
        K = self.uav_confidence_obs_grid
        return self._global_grid_observation(self._confidence_grid_for_observation(agent), K)

    def _local_confidence_observation(self, agent: Agent) -> Tensor:
        """Pooled ego-centric inspection-confidence patch.

        Outside-map cells are filled as high confidence, matching the coverage
        observation convention that non-searchable space should not look
        unexplored.
        """
        if getattr(agent, "is_drone", False) and self.n_drones > 0:
            try:
                drone_idx = int(agent.name.rsplit("_", 1)[1])
            except (AttributeError, IndexError, ValueError):
                drone_idx = 0
            drone_idx = min(max(drone_idx, 0), self.n_drones - 1)
            return self._current_uav_local_confidence_features()[:, drone_idx]
        return self._local_grid_observation(
            agent,
            self._confidence_grid_for_observation(agent),
            K=self.local_confidence_obs_grid,
            radius_m=self.local_confidence_obs_radius_m,
            outside_value=1.0,
        )

    def _current_uav_local_confidence_features(self) -> Tensor:
        """Batched local confidence patches for all UAVs at current positions."""
        if self.n_drones <= 0:
            return torch.zeros(
                self.world.batch_dim,
                0,
                self.local_confidence_obs_grid * self.local_confidence_obs_grid,
                device=self.uav_confidence_grid.device,
                dtype=self.uav_confidence_grid.dtype,
            )
        drone_pos = torch.stack(
            [drone.state.pos for drone in self.world.agents[: self.n_drones]],
            dim=1,
        )
        if self._comms_maps_enabled():
            features = []
            for drone_idx in range(self.n_drones):
                features.append(self._batched_local_grid_observation(
                    drone_pos[:, drone_idx : drone_idx + 1],
                    self._drone_confidence_grid_for_observation(drone_idx),
                    K=self.local_confidence_obs_grid,
                    radius_m=self.local_confidence_obs_radius_m,
                    outside_value=1.0,
                )[:, 0])
            return torch.stack(features, dim=1)
        if not hasattr(self, "_uav_local_confidence_obs_cache"):
            self._uav_local_confidence_obs_cache = {}
        key = (
            getattr(self.uav_confidence_grid, "_version", 0),
            drone_pos.device.type,
            drone_pos.device.index,
            str(drone_pos.dtype),
            tuple(drone_pos.shape),
            int(self.local_confidence_obs_grid),
            float(self.local_confidence_obs_radius_m),
        )
        cached = self._uav_local_confidence_obs_cache.get(key)
        if cached is not None:
            return cached
        features = self._batched_local_grid_observation(
            drone_pos,
            self.uav_confidence_grid.float(),
            K=self.local_confidence_obs_grid,
            radius_m=self.local_confidence_obs_radius_m,
            outside_value=1.0,
        )
        self._uav_local_confidence_obs_cache[key] = features
        return features

    def _batched_local_grid_observation(
        self,
        positions: Tensor,
        grid: Tensor,
        *,
        K: int,
        radius_m: float,
        outside_value: float,
    ) -> Tensor:
        """Vectorized equivalent of `_local_grid_observation` for [B, N, 2] positions."""
        K = int(K)
        B, N, _ = positions.shape
        if K <= 0 or N <= 0:
            return torch.zeros(B, N, 0, device=positions.device, dtype=positions.dtype)
        import torch.nn.functional as F

        device = positions.device
        dtype = positions.dtype
        G = int(grid.shape[-1])
        cell_width_m = 1.0 / (
            self.terrain_sim_units_per_meter.to(device=device, dtype=dtype).clamp_min(1e-9)
            * (float(G) / (2.0 * float(self.x_semidim)))
        )
        radius_cells = torch.round(float(radius_m) / cell_width_m).long().clamp_min(1)
        max_radius = int(radius_cells.max().detach().cpu().item())
        patch_size = 2 * max_radius + 1

        values = grid.to(device=device, dtype=dtype)
        gx, gy = self._positions_to_grid(positions, grid_size=G)
        offsets = torch.arange(-max_radius, max_radius + 1, device=device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        patch_x = gx[..., None, None] + offset_x.view(1, 1, patch_size, patch_size)
        patch_y = gy[..., None, None] + offset_y.view(1, 1, patch_size, patch_size)

        radius = radius_cells.view(B, 1, 1, 1)
        inside_radius = (
            (offset_x.abs().view(1, 1, patch_size, patch_size) <= radius)
            & (offset_y.abs().view(1, 1, patch_size, patch_size) <= radius)
        )
        in_bounds = (
            inside_radius
            & (patch_x >= 0)
            & (patch_x < G)
            & (patch_y >= 0)
            & (patch_y < G)
        )
        safe_x = patch_x.clamp(0, G - 1)
        safe_y = patch_y.clamp(0, G - 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, patch_size, patch_size)
        gathered = values[batch_idx, safe_y, safe_x]
        patches = torch.where(
            in_bounds,
            gathered,
            torch.full_like(gathered, float(outside_value)),
        )
        pooled = F.adaptive_avg_pool2d(
            patches.reshape(B * N, 1, patch_size, patch_size),
            (K, K),
        )
        return pooled.reshape(B, N, K * K)

    def _global_grid_observation(self, grid: Tensor, K: int) -> Tensor:
        if K <= 0:
            return torch.zeros(self.world.batch_dim, 0, device=grid.device, dtype=grid.dtype)
        import torch.nn.functional as F

        values = grid.float().unsqueeze(1)                       # [B, 1, G, G]
        pooled = F.adaptive_avg_pool2d(values, (K, K)).flatten(1) # [B, K*K]
        global_mean = grid.float().mean(dim=(1, 2), keepdim=True).squeeze(1)  # [B, 1]
        return torch.cat([pooled, global_mean], dim=-1)

    def _local_grid_observation(
        self,
        agent: Agent,
        grid: Tensor,
        *,
        K: int,
        radius_m: float,
        outside_value: float,
    ) -> Tensor:
        K = int(K)
        if K <= 0:
            return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)
        import torch.nn.functional as F

        pos = agent.state.pos
        device = pos.device
        dtype = pos.dtype
        G = int(grid.shape[-1])
        cell_width_m = 1.0 / (
            self.terrain_sim_units_per_meter.to(device=device, dtype=dtype).clamp_min(1e-9)
            * (float(G) / (2.0 * float(self.x_semidim)))
        )
        radius_cells = torch.round(float(radius_m) / cell_width_m).long().clamp_min(1)
        max_radius = int(radius_cells.max().detach().cpu().item())
        patch_size = 2 * max_radius + 1

        values = grid.to(device=device, dtype=dtype)
        gx, gy = self._positions_to_grid(pos, grid_size=G)
        out = torch.empty(self.world.batch_dim, K * K, device=device, dtype=dtype)
        for env_idx in range(self.world.batch_dim):
            radius = int(radius_cells[env_idx].detach().cpu().item())
            raw_patch_size = 2 * radius + 1
            patch = torch.full(
                (raw_patch_size, raw_patch_size),
                float(outside_value),
                device=device,
                dtype=dtype,
            )
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
                patch[py0:py1, px0:px1] = values[env_idx, sy0:sy1, sx0:sx1]
            if raw_patch_size != patch_size:
                padded = torch.full(
                    (patch_size, patch_size),
                    float(outside_value),
                    device=device,
                    dtype=dtype,
                )
                offset = max_radius - radius
                padded[offset : offset + raw_patch_size, offset : offset + raw_patch_size] = patch
                patch = padded
            pooled = F.adaptive_avg_pool2d(patch.view(1, 1, patch_size, patch_size), (K, K))
            out[env_idx] = pooled.flatten()
        return out

    def _uav_frontier_observation(self, agent: Agent) -> Tensor:
        """Configured frontier features for nearby uncovered team-coverage cells."""
        dim = self._uav_frontier_obs_dim()
        if not agent.is_drone or self.n_drones <= 0:
            return torch.zeros(self.world.batch_dim, dim, device=agent.state.pos.device, dtype=agent.state.pos.dtype)
        try:
            drone_idx = int(agent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            drone_idx = 0
        drone_idx = min(max(drone_idx, 0), self.n_drones - 1)
        return self._current_uav_frontier_features()[:, drone_idx]

    def _uav_cleanup_target_observation(self, agent: Agent) -> Tensor:
        dim = self._uav_cleanup_target_obs_dim()
        if not agent.is_drone or self.n_drones <= 0:
            return torch.zeros(self.world.batch_dim, dim, device=agent.state.pos.device, dtype=agent.state.pos.dtype)
        try:
            drone_idx = int(agent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            drone_idx = 0
        drone_idx = min(max(drone_idx, 0), self.n_drones - 1)
        drone_pos = torch.stack(
            [drone.state.pos for drone in self.world.agents[: self.n_drones]],
            dim=1,
        )
        self._refresh_uav_cleanup_target_assignments(drone_pos)
        valid = self.uav_cleanup_target_valid[:, drone_idx].to(device=agent.state.pos.device)
        target_vec = (
            self.uav_cleanup_target_pos[:, drone_idx].to(device=agent.state.pos.device, dtype=agent.state.pos.dtype)
            - agent.state.pos
        )
        distance_sim = target_vec.norm(dim=-1)
        unit = target_vec / distance_sim.clamp_min(1e-9).unsqueeze(-1)
        meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(
            device=agent.state.pos.device,
            dtype=agent.state.pos.dtype,
        ).clamp_min(1e-9)
        distance_m = distance_sim * meters_per_sim
        map_diag_m = (
            math.hypot(2.0 * float(self.x_semidim), 2.0 * float(self.y_semidim))
            * meters_per_sim
        ).clamp_min(1e-9)
        distance_norm = (distance_m / map_diag_m).clamp(0.0, 1.0)
        value = self.uav_cleanup_target_value[:, drone_idx].to(
            device=agent.state.pos.device,
            dtype=agent.state.pos.dtype,
        ).clamp(0.0, 1.0)
        out = torch.cat((unit, distance_norm.unsqueeze(-1), value.unsqueeze(-1)), dim=-1)
        return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))

    def _uav_astar_route_observation(self, agent: Agent) -> Tensor:
        dim = self._uav_astar_route_obs_dim()
        if not agent.is_drone or self.n_drones <= 0:
            return torch.zeros(self.world.batch_dim, dim, device=agent.state.pos.device, dtype=agent.state.pos.dtype)
        try:
            drone_idx = int(agent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            drone_idx = 0
        drone_idx = min(max(drone_idx, 0), self.n_drones - 1)
        drone_pos = torch.stack(
            [drone.state.pos for drone in self.world.agents[: self.n_drones]],
            dim=1,
        )
        self._refresh_uav_astar_routes(drone_pos)
        device = agent.state.pos.device
        dtype = agent.state.pos.dtype
        valid = self.uav_astar_waypoint_valid[:, drone_idx].to(device=device)
        route_vec = (
            self.uav_astar_waypoint_pos[:, drone_idx].to(device=device, dtype=dtype)
            - agent.state.pos
        )
        distance_sim = route_vec.norm(dim=-1)
        unit = route_vec / distance_sim.clamp_min(1e-9).unsqueeze(-1)
        meters_per_sim = 1.0 / self.terrain_sim_units_per_meter.to(
            device=device,
            dtype=dtype,
        ).clamp_min(1e-9)
        distance_m = distance_sim * meters_per_sim
        map_diag_m = (
            math.hypot(2.0 * float(self.x_semidim), 2.0 * float(self.y_semidim))
            * meters_per_sim
        ).clamp_min(1e-9)
        distance_norm = (distance_m / map_diag_m).clamp(0.0, 1.0)
        path_cost_norm = self.uav_astar_path_cost_norm[:, drone_idx].to(
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)
        out = torch.cat((unit, distance_norm.unsqueeze(-1), path_cost_norm.unsqueeze(-1)), dim=-1)
        return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))

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

    def _team_reward_comms_up_mask(self, device: torch.device) -> Tensor:
        """Return the per-agent communication state used for immediate team rewards.

        Team rewards are intentionally not replayed after reconnection. We use
        the communication state sampled for the latest observation/action cycle;
        direct observers are handled separately by the event actor mask.
        """
        if self.n_agents <= 0:
            return torch.zeros(self.world.batch_dim, 0, dtype=torch.bool, device=device)
        if self.comms_dropout <= 0.0:
            return torch.ones(
                self.world.batch_dim,
                self.n_agents,
                dtype=torch.bool,
                device=device,
            )

        states = []
        for agent in self.world.agents:
            comms_up = getattr(agent, "comms_up", None)
            if comms_up is None:
                states.append(torch.ones(self.world.batch_dim, dtype=torch.bool, device=device))
            else:
                states.append(comms_up.to(device=device, dtype=torch.bool).view(-1))
        return torch.stack(states, dim=1)

    def _communication_gated_team_event_reward(
        self,
        event_by_target: Tensor,
        actor_events: Tensor,
        reward_weight: float,
        comms_up: Tensor,
    ) -> Tensor:
        """Distribute sparse team event rewards without leaking through dropout.

        ``event_by_target`` marks newly scouted/confirmed survivors. ``actor_events``
        marks the agents that directly observed those same events. An agent gets
        the team reward for an event if it either directly observed the event or
        was connected at that step.
        """
        if reward_weight == 0.0 or event_by_target.numel() == 0 or self.n_agents <= 0:
            return torch.zeros(
                self.world.batch_dim,
                self.n_agents,
                device=comms_up.device,
                dtype=torch.float,
            )
        events = event_by_target.to(device=comms_up.device, dtype=torch.bool)
        direct = actor_events.to(device=comms_up.device, dtype=torch.bool) & events.unsqueeze(1)
        connected = comms_up.unsqueeze(-1) & events.unsqueeze(1)
        recipients = direct | connected
        return recipients.float().sum(dim=2) * float(reward_weight)

    def _communication_keep(self, agent: Agent) -> Tensor:
        """Sample one receiver-level communication state for this observation."""
        if self.comms_dropout <= 0:
            keep = torch.ones(
                self.world.batch_dim, 1, dtype=torch.bool, device=agent.state.pos.device,
            )
        elif self.comms_dropout_mode == "bursty":
            self._advance_bursty_comms_dropout()
            agent_idx = self.world.agents.index(agent)
            keep = (self.comms_dropout_remaining_steps[:, agent_idx] <= 0).view(-1, 1)
        else:
            keep = (
                torch.rand(self.world.batch_dim, 1, device=agent.state.pos.device)
                > self.comms_dropout
            )
        agent.comms_up = keep[:, 0]
        return keep

    def _bursty_comms_start_probability(self) -> float:
        """Per-connected-step outage start probability for target down fraction."""
        target_fraction = min(max(float(self.comms_dropout), 0.0), 1.0)
        if target_fraction <= 0.0:
            return 0.0
        if target_fraction >= 1.0:
            return 1.0
        mean_duration = 0.5 * (
            float(self.comms_dropout_min_steps) + float(self.comms_dropout_max_steps)
        )
        # Outages start on a connected step and consume that same step. With a
        # geometric connected run, q below gives the requested long-run down
        # fraction in expectation for the configured mean outage duration.
        return target_fraction / (
            target_fraction + (1.0 - target_fraction) * max(mean_duration, 1.0)
        )

    def _advance_bursty_comms_dropout(self) -> None:
        """Advance receiver-level burst outage timers once per env step."""
        if self.n_agents <= 0:
            return
        current_step = self.step_count.to(
            device=self.comms_dropout_last_update_step.device,
            dtype=self.comms_dropout_last_update_step.dtype,
        )
        due = current_step != self.comms_dropout_last_update_step
        if not bool(due.any().item()):
            return

        due_idx = due.nonzero(as_tuple=False).flatten()
        previous_initialized = self.comms_dropout_last_update_step[due_idx] >= 0
        if bool(previous_initialized.any().item()):
            initialized_envs = due_idx[previous_initialized]
            elapsed = (
                current_step[initialized_envs]
                - self.comms_dropout_last_update_step[initialized_envs]
            ).clamp_min(1).view(-1, 1)
            self.comms_dropout_remaining_steps[initialized_envs] = (
                self.comms_dropout_remaining_steps[initialized_envs] - elapsed
            ).clamp_min(0)

        start_probability = self._bursty_comms_start_probability()
        if start_probability > 0.0:
            remaining = self.comms_dropout_remaining_steps[due_idx]
            connected = remaining <= 0
            starts = (
                torch.rand(remaining.shape, device=remaining.device) < start_probability
            ) & connected
            if bool(starts.any().item()):
                durations = torch.randint(
                    int(self.comms_dropout_min_steps),
                    int(self.comms_dropout_max_steps) + 1,
                    remaining.shape,
                    device=remaining.device,
                    dtype=remaining.dtype,
                )
                updated = torch.where(starts, durations, remaining)
                self.comms_dropout_remaining_steps[due_idx] = updated

        self.comms_dropout_last_update_step[due_idx] = current_step[due_idx]

    def _survivor_message_observations(self, agent: Agent, comms_keep: Tensor) -> Tensor:
        """Encode known survivor/false-positive candidates.

        Base feature order is [known, dx, dy, ux, uy, distance_norm, confirmed].
        If decoys are enabled, the same candidate block adds
        [status_false_positive] after confirmed. True survivors always have
        status_false_positive=0; decoys flip to 1 only after UGV investigation.
        When survivor_assignment_obs is enabled, two flags are appended:
        [assigned_to_me, assigned_to_other_ugv].
        """
        agent_idx = self.world.agents.index(agent)
        has_decoys = self.n_decoys > 0
        active_survivors = self._active_survivor_mask()
        local_known = self.known_survivors_by_agent[:, agent_idx]
        local_confirmed = self.confirmed_survivors_by_agent[:, agent_idx]

        team_known = self.known_survivors_by_agent.any(dim=1) & active_survivors
        team_confirmed = self.confirmed_survivors_by_agent.any(dim=1) & active_survivors
        connected = comms_keep.expand_as(local_known)
        local_known = torch.where(connected, team_known, local_known & active_survivors)
        local_confirmed = torch.where(connected, team_confirmed, local_confirmed & active_survivors)

        self.known_survivors_by_agent[:, agent_idx] = local_known
        self.confirmed_survivors_by_agent[:, agent_idx] = local_confirmed

        obs_known = local_known
        obs_confirmed = local_confirmed
        obs_false_positive = torch.zeros_like(obs_known)
        decoy_known = None
        decoy_false_positive = None
        if has_decoys:
            active_decoys = self._active_decoy_mask()
            local_decoy_known = self.known_decoys_by_agent[:, agent_idx]
            team_decoy_known = self.known_decoys_by_agent.any(dim=1) & active_decoys
            connected_decoys = comms_keep.expand_as(local_decoy_known)
            decoy_known = torch.where(connected_decoys, team_decoy_known, local_decoy_known & active_decoys)
            self.known_decoys_by_agent[:, agent_idx] = decoy_known
            decoy_false_positive = decoy_known & self.dismissed_decoys & active_decoys

        assigned_to_me = None
        assigned_to_other = None
        decoy_assigned_to_me = None
        decoy_assigned_to_other = None
        if self.survivor_assignment_obs:
            assigned_to_me = torch.zeros_like(obs_known)
            assigned_to_other = torch.zeros_like(obs_known)
            if has_decoys:
                decoy_assigned_to_me = torch.zeros_like(decoy_known)
                decoy_assigned_to_other = torch.zeros_like(decoy_known)
            if self.n_ground > 0 and (self.n_survivors > 0 or has_decoys):
                ground_slice = slice(self.n_drones, self.n_agents)
                ground_known = self.known_survivors_by_agent[:, ground_slice]
                ground_confirmed = self.confirmed_survivors_by_agent[:, ground_slice]
                targetable = ground_known & ~ground_confirmed & active_survivors.unsqueeze(1)
                survivor_pos_for_assignment = torch.stack([s.state.pos for s in self._survivors], dim=1)
                assignment_pos = survivor_pos_for_assignment
                if has_decoys:
                    decoy_pos_for_assignment = torch.stack([d.state.pos for d in self._decoys], dim=1)
                    ground_decoy_known = self.known_decoys_by_agent[:, ground_slice]
                    decoy_targetable = (
                        ground_decoy_known
                        & ~self.dismissed_decoys.unsqueeze(1)
                        & self._active_decoy_mask().unsqueeze(1)
                    )
                    targetable = torch.cat((targetable, decoy_targetable), dim=2)
                    assignment_pos = torch.cat((assignment_pos, decoy_pos_for_assignment), dim=1)
                ground_pos = torch.stack([a.state.pos for a in self.world.agents[ground_slice]], dim=1)
                assigned_idx, _assigned_dist = self._ugv_assigned_target_indices(
                    ground_pos,
                    assignment_pos,
                    targetable,
                )
                assigned_valid = assigned_idx >= 0
                n_assignment_targets = targetable.shape[2]
                assignment_mask = torch.zeros(
                    self.world.batch_dim,
                    self.n_ground,
                    n_assignment_targets,
                    dtype=torch.bool,
                    device=obs_known.device,
                )
                assignment_mask.scatter_(
                    dim=2,
                    index=assigned_idx.clamp(min=0).unsqueeze(-1),
                    src=assigned_valid.unsqueeze(-1),
                )
                if not agent.is_drone:
                    ground_index = agent_idx - self.n_drones
                    assigned_to_me = assignment_mask[:, ground_index, :self.n_survivors]
                    if self.n_ground > 1:
                        other_mask = assignment_mask.clone()
                        other_mask[:, ground_index, :] = False
                        assigned_to_other = other_mask.any(dim=1)[:, :self.n_survivors]
                    if has_decoys:
                        decoy_assigned_to_me = assignment_mask[:, ground_index, self.n_survivors:]
                        if self.n_ground > 1:
                            decoy_assigned_to_other = other_mask.any(dim=1)[:, self.n_survivors:]
                else:
                    assigned_to_other = assignment_mask.any(dim=1)[:, :self.n_survivors]
                    if has_decoys:
                        decoy_assigned_to_other = assignment_mask.any(dim=1)[:, self.n_survivors:]

        survivor_pos = torch.stack([s.state.pos for s in self._survivors], dim=1)
        relative_pos = survivor_pos - agent.state.pos.unsqueeze(1)
        relative_pos = relative_pos * obs_known.unsqueeze(-1).float()
        dist_sim = torch.linalg.norm(relative_pos, dim=-1, keepdim=True)
        unit_direction = relative_pos / dist_sim.clamp_min(1e-9)
        distance_m = dist_sim / self.terrain_sim_units_per_meter.view(-1, 1, 1).clamp_min(1e-9)
        distance_norm = distance_m / self.survivor_message_distance_scale_m
        feature_parts = [
            obs_known.unsqueeze(-1).float(),
            relative_pos,
            unit_direction,
            distance_norm,
            obs_confirmed.unsqueeze(-1).float(),
        ]
        if has_decoys:
            feature_parts.append(obs_false_positive.unsqueeze(-1).float())
        features = torch.cat(feature_parts, dim=-1)
        if self.survivor_assignment_obs:
            features = torch.cat(
                [
                    features,
                    assigned_to_me.unsqueeze(-1).float(),
                    assigned_to_other.unsqueeze(-1).float(),
                ],
                dim=-1,
            )
        if self.obs_schema_n_survivors > self.n_survivors:
            pad = torch.zeros(
                self.world.batch_dim,
                self.obs_schema_n_survivors - self.n_survivors,
                features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            features = torch.cat((features, pad), dim=1)

        if has_decoys:
            decoy_pos = torch.stack([d.state.pos for d in self._decoys], dim=1)
            decoy_relative_pos = decoy_pos - agent.state.pos.unsqueeze(1)
            decoy_relative_pos = decoy_relative_pos * decoy_known.unsqueeze(-1).float()
            decoy_dist_sim = torch.linalg.norm(decoy_relative_pos, dim=-1, keepdim=True)
            decoy_unit_direction = decoy_relative_pos / decoy_dist_sim.clamp_min(1e-9)
            decoy_distance_m = decoy_dist_sim / self.terrain_sim_units_per_meter.view(-1, 1, 1).clamp_min(1e-9)
            decoy_distance_norm = decoy_distance_m / self.survivor_message_distance_scale_m
            decoy_features = torch.cat(
                [
                    decoy_known.unsqueeze(-1).float(),
                    decoy_relative_pos,
                    decoy_unit_direction,
                    decoy_distance_norm,
                    torch.zeros_like(decoy_known.unsqueeze(-1).float()),
                    decoy_false_positive.unsqueeze(-1).float(),
                ],
                dim=-1,
            )
            if self.survivor_assignment_obs:
                decoy_features = torch.cat(
                    [
                        decoy_features,
                        decoy_assigned_to_me.unsqueeze(-1).float(),
                        decoy_assigned_to_other.unsqueeze(-1).float(),
                    ],
                    dim=-1,
                )
            features = torch.cat((features, decoy_features), dim=1)

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
        If ``ugv_planner_detour_obs`` is enabled, detour_needed is appended.
        The planner is constrained to ``ugv_planner_patch_size`` cells around the
        UGV and does not expose the full route.
        """
        if self.ugv_planner_hint == "none":
            return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)
        if self.ugv_planner_hint not in UGV_PLANNER_HINT_MODES:
            raise RuntimeError(f"unsupported ugv_planner_hint: {self.ugv_planner_hint!r}")
        hint_dim = self._ugv_planner_hint_dim()
        agent_idx = self.world.agents.index(agent)
        if agent_idx == self.n_drones:
            self._invalidate_ugv_planner_route_cache()
        if agent.is_drone:
            return torch.zeros(self.world.batch_dim, hint_dim, device=agent.state.pos.device)

        ground_slice = slice(self.n_drones, self.n_agents)
        target_pos_all, targetable, _target_is_decoy = self._ugv_ground_target_candidates()
        if target_pos_all.shape[1] == 0:
            return torch.zeros(self.world.batch_dim, hint_dim, device=agent.state.pos.device)
        ground_pos = torch.stack([a.state.pos for a in self.world.agents[ground_slice]], dim=1)
        assigned_idx, _assigned_dist = self._ugv_assigned_target_indices(
            ground_pos,
            target_pos_all,
            targetable,
        )
        ground_index = agent_idx - self.n_drones
        target_idx = assigned_idx[:, ground_index]
        has_target = target_idx >= 0
        target_idx_safe = target_idx.clamp(min=0)
        target_pos = target_pos_all.gather(1, target_idx_safe.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

        out = torch.zeros(self.world.batch_dim, hint_dim, device=agent.state.pos.device)
        for env_index in range(self.world.batch_dim):
            if not bool(has_target[env_index].item()):
                continue
            out[env_index] = self._local_astar_hint_for_env(
                env_index,
                ground_index,
                agent.state.pos[env_index],
                target_pos[env_index],
                target_idx=int(target_idx[env_index].item()),
            )
        return out

    def _ugv_planner_hint_dim(self) -> int:
        if self.ugv_planner_hint not in UGV_PLANNER_HINT_MODES:
            return 0
        return UGV_PLANNER_HINT_DIM + int(bool(self.ugv_planner_detour_obs))

    def _local_astar_hint_for_env(
        self,
        env_index: int,
        ground_index: int | None,
        pos: Tensor,
        target_pos: Tensor,
        *,
        target_idx: int | None = None,
    ) -> Tensor:
        if (
            self.ugv_dense_reward_mode == "escape_route_switch"
            and ground_index is not None
            and hasattr(self, "ugv_escape_route_active")
            and bool(self.ugv_escape_route_active[env_index, ground_index].item())
        ):
            waypoint = self._ugv_escape_route_waypoint_for_env(
                env_index,
                ground_index,
                pos,
                update_index=True,
            )
            if waypoint is not None:
                return self._ugv_planner_hint_from_waypoint(
                    env_index,
                    pos,
                    waypoint,
                    direct_blocked=True,
                    detour_needed=True,
                )

        route = self._ugv_planner_route_for_env(
            env_index,
            pos,
            target_pos,
            ground_index=ground_index,
            target_idx=target_idx,
        )
        hint = torch.zeros(self._ugv_planner_hint_dim(), device=pos.device)
        if route is None:
            return hint
        waypoint, direct_blocked, detour_needed = route
        return self._ugv_planner_hint_from_waypoint(
            env_index,
            pos,
            waypoint,
            direct_blocked=direct_blocked,
            detour_needed=detour_needed,
            planner_range_m=(
                self.ugv_global_planner_lookahead_m
                if self.ugv_planner_hint == "global_astar"
                else None
            ),
        )

    def _ugv_planner_hint_from_waypoint(
        self,
        env_index: int,
        pos: Tensor,
        waypoint: tuple[int, int],
        *,
        direct_blocked: bool,
        detour_needed: bool,
        planner_range_m: float | None = None,
    ) -> Tensor:
        device = pos.device
        hint = torch.zeros(self._ugv_planner_hint_dim(), device=device)
        waypoint_pos = self._grid_cell_center_to_world(waypoint, device=device, dtype=pos.dtype)
        delta = waypoint_pos - pos
        dist = torch.linalg.norm(delta)
        if float(dist.item()) <= 1e-9:
            return hint

        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        if planner_range_m is None:
            patch_size = self.ugv_planner_patch_size
            radius = patch_size // 2
            cell_w_sim = 2.0 * float(self.x_semidim) / float(self.fire_grid_size)
            cell_h_sim = 2.0 * float(self.y_semidim) / float(self.fire_grid_size)
            planner_range_m = max(radius * max(cell_w_sim, cell_h_sim) / max(scale, 1e-9), 1e-6)
        else:
            planner_range_m = max(float(planner_range_m), 1e-6)
        dist_m = float(dist.detach().cpu().item()) / max(scale, 1e-9)
        unit = delta / dist.clamp_min(1e-9)
        hint[0] = unit[X]
        hint[1] = unit[Y]
        hint[2] = min(max(dist_m / planner_range_m, 0.0), 1.0)
        hint[3] = 1.0
        hint[4] = 1.0 if direct_blocked else 0.0
        if self.ugv_planner_detour_obs:
            hint[5] = 1.0 if detour_needed else 0.0
        return hint

    def _ugv_escape_route_waypoint_for_env(
        self,
        env_index: int,
        ground_index: int,
        pos: Tensor,
        *,
        update_index: bool,
    ) -> tuple[int, int] | None:
        if not hasattr(self, "ugv_escape_route_paths"):
            return None
        if not bool(self.ugv_escape_route_active[env_index, ground_index].item()):
            return None
        path = self.ugv_escape_route_paths[env_index][ground_index]
        if len(path) < 2:
            return None
        pos_cell = self._single_position_to_grid_cell(pos)
        start_idx = int(self.ugv_escape_route_path_index[env_index, ground_index].item())
        start_idx = max(0, min(start_idx, len(path) - 1))
        nearest_offset, _nearest_cell = min(
            enumerate(path[start_idx:]),
            key=lambda item: (
                item[1][0] - pos_cell[0]
            ) ** 2 + (
                item[1][1] - pos_cell[1]
            ) ** 2,
        )
        nearest_idx = start_idx + int(nearest_offset)
        waypoint = self._route_lookahead_cell(
            self.traversable_grid[env_index],
            path,
            nearest_idx,
        )
        waypoint_idx = nearest_idx
        for idx in range(nearest_idx, len(path)):
            if path[idx] == waypoint:
                waypoint_idx = idx
                break
        if update_index:
            self.ugv_escape_route_path_index[env_index, ground_index] = nearest_idx
            self.ugv_escape_route_waypoint_cell[env_index, ground_index, 0] = int(waypoint[0])
            self.ugv_escape_route_waypoint_cell[env_index, ground_index, 1] = int(waypoint[1])
        return path[waypoint_idx]

    def _ugv_planner_route_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        ground_index: int | None = None,
        target_idx: int | None = None,
    ) -> tuple[tuple[int, int], bool, bool] | None:
        if self.ugv_planner_hint == "local_astar":
            return self._local_astar_route_for_env(
                env_index,
                pos,
                target_pos,
                ground_index=ground_index,
            )
        if self.ugv_planner_hint == "local_escape_astar":
            return self._local_escape_astar_route_for_env(
                env_index,
                pos,
                target_pos,
                ground_index=ground_index,
            )
        if self.ugv_planner_hint == "global_astar":
            return self._global_astar_route_for_env(
                env_index,
                pos,
                target_pos,
                ground_index=ground_index,
                target_idx=target_idx,
            )
        return None

    def _global_astar_route_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        ground_index: int | None = None,
        target_idx: int | None = None,
    ) -> tuple[tuple[int, int], bool, bool] | None:
        plan = self._global_astar_route_info_for_env(
            env_index,
            pos,
            target_pos,
            ground_index=ground_index,
            target_idx=target_idx,
            update_index=True,
        )
        return None if plan is None else plan["route"]

    def _global_astar_route_info_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        ground_index: int | None = None,
        target_idx: int | None = None,
        update_index: bool = False,
    ) -> dict | None:
        if ground_index is None or target_idx is None:
            return self._global_astar_plan_uncached_for_env(
                env_index,
                pos,
                target_pos,
                update_state=False,
            )

        current_target = int(target_idx)
        needs_plan = (
            int(self.ugv_global_route_target_idx[env_index, ground_index].item()) != current_target
            or not self.ugv_global_route_paths[env_index][ground_index]
        )
        if needs_plan:
            replanned_after_fire = bool(
                self.ugv_global_route_fire_replan_pending[env_index, ground_index].item()
            )
            self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index] = False
            self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index] = False
            plan = self._global_astar_plan_uncached_for_env(
                env_index,
                pos,
                target_pos,
                update_state=False,
            )
            if plan is None:
                self.ugv_global_route_target_idx[env_index, ground_index] = -1
                self.ugv_global_route_paths[env_index][ground_index] = []
                self.ugv_global_route_fire_replan_pending[env_index, ground_index] = False
                self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index] = replanned_after_fire
                self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index] = (
                    self._ugv_fire_blocked_no_path_active(env_index)
                )
                return None
            self.ugv_global_route_target_idx[env_index, ground_index] = current_target
            self.ugv_global_route_paths[env_index][ground_index] = list(plan["path"])
            gx, gy = plan["goal"]
            self.ugv_global_route_goal_cell[env_index, ground_index, 0] = int(gx)
            self.ugv_global_route_goal_cell[env_index, ground_index, 1] = int(gy)
            self.ugv_global_route_path_index[env_index, ground_index] = 0
            self.ugv_global_route_last_replan_step[env_index, ground_index] = int(
                self.step_count[env_index].item()
            )
            self.ugv_global_route_fire_replan_pending[env_index, ground_index] = False
            self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index] = replanned_after_fire

        path = self.ugv_global_route_paths[env_index][ground_index]
        if len(path) < 2:
            return None
        goal = path[-1]
        waypoint, nearest_idx, waypoint_idx = self._global_route_waypoint_for_env(
            env_index,
            pos,
            path,
            start_idx=int(self.ugv_global_route_path_index[env_index, ground_index].item()),
        )
        if update_index:
            self.ugv_global_route_path_index[env_index, ground_index] = int(nearest_idx)
            self.ugv_global_route_waypoint_cell[env_index, ground_index, 0] = int(waypoint[0])
            self.ugv_global_route_waypoint_cell[env_index, ground_index, 1] = int(waypoint[1])

        start = self._single_position_to_grid_cell(pos)
        planner_traversable, _planner_cost = self._ugv_planner_layer_tensors_for_env(env_index)
        direct_blocked = not self._grid_segment_is_traversable(planner_traversable, start, goal)
        detour_needed = self._local_astar_detour_needed(
            env_index,
            start,
            goal,
            waypoint,
            path[nearest_idx:],
            direct_blocked,
        )
        return {
            "route": (waypoint, bool(direct_blocked), bool(detour_needed)),
            "start": start,
            "goal": goal,
            "waypoint": waypoint,
            "path": path,
            "path_index": int(nearest_idx),
            "waypoint_index": int(waypoint_idx),
            "direct_blocked": bool(direct_blocked),
            "detour_needed": bool(detour_needed),
            "planner_mode": "global_astar",
        }

    def _global_astar_plan_uncached_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        update_state: bool = False,
    ) -> dict | None:
        del update_state
        planner_traversable_tensor, planner_cost_tensor = self._ugv_planner_layer_tensors_for_env(env_index)
        start_guess = self._single_position_to_grid_cell(pos)
        target_cell = self._single_position_to_grid_cell(target_pos)
        bounds = (0, self.fire_grid_size - 1, 0, self.fire_grid_size - 1)
        start = self._nearest_traversable_cell_in_bounds(
            env_index,
            start_guess[0],
            start_guess[1],
            bounds,
            traversable=planner_traversable_tensor,
        )
        if start is None:
            return None

        traversable, movement_cost = self._ugv_planner_layer_arrays_for_env(env_index)
        finite_open = traversable & np.isfinite(movement_cost)
        if not bool(finite_open.any()):
            return None

        goals = self._global_astar_goal_cells_for_env(
            env_index,
            target_pos,
            traversable,
            movement_cost,
        )
        if not goals:
            nearest_goal = self._nearest_traversable_cell_in_bounds(
                env_index,
                target_cell[0],
                target_cell[1],
                bounds,
                traversable=planner_traversable_tensor,
            )
            if nearest_goal is None:
                return None
            goals = [nearest_goal]

        heuristic_goals = None
        if self.ugv_global_planner_heuristic == "terrain":
            static_traversable, static_cost = self._ugv_static_planner_layer_arrays_for_env(env_index)
            heuristic_goals = self._global_astar_goal_cells_for_env(
                env_index,
                target_pos,
                static_traversable,
                static_cost,
            )
            if not heuristic_goals:
                heuristic_goals = goals

        path = self._global_astar_grid_path(
            env_index,
            start,
            goals,
            traversable,
            movement_cost,
            heuristic_goals=heuristic_goals,
        )
        if len(path) < 2:
            return None
        waypoint, nearest_idx, waypoint_idx = self._global_route_waypoint_for_env(
            env_index,
            pos,
            path,
            start_idx=0,
        )
        goal = path[-1]
        direct_blocked = not self._grid_segment_is_traversable(planner_traversable_tensor, start, goal)
        detour_needed = self._local_astar_detour_needed(
            env_index,
            start,
            goal,
            waypoint,
            path,
            direct_blocked,
        )
        return {
            "route": (waypoint, bool(direct_blocked), bool(detour_needed)),
            "start": start,
            "goal": goal,
            "waypoint": waypoint,
            "path": path,
            "path_index": int(nearest_idx),
            "waypoint_index": int(waypoint_idx),
            "direct_blocked": bool(direct_blocked),
            "detour_needed": bool(detour_needed),
            "planner_mode": "global_astar",
        }

    def _global_astar_goal_cells_for_env(
        self,
        env_index: int,
        target_pos: Tensor,
        traversable: np.ndarray,
        movement_cost: np.ndarray,
    ) -> list[tuple[int, int]]:
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        confirm_radius_sim = float(self.detection_range_by_env[env_index].detach().cpu().item())
        if confirm_radius_sim <= 1e-9 and scale > 1e-9:
            confirm_radius_sim = float(self.ground_confirmation_range_m) * scale
        G = int(self.fire_grid_size)
        ys, xs = np.nonzero(traversable & np.isfinite(movement_cost))
        if xs.size == 0:
            return []
        cell_w = 2.0 * float(self.x_semidim) / float(G)
        cell_h = 2.0 * float(self.y_semidim) / float(G)
        centers_x = -float(self.x_semidim) + (xs.astype(np.float64) + 0.5) * cell_w
        centers_y = -float(self.y_semidim) + (ys.astype(np.float64) + 0.5) * cell_h
        target_x = float(target_pos[X].detach().cpu().item())
        target_y = float(target_pos[Y].detach().cpu().item())
        dist2 = (centers_x - target_x) ** 2 + (centers_y - target_y) ** 2
        in_radius = dist2 <= confirm_radius_sim ** 2
        if not bool(np.any(in_radius)):
            return []
        candidate_x = xs[in_radius]
        candidate_y = ys[in_radius]
        candidate_dist2 = dist2[in_radius]
        order = np.argsort(candidate_dist2, kind="stable")
        return [
            (int(candidate_x[i]), int(candidate_y[i]))
            for i in order
        ]

    def _global_astar_euclidean_heuristic_grid(
        self,
        goals: list[tuple[int, int]],
        traversable: np.ndarray,
        movement_cost: np.ndarray,
    ) -> np.ndarray:
        G = int(self.fire_grid_size)
        finite_open_cost = movement_cost[traversable & np.isfinite(movement_cost)]
        if finite_open_cost.size == 0:
            return np.zeros((G, G), dtype=np.float64)
        min_cost = max(float(finite_open_cost.min()), 1e-6)
        grid_y = np.arange(G, dtype=np.float64)[:, None]
        grid_x = np.arange(G, dtype=np.float64)[None, :]
        heuristic_dist2 = np.full((G, G), np.inf, dtype=np.float64)
        for gx, gy in goals:
            dx = grid_x - float(gx)
            dy = grid_y - float(gy)
            heuristic_dist2 = np.minimum(heuristic_dist2, dx * dx + dy * dy)
        return np.sqrt(heuristic_dist2) * min_cost

    def _global_astar_static_cost_to_go_for_env(
        self,
        env_index: int,
        goals: list[tuple[int, int]],
    ) -> np.ndarray:
        version = int(getattr(self, "_ugv_static_planner_cache_version", 0))
        goal_key = tuple((int(x), int(y)) for x, y in goals)
        key = (int(env_index), version, goal_key)
        cache = getattr(self, "_ugv_global_heuristic_cache", None)
        if cache is None:
            self._ugv_global_heuristic_cache = {}
            cache = self._ugv_global_heuristic_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        costs = self._global_astar_static_raw_cost_to_go_for_env(env_index, goals)
        heuristic = np.where(np.isfinite(costs), costs, 0.0)
        cache[key] = heuristic
        return heuristic

    def _global_astar_static_raw_cost_to_go_for_env(
        self,
        env_index: int,
        goals: list[tuple[int, int]],
    ) -> np.ndarray:
        G = int(self.fire_grid_size)
        version = int(getattr(self, "_ugv_static_planner_cache_version", 0))
        goal_key = tuple((int(x), int(y)) for x, y in goals)
        key = (int(env_index), version, goal_key)
        cache = getattr(self, "_ugv_global_raw_cost_to_go_cache", None)
        if cache is None:
            self._ugv_global_raw_cost_to_go_cache = {}
            cache = self._ugv_global_raw_cost_to_go_cache
        cached = cache.get(key)
        if cached is not None:
            return cached

        tools = _scipy_sparse_tools()
        graph = self._ugv_static_planner_graph_for_env(env_index) if tools is not None else None
        if graph is not None:
            _csr_matrix, dijkstra = tools
            traversable, movement_cost = self._ugv_static_planner_layer_arrays_for_env(env_index)
            valid = traversable & np.isfinite(movement_cost)
            goal_indices = []
            for gx, gy in goal_key:
                if 0 <= gx < G and 0 <= gy < G and bool(valid[gy, gx]):
                    goal_indices.append(gy * G + gx)
            if goal_indices:
                distances = dijkstra(
                    graph,
                    directed=False,
                    indices=np.asarray(goal_indices, dtype=np.int32),
                    min_only=True,
                )
                costs = np.asarray(distances, dtype=np.float64).reshape(G, G)
            else:
                costs = np.full((G, G), np.inf, dtype=np.float64)
            cache[key] = costs
            return costs

        traversable, movement_cost = self._ugv_static_planner_layer_arrays_for_env(env_index)
        valid = traversable & np.isfinite(movement_cost)
        valid_flat = valid.reshape(-1)
        cost_flat = movement_cost.reshape(-1)

        costs = np.full((G, G), np.inf, dtype=np.float64)
        costs_flat = costs.reshape(-1)
        open_heap: list[tuple[float, int, int]] = []
        for goal in goal_key:
            gx, gy = goal
            if gx < 0 or gx >= G or gy < 0 or gy >= G:
                continue
            goal_idx = gy * G + gx
            if not bool(valid_flat[goal_idx]):
                continue
            if costs_flat[goal_idx] == 0.0:
                continue
            costs_flat[goal_idx] = 0.0
            heapq.heappush(open_heap, (0.0, gx, gy))

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
            cost_so_far, cx, cy = heapq.heappop(open_heap)
            current_idx = cy * G + cx
            if cost_so_far > costs_flat[current_idx] + 1e-9:
                continue
            current_cell_cost = float(cost_flat[current_idx])
            for ox, oy, step_len in neighbor_offsets:
                nx = cx + ox
                ny = cy + oy
                if nx < 0 or nx >= G or ny < 0 or ny >= G:
                    continue
                next_idx = ny * G + nx
                if not bool(valid_flat[next_idx]):
                    continue
                if ox != 0 and oy != 0 and (
                    not bool(valid[cy, nx]) or not bool(valid[ny, cx])
                ):
                    continue
                edge_cost = step_len * (
                    current_cell_cost + float(cost_flat[next_idx])
                ) * 0.5
                new_cost = cost_so_far + edge_cost
                if new_cost < costs_flat[next_idx]:
                    costs_flat[next_idx] = new_cost
                    heapq.heappush(open_heap, (new_cost, nx, ny))

        cache[key] = costs
        return costs

    def _global_astar_heuristic_grid(
        self,
        env_index: int,
        goals: list[tuple[int, int]],
        traversable: np.ndarray,
        movement_cost: np.ndarray,
    ) -> np.ndarray:
        if self.ugv_global_planner_heuristic == "terrain":
            return self._global_astar_static_cost_to_go_for_env(env_index, goals)
        return self._global_astar_euclidean_heuristic_grid(goals, traversable, movement_cost)

    def _global_astar_grid_path(
        self,
        env_index: int,
        start: tuple[int, int],
        goals: list[tuple[int, int]],
        traversable: np.ndarray,
        movement_cost: np.ndarray,
        *,
        heuristic_goals: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        if not goals:
            return []
        G = int(self.fire_grid_size)
        goal_set = set(goals)

        valid = traversable & np.isfinite(movement_cost)
        sx, sy = start
        if not (0 <= sx < G and 0 <= sy < G and bool(valid[sy, sx])):
            return []
        if start in goal_set:
            return [start]
        goals = [
            (int(gx), int(gy))
            for gx, gy in goals
            if 0 <= int(gx) < G and 0 <= int(gy) < G and bool(valid[int(gy), int(gx)])
        ]
        if not goals:
            return []
        finite_open_cost = movement_cost[valid]
        if finite_open_cost.size == 0:
            return []
        heuristic_grid = self._global_astar_heuristic_grid(
            env_index,
            heuristic_goals if heuristic_goals is not None else goals,
            traversable,
            movement_cost,
        )

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

        total_cells = G * G
        start_idx = sy * G + sx
        goal_mask = np.zeros(total_cells, dtype=bool)
        for gx, gy in goals:
            goal_mask[gy * G + gx] = True

        valid_flat = valid.reshape(-1)
        cost_flat = movement_cost.reshape(-1)
        heuristic_flat = heuristic_grid.reshape(-1)
        best_cost = np.full(total_cells, np.inf, dtype=np.float64)
        predecessor = np.full(total_cells, -1, dtype=np.int32)
        best_cost[start_idx] = 0.0
        open_heap = [(float(heuristic_flat[start_idx]), 0.0, sx, sy, start_idx)]
        while open_heap:
            _, cost_so_far, cx, cy, current_idx = heapq.heappop(open_heap)
            if goal_mask[current_idx]:
                path: list[tuple[int, int]] = []
                cursor = int(current_idx)
                while cursor >= 0:
                    path.append((int(cursor % G), int(cursor // G)))
                    cursor = int(predecessor[cursor])
                path.reverse()
                return path
            if cost_so_far > float(best_cost[current_idx]) + 1e-9:
                continue
            current_cell_cost = float(cost_flat[current_idx])
            for ox, oy, step_len in neighbor_offsets:
                nx = cx + ox
                ny = cy + oy
                if nx < 0 or nx >= G or ny < 0 or ny >= G:
                    continue
                next_idx = ny * G + nx
                if not bool(valid_flat[next_idx]):
                    continue
                if ox != 0 and oy != 0 and (
                    not bool(valid[cy, nx]) or not bool(valid[ny, cx])
                ):
                    continue
                edge_cost = step_len * (
                    current_cell_cost + float(cost_flat[next_idx])
                ) * 0.5
                new_cost = cost_so_far + edge_cost
                if new_cost < float(best_cost[next_idx]):
                    best_cost[next_idx] = new_cost
                    predecessor[next_idx] = int(current_idx)
                    heapq.heappush(
                        open_heap,
                        (new_cost + float(heuristic_flat[next_idx]), new_cost, nx, ny, next_idx),
                    )
        return []

    def _global_route_waypoint_for_env(
        self,
        env_index: int,
        pos: Tensor,
        path: list[tuple[int, int]],
        *,
        start_idx: int,
    ) -> tuple[tuple[int, int], int, int]:
        if len(path) < 2:
            cell = path[0] if path else self._single_position_to_grid_cell(pos)
            return cell, 0, 0
        pos_cell = self._single_position_to_grid_cell(pos)
        start_idx = max(0, min(int(start_idx), len(path) - 1))
        nearest_offset, _nearest_cell = min(
            enumerate(path[start_idx:]),
            key=lambda item: (
                item[1][0] - pos_cell[0]
            ) ** 2 + (
                item[1][1] - pos_cell[1]
            ) ** 2,
        )
        nearest_idx = start_idx + int(nearest_offset)
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        cell_w_m = (2.0 * float(self.x_semidim) / float(self.fire_grid_size)) / max(scale, 1e-9)
        cell_h_m = (2.0 * float(self.y_semidim) / float(self.fire_grid_size)) / max(scale, 1e-9)
        traversable, _movement_cost = self._ugv_planner_layer_tensors_for_env(env_index)
        waypoint = path[min(nearest_idx + 1, len(path) - 1)]
        waypoint_idx = min(nearest_idx + 1, len(path) - 1)
        accumulated_m = 0.0
        previous = path[nearest_idx]
        for idx in range(nearest_idx + 1, len(path)):
            candidate = path[idx]
            dx = abs(candidate[0] - previous[0])
            dy = abs(candidate[1] - previous[1])
            step_m = math.hypot(dx * cell_w_m, dy * cell_h_m)
            next_accumulated = accumulated_m + step_m
            if next_accumulated > self.ugv_global_planner_lookahead_m and idx > nearest_idx + 1:
                break
            if not self._grid_segment_is_traversable(traversable, path[nearest_idx], candidate):
                break
            waypoint = candidate
            waypoint_idx = idx
            accumulated_m = next_accumulated
            previous = candidate
        return waypoint, nearest_idx, waypoint_idx

    def _global_route_remaining_distance_m_for_env(
        self,
        env_index: int,
        pos: Tensor,
        path: list[tuple[int, int]],
        *,
        start_idx: int,
    ) -> float:
        if not path:
            return 0.0
        scale = float(self.terrain_sim_units_per_meter[env_index].detach().cpu().item())
        if scale <= 1e-9:
            return 0.0
        start_idx = max(0, min(int(start_idx), len(path) - 1))
        cell_w_m = (2.0 * float(self.x_semidim) / float(self.fire_grid_size)) / scale
        cell_h_m = (2.0 * float(self.y_semidim) / float(self.fire_grid_size)) / scale
        start_center = self._grid_cell_center_to_world(
            path[start_idx],
            device=pos.device,
            dtype=pos.dtype,
        )
        remaining = float(torch.linalg.norm(start_center - pos).detach().cpu().item()) / scale
        previous = path[start_idx]
        for cell in path[start_idx + 1:]:
            dx = abs(int(cell[0]) - int(previous[0]))
            dy = abs(int(cell[1]) - int(previous[1]))
            remaining += math.hypot(dx * cell_w_m, dy * cell_h_m)
            previous = cell
        return max(float(remaining), 0.0)

    def _local_astar_route_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        ground_index: int | None = None,
    ) -> tuple[tuple[int, int], bool, bool] | None:
        pos_cell = self._single_position_to_grid_cell(pos)
        target_cell = self._single_position_to_grid_cell(target_pos)
        if ground_index is None:
            return self._local_astar_route_uncached_for_env(
                env_index,
                pos,
                target_pos,
                pos_cell=pos_cell,
                target_cell=target_cell,
            )

        key = (
            int(env_index),
            int(ground_index),
            int(pos_cell[0]),
            int(pos_cell[1]),
            int(target_cell[0]),
            int(target_cell[1]),
            int(self.ugv_planner_patch_size),
            int(self.ugv_planner_lookahead_cells),
            int(getattr(self, "_ugv_planner_terrain_cache_version", 0)),
        )
        if not hasattr(self, "_ugv_planner_route_cache"):
            self._ugv_planner_route_cache = {}
        if key not in self._ugv_planner_route_cache:
            self._ugv_planner_route_cache[key] = self._local_astar_route_uncached_for_env(
                env_index,
                pos,
                target_pos,
                pos_cell=pos_cell,
                target_cell=target_cell,
            )
        return self._ugv_planner_route_cache[key]

    def _local_astar_route_uncached_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        pos_cell: tuple[int, int] | None = None,
        target_cell: tuple[int, int] | None = None,
    ) -> tuple[tuple[int, int], bool, bool] | None:
        patch_size = self.ugv_planner_patch_size
        radius = patch_size // 2
        pos_cell = pos_cell if pos_cell is not None else self._single_position_to_grid_cell(pos)
        target_cell = target_cell if target_cell is not None else self._single_position_to_grid_cell(target_pos)
        sx, sy = pos_cell
        x0 = max(0, sx - radius)
        x1 = min(self.fire_grid_size - 1, sx + radius)
        y0 = max(0, sy - radius)
        y1 = min(self.fire_grid_size - 1, sy + radius)
        bounds = (x0, x1, y0, y1)
        planner_traversable, planner_cost = self._ugv_planner_layer_tensors_for_env(env_index)

        traversable_patch = (
            planner_traversable[y0 : y1 + 1, x0 : x1 + 1]
            .detach()
            .cpu()
            .numpy()
            .astype(bool, copy=False)
        )
        movement_cost_patch = planner_cost[y0 : y1 + 1, x0 : x1 + 1].detach().cpu().numpy()

        def in_bounds(cell: tuple[int, int]) -> bool:
            x, y = cell
            return x0 <= x <= x1 and y0 <= y <= y1

        def local(cell: tuple[int, int]) -> tuple[int, int]:
            x, y = cell
            return x - x0, y - y0

        def traversable_at(cell: tuple[int, int]) -> bool:
            if not in_bounds(cell):
                return False
            lx, ly = local(cell)
            return bool(traversable_patch[ly, lx])

        def cost_at(cell: tuple[int, int]) -> float:
            lx, ly = local(cell)
            return float(movement_cost_patch[ly, lx])

        def open_cell(cell: tuple[int, int]) -> bool:
            if not traversable_at(cell):
                return False
            return math.isfinite(cost_at(cell))

        def nearest_traversable_cell(gx: int, gy: int) -> tuple[int, int] | None:
            gx_clamped = max(x0, min(x1, gx))
            gy_clamped = max(y0, min(y1, gy))
            if traversable_at((gx_clamped, gy_clamped)):
                return gx_clamped, gy_clamped
            max_radius = max(x1 - x0, y1 - y0)
            for search_radius in range(1, max_radius + 1):
                candidates = []
                cy0, cy1 = max(y0, gy_clamped - search_radius), min(y1, gy_clamped + search_radius)
                cx0, cx1 = max(x0, gx_clamped - search_radius), min(x1, gx_clamped + search_radius)
                for y in range(cy0, cy1 + 1):
                    candidates.extend(((cx0, y), (cx1, y)))
                for x in range(cx0 + 1, cx1):
                    candidates.extend(((x, cy0), (x, cy1)))
                valid = [(x, y) for x, y in candidates if traversable_at((x, y))]
                if valid:
                    return min(
                        valid,
                        key=lambda cell: (cell[0] - gx_clamped) ** 2 + (cell[1] - gy_clamped) ** 2,
                    )
            return None

        def local_planner_goal_candidates(
            start: tuple[int, int],
            target: tuple[int, int],
        ) -> list[tuple[int, int]]:
            tx, ty = target
            if x0 <= tx <= x1 and y0 <= ty <= y1:
                nearest = nearest_traversable_cell(tx, ty)
                return [] if nearest is None else [nearest]

            start_x, start_y = start
            dir_x = float(tx - start_x)
            dir_y = float(ty - start_y)
            dir_norm = max(math.hypot(dir_x, dir_y), 1e-9)
            dir_x /= dir_norm
            dir_y /= dir_norm
            boundary = []
            interior = []
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if not traversable_at((x, y)) or (x, y) == start:
                        continue
                    dx = x - start_x
                    dy = y - start_y
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

        def grid_segment_cells(
            start: tuple[int, int],
            end: tuple[int, int],
        ) -> list[tuple[int, int]]:
            start_x, start_y = start
            end_x, end_y = end
            steps = max(abs(end_x - start_x), abs(end_y - start_y))
            if steps == 0:
                return [start]
            cells = []
            previous: tuple[int, int] | None = None
            for i in range(steps + 1):
                x = int(round(start_x + (end_x - start_x) * i / steps))
                y = int(round(start_y + (end_y - start_y) * i / steps))
                cell = (x, y)
                if cell != previous:
                    cells.append(cell)
                    previous = cell
            return cells

        def segment_is_traversable(
            start: tuple[int, int],
            end: tuple[int, int],
        ) -> bool:
            cells = grid_segment_cells(start, end)
            previous = cells[0]
            if not traversable_at(previous):
                return False
            for cell in cells[1:]:
                x, y = cell
                px, py = previous
                if not traversable_at(cell):
                    return False
                if x != px and y != py:
                    if not traversable_at((px, y)) or not traversable_at((x, py)):
                        return False
                previous = cell
            return True

        def grid_path_cost(path: list[tuple[int, int]]) -> float:
            if len(path) < 2:
                return 0.0
            total = 0.0
            for (path_x0, path_y0), (path_x1, path_y1) in zip(path[:-1], path[1:]):
                if (
                    not traversable_at((path_x0, path_y0))
                    or not traversable_at((path_x1, path_y1))
                ):
                    return float("inf")
                if path_x0 != path_x1 and path_y0 != path_y1:
                    if (
                        not traversable_at((path_x0, path_y1))
                        or not traversable_at((path_x1, path_y0))
                    ):
                        return float("inf")
                    step_len = math.sqrt(2.0)
                else:
                    step_len = 1.0
                c0 = cost_at((path_x0, path_y0))
                c1 = cost_at((path_x1, path_y1))
                if not math.isfinite(c0) or not math.isfinite(c1):
                    return float("inf")
                total += step_len * (c0 + c1) * 0.5
            return total

        def local_astar_grid_path(
            start: tuple[int, int],
            goal: tuple[int, int],
        ) -> list[tuple[int, int]]:
            if start == goal:
                return [start]
            if not open_cell(start) or not open_cell(goal):
                return []
            finite_open_cost = movement_cost_patch[
                traversable_patch & np.isfinite(movement_cost_patch)
            ]
            if finite_open_cost.size == 0:
                return []
            min_cost = float(finite_open_cost.min())
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
                current_x, current_y = current
                for ox, oy, step_len in neighbor_offsets:
                    nxt = (current_x + ox, current_y + oy)
                    if not open_cell(nxt):
                        continue
                    if ox != 0 and oy != 0 and (
                        not open_cell((current_x + ox, current_y))
                        or not open_cell((current_x, current_y + oy))
                    ):
                        continue
                    nx, ny = nxt
                    edge_cost = step_len * (
                        cost_at((current_x, current_y)) + cost_at((nx, ny))
                    ) * 0.5
                    new_cost = cost_so_far + edge_cost
                    if new_cost < best_cost.get(nxt, float("inf")):
                        best_cost[nxt] = new_cost
                        came_from[nxt] = current
                        heapq.heappush(open_heap, (new_cost + heuristic(nxt), new_cost, nxt))
            return []

        def route_lookahead_cell(path: list[tuple[int, int]]) -> tuple[int, int]:
            start = path[0]
            best = path[min(1, len(path) - 1)]
            stop = min(self.ugv_planner_lookahead_cells, len(path) - 1)
            for idx in range(2, stop + 1):
                candidate = path[idx]
                if not segment_is_traversable(start, candidate):
                    break
                best = candidate
            return best

        def local_astar_detour_needed(
            start: tuple[int, int],
            goal: tuple[int, int],
            waypoint: tuple[int, int],
            path: list[tuple[int, int]],
            direct_blocked: bool,
        ) -> bool:
            if direct_blocked:
                return True
            start_x, start_y = start
            goal_x, goal_y = goal
            waypoint_x, waypoint_y = waypoint
            direct_vec = (goal_x - start_x, goal_y - start_y)
            waypoint_vec = (waypoint_x - start_x, waypoint_y - start_y)
            direct_norm = math.hypot(*direct_vec)
            waypoint_norm = math.hypot(*waypoint_vec)
            direction_detour = False
            if direct_norm > 1e-9 and waypoint_norm > 1e-9:
                cos_to_goal = (
                    direct_vec[0] * waypoint_vec[0] + direct_vec[1] * waypoint_vec[1]
                ) / (direct_norm * waypoint_norm)
                direction_detour = cos_to_goal < math.cos(math.radians(30.0))

            direct_cells = grid_segment_cells(start, goal)
            direct_cost = grid_path_cost(direct_cells)
            astar_cost = grid_path_cost(path)
            cost_detour = (
                math.isfinite(direct_cost)
                and math.isfinite(astar_cost)
                and astar_cost > 1e-9
                and direct_cost > astar_cost * 1.25
            )
            return direction_detour or cost_detour

        start = nearest_traversable_cell(sx, sy)
        if start is None:
            return None

        goal_candidates = local_planner_goal_candidates(
            start,
            target_cell,
        )
        if not goal_candidates:
            return None

        path: list[tuple[int, int]] = []
        goal = goal_candidates[0]
        for candidate in goal_candidates:
            path = local_astar_grid_path(start, candidate)
            if len(path) >= 2:
                goal = candidate
                break
        if len(path) < 2:
            return None

        direct_blocked = not segment_is_traversable(start, goal)
        waypoint = route_lookahead_cell(path)
        detour_needed = local_astar_detour_needed(start, goal, waypoint, path, direct_blocked)
        return waypoint, direct_blocked, detour_needed

    def _local_escape_astar_route_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        ground_index: int | None = None,
    ) -> tuple[tuple[int, int], bool, bool] | None:
        plan = self._local_escape_astar_plan_cached_for_env(
            env_index,
            pos,
            target_pos,
            ground_index=ground_index,
        )
        return None if plan is None else plan["route"]

    def _local_escape_astar_plan_cached_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        ground_index: int | None = None,
    ) -> dict | None:
        pos_cell = self._single_position_to_grid_cell(pos)
        target_cell = self._single_position_to_grid_cell(target_pos)
        if ground_index is None:
            return self._local_escape_astar_plan_for_env(
                env_index,
                pos,
                target_pos,
                pos_cell=pos_cell,
                target_cell=target_cell,
            )

        key = (
            "local_escape_astar_plan",
            int(env_index),
            int(ground_index),
            int(pos_cell[0]),
            int(pos_cell[1]),
            int(target_cell[0]),
            int(target_cell[1]),
            int(self.ugv_planner_patch_size),
            int(self.ugv_planner_lookahead_cells),
            int(getattr(self, "_ugv_planner_terrain_cache_version", 0)),
        )
        if not hasattr(self, "_ugv_planner_route_cache"):
            self._ugv_planner_route_cache = {}
        if key not in self._ugv_planner_route_cache:
            self._ugv_planner_route_cache[key] = self._local_escape_astar_plan_for_env(
                env_index,
                pos,
                target_pos,
                pos_cell=pos_cell,
                target_cell=target_cell,
            )
        return self._ugv_planner_route_cache[key]

    def _local_escape_astar_route_info_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
    ) -> dict | None:
        return self._local_escape_astar_plan_for_env(
            env_index,
            pos,
            target_pos,
            include_details=True,
        )

    def _local_escape_astar_plan_for_env(
        self,
        env_index: int,
        pos: Tensor,
        target_pos: Tensor,
        *,
        pos_cell: tuple[int, int] | None = None,
        target_cell: tuple[int, int] | None = None,
        include_details: bool = False,
    ) -> dict | None:
        patch_size = self.ugv_planner_patch_size
        radius = patch_size // 2
        pos_cell = pos_cell if pos_cell is not None else self._single_position_to_grid_cell(pos)
        target_cell = target_cell if target_cell is not None else self._single_position_to_grid_cell(target_pos)
        sx, sy = pos_cell
        tx, ty = target_cell
        x0 = max(0, sx - radius)
        x1 = min(self.fire_grid_size - 1, sx + radius)
        y0 = max(0, sy - radius)
        y1 = min(self.fire_grid_size - 1, sy + radius)
        bounds = (x0, x1, y0, y1)

        traversable, movement_cost = self._ugv_planner_layer_tensors_for_env(env_index)

        def in_bounds(cell: tuple[int, int]) -> bool:
            x, y = cell
            return x0 <= x <= x1 and y0 <= y <= y1

        def open_cell(cell: tuple[int, int]) -> bool:
            x, y = cell
            if not in_bounds(cell):
                return False
            return bool(traversable[y, x].item()) and math.isfinite(float(movement_cost[y, x].item()))

        def segment_blocked_fraction(start: tuple[int, int], goal: tuple[int, int]) -> float:
            cells = self._grid_segment_cells(start, goal)
            blocked = 0
            checks = 0
            previous = cells[0] if cells else None
            for cell in cells:
                checks += 1
                if not open_cell(cell):
                    blocked += 1
                if previous is not None:
                    px, py = previous
                    x, y = cell
                    if x != px and y != py:
                        checks += 2
                        if not open_cell((px, y)):
                            blocked += 1
                        if not open_cell((x, py)):
                            blocked += 1
                previous = cell
            return float(blocked) / float(max(checks, 1))

        def local_openness(cell: tuple[int, int]) -> float:
            x, y = cell
            total = 0
            open_count = 0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    total += 1
                    if open_cell((x + ox, y + oy)):
                        open_count += 1
            return float(open_count) / float(max(total, 1))

        start = self._nearest_traversable_cell_in_bounds(env_index, sx, sy, bounds)
        if start is None:
            return None

        old_route = self._local_astar_route_uncached_for_env(
            env_index,
            pos,
            target_pos,
            pos_cell=pos_cell,
            target_cell=target_cell,
        )
        if old_route is not None:
            waypoint, direct_blocked, detour_needed = old_route
            if not (direct_blocked and detour_needed):
                old_goal = waypoint
                old_path = [start, waypoint] if waypoint != start else [start]
                if include_details:
                    goal_candidates = self._local_planner_goal_candidates(env_index, start, target_cell, bounds)
                    for candidate in goal_candidates:
                        candidate_path = self._local_astar_grid_path(env_index, start, candidate, bounds)
                        if len(candidate_path) >= 2:
                            old_goal = candidate
                            old_path = candidate_path
                            break
                target_corridor_blocked_fraction = segment_blocked_fraction(start, old_goal)
                path = [start, waypoint] if waypoint != start else [start]
                return {
                    "route": old_route,
                    "start": start,
                    "goal": old_goal,
                    "waypoint": waypoint,
                    "path": old_path,
                    "escape_mode": False,
                    "direct_blocked": bool(direct_blocked),
                    "detour_needed": bool(detour_needed),
                    "exit_clearance_cells": None,
                    "exit_openness": local_openness(waypoint) if include_details else None,
                    "target_corridor_blocked_fraction": target_corridor_blocked_fraction,
                }
        else:
            target_corridor_blocked_fraction = segment_blocked_fraction(start, target_cell)

        goal_candidates = self._local_planner_goal_candidates(env_index, start, target_cell, bounds)
        old_goal = goal_candidates[0] if goal_candidates else start
        if goal_candidates:
            target_corridor_blocked_fraction = segment_blocked_fraction(start, old_goal)
        else:
            target_corridor_blocked_fraction = segment_blocked_fraction(start, target_cell)

        finite_cost = movement_cost[y0 : y1 + 1, x0 : x1 + 1]
        finite_mask = torch.isfinite(finite_cost)
        if not bool(finite_mask.any().item()):
            return None

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
        open_heap = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best_cost: dict[tuple[int, int], float] = {start: 0.0}
        while open_heap:
            cost_so_far, current = heapq.heappop(open_heap)
            if cost_so_far > best_cost.get(current, float("inf")) + 1e-9:
                continue
            cx, cy = current
            for ox, oy, step_len in neighbor_offsets:
                nxt = (cx + ox, cy + oy)
                if not open_cell(nxt):
                    continue
                if ox != 0 and oy != 0 and (
                    not open_cell((cx + ox, cy)) or not open_cell((cx, cy + oy))
                ):
                    continue
                nx, ny = nxt
                edge_cost = step_len * (
                    float(movement_cost[cy, cx].item()) + float(movement_cost[ny, nx].item())
                ) * 0.5
                new_cost = cost_so_far + edge_cost
                if new_cost < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (new_cost, nxt))

        def reconstruct(goal: tuple[int, int]) -> list[tuple[int, int]]:
            path = [goal]
            current = goal
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        reachable = [cell for cell in best_cost if cell != start]
        boundary = [
            cell
            for cell in reachable
            if cell[0] in (x0, x1) or cell[1] in (y0, y1)
        ]
        candidates = boundary or reachable
        if not candidates:
            return None

        target_vec = (float(tx - start[0]), float(ty - start[1]))
        target_norm = max(math.hypot(*target_vec), 1e-9)
        max_cost = max(max(best_cost.values()), 1e-9)
        blocked_cells = [
            (x, y)
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
            if not open_cell((x, y))
        ]

        def clearance_cells(cell: tuple[int, int]) -> float:
            if not blocked_cells:
                return float(radius + 1)
            x, y = cell
            return min(math.hypot(x - bx, y - by) for bx, by in blocked_cells)

        max_clearance = max(float(radius), 1.0)
        best_cell: tuple[int, int] | None = None
        best_score: tuple[float, float, float, float, float, float] | None = None
        best_path: list[tuple[int, int]] = []
        best_clearance = 0.0
        best_openness = 0.0
        for cell in candidates:
            cell_vec = (float(cell[0] - start[0]), float(cell[1] - start[1]))
            cell_norm = max(math.hypot(*cell_vec), 1e-9)
            target_alignment = (
                cell_vec[0] * target_vec[0] + cell_vec[1] * target_vec[1]
            ) / (cell_norm * target_norm)
            target_alignment = max(min(target_alignment, 1.0), -1.0)
            path_cost_norm = min(max(best_cost[cell] / max_cost, 0.0), 1.0)
            target_dist_norm = min(math.hypot(tx - cell[0], ty - cell[1]) / target_norm, 1.0)
            openness = local_openness(cell)
            clearance = clearance_cells(cell)
            clearance_norm = min(max(clearance / max_clearance, 0.0), 1.0)
            dead_end_penalty = 1.0 - openness
            score = (
                -0.25 * path_cost_norm
                + 0.20 * clearance_norm
                + 0.15 * openness
                - 0.10 * dead_end_penalty
                + 0.35 * target_alignment
                - 0.10 * target_dist_norm
            )
            score_tuple = (
                score,
                clearance_norm,
                openness,
                target_alignment,
                -path_cost_norm,
                -target_dist_norm,
            )
            if best_score is None or score_tuple > best_score:
                best_cell = cell
                best_score = score_tuple
                best_path = reconstruct(cell)
                best_clearance = clearance
                best_openness = openness

        if best_cell is None or len(best_path) < 2:
            return None

        selected_direct_blocked = not self._grid_segment_is_traversable(traversable, start, best_cell)
        waypoint = self._route_lookahead_cell(traversable, best_path, 0)
        detour_needed = True
        route = (waypoint, bool(selected_direct_blocked), bool(detour_needed))
        return {
            "route": route,
            "start": start,
            "goal": best_cell,
            "waypoint": waypoint,
            "path": best_path,
            "escape_mode": True,
            "direct_blocked": bool(selected_direct_blocked),
            "detour_needed": bool(detour_needed),
            "exit_clearance_cells": best_clearance,
            "exit_openness": best_openness,
            "target_corridor_blocked_fraction": target_corridor_blocked_fraction,
        }

    def _clear_ugv_escape_route_for_env(self, env_index: int, ground_index: int) -> None:
        self.ugv_escape_route_active[env_index, ground_index] = False
        self.ugv_escape_route_age[env_index, ground_index] = 0
        self.ugv_escape_route_path_index[env_index, ground_index] = 0
        self.ugv_escape_route_target_idx[env_index, ground_index] = -1
        self.ugv_escape_route_goal_cell[env_index, ground_index] = -1
        self.ugv_escape_route_waypoint_cell[env_index, ground_index] = -1
        self.ugv_escape_route_paths[env_index][ground_index] = []

    def _start_ugv_escape_route_for_env(
        self,
        env_index: int,
        ground_index: int,
        target_idx: int,
        path: list[tuple[int, int]],
    ) -> bool:
        if len(path) < 2:
            return False
        self.ugv_escape_route_active[env_index, ground_index] = True
        self.ugv_escape_route_age[env_index, ground_index] = 0
        self.ugv_escape_route_stall_counter[env_index, ground_index] = 0
        self.ugv_escape_route_target_idx[env_index, ground_index] = int(target_idx)
        self.ugv_escape_route_path_index[env_index, ground_index] = 0
        self.ugv_escape_route_paths[env_index][ground_index] = list(path)
        self.ugv_escape_route_goal_cell[env_index, ground_index, 0] = int(path[-1][0])
        self.ugv_escape_route_goal_cell[env_index, ground_index, 1] = int(path[-1][1])
        waypoint = self._route_lookahead_cell(self.traversable_grid[env_index], path, 0)
        self.ugv_escape_route_waypoint_cell[env_index, ground_index, 0] = int(waypoint[0])
        self.ugv_escape_route_waypoint_cell[env_index, ground_index, 1] = int(waypoint[1])
        return True

    def _ugv_escape_route_switch_rewards(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        target_pos: Tensor,
        target_idx: Tensor,
        gate: Tensor,
        ground_progress_m: Tensor,
        movement_alignment: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        progress_m = torch.zeros_like(gate, dtype=start_pos.dtype)
        progress_scaled = torch.zeros_like(progress_m)
        route_movement_alignment = torch.zeros_like(progress_m)
        reward_active = torch.zeros_like(gate, dtype=torch.bool)
        entered = torch.zeros_like(gate, dtype=torch.bool)
        exited = torch.zeros_like(gate, dtype=torch.bool)
        waypoint_distance_m = torch.zeros_like(progress_m)
        path_index = torch.zeros_like(progress_m)
        path_length = torch.zeros_like(progress_m)
        if self.ugv_dense_reward_mode != "escape_route_switch" or start_pos.shape[1] == 0:
            return (
                progress_m,
                progress_scaled,
                route_movement_alignment,
                reward_active,
                entered,
                exited,
                waypoint_distance_m,
                path_index,
                path_length,
            )

        sim_units_per_meter = self.terrain_sim_units_per_meter.to(start_pos.device).clamp_min(1e-9)
        batch_dim, n_ground, _ = start_pos.shape
        for env_index in range(batch_dim):
            scale = sim_units_per_meter[env_index]
            for ground_index in range(n_ground):
                target_valid = bool(gate[env_index, ground_index].item())
                current_target_idx = int(target_idx[env_index, ground_index].item())
                was_active = bool(self.ugv_escape_route_active[env_index, ground_index].item())
                if was_active and (
                    not target_valid
                    or current_target_idx < 0
                    or int(self.ugv_escape_route_target_idx[env_index, ground_index].item()) != current_target_idx
                ):
                    self._clear_ugv_escape_route_for_env(env_index, ground_index)
                    exited[env_index, ground_index] = True
                    was_active = False

                if was_active:
                    waypoint = self._ugv_escape_route_waypoint_for_env(
                        env_index,
                        ground_index,
                        start_pos[env_index, ground_index],
                        update_index=False,
                    )
                    if waypoint is None:
                        self._clear_ugv_escape_route_for_env(env_index, ground_index)
                        exited[env_index, ground_index] = True
                        continue

                    waypoint_pos = self._grid_cell_center_to_world(
                        waypoint,
                        device=start_pos.device,
                        dtype=start_pos.dtype,
                    )
                    before_m = torch.linalg.norm(waypoint_pos - start_pos[env_index, ground_index]) / scale
                    after_m = torch.linalg.norm(waypoint_pos - end_pos[env_index, ground_index]) / scale
                    step_progress_m = before_m - after_m
                    progress_m[env_index, ground_index] = step_progress_m
                    progress_scaled[env_index, ground_index] = (
                        step_progress_m / self.ugv_planner_progress_scale_m
                    ).clamp(-1.0, 1.0)
                    route_movement_alignment[env_index, ground_index] = (
                        step_progress_m
                        / self.step_ugv_actual_displacement_m[env_index, ground_index].clamp_min(1e-6)
                    ).clamp(-1.0, 1.0)
                    waypoint_distance_m[env_index, ground_index] = after_m
                    reward_active[env_index, ground_index] = True

                    self._ugv_escape_route_waypoint_for_env(
                        env_index,
                        ground_index,
                        end_pos[env_index, ground_index],
                        update_index=True,
                    )
                    route_path = self.ugv_escape_route_paths[env_index][ground_index]
                    final_cell = route_path[-1] if route_path else waypoint
                    final_pos = self._grid_cell_center_to_world(
                        final_cell,
                        device=end_pos.device,
                        dtype=end_pos.dtype,
                    )
                    final_distance_m = torch.linalg.norm(
                        final_pos - end_pos[env_index, ground_index]
                    ) / scale
                    normal_route = self._local_astar_route_uncached_for_env(
                        env_index,
                        end_pos[env_index, ground_index],
                        target_pos[env_index, ground_index],
                    )
                    direct_open = normal_route is not None and not bool(normal_route[1])
                    next_age = int(self.ugv_escape_route_age[env_index, ground_index].item()) + 1
                    route_complete = float(final_distance_m.item()) <= self.ugv_escape_waypoint_reached_m
                    timed_out = next_age >= self.ugv_escape_max_steps
                    if route_complete or direct_open or timed_out:
                        self._clear_ugv_escape_route_for_env(env_index, ground_index)
                        exited[env_index, ground_index] = True
                    else:
                        self.ugv_escape_route_age[env_index, ground_index] = next_age
                        self.ugv_escape_route_stall_counter[env_index, ground_index] = 0
                    continue

                if not target_valid or current_target_idx < 0:
                    self.ugv_escape_route_stall_counter[env_index, ground_index] = 0
                    continue

                normal_route = self._local_astar_route_uncached_for_env(
                    env_index,
                    end_pos[env_index, ground_index],
                    target_pos[env_index, ground_index],
                )
                direct_blocked = normal_route is None or bool(normal_route[1])
                poor_progress = (
                    float(ground_progress_m[env_index, ground_index].item())
                    <= self.ugv_escape_progress_threshold_m
                )
                slow_or_sideways = (
                    float(self.step_ugv_actual_displacement_m[env_index, ground_index].item())
                    <= self.ugv_escape_movement_threshold_m
                    or float(movement_alignment[env_index, ground_index].item()) <= 0.25
                )
                if direct_blocked and poor_progress and slow_or_sideways:
                    self.ugv_escape_route_stall_counter[env_index, ground_index] += 1
                else:
                    self.ugv_escape_route_stall_counter[env_index, ground_index] = 0

                if int(self.ugv_escape_route_stall_counter[env_index, ground_index].item()) < self.ugv_escape_stall_steps:
                    continue

                plan = self._local_escape_astar_plan_for_env(
                    env_index,
                    end_pos[env_index, ground_index],
                    target_pos[env_index, ground_index],
                    include_details=False,
                )
                if plan is None or not bool(plan.get("escape_mode", False)):
                    continue
                path = plan.get("path", [])
                if self._start_ugv_escape_route_for_env(
                    env_index,
                    ground_index,
                    current_target_idx,
                    path,
                ):
                    entered[env_index, ground_index] = True

        for env_index in range(batch_dim):
            for ground_index in range(n_ground):
                if bool(self.ugv_escape_route_active[env_index, ground_index].item()):
                    route_path = self.ugv_escape_route_paths[env_index][ground_index]
                    path_index[env_index, ground_index] = float(
                        int(self.ugv_escape_route_path_index[env_index, ground_index].item())
                    )
                    path_length[env_index, ground_index] = float(len(route_path))
                    if float(waypoint_distance_m[env_index, ground_index].item()) == 0.0:
                        waypoint = self._ugv_escape_route_waypoint_for_env(
                            env_index,
                            ground_index,
                            end_pos[env_index, ground_index],
                            update_index=False,
                        )
                        if waypoint is not None:
                            waypoint_pos = self._grid_cell_center_to_world(
                                waypoint,
                                device=end_pos.device,
                                dtype=end_pos.dtype,
                            )
                            waypoint_distance_m[env_index, ground_index] = (
                                torch.linalg.norm(
                                    waypoint_pos - end_pos[env_index, ground_index]
                                )
                                / sim_units_per_meter[env_index]
                            )
        return (
            progress_m,
            progress_scaled,
            route_movement_alignment,
            reward_active,
            entered,
            exited,
            waypoint_distance_m,
            path_index,
            path_length,
        )

    def _ugv_planner_progress_rewards(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        target_pos: Tensor,
        target_idx: Tensor,
        gate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        reward = torch.zeros_like(gate, dtype=start_pos.dtype)
        progress_m = torch.zeros_like(reward)
        progress_scaled = torch.zeros_like(reward)
        active = torch.zeros_like(gate, dtype=torch.bool)
        direct_blocked_out = torch.zeros_like(gate, dtype=torch.bool)
        detour_needed_out = torch.zeros_like(gate, dtype=torch.bool)
        escape_mode_out = torch.zeros_like(gate, dtype=torch.bool)
        required_progress_m = torch.zeros_like(reward)
        shortfall_m = torch.zeros_like(reward)
        remaining_distance_m = torch.zeros_like(reward)
        if (
            self.ugv_planner_hint not in UGV_PLANNER_HINT_MODES
            or (
                self.r_ugv_planner_progress <= 0.0
                and self.r_ugv_route_progress_shortfall_penalty <= 0.0
                and not self.ugv_route_aware_reward
                and self.ugv_dense_reward_mode != "planner_blend"
                and self.ugv_dense_reward_mode != "escape_blend"
                and self.ugv_dense_reward_mode != "planner_follow"
            )
            or start_pos.shape[1] == 0
        ):
            return (
                reward,
                progress_m,
                progress_scaled,
                active,
                direct_blocked_out,
                detour_needed_out,
                escape_mode_out,
                required_progress_m,
                shortfall_m,
                remaining_distance_m,
            )

        sim_units_per_meter = self.terrain_sim_units_per_meter.to(start_pos.device).clamp_min(1e-9)
        batch_dim, n_ground, _ = start_pos.shape
        for env_index in range(batch_dim):
            scale = sim_units_per_meter[env_index]
            for ground_index in range(n_ground):
                if not bool(gate[env_index, ground_index].item()):
                    continue
                escape_mode = False
                plan_info = None
                if self.ugv_planner_hint == "local_escape_astar":
                    plan = self._local_escape_astar_plan_cached_for_env(
                        env_index,
                        start_pos[env_index, ground_index],
                        target_pos[env_index, ground_index],
                        ground_index=ground_index,
                    )
                    route = None if plan is None else plan["route"]
                    escape_mode = bool(plan.get("escape_mode", False)) if plan is not None else False
                elif self.ugv_planner_hint == "global_astar":
                    plan_info = self._global_astar_route_info_for_env(
                        env_index,
                        start_pos[env_index, ground_index],
                        target_pos[env_index, ground_index],
                        ground_index=ground_index,
                        target_idx=int(target_idx[env_index, ground_index].item()),
                        update_index=True,
                    )
                    route = None if plan_info is None else plan_info["route"]
                else:
                    route = self._ugv_planner_route_for_env(
                        env_index,
                        start_pos[env_index, ground_index],
                        target_pos[env_index, ground_index],
                        ground_index=ground_index,
                        target_idx=int(target_idx[env_index, ground_index].item()),
                    )
                if route is None:
                    if self.ugv_planner_hint == "global_astar":
                        if bool(self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index].item()):
                            self.metric_ugv_route_replanned_after_fire[env_index] += 1.0
                            self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index] = False
                        if bool(self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index].item()):
                            self.metric_ugv_route_fire_blocked_no_path[env_index] += 1.0
                            self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index] = False
                    continue
                waypoint, direct_blocked, detour_needed = route
                direct_blocked_out[env_index, ground_index] = direct_blocked
                detour_needed_out[env_index, ground_index] = detour_needed
                escape_mode_out[env_index, ground_index] = escape_mode
                if not detour_needed and self.ugv_dense_reward_mode != "planner_follow":
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
                route_remaining_m = before_m
                if plan_info is not None:
                    route_remaining_m = torch.as_tensor(
                        self._global_route_remaining_distance_m_for_env(
                            env_index,
                            start_pos[env_index, ground_index],
                            plan_info.get("path", []),
                            start_idx=int(plan_info.get("path_index", 0)),
                        ),
                        device=start_pos.device,
                        dtype=start_pos.dtype,
                    )
                remaining_steps = max(
                    int(self.max_steps) - int(self.step_count[env_index].item()),
                    1,
                )
                required_m = route_remaining_m / float(remaining_steps)
                required_progress_m[env_index, ground_index] = required_m
                remaining_distance_m[env_index, ground_index] = route_remaining_m
                shortfall_m[env_index, ground_index] = (required_m - step_progress_m).clamp(min=0.0)
                reward[env_index, ground_index] = step_progress_scaled * self.r_ugv_planner_progress
                active[env_index, ground_index] = True
                if self.ugv_planner_hint == "global_astar":
                    self.metric_ugv_global_route_valid[env_index] += 1.0
                    self.metric_ugv_global_route_active[env_index] += 1.0
                    self.metric_ugv_global_route_waypoint_distance_m[env_index] += before_m
                    self.metric_ugv_global_route_progress_m[env_index] += step_progress_m
                    self.metric_ugv_global_route_progress_scaled[env_index] += step_progress_scaled
                    self.metric_ugv_global_route_direct_blocked[env_index] += float(direct_blocked)
                    self.metric_ugv_global_route_detour_needed[env_index] += float(detour_needed)
                    if hasattr(self, "ugv_global_route_paths"):
                        route_path = self.ugv_global_route_paths[env_index][ground_index]
                        self.metric_ugv_global_route_path_length[env_index] += float(len(route_path))
                        fire_stats = self._ugv_route_fire_stats_for_env(env_index, route_path)
                        self.metric_ugv_route_fire_cells[env_index] += fire_stats["fire_cells"]
                        self.metric_ugv_route_smoke_mean[env_index] += fire_stats["smoke_mean"]
                        self.metric_ugv_route_smolder_mean[env_index] += fire_stats["smolder_mean"]
                        self.metric_ugv_route_fire_buffer_cells[env_index] += fire_stats["fire_buffer_cells"]
                    if bool(self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index].item()):
                        self.metric_ugv_route_replanned_after_fire[env_index] += 1.0
                        self.ugv_global_route_replanned_after_fire_flag[env_index, ground_index] = False
                    if bool(self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index].item()):
                        self.metric_ugv_route_fire_blocked_no_path[env_index] += 1.0
                        self.ugv_global_route_fire_blocked_no_path_flag[env_index, ground_index] = False
                    if hasattr(self, "ugv_global_route_path_index"):
                        self.metric_ugv_global_route_path_index[env_index] += (
                            self.ugv_global_route_path_index[env_index, ground_index].float()
                        )
        return (
            reward,
            progress_m,
            progress_scaled,
            active,
            direct_blocked_out,
            detour_needed_out,
            escape_mode_out,
            required_progress_m,
            shortfall_m,
            remaining_distance_m,
        )

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
        *,
        traversable: Tensor | None = None,
    ) -> tuple[int, int] | None:
        x0, x1, y0, y1 = bounds
        gx = max(x0, min(x1, gx))
        gy = max(y0, min(y1, gy))
        if traversable is None:
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
        *,
        traversable: Tensor | None = None,
    ) -> list[tuple[int, int]]:
        x0, x1, y0, y1 = bounds
        tx, ty = target
        if traversable is None:
            traversable = self.traversable_grid[env_index]
        if x0 <= tx <= x1 and y0 <= ty <= y1:
            nearest = self._nearest_traversable_cell_in_bounds(
                env_index,
                tx,
                ty,
                bounds,
                traversable=traversable,
            )
            return [] if nearest is None else [nearest]

        sx, sy = start
        dir_x = float(tx - sx)
        dir_y = float(ty - sy)
        dir_norm = max(math.hypot(dir_x, dir_y), 1e-9)
        dir_x /= dir_norm
        dir_y /= dir_norm
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
        traversable, movement_cost = self._ugv_planner_layer_tensors_for_env(env_index)

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
        traversable, movement_cost = self._ugv_planner_layer_tensors_for_env(env_index)
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

    def _obs_schema_agent_index(self, agent: Agent) -> int:
        agent_idx = self.world.agents.index(agent)
        if agent.is_drone:
            return agent_idx
        return self.obs_schema_n_drones + (agent_idx - self.n_drones)

    def _neighbor_observations(self, agent: Agent, comms_keep: Tensor) -> Tensor:
        if self.obs_schema_n_agents == self.n_agents:
            deltas = []
            for other in self.world.agents:
                if other is agent:
                    continue
                deltas.append(other.state.pos - agent.state.pos)
            if not deltas:
                return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)
            rel = torch.cat(deltas, dim=-1)
            return rel * comms_keep.float()

        schema_self = self._obs_schema_agent_index(agent)
        out = torch.zeros(
            self.world.batch_dim,
            max(self.obs_schema_n_agents - 1, 0) * 2,
            device=agent.state.pos.device,
            dtype=agent.state.pos.dtype,
        )
        for other in self.world.agents:
            if other is agent:
                continue
            other_schema = self._obs_schema_agent_index(other)
            if other_schema == schema_self or other_schema >= self.obs_schema_n_agents:
                continue
            compact_slot = other_schema if other_schema < schema_self else other_schema - 1
            out[:, 2 * compact_slot : 2 * compact_slot + 2] = other.state.pos - agent.state.pos
        return out * comms_keep.float()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    def done(self) -> Tensor:
        all_found = self._all_active_survivors_found()
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
            "n_found":   (self.found_survivors & self._active_survivor_mask()).sum(dim=1).float(),
            "n_scouted": (self.scouted_survivors & self._active_survivor_mask()).sum(dim=1).float(),
            "n_active_survivors": self._active_survivor_count(),
            "mission/new_scouts": self.metric_new_scouts,
            "mission/new_oracle_reveals": self.metric_survivor_oracle_reveals,
            "mission/n_oracle_revealed": (
                self.survivor_oracle_revealed & self._active_survivor_mask()
            ).sum(dim=1).float(),
            "mission/new_decoy_oracle_reveals": self.metric_decoy_oracle_reveals,
            "mission/n_active_decoys": self._active_decoy_count(),
            "mission/n_decoy_oracle_revealed": (
                self.decoy_oracle_revealed & self._active_decoy_mask()
            ).sum(dim=1).float(),
            "mission/new_confirmations": self.metric_new_confirmations,
            "mission/n_active_survivors": self._active_survivor_count(),
            "mission/n_scouted": (
                self.scouted_survivors & self._active_survivor_mask()
            ).sum(dim=1).float(),
            "mission/n_confirmed": (
                self.found_survivors & self._active_survivor_mask()
            ).sum(dim=1).float(),
            "mission/full_success": self.metric_full_success,
            **(
                {
                    "mission/false_positive_detections": self.metric_false_positive_detections,
                    "mission/false_positive_trips": self.metric_false_positive_trips,
                }
                if self.n_decoys > 0
                else {}
            ),
            "reward/team": self.metric_reward_team,
            "reward/all_survivors_found": self.metric_reward_all_survivors_found,
            "reward/team_scout": self.metric_reward_team_scout,
            "reward/pending_penalty": self.metric_reward_pending_penalty,
            "reward/drone_scout": self.metric_reward_drone_scout,
            "reward/drone_progress": self.metric_reward_drone_progress,
            "reward/uav_move_coverage": self.metric_reward_uav_move_coverage,
            "reward/uav_inefficient_move": self.metric_reward_uav_inefficient_move,
            "reward/uav_coverage_threshold": self.metric_reward_uav_coverage_threshold,
            "reward/uav_frontier_alignment": self.metric_reward_uav_frontier_alignment,
            "reward/uav_confidence": self.metric_reward_uav_confidence,
            "reward/uav_team_confidence": self.metric_reward_uav_team_confidence,
            "reward/uav_team_confidence_overlap": self.metric_reward_uav_team_confidence_overlap,
            "reward/uav_confidence_move": self.metric_reward_uav_confidence_move,
            "reward/uav_confidence_overlap": self.metric_reward_uav_confidence_overlap,
            "reward/uav_cleanup_target_progress": self.metric_reward_uav_cleanup_target_progress,
            "reward/uav_astar_progress": self.metric_reward_uav_astar_progress,
            "reward/uav_overlap": self.metric_reward_uav_overlap,
            "reward/uav_inter_uav_overlap": self.metric_reward_uav_inter_uav_overlap,
            "reward/uav_outside_footprint": self.metric_reward_uav_outside_footprint,
            "reward/ugv_progress": self.metric_reward_ugv_progress,
            "reward/ugv_approach": self.metric_reward_ugv_approach,
            "reward/ugv_movement_alignment": self.metric_reward_ugv_movement_alignment,
            "reward/ugv_planner_progress": self.metric_reward_ugv_planner_progress,
            "reward/ugv_stall_penalty": self.metric_reward_ugv_stall_penalty,
            "reward/ugv_route_progress_floor_penalty": self.metric_reward_ugv_route_progress_floor_penalty,
            "reward/ugv_route_progress_shortfall_penalty": (
                self.metric_reward_ugv_route_progress_shortfall_penalty
            ),
            "reward/ground_confirm": self.metric_reward_ground_confirm,
            "reward/coverage": self.metric_reward_coverage,
            "cost/ugv_fire_exposure": self.metric_cost_ugv_fire_exposure,
            "cost/ugv_travel": self.metric_cost_ugv_travel,
            "cost/drone_energy": self.metric_cost_drone_energy,
            "cost/drone_climb": self.metric_cost_drone_climb,
            "diagnostic/ugv_proposed_path_blocked": self.step_ugv_proposed_path_blocked.float().sum(dim=1),
            "diagnostic/uav_overlap_fraction": self.metric_uav_overlap_fraction,
            "diagnostic/uav_inter_uav_overlap_fraction": self.metric_uav_inter_uav_overlap_fraction,
            "diagnostic/uav_frontier_alignment": self.metric_uav_frontier_alignment,
            "diagnostic/uav_frontier_progress_fraction": self.metric_uav_frontier_progress_fraction,
            "diagnostic/uav_frontier_uncovered_ratio": self.metric_uav_frontier_uncovered_ratio,
            "diagnostic/uav_astar_progress_fraction": self.metric_uav_astar_progress_fraction,
            "diagnostic/uav_astar_frontier_gate": self.metric_uav_astar_frontier_gate,
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
            "diagnostic/ugv_duplicate_assignment_fraction": self.metric_ugv_duplicate_assignment_fraction,
            "diagnostic/ugv_assignment_switches": self.metric_ugv_assignment_switches,
            "diagnostic/ugv_ground_progress_m": self.metric_ugv_ground_progress_m,
            "diagnostic/ugv_ground_progress_scaled": self.metric_ugv_ground_progress_scaled,
            "diagnostic/ugv_route_progress_floor_shortfall_m": self.metric_ugv_route_progress_floor_shortfall_m,
            "diagnostic/ugv_route_progress_required_m": self.metric_ugv_route_progress_required_m,
            "diagnostic/ugv_route_progress_shortfall_m": self.metric_ugv_route_progress_shortfall_m,
            "diagnostic/ugv_route_remaining_distance_m": self.metric_ugv_route_remaining_distance_m,
            "diagnostic/ugv_planner_progress_m": self.metric_ugv_planner_progress_m,
            "diagnostic/ugv_planner_progress_scaled": self.metric_ugv_planner_progress_scaled,
            "diagnostic/ugv_planner_active": self.metric_ugv_planner_active,
            "diagnostic/ugv_planner_direct_blocked": self.metric_ugv_planner_direct_blocked,
            "diagnostic/ugv_planner_detour_needed": self.metric_ugv_planner_detour_needed,
            "diagnostic/ugv_planner_escape_mode": self.metric_ugv_planner_escape_mode,
            "diagnostic/ugv_escape_route_active": self.metric_ugv_escape_route_active,
            "diagnostic/ugv_escape_route_enter": self.metric_ugv_escape_route_enter,
            "diagnostic/ugv_escape_route_exit": self.metric_ugv_escape_route_exit,
            "diagnostic/ugv_escape_route_stall_counter": self.metric_ugv_escape_route_stall_counter,
            "diagnostic/ugv_escape_route_age": self.metric_ugv_escape_route_age,
            "diagnostic/ugv_escape_route_waypoint_progress_m": (
                self.metric_ugv_escape_route_waypoint_progress_m
            ),
            "diagnostic/ugv_escape_route_waypoint_progress_scaled": (
                self.metric_ugv_escape_route_waypoint_progress_scaled
            ),
            "diagnostic/ugv_escape_route_waypoint_distance_m": (
                self.metric_ugv_escape_route_waypoint_distance_m
            ),
            "diagnostic/ugv_escape_route_path_index": self.metric_ugv_escape_route_path_index,
            "diagnostic/ugv_escape_route_path_length": self.metric_ugv_escape_route_path_length,
            "diagnostic/ugv_global_route_valid": self.metric_ugv_global_route_valid,
            "diagnostic/ugv_global_route_active": self.metric_ugv_global_route_active,
            "diagnostic/ugv_global_route_waypoint_distance_m": (
                self.metric_ugv_global_route_waypoint_distance_m
            ),
            "diagnostic/ugv_global_route_progress_m": self.metric_ugv_global_route_progress_m,
            "diagnostic/ugv_global_route_progress_scaled": self.metric_ugv_global_route_progress_scaled,
            "diagnostic/ugv_global_route_path_index": self.metric_ugv_global_route_path_index,
            "diagnostic/ugv_global_route_path_length": self.metric_ugv_global_route_path_length,
            "diagnostic/ugv_global_route_direct_blocked": self.metric_ugv_global_route_direct_blocked,
            "diagnostic/ugv_global_route_detour_needed": self.metric_ugv_global_route_detour_needed,
            "diagnostic/ugv_route_fire_cells": self.metric_ugv_route_fire_cells,
            "diagnostic/ugv_route_smoke_mean": self.metric_ugv_route_smoke_mean,
            "diagnostic/ugv_route_smolder_mean": self.metric_ugv_route_smolder_mean,
            "diagnostic/ugv_route_fire_buffer_cells": self.metric_ugv_route_fire_buffer_cells,
            "diagnostic/ugv_route_replanned_after_fire": self.metric_ugv_route_replanned_after_fire,
            "diagnostic/ugv_route_fire_blocked_no_path": self.metric_ugv_route_fire_blocked_no_path,
            "diagnostic/ugv_route_aware_active": self.metric_ugv_route_aware_active,
            "diagnostic/ugv_action_alignment": self.metric_ugv_action_alignment,
            "diagnostic/ugv_movement_alignment": self.metric_ugv_movement_alignment,
            "diagnostic/uav_final_target_distance_m": self.metric_uav_target_distance_m,
            "diagnostic/uav_min_target_distance_m": self.metric_uav_target_distance_m,
            "diagnostic/uav_footprint_radius_m": self.metric_uav_footprint_radius_m,
            "diagnostic/uav_steps_with_target_in_footprint": self.metric_uav_target_within_footprint,
            "diagnostic/uav_displacement_m": self.metric_uav_displacement_m,
            "diagnostic/uav_new_coverage_cells": self.metric_uav_new_coverage_cells,
            "diagnostic/uav_coverage_opportunity_cells": self.metric_uav_coverage_opportunity_cells,
            "diagnostic/uav_coverage_opportunity_fraction": self.metric_uav_coverage_opportunity_fraction,
            "diagnostic/uav_coverage_opportunity_available_fraction": (
                self.metric_uav_coverage_opportunity_available_fraction
            ),
            "diagnostic/uav_confidence_mean": self.metric_uav_confidence_mean,
            "diagnostic/uav_confidence_gain": self.metric_uav_confidence_gain,
            "diagnostic/uav_weighted_confidence_gain": self.metric_uav_weighted_confidence_gain,
            "diagnostic/uav_confidence_opportunity_fraction": (
                self.metric_uav_confidence_opportunity_fraction
            ),
            "diagnostic/uav_confidence_opportunity_best_gain": (
                self.metric_uav_confidence_opportunity_best_gain
            ),
            "diagnostic/uav_confidence_overlap_fraction": (
                self.metric_uav_confidence_overlap_fraction
            ),
            "diagnostic/uav_cleanup_target_valid_fraction": (
                self.metric_uav_cleanup_target_valid_by_drone.mean(dim=1)
                if self.n_drones > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/uav_cleanup_target_distance_m": (
                self.metric_uav_cleanup_target_distance_m_by_drone.mean(dim=1)
                if self.n_drones > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/uav_cleanup_target_value": (
                self.metric_uav_cleanup_target_value_by_drone.mean(dim=1)
                if self.n_drones > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/uav_cleanup_target_progress_m": (
                self.metric_uav_cleanup_target_progress_m_by_drone.mean(dim=1)
                if self.n_drones > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/uav_cleanup_target_frontier_gate": (
                self.metric_uav_cleanup_target_frontier_gate_by_drone.mean(dim=1)
                if self.n_drones > 0
                else torch.zeros(self.world.batch_dim, device=self.fire_grid.device)
            ),
            "diagnostic/uav_confidence_low_fraction": self.metric_uav_confidence_low_fraction,
            "diagnostic/uav_confidence_high_fraction": self.metric_uav_confidence_high_fraction,
            "diagnostic/uav_step_detection_probability": self.metric_uav_step_detection_probability,
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
    p.add_argument("--comms-dropout-mode", choices=("iid", "bursty"), default="iid")
    p.add_argument("--comms-map-mode", choices=("global", "per_agent", "per-agent"), default="global")
    p.add_argument("--comms-dropout-min-steps", type=int, default=5)
    p.add_argument("--comms-dropout-max-steps", type=int, default=15)
    args = p.parse_args()
    args.comms_map_mode = str(args.comms_map_mode).replace("-", "_")
    render_interactively(
        WildfireSearchScenario(),
        control_two_agents=True,
        n_drones=args.n_drones,
        n_ground=args.n_ground,
        n_survivors=args.n_survivors,
        fire_grid_size=args.grid_size,
        comms_dropout=args.comms_dropout,
        comms_dropout_mode=args.comms_dropout_mode,
        comms_map_mode=args.comms_map_mode,
        comms_dropout_min_steps=args.comms_dropout_min_steps,
        comms_dropout_max_steps=args.comms_dropout_max_steps,
        terrain_cache_path=args.terrain_cache_path,
        terrain_place=args.terrain_place,
    )
