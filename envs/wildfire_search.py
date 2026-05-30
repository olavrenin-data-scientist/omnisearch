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

import math
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


# Indices into agent position tensors
X, Y = 0, 1

# Land-cover types stored in land_cover_grid. Terrain affects ground robots;
# drones fly above it but observe the map to coordinate ground routes.
LAND_ROAD, LAND_OPEN, LAND_BRUSH, LAND_FOREST, LAND_ROCK, LAND_WATER = range(6)
OBJECT_NONE, OBJECT_TREE, OBJECT_HOUSE = range(3)


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
        self.x_semidim = kwargs.pop("x_semidim", 1.0)
        self.y_semidim = kwargs.pop("y_semidim", 1.0)

        # Detection / sensing
        # Drone search uses a downward camera footprint, not a fixed magic
        # radius: altitude * tan(FOV / 2) gives the visible ground radius.
        kwargs.pop("drone_lidar_range", None)  # legacy name; replaced by camera FOV.
        self.drone_camera_fov_deg = kwargs.pop("drone_camera_fov_deg", 65.0)
        if not 0.0 < self.drone_camera_fov_deg < 180.0:
            raise ValueError("drone_camera_fov_deg must be between 0 and 180")
        self.drone_camera_half_angle_tan = math.tan(math.radians(self.drone_camera_fov_deg) / 2.0)
        self.ground_lidar_range = kwargs.pop("ground_lidar_range", 0.20)
        self.n_lidar_rays       = kwargs.pop("n_lidar_rays", 12)
        self.detection_range    = kwargs.pop("detection_range", 0.10)   # ground confirm radius

        # Fire spread (cellular automata on a discrete grid overlay)
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
        self.fire_burnout_min_updates = max(int(kwargs.pop("fire_burnout_min_updates", 5)), 1)
        self.fire_burnout_max_updates = max(
            int(kwargs.pop("fire_burnout_max_updates", 14)),
            self.fire_burnout_min_updates,
        )
        self.fire_step_interval  = kwargs.pop("fire_step_interval", 3)  # spread every N env steps
        self.smoke_emission = kwargs.pop("smoke_emission", 0.18)
        self.smoke_decay = kwargs.pop("smoke_decay", 0.96)
        self.smoke_diffusion = kwargs.pop("smoke_diffusion", 0.16)
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
        self.slope_speed_weight = kwargs.pop("slope_speed_weight", 1.5)
        self.terrain_path_samples = kwargs.pop("terrain_path_samples", 6)
        land_cover_costs = _land_cover_values(
            kwargs.pop("land_cover_costs", (0.65, 1.0, 1.5, 2.2, 4.0, 8.0)),
            water_value=8.0,
            name="land_cover_costs",
        )
        land_cover_speeds = _land_cover_values(
            kwargs.pop("land_cover_speeds", (1.0, 0.9, 0.65, 0.45, 0.0, 0.0)),
            water_value=0.0,
            name="land_cover_speeds",
        )

        # 2.5D drone flight: horizontal VMAS motion plus an automatic safe
        # continuous AGL altitude. MSL altitude is derived from local terrain elevation.
        # Meter-based anchors are converted to sim units after loading terrain metadata.
        drone_flight_levels_m = kwargs.pop("drone_flight_levels_m", (30.0, 60.0, 90.0))
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
        self.drone_safety_clearance_m = max(float(kwargs.pop("drone_safety_clearance_m", 15.0)), 0.0)
        self.drone_climb_rate = max(float(kwargs.pop("drone_climb_rate", 0.035)), 0.0)
        self.drone_descent_rate = max(float(kwargs.pop("drone_descent_rate", 0.020)), 0.0)
        self.drone_climb_rate_m = max(float(kwargs.pop("drone_climb_rate_m", 10.0)), 0.0)
        self.drone_descent_rate_m = max(float(kwargs.pop("drone_descent_rate_m", 8.0)), 0.0)
        self.drone_altitude_release_margin_m = max(float(kwargs.pop("drone_altitude_release_margin_m", 10.0)), 0.0)
        self.drone_altitude_release_margin = max(float(kwargs.pop("drone_altitude_release_margin", 0.04)), 0.0)
        self.r_drone_climb_cost = kwargs.pop("r_drone_climb_cost", -0.02)
        self.drone_sensor_max_range = float(max(drone_flight_levels) * self.drone_camera_half_angle_tan)

        # Communication
        self.comms_dropout = kwargs.pop("comms_dropout", 0.0)

        # Episode
        self.max_steps = kwargs.pop("max_steps", 500)

        # Reward weights
        self.r_found_survivor = kwargs.pop("r_found_survivor", 1.0)
        self.r_drone_scout    = kwargs.pop("r_drone_scout", 0.3)
        self.r_ground_confirm = kwargs.pop("r_ground_confirm", 0.5)
        self.r_time_penalty   = kwargs.pop("r_time_penalty", -0.001)
        self.r_fire_penalty   = kwargs.pop("r_fire_penalty", -1.0)
        self.r_ground_travel_cost = kwargs.pop("r_ground_travel_cost", -0.05)
        self.r_drone_shaping  = kwargs.pop("r_drone_shaping",  0.05)
        self.r_ground_shaping = kwargs.pop("r_ground_shaping", 0.10)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Physical sizes
        self.agent_radius    = 0.04
        self.survivor_radius = 0.03

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
                max_speed=0.5,
                u_range=1.0,
                u_multiplier=0.6,
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
                max_speed=0.2,
                u_range=1.0,
                u_multiplier=0.3,
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
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self._prev_ground_pos = torch.zeros(batch_dim, self.n_ground, 2, device=device)
        self._pre_step_ground_pos = torch.zeros_like(self._prev_ground_pos)
        self._pre_step_drone_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.step_ugv_travel_cost = torch.zeros(batch_dim, self.n_ground, device=device)
        self.prev_drone_dist  = torch.full((batch_dim, self.n_drones),  float("inf"), device=device)
        self.prev_ground_dist = torch.full((batch_dim, self.n_ground), float("inf"), device=device)
        self.terrain_source_description = ["real"] * batch_dim
        self.terrain_source_metadata = [{} for _ in range(batch_dim)]

        # Per-agent reward buffers (filled in _compute_step_rewards)
        for agent in world.agents:
            agent.scenario_reward = torch.zeros(batch_dim, device=device)

        return world

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            entities=self._survivors + self.world.agents,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=2 * self.agent_radius + 0.02,
            x_bounds=(-self.x_semidim, self.x_semidim),
            y_bounds=(-self.y_semidim, self.y_semidim),
        )

        if env_index is None:
            self.found_survivors.zero_()
            self.scouted_survivors.zero_()
            self.fire_grid.zero_()
            self.burned_grid.zero_()
            self.fire_age_grid.zero_()
            self.fire_lifetime_grid.zero_()
            self.fire_intensity_grid.zero_()
            self.smoke_grid.zero_()
            self.step_count.zero_()
            self.prev_drone_dist.fill_(float("inf"))
            self.prev_ground_dist.fill_(float("inf"))
            envs_to_seed = range(self.world.batch_dim)
        else:
            self.found_survivors[env_index] = False
            self.scouted_survivors[env_index] = False
            self.fire_grid[env_index] = False
            self.burned_grid[env_index] = False
            self.fire_age_grid[env_index] = 0
            self.fire_lifetime_grid[env_index] = 0
            self.fire_intensity_grid[env_index] = 0.0
            self.smoke_grid[env_index] = 0.0
            self.step_count[env_index] = 0
            self.prev_drone_dist[env_index]  = float("inf")
            self.prev_ground_dist[env_index] = float("inf")
            envs_to_seed = [env_index]

        H = W = self.fire_grid_size
        for b in envs_to_seed:
            self._generate_terrain(b)
            self._seed_initial_fire(b, H, W)

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
        random_lifetime = torch.randint(
            self.fire_burnout_min_updates,
            self.fire_burnout_max_updates + 1,
            self.fire_lifetime_grid.shape,
            device=self.fire_lifetime_grid.device,
            dtype=self.fire_lifetime_grid.dtype,
        )
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
        self._relocate_land_entities_to_valid_cells(env_index)
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
        )
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
        )
        self.terrain_source_description[env_index] = terrain.source
        self.terrain_source_metadata[env_index] = dict(terrain.metadata)
        sim_units_per_meter = self._terrain_sim_units_per_meter(terrain.metadata)
        self.terrain_sim_units_per_meter[env_index] = sim_units_per_meter
        self._refresh_drone_unit_conversions(env_index, sim_units_per_meter)
        if self.drone_safety_clearance_sim_override is None and sim_units_per_meter > 0.0:
            clearance = self.drone_safety_clearance_m * sim_units_per_meter
        else:
            clearance = self.drone_safety_clearance_by_env[env_index].item()
        self.drone_safety_clearance_by_env[env_index] = max(float(clearance), 0.0)
        self.drone_safety_clearance = float(self.drone_safety_clearance_by_env.mean().item())

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

    def _terrain_sim_units_per_meter(self, metadata: dict) -> float:
        """Return the terrain cache's vertical/horizontal simulation conversion."""
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
                return value_f

        bounds = metadata.get("inputs", {}).get("usgs_3dep", {}).get("projected_bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            west_m, south_m, east_m, north_m = (float(v) for v in bounds)
            extent_m = max(abs(east_m - west_m), abs(north_m - south_m), 1e-6)
            return 2.0 / extent_m

        cell_size_m = metadata.get("inputs", {}).get("usgs_3dep", {}).get("cell_size_m")
        try:
            return 2.0 / max(float(cell_size_m) * float(self.fire_grid_size), 1e-6)
        except (TypeError, ValueError):
            return 0.0

    def _relocate_land_entities_to_valid_cells(self, env_index: int) -> None:
        """Move survivors and UGV starts onto feasible terrain without editing it."""
        candidates = self._land_entity_spawn_candidate_mask(env_index)
        if not bool(candidates.any().item()):
            return

        size = self.fire_grid_size
        device = self.fire_grid.device
        yy, xx = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
        available = candidates.clone()
        entities = self._survivors + self.world.agents[self.n_drones:]
        for entity in entities:
            gx, gy = self._positions_to_grid(entity.state.pos[env_index].view(1, 1, 2))
            x, y = int(gx.item()), int(gy.item())
            if bool(candidates[y, x].item()):
                available[y, x] = False
                continue

            search_mask = available if bool(available.any().item()) else candidates
            dist2 = (xx - x).float().square() + (yy - y).float().square()
            dist2 = torch.where(search_mask, dist2, torch.full_like(dist2, float("inf")))
            best = int(dist2.flatten().argmin().item())
            new_y, new_x = divmod(best, size)
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
        drone_agents = self.world.agents[:self.n_drones]
        if drone_agents:
            self._pre_step_drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1).clone()
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
        """Emit smoke from burning cells, then diffuse, drift, and decay."""
        smoke = self.smoke_grid * self.smoke_decay
        fuel = self._fire_fuel_grid()
        flame_output = self.fire_grid.float() * self.fire_intensity_grid.clamp(0.15, 1.0)
        smoke = smoke + flame_output * self.smoke_emission * fuel.clamp(0.0, 1.0)
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
        """Reduce ground-robot traction/speed on slow surfaces and slopes."""
        if agent.is_drone:
            return
        speed = self._grid_values_at_positions(
            self.speed_multiplier_grid, agent.state.pos.unsqueeze(1),
        ).squeeze(1)
        agent.action.u = agent.action.u * speed.unsqueeze(-1)

    def post_step(self):
        """Apply blocked ground routes and auto-select safe drone altitude."""
        ground_agents = self.world.agents[self.n_drones:]
        if ground_agents:
            ground_pos = torch.stack([a.state.pos for a in ground_agents], dim=1)
            traversable = self._path_is_traversable(self._pre_step_ground_pos, ground_pos)
            for i, agent in enumerate(ground_agents):
                blocked = ~traversable[:, i]
                soft_pos = self._soft_blocked_ground_position(
                    self._pre_step_ground_pos[:, i], agent.state.pos,
                )
                corrected_pos = torch.where(
                    blocked.unsqueeze(-1), soft_pos, agent.state.pos,
                )
                soft_vel = (soft_pos - self._pre_step_ground_pos[:, i]) / self.world.dt
                corrected_vel = torch.where(
                    blocked.unsqueeze(-1), soft_vel, agent.state.vel,
                )
                agent.set_pos(corrected_pos, batch_index=None)
                agent.set_vel(corrected_vel, batch_index=None)

        drone_agents = self.world.agents[:self.n_drones]
        if drone_agents:
            drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1)
            self._update_drone_altitudes(self._pre_step_drone_pos, drone_pos)
        self._clamp_agents_to_world()

    def _clamp_agents_to_world(self) -> None:
        """Keep agent bodies inside the visible world bounds."""
        x_min, x_max = -self.x_semidim + self.agent_radius, self.x_semidim - self.agent_radius
        y_min, y_max = -self.y_semidim + self.agent_radius, self.y_semidim - self.agent_radius
        for agent in self.world.agents:
            pos = agent.state.pos
            clamped = pos.clone()
            clamped[:, X] = clamped[:, X].clamp(x_min, x_max)
            clamped[:, Y] = clamped[:, Y].clamp(y_min, y_max)
            hit_boundary = (clamped != pos).any(dim=-1, keepdim=True)
            vel = torch.where(hit_boundary, torch.zeros_like(agent.state.vel), agent.state.vel)
            agent.set_pos(clamped, batch_index=None)
            agent.set_vel(vel, batch_index=None)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, agent: Agent) -> Tensor:
        if agent is self.world.agents[0]:
            self._compute_step_rewards()
        return agent.scenario_reward

    def _compute_step_rewards(self):
        device = self.fire_grid.device

        agent_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B, A, 2]
        surv_pos  = torch.stack([s.state.pos for s in self._survivors], dim=1)    # [B, S, 2]
        dists = torch.cdist(agent_pos, surv_pos)                                  # [B, A, S]

        drone_pos = agent_pos[:, :self.n_drones, :]
        drone_dists = dists[:, :self.n_drones, :]
        drone_seen = self._drone_survivor_detections(drone_dists, drone_pos, surv_pos)
        seen_by_drone       = drone_seen.any(dim=1)
        within_confirm      = dists < self.detection_range
        confirmed_by_ground = within_confirm[:, self.n_drones:, :].any(dim=1)

        newly_scouted = seen_by_drone       & ~self.scouted_survivors & ~self.found_survivors
        newly_found   = confirmed_by_ground & ~self.found_survivors

        self.scouted_survivors = self.scouted_survivors | newly_scouted
        self.found_survivors   = self.found_survivors   | newly_found

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
        all_scouted = ~unscouted.any(dim=1, keepdim=True)
        prev_known = ~torch.isinf(self.prev_drone_dist) & ~all_scouted
        drone_shaping = torch.where(
            prev_known,
            (self.prev_drone_dist - curr_drone_dist) * self.r_drone_shaping,
            torch.zeros_like(curr_drone_dist),
        )
        self.prev_drone_dist = curr_drone_dist

        # Ground robots: target = scouted-but-unconfirmed survivors
        unconfirmed_scouted = self.scouted_survivors & ~self.found_survivors
        ground_d = torch.where(
            unconfirmed_scouted.unsqueeze(1),
            dists[:, self.n_drones:, :],
            torch.full_like(dists[:, self.n_drones:, :], INF),
        )
        curr_ground_dist, _ = ground_d.min(dim=2)
        all_done = ~unconfirmed_scouted.any(dim=1, keepdim=True)
        prev_known = ~torch.isinf(self.prev_ground_dist) & ~all_done
        ground_shaping = torch.where(
            prev_known,
            (self.prev_ground_dist - curr_ground_dist) * self.r_ground_shaping,
            torch.zeros_like(curr_ground_dist),
        )
        self.prev_ground_dist = curr_ground_dist

        team_reward = (
            newly_found.float().sum(dim=1) * self.r_found_survivor
            + self.r_time_penalty
        )

        scout_credit_mask    = drone_seen & newly_scouted.unsqueeze(1)
        scout_per_drone      = scout_credit_mask.float().sum(dim=2)         # [B, D]

        ground_within        = within_confirm[:, self.n_drones:, :]
        confirm_credit_mask  = ground_within & newly_found.unsqueeze(1)
        confirm_per_ground   = confirm_credit_mask.float().sum(dim=2)       # [B, G]

        ground_agents = self.world.agents[self.n_drones:]
        ground_in_fire = self._agents_in_fire(ground_agents)  # [B, G]
        if ground_agents:
            ground_pos = torch.stack([a.state.pos for a in ground_agents], dim=1)
            self.step_ugv_travel_cost = self._terrain_path_cost(self._prev_ground_pos, ground_pos)
            self._prev_ground_pos = ground_pos.clone()
        else:
            self.step_ugv_travel_cost.zero_()

        for i, agent in enumerate(self.world.agents):
            r = team_reward.clone()
            if agent.is_drone:
                r = r + scout_per_drone[:, i] * self.r_drone_scout
                r = r - self.drone_energy_cost[:, i]
                r = r + self.step_drone_climb[:, i] * self.r_drone_climb_cost
                r = r + drone_shaping[:, i]
            else:
                g = i - self.n_drones
                r = r + confirm_per_ground[:, g] * self.r_ground_confirm
                r = r + ground_in_fire[:, g].float() * self.r_fire_penalty
                r = r + self.step_ugv_travel_cost[:, g] * self.r_ground_travel_cost
                r = r + ground_shaping[:, g]
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
        path = self._sample_path(start_pos, end_pos)
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

    def _drone_camera_ranges(self) -> Tensor:
        """Ground footprint radius for each drone's current flight altitude."""
        return self.drone_altitude * self.drone_camera_half_angle_tan

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
        terrain_local = self._local_terrain_features(agent)  # [B, 27]
        flight_state = self._flight_state(agent)             # [B, 2]
        neighbor   = self._neighbor_observations(agent)     # [B, (A-1)*2]
        return torch.cat(
            [own_pos, own_vel, lidar_obs, fire_local, terrain_local, flight_state, neighbor], dim=-1,
        )

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
        gx = ((pos[..., X] + self.x_semidim) / (2 * self.x_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2
        ).long()
        gy = ((pos[..., Y] + self.y_semidim) / (2 * self.y_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2
        ).long()
        b_idx = torch.arange(self.world.batch_dim, device=pos.device)
        nearby_costs = []
        nearby_blocked = []
        nearby_clearance = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nearby_costs.append(self.mobility_cost_grid[b_idx, gy + dy, gx + dx])
                nearby_blocked.append(~self.traversable_grid[b_idx, gy + dy, gx + dx])
                nearby_clearance.append(self.required_clearance_grid[b_idx, gy + dy, gx + dx])
        costs = torch.stack(nearby_costs, dim=-1)
        blocked = torch.stack(nearby_blocked, dim=-1).float()
        clearance = torch.stack(nearby_clearance, dim=-1)
        normalized_costs = costs / self.mobility_cost_grid.amax(dim=(1, 2), keepdim=False).unsqueeze(-1)
        normalized_clearance = clearance / self.drone_max_altitude_by_env.unsqueeze(-1).clamp_min(1e-6)
        return torch.cat([normalized_costs, blocked, normalized_clearance], dim=-1)

    def _flight_state(self, agent: Agent) -> Tensor:
        state = torch.zeros(self.world.batch_dim, 2, device=self.fire_grid.device)
        if agent.is_drone:
            drone_idx = self.world.agents.index(agent)
            state[:, 0] = self.drone_altitude[:, drone_idx] / self.drone_max_altitude_by_env.clamp_min(1e-6)
            state[:, 1] = self.drone_altitude_quality[:, drone_idx]
        return state

    def _neighbor_observations(self, agent: Agent) -> Tensor:
        deltas = []
        for other in self.world.agents:
            if other is agent:
                continue
            deltas.append(other.state.pos - agent.state.pos)
        if not deltas:
            return torch.zeros(self.world.batch_dim, 0, device=agent.state.pos.device)
        rel = torch.cat(deltas, dim=-1)
        if self.comms_dropout > 0:
            keep = (torch.rand_like(rel[..., :1]) > self.comms_dropout).float()
            rel = rel * keep
        return rel

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
            "n_burning": self.fire_grid.flatten(1).sum(dim=1).float(),
            "n_burned":  self.burned_grid.flatten(1).sum(dim=1).float(),
            "affected_fraction": self.burned_grid.float().flatten(1).mean(dim=1),
            "mean_fire_intensity": self.fire_intensity_grid.flatten(1).mean(dim=1),
            "ugv_step_travel_cost": self.step_ugv_travel_cost.sum(dim=1),
            "mean_drone_altitude": mean_drone_altitude,
            "mean_drone_altitude_msl": mean_drone_altitude_msl,
        }


if __name__ == "__main__":
    from vmas import render_interactively
    render_interactively(
        WildfireSearchScenario(),
        control_two_agents=True,
        n_drones=3,
        n_ground=2,
        n_survivors=5,
    )
