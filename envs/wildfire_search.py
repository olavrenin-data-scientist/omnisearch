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
LAND_ROAD, LAND_OPEN, LAND_BRUSH, LAND_FOREST, LAND_ROCK = range(5)
OBJECT_NONE, OBJECT_TREE, OBJECT_HOUSE = range(3)


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
        self.fire_grid_size      = kwargs.pop("fire_grid_size", 16)
        self.terrain_reference_grid_size = kwargs.pop("terrain_reference_grid_size", 16)
        self.fire_spread_prob    = kwargs.pop("fire_spread_prob", 0.10)
        self.fire_spread_variability = kwargs.pop("fire_spread_variability", 0.55)
        self.fire_spotting_prob = kwargs.pop("fire_spotting_prob", 0.00012)
        self.initial_fire_cells  = kwargs.pop("initial_fire_cells", 1)
        self.initial_fire_area_fraction = kwargs.pop("initial_fire_area_fraction", 0.025)
        self.wildfire_area_fraction_range = kwargs.pop("wildfire_area_fraction_range", (0.20, 0.40))
        if len(self.wildfire_area_fraction_range) != 2:
            raise ValueError("wildfire_area_fraction_range must be (low, high)")
        self.wildfire_area_fraction_range = (
            max(0.0, min(float(self.wildfire_area_fraction_range[0]), 1.0)),
            max(0.0, min(float(self.wildfire_area_fraction_range[1]), 1.0)),
        )
        if self.wildfire_area_fraction_range[1] < self.wildfire_area_fraction_range[0]:
            raise ValueError("wildfire_area_fraction_range high must be >= low")
        self.fire_burnout_min_updates = max(int(kwargs.pop("fire_burnout_min_updates", 5)), 1)
        self.fire_burnout_max_updates = max(
            int(kwargs.pop("fire_burnout_max_updates", 14)),
            self.fire_burnout_min_updates,
        )
        self.fire_step_interval  = kwargs.pop("fire_step_interval", 3)  # spread every N env steps
        self.smoke_emission = kwargs.pop("smoke_emission", 0.18)
        self.smoke_decay = kwargs.pop("smoke_decay", 0.96)
        self.smoke_diffusion = kwargs.pop("smoke_diffusion", 0.16)
        land_cover_fire_fuel = kwargs.pop("land_cover_fire_fuel", (0.05, 0.40, 1.10, 1.35, 0.0))
        object_fire_fuel = kwargs.pop("object_fire_fuel", (0.0, 0.25, 1.00))
        if len(land_cover_fire_fuel) != 5:
            raise ValueError("land_cover_fire_fuel must cover road/open/brush/forest/rock")
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
        land_cover_costs = kwargs.pop("land_cover_costs", (0.65, 1.0, 1.5, 2.2, 4.0))
        land_cover_speeds = kwargs.pop("land_cover_speeds", (1.0, 0.9, 0.65, 0.45, 0.0))
        if len(land_cover_costs) != 5 or len(land_cover_speeds) != 5:
            raise ValueError("land-cover cost and speed values must cover road/open/brush/forest/rock")

        # 2.5D drone flight: horizontal VMAS motion plus an automatic safe
        # AGL flight level. MSL altitude is derived from local terrain elevation.
        drone_flight_levels = kwargs.pop("drone_flight_levels", (0.18, 0.40, 0.70))
        drone_detection_quality = kwargs.pop(
            "drone_detection_quality",
            kwargs.pop("drone_detection_factors", (0.95, 0.75, 0.55)),
        )
        drone_energy_costs = kwargs.pop("drone_energy_costs", (0.0, 0.002, 0.006))
        if not (len(drone_flight_levels) == len(drone_detection_quality) == len(drone_energy_costs)):
            raise ValueError("drone flight levels, detection quality, and energy costs must align")
        drone_cover_detection_factors = kwargs.pop(
            "drone_cover_detection_factors", (1.0, 1.0, 0.72, 0.45, 0.35),
        )
        if len(drone_cover_detection_factors) != 5:
            raise ValueError("drone_cover_detection_factors must cover road/open/brush/forest/rock")
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
        self.drone_safety_clearance = kwargs.pop("drone_safety_clearance", 0.03)
        self.r_drone_climb_cost = kwargs.pop("r_drone_climb_cost", -0.02)
        self.drone_sensor_max_range = float(max(drone_flight_levels) * self.drone_camera_half_angle_tan)

        # Communication
        self.comms_dropout = kwargs.pop("comms_dropout", 0.0)

        # Episode
        self.max_steps = kwargs.pop("max_steps", 200)

        # Reward weights
        self.r_found_survivor = kwargs.pop("r_found_survivor", 1.0)
        self.r_drone_scout    = kwargs.pop("r_drone_scout", 0.3)
        self.r_ground_confirm = kwargs.pop("r_ground_confirm", 0.5)
        self.r_time_penalty   = kwargs.pop("r_time_penalty", -0.001)
        self.r_fire_penalty   = kwargs.pop("r_fire_penalty", -1.0)
        self.r_ground_travel_cost = kwargs.pop("r_ground_travel_cost", -0.05)

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
        target_low, target_high = self.wildfire_area_fraction_range
        self.fire_target_fraction = torch.empty(batch_dim, device=device).uniform_(target_low, target_high)
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
        # Compatibility note: drone_altitude is altitude above ground level
        # (AGL). Absolute MSL altitude is tracked separately.
        self.drone_altitude = torch.zeros(batch_dim, self.n_drones, device=device)
        self.drone_altitude_msl = torch.zeros(batch_dim, self.n_drones, device=device)
        self.drone_altitude_level = torch.zeros(batch_dim, self.n_drones, dtype=torch.long, device=device)
        self.step_drone_climb = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self._prev_ground_pos = torch.zeros(batch_dim, self.n_ground, 2, device=device)
        self._pre_step_ground_pos = torch.zeros_like(self._prev_ground_pos)
        self._pre_step_drone_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.step_ugv_travel_cost = torch.zeros(batch_dim, self.n_ground, device=device)
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
            self.smoke_grid.zero_()
            self.step_count.zero_()
            target_low, target_high = self.wildfire_area_fraction_range
            self.fire_target_fraction.uniform_(target_low, target_high)
            envs_to_seed = range(self.world.batch_dim)
        else:
            self.found_survivors[env_index] = False
            self.scouted_survivors[env_index] = False
            self.fire_grid[env_index] = False
            self.burned_grid[env_index] = False
            self.fire_age_grid[env_index] = 0
            self.fire_lifetime_grid[env_index] = 0
            self.smoke_grid[env_index] = 0.0
            self.step_count[env_index] = 0
            target_low, target_high = self.wildfire_area_fraction_range
            self.fire_target_fraction[env_index] = torch.empty(
                (), device=self.fire_grid.device,
            ).uniform_(target_low, target_high)
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
                self._update_drone_altitudes(drone_pos, drone_pos)
                self.step_drone_climb.zero_()
            else:
                self._pre_step_drone_pos[env_index] = drone_pos[env_index]
                one = torch.tensor([env_index], device=drone_pos.device)
                self._update_drone_altitudes(
                    drone_pos[env_index:env_index + 1], drone_pos[env_index:env_index + 1], one,
                )
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
        """Start a compact fire patch with resolution-independent physical area."""
        fuel = self._fire_fuel_grid(env_index)
        candidates = (fuel > 0.20).flatten().nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(height * width, device=self.fire_grid.device)
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
            ys = torch.arange(height, device=self.fire_grid.device).view(height, 1)
            xs = torch.arange(width, device=self.fire_grid.device).view(1, width)
            dist2 = (xs - seed_x).float().square() + (ys - seed_y).float().square()
            scores = torch.full(
                (height * width,),
                float("inf"),
                device=self.fire_grid.device,
            )
            scores[candidates] = dist2.flatten()[candidates]
            choice = torch.topk(scores, k=n_cells, largest=False).indices
            self._ignite_fire_cells(self._cell_choice_mask(env_index, choice, height, width))

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

    def _generate_terrain(self, env_index: int) -> None:
        """Load real terrain and clear small entity staging areas."""
        self._load_real_terrain(env_index)
        self._clear_entity_staging_areas(env_index)
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

    def _clear_entity_staging_areas(self, env_index: int) -> None:
        """Ensure survivor locations and ground starts have small organic clearings."""
        clear_radius = max(
            self._world_length_to_cells(max(self.agent_radius, self.survivor_radius) * 2.4),
            2,
        )
        size = self.fire_grid_size
        device = self.fire_grid.device
        yy = torch.arange(size, device=device).view(-1, 1)
        xx = torch.arange(size, device=device).view(1, -1)
        entities = self._survivors + self.world.agents[self.n_drones:]
        for entity in entities:
            gx, gy = self._positions_to_grid(entity.state.pos[env_index].view(1, 1, 2))
            x, y = int(gx.item()), int(gy.item())
            dist = torch.sqrt((xx - x).float().square() + (yy - y).float().square())
            mask = dist <= clear_radius
            self.land_cover_grid[env_index][mask] = LAND_OPEN
            self.slope_grid[env_index][mask] *= 0.35
            self.obstacle_type_grid[env_index][mask] = OBJECT_NONE
            self.obstacle_height_grid[env_index][mask] = 0.0

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
            (cover != LAND_ROCK) & (objects == OBJECT_NONE)
            & ((slope <= self.max_ground_slope) | road)
        )
        cost = self.land_cover_cost_values[cover] * (1.0 + self.slope_cost_weight * slope)
        speed = self.land_cover_speed_values[cover] / (1.0 + self.slope_speed_weight * slope)
        self.required_clearance_grid[env_index] = (
            self.obstacle_height_grid[env_index] + self.drone_safety_clearance
        )
        self.required_clearance_msl_grid[env_index] = (
            self.elevation_grid[env_index] + self.required_clearance_grid[env_index]
        )
        if self.required_clearance_grid[env_index].max() > self.drone_flight_levels.max():
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
        fire = self.fire_grid.float()
        neighbors = self._wind_weighted_neighbor_sum(fire)
        fuel = self._fire_fuel_grid()
        burnable = (fuel > 0.0) & ~self.burned_grid
        affected_fraction = self.burned_grid.float().flatten(1).mean(dim=1).view(-1, 1, 1)
        target_fraction = self.fire_target_fraction.view(-1, 1, 1).clamp_min(1e-6)
        target_gap = ((target_fraction - affected_fraction) / target_fraction).clamp(0.0, 1.0)
        random_rate = torch.exp(
            torch.randn(
                self.world.batch_dim, 1, 1,
                device=self.fire_grid.device,
            ) * self.fire_spread_variability
        ).clamp(0.35, 2.25)
        rate = neighbors * fuel * self._grid_scale() * random_rate * (0.15 + 2.35 * target_gap)
        p_ignite = 1.0 - (1.0 - self.fire_spread_prob) ** rate.clamp_min(0.0)

        smoke_spotting = self.smoke_grid > 0.08
        p_spot = (
            self.fire_spotting_prob
            * self._grid_scale()
            * random_rate
            * target_gap
            * fuel.clamp(0.0, 1.0)
        )
        new_burns = (
            ((torch.rand_like(p_ignite) < p_ignite) | ((torch.rand_like(p_spot) < p_spot) & smoke_spotting))
            & burnable
        )
        new_burns = self._cap_new_burns_to_target(new_burns)
        self._ignite_fire_cells(new_burns)

        self.fire_age_grid = torch.where(
            self.fire_grid,
            self.fire_age_grid + 1,
            self.fire_age_grid,
        )
        burned_out = self.fire_grid & (self.fire_age_grid >= self.fire_lifetime_grid.clamp_min(1))
        self.fire_grid = self.fire_grid & ~burned_out

    def _cap_new_burns_to_target(self, new_burns: Tensor) -> Tensor:
        """Keep the affected wildfire area near the sampled scenario target."""
        total_cells = self.fire_grid_size * self.fire_grid_size
        capped = new_burns.clone()
        for b in range(self.world.batch_dim):
            target_cells = int(round(float(self.fire_target_fraction[b]) * total_cells))
            remaining = target_cells - int(self.burned_grid[b].sum().item())
            if remaining <= 0:
                capped[b] = False
                continue
            choices = capped[b].flatten().nonzero(as_tuple=False).flatten()
            if choices.numel() > remaining:
                keep = choices[
                    torch.randperm(choices.numel(), device=choices.device)[:remaining]
                ]
                mask = torch.zeros_like(capped[b])
                mask.view(-1)[keep] = True
                capped[b] = mask
        return capped

    def _update_smoke(self) -> None:
        """Emit smoke from burning cells, then diffuse, drift, and decay."""
        smoke = self.smoke_grid * self.smoke_decay
        fuel = self._fire_fuel_grid()
        smoke = smoke + self.fire_grid.float() * self.smoke_emission * fuel.clamp(0.0, 1.0)
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
                r = r - self.drone_energy_costs[self.drone_altitude_level[:, i]]
                r = r + self.step_drone_climb[:, i] * self.r_drone_climb_cost
            else:
                g = i - self.n_drones
                r = r + confirm_per_ground[:, g] * self.r_ground_confirm
                r = r + ground_in_fire[:, g].float() * self.r_fire_penalty
                r = r + self.step_ugv_travel_cost[:, g] * self.r_ground_travel_cost
            agent.scenario_reward = r

    def _drone_survivor_detections(
        self,
        drone_dists: Tensor,
        drone_pos: Tensor,
        surv_pos: Tensor,
    ) -> Tensor:
        """Stochastic drone scouting from camera footprint and scene quality."""
        if self.n_drones == 0:
            return torch.zeros(
                self.world.batch_dim, 0, self.n_survivors,
                dtype=torch.bool, device=self.fire_grid.device,
            )

        footprint = self._drone_camera_ranges().unsqueeze(-1)
        visible = drone_dists <= footprint
        normalized_distance = (drone_dists / footprint.clamp_min(1e-6)).clamp(0.0, 1.0)
        distance_factor = 1.0 - (1.0 - self.drone_edge_detection_floor) * normalized_distance.square()

        gx, gy = self._positions_to_grid(surv_pos)
        b_idx = torch.arange(self.world.batch_dim, device=surv_pos.device).view(-1, 1).expand_as(gx)
        survivor_cover = self.land_cover_grid[b_idx, gy, gx]
        cover_factor = self.drone_cover_detection_factors[survivor_cover].unsqueeze(1)
        fire_smoke_factor = self._drone_fire_smoke_visibility_factor(drone_pos, surv_pos)
        altitude_quality = self.drone_detection_quality[self.drone_altitude_level].unsqueeze(-1)

        probability = (altitude_quality * distance_factor * cover_factor * fire_smoke_factor).clamp(0.0, 1.0)
        return visible & (torch.rand_like(probability) < probability)

    def _drone_fire_smoke_visibility_factor(self, drone_pos: Tensor, surv_pos: Tensor) -> Tensor:
        """Attenuate camera detections by smoke, flame glare, and heat shimmer."""
        path = self._sample_pair_paths(drone_pos, surv_pos, self.drone_perception_path_samples)
        smoke_path = self._grid_values_at_positions(self.smoke_grid, path)
        fire_path = self._grid_values_at_positions(self.fire_grid, path).float()

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
        """Select the lowest AGL flight level that clears crossed obstacles."""
        path = self._sample_path(start_pos, end_pos)
        required = self._grid_values_at_positions(
            self.required_clearance_grid, path, env_indices,
        ).amax(dim=-1)
        fits = self.drone_flight_levels.view(1, 1, -1) >= required.unsqueeze(-1)
        selected = fits.to(torch.int64).argmax(dim=-1)
        selected = torch.where(
            fits.any(dim=-1), selected, torch.full_like(selected, self.drone_flight_levels.numel() - 1),
        )
        end_ground_msl = self._grid_values_at_positions(self.elevation_grid, end_pos, env_indices)
        if env_indices is None:
            previous_msl = self.drone_altitude_msl.clone()
            self.drone_altitude_level = selected
            self.drone_altitude = self.drone_flight_levels[selected]
            self.drone_altitude_msl = end_ground_msl + self.drone_altitude
            self.step_drone_climb = (self.drone_altitude_msl - previous_msl).abs()
        else:
            previous_msl = self.drone_altitude_msl[env_indices].clone()
            self.drone_altitude_level[env_indices] = selected
            self.drone_altitude[env_indices] = self.drone_flight_levels[selected]
            self.drone_altitude_msl[env_indices] = end_ground_msl + self.drone_altitude[env_indices]
            self.step_drone_climb[env_indices] = (self.drone_altitude_msl[env_indices] - previous_msl).abs()

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
            lidar_obs = torch.full(
                (self.world.batch_dim, self.n_lidar_rays),
                self.drone_sensor_max_range,
                device=self.fire_grid.device,
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
        normalized_clearance = clearance / self.drone_flight_levels.max()
        return torch.cat([normalized_costs, blocked, normalized_clearance], dim=-1)

    def _flight_state(self, agent: Agent) -> Tensor:
        state = torch.zeros(self.world.batch_dim, 2, device=self.fire_grid.device)
        if agent.is_drone:
            drone_idx = self.world.agents.index(agent)
            state[:, 0] = self.drone_altitude[:, drone_idx] / self.drone_flight_levels.max()
            state[:, 1] = self.drone_detection_quality[self.drone_altitude_level[:, drone_idx]]
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
