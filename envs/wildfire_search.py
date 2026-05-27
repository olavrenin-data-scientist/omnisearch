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
        self.fire_spread_prob    = kwargs.pop("fire_spread_prob", 0.04)
        self.initial_fire_cells  = kwargs.pop("initial_fire_cells", 1)
        self.fire_step_interval  = kwargs.pop("fire_step_interval", 5)  # spread every N env steps
        self.smoke_emission = kwargs.pop("smoke_emission", 0.18)
        self.smoke_decay = kwargs.pop("smoke_decay", 0.96)
        self.smoke_diffusion = kwargs.pop("smoke_diffusion", 0.16)
        self.smoke_wind = kwargs.pop("smoke_wind", (1, 0))
        self.smoke_wind_strength = kwargs.pop("smoke_wind_strength", 0.06)

        # Ground terrain: procedural land cover layered on generated elevation.
        # It shares fire resolution so vegetation can later influence spread.
        self.terrain_brush_patches = kwargs.pop("terrain_brush_patches", 4)
        self.terrain_forest_patches = kwargs.pop("terrain_forest_patches", 3)
        self.terrain_rock_patches = kwargs.pop("terrain_rock_patches", 2)
        self.terrain_patch_radius = kwargs.pop("terrain_patch_radius", 3)
        self.terrain_hills = kwargs.pop("terrain_hills", 4)
        self.terrain_elevation_scale = kwargs.pop("terrain_elevation_scale", 0.30)
        self.terrain_road_width = kwargs.pop("terrain_road_width", 1)
        self.max_ground_slope = kwargs.pop("max_ground_slope", 0.70)
        self.slope_cost_weight = kwargs.pop("slope_cost_weight", 2.0)
        self.slope_speed_weight = kwargs.pop("slope_speed_weight", 1.5)
        self.terrain_path_samples = kwargs.pop("terrain_path_samples", 6)
        land_cover_costs = kwargs.pop("land_cover_costs", (0.65, 1.0, 1.5, 2.2, 4.0))
        land_cover_speeds = kwargs.pop("land_cover_speeds", (1.0, 0.9, 0.65, 0.45, 0.0))
        if len(land_cover_costs) != 5 or len(land_cover_speeds) != 5:
            raise ValueError("land-cover cost and speed values must cover road/open/brush/forest/rock")
        self.terrain_houses = kwargs.pop("terrain_houses", 5)
        self.tree_height_range = kwargs.pop("tree_height_range", (0.12, 0.24))
        self.house_height_range = kwargs.pop("house_height_range", (0.08, 0.15))

        # 2.5D drone flight: horizontal VMAS motion plus an automatic safe
        # flight level. Higher flight clears structures but worsens sensing.
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

        # Drones: fast, wide lidar. VMAS requires collide=True for lidar ray
        # casting; in concept these fly above ground (2D-abstracted here).
        for i in range(self.n_drones):
            agent = Agent(
                name=f"drone_{i}",
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                max_speed=0.5,
                u_range=1.0,
                u_multiplier=0.6,
                color=Color.BLUE,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.drone_sensor_max_range,
                        entity_filter=survivor_filter,
                        render_color=Color.RED,
                    ),
                ],
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

        # Survivor landmarks. collide=True so lidar can hit them; movable=False
        # so they don't drift when bumped.
        self._survivors: List[Landmark] = []
        for i in range(self.n_survivors):
            survivor = Landmark(
                name=f"survivor_{i}",
                collide=True,
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
        self.obstacle_type_grid = torch.zeros_like(self.land_cover_grid)
        self.obstacle_height_grid = torch.zeros_like(self.elevation_grid)
        self.required_clearance_grid = torch.zeros_like(self.elevation_grid)
        self.traversable_grid = torch.ones_like(self.fire_grid)
        self.mobility_cost_grid = torch.ones_like(self.elevation_grid)
        self.speed_multiplier_grid = torch.ones_like(self.elevation_grid)
        self.land_cover_cost_values = torch.tensor(land_cover_costs, dtype=torch.float, device=device)
        self.land_cover_speed_values = torch.tensor(land_cover_speeds, dtype=torch.float, device=device)
        self.drone_flight_levels = torch.tensor(drone_flight_levels, dtype=torch.float, device=device)
        self.drone_detection_quality = torch.tensor(drone_detection_quality, dtype=torch.float, device=device)
        self.drone_cover_detection_factors = torch.tensor(
            drone_cover_detection_factors, dtype=torch.float, device=device,
        )
        self.drone_energy_costs = torch.tensor(drone_energy_costs, dtype=torch.float, device=device)
        self.drone_altitude = torch.zeros(batch_dim, self.n_drones, device=device)
        self.drone_altitude_level = torch.zeros(batch_dim, self.n_drones, dtype=torch.long, device=device)
        self.step_drone_climb = torch.zeros(batch_dim, self.n_drones, device=device)
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self._prev_ground_pos = torch.zeros(batch_dim, self.n_ground, 2, device=device)
        self._pre_step_ground_pos = torch.zeros_like(self._prev_ground_pos)
        self._pre_step_drone_pos = torch.zeros(batch_dim, self.n_drones, 2, device=device)
        self.step_ugv_travel_cost = torch.zeros(batch_dim, self.n_ground, device=device)

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
            self.smoke_grid.zero_()
            self.step_count.zero_()
            envs_to_seed = range(self.world.batch_dim)
        else:
            self.found_survivors[env_index] = False
            self.scouted_survivors[env_index] = False
            self.fire_grid[env_index] = False
            self.smoke_grid[env_index] = 0.0
            self.step_count[env_index] = 0
            envs_to_seed = [env_index]

        # Seed initial fire cells (random per-batch)
        H = W = self.fire_grid_size
        for b in envs_to_seed:
            idx = torch.randperm(H * W, device=self.fire_grid.device)[: self.initial_fire_cells]
            self.fire_grid[b].view(-1)[idx] = True
            self._generate_terrain(b)

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

    def _generate_terrain(self, env_index: int) -> None:
        """Create hills, vegetation regions, rocky barriers, and road corridors."""
        grid = self.land_cover_grid[env_index]
        grid.fill_(LAND_OPEN)
        size = self.fire_grid_size
        radius = max(int(self.terrain_patch_radius), 1)

        for land_type, n_patches in (
            (LAND_BRUSH, self.terrain_brush_patches),
            (LAND_FOREST, self.terrain_forest_patches),
            (LAND_ROCK, self.terrain_rock_patches),
        ):
            for _ in range(n_patches):
                cx = int(torch.randint(size, (1,), device=grid.device).item())
                cy = int(torch.randint(size, (1,), device=grid.device).item())
                rx = int(torch.randint(1, radius + 1, (1,), device=grid.device).item())
                ry = int(torch.randint(1, radius + 1, (1,), device=grid.device).item())
                ys = torch.arange(size, device=grid.device).view(-1, 1)
                xs = torch.arange(size, device=grid.device).view(1, -1)
                patch = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
                grid[patch] = land_type

        self._generate_elevation(env_index)
        self._paint_roads(env_index)
        self._generate_obstacles(env_index)
        self._clear_entity_staging_areas(env_index)
        self._refresh_mobility_layers(env_index)

    def _generate_elevation(self, env_index: int) -> None:
        """Blend smooth Gaussian hills and derive a slope raster."""
        size = self.fire_grid_size
        device = self.fire_grid.device
        axis = torch.linspace(-1.0, 1.0, size, device=device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        elevation = torch.zeros(size, size, device=device)
        for _ in range(self.terrain_hills):
            cx = torch.rand((), device=device) * 1.6 - 0.8
            cy = torch.rand((), device=device) * 1.6 - 0.8
            sx = torch.rand((), device=device) * 0.25 + 0.20
            sy = torch.rand((), device=device) * 0.25 + 0.20
            height = torch.rand((), device=device) * 0.55 + 0.45
            elevation += height * torch.exp(
                -(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0,
            )
        elevation -= elevation.min()
        elevation /= elevation.max().clamp_min(1e-6)
        elevation *= self.terrain_elevation_scale
        cell_width = 2.0 / max(size - 1, 1)
        grade_y, grade_x = torch.gradient(elevation, spacing=(cell_width, cell_width))
        self.elevation_grid[env_index] = elevation
        self.slope_grid[env_index] = torch.sqrt(grade_x.square() + grade_y.square())

    def _paint_roads(self, env_index: int) -> None:
        """Lay two traversable winding access tracks across the landscape."""
        grid = self.land_cover_grid[env_index]
        size = self.fire_grid_size
        width = max(int(self.terrain_road_width), 0)
        phase = float(torch.rand((), device=grid.device).item()) * 6.283
        x_mid = int(torch.randint(size // 3, max(2 * size // 3, size // 3 + 1),
                                  (1,), device=grid.device).item())
        y_mid = int(torch.randint(size // 3, max(2 * size // 3, size // 3 + 1),
                                  (1,), device=grid.device).item())
        for i in range(size):
            offset = int(round(1.5 * math.sin(phase + i * 0.35)))
            x = max(0, min(size - 1, x_mid + offset))
            y = max(0, min(size - 1, y_mid + offset))
            grid[i, max(0, x - width): min(size, x + width + 1)] = LAND_ROAD
            grid[max(0, y - width): min(size, y + width + 1), i] = LAND_ROAD

    def _generate_obstacles(self, env_index: int) -> None:
        """Place tree canopy on forest cells and compact house footprints."""
        cover = self.land_cover_grid[env_index]
        objects = self.obstacle_type_grid[env_index]
        heights = self.obstacle_height_grid[env_index]
        objects.zero_()
        heights.zero_()
        forest = cover == LAND_FOREST
        objects[forest] = OBJECT_TREE
        tree_low, tree_high = self.tree_height_range
        heights[forest] = tree_low + torch.rand_like(heights[forest]) * (tree_high - tree_low)

        size = self.fire_grid_size
        house_low, house_high = self.house_height_range
        for _ in range(self.terrain_houses):
            x = int(torch.randint(1, max(size - 1, 2), (1,), device=cover.device).item())
            y = int(torch.randint(1, max(size - 1, 2), (1,), device=cover.device).item())
            footprint_cover = cover[y:y + 2, x:x + 2]
            buildable = (
                (footprint_cover == LAND_OPEN)
                | (footprint_cover == LAND_BRUSH)
                | (footprint_cover == LAND_FOREST)
            )
            if not buildable.any():
                continue
            footprint_cover[buildable] = LAND_OPEN
            objects[y:y + 2, x:x + 2][buildable] = OBJECT_HOUSE
            house_height = float(
                (house_low + torch.rand((), device=cover.device) * (house_high - house_low)).item(),
            )
            heights[y:y + 2, x:x + 2][buildable] = house_height

    def _clear_entity_staging_areas(self, env_index: int) -> None:
        """Ensure survivor locations and ground starts are accessible clearings."""
        entities = self._survivors + self.world.agents[self.n_drones:]
        for entity in entities:
            gx, gy = self._positions_to_grid(entity.state.pos[env_index].view(1, 1, 2))
            x, y = int(gx.item()), int(gy.item())
            ys = slice(max(y - 1, 0), min(y + 2, self.fire_grid_size))
            xs = slice(max(x - 1, 0), min(x + 2, self.fire_grid_size))
            self.land_cover_grid[env_index, ys, xs] = LAND_OPEN
            self.slope_grid[env_index, ys, xs] = 0.0
            self.obstacle_type_grid[env_index, ys, xs] = OBJECT_NONE
            self.obstacle_height_grid[env_index, ys, xs] = 0.0

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
            self.elevation_grid[env_index] + self.obstacle_height_grid[env_index]
            + self.drone_safety_clearance
        )
        if self.required_clearance_grid[env_index].max() > self.drone_flight_levels.max():
            raise ValueError("highest drone_flight_levels entry must clear generated terrain and obstacles")
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
        neighbors = self._neighbor_sum(fire)
        p_ignite = 1.0 - (1.0 - self.fire_spread_prob) ** neighbors
        new_burns = torch.rand_like(p_ignite) < p_ignite
        self.fire_grid = self.fire_grid | new_burns

    def _update_smoke(self) -> None:
        """Emit smoke from burning cells, then diffuse, drift, and decay."""
        smoke = self.smoke_grid * self.smoke_decay
        smoke = smoke + self.fire_grid.float() * self.smoke_emission
        neighbor_mean = self._neighbor_sum(smoke) / 4.0
        smoke = smoke + self.smoke_diffusion * (neighbor_mean - smoke)

        wind_x, wind_y = self.smoke_wind
        if self.smoke_wind_strength > 0 and (wind_x != 0 or wind_y != 0):
            shifted = self._shift_grid_no_wrap(smoke, int(wind_x), int(wind_y))
            smoke = smoke + self.smoke_wind_strength * (shifted - smoke)

        self.smoke_grid = smoke.clamp(0.0, 1.0)

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
                corrected_pos = torch.where(
                    blocked.unsqueeze(-1), self._pre_step_ground_pos[:, i], agent.state.pos,
                )
                corrected_vel = torch.where(
                    blocked.unsqueeze(-1), torch.zeros_like(agent.state.vel), agent.state.vel,
                )
                agent.set_pos(corrected_pos, batch_index=None)
                agent.set_vel(corrected_vel, batch_index=None)

        drone_agents = self.world.agents[:self.n_drones]
        if drone_agents:
            drone_pos = torch.stack([a.state.pos for a in drone_agents], dim=1)
            self._update_drone_altitudes(self._pre_step_drone_pos, drone_pos)

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

        drone_dists = dists[:, :self.n_drones, :]
        drone_seen = self._drone_survivor_detections(drone_dists, surv_pos)
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

    def _drone_survivor_detections(self, drone_dists: Tensor, surv_pos: Tensor) -> Tensor:
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
        smoke = self.smoke_grid[b_idx, gy, gx].unsqueeze(1)
        smoke_factor = 1.0 - smoke * (1.0 - self.drone_smoke_detection_factor)
        altitude_quality = self.drone_detection_quality[self.drone_altitude_level].unsqueeze(-1)

        probability = (altitude_quality * distance_factor * cover_factor * smoke_factor).clamp(0.0, 1.0)
        return visible & (torch.rand_like(probability) < probability)

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

    def _path_is_traversable(self, start_pos: Tensor, end_pos: Tensor) -> Tensor:
        path = self._sample_path(start_pos, end_pos)
        return self._grid_values_at_positions(self.traversable_grid, path).all(dim=-1)

    def _update_drone_altitudes(
        self,
        start_pos: Tensor,
        end_pos: Tensor,
        env_indices: Tensor | None = None,
    ) -> None:
        """Select the lowest flight level that clears each crossed cell."""
        path = self._sample_path(start_pos, end_pos)
        required = self._grid_values_at_positions(
            self.required_clearance_grid, path, env_indices,
        ).amax(dim=-1)
        fits = self.drone_flight_levels.view(1, 1, -1) >= required.unsqueeze(-1)
        selected = fits.to(torch.int64).argmax(dim=-1)
        selected = torch.where(
            fits.any(dim=-1), selected, torch.full_like(selected, self.drone_flight_levels.numel() - 1),
        )
        if env_indices is None:
            previous = self.drone_altitude.clone()
            self.drone_altitude_level = selected
            self.drone_altitude = self.drone_flight_levels[selected]
            self.step_drone_climb = (self.drone_altitude - previous).abs()
        else:
            previous = self.drone_altitude[env_indices].clone()
            self.drone_altitude_level[env_indices] = selected
            self.drone_altitude[env_indices] = self.drone_flight_levels[selected]
            self.step_drone_climb[env_indices] = (self.drone_altitude[env_indices] - previous).abs()

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
        lidar_obs  = agent.sensors[0].measure()     # [B, n_rays]
        if agent.is_drone:
            drone_idx = self.world.agents.index(agent)
            effective_range = self._drone_camera_ranges()[:, drone_idx]
            lidar_obs = torch.where(
                lidar_obs <= effective_range.unsqueeze(-1),
                lidar_obs,
                torch.full_like(lidar_obs, self.drone_sensor_max_range),
            )
        fire_local = self._local_fire_density(agent)        # [B, 1]
        terrain_local = self._local_terrain_features(agent)  # [B, 27]
        flight_state = self._flight_state(agent)             # [B, 2]
        neighbor   = self._neighbor_observations(agent)     # [B, (A-1)*2]
        return torch.cat(
            [own_pos, own_vel, lidar_obs, fire_local, terrain_local, flight_state, neighbor], dim=-1,
        )

    def _local_fire_density(self, agent: Agent) -> Tensor:
        pos = agent.state.pos
        gx = ((pos[..., X] + self.x_semidim) / (2 * self.x_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2
        ).long()
        gy = ((pos[..., Y] + self.y_semidim) / (2 * self.y_semidim) * self.fire_grid_size).clamp(
            1, self.fire_grid_size - 2
        ).long()
        b_idx = torch.arange(self.world.batch_dim, device=pos.device)
        density = torch.zeros(self.world.batch_dim, device=pos.device)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                density = density + self.fire_grid[b_idx, gy + dy, gx + dx].float()
        return (density / 9.0).unsqueeze(-1)

    def _local_terrain_features(self, agent: Agent) -> Tensor:
        """Expose mobility cost, blocked masks, and air-clearance requirements."""
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
        return {
            "n_found":   self.found_survivors.sum(dim=1).float(),
            "n_scouted": self.scouted_survivors.sum(dim=1).float(),
            "n_burning": self.fire_grid.flatten(1).sum(dim=1).float(),
            "ugv_step_travel_cost": self.step_ugv_travel_cost.sum(dim=1),
            "mean_drone_altitude": mean_drone_altitude,
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
