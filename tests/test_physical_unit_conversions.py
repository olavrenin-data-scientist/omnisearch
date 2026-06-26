import math
import sys
import types
import unittest

import torch


def _install_vmas_stubs() -> None:
    try:
        import vmas as installed_vmas

        if hasattr(installed_vmas, "make_env"):
            return
    except ImportError:
        pass

    vmas = types.ModuleType("vmas")
    simulator = types.ModuleType("vmas.simulator")
    core = types.ModuleType("vmas.simulator.core")
    scenario = types.ModuleType("vmas.simulator.scenario")
    sensors = types.ModuleType("vmas.simulator.sensors")
    utils = types.ModuleType("vmas.simulator.utils")

    class Sphere:
        def __init__(self, radius=0.05):
            self._radius = radius

        @property
        def radius(self):
            return self._radius

    class ScenarioUtils:
        pass

    class BaseScenario:
        @property
        def world(self):
            return self._world

    class Color:
        BLUE = "blue"
        GREEN = "green"
        RED = "red"

    core.Agent = type("Agent", (), {})
    core.Landmark = type("Landmark", (), {})
    core.Sphere = Sphere
    core.World = type("World", (), {})
    scenario.BaseScenario = BaseScenario
    sensors.Lidar = type("Lidar", (), {})
    utils.Color = Color
    utils.ScenarioUtils = ScenarioUtils

    sys.modules.setdefault("vmas", vmas)
    sys.modules.setdefault("vmas.simulator", simulator)
    sys.modules.setdefault("vmas.simulator.core", core)
    sys.modules.setdefault("vmas.simulator.scenario", scenario)
    sys.modules.setdefault("vmas.simulator.sensors", sensors)
    sys.modules.setdefault("vmas.simulator.utils", utils)


_install_vmas_stubs()

from envs.wildfire_search import WildfireSearchScenario  # noqa: E402


class _Entity:
    def __init__(self, radius):
        self.shape = sys.modules["vmas.simulator.core"].Sphere(radius)


class _Agent:
    def __init__(self, name, is_drone, pos, vel):
        self.name = name
        self.is_drone = is_drone
        self.state = types.SimpleNamespace(
            pos=torch.tensor(pos, dtype=torch.float32),
            vel=torch.tensor(vel, dtype=torch.float32),
        )

    def set_pos(self, pos, batch_index=None):
        self.state.pos = pos

    def set_vel(self, vel, batch_index=None):
        self.state.vel = vel


class PhysicalUnitConversionTests(unittest.TestCase):
    def _scenario(self):
        scenario = WildfireSearchScenario()
        scenario.agent_radius_sim_override = None
        scenario.survivor_radius_sim_override = None
        scenario.detection_range_sim_override = None
        scenario.agent_radius_m = 0.50
        scenario.survivor_radius_m = 0.35
        scenario.ground_confirmation_range_m = 10.0
        scenario.drone_min_footprint_sim_override = None
        scenario.drone_min_footprint_m = 75.0
        scenario.ground_confirm_min_sim_override = None
        scenario.ground_confirm_min_m = 20.0
        scenario.ground_lidar_range_sim_override = None
        scenario.ground_lidar_range_m = 20.0
        scenario.spawn_padding_m = 1.0
        scenario.ground_min_step_sim_override = None
        scenario.ground_min_step_m = 0.0
        scenario.agent_radius_by_env = torch.zeros(1)
        scenario.survivor_radius_by_env = torch.zeros(1)
        scenario.detection_range_by_env = torch.zeros(1)
        scenario.drone_min_footprint_by_env = torch.zeros(1)
        scenario.spawn_padding_by_env = torch.zeros(1)
        ground = _Entity(0.04)
        ground.sensors = [types.SimpleNamespace(_max_range=0.20)]
        scenario.n_drones = 0
        scenario._world = types.SimpleNamespace(agents=[ground])
        scenario._survivors = [_Entity(0.03)]
        return scenario

    def test_dimensions_are_converted_from_meters(self):
        scenario = self._scenario()
        scale = 2.0 / 13_794.918831077

        scenario._refresh_physical_size_conversions(0, scale)

        self.assertAlmostEqual(scenario.agent_radius, 0.50 * scale)
        self.assertAlmostEqual(scenario.survivor_radius, 0.35 * scale)
        self.assertAlmostEqual(scenario.detection_range, 20.0 * scale)
        self.assertAlmostEqual(float(scenario.drone_min_footprint_by_env[0]), 75.0 * scale)
        self.assertAlmostEqual(scenario.world.agents[0].shape.radius, 0.50 * scale)
        self.assertAlmostEqual(scenario._survivors[0].shape.radius, 0.35 * scale)

    def test_drone_footprint_has_no_simulation_unit_floor(self):
        scenario = self._scenario()
        scale = 2.0 / 13_794.918831077
        scenario.drone_altitude = torch.tensor([[50.0 * scale]])
        scenario.drone_camera_half_angle_tan = math.tan(math.radians(90.0) / 2.0)

        footprint = scenario._drone_camera_ranges()

        self.assertAlmostEqual(float(footprint.item()), 50.0 * scale, places=7)
        self.assertLess(float(footprint.item()), 0.12)

    def test_legacy_floor_sim_overrides_take_precedence(self):
        scenario = self._scenario()
        scenario.drone_min_footprint_sim_override = 0.15
        scenario.ground_confirm_min_sim_override = 0.20

        scenario._refresh_physical_size_conversions(0, 2.0 / 1_000.0)

        self.assertAlmostEqual(float(scenario.drone_min_footprint_by_env[0]), 0.15)
        self.assertAlmostEqual(float(scenario.detection_range_by_env[0]), 0.20)

    def test_terrain_scale_honors_non_default_equal_semidims(self):
        scenario = self._scenario()
        scenario.world_scale = 2.0
        metadata = {"units": {"sim_units_per_meter": 2.0 / 13_794.918831077}}

        scale = scenario._terrain_sim_units_per_meter(metadata)

        self.assertAlmostEqual(scale, 4.0 / 13_794.918831077)

    def test_ground_sensor_distances_are_converted_from_meters(self):
        scenario = self._scenario()
        scale = 2.0 / 13_794.918831077

        scenario._refresh_ground_sensor_conversions(0, scale)

        self.assertAlmostEqual(scenario.ground_lidar_range, 20.0 * scale)
        self.assertAlmostEqual(float(scenario.spawn_padding_by_env[0]), 1.0 * scale)
        self.assertEqual(scenario.ground_min_step_sim, 0.0)
        self.assertAlmostEqual(scenario.world.agents[0].sensors[0]._max_range, 20.0 * scale)

    def _coverage_scenario(self, n_drones=1, grid_size=32):
        scenario = self._scenario()
        scenario._world = types.SimpleNamespace(batch_dim=1)
        scenario.n_drones = n_drones
        scenario.r_coverage = 1.0
        scenario.uav_coverage_normalization = "map"
        scenario.r_uav_move_coverage = 0.001
        scenario.uav_move_coverage_normalization = "raw"
        scenario.r_uav_move_coverage_cap = 0.1
        scenario.uav_coverage_opportunity_cap = 1.0
        scenario.r_uav_frontier_alignment = 0.0
        scenario.uav_frontier_obs = False
        scenario.uav_frontier_obs_radius_m = 10.0
        scenario.uav_frontier_mode = "centroid"
        scenario.uav_frontier_source = "coverage"
        scenario.uav_frontier_sectors = 8
        scenario.uav_frontier_top_k = 2
        scenario.uav_frontier_ownership = False
        scenario.r_uav_confidence = 0.0
        scenario.r_uav_confidence_move = 0.0
        scenario.uav_confidence_gamma = 2.0
        scenario.uav_confidence_eps = 0.05
        scenario.uav_confidence_opportunity_eps = 1e-6
        scenario.uav_confidence_diagnostics = False
        scenario.uav_confidence_obs_grid = 0
        scenario.local_confidence_obs_grid = 0
        scenario.local_confidence_obs_radius_m = 150.0
        scenario.r_uav_overlap = 0.0
        scenario.uav_overlap_allowed = 0.10
        scenario.uav_overlap_penalty_normalization = "raw"
        scenario.r_uav_inter_uav_overlap = 0.0
        scenario.uav_inter_uav_overlap_allowed = 0.20
        scenario.r_uav_outside_footprint = 0.0
        scenario.uav_boundary_soft_margin_m = 25.0
        scenario.sim_step_seconds = 2.0
        scenario.drone_speed_mps = 10.0
        scenario.uav_boundary_escape_m = 0.0
        scenario.uav_boundary_escape_raw_threshold = 0.2
        scenario.uav_boundary_escape_projected_threshold = 0.05
        scenario.fire_grid_size = grid_size
        scenario.x_semidim = 1.0
        scenario.y_semidim = 1.0
        scenario.coverage_grid = torch.zeros(1, grid_size, grid_size, dtype=torch.bool)
        scenario.uav_confidence_grid = torch.zeros(1, grid_size, grid_size)
        scenario.land_cover_grid = torch.zeros(1, grid_size, grid_size, dtype=torch.long)
        scenario.drone_cover_detection_factors = torch.ones(8)
        scenario.disable_fire = True
        scenario.drone_edge_detection_floor = 0.4
        scenario.drone_altitude = torch.full((1, n_drones), 0.10)
        scenario.drone_altitude_quality = torch.ones(1, n_drones)
        scenario.drone_camera_half_angle_tan = 1.0
        scenario.drone_min_footprint_by_env = torch.zeros(1)
        scenario.terrain_sim_units_per_meter = torch.tensor([0.1])
        scenario._pre_step_drone_pos = torch.zeros(1, n_drones, 2)
        return scenario

    def test_coverage_uses_camera_footprint(self):
        scenario = self._coverage_scenario()
        positions = torch.zeros(1, 1, 2)

        scenario.drone_altitude = torch.tensor([[0.05]])
        small_credit, *_ = scenario._coverage_reward(positions)
        small_footprint = float(small_credit.sum())

        scenario.coverage_grid.zero_()
        scenario.drone_altitude = torch.tensor([[0.20]])
        large_credit, *_ = scenario._coverage_reward(positions)
        large_footprint = float(large_credit.sum())

        self.assertGreater(large_footprint, small_footprint)

    def test_coverage_overlap_is_split_without_duplicate_credit(self):
        scenario = self._coverage_scenario(n_drones=2)
        scenario.drone_altitude = torch.tensor([[0.10, 0.10]])
        positions = torch.zeros(1, 2, 2)

        credit, overlap, outside, inter_uav, *_ = scenario._coverage_reward(positions)

        self.assertAlmostEqual(float(credit[0, 0]), float(credit[0, 1]), places=7)
        self.assertEqual(float(overlap.sum()), 0.0)
        self.assertEqual(float(outside.sum()), 0.0)
        self.assertEqual(float(inter_uav[0, 0]), 1.0)
        self.assertEqual(float(inter_uav[0, 1]), 1.0)
        self.assertAlmostEqual(
            float(credit.sum()),
            float(scenario.coverage_grid.float().mean()),
            places=7,
        )

    def test_revisiting_covered_ground_earns_no_reward(self):
        scenario = self._coverage_scenario()
        scenario.drone_altitude = torch.tensor([[0.10]])
        positions = torch.zeros(1, 1, 2)

        first_credit, first_overlap, first_outside, first_inter_uav, *_ = scenario._coverage_reward(positions)
        revisit_credit, revisit_overlap, revisit_outside, revisit_inter_uav, *_ = scenario._coverage_reward(positions)

        self.assertGreater(float(first_credit.sum()), 0.0)
        self.assertEqual(float(first_overlap.sum()), 0.0)
        self.assertEqual(float(first_outside.sum()), 0.0)
        self.assertEqual(float(first_inter_uav.sum()), 0.0)
        self.assertEqual(float(revisit_credit.sum()), 0.0)
        self.assertEqual(float(revisit_overlap[0, 0]), 1.0)
        self.assertEqual(float(revisit_outside.sum()), 0.0)
        self.assertEqual(float(revisit_inter_uav.sum()), 0.0)

    def test_total_episode_coverage_credit_is_bounded(self):
        scenario = self._coverage_scenario(grid_size=16)
        scenario.drone_altitude = torch.tensor([[0.50]])

        total = 0.0
        for x in (-0.75, -0.25, 0.25, 0.75):
            for y in (-0.75, -0.25, 0.25, 0.75):
                credit, *_ = scenario._coverage_reward(torch.tensor([[[x, y]]]))
                total += float(credit.sum())

        self.assertLessEqual(total, 1.0 + 1e-7)

    def test_uav_expected_overlap_fraction_uses_circle_geometry(self):
        scenario = self._coverage_scenario()
        scenario.drone_altitude = torch.tensor([[2.5]])  # 25m footprint radius at 0.1 sim-units/m.

        expected = scenario._uav_expected_overlap_fraction(torch.tensor([[16.0]]))

        self.assertAlmostEqual(float(expected[0, 0]), 0.600, places=3)

    def test_uav_overlap_penalty_uses_expected_overlap_and_allowed_slack(self):
        scenario = self._coverage_scenario()
        scenario.r_uav_overlap = 0.05
        scenario.uav_overlap_allowed = 0.10
        overlap = torch.tensor([[0.50, 0.70, 0.80, 1.00]])
        expected = torch.tensor([[0.60, 0.60, 0.60, 0.60]])
        scenario.n_drones = overlap.shape[1]

        penalty = scenario._uav_overlap_penalty(overlap, expected)

        self.assertAlmostEqual(float(penalty[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(penalty[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(penalty[0, 2]), -0.05 / 3.0, places=6)
        self.assertAlmostEqual(float(penalty[0, 3]), -0.05, places=6)

    def test_uav_overlap_penalty_can_use_opportunity_normalization(self):
        scenario = self._coverage_scenario()
        scenario.r_uav_overlap = 0.05
        scenario.uav_overlap_allowed = 0.10
        scenario.uav_overlap_penalty_normalization = "opportunity"
        overlap = torch.tensor([[1.00]])
        expected = torch.tensor([[0.60]])
        opportunity_available = torch.tensor([[0.25]])

        penalty = scenario._uav_overlap_penalty(
            overlap,
            expected,
            opportunity_available,
        )

        self.assertAlmostEqual(float(penalty[0, 0]), -0.05 * 0.25, places=6)

    def test_uav_inter_uav_overlap_penalty_uses_same_step_overlap_slack(self):
        scenario = self._coverage_scenario(n_drones=2)
        scenario.r_uav_inter_uav_overlap = 0.03
        scenario.uav_inter_uav_overlap_allowed = 0.20
        scenario.drone_altitude = torch.tensor([[0.10, 0.10]])

        _, _, _, inter_uav, *_ = scenario._coverage_reward(torch.zeros(1, 2, 2))
        penalty = scenario._uav_inter_uav_overlap_penalty(inter_uav)

        self.assertEqual(float(inter_uav[0, 0]), 1.0)
        self.assertEqual(float(inter_uav[0, 1]), 1.0)
        self.assertAlmostEqual(float(penalty[0, 0]), -0.03, places=6)
        self.assertAlmostEqual(float(penalty[0, 1]), -0.03, places=6)

    def test_uav_frontier_features_point_toward_uncovered_mass(self):
        scenario = self._coverage_scenario(grid_size=8)
        scenario.uav_frontier_obs = True
        scenario.coverage_grid[:] = True
        scenario.coverage_grid[:, :, 4:] = False

        features = scenario._uav_frontier_features_for_positions(torch.zeros(1, 1, 2))

        self.assertGreater(float(features[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(features[0, 0, 1]), 0.0, places=6)
        self.assertGreater(float(features[0, 0, 2]), 0.0)
        self.assertGreater(float(features[0, 0, 3]), 0.0)

    def test_uav_frontier_alignment_rewards_clamped_progress_toward_uncovered_mass(self):
        scenario = self._coverage_scenario(grid_size=8)
        scenario.uav_frontier_obs = True
        scenario.r_uav_frontier_alignment = 0.2
        scenario.coverage_grid[:] = True
        scenario.coverage_grid[:, :, 4:] = False
        scenario._pre_step_drone_pos = torch.zeros(1, 1, 2)

        toward = torch.tensor([[[0.1, 0.0]]])
        full_step = torch.tensor([[[2.0, 0.0]]])
        overshoot = torch.tensor([[[4.0, 0.0]]])
        away = torch.tensor([[[-0.1, 0.0]]])

        toward_reward, toward_alignment, toward_progress, ratio = scenario._uav_frontier_alignment_reward(toward)
        full_reward, _, full_progress, full_ratio = scenario._uav_frontier_alignment_reward(full_step)
        overshoot_reward, _, overshoot_progress, _ = scenario._uav_frontier_alignment_reward(overshoot)
        away_reward, away_alignment, away_progress, _ = scenario._uav_frontier_alignment_reward(away)

        self.assertGreater(float(ratio[0, 0]), 0.0)
        self.assertGreater(float(toward_alignment[0, 0]), 0.9)
        self.assertLess(float(away_alignment[0, 0]), -0.9)
        self.assertGreater(float(toward_progress[0, 0]), 0.0)
        self.assertLess(float(toward_progress[0, 0]), 1.0)
        self.assertEqual(float(full_progress[0, 0]), 1.0)
        self.assertEqual(float(overshoot_progress[0, 0]), 1.0)
        self.assertEqual(float(away_progress[0, 0]), 0.0)
        self.assertGreater(float(toward_reward[0, 0]), 0.0)
        self.assertLess(float(toward_reward[0, 0]), float(full_reward[0, 0]))
        self.assertAlmostEqual(float(full_reward[0, 0]), 0.2 * float(full_ratio[0, 0]), places=6)
        self.assertAlmostEqual(float(overshoot_reward[0, 0]), float(full_reward[0, 0]), places=6)
        self.assertEqual(float(away_reward[0, 0]), 0.0)

    def test_uav_outside_footprint_penalty_scales_with_footprint_outside_map(self):
        scenario = self._coverage_scenario(grid_size=64)
        scenario.r_uav_outside_footprint = 0.1
        scenario.drone_altitude = torch.tensor([[0.20]])

        _, _, center_outside, *_ = scenario._coverage_reward(torch.tensor([[[0.0, 0.0]]]))
        center_penalty = scenario._uav_outside_footprint_penalty(center_outside)

        scenario.coverage_grid.zero_()
        _, _, corner_outside, *_ = scenario._coverage_reward(torch.tensor([[[0.95, 0.95]]]))
        corner_penalty = scenario._uav_outside_footprint_penalty(corner_outside)

        self.assertAlmostEqual(float(center_outside[0, 0]), 0.0, places=6)
        self.assertEqual(float(center_penalty[0, 0]), 0.0)
        self.assertGreater(float(corner_outside[0, 0]), 0.0)
        self.assertLess(float(corner_penalty[0, 0]), 0.0)
        self.assertGreaterEqual(float(corner_penalty[0, 0]), -0.1)

    def test_local_coverage_observation_pools_physical_window_and_marks_outside(self):
        scenario = self._coverage_scenario(grid_size=8)
        scenario.local_coverage_obs_grid = 3
        scenario.local_coverage_obs_radius_m = 100.0
        scenario.terrain_sim_units_per_meter = torch.tensor([0.01])
        scenario.coverage_grid.zero_()
        scenario.coverage_grid[:, 3:5, 3:5] = True
        agent = types.SimpleNamespace(
            state=types.SimpleNamespace(pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
        )

        center_patch = scenario._local_coverage_observation(agent)

        self.assertEqual(center_patch.shape[-1], 9)
        self.assertGreater(float(center_patch[0, 4]), 0.0)
        self.assertLess(float(center_patch[0, 4]), 1.0)
        self.assertLess(float(center_patch[0, 0]), 1.0)

        agent.state.pos = torch.tensor([[0.95, 0.95]], dtype=torch.float32)
        edge_patch = scenario._local_coverage_observation(agent)

        self.assertGreater(float(edge_patch[0, -1]), 0.5)

    def test_confidence_observations_pool_global_and_local_maps(self):
        scenario = self._coverage_scenario(grid_size=8)
        scenario.uav_confidence_obs_grid = 2
        scenario.local_confidence_obs_grid = 3
        scenario.local_confidence_obs_radius_m = 100.0
        scenario.terrain_sim_units_per_meter = torch.tensor([0.01])
        scenario.uav_confidence_grid = torch.zeros(1, 8, 8)
        scenario.uav_confidence_grid[:, 3:5, 3:5] = 0.8
        agent = types.SimpleNamespace(
            state=types.SimpleNamespace(pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
        )

        global_obs = scenario._uav_confidence_observation()
        center_patch = scenario._local_confidence_observation(agent)

        self.assertEqual(global_obs.shape[-1], 5)
        self.assertAlmostEqual(float(global_obs[0, -1]), float(scenario.uav_confidence_grid.mean()))
        self.assertEqual(center_patch.shape[-1], 9)
        self.assertGreater(float(center_patch[0, 4]), 0.0)
        self.assertLess(float(center_patch[0, 4]), 1.0)

        agent.state.pos = torch.tensor([[0.95, 0.95]], dtype=torch.float32)
        edge_patch = scenario._local_confidence_observation(agent)

        self.assertGreater(float(edge_patch[0, -1]), 0.5)

    def test_confidence_frontier_features_point_toward_low_confidence_mass(self):
        scenario = self._coverage_scenario(grid_size=8)
        scenario.uav_frontier_obs = True
        scenario.uav_frontier_source = "confidence"
        scenario.coverage_grid[:] = True
        scenario.uav_confidence_grid[:] = 1.0
        scenario.uav_confidence_grid[:, :, :4] = 0.0

        features = scenario._uav_frontier_features_for_positions(torch.zeros(1, 1, 2))

        self.assertLess(float(features[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(features[0, 0, 1]), 0.0, places=6)
        self.assertGreater(float(features[0, 0, 2]), 0.0)
        self.assertGreater(float(features[0, 0, 3]), 0.0)

    def test_uav_confidence_move_reward_uses_best_one_step_opportunity(self):
        scenario = self._coverage_scenario(grid_size=16)
        scenario.r_uav_confidence = 0.0
        scenario.r_uav_confidence_move = 0.2
        scenario.uav_confidence_grid.zero_()
        scenario._pre_step_drone_pos = torch.zeros(1, 1, 2)

        confidence_reward, move_reward = scenario._update_uav_confidence(torch.tensor([[[0.5, 0.0]]]))

        self.assertEqual(float(confidence_reward[0, 0]), 0.0)
        self.assertGreater(float(move_reward[0, 0]), 0.0)
        self.assertLessEqual(float(move_reward[0, 0]), 0.2)
        self.assertGreater(float(scenario.metric_uav_confidence_opportunity_best_gain[0]), 0.0)
        self.assertGreater(float(scenario.metric_uav_confidence_opportunity_fraction[0]), 0.0)
        self.assertLessEqual(float(scenario.metric_uav_confidence_opportunity_fraction[0]), 1.0)

    def test_uav_move_coverage_reward_scales_with_new_cells_and_displacement(self):
        scenario = self._coverage_scenario(grid_size=16)
        drone_pos = torch.tensor([[[1.0, 0.0]]])  # 10 meters at 0.1 sim-units/m.
        coverage_new = torch.tensor([[10.0 / (16 * 16)]])

        reward, displacement_m, coverage_cells = scenario._uav_move_coverage_reward(
            drone_pos,
            coverage_new,
        )

        self.assertAlmostEqual(float(displacement_m[0, 0]), 10.0, places=6)
        self.assertAlmostEqual(float(coverage_cells[0, 0]), 10.0, places=6)
        self.assertAlmostEqual(float(reward[0, 0]), 0.1, places=6)

    def test_uav_move_coverage_reward_zero_without_new_coverage(self):
        scenario = self._coverage_scenario(grid_size=16)
        drone_pos = torch.tensor([[[1.0, 0.0]]])
        coverage_new = torch.zeros(1, 1)

        reward, _, coverage_cells = scenario._uav_move_coverage_reward(drone_pos, coverage_new)

        self.assertEqual(float(coverage_cells[0, 0]), 0.0)
        self.assertEqual(float(reward[0, 0]), 0.0)

    def test_uav_move_coverage_reward_can_use_opportunity_normalization(self):
        scenario = self._coverage_scenario(grid_size=16)
        scenario.r_uav_move_coverage = 0.2
        scenario.uav_move_coverage_normalization = "opportunity"
        scenario.r_uav_move_coverage_cap = 0.1
        drone_pos = torch.tensor([[[1.0, 0.0]]])  # 10 meters at 0.1 sim-units/m.
        coverage_new = torch.tensor([[10.0 / (16 * 16)]])
        opportunity_fraction = torch.tensor([[0.25]])

        reward, displacement_m, coverage_cells = scenario._uav_move_coverage_reward(
            drone_pos,
            coverage_new,
            opportunity_fraction,
        )

        self.assertAlmostEqual(float(displacement_m[0, 0]), 10.0, places=6)
        self.assertAlmostEqual(float(coverage_cells[0, 0]), 10.0, places=6)
        self.assertAlmostEqual(float(reward[0, 0]), 0.2 * (10.0 / 20.0) * 0.25, places=6)

        scenario.r_uav_move_coverage_cap = 0.01
        capped_reward, _, _ = scenario._uav_move_coverage_reward(
            drone_pos,
            coverage_new,
            opportunity_fraction,
        )
        self.assertAlmostEqual(float(capped_reward[0, 0]), 0.01, places=6)

    def test_uav_coverage_reward_can_use_reachable_uncovered_cells(self):
        scenario = self._coverage_scenario(grid_size=16)
        scenario.r_coverage = 0.5
        scenario.uav_coverage_normalization = "opportunity"
        scenario.uav_coverage_opportunity_cap = 1.0
        scenario.drone_altitude = torch.tensor([[0.10]])

        (
            credit,
            _,
            _,
            _,
            opportunity_fraction,
            opportunity_cells,
            opportunity_available_fraction,
        ) = scenario._coverage_reward(torch.zeros(1, 1, 2))
        new_cells = credit * float(scenario.fire_grid_size * scenario.fire_grid_size)
        reward = scenario._uav_coverage_reward(credit, opportunity_fraction)

        self.assertGreater(float(new_cells[0, 0]), 0.0)
        self.assertGreaterEqual(float(opportunity_cells[0, 0]), float(new_cells[0, 0]))
        self.assertGreater(float(opportunity_available_fraction[0, 0]), 0.0)
        self.assertLessEqual(float(opportunity_available_fraction[0, 0]), 1.0)
        self.assertAlmostEqual(
            float(opportunity_fraction[0, 0]),
            float(new_cells[0, 0] / opportunity_cells[0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(reward[0, 0]),
            0.5 * float(opportunity_fraction[0, 0]),
            places=6,
        )

        scenario.uav_coverage_opportunity_cap = 0.25
        capped_reward = scenario._uav_coverage_reward(credit, opportunity_fraction)
        self.assertLessEqual(float(capped_reward[0, 0]), 0.5 * 0.25 + 1e-7)

        scenario.uav_coverage_opportunity_cap = 0.0
        zero_capped_reward = scenario._uav_coverage_reward(credit, opportunity_fraction)
        self.assertEqual(float(zero_capped_reward[0, 0]), 0.0)

        scenario.uav_coverage_normalization = "map"
        map_reward = scenario._uav_coverage_reward(credit, opportunity_fraction)
        self.assertAlmostEqual(float(map_reward[0, 0]), 0.5 * float(credit[0, 0]), places=6)

    def test_uav_boundary_risk_metrics_scale_with_meter_distance(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.terrain_sim_units_per_meter = torch.tensor([0.01])

        risk, distance_m = scenario._uav_boundary_risk_metrics(torch.tensor([[[0.0, 0.0]]]))
        self.assertEqual(float(risk[0, 0]), 0.0)
        self.assertGreater(float(distance_m[0, 0]), 25.0)

        # x=0.825 is 0.125 sim-units from the x_max=0.95 body boundary.
        # With 0.1 sim-units/m this is 12.5m, half of the 25m margin.
        risk, distance_m = scenario._uav_boundary_risk_metrics(torch.tensor([[[0.825, 0.0]]]))
        self.assertAlmostEqual(float(distance_m[0, 0]), 12.5, places=6)
        self.assertAlmostEqual(float(risk[0, 0]), 0.5, places=6)

        risk, distance_m = scenario._uav_boundary_risk_metrics(torch.tensor([[[0.95, 0.0]]]))
        self.assertAlmostEqual(float(distance_m[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(risk[0, 0]), 1.0, places=6)

    def test_drone_boundary_projection_removes_outward_component(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.step_uav_boundary_projection_norm = torch.zeros(1, 1)
        scenario.step_uav_boundary_projection_count = torch.zeros(1, 1)
        agent = types.SimpleNamespace(
            name="drone_0",
            is_drone=True,
            state=types.SimpleNamespace(pos=torch.tensor([[-0.94, 0.0]])),
            action=types.SimpleNamespace(u=torch.tensor([[-0.8, 0.3]])),
        )

        scenario._project_drone_action_at_boundary(agent)

        torch.testing.assert_close(agent.action.u, torch.tensor([[0.0, 0.3]]))
        self.assertEqual(float(scenario.step_uav_boundary_projection_count[0, 0]), 1.0)
        self.assertAlmostEqual(float(scenario.step_uav_boundary_projection_norm[0, 0]), 0.8, places=6)

    def test_drone_boundary_escape_uses_meter_scaled_inward_push(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.uav_boundary_escape_m = 2.0
        scenario.step_uav_boundary_projection_norm = torch.zeros(1, 1)
        scenario.step_uav_boundary_projection_count = torch.zeros(1, 1)
        agent = types.SimpleNamespace(
            name="drone_0",
            is_drone=True,
            state=types.SimpleNamespace(pos=torch.tensor([[-0.94, -0.94]])),
            action=types.SimpleNamespace(u=torch.tensor([[-0.8, -0.6]])),
        )

        scenario._project_drone_action_at_boundary(agent)

        expected = 0.1 / math.sqrt(2.0)
        torch.testing.assert_close(
            agent.action.u,
            torch.tensor([[expected, expected]]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_drone_boundary_escape_is_off_by_default(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.step_uav_boundary_projection_norm = torch.zeros(1, 1)
        scenario.step_uav_boundary_projection_count = torch.zeros(1, 1)
        agent = types.SimpleNamespace(
            name="drone_0",
            is_drone=True,
            state=types.SimpleNamespace(pos=torch.tensor([[-0.94, -0.94]])),
            action=types.SimpleNamespace(u=torch.tensor([[-0.8, -0.6]])),
        )

        scenario._project_drone_action_at_boundary(agent)

        torch.testing.assert_close(agent.action.u, torch.zeros(1, 2))
        self.assertAlmostEqual(float(scenario.step_uav_boundary_projection_norm[0, 0]), 1.0, places=6)

    def test_drone_boundary_escape_scales_with_physical_step_distance(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.uav_boundary_escape_m = 2.0
        scenario.drone_speed_mps = 20.0
        scenario.step_uav_boundary_projection_norm = torch.zeros(1, 1)
        scenario.step_uav_boundary_projection_count = torch.zeros(1, 1)
        agent = types.SimpleNamespace(
            name="drone_0",
            is_drone=True,
            state=types.SimpleNamespace(pos=torch.tensor([[0.94, 0.94]])),
            action=types.SimpleNamespace(u=torch.tensor([[0.8, 0.6]])),
        )

        scenario._project_drone_action_at_boundary(agent)

        expected = 0.05 / math.sqrt(2.0)
        torch.testing.assert_close(
            agent.action.u,
            torch.tensor([[-expected, -expected]]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_world_clamp_stops_outward_velocity_at_uav_boundary(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.step_uav_boundary_hit = torch.zeros(1, 1)
        drone = _Agent(
            name="drone_0",
            is_drone=True,
            pos=[[0.95, -0.95]],
            vel=[[0.4, -0.3]],
        )
        scenario._world = types.SimpleNamespace(batch_dim=1, agents=[drone])

        scenario._clamp_agents_to_world()

        torch.testing.assert_close(drone.state.pos, torch.tensor([[0.95, -0.95]]))
        torch.testing.assert_close(drone.state.vel, torch.zeros(1, 2))
        self.assertEqual(float(scenario.step_uav_boundary_hit[0, 0]), 0.0)

    def test_world_clamp_records_uav_boundary_hit_and_clamps_position(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.step_uav_boundary_hit = torch.zeros(1, 1)
        drone = _Agent(
            name="drone_0",
            is_drone=True,
            pos=[[1.10, 0.0]],
            vel=[[0.4, 0.2]],
        )
        scenario._world = types.SimpleNamespace(batch_dim=1, agents=[drone])

        scenario._clamp_agents_to_world()

        torch.testing.assert_close(drone.state.pos, torch.tensor([[0.95, 0.0]]))
        torch.testing.assert_close(drone.state.vel, torch.tensor([[0.0, 0.2]]))
        self.assertEqual(float(scenario.step_uav_boundary_hit[0, 0]), 1.0)

    def test_boundary_observation_is_normalized_by_drone_footprint(self):
        scenario = self._coverage_scenario()
        scenario.agent_radius = 0.05
        scenario.drone_altitude = torch.tensor([[0.2]])
        agent = types.SimpleNamespace(
            name="drone_0",
            is_drone=True,
            state=types.SimpleNamespace(pos=torch.tensor([[-0.85, 0.0]])),
        )

        obs = scenario._boundary_observation(agent)

        self.assertAlmostEqual(float(obs[0, 0]), 0.5, places=6)
        torch.testing.assert_close(obs[0, 1:], torch.ones(3))

    def test_drone_blocked_patch_marks_outside_map_cells(self):
        scenario = self._coverage_scenario(grid_size=8)
        scenario.local_map_patch_size = 3
        scenario.required_clearance_grid = torch.zeros(1, 8, 8)
        scenario.drone_max_altitude_by_env = torch.ones(1)
        agent = types.SimpleNamespace(
            name="drone_0",
            is_drone=True,
            state=types.SimpleNamespace(pos=torch.tensor([[-0.99, 0.99]])),
        )

        features = scenario._local_terrain_features(agent)
        patch_cells = scenario.local_map_patch_size * scenario.local_map_patch_size
        blocked = features[:, patch_cells : 2 * patch_cells]

        expected = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        torch.testing.assert_close(blocked, expected)

    def test_uav_start_sampling_enforces_spacing_and_edge_margin(self):
        scenario = WildfireSearchScenario()
        scenario.n_drones = 3
        scenario.x_semidim = 1.0
        scenario.y_semidim = 1.0
        scenario.agent_radius = 0.0
        scenario.uav_start_min_separation_m = 150.0
        scenario.uav_start_edge_margin_m = 50.0
        scenario.uav_start_max_attempts = 4096
        scenario.terrain_sim_units_per_meter = torch.tensor([2.0 / 500.0])
        drones = [
            _Agent(name=f"drone_{idx}", is_drone=True, pos=[[0.0, 0.0]], vel=[[1.0, -1.0]])
            for idx in range(3)
        ]
        scenario._world = types.SimpleNamespace(agents=drones, batch_dim=1)

        torch.manual_seed(7)
        scenario._place_drones_jointly_uniform_interior(0)

        positions_sim = torch.stack([drone.state.pos[0] for drone in drones])
        positions_m = positions_sim / scenario.terrain_sim_units_per_meter[0]
        pairwise_m = torch.pdist(positions_m)
        edge_distances_m = torch.stack(
            [
                positions_m[:, 0] + 250.0,
                250.0 - positions_m[:, 0],
                positions_m[:, 1] + 250.0,
                250.0 - positions_m[:, 1],
            ],
            dim=1,
        )

        self.assertGreaterEqual(float(pairwise_m.min()), 150.0 - 1e-5)
        self.assertGreaterEqual(float(edge_distances_m.min()), 50.0 - 1e-5)
        for drone in drones:
            torch.testing.assert_close(drone.state.vel, torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
