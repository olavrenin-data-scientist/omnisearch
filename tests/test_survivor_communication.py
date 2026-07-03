import unittest
import math
from pathlib import Path

import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario


ROOT = Path(__file__).resolve().parent.parent
TERRAIN_CACHE = ROOT / "data" / "terrain_cache" / "malibu_creek_state_park_california_128.npz"
TERRAIN_500M_CACHE = ROOT / "data" / "terrain_cache" / "malibu_creek_500m_128.npz"


class SurvivorCommunicationTests(unittest.TestCase):
    def _env(self, *, n_survivors=2, comms_dropout=0.0):
        return vmas.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            seed=1,
            n_drones=1,
            n_ground=1,
            n_survivors=n_survivors,
            comms_dropout=comms_dropout,
            fire_grid_size=128,
            max_steps=10,
            terrain_source="real",
            terrain_cache_path=str(TERRAIN_CACHE),
        )

    def _diagnostic_env(self, **kwargs):
        params = {
            "n_drones": 0,
            "n_ground": 1,
            "n_survivors": 1,
            "known_survivors_at_reset": True,
            "disable_fire": True,
            "comms_dropout": 0.0,
            "fire_grid_size": 128,
            "max_steps": 10,
            "terrain_source": "real",
            "terrain_cache_path": str(TERRAIN_CACHE),
        }
        params.update(kwargs)
        return vmas.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            seed=1,
            **params,
        )

    def _reference_local_astar_route(self, scenario, env_index, pos, target_pos):
        patch_size = scenario.ugv_planner_patch_size
        radius = patch_size // 2
        pos_cell = scenario._single_position_to_grid_cell(pos)
        target_cell = scenario._single_position_to_grid_cell(target_pos)
        sx, sy = pos_cell
        x0 = max(0, sx - radius)
        x1 = min(scenario.fire_grid_size - 1, sx + radius)
        y0 = max(0, sy - radius)
        y1 = min(scenario.fire_grid_size - 1, sy + radius)
        bounds = (x0, x1, y0, y1)

        start = scenario._nearest_traversable_cell_in_bounds(env_index, sx, sy, bounds)
        if start is None:
            return None
        goal_candidates = scenario._local_planner_goal_candidates(
            env_index,
            start,
            target_cell,
            bounds,
        )
        if not goal_candidates:
            return None

        path = []
        goal = goal_candidates[0]
        for candidate in goal_candidates:
            path = scenario._local_astar_grid_path(env_index, start, candidate, bounds)
            if len(path) >= 2:
                goal = candidate
                break
        if len(path) < 2:
            return None

        traversable = scenario.traversable_grid[env_index]
        direct_blocked = not scenario._grid_segment_is_traversable(traversable, start, goal)
        waypoint = scenario._route_lookahead_cell(traversable, path, 0)
        detour_needed = scenario._local_astar_detour_needed(
            env_index,
            start,
            goal,
            waypoint,
            path,
            direct_blocked,
        )
        return waypoint, direct_blocked, detour_needed

    def _set_local_astar_case(self, scenario, start_cell, target_cell, blocked_cells=()):
        ground = scenario.world.agents[scenario.n_drones]
        survivor = scenario._survivors[0]
        device = ground.state.pos.device
        dtype = ground.state.pos.dtype
        scenario.traversable_grid.fill_(True)
        scenario.mobility_cost_grid.fill_(1.0)
        scenario.fire_grid.zero_()
        for x, y in blocked_cells:
            scenario.traversable_grid[0, y, x] = False
        ground.state.pos[:] = scenario._grid_cell_center_to_world(
            start_cell,
            device=device,
            dtype=dtype,
        ).view(1, 2)
        survivor.state.pos[:] = scenario._grid_cell_center_to_world(
            target_cell,
            device=device,
            dtype=dtype,
        ).view(1, 2)
        scenario.scouted_survivors[0, 0] = True
        scenario.known_survivors_by_agent[0, scenario.n_drones, 0] = True
        scenario.found_survivors.zero_()
        scenario._invalidate_ugv_planner_route_cache(terrain_changed=True)
        return ground, survivor

    def test_default_ground_speed_model_uses_spot_like_terrain_values(self):
        env = self._diagnostic_env()
        scenario = env.scenario

        self.assertEqual(scenario.slope_speed_weight, 0.5)
        self.assertTrue(torch.allclose(
            scenario.land_cover_speed_values.cpu(),
            torch.tensor([1.0, 0.95, 0.8, 0.7, 0.0, 0.0]),
        ))

    def test_default_reward_profile_uses_search_values(self):
        env = self._diagnostic_env()
        scenario = env.scenario

        self.assertEqual(scenario.r_found_survivor, 10.0)
        self.assertEqual(scenario.r_all_survivors_found, 0.0)
        self.assertEqual(scenario.r_drone_scout, 2.0)
        self.assertEqual(scenario.r_ground_confirm, 4.0)
        self.assertEqual(scenario.r_drone_shaping, 0.30)
        self.assertEqual(scenario.r_ground_shaping, 0.50)
        self.assertEqual(scenario.r_ground_approach, 0.05)
        self.assertEqual(scenario.r_ugv_movement_alignment, 0.20)
        self.assertEqual(scenario.r_ugv_planner_progress, 0.0)
        self.assertEqual(scenario.r_ugv_stall_penalty, 0.0)
        self.assertEqual(scenario.r_fire_penalty, -0.20)
        self.assertEqual(scenario.r_ground_travel_cost, -0.01)
        self.assertEqual(scenario.r_drone_climb_cost, -0.005)
        self.assertEqual(scenario.r_time_penalty, -0.0005)
        self.assertEqual(scenario.r_coverage, 5.0)
        self.assertEqual(scenario.r_uav_coverage_threshold, 0.0)
        self.assertEqual(scenario.uav_coverage_threshold_fraction, 0.95)
        self.assertEqual(scenario.ground_approach_milestone_radii_m, (75.0, 50.0, 40.0, 30.0, 20.0))
        torch.testing.assert_close(
            scenario.ground_approach_milestone_rewards_tensor.cpu(),
            torch.tensor([0.02, 0.025, 0.03, 0.04, 0.05]),
        )

    def test_legacy_drone_scouts_confirm_alias_enables_drone_confirmation(self):
        env = self._diagnostic_env(
            n_drones=1,
            n_ground=0,
            known_survivors_at_reset=False,
            survivor_spawn_reference="drone",
            drone_scouts_confirm_survivors=True,
        )
        scenario = env.scenario

        self.assertTrue(scenario.drone_can_confirm)
        self.assertFalse(hasattr(scenario, "drone_scouts_confirm_survivors"))

    def test_all_survivors_found_reward_fires_once(self):
        env = self._diagnostic_env(
            n_drones=1,
            n_ground=0,
            n_survivors=1,
            known_survivors_at_reset=False,
            drone_can_confirm=True,
            r_found_survivor=0.0,
            r_all_survivors_found=7.0,
            r_drone_scout=0.0,
            r_drone_shaping=0.0,
            r_time_penalty=0.0,
            r_coverage=0.0,
            drone_detection_quality=(1.0, 1.0, 1.0),
            drone_cover_detection_factors=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            drone_energy_costs=(0.0, 0.0, 0.0),
        )
        env.reset()
        scenario = env.scenario
        drone = env.agents[0]
        survivor = scenario._survivors[0]
        drone.state.pos[:] = torch.tensor([[0.0, 0.0]])
        survivor.state.pos[:] = torch.tensor([[0.0, 0.0]])

        scenario._compute_step_rewards()
        torch.testing.assert_close(drone.scenario_reward, torch.tensor([7.0]))
        torch.testing.assert_close(scenario.metric_reward_all_survivors_found, torch.tensor([7.0]))

        scenario._compute_step_rewards()
        torch.testing.assert_close(drone.scenario_reward, torch.tensor([0.0]))
        torch.testing.assert_close(scenario.metric_reward_all_survivors_found, torch.tensor([0.0]))

    def test_uav_coverage_threshold_reward_fires_once(self):
        env = self._diagnostic_env(
            n_drones=1,
            n_ground=0,
            n_survivors=1,
            known_survivors_at_reset=False,
            r_found_survivor=0.0,
            r_drone_scout=0.0,
            r_drone_shaping=0.0,
            r_time_penalty=0.0,
            r_coverage=0.0,
            r_uav_coverage_threshold=5.0,
            uav_coverage_threshold_fraction=0.0001,
            drone_energy_costs=(0.0, 0.0, 0.0),
        )
        env.reset()
        scenario = env.scenario
        drone = env.agents[0]
        drone.state.pos[:] = torch.tensor([[0.0, 0.0]])

        scenario._compute_step_rewards()
        torch.testing.assert_close(drone.scenario_reward, torch.tensor([5.0]))
        torch.testing.assert_close(scenario.metric_reward_uav_coverage_threshold, torch.tensor([5.0]))

        scenario._compute_step_rewards()
        torch.testing.assert_close(drone.scenario_reward, torch.tensor([0.0]))
        torch.testing.assert_close(scenario.metric_reward_uav_coverage_threshold, torch.tensor([0.0]))

    def test_ground_action_magnitude_is_normalized_before_terrain_speed(self):
        env = self._diagnostic_env()
        env.reset()
        scenario = env.scenario
        ground = env.agents[0]
        scenario.speed_multiplier_grid.fill_(1.0)

        ground.action.u = torch.tensor([[1.0, 0.0]])
        scenario.process_action(ground)
        cardinal_norm = ground.action.u.norm(dim=-1)

        ground.action.u = torch.tensor([[1.0, 1.0]])
        scenario.process_action(ground)
        diagonal_norm = ground.action.u.norm(dim=-1)

        expected_norm = torch.tensor([ground.u_range * ground.u_multiplier])
        torch.testing.assert_close(cardinal_norm, expected_norm)
        torch.testing.assert_close(diagonal_norm, expected_norm)

    def test_connected_ground_agent_receives_scouted_candidate(self):
        env = self._env()
        scenario = env.scenario
        drone, ground = env.agents
        survivor = scenario._survivors[0]
        drone.state.pos[:] = torch.tensor([[0.0, 0.0]])
        ground.state.pos[:] = torch.tensor([[0.25, -0.10]])
        survivor.state.pos[:] = torch.tensor([[0.50, 0.20]])
        scenario.known_survivors_by_agent[0, 0, 0] = True

        message = scenario._survivor_message_observations(
            ground,
            torch.ones(1, 1, dtype=torch.bool),
        ).view(1, scenario.n_survivors, 7)

        expected_unit = torch.tensor([0.25, 0.30]) / torch.linalg.norm(torch.tensor([0.25, 0.30]))
        torch.testing.assert_close(message[0, 0, :3], torch.tensor([1.0, 0.25, 0.30]))
        torch.testing.assert_close(message[0, 0, 3:5], expected_unit)
        self.assertGreater(float(message[0, 0, 5]), 0.0)
        self.assertEqual(float(message[0, 0, 6]), 0.0)
        self.assertTrue(torch.equal(message[0, 1], torch.zeros(7)))
        self.assertTrue(scenario.known_survivors_by_agent[0, 1, 0])

    def test_dropout_blocks_new_team_message_but_keeps_local_memory(self):
        env = self._env()
        scenario = env.scenario
        ground = env.agents[1]
        scenario.known_survivors_by_agent[0, 0, 0] = True
        scenario.known_survivors_by_agent[0, 1, 1] = True

        disconnected = scenario._survivor_message_observations(
            ground,
            torch.zeros(1, 1, dtype=torch.bool),
        ).view(1, scenario.n_survivors, 7)
        self.assertEqual(float(disconnected[0, 0, 0]), 0.0)
        self.assertEqual(float(disconnected[0, 1, 0]), 1.0)

        scenario._survivor_message_observations(
            ground,
            torch.ones(1, 1, dtype=torch.bool),
        )
        disconnected_again = scenario._survivor_message_observations(
            ground,
            torch.zeros(1, 1, dtype=torch.bool),
        ).view(1, scenario.n_survivors, 7)
        self.assertEqual(float(disconnected_again[0, 0, 0]), 1.0)
        self.assertEqual(float(disconnected_again[0, 1, 0]), 1.0)

    def test_ground_cannot_confirm_before_drone_scout(self):
        env = self._env(n_survivors=1)
        scenario = env.scenario
        drone, ground = env.agents
        survivor = scenario._survivors[0]
        drone.state.pos[:] = torch.tensor([[-1.0, -1.0]])
        ground.state.pos[:] = torch.tensor([[0.5, 0.5]])
        survivor.state.pos[:] = ground.state.pos

        scenario._compute_step_rewards()

        self.assertFalse(bool(scenario.scouted_survivors[0, 0]))
        self.assertFalse(bool(scenario.found_survivors[0, 0]))
        self.assertFalse(bool(scenario.step_ground_confirmations[0, 0, 0]))

    def test_known_survivors_at_reset_initializes_ground_mission_memory(self):
        env = self._diagnostic_env()
        scenario = env.scenario

        self.assertTrue(bool(scenario.scouted_survivors[0, 0]))
        self.assertTrue(bool(scenario.known_survivors_by_agent[0, 0, 0]))

        obs = scenario.observation(env.agents[0])
        survivor_block = obs[:, -7:].view(1, 1, 7)
        self.assertEqual(float(survivor_block[0, 0, 0]), 1.0)

    def test_disable_fire_leaves_hazard_fields_empty_after_reset(self):
        env = self._diagnostic_env()
        scenario = env.scenario

        self.assertEqual(int(scenario.fire_grid.sum().item()), 0)
        self.assertEqual(int(scenario.burned_grid.sum().item()), 0)
        self.assertEqual(float(scenario.smoke_grid.sum().item()), 0.0)

    def test_known_survivor_can_be_confirmed_without_drone(self):
        env = self._diagnostic_env(ground_confirm_min_m=20.0)
        scenario = env.scenario
        ground = env.agents[0]
        survivor = scenario._survivors[0]
        ground.state.pos[:] = survivor.state.pos

        scenario._compute_step_rewards()

        self.assertTrue(bool(scenario.found_survivors[0, 0]))
        self.assertTrue(bool(scenario.step_ground_confirmations[0, 0, 0]))

    def test_known_survivor_spawn_distance_places_candidate_near_ground(self):
        env = self._diagnostic_env(
            known_survivor_spawn_distance_m=80.0,
            ground_confirm_min_m=20.0,
        )
        scenario = env.scenario
        ground = env.agents[0]
        survivor = scenario._survivors[0]
        scale = float(scenario.terrain_sim_units_per_meter[0])
        distance_m = float(torch.linalg.norm(ground.state.pos - survivor.state.pos) / scale)

        self.assertGreater(distance_m, 20.0)
        self.assertLess(distance_m, 120.0)

    def test_known_survivor_spawn_distance_range_is_supported(self):
        env = self._diagnostic_env(
            known_survivor_spawn_distance_m=65.0,
            known_survivor_spawn_distance_min_m=30.0,
            known_survivor_spawn_distance_max_m=100.0,
            ground_confirm_min_m=20.0,
        )
        scenario = env.scenario
        ground = env.agents[0]
        survivor = scenario._survivors[0]
        scale = float(scenario.terrain_sim_units_per_meter[0])
        distance_m = float(torch.linalg.norm(ground.state.pos - survivor.state.pos) / scale)

        self.assertEqual(scenario.known_survivor_spawn_distance_min_m, 30.0)
        self.assertEqual(scenario.known_survivor_spawn_distance_max_m, 100.0)
        self.assertGreater(distance_m, 20.0)
        self.assertLess(distance_m, 150.0)

    def test_known_survivor_spawn_distance_min_without_max_is_unbounded(self):
        env = self._diagnostic_env(
            known_survivor_spawn_distance_min_m=30.0,
            terrain_cache_path=str(TERRAIN_500M_CACHE),
        )
        scenario = env.scenario
        ground = env.agents[0]
        survivor = scenario._survivors[0]
        scale = float(scenario.terrain_sim_units_per_meter[0])
        distance_m = float(torch.linalg.norm(ground.state.pos - survivor.state.pos) / scale)

        self.assertEqual(scenario.known_survivor_spawn_distance_min_m, 30.0)
        self.assertTrue(math.isinf(scenario.known_survivor_spawn_distance_max_m))
        self.assertGreaterEqual(distance_m, 30.0)

    def test_ugv_planner_patch_size_must_be_positive_odd(self):
        with self.assertRaises(ValueError):
            self._diagnostic_env(ugv_planner_hint="local_astar", ugv_planner_patch_size=10)

    def test_ugv_planner_lookahead_is_clamped_to_patch_radius(self):
        env = self._diagnostic_env(
            ugv_planner_hint="local_astar",
            ugv_planner_patch_size=11,
            ugv_planner_lookahead_cells=10,
        )

        self.assertEqual(env.scenario.ugv_planner_lookahead_cells, 5)

    def test_local_astar_planner_hint_appends_features(self):
        env = self._diagnostic_env(
            local_map_patch_size=7,
            ugv_planner_hint="local_astar",
            ugv_planner_patch_size=11,
        )
        scenario = env.scenario

        obs = scenario.observation(env.agents[0])

        expected_width = 4 + 12 + 1 + 2 * 7 * 7 + 9 + 5 + 2 + 4 + 7
        self.assertEqual(obs.shape[-1], expected_width)

    def test_local_astar_planner_hint_points_toward_clear_local_target(self):
        env = self._diagnostic_env(
            local_map_patch_size=7,
            ugv_planner_hint="local_astar",
            ugv_planner_patch_size=11,
            ugv_planner_lookahead_cells=10,
        )
        scenario = env.scenario
        ground = env.agents[0]
        survivor = scenario._survivors[0]
        device = ground.state.pos.device
        dtype = ground.state.pos.dtype

        scenario.traversable_grid.fill_(True)
        scenario.mobility_cost_grid.fill_(1.0)
        scenario.fire_grid.zero_()
        ground.state.pos[:] = scenario._grid_cell_center_to_world((64, 64), device=device, dtype=dtype).view(1, 2)
        survivor.state.pos[:] = scenario._grid_cell_center_to_world((67, 64), device=device, dtype=dtype).view(1, 2)
        scenario.known_survivors_by_agent[0, 0, 0] = True
        scenario.confirmed_survivors_by_agent.zero_()

        obs = scenario.observation(ground)
        hint_offset = 4 + 12 + 1 + 2 * 7 * 7 + 9
        hint = obs[0, hint_offset : hint_offset + 5]

        self.assertGreater(float(hint[0]), 0.8)
        self.assertLess(abs(float(hint[1])), 0.2)
        self.assertGreater(float(hint[2]), 0.0)
        self.assertEqual(float(hint[3]), 1.0)
        self.assertEqual(float(hint[4]), 0.0)

    def test_optimized_local_astar_route_matches_reference_cases(self):
        route_cases = (
            ("clear_direct", (64, 64), (67, 64), ()),
            ("blocked_direct", (64, 64), (69, 64), ((66, 64),)),
            ("diagonal_corner_blocked", (64, 64), (67, 67), ((65, 64), (64, 65))),
            ("target_outside_patch", (64, 64), (90, 64), ()),
            ("start_blocked", (64, 64), (69, 64), ((64, 64),)),
            (
                "no_reachable_route",
                (64, 64),
                (69, 64),
                tuple(
                    (x, y)
                    for x in range(59, 70)
                    for y in range(59, 70)
                    if (x, y) != (64, 64)
                ),
            ),
        )
        for patch_size in (7, 11, 15):
            env = self._diagnostic_env(
                ugv_planner_hint="local_astar",
                ugv_planner_patch_size=patch_size,
                ugv_planner_lookahead_cells=patch_size // 2,
            )
            scenario = env.scenario
            ground = env.agents[0]
            survivor = scenario._survivors[0]
            for name, start_cell, target_cell, blocked_cells in route_cases:
                with self.subTest(patch_size=patch_size, case=name):
                    self._set_local_astar_case(
                        scenario,
                        start_cell,
                        target_cell,
                        blocked_cells,
                    )
                    expected = self._reference_local_astar_route(
                        scenario,
                        0,
                        ground.state.pos[0],
                        survivor.state.pos[0],
                    )
                    actual = scenario._local_astar_route_for_env(
                        0,
                        ground.state.pos[0],
                        survivor.state.pos[0],
                    )
                    self.assertEqual(actual, expected)

    def test_ugv_planner_reward_reuses_hint_route_cache(self):
        env = self._diagnostic_env(
            ugv_planner_hint="local_astar",
            ugv_planner_patch_size=11,
            ugv_planner_lookahead_cells=5,
        )
        scenario = env.scenario
        ground, survivor = self._set_local_astar_case(
            scenario,
            (64, 64),
            (69, 64),
            ((66, 64),),
        )
        scenario.r_ugv_planner_progress = 0.05
        scenario.ugv_planner_progress_scale_m = 1.0

        original = scenario._local_astar_route_uncached_for_env
        calls = {"count": 0}

        def counted(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        scenario._local_astar_route_uncached_for_env = counted
        scenario.observation(ground)
        self.assertEqual(calls["count"], 1)

        start_pos = ground.state.pos.unsqueeze(1).clone()
        target_pos = survivor.state.pos.unsqueeze(1).clone()
        route = scenario._local_astar_route_for_env(
            0,
            ground.state.pos[0],
            survivor.state.pos[0],
            ground_index=0,
        )
        self.assertIsNotNone(route)
        waypoint, _direct_blocked, _detour_needed = route
        waypoint_pos = scenario._grid_cell_center_to_world(
            waypoint,
            device=ground.state.pos.device,
            dtype=ground.state.pos.dtype,
        ).view(1, 1, 2)
        direction = waypoint_pos - start_pos
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scale = float(scenario.terrain_sim_units_per_meter[0])
        end_pos = start_pos + direction * scale
        gate = torch.ones(1, 1, dtype=torch.bool)

        reward, progress_m, _progress_scaled, active, direct_blocked, detour_needed = (
            scenario._ugv_planner_progress_rewards(start_pos, end_pos, target_pos, gate)
        )
        self.assertEqual(calls["count"], 1)
        self.assertTrue(bool(active[0, 0]))
        self.assertTrue(bool(direct_blocked[0, 0]))
        self.assertTrue(bool(detour_needed[0, 0]))
        self.assertGreater(float(progress_m[0, 0]), 0.0)
        self.assertGreater(float(reward[0, 0]), 0.0)

        scenario._invalidate_ugv_planner_route_cache()
        scenario._ugv_planner_progress_rewards(start_pos, end_pos, target_pos, gate)
        self.assertEqual(calls["count"], 2)

    def test_known_survivor_spawn_distance_range_samples_angles(self):
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        counts = {label: 0 for label in labels}
        distances_m = []
        torch.manual_seed(123)
        env = self._diagnostic_env(
            known_survivor_spawn_distance_m=55.0,
            known_survivor_spawn_distance_min_m=30.0,
            known_survivor_spawn_distance_max_m=80.0,
            terrain_cache_path=str(TERRAIN_500M_CACHE),
        )
        scenario = env.scenario

        for _ in range(320):
            env.reset()
            ground_pos = env.agents[0].state.pos[0]
            survivor_pos = scenario._survivors[0].state.pos[0]
            offset = survivor_pos - ground_pos
            angle = float(torch.atan2(offset[1], offset[0]))
            bucket = int(((angle + math.pi / 8.0) % (2.0 * math.pi)) / (math.pi / 4.0))
            counts[labels[bucket]] += 1
            scale = float(scenario.terrain_sim_units_per_meter[0])
            distances_m.append(float(torch.linalg.norm(offset) / scale))

        self.assertEqual(set(counts.keys()), set(labels))
        self.assertGreaterEqual(min(counts.values()), 20)
        self.assertLessEqual(max(counts.values()), 70)
        self.assertGreater(sum(counts[label] for label in ("S", "SW")), 0)
        self.assertLessEqual(sum(counts[label] for label in ("S", "SW")), 130)
        self.assertGreater(min(distances_m), 20.0)
        self.assertLess(max(distances_m), 95.0)
        self.assertGreater(sum(distances_m) / len(distances_m), 45.0)
        self.assertLess(sum(distances_m) / len(distances_m), 65.0)

    def test_local_map_patch_size_uses_flattened_patch_features_for_ugv(self):
        env = self._diagnostic_env(local_map_patch_size=11)
        obs = env.scenario.observation(env.agents[0])

        # own pos/vel 4 + lidar 12 + fire 1 + terrain
        # terrain = 11x11 normalized costs + 11x11 blocked indicators + 3x3 clearance
        # flight 2 + boundary 4 + no neighbors + one survivor message 7
        self.assertEqual(obs.shape[-1], 4 + 12 + 1 + 2 * 11 * 11 + 9 + 2 + 4 + 7)

    def test_local_map_patch_size_keeps_drone_and_ugv_observation_widths_equal(self):
        env = self._env(n_survivors=1)
        env.scenario.local_map_patch_size = 11
        obs = env.scenario.observation(env.agents[0])
        ground_obs = env.scenario.observation(env.agents[1])

        # Both roles receive the same terrain block:
        # 11x11 normalized costs + 11x11 blocked indicators + 3x3 clearance.
        # own pos/vel 4 + lidar/dummy lidar 12 + fire 1 + terrain + flight 2 + boundary 4
        # + one teammate relative position 2 + one survivor message 7
        expected = 4 + 12 + 1 + 2 * 11 * 11 + 9 + 2 + 4 + 2 + 7
        self.assertEqual(obs.shape[-1], expected)
        self.assertEqual(ground_obs.shape[-1], expected)

    def test_drone_terrain_features_zero_ground_mobility_channels(self):
        env = self._env(n_survivors=1)
        scenario = env.scenario
        drone = env.agents[0]
        ground = env.agents[1]
        scenario.local_map_patch_size = 3
        scenario.mobility_cost_grid.fill_(2.0)
        scenario.traversable_grid.fill_(False)
        scenario.required_clearance_grid[:] = scenario.drone_max_altitude_by_env.view(-1, 1, 1) * 0.5

        drone_features = scenario._local_terrain_features(drone)
        ground_features = scenario._local_terrain_features(ground)
        patch_cells = scenario.local_map_patch_size * scenario.local_map_patch_size

        torch.testing.assert_close(drone_features[:, :patch_cells], torch.zeros_like(drone_features[:, :patch_cells]))
        torch.testing.assert_close(
            drone_features[:, patch_cells : 2 * patch_cells],
            torch.zeros_like(drone_features[:, patch_cells : 2 * patch_cells]),
        )
        torch.testing.assert_close(
            drone_features[:, 2 * patch_cells :],
            torch.full_like(drone_features[:, 2 * patch_cells :], 0.5),
        )
        torch.testing.assert_close(
            ground_features[:, :patch_cells],
            torch.ones_like(ground_features[:, :patch_cells]),
        )
        torch.testing.assert_close(
            ground_features[:, patch_cells : 2 * patch_cells],
            torch.ones_like(ground_features[:, patch_cells : 2 * patch_cells]),
        )
        torch.testing.assert_close(
            ground_features[:, 2 * patch_cells :],
            torch.zeros_like(ground_features[:, 2 * patch_cells :]),
        )

    def test_local_map_patch_size_must_be_positive_odd(self):
        with self.assertRaises(ValueError):
            self._diagnostic_env(local_map_patch_size=10)

    def _configure_progress_case(self):
        env = self._env(n_survivors=1)
        scenario = env.scenario
        drone, ground = env.agents
        survivor = scenario._survivors[0]
        scenario.r_ground_shaping = 0.5
        scenario.r_ground_approach = 0.0
        scenario.ground_approach_milestone_rewards_tensor.zero_()
        scenario.ground_progress_scale_m = 3.2
        scenario.scouted_survivors[0, 0] = True
        drone.state.pos[:] = torch.tensor([[-1.0, -1.0]])
        survivor.state.pos[:] = torch.tensor([[0.0, 0.0]])
        ground.state.pos[:] = torch.tensor([[-0.4, 0.0]])
        return scenario, ground

    def test_known_ground_progress_reward_is_positive_when_moving_closer(self):
        scenario, ground = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        step_sim = 3.2 * float(scenario.terrain_sim_units_per_meter[0])

        scenario._compute_step_rewards()
        ground.state.pos[:] = torch.tensor([[-0.4 + step_sim, 0.0]])
        scenario._compute_step_rewards()

        self.assertAlmostEqual(float(scenario.metric_reward_ugv_progress[0]), 0.5, places=5)
        self.assertEqual(float(scenario.metric_ugv_known_target_valid[0]), 1.0)
        self.assertEqual(float(scenario.metric_ugv_prev_distance_valid[0]), 1.0)
        self.assertEqual(float(scenario.metric_ugv_same_target[0]), 1.0)
        self.assertEqual(float(scenario.metric_ugv_progress_gate_active[0]), 1.0)
        self.assertAlmostEqual(float(scenario.metric_ugv_ground_progress_m[0]), 3.2, places=3)
        self.assertAlmostEqual(float(scenario.metric_ugv_ground_progress_scaled[0]), 1.0, places=5)

    def test_known_ground_progress_reward_is_negative_when_moving_away(self):
        scenario, ground = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        step_sim = 3.2 * float(scenario.terrain_sim_units_per_meter[0])

        scenario._compute_step_rewards()
        ground.state.pos[:] = torch.tensor([[-0.4 - step_sim, 0.0]])
        scenario._compute_step_rewards()

        self.assertAlmostEqual(float(scenario.metric_reward_ugv_progress[0]), -0.5, places=4)

    def test_ground_progress_reward_is_zero_before_candidate_is_known(self):
        scenario, ground = self._configure_progress_case()
        step_sim = 3.2 * float(scenario.terrain_sim_units_per_meter[0])

        scenario._compute_step_rewards()
        ground.state.pos[:] = torch.tensor([[-0.4 + step_sim, 0.0]])
        scenario._compute_step_rewards()

        self.assertEqual(float(scenario.metric_reward_ugv_progress[0]), 0.0)
        self.assertEqual(float(scenario.metric_ugv_known_target_valid[0]), 0.0)
        self.assertEqual(float(scenario.metric_ugv_progress_gate_active[0]), 0.0)

    def test_ground_progress_reward_is_zero_on_first_known_step(self):
        scenario, ground = self._configure_progress_case()

        scenario._compute_step_rewards()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        scenario._compute_step_rewards()

        self.assertEqual(float(scenario.metric_reward_ugv_progress[0]), 0.0)
        self.assertEqual(float(scenario.metric_ugv_known_target_valid[0]), 1.0)
        self.assertEqual(float(scenario.metric_ugv_prev_distance_valid[0]), 0.0)
        self.assertEqual(float(scenario.metric_ugv_progress_gate_active[0]), 0.0)

    def test_ground_progress_reward_is_clipped(self):
        scenario, ground = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        large_step_sim = 10.0 * float(scenario.terrain_sim_units_per_meter[0])

        scenario._compute_step_rewards()
        ground.state.pos[:] = torch.tensor([[-0.4 + large_step_sim, 0.0]])
        scenario._compute_step_rewards()

        self.assertAlmostEqual(float(scenario.metric_reward_ugv_progress[0]), 0.5, places=5)

    def test_ground_approach_reward_pays_once_when_crossing_milestones(self):
        scenario, ground = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        scenario.ground_approach_milestone_radii_m_tensor = torch.tensor(
            [75.0, 50.0, 40.0, 30.0, 20.0],
            device=scenario.fire_grid.device,
        )
        scenario.ground_approach_milestone_rewards_tensor = torch.tensor(
            [0.04, 0.05, 0.06, 0.08, 0.10],
            device=scenario.fire_grid.device,
        )
        scenario.ground_approach_milestones_reached.zero_()
        scale = float(scenario.terrain_sim_units_per_meter[0])

        ground.state.pos[:] = torch.tensor([[-80.0 * scale, 0.0]])
        scenario._compute_step_rewards()
        self.assertEqual(float(scenario.metric_reward_ugv_approach[0]), 0.0)

        scenario._pre_step_ground_pos[:, 0, :] = ground.state.pos
        ground.state.pos[:] = torch.tensor([[-45.0 * scale, 0.0]])
        scenario._compute_step_rewards()
        self.assertAlmostEqual(float(scenario.metric_reward_ugv_approach[0]), 0.09, places=5)

        scenario._compute_step_rewards()
        self.assertEqual(float(scenario.metric_reward_ugv_approach[0]), 0.0)

        scenario._pre_step_ground_pos[:, 0, :] = ground.state.pos
        ground.state.pos[:] = torch.tensor([[-19.0 * scale, 0.0]])
        scenario._compute_step_rewards()
        self.assertAlmostEqual(float(scenario.metric_reward_ugv_approach[0]), 0.24, places=5)

    def test_ground_approach_reward_requires_aligned_progress(self):
        scenario, ground = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        scenario.ground_approach_milestone_radii_m_tensor = torch.tensor(
            [75.0, 50.0],
            device=scenario.fire_grid.device,
        )
        scenario.ground_approach_milestone_rewards_tensor = torch.tensor(
            [0.04, 0.05],
            device=scenario.fire_grid.device,
        )
        scenario.ground_approach_milestones_reached.zero_()
        scale = float(scenario.terrain_sim_units_per_meter[0])

        ground.state.pos[:] = torch.tensor([[-80.0 * scale, 0.0]])
        scenario._compute_step_rewards()

        scenario._pre_step_ground_pos[:, 0, :] = torch.tensor([[-45.0 * scale, -10.0 * scale]])
        ground.state.pos[:] = torch.tensor([[-45.0 * scale, 0.0]])
        scenario._compute_step_rewards()

        self.assertEqual(float(scenario.metric_reward_ugv_approach[0]), 0.0)
        self.assertFalse(bool(scenario.ground_approach_milestones_reached.any()))

    def test_ugv_stall_penalty_applies_after_known_target_gate(self):
        scenario, _ = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        scenario.r_ugv_stall_penalty = 0.02
        scenario.ugv_stall_displacement_threshold_m = 0.05

        scenario._compute_step_rewards()
        self.assertEqual(float(scenario.metric_reward_ugv_stall_penalty[0]), 0.0)

        scenario.step_ugv_actual_displacement_m[0, 0] = 0.01
        scenario._compute_step_rewards()

        self.assertAlmostEqual(float(scenario.metric_reward_ugv_stall_penalty[0]), -0.02, places=5)

    def test_ugv_planner_progress_reward_uses_astar_waypoint_when_detouring(self):
        env = self._diagnostic_env(
            ugv_planner_hint="local_astar",
            ugv_planner_patch_size=11,
            ugv_planner_lookahead_cells=5,
        )
        scenario = env.scenario
        ground = env.agents[0]
        survivor = scenario._survivors[0]
        device = ground.state.pos.device
        dtype = ground.state.pos.dtype
        scenario.r_ground_shaping = 0.0
        scenario.r_ground_approach = 0.0
        scenario.ground_approach_milestone_rewards_tensor.zero_()
        scenario.r_ugv_movement_alignment = 0.0
        scenario.r_ugv_planner_progress = 0.05
        scenario.ugv_planner_progress_scale_m = 1.0
        scenario.detection_range_by_env.zero_()
        scenario.traversable_grid.fill_(True)
        scenario.mobility_cost_grid.fill_(1.0)
        scenario.fire_grid.zero_()

        start_cell = (64, 64)
        target_cell = (69, 64)
        scenario.traversable_grid[0, 64, 66] = False
        ground.state.pos[:] = scenario._grid_cell_center_to_world(
            start_cell, device=device, dtype=dtype,
        ).view(1, 2)
        survivor.state.pos[:] = scenario._grid_cell_center_to_world(
            target_cell, device=device, dtype=dtype,
        ).view(1, 2)
        scenario.scouted_survivors[0, 0] = True
        scenario.known_survivors_by_agent[0, 0, 0] = True
        scenario.found_survivors.zero_()

        route = scenario._local_astar_route_for_env(0, ground.state.pos[0], survivor.state.pos[0])
        self.assertIsNotNone(route)
        waypoint, direct_blocked, detour_needed = route
        self.assertTrue(direct_blocked)
        self.assertTrue(detour_needed)

        scenario._compute_step_rewards()
        waypoint_pos = scenario._grid_cell_center_to_world(waypoint, device=device, dtype=dtype)
        direction = waypoint_pos - ground.state.pos[0]
        direction = direction / direction.norm().clamp_min(1e-9)
        scale = float(scenario.terrain_sim_units_per_meter[0])
        scenario._pre_step_ground_pos[:, 0, :] = ground.state.pos
        ground.state.pos[:] = ground.state.pos + direction.view(1, 2) * scale
        scenario.step_ugv_actual_displacement_m[0, 0] = 1.0
        scenario._compute_step_rewards()

        self.assertGreater(float(scenario.metric_reward_ugv_planner_progress[0]), 0.0)
        self.assertEqual(float(scenario.metric_ugv_planner_active[0]), 1.0)
        self.assertEqual(float(scenario.metric_ugv_planner_direct_blocked[0]), 1.0)
        self.assertEqual(float(scenario.metric_ugv_planner_detour_needed[0]), 1.0)

    def test_info_contains_training_debug_metrics(self):
        env = self._env(n_survivors=1)
        scenario = env.scenario

        scenario._compute_step_rewards()
        info = scenario.info(env.agents[0])

        for key in (
            "mission/new_scouts",
            "mission/new_confirmations",
            "mission/n_scouted",
            "mission/n_confirmed",
            "mission/full_success",
            "reward/team",
            "reward/drone_scout",
            "reward/drone_progress",
            "reward/ugv_progress",
            "reward/ugv_approach",
            "reward/ugv_movement_alignment",
            "reward/ugv_planner_progress",
            "reward/ugv_stall_penalty",
            "reward/ground_confirm",
            "reward/coverage",
            "cost/ugv_fire_exposure",
            "cost/ugv_travel",
            "cost/drone_energy",
            "cost/drone_climb",
            "diagnostic/ugv_known_target_valid",
            "diagnostic/ugv_same_target",
            "diagnostic/ugv_prev_distance_valid",
            "diagnostic/ugv_progress_gate_active",
            "diagnostic/ugv_target_index",
            "diagnostic/ugv_ground_progress_m",
            "diagnostic/ugv_ground_progress_scaled",
            "diagnostic/ugv_planner_progress_m",
            "diagnostic/ugv_planner_progress_scaled",
            "diagnostic/ugv_planner_active",
            "diagnostic/ugv_planner_direct_blocked",
            "diagnostic/ugv_planner_detour_needed",
            "diagnostic/ugv_action_alignment",
            "diagnostic/ugv_movement_alignment",
        ):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
