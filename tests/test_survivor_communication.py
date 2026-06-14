import unittest
from pathlib import Path

import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario


ROOT = Path(__file__).resolve().parent.parent
TERRAIN_CACHE = ROOT / "data" / "terrain_cache" / "malibu_creek_state_park_california_128.npz"


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
        ).view(1, scenario.n_survivors, 4)

        self.assertTrue(torch.equal(message[0, 0], torch.tensor([1.0, 0.25, 0.30, 0.0])))
        self.assertTrue(torch.equal(message[0, 1], torch.zeros(4)))
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
        ).view(1, scenario.n_survivors, 4)
        self.assertEqual(float(disconnected[0, 0, 0]), 0.0)
        self.assertEqual(float(disconnected[0, 1, 0]), 1.0)

        scenario._survivor_message_observations(
            ground,
            torch.ones(1, 1, dtype=torch.bool),
        )
        disconnected_again = scenario._survivor_message_observations(
            ground,
            torch.zeros(1, 1, dtype=torch.bool),
        ).view(1, scenario.n_survivors, 4)
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
        survivor_block = obs[:, -4:].view(1, 1, 4)
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

    def test_local_map_patch_size_expands_mobility_and_blocked_only(self):
        env = self._diagnostic_env(local_map_patch_size=11)
        obs = env.scenario.observation(env.agents[0])

        # own pos/vel 4 + lidar 12 + fire 1 + terrain
        # terrain = mobility 11x11 + blocked 11x11 + clearance 3x3
        # flight 2 + no neighbors + one survivor message 4
        self.assertEqual(obs.shape[-1], 4 + 12 + 1 + 121 + 121 + 9 + 2 + 4)

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

    def test_ground_progress_reward_is_zero_on_first_known_step(self):
        scenario, ground = self._configure_progress_case()

        scenario._compute_step_rewards()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        scenario._compute_step_rewards()

        self.assertEqual(float(scenario.metric_reward_ugv_progress[0]), 0.0)

    def test_ground_progress_reward_is_clipped(self):
        scenario, ground = self._configure_progress_case()
        scenario.known_survivors_by_agent[0, 1, 0] = True
        large_step_sim = 10.0 * float(scenario.terrain_sim_units_per_meter[0])

        scenario._compute_step_rewards()
        ground.state.pos[:] = torch.tensor([[-0.4 + large_step_sim, 0.0]])
        scenario._compute_step_rewards()

        self.assertAlmostEqual(float(scenario.metric_reward_ugv_progress[0]), 0.5, places=5)

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
            "reward/ground_confirm",
            "reward/coverage",
            "cost/ugv_fire_exposure",
            "cost/ugv_travel",
            "cost/drone_energy",
            "cost/drone_climb",
        ):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
