import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from agents.baselines import (
    BASELINES,
    RandomActionPolicy,
    RandomWalkPolicy,
    _reflect_random_walk_directions,
)


class RandomWalkTests(unittest.TestCase):
    def test_reflects_nominal_step_at_world_boundary(self):
        reflected = _reflect_random_walk_directions(
            positions=torch.tensor([[0.95, 0.0]]),
            directions=torch.tensor([[1.0, 0.0]]),
            step_distance=torch.tensor([[0.10]]),
            x_bound=0.96,
            y_bound=0.96,
        )

        torch.testing.assert_close(reflected, torch.tensor([[-1.0, 0.0]]))

    def test_ground_robot_uses_smallest_safe_turn(self):
        scenario = self._scenario(n_drones=0, path_rule="positive_y")
        env = SimpleNamespace(scenario=scenario)
        policy = RandomWalkPolicy(env, persistence_s=20.0)
        policy.headings.zero_()

        with patch(
            "agents.baselines.torch.randn_like",
            side_effect=lambda tensor: torch.zeros_like(tensor),
        ):
            action = policy(env)[0]

        expected = 0.95 * torch.tensor(
            [[math.cos(math.pi / 6), math.sin(math.pi / 6)]],
        )
        torch.testing.assert_close(action, expected)
        self.assertAlmostEqual(
            float(policy.headings[0, 0]),
            math.pi / 6,
            places=6,
        )

    def test_registered_as_random_walk(self):
        self.assertIs(BASELINES["random_walk"], RandomWalkPolicy)

    def test_nearest_candidate_removed_from_registry(self):
        self.assertNotIn("nearest_candidate", BASELINES)

    def test_random_walk_uses_random_walk_for_uavs_and_nearest_confirm_for_ugvs(self):
        agents = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.zeros(1, 2))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.zeros(1, 2))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.zeros(1, 2))),
        ]
        scenario = SimpleNamespace(
            world=SimpleNamespace(batch_dim=1, agents=agents),
            n_drones=2,
            n_ground=1,
            n_survivors=1,
            x_semidim=1.0,
            y_semidim=1.0,
            agent_radius=0.04,
            scouted_survivors=torch.tensor([[True]]),
            found_survivors=torch.tensor([[False]]),
            _path_is_traversable=lambda starts, endpoints: torch.ones(
                endpoints.shape[:-1], dtype=torch.bool,
            ),
        )
        policy = RandomWalkPolicy.__new__(RandomWalkPolicy)
        policy.scenario = scenario
        policy.ground_route_cache = [dict()]
        policy.headings = torch.zeros(1, 3)
        walk_actions = [
            torch.tensor([[0.8, 0.0]]),
            torch.tensor([[0.0, 0.8]]),
            torch.tensor([[0.4, 0.4]]),
        ]

        with patch(
            "agents.baselines._coordinated_ground_actions",
            return_value=[torch.tensor([[0.3, 0.3]])],
        ) as coordinated:
            ground = policy._ground_actions(
                directions=torch.stack(walk_actions, dim=1),
                step_distance=torch.ones(1, 3, 1),
            )

        coordinated.assert_called_once()
        torch.testing.assert_close(ground[0], torch.tensor([[0.3, 0.3]]))

    def test_random_action_uses_random_for_uavs_and_nearest_confirm_for_ugvs(self):
        scenario = SimpleNamespace(
            world=SimpleNamespace(batch_dim=1),
            n_drones=2,
            n_ground=1,
            n_survivors=1,
            step_count=torch.zeros(1, dtype=torch.long),
            scouted_survivors=torch.tensor([[True]]),
            found_survivors=torch.tensor([[False]]),
        )
        random_actions = [
            torch.tensor([[0.1, 0.1]]),
            torch.tensor([[0.2, 0.2]]),
            torch.tensor([[0.9, 0.9]]),
        ]
        env = SimpleNamespace(
            scenario=scenario,
            get_random_actions=lambda: random_actions,
        )
        policy = RandomActionPolicy(env)

        with patch(
            "agents.baselines._coordinated_ground_actions",
            return_value=[torch.tensor([[0.3, 0.3]])],
        ) as coordinated:
            actions = policy(env)

        coordinated.assert_called_once()
        torch.testing.assert_close(actions[0], random_actions[0])
        torch.testing.assert_close(actions[1], random_actions[1])
        torch.testing.assert_close(actions[2], torch.tensor([[0.3, 0.3]]))

    @staticmethod
    def _scenario(n_drones: int, path_rule: str):
        state = SimpleNamespace(
            pos=torch.zeros(1, 2),
            vel=torch.zeros(1, 2),
        )
        agent = SimpleNamespace(state=state)

        def path_is_traversable(starts, endpoints):
            if path_rule == "positive_y":
                delta = endpoints - starts
                return delta[..., 1] > 1e-6
            return torch.ones(endpoints.shape[:-1], dtype=torch.bool)

        return SimpleNamespace(
            world=SimpleNamespace(batch_dim=1, agents=[agent]),
            fire_grid=torch.zeros(1, 4, 4),
            step_count=torch.zeros(1, dtype=torch.long),
            n_drones=n_drones,
            n_ground=1 - n_drones,
            sim_step_seconds=1.0,
            drone_speed_mps=1.0,
            ground_speed_mps=1.0,
            terrain_sim_units_per_meter=torch.tensor([0.1]),
            x_semidim=1.0,
            y_semidim=1.0,
            agent_radius=0.04,
            _path_is_traversable=path_is_traversable,
        )


if __name__ == "__main__":
    unittest.main()
