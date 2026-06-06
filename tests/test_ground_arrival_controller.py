import unittest

import torch

from agents.baselines import _ground_arrival_action


class GroundArrivalControllerTests(unittest.TestCase):
    def test_action_is_invariant_to_simulation_scale(self):
        actions = []
        for scale in (0.5, 1.0, 2.0, 3.0):
            sim_units_per_meter = 0.0002 * scale
            direction = torch.tensor([[1.0, 0.0]])
            distance = torch.tensor([[5.0 * sim_units_per_meter]])
            velocity = torch.tensor([[0.8 * sim_units_per_meter, 0.0]])
            max_speed = 1.6 * sim_units_per_meter
            slowdown = torch.tensor([[10.0 * sim_units_per_meter]])
            actions.append(
                _ground_arrival_action(
                    direction=direction,
                    distance=distance,
                    velocity=velocity,
                    max_speed=max_speed,
                    slowdown_distance=slowdown,
                    damping=0.6,
                )
            )

        for action in actions[1:]:
            torch.testing.assert_close(action, actions[0])

    def test_proportional_action_uses_physical_slowdown_distance(self):
        action = _ground_arrival_action(
            direction=torch.tensor([[1.0, 0.0]]),
            distance=torch.tensor([[0.005]]),
            velocity=torch.zeros(1, 2),
            max_speed=0.01,
            slowdown_distance=torch.tensor([[0.01]]),
            damping=0.6,
        )

        torch.testing.assert_close(action, torch.tensor([[0.5, 0.0]]))

    def test_velocity_term_brakes_motion(self):
        action = _ground_arrival_action(
            direction=torch.tensor([[1.0, 0.0]]),
            distance=torch.tensor([[0.005]]),
            velocity=torch.tensor([[0.005, 0.0]]),
            max_speed=0.01,
            slowdown_distance=torch.tensor([[0.01]]),
            damping=0.6,
        )

        torch.testing.assert_close(action, torch.tensor([[0.2, 0.0]]))


if __name__ == "__main__":
    unittest.main()
