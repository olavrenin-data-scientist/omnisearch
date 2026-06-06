import unittest

import torch

from agents.baselines import _drone_arrival_action


class DroneArrivalControllerTests(unittest.TestCase):
    def test_cruises_toward_distant_waypoint(self):
        action = _drone_arrival_action(
            direction=torch.tensor([1.0, 0.0]),
            distance=0.20,
            velocity=torch.zeros(2),
            max_speed=0.20,
            slowdown_distance=0.04,
        )

        torch.testing.assert_close(action, torch.tensor([0.95, 0.0]))

    def test_brakes_existing_velocity_near_waypoint(self):
        action = _drone_arrival_action(
            direction=torch.tensor([0.0, 1.0]),
            distance=0.01,
            velocity=torch.tensor([0.20, 0.0]),
            max_speed=0.20,
            slowdown_distance=0.04,
        )

        self.assertLess(float(action[0]), 0.0)
        self.assertGreater(float(action[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
