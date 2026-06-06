import unittest

import torch

from envs.wildfire_search import WildfireSearchScenario


class PathTraversabilityTests(unittest.TestCase):
    def test_long_path_does_not_skip_thin_obstacle(self):
        scenario = WildfireSearchScenario.__new__(WildfireSearchScenario)
        scenario.x_semidim = 1.0
        scenario.y_semidim = 1.0
        scenario.fire_grid_size = 128
        scenario.terrain_path_samples = 6
        scenario.traversable_grid = torch.ones(1, 128, 128, dtype=torch.bool)
        scenario.traversable_grid[:, :, 64] = False

        traversable = scenario._path_is_traversable(
            torch.tensor([[[-0.9, 0.0]]]),
            torch.tensor([[[0.9, 0.0]]]),
        )

        self.assertFalse(bool(traversable[0, 0]))

    def test_short_open_path_remains_traversable(self):
        scenario = WildfireSearchScenario.__new__(WildfireSearchScenario)
        scenario.x_semidim = 1.0
        scenario.y_semidim = 1.0
        scenario.fire_grid_size = 128
        scenario.terrain_path_samples = 6
        scenario.traversable_grid = torch.ones(1, 128, 128, dtype=torch.bool)

        traversable = scenario._path_is_traversable(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[0.01, 0.01]]]),
        )

        self.assertTrue(bool(traversable[0, 0]))


if __name__ == "__main__":
    unittest.main()
