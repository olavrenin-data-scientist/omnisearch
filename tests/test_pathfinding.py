import unittest

import numpy as np
import torch

from agents.baselines import _route_lookahead_cell
from agents.pathfinding import find_ground_route


class GroundRouteTests(unittest.TestCase):
    def test_diagonal_route_does_not_cut_blocked_corner(self):
        traversable = np.array(
            [
                [True, False, True],
                [False, True, True],
                [True, True, True],
            ],
            dtype=bool,
        )

        route = find_ground_route(
            traversable=traversable,
            movement_cost=np.ones((3, 3)),
            start=(0, 0),
            goal=(1, 1),
        )

        self.assertEqual(route, [])

    def test_diagonal_route_remains_available_through_open_corner(self):
        route = find_ground_route(
            traversable=np.ones((2, 2), dtype=bool),
            movement_cost=np.ones((2, 2)),
            start=(0, 0),
            goal=(1, 1),
        )

        self.assertEqual(route, [(0, 0), (1, 1)])

    def test_route_lookahead_stops_before_blocked_bend(self):
        traversable = torch.tensor(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]

        self.assertEqual(_route_lookahead_cell(traversable, path, 0), (3, 0))


if __name__ == "__main__":
    unittest.main()
