import unittest

import torch

from agents.baselines import (
    _merge_local_bool_knowledge,
    _merge_local_timestamp_maps,
)


class AntColonyMemoryTests(unittest.TestCase):
    def test_connected_receiver_gets_latest_team_pheromone(self):
        local_maps = torch.tensor(
            [[
                [[4, -1], [-1, -1]],
                [[-1, -1], [-1, 7]],
            ]],
            dtype=torch.long,
        )
        comms_up = torch.tensor([[True, False]])

        merged = _merge_local_timestamp_maps(local_maps, comms_up)

        torch.testing.assert_close(
            merged[0, 0],
            torch.tensor([[4, -1], [-1, 7]]),
        )
        torch.testing.assert_close(merged[0, 1], local_maps[0, 1])

    def test_disconnected_receiver_keeps_stale_pheromone_map(self):
        local_maps = torch.tensor(
            [[
                [[2, -1], [-1, -1]],
                [[-1, 5], [-1, -1]],
            ]],
            dtype=torch.long,
        )
        comms_up = torch.tensor([[False, False]])

        merged = _merge_local_timestamp_maps(local_maps, comms_up)

        torch.testing.assert_close(merged, local_maps)

    def test_survivor_events_merge_only_into_connected_agents(self):
        knowledge = torch.tensor(
            [[
                [True, False],
                [False, True],
                [False, False],
            ]],
        )
        comms_up = torch.tensor([[True, False, True]])

        merged = _merge_local_bool_knowledge(knowledge, comms_up)

        torch.testing.assert_close(merged[0, 0], torch.tensor([True, True]))
        torch.testing.assert_close(merged[0, 1], torch.tensor([False, True]))
        torch.testing.assert_close(merged[0, 2], torch.tensor([True, True]))


if __name__ == "__main__":
    unittest.main()
