import unittest

import torch

from agents.baselines import (
    _merge_local_bool_knowledge,
    _merge_local_timestamp_maps,
)


class AntColonyMemoryTests(unittest.TestCase):
    def test_disconnected_sender_does_not_publish_pheromone(self):
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
            local_maps[0, 0],
        )
        torch.testing.assert_close(merged[0, 1], local_maps[0, 1])

    def test_connected_agents_merge_pheromone_after_reconnection(self):
        local_maps = torch.tensor(
            [[
                [[4, -1], [-1, -1]],
                [[-1, -1], [-1, 7]],
            ]],
            dtype=torch.long,
        )
        comms_up = torch.tensor([[True, True]])

        merged = _merge_local_timestamp_maps(local_maps, comms_up)

        expected = torch.tensor([[4, -1], [-1, 7]])
        torch.testing.assert_close(merged[0, 0], expected)
        torch.testing.assert_close(merged[0, 1], expected)

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

    def test_disconnected_sender_does_not_publish_survivor_events(self):
        knowledge = torch.tensor(
            [[
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]],
        )
        comms_up = torch.tensor([[True, False, True]])

        merged = _merge_local_bool_knowledge(knowledge, comms_up)

        connected_expected = torch.tensor([True, False, True])
        torch.testing.assert_close(merged[0, 0], connected_expected)
        torch.testing.assert_close(merged[0, 1], knowledge[0, 1])
        torch.testing.assert_close(merged[0, 2], connected_expected)

    def test_survivor_events_sync_after_reconnection(self):
        knowledge = torch.tensor(
            [[
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]],
        )
        comms_up = torch.tensor([[True, True, True]])

        merged = _merge_local_bool_knowledge(knowledge, comms_up)

        expected = torch.tensor([True, True, True]).expand_as(merged[0])
        torch.testing.assert_close(merged[0], expected)


if __name__ == "__main__":
    unittest.main()
