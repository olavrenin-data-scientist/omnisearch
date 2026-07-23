import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from agents.baselines import (
    AntColonyPolicy,
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

    def test_ground_dispatch_uses_persistent_communication_lease_state(self):
        scenario = SimpleNamespace(
            world=SimpleNamespace(batch_dim=1),
            n_drones=1,
            n_ground=2,
            n_agents=3,
            n_survivors=2,
            fire_grid=torch.zeros(1, 2, 2),
            _active_survivor_mask=lambda: torch.ones(1, 2, dtype=torch.bool),
        )
        policy = AntColonyPolicy.__new__(AntColonyPolicy)
        policy.scenario = scenario
        policy.known_survivors = torch.ones(1, 3, 2, dtype=torch.bool)
        policy.known_confirmed = torch.zeros_like(policy.known_survivors)
        policy.ground_target_indices = torch.tensor([[0, -1]])
        policy.ground_route_cache = [dict(), dict()]

        with patch(
            "agents.baselines._communication_aware_ground_actions",
            return_value=[torch.zeros(1, 2), torch.zeros(1, 2)],
        ) as dispatch:
            policy._ground_actions()

        self.assertIs(dispatch.call_args.args[3], policy.ground_target_indices)


if __name__ == "__main__":
    unittest.main()
