import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from agents.baselines import LawnmowerPolicy, _communication_aware_ground_actions


class LawnmowerCommunicationTests(unittest.TestCase):
    @staticmethod
    def _scenario(comms_up: torch.Tensor):
        uav = SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.0, -1.0]])))
        ugv0 = SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.0, 0.0]])))
        ugv1 = SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.1, 0.0]])))
        survivors = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.2, 0.0]]))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.9, 0.0]]))),
        ]
        return SimpleNamespace(
            world=SimpleNamespace(batch_dim=1, agents=[uav, ugv0, ugv1]),
            n_drones=1,
            n_ground=2,
            n_agents=3,
            _survivors=survivors,
            _latest_comms_up_mask=lambda device=None: comms_up.to(device=device),
        )

    @staticmethod
    def _actions(scenario, targetable, current_targets):
        fallback = [torch.zeros(1, 2), torch.zeros(1, 2)]
        with (
            patch(
                "agents.baselines._route_ground_waypoints",
                side_effect=lambda _sc, _gi, target, _idx, _cache: target,
            ),
            patch(
                "agents.baselines._terrain_safe_ground_action",
                side_effect=lambda _sc, _gi, target, _fallback: target,
            ),
        ):
            return _communication_aware_ground_actions(
                scenario,
                targetable,
                fallback,
                current_targets,
            )

    def test_connected_ugvs_split_locally_known_targets(self):
        scenario = self._scenario(torch.tensor([[True, True, True]]))
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        current_targets = torch.full((1, 2), -1, dtype=torch.long)

        self._actions(scenario, targetable, current_targets)

        torch.testing.assert_close(current_targets, torch.tensor([[0, 1]]))

    def test_disconnected_ugv_target_is_reserved_for_connected_team(self):
        scenario = self._scenario(torch.tensor([[True, False, True]]))
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        current_targets = torch.tensor([[0, -1]])

        self._actions(scenario, targetable, current_targets)

        torch.testing.assert_close(current_targets, torch.tensor([[0, 1]]))

    def test_disconnected_ugv_keeps_valid_target_instead_of_switching(self):
        scenario = self._scenario(torch.tensor([[True, False, True]]))
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        current_targets = torch.tensor([[1, -1]])

        self._actions(scenario, targetable, current_targets)

        self.assertEqual(int(current_targets[0, 0]), 1)
        self.assertEqual(int(current_targets[0, 1]), 0)

    def test_reconnected_ugv_rejoins_distinct_assignment(self):
        scenario = self._scenario(torch.tensor([[True, False, True]]))
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        current_targets = torch.tensor([[0, -1]])
        self._actions(scenario, targetable, current_targets)
        torch.testing.assert_close(current_targets, torch.tensor([[0, 1]]))

        scenario._latest_comms_up_mask = lambda device=None: torch.tensor(
            [[True, True, True]], device=device,
        )
        self._actions(scenario, targetable, current_targets)

        torch.testing.assert_close(current_targets, torch.tensor([[0, 1]]))

    def test_disconnected_unassigned_ugv_receives_no_new_target(self):
        scenario = self._scenario(torch.tensor([[True, False, True]]))
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        current_targets = torch.full((1, 2), -1, dtype=torch.long)

        self._actions(scenario, targetable, current_targets)

        torch.testing.assert_close(current_targets, torch.tensor([[-1, 0]]))

    def test_private_confirmation_stops_motion_without_releasing_lease(self):
        scenario = self._scenario(torch.tensor([[True, False, True]]))
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        targetable[0, 0, 0] = False
        current_targets = torch.tensor([[0, -1]])

        actions = self._actions(scenario, targetable, current_targets)

        torch.testing.assert_close(current_targets, torch.tensor([[0, 1]]))
        torch.testing.assert_close(actions[0], torch.zeros_like(actions[0]))

    def test_lawnmower_does_not_use_global_scout_mask_for_ugv_targets(self):
        agents = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.zeros(1, 2))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.zeros(1, 2))),
        ]
        scenario = SimpleNamespace(
            world=SimpleNamespace(batch_dim=1, agents=agents),
            n_drones=1,
            n_ground=1,
            n_agents=2,
            n_survivors=1,
            fire_grid=torch.zeros(1, 2, 2),
            step_count=torch.zeros(1, dtype=torch.long),
            scouted_survivors=torch.ones(1, 1, dtype=torch.bool),
            found_survivors=torch.zeros(1, 1, dtype=torch.bool),
            known_survivors_by_agent=torch.zeros(1, 2, 1, dtype=torch.bool),
            confirmed_survivors_by_agent=torch.zeros(1, 2, 1, dtype=torch.bool),
            _active_survivor_mask=lambda: torch.ones(1, 1, dtype=torch.bool),
        )
        policy = LawnmowerPolicy(SimpleNamespace(scenario=scenario))

        with (
            patch.object(
                policy,
                "_drone_lawnmower_actions",
                return_value=[torch.zeros(1, 2)],
            ),
            patch(
                "agents.baselines._communication_aware_ground_actions",
                return_value=[torch.zeros(1, 2)],
            ) as coordinated,
        ):
            policy(SimpleNamespace(scenario=scenario))

        targetable = coordinated.call_args.args[1]
        self.assertFalse(bool(targetable.any()))

        scenario.known_survivors_by_agent[0, 1, 0] = True
        with (
            patch.object(
                policy,
                "_drone_lawnmower_actions",
                return_value=[torch.zeros(1, 2)],
            ),
            patch(
                "agents.baselines._communication_aware_ground_actions",
                return_value=[torch.zeros(1, 2)],
            ) as coordinated,
        ):
            policy(SimpleNamespace(scenario=scenario))

        targetable = coordinated.call_args.args[1]
        self.assertTrue(bool(targetable[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
