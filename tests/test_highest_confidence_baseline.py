import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from agents.baselines import (
    HighestConfidencePolicy,
    _communication_aware_ground_actions,
    _priority_nearest_ground_assignments,
)
from envs.wildfire_search import WildfireSearchScenario


class HighestConfidenceBaselineTests(unittest.TestCase):
    @staticmethod
    def _assignment_scenario():
        agents = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.0, -5.0]]))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[0.0, 0.0]]))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[9.0, 0.0]]))),
        ]
        survivors = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[10.0, 0.0]]))),
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[1.0, 0.0]]))),
        ]
        return SimpleNamespace(
            world=SimpleNamespace(agents=agents),
            n_drones=1,
            n_ground=2,
            _survivors=survivors,
        )

    def test_highest_confidence_target_gets_nearest_available_ugv(self):
        scenario = self._assignment_scenario()
        targetable = torch.tensor([[True, True]])
        confidence = torch.tensor([[0.90, 0.70]])

        assignments = _priority_nearest_ground_assignments(
            scenario,
            targetable,
            confidence,
        )

        # Survivor 0 has highest confidence and is nearest UGV 1. Survivor 1
        # is then assigned to the remaining UGV 0.
        torch.testing.assert_close(assignments, torch.tensor([[1, 0]]))

    def test_single_target_is_assigned_to_nearest_ugv(self):
        scenario = self._assignment_scenario()
        targetable = torch.tensor([[True, False]])
        confidence = torch.tensor([[0.90, 0.70]])

        assignments = _priority_nearest_ground_assignments(
            scenario,
            targetable,
            confidence,
        )

        torch.testing.assert_close(assignments, torch.tensor([[-1, 0]]))

    def test_disconnected_high_confidence_target_remains_reserved(self):
        scenario = self._assignment_scenario()
        scenario.world.batch_dim = 1
        scenario.n_agents = 3
        scenario._latest_comms_up_mask = lambda device=None: torch.tensor(
            [[True, False, True]], device=device,
        )
        targetable = torch.ones(1, 2, 2, dtype=torch.bool)
        current_targets = torch.tensor([[0, -1]])

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
            _communication_aware_ground_actions(
                scenario,
                targetable,
                [torch.zeros(1, 2), torch.zeros(1, 2)],
                current_targets,
                priority=torch.tensor([[0.9, 0.7]]),
            )

        torch.testing.assert_close(current_targets, torch.tensor([[0, 1]]))

    def test_detector_confidence_is_shared_only_with_connected_ugvs(self):
        agents = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.zeros(1, 2)))
            for _ in range(3)
        ]
        scenario = SimpleNamespace(
            world=SimpleNamespace(batch_dim=1, agents=agents),
            n_drones=1,
            n_ground=2,
            n_agents=3,
            n_survivors=2,
            fire_grid=torch.zeros(1, 2, 2),
            step_count=torch.zeros(1, dtype=torch.long),
            step_drone_detection_confidence=torch.tensor([[[0.8, 0.0]]]),
            known_survivors_by_agent=torch.ones(1, 3, 2, dtype=torch.bool),
            confirmed_survivors_by_agent=torch.zeros(1, 3, 2, dtype=torch.bool),
            _active_survivor_mask=lambda: torch.ones(1, 2, dtype=torch.bool),
            _latest_comms_up_mask=lambda device=None: torch.tensor(
                [[True, False, True]], device=device,
            ),
        )
        env = SimpleNamespace(
            scenario=scenario,
            get_random_actions=lambda: [torch.zeros(1, 2) for _ in range(3)],
        )
        policy = HighestConfidencePolicy(env)
        policy.survivor_confidence_by_agent[0, 1, 1] = 0.95

        with (
            patch.object(
                policy._lawnmower,
                "_drone_lawnmower_actions",
                return_value=[torch.zeros(1, 2)],
            ),
            patch(
                "agents.baselines._communication_aware_ground_actions",
                return_value=[torch.zeros(1, 2), torch.zeros(1, 2)],
            ) as ground_actions,
        ):
            policy(env)

        self.assertAlmostEqual(float(policy.survivor_confidence_by_agent[0, 0, 0]), 0.8)
        self.assertAlmostEqual(float(policy.survivor_confidence_by_agent[0, 1, 0]), 0.0)
        self.assertAlmostEqual(float(policy.survivor_confidence_by_agent[0, 2, 0]), 0.8)
        self.assertAlmostEqual(float(policy.survivor_confidence_by_agent[0, 1, 1]), 0.95)
        torch.testing.assert_close(
            ground_actions.call_args.kwargs["priority"],
            torch.tensor([[0.8, 0.0]]),
        )

    def test_abstract_detector_retains_score_only_for_successful_detection(self):
        probability = torch.tensor([[[0.80, 0.30]]])
        scenario = SimpleNamespace(
            detection_backend="abstract",
            _drone_detection_components=lambda *_args: {"probability": probability},
        )
        random_draw = torch.tensor([[[0.20, 0.50]]])

        with patch("envs.wildfire_search.torch.rand_like", return_value=random_draw):
            detected = WildfireSearchScenario._drone_survivor_detections(
                scenario,
                torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 2),
                torch.zeros(1, 2, 2),
            )

        torch.testing.assert_close(detected, torch.tensor([[[True, False]]]))
        torch.testing.assert_close(
            scenario.step_drone_detection_confidence,
            torch.tensor([[[0.80, 0.00]]]),
        )


if __name__ == "__main__":
    unittest.main()
