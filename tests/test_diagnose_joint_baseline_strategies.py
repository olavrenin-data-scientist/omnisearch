import argparse
import unittest
from pathlib import Path

import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario
from scripts.diagnose_joint_baseline_strategies import (
    StrategySpec,
    build_scenario_kwargs,
    make_policy,
    parse_strategy_specs,
    run_rollout,
)


def _args(**overrides):
    defaults = {
        "steps": 6,
        "time_bins": 2,
        "joint_diagnostic_ugvs": 2,
        "terrain_cache_path": None,
        "enable_fire": False,
        "disable_fire": False,
        "ugv_target_assignment_mode": None,
        "ugv_planner_fire_mode": None,
        "ugv_planner_fire_replan_policy": None,
        "ugv_planner_fire_replan_interval_steps": None,
        "ugv_planner_fire_cost": None,
        "ugv_planner_fire_block_threshold": None,
        "ugv_planner_smoke_cost": None,
        "ugv_planner_smolder_cost": None,
        "ugv_planner_fire_buffer_m": None,
        "ugv_planner_fire_buffer_cost": None,
        "ugv_planner_land_cover_costs": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class DiagnoseJointBaselineStrategiesTests(unittest.TestCase):
    def test_strategy_parser_accepts_aliases_all_and_happo(self):
        specs = parse_strategy_specs(["lawnmower", "ant-colony-astar"])
        self.assertEqual([spec.name for spec in specs], ["lawnmower_astar", "ant_colony_astar"])
        self.assertEqual([spec.label for spec in specs], ["lawnmower_astar", "ant_colony_astar"])

        all_specs = parse_strategy_specs(["all"])
        self.assertEqual([spec.name for spec in all_specs], ["lawnmower_astar", "ant_colony_astar"])

        duplicate = parse_strategy_specs(["lawnmower_astar", "lawnmower_astar"])
        self.assertEqual([spec.label for spec in duplicate], ["lawnmower_astar", "lawnmower_astar_2"])

        happo = parse_strategy_specs(["happo:/tmp/models"])
        self.assertEqual(happo[0].name, "happo")
        self.assertEqual(happo[0].checkpoint_dir, Path("/tmp/models"))

        with self.assertRaises(ValueError):
            parse_strategy_specs(["not_a_strategy"])

    def test_happo_checkpoint_flag_supplies_happo_strategy(self):
        specs = parse_strategy_specs(["happo"], happo_checkpoint="/tmp/joint/models")

        self.assertEqual(specs[0].name, "happo")
        self.assertEqual(specs[0].checkpoint_dir, Path("/tmp/joint/models"))

    def test_build_scenario_kwargs_match_joint_defaults(self):
        kwargs = build_scenario_kwargs(_args())

        self.assertEqual(kwargs["n_drones"], 3)
        self.assertEqual(kwargs["n_ground"], 2)
        self.assertEqual(kwargs["n_survivors"], 5)
        self.assertFalse(kwargs["known_survivors_at_reset"])
        self.assertFalse(kwargs["delayed_survivor_knowledge"])
        self.assertFalse(kwargs["drone_can_confirm"])
        self.assertEqual(kwargs["ugv_planner_hint"], "global_astar")
        self.assertEqual(kwargs["ugv_dense_reward_mode"], "planner_follow")
        self.assertEqual(kwargs["ugv_target_assignment_mode"], "greedy_sticky")
        self.assertTrue(kwargs["uav_confidence_diagnostics"])

    def test_rollout_smoke_contains_joint_metrics(self):
        kwargs = build_scenario_kwargs(_args(steps=3))
        row = run_rollout(
            StrategySpec(label="lawnmower_astar", name="lawnmower_astar"),
            kwargs,
            seed=1000,
            time_bins=2,
        )

        self.assertEqual(row["survivors"], 5)
        self.assertIn("scout_recall", row)
        self.assertIn("confirm_recall", row)
        self.assertIn("final_coverage_fraction", row)
        self.assertIn("final_confidence_mean", row)
        self.assertEqual(len(row["uav_path_length_by_agent_m"]), 3)
        self.assertEqual(len(row["ugv_path_length_by_agent_m"]), 2)
        self.assertEqual(len(row["time_bins"]), 2)

    def test_baseline_policies_produce_drone_actions_and_hold_ugvs_before_scouts(self):
        for spec in parse_strategy_specs(["lawnmower_astar", "ant_colony_astar"]):
            with self.subTest(strategy=spec.name):
                env = self._env()
                try:
                    scenario = env.scenario
                    policy = make_policy(spec, env)
                    actions = policy(env)

                    self.assertEqual(len(actions), scenario.n_agents)
                    self.assertTrue(any(float(action.norm().item()) > 1e-6 for action in actions[:scenario.n_drones]))
                    self.assertTrue(all(float(action.norm().item()) == 0.0 for action in actions[scenario.n_drones:]))
                finally:
                    close = getattr(env, "close", None)
                    if close is not None:
                        close()

    def _env(self):
        kwargs = build_scenario_kwargs(_args(steps=4))
        env = WildfireSearchScenario.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            seed=7,
            **kwargs,
        )
        env.reset()
        return env


if __name__ == "__main__":
    unittest.main()
