import argparse
import unittest
from pathlib import Path

import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario
from scripts.diagnose_ugv_baseline_strategies import (
    MatchedRevealAntColonyPolicy,
    StrategySpec,
    build_scenario_kwargs,
    make_policy,
    parse_strategy_specs,
    run_rollout,
    sync_ant_colony_oracle_knowledge,
)


def _args(**overrides):
    defaults = {
        "steps": 8,
        "terrain_cache_path": None,
        "enable_fire": False,
        "disable_fire": False,
        "survivor_reveal_initial_count": 1,
        "survivor_reveal_start_step": 2,
        "survivor_reveal_end_step": 6,
        "time_bins": 2,
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


class DiagnoseUgvBaselineStrategiesTests(unittest.TestCase):
    def test_strategy_parser_accepts_aliases_all_and_duplicate_labels(self):
        specs = parse_strategy_specs(["lawnmower_astar", "ant-colony-astar"])
        self.assertEqual([spec.name for spec in specs], ["lawnmower_astar", "ant_colony_astar"])
        self.assertEqual([spec.label for spec in specs], ["lawnmower_astar", "ant_colony_astar"])

        all_specs = parse_strategy_specs(["all"])
        self.assertEqual([spec.name for spec in all_specs], ["lawnmower_astar", "ant_colony_astar"])

        duplicate = parse_strategy_specs(["lawnmower_astar", "lawnmower_astar"])
        self.assertEqual([spec.label for spec in duplicate], ["lawnmower_astar", "lawnmower_astar_2"])

        happo = parse_strategy_specs(["happo:/tmp/models"])
        self.assertEqual(happo[0].name, "happo")
        self.assertEqual(happo[0].label, "happo")
        self.assertEqual(happo[0].checkpoint_dir, Path("/tmp/models"))

        with self.assertRaises(ValueError):
            parse_strategy_specs(["not_a_strategy"])

    def test_happo_checkpoint_flag_supplies_happo_strategy(self):
        specs = parse_strategy_specs(["happo"], happo_checkpoint="/tmp/ugv/models")

        self.assertEqual(specs[0].name, "happo")
        self.assertEqual(specs[0].checkpoint_dir, Path("/tmp/ugv/models"))

    def test_build_scenario_kwargs_match_joint_schema_ugv_defaults(self):
        kwargs = build_scenario_kwargs(_args())

        self.assertEqual(kwargs["n_drones"], 0)
        self.assertEqual(kwargs["n_ground"], 2)
        self.assertEqual(kwargs["n_survivors"], 5)
        self.assertFalse(kwargs["known_survivors_at_reset"])
        self.assertTrue(kwargs["delayed_survivor_knowledge"])
        self.assertEqual(kwargs["survivor_reveal_initial_count"], 1)
        self.assertEqual(kwargs["survivor_reveal_start_step"], 2)
        self.assertEqual(kwargs["survivor_reveal_end_step"], 6)
        self.assertEqual(kwargs["ugv_planner_hint"], "global_astar")
        self.assertEqual(kwargs["ugv_dense_reward_mode"], "planner_follow")
        self.assertEqual(kwargs["ugv_target_assignment_mode"], "greedy_sticky")

    def test_same_seed_uses_same_reveal_steps_for_both_strategies(self):
        kwargs = build_scenario_kwargs(_args(
            steps=6,
            survivor_reveal_start_step=2,
            survivor_reveal_end_step=4,
        ))
        lawn = run_rollout(
            StrategySpec(label="lawnmower_astar", name="lawnmower_astar"),
            kwargs,
            seed=1000,
            time_bins=2,
        )
        ant = run_rollout(
            StrategySpec(label="ant_colony_astar", name="ant_colony_astar"),
            kwargs,
            seed=1000,
            time_bins=2,
        )

        self.assertEqual(lawn["survivor_reveal_steps"], ant["survivor_reveal_steps"])
        self.assertIn("confirmation_recall", lawn)
        self.assertIn("pending_target_time_fraction", ant)

    def test_ant_colony_adapter_receives_oracle_reveals_without_uavs(self):
        env = self._env(survivor_reveal_initial_count=0)
        try:
            scenario = env.scenario
            wrapper = MatchedRevealAntColonyPolicy(env)

            self.assertFalse(bool(wrapper.policy.known_survivors.any().item()))
            scenario.scouted_survivors[0, 0] = True
            sync_ant_colony_oracle_knowledge(wrapper.policy, scenario)

            self.assertTrue(bool(wrapper.policy.known_survivors[0, :, 0].all().item()))
            self.assertFalse(bool(wrapper.policy.known_confirmed[0, :, 0].any().item()))

            scenario.found_survivors[0, 0] = True
            sync_ant_colony_oracle_knowledge(wrapper.policy, scenario)
            self.assertTrue(bool(wrapper.policy.known_confirmed[0, :, 0].all().item()))
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    def test_baselines_hold_before_reveal_and_move_after_target_is_known(self):
        for spec in parse_strategy_specs(["lawnmower_astar", "ant_colony_astar"]):
            with self.subTest(strategy=spec.name):
                env = self._env(survivor_reveal_initial_count=0)
                try:
                    scenario = env.scenario
                    policy = make_policy(spec, env)

                    actions = policy(env)
                    self.assertTrue(all(float(action.norm().item()) == 0.0 for action in actions))

                    target_idx = self._furthest_survivor_from_any_ugv(scenario)
                    scenario.scouted_survivors[0, target_idx] = True
                    scenario.known_survivors_by_agent[0, :, target_idx] = True

                    actions = policy(env)
                    self.assertTrue(any(float(action.norm().item()) > 1e-6 for action in actions))
                finally:
                    close = getattr(env, "close", None)
                    if close is not None:
                        close()

    def _env(self, **overrides):
        kwargs = build_scenario_kwargs(_args(
            steps=8,
            survivor_reveal_start_step=3,
            survivor_reveal_end_step=5,
            **overrides,
        ))
        return vmas.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            seed=7,
            **kwargs,
        )

    @staticmethod
    def _furthest_survivor_from_any_ugv(scenario):
        ground_pos = torch.stack(
            [agent.state.pos[0] for agent in scenario.world.agents[scenario.n_drones:]],
            dim=0,
        )
        survivor_pos = torch.stack([survivor.state.pos[0] for survivor in scenario._survivors], dim=0)
        distances = torch.linalg.norm(ground_pos.unsqueeze(1) - survivor_pos.unsqueeze(0), dim=-1)
        return int(distances.min(dim=0).values.argmax().item())


if __name__ == "__main__":
    unittest.main()
