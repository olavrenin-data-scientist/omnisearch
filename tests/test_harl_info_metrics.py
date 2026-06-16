import unittest
from pathlib import Path

import numpy as np

from agents.action_transform import transform_continuous_action
from agents.harl_env import WildfireHARLEnv
from agents.harl_metrics import (
    accumulate_env_metrics,
    init_env_metric_storage,
    log_done_env_metrics,
)
from agents.harl_vec_env import BatchedVMASVecEnv


ROOT = Path(__file__).resolve().parent.parent
TERRAIN_CACHE = ROOT / "data" / "terrain_cache" / "malibu_creek_state_park_california_128.npz"


class HARLInfoMetricTests(unittest.TestCase):
    def _env_args(self):
        return {
            "max_cycles": 2,
            "scenario_kwargs": {
                "max_steps": 2,
                "n_drones": 1,
                "n_ground": 1,
                "n_survivors": 1,
                "local_map_patch_size": 11,
                "fire_grid_size": 128,
                "terrain_source": "real",
                "terrain_cache_path": str(TERRAIN_CACHE),
            },
        }

    def test_tanh_action_transform_bounds_actions_smoothly(self):
        raw = np.array([[-3.0, 0.0, 3.0]], dtype=np.float32)

        transformed = transform_continuous_action(raw, "tanh")

        self.assertLess(transformed[0, 0], -0.99)
        self.assertEqual(transformed[0, 1], 0.0)
        self.assertGreater(transformed[0, 2], 0.99)
        self.assertTrue(np.all(np.abs(transformed) < 1.0))

    def test_single_harl_adapter_preserves_scenario_info(self):
        env = WildfireHARLEnv(self._env_args())
        actions = np.zeros((env.n_agents, 2), dtype=np.float32)

        _, _, _, _, infos, _ = env.step(actions)

        self.assertIn("mission/n_scouted", infos[0])
        self.assertIn("reward/ugv_progress", infos[0])
        self.assertIn("diagnostic/ugv_progress_gate_active", infos[0])
        self.assertIn("bad_transition", infos[0])

    def test_single_harl_adapter_distinguishes_natural_done_from_truncation(self):
        env_args = self._env_args()
        env_args["max_cycles"] = 5
        env_args["scenario_kwargs"]["max_steps"] = 1
        env = WildfireHARLEnv(env_args)
        actions = np.zeros((env.n_agents, 2), dtype=np.float32)

        _, _, _, dones, infos, _ = env.step(actions)

        self.assertTrue(all(dones))
        self.assertFalse(infos[0]["bad_transition"])

        env_args = self._env_args()
        env_args["max_cycles"] = 1
        env_args["scenario_kwargs"]["max_steps"] = 5
        env = WildfireHARLEnv(env_args)

        _, _, _, dones, infos, _ = env.step(actions)

        self.assertTrue(all(dones))
        self.assertTrue(infos[0]["bad_transition"])

    def test_batched_harl_adapter_preserves_scenario_info(self):
        env = BatchedVMASVecEnv(
            num_envs=2,
            seed=1,
            max_cycles=2,
            scenario_kwargs=self._env_args()["scenario_kwargs"],
        )
        obs, share_obs, _ = env.reset()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (2, env.n_agents, env.observation_space[0].shape[0]))
        self.assertEqual(obs[:, 0].shape, (2, env.observation_space[0].shape[0]))
        self.assertEqual(obs[:, 1].shape, (2, env.observation_space[1].shape[0]))
        self.assertEqual(env.observation_space[0].shape[0], env.observation_space[1].shape[0])
        self.assertEqual(share_obs.shape, (2, env.n_agents, sum(s.shape[0] for s in env.observation_space)))

        actions = np.zeros((2, env.n_agents, 2), dtype=np.float32)

        env.step_async(actions)
        _, _, _, _, infos, _ = env.step_wait()

        self.assertIn("mission/n_scouted", infos[0, 0])
        self.assertIn("reward/ugv_progress", infos[0, 0])
        self.assertIn("diagnostic/ugv_progress_gate_active", infos[0, 0])
        self.assertIn("bad_transition", infos[0, 0])

    def test_batched_harl_adapter_distinguishes_natural_done_from_truncation(self):
        scenario_kwargs = self._env_args()["scenario_kwargs"]
        scenario_kwargs["max_steps"] = 1
        env = BatchedVMASVecEnv(
            num_envs=2,
            seed=1,
            max_cycles=5,
            scenario_kwargs=scenario_kwargs,
        )
        env.reset()
        actions = np.zeros((2, env.n_agents, 2), dtype=np.float32)

        env.step_async(actions)
        _, _, _, dones, infos, _ = env.step_wait()

        self.assertTrue(np.all(dones))
        self.assertFalse(infos[0, 0]["bad_transition"])

        scenario_kwargs = self._env_args()["scenario_kwargs"]
        scenario_kwargs["max_steps"] = 5
        env = BatchedVMASVecEnv(
            num_envs=2,
            seed=1,
            max_cycles=1,
            scenario_kwargs=scenario_kwargs,
        )
        env.reset()

        env.step_async(actions)
        _, _, _, dones, infos, _ = env.step_wait()

        self.assertTrue(np.all(dones))
        self.assertTrue(infos[0, 0]["bad_transition"])

    def test_batched_harl_adapter_auto_reset_clears_vmas_step_counter(self):
        scenario_kwargs = self._env_args()["scenario_kwargs"]
        scenario_kwargs["max_steps"] = 5
        env = BatchedVMASVecEnv(
            num_envs=1,
            seed=1,
            max_cycles=2,
            scenario_kwargs=scenario_kwargs,
        )
        env.reset()
        actions = np.zeros((1, env.n_agents, 2), dtype=np.float32)

        env.step_async(actions)
        _, _, _, dones, _, _ = env.step_wait()
        self.assertFalse(bool(dones[0, 0]))

        env.step_async(actions)
        _, _, _, dones, _, _ = env.step_wait()
        self.assertTrue(bool(dones[0, 0]))

        env.step_async(actions)
        _, _, _, dones, _, _ = env.step_wait()
        self.assertFalse(bool(dones[0, 0]))

    def test_metric_logger_accumulates_and_emits_episode_values(self):
        class DummyLogger:
            algo_args = {"train": {"n_rollout_threads": 1}}

            def __init__(self):
                self.logged = {}

            def log_env(self, env_infos):
                self.logged.update(env_infos)

        logger = DummyLogger()
        init_env_metric_storage(logger)
        infos = np.array([
            [
                {
                    "mission/new_scouts": 1.0,
                    "mission/n_scouted": 3.0,
                    "mission/n_confirmed": 2.0,
                    "mission/full_success": 0.0,
                    "reward/ugv_progress": 0.5,
                    "diagnostic/ugv_progress_gate_active": 1.0,
                    "diagnostic/ugv_ground_progress_m": 3.2,
                    "diagnostic/ugv_target_index": 0.0,
                },
            ],
        ], dtype=object)
        dones = np.array([[True]])

        accumulate_env_metrics(logger, infos, dones)
        log_done_env_metrics(logger)

        self.assertEqual(logger.logged["mission/new_scouts"], [1.0])
        self.assertEqual(logger.logged["mission/n_scouted"], [3.0])
        self.assertEqual(logger.logged["mission/n_confirmed"], [2.0])
        self.assertEqual(logger.logged["reward/ugv_progress"], [0.5])
        self.assertEqual(logger.logged["diagnostic/ugv_progress_gate_active"], [1.0])
        self.assertEqual(logger.logged["diagnostic/ugv_ground_progress_m"], [3.2])
        self.assertEqual(logger.logged["diagnostic/ugv_target_index"], [0.0])


if __name__ == "__main__":
    unittest.main()
