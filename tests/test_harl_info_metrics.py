import unittest
from pathlib import Path

import numpy as np

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
                "fire_grid_size": 128,
                "terrain_source": "real",
                "terrain_cache_path": str(TERRAIN_CACHE),
            },
        }

    def test_single_harl_adapter_preserves_scenario_info(self):
        env = WildfireHARLEnv(self._env_args())
        actions = np.zeros((env.n_agents, 2), dtype=np.float32)

        _, _, _, _, infos, _ = env.step(actions)

        self.assertIn("mission/n_scouted", infos[0])
        self.assertIn("reward/ugv_progress", infos[0])
        self.assertIn("bad_transition", infos[0])

    def test_batched_harl_adapter_preserves_scenario_info(self):
        env = BatchedVMASVecEnv(
            num_envs=2,
            seed=1,
            max_cycles=2,
            scenario_kwargs=self._env_args()["scenario_kwargs"],
        )
        actions = np.zeros((2, env.n_agents, 2), dtype=np.float32)

        env.step_async(actions)
        _, _, _, _, infos, _ = env.step_wait()

        self.assertIn("mission/n_scouted", infos[0, 0])
        self.assertIn("reward/ugv_progress", infos[0, 0])
        self.assertIn("bad_transition", infos[0, 0])

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


if __name__ == "__main__":
    unittest.main()
