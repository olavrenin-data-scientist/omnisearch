import unittest

from agents.baselines import BASELINES
from scripts.export_trajectories import (
    DEFAULT_EXPORT_DRONES,
    DEFAULT_EXPORT_GRID_SIZE,
    DEFAULT_EXPORT_STEPS,
    DEFAULT_EXPORT_SURVIVORS,
    DEFAULT_EXPORT_UGVS,
    DEFAULT_HAPPO_CHECKPOINT,
    DEFAULT_TERRAIN_CACHE_PATH,
    ROOT,
    _reference_scenario_defaults,
    _selected_baselines,
)


class ExportTrajectorySelectionTests(unittest.TestCase):
    def test_all_selects_every_baseline(self):
        self.assertEqual(_selected_baselines("all"), list(BASELINES))

    def test_happo_selects_no_baselines(self):
        self.assertEqual(_selected_baselines("happo"), [])

    def test_named_baseline_selects_only_that_baseline(self):
        baseline = next(iter(BASELINES))
        self.assertEqual(_selected_baselines(baseline), [baseline])

    def test_reference_checkpoint_and_export_dimensions(self):
        self.assertEqual(
            DEFAULT_HAPPO_CHECKPOINT.relative_to(ROOT).as_posix(),
            "results/harl_runs/wildfire/wildfire_search/happo/"
            "happo_uav4_ugv3_area_1sqkm_malibu_grid256_steps900_fire_survivors_10/"
            "seed-00001-2026-07-19-00-15-35/models",
        )
        self.assertEqual(DEFAULT_EXPORT_STEPS, 900)
        self.assertEqual(DEFAULT_EXPORT_GRID_SIZE, 256)
        self.assertEqual(DEFAULT_EXPORT_DRONES, 4)
        self.assertEqual(DEFAULT_EXPORT_UGVS, 3)
        self.assertEqual(DEFAULT_EXPORT_SURVIVORS, 10)

    def test_reference_scenario_matches_joint_training_profile(self):
        scenario = _reference_scenario_defaults()

        self.assertEqual(scenario["n_drones"], 4)
        self.assertEqual(scenario["n_ground"], 3)
        self.assertEqual(scenario["n_survivors"], 10)
        self.assertEqual(scenario["active_survivors_min"], 10)
        self.assertEqual(scenario["active_survivors_max"], 10)
        self.assertFalse(scenario["disable_fire"])
        self.assertEqual(scenario["fire_grid_size"], 256)
        self.assertEqual(
            scenario["terrain_cache_path"],
            str(DEFAULT_TERRAIN_CACHE_PATH),
        )
        self.assertEqual(scenario["drone_perception_mode"], "rgb_thermal")
        self.assertEqual(scenario["drone_flight_levels_m"], (30.0, 50.0, 75.0))
        self.assertEqual(scenario["uav_confidence_obs_grid"], 32)
        self.assertEqual(scenario["local_confidence_obs_grid"], 9)
        self.assertEqual(scenario["uav_frontier_mode"], "local_global")
        self.assertEqual(scenario["ugv_planner_hint"], "global_astar")
        self.assertTrue(scenario["survivor_assignment_obs"])


if __name__ == "__main__":
    unittest.main()
