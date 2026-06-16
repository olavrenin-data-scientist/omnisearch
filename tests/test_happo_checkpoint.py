import json
import tempfile
import types
import unittest
from pathlib import Path

from agents.happo_checkpoint import (
    MANIFEST_FILENAME,
    load_training_manifest,
    merge_training_scenario,
    save_training_manifest,
)
from agents.happo_policy import _action_transform_from_manifest, _scenario_kwargs_from_manifest
from scripts.train_happo_smoke import build_args


class HappoCheckpointTests(unittest.TestCase):
    def test_manifest_round_trip_beside_models_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True)
            runner = types.SimpleNamespace(save_dir=models_dir)
            env_args = {
                "scenario_kwargs": {
                    "drone_min_footprint_m": 75.0,
                    "ground_confirm_min_m": 10.0,
                },
            }

            path = save_training_manifest(
                runner,
                harl_args={"algo": "happo"},
                algo_args={"model": {"hidden_sizes": [128, 128]}},
                env_args=env_args,
            )

            self.assertEqual(path, run_dir / MANIFEST_FILENAME)
            self.assertEqual(load_training_manifest(models_dir)["env_args"], env_args)

    def test_missing_manifest_is_supported_for_legacy_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_training_manifest(Path(tmp) / "models"))

    def test_rejects_unknown_manifest_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            models_dir = run_dir / "models"
            models_dir.mkdir()
            (run_dir / MANIFEST_FILENAME).write_text(
                json.dumps({"version": 999}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_training_manifest(models_dir)

    def test_training_scenario_is_restored_with_evaluation_controls(self):
        export_scenario = {
            "terrain_cache_path": "export.npz",
            "drone_min_footprint": 0.0,
            "max_steps": 100,
            "comms_dropout": 0.0,
        }
        manifest = {
            "env_args": {
                "scenario_kwargs": {
                    "terrain_cache_path": "training.npz",
                    "drone_min_footprint": 0.15,
                    "ground_confirm_min": 0.12,
                    "max_steps": 500,
                    "comms_dropout": 0.1,
                },
            },
        }

        merged = merge_training_scenario(
            export_scenario,
            manifest,
            max_steps=1_000,
            comms_dropout=0.8,
        )

        self.assertEqual(merged["terrain_cache_path"], "training.npz")
        self.assertEqual(merged["drone_min_footprint"], 0.15)
        self.assertEqual(merged["ground_confirm_min"], 0.12)
        self.assertEqual(merged["max_steps"], 1_000)
        self.assertEqual(merged["comms_dropout"], 0.8)

    def test_policy_loader_extracts_manifest_scenario_kwargs(self):
        manifest = {
            "env_args": {
                "scenario_kwargs": {
                    "n_drones": 0,
                    "n_ground": 1,
                    "n_survivors": 1,
                    "known_survivors_at_reset": True,
                },
            },
        }

        scenario_kwargs = _scenario_kwargs_from_manifest(manifest)

        self.assertEqual(scenario_kwargs["n_drones"], 0)
        self.assertEqual(scenario_kwargs["n_ground"], 1)
        self.assertEqual(scenario_kwargs["n_survivors"], 1)
        self.assertTrue(scenario_kwargs["known_survivors_at_reset"])

    def test_policy_loader_extracts_manifest_action_transform(self):
        manifest = {
            "env_args": {
                "action_transform": "tanh",
                "scenario_kwargs": {},
            },
        }

        self.assertEqual(_action_transform_from_manifest(manifest), "tanh")
        self.assertEqual(_action_transform_from_manifest(None), "clip")

    def test_default_and_reward_search_use_same_reward_profile(self):
        common_kwargs = dict(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
        )
        _, _, default_env_args = build_args(exp_name="default", **common_kwargs)
        _, _, search_env_args = build_args(exp_name="search", reward_search=True, **common_kwargs)

        expected = {
            "r_found_survivor": 10.0,
            "r_drone_scout": 2.0,
            "r_ground_confirm": 4.0,
            "r_drone_shaping": 0.30,
            "r_ground_shaping": 0.50,
            "r_ground_approach": 0.05,
            "ground_approach_milestone_radii_m": (75.0, 50.0, 40.0, 30.0, 20.0),
            "r_ugv_movement_alignment": 0.20,
            "r_ugv_stall_penalty": 0.0,
            "r_fire_penalty": -0.20,
            "r_ground_travel_cost": -0.01,
            "r_drone_climb_cost": -0.005,
            "r_time_penalty": -0.0005,
            "r_coverage": 5.0,
        }
        for scenario in (
            default_env_args["scenario_kwargs"],
            search_env_args["scenario_kwargs"],
        ):
            for key, value in expected.items():
                self.assertEqual(scenario[key], value)

    def test_build_args_exposes_learning_rate_schedule(self):
        _, algo_args, _ = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="lr",
            lr=2.5e-4,
            critic_lr=5e-4,
            linear_lr_decay=True,
        )

        self.assertEqual(algo_args["model"]["lr"], 2.5e-4)
        self.assertEqual(algo_args["model"]["critic_lr"], 5e-4)
        self.assertTrue(algo_args["train"]["use_linear_lr_decay"])

    def test_ugv_known_survivor_diagnostic_build_args(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="diag",
            terrain_cache_path="terrain.npz",
            ground_confirm_min_m=20.0,
            ugv_known_survivor_diagnostic=True,
            ugv_diagnostic_target_distance_min_m=30.0,
            ugv_diagnostic_target_distance_max_m=100.0,
            local_map_patch_size=11,
            slope_speed_weight=0.5,
            land_cover_speeds=(1.0, 0.95, 0.8, 0.7, 0.0, 0.0),
            action_transform="tanh",
        )

        self.assertEqual(env_args["action_transform"], "tanh")
        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["n_drones"], 0)
        self.assertEqual(scenario["n_ground"], 1)
        self.assertEqual(scenario["n_survivors"], 1)
        self.assertEqual(scenario["local_map_patch_size"], 11)
        self.assertTrue(scenario["known_survivors_at_reset"])
        self.assertEqual(scenario["known_survivor_spawn_distance_m"], 65.0)
        self.assertEqual(scenario["known_survivor_spawn_distance_min_m"], 30.0)
        self.assertEqual(scenario["known_survivor_spawn_distance_max_m"], 100.0)
        self.assertTrue(scenario["disable_fire"])
        self.assertEqual(scenario["comms_dropout"], 0.0)
        self.assertEqual(scenario["r_drone_scout"], 0.0)
        self.assertEqual(scenario["r_drone_shaping"], 0.0)
        self.assertEqual(scenario["r_coverage"], 0.0)
        self.assertEqual(scenario["r_fire_penalty"], 0.0)
        self.assertEqual(scenario["r_ground_travel_cost"], 0.0)
        self.assertEqual(scenario["r_ground_shaping"], 0.50)
        self.assertEqual(scenario["r_ground_approach"], 0.05)
        self.assertEqual(scenario["ground_approach_milestone_radii_m"], (75.0, 50.0, 40.0, 30.0, 20.0))
        self.assertEqual(scenario["r_ugv_movement_alignment"], 0.20)
        self.assertEqual(scenario["r_ugv_stall_penalty"], 0.0)
        self.assertEqual(scenario["ugv_stall_displacement_threshold_m"], 0.05)
        self.assertEqual(scenario["ground_confirm_min_m"], 20.0)
        self.assertEqual(scenario["slope_speed_weight"], 0.5)
        self.assertEqual(scenario["land_cover_speeds"], (1.0, 0.95, 0.8, 0.7, 0.0, 0.0))

    def test_ugv_known_survivor_diagnostic_uses_normal_placement_by_default(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="diag",
            ugv_known_survivor_diagnostic=True,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertTrue(scenario["known_survivors_at_reset"])
        self.assertNotIn("known_survivor_spawn_distance_m", scenario)
        self.assertNotIn("known_survivor_spawn_distance_min_m", scenario)
        self.assertNotIn("known_survivor_spawn_distance_max_m", scenario)

    def test_ugv_known_survivor_exact_distance_uses_min_equals_max(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="diag",
            ugv_known_survivor_diagnostic=True,
            ugv_diagnostic_target_distance_min_m=80.0,
            ugv_diagnostic_target_distance_max_m=80.0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["known_survivor_spawn_distance_m"], 80.0)
        self.assertEqual(scenario["known_survivor_spawn_distance_min_m"], 80.0)
        self.assertEqual(scenario["known_survivor_spawn_distance_max_m"], 80.0)

    def test_ugv_known_survivor_min_distance_can_omit_max(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="diag",
            ugv_known_survivor_diagnostic=True,
            ugv_diagnostic_target_distance_min_m=80.0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["known_survivor_spawn_distance_min_m"], 80.0)
        self.assertNotIn("known_survivor_spawn_distance_m", scenario)
        self.assertNotIn("known_survivor_spawn_distance_max_m", scenario)


if __name__ == "__main__":
    unittest.main()
