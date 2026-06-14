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
from agents.happo_policy import _scenario_kwargs_from_manifest
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
            ugv_diagnostic_target_distance_m=80.0,
            local_map_patch_size=11,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["n_drones"], 0)
        self.assertEqual(scenario["n_ground"], 1)
        self.assertEqual(scenario["n_survivors"], 1)
        self.assertEqual(scenario["local_map_patch_size"], 11)
        self.assertTrue(scenario["known_survivors_at_reset"])
        self.assertEqual(scenario["known_survivor_spawn_distance_m"], 80.0)
        self.assertTrue(scenario["disable_fire"])
        self.assertEqual(scenario["comms_dropout"], 0.0)
        self.assertEqual(scenario["r_drone_scout"], 0.0)
        self.assertEqual(scenario["r_drone_shaping"], 0.0)
        self.assertEqual(scenario["r_coverage"], 0.0)
        self.assertEqual(scenario["r_fire_penalty"], 0.0)
        self.assertEqual(scenario["r_ground_travel_cost"], 0.0)
        self.assertEqual(scenario["r_ground_shaping"], 0.50)
        self.assertEqual(scenario["ground_confirm_min_m"], 20.0)


if __name__ == "__main__":
    unittest.main()
