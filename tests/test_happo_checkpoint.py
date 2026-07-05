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
from scripts.diagnose_uav_happo import (
    _scenario_kwargs as diagnose_uav_scenario_kwargs,
    _summarize_per_drone,
)
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
            "r_ugv_planner_progress": 0.0,
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

    def test_build_args_exposes_global_parameter_sharing(self):
        _, algo_args, _ = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="share",
            share_param=True,
        )

        self.assertTrue(algo_args["algo"]["share_param"])

    def test_build_args_exposes_terrain_cnn_encoder(self):
        _, algo_args, _ = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="cnn",
            ugv_known_survivor_diagnostic=True,
            local_map_patch_size=7,
            terrain_cnn_encoder=True,
            terrain_cnn_embed_dim=12,
        )

        model = algo_args["model"]
        self.assertTrue(model["use_terrain_cnn_encoder"])
        self.assertEqual(model["terrain_cnn_patch_size"], 7)
        self.assertEqual(model["terrain_cnn_embed_dim"], 12)
        self.assertEqual(model["terrain_cnn_single_obs_dim"], 4 + 12 + 1 + 2 * 7 * 7 + 9 + 2 + 4 + 7 + 5)

    def test_build_args_exposes_ugv_planner_hint(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="planner",
            ugv_known_survivor_diagnostic=True,
            local_map_patch_size=7,
            ugv_planner_hint="local-astar",
            ugv_planner_detour_obs=True,
            ugv_planner_patch_size=11,
            ugv_planner_lookahead_cells=6,
            ugv_planner_progress_reward=0.05,
            ugv_route_aware_reward=True,
            ugv_dense_reward_mode="target",
            ugv_planner_blend_weight=0.5,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["ugv_planner_hint"], "local_astar")
        self.assertTrue(scenario["ugv_planner_detour_obs"])
        self.assertEqual(scenario["ugv_planner_patch_size"], 11)
        self.assertEqual(scenario["ugv_planner_lookahead_cells"], 5)
        self.assertEqual(scenario["r_ugv_planner_progress"], 0.05)
        self.assertTrue(scenario["ugv_route_aware_reward"])

    def test_build_args_accepts_escape_ugv_planner_hint(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="escape-planner",
            ugv_known_survivor_diagnostic=True,
            local_map_patch_size=7,
            ugv_planner_hint="local-escape-astar",
            ugv_planner_detour_obs=True,
            ugv_planner_patch_size=11,
            ugv_planner_lookahead_cells=6,
            ugv_planner_progress_reward=0.05,
            ugv_planner_blend_weight=0.5,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["ugv_planner_hint"], "local_escape_astar")
        self.assertTrue(scenario["ugv_planner_detour_obs"])
        self.assertEqual(scenario["ugv_planner_lookahead_cells"], 5)
        self.assertEqual(scenario["ugv_dense_reward_mode"], "target")
        self.assertEqual(scenario["ugv_planner_blend_weight"], 0.5)
        self.assertEqual(algo_args["model"]["terrain_cnn_single_obs_dim"], 4 + 12 + 1 + 2 * 7 * 7 + 9 + 6 + 2 + 4 + 7)

    def test_build_args_accepts_escape_blend_dense_reward_mode(self):
        _, _algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="escape-blend",
            ugv_known_survivor_diagnostic=True,
            local_map_patch_size=7,
            ugv_planner_hint="local-escape-astar",
            ugv_dense_reward_mode="escape-blend",
            ugv_planner_blend_weight=0.5,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["ugv_planner_hint"], "local_escape_astar")
        self.assertEqual(scenario["ugv_dense_reward_mode"], "escape_blend")
        self.assertEqual(scenario["ugv_planner_blend_weight"], 0.5)

    def test_build_args_accepts_escape_route_switch_dense_reward_mode(self):
        _, _algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="escape-route-switch",
            ugv_known_survivor_diagnostic=True,
            local_map_patch_size=7,
            ugv_planner_hint="local-astar",
            ugv_dense_reward_mode="escape-route-switch",
            ugv_escape_stall_steps=3,
            ugv_escape_progress_threshold_m=0.2,
            ugv_escape_movement_threshold_m=0.4,
            ugv_escape_waypoint_reached_m=5.0,
            ugv_escape_max_steps=12,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["ugv_planner_hint"], "local_astar")
        self.assertEqual(scenario["ugv_dense_reward_mode"], "escape_route_switch")
        self.assertEqual(scenario["ugv_escape_stall_steps"], 3)
        self.assertEqual(scenario["ugv_escape_progress_threshold_m"], 0.2)
        self.assertEqual(scenario["ugv_escape_movement_threshold_m"], 0.4)
        self.assertEqual(scenario["ugv_escape_waypoint_reached_m"], 5.0)
        self.assertEqual(scenario["ugv_escape_max_steps"], 12)

    def test_build_args_accepts_global_astar_planner_follow(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="global-planner",
            ugv_known_survivor_diagnostic=True,
            local_map_patch_size=7,
            ugv_planner_hint="global-astar",
            ugv_dense_reward_mode="planner-follow",
            ugv_global_planner_lookahead_m=25.0,
            ugv_global_planner_heuristic="terrain",
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["ugv_planner_hint"], "global_astar")
        self.assertEqual(scenario["ugv_dense_reward_mode"], "planner_follow")
        self.assertEqual(scenario["ugv_global_planner_lookahead_m"], 25.0)
        self.assertEqual(scenario["ugv_global_planner_heuristic"], "terrain")
        self.assertEqual(algo_args["model"]["terrain_cnn_single_obs_dim"], 4 + 12 + 1 + 2 * 7 * 7 + 9 + 5 + 2 + 4 + 7)

    def test_build_args_exposes_fire_aware_ugv_planner_settings(self):
        _, _algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.0,
            entropy_coef=0.01,
            exp_name="fire-planner",
            ugv_known_survivor_diagnostic=True,
            ugv_planner_hint="global-astar",
            ugv_dense_reward_mode="planner-follow",
            ugv_planner_fire_mode="block",
            ugv_planner_fire_replan_policy="lazy",
            ugv_planner_fire_replan_interval_steps=17,
            ugv_planner_fire_cost=30.0,
            ugv_planner_fire_block_threshold=0.6,
            ugv_planner_smoke_cost=6.0,
            ugv_planner_smolder_cost=4.0,
            ugv_planner_fire_buffer_m=12.0,
            ugv_planner_fire_buffer_cost=9.0,
            ugv_planner_land_cover_costs=(0.85, 1.0, 1.15, 1.35, 4.0, 8.0),
            enable_fire=True,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["ugv_planner_fire_mode"], "block")
        self.assertEqual(scenario["ugv_planner_fire_replan_policy"], "lazy")
        self.assertEqual(scenario["ugv_planner_fire_replan_interval_steps"], 17)
        self.assertEqual(scenario["ugv_planner_fire_cost"], 30.0)
        self.assertEqual(scenario["ugv_planner_fire_block_threshold"], 0.6)
        self.assertEqual(scenario["ugv_planner_smoke_cost"], 6.0)
        self.assertEqual(scenario["ugv_planner_smolder_cost"], 4.0)
        self.assertEqual(scenario["ugv_planner_fire_buffer_m"], 12.0)
        self.assertEqual(scenario["ugv_planner_fire_buffer_cost"], 9.0)
        self.assertEqual(
            scenario["ugv_planner_land_cover_costs"],
            (0.85, 1.0, 1.15, 1.35, 4.0, 8.0),
        )
        self.assertFalse(scenario["disable_fire"])

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
            enable_fire=False,
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
        self.assertEqual(scenario["r_ugv_planner_progress"], 0.0)
        self.assertEqual(scenario["r_ugv_stall_penalty"], 0.0)
        self.assertEqual(scenario["ugv_stall_displacement_threshold_m"], 0.05)
        self.assertEqual(scenario["ground_confirm_min_m"], 20.0)
        self.assertEqual(scenario["slope_speed_weight"], 0.5)
        self.assertEqual(scenario["land_cover_speeds"], (1.0, 0.95, 0.8, 0.7, 0.0, 0.0))

    def test_ugv_known_survivor_diagnostic_global_fire_defaults(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="ugv_global_fire_diag",
            ugv_known_survivor_diagnostic=True,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(algo_args["model"]["lr"], 0.00025)
        self.assertEqual(algo_args["model"]["critic_lr"], 0.0005)
        self.assertTrue(algo_args["train"]["use_linear_lr_decay"])
        self.assertEqual(env_args["action_transform"], "radial_tanh")
        self.assertEqual(scenario["local_map_patch_size"], 7)
        self.assertTrue(scenario["terrain_cache_path"].endswith("malibu_creek_500m_128.npz"))
        self.assertEqual(scenario["known_survivor_spawn_distance_min_m"], 30.0)
        self.assertNotIn("known_survivor_spawn_distance_m", scenario)
        self.assertFalse(scenario["disable_fire"])
        self.assertEqual(scenario["ugv_planner_hint"], "global_astar")
        self.assertEqual(scenario["ugv_dense_reward_mode"], "planner_follow")
        self.assertEqual(scenario["ugv_global_planner_heuristic"], "euclidean")
        self.assertEqual(scenario["ugv_global_planner_lookahead_m"], 20.0)
        self.assertEqual(scenario["r_ugv_planner_progress"], 0.0)
        self.assertEqual(scenario["r_ground_approach"], 0.05)
        self.assertEqual(scenario["ugv_planner_fire_mode"], "block")
        self.assertEqual(scenario["ugv_planner_fire_cost"], 25.0)
        self.assertEqual(scenario["ugv_planner_smoke_cost"], 5.0)
        self.assertEqual(scenario["ugv_planner_smolder_cost"], 3.0)
        self.assertEqual(scenario["ugv_planner_fire_buffer_m"], 10.0)
        self.assertEqual(scenario["ugv_planner_fire_buffer_cost"], 8.0)
        self.assertEqual(scenario["ugv_planner_fire_replan_policy"], "lazy")
        self.assertEqual(scenario["ugv_planner_fire_replan_interval_steps"], 15)
        self.assertEqual(scenario["ugv_planner_fire_block_threshold"], 0.6)
        self.assertEqual(
            scenario["ugv_planner_land_cover_costs"],
            (0.85, 1.0, 1.15, 1.35, 4.0, 8.0),
        )

    def test_uav_survivor_diagnostic_build_args(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag",
            terrain_cache_path="terrain.npz",
            coverage_obs_grid=6,
            local_coverage_obs_grid=9,
            local_coverage_obs_radius_m=150.0,
            uav_confidence_obs_grid=4,
            uav_local_confidence_obs_grid=5,
            uav_local_confidence_obs_radius_m=100.0,
            uav_frontier_obs=True,
            uav_frontier_obs_radius_m=120.0,
            uav_frontier_mode="local_global",
            uav_frontier_source="confidence",
            uav_survivor_diagnostic=True,
            uav_coverage_reward=7.5,
            uav_move_coverage_reward=0.001,
            uav_move_coverage_cap=0.2,
            uav_frontier_alignment_reward=0.04,
            uav_confidence_reward=0.03,
            uav_confidence_move_reward=0.07,
            uav_confidence_overlap_penalty=0.04,
            uav_confidence_overlap_mode="opportunity-regret",
            uav_confidence_overlap_allowed_regret=0.12,
            uav_confidence_overlap_threshold=0.70,
            uav_confidence_opportunity_eps=1e-5,
            uav_overlap_penalty=0.05,
            uav_overlap_allowed=0.60,
            uav_inter_uav_overlap_penalty=0.03,
            uav_inter_uav_overlap_allowed=0.25,
            uav_outside_footprint_penalty=0.1,
            uav_boundary_soft_margin_m=30.0,
            action_transform="radial_tanh",
        )

        self.assertEqual(env_args["action_transform"], "radial_tanh")
        scenario = env_args["scenario_kwargs"]
        self.assertTrue(algo_args["algo"]["share_param"])
        self.assertEqual(scenario["n_drones"], 3)
        self.assertEqual(scenario["n_ground"], 0)
        self.assertEqual(scenario["n_survivors"], 5)
        self.assertFalse(scenario["known_survivors_at_reset"])
        self.assertNotIn("survivor_spawn_reference", scenario)
        self.assertTrue(scenario["drone_can_confirm"])
        self.assertNotIn("known_survivor_spawn_distance_m", scenario)
        self.assertNotIn("known_survivor_spawn_distance_min_m", scenario)
        self.assertNotIn("known_survivor_spawn_distance_max_m", scenario)
        self.assertTrue(scenario["disable_fire"])
        self.assertEqual(scenario["comms_dropout"], 0.0)
        self.assertEqual(scenario["r_drone_scout"], 2.0)
        self.assertEqual(scenario["r_drone_shaping"], 0.0)
        self.assertEqual(scenario["r_ground_confirm"], 0.0)
        self.assertEqual(scenario["r_ground_shaping"], 0.0)
        self.assertEqual(scenario["r_coverage"], 7.5)
        self.assertEqual(scenario["uav_coverage_normalization"], "map")
        self.assertEqual(scenario["coverage_obs_grid"], 6)
        self.assertEqual(scenario["local_coverage_obs_grid"], 9)
        self.assertEqual(scenario["local_coverage_obs_radius_m"], 150.0)
        self.assertEqual(scenario["uav_confidence_obs_grid"], 4)
        self.assertEqual(scenario["local_confidence_obs_grid"], 5)
        self.assertEqual(scenario["local_confidence_obs_radius_m"], 100.0)
        self.assertTrue(scenario["uav_frontier_obs"])
        self.assertEqual(scenario["uav_frontier_obs_radius_m"], 120.0)
        self.assertEqual(scenario["uav_frontier_mode"], "local_global")
        self.assertEqual(scenario["uav_frontier_source"], "confidence")
        self.assertEqual(scenario["r_uav_move_coverage"], 0.001)
        self.assertEqual(scenario["uav_move_coverage_normalization"], "raw")
        self.assertEqual(scenario["r_uav_move_coverage_cap"], 0.2)
        self.assertEqual(scenario["r_uav_frontier_alignment"], 0.04)
        self.assertEqual(scenario["r_uav_confidence"], 0.03)
        self.assertEqual(scenario["r_uav_confidence_move"], 0.07)
        self.assertEqual(scenario["r_uav_confidence_overlap"], 0.04)
        self.assertEqual(scenario["uav_confidence_overlap_mode"], "opportunity_regret")
        self.assertEqual(scenario["uav_confidence_overlap_allowed_regret"], 0.12)
        self.assertEqual(scenario["uav_confidence_overlap_threshold"], 0.70)
        self.assertEqual(scenario["uav_confidence_opportunity_eps"], 1e-5)
        self.assertEqual(scenario["r_uav_overlap"], 0.05)
        self.assertEqual(scenario["uav_overlap_allowed"], 0.60)
        self.assertEqual(scenario["uav_overlap_penalty_normalization"], "raw")
        self.assertEqual(scenario["r_uav_inter_uav_overlap"], 0.03)
        self.assertEqual(scenario["uav_inter_uav_overlap_allowed"], 0.25)
        self.assertEqual(scenario["r_uav_outside_footprint"], 0.1)
        self.assertEqual(scenario["uav_boundary_soft_margin_m"], 30.0)
        self.assertEqual(scenario["uav_start_min_separation_m"], 150.0)
        self.assertEqual(scenario["uav_start_edge_margin_m"], 50.0)
        self.assertEqual(scenario["r_fire_penalty"], 0.0)
        self.assertEqual(scenario["r_drone_climb_cost"], 0.0)

    def test_uav_survivor_diagnostic_uses_current_defaults(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_defaults",
            uav_survivor_diagnostic=True,
        )

        self.assertEqual(env_args["action_transform"], "radial_tanh")
        scenario = env_args["scenario_kwargs"]
        self.assertTrue(algo_args["algo"]["share_param"])
        self.assertEqual(scenario["n_drones"], 3)
        self.assertEqual(scenario["n_ground"], 0)
        self.assertEqual(scenario["n_survivors"], 5)
        self.assertEqual(scenario["r_drone_scout"], 2.0)
        self.assertEqual(scenario["r_found_survivor"], 0.0)
        self.assertEqual(scenario["r_all_survivors_found"], 0.0)
        self.assertEqual(scenario["r_time_penalty"], 0.0)
        self.assertEqual(scenario["r_coverage"], 0.0)
        self.assertEqual(scenario["uav_coverage_normalization"], "map")
        self.assertEqual(scenario["coverage_obs_grid"], 6)
        self.assertEqual(scenario["local_coverage_obs_grid"], 9)
        self.assertEqual(scenario["local_coverage_obs_radius_m"], 150.0)
        self.assertEqual(scenario["uav_confidence_obs_grid"], 6)
        self.assertNotIn("local_confidence_obs_grid", scenario)
        self.assertEqual(scenario["r_uav_move_coverage"], 0.0)
        self.assertEqual(scenario["uav_move_coverage_normalization"], "raw")
        self.assertEqual(scenario["r_uav_move_coverage_cap"], 0.1)
        self.assertEqual(scenario["r_uav_coverage_threshold"], 0.0)
        self.assertEqual(scenario["uav_coverage_threshold_fraction"], 0.95)
        self.assertEqual(scenario["r_uav_overlap"], 0.0)
        self.assertEqual(scenario["uav_overlap_allowed"], 0.10)
        self.assertEqual(scenario["uav_overlap_penalty_normalization"], "raw")
        self.assertEqual(scenario["r_uav_inter_uav_overlap"], 0.0)
        self.assertEqual(scenario["uav_inter_uav_overlap_allowed"], 0.20)
        self.assertTrue(scenario["uav_frontier_obs"])
        self.assertEqual(scenario["uav_frontier_obs_radius_m"], 60.0)
        self.assertEqual(scenario["uav_frontier_mode"], "local_global")
        self.assertEqual(scenario["uav_frontier_source"], "confidence")
        self.assertEqual(scenario["uav_frontier_sectors"], 8)
        self.assertEqual(scenario["uav_frontier_top_k"], 2)
        self.assertTrue(scenario["uav_frontier_ownership"])
        self.assertEqual(scenario["r_uav_frontier_alignment"], 0.05)
        self.assertEqual(scenario["r_uav_confidence"], 30.0)
        self.assertEqual(scenario["r_uav_confidence_move"], 0.10)
        self.assertEqual(scenario["r_uav_inefficient_move"], 0.01)
        self.assertEqual(scenario["uav_inefficient_move_source"], "coverage")
        self.assertEqual(scenario["r_uav_confidence_overlap"], 0.06)
        self.assertEqual(scenario["uav_confidence_overlap_mode"], "raw")
        self.assertEqual(scenario["uav_confidence_overlap_allowed_regret"], 0.10)
        self.assertEqual(scenario["uav_confidence_overlap_threshold"], 0.80)
        self.assertEqual(scenario["uav_confidence_opportunity_eps"], 1e-6)
        self.assertTrue(scenario["uav_cleanup_target_obs"])
        self.assertEqual(scenario["r_uav_cleanup_target_progress"], 0.10)
        self.assertEqual(scenario["uav_cleanup_target_refresh_mode"], "fixed_hold")
        self.assertEqual(scenario["r_uav_outside_footprint"], 0.10)
        self.assertEqual(scenario["uav_start_min_separation_m"], 150.0)
        self.assertEqual(scenario["uav_start_edge_margin_m"], 50.0)
        self.assertEqual(algo_args["model"]["terrain_cnn_single_obs_dim"], 336)

    def test_uav_survivor_diagnostic_can_use_opportunity_coverage_normalization(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_opp_cov",
            uav_survivor_diagnostic=True,
            uav_coverage_reward=0.5,
            uav_coverage_normalization="opportunity",
            uav_coverage_opportunity_cap=1.0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_coverage"], 0.5)
        self.assertEqual(scenario["uav_coverage_normalization"], "opportunity")
        self.assertEqual(scenario["uav_coverage_opportunity_cap"], 1.0)
        self.assertNotIn("r_uav_coverage_opportunity", scenario)

    def test_legacy_uav_opportunity_reward_alias_sets_coverage_normalization(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_legacy_opp_cov",
            uav_survivor_diagnostic=True,
            uav_coverage_reward=0.0,
            uav_coverage_opportunity_reward=0.5,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_coverage"], 0.5)
        self.assertEqual(scenario["uav_coverage_normalization"], "opportunity")
        self.assertNotIn("r_uav_coverage_opportunity", scenario)

    def test_uav_survivor_diagnostic_can_use_opportunity_move_coverage_normalization(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_opp_move_cov",
            uav_survivor_diagnostic=True,
            uav_move_coverage_reward=0.2,
            uav_move_coverage_normalization="opportunity",
            uav_move_coverage_cap=0.05,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_uav_move_coverage"], 0.2)
        self.assertEqual(scenario["uav_move_coverage_normalization"], "opportunity")
        self.assertEqual(scenario["r_uav_move_coverage_cap"], 0.05)

    def test_uav_survivor_diagnostic_can_use_opportunity_overlap_penalty_normalization(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_opp_overlap",
            uav_survivor_diagnostic=True,
            uav_overlap_penalty=0.05,
            uav_overlap_allowed=0.60,
            uav_overlap_penalty_normalization="opportunity",
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_uav_overlap"], 0.05)
        self.assertEqual(scenario["uav_overlap_allowed"], 0.60)
        self.assertEqual(scenario["uav_overlap_penalty_normalization"], "opportunity")

    def test_uav_survivor_diagnostic_preserves_explicit_zero_reward_overrides(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_zeroes",
            uav_survivor_diagnostic=True,
            uav_coverage_reward=0.0,
            uav_move_coverage_reward=0.0,
            uav_overlap_penalty=0.0,
            uav_overlap_allowed=0.0,
            uav_inter_uav_overlap_penalty=0.0,
            uav_inter_uav_overlap_allowed=0.0,
            uav_outside_footprint_penalty=0.0,
            uav_frontier_alignment_reward=0.0,
            uav_confidence_reward=0.0,
            uav_confidence_move_reward=0.0,
            uav_inefficient_move_penalty=0.0,
            uav_confidence_overlap_penalty=0.0,
            uav_cleanup_target_progress_reward=0.0,
            uav_cleanup_target_obs=False,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_coverage"], 0.0)
        self.assertEqual(scenario["r_uav_move_coverage"], 0.0)
        self.assertEqual(scenario["r_uav_overlap"], 0.0)
        self.assertEqual(scenario["uav_overlap_allowed"], 0.0)
        self.assertEqual(scenario["r_uav_inter_uav_overlap"], 0.0)
        self.assertEqual(scenario["uav_inter_uav_overlap_allowed"], 0.0)
        self.assertEqual(scenario["r_uav_outside_footprint"], 0.0)
        self.assertEqual(scenario["r_uav_frontier_alignment"], 0.0)
        self.assertEqual(scenario["r_uav_confidence"], 0.0)
        self.assertEqual(scenario["r_uav_confidence_move"], 0.0)
        self.assertEqual(scenario["r_uav_inefficient_move"], 0.0)
        self.assertEqual(scenario["r_uav_confidence_overlap"], 0.0)
        self.assertEqual(scenario["r_uav_cleanup_target_progress"], 0.0)
        self.assertNotIn("uav_cleanup_target_obs", scenario)

    def test_uav_survivor_diagnostic_can_use_two_drones(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_2d",
            uav_survivor_diagnostic=True,
            uav_diagnostic_drones=2,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["n_drones"], 2)
        self.assertEqual(scenario["n_ground"], 0)
        self.assertEqual(scenario["n_survivors"], 5)
        self.assertEqual(scenario["r_drone_scout"], 2.0)
        self.assertEqual(scenario["r_found_survivor"], 0.0)
        self.assertEqual(scenario["r_time_penalty"], 0.0)
        self.assertEqual(scenario["uav_start_min_separation_m"], 150.0)
        self.assertEqual(scenario["uav_start_edge_margin_m"], 50.0)

    def test_uav_survivor_diagnostic_start_constraints_can_be_disabled(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_diag_no_start_constraints",
            uav_survivor_diagnostic=True,
            uav_start_min_separation_m=0.0,
            uav_start_edge_margin_m=0.0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["uav_start_min_separation_m"], 0.0)
        self.assertEqual(scenario["uav_start_edge_margin_m"], 0.0)

    def test_uav_diagnostics_preserves_manifest_drone_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True)
            runner = types.SimpleNamespace(save_dir=models_dir)
            save_training_manifest(
                runner,
                harl_args={},
                algo_args={},
                env_args={
                    "scenario_kwargs": {
                        "n_drones": 2,
                        "n_ground": 0,
                        "n_survivors": 5,
                        "max_steps": 300,
                    },
                },
            )
            args = types.SimpleNamespace(
                steps=123,
                n_drones=None,
                terrain_cache_path=None,
                local_map_patch_size=None,
                drone_min_footprint_radius_m=None,
            )

            scenario = diagnose_uav_scenario_kwargs(models_dir, args)

        self.assertEqual(scenario["n_drones"], 2)
        self.assertEqual(scenario["n_ground"], 0)
        self.assertEqual(scenario["n_survivors"], 5)
        self.assertEqual(scenario["max_steps"], 123)

    def test_uav_diagnostics_summarizes_per_drone_metrics(self):
        rows = [
            {
                "per_drone": [
                    {
                        "drone": 0,
                        "scout_credit_count": 2,
                        "avg_action_norm": 0.5,
                        "avg_displacement_m": 10.0,
                        "path_length_m": 100.0,
                        "avg_action_displacement_alignment": 0.8,
                        "avg_new_coverage_cells": 20.0,
                        "total_new_coverage_cells": 200.0,
                        "new_coverage_step_frac": 0.9,
                        "avg_outside_footprint_fraction": 0.01,
                        "avg_overlap_fraction": 0.7,
                        "avg_expected_overlap_fraction": 0.6,
                        "avg_excess_overlap_fraction": 0.1,
                        "excess_overlap_step_frac_10": 0.2,
                        "edge_step_frac": 0.1,
                        "corner_step_frac": 0.0,
                        "stalled_step_frac": 0.0,
                        "longest_stall_steps": 1.0,
                        "moving_no_new_coverage_frac": 0.05,
                        "mean_boundary_distance_m": 100.0,
                        "min_boundary_distance_m": 50.0,
                    },
                    {
                        "drone": 1,
                        "scout_credit_count": 0,
                        "avg_action_norm": 0.2,
                        "avg_displacement_m": 2.0,
                        "path_length_m": 20.0,
                        "avg_action_displacement_alignment": 0.1,
                        "avg_new_coverage_cells": 2.0,
                        "total_new_coverage_cells": 20.0,
                        "new_coverage_step_frac": 0.2,
                        "avg_outside_footprint_fraction": 0.2,
                        "avg_overlap_fraction": 0.9,
                        "avg_expected_overlap_fraction": 0.8,
                        "avg_excess_overlap_fraction": 0.1,
                        "excess_overlap_step_frac_10": 0.3,
                        "edge_step_frac": 0.8,
                        "corner_step_frac": 0.4,
                        "stalled_step_frac": 0.3,
                        "longest_stall_steps": 20.0,
                        "moving_no_new_coverage_frac": 0.4,
                        "mean_boundary_distance_m": 10.0,
                        "min_boundary_distance_m": 0.0,
                    },
                ],
            },
            {
                "per_drone": [
                    {
                        "drone": 0,
                        "scout_credit_count": 4,
                        "avg_action_norm": 0.7,
                        "avg_displacement_m": 12.0,
                        "path_length_m": 120.0,
                        "avg_action_displacement_alignment": 0.9,
                        "avg_new_coverage_cells": 30.0,
                        "total_new_coverage_cells": 300.0,
                        "new_coverage_step_frac": 1.0,
                        "avg_outside_footprint_fraction": 0.02,
                        "avg_overlap_fraction": 0.6,
                        "avg_expected_overlap_fraction": 0.5,
                        "avg_excess_overlap_fraction": 0.1,
                        "excess_overlap_step_frac_10": 0.1,
                        "edge_step_frac": 0.2,
                        "corner_step_frac": 0.0,
                        "stalled_step_frac": 0.0,
                        "longest_stall_steps": 2.0,
                        "moving_no_new_coverage_frac": 0.03,
                        "mean_boundary_distance_m": 120.0,
                        "min_boundary_distance_m": 60.0,
                    },
                ],
            },
        ]

        summary = _summarize_per_drone(rows)

        self.assertEqual([row["drone"] for row in summary], [0, 1])
        self.assertEqual(summary[0]["episodes"], 2.0)
        self.assertEqual(summary[1]["episodes"], 1.0)
        self.assertAlmostEqual(summary[0]["mean_scout_credit_count"], 3.0)
        self.assertAlmostEqual(summary[0]["mean_path_length_m"], 110.0)
        self.assertAlmostEqual(summary[1]["mean_edge_step_frac"], 0.8)

    def test_uav_coverage_only_diagnostic_build_args(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_coverage_only",
            uav_survivor_diagnostic=True,
            uav_coverage_only=True,
            uav_coverage_reward=20.0,
            uav_move_coverage_reward=0.001,
            uav_move_coverage_cap=0.2,
            uav_overlap_penalty=0.05,
            uav_overlap_allowed=0.60,
            uav_outside_footprint_penalty=0.1,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["n_drones"], 3)
        self.assertEqual(scenario["n_ground"], 0)
        self.assertEqual(scenario["n_survivors"], 5)
        self.assertEqual(scenario["r_found_survivor"], 0.0)
        self.assertEqual(scenario["r_all_survivors_found"], 0.0)
        self.assertEqual(scenario["r_drone_scout"], 0.0)
        self.assertEqual(scenario["r_time_penalty"], 0.0)
        self.assertEqual(scenario["r_coverage"], 20.0)
        self.assertEqual(scenario["uav_coverage_normalization"], "map")
        self.assertEqual(scenario["r_uav_move_coverage"], 0.001)
        self.assertEqual(scenario["uav_move_coverage_normalization"], "raw")
        self.assertEqual(scenario["r_uav_move_coverage_cap"], 0.2)
        self.assertEqual(scenario["r_uav_coverage_threshold"], 0.0)
        self.assertEqual(scenario["uav_coverage_threshold_fraction"], 0.95)
        self.assertEqual(scenario["r_uav_overlap"], 0.05)
        self.assertEqual(scenario["uav_overlap_allowed"], 0.60)
        self.assertEqual(scenario["uav_overlap_penalty_normalization"], "raw")
        self.assertEqual(scenario["r_uav_outside_footprint"], 0.1)

    def test_uav_diagnostic_can_disable_team_found_and_time_reward(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_scout_only",
            uav_survivor_diagnostic=True,
            uav_found_survivor_reward=0.0,
            uav_time_penalty=0.0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_drone_scout"], 2.0)
        self.assertEqual(scenario["r_found_survivor"], 0.0)
        self.assertEqual(scenario["r_all_survivors_found"], 0.0)
        self.assertEqual(scenario["r_time_penalty"], 0.0)
        self.assertEqual(scenario["r_coverage"], 0.0)
        self.assertEqual(scenario["coverage_obs_grid"], 6)
        self.assertEqual(scenario["local_coverage_obs_grid"], 9)
        self.assertEqual(scenario["r_uav_confidence"], 30.0)
        self.assertEqual(scenario["r_uav_confidence_overlap"], 0.06)

    def test_uav_diagnostic_can_set_all_survivors_reward(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_completion_bonus",
            uav_survivor_diagnostic=True,
            uav_all_survivors_reward=8.0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_found_survivor"], 0.0)
        self.assertEqual(scenario["r_all_survivors_found"], 8.0)
        self.assertEqual(scenario["r_drone_scout"], 2.0)

    def test_uav_diagnostic_can_set_coverage_threshold_reward(self):
        _, _, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_coverage_threshold_bonus",
            uav_survivor_diagnostic=True,
            uav_coverage_threshold_reward=6.0,
            uav_coverage_threshold_fraction=0.90,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertEqual(scenario["r_uav_coverage_threshold"], 6.0)
        self.assertEqual(scenario["uav_coverage_threshold_fraction"], 0.90)

    def test_uav_diagnostic_can_disable_default_global_coverage_observation(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_local_only",
            uav_survivor_diagnostic=True,
            uav_coverage_only=True,
            uav_no_global_coverage_obs=True,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertNotIn("coverage_obs_grid", scenario)
        self.assertEqual(scenario["local_coverage_obs_grid"], 9)
        self.assertEqual(scenario["local_coverage_obs_radius_m"], 150.0)
        self.assertEqual(scenario["local_map_patch_size"], 7)
        self.assertEqual(scenario["terrain_source"], "real")
        self.assertTrue(scenario["terrain_cache_path"].endswith("malibu_creek_500m_128.npz"))
        self.assertEqual(scenario["r_found_survivor"], 0.0)
        self.assertEqual(scenario["r_drone_scout"], 0.0)
        self.assertTrue(scenario["uav_frontier_obs"])
        self.assertEqual(scenario["r_uav_frontier_alignment"], 0.05)
        self.assertEqual(scenario["uav_confidence_obs_grid"], 6)
        self.assertTrue(scenario["uav_cleanup_target_obs"])
        self.assertEqual(algo_args["model"]["terrain_cnn_single_obs_dim"], 299)

    def test_uav_diagnostic_explicit_zero_disables_coverage_observations(self):
        _, algo_args, env_args = build_args(
            num_env_steps=100,
            episode_length=50,
            seed=1,
            comms_dropout=0.5,
            entropy_coef=0.01,
            exp_name="uav_no_coverage_obs",
            uav_survivor_diagnostic=True,
            coverage_obs_grid=0,
            local_coverage_obs_grid=0,
        )

        scenario = env_args["scenario_kwargs"]
        self.assertNotIn("coverage_obs_grid", scenario)
        self.assertNotIn("local_coverage_obs_grid", scenario)
        self.assertEqual(scenario["uav_confidence_obs_grid"], 6)
        self.assertLess(algo_args["model"]["terrain_cnn_single_obs_dim"], 258)

    def test_diagnostic_modes_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            build_args(
                num_env_steps=100,
                episode_length=50,
                seed=1,
                comms_dropout=0.0,
                entropy_coef=0.01,
                exp_name="both_diag",
                ugv_known_survivor_diagnostic=True,
                uav_survivor_diagnostic=True,
            )

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
        self.assertEqual(scenario["known_survivor_spawn_distance_min_m"], 30.0)
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
