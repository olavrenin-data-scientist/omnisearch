import unittest

import torch

from agents.harl_terrain_cnn import TerrainCNNMLPBase, wildfire_single_observation_dim
from harl.models.base.mlp import MLPBase


def _model_args(**overrides):
    args = {
        "hidden_sizes": [32, 32],
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "use_terrain_cnn_encoder": False,
    }
    args.update(overrides)
    return args


class HARLTerrainCNNTests(unittest.TestCase):
    def test_disabled_encoder_matches_harl_mlp_state_dict_keys(self):
        args = _model_args()
        patched = TerrainCNNMLPBase(args, (64,))
        original = MLPBase(args, (64,))

        self.assertEqual(set(patched.state_dict()), set(original.state_dict()))
        self.assertEqual(patched(torch.zeros(3, 64)).shape, (3, 32))

    def test_wildfire_observation_dim_includes_optional_planner_hint(self):
        base = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
        )
        with_planner = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            ugv_planner_hint="local-astar",
        )
        with_planner_detour = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            ugv_planner_hint="local-astar",
            ugv_planner_detour_obs=True,
        )
        with_escape_planner = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            ugv_planner_hint="local-escape-astar",
        )

        self.assertEqual(base, 4 + 12 + 1 + 2 * 7 * 7 + 9 + 2 + 4 + 7)
        self.assertEqual(with_planner, base + 5)
        self.assertEqual(with_planner_detour, base + 6)
        self.assertEqual(with_escape_planner, base + 5)

        with_coverage = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            coverage_obs_grid=6,
        )
        self.assertEqual(with_coverage, base + 6 * 6 + 1)

        with_local_coverage = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            local_coverage_obs_grid=9,
        )
        self.assertEqual(with_local_coverage, base + 9 * 9)

        with_confidence = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            uav_confidence_obs_grid=6,
        )
        self.assertEqual(with_confidence, base + 6 * 6 + 1)

        with_local_confidence = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            local_confidence_obs_grid=9,
        )
        self.assertEqual(with_local_confidence, base + 9 * 9)

        with_frontier = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            uav_frontier_obs=True,
        )
        self.assertEqual(with_frontier, base + 4)

        with_local_global_frontier = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            uav_frontier_obs=True,
            uav_frontier_mode="local_global",
        )
        self.assertEqual(with_local_global_frontier, base + 8)

        with_cleanup_target = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            uav_cleanup_target_obs=True,
        )
        self.assertEqual(with_cleanup_target, base + 4)

        with_astar_route = wildfire_single_observation_dim(
            local_map_patch_size=7,
            n_agents=1,
            n_survivors=1,
            uav_astar_route_obs=True,
        )
        self.assertEqual(with_astar_route, base + 4)

    def test_enabled_encoder_replaces_patch_with_embedding(self):
        patch_size = 7
        obs_dim = wildfire_single_observation_dim(
            local_map_patch_size=patch_size,
            n_agents=1,
            n_survivors=1,
        )
        args = _model_args(
            use_terrain_cnn_encoder=True,
            terrain_cnn_patch_size=patch_size,
            terrain_cnn_channels=2,
            terrain_cnn_obs_offset=17,
            terrain_cnn_embed_dim=12,
            terrain_cnn_hidden_channels=4,
            terrain_cnn_single_obs_dim=obs_dim,
        )

        model = TerrainCNNMLPBase(args, (obs_dim,))
        out = model(torch.zeros(5, obs_dim))

        self.assertEqual(out.shape, (5, 32))
        self.assertTrue(any(key.startswith("terrain_encoder.") for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
