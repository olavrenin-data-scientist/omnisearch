"""Optional terrain-patch CNN encoder for HARL's flat MLP observations."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from harl.models.base.mlp import MLPLayer
from harl.utils.models_tools import get_active_func, get_init_method, init


TERRAIN_CNN_OBS_OFFSET = 4 + 12 + 1
TERRAIN_CNN_CHANNELS = 2
UGV_PLANNER_HINT_DIM = 5
BOUNDARY_OBS_DIM = 4
SURVIVOR_MESSAGE_BASE_DIM = 7
SURVIVOR_ASSIGNMENT_OBS_DIM = 2
UAV_FRONTIER_OBS_DIM = 4
UAV_CLEANUP_TARGET_OBS_DIM = 4
UAV_ASTAR_ROUTE_OBS_DIM = 4


def uav_frontier_observation_dim(
    *,
    uav_frontier_obs: bool = False,
    uav_frontier_mode: str = "centroid",
    uav_frontier_top_k: int = 2,
) -> int:
    if not bool(uav_frontier_obs):
        return 0
    mode = str(uav_frontier_mode).replace("-", "_")
    if mode in {"sector_topk", "local_global"}:
        if mode == "local_global":
            return 8
        return 4 * max(int(uav_frontier_top_k), 1)
    return UAV_FRONTIER_OBS_DIM


def wildfire_single_observation_dim(
    *,
    local_map_patch_size: int,
    n_agents: int,
    n_survivors: int,
    n_decoys: int = 0,
    ugv_planner_hint: str = "none",
    ugv_planner_detour_obs: bool = False,
    coverage_obs_grid: int = 0,
    local_coverage_obs_grid: int = 0,
    uav_confidence_obs_grid: int = 0,
    local_confidence_obs_grid: int = 0,
    uav_frontier_obs: bool = False,
    uav_frontier_mode: str = "centroid",
    uav_frontier_top_k: int = 2,
    uav_cleanup_target_obs: bool = False,
    uav_astar_route_obs: bool = False,
    survivor_assignment_obs: bool = False,
) -> int:
    """Return the per-agent wildfire observation width for the current layout."""
    patch_size = int(local_map_patch_size)
    planner_hint_dim = 0
    if str(ugv_planner_hint).replace("-", "_") in {"local_astar", "local_escape_astar", "global_astar"}:
        planner_hint_dim = UGV_PLANNER_HINT_DIM + int(bool(ugv_planner_detour_obs))
    coverage_grid = max(int(coverage_obs_grid), 0)
    coverage_obs_dim = coverage_grid * coverage_grid + 1 if coverage_grid > 0 else 0
    local_coverage_grid = max(int(local_coverage_obs_grid), 0)
    local_coverage_obs_dim = local_coverage_grid * local_coverage_grid
    confidence_grid = max(int(uav_confidence_obs_grid), 0)
    confidence_obs_dim = confidence_grid * confidence_grid + 1 if confidence_grid > 0 else 0
    local_confidence_grid = max(int(local_confidence_obs_grid), 0)
    local_confidence_obs_dim = local_confidence_grid * local_confidence_grid
    uav_frontier_obs_dim = uav_frontier_observation_dim(
        uav_frontier_obs=uav_frontier_obs,
        uav_frontier_mode=uav_frontier_mode,
        uav_frontier_top_k=uav_frontier_top_k,
    )
    uav_cleanup_target_obs_dim = UAV_CLEANUP_TARGET_OBS_DIM if bool(uav_cleanup_target_obs) else 0
    uav_astar_route_obs_dim = UAV_ASTAR_ROUTE_OBS_DIM if bool(uav_astar_route_obs) else 0
    decoys_enabled = max(int(n_decoys), 0) > 0
    survivor_message_dim = SURVIVOR_MESSAGE_BASE_DIM + int(decoys_enabled) + (
        SURVIVOR_ASSIGNMENT_OBS_DIM if bool(survivor_assignment_obs) else 0
    )
    candidate_slots = max(int(n_survivors), 0) + max(int(n_decoys), 0)
    return (
        4  # own pos + velocity
        + 12  # lidar or drone dummy lidar
        + 1  # local fire density
        + TERRAIN_CNN_CHANNELS * patch_size * patch_size  # mobility + blocked patch
        + 9  # fixed 3x3 air-clearance patch
        + planner_hint_dim  # optional local A* waypoint hint
        + 2  # flight state
        + BOUNDARY_OBS_DIM  # distances to left, right, bottom, top boundary
        + max(int(n_agents) - 1, 0) * 2  # teammate relative positions
        + candidate_slots * survivor_message_dim  # survivor/decoy candidate messages
        + coverage_obs_dim  # optional downsampled team coverage map + global fraction
        + local_coverage_obs_dim  # optional pooled local coverage map around this agent
        + confidence_obs_dim  # optional downsampled UAV inspection-confidence map + global mean
        + local_confidence_obs_dim  # optional pooled local confidence map around this agent
        + uav_frontier_obs_dim  # optional direction/distance/strength toward uncovered coverage frontier
        + uav_cleanup_target_obs_dim  # optional persistent cleanup target destination
        + uav_astar_route_obs_dim  # optional A* route waypoint toward cleanup target
    )


def wildfire_observation_slices(
    *,
    local_map_patch_size: int,
    n_agents: int,
    n_survivors: int,
    n_decoys: int = 0,
    ugv_planner_hint: str = "none",
    ugv_planner_detour_obs: bool = False,
    coverage_obs_grid: int = 0,
    local_coverage_obs_grid: int = 0,
    uav_confidence_obs_grid: int = 0,
    local_confidence_obs_grid: int = 0,
    uav_frontier_obs: bool = False,
    uav_frontier_mode: str = "centroid",
    uav_frontier_top_k: int = 2,
    uav_cleanup_target_obs: bool = False,
    uav_astar_route_obs: bool = False,
    survivor_assignment_obs: bool = False,
) -> "OrderedDict[str, slice]":
    """Named slices for the flat wildfire observation layout."""
    patch_size = int(local_map_patch_size)
    planner_hint_dim = 0
    if str(ugv_planner_hint).replace("-", "_") in {"local_astar", "local_escape_astar", "global_astar"}:
        planner_hint_dim = UGV_PLANNER_HINT_DIM + int(bool(ugv_planner_detour_obs))
    coverage_grid = max(int(coverage_obs_grid), 0)
    coverage_obs_dim = coverage_grid * coverage_grid + 1 if coverage_grid > 0 else 0
    local_coverage_grid = max(int(local_coverage_obs_grid), 0)
    local_coverage_obs_dim = local_coverage_grid * local_coverage_grid
    confidence_grid = max(int(uav_confidence_obs_grid), 0)
    confidence_obs_dim = confidence_grid * confidence_grid + 1 if confidence_grid > 0 else 0
    local_confidence_grid = max(int(local_confidence_obs_grid), 0)
    local_confidence_obs_dim = local_confidence_grid * local_confidence_grid
    uav_frontier_obs_dim = uav_frontier_observation_dim(
        uav_frontier_obs=uav_frontier_obs,
        uav_frontier_mode=uav_frontier_mode,
        uav_frontier_top_k=uav_frontier_top_k,
    )
    decoys_enabled = max(int(n_decoys), 0) > 0
    survivor_message_dim = SURVIVOR_MESSAGE_BASE_DIM + int(decoys_enabled) + (
        SURVIVOR_ASSIGNMENT_OBS_DIM if bool(survivor_assignment_obs) else 0
    )
    candidate_slots = max(int(n_survivors), 0) + max(int(n_decoys), 0)

    widths = [
        ("kinematics", 4),
        ("lidar", 12),
        ("local_fire_density", 1),
        ("terrain_mobility_blocked", TERRAIN_CNN_CHANNELS * patch_size * patch_size),
        ("air_clearance", 9),
        ("ugv_planner_hint", planner_hint_dim),
        ("flight_state", 2),
        ("boundary", BOUNDARY_OBS_DIM),
        ("neighbors", max(int(n_agents) - 1, 0) * 2),
        ("survivor_messages", candidate_slots * survivor_message_dim),
        ("coverage", coverage_obs_dim),
        ("local_coverage", local_coverage_obs_dim),
        ("uav_confidence", confidence_obs_dim),
        ("local_confidence", local_confidence_obs_dim),
        ("uav_frontier", uav_frontier_obs_dim),
        ("uav_cleanup_target", UAV_CLEANUP_TARGET_OBS_DIM if bool(uav_cleanup_target_obs) else 0),
        ("uav_astar_route", UAV_ASTAR_ROUTE_OBS_DIM if bool(uav_astar_route_obs) else 0),
    ]
    out: "OrderedDict[str, slice]" = OrderedDict()
    offset = 0
    for name, width in widths:
        width = max(int(width), 0)
        out[name] = slice(offset, offset + width)
        offset += width
    return out


class TerrainCNNMLPBase(nn.Module):
    """HARL MLPBase-compatible module with optional terrain-patch CNN.

    When ``use_terrain_cnn_encoder`` is false, this intentionally exposes the
    same state-dict keys as HARL's original ``MLPBase`` so older checkpoints
    still load after the monkey patch is installed.
    """

    def __init__(self, args, obs_shape):
        super().__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        self.use_terrain_cnn_encoder = bool(args.get("use_terrain_cnn_encoder", False))
        self.obs_dim = int(obs_shape[0])

        if not self.use_terrain_cnn_encoder:
            if self.use_feature_normalization:
                self.feature_norm = nn.LayerNorm(self.obs_dim)
            self.mlp = MLPLayer(
                self.obs_dim,
                self.hidden_sizes,
                self.initialization_method,
                self.activation_func,
            )
            return

        self.patch_size = int(args.get("terrain_cnn_patch_size", 0))
        self.patch_channels = int(args.get("terrain_cnn_channels", TERRAIN_CNN_CHANNELS))
        self.patch_offset = int(args.get("terrain_cnn_obs_offset", TERRAIN_CNN_OBS_OFFSET))
        self.single_obs_dim = int(args.get("terrain_cnn_single_obs_dim", self.obs_dim))
        self.embed_dim = int(args.get("terrain_cnn_embed_dim", 16))
        hidden_channels = int(args.get("terrain_cnn_hidden_channels", 8))

        if self.patch_size < 1 or self.patch_size % 2 != 1:
            raise ValueError("terrain_cnn_patch_size must be a positive odd integer")
        if self.patch_channels <= 0:
            raise ValueError("terrain_cnn_channels must be positive")
        if self.embed_dim <= 0:
            raise ValueError("terrain_cnn_embed_dim must be positive")
        if self.obs_dim % self.single_obs_dim != 0:
            raise ValueError(
                "terrain_cnn_single_obs_dim must evenly divide the observation width "
                f"({self.single_obs_dim} vs {self.obs_dim})",
            )

        self.patch_len = self.patch_channels * self.patch_size * self.patch_size
        if self.patch_offset < 0 or self.patch_offset + self.patch_len > self.single_obs_dim:
            raise ValueError("terrain CNN patch slice does not fit inside one observation block")

        self.n_obs_blocks = self.obs_dim // self.single_obs_dim
        nonterrain_per_block = self.single_obs_dim - self.patch_len
        self.nonterrain_dim = self.n_obs_blocks * nonterrain_per_block
        if self.use_feature_normalization:
            self.scalar_norm = nn.LayerNorm(self.nonterrain_dim)

        active_func = get_active_func(self.activation_func)
        init_method = get_init_method(self.initialization_method)
        gain = nn.init.calculate_gain(self.activation_func)

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.terrain_encoder = nn.Sequential(
            init_(nn.Conv2d(self.patch_channels, hidden_channels, kernel_size=3, padding=1)),
            active_func,
            init_(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            active_func,
            nn.Flatten(),
            init_(nn.Linear(hidden_channels * self.patch_size * self.patch_size, self.embed_dim)),
            active_func,
            nn.LayerNorm(self.embed_dim),
        )
        mlp_input_dim = self.nonterrain_dim + self.n_obs_blocks * self.embed_dim
        self.mlp = MLPLayer(
            mlp_input_dim,
            self.hidden_sizes,
            self.initialization_method,
            self.activation_func,
        )

    def forward(self, x):
        if not self.use_terrain_cnn_encoder:
            if self.use_feature_normalization:
                x = self.feature_norm(x)
            return self.mlp(x)

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.obs_dim)
        scalar_parts = []
        patch_parts = []
        for block_idx in range(self.n_obs_blocks):
            block_start = block_idx * self.single_obs_dim
            patch_start = block_start + self.patch_offset
            patch_end = patch_start + self.patch_len
            block_end = block_start + self.single_obs_dim
            scalar_parts.append(x_flat[:, block_start:patch_start])
            scalar_parts.append(x_flat[:, patch_end:block_end])
            patch_parts.append(
                x_flat[:, patch_start:patch_end].reshape(
                    x_flat.shape[0],
                    self.patch_channels,
                    self.patch_size,
                    self.patch_size,
                ),
            )

        scalars = torch.cat(scalar_parts, dim=-1)
        if self.use_feature_normalization:
            scalars = self.scalar_norm(scalars)

        patches = torch.cat(patch_parts, dim=0)
        patch_embeddings = self.terrain_encoder(patches).reshape(
            self.n_obs_blocks,
            x_flat.shape[0],
            self.embed_dim,
        )
        patch_embeddings = patch_embeddings.permute(1, 0, 2).reshape(x_flat.shape[0], -1)
        features = torch.cat([scalars, patch_embeddings], dim=-1)
        out = self.mlp(features)
        return out.reshape(*leading_shape, self.hidden_sizes[-1])


def install_harl_terrain_cnn_patch() -> None:
    """Install the optional CNN base into HARL actor/critic modules."""
    import harl.models.policy_models.stochastic_policy as stochastic_policy
    import harl.models.value_function_models.v_net as v_net

    stochastic_policy.MLPBase = TerrainCNNMLPBase
    v_net.MLPBase = TerrainCNNMLPBase
