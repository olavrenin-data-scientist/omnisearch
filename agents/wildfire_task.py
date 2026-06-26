"""
BenchMARL integration for the WildfireSearchScenario.

BenchMARL's stock VMAS path looks up scenarios by string name from VMAS's
built-in registry. To use our custom scenario we subclass VmasClass and
override get_env_fun so it passes our scenario *instance* through to
torchrl.envs.libs.vmas.VmasEnv (which accepts either a name or an instance).

Usage
-----
    from agents.wildfire_task import make_wildfire_task

    task = make_wildfire_task(max_steps=200, n_drones=3, n_ground=2)
    # task is a TaskClass instance that Experiment(task=...) accepts directly.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

from benchmarl.environments.common import Task
from benchmarl.environments.vmas.common import VmasClass
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from envs.wildfire_defaults import (
    DRONE_CAMERA_FOV_DEG,
    DRONE_FLIGHT_LEVELS_M,
    DRONE_SAFETY_CLEARANCE_M,
    DRONE_SPEED_MPS,
    DRONE_U_MULTIPLIER,
    GROUND_ACCEL_MPS2,
    GROUND_ARRIVAL_DAMPING,
    GROUND_ARRIVAL_SLOWDOWN_M,
    GROUND_CONFIRMATION_RANGE_M,
    GROUND_LIDAR_RANGE_M,
    GROUND_SPEED_MPS,
    SIM_STEP_SECONDS,
)
from envs.wildfire_search import WildfireSearchScenario


class WildfireVmasClass(VmasClass):
    """VMAS task class that uses a fresh WildfireSearchScenario instance."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: str,
    ) -> Callable[[], EnvBase]:
        # Strip our scenario kwargs from the config (the rest goes to VmasEnv).
        config = copy.deepcopy(self.config)
        # Everything in config is a scenario kwarg for WildfireSearchScenario;
        # VmasEnv passes **kwargs through to the scenario's make_world.
        return lambda: VmasEnv(
            scenario=WildfireSearchScenario(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def max_steps(self, env: EnvBase) -> int:
        # Used by BenchMARL to size the rollout buffer.
        return self.config.get("max_steps", 500)


class WildfireTask(Task):
    """Single-task enum for OmniSearch's wildfire search environment."""

    WILDFIRE_SEARCH = None

    @staticmethod
    def associated_class():
        return WildfireVmasClass


# Default scenario kwargs for a small/fast smoke training run.
DEFAULT_CONFIG: Dict[str, Any] = {
    "max_steps":          500,
    "n_drones":             3,
    "n_ground":             2,
    "n_survivors":          5,
    "x_semidim":          1.0,
    "y_semidim":          1.0,
    "drone_camera_fov_deg": DRONE_CAMERA_FOV_DEG,
    "ground_lidar_range_m": GROUND_LIDAR_RANGE_M,
    "n_lidar_rays":        12,
    "agent_radius_m":      0.50,
    "survivor_radius_m":   0.35,
    "ground_confirmation_range_m": GROUND_CONFIRMATION_RANGE_M,
    "spawn_padding_m":      1.0,
    "fire_grid_size":     128,
    "terrain_reference_grid_size": 16,
    "fire_spread_prob":  0.03,
    "fire_wind_spread_weight": 1.25,
    "fire_slope_spread_weight": 1.65,
    "fire_moisture_damping": 1.15,
    "fire_intensity_decay": 0.82,
    "initial_fire_cells":   1,
    "fire_step_interval":   5,
    "land_cover_fire_burnout_min_updates": (5, 5, 20, 60, 0, 0),
    "land_cover_fire_burnout_max_updates": (20, 20, 60, 200, 0, 0),
    "smoke_emission":    0.18,
    "smoke_decay":       0.985,
    "smoke_diffusion":   0.16,
    "smolder_smoke_emission": 0.04,
    "smolder_decay":     0.995,
    "smolder_start_fraction": 0.65,
    "land_cover_fire_fuel": (0.05, 0.40, 1.10, 1.35, 0.0, 0.0),
    "object_fire_fuel": (0.0, 0.25, 1.00),
    "wind_direction":   (1, 0),
    "wind_strength":     0.06,
    "terrain_source":      "real",
    "terrain_cache_dir":   "data/terrain_cache",
    "max_ground_slope":    0.70,
    "slope_cost_weight":    2.0,
    "slope_speed_weight":   0.5,
    "terrain_path_samples":   6,
    "land_cover_costs": (0.65, 1.0, 1.5, 2.2, 4.0, 8.0),
    "land_cover_speeds": (1.0, 0.95, 0.8, 0.7, 0.0, 0.0),
    "sim_step_seconds": SIM_STEP_SECONDS,
    "ground_speed_mps": GROUND_SPEED_MPS,
    "ground_accel_mps2": GROUND_ACCEL_MPS2,
    "ground_arrival_slowdown_m": GROUND_ARRIVAL_SLOWDOWN_M,
    "ground_arrival_damping": GROUND_ARRIVAL_DAMPING,
    "drone_speed_mps": DRONE_SPEED_MPS,
    "drone_u_multiplier": DRONE_U_MULTIPLIER,
    "drone_flight_levels_m": DRONE_FLIGHT_LEVELS_M,
    "drone_detection_quality": (0.95, 0.75, 0.55),
    "drone_cover_detection_factors": (1.0, 0.95, 0.75, 0.55, 0.45, 0.90),
    "drone_smoke_detection_factor": 0.55,
    "drone_edge_detection_floor": 0.40,
    "drone_energy_costs": (0.0, 0.002, 0.006),
    "drone_safety_clearance_m": DRONE_SAFETY_CLEARANCE_M,
    "comms_dropout":      0.0,
    "r_found_survivor":   1.0,
    "r_drone_scout":      0.3,
    "r_ground_confirm":   0.5,
    "r_time_penalty":  -0.001,
    "r_fire_penalty":    -1.0,
    "r_ground_travel_cost": -0.05,
    "r_drone_climb_cost": -0.02,
    "r_drone_shaping":    0.05,
    "r_ground_shaping":   0.10,
}


def make_wildfire_task(**overrides: Any) -> WildfireVmasClass:
    """Build the BenchMARL task with default scenario config + any overrides."""
    config = {**DEFAULT_CONFIG, **overrides}
    return WildfireVmasClass(name="WILDFIRE_SEARCH", config=config)
