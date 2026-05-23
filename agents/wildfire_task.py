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
        return self.config.get("max_steps", 200)


class WildfireTask(Task):
    """Single-task enum for OmniSearch's wildfire search environment."""

    WILDFIRE_SEARCH = None

    @staticmethod
    def associated_class():
        return WildfireVmasClass


# Default scenario kwargs for a small/fast smoke training run.
DEFAULT_CONFIG: Dict[str, Any] = {
    "max_steps":          200,
    "n_drones":             3,
    "n_ground":             2,
    "n_survivors":          5,
    "x_semidim":          1.0,
    "y_semidim":          1.0,
    "drone_lidar_range":  0.50,
    "ground_lidar_range": 0.20,
    "n_lidar_rays":        12,
    "detection_range":   0.10,
    "fire_grid_size":      16,
    "fire_spread_prob":  0.04,
    "initial_fire_cells":   1,
    "fire_step_interval":   5,
    "comms_dropout":      0.0,
    "r_found_survivor":   1.0,
    "r_drone_scout":      0.3,
    "r_ground_confirm":   0.5,
    "r_time_penalty":  -0.001,
    "r_fire_penalty":    -1.0,
}


def make_wildfire_task(**overrides: Any) -> WildfireVmasClass:
    """Build the BenchMARL task with default scenario config + any overrides."""
    config = {**DEFAULT_CONFIG, **overrides}
    return WildfireVmasClass(name="WILDFIRE_SEARCH", config=config)
