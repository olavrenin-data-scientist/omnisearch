import math
import sys
import types
import unittest

import torch


def _install_vmas_stubs() -> None:
    vmas = types.ModuleType("vmas")
    simulator = types.ModuleType("vmas.simulator")
    core = types.ModuleType("vmas.simulator.core")
    scenario = types.ModuleType("vmas.simulator.scenario")
    sensors = types.ModuleType("vmas.simulator.sensors")
    utils = types.ModuleType("vmas.simulator.utils")

    class Sphere:
        def __init__(self, radius=0.05):
            self._radius = radius

        @property
        def radius(self):
            return self._radius

    class ScenarioUtils:
        pass

    class Color:
        BLUE = "blue"
        GREEN = "green"
        RED = "red"

    core.Agent = type("Agent", (), {})
    core.Landmark = type("Landmark", (), {})
    core.Sphere = Sphere
    core.World = type("World", (), {})
    scenario.BaseScenario = type("BaseScenario", (), {})
    sensors.Lidar = type("Lidar", (), {})
    utils.Color = Color
    utils.ScenarioUtils = ScenarioUtils

    sys.modules.setdefault("vmas", vmas)
    sys.modules.setdefault("vmas.simulator", simulator)
    sys.modules.setdefault("vmas.simulator.core", core)
    sys.modules.setdefault("vmas.simulator.scenario", scenario)
    sys.modules.setdefault("vmas.simulator.sensors", sensors)
    sys.modules.setdefault("vmas.simulator.utils", utils)


_install_vmas_stubs()

from envs.wildfire_search import WildfireSearchScenario  # noqa: E402


class _Entity:
    def __init__(self, radius):
        self.shape = sys.modules["vmas.simulator.core"].Sphere(radius)


class PhysicalUnitConversionTests(unittest.TestCase):
    def _scenario(self):
        scenario = WildfireSearchScenario()
        scenario.agent_radius_sim_override = None
        scenario.survivor_radius_sim_override = None
        scenario.detection_range_sim_override = None
        scenario.agent_radius_m = 0.50
        scenario.survivor_radius_m = 0.35
        scenario.ground_confirmation_range_m = 10.0
        scenario.ground_lidar_range_sim_override = None
        scenario.ground_lidar_range_m = 20.0
        scenario.spawn_padding_m = 1.0
        scenario.ground_min_step_sim_override = None
        scenario.ground_min_step_m = 0.0
        scenario.agent_radius_by_env = torch.zeros(1)
        scenario.survivor_radius_by_env = torch.zeros(1)
        scenario.detection_range_by_env = torch.zeros(1)
        scenario.spawn_padding_by_env = torch.zeros(1)
        ground = _Entity(0.04)
        ground.sensors = [types.SimpleNamespace(_max_range=0.20)]
        scenario.n_drones = 0
        scenario.world = types.SimpleNamespace(agents=[ground])
        scenario._survivors = [_Entity(0.03)]
        return scenario

    def test_dimensions_are_converted_from_meters(self):
        scenario = self._scenario()
        scale = 2.0 / 13_794.918831077

        scenario._refresh_physical_size_conversions(0, scale)

        self.assertAlmostEqual(scenario.agent_radius, 0.50 * scale)
        self.assertAlmostEqual(scenario.survivor_radius, 0.35 * scale)
        self.assertAlmostEqual(scenario.detection_range, 10.0 * scale)
        self.assertAlmostEqual(scenario.world.agents[0].shape.radius, 0.50 * scale)
        self.assertAlmostEqual(scenario._survivors[0].shape.radius, 0.35 * scale)

    def test_drone_footprint_has_no_simulation_unit_floor(self):
        scenario = self._scenario()
        scale = 2.0 / 13_794.918831077
        scenario.drone_altitude = torch.tensor([[50.0 * scale]])
        scenario.drone_camera_half_angle_tan = math.tan(math.radians(90.0) / 2.0)

        footprint = scenario._drone_camera_ranges()

        self.assertAlmostEqual(float(footprint.item()), 50.0 * scale, places=7)
        self.assertLess(float(footprint.item()), 0.12)

    def test_terrain_scale_honors_non_default_equal_semidims(self):
        scenario = self._scenario()
        scenario.world_scale = 2.0
        metadata = {"units": {"sim_units_per_meter": 2.0 / 13_794.918831077}}

        scale = scenario._terrain_sim_units_per_meter(metadata)

        self.assertAlmostEqual(scale, 4.0 / 13_794.918831077)

    def test_ground_sensor_distances_are_converted_from_meters(self):
        scenario = self._scenario()
        scale = 2.0 / 13_794.918831077

        scenario._refresh_ground_sensor_conversions(0, scale)

        self.assertAlmostEqual(scenario.ground_lidar_range, 20.0 * scale)
        self.assertAlmostEqual(float(scenario.spawn_padding_by_env[0]), 1.0 * scale)
        self.assertEqual(scenario.ground_min_step_sim, 0.0)
        self.assertAlmostEqual(scenario.world.agents[0].sensors[0]._max_range, 20.0 * scale)


if __name__ == "__main__":
    unittest.main()
