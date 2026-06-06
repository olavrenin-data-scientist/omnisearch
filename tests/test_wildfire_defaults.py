import unittest
from pathlib import Path

from envs.wildfire_defaults import (
    DRONE_CAMERA_FOV_DEG,
    DRONE_FLIGHT_LEVELS_M,
    DRONE_SAFETY_CLEARANCE_M,
    DRONE_SPEED_MPS,
    GROUND_ACCEL_MPS2,
    GROUND_SPEED_MPS,
)


ROOT = Path(__file__).resolve().parent.parent


class WildfireDefaultsTests(unittest.TestCase):
    def test_happo_operational_defaults_are_canonical(self):
        self.assertEqual(DRONE_FLIGHT_LEVELS_M, (20.0, 35.0, 50.0))
        self.assertEqual(DRONE_CAMERA_FOV_DEG, 90.0)
        self.assertEqual(DRONE_SAFETY_CLEARANCE_M, 3.0)
        self.assertEqual(DRONE_SPEED_MPS, 10.0)
        self.assertEqual(GROUND_SPEED_MPS, 1.6)
        self.assertEqual(GROUND_ACCEL_MPS2, 2.0)

    def test_entrypoints_import_shared_defaults(self):
        sources = {
            "scenario": ROOT / "envs" / "wildfire_search.py",
            "benchmarl": ROOT / "agents" / "wildfire_task.py",
            "exporter": ROOT / "scripts" / "export_trajectories.py",
        }
        for name, path in sources.items():
            with self.subTest(entrypoint=name):
                self.assertIn(
                    "from envs.wildfire_defaults import",
                    path.read_text(encoding="utf-8"),
                )

        benchmarl_source = sources["benchmarl"].read_text(encoding="utf-8")
        exporter_source = sources["exporter"].read_text(encoding="utf-8")
        self.assertIn('"drone_flight_levels_m": DRONE_FLIGHT_LEVELS_M', benchmarl_source)
        self.assertIn('"drone_camera_fov_deg": DRONE_CAMERA_FOV_DEG', benchmarl_source)
        self.assertIn('"drone_safety_clearance_m": DRONE_SAFETY_CLEARANCE_M', benchmarl_source)
        self.assertIn("default=DRONE_FLIGHT_LEVELS_M", exporter_source)
        self.assertIn("default=DRONE_CAMERA_FOV_DEG", exporter_source)
        self.assertIn("default=DRONE_SAFETY_CLEARANCE_M", exporter_source)


if __name__ == "__main__":
    unittest.main()
