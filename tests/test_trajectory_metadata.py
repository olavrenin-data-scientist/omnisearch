import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "trajectory_metadata.py"
SPEC = importlib.util.spec_from_file_location("trajectory_metadata", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
trajectory_timing = MODULE.trajectory_timing


class TrajectoryMetadataTests(unittest.TestCase):
    def test_stride_uses_simulation_step_not_recorded_frame_count(self):
        frames = [{"step": step} for step in (0, 3, 6, 9, 1_000)]

        timing = trajectory_timing(frames, sim_step_seconds=2.0)

        self.assertEqual(timing["actual_n_steps"], 1_000)
        self.assertEqual(timing["recorded_frame_count"], 5)
        self.assertEqual(timing["actual_duration_seconds"], 2_000.0)

    def test_empty_frames_have_zero_duration(self):
        timing = trajectory_timing([], sim_step_seconds=2.0)

        self.assertEqual(timing["actual_n_steps"], 0)
        self.assertEqual(timing["recorded_frame_count"], 0)
        self.assertEqual(timing["actual_duration_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
