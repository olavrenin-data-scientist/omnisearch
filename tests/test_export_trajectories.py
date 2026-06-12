import unittest

from agents.baselines import BASELINES
from scripts.export_trajectories import _selected_baselines


class ExportTrajectorySelectionTests(unittest.TestCase):
    def test_all_selects_every_baseline(self):
        self.assertEqual(_selected_baselines("all"), list(BASELINES))

    def test_happo_selects_no_baselines(self):
        self.assertEqual(_selected_baselines("happo"), [])

    def test_named_baseline_selects_only_that_baseline(self):
        baseline = next(iter(BASELINES))
        self.assertEqual(_selected_baselines(baseline), [baseline])


if __name__ == "__main__":
    unittest.main()
