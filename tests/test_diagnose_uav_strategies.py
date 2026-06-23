import unittest
from pathlib import Path

from agents.baselines import BASELINES
from scripts.diagnose_uav_strategies import parse_strategy_specs


class DiagnoseUavStrategiesTests(unittest.TestCase):
    def test_all_expands_to_every_baseline(self):
        specs = parse_strategy_specs(["all"])

        self.assertEqual([spec.baseline_name for spec in specs], list(BASELINES))
        self.assertTrue(all(spec.kind == "baseline" for spec in specs))

    def test_named_baselines_are_preserved(self):
        specs = parse_strategy_specs(["lawnmower", "ant_colony"])

        self.assertEqual([spec.label for spec in specs], ["lawnmower", "ant_colony"])
        self.assertEqual([spec.baseline_name for spec in specs], ["lawnmower", "ant_colony"])

    def test_happo_path_strategy_is_supported(self):
        specs = parse_strategy_specs(["happo:/tmp/models"])

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].label, "happo")
        self.assertEqual(specs[0].kind, "happo")
        self.assertEqual(specs[0].checkpoint_dir, Path("/tmp/models"))

    def test_duplicate_labels_are_made_unique(self):
        specs = parse_strategy_specs(["happo:/tmp/a", "happo:/tmp/b"])

        self.assertEqual([spec.label for spec in specs], ["happo", "happo_2"])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            parse_strategy_specs(["not_a_strategy"])
