import unittest

from terrain.usgs_osm_builder import _validate_square_projected_bounds


class SquareBboxTests(unittest.TestCase):
    def test_accepts_square_projected_bounds(self):
        _validate_square_projected_bounds((0.0, 0.0, 1000.0, 990.0), tolerance=0.02)

    def test_rejects_rectangular_projected_bounds(self):
        with self.assertRaisesRegex(ValueError, "bbox must be square"):
            _validate_square_projected_bounds((0.0, 0.0, 1000.0, 800.0), tolerance=0.02)

    def test_rejects_invalid_tolerance(self):
        with self.assertRaisesRegex(ValueError, "square_bbox_tolerance"):
            _validate_square_projected_bounds((0.0, 0.0, 1000.0, 1000.0), tolerance=1.0)


if __name__ == "__main__":
    unittest.main()
