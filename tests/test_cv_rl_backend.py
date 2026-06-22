"""Tests for the CV detection backend in the RL scenario.

Verifies that detection_backend="cv" produces a valid [B, D, S] boolean tensor
and that the abstract backend still works as before.

These tests mock the heavy VMAS dependency and test only the CV detection logic.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The scenario file imports vmas which may not be installed in CI/test envs.
# We test the CV detection logic by extracting the method and running it
# against a lightweight stub, avoiding the full vmas import.
try:
    from envs.wildfire_search import WildfireSearchScenario
    HAS_VMAS = True
except ImportError:
    HAS_VMAS = False

skip_no_vmas = pytest.mark.skipif(not HAS_VMAS, reason="vmas not installed")


def _drone_survivor_detections_cv(self, drone_pos, surv_pos):
    """Extracted CV detection logic (mirrors envs/wildfire_search.py)."""
    if self._cv_adapter is None:
        raise RuntimeError("CV adapter not initialized")

    device = drone_pos.device
    B = self.world.batch_dim
    D = self.n_drones
    S = self.n_survivors
    result = torch.zeros(B, D, S, dtype=torch.bool, device=device)
    self.step_cv_false_positives = 0

    for b in range(B):
        survivors = [
            {"index": s, "world_xy": (float(surv_pos[b, s, 0]), float(surv_pos[b, s, 1]))}
            for s in range(S)
        ]
        for d in range(D):
            altitude_agl = float(self.drone_altitude[b, d])
            det_result = self._cv_adapter.render_and_detect(
                drone=MagicMock(
                    index=d, name=f"drone_{d}",
                    world_xy=(float(drone_pos[b, d, 0]), float(drone_pos[b, d, 1])),
                    altitude_agl=altitude_agl,
                ),
                survivors=survivors,
                wildfire_state=None,
            )
            for det in det_result.get("detections", []):
                matched_idx = det.get("matched_survivor_index")
                if matched_idx is not None and 0 <= matched_idx < S:
                    result[b, d, matched_idx] = True
                else:
                    self.step_cv_false_positives += 1

    return result


class TestCvDetectionLogic:
    """Test CV detection method logic using a minimal mock scenario."""

    def _make_scenario_stub(self, B, D, S):
        """Create a minimal scenario-like object with just the CV fields."""
        scenario = MagicMock()
        scenario.detection_backend = "cv"
        scenario.cv_image_size = 512
        scenario.cv_person_model = None
        scenario.cv_conf_threshold = 0.35
        scenario._cv_adapter = None
        scenario.terrain_cache_path = "data/terrain_cache/malibu_128.npz"
        scenario.n_drones = D
        scenario.n_survivors = S
        scenario.world = MagicMock()
        scenario.world.batch_dim = B
        scenario.drone_altitude = torch.ones(B, D)
        scenario.fire_grid = None
        # Bind the extracted method
        import types
        scenario._drone_survivor_detections_cv = types.MethodType(_drone_survivor_detections_cv, scenario)
        # Mock world.agents
        agents = [MagicMock(name=f"drone_{i}") for i in range(D)]
        for i, a in enumerate(agents):
            a.name = f"drone_{i}"
        scenario.world.agents = agents
        return scenario

    def test_cv_detection_returns_correct_shape(self):
        """_drone_survivor_detections_cv returns [B, D, S] bool tensor."""
        B, D, S = 1, 2, 3
        scenario = self._make_scenario_stub(B, D, S)

        mock_adapter = MagicMock()
        mock_adapter.render_and_detect.return_value = {
            "detections": [
                {"matched_survivor_index": 0, "confidence": 0.85},
                {"matched_survivor_index": None, "confidence": 0.4},
            ],
            "truth": [],
        }
        scenario._cv_adapter = mock_adapter

        drone_pos = torch.rand(B, D, 2)
        surv_pos = torch.rand(B, S, 2)

        result = scenario._drone_survivor_detections_cv(drone_pos, surv_pos)

        assert result.shape == (B, D, S)
        assert result.dtype == torch.bool
        assert result[0, 0, 0].item() is True
        assert result[0, 0, 1].item() is False
        assert result[0, 0, 2].item() is False
        assert scenario.step_cv_false_positives == D  # 1 FP per drone

    def test_cv_detection_handles_no_detections(self):
        """Returns all-False when YOLO finds nothing."""
        B, D, S = 1, 2, 4
        scenario = self._make_scenario_stub(B, D, S)

        mock_adapter = MagicMock()
        mock_adapter.render_and_detect.return_value = {"detections": [], "truth": []}
        scenario._cv_adapter = mock_adapter

        result = scenario._drone_survivor_detections_cv(torch.rand(B, D, 2), torch.rand(B, S, 2))

        assert result.shape == (B, D, S)
        assert not result.any()
        assert scenario.step_cv_false_positives == 0

    def test_cv_detection_multi_batch(self):
        """Processes all batch envs and all drones."""
        B, D, S = 3, 2, 2
        scenario = self._make_scenario_stub(B, D, S)

        call_count = [0]

        def mock_detect(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return {"detections": [{"matched_survivor_index": 1, "confidence": 0.9}], "truth": []}
            return {"detections": [], "truth": []}

        mock_adapter = MagicMock()
        mock_adapter.render_and_detect.side_effect = mock_detect
        scenario._cv_adapter = mock_adapter

        result = scenario._drone_survivor_detections_cv(torch.rand(B, D, 2), torch.rand(B, S, 2))

        assert result.shape == (B, D, S)
        assert mock_adapter.render_and_detect.call_count == B * D

    def test_cv_detection_out_of_range_index_ignored(self):
        """Survivor indices outside [0, S) are treated as false positives."""
        B, D, S = 1, 1, 2
        scenario = self._make_scenario_stub(B, D, S)

        mock_adapter = MagicMock()
        mock_adapter.render_and_detect.return_value = {
            "detections": [
                {"matched_survivor_index": 99, "confidence": 0.7},
                {"matched_survivor_index": -1, "confidence": 0.5},
            ],
            "truth": [],
        }
        scenario._cv_adapter = mock_adapter

        result = scenario._drone_survivor_detections_cv(torch.rand(B, D, 2), torch.rand(B, S, 2))

        assert not result.any()
        assert scenario.step_cv_false_positives == 2


@skip_no_vmas
class TestCvBackendIntegration:
    """Full integration tests requiring vmas."""

    def test_abstract_backend_default(self):
        """Default detection_backend is abstract."""
        scenario = WildfireSearchScenario()
        scenario.make_world(batch_dim=1, device=torch.device("cpu"),
                           n_drones=2, n_ground=1, n_survivors=3)
        assert scenario.detection_backend == "abstract"

    def test_cv_backend_init(self):
        """CV backend kwarg stores correctly."""
        scenario = WildfireSearchScenario()
        scenario.make_world(batch_dim=1, device=torch.device("cpu"),
                           n_drones=2, n_ground=1, n_survivors=3,
                           detection_backend="cv",
                           cv_image_size=1024,
                           cv_conf_threshold=0.5)
        assert scenario.detection_backend == "cv"
        assert scenario.cv_image_size == 1024
        assert scenario.cv_conf_threshold == 0.5
        assert scenario._cv_adapter is None

    def test_invalid_backend_raises(self):
        """Invalid detection_backend raises ValueError."""
        scenario = WildfireSearchScenario()
        with pytest.raises(ValueError, match="detection_backend"):
            scenario.make_world(batch_dim=1, device=torch.device("cpu"),
                               n_drones=2, n_ground=1, n_survivors=3,
                               detection_backend="bogus")
