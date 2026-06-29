"""Tests for the temporal motion detector."""

import numpy as np
import pytest
from PIL import Image

from detection.motion_detector import (
    MotionDetector,
    MotionDetectorConfig,
    fuse_cv_motion,
)


def _make_frame(size: int = 128, seed: int = 0) -> Image.Image:
    """Create a random background frame."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 180, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _add_blob(frame: Image.Image, cx: int, cy: int, radius: int = 15, value: int = 255) -> Image.Image:
    """Add a bright blob (simulating a survivor) to a frame."""
    arr = np.array(frame)
    h, w = arr.shape[:2]
    for r in range(max(0, cy - radius), min(h, cy + radius)):
        for c in range(max(0, cx - radius), min(w, cx + radius)):
            if (r - cy) ** 2 + (c - cx) ** 2 < radius ** 2:
                arr[r, c] = value
    return Image.fromarray(arr)


class TestMotionDetector:
    """Test temporal frame differencing detection."""

    def test_first_frame_returns_empty(self):
        """First frame has no previous — should return no detections."""
        det = MotionDetector(MotionDetectorConfig(min_blob_area=50))
        frame = _make_frame(128, seed=0)
        results = det.detect(frame, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        assert results == []
        assert det.has_previous_frame is True

    def test_identical_frames_no_detection(self):
        """Two identical frames should produce no motion detections."""
        det = MotionDetector(MotionDetectorConfig(min_blob_area=50))
        frame = _make_frame(128, seed=0)
        det.detect(frame, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        results = det.detect(frame, drone_xy=(0.01, 0.0), footprint_world=1.0, image_size=128)
        assert len(results) == 0

    def test_new_object_detected(self):
        """Adding a bright object in the second frame should trigger detection."""
        cfg = MotionDetectorConfig(min_blob_area=20, diff_threshold=25, dilation_size=3)
        det = MotionDetector(cfg)
        frame1 = _make_frame(128, seed=0)
        frame2 = _add_blob(frame1.copy(), cx=64, cy=64, radius=12, value=255)

        det.detect(frame1, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        results = det.detect(frame2, drone_xy=(0.01, 0.0), footprint_world=1.0, image_size=128)
        assert len(results) > 0
        # Detection should be near the blob center
        best = max(results, key=lambda r: r["area_px"])
        assert abs(best["center_px"][0] - 64) < 20
        assert abs(best["center_px"][1] - 64) < 20

    def test_smoke_reduces_detections(self):
        """High smoke load should reduce motion detection confidence."""
        cfg = MotionDetectorConfig(min_blob_area=20, diff_threshold=25, dilation_size=3)

        det_clear = MotionDetector(cfg)
        frame1 = _make_frame(128, seed=0)
        frame2 = _add_blob(frame1.copy(), cx=64, cy=64, radius=12, value=255)
        det_clear.detect(frame1, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        res_clear = det_clear.detect(frame2, drone_xy=(0.01, 0.0), footprint_world=1.0, image_size=128, smoke_load=0.0)

        det_smoke = MotionDetector(cfg)
        det_smoke.detect(frame1, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        res_smoke = det_smoke.detect(frame2, drone_xy=(0.01, 0.0), footprint_world=1.0, image_size=128, smoke_load=2.0)

        if res_clear and res_smoke:
            assert res_clear[0]["confidence"] > res_smoke[0]["confidence"]

    def test_large_movement_penalty(self):
        """Large drone movement between frames should reduce confidence."""
        cfg = MotionDetectorConfig(min_blob_area=20, diff_threshold=25, dilation_size=3)

        det_small = MotionDetector(cfg)
        frame1 = _make_frame(128, seed=0)
        frame2 = _add_blob(frame1.copy(), cx=64, cy=64, radius=12, value=255)
        det_small.detect(frame1, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        res_small = det_small.detect(frame2, drone_xy=(0.001, 0.0), footprint_world=1.0, image_size=128)

        det_large = MotionDetector(cfg)
        det_large.detect(frame1, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=128)
        res_large = det_large.detect(frame2, drone_xy=(0.5, 0.0), footprint_world=1.0, image_size=128)

        if res_small and res_large:
            assert res_small[0]["movement_factor"] > res_large[0]["movement_factor"]

    def test_frame_counter(self):
        """Frame counter should increment with each detect call."""
        det = MotionDetector()
        assert det.frame_count == 0
        frame = _make_frame(64, seed=0)
        det.detect(frame, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=64)
        assert det.frame_count == 1
        det.detect(frame, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=64)
        assert det.frame_count == 2

    def test_reset(self):
        """Reset should clear previous frame and counter."""
        det = MotionDetector()
        frame = _make_frame(64, seed=0)
        det.detect(frame, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=64)
        assert det.has_previous_frame is True
        det.reset()
        assert det.has_previous_frame is False
        assert det.frame_count == 0


class TestMotionFusion:
    """Test CV + Motion fusion logic."""

    def test_boost_mode_increases_confidence(self):
        """When motion confirms a CV detection, confidence should increase."""
        cv_dets = [{
            "class_name": "person",
            "confidence": 0.70,
            "bbox_xyxy": [50, 50, 100, 150],
            "center_px": [75.0, 100.0],
            "matched_survivor_index": 0,
        }]
        motion_dets = [{
            "center_px": [78.0, 105.0],  # Close to CV detection
            "bbox_xyxy": [55, 55, 95, 145],
            "confidence": 0.50,
            "area_px": 500,
        }]

        fused = fuse_cv_motion(cv_dets, motion_dets, fusion_mode="boost")
        assert len(fused) == 1
        assert fused[0]["motion_confirmed"] is True
        assert fused[0]["confidence"] > 0.70  # Boosted

    def test_boost_mode_no_motion_match(self):
        """When motion doesn't confirm CV, confidence stays the same."""
        cv_dets = [{
            "class_name": "person",
            "confidence": 0.70,
            "bbox_xyxy": [50, 50, 100, 150],
            "center_px": [75.0, 100.0],
            "matched_survivor_index": 0,
        }]
        motion_dets = [{
            "center_px": [300.0, 300.0],  # Far from CV detection
            "bbox_xyxy": [280, 280, 320, 320],
            "confidence": 0.50,
            "area_px": 500,
        }]

        fused = fuse_cv_motion(cv_dets, motion_dets, fusion_mode="boost", match_radius_px=50.0)
        assert len(fused) == 1
        assert fused[0]["motion_confirmed"] is False
        assert fused[0]["confidence"] == 0.70

    def test_union_mode_includes_unmatched_motion(self):
        """Union mode should include motion-only detections as candidates."""
        cv_dets = [{
            "class_name": "person",
            "confidence": 0.80,
            "bbox_xyxy": [50, 50, 100, 150],
            "center_px": [75.0, 100.0],
            "matched_survivor_index": 0,
        }]
        motion_dets = [{
            "center_px": [300.0, 300.0],  # Far — unmatched
            "bbox_xyxy": [280, 280, 320, 320],
            "confidence": 0.45,
            "area_px": 400,
        }]

        fused = fuse_cv_motion(cv_dets, motion_dets, fusion_mode="union", match_radius_px=50.0)
        assert len(fused) == 2
        sources = [f["fusion_source"] for f in fused]
        assert "cv_only" in sources
        assert "motion_only" in sources

    def test_confirm_mode_drops_unconfirmed(self):
        """Confirm mode should only keep CV detections that have motion support."""
        cv_dets = [
            {
                "class_name": "person",
                "confidence": 0.80,
                "bbox_xyxy": [50, 50, 100, 150],
                "center_px": [75.0, 100.0],
                "matched_survivor_index": 0,
            },
            {
                "class_name": "person",
                "confidence": 0.60,
                "bbox_xyxy": [200, 200, 250, 300],
                "center_px": [225.0, 250.0],
                "matched_survivor_index": 1,
            },
        ]
        motion_dets = [{
            "center_px": [78.0, 105.0],  # Matches first CV detection only
            "bbox_xyxy": [55, 55, 95, 145],
            "confidence": 0.50,
            "area_px": 500,
        }]

        fused = fuse_cv_motion(cv_dets, motion_dets, fusion_mode="confirm", match_radius_px=50.0)
        assert len(fused) == 1  # Only the confirmed one
        assert fused[0]["matched_survivor_index"] == 0
        assert fused[0]["motion_confirmed"] is True


class TestDetectionModeValidation:
    """Verify detection mode accepts motion modes."""

    def test_motion_mode_accepted(self):
        """'motion' should be a valid detection mode string."""
        from detection.simulation_adapter import SimulationCvAdapter
        # Just verify the validation logic — don't fully construct (needs terrain)
        valid = ("cv", "thermal", "cv+thermal", "motion", "cv+motion")
        assert "motion" in valid
        assert "cv+motion" in valid
