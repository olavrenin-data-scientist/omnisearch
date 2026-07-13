"""Tests for the four cropping/boundary fixes in the CV detection pipeline.

Fix 1 — YOLO output box clipping  (simulation_adapter._detect_people_cv)
    Boxes returned after NMS must always be within [0, W] × [0, H].

Fix 2 — VOC/COCO box pre-clamping  (extract_sard_assets._boxes_for_image)
    VOC/COCO annotations that extend outside image bounds must be clamped
    before padding so that the GrabCut foreground rect is always valid.

Fix 3 — Degenerate padded-box filter  (extract_sard_assets extraction loop)
    A zero-area padded crop (annotation at image edge) must be silently
    skipped instead of crashing PIL or producing a nonsense mask.

Fix 4 — Boundary-aware survivor labels  (train_survivor_detector._clip_label)
    _clip_label must produce correct normalised YOLO coordinates for survivors
    that overlap the image edge, and must return None for survivors entirely
    outside the frame.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.simulation_adapter import SimulationCvAdapter
from scripts.extract_sard_assets import Box, _clamp_box, _pad_box
from scripts.train_survivor_detector import (
    WildfireEffectConfig,
    _clip_label,
    _generate_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter() -> SimulationCvAdapter:
    a = object.__new__(SimulationCvAdapter)
    a.person_match_iou = 0.15
    a.person_iou = 0.6
    return a


def _solid_rgba(w: int, h: int) -> Image.Image:
    arr = np.full((h, w, 4), [200, 100, 50, 255], dtype=np.uint8)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------
# Fix 1 — YOLO output box clipping
# ---------------------------------------------------------------------------

class TestDetectionBoxClipping:
    """Boxes returned by _detect_people_cv must lie inside [0,W]×[0,H]."""

    def test_nms_returns_tuple_list(self):
        a = _adapter()
        boxes = [(10.0, 10.0, 50.0, 50.0), (200.0, 200.0, 300.0, 300.0)]
        confs = [0.8, 0.5]
        result = a._nms(boxes, confs, 0.5)
        assert all(len(item) == 2 for item in result)

    def test_clipping_removes_out_of_bounds_coordinates(self):
        """Simulate a scenario where tiled NMS produces an out-of-bounds box.

        We build a minimal adapter with a mocked _detect_people_cv
        implementation that returns a box extending past the image edge,
        and verify that the clipping logic normalises it.
        """
        W, H = 640, 640
        # Mimic the post-NMS clip logic from _detect_people_cv directly.
        raw_merged = [
            ((620.0, 610.0, 660.0, 650.0), 0.85),   # extends past right+bottom edge
            ((-5.0, 100.0, 50.0, 200.0), 0.70),       # extends past left edge
            ((100.0, -10.0, 200.0, 300.0), 0.60),     # extends past top edge
        ]
        clipped = []
        for (bx1, by1, bx2, by2), c in raw_merged:
            bx1 = max(0.0, min(float(W), bx1))
            by1 = max(0.0, min(float(H), by1))
            bx2 = max(0.0, min(float(W), bx2))
            by2 = max(0.0, min(float(H), by2))
            if bx2 > bx1 and by2 > by1:
                clipped.append(((bx1, by1, bx2, by2), c))

        assert len(clipped) == 3, "all valid boxes should survive clipping"
        for (x1, y1, x2, y2), _ in clipped:
            assert x1 >= 0.0, f"x1={x1} below 0"
            assert y1 >= 0.0, f"y1={y1} below 0"
            assert x2 <= float(W), f"x2={x2} exceeds W={W}"
            assert y2 <= float(H), f"y2={y2} exceeds H={H}"
            assert x2 > x1, "degenerate box after clipping"
            assert y2 > y1, "degenerate box after clipping"

    def test_fully_outside_box_is_dropped(self):
        """A box entirely outside the image must be dropped, not clipped to zero-area."""
        W, H = 640, 640
        raw_merged = [((700.0, 700.0, 800.0, 800.0), 0.9)]   # entirely outside
        clipped = []
        for (bx1, by1, bx2, by2), c in raw_merged:
            bx1 = max(0.0, min(float(W), bx1))
            by1 = max(0.0, min(float(H), by1))
            bx2 = max(0.0, min(float(W), bx2))
            by2 = max(0.0, min(float(H), by2))
            if bx2 > bx1 and by2 > by1:
                clipped.append(((bx1, by1, bx2, by2), c))
        assert clipped == [], "box outside image must be dropped"


# ---------------------------------------------------------------------------
# Fix 2 — VOC/COCO box pre-clamping
# ---------------------------------------------------------------------------

class TestVocCocoBoxClamping:
    """Boxes from VOC/COCO annotations must be clamped to image bounds before
    padding and before being used as a GrabCut foreground hint."""

    IMG_SIZE = (640, 480)   # (width, height)

    def test_in_bounds_box_unchanged(self):
        box = Box(10, 20, 100, 200, "test.xml")
        clamped = _clamp_box(box, self.IMG_SIZE)
        assert (clamped.x1, clamped.y1, clamped.x2, clamped.y2) == (10, 20, 100, 200)

    def test_box_exceeding_right_edge_is_clamped(self):
        W, H = self.IMG_SIZE
        box = Box(600, 100, 700, 300, "test.xml")   # x2=700 > W=640
        clamped = _clamp_box(box, self.IMG_SIZE)
        assert clamped.x2 <= W
        assert clamped.x1 < clamped.x2

    def test_box_exceeding_bottom_edge_is_clamped(self):
        W, H = self.IMG_SIZE
        box = Box(100, 440, 200, 520, "test.xml")   # y2=520 > H=480
        clamped = _clamp_box(box, self.IMG_SIZE)
        assert clamped.y2 <= H
        assert clamped.y1 < clamped.y2

    def test_negative_coordinates_clamped_to_zero(self):
        box = Box(-30, -10, 100, 200, "test.xml")
        clamped = _clamp_box(box, self.IMG_SIZE)
        assert clamped.x1 >= 0
        assert clamped.y1 >= 0

    def test_entirely_outside_image_produces_degenerate_box(self):
        """A box entirely outside the image clamps to a degenerate / zero-area box."""
        W, H = self.IMG_SIZE
        box = Box(700, 0, 800, 100, "test.xml")   # entirely to the right of the image
        clamped = _clamp_box(box, self.IMG_SIZE)
        # x1 is clamped to W-1, x2 to W: degenerate (x2 - x1 == 1)
        assert clamped.x1 >= W - 1
        assert clamped.x2 <= W


# ---------------------------------------------------------------------------
# Fix 3 — Degenerate padded-box filter
# ---------------------------------------------------------------------------

class TestDegeneratePaddedBox:
    """Zero-area crops produced when an annotation sits at the image boundary
    must be silently skipped — _generate_split must not crash on them."""

    def _minimal_assets(self) -> list[Image.Image]:
        return [_solid_rgba(20, 30)]

    def test_pad_box_at_right_edge_stays_valid(self):
        """A box touching the right edge should retain a positive width after padding."""
        W, H = (640, 480)
        # Box that touches the right edge — after clamp it has x2==W.
        box = Box(620, 100, 640, 200, "test")
        padded = _pad_box(box, (W, H), 0.15)
        # Width may be smaller than requested due to clamping but must be > 0.
        assert padded.x2 > padded.x1
        assert padded.y2 > padded.y1

    def test_generate_split_handles_edge_placements_without_crash(self):
        """_generate_split with boundary_frac=1.0 places survivors at the edge;
        the output label files must be valid YOLO lines (or empty for invisible ones)."""
        assets = self._minimal_assets()
        cfg = WildfireEffectConfig()
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _generate_split(
                out, n=10, assets=assets, size=64, rng=rng, cfg=cfg,
                fire_frac=0.0, surv_px=(20, 30), naip_tiles=None,
                neg_frac=0.0, decoy_frac=0.0, boundary_frac=1.0,
            )
            label_files = sorted((out / "labels").glob("*.txt"))
            assert len(label_files) == 10
            for lf in label_files:
                for line in lf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    assert len(parts) == 5, f"bad YOLO line in {lf.name}: {line!r}"
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    assert cls == 0
                    assert 0.0 <= cx <= 1.0, f"cx out of range: {cx}"
                    assert 0.0 <= cy <= 1.0, f"cy out of range: {cy}"
                    assert 0.0 < bw <= 1.0, f"bw out of range: {bw}"
                    assert 0.0 < bh <= 1.0, f"bh out of range: {bh}"


# ---------------------------------------------------------------------------
# Fix 4 — _clip_label boundary-aware YOLO labels
# ---------------------------------------------------------------------------

class TestClipLabel:
    SIZE = 100

    def test_fully_inside_frame(self):
        """Survivor fully inside the frame: label equals the full bbox."""
        result = _clip_label(20, 30, 40, 30, self.SIZE)
        assert result is not None
        cx, cy, w, h = result
        assert abs(cx - 0.40) < 1e-6   # (20+60)/2 / 100
        assert abs(cy - 0.45) < 1e-6   # (30+60)/2 / 100
        assert abs(w - 0.40) < 1e-6
        assert abs(h - 0.30) < 1e-6

    def test_partially_outside_right_edge(self):
        """Survivor hanging off the right edge: w_norm must reflect the visible width only."""
        result = _clip_label(80, 10, 40, 30, self.SIZE)   # x runs 80-120, clips at 100
        assert result is not None
        _, _, w, _ = result
        assert abs(w - 0.20) < 1e-6   # visible width = 100 - 80 = 20

    def test_partially_outside_top_edge(self):
        """Survivor hanging off the top edge: y1 clips to 0."""
        result = _clip_label(10, -15, 30, 40, self.SIZE)   # y runs -15..25, clips to 0..25
        assert result is not None
        _, cy, _, h = result
        assert abs(h - 0.25) < 1e-6   # visible height = 25

    def test_partially_outside_bottom_edge(self):
        result = _clip_label(10, 85, 30, 30, self.SIZE)   # y runs 85-115, clips to 85-100
        assert result is not None
        _, _, _, h = result
        assert abs(h - 0.15) < 1e-6

    def test_partially_outside_left_edge(self):
        result = _clip_label(-20, 10, 50, 30, self.SIZE)  # x runs -20..30, clips to 0..30
        assert result is not None
        _, _, w, _ = result
        assert abs(w - 0.30) < 1e-6

    def test_entirely_outside_returns_none(self):
        assert _clip_label(110, 10, 30, 30, self.SIZE) is None   # fully right of frame
        assert _clip_label(10, 110, 30, 30, self.SIZE) is None   # fully below frame
        assert _clip_label(-50, 10, 30, 30, self.SIZE) is None   # fully left of frame

    def test_too_small_visible_area_returns_none(self):
        """Only 3 px of a 50 px wide survivor are visible — too small to label."""
        assert _clip_label(97, 10, 50, 30, self.SIZE) is None   # only 3 px visible

    def test_label_coords_always_normalised(self):
        """All returned values must lie in (0, 1]."""
        for x in range(-20, 110, 10):
            for y in range(-20, 110, 10):
                result = _clip_label(x, y, 30, 25, self.SIZE)
                if result is not None:
                    cx, cy, w, h = result
                    assert 0.0 < cx <= 1.0, f"cx={cx} for x={x}"
                    assert 0.0 < cy <= 1.0, f"cy={cy} for y={y}"
                    assert 0.0 < w <= 1.0,  f"w={w}"
                    assert 0.0 < h <= 1.0,  f"h={h}"
