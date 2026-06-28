"""Tests for the compositing helpers and hard-negative pipeline added to
train_survivor_detector.py.

Covers:
  - _synthetic_decoy_rgba        : produces a valid RGBA image with non-trivial alpha
  - _erode_alpha_edge            : softens hard edges (mean alpha decreases)
  - _harmonize_color             : shifts survivor mean toward background mean
  - _resolution_blur             : blurs RGB (sharpness decreases)
  - _generate_split (neg_frac=1) : 100% negative images → all label files are empty
  - _generate_split (decoy_frac) : decoy images are composited but produce no label
  - Decoy-on-terrain detector    : OPTIONAL — with model weights, a synthetic decoy
                                   pasted on a procedural background must NOT match
                                   any ground-truth survivor box.

All tests are model-free and run without any downloads.  The detector test is
automatically skipped if no fine-tuned weights are present.
"""

from __future__ import annotations

import glob
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
_FINETUNED = next(
    (ROOT / "models" / n for n in ("survivor_naip_yolov8s.pt", "survivor_yolov8s.pt", "survivor_yolov8n.pt")
     if (ROOT / "models" / n).exists()),
    None,
)

# Import compositing helpers from the training script.
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_survivor_detector import (
    _erode_alpha_edge,
    _harmonize_color,
    _resolution_blur,
    _synthetic_decoy_rgba,
    _generate_split,
    _procedural_background,
    altitude_to_survivor_px,
    altitude_to_gsd,
    sample_altitude,
    DRONE_FLIGHT_LEVELS_M,
    DRONE_CAMERA_FOV_DEG,
    SURVIVOR_BODY_WIDTH_M,
    WildfireEffectConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _solid_rgba(w: int, h: int, r: int, g: int, b: int, a: int = 255) -> Image.Image:
    arr = np.full((h, w, 4), [r, g, b, a], dtype=np.uint8)
    return Image.fromarray(arr, "RGBA")


def _sharpness(img: Image.Image) -> float:
    """Mean absolute gradient magnitude — proxy for sharpness."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    return float(gx.mean() + gy.mean())


# ---------------------------------------------------------------------------
# _synthetic_decoy_rgba
# ---------------------------------------------------------------------------

class TestSyntheticDecoyRgba:
    def test_output_size_matches_request(self):
        rng = _rng()
        decoy = _synthetic_decoy_rgba(80, 60, rng)
        assert decoy.size == (80, 60)

    def test_output_is_rgba(self):
        rng = _rng()
        decoy = _synthetic_decoy_rgba(64, 64, rng)
        assert decoy.mode == "RGBA"

    def test_alpha_channel_is_non_trivial(self):
        """Blob mask must cover a meaningful fraction of the crop — not blank, not fully opaque."""
        rng = _rng(42)
        decoy = _synthetic_decoy_rgba(100, 80, rng)
        alpha = np.asarray(decoy.getchannel("A"), dtype=np.float32)
        fg_frac = (alpha > 127).mean()
        assert 0.05 < fg_frac < 0.95, f"alpha foreground fraction {fg_frac:.2f} out of expected range"

    def test_rgb_in_earth_tone_range(self):
        """All four archetypes should stay in muted earth tones (no neon colours)."""
        for seed in range(8):
            decoy = _synthetic_decoy_rgba(60, 60, _rng(seed))
            rgb = np.asarray(decoy.convert("RGB"), dtype=np.float32)
            alpha = np.asarray(decoy.getchannel("A")) > 127
            if alpha.sum() == 0:
                continue
            mean_sat = float(rgb[alpha].std(axis=1).mean())   # rough saturation proxy
            assert mean_sat < 80, f"seed {seed}: foreground looks too saturated ({mean_sat:.1f})"

    def test_different_seeds_produce_different_shapes(self):
        d1 = _synthetic_decoy_rgba(60, 60, _rng(0))
        d2 = _synthetic_decoy_rgba(60, 60, _rng(99))
        a1 = np.asarray(d1.getchannel("A"))
        a2 = np.asarray(d2.getchannel("A"))
        # They should not be pixel-identical.
        assert not np.array_equal(a1, a2)


# ---------------------------------------------------------------------------
# _erode_alpha_edge
# ---------------------------------------------------------------------------

class TestErodeAlphaEdge:
    def test_output_size_unchanged(self):
        sprite = _solid_rgba(50, 80, 200, 100, 50)
        rng = _rng()
        result = _erode_alpha_edge(sprite, rng)
        assert result.size == sprite.size

    def test_output_is_rgba(self):
        sprite = _solid_rgba(40, 40, 100, 100, 100)
        result = _erode_alpha_edge(sprite, _rng())
        assert result.mode == "RGBA"

    def test_mean_alpha_does_not_increase(self):
        """Erosion + blur should reduce or maintain mean alpha, never increase it."""
        sprite = _solid_rgba(60, 60, 150, 100, 80, a=255)
        rng = _rng(7)
        result = _erode_alpha_edge(sprite, rng)
        before = np.asarray(sprite.getchannel("A"), dtype=np.float32).mean()
        after = np.asarray(result.getchannel("A"), dtype=np.float32).mean()
        assert after <= before + 1.0, f"alpha increased from {before:.1f} to {after:.1f}"

    def test_rgb_channels_preserved(self):
        """RGB must not change — only the alpha channel is modified."""
        sprite = _solid_rgba(50, 50, 123, 200, 77)
        result = _erode_alpha_edge(sprite, _rng())
        before_rgb = np.asarray(sprite.convert("RGB"))
        after_rgb = np.asarray(result.convert("RGB"))
        assert np.array_equal(before_rgb, after_rgb)


# ---------------------------------------------------------------------------
# _harmonize_color
# ---------------------------------------------------------------------------

class TestHarmonizeColor:
    def test_output_size_unchanged(self):
        sprite = _solid_rgba(40, 60, 200, 50, 50)
        bg_patch = Image.fromarray(np.full((60, 40, 3), [80, 80, 80], dtype=np.uint8))
        result = _harmonize_color(sprite, bg_patch, _rng())
        assert result.size == sprite.size

    def test_output_is_rgba(self):
        sprite = _solid_rgba(30, 30, 200, 50, 50)
        bg_patch = Image.fromarray(np.full((30, 30, 3), [100, 100, 100], dtype=np.uint8))
        result = _harmonize_color(sprite, bg_patch, _rng())
        assert result.mode == "RGBA"

    def test_mean_shifted_toward_background(self):
        """If the survivor is bright red and the background is dark gray, the
        harmonized survivor's mean red channel should be lower than the original."""
        sprite = _solid_rgba(50, 50, 240, 10, 10, a=255)   # very bright red foreground
        bg_patch = Image.fromarray(np.full((50, 50, 3), [40, 40, 40], dtype=np.uint8))
        result = _harmonize_color(sprite, bg_patch, np.random.default_rng(0))
        orig_r = float(np.asarray(sprite.convert("RGB"))[:, :, 0].mean())
        new_r = float(np.asarray(result.convert("RGB"))[:, :, 0].mean())
        assert new_r < orig_r, (
            f"Red channel did not shift toward darker background: {orig_r:.1f} → {new_r:.1f}"
        )


# ---------------------------------------------------------------------------
# _resolution_blur
# ---------------------------------------------------------------------------

class TestResolutionBlur:
    def test_output_size_unchanged(self):
        sprite = _solid_rgba(80, 80, 100, 100, 100)
        result = _resolution_blur(sprite, 80, _rng())
        assert result.size == sprite.size

    def test_output_is_rgba(self):
        sprite = _solid_rgba(60, 60, 100, 100, 100)
        result = _resolution_blur(sprite, 60, _rng())
        assert result.mode == "RGBA"

    def test_small_survivor_gets_blurred(self):
        """A 15 px wide survivor (tiny at aerial scale) should be blurred."""
        # Checkerboard: maximum high-frequency content → easily measurable sharpness drop.
        arr = np.indices((64, 64)).sum(axis=0) % 2   # 0/1 checkerboard
        rgb_arr = np.stack([arr * 200] * 3, axis=-1).astype(np.uint8)
        alpha_arr = np.full((64, 64), 255, dtype=np.uint8)
        rgba_arr = np.dstack([rgb_arr, alpha_arr])
        sprite = Image.fromarray(rgba_arr, "RGBA")
        rng = np.random.default_rng(0)  # seed 0 → blur radius in [0.9, 1.8] for w<30
        result = _resolution_blur(sprite, 15, rng)
        assert _sharpness(result.convert("RGB")) < _sharpness(sprite.convert("RGB")), (
            "Small survivor was not blurred by _resolution_blur"
        )

    def test_large_survivor_may_be_unblurred(self):
        """For large survivors (≥200 px) blur radius can be 0 — no assertion needed,
        just verify no crash and correct output type/size."""
        sprite = _solid_rgba(200, 200, 100, 150, 80)
        rng = np.random.default_rng(99)
        result = _resolution_blur(sprite, 200, rng)
        assert result.size == (200, 200)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# _generate_split: negative images produce empty label files
# ---------------------------------------------------------------------------

class TestGenerateSplitNegatives:
    def _minimal_assets(self, n: int = 3) -> list[Image.Image]:
        """Generate tiny solid-color RGBA assets as a stand-in for SARD PNGs."""
        return [_solid_rgba(20, 30, 180 - i * 20, 100, 80 + i * 10) for i in range(n)]

    def test_all_negatives_produce_empty_label_files(self):
        """When neg_frac=1.0, every label file must be empty (YOLO negative convention)."""
        assets = self._minimal_assets()
        cfg = WildfireEffectConfig()
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _generate_split(out, n=8, assets=assets, size=64, rng=rng, cfg=cfg,
                            fire_frac=0.0, surv_px=(10, 20), naip_tiles=None,
                            neg_frac=1.0, decoy_frac=0.0)
            label_files = sorted((out / "labels").glob("*.txt"))
            assert len(label_files) == 8
            for lf in label_files:
                content = lf.read_text(encoding="utf-8").strip()
                assert content == "", f"{lf.name} should be empty for a negative image, got: {content!r}"

    def test_positive_images_produce_non_empty_label_files(self):
        """When neg_frac=0.0, every label file must have at least one annotation."""
        assets = self._minimal_assets()
        cfg = WildfireEffectConfig()
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _generate_split(out, n=6, assets=assets, size=64, rng=rng, cfg=cfg,
                            fire_frac=0.0, surv_px=(10, 20), naip_tiles=None,
                            neg_frac=0.0, decoy_frac=0.0)
            label_files = sorted((out / "labels").glob("*.txt"))
            assert len(label_files) == 6
            for lf in label_files:
                content = lf.read_text(encoding="utf-8").strip()
                assert content != "", f"{lf.name} should have at least one label for a positive image"

    def test_decoy_images_never_add_labels(self):
        """When decoy_frac=1.0 and neg_frac=0.0, decoy objects are pasted but
        must never appear in the label files (they are intentionally unlabeled)."""
        assets = self._minimal_assets()
        cfg = WildfireEffectConfig()
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _generate_split(out, n=6, assets=assets, size=64, rng=rng, cfg=cfg,
                            fire_frac=0.0, surv_px=(10, 20), naip_tiles=None,
                            neg_frac=0.0, decoy_frac=1.0)
            label_files = sorted((out / "labels").glob("*.txt"))
            for lf in label_files:
                lines = [l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
                # Labels must only be class 0 (person) — no extra entries from decoys.
                for line in lines:
                    cls = int(line.split()[0])
                    assert cls == 0, f"Unexpected class {cls} in {lf.name} — decoy was labeled!"


# ---------------------------------------------------------------------------
# Detector test: synthetic decoy on background should NOT match a survivor box
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _FINETUNED is None,
    reason="fine-tuned survivor weights not present (run scripts/train_survivor_detector.py)",
)
def test_decoy_does_not_match_survivor_ground_truth():
    """A synthetic non-human decoy pasted on a procedural background should not
    be matched to any survivor ground-truth box.

    The decoy occupies a known region; we assert no YOLO detection overlaps it
    with IoU ≥ 0.15 at the ground-truth box (i.e. the model does not fire a
    labeled survivor detection on it).  Isolated false positives are allowed —
    the test only checks that the ground-truth SURVIVOR box logic is not
    contaminated.
    """
    from detection.simulation_adapter import SimulationCvAdapter

    SZ = 640
    rng = np.random.default_rng(42)

    # Procedural background (no NAIP needed).
    bg = _procedural_background(SZ, rng)

    # Paste a synthetic decoy in the centre.
    dw, dh = 80, 55
    dx, dy = SZ // 2 - dw // 2, SZ // 2 - dh // 2
    decoy = _synthetic_decoy_rgba(dw, dh, rng)
    bg.paste(decoy, (dx, dy), decoy)

    # Ground-truth box: the exact region where the decoy was placed.
    decoy_box = (dx, dy, dx + dw, dy + dh)

    # Run the YOLO detector.
    det = object.__new__(SimulationCvAdapter)
    det.detector_backend = "yolo"
    det.person_model_name = str(_FINETUNED)
    det.person_conf = 0.30
    det.person_iou = 0.6
    det.person_imgsz = 640
    det.person_tiled = False
    det.person_tile_grid = 2
    det.person_tile_overlap = 0.25
    det.person_match_iou = 0.15
    det.person_device = None
    det.person_augment = False
    det._person_detector = None
    det.image_size = SZ

    dets = det._detect_people_cv(bg)

    # None of the detections should overlap the decoy box at match_iou ≥ 0.15.
    for (bx1, by1, bx2, by2), conf in dets:
        inter_x = max(0, min(bx2, decoy_box[2]) - max(bx1, decoy_box[0]))
        inter_y = max(0, min(by2, decoy_box[3]) - max(by1, decoy_box[1]))
        inter = inter_x * inter_y
        area_det = max(1, (bx2 - bx1) * (by2 - by1))
        area_gt = max(1, (decoy_box[2] - decoy_box[0]) * (decoy_box[3] - decoy_box[1]))
        iou = inter / (area_det + area_gt - inter)
        assert iou < det.person_match_iou, (
            f"Detector fired on the synthetic decoy at IoU={iou:.3f} (conf={conf:.2f}). "
            "Model may be learning compositing artifacts rather than human shape."
        )


# ---------------------------------------------------------------------------
# Altitude-aware physics tests
# ---------------------------------------------------------------------------

class TestAltitudePhysics:
    """Verify the altitude → pixel-size model is physically consistent."""

    def test_higher_altitude_produces_smaller_survivors(self):
        """Survivor pixel width must decrease monotonically with altitude."""
        sizes = [altitude_to_survivor_px(alt) for alt in (20.0, 35.0, 50.0)]
        assert sizes[0] > sizes[1] > sizes[2]

    def test_known_values_at_default_config(self):
        """Sanity check pixel sizes at the three default flight levels (640px, 65° FOV).

        With SURVIVOR_BODY_WIDTH_M=2.4 (full bounding box from above, matching
        detection/simulation_adapter.py):
          20m: footprint ≈ 25.5m → 2.4/25.5*640 ≈ 60 px
          35m: footprint ≈ 44.6m → 2.4/44.6*640 ≈ 34 px
          50m: footprint ≈ 63.7m → 2.4/63.7*640 ≈ 24 px
        """
        px_20 = altitude_to_survivor_px(20.0, image_size=640)
        px_35 = altitude_to_survivor_px(35.0, image_size=640)
        px_50 = altitude_to_survivor_px(50.0, image_size=640)
        assert 45 < px_20 < 75, f"Expected ~60px at 20m, got {px_20:.1f}"
        assert 25 < px_35 < 45, f"Expected ~34px at 35m, got {px_35:.1f}"
        assert 18 < px_50 < 32, f"Expected ~24px at 50m, got {px_50:.1f}"

    def test_gsd_increases_with_altitude(self):
        """Ground sample distance must increase with altitude."""
        gsd_20 = altitude_to_gsd(20.0, image_size=640)
        gsd_50 = altitude_to_gsd(50.0, image_size=640)
        assert gsd_50 > gsd_20
        # At 20m: footprint ~25.5m / 640px ≈ 0.040 m/px
        assert 0.03 < gsd_20 < 0.06
        # At 50m: footprint ~63.7m / 640px ≈ 0.100 m/px
        assert 0.07 < gsd_50 < 0.13

    def test_sample_altitude_within_envelope(self):
        """Sampled altitudes must be within the operational flight envelope."""
        rng = _rng(42)
        for _ in range(200):
            alt = sample_altitude(rng)
            assert DRONE_FLIGHT_LEVELS_M[0] <= alt <= DRONE_FLIGHT_LEVELS_M[-1]

    def test_altitude_aware_generation_produces_metadata(self):
        """altitude_aware=True writes per-image .json sidecar with altitude info."""
        rng = _rng(7)
        cfg = WildfireEffectConfig()
        # Minimal 1-pixel "survivor" asset for fast generation.
        asset = Image.new("RGBA", (10, 20), (200, 100, 80, 255))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test_alt"
            _generate_split(
                out, n=5, assets=[asset], size=64, rng=rng, cfg=cfg,
                fire_frac=0.0, neg_frac=0.0, decoy_frac=0.0,
                altitude_aware=True,
            )
            json_files = list((out / "labels").glob("*.json"))
            assert len(json_files) == 5
            import json
            meta = json.loads(json_files[0].read_text())
            assert "altitude_m" in meta
            assert "gsd_m" in meta
            assert "survivor_base_px" in meta
            assert DRONE_FLIGHT_LEVELS_M[0] <= meta["altitude_m"] <= DRONE_FLIGHT_LEVELS_M[-1]

    def test_legacy_mode_no_metadata(self):
        """altitude_aware=False should NOT produce .json sidecars."""
        rng = _rng(7)
        cfg = WildfireEffectConfig()
        asset = Image.new("RGBA", (10, 20), (200, 100, 80, 255))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test_noalt"
            _generate_split(
                out, n=3, assets=[asset], size=64, rng=rng, cfg=cfg,
                fire_frac=0.0, neg_frac=0.0, decoy_frac=0.0,
                altitude_aware=False,
            )
            json_files = list((out / "labels").glob("*.json"))
            assert len(json_files) == 0
