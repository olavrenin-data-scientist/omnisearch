"""Unit tests for the YOLO CV-detection helpers in SimulationCvAdapter.

These exercise the box geometry (IoU), duplicate-merging (NMS), and the
ground-truth matching that turns raw YOLO boxes into survivor indices. They
build the adapter via ``object.__new__`` and set only the relevant attributes,
so they need no terrain cache, NAIP imagery, or model download.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from detection.simulation_adapter import SimulationCvAdapter

ROOT = Path(__file__).resolve().parent.parent
_FINETUNED = ROOT / "models" / "survivor_yolov8n.pt"


def _adapter(match_iou: float = 0.15, nms_iou: float = 0.6) -> SimulationCvAdapter:
    a = object.__new__(SimulationCvAdapter)
    a.person_match_iou = match_iou
    a.person_iou = nms_iou
    return a


def test_iou_basic():
    a = _adapter()
    assert a._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert a._iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0
    # Half-overlap on x: intersection 5x10=50, union 100+100-50=150.
    assert abs(a._iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_nms_merges_duplicate_tile_detections():
    a = _adapter(nms_iou=0.5)
    # Two near-identical boxes (overlapping tiles) + one far-away box.
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (100, 100, 120, 120)]
    confs = [0.4, 0.9, 0.3]
    kept = a._nms(boxes, confs, a.person_iou)
    assert len(kept) == 2                      # the two duplicates collapse to one
    assert kept[0][1] == 0.9                   # highest-confidence duplicate survives
    assert (100, 100, 120, 120) in [b for b, _ in kept]


def test_match_truth_index_by_overlap():
    a = _adapter(match_iou=0.15)
    truth_boxes = [(0, 0, 20, 40), (200, 200, 220, 240)]
    truth = [{"survivor_index": 7}, {"survivor_index": 3}]
    # A detection overlapping the second survivor box.
    det = (202, 202, 222, 242)
    assert a._match_truth_index(det, truth_boxes, truth) == 3
    # A detection overlapping nothing -> no match (false positive).
    assert a._match_truth_index((500, 500, 520, 540), truth_boxes, truth) is None


def test_match_truth_index_requires_min_overlap():
    a = _adapter(match_iou=0.5)
    truth_boxes = [(0, 0, 100, 100)]
    truth = [{"survivor_index": 1}]
    # Tiny overlap in the corner -> below threshold -> unmatched.
    assert a._match_truth_index((90, 90, 200, 200), truth_boxes, truth) is None
    # Large overlap -> matched.
    assert a._match_truth_index((0, 0, 90, 90), truth_boxes, truth) == 1


def _disc(size, cx, cy, r):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1).astype(np.float32)


@pytest.mark.skipif(not _FINETUNED.exists(), reason="fine-tuned survivor weights not present (run scripts/train_survivor_detector.py)")
def test_survivor_detected_under_smoke():
    """Regression: a survivor under heavy smoke is still detected and matched.

    Skipped unless the fine-tuned detector is available locally, so the default
    suite needs no model download or network.
    """
    from PIL import Image
    from detection.wildfire_effects import WildfireEffectConfig, WildfireMasks, apply_wildfire_effects_to_pil

    SZ = 1024
    det = object.__new__(SimulationCvAdapter)
    det.detector_backend = "yolo"; det.person_model_name = str(_FINETUNED); det.person_conf = 0.2
    det.person_iou = 0.6; det.person_imgsz = 1280; det.person_tiled = True; det.person_tile_grid = 2
    det.person_tile_overlap = 0.25; det.person_match_iou = 0.15; det.person_device = None
    det.person_augment = False
    det._person_detector = None; det.image_size = SZ

    asset = Image.open(sorted(glob.glob(str(ROOT / "data/cv_assets/sard_grabcut/*.png")))[0]).convert("RGBA")
    bg = Image.fromarray(np.random.default_rng(2).integers(70, 120, (SZ, SZ, 3)).astype("uint8")).convert("RGB")
    # 96px on a 1024px frame matches the largest realistic survivor size at the
    # lowest flight level (20m, 65 deg FOV: 2.4m / 25.5m * 1024 ~ 96px). The model
    # is trained altitude-aware (24-60px at 640px), so unrealistically large
    # sizes (e.g. the old 150px) are out of distribution and must not be tested.
    surv = 96; h = int(surv * asset.height / asset.width)
    s = asset.resize((surv, h), Image.Resampling.LANCZOS)
    bg.paste(s, (SZ // 2 - surv // 2, SZ // 2 - h // 2), s)
    gt = (SZ // 2 - surv // 2, SZ // 2 - h // 2, SZ // 2 + surv // 2, SZ // 2 + h // 2)
    m = WildfireMasks(burned=_disc(SZ, SZ//2, SZ//2, SZ*0.4), active=_disc(SZ, SZ//2, SZ//2, SZ*0.3),
                      intensity=_disc(SZ, SZ//2, SZ//2, SZ*0.3), smoke=_disc(SZ, SZ//2, SZ//2, SZ*0.45))
    view, _ = apply_wildfire_effects_to_pil(bg, m, config=WildfireEffectConfig(),
                                            include_burn=False, include_flame=False, include_smoke=True)
    dets = det._detect_people_cv(view)
    assert dets, "no detections under smoke"
    assert any(det._match_truth_index(b, [gt], [{"survivor_index": 0}]) == 0 for b, _ in dets)
