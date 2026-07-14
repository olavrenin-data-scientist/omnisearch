"""Physical-plausibility tests for object sizes vs. terrain ground resolution.

For every altitude-aware dataset (``survivor/`` and ``thermal/``) each image
records its ground-sample-distance (GSD, metres-per-pixel). Multiplying a YOLO
box's pixel size by the GSD yields the object's implied *real-world* size in
metres. A survivor bounding box must therefore correspond to a human-scale
object — a person is ~0.55 m across the shoulders and at most ~1.75 m long
(standing height / prone body length), so no box should exceed ~2 m in any
dimension, and the set as a whole must not be dominated by large near-square
"blobs" (which imply an impossible multi-square-metre footprint for one person).

These tests convert every labelled box to metres and assert it falls inside a
physical human envelope. They also sanity-check the generator's own physics
helpers so the size model itself stays honest.
"""

from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path

import pytest

from scripts.train_survivor_detector import (
    DRONE_FLIGHT_LEVELS_M,
    MAX_SURVIVOR_LONG_AXIS_M,
    SURVIVOR_BODY_WIDTH_M,
    altitude_to_gsd,
    altitude_to_survivor_px,
    oblique_survivor_size,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cv_train"

# Physical envelope for a human detection bounding box, in metres. A person is
# ~0.55 m across and at most ~1.75 m long; the generator caps any dimension at
# MAX_SURVIVOR_LONG_AXIS_M (2.0 m). Allow a little headroom for pixel rounding
# at low altitude. The floor guards against degenerate zero-area boxes.
MIN_DIM_M = 0.1
MAX_DIM_M = MAX_SURVIVOR_LONG_AXIS_M + 0.15  # 2.15 m

# Datasets carrying per-image GSD metadata.
GSD_DATASETS = [
    ("survivor/train", 640),
    ("survivor/val", 640),
    ("thermal/train", 512),
    ("thermal/val", 512),
]
# The silhouette-shape rules apply to the composited-person (CV) dataset. Thermal
# targets are radial heat blobs, so "square" is expected and not a defect there.
SURVIVOR_SPLITS = [("survivor/train", 640), ("survivor/val", 640)]


def _image_gsd(meta: dict, image_size: int) -> float | None:
    """Metres-per-pixel for an image from its metadata sidecar."""
    if meta.get("gsd_m"):
        return float(meta["gsd_m"])
    if meta.get("footprint_m"):  # thermal stores footprint instead of gsd
        return float(meta["footprint_m"]) / image_size
    return None


def _boxes_in_metres(split: str, image_size: int):
    """Yield (stem, width_m, height_m) for every labelled box in a split."""
    label_dir = DATA / split / "labels"
    for txt in glob.glob(str(label_dir / "*.txt")):
        stem = Path(txt).stem
        js = label_dir / f"{stem}.json"
        if not js.exists():
            continue
        meta = json.loads(js.read_text())
        gsd = _image_gsd(meta, image_size)
        if not gsd:
            continue
        for line in Path(txt).read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            w_m = float(parts[3]) * image_size * gsd
            h_m = float(parts[4]) * image_size * gsd
            yield stem, w_m, h_m


# ---------------------------------------------------------------------------
# Generator physics helpers — the size model itself must be human-scale.
# ---------------------------------------------------------------------------
def test_survivor_width_round_trips_to_body_width():
    """px shoulder-width x gsd must recover the modelled body width."""
    lo, hi = DRONE_FLIGHT_LEVELS_M[0], DRONE_FLIGHT_LEVELS_M[-1]
    for alt in (lo, (lo + hi) / 2, hi):
        gsd = altitude_to_gsd(alt)
        width_px = altitude_to_survivor_px(alt)
        assert math.isclose(width_px * gsd, SURVIVOR_BODY_WIDTH_M, rel_tol=1e-6)


def test_modelled_body_width_is_human_scale():
    # Shoulder width should be a realistic ~0.5 m, not a vehicle-sized 2.4 m.
    assert 0.3 <= SURVIVOR_BODY_WIDTH_M <= 0.8


def test_gsd_increases_with_altitude():
    assert altitude_to_gsd(20.0) < altitude_to_gsd(35.0) < altitude_to_gsd(50.0)


def test_oblique_height_never_exceeds_envelope():
    """A tilted-camera survivor height must still map to a human-scale object."""
    for alt in DRONE_FLIGHT_LEVELS_M:
        gsd = altitude_to_gsd(alt)
        for tilt in (0.0, 15.0, 30.0, 45.0):
            _, height_px = oblique_survivor_size(alt, tilt)
            real_h = height_px * gsd
            assert real_h <= MAX_DIM_M, (
                f"oblique height {real_h:.1f} m at alt={alt} tilt={tilt} "
                f"exceeds {MAX_DIM_M} m"
            )


# ---------------------------------------------------------------------------
# Data tests — every labelled box in the generated datasets must be human-scale.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("split,image_size", GSD_DATASETS)
def test_object_dimensions_are_human_scale(split, image_size):
    violations = []
    for stem, w_m, h_m in _boxes_in_metres(split, image_size):
        if not (MIN_DIM_M <= w_m <= MAX_DIM_M and MIN_DIM_M <= h_m <= MAX_DIM_M):
            violations.append((stem, round(w_m, 2), round(h_m, 2)))
    assert not violations, (
        f"{len(violations)} boxes in {split} exceed the human size envelope "
        f"[{MIN_DIM_M}, {MAX_DIM_M}] m (w,h); examples: {violations[:5]}"
    )


@pytest.mark.parametrize("split,image_size", GSD_DATASETS)
def test_object_long_axis_within_body_length(split, image_size):
    violations = [
        (stem, round(max(w_m, h_m), 2))
        for stem, w_m, h_m in _boxes_in_metres(split, image_size)
        if max(w_m, h_m) > MAX_DIM_M
    ]
    assert not violations, (
        f"{len(violations)} boxes in {split} have a long axis > {MAX_DIM_M} m "
        f"(a person is at most ~1.75 m long); examples: {violations[:5]}"
    )


def _frame_metas(split: str):
    """Yield (stem, meta, label_lines) for every frame with a JSON sidecar."""
    label_dir = DATA / split / "labels"
    for js in glob.glob(str(label_dir / "*.json")):
        stem = Path(js).stem
        meta = json.loads(Path(js).read_text())
        txt = label_dir / f"{stem}.txt"
        lines = txt.read_text().strip().splitlines() if txt.exists() else []
        yield stem, meta, [l for l in lines if l.strip()]


@pytest.mark.parametrize("split,image_size", SURVIVOR_SPLITS)
def test_boxes_are_tight_against_alpha_mask(split, image_size):
    """Every stored box must have IoU >= 0.9 against its sprite alpha bbox."""
    loose = []
    checked = 0
    for stem, meta, _lines in _frame_metas(split):
        for bi, box in enumerate(meta.get("boxes") or []):
            checked += 1
            if box.get("mask_iou") is not None and box["mask_iou"] < 0.9:
                loose.append((stem, bi, box["mask_iou"]))
    if checked == 0:
        pytest.skip(f"{split} has no per-box metadata yet (legacy data)")
    assert not loose, f"{len(loose)} loose boxes in {split}: {loose[:5]}"


@pytest.mark.parametrize("split,image_size", SURVIVOR_SPLITS)
def test_within_frame_pixel_sizes_are_coherent(split, image_size):
    """At one GSD, survivors may differ in pixel size only by pose (< ~2x)."""
    offenders = []
    for stem, meta, lines in _frame_metas(split):
        sizes = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                sizes.append(
                    math.sqrt(float(parts[3]) * float(parts[4])) * image_size
                )
        if len(sizes) >= 2 and max(sizes) / max(1e-6, min(sizes)) > 2.0:
            offenders.append((stem, round(max(sizes) / min(sizes), 2)))
    assert not offenders, (
        f"{len(offenders)} frames in {split} mix survivor pixel sizes beyond "
        f"2x at a single GSD: {offenders[:5]}"
    )


@pytest.mark.parametrize("split,image_size", SURVIVOR_SPLITS)
def test_nadir_short_axis_is_body_scale(split, image_size):
    """Straight-down boxes must not be wider than a body cross-section (0.7 m)."""
    offenders = []
    for stem, meta, lines in _frame_metas(split):
        if meta.get("oblique") or meta.get("gsd_m") is None:
            continue
        gsd = float(meta["gsd_m"])
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            w_m = float(parts[3]) * image_size * gsd
            h_m = float(parts[4]) * image_size * gsd
            if min(w_m, h_m) > 0.7:
                offenders.append((stem, round(min(w_m, h_m), 2)))
    assert not offenders, (
        f"{len(offenders)} nadir boxes in {split} have short axis > 0.7 m: "
        f"{offenders[:5]}"
    )


UGV_SPLITS = ["ugv/front/train", "ugv/front/val", "ugv/mast/train", "ugv/mast/val"]


@pytest.mark.parametrize("split", UGV_SPLITS)
def test_ugv_implied_person_size(split):
    """Perspective boxes must imply a ~1.75 m person via per-object m/px."""
    if not (DATA / split / "labels").is_dir():
        pytest.skip(f"{split} not generated")
    offenders = []
    checked = 0
    for stem, meta, _lines in _frame_metas(split):
        for bi, box in enumerate(meta.get("boxes") or []):
            if "m_per_px" not in box:
                continue
            checked += 1
            fore = box.get("foreshortening") or 1.0
            implied = max(box["w_px"], box["h_px"]) * box["m_per_px"] / fore
            if not (1.3 <= implied <= 2.2):
                offenders.append((stem, bi, round(implied, 2)))
    if checked == 0:
        pytest.skip(f"{split} has no per-box scale metadata yet (legacy data)")
    assert not offenders, (
        f"{len(offenders)} UGV boxes in {split} imply a non-human person size: "
        f"{offenders[:5]}"
    )


@pytest.mark.parametrize("split,image_size", SURVIVOR_SPLITS)
def test_survivor_boxes_are_anisotropic_on_average(split, image_size):
    """The survivor set must show orientation variety, not uniform blobs.

    A person seen at random headings/poses yields a spread of aspect ratios;
    the median short/long ratio should be clearly below 1.0. (The old 2.4 m
    model produced a median of ~0.84 — nearly all square.) Combined with the
    long-axis cap, this rejects a regression to constant large square blobs
    without penalising a legitimately near-square diagonally-prone body.
    """
    aspects = []
    for _stem, w_m, h_m in _boxes_in_metres(split, image_size):
        lo, hi = min(w_m, h_m), max(w_m, h_m)
        if hi > 0:
            aspects.append(lo / hi)
    assert aspects, f"no boxes found in {split}"
    median_aspect = statistics.median(aspects)
    assert median_aspect < 0.82, (
        f"survivor boxes in {split} are too square on average "
        f"(median short/long = {median_aspect:.2f}); expected anisotropic bodies"
    )
