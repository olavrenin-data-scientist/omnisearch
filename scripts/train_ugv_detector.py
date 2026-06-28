"""Fine-tune YOLOv8 person detectors for UGV (ground robot) cameras.

Unlike the top-down aerial detector trained in train_survivor_detector.py,
UGVs see upright persons from ground-level or from a short mast (~3-5 m).
Two models are produced:

  1. **Front camera** (yolov8s): forward-looking at ground level, persons at
     5-30 m range, partially occluded by vegetation/brush. Used for forward
     detection while driving.

  2. **Mast camera** (yolov8n): elevated 3-5 m, angled ~45° down, 3-15 m range.
     Used for visual confirmation once the UGV arrives near a scouted survivor.

Both reuse the existing SARD assets (upright orientation) and NAIP backgrounds
but with completely different camera geometry and compositing logic.

Usage:
    python scripts/train_ugv_detector.py                        # both cameras
    python scripts/train_ugv_detector.py --camera front         # front only
    python scripts/train_ugv_detector.py --camera mast          # mast only

Output weights:
    models/ugv_front_yolov8s.pt
    models/ugv_mast_yolov8n.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.wildfire_effects import (
    WildfireEffectConfig,
    WildfireMasks,
    apply_wildfire_effects_to_pil,
)


# ---------------------------------------------------------------------------
# UGV camera physics
# ---------------------------------------------------------------------------

# Physical constants
PERSON_HEIGHT_M = 1.75
PERSON_WIDTH_M = 0.55

# Front camera: mounted at ~0.8 m height, horizontal FOV ~70°
FRONT_CAMERA_HEIGHT_M = 0.8
FRONT_CAMERA_HFOV_DEG = 70.0
FRONT_RANGE_MIN_M = 5.0
FRONT_RANGE_MAX_M = 30.0

# Mast camera: mounted at 3-5 m height, tilted ~45° down, FOV ~60°
MAST_HEIGHT_MIN_M = 3.0
MAST_HEIGHT_MAX_M = 5.0
MAST_TILT_DEG = 45.0
MAST_HFOV_DEG = 60.0
MAST_RANGE_MIN_M = 3.0
MAST_RANGE_MAX_M = 15.0


def front_range_to_person_px(
    range_m: float,
    image_width: int = 640,
    hfov_deg: float = FRONT_CAMERA_HFOV_DEG,
    person_height_m: float = PERSON_HEIGHT_M,
    person_width_m: float = PERSON_WIDTH_M,
) -> tuple[float, float]:
    """Compute person pixel height and width at a given range for the front camera.

    Uses the pinhole model: at distance d, an object of height h subtends
    h / (2*d*tan(fov/2)) * image_width pixels vertically (scaled by aspect ratio).
    """
    ground_width_m = 2.0 * range_m * np.tan(np.radians(hfov_deg) / 2.0)
    px_per_m = image_width / ground_width_m
    px_height = person_height_m * px_per_m
    px_width = person_width_m * px_per_m
    return px_height, px_width


def mast_range_to_person_px(
    range_m: float,
    mast_height_m: float = 4.0,
    image_width: int = 640,
    hfov_deg: float = MAST_HFOV_DEG,
    tilt_deg: float = MAST_TILT_DEG,
    person_height_m: float = PERSON_HEIGHT_M,
    person_width_m: float = PERSON_WIDTH_M,
) -> tuple[float, float]:
    """Compute person pixel height and width for the mast camera.

    The mast camera is elevated and tilted down. The effective distance to the
    person is the slant range. The person's apparent height is foreshortened
    by the viewing angle.
    """
    slant_range = np.sqrt(range_m**2 + mast_height_m**2)
    view_angle_rad = np.arctan2(mast_height_m, range_m)
    foreshortening = np.cos(view_angle_rad - np.radians(90 - tilt_deg))
    foreshortening = max(0.3, min(1.0, abs(foreshortening)))

    ground_width_m = 2.0 * slant_range * np.tan(np.radians(hfov_deg) / 2.0)
    px_per_m = image_width / ground_width_m
    px_height = person_height_m * foreshortening * px_per_m
    px_width = person_width_m * px_per_m
    return px_height, px_width


def sample_front_range(rng: np.random.Generator) -> float:
    """Sample a target range for the front camera (biased toward closer ranges)."""
    return float(rng.triangular(FRONT_RANGE_MIN_M, FRONT_RANGE_MIN_M + 5, FRONT_RANGE_MAX_M))


def sample_mast_range(rng: np.random.Generator) -> float:
    """Sample a target range for the mast camera."""
    return float(rng.uniform(MAST_RANGE_MIN_M, MAST_RANGE_MAX_M))


def sample_mast_height(rng: np.random.Generator) -> float:
    """Sample a mast height."""
    return float(rng.uniform(MAST_HEIGHT_MIN_M, MAST_HEIGHT_MAX_M))


# ---------------------------------------------------------------------------
# Background generation for ground-level views
# ---------------------------------------------------------------------------

def _ground_level_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Generate a procedural ground-level background.

    Simulates a vegetation/terrain scene viewed from ground level: a gradient
    from soil/ground at the bottom to vegetation/sky at the top, with noise
    texturing throughout.
    """
    arr = np.zeros((size, size, 3), dtype=np.float32)

    # Vertical gradient: bottom is soil (brown), top is vegetation (green/dark)
    for y in range(size):
        t = y / size
        # Sky/canopy at top, ground at bottom
        r = rng.uniform(30, 60) * (1 - t) + rng.uniform(80, 140) * t
        g = rng.uniform(50, 90) * (1 - t) + rng.uniform(60, 110) * t
        b = rng.uniform(20, 50) * (1 - t) + rng.uniform(40, 80) * t
        arr[y, :] = [r, g, b]

    # Low-frequency colour variation (patches of different vegetation)
    lf = rng.uniform(-20, 20, (16, 16, 3)).astype(np.float32)
    lf_up = np.asarray(
        Image.fromarray(np.clip(lf + 128, 0, 255).astype(np.uint8)).resize(
            (size, size), Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    )
    arr = arr + (lf_up - 128)

    # Fine noise texture
    arr += rng.normal(0, 10, arr.shape)

    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")


def _naip_ground_background(
    tiles: list[Image.Image], size: int, rng: np.random.Generator
) -> Image.Image:
    """Crop a NAIP tile at larger scale (ground-level equivalent).

    For ground-level views we take a very tight crop from NAIP and upscale it
    significantly, simulating what the terrain looks like from ~1m height
    rather than from 100+ m altitude.
    """
    tile = tiles[int(rng.integers(0, len(tiles)))]
    W, H = tile.size
    # Very small crop (simulating close-up ground view)
    crop_size = max(16, int(min(W, H) * rng.uniform(0.05, 0.15)))
    x = int(rng.integers(0, max(1, W - crop_size)))
    y = int(rng.integers(0, max(1, H - crop_size)))
    crop = tile.crop((x, y, x + crop_size, y + crop_size))
    crop = crop.resize((size, size), Image.Resampling.BILINEAR)

    # Heavy augmentation since we're upscaling a lot
    arr = np.asarray(crop, dtype=np.float32)
    arr *= rng.uniform(0.7, 1.3)
    for c in range(3):
        arr[..., c] += rng.uniform(-25, 25)
    arr += rng.normal(0, 8, arr.shape)
    crop = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))

    # Random rotation
    rot_k = int(rng.integers(0, 4))
    if rot_k:
        crop = crop.rotate(rot_k * 90, expand=False)
    return crop


# ---------------------------------------------------------------------------
# Vegetation occlusion for front camera
# ---------------------------------------------------------------------------

def _add_vegetation_occlusion(
    img: Image.Image,
    person_bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
    occlusion_prob: float = 0.5,
) -> Image.Image:
    """Add procedural vegetation/brush occlusion in front of a person.

    Simulates branches, leaves, or tall grass partially obscuring the person
    as seen from a ground-level camera in a wildfire/forest setting.
    """
    if rng.random() > occlusion_prob:
        return img

    x1, y1, x2, y2 = person_bbox
    pw, ph = x2 - x1, y2 - y1
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    n_elements = int(rng.integers(3, 12))
    for _ in range(n_elements):
        # Vegetation elements near the person bbox
        vx_lo = min(x1 - pw * 0.3, x2 + pw * 0.3)
        vx_hi = max(x1 - pw * 0.3, x2 + pw * 0.3)
        vx = int(rng.uniform(vx_lo, vx_hi + 1))
        vy_lo = min(y1 + ph * 0.3, y2 + ph * 0.2)
        vy_hi = max(y1 + ph * 0.3, y2 + ph * 0.2)
        vy = int(rng.uniform(vy_lo, vy_hi + 1))

        # Dark green/brown strokes simulating branches or leaves
        g = int(rng.integers(30, 100))
        r = int(rng.integers(20, 70))
        b = int(rng.integers(10, 50))
        alpha = int(rng.integers(120, 220))

        # Random stroke
        stroke_lo = max(1, pw // 4)
        stroke_hi = max(stroke_lo + 1, pw)
        stroke_len = int(rng.integers(stroke_lo, stroke_hi))
        angle = rng.uniform(-0.5, 0.5)
        vx2 = int(vx + stroke_len * np.cos(angle))
        vy2 = int(vy + stroke_len * np.sin(angle))
        width = int(rng.integers(2, max(3, pw // 8)))
        draw.line([(vx, vy), (vx2, vy2)], fill=(r, g, b, alpha), width=width)

    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    return img_rgba.convert("RGB")


# ---------------------------------------------------------------------------
# Compositing helpers (shared with drone pipeline)
# ---------------------------------------------------------------------------

def _erode_alpha_edge(sprite: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Erode and feather alpha to soften hard edges."""
    a = sprite.getchannel("A")
    erode_px = int(rng.integers(0, 3))
    if erode_px > 0:
        a = a.filter(ImageFilter.MinFilter(size=erode_px * 2 + 1))
    a = a.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return Image.merge("RGBA", (*sprite.convert("RGB").split(), a))


def _harmonize_color(
    sprite: Image.Image, bg_patch: Image.Image, rng: np.random.Generator
) -> Image.Image:
    """Shift sprite color mean toward background patch."""
    alpha = np.asarray(sprite.getchannel("A"), dtype=np.float32) / 255.0
    fg_mask = alpha > 0.5
    if fg_mask.sum() < 5:
        return sprite
    spr = np.asarray(sprite.convert("RGB"), dtype=np.float32)
    bg = np.asarray(bg_patch.convert("RGB"), dtype=np.float32)
    blend = rng.uniform(0.15, 0.40)
    for c in range(3):
        src_mean = float(spr[:, :, c][fg_mask].mean())
        tgt_mean = float(bg[:, :, c].mean())
        spr[:, :, c] = np.clip(spr[:, :, c] + (tgt_mean - src_mean) * blend, 0, 255)
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.90, 1.10)
    mid = 127.5
    spr = np.clip((spr - mid) * contrast + mid * brightness, 0, 255)
    result = Image.fromarray(spr.astype(np.uint8))
    return Image.merge("RGBA", (*result.split(), sprite.getchannel("A")))


def _range_blur(
    sprite: Image.Image, range_m: float, rng: np.random.Generator
) -> Image.Image:
    """Apply distance-dependent blur (farther = more blur from atmosphere/focus)."""
    base_r = range_m * 0.04  # ~0.2 at 5m, ~1.2 at 30m
    r = float(rng.uniform(base_r * 0.6, base_r * 1.4))
    if r < 0.1:
        return sprite
    rgb = sprite.convert("RGB").filter(ImageFilter.GaussianBlur(radius=r))
    return Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))


# ---------------------------------------------------------------------------
# Hard-negative decoys
# ---------------------------------------------------------------------------

def _load_decoy_assets(decoy_dir: str) -> list[Image.Image]:
    """Load pre-extracted RGBA decoy PNGs."""
    paths = sorted(glob.glob(str(Path(decoy_dir) / "*.png")))
    assets = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGBA")
        except Exception:
            continue
        if min(img.size) >= 8:
            assets.append(img)
    return assets


def _synthetic_decoy_rgba(
    target_w: int, target_h: int, rng: np.random.Generator
) -> Image.Image:
    """Procedural non-human object (rock / stump / gear) as RGBA."""
    w = max(8, int(target_w * rng.uniform(0.6, 1.4)))
    h = max(8, int(target_h * rng.uniform(0.5, 1.6)))

    archetype = int(rng.integers(0, 4))
    if archetype == 0:
        r0, g0, b0 = int(rng.integers(80, 160)), int(rng.integers(75, 155)), int(rng.integers(70, 150))
    elif archetype == 1:
        r0, g0, b0 = int(rng.integers(100, 170)), int(rng.integers(65, 125)), int(rng.integers(35, 85))
    elif archetype == 2:
        r0, g0, b0 = int(rng.integers(140, 200)), int(rng.integers(115, 165)), int(rng.integers(55, 105))
    else:
        r0, g0, b0 = int(rng.integers(45, 105)), int(rng.integers(50, 110)), int(rng.integers(40, 95))

    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(rng.integers(2, 6))):
        cx = rng.uniform(w * 0.15, w * 0.85)
        cy = rng.uniform(h * 0.15, h * 0.85)
        rx = max(1.0, rng.uniform(w * 0.12, w * 0.48))
        ry = max(1.0, rng.uniform(h * 0.12, h * 0.48))
        blob = np.maximum(blob, np.clip(1.0 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0, 1))
    alpha = ((blob > 0.38).astype(np.uint8) * 255)
    alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=1.0))

    rgb = np.full((h, w, 3), [r0, g0, b0], dtype=np.float32)
    rgb += rng.normal(0, 14, rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    result = Image.fromarray(rgb)
    result = Image.merge("RGBA", (*result.split(), alpha_img)).resize(
        (target_w, target_h), Image.Resampling.LANCZOS
    )
    return result


# ---------------------------------------------------------------------------
# YOLO label utility
# ---------------------------------------------------------------------------

def _clip_label(x: int, y: int, w: int, h: int, size: int):
    """Clip placement to image boundary; return YOLO label or None."""
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(size, x + w); y2 = min(size, y + h)
    cw = x2 - x1; ch = y2 - y1
    if cw < 4 or ch < 4:
        return None
    return (x1 + x2) / 2.0 / size, (y1 + y2) / 2.0 / size, cw / size, ch / size


# ---------------------------------------------------------------------------
# Front camera dataset generation
# ---------------------------------------------------------------------------

def _generate_front_split(
    out_dir: Path,
    n: int,
    assets: list[Image.Image],
    size: int,
    rng: np.random.Generator,
    cfg: WildfireEffectConfig,
    naip_tiles: list[Image.Image] | None = None,
    neg_frac: float = 0.10,
    decoy_frac: float = 0.15,
    decoy_assets: list[Image.Image] | None = None,
    fire_frac: float = 0.40,
    vegetation_occlusion_prob: float = 0.50,
) -> None:
    """Generate front-camera training composites.

    Persons are rendered upright (as seen from ground level), sized according
    to the pinhole camera model at sampled ranges (5-30 m). Vegetation
    occlusion is applied probabilistically.
    """
    img_dir = out_dir / "images"; lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    n_neg = 0

    for i in range(n):
        if naip_tiles and rng.random() < 0.7:
            bg = _naip_ground_background(naip_tiles, size, rng)
        else:
            bg = _ground_level_background(size, rng)

        is_negative = rng.random() < neg_frac
        labels: list[str] = []
        placements: list[tuple[int, int, int, int]] = []

        if not is_negative:
            n_persons = int(rng.integers(1, 4))
            for _ in range(n_persons):
                range_m = sample_front_range(rng)
                px_h, px_w = front_range_to_person_px(range_m, image_width=size)

                # Add jitter for body variation
                px_h = int(px_h * rng.uniform(0.8, 1.2))
                px_w = int(px_w * rng.uniform(0.8, 1.2))
                px_h = max(20, min(size - 10, px_h))
                px_w = max(8, min(size // 2, px_w))

                asset = assets[int(rng.integers(0, len(assets)))]
                sprite = asset.resize((px_w, px_h), Image.Resampling.LANCZOS)

                # Place person: bottom-aligned (feet near bottom of frame for close,
                # higher up for farther)
                range_frac = (range_m - FRONT_RANGE_MIN_M) / (FRONT_RANGE_MAX_M - FRONT_RANGE_MIN_M)
                # Farther persons appear higher in frame (perspective)
                base_y = int(size * (0.95 - range_frac * 0.5) - px_h)
                y = max(0, min(size - px_h, base_y + int(rng.integers(-20, 20))))
                x = int(rng.integers(0, max(1, size - px_w)))

                # Compositing pipeline
                bg_patch = bg.crop((
                    max(0, x), max(0, y),
                    min(size, x + px_w), min(size, y + px_h)
                ))
                sprite = _harmonize_color(sprite, bg_patch, rng)
                sprite = _erode_alpha_edge(sprite, rng)
                sprite = _range_blur(sprite, range_m, rng)
                bg.paste(sprite, (x, y), sprite)

                placements.append((x, y, px_w, px_h))
                clipped = _clip_label(x, y, px_w, px_h, size)
                if clipped is not None:
                    cx_n, cy_n, w_n, h_n = clipped
                    labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

            # Vegetation occlusion (applied after all persons are placed)
            for bbox in placements:
                bg = _add_vegetation_occlusion(bg, bbox, rng, vegetation_occlusion_prob)

        # Fire/smoke effects
        if rng.random() < fire_frac and placements:
            centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in placements]
            mask = _wildfire_masks_ground(size, rng, centers)
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg)

        # Hard-negative decoys
        if rng.random() < decoy_frac:
            n_decoys = int(rng.integers(1, 3))
            for _ in range(n_decoys):
                dw = int(rng.integers(15, 80))
                dh = int(dw * rng.uniform(1.0, 2.5))
                dx = int(rng.integers(0, max(1, size - dw)))
                dy = int(rng.integers(size // 3, max(size // 3 + 1, size - dh)))
                if decoy_assets:
                    raw = decoy_assets[int(rng.integers(0, len(decoy_assets)))]
                    decoy = raw.resize((dw, dh), Image.Resampling.LANCZOS)
                else:
                    decoy = _synthetic_decoy_rgba(dw, dh, rng)
                bg_patch = bg.crop((dx, dy, min(size, dx + dw), min(size, dy + dh)))
                decoy = _harmonize_color(decoy, bg_patch, rng)
                decoy = _erode_alpha_edge(decoy, rng)
                bg.paste(decoy, (dx, dy), decoy)

        bg.save(img_dir / f"{i:05d}.jpg", quality=92)
        label_text = "\n".join(labels) + ("\n" if labels else "")
        (lbl_dir / f"{i:05d}.txt").write_text(label_text, encoding="utf-8")

        meta = {
            "camera": "front",
            "n_persons": len(placements),
            "is_negative": is_negative,
        }
        (lbl_dir / f"{i:05d}.json").write_text(json.dumps(meta), encoding="utf-8")
        if is_negative:
            n_neg += 1

    print(f"  {out_dir.name}: {n} images ({n_neg} negatives, {100*n_neg/max(1,n):.0f}%)")


# ---------------------------------------------------------------------------
# Mast camera dataset generation
# ---------------------------------------------------------------------------

def _generate_mast_split(
    out_dir: Path,
    n: int,
    assets: list[Image.Image],
    size: int,
    rng: np.random.Generator,
    cfg: WildfireEffectConfig,
    naip_tiles: list[Image.Image] | None = None,
    neg_frac: float = 0.10,
    decoy_frac: float = 0.15,
    decoy_assets: list[Image.Image] | None = None,
    fire_frac: float = 0.40,
) -> None:
    """Generate mast-camera training composites.

    The mast camera is elevated (3-5 m) and angled down ~45°. Persons appear
    with foreshortened height (oblique view) and are placed more centrally
    since the camera is aimed at the confirmation zone directly below/ahead.
    """
    img_dir = out_dir / "images"; lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    n_neg = 0

    for i in range(n):
        if naip_tiles and rng.random() < 0.8:
            bg = _naip_ground_background(naip_tiles, size, rng)
        else:
            bg = _ground_level_background(size, rng)

        is_negative = rng.random() < neg_frac
        labels: list[str] = []
        placements: list[tuple[int, int, int, int]] = []

        if not is_negative:
            mast_h = sample_mast_height(rng)
            n_persons = int(rng.integers(1, 3))
            for _ in range(n_persons):
                range_m = sample_mast_range(rng)
                px_h, px_w = mast_range_to_person_px(
                    range_m, mast_height_m=mast_h, image_width=size
                )

                px_h = int(px_h * rng.uniform(0.8, 1.2))
                px_w = int(px_w * rng.uniform(0.8, 1.2))
                px_h = max(15, min(size - 10, px_h))
                px_w = max(8, min(size // 2, px_w))

                asset = assets[int(rng.integers(0, len(assets)))]
                sprite = asset.resize((px_w, px_h), Image.Resampling.LANCZOS)

                # Mast view: persons anywhere in frame (camera looks down at zone)
                x = int(rng.integers(0, max(1, size - px_w)))
                y = int(rng.integers(0, max(1, size - px_h)))

                bg_patch = bg.crop((
                    max(0, x), max(0, y),
                    min(size, x + px_w), min(size, y + px_h)
                ))
                sprite = _harmonize_color(sprite, bg_patch, rng)
                sprite = _erode_alpha_edge(sprite, rng)
                sprite = _range_blur(sprite, range_m, rng)
                bg.paste(sprite, (x, y), sprite)

                placements.append((x, y, px_w, px_h))
                clipped = _clip_label(x, y, px_w, px_h, size)
                if clipped is not None:
                    cx_n, cy_n, w_n, h_n = clipped
                    labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        # Fire/smoke effects (less common for mast since it's confirmation range)
        if rng.random() < fire_frac and placements:
            centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in placements]
            mask = _wildfire_masks_ground(size, rng, centers)
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg)

        # Hard-negative decoys
        if rng.random() < decoy_frac:
            n_decoys = int(rng.integers(1, 3))
            for _ in range(n_decoys):
                dw = int(rng.integers(15, 80))
                dh = int(dw * rng.uniform(0.8, 2.0))
                dx = int(rng.integers(0, max(1, size - dw)))
                dy = int(rng.integers(0, max(1, size - dh)))
                if decoy_assets:
                    raw = decoy_assets[int(rng.integers(0, len(decoy_assets)))]
                    decoy = raw.resize((dw, dh), Image.Resampling.LANCZOS)
                else:
                    decoy = _synthetic_decoy_rgba(dw, dh, rng)
                bg_patch = bg.crop((dx, dy, min(size, dx + dw), min(size, dy + dh)))
                decoy = _harmonize_color(decoy, bg_patch, rng)
                decoy = _erode_alpha_edge(decoy, rng)
                bg.paste(decoy, (dx, dy), decoy)

        bg.save(img_dir / f"{i:05d}.jpg", quality=92)
        label_text = "\n".join(labels) + ("\n" if labels else "")
        (lbl_dir / f"{i:05d}.txt").write_text(label_text, encoding="utf-8")

        meta = {
            "camera": "mast",
            "n_persons": len(placements),
            "is_negative": is_negative,
        }
        (lbl_dir / f"{i:05d}.json").write_text(json.dumps(meta), encoding="utf-8")
        if is_negative:
            n_neg += 1

    print(f"  {out_dir.name}: {n} images ({n_neg} negatives, {100*n_neg/max(1,n):.0f}%)")


# ---------------------------------------------------------------------------
# Wildfire masks adapted for ground-level views
# ---------------------------------------------------------------------------

def _disc(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0), 0, 1).astype(np.float32)


def _wildfire_masks_ground(
    size: int, rng: np.random.Generator, centers: list
) -> WildfireMasks:
    """Build fire/smoke masks for ground-level views.

    Fire tends to be lower in the frame (ground fires) and smoke drifts
    across the entire frame from below.
    """
    burned = np.zeros((size, size), np.float32)
    active = np.zeros((size, size), np.float32)
    smoke = np.zeros((size, size), np.float32)

    for _ in range(int(rng.integers(1, 3))):
        if centers and rng.random() < 0.6:
            cx, cy = centers[int(rng.integers(0, len(centers)))]
            cx += rng.uniform(-size * 0.1, size * 0.1)
            cy += rng.uniform(-size * 0.05, size * 0.1)
        else:
            cx = rng.uniform(0, size)
            cy = rng.uniform(size * 0.5, size)  # Fire near bottom

        burned = np.maximum(burned, _disc(size, cx, cy, size * rng.uniform(0.15, 0.4)) * rng.uniform(0.4, 0.9))
        active = np.maximum(active, _disc(size, cx, cy, size * rng.uniform(0.1, 0.3)) * rng.uniform(0.5, 1.0))
        # Smoke rises: centered higher than fire
        smoke_cy = cy - size * rng.uniform(0.1, 0.3)
        smoke = np.maximum(smoke, _disc(size, cx, smoke_cy, size * rng.uniform(0.2, 0.5)) * rng.uniform(0.3, 0.8))

    return WildfireMasks(burned=burned, active=active, intensity=active.copy(), smoke=smoke)


# ---------------------------------------------------------------------------
# NAIP tile loading (reuse from drone pipeline)
# ---------------------------------------------------------------------------

def _load_naip_tiles(naip_dir: str) -> list[Image.Image]:
    """Load NAIP tiles for background generation."""
    paths = sorted(glob.glob(str(Path(naip_dir) / "**" / "*.png"), recursive=True))
    tiles = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        if min(im.size) >= 64:
            tiles.append(im)
    return tiles


def _split_tiles(tiles: list[Image.Image], val_frac: float = 0.20):
    """Geographic train/val split."""
    n_train = max(1, int(len(tiles) * (1.0 - val_frac)))
    return tiles[:n_train], tiles[n_train:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train UGV person detectors (front + mast cameras)")
    ap.add_argument("--camera", choices=["front", "mast", "both"], default="both",
                    help="Which camera model(s) to train.")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--front-model", default="yolov8s.pt",
                    help="Base model for front camera (larger for harder task).")
    ap.add_argument("--mast-model", default="yolov8n.pt",
                    help="Base model for mast camera (lighter, closer targets).")
    ap.add_argument("--naip-dir", default=None,
                    help="NAIP tiles for background generation.")
    ap.add_argument("--assets-dir", default=str(ROOT / "data/cv_assets/sard_grabcut"))
    ap.add_argument("--review-json",
                    default=str(ROOT / "configs/cv/sard_grabcut_asset_review.json"))
    ap.add_argument("--decoy-assets-dir", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data/cv_train/ugv"))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    cfg = WildfireEffectConfig()

    # Load SARD survivor assets
    all_asset_paths = sorted(glob.glob(str(Path(args.assets_dir) / "*.png")))
    if not all_asset_paths:
        raise SystemExit(f"No survivor assets in {args.assets_dir}")

    review_path = Path(args.review_json)
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        accepted = set(review.get("accepted_assets", []))
        if accepted:
            filtered = [p for p in all_asset_paths if Path(p).name in accepted]
            if filtered:
                print(f"Using {len(filtered)} accepted assets (of {len(all_asset_paths)} total)")
                all_asset_paths = filtered

    assets = [Image.open(p).convert("RGBA") for p in all_asset_paths]
    n_hold = max(1, len(assets) // 5)
    train_assets, val_assets = assets[:-n_hold], assets[-n_hold:]

    # Load decoy assets
    decoy_assets = None
    if args.decoy_assets_dir:
        decoy_assets = _load_decoy_assets(args.decoy_assets_dir)
        if decoy_assets:
            print(f"Loaded {len(decoy_assets)} decoy assets")

    # NAIP tiles
    naip_train_tiles = None
    naip_val_tiles = None
    if args.naip_dir:
        all_tiles = _load_naip_tiles(args.naip_dir)
        if all_tiles:
            naip_train_tiles, naip_val_tiles = _split_tiles(all_tiles)
            print(f"NAIP tiles: {len(naip_train_tiles)} train / {len(naip_val_tiles)} val")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_to_train = []
    if args.camera in ("front", "both"):
        cameras_to_train.append("front")
    if args.camera in ("mast", "both"):
        cameras_to_train.append("mast")

    for camera in cameras_to_train:
        print(f"\n{'='*60}")
        print(f"  UGV {camera.upper()} camera detector")
        print(f"{'='*60}")

        cam_data_dir = data_dir / camera

        if camera == "front":
            print(f"Generating {args.n_train} train / {args.n_val} val front-camera composites ...")
            _generate_front_split(
                cam_data_dir / "train", args.n_train, train_assets, args.size, rng, cfg,
                naip_tiles=naip_train_tiles, decoy_assets=decoy_assets,
            )
            _generate_front_split(
                cam_data_dir / "val", args.n_val, val_assets, args.size, rng, cfg,
                naip_tiles=naip_val_tiles, decoy_assets=decoy_assets,
            )
            base_model = args.front_model
            out_name = "ugv_front_yolov8s.pt"
        else:
            print(f"Generating {args.n_train} train / {args.n_val} val mast-camera composites ...")
            _generate_mast_split(
                cam_data_dir / "train", args.n_train, train_assets, args.size, rng, cfg,
                naip_tiles=naip_train_tiles, decoy_assets=decoy_assets,
            )
            _generate_mast_split(
                cam_data_dir / "val", args.n_val, val_assets, args.size, rng, cfg,
                naip_tiles=naip_val_tiles, decoy_assets=decoy_assets,
            )
            base_model = args.mast_model
            out_name = "ugv_mast_yolov8n.pt"

        yaml_path = cam_data_dir / "person.yaml"
        yaml_path.write_text(
            f"path: {cam_data_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n",
            encoding="utf-8",
        )

        from ultralytics import YOLO
        model = YOLO(base_model)
        print(f"Fine-tuning {base_model} for {args.epochs} epochs ...")
        model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=16,
            device="cpu", verbose=True, seed=args.seed,
            project=str(cam_data_dir.resolve() / "runs"),
            name=camera, exist_ok=True, plots=False,
        )

        best = Path(getattr(model.trainer, "best", cam_data_dir / "runs" / camera / "weights" / "best.pt"))
        if not best.exists():
            found = list(cam_data_dir.rglob("weights/best.pt"))
            if found:
                best = max(found, key=lambda p: p.stat().st_mtime)

        import shutil
        out_path = out_dir / out_name
        shutil.copy(best, out_path)
        print(f"\nWrote {camera} detector to: {out_path}")

    print("\nDone! UGV detectors ready.")


if __name__ == "__main__":
    main()
