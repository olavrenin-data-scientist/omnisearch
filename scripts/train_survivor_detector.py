"""Fine-tune a YOLOv8 person detector on OmniSearch's own render distribution.

Stock COCO YOLO is trained on eye-level photos and is unreliable on small,
top-down aerial survivors (confidence ~0.2-0.7, often missed). Because we
*control* the renderer, we can generate labelled training data that matches
exactly what the drone camera produces — SARD survivor assets composited on
terrain-like backgrounds with the same wildfire smoke/flame/burn effects — and
fine-tune YOLO on it. The detector then sees in-distribution data and reaches
high confidence (~0.9) on survivors, including those under fire and smoke.

Usage:
    python scripts/train_survivor_detector.py                 # quick default run
    python scripts/train_survivor_detector.py --epochs 25 --n-train 800 --model yolov8s.pt

Output weights: models/survivor_yolov8n.pt (gitignored). Point the CV adapter
at them with --cv-person-model models/survivor_yolov8n.pt.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.wildfire_effects import (
    WildfireEffectConfig,
    WildfireMasks,
    apply_wildfire_effects_to_pil,
)


# ---------------------------------------------------------------------------
# Altitude-aware survivor size model
# ---------------------------------------------------------------------------
# Physical parameters matching the RL scenario (envs/wildfire_defaults.py):
#   drone_flight_levels_m = (20, 35, 50)
#   drone_camera_fov_deg = 65
#   image_size = 512 or 640
#   survivor_width_m ≈ 0.5 (shoulder width, top-down)
#   survivor_height_m ≈ 0.5 (top-down footprint at nadir)

DRONE_FLIGHT_LEVELS_M = (20.0, 35.0, 50.0)
DRONE_CAMERA_FOV_DEG = 65.0
# Full survivor bounding box width as seen from nadir (top-down). This is NOT
# just shoulder width — it's the full detection bbox encompassing a person who
# may be sprawled, carrying gear, or with arms extended.  Matches the value in
# detection/simulation_adapter.py (survivor_width_m=2.4).
SURVIVOR_BODY_WIDTH_M = 2.4

# Oblique (side-angle) drone camera parameters.
# When oblique_frac > 0, a fraction of training images simulate a tilted camera
# (15-45° from nadir), producing partially foreshortened survivors that appear
# more elongated/upright than pure top-down views.
OBLIQUE_TILT_MIN_DEG = 15.0   # Minimum tilt from nadir (shallow angle)
OBLIQUE_TILT_MAX_DEG = 45.0   # Maximum tilt from nadir (steep side view)
PERSON_HEIGHT_M = 1.75         # Used for oblique foreshortening calculation


def altitude_to_survivor_px(
    altitude_m: float,
    image_size: int = 640,
    fov_deg: float = DRONE_CAMERA_FOV_DEG,
    body_width_m: float = SURVIVOR_BODY_WIDTH_M,
) -> float:
    """Compute the expected pixel width of a survivor at a given drone altitude.

    Uses the pinhole camera model: the ground footprint of the image is
    ``2 * altitude * tan(fov/2)`` meters wide, spread across ``image_size``
    pixels.  A survivor of width ``body_width_m`` therefore occupies
    ``body_width_m / footprint_m * image_size`` pixels.

    At 20 m → ~25 px, 35 m → ~14 px, 50 m → ~10 px (for 640px image, 65° FOV).
    These are *small* — close to the detection limit — which is why we train
    heavily at this scale.
    """
    footprint_m = 2.0 * altitude_m * np.tan(np.radians(fov_deg) / 2.0)
    return body_width_m / footprint_m * image_size


def altitude_to_gsd(
    altitude_m: float,
    image_size: int = 640,
    fov_deg: float = DRONE_CAMERA_FOV_DEG,
) -> float:
    """Ground sample distance (meters per pixel) at a given altitude."""
    footprint_m = 2.0 * altitude_m * np.tan(np.radians(fov_deg) / 2.0)
    return footprint_m / image_size


def oblique_survivor_size(
    altitude_m: float,
    tilt_deg: float,
    image_size: int = 640,
    fov_deg: float = DRONE_CAMERA_FOV_DEG,
    person_height_m: float = PERSON_HEIGHT_M,
    body_width_m: float = SURVIVOR_BODY_WIDTH_M,
) -> tuple[float, float]:
    """Compute survivor pixel width and height for an oblique (tilted) drone camera.

    At nadir (tilt=0), the person appears as a foreshortened circle/blob.
    At oblique angles (tilt>0), the person appears more upright — the apparent
    height increases as the camera tilts toward the horizon.

    Returns (width_px, height_px).
    """
    footprint_m = 2.0 * altitude_m * np.tan(np.radians(fov_deg) / 2.0)
    px_per_m = image_size / footprint_m

    # Width: same as nadir (shoulder width projected horizontally)
    width_px = body_width_m * px_per_m

    # Height: person_height projected at the tilt angle.
    # At nadir (tilt=0): see top of head, apparent height ≈ body_width (circular blob)
    # At tilt_deg: apparent height = person_height * sin(tilt_deg)
    tilt_rad = np.radians(tilt_deg)
    apparent_height_m = person_height_m * np.sin(tilt_rad) + body_width_m * np.cos(tilt_rad)
    height_px = apparent_height_m * px_per_m

    return float(width_px), float(height_px)


def sample_altitude(rng: np.random.Generator) -> float:
    """Sample a drone altitude from the operational flight envelope.

    Uniformly samples between min and max flight levels (20–50 m) to produce
    a realistic altitude distribution matching RL training.
    """
    return float(rng.uniform(DRONE_FLIGHT_LEVELS_M[0], DRONE_FLIGHT_LEVELS_M[-1]))


def _procedural_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Terrain-like background: low-frequency green/brown/tan colour field."""
    low = rng.integers(40, 150, size=(8, 8, 3)).astype(np.float32)
    # Bias toward vegetation/soil hues.
    low[..., 1] *= rng.uniform(1.0, 1.4)   # green
    low[..., 0] *= rng.uniform(0.8, 1.2)   # red/soil
    img = Image.fromarray(np.clip(low, 0, 255).astype("uint8")).resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, 12, arr.shape)    # fine texture
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")


def _load_naip_tiles(naip_dir: str) -> list[Image.Image]:
    """Load cached NAIP tiles to sample real aerial backgrounds from."""
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


def _split_tiles(
    tiles: list[Image.Image], val_frac: float = 0.20
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Deterministic geographic split so train and val never share the same tile.

    The tiles list is already sorted by path, so consecutive tiles from the same
    geographic area stay together.  The first (1-val_frac) fraction is used for
    training; the remainder provides an independent validation background pool.
    """
    n_train = max(1, int(len(tiles) * (1.0 - val_frac)))
    return tiles[:n_train], tiles[n_train:]


def _naip_background(tiles: list[Image.Image], size: int, rng: np.random.Generator) -> Image.Image:
    """Random crop from a real NAIP tile with aggressive augmentation.

    Since only a few NAIP tiles are available, we maximize visual diversity via
    random scale, flip, 90-degree rotation, and color jitter (hue/saturation/
    brightness shifts).  This lets the model generalise beyond the specific
    colour palette of the 4 available tiles.
    """
    tile = tiles[int(rng.integers(0, len(tiles)))]
    W, H = tile.size
    scale = rng.uniform(0.5, 1.0)
    cw = max(8, int(min(W, H) * scale))
    x = int(rng.integers(0, max(1, W - cw))); y = int(rng.integers(0, max(1, H - cw)))
    crop = tile.crop((x, y, x + cw, y + cw)).resize((size, size), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    # Random 90-degree rotation
    rot_k = int(rng.integers(0, 4))
    if rot_k:
        crop = crop.rotate(rot_k * 90, expand=False)
    # Color jitter: shift hue/saturation/brightness to simulate different regions
    arr = np.asarray(crop, dtype=np.float32)
    brightness = rng.uniform(0.85, 1.15)
    arr = arr * brightness
    # Per-channel shift (simulates different soil/vegetation tones)
    for c in range(3):
        arr[..., c] += rng.uniform(-15, 15)
    crop = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    return crop


def _disc(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0), 0, 1).astype(np.float32)


def _wildfire_masks(
    size: int, rng: np.random.Generator, centers: list, heavy_occlude: bool = False
) -> WildfireMasks:
    """Build burn/flame/smoke masks, biased to sit ON survivors.

    ``centers`` are pixel (cx, cy) of placed survivors; fire is centered on a
    random subset so the model sees plenty of survivors *inside* active fire
    and smoke — the hard case stock YOLO fails on.

    When ``heavy_occlude`` is True, a thick smoke column is placed directly over
    each survivor to force the model to learn partial-occlusion detection.
    """
    burned = np.zeros((size, size), np.float32)
    active = np.zeros((size, size), np.float32)
    smoke = np.zeros((size, size), np.float32)
    # 1-2 fire blobs, each centered on a survivor (mostly) or random (sometimes).
    for _ in range(int(rng.integers(1, 3))):
        if centers and rng.random() < 0.8:
            cx, cy = centers[int(rng.integers(0, len(centers)))]
            cx += rng.uniform(-size * 0.05, size * 0.05); cy += rng.uniform(-size * 0.05, size * 0.05)
        else:
            cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        burned = np.maximum(burned, _disc(size, cx, cy, size * rng.uniform(0.18, 0.5)) * rng.uniform(0.5, 1.0))
        active = np.maximum(active, _disc(size, cx, cy, size * rng.uniform(0.12, 0.38)) * rng.uniform(0.5, 1.0))
        smoke_opacity = rng.uniform(0.6, 1.0)
        smoke = np.maximum(smoke, _disc(size, cx, cy, size * rng.uniform(0.25, 0.6)) * smoke_opacity)

    if heavy_occlude and centers:
        for cx, cy in centers:
            col_r = size * rng.uniform(0.06, 0.15)
            smoke = np.maximum(smoke, _disc(size, cx, cy, col_r) * rng.uniform(0.8, 1.0))
            burned = np.maximum(burned, _disc(size, cx, cy, col_r * 1.3) * rng.uniform(0.6, 0.9))
            active = np.maximum(active, _disc(size, cx, cy, col_r * 0.8) * rng.uniform(0.7, 1.0))

    return WildfireMasks(burned=burned, active=active, intensity=active.copy(), smoke=smoke)


# ---------------------------------------------------------------------------
# Compositing helpers — each reduces a specific class of artifact that arises
# when sharp close-up SARD cutouts are pasted onto 0.6 m/px NAIP terrain.
# ---------------------------------------------------------------------------

def _erode_alpha_edge(sprite: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Erode and feather the alpha channel to soften hard GrabCut boundary pixels.

    GrabCut produces a binary silhouette with an abrupt edge.  Varying the
    erosion (0-2 px) and Gaussian feather prevents the model from memorising
    a fixed ring of transitional pixels as a person cue.
    """
    a = sprite.getchannel("A")
    erode_px = int(rng.integers(0, 3))
    if erode_px > 0:
        a = a.filter(ImageFilter.MinFilter(size=erode_px * 2 + 1))
    a = a.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return Image.merge("RGBA", (*sprite.convert("RGB").split(), a))


def _harmonize_color(
    sprite: Image.Image, bg_patch: Image.Image, rng: np.random.Generator
) -> Image.Image:
    """Shift the survivor's per-channel mean toward the local background patch.

    SARD images are taken in a range of lighting conditions unrelated to the
    NAIP tile.  Blending a fraction of the background mean into the survivor
    makes the composite look less like a foreign object, reducing the risk
    that the model learns 'bright/saturated blob on muted NAIP' as the signal.
    """
    alpha = np.asarray(sprite.getchannel("A"), dtype=np.float32) / 255.0
    fg_mask = alpha > 0.5
    if fg_mask.sum() < 5:
        return sprite
    spr = np.asarray(sprite.convert("RGB"), dtype=np.float32)
    bg = np.asarray(bg_patch.convert("RGB"), dtype=np.float32)
    blend = rng.uniform(0.25, 0.55)
    for c in range(3):
        src_mean = float(spr[:, :, c][fg_mask].mean())
        tgt_mean = float(bg[:, :, c].mean())
        spr[:, :, c] = np.clip(spr[:, :, c] + (tgt_mean - src_mean) * blend, 0, 255)
    # Per-image random brightness / contrast so all survivors do not look the same tone.
    brightness = rng.uniform(0.82, 1.18)
    contrast = rng.uniform(0.88, 1.12)
    mid = 127.5
    spr = np.clip((spr - mid) * contrast + mid * brightness, 0, 255)
    result = Image.fromarray(spr.astype(np.uint8))
    return Image.merge("RGBA", (*result.split(), sprite.getchannel("A")))


def _resolution_blur(sprite: Image.Image, w: int, rng: np.random.Generator) -> Image.Image:
    """Blur the survivor to approximate NAIP's 0.6 m/px ground-sample distance.

    A SARD image is captured at close range and is much sharper than anything
    in a NAIP tile at the same pixel size.  Blurring proportionally to apparent
    size removes the sharpness discontinuity at the survivor boundary so the
    model cannot rely on 'sharp island in a blurry tile' as a shortcut cue.
    """
    if w < 30:
        r = rng.uniform(0.9, 1.8)
    elif w < 80:
        r = rng.uniform(0.4, 1.2)
    else:
        r = rng.uniform(0.0, 0.7)
    if r < 0.05:
        return sprite
    rgb = sprite.convert("RGB").filter(ImageFilter.GaussianBlur(radius=r))
    return Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))


# ---------------------------------------------------------------------------
# Real hard-negative decoy asset loading
# ---------------------------------------------------------------------------

def _load_decoy_assets(decoy_dir: str) -> list[Image.Image]:
    """Load pre-extracted RGBA decoy PNGs (VisDrone vehicles, SARD rejected, etc.)."""
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


# ---------------------------------------------------------------------------
# Synthetic hard-negative decoy objects
# ---------------------------------------------------------------------------

def _synthetic_decoy_rgba(
    target_w: int, target_h: int, rng: np.random.Generator
) -> Image.Image:
    """Generate a procedural non-human top-down object (rock / stump / debris) as RGBA.

    These are composited onto training images using the same pipeline as
    survivor assets (harmonize → erode → blur) but carry no YOLO label.
    This forces the model to learn a genuine human silhouette rather than the
    shortcut cue of 'any composited object on NAIP terrain'.

    Four synthetic object archetypes are drawn from overlapping ellipses with
    earth-tone colour palettes that approximate what rocks, stumps, gear piles,
    and debris look like from 30–300 m altitude.
    """
    # Randomise dimensions slightly from the requested target so objects are not
    # all exactly person-width — rocks and stumps vary more in aspect ratio.
    w = max(8, int(target_w * rng.uniform(0.6, 1.4)))
    h = max(8, int(target_h * rng.uniform(0.5, 1.6)))

    # Earth-tone palette.
    archetype = int(rng.integers(0, 4))
    if archetype == 0:      # rock — cool gray
        r0 = int(rng.integers(80, 160)); g0 = int(rng.integers(75, 155)); b0 = int(rng.integers(70, 150))
    elif archetype == 1:    # soil / dirt — warm brown
        r0 = int(rng.integers(100, 170)); g0 = int(rng.integers(65, 125)); b0 = int(rng.integers(35, 85))
    elif archetype == 2:    # dead vegetation / stump — tan / ochre
        r0 = int(rng.integers(140, 200)); g0 = int(rng.integers(115, 165)); b0 = int(rng.integers(55, 105))
    else:                   # dark gear / debris — dark olive / gray-green
        r0 = int(rng.integers(45, 105)); g0 = int(rng.integers(50, 110)); b0 = int(rng.integers(40, 95))

    # Alpha mask: 2–5 overlapping ellipses → irregular blob silhouette.
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

    # Textured RGB: base colour + Gaussian noise + low-frequency shading.
    rgb = np.full((h, w, 3), [r0, g0, b0], dtype=np.float32)
    rgb += rng.normal(0, 14, rgb.shape)
    lf = rng.uniform(-18, 18, (4, 4, 3)).astype(np.float32)
    lf_up = np.asarray(
        Image.fromarray(np.clip(lf + 128, 0, 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    rgb = np.clip(rgb + lf_up - 128, 0, 255).astype(np.uint8)

    result = Image.fromarray(rgb)
    # Resize to the originally requested dimensions so placement arithmetic stays simple.
    result = Image.merge("RGBA", (*result.split(), alpha_img)).resize(
        (target_w, target_h), Image.Resampling.LANCZOS
    )
    return result


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _clip_label(x: int, y: int, w: int, h: int, size: int) -> tuple[float, float, float, float] | None:
    """Clip a survivor placement to the image boundary and return a YOLO label tuple.

    Returns (cx_norm, cy_norm, w_norm, h_norm) using the visible (clipped) bbox,
    or None if the visible area is too small to label (< 4 px on either axis).
    """
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(size, x + w); y2 = min(size, y + h)
    cw = x2 - x1; ch = y2 - y1
    if cw < 4 or ch < 4:
        return None
    return (x1 + x2) / 2.0 / size, (y1 + y2) / 2.0 / size, cw / size, ch / size


def _generate_split(
    out_dir: Path,
    n: int,
    assets: list[Image.Image],
    size: int,
    rng: np.random.Generator,
    cfg: WildfireEffectConfig,
    fire_frac: float = 0.65,
    surv_px: tuple[int, int] = (25, 230),
    naip_tiles: list[Image.Image] | None = None,
    neg_frac: float = 0.12,
    decoy_frac: float = 0.20,
    decoy_assets: list[Image.Image] | None = None,
    boundary_frac: float = 0.10,
    small_frac: float = 0.30,
    heavy_occlude_frac: float = 0.20,
) -> None:
    """Generate *n* composite images into *out_dir*.

    neg_frac:      fraction of images that are pure background (no survivor label).
                   Essential so the model learns that not every NAIP crop contains a
                   person.

    decoy_frac:    fraction of images that contain 1–3 hard-negative decoy objects
                   composited with the SAME pipeline as survivors but carrying NO
                   label.  Decoys are drawn from *decoy_assets* when provided
                   (real VisDrone vehicles / SARD rejected crops), otherwise from
                   procedural synthetic blobs.

    boundary_frac: fraction of survivor placements that are allowed to overlap the
                   image edge (partially outside the frame).  Labels use the visible
                   (clipped) bbox.  Closes the train/deploy mismatch where deployment
                   can render edge-truncated survivors but training never did.

    small_frac:    fraction of survivor placements forced to 25–50 px width (the
                   hardest detection scale at realistic drone altitudes).

    heavy_occlude_frac: fraction of fire images that use heavy smoke directly on
                   survivors to train partial-occlusion robustness.
    """
    img_dir = out_dir / "images"; lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = int(surv_px[0]), int(surv_px[1])
    n_neg = 0
    for i in range(n):
        # Real NAIP crops when available (matches deployment imagery and kills
        # the false positives a procedural-only model fires on real terrain);
        # mix in some procedural for variety.
        if naip_tiles and rng.random() < 0.85:
            bg = _naip_background(naip_tiles, size, rng)
        else:
            bg = _procedural_background(size, rng)

        # Negative-only images: survivor-free backgrounds teach the model that
        # NAIP terrain does not automatically imply a person is present.
        is_negative = rng.random() < neg_frac

        placements: list[tuple[Image.Image, int, int, int, int]] = []
        if not is_negative:
            for _ in range(int(rng.integers(1, 4))):   # 1-3 survivors
                asset = assets[int(rng.integers(0, len(assets)))]
                # Force small survivors for a fraction of placements to improve
                # recall at the hardest detection scale (20-50px at altitude).
                if rng.random() < small_frac:
                    w = int(rng.integers(max(lo, 20), min(50, hi)))
                else:
                    w = int(rng.integers(lo, hi))
                h = int(w * asset.height / asset.width)
                if rng.random() < boundary_frac:
                    x = int(rng.integers(-w // 2, max(1, size - w // 2)))
                    y = int(rng.integers(-h // 2, max(1, size - h // 2)))
                else:
                    x = int(rng.integers(0, max(1, size - w)))
                    y = int(rng.integers(0, max(1, size - h)))
                placements.append((asset, w, h, x, y))
            n_neg -= 1   # track below

        centers = [(x + w / 2, y + h / 2) for (_, w, h, x, y) in placements]
        has_fire = rng.random() < fire_frac
        heavy_occlude = has_fire and rng.random() < heavy_occlude_frac
        mask = _wildfire_masks(size, rng, centers, heavy_occlude=heavy_occlude) if has_fire else None
        if mask is not None:    # burn + flame UNDER survivors (production order)
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg, include_burn=True, include_flame=True, include_smoke=False)

        labels: list[str] = []
        for asset, w, h, x, y in placements:
            s = asset.resize((w, h), Image.Resampling.LANCZOS)
            # Sample only the visible patch of the background for color harmonization.
            px1 = max(0, x); py1 = max(0, y)
            px2 = min(size, x + w); py2 = min(size, y + h)
            bg_patch = bg.crop((px1, py1, px2, py2)) if px2 > px1 and py2 > py1 else bg.crop((0, 0, w, h))
            s = _harmonize_color(s, bg_patch, rng)   # tone match to local BG
            s = _erode_alpha_edge(s, rng)             # soften GrabCut hard edges
            s = _resolution_blur(s, w, rng)           # match NAIP sharpness
            bg.paste(s, (x, y), s)
            # Use the visible (clipped) bbox for the YOLO label so that partially
            # out-of-frame survivors get correct annotations.
            clipped = _clip_label(x, y, w, h, size)
            if clipped is not None:
                cx_n, cy_n, w_n, h_n = clipped
                labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        if mask is not None:    # smoke drifts OVER survivors
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg, include_burn=False, include_flame=False, include_smoke=True)

        # Hard-negative decoys: non-human objects composited the same way as
        # survivors but never labeled.  Applied regardless of whether the image is
        # positive, negative, or mixed — the model must learn to ignore them.
        # Prefer REAL assets (VisDrone vehicles, SARD rejected) when available;
        # fall back to procedural synthetic blobs otherwise.
        if rng.random() < decoy_frac:
            n_decoys = int(rng.integers(1, 4))
            for _ in range(n_decoys):
                dw = int(rng.integers(lo, hi))
                dx = int(rng.integers(0, max(1, size - dw)))
                if decoy_assets:
                    # Real asset: resize to target width preserving aspect ratio.
                    raw = decoy_assets[int(rng.integers(0, len(decoy_assets)))]
                    dh = max(8, int(dw * raw.height / raw.width))
                    dy = int(rng.integers(0, max(1, size - dh)))
                    decoy = raw.resize((dw, dh), Image.Resampling.LANCZOS)
                else:
                    # Fallback: procedural blob (non-human aspect ratios).
                    dh = int(dw * rng.uniform(0.5, 1.8))
                    dy = int(rng.integers(0, max(1, size - dh)))
                    decoy = _synthetic_decoy_rgba(dw, dh, rng)
                bg_patch = bg.crop((dx, dy, dx + dw, dy + dh))
                decoy = _harmonize_color(decoy, bg_patch, rng)
                decoy = _erode_alpha_edge(decoy, rng)
                decoy = _resolution_blur(decoy, dw, rng)
                bg.paste(decoy, (dx, dy), decoy)
                # Intentionally no label — this is the hard-negative signal.

        bg.save(img_dir / f"{i:05d}.jpg", quality=92)
        # Empty label file is the YOLO convention for a negative image.
        label_text = "\n".join(labels) + ("\n" if labels else "")
        (lbl_dir / f"{i:05d}.txt").write_text(label_text, encoding="utf-8")
        if is_negative:
            n_neg += 1

    neg_pct = 100 * n_neg / n if n else 0
    print(f"  {out_dir.name}: {n} images ({n_neg} negatives, {neg_pct:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--size", type=int, default=640, help="Generated training image size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fire-frac", type=float, default=0.65,
                    help="Fraction of training images with fire/smoke centered on survivors.")
    ap.add_argument("--neg-frac", type=float, default=0.12,
                    help="Fraction of training images that are negative-only (no survivor label). "
                         "These teach the model that not every NAIP crop contains a person.")
    ap.add_argument("--boundary-frac", type=float, default=0.10,
                    help="Fraction of survivor placements allowed to overlap the image edge. "
                         "Labels use the visible (clipped) bbox. Closes the train/deploy gap "
                         "where deployment renders partially cropped survivors at the frame edge.")
    ap.add_argument("--decoy-frac", type=float, default=0.20,
                    help="Fraction of training images that contain 1-3 synthetic non-human objects "
                         "(rocks, stumps, debris) composited identically to survivors but with no label. "
                         "Hard negatives: forces the model to learn genuine human silhouette rather "
                         "than 'any composited blob on NAIP'.")
    ap.add_argument("--min-surv-px", type=int, default=25,
                    help="Min survivor width in the generated image. Default 25px covers "
                         "realistic drone altitudes. Use larger (e.g. 120) for ground-robot model.")
    ap.add_argument("--max-surv-px", type=int, default=230,
                    help="Max survivor width in the generated image.")
    ap.add_argument("--small-frac", type=float, default=0.30,
                    help="Fraction of survivor placements forced to 25-50px width (the hardest "
                         "detection scale). Higher values improve small-object recall.")
    ap.add_argument("--heavy-occlude-frac", type=float, default=0.20,
                    help="Fraction of fire images with thick smoke placed directly on survivors "
                         "to train partial-occlusion detection.")
    ap.add_argument("--naip-dir", default=None,
                    help="Cached NAIP tile dir to sample REAL aerial backgrounds from "
                         "(e.g. data/source_cache/naip/naip_tiles_*). Strongly recommended for real deployment. "
                         "Tiles are sorted alphabetically and split 80/20 into train/val pools so the "
                         "two splits never share the same geographic crop.")
    ap.add_argument("--naip-val-dir", default=None,
                    help="Optional separate NAIP tile directory used ONLY for validation composites. "
                         "Use a different geographic area than --naip-dir for the strongest domain-shift test.")
    ap.add_argument("--assets-dir", default=str(ROOT / "data/cv_assets/sard_grabcut"))
    ap.add_argument(
        "--review-json",
        default=str(ROOT / "configs/cv/sard_grabcut_asset_review.json"),
        help="SARD asset review JSON.  When present, only 'accepted_assets' are used "
             "for training so rejected/ambiguous cutouts stay out of the positive class.",
    )
    ap.add_argument(
        "--decoy-assets-dir",
        default=None,
        help="Directory of real non-human RGBA PNGs to use as hard-negative decoys "
             "(e.g. data/cv_assets/visdrone_decoys or data/cv_assets/sard_decoys). "
             "Generate these first with extract_visdrone_decoys.py or "
             "extract_sard_rejected_decoys.py.  Falls back to procedural blobs if omitted.",
    )
    ap.add_argument("--data-dir", default=str(ROOT / "data/cv_train/survivor"))
    ap.add_argument("--out", default=str(ROOT / "models/survivor_yolov8n.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    cfg = WildfireEffectConfig()

    # Load SARD survivor assets, filtering to accepted-only when a review JSON exists.
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
                print(f"Filtering survivor assets to {len(filtered)} accepted "
                      f"(of {len(all_asset_paths)} total) using {review_path.name}")
                all_asset_paths = filtered
            else:
                print(f"WARNING: review JSON found but no paths matched; using all {len(all_asset_paths)} assets")
    else:
        print(f"No review JSON at {review_path}; using all {len(all_asset_paths)} survivor assets")

    assets = [Image.open(p).convert("RGBA") for p in all_asset_paths]
    # Hold out last ~20% of SARD assets for val so we test pose generalisation.
    n_hold = max(1, len(assets) // 5)
    train_assets, val_assets = assets[:-n_hold], assets[-n_hold:]

    # Load real non-human decoy assets if a directory was provided.
    decoy_assets: list[Image.Image] | None = None
    if args.decoy_assets_dir:
        decoy_assets = _load_decoy_assets(args.decoy_assets_dir)
        if decoy_assets:
            print(f"Loaded {len(decoy_assets)} real decoy assets from {args.decoy_assets_dir}")
        else:
            print(f"WARNING: no decoy assets found in {args.decoy_assets_dir}; falling back to procedural blobs")

    # NAIP tile pools — split geographically so train and val backgrounds are
    # from different areas, mirroring the out-of-distribution concern.
    naip_train_tiles: list[Image.Image] | None = None
    naip_val_tiles: list[Image.Image] | None = None
    if args.naip_dir:
        all_tiles = _load_naip_tiles(args.naip_dir)
        if all_tiles:
            if args.naip_val_dir:
                # Separate directory: use all tiles from naip_dir for train only.
                naip_train_tiles = all_tiles
                naip_val_tiles = _load_naip_tiles(args.naip_val_dir) or all_tiles
                print(f"NAIP train tiles: {len(naip_train_tiles)} (from {args.naip_dir})")
                print(f"NAIP  val tiles: {len(naip_val_tiles)} (from {args.naip_val_dir})")
            else:
                # Single directory: 80/20 geographic split by tile index.
                naip_train_tiles, naip_val_tiles = _split_tiles(all_tiles, val_frac=0.20)
                if not naip_val_tiles:
                    naip_val_tiles = naip_train_tiles   # fallback if only 1 tile
                print(f"NAIP tiles split: {len(naip_train_tiles)} train / {len(naip_val_tiles)} val "
                      f"(geographic holdout from sorted tile list)")
        else:
            print(f"WARNING: no NAIP tiles found in {args.naip_dir}; using procedural backgrounds.")

    data_dir = Path(args.data_dir)
    surv_px = (args.min_surv_px, args.max_surv_px)
    print(f"Generating {args.n_train} train / {args.n_val} val composites at {args.size}px "
          f"({int(args.fire_frac*100)}% with fire/smoke, {int(args.neg_frac*100)}% negatives, "
          f"{int(args.decoy_frac*100)}% with hard-negative decoys) ...")
    _generate_split(
        data_dir / "train", args.n_train, train_assets, args.size, rng, cfg,
        fire_frac=args.fire_frac, surv_px=surv_px,
        naip_tiles=naip_train_tiles, neg_frac=args.neg_frac, decoy_frac=args.decoy_frac,
        decoy_assets=decoy_assets, boundary_frac=args.boundary_frac,
        small_frac=args.small_frac, heavy_occlude_frac=args.heavy_occlude_frac,
    )
    _generate_split(
        data_dir / "val", args.n_val, val_assets, args.size, rng, cfg,
        fire_frac=args.fire_frac, surv_px=surv_px,
        naip_tiles=naip_val_tiles, neg_frac=args.neg_frac, decoy_frac=args.decoy_frac,
        decoy_assets=decoy_assets, boundary_frac=args.boundary_frac,
        small_frac=args.small_frac, heavy_occlude_frac=args.heavy_occlude_frac,
    )

    yaml = data_dir / "survivor.yaml"
    yaml.write_text(
        f"path: {data_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n",
        encoding="utf-8",
    )

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"Fine-tuning {args.model} for {args.epochs} epochs ...")
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=16,
        device="cpu", verbose=True, seed=args.seed, project=str(data_dir.resolve() / "runs"),
        name="survivor", exist_ok=True, plots=False,
    )
    # Ultralytics records the best-weights path on the trainer; fall back to a
    # search (relative project paths get re-rooted under runs/detect/).
    best = Path(getattr(model.trainer, "best", data_dir / "runs" / "survivor" / "weights" / "best.pt"))
    if not best.exists():
        found = list(ROOT.rglob("survivor/weights/best.pt"))
        if found:
            best = max(found, key=lambda p: p.stat().st_mtime)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(best, out)
    print(f"\nWrote fine-tuned survivor detector to: {out}")
    print(f"Use it with: python scripts/export_trajectories.py --enable-cv --cv-detector yolo --cv-person-model {out}")


if __name__ == "__main__":
    main()
