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
from PIL import Image, ImageDraw, ImageFilter

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

# Physical survivor dimensions — a real human, not a vehicle-sized blob.
# A person is modelled as an oriented box: shoulder width ~0.55 m, front-to-back
# depth ~0.35 m, and standing height / prone body-length ~1.75 m. The detection
# bounding box is the axis-aligned box of that body at a random heading and pose
# (standing vs prone), so it is anisotropic and pose-dependent — NOT the old
# ~2.4 m near-square blob (which implied a 6-8 m^2 footprint for one person).
PERSON_HEIGHT_M = 1.75          # standing height / prone body length (long axis)
PERSON_SHOULDER_M = 0.55        # shoulder width (short axis, top-down)
PERSON_DEPTH_M = 0.35           # front-to-back thickness
PRONE_FRAC = 0.45               # fraction of survivors lying prone
MAX_SURVIVOR_LONG_AXIS_M = 2.0  # hard ceiling on any survivor bbox dimension
MAX_NADIR_SHORT_AXIS_M = 0.7    # nadir footprint short axis (audit rule)
SURVIVOR_MIN_PX = 5             # floor so far-away survivors aren't zero-size

# Hard-negative decoys, sized by real-world footprint through the frame GSD.
# Vehicle mix is biased toward passenger cars (4.2-5 m in reality): the old
# uniform 3-8 m + 15% buses read as visually oversized next to 0.5-2 m people.
DECOY_VEHICLE_LONG_M = (3.5, 4.8)   # compact sedans — the only vehicle class now.
# Vans and buses are removed: they read as visually oversized next to 0.5-1.8 m
# survivors, letting the model reject them by size alone. The human-scale decoy
# classes (animal, colorful_object) cover the same-size case; compact cars cover
# the "larger-but-not-huge" real-world vehicle that appears on Malibu fire roads.
DECOY_VAN_LONG_M = (5.2, 8.0)       # kept for reference but not used
DECOY_LARGE_VEHICLE_M = (8.0, 13.0) # kept for reference but not used
DECOY_VAN_FRAC = 0.0                 # disabled
DECOY_LARGE_FRAC = 0.0               # disabled
DECOY_MIN_PX = 6

# Human-scale decoys: animals and colourful human-sized objects (tarps,
# jackets, gear) that match survivors in SIZE, COLOUR and CONTRAST. Vehicles
# alone are too easy to reject (large, white/gray) — the model could pass by
# learning "colourful small blob = human" or by keying on paste artifacts.
# These decoys share the survivors' size band and clothing palette and go
# through the identical composite pipeline, so the only remaining separator
# is genuine human shape.
DECOY_HUMAN_SCALE_FRAC = 0.5        # fraction of decoys at human scale
DECOY_HUMAN_SCALE_LONG_M = (0.5, 1.8)

# Back-compat alias: the "body width" used by the pinhole helpers is the
# shoulder width (short axis, top-down), NOT the old oversized 2.4 m value.
SURVIVOR_BODY_WIDTH_M = PERSON_SHOULDER_M

# Oblique (side-angle) drone camera parameters.
# When oblique_frac > 0, a fraction of training images simulate a tilted camera
# (15-45° from nadir), producing partially foreshortened survivors that appear
# more elongated/upright than pure top-down views.
OBLIQUE_TILT_MIN_DEG = 15.0   # Minimum tilt from nadir (shallow angle)
OBLIQUE_TILT_MAX_DEG = 45.0   # Maximum tilt from nadir (steep side view)


def altitude_to_survivor_px(
    altitude_m: float,
    image_size: int = 640,
    fov_deg: float = DRONE_CAMERA_FOV_DEG,
    body_width_m: float = SURVIVOR_BODY_WIDTH_M,
) -> float:
    """Expected pixel *shoulder width* of a survivor at a given drone altitude.

    Uses the pinhole camera model: the ground footprint of the image is
    ``2 * altitude * tan(fov/2)`` meters wide, spread across ``image_size``
    pixels.  A survivor of shoulder width ``body_width_m`` (~0.55 m) therefore
    occupies ``body_width_m / footprint_m * image_size`` pixels.

    At 20 m → ~14 px, 35 m → ~8 px, 50 m → ~5 px (for 640px image, 65° FOV).
    These are the *short axis*; a prone body's long axis is ~3x larger. Survivors
    are genuinely tiny at altitude — close to the detection limit — which is why
    we train heavily at this scale.
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


def survivor_footprint_m(
    rng: np.random.Generator,
    *,
    is_oblique: bool = False,
    tilt_deg: float = 0.0,
) -> tuple[float, float, str, float]:
    """Physically-grounded survivor bounding-box footprint in metres.

    Models a person as an oriented box (shoulder x depth x height) at a random
    heading and pose, and returns the axis-aligned bounding box ``(w_m, h_m)``
    plus the sampled ``pose`` and ``heading_deg``:

      * prone  — lying flat: a ``1.75 x 0.55 m`` rectangle rotated by heading,
        so the axis-aligned box is elongated (long axis up to ~1.75 m).
      * standing, nadir — compact head+shoulders blob (~0.4-0.55 m each way).
      * standing, oblique — apparent height grows with camera tilt (person seen
        more side-on), up to ~1.4 m tall at 45°.

    A ±15% jitter models body size/pose variation, and the long axis is capped
    at ``MAX_SURVIVOR_LONG_AXIS_M`` (2.0 m). The result is anisotropic and
    pose-dependent — never a fixed ~2.4 m near-square blob.
    """
    prone = bool(rng.random() < PRONE_FRAC)
    if prone and not is_oblique:
        # Nadir prone bodies keep a near-axis-aligned heading so the emitted
        # axis-aligned box stays elongated (~0.5 x 1.8 m) instead of degrading
        # into a large square AABB at diagonal headings. The audit rule caps
        # the nadir short axis at 0.7 m (MAX_NADIR_SHORT_AXIS_M).
        axis = 0.0 if rng.random() < 0.5 else np.pi / 2.0
        heading = float(axis + np.radians(rng.uniform(-8.0, 8.0)))
    else:
        heading = float(rng.uniform(0.0, np.pi))
    c, s = abs(np.cos(heading)), abs(np.sin(heading))
    if prone:
        length, width = PERSON_HEIGHT_M, PERSON_SHOULDER_M
        w_m = length * c + width * s
        h_m = length * s + width * c
        pose = "prone"
    else:
        cross = PERSON_SHOULDER_M * s + PERSON_DEPTH_M * c
        if is_oblique and tilt_deg > 0.0:
            tilt = np.radians(tilt_deg)
            vert = PERSON_HEIGHT_M * np.sin(tilt) + PERSON_DEPTH_M * np.cos(tilt)
            w_m, h_m = cross, max(cross, float(vert))
        else:
            w_m = cross
            h_m = PERSON_SHOULDER_M * c + PERSON_DEPTH_M * s
        pose = "standing"
    w_m *= float(rng.uniform(0.85, 1.15))
    h_m *= float(rng.uniform(0.85, 1.15))
    # Cap with a small margin under the 2.0 m audit ceiling so pixel rounding
    # and the resolution-blur alpha halo cannot push an emitted box past it.
    scale = min(1.0, (MAX_SURVIVOR_LONG_AXIS_M * 0.96) / max(w_m, h_m))
    w_m *= scale
    h_m *= scale
    if not is_oblique:
        # From straight above, a body's cross-section never exceeds ~0.5-0.6 m
        # (prone width / shoulder span); clamp the short axis below the 0.7 m
        # audit ceiling regardless of heading deviation and jitter.
        cap = MAX_NADIR_SHORT_AXIS_M * 0.93
        if min(w_m, h_m) > cap:
            if w_m <= h_m:
                w_m = cap
            else:
                h_m = cap
    return w_m, h_m, pose, float(np.degrees(heading))


def survivor_box_px(
    rng: np.random.Generator,
    gsd_m: float,
    *,
    is_oblique: bool = False,
    tilt_deg: float = 0.0,
) -> tuple[int, int, str, float]:
    """Survivor pixel box ``(w_px, h_px, pose, heading_deg)`` at a given GSD."""
    w_m, h_m, pose, heading = survivor_footprint_m(
        rng, is_oblique=is_oblique, tilt_deg=tilt_deg
    )
    w_px = max(SURVIVOR_MIN_PX, int(round(w_m / gsd_m)))
    h_px = max(SURVIVOR_MIN_PX, int(round(h_m / gsd_m)))
    return w_px, h_px, pose, heading


def sample_altitude(rng: np.random.Generator) -> float:
    """Sample a drone altitude from the operational flight envelope.

    Uniformly samples between min and max flight levels (20–50 m) to produce
    a realistic altitude distribution matching RL training.
    """
    return float(rng.uniform(DRONE_FLIGHT_LEVELS_M[0], DRONE_FLIGHT_LEVELS_M[-1]))


# Within one frame every survivor shares the same GSD, so their pixel sizes may
# only differ by pose/body variation. Cap the max/min sqrt(box-area) ratio; a
# frame mixing a prone adult and a standing child stays under ~2x. The generator
# enforces a tighter bound so that residual erosion/blur shrinkage of the final
# alpha mask cannot push emitted boxes past the 2.0 audit rule.
FRAME_SIZE_RATIO_MAX = 2.0
GEN_SIZE_RATIO_MAX = 1.7
ALPHA_BBOX_THRESHOLD = 128
# Below this sprite size, alpha erosion can wipe the silhouette entirely
# (MinFilter of 2 px on a 6 px blob leaves nothing above threshold).
MIN_ERODE_SPRITE_PX = 16


def _tight_alpha_bbox(
    sprite: Image.Image, threshold: int = ALPHA_BBOX_THRESHOLD
) -> tuple[int, int, int, int] | None:
    """Tight bbox (x1, y1, x2, y2) of the sprite's alpha > threshold, or None."""
    a = np.asarray(sprite.getchannel("A"))
    mask = a > threshold
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _crop_to_alpha(sprite: Image.Image) -> Image.Image:
    """Crop the sprite to its tight alpha bbox (removes transparent margins)."""
    bb = _tight_alpha_bbox(sprite)
    return sprite.crop(bb) if bb else sprite


def _scale_sprite_into_box(sprite: Image.Image, w_px: int, h_px: int) -> Image.Image:
    """Uniformly scale the sprite so its tight alpha extent fits (w_px, h_px).

    Aspect-preserving: the sprite is never stretched to fill the axis-aligned
    box. The emitted label is later derived from the pasted sprite's actual
    alpha bbox, so labels stay tight regardless of the residual short-axis gap.
    """
    sprite = _crop_to_alpha(sprite)
    tw, th = sprite.size
    scale = min(w_px / max(1, tw), h_px / max(1, th))
    nw = max(2, int(round(tw * scale)))
    nh = max(2, int(round(th * scale)))
    return sprite.resize((nw, nh), Image.Resampling.LANCZOS)


def _top_down_standing_sprite(asset: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Synthesize a plausible top-down view of a STANDING person.

    A drone at nadir sees only the head and shoulders of a standing person — a
    compact roughly-elliptical blob — never the recognizable front/side profile
    the SARD ground-level photos show. We approximate that view by cropping the
    upper (head + shoulders) region of the cutout, squashing it slightly, and
    masking with an ellipse so the silhouette reads as a foreshortened blob
    with realistic hair/clothing colours.
    """
    asset = _crop_to_alpha(asset)
    w, h = asset.size
    crop = asset.crop((0, 0, w, max(2, int(h * 0.35))))
    cw, ch = crop.size
    # Squash vertically: from above, head+shoulders depth < shoulder width.
    target_h = max(2, int(cw * rng.uniform(0.65, 0.95)))
    crop = crop.resize((cw, target_h), Image.Resampling.LANCZOS)
    ellipse = Image.new("L", crop.size, 0)
    ImageDraw.Draw(ellipse).ellipse([0, 0, crop.size[0] - 1, crop.size[1] - 1], fill=255)
    a = np.asarray(crop.getchannel("A"), dtype=np.float32)
    e = np.asarray(ellipse, dtype=np.float32) / 255.0
    new_alpha = Image.fromarray(np.clip(a * e, 0, 255).astype(np.uint8))
    return Image.merge("RGBA", (*crop.convert("RGB").split(), new_alpha))


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


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


def _resolution_blur(
    sprite: Image.Image,
    w: int,
    rng: np.random.Generator,
    gsd_m: float | None = None,
) -> Image.Image:
    """Blur the survivor to approximate the effective resolution at flight altitude.

    When ``gsd_m`` (ground sample distance in meters/pixel) is provided, the blur
    radius is derived from the physical resolution: higher altitude → larger GSD →
    more blur.  The relationship is:
      - NAIP baseline GSD: 0.6 m/px (reference sharpness of the background)
      - Drone at 20 m, 65° FOV, 640px: GSD ≈ 0.020 m/px (sharp — almost no blur)
      - Drone at 50 m: GSD ≈ 0.050 m/px (still sharper than NAIP)

    Even though the drone's own GSD is better than NAIP, the SARD source images
    are captured at *much* closer range (~2–5 m) so they need downsampling blur.
    The amount scales with altitude: at 50 m objects are fuzzier than at 20 m.

    Falls back to the size-heuristic when altitude is not available.
    """
    if gsd_m is not None:
        # Higher GSD means more blur. Scale: at GSD 0.02 (20m alt) → mild blur,
        # at GSD 0.05 (50m alt) → moderate blur.  SARD close-range images need
        # the most softening when placed small (high altitude).
        # Empirical mapping: radius ≈ 25 * gsd_m + jitter
        base_r = 25.0 * gsd_m  # ~0.5 at 20m, ~1.25 at 50m
        r = float(rng.uniform(base_r * 0.7, base_r * 1.3))
    else:
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


def _animal_decoy_rgba(
    target_w: int, target_h: int, rng: np.random.Generator
) -> Image.Image:
    """Procedural top-down ANIMAL decoy (deer / dog / coyote scale).

    An elongated body ellipse with a smaller head blob at one end and a fur-
    tone palette. From 20-50 m a quadruped is a human-sized elongated blob —
    the closest natural confuser for a prone survivor — so these force the
    model to separate people from animals by silhouette, not by size.
    """
    w, h = max(8, target_w), max(8, target_h)
    # Fur palette: tan / brown / grey / near-black / off-white.
    palettes = [
        (168, 132, 92), (120, 88, 58), (140, 140, 135),
        (60, 52, 46), (205, 198, 185),
    ]
    r0, g0, b0 = palettes[int(rng.integers(0, len(palettes)))]

    yy, xx = np.mgrid[0:h, 0:w]
    horizontal = w >= h
    body_len = (w if horizontal else h)
    # Body: ellipse spanning ~70% of the long axis.
    bx = w * (0.42 if horizontal else 0.5)
    by = h * (0.5 if horizontal else 0.42)
    brx = body_len * 0.36 if horizontal else w * 0.38
    bry = h * 0.38 if horizontal else body_len * 0.36
    blob = np.clip(1.0 - np.sqrt(((xx - bx) / max(1.5, brx)) ** 2
                                 + ((yy - by) / max(1.5, bry)) ** 2), 0, 1)
    # Head: smaller blob overlapping one end of the body ellipse.
    hx = w * (0.80 if horizontal else 0.5)
    hy = h * (0.5 if horizontal else 0.80)
    hr = max(1.5, body_len * 0.14)
    head = np.clip(1.0 - np.sqrt(((xx - hx) / hr) ** 2
                                 + ((yy - hy) / hr) ** 2), 0, 1)
    blob = np.maximum(blob, head)
    alpha = ((blob > 0.28).astype(np.uint8) * 255)
    alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=0.7))

    rgb = np.full((h, w, 3), [r0, g0, b0], dtype=np.float32)
    rgb += rng.normal(0, 12, rgb.shape)          # fur texture
    # Darker dorsal stripe along the spine.
    stripe = np.exp(-(((yy - by) if horizontal else (xx - bx)) ** 2)
                    / max(1.0, (0.12 * (h if horizontal else w)) ** 2))
    rgb -= (stripe * 22)[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.merge("RGBA", (*Image.fromarray(rgb).split(), alpha_img))


def _colorful_decoy_rgba(
    target_w: int, target_h: int, rng: np.random.Generator
) -> Image.Image:
    """Procedural COLOURFUL human-scale decoy (tarp, jacket, tent, gear).

    Saturated clothing-like colours at survivor size and contrast. Survivor
    sprites often wear bright clothing; without equally bright non-human
    objects the model can shortcut on "saturated small pixels = person".
    Optionally two-tone (like a jacket + pants) to mimic clothing colour
    boundaries without human anatomy.
    """
    import colorsys
    w, h = max(8, target_w), max(8, target_h)

    def clothing_color(rng) -> tuple[int, int, int]:
        hue = float(rng.uniform(0.0, 1.0))                 # any hue
        sat = float(rng.uniform(0.55, 0.95))               # saturated
        val = float(rng.uniform(0.45, 0.95))               # visible
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return int(r * 255), int(g * 255), int(b * 255)

    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.zeros((h, w), dtype=np.float32)
    # Shape families. "blob" alone was too narrow a distribution — the model
    # rejected round tarps but still fired on anything elongated or bent
    # (the shapes that most resemble a prone person). Cover those too.
    shape = rng.choice(["blob", "elongated", "bent", "lobed"])
    if shape == "elongated":
        # Rolled tarp / sleeping bag: one long thin ellipse at a random angle.
        ang = rng.uniform(0, np.pi)
        ca, sa = np.cos(ang), np.sin(ang)
        u = (xx - w / 2) * ca + (yy - h / 2) * sa
        v = -(xx - w / 2) * sa + (yy - h / 2) * ca
        ru = max(2.0, 0.48 * max(w, h))
        rv = max(1.0, rng.uniform(0.12, 0.24) * min(w, h))
        blob = np.clip(1.0 - np.sqrt((u / ru) ** 2 + (v / rv) ** 2), 0, 1)
    elif shape == "bent":
        # Bent/L shape: two thin ellipses joined at an end (limb-like geometry
        # without anatomy).
        for k in range(2):
            ang = rng.uniform(0, np.pi)
            ca, sa = np.cos(ang), np.sin(ang)
            cx = rng.uniform(w * 0.35, w * 0.65)
            cy = rng.uniform(h * 0.35, h * 0.65)
            u = (xx - cx) * ca + (yy - cy) * sa
            v = -(xx - cx) * sa + (yy - cy) * ca
            ru = max(2.0, rng.uniform(0.3, 0.5) * max(w, h))
            rv = max(1.0, rng.uniform(0.10, 0.2) * min(w, h))
            blob = np.maximum(blob, np.clip(
                1.0 - np.sqrt((u / ru) ** 2 + (v / rv) ** 2), 0, 1))
    elif shape == "lobed":
        # Torso+limb proportions: one big lobe with 1-2 small satellites.
        cx, cy = w * 0.5, h * 0.5
        blob = np.clip(1.0 - np.sqrt(((xx - cx) / (w * 0.32)) ** 2
                                     + ((yy - cy) / (h * 0.32)) ** 2), 0, 1)
        for _ in range(int(rng.integers(1, 3))):
            sx = rng.uniform(w * 0.15, w * 0.85)
            sy = rng.uniform(h * 0.15, h * 0.85)
            sr = max(1.0, rng.uniform(0.10, 0.2) * min(w, h))
            blob = np.maximum(blob, np.clip(
                1.0 - np.sqrt(((xx - sx) / sr) ** 2 + ((yy - sy) / sr) ** 2), 0, 1))
    else:
        # Irregular blob (2-4 overlapping ellipses), like crumpled fabric.
        for _ in range(int(rng.integers(2, 5))):
            cx = rng.uniform(w * 0.2, w * 0.8)
            cy = rng.uniform(h * 0.2, h * 0.8)
            rx = max(1.0, rng.uniform(w * 0.18, w * 0.45))
            ry = max(1.0, rng.uniform(h * 0.18, h * 0.45))
            blob = np.maximum(blob, np.clip(
                1.0 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0, 1))
    alpha = ((blob > 0.38).astype(np.uint8) * 255)
    alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=1.0))

    c1 = clothing_color(rng)
    rgb = np.full((h, w, 3), c1, dtype=np.float32)
    pattern = rng.choice(["solid", "twotone", "stripes", "patch"])
    if pattern == "twotone":
        # Two-tone split at an ARBITRARY angle (jacket/pants boundary is
        # rarely axis-aligned in an aerial view).
        c2 = clothing_color(rng)
        ang = rng.uniform(0, np.pi)
        boundary = (xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)
        rgb[boundary > rng.uniform(-0.15, 0.15) * max(w, h)] = c2
    elif pattern == "stripes":
        # 2-4 colour bands (folded tarp / gear straps).
        n_bands = int(rng.integers(2, 5))
        ang = rng.uniform(0, np.pi)
        band = ((xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang))
        band = ((band - band.min()) / max(1e-6, band.max() - band.min()) * n_bands).astype(int)
        colors = [clothing_color(rng) for _ in range(n_bands + 1)]
        for b in range(n_bands + 1):
            rgb[band == b] = colors[b]
    elif pattern == "patch":
        # Small contrasting patch on a larger body (backpack-on-jacket look).
        c2 = clothing_color(rng)
        px = rng.uniform(w * 0.25, w * 0.75)
        py = rng.uniform(h * 0.25, h * 0.75)
        pr = max(1.0, rng.uniform(0.15, 0.3) * min(w, h))
        patch_mask = ((xx - px) ** 2 + (yy - py) ** 2) < pr ** 2
        rgb[patch_mask] = c2
    rgb += rng.normal(0, float(rng.uniform(6, 18)), rgb.shape)  # fabric texture
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.merge("RGBA", (*Image.fromarray(rgb).split(), alpha_img))


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
    altitude_aware: bool = True,
    oblique_frac: float = 0.25,
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

    altitude_aware: when True (default), each image simulates a specific drone
                   altitude sampled from the operational flight envelope (20–50 m).
                   Survivor pixel size and resolution blur are derived from the
                   altitude via the pinhole camera model, producing a physically
                   realistic size distribution instead of a uniform random range.
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

        # Sample a drone altitude for this image — drives the ground-sample
        # distance and hence survivor pixel size.
        is_oblique = altitude_aware and rng.random() < oblique_frac
        img_tilt_deg = 0.0
        if altitude_aware:
            img_altitude_m = sample_altitude(rng)
            img_gsd = altitude_to_gsd(img_altitude_m, image_size=size)
            if is_oblique:
                img_tilt_deg = float(rng.uniform(OBLIQUE_TILT_MIN_DEG, OBLIQUE_TILT_MAX_DEG))
        else:
            img_altitude_m = None
            img_gsd = None

        # Negative-only images: survivor-free backgrounds teach the model that
        # NAIP terrain does not automatically imply a person is present.
        is_negative = rng.random() < neg_frac

        # Phase 1 — sample survivors: physically-sized, viewpoint-correct
        # sprites at non-overlapping positions. All alpha-affecting transforms
        # (erode, blur) are applied HERE so the emitted tight box is known
        # before acceptance; the per-frame size-ratio and overlap checks then
        # operate on the FINAL label boxes, not the physics targets. Colour
        # harmonisation (RGB-only, alpha-preserving) happens at paste time.
        placements: list[dict] = []
        sizes_sqrt: list[float] = []
        if not is_negative:
            for _ in range(int(rng.integers(1, 4))):   # 1-3 survivors
                for _attempt in range(8):
                    asset = assets[int(rng.integers(0, len(assets)))]
                    if altitude_aware:
                        w, h, pose, heading = survivor_box_px(
                            rng, img_gsd, is_oblique=is_oblique, tilt_deg=img_tilt_deg
                        )
                        # Viewpoint-correct sprite:
                        #  - prone: full body rotated to its ground heading
                        #    (plausible from above)
                        #  - standing at nadir: top-down head+shoulders blob,
                        #    never an upright ground-level photo
                        #  - oblique: upright photo is legitimately visible
                        if pose == "prone":
                            # The upright cutout's long axis is vertical (90°);
                            # rotate so it lies along the sampled ground heading
                            # (measured from the x-axis, like the box math).
                            sprite = asset.rotate(
                                heading - 90.0, expand=True,
                                resample=Image.Resampling.BICUBIC,
                            )
                        elif not is_oblique:
                            sprite = _top_down_standing_sprite(asset, rng)
                        else:
                            sprite = asset
                    elif rng.random() < small_frac:
                        pose, heading = "photo", 0.0
                        w = int(rng.integers(max(lo, 20), min(50, hi)))
                        h = int(w * asset.height / asset.width)
                        sprite = asset
                    else:
                        pose, heading = "photo", 0.0
                        w = int(rng.integers(lo, hi))
                        h = int(w * asset.height / asset.width)
                        sprite = asset
                    # Aspect-preserving scale: sprite alpha extent fits the
                    # physics box; the sprite is never stretched to fill it.
                    sprite = _scale_sprite_into_box(sprite, max(2, int(w)), max(2, int(h)))
                    sw, sh = sprite.size
                    # Alpha-final transforms (colour harmonisation later does
                    # not touch alpha, so the tight box computed now is exact).
                    if min(sw, sh) >= MIN_ERODE_SPRITE_PX:
                        sprite = _erode_alpha_edge(sprite, rng)
                    sprite = _resolution_blur(sprite, sw, rng, gsd_m=img_gsd)
                    if _tight_alpha_bbox(sprite) is None:
                        continue
                    if rng.random() < boundary_frac:
                        x = int(rng.integers(-sw // 2, max(1, size - sw // 2)))
                        y = int(rng.integers(-sh // 2, max(1, size - sh // 2)))
                    else:
                        x = int(rng.integers(0, max(1, size - sw)))
                        y = int(rng.integers(0, max(1, size - sh)))
                    # Emitted label = tight bbox of the VISIBLE (in-frame) part
                    # of the final alpha mask. For edge-truncated sprites this
                    # is tighter than clipping the full-mask bbox (a protruding
                    # limb outside the frame must not widen the visible box).
                    vx1, vy1 = max(0, -x), max(0, -y)
                    vx2, vy2 = min(sw, size - x), min(sh, size - y)
                    if vx2 - vx1 < 4 or vy2 - vy1 < 4:
                        continue
                    alpha = np.asarray(sprite.getchannel("A"))
                    vis = alpha[vy1:vy2, vx1:vx2] > ALPHA_BBOX_THRESHOLD
                    if not vis.any():
                        continue
                    rows = np.where(vis.any(axis=1))[0]
                    cols = np.where(vis.any(axis=0))[0]
                    bx1 = x + vx1 + int(cols[0])
                    bx2 = x + vx1 + int(cols[-1]) + 1
                    by1 = y + vy1 + int(rows[0])
                    by2 = y + vy1 + int(rows[-1]) + 1
                    if bx2 - bx1 < 4 or by2 - by1 < 4:
                        continue
                    # Per-frame size coherence on the FINAL emitted boxes.
                    s_size = float(np.sqrt((bx2 - bx1) * (by2 - by1)))
                    if sizes_sqrt:
                        s_max = max(sizes_sqrt + [s_size])
                        s_min = min(sizes_sqrt + [s_size])
                        if s_max / s_min > GEN_SIZE_RATIO_MAX:
                            continue
                    emitted = (bx1, by1, bx2 - bx1, by2 - by1)
                    # Labels must not overlap: reject and resample the position.
                    if any(_boxes_intersect(emitted, p["emitted"]) for p in placements):
                        continue
                    placements.append({
                        "sprite": sprite, "x": x, "y": y,
                        "pose": pose, "heading": heading,
                        "emitted": emitted,
                        # The label IS the visible-mask tight box, so its IoU
                        # against the mask bbox is 1.0 by construction; it is
                        # persisted so the validator can audit tightness after
                        # the alpha channel is flattened into a JPG.
                        "mask_iou": 1.0,
                    })
                    sizes_sqrt.append(s_size)
                    break

        centers = [
            (p["emitted"][0] + p["emitted"][2] / 2, p["emitted"][1] + p["emitted"][3] / 2)
            for p in placements
        ]
        has_fire = rng.random() < fire_frac
        heavy_occlude = has_fire and rng.random() < heavy_occlude_frac
        mask = _wildfire_masks(size, rng, centers, heavy_occlude=heavy_occlude) if has_fire else None
        if mask is not None:    # burn + flame UNDER survivors (production order)
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg, include_burn=True, include_flame=True, include_smoke=False)

        # Phase 2 — composite and emit TIGHT labels: the YOLO box is the tight
        # bbox of the sprite's final alpha mask (computed in phase 1; colour
        # harmonisation below is RGB-only and does not alter the alpha), clipped
        # to the frame — never the padded paste rectangle.
        labels: list[str] = []
        box_records: list[dict] = []
        for p in placements:
            sprite, x, y = p["sprite"], p["x"], p["y"]
            sw, sh = sprite.size
            # Sample only the visible patch of the background for color harmonization.
            px1 = max(0, x); py1 = max(0, y)
            px2 = min(size, x + sw); py2 = min(size, y + sh)
            if px2 <= px1 or py2 <= py1:
                continue
            bg_patch = bg.crop((px1, py1, px2, py2))
            s = _harmonize_color(sprite, bg_patch, rng)  # tone match to local BG
            bg.paste(s, (x, y), s)
            bx1, by1, bw, bh = p["emitted"]
            cx_n = (bx1 + bw / 2.0) / size
            cy_n = (by1 + bh / 2.0) / size
            labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {bw / size:.6f} {bh / size:.6f}")
            box_records.append({
                "pose": p["pose"],
                "heading_deg": round(p["heading"], 1),
                "x_px": bx1, "y_px": by1, "w_px": bw, "h_px": bh,
                "w_m": round(bw * img_gsd, 2) if img_gsd else None,
                "h_m": round(bh * img_gsd, 2) if img_gsd else None,
                "mask_iou": round(p["mask_iou"], 3),
            })

        if mask is not None:    # smoke drifts OVER survivors
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg, include_burn=False, include_flame=False, include_smoke=True)

        # Hard-negative decoys: non-human objects composited the same way as
        # survivors but never labeled.  Applied regardless of whether the image is
        # positive, negative, or mixed — the model must learn to ignore them.
        # Prefer REAL assets (VisDrone vehicles, SARD rejected) when available;
        # fall back to procedural synthetic blobs otherwise.
        decoy_records: list[dict] = []
        if rng.random() < decoy_frac:
            survivor_rects = [p["emitted"] for p in placements]
            n_decoys = int(rng.integers(1, 4))
            for _ in range(n_decoys):
                decoy_type = "vehicle"
                if altitude_aware and img_gsd:
                    if rng.random() < DECOY_HUMAN_SCALE_FRAC:
                        # Human-scale confuser: animal or colourful object in the
                        # SAME size band as survivors (0.5-1.8 m) so size alone
                        # cannot separate people from decoys.
                        obj_long_m = float(rng.uniform(*DECOY_HUMAN_SCALE_LONG_M))
                        long_px = max(DECOY_MIN_PX, obj_long_m / img_gsd)
                        aspect = float(rng.uniform(0.35, 0.9))  # elongated, like a prone body
                        if rng.random() < 0.5:
                            dw = int(round(long_px)); dh = max(DECOY_MIN_PX, int(round(long_px * aspect)))
                        else:
                            dh = int(round(long_px)); dw = max(DECOY_MIN_PX, int(round(long_px * aspect)))
                        if rng.random() < 0.5:
                            decoy = _animal_decoy_rgba(dw, dh, rng)
                            decoy_type = "animal"
                        else:
                            decoy = _colorful_decoy_rgba(dw, dh, rng)
                            decoy_type = "colorful_object"
                    else:
                        # Physically-sized vehicle: sample a real-world long axis
                        # and scale through the camera GSD, preserving the asset
                        # aspect. Mix biased to passenger cars (~3.5-5.2 m).
                        u = rng.random()
                        if u < DECOY_LARGE_FRAC:
                            veh_long_m = float(rng.uniform(*DECOY_LARGE_VEHICLE_M))
                        elif u < DECOY_LARGE_FRAC + DECOY_VAN_FRAC:
                            veh_long_m = float(rng.uniform(*DECOY_VAN_LONG_M))
                        else:
                            veh_long_m = float(rng.uniform(*DECOY_VEHICLE_LONG_M))
                        long_px = max(DECOY_MIN_PX, veh_long_m / img_gsd)
                        if decoy_assets:
                            raw = decoy_assets[int(rng.integers(0, len(decoy_assets)))]
                            aspect = raw.height / raw.width
                            if aspect >= 1.0:
                                dh = int(round(long_px)); dw = max(DECOY_MIN_PX, int(round(long_px / aspect)))
                            else:
                                dw = int(round(long_px)); dh = max(DECOY_MIN_PX, int(round(long_px * aspect)))
                            decoy = raw.resize((dw, dh), Image.Resampling.LANCZOS)
                        else:
                            dw = int(round(long_px)); dh = max(DECOY_MIN_PX, int(dw * rng.uniform(0.5, 1.8)))
                            decoy = _synthetic_decoy_rgba(dw, dh, rng)
                    # Never let an UNLABELED decoy overlap a LABELED survivor box:
                    # an unlabeled human-sized blob on top of a positive corrupts
                    # both the positive and the hard-negative signal.
                    dx = dy = None
                    for _attempt in range(8):
                        tx = int(rng.integers(0, max(1, size - dw)))
                        ty = int(rng.integers(0, max(1, size - dh)))
                        clash = any(
                            tx < bx + bw_ and tx + dw > bx and ty < by + bh_ and ty + dh > by
                            for (bx, by, bw_, bh_) in survivor_rects
                        )
                        if not clash:
                            dx, dy = tx, ty
                            break
                    if dx is None:
                        continue  # frame too crowded — skip this decoy
                elif decoy_assets:
                    # Legacy (uniform-size) mode: random width, preserve aspect.
                    dw = int(rng.integers(lo, hi))
                    raw = decoy_assets[int(rng.integers(0, len(decoy_assets)))]
                    dh = max(8, int(dw * raw.height / raw.width))
                    dx = int(rng.integers(0, max(1, size - dw)))
                    dy = int(rng.integers(0, max(1, size - dh)))
                    decoy = raw.resize((dw, dh), Image.Resampling.LANCZOS)
                else:
                    # Fallback: procedural blob (non-human aspect ratios).
                    dw = int(rng.integers(lo, hi))
                    dh = int(dw * rng.uniform(0.5, 1.8))
                    dx = int(rng.integers(0, max(1, size - dw)))
                    dy = int(rng.integers(0, max(1, size - dh)))
                    decoy = _synthetic_decoy_rgba(dw, dh, rng)
                    decoy_type = "synthetic_blob"
                bg_patch = bg.crop((dx, dy, dx + dw, dy + dh))
                decoy = _harmonize_color(decoy, bg_patch, rng)
                # Same erosion guard as survivors: skipping erosion on small
                # sprites must apply to BOTH classes, or the edge treatment
                # itself becomes a human/non-human shortcut cue.
                if min(dw, dh) >= MIN_ERODE_SPRITE_PX:
                    decoy = _erode_alpha_edge(decoy, rng)
                decoy = _resolution_blur(decoy, dw, rng, gsd_m=img_gsd)
                bg.paste(decoy, (dx, dy), decoy)
                # Intentionally no YOLO label — this is the hard-negative signal.
                # Recorded in metadata so decoy composition is auditable.
                decoy_records.append({
                    "type": decoy_type,
                    "x_px": dx, "y_px": dy, "w_px": dw, "h_px": dh,
                    "long_m": round(max(dw, dh) * img_gsd, 2) if img_gsd else None,
                })

        bg.save(img_dir / f"{i:05d}.jpg", quality=92)
        # Empty label file is the YOLO convention for a negative image.
        label_text = "\n".join(labels) + ("\n" if labels else "")
        (lbl_dir / f"{i:05d}.txt").write_text(label_text, encoding="utf-8")
        # Metadata sidecar — written for EVERY frame so each label's physical
        # size is verifiable (real_size_m = px * gsd_m). Legacy (non-altitude-
        # aware) frames carry gsd_m: null and are flagged by the validator.
        meta = {
            "altitude_m": round(img_altitude_m, 1) if img_altitude_m is not None else None,
            "gsd_m": round(img_gsd, 4) if img_gsd is not None else None,
            "view": ("oblique" if is_oblique else "nadir") if altitude_aware else "photo",
            "n_survivors": len(labels),
            "poses": [b["pose"] for b in box_records],
            "boxes": box_records,
            "has_fire": has_fire,
            "oblique": is_oblique,
            "tilt_deg": round(img_tilt_deg, 1),
            "n_decoys": len(decoy_records),
            "decoys": decoy_records,
        }
        (lbl_dir / f"{i:05d}.json").write_text(json.dumps(meta), encoding="utf-8")
        if not labels:
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
    ap.add_argument("--device", default="cpu",
                    help="Training device: 'cpu' (default) or 'mps' (Apple GPU, much faster on M-series).")
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
    ap.add_argument("--altitude-aware", action=argparse.BooleanOptionalAction, default=True,
                    help="Sample drone altitude from operational envelope (20-50m) and derive "
                         "survivor pixel size + blur from camera physics. Produces a realistic "
                         "altitude-dependent size distribution. Disable with --no-altitude-aware "
                         "to use the legacy uniform random size range.")
    ap.add_argument("--oblique-frac", type=float, default=0.25,
                    help="Fraction of images using an oblique (side-angle) drone camera "
                         "(15-45° tilt from nadir). Survivors appear more elongated/upright. "
                         "Default 0.25 (25%% of images). Set 0 for pure nadir only.")
    ap.add_argument("--data-dir", default=str(ROOT / "data/cv_train/survivor"))
    ap.add_argument("--out", default=str(ROOT / "models/survivor_yolov8n.pt"))
    ap.add_argument("--skip-generation", action="store_true",
                    help="Skip dataset generation and train on the existing --data-dir "
                         "(use after a prior --epochs 0 generate-only run).")
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
    alt_info = (
        f"altitude-aware ({DRONE_FLIGHT_LEVELS_M[0]:.0f}–{DRONE_FLIGHT_LEVELS_M[-1]:.0f}m, "
        f"{DRONE_CAMERA_FOV_DEG:.0f}° FOV)"
        if args.altitude_aware else "uniform random size"
    )
    if args.skip_generation:
        print(f"--skip-generation: training on existing dataset in {data_dir}")
    else:
        print(f"Generating {args.n_train} train / {args.n_val} val composites at {args.size}px "
              f"({int(args.fire_frac*100)}% with fire/smoke, {int(args.neg_frac*100)}% negatives, "
              f"{int(args.decoy_frac*100)}% with hard-negative decoys, {alt_info}) ...")
        _generate_split(
            data_dir / "train", args.n_train, train_assets, args.size, rng, cfg,
            fire_frac=args.fire_frac, surv_px=surv_px,
            naip_tiles=naip_train_tiles, neg_frac=args.neg_frac, decoy_frac=args.decoy_frac,
            decoy_assets=decoy_assets, boundary_frac=args.boundary_frac,
            small_frac=args.small_frac, heavy_occlude_frac=args.heavy_occlude_frac,
            altitude_aware=args.altitude_aware, oblique_frac=args.oblique_frac,
        )
        _generate_split(
            data_dir / "val", args.n_val, val_assets, args.size, rng, cfg,
            fire_frac=args.fire_frac, surv_px=surv_px,
            naip_tiles=naip_val_tiles, neg_frac=args.neg_frac, decoy_frac=args.decoy_frac,
            decoy_assets=decoy_assets, boundary_frac=args.boundary_frac,
            small_frac=args.small_frac, heavy_occlude_frac=args.heavy_occlude_frac,
            altitude_aware=args.altitude_aware, oblique_frac=args.oblique_frac,
        )

    yaml = data_dir / "survivor.yaml"
    yaml.write_text(
        f"path: {data_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n",
        encoding="utf-8",
    )

    if args.epochs <= 0:
        print("epochs<=0: generated dataset only, skipping training.")
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"Fine-tuning {args.model} for {args.epochs} epochs ...")
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=16,
        device=args.device, verbose=True, seed=args.seed, project=str(data_dir.resolve() / "runs"),
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
