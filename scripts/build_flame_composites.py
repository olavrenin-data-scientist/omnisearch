#!/usr/bin/env python3
"""Composite REAL person cutouts onto REAL wildfire drone imagery (FLAME 3).

Purpose: the synthetic training/eval pipeline applies *synthetic* smoke to
NAIP tiles, so "recall under smoke" has only been measured against our own
smoke model. FLAME 3 (Sycan Marsh prescribed burn, DJI M30T, ~55-100 m AGL)
provides real fire/smoke aerial frames; pasting real SARD person cutouts onto
them gives ground truth for "RGB detection through REAL smoke".

Physics: FLAME frames are oblique (gimbal pitch -7..-26 deg below horizon).
For each paste row y we compute the depression angle, slant range and local
angular GSD, then size the person from the same physical body model used by
scripts/train_survivor_detector.py. People are only placed where the ground
is closer than --max-range and the depression angle is steep enough to see
ground (not sky/horizon).

Smoke occlusion: pasting a sprite ON TOP of smoke would be unphysical (the
smoke sits between camera and ground). We estimate per-patch smoke opacity
(bright + desaturated pixels) and attenuate the sprite's alpha accordingly,
recording the opacity in the per-image JSON so the eval can report recall as
a function of REAL smoke density.

Usage:
    python scripts/build_flame_composites.py \
        --flame-dir "~/.cache/kagglehub/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh/versions/1/FLAME 3 CV Dataset (Sycan Marsh)" \
        --out data/cv_train/flame_composites
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from train_survivor_detector import (  # noqa: E402
    PERSON_HEIGHT_M,
    PERSON_SHOULDER_M,
    PERSON_DEPTH_M,
    PRONE_FRAC,
    SURVIVOR_MIN_PX,
    _crop_to_alpha,
    _erode_alpha_edge,
    _harmonize_color,
    _scale_sprite_into_box,
    _tight_alpha_bbox,
    MIN_ERODE_SPRITE_PX,
)

Image.MAX_IMAGE_PIXELS = None

# DJI M30T wide camera: 4.4 mm focal length on a 1/2" CMOS (6.4 x 4.8 mm),
# 4000 x 3000 px. HFOV = 2*atan(3.2/4.4) = 72.0 deg, VFOV = 57.2 deg.
SENSOR_W_MM = 6.4
SENSOR_H_MM = 4.8
FOCAL_MM = 4.4
HFOV_RAD = 2.0 * np.arctan((SENSOR_W_MM / 2.0) / FOCAL_MM)
VFOV_RAD = 2.0 * np.arctan((SENSOR_H_MM / 2.0) / FOCAL_MM)

MIN_DEPRESSION_DEG = 18.0   # only paste where the ground is clearly visible
MAX_SLANT_RANGE_M = 220.0   # beyond this the person is < ~4 px — pointless
MAX_SMOKE_ALPHA_ATTEN = 0.88  # never erase the person entirely

# Target smoke-density bands for stratified placement, so the eval measures
# recall vs REAL smoke opacity instead of mostly-clear foreground pastes:
# (name, opacity_lo, opacity_hi, sampling weight)
SMOKE_BANDS = [
    ("clear", 0.00, 0.12, 0.40),
    ("light", 0.12, 0.35, 0.35),
    ("heavy", 0.35, 1.01, 0.25),
]


def parse_dji_xmp(path: Path) -> tuple[float | None, float | None]:
    """Extract (relative_altitude_m, gimbal_pitch_deg) from DJI XMP header."""
    data = path.open("rb").read(65536)
    alt = re.search(rb'RelativeAltitude="\+?([-0-9.]+)"', data)
    pitch = re.search(rb'GimbalPitchDegree="\+?([-0-9.]+)"', data)
    return (
        float(alt.group(1)) if alt else None,
        float(pitch.group(1)) if pitch else None,
    )


def depression_at_row(y_px: float, img_h: int, gimbal_pitch_deg: float) -> float:
    """Depression angle (deg below horizon) of the ray through image row y.

    gimbal_pitch_deg is DJI convention: negative = camera axis below horizon
    (-90 = nadir). Rows below image centre look further down.
    """
    axis_depression = -gimbal_pitch_deg
    frac = (y_px - img_h / 2.0) / img_h  # -0.5 (top) .. +0.5 (bottom)
    return axis_depression + np.degrees(VFOV_RAD) * frac


def geometry_at_row(
    y_px: float, img_w: int, img_h: int, altitude_m: float, gimbal_pitch_deg: float
) -> tuple[float, float, float] | None:
    """Return (depression_deg, slant_range_m, gsd_m_per_px) or None if sky/too far."""
    dep = depression_at_row(y_px, img_h, gimbal_pitch_deg)
    if dep < MIN_DEPRESSION_DEG:
        return None
    slant = altitude_m / np.sin(np.radians(dep))
    if slant > MAX_SLANT_RANGE_M:
        return None
    gsd = slant * (HFOV_RAD / img_w)  # angular pixel size * range
    return dep, slant, gsd


def smoke_opacity(patch: Image.Image) -> float:
    """Estimate smoke opacity in [0,1] for a background patch.

    Smoke reads as bright, desaturated AND featureless; dry grass is bright
    but saturated and textured, trees/burned ground are dark. The score
    combines a brightness gate with desaturation and low-texture terms, so
    both white plumes and the bluish haze FLAME frames show read as smoke.
    """
    hsv = np.asarray(patch.convert("HSV"), dtype=np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    v_gate = np.clip((v - 0.40) / 0.60, 0.0, 1.0)
    desat = np.clip((0.45 - s) / 0.45, 0.0, 1.0)
    gray = np.asarray(patch.convert("L"), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    smooth = float(np.clip((0.035 - (gx + gy) / 2.0) / 0.035, 0.0, 1.0))
    score = (v_gate * (0.65 * desat + 0.45 * smooth)).mean()
    return float(np.clip(score, 0.0, 1.0))


def patch_is_placeable(patch: Image.Image) -> bool:
    """Reject deep-shadow / dark canopy patches where a ground target is implausible."""
    v = np.asarray(patch.convert("L"), dtype=np.float32) / 255.0
    return float(v.mean()) >= 0.18


def person_box_px(
    rng: np.random.Generator, dep_deg: float, gsd: float
) -> tuple[int, int, str]:
    """Pixel (w, h, pose) of a person seen from depression angle dep_deg.

    Sizes are perpendicular-to-line-of-sight extents divided by the angular
    GSD, which is the correct projection for a compact object at range.
    """
    dep = np.radians(dep_deg)
    prone = bool(rng.random() < PRONE_FRAC)
    if prone:
        heading = rng.uniform(0.0, np.pi)
        c, s = abs(np.cos(heading)), abs(np.sin(heading))
        w_m = PERSON_HEIGHT_M * c + PERSON_SHOULDER_M * s
        # The along-view axis of a ground-lying body is foreshortened by sin(dep).
        h_m = (PERSON_HEIGHT_M * s + PERSON_SHOULDER_M * c) * np.sin(dep) \
            + 0.30 * np.cos(dep)  # body thickness seen side-on
        pose = "prone"
    else:
        w_m = PERSON_SHOULDER_M
        # Standing: full height visible at shallow angles, head+shoulders at nadir.
        h_m = PERSON_HEIGHT_M * np.cos(dep) + PERSON_DEPTH_M * np.sin(dep)
        pose = "standing"
    w_m *= rng.uniform(0.85, 1.15)
    h_m *= rng.uniform(0.85, 1.15)
    w_px = max(SURVIVOR_MIN_PX, int(round(w_m / gsd)))
    h_px = max(SURVIVOR_MIN_PX, int(round(h_m / gsd)))
    return w_px, h_px, pose


def range_blur(sprite: Image.Image, slant_m: float, rng: np.random.Generator) -> Image.Image:
    """Soften close-range SARD photos to the effective resolution at range."""
    from PIL import ImageFilter

    r = float(np.clip(slant_m / 120.0, 0.3, 2.0)) * rng.uniform(0.8, 1.2)
    rgb = sprite.convert("RGB").filter(ImageFilter.GaussianBlur(radius=r))
    return Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))


def load_person_assets(assets_dir: Path) -> list[Image.Image]:
    out = []
    for p in sorted(assets_dir.glob("*.png")):
        try:
            img = Image.open(p).convert("RGBA")
        except Exception:
            continue
        if min(img.size) >= 30:
            out.append(img)
    return out


def composite_frame(
    frame: Image.Image,
    assets: list[Image.Image],
    altitude_m: float,
    gimbal_pitch_deg: float,
    rng: np.random.Generator,
    n_people: int,
) -> tuple[Image.Image, list[dict]]:
    """Paste n_people onto the frame; returns (image, per-person records)."""
    W, H = frame.size
    out = frame.convert("RGB")
    records: list[dict] = []
    placed: list[tuple[int, int, int, int]] = []

    band_w = np.array([b[3] for b in SMOKE_BANDS])
    band_w = band_w / band_w.sum()

    for _person in range(n_people):
        band_name, op_lo, op_hi, _ = SMOKE_BANDS[int(rng.choice(len(SMOKE_BANDS), p=band_w))]
        placed_ok = False
        best = None  # fallback: any valid placement if the band can't be hit
        for _try in range(60):
            y = rng.uniform(H * 0.30, H * 0.98)
            geo = geometry_at_row(y, W, H, altitude_m, gimbal_pitch_deg)
            if geo is None:
                continue
            dep, slant, gsd = geo
            w_px, h_px, pose = person_box_px(rng, dep, gsd)
            if w_px >= W // 4 or h_px >= H // 4:
                continue
            x = int(rng.uniform(0, W - w_px - 1))
            y0 = int(min(y, H - h_px - 1))
            if any(
                not (x + w_px <= bx or bx + bw <= x or y0 + h_px <= by or by + bh <= y0)
                for bx, by, bw, bh in placed
            ):
                continue
            patch0 = out.crop((x, y0, min(W, x + w_px), min(H, y0 + h_px)))
            if not patch_is_placeable(patch0):
                continue
            op0 = smoke_opacity(patch0)
            cand = (x, y0, w_px, h_px, pose, dep, slant, gsd)
            if best is None:
                best = cand
            if op_lo <= op0 < op_hi:
                best = cand
                placed_ok = True
                break
        if best is None:
            continue
        x, y0, w_px, h_px, pose, dep, slant, gsd = best

        asset = assets[int(rng.integers(len(assets)))]
        sprite = _crop_to_alpha(asset)
        if pose == "prone":
            sprite = sprite.rotate(90 if rng.random() < 0.5 else -90, expand=True)
        sprite = _scale_sprite_into_box(sprite, w_px, h_px)
        sw, sh = sprite.size

        patch = out.crop((x, y0, min(W, x + sw), min(H, y0 + sh)))
        opacity = smoke_opacity(patch)

        sprite = _harmonize_color(sprite, patch, rng)
        if min(sprite.size) >= MIN_ERODE_SPRITE_PX:
            sprite = _erode_alpha_edge(sprite, rng)
        sprite = range_blur(sprite, slant, rng)

        # Real smoke sits between camera and ground: attenuate the person by
        # the estimated smoke opacity instead of pasting over the plume.
        atten = min(MAX_SMOKE_ALPHA_ATTEN, opacity)
        if atten > 0.01:
            a = np.asarray(sprite.getchannel("A"), dtype=np.float32)
            a *= 1.0 - atten
            sprite = Image.merge(
                "RGBA", (*sprite.convert("RGB").split(),
                         Image.fromarray(a.astype(np.uint8)))
            )

        out.paste(sprite, (x, y0), sprite)
        bb = _tight_alpha_bbox(sprite, threshold=24)
        if bb is None:
            continue
        bx1, by1, bx2, by2 = bb
        lx, ly = x + bx1, y0 + by1
        lw, lh = bx2 - bx1, by2 - by1
        placed.append((lx, ly, lw, lh))
        records.append({
            "x": lx, "y": ly, "w": lw, "h": lh,
            "pose": pose,
            "depression_deg": round(dep, 2),
            "slant_range_m": round(slant, 1),
            "gsd_m_per_px": round(gsd, 4),
            "smoke_opacity": round(opacity, 3),
            "target_band": band_name,
            "band_hit": placed_ok,
        })
    return out, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame-dir", required=True)
    ap.add_argument("--assets-dir", default=str(ROOT / "data/cv_assets/sard_grabcut"))
    ap.add_argument("--out", default=str(ROOT / "data/cv_train/flame_composites"))
    ap.add_argument("--n-fire", type=int, default=150, help="Fire frames WITH people")
    ap.add_argument("--n-nofire", type=int, default=40, help="No-fire frames WITH people (clear control)")
    ap.add_argument("--n-negative", type=int, default=40, help="Fire frames WITHOUT people (FP control)")
    ap.add_argument("--people-min", type=int, default=1)
    ap.add_argument("--people-max", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    flame = Path(args.flame_dir).expanduser()
    fire_dir = flame / "Fire" / "RGB" / "Raw"
    nofire_dir = flame / "No Fire" / "RGB" / "Raw"
    assets = load_person_assets(Path(args.assets_dir))
    if not assets:
        raise SystemExit(f"No person assets in {args.assets_dir}")
    print(f"{len(assets)} person cutouts loaded")

    out_root = Path(args.out)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    meta_dir = out_root / "meta"
    for d in (img_dir, lbl_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    def usable(paths: list[Path]) -> list[tuple[Path, float, float]]:
        ok = []
        for p in paths:
            alt, pitch = parse_dji_xmp(p)
            if alt is None or pitch is None or alt < 20:
                continue
            # Need SOME ground within range: check the bottom row.
            if geometry_at_row(2999, 4000, 3000, alt, pitch) is None:
                continue
            ok.append((p, alt, pitch))
        return ok

    fire_all = usable(sorted(fire_dir.glob("*.JPG")))
    nofire_all = usable(sorted(nofire_dir.glob("*.JPG")))
    print(f"usable: {len(fire_all)} fire, {len(nofire_all)} no-fire frames")

    def pick(pool, n):
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]

    jobs = (
        [("fire", *t, True) for t in pick(fire_all, args.n_fire)]
        + [("nofire", *t, True) for t in pick(nofire_all, args.n_nofire)]
        + [("fireneg", *t, False) for t in pick(fire_all, args.n_negative)]
    )

    n_boxes = 0
    for i, (tag, path, alt, pitch, with_people) in enumerate(jobs):
        frame = Image.open(path).convert("RGB")
        if with_people:
            n_people = int(rng.integers(args.people_min, args.people_max + 1))
            img, recs = composite_frame(frame, assets, alt, pitch, rng, n_people)
        else:
            img, recs = frame, []
        stem = f"flame_{tag}_{path.stem}"
        img.save(img_dir / f"{stem}.jpg", quality=92)
        W, H = img.size
        lines = []
        for r in recs:
            cx = (r["x"] + r["w"] / 2) / W
            cy = (r["y"] + r["h"] / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {r['w'] / W:.6f} {r['h'] / H:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (meta_dir / f"{stem}.json").write_text(json.dumps({
            "source": str(path), "category": tag,
            "altitude_m": alt, "gimbal_pitch_deg": pitch,
            "people": recs,
        }, indent=1))
        n_boxes += len(recs)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(jobs)} frames, {n_boxes} people placed")

    print(f"Done: {len(jobs)} frames, {n_boxes} person boxes -> {out_root}")


if __name__ == "__main__":
    main()
