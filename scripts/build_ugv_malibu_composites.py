#!/usr/bin/env python3
"""Composite REAL person cutouts onto REAL Malibu-area terrain photos (UGV eval).

Purpose: the UGV real-data evaluation previously used COCO (indoor kitchens,
offices, city streets) as the "real image" stand-in, which has essentially no
visual relationship to a ground robot operating in coastal-California
wildland (dry chaparral, fire roads, oak-lined canyons). This script instead
composites real SARD person cutouts onto real ground-level photographs of
Malibu / Santa Monica Mountains terrain (Malibu Canyon, Malibu Creek State
Park, Topanga State Park, Point Mugu State Park — sourced from Wikimedia
Commons, CC-licensed), sized with the SAME front-camera pinhole physics used
to generate the synthetic UGV training data
(:mod:`scripts.train_ugv_detector`), so the eval measures "does the person
look right against REAL Malibu vegetation/lighting" rather than "does the
detector generalize to a kitchen".

This is honest about what it is: the *terrain* and the *person* are both
real photographs, but the compositing (choice of scale/placement) is
synthetic, same caveat as scripts/build_flame_composites.py. Unlike FLAME
(DJI XMP gives exact altitude/gimbal pitch), these Commons photos have no
pose metadata, so range is sampled from the UGV operating envelope
(5-30 m) and placed with the same bottom-aligned perspective heuristic
used in training, rather than derived from per-pixel geometry.

Usage:
    python scripts/build_ugv_malibu_composites.py \
        --backgrounds data/cv_assets/malibu_terrain_photos \
        --assets-dir data/cv_assets/sard_grabcut \
        --out data/cv_train/ugv_malibu_real \
        --variants-per-bg 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from train_survivor_detector import _crop_to_alpha, _tight_alpha_bbox  # noqa: E402
from train_ugv_detector import (  # noqa: E402
    FRONT_CAMERA_HFOV_DEG,
    FRONT_CAMERA_HEIGHT_M,
    FRONT_RANGE_MIN_M,
    FRONT_RANGE_MAX_M,
    PERSON_HEIGHT_M,
    front_m_per_px,
    front_range_to_person_px,
    sample_front_range,
    _scale_sprite_long_axis,
    _erode_alpha_edge,
    _harmonize_color,
    _range_blur,
    _add_vegetation_occlusion,
)

Image.MAX_IMAGE_PIXELS = None
SIZE = 640


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


def load_backgrounds(bg_dir: Path) -> list[tuple[str, Image.Image]]:
    out = []
    for p in sorted(bg_dir.glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        out.append((p.name, img))
    return out


def _square_crop(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Random square crop (biased toward the lower 2/3 = ground plane) then resize to SIZE."""
    W, H = img.size
    side = min(W, H)
    max_x = W - side
    # Bias the crop's vertical origin toward the lower part of the frame so we
    # keep ground/vegetation rather than mostly sky.
    max_y = H - side
    y0 = int(max_y * rng.uniform(0.35, 1.0)) if max_y > 0 else 0
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    crop = img.crop((x0, y0, x0 + side, y0 + side))
    return crop.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def composite_one(
    bg_photo: Image.Image,
    source_name: str,
    assets: list[Image.Image],
    rng: np.random.Generator,
    is_negative: bool,
) -> tuple[Image.Image, list[str], list[dict]]:
    bg = _square_crop(bg_photo, rng).copy()
    labels: list[str] = []
    box_records: list[dict] = []
    placements: list[tuple[int, int, int, int]] = []

    if is_negative:
        return bg, labels, box_records

    n_persons = int(rng.integers(1, 3))
    for _ in range(n_persons):
        sprite = None
        range_m = FRONT_RANGE_MIN_M
        for _attempt in range(6):
            range_m = sample_front_range(rng)
            px_h, _px_w = front_range_to_person_px(range_m, image_width=SIZE)
            body_jitter = float(rng.uniform(0.85, 1.15))
            px_h = int(max(15, min(SIZE - 10, px_h * body_jitter)))

            asset = assets[int(rng.integers(0, len(assets)))]
            cand = _scale_sprite_long_axis(asset, px_h)
            if min(cand.size) >= 24:
                cand = _erode_alpha_edge(cand, rng)
            cand = _range_blur(cand, range_m, rng)
            bb = _tight_alpha_bbox(cand)
            if bb is None:
                continue
            long_final = max(bb[2] - bb[0], bb[3] - bb[1])
            implied = long_final * front_m_per_px(range_m, image_width=SIZE)
            if not (1.35 <= implied <= 2.15):
                continue
            sprite = cand
            break
        if sprite is None:
            continue
        sw, sh = sprite.size

        range_frac = (range_m - FRONT_RANGE_MIN_M) / (FRONT_RANGE_MAX_M - FRONT_RANGE_MIN_M)
        base_y = int(SIZE * (0.95 - range_frac * 0.5) - sh)
        y = max(0, min(SIZE - sh, base_y + int(rng.integers(-20, 20))))
        x = int(rng.integers(0, max(1, SIZE - sw)))

        overlap = any(
            not (x + sw <= px or px + pw <= x or y + sh <= py or py + ph <= y)
            for px, py, pw, ph in placements
        )
        if overlap:
            continue

        bg_patch = bg.crop((max(0, x), max(0, y), min(SIZE, x + sw), min(SIZE, y + sh)))
        sprite = _harmonize_color(sprite, bg_patch, rng)
        bg.paste(sprite, (x, y), sprite)

        fx1, fy1, fx2, fy2 = x + bb[0], y + bb[1], x + bb[2], y + bb[3]
        bx1, by1 = max(0, fx1), max(0, fy1)
        bx2, by2 = min(SIZE, fx2), min(SIZE, fy2)
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        placements.append((bx1, by1, bx2 - bx1, by2 - by1))
        cx_n = (bx1 + bx2) / 2.0 / SIZE
        cy_n = (by1 + by2) / 2.0 / SIZE
        labels.append(
            f"0 {cx_n:.6f} {cy_n:.6f} {(bx2 - bx1) / SIZE:.6f} {(by2 - by1) / SIZE:.6f}"
        )
        m_px = front_m_per_px(range_m, image_width=SIZE)
        inter = (bx2 - bx1) * (by2 - by1)
        union = inter + (fx2 - fx1) * (fy2 - fy1) - inter
        box_records.append({
            "range_m": round(range_m, 1),
            "m_per_px": round(m_px, 4),
            "x_px": bx1, "y_px": by1,
            "w_px": bx2 - bx1, "h_px": by2 - by1,
            "implied_long_m": round(max(bx2 - bx1, by2 - by1) * m_px, 2),
            "mask_iou": round(inter / union if union > 0 else 0.0, 3),
        })

    for bbox in placements:
        bg = _add_vegetation_occlusion(bg, bbox, rng, occlusion_prob=0.45)

    return bg, labels, box_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds", default=str(ROOT / "data/cv_assets/malibu_terrain_photos"))
    ap.add_argument("--assets-dir", default=str(ROOT / "data/cv_assets/sard_grabcut"))
    ap.add_argument("--out", default=str(ROOT / "data/cv_train/ugv_malibu_real"))
    ap.add_argument("--variants-per-bg", type=int, default=4)
    ap.add_argument("--negative-variants-per-bg", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    assets = load_person_assets(Path(args.assets_dir))
    backgrounds = load_backgrounds(Path(args.backgrounds))
    print(f"Loaded {len(assets)} person cutouts, {len(backgrounds)} real Malibu-area backgrounds")

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    n_neg = 0
    for name, photo in backgrounds:
        for v in range(args.variants_per_bg):
            img, labels, boxes = composite_one(photo, name, assets, rng, is_negative=False)
            stem = f"{idx:05d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )
            meta = {
                "camera": "front",
                "camera_height_m": FRONT_CAMERA_HEIGHT_M,
                "hfov_deg": FRONT_CAMERA_HFOV_DEG,
                "person_height_m": PERSON_HEIGHT_M,
                "source_background": name,
                "source": "Wikimedia Commons (real Malibu / Santa Monica Mountains photo)",
                "n_persons": len(labels),
                "boxes": boxes,
                "is_negative": False,
            }
            (lbl_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
            if not labels:
                n_neg += 1
            idx += 1
        for v in range(args.negative_variants_per_bg):
            img, labels, boxes = composite_one(photo, name, assets, rng, is_negative=True)
            stem = f"{idx:05d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            meta = {
                "camera": "front",
                "source_background": name,
                "source": "Wikimedia Commons (real Malibu / Santa Monica Mountains photo)",
                "n_persons": 0,
                "boxes": [],
                "is_negative": True,
            }
            (lbl_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
            n_neg += 1
            idx += 1

    print(f"Wrote {idx} images ({n_neg} negative/no-person) to {out_dir}")


if __name__ == "__main__":
    main()
