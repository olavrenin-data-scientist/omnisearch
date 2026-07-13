#!/usr/bin/env python3
"""Annotate FLAME 3 real-wildfire composites with box, pose, range and smoke info.

For every frame in data/cv_train/flame_composites/, draws each pasted person's
box color-coded by real-smoke opacity band, with a label showing pose, slant
range, GSD and smoke opacity, plus a header banner with altitude/gimbal pitch
and a legend. Person-free fire frames (FP controls) get a banner instead.

Usage:
    python scripts/annotate_flame_composites.py \
        --src data/cv_train/flame_composites \
        --dst ~/Documents/omnisearch_capstone/flame_composites_annotated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


BAND_COLORS = {
    "clear": (57, 255, 20),    # lime — < 0.12 smoke opacity
    "light": (255, 210, 60),   # amber — 0.12-0.35
    "heavy": (255, 60, 60),    # red — >= 0.35
}
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)
BANNER_BG = (15, 40, 20)
NEG_BANNER_BG = (60, 20, 20)


def band_of(opacity: float) -> str:
    if opacity < 0.12:
        return "clear"
    if opacity < 0.35:
        return "light"
    return "heavy"


def _text_with_bg(draw, xy, text, font, fg=TEXT_FG, bg=TEXT_BG, pad=3):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle([l - pad, t - pad, r + pad, b + pad], fill=bg)
    draw.text((x, y), text, fill=fg, font=font)


def annotate_image(img_path: Path, meta_path: Path, dst_path: Path, font, font_sm) -> None:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    meta = json.loads(meta_path.read_text())
    people = meta.get("people", [])
    draw = ImageDraw.Draw(img)

    for p in people:
        b = band_of(p["smoke_opacity"])
        color = BAND_COLORS[b]
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        pad = max(4, int(w * 0.6))
        draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], outline=color, width=3)
        label = (f"{p['pose']} | {p['slant_range_m']:.0f} m | "
                 f"gsd {p['gsd_m_per_px']:.3f} m/px | smoke {p['smoke_opacity']:.2f} ({b})")
        ly = y - pad - 22 if y - pad - 22 > 40 else y + h + pad + 4
        _text_with_bg(draw, (x - pad, ly), label, font_sm, bg=(0, 0, 0), fg=color)

    header = (f"{meta['category'].upper()}  |  alt {meta['altitude_m']:.1f} m  |  "
              f"gimbal {meta['gimbal_pitch_deg']:.1f}°  |  {len(people)} people placed")
    banner_bg = NEG_BANNER_BG if not people else BANNER_BG
    draw.rectangle([0, 0, W, 40], fill=banner_bg)
    _text_with_bg(draw, (10, 8), header, font, bg=banner_bg, fg=(255, 255, 255))
    if not people:
        _text_with_bg(draw, (10, H - 46), "FALSE-POSITIVE CONTROL: no person pasted in this frame",
                       font_sm, bg=(0, 0, 0), fg=(255, 120, 120))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, quality=90)


def write_legend(dst_root: Path, font, font_sm) -> None:
    W, H = 900, 360
    img = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    _text_with_bg(draw, (20, 15), "FLAME 3 real-wildfire composites — annotation legend",
                   font, bg=(20, 20, 20), fg=(255, 255, 255))
    y = 65
    lines = [
        (BAND_COLORS["clear"], "Box outline: CLEAR — real-smoke opacity < 0.12 at paste site"),
        (BAND_COLORS["light"], "Box outline: LIGHT SMOKE — opacity 0.12-0.35"),
        (BAND_COLORS["heavy"], "Box outline: HEAVY SMOKE — opacity >= 0.35 (person barely visible)"),
    ]
    for color, text in lines:
        draw.rectangle([20, y, 60, y + 24], outline=color, width=4)
        draw.text((72, y + 2), text, font=font_sm, fill=(255, 255, 255))
        y += 40
    y += 10
    draw.text((20, y), "Label per box: pose | slant range (m) | GSD (m/px) | smoke opacity (band)",
               font=font_sm, fill=(200, 200, 200))
    y += 30
    draw.text((20, y), "Green header banner: frame with people pasted. Red banner: person-free", font=font_sm, fill=(200, 200, 200))
    y += 26
    draw.text((20, y), "fire frame used as a false-positive control (no ground truth boxes).", font=font_sm, fill=(200, 200, 200))
    y += 34
    draw.text((20, y), "Source: FLAME 3 (Sycan Marsh prescribed burn, DJI M30T) + real SARD person cutouts.",
               font=font_sm, fill=(150, 150, 150))
    y += 26
    draw.text((20, y), "Person size derived from per-frame EXIF altitude + gimbal pitch; smoke opacity", font=font_sm, fill=(150, 150, 150))
    y += 26
    draw.text((20, y), "estimated from the scene and used to attenuate the pasted person's visibility.", font=font_sm, fill=(150, 150, 150))
    img.save(dst_root / "_LEGEND.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/cv_train/flame_composites"))
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    font = _load_font(18)
    font_sm = _load_font(15)

    metas = sorted((src / "meta").glob("*.json"))
    n_people = 0
    n_neg = 0
    for i, mp in enumerate(metas):
        img_path = src / "images" / (mp.stem + ".jpg")
        if not img_path.exists():
            continue
        meta = json.loads(mp.read_text())
        n_people += len(meta.get("people", []))
        if not meta.get("people"):
            n_neg += 1
        annotate_image(img_path, mp, dst / (mp.stem + ".jpg"), font, font_sm)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(metas)} annotated")

    write_legend(dst, font, font_sm)
    print(f"Done: {len(metas)} frames ({n_people} people, {n_neg} FP-control frames) -> {dst}")


if __name__ == "__main__":
    main()
