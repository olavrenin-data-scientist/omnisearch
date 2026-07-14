"""Annotate survivor dataset images with physical object sizes and terrain scale.

For every image in a YOLO split (``images/`` + ``labels/`` with a ``.json``
metadata sidecar carrying ``gsd_m``), this draws:

  * each survivor bounding box,
  * the box's real-world size in metres (px x gsd),
  * a header with the ground-sample-distance (m/px), altitude and camera mode,
  * a scale bar calibrated to the terrain resolution.

This is the same "size vs. terrain" visualisation used for the presentation
panels, applied to a whole folder.

Usage:
    python scripts/annotate_survivor_sizes.py \
        --src exports/survivor_fixed_20260708 \
        --dst exports/survivor_fixed_20260708/annotated
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


BOX_COLOR = (57, 255, 20)      # lime — labeled survivors
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)
BANNER_BG = (15, 40, 20)
WARN_BG = (140, 20, 20)        # red banner for physically unverifiable frames
SCALE_COLOR = (255, 210, 60)   # amber
DECOY_COLORS = {               # unlabeled hard-negative decoys, by class
    "vehicle": (255, 150, 40),         # orange
    "animal": (255, 60, 60),           # red
    "colorful_object": (255, 80, 255), # magenta
    "synthetic_blob": (200, 200, 80),  # olive
}


def _image_gsd(meta: dict, image_size: int) -> float | None:
    if meta.get("gsd_m"):
        return float(meta["gsd_m"])
    if meta.get("footprint_m"):
        return float(meta["footprint_m"]) / image_size
    return None


def _text_with_bg(draw, xy, text, font, fg=TEXT_FG, bg=TEXT_BG, pad=2):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle([l - pad, t - pad, r + pad, b + pad], fill=bg)
    draw.text((x, y), text, fill=fg, font=font)


def _nice_scale_metres(gsd: float, image_w: int) -> float:
    """Pick a round scale-bar length (m) that spans ~1/4 of the image width."""
    target_px = image_w * 0.25
    target_m = target_px * gsd
    for step in (1, 2, 5, 10, 20, 50, 100):
        if step >= target_m:
            return float(step)
    return 100.0


def annotate_image(img_path: Path, label_path: Path, meta_path: Path, dst_path: Path) -> bool:
    """Annotate one frame. Returns True if the frame carried scale metadata."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    gsd = _image_gsd(meta, W)
    meta_boxes = meta.get("boxes") or []

    draw = ImageDraw.Draw(img)
    font_box = _load_font(max(12, W // 45))
    font_hdr = _load_font(max(13, W // 40))

    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _cls, cx, cy, bw, bh = (float(v) for v in p)
            boxes.append((cx * W, cy * H, bw * W, bh * H))

    # Perspective (UGV) frames have per-object scale rather than a frame GSD.
    per_box_scale = [
        float(mb["m_per_px"]) if "m_per_px" in mb else None for mb in meta_boxes
    ]
    has_scale = gsd is not None or (
        bool(boxes) and len(per_box_scale) == len(boxes)
        and all(s is not None for s in per_box_scale)
    ) or (not boxes and bool(meta))

    for bi, (cx, cy, bw, bh) in enumerate(boxes):
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
        scale = gsd if gsd is not None else (
            per_box_scale[bi] if bi < len(per_box_scale) else None
        )
        if scale:
            label = f"{bw:.0f}x{bh:.0f} px | {bw * scale:.1f}x{bh * scale:.1f} m"
            if gsd is None:  # perspective frame: scale is per-object, show it
                label += f" ({scale:.3f} m/px)"
            if bi < len(meta_boxes) and meta_boxes[bi].get("range_m"):
                label += f" @{meta_boxes[bi]['range_m']:.0f}m"
        else:
            label = f"{bw:.0f}x{bh:.0f} px (NO SCALE)"
        ty = y1 - font_box.size - 6
        if ty < 2:
            ty = y2 + 4
        _text_with_bg(draw, (max(2, x1), ty), label, font_box)

    # Hard-negative decoys (recorded in metadata, never in YOLO labels):
    # drawn in a per-class colour so they are visibly NOT survivor labels.
    for d in meta.get("decoys", []):
        color = DECOY_COLORS.get(d.get("type", ""), (255, 255, 0))
        dx1, dy1 = d["x_px"], d["y_px"]
        dx2, dy2 = dx1 + d["w_px"], dy1 + d["h_px"]
        draw.rectangle([dx1, dy1, dx2, dy2], outline=color, width=2)
        dlabel = d.get("type", "decoy")
        if gsd is not None:
            dlabel += f" {d['w_px']}x{d['h_px']} px | {d['w_px']*gsd:.1f}x{d['h_px']*gsd:.1f} m"
        else:
            dlabel += f" {d['w_px']}x{d['h_px']} px"
        ty = dy2 + 4 if dy2 + font_box.size + 6 < H else dy1 - font_box.size - 6
        _text_with_bg(draw, (max(2, dx1), ty), dlabel, font_box, bg=(40, 20, 40))

    # Header banner with terrain scale + capture geometry.
    parts = []
    if meta.get("camera"):        # UGV perspective frame
        parts.append(f"cam {meta['camera']}")
        if meta.get("mast_height_m"):
            parts.append(f"mast {meta['mast_height_m']:.0f} m")
    else:
        mode = "oblique" if meta.get("oblique") else "nadir"
        parts.append(mode + (f" {meta.get('tilt_deg', 0):.0f}deg" if meta.get("oblique") else ""))
    if gsd:
        parts.insert(0, f"GSD {gsd:.3f} m/px")
    if meta.get("altitude_m") is not None:
        parts.append(f"alt {meta['altitude_m']:.0f} m")
    parts.append(f"{len(boxes)} survivor" + ("s" if len(boxes) != 1 else ""))
    if meta.get("n_decoys"):
        parts.append(f"{meta['n_decoys']} decoy" + ("s" if meta["n_decoys"] != 1 else ""))
    if not has_scale:
        parts.append("!! NO GSD / SCALE METADATA — sizes unverifiable")
    header = "   ".join(parts)
    hb = draw.textbbox((0, 0), header, font=font_hdr)
    draw.rectangle([0, 0, W, hb[3] - hb[1] + 8],
                   fill=BANNER_BG if has_scale else WARN_BG)
    draw.text((6, 3), header, fill=(230, 255, 230), font=font_hdr)

    # Scale bar (bottom-left) calibrated to the terrain resolution.
    if gsd:
        bar_m = _nice_scale_metres(gsd, W)
        bar_px = bar_m / gsd
        bx, by = 10, H - 18
        draw.line([bx, by, bx + bar_px, by], fill=SCALE_COLOR, width=4)
        draw.line([bx, by - 5, bx, by + 5], fill=SCALE_COLOR, width=4)
        draw.line([bx + bar_px, by - 5, bx + bar_px, by + 5], fill=SCALE_COLOR, width=4)
        _text_with_bg(draw, (bx, by - font_box.size - 8), f"{bar_m:.0f} m",
                      font_box, fg=(20, 20, 20), bg=SCALE_COLOR)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, quality=90)
    return has_scale


def process_split(src_split: Path, dst_split: Path) -> tuple[int, int]:
    """Annotate one split; returns (n_images, n_missing_scale)."""
    img_dir = src_split / "images"
    lbl_dir = src_split / "labels"
    paths = sorted(
        glob.glob(str(img_dir / "*.jpg"))
        + glob.glob(str(img_dir / "*.jpeg"))
        + glob.glob(str(img_dir / "*.png"))
    )
    n = 0
    missing = 0
    for img_path in paths:
        stem = Path(img_path).stem
        ok = annotate_image(
            Path(img_path),
            lbl_dir / f"{stem}.txt",
            lbl_dir / f"{stem}.json",
            dst_split / f"{stem}.jpg",
        )
        if not ok:
            missing += 1
        n += 1
    return n, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Dataset root containing train/ and/or val/")
    ap.add_argument("--dst", required=True, help="Output root for annotated images")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    total = 0
    total_missing = 0
    # Split dirs are any directories containing both images/ and labels/
    # (handles nested layouts like ugv/front/train).
    for lbl in sorted(src.rglob("labels")):
        s = lbl.parent
        if not (s / "images").is_dir():
            continue
        rel = s.relative_to(src)
        k, missing = process_split(s, dst / rel)
        note = f" ({missing} frames MISSING scale metadata)" if missing else ""
        print(f"  {rel}: annotated {k} images -> {dst / rel}{note}")
        total += k
        total_missing += missing
    print(f"Done. {total} annotated images written to {dst}"
          + (f"; {total_missing} frames lacked scale metadata" if total_missing else ""))


if __name__ == "__main__":
    main()
