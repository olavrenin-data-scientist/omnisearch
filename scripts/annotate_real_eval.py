#!/usr/bin/env python3
"""Annotate a real-image YOLO eval split with GT vs. model predictions (TP/FN/FP).

Draws:
  * green box  = true positive  (GT matched by a detection >= --conf)
  * yellow box = false negative (GT missed by the model)
  * red box    = false positive (detection with no matching GT)

Used to visually inspect the real-data eval sets (HIT-UAV thermal, HERIDAL
drone, COCO UGV) alongside the numbers in docs/real_data_eval_report.md.

Usage:
    python scripts/annotate_real_eval.py \
        --images data/cv_train/thermal_real/test/images \
        --labels data/cv_train/thermal_real/test/labels \
        --model models/thermal_real_yolov8n.pt \
        --dst ~/Documents/omnisearch_capstone/thermal_hituav_annotated \
        --title "Thermal IR — HIT-UAV real test split" \
        --imgsz 640 --conf 0.25 --n 80
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = None


def _load_font(size: int):
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


TP_COLOR = (57, 255, 20)
FN_COLOR = (255, 210, 60)
FP_COLOR = (255, 60, 60)
BANNER_BG = (15, 20, 40)


def _text_with_bg(draw, xy, text, font, fg=(255, 255, 255), bg=(0, 0, 0), pad=3):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle([l - pad, t - pad, r + pad, b + pad], fill=bg)
    draw.text((x, y), text, fill=fg, font=font)


def _yolo_to_xyxy(line: str, w: int, h: int) -> tuple[float, float, float, float]:
    _, cx, cy, bw, bh = (float(v) for v in line.split())
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def write_legend(dst_root: Path, title: str, model: str, extra_lines: list[str], font, font_sm) -> None:
    W, H = 900, 300 + 26 * len(extra_lines)
    img = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    _text_with_bg(draw, (20, 15), title, font, bg=(20, 20, 20))
    y = 60
    for color, text in (
        (TP_COLOR, "Green: TRUE POSITIVE — a real ground-truth person, correctly detected"),
        (FN_COLOR, "Yellow: FALSE NEGATIVE — a real ground-truth person the model MISSED"),
        (FP_COLOR, "Red: FALSE POSITIVE — a detection with no matching ground-truth person"),
    ):
        draw.rectangle([20, y, 60, y + 24], outline=color, width=4)
        draw.text((72, y + 2), text, font=font_sm, fill=(255, 255, 255))
        y += 38
    y += 8
    draw.text((20, y), f"Model: {model}", font=font_sm, fill=(200, 200, 200))
    y += 28
    for line in extra_lines:
        draw.text((20, y), line, font=font_sm, fill=(150, 150, 150))
        y += 26
    img.save(dst_root / "_LEGEND.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--title", default="Real-data evaluation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=80, help="Number of images to annotate (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--extra-note", default="", help="Extra legend line, e.g. dataset provenance")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.device:
        model.overrides["device"] = args.device

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.n:
        images = images[: args.n]

    font = _load_font(20)
    font_sm = _load_font(15)

    n_tp = n_fn = n_fp = 0
    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gts = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if line.strip():
                    gts.append(_yolo_to_xyxy(line, W, H))

        res = model.predict(source=img, classes=[0], conf=args.conf, imgsz=args.imgsz,
                             max_det=300, verbose=False)
        dets = []
        r = res[0] if res else None
        if r is not None and r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                dets.append(((float(x1), float(y1), float(x2), float(y2)), float(c)))
        dets.sort(key=lambda d: -d[1])

        matched_gt = [False] * len(gts)
        matched_det = [False] * len(dets)
        for di, (dbox, _c) in enumerate(dets):
            best, best_v = None, args.match_iou
            for gi, gbox in enumerate(gts):
                if matched_gt[gi]:
                    continue
                v = _iou(gbox, dbox)
                if v >= best_v:
                    best, best_v = gi, v
            if best is not None:
                matched_gt[best] = True
                matched_det[di] = True

        draw = ImageDraw.Draw(img)
        pad = max(2, int(min(W, H) * 0.004))
        for gi, gbox in enumerate(gts):
            color = TP_COLOR if matched_gt[gi] else FN_COLOR
            n_tp += int(matched_gt[gi])
            n_fn += int(not matched_gt[gi])
            draw.rectangle([gbox[0] - pad, gbox[1] - pad, gbox[2] + pad, gbox[3] + pad],
                            outline=color, width=max(2, pad))
        for di, (dbox, c) in enumerate(dets):
            if matched_det[di]:
                continue
            n_fp += 1
            draw.rectangle([dbox[0] - pad, dbox[1] - pad, dbox[2] + pad, dbox[3] + pad],
                            outline=FP_COLOR, width=max(2, pad))
            _text_with_bg(draw, (dbox[0], max(0, dbox[1] - 20)), f"{c:.2f}", font_sm, fg=FP_COLOR)

        n_tp_img = sum(matched_gt)
        n_fn_img = len(gts) - n_tp_img
        n_fp_img = sum(1 for d in matched_det if not d)
        header = f"{img_path.name}  |  GT {len(gts)}  TP {n_tp_img}  FN {n_fn_img}  FP {n_fp_img}"
        bh = 34
        draw.rectangle([0, 0, W, bh], fill=BANNER_BG)
        _text_with_bg(draw, (8, 6), header, font_sm, bg=BANNER_BG)

        img.save(dst / img_path.name, quality=90)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(images)} annotated")

    extra = [f"Eval set: {img_dir}", f"conf >= {args.conf}, match IoU >= {args.match_iou}",
             f"Totals over {len(images)} images: TP {n_tp}  FN {n_fn}  FP {n_fp}"]
    if args.extra_note:
        extra.append(args.extra_note)
    write_legend(dst, args.title, args.model, extra, font, font_sm)
    print(f"Done: {len(images)} images, TP {n_tp} FN {n_fn} FP {n_fp} -> {dst}")


if __name__ == "__main__":
    main()
