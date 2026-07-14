#!/usr/bin/env python3
"""Evaluate the trained drone survivor detector on REAL aerial SAR imagery.

Dataset: HERIDAL (University of Split) — 4000x3000 px wilderness photos taken
from a real drone at ~50 m altitude, with PASCAL VOC person annotations. This
is the closest public match to our deployment domain (nadir/oblique wilderness
person search), so recall/precision here measures the sim-to-real gap of the
synthetic-composite training pipeline.

Inference configs mirror scripts/eval_recall_by_size.py:
  A) full frame at imgsz 1280
  B) tiled 4x4 with 20% overlap, imgsz 1280 per tile (a 4000px frame maps to
     ~1000px tiles - the same effective px-per-tile as the deployment 2x2 grid
     on 2048px simulator frames)

Usage:
    python scripts/eval_heridal.py --model models/survivor_yolov8s_1280.pt
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_recall_by_size import _predict_full, _predict_tiled, _match, wilson

Image.MAX_IMAGE_PIXELS = None

MATCH_IOU = 0.15
CONF_CUTS = [0.05, 0.15, 0.35, 0.50]


def _voc_boxes(xml_path: Path) -> list[tuple[float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").lower()
        if name not in ("person", "human"):
            continue
        bb = obj.find("bndbox")
        out.append((float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                    float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/survivor_yolov8s_1280.pt"))
    ap.add_argument("--images", required=True, help="Dir with HERIDAL test JPGs")
    ap.add_argument("--annotations", required=True, help="Dir with VOC XMLs")
    ap.add_argument("--n", type=int, default=0, help="Limit images (0 = all)")
    ap.add_argument("--split-file", default=None,
                    help="Optional VOC ImageSets file listing image stems to use")
    ap.add_argument("--tile-grid", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "reports/heridal_real_eval.json"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    img_dir = Path(args.images)
    ann_dir = Path(args.annotations)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.split_file:
        stems = {line.strip() for line in open(args.split_file) if line.strip()}
        images = [p for p in images if p.stem in stems]
    if args.n:
        images = images[: args.n]

    configs = {
        "full_1280": lambda im: _predict_full(model, im, 1280, 0.7),
        f"tiled_{args.tile_grid}x{args.tile_grid}_1280": lambda im: _predict_tiled(
            model, im, args.tile_grid, 0.2, 1280, 0.7),
    }

    # Collect detections once per config at low conf; sweep conf cuts after.
    per_cfg: dict[str, list[tuple[list, list]]] = {k: [] for k in configs}
    n_gt_total = 0
    n_used = 0
    for i, img_path in enumerate(images):
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue
        gts = _voc_boxes(xml_path)
        if not gts:
            continue
        img = Image.open(img_path).convert("RGB")
        n_gt_total += len(gts)
        n_used += 1
        for cfg, fn in configs.items():
            per_cfg[cfg].append((gts, fn(img)))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(images)} images (used {n_used}, {n_gt_total} GT boxes)")

    print(f"Evaluated {n_used} images with {n_gt_total} GT person boxes")
    results = {}
    for cfg, frames in per_cfg.items():
        results[cfg] = {}
        for conf in CONF_CUTS:
            tp = fp = ndet = 0
            for gts, dets in frames:
                flags, n_fp, n_det = _match(gts, dets, conf, MATCH_IOU)
                tp += sum(flags)
                fp += n_fp
                ndet += n_det
            rec = tp / n_gt_total if n_gt_total else 0.0
            prec = tp / ndet if ndet else 0.0
            lo, hi = wilson(tp, n_gt_total)
            results[cfg][f"conf_{conf}"] = {
                "recall": round(rec, 4), "recall_ci95": [round(lo, 4), round(hi, 4)],
                "precision": round(prec, 4), "tp": tp, "fp": fp, "n_det": ndet,
            }
            print(f"  {cfg:22s} conf {conf:4}: R {rec:6.1%} [{lo:.1%}-{hi:.1%}]  P {prec:6.1%}  (TP {tp}/{n_gt_total})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": f"HERIDAL real aerial SAR ({n_used} images, {n_gt_total} GT boxes)",
        "model": args.model, "match_iou": MATCH_IOU, "results": results,
    }, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
