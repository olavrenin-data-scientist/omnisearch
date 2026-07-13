#!/usr/bin/env python3
"""Measure how often each hard-negative decoy class triggers a false 'person' detection.

The generator records every composited decoy (type, bbox, physical size) in the
per-image metadata sidecar. This script runs the detector on all val images that
contain decoys and reports, per decoy class, the fraction that overlap a
detection (IoU > 0.3). A high rate on 'colorful_object' means the model is using
the shortcut "saturated human-sized blob = person" instead of human shape —
exactly the failure mode the human-scale decoys were added to expose and train
away.

Usage:
    python3 scripts/eval_decoy_fp.py --model models/survivor_yolov8s.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/survivor_yolov8s.pt"))
    ap.add_argument("--val-dir", default=str(ROOT / "data/cv_train/survivor/val"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "reports/decoy_fp_rates.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    val = Path(args.val_dir)
    hits: Counter = Counter()
    totals: Counter = Counter()
    fp_other = 0
    n_imgs = 0

    for meta_path in sorted((val / "labels").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        decoys = meta.get("decoys", [])
        if not decoys:
            continue
        n_imgs += 1
        img = str(val / "images" / f"{meta_path.stem}.jpg")
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
        gt = [
            (b["x_px"], b["y_px"], b["x_px"] + b["w_px"], b["y_px"] + b["h_px"])
            for b in meta.get("boxes", [])
        ]
        decoy_boxes = [
            (d["x_px"], d["y_px"], d["x_px"] + d["w_px"], d["y_px"] + d["h_px"])
            for d in decoys
        ]
        for d, dbox in zip(decoys, decoy_boxes):
            totals[d["type"]] += 1
            if any(_iou(dbox, det) > args.match_iou for det in dets):
                hits[d["type"]] += 1
        for det in dets:
            if not any(_iou(det, g) > args.match_iou for g in gt) and not any(
                _iou(det, db) > args.match_iou for db in decoy_boxes
            ):
                fp_other += 1

    print(f"Model: {args.model}")
    print(f"Images with decoys: {n_imgs}  (conf {args.conf}, imgsz {args.imgsz}, match IoU {args.match_iou})")
    results = {"model": args.model, "conf": args.conf, "imgsz": args.imgsz,
               "n_images": n_imgs, "per_type": {}, "other_background_fps": fp_other}
    for t in sorted(totals):
        rate = hits[t] / totals[t]
        results["per_type"][t] = {"hits": hits[t], "total": totals[t], "fp_rate": round(rate, 4)}
        print(f"  {t:16s}: {hits[t]}/{totals[t]} triggered a 'person' detection ({100*rate:.1f}%)")
    print(f"  other background FPs: {fp_other}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
