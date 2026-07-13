#!/usr/bin/env python3
"""Measure the REAL trained YOLOv8 survivor detector's recall/precision,
split by the one condition axis our generated val set actually has ground
truth for: ``has_fire`` (clear vs. fire/smoke present).

This replaces the synthetic ``PreliminaryPersonDetector`` stand-in used by
``compare_detection_modes.py`` for the "CV" column of the modality-comparison
heatmap. That stand-in starts from ground-truth boxes and coin-flips each one
with a hardcoded probability — it never runs the actual model, so its 100%
"recall" numbers do not reflect real detector performance. This script runs
the real ``models/survivor_yolov8s.pt`` on real held-out frames instead.

Usage:
    python3 scripts/eval_cv_by_fire_condition.py --model models/survivor_yolov8s.pt
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--match-iou", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "reports/cv_recall_by_fire_condition.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    val = Path(args.val_dir)

    buckets = {
        "clear": {"tp": 0, "gt": 0, "det": 0},
        "fire_or_smoke": {"tp": 0, "gt": 0, "det": 0},
    }

    for meta_path in sorted((val / "labels").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        bucket_name = "fire_or_smoke" if meta.get("has_fire") else "clear"
        bucket = buckets[bucket_name]

        gt = [
            (b["x_px"], b["y_px"], b["x_px"] + b["w_px"], b["y_px"] + b["h_px"])
            for b in meta.get("boxes", [])
        ]
        bucket["gt"] += len(gt)
        if not gt:
            continue  # negative frame: nothing to recall, skip detection cost

        img = str(val / "images" / f"{meta_path.stem}.jpg")
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
        bucket["det"] += len(dets)

        matched_gt = set()
        for det in dets:
            best_iou, best_i = 0.0, None
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                iou = _iou(det, g)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i is not None and best_iou >= args.match_iou:
                matched_gt.add(best_i)
        bucket["tp"] += len(matched_gt)

    print(f"Model: {args.model}  (conf {args.conf}, imgsz {args.imgsz}, match IoU {args.match_iou})")
    results = {"model": args.model, "conf": args.conf, "imgsz": args.imgsz,
               "match_iou": args.match_iou, "buckets": {}}
    for name, b in buckets.items():
        recall = b["tp"] / b["gt"] if b["gt"] else 0.0
        precision = b["tp"] / b["det"] if b["det"] else 0.0
        results["buckets"][name] = {
            "recall": round(recall, 4), "precision": round(precision, 4),
            "tp": b["tp"], "gt": b["gt"], "detections": b["det"],
        }
        print(f"  {name:14s}: recall {recall*100:.1f}%  precision {precision*100:.1f}%  (GT={b['gt']}, TP={b['tp']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
