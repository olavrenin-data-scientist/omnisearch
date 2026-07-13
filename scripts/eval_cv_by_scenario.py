#!/usr/bin/env python3
"""Measure the REAL trained YOLOv8 survivor detector across the six wildfire
scenarios used by the modality-comparison heatmap.

Method: take the CLEAR held-out validation frames (has_fire=False, so they
carry no baked-in effects), synthesize each scenario's fire/smoke/burn onto
them with the same wildfire-effect renderer the simulator uses
(detection/wildfire_effects.py), and run the real model on each variant.
Ground-truth boxes come from the generation metadata, so recall/precision are
measured, not simulated. This replaces the PreliminaryPersonDetector coin-flip
proxy that produced the misleading 100% "CV" column in
data/detection_mode_comparison.json.

Caveat recorded in the output: effects are composited over the full frame,
including survivor pixels (in the training generator, burn/flame go UNDER
survivors and only smoke goes over). This makes fire/burn scenarios slightly
pessimistic for survivors standing inside the affected region.

Usage:
    python3 scripts/eval_cv_by_scenario.py --model models/survivor_yolov8s.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from detection.wildfire_effects import (
    WildfireEffectConfig,
    apply_wildfire_effects_to_pil,
    masks_from_simulation_grids,
)

SCENARIOS = ["clear", "light_smoke", "heavy_smoke", "active_fire", "burned_ground", "mixed"]


def make_grids(grid_size: int, scenario: str) -> dict:
    """Same scenario definitions as scripts/compare_detection_modes.py."""
    fire = np.zeros((grid_size, grid_size))
    fire_intensity = np.zeros((grid_size, grid_size))
    burned = np.zeros((grid_size, grid_size))
    smoke = np.zeros((grid_size, grid_size))

    if scenario == "clear":
        pass
    elif scenario == "light_smoke":
        smoke[:] = 0.5
    elif scenario == "heavy_smoke":
        smoke[:] = 2.5
    elif scenario == "active_fire":
        fire[3:7, 3:7] = 1.0
        fire_intensity[3:7, 3:7] = 0.8
        smoke[:] = 1.5
    elif scenario == "burned_ground":
        burned[:] = 0.46
        smoke[:] = 0.3
    elif scenario == "mixed":
        fire[0:3, 0:3] = 1.0
        fire_intensity[0:3, 0:3] = 0.6
        burned[3:7, 3:7] = 0.4
        smoke[:] = 1.0

    return {
        "fire_grid": fire,
        "fire_intensity_grid": fire_intensity,
        "burned_grid": burned,
        "smoke_grid": smoke,
    }


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
    ap.add_argument("--grid-size", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "reports/cv_recall_by_scenario.json"))
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.model)
    val = Path(args.val_dir)

    # Base frames: CLEAR val frames with at least one survivor. Frames that
    # already have baked-in fire/smoke would double-apply effects, so skip them.
    frames = []
    for meta_path in sorted((val / "labels").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("has_fire"):
            continue
        gt = [
            (b["x_px"], b["y_px"], b["x_px"] + b["w_px"], b["y_px"] + b["h_px"])
            for b in meta.get("boxes", [])
        ]
        if not gt:
            continue
        frames.append((val / "images" / f"{meta_path.stem}.jpg", gt))
    print(f"Base frames: {len(frames)} clear val frames, "
          f"{sum(len(g) for _, g in frames)} GT boxes")

    cfg = WildfireEffectConfig(seed=7)
    results = {"model": args.model, "conf": args.conf, "imgsz": args.imgsz,
               "match_iou": args.match_iou, "n_frames": len(frames),
               "note": ("Scenario effects synthesized onto clear val frames with "
                        "detection/wildfire_effects.py; effects composited over the "
                        "full frame including survivor pixels (slightly pessimistic "
                        "for fire/burn scenarios)."),
               "scenarios": {}}

    for scenario in SCENARIOS:
        grids = make_grids(args.grid_size, scenario)
        tp = gt_total = det_total = 0
        masks = None
        for img_path, gt in frames:
            img = Image.open(img_path).convert("RGB")
            if scenario != "clear":
                if masks is None or masks.smoke.shape[::-1] != img.size:
                    masks = masks_from_simulation_grids(
                        image_size=img.size,
                        center_world=(0.0, 0.0),
                        footprint_world=2.0,  # crop spans the whole grid
                        **grids,
                    )
                img, _ = apply_wildfire_effects_to_pil(img, masks, config=cfg)

            res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            dets = [tuple(b) for b in res.boxes.xyxy.cpu().numpy()]
            det_total += len(dets)
            gt_total += len(gt)

            matched = set()
            for det in dets:
                best_iou, best_i = 0.0, None
                for i, g in enumerate(gt):
                    if i in matched:
                        continue
                    iou = _iou(det, g)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_i is not None and best_iou >= args.match_iou:
                    matched.add(best_i)
            tp += len(matched)

        recall = tp / gt_total if gt_total else 0.0
        precision = tp / det_total if det_total else 0.0
        results["scenarios"][scenario] = {
            "recall": round(recall, 4), "precision": round(precision, 4),
            "tp": tp, "gt": gt_total, "detections": det_total,
        }
        print(f"  {scenario:14s}: recall {recall*100:5.1f}%  precision {precision*100:5.1f}%  "
              f"(TP {tp}/{gt_total}, dets {det_total})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
