#!/usr/bin/env python3
"""Evaluate drone survivor detectors on FLAME 3 real-wildfire composites.

The set (built by scripts/build_flame_composites.py) pastes real SARD person
cutouts onto REAL prescribed-burn drone frames, with per-person real-smoke
opacity recorded at paste time. This answers the question the synthetic val
set cannot: does the RGB model detect people through REAL smoke?

Reports, per inference config and conf cut:
  * recall overall and stratified by smoke band (clear / light / heavy)
  * false positives per image on the person-free fire frames (does real
    fire/smoke texture trigger phantom "survivors"?)

Usage:
    python scripts/eval_flame_composites.py --model models/survivor_yolov8s_1280.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_recall_by_size import _predict_full, _predict_tiled, _match, wilson

Image.MAX_IMAGE_PIXELS = None

MATCH_IOU = 0.15
CONF_CUTS = [0.05, 0.15, 0.35, 0.50]
BANDS = [("clear", 0.00, 0.12), ("light", 0.12, 0.35), ("heavy", 0.35, 1.01)]

# FLAME 3 frames have a DJI on-screen display (model / timestamp / GPS text)
# burned into the top-left corner; the detector fires on the glyphs. That is
# a source-video artifact, not a scene false positive, so detections whose
# centre falls inside the OSD region are dropped. Fractions of image size.
OSD_X_FRAC = 0.30
OSD_Y_FRAC = 0.10


def drop_osd_dets(dets, img_w: int, img_h: int):
    ox, oy = img_w * OSD_X_FRAC, img_h * OSD_Y_FRAC
    out = []
    for (x1, y1, x2, y2), c in dets:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cx < ox and cy < oy:
            continue
        out.append(((x1, y1, x2, y2), c))
    return out


def band_of(opacity: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= opacity < hi:
            return name
    return "heavy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/survivor_yolov8s_1280.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/cv_train/flame_composites"))
    ap.add_argument("--out", default=str(ROOT / "reports/flame_composite_eval.json"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default=None, help="e.g. cpu / mps (default: auto)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.device:
        model.overrides["device"] = args.device

    data = Path(args.data)
    metas = sorted((data / "meta").glob("*.json"))
    if args.n:
        metas = metas[: args.n]

    configs = {
        "full_1280": lambda im: _predict_full(model, im, 1280, 0.7),
        "tiled_2x2_1280": lambda im: _predict_tiled(model, im, 2, 0.2, 1280, 0.7),
        "tiled_4x4_1280": lambda im: _predict_tiled(model, im, 4, 0.2, 1280, 0.7),
    }

    # frames[cfg] = list of (gt_boxes, gt_bands, dets, is_negative_frame)
    frames: dict[str, list] = {k: [] for k in configs}
    n_people = 0
    n_neg_frames = 0
    for i, mp in enumerate(metas):
        meta = json.loads(mp.read_text())
        img_path = data / "images" / (mp.stem + ".jpg")
        if not img_path.exists():
            continue
        gts = [(p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]) for p in meta["people"]]
        bands = [band_of(p["smoke_opacity"]) for p in meta["people"]]
        img = Image.open(img_path).convert("RGB")
        n_people += len(gts)
        if not gts:
            n_neg_frames += 1
        for cfg, fn in configs.items():
            dets = drop_osd_dets(fn(img), *img.size)
            frames[cfg].append((gts, bands, dets, not gts))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(metas)} frames")

    print(f"Evaluated {len(metas)} frames, {n_people} people, {n_neg_frames} person-free fire frames")
    results: dict = {}
    for cfg, rows in frames.items():
        results[cfg] = {}
        for conf in CONF_CUTS:
            tp_band = {b: 0 for b, _, _ in BANDS}
            n_band = {b: 0 for b, _, _ in BANDS}
            fp_total = fp_neg = 0
            ndet = 0
            for gts, bands, dets, is_neg in rows:
                flags, n_fp, n_det = _match(gts, dets, conf, MATCH_IOU)
                for f, b in zip(flags, bands):
                    n_band[b] += 1
                    tp_band[b] += int(f)
                fp_total += n_fp
                ndet += n_det
                if is_neg:
                    fp_neg += n_fp
            tp = sum(tp_band.values())
            n_gt = sum(n_band.values())
            rec = tp / n_gt if n_gt else 0.0
            prec = tp / ndet if ndet else 0.0
            lo, hi = wilson(tp, n_gt)
            entry = {
                "recall": round(rec, 4), "recall_ci95": [round(lo, 4), round(hi, 4)],
                "precision": round(prec, 4), "tp": tp, "n_gt": n_gt,
                "fp_total": fp_total,
                "fp_per_negative_frame": round(fp_neg / n_neg_frames, 3) if n_neg_frames else None,
                "by_smoke_band": {},
            }
            parts = []
            for b, _, _ in BANDS:
                if n_band[b]:
                    blo, bhi = wilson(tp_band[b], n_band[b])
                    entry["by_smoke_band"][b] = {
                        "recall": round(tp_band[b] / n_band[b], 4),
                        "ci95": [round(blo, 4), round(bhi, 4)],
                        "tp": tp_band[b], "n": n_band[b],
                    }
                    parts.append(f"{b} {tp_band[b] / n_band[b]:.0%} ({tp_band[b]}/{n_band[b]})")
            results[cfg][f"conf_{conf}"] = entry
            print(f"  {cfg:16s} conf {conf:4}: R {rec:6.1%} [{lo:.1%}-{hi:.1%}]  "
                  f"P {prec:6.1%}  FP/negframe {entry['fp_per_negative_frame']}  | {'  '.join(parts)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": f"FLAME3 real-smoke composites ({len(metas)} frames, {n_people} people)",
        "model": args.model, "match_iou": MATCH_IOU, "results": results,
    }, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
