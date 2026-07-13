#!/usr/bin/env python3
"""Measure detector recall stratified by raw target pixel size.

Runs the drone survivor detector over a val split at three inference
configurations and reports recall per ground-truth size bucket, so the
"tiled inference rescues small targets" claim is a measured number:

  A) imgsz 640, full frame        (naive baseline)
  B) imgsz 1280, full frame       (upsampled)
  C) 2x2 tiles, 20% overlap, imgsz 1280 per tile  (deployment config,
     mirrors SimulationCvAdapter._detect_people_cv)

Protocol (per external review):
  * Detections are collected once at conf 0.001 so recall reflects "can the
    model see it" rather than a fixed operating point; the PR summary at
    several conf cuts is what you pick the deployment threshold from.
  * Recall is reported at TWO match-IoU thresholds (0.15 = deployment
    person_match_iou, and 0.30). A large gap between them means targets are
    being found but poorly localized — a different fix than "not found".
  * Wilson 95% intervals are reported per bucket, since a 100-200-box bucket
    moves whole points per hit.

Usage:
    python scripts/eval_recall_by_size.py                 # full val split
    python scripts/eval_recall_by_size.py --model models/survivor_yolov8s.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = [(0, 8), (8, 15), (15, 10_000)]
BUCKET_NAMES = ["<8px", "8-15px", ">15px"]
MATCH_IOUS = [0.15, 0.30]
DETECT_CONF = 0.001
PR_CONF_CUTS = [0.001, 0.05, 0.15, 0.35, 0.50]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _nms(boxes, confs, thr=0.5):
    order = np.argsort(confs)[::-1]
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _predict_full(model, img, imgsz, nms_iou):
    res = model.predict(source=img, classes=[0], conf=DETECT_CONF, iou=nms_iou,
                        imgsz=imgsz, max_det=300, verbose=False)
    out = []
    r = res[0] if res else None
    if r is not None and r.boxes is not None and len(r.boxes):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy()):
            out.append(((float(x1), float(y1), float(x2), float(y2)), float(c)))
    return out


def _predict_tiled(model, img, grid, overlap, imgsz, nms_iou):
    """Mirror of SimulationCvAdapter tiled inference (2x2, 20% overlap)."""
    W, H = img.size
    step_x, step_y = W / grid, H / grid
    ov_x, ov_y = step_x * overlap, step_y * overlap
    boxes, confs = [], []
    for ty in range(grid):
        for tx in range(grid):
            x0 = max(0, int(tx * step_x - ov_x))
            y0 = max(0, int(ty * step_y - ov_y))
            x1 = min(W, int((tx + 1) * step_x + ov_x))
            y1 = min(H, int((ty + 1) * step_y + ov_y))
            tile = img.crop((x0, y0, x1, y1))
            for (bx1, by1, bx2, by2), c in _predict_full(model, tile, imgsz, nms_iou):
                boxes.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0))
                confs.append(c)
    keep = _nms(boxes, confs, thr=0.5)
    return [(boxes[i], confs[i]) for i in keep]


def _match(gts, dets, conf_cut, match_iou):
    """Greedy conf-descending matching. Returns (tp_flags_per_gt, n_fp, n_det)."""
    live = [(b, c) for b, c in dets if c >= conf_cut]
    live.sort(key=lambda x: -x[1])
    matched_gt = [False] * len(gts)
    n_fp = 0
    for box, _c in live:
        best, best_v = None, match_iou
        for gi, g in enumerate(gts):
            if matched_gt[gi]:
                continue
            v = _iou(g, box)
            if v >= best_v:
                best, best_v = gi, v
        if best is not None:
            matched_gt[best] = True
        else:
            n_fp += 1
    return matched_gt, n_fp, len(live)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "data/cv_train/survivor/val"))
    ap.add_argument("--model", default=str(ROOT / "models/survivor_yolov8s.pt"))
    ap.add_argument("--n", type=int, default=0, help="Limit images (0 = all).")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="Deployment person_iou.")
    ap.add_argument("--out", default=str(ROOT / "reports/recall_by_size.json"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    split = Path(args.split)
    img_paths = sorted((split / "images").glob("*.jpg"))
    if args.n:
        img_paths = img_paths[: args.n]

    configs = {
        "A_640_full":   lambda im: _predict_full(model, im, 640, args.nms_iou),
        "B_1280_full":  lambda im: _predict_full(model, im, 1280, args.nms_iou),
        "C_1280_tiled": lambda im: _predict_tiled(model, im, 2, 0.2, 1280, args.nms_iou),
    }

    # Collect GT + raw detections once (at DETECT_CONF); threshold in post.
    frames = []
    for k, ip in enumerate(img_paths):
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        gts, buckets = [], []
        txt = split / "labels" / f"{ip.stem}.txt"
        for line in txt.read_text().strip().splitlines() if txt.exists() else []:
            p = line.split()
            if len(p) == 5:
                _, cx, cy, w, h = (float(v) for v in p)
                g = ((cx - w / 2) * W, (cy - h / 2) * H,
                     (cx + w / 2) * W, (cy + h / 2) * H)
                gts.append(g)
                long_px = max(g[2] - g[0], g[3] - g[1])
                buckets.append(next(bi for bi, (lo, hi) in enumerate(BUCKETS)
                                    if lo <= long_px < hi))
        dets = {name: fn(img) for name, fn in configs.items()}
        frames.append((gts, buckets, dets))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(img_paths)} images", flush=True)

    print(f"\nModel: {args.model}")
    print(f"Val split: {split}  ({len(img_paths)} images, detections at conf {DETECT_CONF})")

    results = {}
    gt_counts = [0] * len(BUCKETS)
    for gts, buckets, _d in frames:
        for b in buckets:
            gt_counts[b] += 1
    print(f"GT boxes per bucket: {dict(zip(BUCKET_NAMES, gt_counts))}\n")

    for name in configs:
        results[name] = {}
        print(f"== {name}")
        # Recall by bucket at the low-conf end, both match IoUs, with Wilson CI.
        for miou in MATCH_IOUS:
            tp = [0] * len(BUCKETS)
            for gts, buckets, dets in frames:
                flags, _fp, _n = _match(gts, dets[name], DETECT_CONF, miou)
                for gi, hit in enumerate(flags):
                    if hit:
                        tp[buckets[gi]] += 1
            row = {}
            cells = []
            for bi, bn in enumerate(BUCKET_NAMES):
                r = tp[bi] / gt_counts[bi] if gt_counts[bi] else float("nan")
                lo, hi = wilson(tp[bi], gt_counts[bi])
                row[bn] = {"recall": r, "ci95": [lo, hi], "tp": tp[bi], "gt": gt_counts[bi]}
                cells.append(f"{bn} {r:6.1%} [{lo:.0%}-{hi:.0%}]")
            overall = sum(tp) / sum(gt_counts)
            row["overall"] = overall
            results[name][f"recall@iou{miou}"] = row
            print(f"  match IoU {miou:.2f}:  " + "   ".join(cells)
                  + f"   overall {overall:6.1%}")
        # PR summary across conf cuts (match IoU 0.15) for operating-point choice.
        pr = []
        for cut in PR_CONF_CUTS:
            tp_all, fp_all, det_all = 0, 0, 0
            for gts, buckets, dets in frames:
                flags, fp, n = _match(gts, dets[name], cut, 0.15)
                tp_all += sum(flags); fp_all += fp; det_all += n
            rec = tp_all / sum(gt_counts)
            prec = tp_all / det_all if det_all else float("nan")
            pr.append({"conf": cut, "recall": rec, "precision": prec, "fp": fp_all})
        results[name]["pr_curve_iou0.15"] = pr
        print("  PR (IoU 0.15): " + "  ".join(
            f"conf {p['conf']:g}: R {p['recall']:.1%}/P {p['precision']:.1%}" for p in pr))
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "split": str(split), "n_images": len(img_paths),
        "detect_conf": DETECT_CONF, "match_ious": MATCH_IOUS,
        "gt_by_bucket": dict(zip(BUCKET_NAMES, gt_counts)),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
