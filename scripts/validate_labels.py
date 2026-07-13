#!/usr/bin/env python3
"""Standalone physical-scale validator for the synthetic CV datasets.

For every frame in a dataset directory the validator flags:

  R1  size      — any label whose long axis exceeds 2.0 m, or (nadir aerial)
                  whose short axis exceeds 0.7 m.
  R2  coherence — aerial: frame max/min survivor pixel-size (sqrt box area)
                  ratio > 2.0. UGV (perspective): per-object implied person
                  long axis outside [1.3, 2.2] m after de-foreshortening.
  R3  metadata  — frame missing GSD / per-object scale metadata, so label
                  sizes are physically unverifiable.
  R4  loose-box — any box whose stored box-vs-alpha-mask IoU < 0.9. Legacy
                  frames without a stored IoU are counted as "unknown".

Writes a CSV of offending frame/box IDs and prints per-dataset summary counts.

Usage:
    python scripts/validate_labels.py                         # all datasets
    python scripts/validate_labels.py --datasets survivor ugv
    python scripts/validate_labels.py --out my_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MAX_LONG_AXIS_M = 2.0
MAX_NADIR_SHORT_AXIS_M = 0.7
MAX_FRAME_SIZE_RATIO = 2.0
UGV_LONG_AXIS_RANGE_M = (1.3, 2.2)
MIN_MASK_IOU = 0.9

# kind: "aerial" (one GSD per frame, nadir/oblique), "thermal" (aerial but
# boxes are ~1.5 m WARM SPOTS — heat spreads beyond the body, so near-square
# boxes are physically correct and the 0.7 m body short-axis rule is not
# applied), or "ugv" (perspective, per-object scale).
DATASET_KINDS = {
    "survivor": "aerial",
    "survivor_naip": "aerial",
    "thermal": "thermal",
    "survivor_ground": "ugv",
    "ugv": "ugv",
}


@dataclass
class Summary:
    frames: int = 0
    boxes: int = 0
    rule_frames: dict[str, set] = field(default_factory=dict)
    rule_boxes: dict[str, int] = field(default_factory=dict)
    iou_unknown: int = 0

    def flag(self, rule: str, frame: str, n_boxes: int = 0) -> None:
        self.rule_frames.setdefault(rule, set()).add(frame)
        self.rule_boxes[rule] = self.rule_boxes.get(rule, 0) + n_boxes


def _find_splits(ds_dir: Path) -> list[Path]:
    """Split dirs are any directories containing both images/ and labels/."""
    splits = []
    for cand in sorted(ds_dir.rglob("labels")):
        if (cand.parent / "images").is_dir():
            splits.append(cand.parent)
    return splits


def _read_labels(txt: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            _, cx, cy, w, h = parts[:5]
            boxes.append((float(cx), float(cy), float(w), float(h)))
    return boxes


def _frame_gsd(meta: dict | None, img_size: int) -> float | None:
    """Per-frame GSD in m/px from metadata (aerial datasets)."""
    if not meta:
        return None
    if meta.get("gsd_m"):
        return float(meta["gsd_m"])
    if meta.get("footprint_m"):  # thermal sidecars store the ground footprint
        return float(meta["footprint_m"]) / img_size
    return None


def validate_dataset(ds_dir: Path, kind: str, writer: csv.writer) -> Summary:
    summary = Summary()
    ds_name = ds_dir.name

    for split_dir in _find_splits(ds_dir):
        split_label = str(split_dir.relative_to(ds_dir))
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        images = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for img_path in images:
            stem = img_path.stem
            frame_id = f"{ds_name}/{split_label}/{stem}"
            summary.frames += 1
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            img_size = max(img_w, img_h)

            boxes = _read_labels(lbl_dir / f"{stem}.txt")
            summary.boxes += len(boxes)

            meta = None
            meta_path = lbl_dir / f"{stem}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_boxes = (meta or {}).get("boxes") or []

            # ---- R3: missing scale metadata -----------------------------
            if kind in ("aerial", "thermal"):
                gsd = _frame_gsd(meta, img_size)
                has_scale = gsd is not None
            else:
                gsd = None
                # UGV frames need per-object m_per_px (perspective camera has
                # no single frame GSD). Negative frames need no per-box scale.
                has_scale = bool(meta) and (
                    not boxes or (len(meta_boxes) == len(boxes)
                                  and all("m_per_px" in b for b in meta_boxes))
                )
            if not has_scale:
                summary.flag("R3_missing_metadata", frame_id)
                writer.writerow([ds_name, split_label, stem, "", "R3_missing_metadata",
                                 "", "no GSD / per-object scale metadata"])

            is_oblique = bool((meta or {}).get("oblique", False))

            px_sizes = []
            for bi, (cx, cy, wn, hn) in enumerate(boxes):
                w_px, h_px = wn * img_w, hn * img_h
                px_sizes.append((w_px * h_px) ** 0.5)
                mb = meta_boxes[bi] if bi < len(meta_boxes) else {}

                # ---- R1: physical size ----------------------------------
                if kind in ("aerial", "thermal") and gsd is not None:
                    w_m, h_m = w_px * gsd, h_px * gsd
                    long_m, short_m = max(w_m, h_m), min(w_m, h_m)
                    if long_m > MAX_LONG_AXIS_M:
                        summary.flag("R1_size", frame_id, 1)
                        writer.writerow([ds_name, split_label, stem, bi, "R1_size",
                                         f"{long_m:.2f}", "long axis > 2.0 m"])
                    elif (kind == "aerial" and not is_oblique
                          and short_m > MAX_NADIR_SHORT_AXIS_M):
                        # A warm spot (thermal) legitimately spreads past the
                        # 0.7 m body cross-section, so this body-geometry rule
                        # only applies to visual aerial boxes.
                        summary.flag("R1_size", frame_id, 1)
                        writer.writerow([ds_name, split_label, stem, bi, "R1_size",
                                         f"{short_m:.2f}", "nadir short axis > 0.7 m"])
                elif kind == "ugv" and mb.get("m_per_px"):
                    # ---- R2 (UGV form): implied person size -------------
                    m_px = float(mb["m_per_px"])
                    foreshort = float(mb.get("foreshortening", 1.0)) or 1.0
                    implied = max(w_px, h_px) * m_px / foreshort
                    lo, hi = UGV_LONG_AXIS_RANGE_M
                    if not (lo <= implied <= hi):
                        summary.flag("R2_coherence", frame_id, 1)
                        writer.writerow([ds_name, split_label, stem, bi, "R2_coherence",
                                         f"{implied:.2f}",
                                         "implied person long axis outside [1.3, 2.2] m"])

                # ---- R4: loose box (stored mask IoU) --------------------
                iou = mb.get("mask_iou")
                if iou is None:
                    summary.iou_unknown += 1
                elif float(iou) < MIN_MASK_IOU:
                    summary.flag("R4_loose_box", frame_id, 1)
                    writer.writerow([ds_name, split_label, stem, bi, "R4_loose_box",
                                     f"{float(iou):.3f}", "box-vs-mask IoU < 0.9"])

            # ---- R2 (aerial form): per-frame pixel-size ratio -----------
            if kind in ("aerial", "thermal") and len(px_sizes) >= 2:
                ratio = max(px_sizes) / max(1e-6, min(px_sizes))
                if ratio > MAX_FRAME_SIZE_RATIO:
                    summary.flag("R2_coherence", frame_id, len(boxes))
                    writer.writerow([ds_name, split_label, stem, "", "R2_coherence",
                                     f"{ratio:.2f}",
                                     "frame max/min survivor pixel-size ratio > 2.0"])

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate physical scale of CV dataset labels")
    ap.add_argument("--data-dir", default=str(ROOT / "data/cv_train"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASET_KINDS),
                    help="Dataset directory names under --data-dir.")
    ap.add_argument("--out", default=str(ROOT / "reports/label_validation.csv"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rules = ["R1_size", "R2_coherence", "R3_missing_metadata", "R4_loose_box"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "split", "frame", "box", "rule", "value", "detail"])

        grand = {}
        for name in args.datasets:
            ds_dir = data_dir / name
            if not ds_dir.is_dir():
                print(f"-- {name}: not found, skipping")
                continue
            kind = DATASET_KINDS.get(name, "aerial")
            s = validate_dataset(ds_dir, kind, writer)
            grand[name] = s
            print(f"\n== {name} ({kind}): {s.frames} frames, {s.boxes} boxes")
            for rule in rules:
                frames = len(s.rule_frames.get(rule, ()))
                boxes = s.rule_boxes.get(rule, 0)
                print(f"   {rule:22s} {frames:5d} frames  {boxes:5d} boxes")
            print(f"   {'iou_unknown (legacy)':22s} {'':5s}        {s.iou_unknown:5d} boxes")

    total_offending = sum(
        len(f) for s in grand.values() for f in s.rule_frames.values()
    )
    print(f"\nTotal offending frame-flags: {total_offending}")
    print(f"CSV report: {out_path}")


if __name__ == "__main__":
    main()
