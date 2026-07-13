"""Ingest the HIT-UAV real infrared-thermal drone dataset as a person-only YOLO set.

HIT-UAV (Suo et al., Scientific Data 2023, CC-BY-4.0):
  2,898 real thermal images from a UAV at 60-130 m altitude, 30-90 deg camera
  angle, day + night, with Person/Car/Bicycle/OtherVehicle/DontCare labels.
  https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset

This script copies the images, keeps only the Person class (id 0), drops all
vehicle/DontCare boxes, and writes per-image metadata JSON (altitude, camera
angle, day/night) parsed from the HIT-UAV filename convention:
    <daylight>_<altitude_m>_<camera_angle_deg>_<?>_<frame>.jpg
so our annotation/validation tooling can display capture conditions.

Usage:
    python scripts/ingest_hit_uav.py --src /tmp/hit_uav_repo \
        --dst data/cv_train/thermal_real
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PERSON_CLASS = 0


def ingest_split(src_root: Path, dst_root: Path, split: str) -> tuple[int, int, int]:
    img_src = src_root / "normal_json" / split
    lbl_src = src_root / "yolo_labels" / split
    img_dst = dst_root / split / "images"
    lbl_dst = dst_root / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = n_negatives = 0
    for lbl_file in sorted(lbl_src.glob("*.txt")):
        img_file = img_src / (lbl_file.stem + ".jpg")
        if not img_file.exists():
            continue
        person_lines = []
        for line in lbl_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5 and int(parts[0]) == PERSON_CLASS:
                person_lines.append("0 " + " ".join(parts[1:]))

        shutil.copy(img_file, img_dst / img_file.name)
        (lbl_dst / lbl_file.name).write_text("\n".join(person_lines) + ("\n" if person_lines else ""))

        # Filename convention: daylight_altitude_angle_?_frame
        meta: dict = {"source": "HIT-UAV", "modality": "thermal_real", "gsd_m": None}
        toks = lbl_file.stem.split("_")
        if len(toks) >= 3:
            try:
                meta["daylight"] = "night" if int(toks[0]) == 1 else "day"
                meta["altitude_m"] = float(toks[1])
                meta["camera_angle_deg"] = float(toks[2])
            except ValueError:
                pass
        (img_dst / (lbl_file.stem + ".json")).write_text(json.dumps(meta, indent=1))

        n_images += 1
        n_boxes += len(person_lines)
        if not person_lines:
            n_negatives += 1
    return n_images, n_boxes, n_negatives


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/hit_uav_repo")
    ap.add_argument("--dst", default="data/cv_train/thermal_real")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    for split in ("train", "val", "test"):
        n, b, neg = ingest_split(src, dst, split)
        print(f"{split}: {n} images, {b} person boxes, {neg} negatives (no person)")

    yaml = dst / "data.yaml"
    yaml.write_text(
        f"path: {dst.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n  0: person\n"
    )
    print(f"Wrote {yaml}")


if __name__ == "__main__":
    main()
