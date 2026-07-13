#!/usr/bin/env python3
"""Convert the HERIDAL dataset (PASCAL VOC) to YOLO format for fine-tuning.

Source: Zenodo record 5662351 ("HERIDAL dataset in keras-retinanet PASCAL VOC
format"), extracted under data/source_cache/heridal/. Real drone photos
(4000x3000, ~50 m altitude) over Mediterranean wilderness with person boxes —
the closest public match to the OmniSearch drone deployment domain.

Output: data/cv_train/heridal/{train,val,test}/{images,labels} with class-0
person labels. Images are symlinked (not copied) to save 8 GB of disk.

Usage:
    python scripts/ingest_heridal.py
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/source_cache/heridal/extracted/heridal_keras_retinanet_voc"
DST = ROOT / "data/cv_train/heridal"

Image.MAX_IMAGE_PIXELS = None


def _voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    lines = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        if obj.findtext("name", "").lower() not in ("person", "human"):
            continue
        bb = obj.find("bndbox")
        x1, x2 = float(bb.findtext("xmin")), float(bb.findtext("xmax"))
        y1, y2 = float(bb.findtext("ymin")), float(bb.findtext("ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(img_w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(img_h), max(y1, y2))
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        lines.append(f"0 {(x1 + w / 2) / img_w:.6f} {(y1 + h / 2) / img_h:.6f} "
                     f"{w / img_w:.6f} {h / img_h:.6f}")
    return lines


def main() -> None:
    for split in ("train", "val", "test"):
        stems = {s.strip() for s in open(SRC / "ImageSets/Main" / f"{split}.txt") if s.strip()}
        img_out = DST / split / "images"
        lbl_out = DST / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for stem in sorted(stems):
            src_img = SRC / "JPEGImages" / f"{stem}.jpg"
            xml = SRC / "Annotations" / f"{stem}.xml"
            if not src_img.exists() or not xml.exists():
                continue
            with Image.open(src_img) as im:
                w, h = im.size
            lines = _voc_to_yolo(xml, w, h)
            link = img_out / src_img.name
            if not link.exists():
                link.symlink_to(src_img.resolve())
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_box += len(lines)
        print(f"{split}: {n_img} images, {n_box} person boxes")

    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "names:\n  0: person\n")
    print(f"Wrote {DST}/data.yaml")


if __name__ == "__main__":
    main()
