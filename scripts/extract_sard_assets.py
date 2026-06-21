"""Extract reusable survivor PNG assets from a local SARD dataset checkout.

The extractor supports common object-detection layouts:

* YOLO labels: ``images/.../*.jpg`` with matching ``labels/.../*.txt``
* Pascal VOC XML files next to, or parallel to, the images
* COCO JSON annotation files

It creates feathered RGBA crops that can be pasted into NAIP drone crops with
``render_highres_naip_yolo_crops.py --human-assets-dir data/cv_assets/sard``.
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for better masks.
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sard-root",
        required=True,
        help="Local SARD dataset folder, for example data/source_cache/sard/SARD.",
    )
    parser.add_argument("--out-dir", default="data/cv_assets/sard")
    parser.add_argument("--max-assets", type=int, default=200)
    parser.add_argument("--padding", type=float, default=0.15, help="Padding as fraction of bbox size.")
    parser.add_argument("--min-box-px", type=int, default=12)
    parser.add_argument(
        "--mask-mode",
        choices=("rounded", "background", "grabcut"),
        default="grabcut",
        help="How to create alpha masks for extracted person crops.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=34.0,
        help="RGB distance threshold for --mask-mode background.",
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    sard_root = Path(args.sard_root)
    if not sard_root.exists() and (root / sard_root).exists():
        sard_root = root / sard_root
    if not sard_root.exists():
        raise SystemExit(f"SARD root not found: {sard_root}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in sard_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    coco_boxes = _load_coco_boxes(sard_root)
    rng = random.Random(int(args.seed))
    rng.shuffle(image_paths)

    manifest = []
    asset_index = 0
    for image_path in image_paths:
        if asset_index >= int(args.max_assets):
            break
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            continue
        boxes = list(_boxes_for_image(image_path, image.size, sard_root, coco_boxes))
        if not boxes:
            continue

        for box in boxes:
            if asset_index >= int(args.max_assets):
                break
            if min(box.x2 - box.x1, box.y2 - box.y1) < int(args.min_box_px):
                continue
            padded = _pad_box(box, image.size, float(args.padding))
            # After clamping to image bounds the padded box can become degenerate
            # (zero or negative area) when the annotation was near or outside an
            # image edge.  Skip rather than passing an invalid region to PIL.crop.
            if padded.x2 - padded.x1 < int(args.min_box_px) or padded.y2 - padded.y1 < int(args.min_box_px):
                continue
            crop = image.crop((padded.x1, padded.y1, padded.x2, padded.y2))
            asset = _to_rgba_asset(
                crop,
                mask_mode=args.mask_mode,
                background_threshold=float(args.background_threshold),
                foreground_box=(
                    box.x1 - padded.x1,
                    box.y1 - padded.y1,
                    box.x2 - padded.x1,
                    box.y2 - padded.y1,
                ),
            )
            asset_index += 1
            asset_path = out_dir / f"sard_survivor_{asset_index:04d}.png"
            asset.save(asset_path)
            manifest.append(
                {
                    "asset_path": str(asset_path),
                    "source_image": str(image_path),
                    "source_annotation": box.source,
                    "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                    "padded_bbox_xyxy": [padded.x1, padded.y1, padded.x2, padded.y2],
                    "mask_mode": args.mask_mode,
                }
            )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"assets": manifest}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} SARD assets to {out_dir}")
    print(f"Wrote manifest: {manifest_path}")


def _boxes_for_image(
    image_path: Path,
    image_size: tuple[int, int],
    sard_root: Path,
    coco_boxes: dict[str, list[Box]],
) -> Iterable[Box]:
    # YOLO boxes are already clamped inside _yolo_boxes_for_image.
    yield from _yolo_boxes_for_image(image_path, image_size, sard_root)
    # VOC/COCO boxes are NOT pre-clamped — clamp here so that _pad_box and
    # the GrabCut foreground-box coordinates are always within the image.
    for box in _voc_boxes_for_image(image_path, sard_root):
        yield _clamp_box(box, image_size)
    try:
        relative = str(image_path.relative_to(sard_root))
    except ValueError:
        relative = image_path.name
    for box in coco_boxes.get(image_path.name, []):
        yield _clamp_box(box, image_size)
    for box in coco_boxes.get(relative, []):
        yield _clamp_box(box, image_size)


def _yolo_boxes_for_image(image_path: Path, image_size: tuple[int, int], sard_root: Path) -> Iterable[Box]:
    width, height = image_size
    label_paths = _candidate_label_paths(image_path, sard_root, ".txt")
    for label_path in label_paths:
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                _class_id, cx, cy, bw, bh = (float(value) for value in parts[:5])
            except ValueError:
                continue
            x1 = int(round((cx - bw / 2.0) * width))
            y1 = int(round((cy - bh / 2.0) * height))
            x2 = int(round((cx + bw / 2.0) * width))
            y2 = int(round((cy + bh / 2.0) * height))
            yield _clamp_box(Box(x1, y1, x2, y2, str(label_path)), image_size)
        return


def _voc_boxes_for_image(image_path: Path, sard_root: Path) -> Iterable[Box]:
    for xml_path in _candidate_label_paths(image_path, sard_root, ".xml"):
        if not xml_path.exists():
            continue
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        for obj in tree.findall(".//object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                box = Box(
                    int(float(bnd.findtext("xmin", "0"))),
                    int(float(bnd.findtext("ymin", "0"))),
                    int(float(bnd.findtext("xmax", "0"))),
                    int(float(bnd.findtext("ymax", "0"))),
                    str(xml_path),
                )
            except ValueError:
                continue
            yield box
        return


def _load_coco_boxes(sard_root: Path) -> dict[str, list[Box]]:
    boxes: dict[str, list[Box]] = {}
    for json_path in sard_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            continue
        images = {image["id"]: image.get("file_name", "") for image in data.get("images", []) if "id" in image}
        for annotation in data.get("annotations", []):
            image_name = images.get(annotation.get("image_id"))
            bbox = annotation.get("bbox")
            if not image_name or not bbox or len(bbox) < 4:
                continue
            x, y, width, height = (float(value) for value in bbox[:4])
            box = Box(int(round(x)), int(round(y)), int(round(x + width)), int(round(y + height)), str(json_path))
            boxes.setdefault(Path(image_name).name, []).append(box)
            boxes.setdefault(image_name, []).append(box)
    return boxes


def _candidate_label_paths(image_path: Path, sard_root: Path, suffix: str) -> list[Path]:
    candidates = [image_path.with_suffix(suffix)]
    parts = list(image_path.parts)
    for token in ("images", "Images", "JPEGImages"):
        if token in parts:
            idx = parts.index(token)
            for label_dir in ("labels", "Labels", "annotations", "Annotations"):
                replaced = parts[:]
                replaced[idx] = label_dir
                candidates.append(Path(*replaced).with_suffix(suffix))
    candidates.extend(sard_root.rglob(f"{image_path.stem}{suffix}"))
    seen = set()
    unique = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _pad_box(box: Box, image_size: tuple[int, int], padding: float) -> Box:
    pad_x = int(round((box.x2 - box.x1) * padding))
    pad_y = int(round((box.y2 - box.y1) * padding))
    return _clamp_box(
        Box(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y, box.source),
        image_size,
    )


def _clamp_box(box: Box, image_size: tuple[int, int]) -> Box:
    width, height = image_size
    x1 = max(0, min(width - 1, box.x1))
    y1 = max(0, min(height - 1, box.y1))
    x2 = max(0, min(width, box.x2))
    y2 = max(0, min(height, box.y2))
    return Box(x1, y1, x2, y2, box.source)


def _to_rgba_asset(
    crop: Image.Image,
    *,
    mask_mode: str,
    background_threshold: float,
    foreground_box: tuple[int, int, int, int],
) -> Image.Image:
    if mask_mode == "rounded":
        return _rounded_rgba(crop)
    if mask_mode == "grabcut":
        return _grabcut_rgba(crop, foreground_box=foreground_box, fallback_threshold=background_threshold)
    return _background_mask_rgba(crop, threshold=background_threshold)


def _rounded_rgba(crop: Image.Image) -> Image.Image:
    rgba = crop.convert("RGBA")
    width, height = rgba.size
    mask = Image.new("L", rgba.size, 0)
    draw = ImageDraw.Draw(mask)
    radius = max(2, min(width, height) // 10)
    inset = max(1, min(width, height) // 20)
    draw.rounded_rectangle((inset, inset, width - inset, height - inset), radius=radius, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(width, height) // 40)))
    rgba.putalpha(mask)
    return rgba


def _background_mask_rgba(crop: Image.Image, *, threshold: float) -> Image.Image:
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    border = max(2, min(width, height) // 10)
    samples = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(samples, axis=0)
    diff = np.linalg.norm(rgb - background, axis=2)

    # Body prior: person is usually near the annotated box center. This prevents
    # isolated high-contrast background texture from becoming opaque.
    yy, xx = np.mgrid[:height, :width]
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    rx = max(width * 0.48, 1.0)
    ry = max(height * 0.48, 1.0)
    body_prior = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    mask = ((diff > float(threshold)) & body_prior).astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=3))
    pil_mask = pil_mask.filter(ImageFilter.MinFilter(size=3))
    pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(width, height) // 45)))

    alpha = np.asarray(pil_mask, dtype=np.float32)
    if alpha.max() < 16:
        return _rounded_rgba(crop)
    alpha = np.clip(alpha * 1.4, 0, 255).astype(np.uint8)
    rgba = crop.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def _grabcut_rgba(
    crop: Image.Image,
    *,
    foreground_box: tuple[int, int, int, int],
    fallback_threshold: float,
) -> Image.Image:
    if cv2 is None:
        return _background_mask_rgba(crop, threshold=fallback_threshold)

    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = foreground_box
    x1 = max(1, min(width - 2, int(x1)))
    y1 = max(1, min(height - 2, int(y1)))
    x2 = max(x1 + 2, min(width - 1, int(x2)))
    y2 = max(y1 + 2, min(height - 1, int(y2)))
    rect = (x1, y1, x2 - x1, y2 - y1)

    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)

    # Mark the padded border as definite background and the annotation box as
    # probable foreground. GrabCut then chooses the actual person pixels.
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _background_mask_rgba(crop, threshold=fallback_threshold)

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    foreground_area = int(np.count_nonzero(alpha))
    box_area = max(1, (x2 - x1) * (y2 - y1))
    if foreground_area < box_area * 0.12:
        return _background_mask_rgba(crop, threshold=fallback_threshold)

    pil_alpha = Image.fromarray(alpha)
    pil_alpha = pil_alpha.filter(ImageFilter.MaxFilter(size=3))
    pil_alpha = pil_alpha.filter(ImageFilter.GaussianBlur(radius=max(1, min(width, height) // 80)))
    rgba = crop.convert("RGBA")
    rgba.putalpha(pil_alpha)
    return rgba


if __name__ == "__main__":
    main()
