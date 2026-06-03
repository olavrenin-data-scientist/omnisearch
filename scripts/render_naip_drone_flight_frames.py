"""Render a fixed-altitude drone flight as sequential NAIP camera crops."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.camera_renderer import DroneCameraRenderer
from detection.naip import fetch_naip_tiled_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-cache", default="data/terrain_cache/big_sur_128.npz")
    parser.add_argument("--out-dir", default="results/cv_demo_drone_flight_30m")
    parser.add_argument("--center-cell", nargs=2, type=int, default=(64, 64), metavar=("GX", "GY"))
    parser.add_argument("--altitude-agl-m", type=float, default=30.0)
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--patch-size-m", type=float, default=2000.0)
    parser.add_argument("--patch-image-size", type=int, default=8192)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--yolo-image-size", type=int, default=512)
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--flight-length-m", type=float, default=180.0)
    parser.add_argument("--flight-angle-deg", type=float, default=0.0)
    parser.add_argument("--survivor-position-m", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--survivor-width-m", type=float, default=2.4)
    parser.add_argument("--survivor-height-m", type=float, default=1.4)
    parser.add_argument("--survivor-rotation-deg", type=float, default=0.0)
    parser.add_argument(
        "--asset-resample",
        choices=("nearest", "bilinear"),
        default="nearest",
        help="Resize method for the pasted survivor asset.",
    )
    parser.add_argument("--human-asset", default="data/cv_assets/sard_grabcut/sard_survivor_0280.png")
    parser.add_argument("--human-assets-dir", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-naip-download", action="store_true")
    args = parser.parse_args()

    terrain_cache = Path(args.terrain_cache)
    if not terrain_cache.exists() and (ROOT / terrain_cache).exists():
        terrain_cache = ROOT / terrain_cache

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    frames_dir = out_dir / "frames"
    labels_dir = out_dir / "labels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    renderer = DroneCameraRenderer(terrain_cache=terrain_cache)
    if renderer.bbox_lonlat is None:
        raise SystemExit("Terrain cache does not contain bbox metadata; cannot fetch matching NAIP image.")

    center_cell = tuple(args.center_cell)
    center_world = renderer.cell_to_world(center_cell)
    center_lonlat = _world_to_lonlat(center_world, renderer.bbox_lonlat)
    patch_bbox_lonlat = _local_bbox_lonlat(center_lonlat, float(args.patch_size_m))
    patch_path = fetch_naip_tiled_image(
        bbox_lonlat=patch_bbox_lonlat,
        out_dir=ROOT / "data/source_cache/naip",
        size=int(args.patch_image_size),
        tile_size=int(args.tile_size),
        force=bool(args.force_naip_download),
    )

    patch = Image.open(patch_path).convert("RGB")
    patch_gsd_m_per_px = float(args.patch_size_m) / float(args.patch_image_size)
    yolo_size = int(args.yolo_image_size)
    altitude_m = float(args.altitude_agl_m)
    footprint_m = 2.0 * altitude_m * math.tan(math.radians(float(args.fov_deg)) / 2.0)
    source_crop_px = max(2, int(round(footprint_m / patch_gsd_m_per_px)))
    if source_crop_px > min(patch.size):
        raise SystemExit(
            f"Altitude {altitude_m:g} m needs {source_crop_px}px from the local patch; "
            f"increase --patch-size-m or --patch-image-size."
        )

    rng = random.Random(int(args.seed))
    human_assets = _load_human_assets(asset_path=args.human_asset, assets_dir=args.human_assets_dir)
    human_asset_path, human_asset = rng.choice(human_assets)
    survivor_m = (float(args.survivor_position_m[0]), float(args.survivor_position_m[1]))
    flight_positions = _flight_positions(
        frame_count=int(args.frame_count),
        length_m=float(args.flight_length_m),
        angle_deg=float(args.flight_angle_deg),
    )

    frames = []
    for index, drone_m in enumerate(flight_positions):
        crop = _crop_patch_by_meters(
            patch,
            center_m=drone_m,
            crop_size_px=source_crop_px,
            patch_size_m=float(args.patch_size_m),
        )
        yolo_crop = crop.resize((yolo_size, yolo_size), Image.Resampling.BILINEAR)
        label_lines: list[str] = []

        dx_m = survivor_m[0] - drone_m[0]
        dy_m = survivor_m[1] - drone_m[1]
        bbox = _survivor_box(
            dx_m=dx_m,
            dy_m=dy_m,
            footprint_m=footprint_m,
            image_size=yolo_size,
            width_m=float(args.survivor_width_m),
            height_m=float(args.survivor_height_m),
        )
        if bbox is not None:
            _paste_survivor_asset(
                yolo_crop,
                human_asset,
                bbox,
                rotation_deg=float(args.survivor_rotation_deg),
                brightness=0.92,
                contrast=1.08,
                resample=args.asset_resample,
            )
            label_lines.append(_to_yolo_line(class_id=0, bbox_xyxy=bbox, image_size=yolo_size))

        stem = f"frame_{index:04d}"
        image_path = frames_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"
        yolo_crop.save(image_path)
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        frames.append(
            {
                "index": index,
                "drone_position_m": [drone_m[0], drone_m[1]],
                "survivor_position_m": [survivor_m[0], survivor_m[1]],
                "survivor_relative_m": [dx_m, dy_m],
                "survivor_visible": bbox is not None,
                "survivor_bbox_xyxy": bbox,
                "image_path": str(image_path),
                "label_path": str(label_path),
            }
        )

    strip_path = out_dir / "flight_contact_strip.png"
    _write_contact_strip(frames, strip_path)
    metadata = {
        "terrain_cache": str(terrain_cache),
        "naip_patch_path": str(patch_path),
        "patch_bbox_lonlat": patch_bbox_lonlat,
        "patch_size_m": float(args.patch_size_m),
        "patch_image_size_px": int(args.patch_image_size),
        "patch_gsd_m_per_px": patch_gsd_m_per_px,
        "patch_gsd_cm_per_px": patch_gsd_m_per_px * 100.0,
        "center_cell": center_cell,
        "center_world_xy": center_world,
        "center_lonlat": center_lonlat,
        "altitude_agl_m": altitude_m,
        "fov_deg": float(args.fov_deg),
        "footprint_m": footprint_m,
        "source_crop_px": source_crop_px,
        "yolo_image_size_px": yolo_size,
        "yolo_resize_method": "bilinear",
        "human_asset": str(human_asset_path),
        "asset_resample": args.asset_resample,
        "comparison_path": str(strip_path),
        "frames": frames,
    }
    metadata_path = out_dir / "flight_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    visible_count = sum(1 for frame in frames if frame["survivor_visible"])
    print(f"Wrote {len(frames)} frames to {frames_dir}")
    print(f"Wrote labels to {labels_dir}")
    print(f"Wrote contact strip: {strip_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Survivor visible in {visible_count}/{len(frames)} frames")


def _flight_positions(*, frame_count: int, length_m: float, angle_deg: float) -> list[tuple[float, float]]:
    frame_count = max(1, int(frame_count))
    angle = math.radians(float(angle_deg))
    ux = math.cos(angle)
    uy = math.sin(angle)
    if frame_count == 1:
        distances = [0.0]
    else:
        distances = [
            -length_m * 0.5 + index * (length_m / (frame_count - 1))
            for index in range(frame_count)
        ]
    return [(distance * ux, distance * uy) for distance in distances]


def _crop_patch_by_meters(
    patch: Image.Image,
    *,
    center_m: tuple[float, float],
    crop_size_px: int,
    patch_size_m: float,
) -> Image.Image:
    cx = patch.width * 0.5 + (center_m[0] / patch_size_m) * patch.width
    cy = patch.height * 0.5 - (center_m[1] / patch_size_m) * patch.height
    half = crop_size_px * 0.5
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + crop_size_px
    bottom = top + crop_size_px
    return _crop_with_padding(patch, (left, top, right, bottom))


def _crop_with_padding(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    crop = Image.new("RGB", (width, height), (0, 0, 0))
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        return crop
    paste_xy = (source_box[0] - left, source_box[1] - top)
    crop.paste(image.crop(source_box), paste_xy)
    return crop


def _survivor_box(
    *,
    dx_m: float,
    dy_m: float,
    footprint_m: float,
    image_size: int,
    width_m: float,
    height_m: float,
) -> tuple[int, int, int, int] | None:
    half = footprint_m * 0.5
    if not (-half <= dx_m <= half and -half <= dy_m <= half):
        return None
    cx = image_size * 0.5 + (dx_m / footprint_m) * image_size
    cy = image_size * 0.5 - (dy_m / footprint_m) * image_size
    width_px = max(2, int(round(width_m / footprint_m * image_size)))
    height_px = max(2, int(round(height_m / footprint_m * image_size)))
    x1 = int(round(cx - width_px / 2))
    y1 = int(round(cy - height_px / 2))
    x2 = int(round(cx + width_px / 2))
    y2 = int(round(cy + height_px / 2))
    x1 = max(0, min(image_size - 1, x1))
    y1 = max(0, min(image_size - 1, y1))
    x2 = max(0, min(image_size - 1, x2))
    y2 = max(0, min(image_size - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _load_human_assets(
    *,
    asset_path: str | None,
    assets_dir: str | None,
) -> list[tuple[str, Image.Image]]:
    if assets_dir:
        directory = Path(assets_dir)
        if not directory.exists() and (ROOT / directory).exists():
            directory = ROOT / directory
        paths = sorted(directory.glob("*.png"))
        if not paths:
            raise SystemExit(f"No PNG human assets found in {directory}")
        return [(str(path), Image.open(path).convert("RGBA")) for path in paths]
    if asset_path:
        path = Path(asset_path)
        if not path.exists() and (ROOT / path).exists():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"Human asset not found: {asset_path}")
        return [(str(path), Image.open(path).convert("RGBA"))]
    raise SystemExit("Provide --human-asset or --human-assets-dir")


def _paste_survivor_asset(
    image: Image.Image,
    asset: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    rotation_deg: float,
    brightness: float,
    contrast: float,
    resample: str,
) -> None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    resample_filter = Image.Resampling.NEAREST if resample == "nearest" else Image.Resampling.BILINEAR
    sprite = asset.resize((width, height), resample_filter)
    rgb = sprite.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    sprite = Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))
    sprite = sprite.rotate(rotation_deg, expand=True, resample=resample_filter)

    px = x1 + width // 2 - sprite.width // 2
    py = y1 + height // 2 - sprite.height // 2
    image.paste(sprite, (px, py), sprite)


def _to_yolo_line(*, class_id: int, bbox_xyxy: tuple[int, int, int, int], image_size: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) * 0.5) / image_size
    cy = ((y1 + y2) * 0.5) / image_size
    width = max(x2 - x1, 0) / image_size
    height = max(y2 - y1, 0) / image_size
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def _write_contact_strip(frames: list[dict], path: Path) -> None:
    images = [Image.open(frame["image_path"]).convert("RGB") for frame in frames]
    if not images:
        return
    thumb_size = 256
    thumbs = []
    for image in images:
        thumb = image.copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.BILINEAR)
        tile = Image.new("RGB", (thumb_size, thumb_size), (255, 255, 255))
        tile.paste(thumb, ((thumb_size - thumb.width) // 2, (thumb_size - thumb.height) // 2))
        thumbs.append(tile)
    strip = Image.new("RGB", (thumb_size * len(thumbs), thumb_size), (255, 255, 255))
    for index, thumb in enumerate(thumbs):
        strip.paste(thumb, (index * thumb_size, 0))
    strip.save(path)


def _world_to_lonlat(
    world_xy: tuple[float, float],
    bbox_lonlat: tuple[float, float, float, float],
) -> tuple[float, float]:
    west, south, east, north = bbox_lonlat
    x, y = world_xy
    u = (x + 1.0) / 2.0
    v = (y + 1.0) / 2.0
    lon = west + u * (east - west)
    lat = south + v * (north - south)
    return float(lon), float(lat)


def _local_bbox_lonlat(center_lonlat: tuple[float, float], size_m: float) -> tuple[float, float, float, float]:
    lon, lat = center_lonlat
    half = size_m * 0.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * max(math.cos(math.radians(lat)), 1e-6)
    dlon = half / meters_per_deg_lon
    dlat = half / meters_per_deg_lat
    return float(lon - dlon), float(lat - dlat), float(lon + dlon), float(lat + dlat)


if __name__ == "__main__":
    main()
