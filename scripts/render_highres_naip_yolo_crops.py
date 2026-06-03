"""Render altitude-scaled YOLO crops from the highest practical local NAIP patch."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.camera_renderer import DroneCameraRenderer
from detection.naip import fetch_naip_tiled_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-cache", default="data/terrain_cache/big_sur_128.npz")
    parser.add_argument("--out-dir", default="results/cv_demo_highres_naip_yolo")
    parser.add_argument("--drone-cell", nargs=2, type=int, default=(64, 64), metavar=("GX", "GY"))
    parser.add_argument("--altitude-agl-m", nargs="+", type=float, default=(30.0, 100.0, 300.0))
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--patch-size-m", type=float, default=2000.0)
    parser.add_argument("--patch-image-size", type=int, default=8192)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--yolo-image-size", type=int, default=512)
    parser.add_argument("--survivor-offset-m", nargs=2, type=float, default=(8.0, 8.0), metavar=("DX", "DY"))
    parser.add_argument("--survivor-width-m", type=float, default=0.8)
    parser.add_argument("--survivor-height-m", type=float, default=1.8)
    parser.add_argument(
        "--human-asset",
        default=None,
        help="Optional transparent PNG survivor asset. If omitted, a simple Pillow demo asset is generated.",
    )
    parser.add_argument(
        "--human-assets-dir",
        default=None,
        help="Optional folder of transparent PNG survivor assets. Overrides --human-asset.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-naip-download", action="store_true")
    args = parser.parse_args()

    terrain_cache = Path(args.terrain_cache)
    if not terrain_cache.exists() and (ROOT / terrain_cache).exists():
        terrain_cache = ROOT / terrain_cache

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute() and args.out_dir == parser.get_default("out_dir"):
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = DroneCameraRenderer(terrain_cache=terrain_cache)
    if renderer.bbox_lonlat is None:
        raise SystemExit("Terrain cache does not contain bbox metadata; cannot fetch matching NAIP image.")

    drone_cell = tuple(args.drone_cell)
    drone_world = renderer.cell_to_world(drone_cell)
    center_lonlat = _world_to_lonlat(drone_world, renderer.bbox_lonlat)
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
    dx_m, dy_m = (float(v) for v in args.survivor_offset_m)
    rng = random.Random(int(args.seed))
    human_assets = _load_human_assets(asset_path=args.human_asset, assets_dir=args.human_assets_dir)
    outputs = []

    for altitude_m in args.altitude_agl_m:
        altitude_m = float(altitude_m)
        footprint_m = 2.0 * altitude_m * math.tan(math.radians(float(args.fov_deg)) / 2.0)
        source_crop_px = max(2, int(round(footprint_m / patch_gsd_m_per_px)))
        if source_crop_px > min(patch.size):
            raise SystemExit(
                f"Altitude {altitude_m:g} m needs {source_crop_px}px from the local patch; "
                f"increase --patch-size-m or --patch-image-size."
            )

        source_crop = _center_crop(patch, source_crop_px)
        yolo_crop = source_crop.resize((yolo_size, yolo_size), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(yolo_crop, "RGBA")

        bbox = _survivor_box(
            dx_m=dx_m,
            dy_m=dy_m,
            footprint_m=footprint_m,
            image_size=yolo_size,
            width_m=float(args.survivor_width_m),
            height_m=float(args.survivor_height_m),
        )
        label_lines: list[str] = []
        mapped_cell = None
        survivor_world = None
        if bbox is not None:
            human_asset_path, human_asset = rng.choice(human_assets)
            _paste_survivor_asset(yolo_crop, human_asset, bbox, rotation_deg=-18.0, brightness=0.92, contrast=1.08)
            label_lines.append(_to_yolo_line(class_id=0, bbox_xyxy=bbox, image_size=yolo_size))
            sim_units_per_meter = _sim_units_per_meter(terrain_cache)
            if sim_units_per_meter is not None:
                survivor_world = (
                    drone_world[0] + dx_m * sim_units_per_meter,
                    drone_world[1] + dy_m * sim_units_per_meter,
                )
                mapped_cell = renderer.world_to_cell(survivor_world)

        stem = f"drone_crop_{int(round(altitude_m))}m"
        image_path = out_dir / f"{stem}_512.png"
        label_path = out_dir / f"{stem}.txt"
        yolo_crop.save(image_path)
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        outputs.append(
            {
                "altitude_agl_m": altitude_m,
                "fov_deg": float(args.fov_deg),
                "footprint_m": footprint_m,
                "source_crop_px": source_crop_px,
                "source_gsd_m_per_px": patch_gsd_m_per_px,
                "yolo_image_size_px": yolo_size,
                "yolo_resize_method": "bilinear",
                "survivor_offset_m": [dx_m, dy_m],
                "survivor_bbox_xyxy": bbox,
                "survivor_renderer": "pillow_alpha_composite",
                "human_asset": str(human_asset_path),
                "survivor_world_xy": survivor_world,
                "mapped_cell": mapped_cell,
                "image_path": str(image_path),
                "label_path": str(label_path),
            }
        )

    comparison_path = out_dir / "altitude_yolo_comparison.png"
    _write_comparison(outputs, comparison_path)
    metadata = {
        "terrain_cache": str(terrain_cache),
        "naip_patch_path": str(patch_path),
        "patch_bbox_lonlat": patch_bbox_lonlat,
        "patch_size_m": float(args.patch_size_m),
        "patch_image_size_px": int(args.patch_image_size),
        "patch_gsd_m_per_px": patch_gsd_m_per_px,
        "patch_gsd_cm_per_px": patch_gsd_m_per_px * 100.0,
        "drone_cell": drone_cell,
        "drone_world_xy": drone_world,
        "center_lonlat": center_lonlat,
        "comparison_path": str(comparison_path),
        "crops": outputs,
    }
    metadata_path = out_dir / "highres_naip_yolo_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote metadata: {metadata_path}")
    for item in outputs:
        print(
            item["image_path"],
            f"altitude={item['altitude_agl_m']:g}m",
            f"footprint={item['footprint_m']:.1f}m",
            f"source_crop={item['source_crop_px']}px",
        )


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    cx = image.width // 2
    cy = image.height // 2
    half = size // 2
    left = cx - half
    top = cy - half
    return image.crop((left, top, left + size, top + size))


def _survivor_box(
    *,
    dx_m: float,
    dy_m: float,
    footprint_m: float,
    image_size: int,
    width_m: float,
    height_m: float,
) -> tuple[int, int, int, int] | None:
    cx = image_size * 0.5 + (dx_m / footprint_m) * image_size
    cy = image_size * 0.5 - (dy_m / footprint_m) * image_size
    if not (0.0 <= cx < image_size and 0.0 <= cy < image_size):
        return None
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
        path = asset_path
        asset_path = Path(path)
        if not asset_path.exists() and (ROOT / asset_path).exists():
            asset_path = ROOT / asset_path
        return [(str(asset_path), Image.open(asset_path).convert("RGBA"))]
    return [("generated_demo_asset", _generated_demo_human_asset())]


def _generated_demo_human_asset() -> Image.Image:
    asset = Image.new("RGBA", (80, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(asset, "RGBA")
    skin = (207, 154, 104, 255)
    shirt = (198, 52, 59, 255)
    pants = (45, 82, 136, 255)
    boots = (35, 30, 25, 255)
    outline = (20, 24, 28, 110)

    draw.ellipse((28, 5, 52, 29), fill=skin, outline=outline, width=2)
    draw.rounded_rectangle((23, 32, 57, 88), radius=11, fill=shirt, outline=outline, width=2)
    draw.rounded_rectangle((7, 42, 27, 112), radius=8, fill=shirt, outline=outline, width=2)
    draw.rounded_rectangle((53, 42, 73, 112), radius=8, fill=shirt, outline=outline, width=2)
    draw.rounded_rectangle((25, 84, 39, 158), radius=6, fill=pants, outline=outline, width=2)
    draw.rounded_rectangle((41, 84, 55, 158), radius=6, fill=pants, outline=outline, width=2)
    draw.rounded_rectangle((21, 153, 39, 174), radius=4, fill=boots)
    draw.rounded_rectangle((41, 153, 59, 174), radius=4, fill=boots)
    return asset


def _paste_survivor_asset(
    image: Image.Image,
    asset: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    rotation_deg: float,
    brightness: float,
    contrast: float,
) -> None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    sprite = asset.resize((width, height), Image.Resampling.BILINEAR)
    rgb = sprite.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    sprite = Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))
    sprite = sprite.rotate(rotation_deg, expand=True, resample=Image.Resampling.BILINEAR)

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


def _write_comparison(outputs: list[dict], path: Path) -> None:
    images = [Image.open(item["image_path"]).convert("RGB") for item in outputs]
    if not images:
        return
    width, height = images[0].size
    strip = Image.new("RGB", (width * len(images), height), (255, 255, 255))
    for idx, image in enumerate(images):
        strip.paste(image, (idx * width, 0))
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


def _sim_units_per_meter(terrain_cache: Path) -> float | None:
    import numpy as np

    with np.load(terrain_cache, allow_pickle=False) as data:
        if "sim_units_per_meter" not in data:
            return None
        return float(np.asarray(data["sim_units_per_meter"]).item())


if __name__ == "__main__":
    main()
