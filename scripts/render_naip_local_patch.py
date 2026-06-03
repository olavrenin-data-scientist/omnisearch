"""Render a high-resolution local NAIP patch for visual cross-checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.camera_renderer import DroneCameraRenderer
from detection.naip import fetch_naip_tiled_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-cache", default="data/terrain_cache/big_sur_128.npz")
    parser.add_argument("--out-dir", default="results/cv_demo_naip_local_patch")
    parser.add_argument("--drone-cell", nargs=2, type=int, default=(64, 64), metavar=("GX", "GY"))
    parser.add_argument("--patch-size-m", type=float, default=2000.0)
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=1024)
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
    bbox_lonlat = _local_bbox_lonlat(center_lonlat, float(args.patch_size_m))

    naip_path = fetch_naip_tiled_image(
        bbox_lonlat=bbox_lonlat,
        out_dir=ROOT / "data/source_cache/naip",
        size=int(args.image_size),
        tile_size=int(args.tile_size),
        force=bool(args.force_naip_download),
    )

    image = Image.open(naip_path).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    cx = annotated.width / 2
    cy = annotated.height / 2
    r = max(24, annotated.width * 0.018)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 230, 40), width=8)
    draw.line((cx - r * 1.6, cy, cx + r * 1.6, cy), fill=(255, 50, 50), width=5)
    draw.line((cx, cy - r * 1.6, cx, cy + r * 1.6), fill=(255, 50, 50), width=5)

    raw_out = out_dir / "naip_local_patch.png"
    annotated_out = out_dir / "naip_local_patch_annotated.png"
    metadata_out = out_dir / "naip_local_patch_metadata.json"
    image.save(raw_out)
    annotated.save(annotated_out)

    gsd_m_per_px = float(args.patch_size_m) / float(args.image_size)
    metadata = {
        "terrain_cache": str(terrain_cache),
        "source_image": str(naip_path),
        "drone_cell": drone_cell,
        "drone_world_xy": drone_world,
        "center_lonlat": center_lonlat,
        "bbox_lonlat": bbox_lonlat,
        "patch_size_m": float(args.patch_size_m),
        "image_size_px": int(args.image_size),
        "approx_gsd_m_per_px": gsd_m_per_px,
        "approx_gsd_cm_per_px": gsd_m_per_px * 100.0,
        "tile_size": int(args.tile_size),
    }
    metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote raw image: {raw_out}")
    print(f"Wrote annotated image: {annotated_out}")
    print(f"Wrote metadata: {metadata_out}")
    print(f"Approx GSD: {gsd_m_per_px:.3f} m/px")


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
