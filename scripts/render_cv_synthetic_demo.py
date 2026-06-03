"""Render a standalone synthetic drone camera crop for CV integration.

This script intentionally does not run the VMAS simulator. It validates the
geometry that a full CV pipeline needs:

    terrain cell -> drone crop pixel -> synthetic label -> mapped terrain cell

Example:
    python scripts/render_cv_synthetic_demo.py \
      --terrain-cache data/terrain_cache/big_sur_128.npz \
      --out-dir results/cv_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.camera_renderer import DroneCameraRenderer, FireCell, SmokeCell, SurvivorObject
from detection.naip import fetch_naip_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terrain-cache",
        default="data/terrain_cache/big_sur_128.npz",
        help="Terrain cache path, relative to omnisearch/ when run from that directory.",
    )
    parser.add_argument("--out-dir", default="results/cv_demo")
    parser.add_argument("--drone-cell", nargs=2, type=int, default=(64, 64), metavar=("GX", "GY"))
    parser.add_argument("--survivor-cell", nargs=2, type=int, default=(69, 67), metavar=("GX", "GY"))
    parser.add_argument(
        "--survivor-offset-m",
        nargs=2,
        type=float,
        default=None,
        metavar=("DX", "DY"),
        help="Place survivor this many meters east/north from the drone. Overrides --survivor-cell.",
    )
    parser.add_argument("--altitude-agl", type=float, default=0.18)
    parser.add_argument(
        "--altitude-agl-m",
        type=float,
        default=None,
        help="Drone altitude in meters AGL. Overrides --altitude-agl when terrain metadata has sim_units_per_meter.",
    )
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--global-image-size", type=int, default=2048)
    parser.add_argument(
        "--native-naip-resolution",
        action="store_true",
        help="Keep the camera crop at native NAIP source pixels instead of resizing to --image-size.",
    )
    parser.add_argument(
        "--use-naip",
        action="store_true",
        help="Use USGS NAIP Plus imagery as the global background instead of terrain colors.",
    )
    parser.add_argument(
        "--naip-cache-dir",
        default="data/source_cache/naip",
        help="Directory for cached NAIP background images.",
    )
    parser.add_argument("--force-naip-download", action="store_true")
    args = parser.parse_args()

    terrain_cache = Path(args.terrain_cache)
    if not terrain_cache.exists() and (ROOT / terrain_cache).exists():
        terrain_cache = ROOT / terrain_cache

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute() and args.out_dir == parser.get_default("out_dir"):
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    global_image_path = None
    if args.use_naip:
        probe = DroneCameraRenderer(
            terrain_cache=terrain_cache,
            global_image_size=args.global_image_size,
            camera_image_size=args.image_size,
        )
        if probe.bbox_lonlat is None:
            raise SystemExit("Terrain cache does not contain bbox metadata; cannot fetch matching NAIP image.")
        naip_cache = Path(args.naip_cache_dir)
        if not naip_cache.is_absolute():
            naip_cache = ROOT / naip_cache
        global_image_path = fetch_naip_image(
            bbox_lonlat=probe.bbox_lonlat,
            out_dir=naip_cache,
            size=args.global_image_size,
            force=args.force_naip_download,
        )

    renderer = DroneCameraRenderer(
        terrain_cache=terrain_cache,
        global_image_path=global_image_path,
        global_image_size=args.global_image_size,
        camera_image_size=args.image_size,
        resize_camera_crop=not args.native_naip_resolution,
    )
    altitude_agl = float(args.altitude_agl)
    sim_units_per_meter = _sim_units_per_meter(terrain_cache)
    if args.altitude_agl_m is not None:
        if sim_units_per_meter is None:
            raise SystemExit("Terrain cache does not contain sim_units_per_meter; cannot convert altitude meters.")
        altitude_agl = float(args.altitude_agl_m) * sim_units_per_meter

    drone_cell = tuple(args.drone_cell)
    survivor_cell = tuple(args.survivor_cell)
    survivor = SurvivorObject(survivor_cell)
    if args.survivor_offset_m is not None:
        if sim_units_per_meter is None:
            raise SystemExit("Terrain cache does not contain sim_units_per_meter; cannot convert survivor meters.")
        dx_m, dy_m = (float(v) for v in args.survivor_offset_m)
        drone_world = renderer.cell_to_world(drone_cell)
        survivor_world = (
            drone_world[0] + dx_m * sim_units_per_meter,
            drone_world[1] + dy_m * sim_units_per_meter,
        )
        survivor = SurvivorObject(world_xy=survivor_world)
    fire_cells = _nearby_cells((60, 61), radius=2, intensity=0.95)
    smoke_cells = [
        SmokeCell((gx + 2, gy + 1), density=max(0.2, 0.8 - 0.08 * idx))
        for idx, (gx, gy) in enumerate([(60, 61), (61, 61), (62, 62), (63, 63), (64, 64), (65, 65)])
    ]

    render = renderer.render(
        drone_cell=drone_cell,
        altitude_agl=altitude_agl,
        fov_deg=args.fov_deg,
        survivors=[survivor],
        fire_cells=fire_cells,
        smoke_cells=smoke_cells,
    )

    image_path = out_dir / "drone_crop.png"
    label_path = out_dir / "drone_crop.txt"
    metadata_path = out_dir / "drone_crop_metadata.json"

    render.image.save(image_path)
    renderer.save_yolo_labels(render, label_path)

    metadata = {
        "terrain_cache": str(terrain_cache),
        "global_image_path": str(global_image_path) if global_image_path is not None else None,
        "background": "naip" if global_image_path is not None else "terrain_land_cover_colors",
        "native_naip_resolution": bool(args.native_naip_resolution),
        "image_size_px": list(render.image.size),
        "source": renderer.source,
        "bbox_lonlat": renderer.bbox_lonlat,
        "drone_cell": render.drone_cell,
        "drone_world_xy": render.drone_world_xy,
        "altitude_agl": altitude_agl,
        "altitude_agl_m": float(args.altitude_agl_m) if args.altitude_agl_m is not None else None,
        "sim_units_per_meter": sim_units_per_meter,
        "requested_survivor_cell": survivor_cell,
        "survivor_offset_m": args.survivor_offset_m,
        "fov_deg": args.fov_deg,
        "crop_radius_world": render.crop_radius_world,
        "crop_world_bounds": render.crop_world_bounds,
        "detections": [
            {
                "class": det.class_name,
                "bbox_xyxy": det.bbox_xyxy,
                "source_cell": det.source_cell,
                "mapped_cell": det.mapped_cell,
                "cell_error_l1": abs(det.source_cell[0] - det.mapped_cell[0])
                + abs(det.source_cell[1] - det.mapped_cell[1]),
                "world_xy": det.world_xy,
            }
            for det in render.detections
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote image: {image_path}")
    print(f"Wrote labels: {label_path}")
    print(f"Wrote metadata: {metadata_path}")
    if render.detections:
        for det in render.detections:
            err = abs(det.source_cell[0] - det.mapped_cell[0]) + abs(det.source_cell[1] - det.mapped_cell[1])
            print(
                "mapping check:",
                f"true={det.source_cell}",
                f"mapped={det.mapped_cell}",
                f"l1_error={err}",
            )
    else:
        print("mapping check: no survivor visible in crop")


def _nearby_cells(center: tuple[int, int], *, radius: int, intensity: float) -> list[FireCell]:
    cx, cy = center
    cells = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                cells.append(FireCell((cx + dx, cy + dy), intensity=intensity))
    return cells


def _sim_units_per_meter(terrain_cache: Path) -> float | None:
    import numpy as np

    with np.load(terrain_cache, allow_pickle=False) as data:
        if "sim_units_per_meter" not in data:
            return None
        return float(np.asarray(data["sim_units_per_meter"]).item())


if __name__ == "__main__":
    main()
