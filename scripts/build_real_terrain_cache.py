"""
Build an OmniSearch real-terrain cache from USGS 3DEP and OpenStreetMap.

Examples:

    python scripts/build_real_terrain_cache.py \
      --place "Malibu Creek State Park, California" \
      --grid-size 128

    python scripts/build_real_terrain_cache.py \
      --bbox -118.78 34.08 -118.68 34.16 \
      --grid-size 128
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terrain.usgs_osm_builder import build_real_terrain_cache


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Malibu Creek State Park, California")
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Use an explicit lon/lat bbox instead of geocoding --place.",
    )
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--cache-dir", default=str(ROOT / "data" / "terrain_cache"))
    p.add_argument("--out", default=None, help="Optional explicit .npz output path.")
    p.add_argument("--dem-resolution-m", type=int, default=10)
    p.add_argument("--road-width-m", type=float, default=8.0)
    p.add_argument("--building-height", type=float, default=0.10)
    p.add_argument("--osm-timeout", type=int, default=180)
    p.add_argument(
        "--fuel-source",
        choices=("derived", "landfire"),
        default="derived",
        help="Use derived fuel/vegetation fields or download LANDFIRE LFPS data.",
    )
    p.add_argument("--source-cache-dir", default=str(ROOT / "data" / "source_cache"))
    p.add_argument(
        "--landfire-layer-list",
        default="LF2025_FBFM40;LF2025_CC;LF2025_CH;LF2025_CBH;LF2025_CBD",
        help="LFPS layer list. Default uses LF2025 FBFM40 + canopy layers.",
    )
    p.add_argument(
        "--landfire-email",
        default=os.environ.get("LANDFIRE_EMAIL"),
        help="Required by LFPS when --fuel-source landfire. Can also use LANDFIRE_EMAIL.",
    )
    p.add_argument("--landfire-resample-resolution", type=int, default=31)
    p.add_argument("--landfire-output-projection", default=None)
    p.add_argument("--landfire-timeout-s", type=int, default=1800)
    p.add_argument("--landfire-poll-interval-s", type=float, default=10.0)
    p.add_argument(
        "--landfire-force-download",
        action="store_true",
        help="Ignore cached raw LFPS files and submit a fresh LANDFIRE job.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the final .npz even when a matching terrain cache exists.",
    )
    args = p.parse_args()

    if args.grid_size < 8:
        raise SystemExit("--grid-size must be at least 8")
    if args.dem_resolution_m <= 0:
        raise SystemExit("--dem-resolution-m must be positive")
    print(f"Place:          {args.place}")
    if args.bbox is not None:
        print(f"BBox:           {tuple(args.bbox)}")
    print(f"Grid:           {args.grid_size}x{args.grid_size}")
    print(f"DEM resolution: {args.dem_resolution_m} m")
    print(f"Fuel source:    {args.fuel_source}")
    print("-" * 60)

    try:
        path = build_real_terrain_cache(
            grid_size=args.grid_size,
            place=args.place,
            bbox=args.bbox,
            cache_dir=args.cache_dir,
            out=args.out,
            dem_resolution_m=args.dem_resolution_m,
            road_width_m=args.road_width_m,
            building_height=args.building_height,
            osm_timeout=args.osm_timeout,
            fuel_source=args.fuel_source,
            source_cache_dir=args.source_cache_dir,
            landfire_layer_list=args.landfire_layer_list,
            landfire_email=args.landfire_email,
            landfire_resample_resolution=args.landfire_resample_resolution,
            landfire_output_projection=args.landfire_output_projection,
            landfire_timeout_s=args.landfire_timeout_s,
            landfire_poll_interval_s=args.landfire_poll_interval_s,
            landfire_force_download=args.landfire_force_download,
            force_rebuild=args.force_rebuild,
        )
    except (ImportError, RuntimeError, TimeoutError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote: {_display_path(path)}")
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.exists():
        print(f"Metadata: {_display_path(metadata_path)}")
    print()
    print("Use it with:")
    print(
        "  python scripts/export_trajectories.py "
        f"--grid-size {args.grid_size} --terrain-source real "
        f"--terrain-cache-path {_display_path(path)}"
    )


if __name__ == "__main__":
    main()
