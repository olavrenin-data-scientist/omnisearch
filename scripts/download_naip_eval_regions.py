"""Download NAIP tiles from diverse geographic regions for cross-terrain CV evaluation.

Each region provides a different terrain type to test detector generalization:
  - Arizona desert: sparse vegetation, sandy/rocky terrain
  - Pacific NW forest: dense tree canopy, dark shadows
  - Texas grassland: open grass with scattered brush

Usage:
    python scripts/download_naip_eval_regions.py
    python scripts/download_naip_eval_regions.py --regions arizona texas
    python scripts/download_naip_eval_regions.py --tile-size 1024 --target-gsd 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.naip import NaipTileCache, estimate_bbox_size_m

REGIONS = {
    "arizona_desert": {
        "description": "Sonoran Desert, AZ — sparse vegetation, sandy/rocky terrain",
        "bbox_lonlat": (-111.95, 33.40, -111.94, 33.41),
    },
    "pnw_forest": {
        "description": "Olympic National Forest, WA — dense conifer canopy, shadows",
        "bbox_lonlat": (-123.80, 47.75, -123.79, 47.76),
    },
    "texas_grassland": {
        "description": "Texas Hill Country — open grassland with scattered brush",
        "bbox_lonlat": (-98.50, 30.25, -98.49, 30.26),
    },
}


def main():
    ap = argparse.ArgumentParser(description="Download NAIP tiles for cross-terrain CV evaluation")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS.keys()),
                    choices=list(REGIONS.keys()) + ["all"],
                    help="Which regions to download (default: all)")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "source_cache" / "naip"),
                    help="Output directory for tile caches")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--target-gsd", type=float, default=0.6,
                    help="Target ground sample distance in meters per pixel")
    args = ap.parse_args()

    if "all" in args.regions:
        args.regions = list(REGIONS.keys())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for region_name in args.regions:
        info = REGIONS[region_name]
        bbox = info["bbox_lonlat"]
        width_m, height_m = estimate_bbox_size_m(bbox)
        print(f"\n{'='*60}")
        print(f"  Region: {region_name}")
        print(f"  {info['description']}")
        print(f"  BBox: {bbox}")
        print(f"  Size: {width_m:.0f} x {height_m:.0f} m")
        print(f"  Target GSD: {args.target_gsd} m/px")
        print(f"{'='*60}")

        try:
            cache = NaipTileCache(
                bbox_lonlat=bbox,
                out_dir=out_dir,
                target_gsd_m=args.target_gsd,
                tile_size=args.tile_size,
                force=False,
            )
            print(f"  Cache dir: {cache.cache_dir}")
            print(f"  Grid: {cache.cols} x {cache.rows} tiles ({cache.cols * cache.rows} total)")
            print(f"  Image: {cache.width_px} x {cache.height_px} px")

            # Pre-fetch all tiles by accessing corners
            n_fetched = 0
            for row in range(cache.rows):
                for col in range(cache.cols):
                    tile_path = cache.cache_dir / f"tile_r{row:04d}_c{col:04d}.png"
                    if not tile_path.exists():
                        # Trigger fetch by requesting a crop that includes this tile
                        x_frac = (col + 0.5) / cache.cols
                        y_frac = (row + 0.5) / cache.rows
                        world_x = x_frac * 2.0 - 1.0
                        world_y = 1.0 - y_frac * 2.0
                        try:
                            cache.crop_world(
                                center_world=(world_x, world_y),
                                size_world=0.01,
                            )
                            n_fetched += 1
                        except Exception as e:
                            print(f"  WARNING: tile r{row} c{col} failed: {e}")
                    else:
                        n_fetched += 1

            existing = list(cache.cache_dir.glob("tile_*.png"))
            print(f"  Tiles downloaded: {len(existing)}/{cache.cols * cache.rows}")
            print(f"  Done: {cache.cache_dir}")

        except Exception as e:
            print(f"  ERROR downloading {region_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print("All regions processed.")
    print(f"Use --naip-val-dir <region_cache_dir> when training to evaluate on unseen terrain.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
