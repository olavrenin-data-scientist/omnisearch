"""Generate a synthetic Thermal Infrared (TIR) image dataset for survivor detection.

Produces YOLO-format labeled grayscale thermal images simulating what a drone-mounted
uncooled microbolometer (LWIR 8-14μm) would see during wildfire search-and-rescue.

Scenarios vary:
  - Clear terrain (easy: survivors are obvious bright blobs)
  - Active fire (hard: thermal crossover confuses survivor/ground contrast)
  - Burned/smoldering ground (medium: warm background reduces ΔT)
  - Smoke (easy for TIR: smoke is mostly transparent at LWIR wavelengths)
  - Multiple survivors at different distances from fire

Output structure (YOLO format):
  data/cv_train/thermal/
    train/images/00000.png   (grayscale 512×512)
    train/labels/00000.txt   (YOLO: class cx cy w h)
    val/images/...
    val/labels/...
    thermal.yaml             (YOLO dataset config)

Usage:
    python scripts/generate_thermal_dataset.py
    python scripts/generate_thermal_dataset.py --n-train 2000 --n-val 400
    python scripts/generate_thermal_dataset.py --colormap iron  # false-color output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.thermal_renderer import render_thermal_frame, render_thermal_with_colormap


DRONE_FLIGHT_LEVELS_M = (20.0, 35.0, 50.0)
DRONE_CAMERA_FOV_DEG = 65.0
# World coordinate convention (matches the simulator): the world spans [-1, 1]
# in both axes and represents a 100 m x 100 m area, so 1 world unit = 50 m.
WORLD_UNIT_M = 50.0


def _make_fire_grid(grid_size: int, rng: np.random.Generator, intensity: float = 0.0) -> np.ndarray:
    """Create a fire intensity grid with random hotspot placement."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if intensity <= 0:
        return grid
    n_hotspots = rng.integers(1, 5)
    for _ in range(n_hotspots):
        cx = rng.integers(10, grid_size - 10)
        cy = rng.integers(10, grid_size - 10)
        radius = rng.integers(3, 15)
        for r in range(max(0, cy - radius), min(grid_size, cy + radius)):
            for c in range(max(0, cx - radius), min(grid_size, cx + radius)):
                dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
                if dist < radius:
                    val = intensity * (1.0 - dist / radius) * rng.uniform(0.6, 1.0)
                    grid[r, c] = max(grid[r, c], val)
    return grid


def _make_burned_grid(grid_size: int, rng: np.random.Generator, coverage: float = 0.0) -> np.ndarray:
    """Create a burned ground grid (fraction of area that is burned)."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if coverage <= 0:
        return grid
    n_patches = rng.integers(2, 8)
    for _ in range(n_patches):
        cx = rng.integers(5, grid_size - 5)
        cy = rng.integers(5, grid_size - 5)
        radius = rng.integers(5, 20)
        for r in range(max(0, cy - radius), min(grid_size, cy + radius)):
            for c in range(max(0, cx - radius), min(grid_size, cx + radius)):
                dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
                if dist < radius:
                    val = coverage * (1.0 - dist / radius) * rng.uniform(0.5, 1.0)
                    grid[r, c] = max(grid[r, c], val)
    return grid


def _sample_scenario(rng: np.random.Generator) -> dict:
    """Sample a random scenario with fire/burn/smoke parameters."""
    scenario_type = rng.choice(["clear", "fire", "burned", "smoke", "fire+smoke", "burned+smoke"])
    params = {"fire_intensity": 0.0, "burn_coverage": 0.0, "smoke_load": 0.0}

    if scenario_type == "clear":
        pass
    elif scenario_type == "fire":
        params["fire_intensity"] = float(rng.uniform(0.3, 1.0))
    elif scenario_type == "burned":
        params["burn_coverage"] = float(rng.uniform(0.3, 0.9))
    elif scenario_type == "smoke":
        params["smoke_load"] = float(rng.uniform(0.2, 0.8))
    elif scenario_type == "fire+smoke":
        params["fire_intensity"] = float(rng.uniform(0.3, 1.0))
        params["smoke_load"] = float(rng.uniform(0.2, 0.7))
    elif scenario_type == "burned+smoke":
        params["burn_coverage"] = float(rng.uniform(0.3, 0.8))
        params["smoke_load"] = float(rng.uniform(0.1, 0.5))

    params["scenario"] = scenario_type
    return params


def _generate_thermal_split(
    out_dir: Path,
    n: int,
    rng: np.random.Generator,
    image_size: int = 512,
    grid_size: int = 100,
    neg_frac: float = 0.10,
    colormap: str | None = None,
) -> None:
    """Generate n thermal images with YOLO labels."""
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_neg = 0
    for i in range(n):
        altitude_m = float(rng.uniform(DRONE_FLIGHT_LEVELS_M[0], DRONE_FLIGHT_LEVELS_M[-1]))
        footprint_m = 2.0 * altitude_m * np.tan(np.radians(DRONE_CAMERA_FOV_DEG) / 2.0)
        # Footprint expressed in world units, consistent with drone/survivor
        # positions in [-1, 1].  Everything passed to the renderer must share
        # this unit or survivors collapse to the image center.
        footprint_world = footprint_m / WORLD_UNIT_M

        # Drone position (random within world bounds)
        drone_x = float(rng.uniform(-0.5, 0.5))
        drone_y = float(rng.uniform(-0.5, 0.5))

        # Scenario (fire/smoke/clear)
        scenario = _sample_scenario(rng)

        # Fire and burned grids
        fire_grid = _make_fire_grid(grid_size, rng, scenario["fire_intensity"])
        burned_grid = _make_burned_grid(grid_size, rng, scenario["burn_coverage"])

        # Survivors placement
        is_negative = rng.random() < neg_frac
        survivors = []
        labels = []

        if not is_negative:
            n_survivors = int(rng.integers(1, 4))
            for _ in range(n_survivors):
                # Place survivor anywhere within the camera footprint (uniform
                # across the frame, with a small margin so the blob fits).
                sx = drone_x + float(rng.uniform(-0.45, 0.45)) * footprint_world
                sy = drone_y + float(rng.uniform(-0.45, 0.45)) * footprint_world
                survivors.append({"world_xy": (sx, sy)})

                # Compute YOLO label (pixel position + normalized bbox)
                px = (sx - drone_x) / footprint_world * image_size + image_size / 2
                py = (sy - drone_y) / footprint_world * image_size + image_size / 2

                # Body radius scales with altitude (closer = larger blob)
                body_radius_px = max(4, int(12 * (20.0 / altitude_m)))
                bbox_size = body_radius_px * 3

                cx_n = px / image_size
                cy_n = py / image_size
                w_n = bbox_size / image_size
                h_n = bbox_size / image_size

                if 0.0 < cx_n < 1.0 and 0.0 < cy_n < 1.0:
                    labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
        else:
            n_neg += 1

        # Body radius for rendering
        body_radius_px = max(4, int(12 * (20.0 / altitude_m)))

        # Render thermal image
        img = render_thermal_frame(
            image_size=image_size,
            drone_xy=(drone_x, drone_y),
            footprint_world=footprint_world,
            survivors=survivors,
            fire_intensity_grid=fire_grid,
            burned_grid=burned_grid,
            grid_size=grid_size,
            noise_std=float(rng.uniform(1.0, 3.5)),
            body_radius_px=body_radius_px,
            seed=int(rng.integers(0, 2**31)),
        )

        # Optionally apply colormap
        if colormap and colormap != "grayscale":
            img = render_thermal_with_colormap(img, colormap=colormap)

        # Save image
        img.save(img_dir / f"{i:05d}.png")

        # Save label
        label_text = "\n".join(labels) + ("\n" if labels else "")
        (lbl_dir / f"{i:05d}.txt").write_text(label_text, encoding="utf-8")

        # Metadata sidecar
        meta = {
            "altitude_m": round(altitude_m, 1),
            "footprint_m": round(footprint_m, 1),
            "scenario": scenario["scenario"],
            "fire_intensity": round(scenario["fire_intensity"], 2),
            "burn_coverage": round(scenario["burn_coverage"], 2),
            "smoke_load": round(scenario["smoke_load"], 2),
            "n_survivors": len(survivors),
            "body_radius_px": body_radius_px,
        }
        (lbl_dir / f"{i:05d}.json").write_text(json.dumps(meta), encoding="utf-8")

    print(f"  {out_dir.name}: {n} images ({n_neg} negatives, {100*n_neg/max(1,n):.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic TIR training dataset")
    ap.add_argument("--n-train", type=int, default=1000, help="Number of training images")
    ap.add_argument("--n-val", type=int, default=200, help="Number of validation images")
    ap.add_argument("--image-size", type=int, default=512, help="Output image size (square)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-frac", type=float, default=0.10,
                    help="Fraction of images with no survivors (negatives)")
    ap.add_argument("--colormap", default=None, choices=["grayscale", "iron", "white_hot", "black_hot"],
                    help="Output colormap. Default: grayscale (raw thermal)")
    ap.add_argument("--data-dir", default=str(ROOT / "data/cv_train/thermal"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)

    print(f"Generating thermal TIR dataset in {data_dir}")
    print(f"  Image size: {args.image_size}×{args.image_size}")
    print(f"  Colormap: {args.colormap or 'grayscale'}")
    print(f"  Scenarios: clear, fire, burned, smoke, fire+smoke, burned+smoke")
    print()

    print("Generating training split...")
    _generate_thermal_split(
        data_dir / "train", args.n_train, rng,
        image_size=args.image_size, neg_frac=args.neg_frac, colormap=args.colormap,
    )

    print("Generating validation split...")
    _generate_thermal_split(
        data_dir / "val", args.n_val, rng,
        image_size=args.image_size, neg_frac=args.neg_frac, colormap=args.colormap,
    )

    # Write YOLO dataset config
    yaml_path = data_dir / "thermal.yaml"
    yaml_path.write_text(
        f"path: {data_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['person']\n",
        encoding="utf-8",
    )

    total = args.n_train + args.n_val
    print(f"\nDone! Generated {total} thermal images.")
    print(f"  Dataset config: {yaml_path}")
    print(f"  Train: {data_dir / 'train'}")
    print(f"  Val:   {data_dir / 'val'}")
    print(f"\nTo train a thermal detector:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=30 imgsz={args.image_size}")


if __name__ == "__main__":
    main()
