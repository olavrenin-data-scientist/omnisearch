"""Simulated thermal image renderer for visualization and training.

Generates grayscale thermal-look images from simulator state, where:
- Human body appears as a bright (hot) blob against cooler terrain
- Fire appears as intense white hotspots
- Burned ground appears as warm gray
- Smoke has minimal visual effect (TIR penetrates smoke)

These images can be used for:
1. Visualization in the web viewer (thermal overlay)
2. Future training of a thermal-specific detector (if real TIR data becomes available)
3. Comparison visualizations in reports
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def render_thermal_frame(
    *,
    image_size: int = 512,
    drone_xy: tuple[float, float],
    footprint_world: float,
    survivors: list[dict],
    fire_grid: np.ndarray | None = None,
    fire_intensity_grid: np.ndarray | None = None,
    burned_grid: np.ndarray | None = None,
    smoke_grid: np.ndarray | None = None,
    grid_size: int = 100,
    ambient_temp_k: float = 293.0,
    body_temp_k: float = 310.0,
    fire_temp_k: float = 600.0,
    burned_temp_k: float = 330.0,
    noise_std: float = 2.0,
    body_radius_px: int = 8,
    seed: int | None = None,
) -> Image.Image:
    """Render a simulated thermal infrared image.

    Returns a grayscale PIL Image where pixel intensity represents temperature.
    Brighter = hotter.

    Parameters
    ----------
    image_size : output image dimensions (square)
    drone_xy : drone world position (center of view)
    footprint_world : camera footprint in world units
    survivors : list of dicts with 'world_xy' keys
    fire_grid, fire_intensity_grid, burned_grid, smoke_grid : sim grids
    grid_size : simulation grid dimension
    ambient_temp_k : background temperature (no fire)
    body_temp_k : human body temperature
    fire_temp_k : active fire temperature
    burned_temp_k : recently burned ground temperature
    noise_std : sensor noise (temperature units)
    body_radius_px : survivor blob radius in pixels
    seed : random seed for noise
    """
    rng = np.random.default_rng(seed)

    # Start with ambient temperature background
    temp_map = np.full((image_size, image_size), ambient_temp_k, dtype=np.float32)

    # Add fire/burned ground heating from simulation grids (vectorized)
    if fire_intensity_grid is not None or burned_grid is not None:
        cols = np.arange(image_size, dtype=np.float32)
        rows = np.arange(image_size, dtype=np.float32)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # Map pixels to world coordinates
        wx = drone_xy[0] + (col_grid / image_size - 0.5) * footprint_world
        wy = drone_xy[1] + (row_grid / image_size - 0.5) * footprint_world

        # Map world to grid cells
        gc = np.clip(((wx + 1.0) * 0.5 * grid_size).astype(int), 0, grid_size - 1)
        gr = np.clip(((wy + 1.0) * 0.5 * grid_size).astype(int), 0, grid_size - 1)

        fire_vals = fire_intensity_grid[gr, gc] if fire_intensity_grid is not None else np.zeros_like(temp_map)
        burn_vals = burned_grid[gr, gc] if burned_grid is not None else np.zeros_like(temp_map)

        fire_temps = fire_vals * (fire_temp_k - ambient_temp_k)
        burn_temps = burn_vals * (burned_temp_k - ambient_temp_k)
        temp_map += np.maximum(fire_temps, burn_temps)

    # Add survivor heat signatures as Gaussian blobs (vectorized per survivor)
    sigma = body_radius_px * 0.7
    radius = body_radius_px * 2
    for survivor in survivors:
        sx, sy = survivor["world_xy"]
        px = int((sx - drone_xy[0]) / footprint_world * image_size + image_size / 2)
        py = int((sy - drone_xy[1]) / footprint_world * image_size + image_size / 2)

        if 0 <= px < image_size and 0 <= py < image_size:
            r_min = max(0, py - radius)
            r_max = min(image_size, py + radius + 1)
            c_min = max(0, px - radius)
            c_max = min(image_size, px + radius + 1)

            rr = np.arange(r_min, r_max) - py
            cc = np.arange(c_min, c_max) - px
            cc_grid, rr_grid = np.meshgrid(cc, rr)
            dist_sq = rr_grid ** 2 + cc_grid ** 2
            mask = dist_sq < (radius ** 2)
            intensity = np.exp(-dist_sq / (2 * sigma ** 2)) * mask
            body_temps = ambient_temp_k + (body_temp_k - ambient_temp_k) * intensity
            temp_map[r_min:r_max, c_min:c_max] = np.maximum(
                temp_map[r_min:r_max, c_min:c_max], body_temps
            )

    # Add sensor noise
    temp_map += rng.normal(0, noise_std, temp_map.shape).astype(np.float32)

    # Normalize to 0-255 grayscale using scene-adaptive contrast (AGC), like a
    # real thermal camera: stretch the actual scene temperature range so warm
    # targets are clearly visible even without fire in frame.  Percentiles
    # protect against a few extreme fire pixels crushing the rest to black.
    t_min = float(np.percentile(temp_map, 1.0))
    t_max = float(np.percentile(temp_map, 99.9))
    if t_max - t_min < 10.0:
        # Nearly uniform scene: keep a minimum span so noise isn't amplified
        # into full-range static.
        t_max = t_min + 10.0
    normalized = (temp_map - t_min) / (t_max - t_min)
    normalized = np.clip(normalized, 0, 1)
    pixel_values = (normalized * 255).astype(np.uint8)

    img = Image.fromarray(pixel_values)

    # Apply slight blur to simulate TIR sensor point spread function
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    return img


def render_thermal_with_colormap(
    thermal_gray: Image.Image,
    colormap: str = "iron",
) -> Image.Image:
    """Apply a false-color thermal colormap to a grayscale thermal image.

    Common thermal camera colormaps:
    - 'iron': black → blue → red → yellow → white (hot iron)
    - 'white_hot': black = cold, white = hot (standard thermal)
    - 'black_hot': white = cold, black = hot (inverted)
    """
    arr = np.array(thermal_gray, dtype=np.float32) / 255.0

    if colormap == "white_hot":
        return thermal_gray.convert("RGB")

    elif colormap == "black_hot":
        inverted = (255 - np.array(thermal_gray)).astype(np.uint8)
        return Image.fromarray(inverted, mode="L").convert("RGB")

    elif colormap == "iron":
        # Iron colormap: black → purple → red → orange → yellow → white
        rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
        # Red channel: rises early
        rgb[..., 0] = np.clip(arr * 3.0 * 255, 0, 255).astype(np.uint8)
        # Green channel: rises mid-range
        rgb[..., 1] = np.clip((arr - 0.33) * 3.0 * 255, 0, 255).astype(np.uint8)
        # Blue channel: only at very low temps (purple tint) and very high (white)
        blue = np.where(arr < 0.2, arr * 5.0 * 128, 0) + np.where(arr > 0.85, (arr - 0.85) * 6.67 * 255, 0)
        rgb[..., 2] = np.clip(blue, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    else:
        return thermal_gray.convert("RGB")
