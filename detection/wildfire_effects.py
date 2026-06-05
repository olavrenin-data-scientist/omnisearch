"""Geospatially consistent wildfire image effects for NAIP-style imagery.

The functions in this module deliberately keep the image math separate from
the simulator.  They can be used in two modes:

* crop mode: render the simulator's active fire, burned, and smoke grids into
  a drone-view RGB crop;
* GeoTIFF mode: apply procedural wildfire masks to a 3- or 4-band raster while
  preserving CRS, affine transform, and profile metadata through rasterio.

The fourth NAIP band is near-infrared, not thermal infrared.  We therefore use
it for vegetation stress/burn consistency, not for true heat signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class WildfireEffectConfig:
    """Parameters controlling visual wildfire rendering."""

    burn_rgb_drop: float = 0.70
    burn_nir_drop: float = 0.90
    smoke_nir_drop: float = 0.22
    smoke_color_rgb: tuple[float, float, float] = (0.70, 0.65, 0.58)
    smoke_alpha: float = 0.62
    smoke_blur_px: float = 5.0
    smoke_noise_strength: float = 0.28
    flame_gain: float = 0.95
    flame_core_rgb: tuple[float, float, float] = (1.00, 0.72, 0.16)
    flame_edge_rgb: tuple[float, float, float] = (0.95, 0.22, 0.02)
    flame_bloom_px: float = 1.8
    flame_hotspot_noise: float = 0.42
    seed: int = 7


@dataclass(frozen=True)
class WildfireMasks:
    """Image-space wildfire masks in float32 [0, 1]."""

    burned: np.ndarray
    active: np.ndarray
    intensity: np.ndarray
    smoke: np.ndarray


def latlon_to_rowcol(dataset, lon: float, lat: float) -> tuple[int, int]:
    """Map WGS84 lon/lat to raster row/col using the raster CRS and transform."""

    import rasterio.transform
    from pyproj import Transformer

    if dataset.crs is None:
        raise ValueError("Raster has no CRS; cannot map lon/lat to pixels")
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    row, col = rasterio.transform.rowcol(dataset.transform, x, y)
    return int(row), int(col)


def masks_from_simulation_grids(
    *,
    image_size: tuple[int, int],
    center_world: tuple[float, float],
    footprint_world: float,
    fire_grid: np.ndarray | None,
    fire_intensity_grid: np.ndarray | None,
    burned_grid: np.ndarray | None,
    smoke_grid: np.ndarray | None,
    x_semidim: float = 1.0,
    y_semidim: float = 1.0,
) -> WildfireMasks:
    """Sample simulator fire/smoke grids into a drone crop's image space."""

    width, height = int(image_size[0]), int(image_size[1])
    shape = (height, width)
    active = np.zeros(shape, dtype=np.float32)
    intensity = np.zeros(shape, dtype=np.float32)
    burned = np.zeros(shape, dtype=np.float32)
    smoke = np.zeros(shape, dtype=np.float32)

    source_grid = _first_grid(fire_grid, fire_intensity_grid, burned_grid, smoke_grid)
    if source_grid is None:
        return WildfireMasks(burned=burned, active=active, intensity=intensity, smoke=smoke)

    grid_h, grid_w = source_grid.shape
    xs = np.linspace(
        center_world[0] - footprint_world * 0.5,
        center_world[0] + footprint_world * 0.5,
        width,
        dtype=np.float32,
    )
    ys = np.linspace(
        center_world[1] + footprint_world * 0.5,
        center_world[1] - footprint_world * 0.5,
        height,
        dtype=np.float32,
    )
    xx, yy = np.meshgrid(xs, ys)
    cols = np.floor((xx + x_semidim) / (2.0 * x_semidim) * grid_w).astype(np.int32)
    rows = np.floor((yy + y_semidim) / (2.0 * y_semidim) * grid_h).astype(np.int32)
    in_bounds = (cols >= 0) & (cols < grid_w) & (rows >= 0) & (rows < grid_h)
    rows_clipped = np.clip(rows, 0, grid_h - 1)
    cols_clipped = np.clip(cols, 0, grid_w - 1)

    if fire_grid is not None:
        active[in_bounds] = np.asarray(fire_grid, dtype=np.float32)[rows_clipped[in_bounds], cols_clipped[in_bounds]]
        active = (active > 0.0).astype(np.float32)
    if fire_intensity_grid is not None:
        intensity[in_bounds] = np.asarray(fire_intensity_grid, dtype=np.float32)[
            rows_clipped[in_bounds],
            cols_clipped[in_bounds],
        ]
    else:
        intensity = active.copy()
    if burned_grid is not None:
        burned[in_bounds] = np.asarray(burned_grid, dtype=np.float32)[rows_clipped[in_bounds], cols_clipped[in_bounds]]
        burned = (burned > 0.0).astype(np.float32)
    if smoke_grid is not None:
        smoke[in_bounds] = np.asarray(smoke_grid, dtype=np.float32)[rows_clipped[in_bounds], cols_clipped[in_bounds]]

    cell_px = max(width * (2.0 * x_semidim / grid_w) / max(footprint_world, 1e-6), 1.0)
    edge_sigma = min(max(cell_px * 0.14, 0.8), 5.0)
    return WildfireMasks(
        burned=_soft_grid_mask(burned, sigma=edge_sigma),
        active=_soft_grid_mask(active, sigma=edge_sigma),
        intensity=np.clip(intensity, 0.0, 1.0),
        smoke=np.clip(smoke, 0.0, 1.0),
    )


def apply_wildfire_effects(
    image: np.ndarray,
    masks: WildfireMasks,
    *,
    config: WildfireEffectConfig | None = None,
    include_burn: bool = True,
    include_flame: bool = True,
    include_smoke: bool = True,
) -> tuple[np.ndarray, dict]:
    """Apply wildfire effects to an RGB or RGB+NIR image array.

    The returned array has the same dtype family as the input.  All blending is
    performed in float32 [0, 1] to avoid uint8 clipping artifacts.
    """

    cfg = config or WildfireEffectConfig()
    original_dtype = image.dtype
    arr = _to_float_image(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("image must have shape HxWx3 or HxWx4")

    rgb = arr[..., :3]
    nir = arr[..., 3] if arr.shape[2] == 4 else None
    burned = _safe_mask(masks.burned, rgb.shape[:2])
    active = _safe_mask(masks.active, rgb.shape[:2])
    intensity = _safe_mask(masks.intensity, rgb.shape[:2])
    smoke = _safe_mask(masks.smoke, rgb.shape[:2])

    if include_burn:
        soft_burn = ndimage.gaussian_filter(np.maximum(burned, active * 0.45), sigma=1.2).clip(0.0, 1.0)
        rgb *= (1.0 - cfg.burn_rgb_drop * soft_burn[..., None]).clip(0.0, 1.0)
        if nir is not None:
            nir *= (1.0 - cfg.burn_nir_drop * soft_burn).clip(0.0, 1.0)

    if include_flame:
        flame = (active * intensity).clip(0.0, 1.0)
        if flame.max() > 0:
            edge = _perimeter_mask(active)
            hotspots = _hotspot_texture(active.shape, cfg)
            core_seed = flame * (0.78 * edge + cfg.flame_hotspot_noise * hotspots * active)
            core = ndimage.gaussian_filter(core_seed, sigma=0.65).clip(0.0, 1.0)
            bloom = ndimage.gaussian_filter(core, sigma=max(float(cfg.flame_bloom_px), 0.0)).clip(0.0, 1.0)
            edge_color = np.asarray(cfg.flame_edge_rgb, dtype=np.float32)
            core_color = np.asarray(cfg.flame_core_rgb, dtype=np.float32)
            rgb += bloom[..., None] * edge_color * (cfg.flame_gain * 0.55)
            rgb += core[..., None] * core_color * (cfg.flame_gain * 0.85)
            rgb[:] = np.clip(rgb, 0.0, 1.0)

    if include_smoke:
        smoke_alpha = _textured_smoke(smoke, cfg).clip(0.0, 1.0)
        if smoke_alpha.max() > 0:
            smoke_color = np.asarray(cfg.smoke_color_rgb, dtype=np.float32)
            luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
            desaturated = 0.45 * rgb + 0.55 * luminance
            rgb[:] = rgb * (1.0 - 0.35 * smoke_alpha[..., None]) + desaturated * (0.35 * smoke_alpha[..., None])
            rgb[:] = rgb * (1.0 - smoke_alpha[..., None]) + smoke_color * smoke_alpha[..., None]
            if nir is not None:
                nir *= (1.0 - cfg.smoke_nir_drop * smoke_alpha).clip(0.0, 1.0)

    arr[..., :3] = np.clip(rgb, 0.0, 1.0)
    if nir is not None:
        arr[..., 3] = np.clip(nir, 0.0, 1.0)

    stats = {
        "burned_mean": round(float(burned.mean()), 6),
        "active_mean": round(float(active.mean()), 6),
        "smoke_mean": round(float(smoke.mean()), 6),
        "max_fire_intensity": round(float((active * intensity).max(initial=0.0)), 6),
        "nir_updated": bool(nir is not None),
    }
    return _from_float_image(arr, original_dtype), stats


def apply_wildfire_effects_to_pil(
    image,
    masks: WildfireMasks,
    *,
    config: WildfireEffectConfig | None = None,
    include_burn: bool = True,
    include_flame: bool = True,
    include_smoke: bool = True,
):
    """Pillow convenience wrapper for RGB drone crops."""

    from PIL import Image

    rendered, stats = apply_wildfire_effects(
        np.asarray(image.convert("RGB")),
        masks,
        config=config,
        include_burn=include_burn,
        include_flame=include_flame,
        include_smoke=include_smoke,
    )
    return Image.fromarray(rendered, "RGB"), stats


def inject_wildfire_effects_geotiff(
    *,
    input_path: str | Path,
    output_path: str | Path,
    ignition_lonlat: tuple[float, float],
    radius_m: float = 60.0,
    wind_xy: tuple[float, float] = (1.0, 0.0),
    config: WildfireEffectConfig | None = None,
) -> Path:
    """Write a fire-augmented GeoTIFF while preserving CRS/transform metadata."""

    import rasterio

    in_path = Path(input_path)
    out_path = Path(output_path)
    cfg = config or WildfireEffectConfig()

    with rasterio.open(in_path) as src:
        if src.count < 3:
            raise ValueError("Expected at least RGB bands")
        row, col = latlon_to_rowcol(src, lon=ignition_lonlat[0], lat=ignition_lonlat[1])
        bands = min(src.count, 4)
        data = np.moveaxis(src.read(indexes=list(range(1, bands + 1))), 0, -1)
        px_m = _meters_per_pixel(src, row=row, col=col)
        masks = procedural_wildfire_masks(
            shape=data.shape[:2],
            center_rc=(row, col),
            radius_px=max(float(radius_m) / max(px_m, 1e-6), 1.0),
            wind_xy=wind_xy,
            seed=cfg.seed,
        )
        rendered, _stats = apply_wildfire_effects(data, masks, config=cfg)

        profile = src.profile.copy()
        profile.update(count=bands, dtype=rendered.dtype)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.moveaxis(rendered, -1, 0))
            dst.update_tags(
                wildfire_effects="procedural_rgb_nir",
                source_path=str(in_path),
                ignition_lon=str(float(ignition_lonlat[0])),
                ignition_lat=str(float(ignition_lonlat[1])),
            )
    return out_path


def procedural_wildfire_masks(
    *,
    shape: tuple[int, int],
    center_rc: tuple[int, int],
    radius_px: float,
    wind_xy: tuple[float, float] = (1.0, 0.0),
    seed: int = 7,
) -> WildfireMasks:
    """Generate an irregular burn scar, fire core, and downwind smoke plume."""

    height, width = int(shape[0]), int(shape[1])
    cy, cx = float(center_rc[0]), float(center_rc[1])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    distance = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)

    rng = np.random.default_rng(int(seed))
    coarse = rng.random((max(4, height // 32), max(4, width // 32)), dtype=np.float32)
    noise = ndimage.zoom(coarse, (height / coarse.shape[0], width / coarse.shape[1]), order=1)
    noise = noise[:height, :width]
    perimeter = float(radius_px) * (0.78 + 0.34 * noise + 0.08 * np.sin(5.0 * angle))
    burned = (distance <= perimeter).astype(np.float32)

    active_band = np.exp(-((distance - perimeter * 0.82) ** 2) / (2.0 * max(radius_px * 0.13, 1.0) ** 2))
    active = (active_band * burned).clip(0.0, 1.0)
    intensity = (0.35 + 0.65 * active_band).clip(0.0, 1.0)

    wind_x, wind_y = _normalize_wind(wind_xy)
    downwind = dx * wind_x + dy * wind_y
    crosswind = np.abs(-dx * wind_y + dy * wind_x)
    plume_len = max(radius_px * 6.0, 1.0)
    plume_width = radius_px * (0.55 + 1.75 * (downwind.clip(0.0) / plume_len))
    smoke = np.exp(-crosswind * crosswind / (2.0 * np.maximum(plume_width, 1.0) ** 2))
    smoke *= np.exp(-downwind.clip(0.0) / plume_len)
    smoke *= (downwind > 0).astype(np.float32)
    smoke += active * 0.45
    return WildfireMasks(
        burned=burned.clip(0.0, 1.0),
        active=active.clip(0.0, 1.0),
        intensity=intensity.clip(0.0, 1.0),
        smoke=smoke.clip(0.0, 1.0),
    )


def _first_grid(*grids: np.ndarray | None) -> np.ndarray | None:
    for grid in grids:
        if grid is not None:
            arr = np.asarray(grid)
            if arr.ndim != 2:
                raise ValueError("wildfire grids must be 2D")
            return arr
    return None


def _safe_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"mask shape {arr.shape} does not match image shape {shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _perimeter_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0.25
    if not binary.any():
        return np.zeros_like(mask, dtype=np.float32)
    eroded = ndimage.binary_erosion(binary, iterations=2, border_value=0)
    perimeter = (binary & ~eroded).astype(np.float32)
    return ndimage.gaussian_filter(perimeter, sigma=1.2).clip(0.0, 1.0)


def _soft_grid_mask(mask: np.ndarray, *, sigma: float) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)
    if arr.max(initial=0.0) <= 0.0:
        return arr
    return ndimage.gaussian_filter(arr, sigma=float(sigma)).clip(0.0, 1.0)


def _hotspot_texture(shape: tuple[int, int], cfg: WildfireEffectConfig) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.seed) + 211)
    coarse_shape = (max(4, shape[0] // 18), max(4, shape[1] // 18))
    coarse = rng.random(coarse_shape, dtype=np.float32)
    texture = ndimage.zoom(coarse, (shape[0] / coarse_shape[0], shape[1] / coarse_shape[1]), order=1)
    texture = texture[: shape[0], : shape[1]]
    texture = ndimage.gaussian_filter(texture, sigma=1.1)
    lo, hi = float(texture.min()), float(texture.max())
    if hi - lo <= 1e-6:
        return np.zeros(shape, dtype=np.float32)
    texture = (texture - lo) / (hi - lo)
    return np.clip((texture - 0.48) / 0.52, 0.0, 1.0)


def _textured_smoke(smoke: np.ndarray, cfg: WildfireEffectConfig) -> np.ndarray:
    if smoke.max(initial=0.0) <= 0:
        return np.zeros_like(smoke, dtype=np.float32)
    rng = np.random.default_rng(int(cfg.seed))
    coarse_shape = (max(4, smoke.shape[0] // 24), max(4, smoke.shape[1] // 24))
    coarse = rng.random(coarse_shape, dtype=np.float32)
    texture = ndimage.zoom(coarse, (smoke.shape[0] / coarse_shape[0], smoke.shape[1] / coarse_shape[1]), order=1)
    texture = texture[: smoke.shape[0], : smoke.shape[1]]
    texture = 1.0 + cfg.smoke_noise_strength * (texture - 0.5) * 2.0
    soft = ndimage.gaussian_filter(smoke * texture, sigma=max(float(cfg.smoke_blur_px), 0.0))
    return (soft * cfg.smoke_alpha).clip(0.0, 1.0)


def _to_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=True).clip(0.0, 1.0)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32) / 255.0


def _from_float_image(image: np.ndarray, dtype) -> np.ndarray:
    arr = np.clip(image, 0.0, 1.0)
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype)
    if dtype == np.uint16:
        return np.round(arr * 65535.0).astype(np.uint16)
    return np.round(arr * 255.0).astype(np.uint8)


def _normalize_wind(wind_xy: tuple[float, float]) -> tuple[float, float]:
    x, y = float(wind_xy[0]), float(wind_xy[1])
    mag = math.hypot(x, y)
    if mag <= 1e-9:
        return 1.0, 0.0
    return x / mag, y / mag


def _meters_per_pixel(dataset, *, row: int, col: int) -> float:
    from pyproj import Geod, Transformer

    row = int(np.clip(row, 0, dataset.height - 1))
    col = int(np.clip(col, 0, dataset.width - 1))
    x0, y0 = dataset.xy(row, col)
    x1, y1 = dataset.xy(min(row + 1, dataset.height - 1), col)
    x2, y2 = dataset.xy(row, min(col + 1, dataset.width - 1))
    if dataset.crs is not None and dataset.crs.is_geographic:
        geod = Geod(ellps="WGS84")
        _, _, dy = geod.inv(x0, y0, x1, y1)
        _, _, dx = geod.inv(x0, y0, x2, y2)
    elif dataset.crs is not None:
        transformer = Transformer.from_crs(dataset.crs, "EPSG:3857", always_xy=True)
        mx0, my0 = transformer.transform(x0, y0)
        mx1, my1 = transformer.transform(x1, y1)
        mx2, my2 = transformer.transform(x2, y2)
        dy = math.hypot(mx1 - mx0, my1 - my0)
        dx = math.hypot(mx2 - mx0, my2 - my0)
    else:
        dx = abs(float(dataset.transform.a))
        dy = abs(float(dataset.transform.e))
    values = [v for v in (dx, dy) if v > 0]
    return float(np.mean(values)) if values else 1.0
