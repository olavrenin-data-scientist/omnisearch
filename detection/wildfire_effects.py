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

    burn_rgb_drop: float = 0.78
    burn_nir_drop: float = 0.90
    smoke_nir_drop: float = 0.22
    char_rgb: tuple[float, float, float] = (0.034, 0.031, 0.024)
    ember_rgb: tuple[float, float, float] = (0.70, 0.15, 0.08)
    smoke_color_rgb: tuple[float, float, float] = (0.47, 0.48, 0.43)
    haze_rgb: tuple[float, float, float] = (0.33, 0.32, 0.28)
    smoke_alpha: float = 0.36
    smoke_blur_px: float = 9.0
    smoke_noise_strength: float = 0.55
    flame_gain: float = 0.95
    flame_bloom_px: float = 4.0
    flame_hotspot_noise: float = 0.70
    heat_ramp_rgb: tuple[tuple[float, tuple[float, float, float]], ...] = (
        (0.00, (0.36, 0.07, 0.05)),
        (0.24, (0.58, 0.12, 0.06)),
        (0.48, (0.88, 0.28, 0.08)),
        (0.72, (1.00, 0.56, 0.16)),
        (0.90, (1.00, 0.80, 0.38)),
        (1.00, (1.00, 0.96, 0.76)),
    )
    seed: int = 7


@dataclass(frozen=True)
class WildfireMasks:
    """Image-space wildfire masks in float32 [0, 1]."""

    burned: np.ndarray
    active: np.ndarray
    intensity: np.ndarray
    smoke: np.ndarray
    cell_px: float = 1.0
    front: np.ndarray | None = None
    interior_active: np.ndarray | None = None


@dataclass(frozen=True)
class _SubcellFireFields:
    """Internal high-resolution-looking fire fields derived from coarse grids."""

    stress: np.ndarray
    char: np.ndarray
    scorch: np.ndarray
    flame_alpha: np.ndarray
    hot_core: np.ndarray
    white_core: np.ndarray
    ember_alpha: np.ndarray
    smoke_alpha: np.ndarray
    warm_smoke: np.ndarray


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
    front = np.zeros(shape, dtype=np.float32)
    interior_active = np.zeros(shape, dtype=np.float32)
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

    fire_source = np.zeros((grid_h, grid_w), dtype=bool)
    if fire_grid is not None:
        fire_source = np.asarray(fire_grid) > 0
        active[in_bounds] = fire_source[rows_clipped[in_bounds], cols_clipped[in_bounds]]
    if fire_intensity_grid is not None:
        intensity[in_bounds] = np.asarray(fire_intensity_grid, dtype=np.float32)[
            rows_clipped[in_bounds],
            cols_clipped[in_bounds],
        ]
    else:
        intensity = active.copy()
    burned_source = fire_source.copy()
    if burned_grid is not None:
        burned_source |= np.asarray(burned_grid) > 0
    burned_out_source = burned_source & ~fire_source
    adjacent_unburned = ndimage.binary_dilation(
        ~burned_source,
        structure=np.ones((3, 3), dtype=bool),
    )
    front_source = fire_source & adjacent_unburned
    interior_source = fire_source & ~front_source
    burned[in_bounds] = burned_out_source[rows_clipped[in_bounds], cols_clipped[in_bounds]]
    front[in_bounds] = front_source[rows_clipped[in_bounds], cols_clipped[in_bounds]]
    interior_active[in_bounds] = interior_source[rows_clipped[in_bounds], cols_clipped[in_bounds]]
    if smoke_grid is not None:
        smoke[in_bounds] = np.asarray(smoke_grid, dtype=np.float32)[rows_clipped[in_bounds], cols_clipped[in_bounds]]

    cell_px = max(width * (2.0 * x_semidim / grid_w) / max(footprint_world, 1e-6), 1.0)
    edge_sigma = min(max(cell_px * 0.14, 0.8), 5.0)
    active_soft = _soft_grid_mask(active, sigma=edge_sigma)
    front_soft = _soft_grid_mask(front, sigma=edge_sigma)
    interior_soft = _soft_grid_mask(interior_active, sigma=edge_sigma)
    intensity_soft = _soft_grid_mask(intensity, sigma=edge_sigma)
    if intensity_soft.max(initial=0.0) <= 0.0 and active_soft.max(initial=0.0) > 0.0:
        intensity_soft = active_soft.copy()
    smoke_sigma = min(max(edge_sigma * 2.8, 2.0), 10.0)
    return WildfireMasks(
        burned=_soft_grid_mask(burned, sigma=edge_sigma),
        active=active_soft,
        intensity=intensity_soft.clip(0.0, 1.0),
        smoke=_soft_grid_mask(smoke, sigma=smoke_sigma),
        cell_px=float(cell_px),
        front=front_soft,
        interior_active=interior_soft,
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
    front = (
        _safe_mask(masks.front, rgb.shape[:2])
        if masks.front is not None
        else active
    )
    interior_active = (
        _safe_mask(masks.interior_active, rgb.shape[:2])
        if masks.interior_active is not None
        else np.zeros_like(active)
    )
    cell_px = max(float(getattr(masks, "cell_px", 1.0)), 1.0)

    if include_burn or include_flame or include_smoke:
        fields = _build_subcell_fire_fields(
            burned=burned,
            active=active,
            front=front,
            interior_active=interior_active,
            intensity=intensity,
            smoke=smoke,
            cfg=cfg,
            cell_px=cell_px,
        )

    if include_burn:
        luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
        desaturated = 0.38 * rgb + 0.62 * luminance
        stress_alpha = (0.38 * fields.stress).clip(0.0, 0.56)
        rgb[:] = rgb * (1.0 - stress_alpha[..., None]) + desaturated * stress_alpha[..., None]
        rgb[:] = _blend_rgb(rgb, (0.25, 0.15, 0.08), (0.18 * fields.scorch).clip(0.0, 0.30))
        rgb[:] = _blend_rgb(rgb, cfg.char_rgb, (cfg.burn_rgb_drop * np.sqrt(fields.char)).clip(0.0, 0.90))
        rgb[:] = _blend_rgb(rgb, (0.018, 0.016, 0.012), (0.30 * fields.char ** 1.18).clip(0.0, 0.40))
        if nir is not None:
            nir *= (1.0 - cfg.burn_nir_drop * np.maximum(fields.char, 0.62 * fields.scorch)).clip(0.0, 1.0)

    if include_flame:
        if fields.flame_alpha.max() > 0 or fields.hot_core.max() > 0 or fields.ember_alpha.max() > 0:
            flame_alpha = (fields.flame_alpha * cfg.flame_gain).clip(0.0, 1.0)
            hot_core = fields.hot_core
            white_core = fields.white_core
            heat_color = np.zeros_like(rgb, dtype=np.float32)
            heat_color[..., 0] = 0.70 + 0.30 * hot_core
            heat_color[..., 1] = 0.23 + 0.55 * hot_core
            heat_color[..., 2] = 0.07 + 0.12 * hot_core
            rgb[:] = _blend_rgb(rgb, heat_color, (0.42 * flame_alpha).clip(0.0, 0.68))
            rgb[:] = (rgb + heat_color * (0.18 * flame_alpha)[..., None]).clip(0.0, 1.0)
            rgb[:] = _blend_rgb(rgb, (1.0, 0.78, 0.28), (0.40 * hot_core).clip(0.0, 0.58))
            rgb[:] = _blend_rgb(rgb, (1.0, 0.98, 0.72), (0.55 * white_core).clip(0.0, 0.62))
            glow_alpha = ndimage.gaussian_filter(flame_alpha, sigma=max(cell_px * 0.10, 1.0))
            rgb[:] = _blend_rgb(rgb, (0.88, 0.34, 0.10), (0.12 * glow_alpha).clip(0.0, 0.18))
            rgb[:] = _blend_rgb(rgb, cfg.ember_rgb, fields.ember_alpha)
            rgb[:] = np.clip(rgb, 0.0, 1.0)

    if include_smoke:
        smoke_alpha = fields.smoke_alpha
        if smoke_alpha.max() > 0:
            heat_visibility = ndimage.gaussian_filter(fields.hot_core, sigma=max(cell_px * 0.16, 1.4)).clip(0.0, 1.0)
            smoke_occlusion = (smoke_alpha * (1.0 - 0.36 * heat_visibility)).clip(0.0, 1.0)
            luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
            desaturated = 0.28 * rgb + 0.72 * luminance
            contrast_alpha = (0.38 * smoke_occlusion).clip(0.0, 0.56)
            rgb[:] = rgb * (1.0 - contrast_alpha[..., None]) + desaturated * contrast_alpha[..., None]
            warm_smoke = fields.warm_smoke
            smoke_color = _color_ramp(
                (
                    (0.0, cfg.smoke_color_rgb),
                    (1.0, (0.54, 0.42, 0.30)),
                ),
                warm_smoke,
            )
            rgb[:] = _blend_rgb(rgb, smoke_color, smoke_occlusion * 0.78)
            haze_alpha = (smoke_occlusion * 0.12).clip(0.0, 0.18)
            rgb[:] = _blend_rgb(rgb, cfg.haze_rgb, haze_alpha)
            through_glow = (warm_smoke * smoke_alpha * 0.13).clip(0.0, 0.16)
            rgb[:] = _blend_rgb(rgb, (0.92, 0.38, 0.12), through_glow)
            if nir is not None:
                nir *= (1.0 - cfg.smoke_nir_drop * smoke_occlusion).clip(0.0, 1.0)

    if (include_burn or include_flame or include_smoke) and (
        burned.max(initial=0.0) > 0.0 or active.max(initial=0.0) > 0.0 or smoke.max(initial=0.0) > 0.0
    ):
        rgb[:] = ndimage.gaussian_filter(rgb, sigma=(0.28, 0.28, 0.0))

    arr[..., :3] = np.clip(rgb, 0.0, 1.0)
    if nir is not None:
        arr[..., 3] = np.clip(nir, 0.0, 1.0)

    stats = {
        "burned_mean": round(float(burned.mean()), 6),
        "active_mean": round(float(active.mean()), 6),
        "front_mean": round(float(front.mean()), 6),
        "interior_active_mean": round(float(interior_active.mean()), 6),
        "smoke_mean": round(float(smoke.mean()), 6),
        "max_fire_intensity": round(float((active * intensity).max(initial=0.0)), 6),
        "fire_cell_px": round(float(cell_px), 3),
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
    wind_x, wind_y = _normalize_wind(wind_xy)
    perimeter = float(radius_px) * (0.78 + 0.34 * noise + 0.08 * np.sin(5.0 * angle))
    burned = (distance <= perimeter).astype(np.float32)

    active_band = np.exp(-((distance - perimeter * 0.82) ** 2) / (2.0 * max(radius_px * 0.13, 1.0) ** 2))
    radial_x = dx / np.maximum(distance, 1.0)
    radial_y = dy / np.maximum(distance, 1.0)
    wind_front = (radial_x * wind_x + radial_y * wind_y).clip(-1.0, 1.0)
    arc_noise = _smooth_noise((height, width), grid=max(int(radius_px * 0.45), 7), seed=seed + 71)
    fine_noise = _smooth_noise((height, width), grid=max(int(radius_px * 0.12), 3), seed=seed + 109)
    angular_breaks = 0.5 + 0.5 * np.sin(3.0 * angle + 0.013 * seed)
    front_gate = (
        0.46 * arc_noise
        + 0.25 * fine_noise
        + 0.18 * angular_breaks
        + 0.34 * ((wind_front + 0.18) / 1.18).clip(0.0, 1.0)
        - 0.42
    )
    front_gate = (front_gate / 0.42).clip(0.0, 1.0)
    active = (active_band * burned * front_gate).clip(0.0, 1.0)
    intensity = (0.25 + 0.75 * active).clip(0.0, 1.0)

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


def _build_subcell_fire_fields(
    *,
    burned: np.ndarray,
    active: np.ndarray,
    front: np.ndarray,
    interior_active: np.ndarray,
    intensity: np.ndarray,
    smoke: np.ndarray,
    cfg: WildfireEffectConfig,
    cell_px: float,
) -> _SubcellFireFields:
    """Convert coarse simulator masks into fragmented fire-render fields.

    The simulator grid remains the physics source of truth.  This function only
    changes the image representation: outer-front cells create visible heat,
    while interior-active cells become charred ground with sparse embers.
    """

    shape = active.shape
    zeros = np.zeros(shape, dtype=np.float32)
    cell_px = max(float(cell_px), 1.0)
    heat_seed = (front * intensity).clip(0.0, 1.0)
    interior_heat = (interior_active * intensity).clip(0.0, 1.0)

    large = _smooth_noise_px(shape, scale_px=max(cell_px * 0.90, 10.0), seed=cfg.seed + 301)
    medium = _smooth_noise_px(shape, scale_px=max(cell_px * 0.30, 5.0), seed=cfg.seed + 307)
    fine = _smooth_noise_px(shape, scale_px=max(cell_px * 0.09, 2.5), seed=cfg.seed + 313, order=1)
    micro = _smooth_noise_px(shape, scale_px=max(cell_px * 0.045, 2.0), seed=cfg.seed + 317, order=1)

    if heat_seed.max(initial=0.0) > 0.0:
        affected_soft = ndimage.gaussian_filter(
            np.maximum(burned, active),
            sigma=max(cell_px * 0.22, 1.8),
        )
        outside_edge = ndimage.gaussian_gradient_magnitude(
            affected_soft,
            sigma=max(cell_px * 0.08, 0.75),
        )
        outside_edge = _normalize_mask(outside_edge)
        front_support = ndimage.gaussian_filter(
            front,
            sigma=max(cell_px * 0.18, 1.4),
        )
        front_support = _normalize_mask(front_support)
        line_gate = ((0.28 * large + 0.34 * medium + 0.38 * fine - 0.38) / 0.44).clip(0.0, 1.0)
        line_gate = ndimage.gaussian_filter(line_gate, sigma=0.35)
        front_seed = (
            outside_edge
            * front_support
            * intensity
            * (0.62 + 0.38 * line_gate)
        ).clip(0.0, 1.0)
        if front_seed.max(initial=0.0) <= 1e-6:
            front_seed = (heat_seed * line_gate).clip(0.0, 1.0)

        fragment = ((0.30 * large + 0.36 * medium + 0.34 * fine - 0.36) / 0.48).clip(0.0, 1.0)
        fragment = ndimage.gaussian_filter(fragment, sigma=0.45)
        heat_near = ndimage.gaussian_filter(front_seed, sigma=max(cell_px * 0.075, 0.90))
        heat_bloom = ndimage.gaussian_filter(front_seed, sigma=max(cell_px * 0.20, 2.2))
        heat = _normalize_mask(1.18 * heat_near + 0.22 * heat_bloom)
        heat = (heat * (0.74 + 0.26 * medium)).clip(0.0, 1.0)

        ridge = (heat - 0.54 * ndimage.gaussian_filter(heat, sigma=max(cell_px * 0.14, 1.5))).clip(0.0, 1.0)
        ridge = _normalize_mask(ridge)
        broken_front = (0.72 * ridge + 0.28 * heat) * (0.64 + 0.36 * fragment)
        flame_alpha = ((broken_front - 0.075) / 0.40).clip(0.0, 1.0)
        flame_alpha = ndimage.gaussian_filter(flame_alpha, sigma=max(cell_px * 0.045, 0.65)).clip(0.0, 1.0)

        hot_core = ((0.76 * ridge + 0.24 * heat) * (0.20 + 0.80 * (fine ** 2.35)) - 0.38) / 0.34
        hot_core = hot_core.clip(0.0, 1.0)
        hot_core = ndimage.gaussian_filter(hot_core, sigma=max(cell_px * 0.014, 0.25)).clip(0.0, 1.0)
        white_core = ((hot_core * (micro ** 1.9) - 0.56) / 0.34).clip(0.0, 1.0)
        white_core = ndimage.gaussian_filter(white_core, sigma=0.35).clip(0.0, 1.0)
    else:
        heat_bloom = zeros
        flame_alpha = zeros
        hot_core = zeros
        white_core = zeros

    active_scorch_source = np.maximum(heat_seed, 0.62 * interior_heat)
    scorch = _normalize_mask(
        ndimage.gaussian_filter(active_scorch_source, sigma=max(cell_px * 0.42, 3.4))
    )
    burn_source = np.maximum(burned, 0.78 * interior_active)
    burned_soft = ndimage.gaussian_filter(burn_source, sigma=max(cell_px * 0.20, 1.5))
    stress = ((0.58 * burned_soft + 0.42 * scorch) * (0.74 + 0.26 * large)).clip(0.0, 1.0)
    char = ((0.68 * burned_soft + 0.26 * scorch) * (0.68 + 0.32 * medium)).clip(0.0, 1.0)
    char = np.maximum(
        char,
        0.68 * ndimage.gaussian_filter(burn_source, sigma=max(cell_px * 0.32, 3.5)),
    ).clip(0.0, 1.0)

    ember_gate = ((fine * micro - 0.68) / 0.32).clip(0.0, 1.0)
    ember_alpha = (
        interior_heat
        * (ember_gate ** 2.2)
        * (0.06 + 0.14 * medium)
    ).clip(0.0, 0.16)
    ember_alpha = ndimage.gaussian_filter(
        ember_alpha,
        sigma=max(cell_px * 0.012, 0.22),
    ).clip(0.0, 0.16)

    # Smoke opacity comes from the simulator smoke field only. Burn scars and
    # active cells must not create a second dark haze layer on their own.
    smoke_source = smoke
    if smoke_source.max(initial=0.0) > 0.0:
        smoke_alpha = ndimage.gaussian_filter(
            smoke_source * (0.78 + 0.46 * (large - 0.5)),
            sigma=max(cell_px * 0.28, 2.8),
        )
        smoke_alpha = _normalize_mask(smoke_alpha)
        smoke_alpha = (cfg.smoke_alpha * 0.84 * smoke_alpha).clip(0.0, min(cfg.smoke_alpha, 0.38))
    else:
        smoke_alpha = zeros

    warm_smoke = ndimage.gaussian_filter(flame_alpha, sigma=max(cell_px * 0.32, 3.2)).clip(0.0, 1.0)
    return _SubcellFireFields(
        stress=stress.astype(np.float32),
        char=char.astype(np.float32),
        scorch=scorch.astype(np.float32),
        flame_alpha=flame_alpha.astype(np.float32),
        hot_core=hot_core.astype(np.float32),
        white_core=white_core.astype(np.float32),
        ember_alpha=ember_alpha.astype(np.float32),
        smoke_alpha=smoke_alpha.astype(np.float32),
        warm_smoke=warm_smoke.astype(np.float32),
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


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)
    high = float(arr.max(initial=0.0))
    if high <= 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / high).clip(0.0, 1.0)


def _blend_rgb(image: np.ndarray, color, alpha: np.ndarray) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    if color_arr.ndim == 1:
        color_arr = color_arr.reshape(1, 1, 3)
    alpha_arr = np.asarray(alpha, dtype=np.float32).clip(0.0, 1.0)
    if alpha_arr.ndim == 2:
        alpha_arr = alpha_arr[..., None]
    return (base * (1.0 - alpha_arr) + color_arr * alpha_arr).clip(0.0, 1.0)


def _color_ramp(
    stops: tuple[tuple[float, tuple[float, float, float]], ...],
    values: np.ndarray,
) -> np.ndarray:
    positions = np.asarray([stop[0] for stop in stops], dtype=np.float32)
    colors = np.asarray([stop[1] for stop in stops], dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).clip(0.0, 1.0).ravel()
    channels = [np.interp(flat, positions, colors[:, idx]) for idx in range(3)]
    return np.stack(channels, axis=-1).reshape((*values.shape, 3)).astype(np.float32)


def _smooth_noise(shape: tuple[int, int], *, grid: int, seed: int) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    grid = max(int(grid), 1)
    small_h = max(2, int(math.ceil(height / grid)))
    small_w = max(2, int(math.ceil(width / grid)))
    rng = np.random.default_rng(int(seed))
    coarse = rng.random((small_h, small_w), dtype=np.float32)
    zoom = (height / small_h, width / small_w)
    noise = ndimage.zoom(coarse, zoom, order=3)
    return noise[:height, :width].clip(0.0, 1.0)


def _smooth_noise_px(
    shape: tuple[int, int],
    *,
    scale_px: float,
    seed: int,
    order: int = 3,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    scale_px = max(float(scale_px), 1.0)
    small_h = max(2, int(math.ceil(height / scale_px)))
    small_w = max(2, int(math.ceil(width / scale_px)))
    rng = np.random.default_rng(int(seed))
    coarse = rng.random((small_h, small_w), dtype=np.float32)
    zoom = (height / small_h, width / small_w)
    noise = ndimage.zoom(coarse, zoom, order=int(order))
    noise = noise[:height, :width]
    low = float(noise.min())
    high = float(noise.max())
    if high - low <= 1e-6:
        return np.zeros(shape, dtype=np.float32)
    return ((noise - low) / (high - low)).clip(0.0, 1.0).astype(np.float32)


def _fractal_noise(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    large = _smooth_noise(shape, grid=56, seed=seed + 1)
    medium = _smooth_noise(shape, grid=22, seed=seed + 2)
    fine = _smooth_noise(shape, grid=9, seed=seed + 3)
    noise = 0.52 * large + 0.30 * medium + 0.18 * fine
    low = float(noise.min())
    high = float(noise.max())
    if high - low <= 1e-6:
        return np.zeros(shape, dtype=np.float32)
    return ((noise - low) / (high - low)).astype(np.float32)


def _perimeter_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0.25
    if not binary.any():
        return np.zeros_like(mask, dtype=np.float32)
    eroded = ndimage.binary_erosion(binary, iterations=2, border_value=0)
    perimeter = (binary & ~eroded).astype(np.float32)
    return ndimage.gaussian_filter(perimeter, sigma=1.2).clip(0.0, 1.0)


def _ridge_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32).clip(0.0, 1.0)
    if arr.max(initial=0.0) <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    inner = ndimage.gaussian_filter(arr, sigma=0.55)
    outer = ndimage.gaussian_filter(arr, sigma=2.2)
    ridge = (inner - 0.72 * outer).clip(0.0, 1.0)
    high = float(ridge.max(initial=0.0))
    if high <= 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (ridge / high).clip(0.0, 1.0)


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
