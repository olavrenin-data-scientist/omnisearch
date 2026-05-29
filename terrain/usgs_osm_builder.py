"""Build real-terrain caches from USGS 3DEP and OpenStreetMap.

The output format is the ``.npz`` schema consumed by ``terrain.real_terrain``.
USGS provides bare-earth elevation; OpenStreetMap provides roads/buildings.
Vegetation fields remain derived placeholders until a fuel/land-cover source
such as LANDFIRE or NLCD is added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import hashlib

import numpy as np

from .real_terrain import (
    LAND_BRUSH,
    LAND_FOREST,
    LAND_OPEN,
    LAND_ROAD,
    LAND_ROCK,
    OBJECT_HOUSE,
    OBJECT_NONE,
    OBJECT_TREE,
    _default_cache_path,
)
from .landfire_client import (
    DEFAULT_LANDFIRE_LAYER_LIST,
    ensure_landfire_geotiff,
    has_cached_landfire_source,
    read_landfire_grid,
)


USGS_3DEP_EXPORT_IMAGE_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)


def build_real_terrain_cache(
    *,
    grid_size: int,
    place: str = "Malibu Creek State Park, California",
    bbox: Optional[Sequence[float]] = None,
    cache_dir: str | Path = "data/terrain_cache",
    out: str | Path | None = None,
    dem_resolution_m: int = 10,
    terrain_elevation_scale: float = 0.30,
    road_width_m: float = 8.0,
    building_height: float = 0.10,
    osm_timeout: int = 180,
    fuel_source: str = "derived",
    source_cache_dir: str | Path = "data/source_cache",
    landfire_layer_list: str = DEFAULT_LANDFIRE_LAYER_LIST,
    landfire_email: str | None = None,
    landfire_resample_resolution: int = 31,
    landfire_output_projection: str | None = None,
    landfire_timeout_s: int = 1800,
    landfire_poll_interval_s: float = 10.0,
    landfire_force_download: bool = False,
    force_rebuild: bool = False,
    source_note: str | None = None,
) -> Path:
    """Create an OmniSearch terrain cache from USGS 3DEP and OSM layers."""

    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox must be (west, south, east, north)")
    if fuel_source not in {"derived", "landfire"}:
        raise ValueError("fuel_source must be 'derived' or 'landfire'")

    deps = _import_geospatial_dependencies()
    ox = deps["osmnx"]
    pyproj = deps["pyproj"]
    requests = deps["requests"]
    MemoryFile = deps["MemoryFile"]
    rasterize = deps["rasterize"]
    from_bounds = deps["from_bounds"]

    if osm_timeout is not None and hasattr(ox, "settings"):
        ox.settings.requests_timeout = osm_timeout

    bbox = tuple(float(v) for v in (bbox if bbox is not None else _bbox_from_place(ox, place)))
    west, south, east, north = bbox
    if west >= east or south >= north:
        raise ValueError("bbox must be ordered as west, south, east, north")

    out_path = Path(out) if out is not None else _default_cache_path(
        place=place,
        bbox=bbox,
        grid_size=grid_size,
        cache_dir=cache_dir,
    )
    if out_path.exists() and not force_rebuild and _cache_matches_options(
        out_path,
        fuel_source=fuel_source,
        landfire_layer_list=landfire_layer_list,
    ):
        return out_path
    if (
        fuel_source == "landfire"
        and not landfire_email
        and not has_cached_landfire_source(
            bbox=bbox,
            source_cache_dir=source_cache_dir,
            layer_list=landfire_layer_list,
            resample_resolution=landfire_resample_resolution,
            output_projection=landfire_output_projection,
        )
    ):
        raise ValueError(
            "LANDFIRE LFPS requires a requestor email for new downloads. "
            "Pass --landfire-email or set LANDFIRE_EMAIL."
        )

    dem_grid, projected_bounds, cell_size_m = _download_usgs_dem_grid(
        requests=requests,
        pyproj=pyproj,
        MemoryFile=MemoryFile,
        bbox=bbox,
        grid_size=grid_size,
    )

    elevation = _normalize(dem_grid) * float(terrain_elevation_scale)
    slope = _slope_m_per_m(dem_grid, cell_size_m)
    slope_norm = _normalize(slope)

    seed = int(hashlib.sha1(f"{bbox}:{grid_size}:{dem_resolution_m}".encode("utf-8")).hexdigest()[:8], 16)
    moisture, fuel_density, rockiness = _derived_surface_fields(elevation, slope_norm, seed)
    landfire = None
    if fuel_source == "landfire":
        landfire_tif = ensure_landfire_geotiff(
            bbox=bbox,
            source_cache_dir=source_cache_dir,
            layer_list=landfire_layer_list,
            email=landfire_email,
            resample_resolution=landfire_resample_resolution,
            output_projection=landfire_output_projection,
            timeout_s=landfire_timeout_s,
            poll_interval_s=landfire_poll_interval_s,
            force_download=landfire_force_download,
        )
        landfire = read_landfire_grid(
            geotiff_path=landfire_tif,
            projected_bounds=projected_bounds,
            grid_size=grid_size,
            layer_list=landfire_layer_list,
        )
        fuel_density = landfire.fuel_density

    land_cover = np.full((grid_size, grid_size), LAND_OPEN, dtype=np.int64)
    if landfire is None:
        land_cover[(fuel_density > np.quantile(fuel_density, 0.64)) & (moisture < np.quantile(moisture, 0.68))] = LAND_BRUSH
        land_cover[(fuel_density > np.quantile(fuel_density, 0.74)) & (moisture > np.quantile(moisture, 0.52))] = LAND_FOREST
    else:
        forest, brush = _landfire_cover_masks(landfire.fuel_model, landfire.fuel_density, landfire.canopy_cover)
        land_cover[brush] = LAND_BRUSH
        land_cover[forest] = LAND_FOREST

    rock_score = 0.72 * rockiness + 0.45 * slope_norm
    land_cover[rock_score > np.quantile(rock_score, 0.92)] = LAND_ROCK

    transform = from_bounds(*projected_bounds, grid_size, grid_size)
    roads = _osm_features_from_bbox(ox, bbox, {"highway": True})
    buildings = _osm_features_from_bbox(ox, bbox, {"building": True})

    road_mask = _rasterize_roads(roads, projected_bounds, transform, grid_size, road_width_m, rasterize)
    building_mask = _rasterize_buildings(buildings, projected_bounds, transform, grid_size, rasterize)

    land_cover[road_mask] = LAND_ROAD
    obstacle_type = np.full((grid_size, grid_size), OBJECT_NONE, dtype=np.int64)
    obstacle_height = np.zeros((grid_size, grid_size), dtype=np.float32)
    if landfire is not None:
        canopy_mask, tree_mask, canopy_height = _landfire_canopy_obstacles(landfire)
        obstacle_height[canopy_mask] = canopy_height[canopy_mask]
        obstacle_type[tree_mask & ~road_mask] = OBJECT_TREE
    buildable_buildings = building_mask & ~road_mask
    obstacle_type[buildable_buildings] = OBJECT_HOUSE
    obstacle_height[buildable_buildings] = float(building_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = source_note or (
        f"USGS 3DEP DEM {dem_resolution_m}m + OpenStreetMap; "
        f"fuel_source={fuel_source}; place={place!r}; "
        f"bbox={tuple(round(v, 6) for v in bbox)}"
    )
    np.savez_compressed(
        out_path,
        land_cover=land_cover,
        elevation=elevation.astype(np.float32),
        slope=slope.astype(np.float32),
        moisture=moisture.astype(np.float32),
        fuel_density=fuel_density.astype(np.float32),
        rockiness=rockiness.astype(np.float32),
        obstacle_type=obstacle_type,
        obstacle_height=obstacle_height.astype(np.float32),
        source=np.asarray(source),
        bbox=np.asarray(bbox, dtype=np.float64),
        dem_resolution_m=np.asarray(dem_resolution_m, dtype=np.int32),
        projected_crs=np.asarray("EPSG:3857"),
        projected_bounds=np.asarray(projected_bounds, dtype=np.float64),
        fuel_source=np.asarray(fuel_source),
        landfire_layer_list=np.asarray(landfire_layer_list if fuel_source == "landfire" else ""),
    )
    return out_path


def _cache_matches_options(path: Path, *, fuel_source: str, landfire_layer_list: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_fuel_source = str(data["fuel_source"].item()) if "fuel_source" in data else "derived"
            cached_layers = str(data["landfire_layer_list"].item()) if "landfire_layer_list" in data else ""
    except Exception:
        return False
    if cached_fuel_source != fuel_source:
        return False
    if fuel_source == "landfire" and cached_layers != landfire_layer_list:
        return False
    return True


def _import_geospatial_dependencies() -> dict:
    missing = []
    modules = {}
    for name in ("osmnx", "pyproj", "requests"):
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    try:
        from rasterio.io import MemoryFile
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
    except ImportError:
        missing.append("rasterio")
    else:
        modules["MemoryFile"] = MemoryFile
        modules["rasterize"] = rasterize
        modules["from_bounds"] = from_bounds

    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise ImportError(
            "Missing geospatial dependencies: "
            f"{missing_list}. Install them in the project venv, for example: "
            "pip install osmnx rasterio geopandas shapely pyproj requests"
        )
    return modules


def _bbox_from_place(ox, place: str) -> tuple[float, float, float, float]:
    gdf = ox.geocode_to_gdf(place)
    west, south, east, north = gdf.total_bounds
    return float(west), float(south), float(east), float(north)


def _download_usgs_dem_grid(
    *,
    requests,
    pyproj,
    MemoryFile,
    bbox: Sequence[float],
    grid_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    west, south, east, north = bbox
    params = {
        "f": "image",
        "format": "tiff",
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "imageSR": "3857",
        "size": f"{grid_size},{grid_size}",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "noDataInterpretation": "esriNoDataMatchAny",
    }
    try:
        response = requests.get(USGS_3DEP_EXPORT_IMAGE_URL, params=params, timeout=180)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Could not download USGS 3DEP elevation data from "
            f"{USGS_3DEP_EXPORT_IMAGE_URL}. Check internet access, VPN/firewall, "
            "and DNS resolution for elevation.nationalmap.gov."
        ) from exc
    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower() or response.content[:1] == b"{":
        raise RuntimeError(f"USGS 3DEP returned an error response: {response.text[:500]}")

    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            grid = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata
            if nodata is not None:
                grid = np.where(grid == nodata, np.nan, grid)

    if grid.shape != (grid_size, grid_size):
        grid = _resize_nearest(grid, (grid_size, grid_size))
    grid = _fill_nan_grid(grid)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    west_m, south_m = transformer.transform(west, south)
    east_m, north_m = transformer.transform(east, north)
    projected_bounds = (float(west_m), float(south_m), float(east_m), float(north_m))
    cell_size_m = max(
        abs(float(east_m - west_m)) / max(grid_size - 1, 1),
        abs(float(north_m - south_m)) / max(grid_size - 1, 1),
    )
    return grid, projected_bounds, float(cell_size_m)


def _resize_nearest(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y_idx = np.linspace(0, grid.shape[0] - 1, shape[0]).round().astype(int)
    x_idx = np.linspace(0, grid.shape[1] - 1, shape[1]).round().astype(int)
    return grid[np.ix_(y_idx, x_idx)]


def _fill_nan_grid(grid: np.ndarray) -> np.ndarray:
    if np.isfinite(grid).all():
        return grid
    finite = np.isfinite(grid)
    if not finite.any():
        raise ValueError("USGS DEM returned no finite elevation values for this area")
    filled = grid.copy()
    filled[~finite] = float(np.nanmean(grid))
    for _ in range(6):
        neighbors = _neighbor_average(filled)
        filled[~finite] = neighbors[~finite]
    return filled


def _slope_m_per_m(elevation_m: np.ndarray, cell_size_m: float) -> np.ndarray:
    grade_y, grade_x = np.gradient(elevation_m, max(cell_size_m, 1e-6), max(cell_size_m, 1e-6))
    return np.clip(np.sqrt(grade_x ** 2 + grade_y ** 2), 0.0, 3.0).astype(np.float32)


def _derived_surface_fields(
    elevation: np.ndarray,
    slope_norm: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise_a = _smooth_noise(rng, elevation.shape, passes=8)
    noise_b = _smooth_noise(rng, elevation.shape, passes=5)
    elevation_norm = _normalize(elevation)
    moisture = _normalize(0.52 * (1.0 - elevation_norm) + 0.34 * noise_a - 0.18 * slope_norm)
    rockiness = _normalize(0.64 * slope_norm + 0.24 * elevation_norm + 0.12 * noise_b)
    fuel_density = _normalize(0.50 * moisture + 0.36 * noise_a + 0.18 * noise_b - 0.28 * rockiness)
    return moisture, fuel_density, rockiness


def _landfire_cover_masks(
    fuel_model: np.ndarray,
    fuel_density: np.ndarray,
    canopy_cover: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fm = np.rint(np.nan_to_num(fuel_model, nan=0.0)).astype(np.int32)
    timber_or_slash = ((fm >= 161) & (fm <= 189)) | ((fm >= 201) & (fm <= 204))
    shrub_or_grass_shrub = ((fm >= 121) & (fm <= 149))
    forest = (canopy_cover >= 35.0) | (timber_or_slash & (canopy_cover >= 18.0))
    brush = shrub_or_grass_shrub | ((fuel_density >= 0.50) & (canopy_cover < 35.0))
    brush &= ~forest
    no_fuel = fuel_density <= 0.08
    return forest & ~no_fuel, brush & ~no_fuel


def _landfire_canopy_obstacles(landfire) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canopy_cover = np.nan_to_num(landfire.canopy_cover, nan=0.0)
    canopy_height_m = np.nan_to_num(landfire.canopy_height_m, nan=0.0)
    if float(canopy_height_m.max(initial=0.0)) <= 1e-6:
        canopy_height_m = canopy_cover / 100.0 * 24.0
    canopy_mask = (canopy_cover >= 20.0) & (canopy_height_m >= 1.0)
    height_norm = np.clip(canopy_height_m / 35.0, 0.0, 1.0)
    sim_height = (0.04 + 0.20 * height_norm).astype(np.float32)

    # Tree trunks/very dense crowns are physical UGV blockers; ordinary canopy
    # still contributes to drone clearance through obstacle_height.
    density_noise = _deterministic_grid_noise(canopy_cover.shape)
    tree_mask = (
        (canopy_cover >= 75.0)
        & (canopy_height_m >= 4.0)
        & (landfire.fuel_density >= 0.25)
        & (density_noise > 0.58)
    )
    return canopy_mask, tree_mask, sim_height


def _deterministic_grid_noise(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.uint32)
    values = xx * np.uint32(374761393) + yy * np.uint32(668265263) + np.uint32(2246822519)
    values = (values ^ (values >> np.uint32(13))) * np.uint32(1274126177)
    values = values ^ (values >> np.uint32(16))
    return (values.astype(np.float32) / np.float32(np.iinfo(np.uint32).max)).astype(np.float32)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], passes: int) -> np.ndarray:
    field = rng.random(shape, dtype=np.float32)
    for _ in range(max(passes, 0)):
        field = _neighbor_average(field)
    return _normalize(field)


def _neighbor_average(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    return (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0


def _osm_features_from_bbox(ox, bbox: Sequence[float], tags: dict):
    try:
        return ox.features_from_bbox(tuple(bbox), tags)
    except TypeError:
        west, south, east, north = bbox
        return ox.features_from_bbox(north, south, east, west, tags)
    except Exception:
        return None


def _rasterize_roads(
    roads,
    projected_bounds: tuple[float, float, float, float],
    transform,
    grid_size: int,
    road_width_m: float,
    rasterize,
) -> np.ndarray:
    if roads is None or len(roads) == 0:
        return np.zeros((grid_size, grid_size), dtype=bool)
    roads = roads[roads.geometry.notna()].to_crs("EPSG:3857")
    buffered = roads.geometry.buffer(max(float(road_width_m), 0.5))
    return _rasterize_geometry(buffered, projected_bounds, transform, grid_size, rasterize)


def _rasterize_buildings(
    buildings,
    projected_bounds: tuple[float, float, float, float],
    transform,
    grid_size: int,
    rasterize,
) -> np.ndarray:
    if buildings is None or len(buildings) == 0:
        return np.zeros((grid_size, grid_size), dtype=bool)
    buildings = buildings[buildings.geometry.notna()].to_crs("EPSG:3857")
    return _rasterize_geometry(buildings.geometry, projected_bounds, transform, grid_size, rasterize)


def _rasterize_geometry(
    geometries,
    projected_bounds: tuple[float, float, float, float],
    transform,
    grid_size: int,
    rasterize,
) -> np.ndarray:
    west_m, south_m, east_m, north_m = projected_bounds
    shapes = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if maxx < west_m or minx > east_m or maxy < south_m or miny > north_m:
            continue
        shapes.append((geom, 1))
    if not shapes:
        return np.zeros((grid_size, grid_size), dtype=bool)
    return rasterize(
        shapes,
        out_shape=(grid_size, grid_size),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=float(np.nanmean(array)) if np.isfinite(array).any() else 0.0)
    array = array - float(array.min())
    denom = float(array.max())
    if denom <= 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return (array / denom).astype(np.float32)
