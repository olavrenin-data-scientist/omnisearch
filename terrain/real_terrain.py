"""Cache-first real terrain adapter.

This module is intentionally small: it defines the stable handoff between
geospatial preprocessing and the VMAS wildfire scenario. A future downloader
can fill the same ``.npz`` cache from USGS 3DEP and OpenStreetMap data without
changing the environment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import hashlib
import re

import numpy as np


LAND_ROAD, LAND_OPEN, LAND_BRUSH, LAND_FOREST, LAND_ROCK = range(5)
OBJECT_NONE, OBJECT_TREE, OBJECT_HOUSE = range(3)


@dataclass(frozen=True)
class RealTerrainMap:
    """Grid layers consumed by ``WildfireSearchScenario``."""

    land_cover: np.ndarray
    elevation: np.ndarray
    slope: np.ndarray
    moisture: np.ndarray
    fuel_density: np.ndarray
    rockiness: np.ndarray
    obstacle_type: np.ndarray
    obstacle_height: np.ndarray
    source: str


def load_real_terrain(
    *,
    grid_size: int,
    place: str = "Malibu Creek State Park, California",
    bbox: Optional[Sequence[float]] = None,
    cache_dir: str | Path = "data/terrain_cache",
    cache_path: str | Path | None = None,
) -> RealTerrainMap:
    """Load a preprocessed real-terrain grid.

    The expected cache format is an ``.npz`` file with these arrays:

    - ``land_cover``: integer land-cover ids, shape ``[grid_size, grid_size]``
    - ``elevation``: normalized world-height layer
    - ``slope``: terrain slope in simulation units
    - ``moisture``, ``fuel_density``, ``rockiness``: continuous fields
    - ``obstacle_type`` and ``obstacle_height``: houses/trees/clearance inputs

    Missing caches are treated as setup errors. Build one with
    ``scripts/build_real_terrain_cache.py`` before running the simulation.
    """

    path = Path(cache_path) if cache_path is not None else _default_cache_path(
        place=place,
        bbox=bbox,
        grid_size=grid_size,
        cache_dir=cache_dir,
    )
    if path.exists():
        return _load_npz(path, grid_size)

    raise FileNotFoundError(
        "No cached real terrain was found. Expected an .npz at "
        f"{path}. Generate a USGS 3DEP + OpenStreetMap/LANDFIRE cache first "
        "with scripts/build_real_terrain_cache.py."
    )


def _default_cache_path(
    *,
    place: str,
    bbox: Optional[Sequence[float]],
    grid_size: int,
    cache_dir: str | Path,
) -> Path:
    cache_dir = Path(cache_dir)
    if bbox is None:
        slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")[:64]
    else:
        bbox_key = ",".join(f"{float(v):.6f}" for v in bbox)
        slug = hashlib.sha1(bbox_key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{slug}_{int(grid_size)}.npz"


def _load_npz(path: Path, grid_size: int) -> RealTerrainMap:
    data = np.load(path, allow_pickle=False)
    source = str(data["source"].item()) if "source" in data else f"cache:{path}"
    terrain = RealTerrainMap(
        land_cover=_require_grid(data, "land_cover", grid_size).astype(np.int64),
        elevation=_require_grid(data, "elevation", grid_size).astype(np.float32),
        slope=_require_grid(data, "slope", grid_size).astype(np.float32),
        moisture=_require_grid(data, "moisture", grid_size).astype(np.float32),
        fuel_density=_require_grid(data, "fuel_density", grid_size).astype(np.float32),
        rockiness=_require_grid(data, "rockiness", grid_size).astype(np.float32),
        obstacle_type=_require_grid(data, "obstacle_type", grid_size).astype(np.int64),
        obstacle_height=_require_grid(data, "obstacle_height", grid_size).astype(np.float32),
        source=source,
    )
    return _sanitize(terrain)


def _require_grid(data: np.lib.npyio.NpzFile, name: str, grid_size: int) -> np.ndarray:
    if name not in data:
        raise ValueError(f"real terrain cache is missing required array '{name}'")
    array = np.asarray(data[name])
    expected = (grid_size, grid_size)
    if array.shape != expected:
        raise ValueError(f"real terrain cache array '{name}' has shape {array.shape}, expected {expected}")
    return array


def _sanitize(terrain: RealTerrainMap) -> RealTerrainMap:
    land_cover = np.clip(np.nan_to_num(terrain.land_cover, nan=LAND_OPEN), LAND_ROAD, LAND_ROCK).astype(np.int64)
    obstacle_type = np.clip(
        np.nan_to_num(terrain.obstacle_type, nan=OBJECT_NONE),
        OBJECT_NONE,
        OBJECT_HOUSE,
    ).astype(np.int64)
    return RealTerrainMap(
        land_cover=land_cover,
        elevation=_finite01ish(terrain.elevation),
        slope=np.clip(np.nan_to_num(terrain.slope, nan=0.0), 0.0, None).astype(np.float32),
        moisture=_finite01ish(terrain.moisture),
        fuel_density=_finite01ish(terrain.fuel_density),
        rockiness=_finite01ish(terrain.rockiness),
        obstacle_type=obstacle_type,
        obstacle_height=np.clip(np.nan_to_num(terrain.obstacle_height, nan=0.0), 0.0, None).astype(np.float32),
        source=terrain.source,
    )


def _finite01ish(array: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)

