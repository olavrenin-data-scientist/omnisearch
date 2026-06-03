"""NAIP imagery helpers for CV rendering experiments."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import urllib.parse
import urllib.request
from PIL import Image, UnidentifiedImageError


NAIP_PLUS_EXPORT_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPPlus/ImageServer/exportImage"
)


class NaipTileCache:
    """Lazy NAIP tile cache for a bbox at a target meters-per-pixel resolution."""

    def __init__(
        self,
        *,
        bbox_lonlat: tuple[float, float, float, float],
        out_dir: str | Path,
        target_gsd_m: float,
        tile_size: int = 1024,
        force: bool = False,
    ):
        self.bbox_lonlat = tuple(float(v) for v in bbox_lonlat)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_gsd_m = max(float(target_gsd_m), 1e-6)
        self.tile_size = int(tile_size)
        self.force = bool(force)
        self.width_m, self.height_m = estimate_bbox_size_m(self.bbox_lonlat)
        self.width_px = max(1, int(math.ceil(self.width_m / self.target_gsd_m)))
        self.height_px = max(1, int(math.ceil(self.height_m / self.target_gsd_m)))
        self.cols = int(math.ceil(self.width_px / self.tile_size))
        self.rows = int(math.ceil(self.height_px / self.tile_size))
        self.key = _cache_key(self.bbox_lonlat, self.width_px, self.height_px, self.tile_size)
        self.cache_dir = self.out_dir / f"naip_tiles_{self.key}_{self.width_px}x{self.height_px}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self._write_manifest()

    @property
    def size_px(self) -> tuple[int, int]:
        return self.width_px, self.height_px

    @property
    def gsd_m_per_px(self) -> tuple[float, float]:
        return self.width_m / max(self.width_px, 1), self.height_m / max(self.height_px, 1)

    def crop_world(self, *, center_world: tuple[float, float], size_world: float) -> Image.Image:
        """Return a square crop centered in simulator world coordinates."""

        cx = (center_world[0] + 1.0) * 0.5 * self.width_px
        cy = (1.0 - (center_world[1] + 1.0) * 0.5) * self.height_px
        crop_w = max(2, int(round(size_world / 2.0 * self.width_px)))
        crop_h = max(2, int(round(size_world / 2.0 * self.height_px)))
        left = int(round(cx - crop_w * 0.5))
        top = int(round(cy - crop_h * 0.5))
        return self.crop_pixels((left, top, left + crop_w, top + crop_h))

    def crop_pixels(self, box: tuple[int, int, int, int]) -> Image.Image:
        """Return a pixel crop from the virtual tiled NAIP raster."""

        left, top, right, bottom = box
        crop_w = right - left
        crop_h = bottom - top
        crop = Image.new("RGB", (crop_w, crop_h), (0, 0, 0))
        source_left = max(0, left)
        source_top = max(0, top)
        source_right = min(self.width_px, right)
        source_bottom = min(self.height_px, bottom)
        if source_right <= source_left or source_bottom <= source_top:
            return crop

        col0 = source_left // self.tile_size
        col1 = (source_right - 1) // self.tile_size
        row0 = source_top // self.tile_size
        row1 = (source_bottom - 1) // self.tile_size
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                tile = self._load_tile(row, col)
                tile_x0 = col * self.tile_size
                tile_y0 = row * self.tile_size
                overlap = (
                    max(source_left, tile_x0),
                    max(source_top, tile_y0),
                    min(source_right, tile_x0 + tile.width),
                    min(source_bottom, tile_y0 + tile.height),
                )
                if overlap[2] <= overlap[0] or overlap[3] <= overlap[1]:
                    continue
                tile_box = (
                    overlap[0] - tile_x0,
                    overlap[1] - tile_y0,
                    overlap[2] - tile_x0,
                    overlap[3] - tile_y0,
                )
                paste_xy = (overlap[0] - left, overlap[1] - top)
                crop.paste(tile.crop(tile_box), paste_xy)
        return crop

    def _load_tile(self, row: int, col: int) -> Image.Image:
        path = self._tile_path(row, col)
        if not path.exists() or self.force:
            self._download_tile(row, col, path)
        return Image.open(path).convert("RGB")

    def _download_tile(self, row: int, col: int, path: Path) -> None:
        x0 = col * self.tile_size
        y0 = row * self.tile_size
        width = min(self.tile_size, self.width_px - x0)
        height = min(self.tile_size, self.height_px - y0)
        west, south, east, north = self.bbox_lonlat
        tile_west = west + (east - west) * (x0 / self.width_px)
        tile_east = west + (east - west) * ((x0 + width) / self.width_px)
        tile_north = north - (north - south) * (y0 / self.height_px)
        tile_south = north - (north - south) * ((y0 + height) / self.height_px)
        data, _content_type, _url = _download_naip_export(
            bbox_lonlat=(tile_west, tile_south, tile_east, tile_north),
            size=(width, height),
        )
        _write_verified_image(path, data)

    def _tile_path(self, row: int, col: int) -> Path:
        return self.cache_dir / f"tile_r{row:04d}_c{col:04d}.png"

    def _write_manifest(self) -> None:
        if self.manifest_path.exists():
            return
        manifest = {
            "provider": "USGS The National Map NAIP Plus ImageServer",
            "bbox_lonlat": list(self.bbox_lonlat),
            "target_gsd_m": self.target_gsd_m,
            "estimated_gsd_m_per_px": list(self.gsd_m_per_px),
            "ground_dimensions_m": [self.width_m, self.height_m],
            "size_px": [self.width_px, self.height_px],
            "tile_size": self.tile_size,
            "rows": self.rows,
            "cols": self.cols,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def fetch_naip_image(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    out_dir: str | Path,
    size: int = 2048,
    force: bool = False,
) -> Path:
    """Download/cache a NAIP image matching a lon/lat bbox.

    The returned image is intended as a visual background for the same bbox as
    the terrain cache. It is not a GeoTIFF; the georeferencing is carried by the
    bbox metadata that both the renderer and terrain cache share.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(bbox_lonlat, size)
    image_path = out_dir / f"naip_{key}_{size}.png"
    metadata_path = image_path.with_suffix(".metadata.json")
    if image_path.exists() and not force:
        return image_path

    data, content_type, url = _download_naip_export(bbox_lonlat=bbox_lonlat, size=(int(size), int(size)))
    _write_verified_image(image_path, data)
    metadata = {
        "provider": "USGS The National Map NAIP Plus ImageServer",
        "url": NAIP_PLUS_EXPORT_URL,
        "request_url": url,
        "bbox_lonlat": list(bbox_lonlat),
        "size": int(size),
        "content_type": content_type,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path


def fetch_naip_tiled_image(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    out_dir: str | Path,
    size: int | tuple[int, int] = 4096,
    tile_size: int = 1024,
    force: bool = False,
) -> Path:
    """Download/cache a higher-resolution NAIP image by tiling export requests."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = _normalize_size(size)
    key = _cache_key(bbox_lonlat, width, height, tile_size)
    image_path = out_dir / f"naip_tiled_{key}_{width}x{height}.png"
    metadata_path = image_path.with_suffix(".metadata.json")
    if image_path.exists() and not force:
        return image_path

    tile_size = int(tile_size)
    cols = int(math.ceil(width / tile_size))
    rows = int(math.ceil(height / tile_size))
    west, south, east, north = (float(v) for v in bbox_lonlat)
    output = Image.new("RGB", (width, height))
    request_urls: list[str] = []

    for row in range(rows):
        for col in range(cols):
            x0 = col * tile_size
            y0 = row * tile_size
            w = min(tile_size, width - x0)
            h = min(tile_size, height - y0)
            tile_west = west + (east - west) * (x0 / width)
            tile_east = west + (east - west) * ((x0 + w) / width)
            tile_north = north - (north - south) * (y0 / height)
            tile_south = north - (north - south) * ((y0 + h) / height)
            data, _content_type, url = _download_naip_export(
                bbox_lonlat=(tile_west, tile_south, tile_east, tile_north),
                size=(w, h),
            )
            tile_path = out_dir / f"{image_path.stem}_tile_{row:02d}_{col:02d}.png"
            _write_verified_image(tile_path, data)
            with Image.open(tile_path) as tile:
                output.paste(tile.convert("RGB"), (x0, y0))
            tile_path.unlink(missing_ok=True)
            request_urls.append(url)

    output.save(image_path)
    metadata = {
        "provider": "USGS The National Map NAIP Plus ImageServer",
        "url": NAIP_PLUS_EXPORT_URL,
        "bbox_lonlat": list(bbox_lonlat),
        "size": [int(width), int(height)],
        "tile_size": int(tile_size),
        "rows": rows,
        "cols": cols,
        "ground_dimensions_m": list(estimate_bbox_size_m(bbox_lonlat)),
        "estimated_gsd_m_per_px": [
            estimate_bbox_size_m(bbox_lonlat)[0] / max(width, 1),
            estimate_bbox_size_m(bbox_lonlat)[1] / max(height, 1),
        ],
        "request_urls": request_urls,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path


def fetch_naip_tiled_image_for_gsd(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    out_dir: str | Path,
    target_gsd_m: float,
    tile_size: int = 1024,
    force: bool = False,
) -> Path:
    """Download/cache a tiled NAIP image sized by target meters-per-pixel."""

    width_m, height_m = estimate_bbox_size_m(bbox_lonlat)
    gsd = max(float(target_gsd_m), 1e-6)
    width_px = max(1, int(math.ceil(width_m / gsd)))
    height_px = max(1, int(math.ceil(height_m / gsd)))
    return fetch_naip_tiled_image(
        bbox_lonlat=bbox_lonlat,
        out_dir=out_dir,
        size=(width_px, height_px),
        tile_size=tile_size,
        force=force,
    )


def estimate_bbox_size_m(bbox_lonlat: tuple[float, float, float, float]) -> tuple[float, float]:
    """Approximate lon/lat bbox size in meters."""

    west, south, east, north = (float(v) for v in bbox_lonlat)
    mid_lat = (south + north) * 0.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * max(math.cos(math.radians(mid_lat)), 1e-6)
    return abs(east - west) * meters_per_deg_lon, abs(north - south) * meters_per_deg_lat


def _download_naip_export(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[bytes, str, str]:
    params = {
        "bbox": ",".join(f"{float(v):.8f}" for v in bbox_lonlat),
        "bboxSR": "4326",
        "size": f"{int(size[0])},{int(size[1])}",
        "imageSR": "4326",
        "format": "png",
        "f": "image",
    }
    url = f"{NAIP_PLUS_EXPORT_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "omnisearch-cv-demo/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if not data:
        raise RuntimeError("NAIP service returned an empty image")
    return data, content_type, url


def _write_verified_image(image_path: Path, data: bytes) -> None:
    image_path.write_bytes(data)
    try:
        with Image.open(image_path) as image:
            image.verify()
    except UnidentifiedImageError as exc:
        message = data[:1000].decode("utf-8", errors="replace")
        image_path.unlink(missing_ok=True)
        raise RuntimeError(f"NAIP service did not return an image: {message}") from exc


def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, tuple):
        width, height = size
        return int(width), int(height)
    value = int(size)
    return value, value


def _cache_key(bbox_lonlat: tuple[float, float, float, float], *parts: int) -> str:
    raw = ",".join(f"{float(v):.8f}" for v in bbox_lonlat) + "|" + "|".join(str(int(v)) for v in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
