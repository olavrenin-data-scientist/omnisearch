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
    size: int = 4096,
    tile_size: int = 1024,
    force: bool = False,
) -> Path:
    """Download/cache a higher-resolution NAIP image by tiling export requests."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(bbox_lonlat, size, tile_size)
    image_path = out_dir / f"naip_tiled_{key}_{size}.png"
    metadata_path = image_path.with_suffix(".metadata.json")
    if image_path.exists() and not force:
        return image_path

    size = int(size)
    tile_size = int(tile_size)
    cols = int(math.ceil(size / tile_size))
    rows = int(math.ceil(size / tile_size))
    west, south, east, north = (float(v) for v in bbox_lonlat)
    output = Image.new("RGB", (size, size))
    request_urls: list[str] = []

    for row in range(rows):
        for col in range(cols):
            x0 = col * tile_size
            y0 = row * tile_size
            w = min(tile_size, size - x0)
            h = min(tile_size, size - y0)
            tile_west = west + (east - west) * (x0 / size)
            tile_east = west + (east - west) * ((x0 + w) / size)
            tile_north = north - (north - south) * (y0 / size)
            tile_south = north - (north - south) * ((y0 + h) / size)
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
        "size": int(size),
        "tile_size": int(tile_size),
        "rows": rows,
        "cols": cols,
        "request_urls": request_urls,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path


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


def _cache_key(bbox_lonlat: tuple[float, float, float, float], *parts: int) -> str:
    raw = ",".join(f"{float(v):.8f}" for v in bbox_lonlat) + "|" + "|".join(str(int(v)) for v in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
