"""LANDFIRE Product Service download and raster helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile
import hashlib
import re
import time

import numpy as np


LFPS_API_BASE = "https://lfps.usgs.gov/api"
LFPS_ARCGIS_JOB_BASE = (
    "https://lfps.usgs.gov/arcgis/rest/directories/arcgisjobs/"
    "landfireproductservice_gpserver"
)
DEFAULT_LANDFIRE_LAYER_LIST = "LF2025_FBFM40;LF2025_CC;LF2025_CH;LF2025_CBH;LF2025_CBD"


@dataclass(frozen=True)
class LandfireGrid:
    """LANDFIRE layers resampled onto the simulation grid."""

    fuel_model: np.ndarray
    fuel_density: np.ndarray
    canopy_cover: np.ndarray
    canopy_height_m: np.ndarray
    source: str


def ensure_landfire_geotiff(
    *,
    bbox: Sequence[float],
    source_cache_dir: str | Path,
    layer_list: str = DEFAULT_LANDFIRE_LAYER_LIST,
    email: str | None = None,
    resample_resolution: int = 31,
    output_projection: str | None = None,
    timeout_s: int = 1800,
    poll_interval_s: float = 10.0,
    force_download: bool = False,
) -> Path:
    """Return a cached LANDFIRE GeoTIFF, downloading from LFPS if needed."""

    cache_key = _landfire_cache_key(
        bbox=bbox,
        layer_list=layer_list,
        resample_resolution=resample_resolution,
        output_projection=output_projection,
    )
    cache_dir = Path(source_cache_dir) / "landfire" / cache_key
    zip_path = cache_dir / "landfire.zip"
    tif_path = cache_dir / "landfire.tif"
    if tif_path.exists() and not force_download:
        return tif_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or force_download:
        if not email:
            raise ValueError(
                "LANDFIRE LFPS requires a requestor email. Pass "
                "--landfire-email or set LANDFIRE_EMAIL."
            )
        _download_lfps_zip(
            bbox=bbox,
            layer_list=layer_list,
            email=email,
            resample_resolution=resample_resolution,
            output_projection=output_projection,
            zip_path=zip_path,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
    _extract_first_geotiff(zip_path, tif_path)
    return tif_path


def has_cached_landfire_source(
    *,
    bbox: Sequence[float],
    source_cache_dir: str | Path,
    layer_list: str = DEFAULT_LANDFIRE_LAYER_LIST,
    resample_resolution: int = 31,
    output_projection: str | None = None,
) -> bool:
    """Return whether raw/extracted LFPS data exists for this request."""

    cache_key = _landfire_cache_key(
        bbox=bbox,
        layer_list=layer_list,
        resample_resolution=resample_resolution,
        output_projection=output_projection,
    )
    cache_dir = Path(source_cache_dir) / "landfire" / cache_key
    return (cache_dir / "landfire.tif").exists() or (cache_dir / "landfire.zip").exists()


def read_landfire_grid(
    *,
    geotiff_path: str | Path,
    projected_bounds: Sequence[float],
    grid_size: int,
    layer_list: str = DEFAULT_LANDFIRE_LAYER_LIST,
) -> LandfireGrid:
    """Read LANDFIRE bands and resample them to the OmniSearch grid."""

    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise ImportError(
            "LANDFIRE processing requires rasterio. Install optional GIS "
            "dependencies with: pip install -r requirements-geo.txt"
        ) from exc

    names = [name.strip().upper() for name in layer_list.split(";") if name.strip()]
    if not names:
        raise ValueError("LANDFIRE layer list cannot be empty")

    dst_transform = from_bounds(*projected_bounds, grid_size, grid_size)
    dst_crs = "EPSG:3857"

    with rasterio.open(geotiff_path) as dataset:
        bands = {}
        for idx, name in enumerate(names, start=1):
            if idx > dataset.count:
                break
            data = dataset.read(idx).astype(np.float32)
            nodata = dataset.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            resampling = Resampling.nearest if _is_categorical_layer(name) else Resampling.bilinear
            dst = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
            reproject(
                source=data,
                destination=dst,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                dst_nodata=np.nan,
            )
            bands[name] = dst

    fuel_model = _first_matching_band(bands, ("FBFM40", "F40", "FBFM13", "F13"))
    canopy_cover = _first_matching_band(bands, ("CC", "EVC", "CANCOV"))
    canopy_height = _first_matching_band(bands, ("CH", "CANHT"))

    if fuel_model is None:
        raise ValueError(
            "LANDFIRE GeoTIFF did not include a fuel model band. Include a "
            "layer such as 250FBFM40 in --landfire-layer-list."
        )
    if canopy_cover is None:
        canopy_cover = np.zeros_like(fuel_model, dtype=np.float32)
    if canopy_height is None:
        canopy_height = np.zeros_like(fuel_model, dtype=np.float32)

    fuel_model = _fill_nan(fuel_model, fill=0.0)
    canopy_cover = np.clip(_fill_nan(canopy_cover, fill=0.0), 0.0, 100.0)
    canopy_height = _clean_canopy_height(canopy_height)
    fuel_density = _fuel_density_from_fbfm40(fuel_model)

    return LandfireGrid(
        fuel_model=fuel_model.astype(np.float32),
        fuel_density=fuel_density.astype(np.float32),
        canopy_cover=canopy_cover.astype(np.float32),
        canopy_height_m=canopy_height.astype(np.float32),
        source=f"LANDFIRE layers {layer_list}; source={Path(geotiff_path)}",
    )


def _download_lfps_zip(
    *,
    bbox: Sequence[float],
    layer_list: str,
    email: str,
    resample_resolution: int,
    output_projection: str | None,
    zip_path: Path,
    timeout_s: int,
    poll_interval_s: float,
) -> None:
    import requests

    params = {
        "Email": email,
        "Layer_List": layer_list,
        "Include_Layer_List_XML_File": "false",
        "Area_of_Interest": " ".join(f"{float(v):.8f}" for v in bbox),
        "Resample_Resolution": int(max(resample_resolution, 31)),
    }
    if output_projection:
        params["Output_Projection"] = output_projection

    try:
        submit = requests.get(
            f"{LFPS_API_BASE}/job/submit",
            params=params,
            timeout=120,
            headers={"Accept": "application/json", "User-Agent": "omnisearch"},
        )
        submit.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Could not submit LANDFIRE LFPS job. Check internet access, "
            "VPN/firewall, and DNS resolution for lfps.usgs.gov."
        ) from exc
    payload = _json_or_raise(submit, "LANDFIRE LFPS submit")
    if payload.get("success") is False:
        raise RuntimeError(f"LANDFIRE LFPS submit failed: {payload}")
    job_id = payload.get("jobId")
    if not job_id:
        raise RuntimeError(f"LANDFIRE LFPS did not return a job id: {payload}")

    deadline = time.time() + timeout_s
    status_payload = payload
    while True:
        try:
            status_response = requests.get(
                f"{LFPS_API_BASE}/job/status",
                params={"JobId": job_id},
                timeout=120,
            )
            status_response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Could not poll LANDFIRE LFPS job {job_id}.") from exc
        status_payload = _json_or_raise(status_response, f"LANDFIRE LFPS status for {job_id}")
        job_status = status_payload.get("status") or status_payload.get("jobStatus")
        if job_status in {"Succeeded", "esriJobSucceeded"}:
            break
        if job_status in {"Failed", "Canceled", "Cancelled", "TimedOut", "esriJobFailed", "esriJobCancelled", "esriJobTimedOut"}:
            messages = status_payload.get("messages", [])
            raise RuntimeError(f"LANDFIRE LFPS job {job_id} ended with {job_status}: {messages}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for LANDFIRE LFPS job {job_id}")
        time.sleep(max(float(poll_interval_s), 1.0))

    download_url = _extract_result_url(status_payload, job_id)
    try:
        download = requests.get(download_url, timeout=300)
        download.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Could not download completed LANDFIRE LFPS result from {download_url}.") from exc
    zip_path.write_bytes(download.content)


def _json_or_raise(response, context: str) -> dict:
    try:
        return response.json()
    except ValueError as exc:
        snippet = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"{context} returned non-JSON response: {snippet}") from exc


def _extract_result_url(result_payload: dict, job_id: str) -> str:
    url = _find_url_value(result_payload)
    if not url:
        candidate = (
            result_payload.get("esriJobId")
            or result_payload.get("esriJobID")
            or result_payload.get("arcgisJobId")
            or result_payload.get("arcgisJobID")
            or job_id
        )
        url = f"{LFPS_ARCGIS_JOB_BASE}/{candidate}/scratch/{candidate}.zip"
    if url.startswith("/"):
        url = "https://lfps.usgs.gov" + url
    return str(url)


def _find_url_value(payload) -> str | None:
    if isinstance(payload, str):
        match = re.search(r"https://[^\s\"']+?\.zip", payload)
        return match.group(0) if match else None
    if isinstance(payload, dict):
        for key in (
            "downloadUrl", "downloadURL", "url", "outputUrl", "outputURL",
            "outputFile", "Output_File", "zipUrl", "zipURL", "fileUrl", "fileURL",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = _find_url_value(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_url_value(item)
            if found:
                return found
    return None


def _extract_first_geotiff(zip_path: Path, tif_path: Path) -> None:
    with ZipFile(zip_path) as archive:
        names = [
            name for name in archive.namelist()
            if name.lower().endswith((".tif", ".tiff")) and "__macosx" not in name.lower()
        ]
        if not names:
            raise RuntimeError(f"LANDFIRE zip did not contain a GeoTIFF: {zip_path}")
        with archive.open(names[0]) as src, tif_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)


def _landfire_cache_key(
    *,
    bbox: Sequence[float],
    layer_list: str,
    resample_resolution: int,
    output_projection: str | None,
) -> str:
    payload = "|".join(
        [
            ",".join(f"{float(v):.6f}" for v in bbox),
            layer_list,
            str(int(max(resample_resolution, 31))),
            output_projection or "",
        ],
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _first_matching_band(bands: dict[str, np.ndarray], needles: tuple[str, ...]) -> np.ndarray | None:
    for name, band in bands.items():
        clean_name = name.upper()
        if any(needle in clean_name for needle in needles):
            return band
    return None


def _is_categorical_layer(name: str) -> bool:
    return any(token in name.upper() for token in ("FBFM", "F40", "F13"))


def _fill_nan(array: np.ndarray, fill: float) -> np.ndarray:
    return np.nan_to_num(array, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def _clean_canopy_height(array: np.ndarray) -> np.ndarray:
    height = np.clip(_fill_nan(array, fill=0.0), 0.0, None)
    finite = height[np.isfinite(height)]
    if finite.size and float(np.nanpercentile(finite, 99)) > 100.0:
        height = height / 10.0
    return np.clip(height, 0.0, 70.0).astype(np.float32)


def _fuel_density_from_fbfm40(fuel_model: np.ndarray) -> np.ndarray:
    fm = np.rint(_fill_nan(fuel_model, fill=0.0)).astype(np.int32)
    density = np.zeros(fm.shape, dtype=np.float32)

    nonburnable = (fm <= 0) | ((fm >= 90) & (fm < 100))
    density[nonburnable] = 0.02

    grass = (fm >= 101) & (fm <= 109)
    density[grass] = 0.25 + 0.055 * (fm[grass] - 101)

    grass_shrub = (fm >= 121) & (fm <= 124)
    density[grass_shrub] = 0.55 + 0.08 * (fm[grass_shrub] - 121)

    shrub = (fm >= 141) & (fm <= 149)
    density[shrub] = 0.58 + 0.045 * (fm[shrub] - 141)

    timber_understory = (fm >= 161) & (fm <= 165)
    density[timber_understory] = 0.62 + 0.07 * (fm[timber_understory] - 161)

    timber_litter = (fm >= 181) & (fm <= 189)
    density[timber_litter] = 0.45 + 0.045 * (fm[timber_litter] - 181)

    slash_blowdown = (fm >= 201) & (fm <= 204)
    density[slash_blowdown] = 0.75 + 0.06 * (fm[slash_blowdown] - 201)

    unknown_burnable = (density <= 0.0) & ~nonburnable
    density[unknown_burnable] = 0.45
    return np.clip(density, 0.0, 1.0)
