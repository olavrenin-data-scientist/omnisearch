"""Simulator-facing CV renderer and preliminary detector adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance

from .naip import (
    NaipTileCache,
    estimate_bbox_size_m,
    fetch_naip_image,
    fetch_naip_tiled_image,
    fetch_naip_tiled_image_for_gsd,
)
from .preliminary_detector import PreliminaryPersonDetector
from .wildfire_effects import (
    WildfireEffectConfig,
    apply_wildfire_effects_to_pil,
    masks_from_simulation_grids,
)


@dataclass(frozen=True)
class SimEntity:
    """Continuous simulation entity position."""

    index: int
    world_xy: tuple[float, float]


@dataclass(frozen=True)
class SimDrone:
    """Continuous simulation drone state."""

    index: int
    name: str
    world_xy: tuple[float, float]
    altitude_agl: float


@dataclass(frozen=True)
class SimWildfireState:
    """Simulator wildfire fields needed for image-space rendering."""

    fire_grid: np.ndarray
    fire_intensity_grid: np.ndarray
    burned_grid: np.ndarray
    smoke_grid: np.ndarray | None = None
    wind_direction: tuple[float, float] = (1.0, 0.0)


class SimulationCvAdapter:
    """Render drone camera images from simulator state and produce detections.

    This is the bridge between the abstract simulator and the CV workflow. It
    uses NAIP as the area background, places SARD survivor cutouts using the
    known simulator state, then runs a preliminary detector over the generated
    ground-truth boxes. YOLO can later replace ``PreliminaryPersonDetector``
    without changing the coordinate-mapping contract.
    """

    def __init__(
        self,
        *,
        terrain_cache_path: str | Path,
        naip_image_path: str | Path | None = None,
        naip_cache_dir: str | Path = "data/source_cache/naip",
        target_gsd_m: float | None = 0.5,
        lazy_tile_cache: bool = True,
        naip_size: int = 8192,
        tiled_naip: bool = True,
        tile_size: int = 1024,
        image_size: int = 512,
        fov_deg: float = 65.0,
        human_asset_path: str | Path | None = "data/cv_assets/sard_grabcut/sard_survivor_0280.png",
        human_assets_dir: str | Path | None = None,
        human_asset_list_path: str | Path | None = None,
        survivor_width_m: float = 2.4,
        survivor_height_m: float = 1.4,
        survivor_rotation_deg: float = 0.0,
        asset_resample: str = "nearest",
        detection_probability: float = 1.0,
        pixel_noise_std: float = 0.0,
        confidence: float = 0.95,
        confidence_jitter: float = 0.0,
        render_wildfire_effects: bool = True,
        wildfire_effect_seed: int | None = None,
        seed: int = 7,
        root: str | Path | None = None,
    ):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
        self.terrain_cache_path = self._resolve(terrain_cache_path)
        self.image_size = int(image_size)
        self.fov_deg = float(fov_deg)
        self.survivor_width_m = float(survivor_width_m)
        self.survivor_height_m = float(survivor_height_m)
        self.survivor_rotation_deg = float(survivor_rotation_deg)
        self.asset_resample = asset_resample
        self.render_wildfire_effects = bool(render_wildfire_effects)
        self.wildfire_effect_config = WildfireEffectConfig(
            seed=int(seed if wildfire_effect_seed is None else wildfire_effect_seed)
        )
        self.rng = random.Random(int(seed))
        self.detector = PreliminaryPersonDetector(
            detection_probability=detection_probability,
            pixel_noise_std=pixel_noise_std,
            confidence=confidence,
            confidence_jitter=confidence_jitter,
            seed=seed,
        )

        terrain = np.load(self.terrain_cache_path, allow_pickle=False)
        try:
            if "bbox" not in terrain:
                raise ValueError(f"Terrain cache has no bbox metadata: {self.terrain_cache_path}")
            self.grid_size = int(np.asarray(terrain["land_cover"]).shape[0])
            self.bbox_lonlat = tuple(float(v) for v in terrain["bbox"].tolist())
            if "sim_units_per_meter" not in terrain:
                raise ValueError(f"Terrain cache has no sim_units_per_meter metadata: {self.terrain_cache_path}")
            self.sim_units_per_meter = float(np.asarray(terrain["sim_units_per_meter"]).item())
        finally:
            terrain.close()
        if self.sim_units_per_meter <= 0.0:
            raise ValueError("sim_units_per_meter must be positive for CV coordinate conversion")

        self.tile_cache: NaipTileCache | None = None
        self.naip_image_path = self._load_or_fetch_naip(
            naip_image_path=naip_image_path,
            cache_dir=naip_cache_dir,
            target_gsd_m=target_gsd_m,
            lazy_tile_cache=lazy_tile_cache,
            naip_size=naip_size,
            tiled=bool(tiled_naip),
            tile_size=int(tile_size),
        )
        width_m, height_m = estimate_bbox_size_m(self.bbox_lonlat)
        if self.naip_image_path is not None:
            self.background = Image.open(self.naip_image_path).convert("RGB")
            self.background_size_px = self.background.size
            self.background_gsd_m_per_px = (
                width_m / max(self.background.width, 1),
                height_m / max(self.background.height, 1),
            )
        elif self.tile_cache is not None:
            self.background = None
            self.background_size_px = self.tile_cache.size_px
            self.background_gsd_m_per_px = self.tile_cache.gsd_m_per_px
        else:
            raise RuntimeError("CV adapter must have either a NAIP image or a tile cache")
        self.human_assets_dir = self._resolve(human_assets_dir) if human_assets_dir is not None else None
        self.human_asset_list_path = self._resolve(human_asset_list_path) if human_asset_list_path is not None else None
        self.human_assets = self._load_human_assets(
            human_asset_path=human_asset_path,
            human_assets_dir=human_assets_dir,
            human_asset_list_path=human_asset_list_path,
        )
        self.human_asset_path = self.human_assets[0][0] if self.human_assets else None
        self.human_asset = self.human_assets[0][1] if self.human_assets else None
        self._asset_order = list(range(len(self.human_assets)))
        self.rng.shuffle(self._asset_order)

    def render_and_detect(
        self,
        *,
        drone: SimDrone,
        survivors: Iterable[SimEntity],
        wildfire_state: SimWildfireState | None = None,
        image_path: str | Path | None = None,
    ) -> dict:
        altitude_m = max(float(drone.altitude_agl) / self.sim_units_per_meter, 1e-6)
        footprint_m = 2.0 * altitude_m * math.tan(math.radians(self.fov_deg) / 2.0)
        footprint_world = footprint_m * self.sim_units_per_meter
        source_crop_size_px = (
            max(2, int(round(footprint_world / 2.0 * self.background_size_px[0]))),
            max(2, int(round(footprint_world / 2.0 * self.background_size_px[1]))),
        )

        crop = self._crop_background(drone.world_xy, source_crop_size_px, footprint_world=footprint_world)
        view = crop.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        wildfire_stats = None
        wildfire_masks = None
        if self.render_wildfire_effects and wildfire_state is not None and image_path is not None:
            wildfire_masks = masks_from_simulation_grids(
                image_size=view.size,
                center_world=drone.world_xy,
                footprint_world=footprint_world,
                fire_grid=wildfire_state.fire_grid,
                fire_intensity_grid=wildfire_state.fire_intensity_grid,
                burned_grid=wildfire_state.burned_grid,
                smoke_grid=wildfire_state.smoke_grid,
            )
            view, ground_stats = apply_wildfire_effects_to_pil(
                view,
                wildfire_masks,
                config=self.wildfire_effect_config,
                include_burn=True,
                include_flame=True,
                include_smoke=False,
            )
            wildfire_stats = {
                "source": "simulation_fire_and_burned_grids",
                "smoke_rendered": True,
                "ground_and_flame": ground_stats,
            }

        truth: list[dict] = []
        truth_boxes: list[tuple[int, int, int, int]] = []
        for survivor in survivors:
            dx_world = survivor.world_xy[0] - drone.world_xy[0]
            dy_world = survivor.world_xy[1] - drone.world_xy[1]
            bbox = self._survivor_box(dx_world=dx_world, dy_world=dy_world, footprint_world=footprint_world)
            if bbox is None:
                continue
            human_asset_path, human_asset = self._asset_for_survivor(survivor.index)
            if human_asset is not None:
                self._paste_survivor(view, bbox, human_asset)
            truth_boxes.append(bbox)
            truth.append(
                {
                    "survivor_index": survivor.index,
                    "bbox_xyxy": list(bbox),
                    "true_world_xy": [survivor.world_xy[0], survivor.world_xy[1]],
                    "human_asset_path": str(human_asset_path) if human_asset_path is not None else None,
                }
            )

        if wildfire_masks is not None:
            view, smoke_stats = apply_wildfire_effects_to_pil(
                view,
                wildfire_masks,
                config=self.wildfire_effect_config,
                include_burn=False,
                include_flame=False,
                include_smoke=True,
            )
            if wildfire_stats is not None:
                wildfire_stats["smoke"] = smoke_stats

        detections = self.detector.detect_boxes(truth_boxes, image_size=self.image_size)
        detection_records = []
        for det_idx, detection in enumerate(detections.detections):
            center_px = detection.center_xy
            relative_world = self._pixel_center_to_relative_world(center_px, footprint_world=footprint_world)
            estimated_world = (
                drone.world_xy[0] + relative_world[0],
                drone.world_xy[1] + relative_world[1],
            )
            survivor_index = truth[det_idx]["survivor_index"] if det_idx < len(truth) else None
            detection_records.append(
                {
                    "class_name": detection.class_name,
                    "confidence": round(float(detection.confidence), 4),
                    "bbox_xyxy": list(detection.box),
                    "center_px": [round(center_px[0], 3), round(center_px[1], 3)],
                    "relative_to_drone_world": [round(relative_world[0], 6), round(relative_world[1], 6)],
                    "estimated_world_xy": [round(estimated_world[0], 6), round(estimated_world[1], 6)],
                    "estimated_cell": list(self._world_to_cell(estimated_world)),
                    "matched_survivor_index": survivor_index,
                }
            )

        saved_path = None
        if image_path is not None:
            out = Path(image_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            view.save(out)
            saved_path = str(out)

        return {
            "name": drone.name,
            "drone_index": drone.index,
            "x": round(float(drone.world_xy[0]), 6),
            "y": round(float(drone.world_xy[1]), 6),
            "altitude_agl": round(float(drone.altitude_agl), 6),
            "altitude_agl_m": round(float(altitude_m), 3),
            "footprint": round(float(footprint_world * 0.5), 6),
            "footprint_m": round(float(footprint_m), 3),
            "image_size_px": self.image_size,
            "source_crop_px": list(source_crop_size_px),
            "background_size_px": list(self.background_size_px),
            "background_gsd_m_per_px": [
                round(float(self.background_gsd_m_per_px[0]), 4),
                round(float(self.background_gsd_m_per_px[1]), 4),
            ],
            "image_path": saved_path,
            "wildfire_effects": wildfire_stats,
            "truth": truth,
            "detections": detection_records,
        }

    def render_survivor_preview(
        self,
        *,
        survivor: SimEntity,
        altitude_m: float = 20.0,
        image_path: str | Path | None = None,
    ) -> dict:
        """Render a centered approximate drone view for one survivor.

        The preview uses the same NAIP crop, camera FOV, image size, survivor
        scaling, and pasted SARD asset as normal drone CV frames. The only
        difference is that the synthetic camera is centered directly over the
        survivor so the preview exists even when no drone has reached the area.
        """

        pseudo_drone = SimDrone(
            index=-1,
            name=f"survivor_{survivor.index}_preview",
            world_xy=survivor.world_xy,
            altitude_agl=float(altitude_m) * self.sim_units_per_meter,
        )
        record = self.render_and_detect(drone=pseudo_drone, survivors=[survivor], image_path=image_path)
        asset_path, _asset = self._asset_for_survivor(survivor.index)
        return {
            "survivor_index": int(survivor.index),
            "image_path": record.get("image_path"),
            "altitude_m": round(float(altitude_m), 3),
            "footprint_m": record.get("footprint_m"),
            "image_size_px": record.get("image_size_px"),
            "source_crop_px": record.get("source_crop_px"),
            "background_gsd_m_per_px": record.get("background_gsd_m_per_px"),
            "human_asset_path": str(asset_path) if asset_path is not None else None,
            "truth": record.get("truth", []),
        }

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

    def _load_human_assets(
        self,
        *,
        human_asset_path: str | Path | None,
        human_assets_dir: str | Path | None,
        human_asset_list_path: str | Path | None,
    ) -> list[tuple[Path, Image.Image]]:
        paths: list[Path] = []
        directory = self._resolve(human_assets_dir) if human_assets_dir is not None else None
        if human_asset_list_path is not None:
            list_path = self._resolve(human_asset_list_path)
            paths = self._paths_from_asset_list(list_path, directory=directory)
        elif directory is not None:
            if directory.exists():
                paths = sorted(directory.glob("*.png"))
            elif human_asset_path is None:
                raise FileNotFoundError(f"Human assets directory not found: {directory}")
        if not paths and human_asset_path is not None:
            path = self._resolve(human_asset_path)
            if path.exists():
                paths = [path]
            else:
                raise FileNotFoundError(f"Human asset not found: {path}")
        return [(path, Image.open(path).convert("RGBA")) for path in paths]

    def _paths_from_asset_list(self, list_path: Path, *, directory: Path | None) -> list[Path]:
        if not list_path.exists():
            raise FileNotFoundError(f"Human asset review list not found: {list_path}")
        data = json.loads(list_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("accepted_assets") or data.get("accepted") or []
            if directory is None and data.get("asset_dir"):
                directory = self._resolve(data["asset_dir"])
        else:
            raise ValueError(f"Unsupported human asset list format: {list_path}")

        paths: list[Path] = []
        seen: set[Path] = set()
        for item in items:
            raw_path = item.get("path") if isinstance(item, dict) else item
            if raw_path is None:
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = directory / path if directory is not None and len(path.parts) == 1 else self._resolve(path)
            path = path.resolve()
            if path in seen:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Accepted human asset not found: {path}")
            paths.append(path)
            seen.add(path)
        if not paths:
            raise ValueError(f"No accepted human assets listed in {list_path}")
        return paths

    def _asset_for_survivor(self, survivor_index: int) -> tuple[Path | None, Image.Image | None]:
        if not self.human_assets:
            return None, None
        asset_idx = self._asset_order[int(survivor_index) % len(self._asset_order)]
        return self.human_assets[asset_idx]

    def _load_or_fetch_naip(
        self,
        *,
        naip_image_path: str | Path | None,
        cache_dir: str | Path,
        target_gsd_m: float | None,
        lazy_tile_cache: bool,
        naip_size: int,
        tiled: bool,
        tile_size: int,
    ) -> Path | None:
        if naip_image_path is not None:
            path = self._resolve(naip_image_path)
            if not path.exists():
                raise FileNotFoundError(f"NAIP image not found: {path}")
            return path

        if target_gsd_m is not None and tiled and lazy_tile_cache:
            self.tile_cache = NaipTileCache(
                bbox_lonlat=self.bbox_lonlat,
                out_dir=self._resolve(cache_dir),
                target_gsd_m=float(target_gsd_m),
                tile_size=int(tile_size),
                force=False,
            )
            return None

        if target_gsd_m is not None and tiled:
            return fetch_naip_tiled_image_for_gsd(
                bbox_lonlat=self.bbox_lonlat,
                out_dir=self._resolve(cache_dir),
                target_gsd_m=float(target_gsd_m),
                tile_size=int(tile_size),
                force=False,
            )

        fetcher = fetch_naip_tiled_image if tiled else fetch_naip_image
        kwargs = {
            "bbox_lonlat": self.bbox_lonlat,
            "out_dir": self._resolve(cache_dir),
            "size": int(naip_size),
            "force": False,
        }
        if tiled:
            kwargs["tile_size"] = int(tile_size)
        return fetcher(**kwargs)

    def _crop_background(
        self,
        center_world: tuple[float, float],
        crop_size_px: tuple[int, int],
        *,
        footprint_world: float,
    ) -> Image.Image:
        if self.tile_cache is not None:
            return self.tile_cache.crop_world(center_world=center_world, size_world=footprint_world)
        if self.background is None:
            raise RuntimeError("No NAIP background image is loaded")
        cx = (center_world[0] + 1.0) * 0.5 * self.background.width
        cy = (1.0 - (center_world[1] + 1.0) * 0.5) * self.background.height
        half_w = crop_size_px[0] * 0.5
        half_h = crop_size_px[1] * 0.5
        left = int(round(cx - half_w))
        top = int(round(cy - half_h))
        return self._crop_with_padding((left, top, left + crop_size_px[0], top + crop_size_px[1]))

    def _crop_with_padding(self, box: tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        crop = Image.new("RGB", (width, height), (0, 0, 0))
        source_box = (
            max(0, left),
            max(0, top),
            min(self.background.width, right),
            min(self.background.height, bottom),
        )
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            return crop
        crop.paste(self.background.crop(source_box), (source_box[0] - left, source_box[1] - top))
        return crop

    def _survivor_box(
        self,
        *,
        dx_world: float,
        dy_world: float,
        footprint_world: float,
    ) -> tuple[int, int, int, int] | None:
        half = footprint_world * 0.5
        if not (-half <= dx_world <= half and -half <= dy_world <= half):
            return None
        cx = self.image_size * 0.5 + (dx_world / footprint_world) * self.image_size
        cy = self.image_size * 0.5 - (dy_world / footprint_world) * self.image_size
        width_px = max(2, int(round(self.survivor_width_m * self.sim_units_per_meter / footprint_world * self.image_size)))
        height_px = max(2, int(round(self.survivor_height_m * self.sim_units_per_meter / footprint_world * self.image_size)))
        x1 = int(round(cx - width_px / 2))
        y1 = int(round(cy - height_px / 2))
        x2 = int(round(cx + width_px / 2))
        y2 = int(round(cy + height_px / 2))
        x1 = max(0, min(self.image_size - 1, x1))
        y1 = max(0, min(self.image_size - 1, y1))
        x2 = max(0, min(self.image_size, x2))
        y2 = max(0, min(self.image_size, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _paste_survivor(self, image: Image.Image, bbox: tuple[int, int, int, int], asset: Image.Image) -> None:
        if asset is None:
            return
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        resample = Image.Resampling.NEAREST if self.asset_resample == "nearest" else Image.Resampling.BILINEAR
        sprite = asset.resize((width, height), resample)
        rgb = ImageEnhance.Brightness(sprite.convert("RGB")).enhance(0.92)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
        sprite = Image.merge("RGBA", (*rgb.split(), sprite.getchannel("A")))
        sprite = sprite.rotate(self.survivor_rotation_deg, expand=True, resample=resample)
        px = x1 + width // 2 - sprite.width // 2
        py = y1 + height // 2 - sprite.height // 2
        image.paste(sprite, (px, py), sprite)

    def _pixel_center_to_relative_world(
        self,
        center_px: tuple[float, float],
        *,
        footprint_world: float,
    ) -> tuple[float, float]:
        cx, cy = center_px
        dx = ((cx / self.image_size) - 0.5) * footprint_world
        dy = (0.5 - (cy / self.image_size)) * footprint_world
        return float(dx), float(dy)

    def _world_to_cell(self, world_xy: tuple[float, float]) -> tuple[int, int]:
        x, y = world_xy
        gx = int(math.floor((x + 1.0) * 0.5 * self.grid_size))
        gy = int(math.floor((y + 1.0) * 0.5 * self.grid_size))
        return max(0, min(self.grid_size - 1, gx)), max(0, min(self.grid_size - 1, gy))
