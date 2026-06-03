"""Standalone drone-camera renderer for CV integration experiments.

This module does not depend on VMAS. It takes a cached terrain grid plus a small
synthetic scene state, renders a top-down drone camera crop, and records the
coordinate transforms needed to map detections back into terrain cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


LAND_COVER_COLORS = {
    0: (181, 150, 101),  # road
    1: (71, 120, 61),   # open
    2: (49, 90, 46),    # brush
    3: (32, 61, 36),    # forest
    4: (85, 89, 99),    # rock
    5: (37, 99, 235),   # water
}


@dataclass(frozen=True)
class SurvivorObject:
    """Synthetic survivor placed by terrain cell or continuous world position."""

    cell: tuple[int, int] | None = None
    label: str = "survivor"
    visible: bool = True
    world_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class FireCell:
    """Synthetic active-fire cell in terrain-grid coordinates."""

    cell: tuple[int, int]
    intensity: float = 1.0


@dataclass(frozen=True)
class SmokeCell:
    """Synthetic smoke cell in terrain-grid coordinates."""

    cell: tuple[int, int]
    density: float = 0.5


@dataclass
class CameraDetection:
    """Ground-truth object rendered into the camera crop."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    source_cell: tuple[int, int]
    mapped_cell: tuple[int, int]
    world_xy: tuple[float, float]

    def to_yolo_line(self, class_id: int, image_size: int) -> str:
        x1, y1, x2, y2 = self.bbox_xyxy
        cx = ((x1 + x2) * 0.5) / image_size
        cy = ((y1 + y2) * 0.5) / image_size
        w = max(x2 - x1, 0) / image_size
        h = max(y2 - y1, 0) / image_size
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass
class CameraRender:
    """Rendered crop plus metadata needed for validation and downstream CV."""

    image: Image.Image
    detections: list[CameraDetection] = field(default_factory=list)
    drone_cell: tuple[int, int] = (0, 0)
    drone_world_xy: tuple[float, float] = (0.0, 0.0)
    crop_world_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    crop_radius_world: float = 0.0


class DroneCameraRenderer:
    """Render top-down drone camera crops from a terrain cache.

    The first version assumes a nadir/top-down square crop. Altitude controls
    crop radius via ``altitude_agl * tan(FOV/2)``. This intentionally matches
    the simulator's current abstract camera-footprint model.
    """

    def __init__(
        self,
        terrain_cache: str | Path,
        *,
        global_image_path: str | Path | None = None,
        global_image_size: int = 2048,
        camera_image_size: int = 512,
        resize_camera_crop: bool = True,
        x_semidim: float = 1.0,
        y_semidim: float = 1.0,
    ):
        self.terrain_cache = Path(terrain_cache)
        self.global_image_size = int(global_image_size)
        self.camera_image_size = int(camera_image_size)
        self.resize_camera_crop = bool(resize_camera_crop)
        self.x_semidim = float(x_semidim)
        self.y_semidim = float(y_semidim)

        with np.load(self.terrain_cache, allow_pickle=False) as data:
            self.land_cover = np.asarray(data["land_cover"]).astype(np.int64)
            self.elevation = np.asarray(data["elevation"]).astype(np.float32)
            self.grid_size = int(self.land_cover.shape[0])
            self.source = str(data["source"].item()) if "source" in data else str(self.terrain_cache)
            if "bbox" in data:
                self.bbox_lonlat = tuple(float(v) for v in np.asarray(data["bbox"]).tolist())
            else:
                self.bbox_lonlat = None

        self.global_image = self._load_global_image(global_image_path)

    # ------------------------------------------------------------------
    # Public coordinate helpers
    # ------------------------------------------------------------------
    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        gx, gy = self._clamp_cell(cell)
        x = -self.x_semidim + (gx + 0.5) * (2.0 * self.x_semidim / self.grid_size)
        y = -self.y_semidim + (gy + 0.5) * (2.0 * self.y_semidim / self.grid_size)
        return float(x), float(y)

    def world_to_cell(self, world_xy: tuple[float, float]) -> tuple[int, int]:
        x, y = world_xy
        gx = int(math.floor((x + self.x_semidim) / (2.0 * self.x_semidim) * self.grid_size))
        gy = int(math.floor((y + self.y_semidim) / (2.0 * self.y_semidim) * self.grid_size))
        return self._clamp_cell((gx, gy))

    def world_to_global_pixel(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        x, y = world_xy
        u = (x + self.x_semidim) / (2.0 * self.x_semidim)
        v = 1.0 - ((y + self.y_semidim) / (2.0 * self.y_semidim))
        scale = self.global_image_size - 1
        return float(u * scale), float(v * scale)

    def global_pixel_to_world(self, pixel_xy: tuple[float, float]) -> tuple[float, float]:
        px, py = pixel_xy
        scale = self.global_image_size - 1
        u = px / scale
        v = py / scale
        x = -self.x_semidim + u * (2.0 * self.x_semidim)
        y = -self.y_semidim + (1.0 - v) * (2.0 * self.y_semidim)
        return float(x), float(y)

    def crop_pixel_to_world(
        self,
        crop_pixel_xy: tuple[float, float],
        crop_world_bounds: tuple[float, float, float, float],
        image_size: int | None = None,
    ) -> tuple[float, float]:
        west, south, east, north = crop_world_bounds
        px, py = crop_pixel_xy
        size = int(image_size or self.camera_image_size)
        u = px / max(size - 1, 1)
        v = py / max(size - 1, 1)
        x = west + u * (east - west)
        y = north - v * (north - south)
        return float(x), float(y)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(
        self,
        *,
        drone_cell: tuple[int, int],
        altitude_agl: float,
        fov_deg: float = 65.0,
        survivors: Iterable[SurvivorObject] = (),
        fire_cells: Iterable[FireCell] = (),
        smoke_cells: Iterable[SmokeCell] = (),
    ) -> CameraRender:
        drone_cell = self._clamp_cell(drone_cell)
        drone_world = self.cell_to_world(drone_cell)
        radius = max(float(altitude_agl), 1e-6) * math.tan(math.radians(float(fov_deg)) / 2.0)
        radius = min(radius, max(self.x_semidim, self.y_semidim))
        crop_bounds = (
            max(-self.x_semidim, drone_world[0] - radius),
            max(-self.y_semidim, drone_world[1] - radius),
            min(self.x_semidim, drone_world[0] + radius),
            min(self.y_semidim, drone_world[1] + radius),
        )
        crop = self._crop_global(crop_bounds).convert("RGBA")
        draw = ImageDraw.Draw(crop, "RGBA")
        image_size = crop.size[0]

        for smoke in smoke_cells:
            self._draw_smoke_cell(draw, crop, crop_bounds, smoke, image_size=image_size)
        for fire in fire_cells:
            self._draw_fire_cell(draw, crop_bounds, fire, image_size=image_size)

        detections: list[CameraDetection] = []
        for survivor in survivors:
            detection = self._draw_survivor(draw, crop_bounds, survivor, image_size=image_size)
            if detection is not None:
                detections.append(detection)

        return CameraRender(
            image=crop.convert("RGB"),
            detections=detections,
            drone_cell=drone_cell,
            drone_world_xy=drone_world,
            crop_world_bounds=crop_bounds,
            crop_radius_world=float(radius),
        )

    def save_yolo_labels(self, render: CameraRender, path: str | Path, class_map: dict[str, int] | None = None) -> Path:
        class_map = class_map or {"survivor": 0}
        image_size = render.image.size[0]
        lines = [
            det.to_yolo_line(class_map[det.class_name], image_size)
            for det in render.detections
            if det.class_name in class_map
        ]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _terrain_image(self) -> Image.Image:
        colors = np.zeros((*self.land_cover.shape, 3), dtype=np.uint8)
        for value, color in LAND_COVER_COLORS.items():
            colors[self.land_cover == value] = color
        # Terrain row 0 is low world-y; image row 0 is high world-y.
        colors = np.flipud(colors)
        return Image.fromarray(colors, "RGB").resize(
            (self.global_image_size, self.global_image_size),
            Image.Resampling.NEAREST,
        )

    def _load_global_image(self, global_image_path: str | Path | None) -> Image.Image:
        if global_image_path is None:
            return self._terrain_image()
        image = Image.open(global_image_path).convert("RGB")
        if image.size != (self.global_image_size, self.global_image_size):
            image = image.resize((self.global_image_size, self.global_image_size), Image.Resampling.BILINEAR)
        return image

    def _crop_global(self, bounds: tuple[float, float, float, float]) -> Image.Image:
        west, south, east, north = bounds
        px_w, py_n = self.world_to_global_pixel((west, north))
        px_e, py_s = self.world_to_global_pixel((east, south))
        box = (
            int(round(px_w)),
            int(round(py_n)),
            int(round(px_e)),
            int(round(py_s)),
        )
        crop = self.global_image.crop(box)
        if not self.resize_camera_crop:
            return crop
        return crop.resize((self.camera_image_size, self.camera_image_size), Image.Resampling.BILINEAR)

    def _cell_bounds_world(self, cell: tuple[int, int]) -> tuple[float, float, float, float]:
        gx, gy = self._clamp_cell(cell)
        west = -self.x_semidim + gx * (2.0 * self.x_semidim / self.grid_size)
        east = -self.x_semidim + (gx + 1) * (2.0 * self.x_semidim / self.grid_size)
        south = -self.y_semidim + gy * (2.0 * self.y_semidim / self.grid_size)
        north = -self.y_semidim + (gy + 1) * (2.0 * self.y_semidim / self.grid_size)
        return west, south, east, north

    def _world_bounds_to_crop_box(
        self,
        world_bounds: tuple[float, float, float, float],
        crop_bounds: tuple[float, float, float, float],
        image_size: int | None = None,
    ) -> tuple[int, int, int, int] | None:
        west, south, east, north = world_bounds
        crop_west, crop_south, crop_east, crop_north = crop_bounds
        clipped_w = max(west, crop_west)
        clipped_e = min(east, crop_east)
        clipped_s = max(south, crop_south)
        clipped_n = min(north, crop_north)
        if clipped_w >= clipped_e or clipped_s >= clipped_n:
            return None
        size = int(image_size or self.camera_image_size)
        x1 = int(round((clipped_w - crop_west) / (crop_east - crop_west) * (size - 1)))
        x2 = int(round((clipped_e - crop_west) / (crop_east - crop_west) * (size - 1)))
        y1 = int(round((crop_north - clipped_n) / (crop_north - crop_south) * (size - 1)))
        y2 = int(round((crop_north - clipped_s) / (crop_north - crop_south) * (size - 1)))
        return self._clamp_box((x1, y1, x2, y2), image_size=size)

    def _draw_fire_cell(
        self,
        draw: ImageDraw.ImageDraw,
        crop_bounds,
        fire: FireCell,
        *,
        image_size: int | None = None,
    ) -> None:
        box = self._world_bounds_to_crop_box(self._cell_bounds_world(fire.cell), crop_bounds, image_size=image_size)
        if box is None:
            return
        alpha = int(90 + 130 * max(0.0, min(float(fire.intensity), 1.0)))
        draw.rectangle(box, fill=(245, 96, 36, alpha), outline=(255, 230, 110, min(alpha + 30, 255)), width=1)

    def _draw_smoke_cell(
        self,
        draw: ImageDraw.ImageDraw,
        crop: Image.Image,
        crop_bounds,
        smoke: SmokeCell,
        *,
        image_size: int | None = None,
    ) -> None:
        box = self._world_bounds_to_crop_box(self._cell_bounds_world(smoke.cell), crop_bounds, image_size=image_size)
        if box is None:
            return
        density = max(0.0, min(float(smoke.density), 1.0))
        alpha = int(35 + 120 * density)
        overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay, "RGBA")
        pad = max(4, int((box[2] - box[0]) * 0.35))
        odraw.ellipse((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(116, 121, 126, alpha))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(3, pad // 3)))
        crop.alpha_composite(overlay)

    def _draw_survivor(
        self,
        draw: ImageDraw.ImageDraw,
        crop_bounds: tuple[float, float, float, float],
        survivor: SurvivorObject,
        *,
        image_size: int | None = None,
    ) -> CameraDetection | None:
        if not survivor.visible:
            return None
        world, source_cell = self._survivor_world_and_cell(survivor)
        crop_west, crop_south, crop_east, crop_north = crop_bounds
        if not (crop_west <= world[0] <= crop_east and crop_south <= world[1] <= crop_north):
            return None

        size = int(image_size or self.camera_image_size)
        px = int(round((world[0] - crop_west) / (crop_east - crop_west) * (size - 1)))
        py = int(round((crop_north - world[1]) / (crop_north - crop_south) * (size - 1)))
        marker_h = max(16, int(size * 0.055))
        marker_w = max(10, int(marker_h * 0.55))
        x1 = px - marker_w // 2
        y1 = py - marker_h // 2
        x2 = x1 + marker_w
        y2 = y1 + marker_h
        box = self._clamp_box((x1, y1, x2, y2), image_size=size)
        if box[2] <= box[0] or box[3] <= box[1]:
            return None

        # Simple high-contrast survivor marker. This is intentionally not the
        # final photorealistic asset; it validates geometry and labeling first.
        cx = (box[0] + box[2]) // 2
        head_r = max(3, marker_w // 4)
        draw.ellipse((cx - head_r, box[1], cx + head_r, box[1] + 2 * head_r), fill=(244, 194, 64, 255))
        draw.rounded_rectangle((box[0], box[1] + 2 * head_r, box[2], box[3]), radius=3, fill=(244, 194, 64, 245))
        draw.rectangle(box, outline=(255, 255, 255, 230), width=2)

        mapped_world = self.crop_pixel_to_world(
            ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
            crop_bounds,
            image_size=size,
        )
        mapped_cell = self.world_to_cell(mapped_world)
        return CameraDetection(
            class_name=survivor.label,
            confidence=1.0,
            bbox_xyxy=box,
            source_cell=source_cell,
            mapped_cell=mapped_cell,
            world_xy=world,
        )

    def _survivor_world_and_cell(self, survivor: SurvivorObject) -> tuple[tuple[float, float], tuple[int, int]]:
        if survivor.world_xy is not None:
            world = (float(survivor.world_xy[0]), float(survivor.world_xy[1]))
            return world, self.world_to_cell(world)
        if survivor.cell is None:
            raise ValueError("SurvivorObject requires either cell or world_xy.")
        cell = self._clamp_cell(survivor.cell)
        return self.cell_to_world(cell), cell

    def _clamp_cell(self, cell: tuple[int, int]) -> tuple[int, int]:
        gx = max(0, min(self.grid_size - 1, int(cell[0])))
        gy = max(0, min(self.grid_size - 1, int(cell[1])))
        return gx, gy

    def _clamp_box(
        self,
        box: tuple[int, int, int, int],
        *,
        image_size: int | None = None,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        size = int(image_size or self.camera_image_size) - 1
        x1 = max(0, min(size, int(x1)))
        y1 = max(0, min(size, int(y1)))
        x2 = max(0, min(size, int(x2)))
        y2 = max(0, min(size, int(y2)))
        return x1, y1, x2, y2
