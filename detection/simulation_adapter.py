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
        detector_backend: str = "preliminary",
        person_model: str = "yolov8n.pt",
        person_conf: float = 0.35,
        person_iou: float = 0.7,
        person_imgsz: int | None = None,
        person_tiled: bool = True,
        person_tile_grid: int = 2,
        person_tile_overlap: float = 0.2,
        person_match_iou: float = 0.15,
        person_device: str | None = None,
        person_augment: bool = False,
        adaptive_conf: bool = False,
        adaptive_conf_high_alt_m: float = 50.0,
        adaptive_conf_low_alt_m: float = 20.0,
        adaptive_conf_high_alt_threshold: float = 0.20,
        adaptive_conf_low_alt_threshold: float = 0.45,
        render_wildfire_effects: bool = True,
        wildfire_effect_seed: int | None = None,
        detection_mode: str = "cv",
        thermal_seed: int | None = None,
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

        # Real computer-vision detection backend. "preliminary" (default) echoes
        # renderer ground-truth boxes — fast, deterministic, no model download —
        # and stays the default for tests. "yolo" runs a real YOLOv8 person
        # detector over the rendered crop. Aerial survivors are tiny (~20-100 px),
        # well below YOLO's reliable floor, so small-object inference (tiling +
        # high imgsz) is enabled by default for the yolo backend.
        self.detector_backend = str(detector_backend).lower()
        if self.detector_backend not in ("preliminary", "yolo"):
            raise ValueError(f"detector_backend must be 'preliminary' or 'yolo', got {detector_backend!r}")
        # Prefer the OmniSearch-fine-tuned survivor detector when the caller
        # left the stock default and the trained weights exist. Stock COCO YOLO
        # is unreliable on top-down aerial survivors; the fine-tuned model
        # (scripts/train_survivor_detector.py) is in-distribution and far more
        # confident. Pass an explicit --cv-person-model to override.
        if str(person_model) == "yolov8n.pt":
            # Prefer the NAIP-trained model (real aerial backgrounds, far fewer
            # false positives on real terrain), then the procedural-trained
            # models, then stock.
            for candidate in ("survivor_naip_yolov8s.pt", "survivor_yolov8s.pt", "survivor_yolov8n.pt"):
                path = self.root / "models" / candidate
                if path.exists():
                    self.person_model_name = str(path)
                    break
            else:
                self.person_model_name = str(person_model)
        else:
            self.person_model_name = str(person_model)
        self.person_conf = float(person_conf)
        self.person_iou = float(person_iou)
        self.person_imgsz = int(person_imgsz) if person_imgsz else max(int(image_size), 1280)
        self.person_tiled = bool(person_tiled)
        self.person_tile_grid = max(int(person_tile_grid), 1)
        self.person_tile_overlap = min(max(float(person_tile_overlap), 0.0), 0.9)
        self.person_match_iou = float(person_match_iou)
        self.person_device = person_device
        self.person_augment = bool(person_augment)
        self.adaptive_conf = bool(adaptive_conf)
        self.adaptive_conf_high_alt_m = float(adaptive_conf_high_alt_m)
        self.adaptive_conf_low_alt_m = float(adaptive_conf_low_alt_m)
        self.adaptive_conf_high_alt_threshold = float(adaptive_conf_high_alt_threshold)
        self.adaptive_conf_low_alt_threshold = float(adaptive_conf_low_alt_threshold)
        self._person_detector = None  # lazily built on first yolo detection

        # UGV-specific detectors for ground confirmation. When trained weights
        # exist (from scripts/train_ugv_detector.py), use them for the appropriate
        # camera mode instead of the drone aerial model.
        self.ugv_front_model_name = self._resolve_ugv_model("front")
        self.ugv_mast_model_name = self._resolve_ugv_model("mast")
        self._ugv_front_detector = None
        self._ugv_mast_detector = None

        # Legacy fallback: if no UGV-specific models exist, ground confirmation
        # uses the drone model (previous behavior).
        self.ground_person_model_name = self.person_model_name
        self._ground_detector = None

        # Multi-object tracker for temporal consistency across frames.
        # Requires min_hits consecutive detections before confirming a track,
        # which suppresses transient false positives.
        self._tracker = None
        self._tracker_enabled = False

        # Detection mode: "cv" (pure CV, default), "thermal" (simulated TIR only),
        # "cv+thermal" (sensor fusion — both sensors run, results merged).
        self.detection_mode = str(detection_mode).lower()
        if self.detection_mode not in ("cv", "thermal", "cv+thermal"):
            raise ValueError(f"detection_mode must be 'cv', 'thermal', or 'cv+thermal', got {detection_mode!r}")
        self._thermal_model = None
        self._thermal_seed = int(thermal_seed if thermal_seed is not None else seed)

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

    def enable_tracking(
        self,
        *,
        min_hits: int = 2,
        lost_track_buffer: int = 5,
        track_activation_threshold: float = 0.25,
    ):
        """Enable multi-object tracking for temporal FP suppression.

        When enabled, detections from `render_and_detect` are passed through
        ByteTrack. Only tracks with `min_hits` consecutive detections are
        reported as confirmed, filtering out transient false positives.
        """
        from .tracker import SurvivorTracker
        self._tracker = SurvivorTracker(
            min_hits=min_hits,
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
        )
        self._tracker_enabled = True

    def disable_tracking(self):
        """Disable tracking and reset tracker state."""
        self._tracker_enabled = False
        if self._tracker is not None:
            self._tracker.reset()

    def reset_tracker(self):
        """Reset tracker state (call between episodes)."""
        if self._tracker is not None:
            self._tracker.reset()

    def _get_thermal_model(self):
        """Lazily initialize the thermal sensor model."""
        if self._thermal_model is None:
            from .thermal_model import ThermalSensorModel, ThermalSensorConfig
            self._thermal_model = ThermalSensorModel(
                ThermalSensorConfig(seed=self._thermal_seed)
            )
        return self._thermal_model

    def _run_thermal_detection(
        self,
        *,
        drone: "SimDrone",
        survivors: list,
        wildfire_state: "SimWildfireState | None",
        altitude_m: float,
    ) -> list[dict]:
        """Run simulated thermal detection on the current frame."""
        thermal = self._get_thermal_model()
        survivor_dicts = [
            {"index": s.index, "world_xy": s.world_xy} for s in survivors
        ]
        return thermal.detect_survivors(
            drone_xy=drone.world_xy,
            drone_altitude_m=altitude_m,
            fov_deg=self.fov_deg,
            survivors=survivor_dicts,
            fire_grid=wildfire_state.fire_grid if wildfire_state else None,
            fire_intensity_grid=wildfire_state.fire_intensity_grid if wildfire_state else None,
            burned_grid=wildfire_state.burned_grid if wildfire_state else None,
            smoke_grid=wildfire_state.smoke_grid if wildfire_state else None,
            sim_units_per_meter=self.sim_units_per_meter,
            grid_size=self.grid_size,
        )

    def _altitude_adjusted_conf(self, altitude_m: float) -> float:
        """Compute confidence threshold interpolated by altitude.

        At high altitude (small survivors, low YOLO confidence), use a lower
        threshold to maintain recall. At low altitude (large, high-confidence
        detections), use a higher threshold to suppress false positives.
        """
        if not self.adaptive_conf:
            return self.person_conf
        low_alt = self.adaptive_conf_low_alt_m
        high_alt = self.adaptive_conf_high_alt_m
        low_thresh = self.adaptive_conf_low_alt_threshold
        high_thresh = self.adaptive_conf_high_alt_threshold
        if altitude_m <= low_alt:
            return low_thresh
        if altitude_m >= high_alt:
            return high_thresh
        t = (altitude_m - low_alt) / (high_alt - low_alt)
        return low_thresh + t * (high_thresh - low_thresh)

    def _resolve_ugv_model(self, camera: str) -> str | None:
        """Find trained UGV model weights for the given camera mode."""
        candidates = {
            "front": ["ugv_front_yolov8s.pt", "ugv_front_yolov8n.pt"],
            "mast": ["ugv_mast_yolov8n.pt", "ugv_mast_yolov8s.pt"],
        }
        for name in candidates.get(camera, []):
            path = self.root / "models" / name
            if path.exists():
                return str(path)
        return None

    def _get_person_detector(self):
        """Lazily construct the YOLOv8 person detector on first use."""
        if self._person_detector is None:
            from .person_detector import PersonDetector
            self._person_detector = PersonDetector(
                model_name=self.person_model_name,
                conf=self.person_conf,
                iou=self.person_iou,
                device=self.person_device,
                augment=self.person_augment,
            )
        return self._person_detector

    def _get_ugv_detector(self, camera: str):
        """Lazily construct UGV-specific detector for front or mast camera."""
        from .person_detector import PersonDetector
        if camera == "front":
            if self._ugv_front_detector is None and self.ugv_front_model_name:
                self._ugv_front_detector = PersonDetector(
                    model_name=self.ugv_front_model_name,
                    conf=self.person_conf,
                    iou=self.person_iou,
                    device=self.person_device,
                    augment=self.person_augment,
                )
            return self._ugv_front_detector
        else:
            if self._ugv_mast_detector is None and self.ugv_mast_model_name:
                self._ugv_mast_detector = PersonDetector(
                    model_name=self.ugv_mast_model_name,
                    conf=self.person_conf,
                    iou=self.person_iou,
                    device=self.person_device,
                    augment=self.person_augment,
                )
            return self._ugv_mast_detector

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, boxes, confs, iou_thresh):
        """Greedy non-maximum suppression (merges duplicate tile detections)."""
        order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            order = [j for j in order if self._iou(boxes[i], boxes[j]) < iou_thresh]
        return [(boxes[i], confs[i]) for i in keep]

    def _detect_people_cv(self, view: Image.Image):
        """Run real YOLO person detection over the rendered crop.

        Aerial survivors occupy only tens of pixels — below YOLO's reliable
        detection floor. When ``person_tiled`` is set we slice the frame into an
        overlapping grid and run each tile at high ``imgsz`` so each survivor
        gets more effective resolution, then map boxes back to full-frame pixel
        coordinates and de-duplicate with NMS.
        """
        detector = self._get_person_detector()
        W, H = view.size
        boxes: list[tuple[float, float, float, float]] = []
        confs: list[float] = []

        if not self.person_tiled or self.person_tile_grid <= 1:
            res = detector.model.predict(
                source=view, classes=[0], conf=self.person_conf, iou=self.person_iou,
                imgsz=self.person_imgsz, device=self.person_device,
                augment=self.person_augment, verbose=False,
            )
            r = res[0] if res else None
            if r is not None and r.boxes is not None and len(r.boxes):
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    confs.append(float(c))
        else:
            g = self.person_tile_grid
            step_x, step_y = W / g, H / g
            ov_x, ov_y = step_x * self.person_tile_overlap, step_y * self.person_tile_overlap
            for ty in range(g):
                for tx in range(g):
                    x0 = max(0, int(tx * step_x - ov_x)); y0 = max(0, int(ty * step_y - ov_y))
                    x1 = min(W, int((tx + 1) * step_x + ov_x)); y1 = min(H, int((ty + 1) * step_y + ov_y))
                    tile = view.crop((x0, y0, x1, y1))
                    res = detector.model.predict(
                        source=tile, classes=[0], conf=self.person_conf, iou=self.person_iou,
                        imgsz=self.person_imgsz, device=self.person_device,
                        augment=self.person_augment, verbose=False,
                    )
                    r = res[0] if res else None
                    if r is None or r.boxes is None or not len(r.boxes):
                        continue
                    for (bx1, by1, bx2, by2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        boxes.append((float(bx1) + x0, float(by1) + y0, float(bx2) + x0, float(by2) + y0))
                        confs.append(float(c))

        merged = self._nms(boxes, confs, self.person_iou)
        # Clip every box to the actual image boundary.  YOLO can return slightly
        # out-of-bounds coordinates due to letter-boxing artefacts in the resize
        # path, and tiled inference adds per-tile offsets that can push boxes past
        # the right/bottom edge.  Drop degenerate boxes that collapse to zero area
        # after clipping rather than passing them to the IoU matcher.
        clipped: list[tuple[tuple, float]] = []
        for (bx1, by1, bx2, by2), c in merged:
            bx1 = max(0.0, min(float(W), bx1))
            by1 = max(0.0, min(float(H), by1))
            bx2 = max(0.0, min(float(W), bx2))
            by2 = max(0.0, min(float(H), by2))
            if bx2 > bx1 and by2 > by1:
                clipped.append(((bx1, by1, bx2, by2), c))
        clipped.sort(key=lambda bc: bc[1], reverse=True)
        return clipped

    def _match_truth_index(self, box, truth_boxes, truth):
        """Return the survivor_index of the best-overlapping ground-truth box."""
        best_idx, best_iou = None, self.person_match_iou
        for i, tb in enumerate(truth_boxes):
            iou = self._iou(box, tb)
            if iou >= best_iou:
                best_iou, best_idx = iou, i
        if best_idx is None:
            return None
        return truth[best_idx]["survivor_index"] if best_idx < len(truth) else None

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
        if self.render_wildfire_effects and wildfire_state is not None:
            grid_h, grid_w = wildfire_state.fire_grid.shape
            half = footprint_world * 0.5
            cx, cy = drone.world_xy
            c0 = max(0, int(np.floor((cx - half + 1.0) * 0.5 * grid_w)))
            c1 = min(grid_w, int(np.ceil((cx + half + 1.0) * 0.5 * grid_w)))
            r0 = max(0, int(np.floor((cy - half + 1.0) * 0.5 * grid_h)))
            r1 = min(grid_h, int(np.ceil((cy + half + 1.0) * 0.5 * grid_h)))
            footprint_has_fire = (
                wildfire_state.fire_grid[r0:r1, c0:c1].any()
                or wildfire_state.burned_grid[r0:r1, c0:c1].any()
            )
            if footprint_has_fire:
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
        survivors_list = list(survivors)
        for survivor in survivors_list:
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

        # Build detections either from the renderer ground truth (preliminary
        # stub) or by running real computer vision over the rendered crop.
        if self.detector_backend == "yolo":
            # Apply adaptive confidence threshold based on drone altitude.
            prev_conf = self.person_conf
            self.person_conf = self._altitude_adjusted_conf(altitude_m)
            cv_boxes = self._detect_people_cv(view)
            self.person_conf = prev_conf
            detection_records = []
            for box, conf in cv_boxes:
                cx = (box[0] + box[2]) * 0.5
                cy = (box[1] + box[3]) * 0.5
                center_px = (cx, cy)
                relative_world = self._pixel_center_to_relative_world(center_px, footprint_world=footprint_world)
                estimated_world = (
                    drone.world_xy[0] + relative_world[0],
                    drone.world_xy[1] + relative_world[1],
                )
                # Match the CV box to a ground-truth survivor by overlap so the
                # downstream pipeline still knows which survivor (if any) this is.
                survivor_index = self._match_truth_index(box, truth_boxes, truth)
                detection_records.append(
                    {
                        "class_name": "person",
                        "confidence": round(float(conf), 4),
                        "bbox_xyxy": [int(v) for v in box],
                        "center_px": [round(center_px[0], 3), round(center_px[1], 3)],
                        "relative_to_drone_world": [round(relative_world[0], 6), round(relative_world[1], 6)],
                        "estimated_world_xy": [round(estimated_world[0], 6), round(estimated_world[1], 6)],
                        "estimated_cell": list(self._world_to_cell(estimated_world)),
                        "matched_survivor_index": survivor_index,
                    }
                )
        else:
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

        # Thermal / fusion detection modes.
        # In "thermal" mode, CV detections are replaced by thermal results.
        # In "cv+thermal" mode, both run and results are fused.
        thermal_detections = None
        if self.detection_mode in ("thermal", "cv+thermal"):
            thermal_detections = self._run_thermal_detection(
                drone=drone,
                survivors=list(survivors_list),
                wildfire_state=wildfire_state,
                altitude_m=altitude_m,
            )

        if self.detection_mode == "thermal":
            # Replace CV detections entirely with thermal results
            detection_records = []
            for td in thermal_detections:
                if td.get("detected", False):
                    detection_records.append({
                        "class_name": "person",
                        "confidence": td.get("confidence", 0.5),
                        "bbox_xyxy": [0, 0, 0, 0],  # No pixel box for thermal
                        "center_px": [self.image_size / 2, self.image_size / 2],
                        "estimated_world_xy": td.get("estimated_world_xy", td.get("true_world_xy")),
                        "matched_survivor_index": td.get("survivor_index"),
                        "sensor": "thermal",
                        "delta_t_k": td.get("delta_t_k"),
                        "thermal_crossover": td.get("thermal_crossover", False),
                        "detection_probability": td.get("detection_probability"),
                    })
        elif self.detection_mode == "cv+thermal":
            from .thermal_model import fuse_cv_thermal
            fused = fuse_cv_thermal(
                cv_detections=detection_records,
                thermal_detections=thermal_detections,
                fusion_mode="union",
            )
            detection_records = fused

        # Apply multi-object tracking if enabled. Tracks accumulate hits over
        # consecutive frames; only tracks exceeding min_hits are "confirmed",
        # filtering transient false positives.
        tracking_info = None
        if self._tracker_enabled and self._tracker is not None:
            tracked = self._tracker.update(detection_records)
            confirmed_tracks = self._tracker.get_confirmed_tracks(tracked)
            tracking_info = {
                "active_tracks": len(tracked),
                "confirmed_tracks": len(confirmed_tracks),
                "frame": self._tracker.frame_count,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "bbox_xyxy": list(t.bbox_xyxy),
                        "confidence": round(t.confidence, 4),
                        "hit_count": t.hit_count,
                        "confirmed": t.confirmed,
                    }
                    for t in tracked
                ],
            }
            # Annotate detection records with track IDs
            for det, trk in zip(detection_records, tracked):
                det["track_id"] = trk.track_id
                det["track_confirmed"] = trk.confirmed
                det["track_hits"] = trk.hit_count

        saved_path = None
        if image_path is not None:
            out = Path(image_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            annotated = view.copy()
            if detection_records:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(annotated)
                for d in detection_records:
                    box = d.get("bbox_xyxy", [0, 0, 0, 0])
                    if box != [0, 0, 0, 0]:
                        x1, y1, x2, y2 = box
                        color = (0, 255, 0) if d.get("track_confirmed", True) else (255, 165, 0)
                        if d.get("sensor") == "thermal" or d.get("fusion_source") == "thermal_only":
                            color = (255, 0, 255)  # Magenta for thermal-only
                        elif d.get("fusion_source") == "both":
                            color = (0, 255, 255)  # Cyan for fused
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        label = f"{d['confidence']:.2f}"
                        if "track_id" in d:
                            label = f"T{d['track_id']} {label}"
                        draw.text((x1, max(0, y1 - 12)), label, fill=color)
            annotated.save(out)
            saved_path = str(out)

        result = {
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
            "detection_mode": self.detection_mode,
            "image_path": saved_path,
            "wildfire_effects": wildfire_stats,
            "truth": truth,
            "detections": detection_records,
        }
        if thermal_detections is not None:
            result["thermal_raw"] = thermal_detections
        if tracking_info is not None:
            result["tracking"] = tracking_info
        return result

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

    def render_ground_confirmation(
        self,
        *,
        robot: SimEntity,
        survivor: SimEntity,
        wildfire_state: SimWildfireState | None = None,
        view_radius_m: float = 4.0,
        camera_mode: str = "auto",
        image_path: str | Path | None = None,
    ) -> dict:
        """Render a ground robot's close-range confirmation view and detect.

        A ground robot confirms a survivor from a few metres away, so the
        survivor fills a large fraction of the frame — the opposite of the
        drone's tiny aerial target. We model that as a small-footprint camera
        view (``2 * view_radius_m`` across) centred on the survivor, reusing the
        full render + CV pipeline. Because the survivor is large, detection is
        high-confidence. (The sim is top-down, so this is a close-range
        approximation of an eye-level confirmation, not a true perspective view.)

        Parameters
        ----------
        camera_mode : str
            Which UGV camera model to use for detection:
            - "auto": use mast model if distance <= 15m, front model otherwise
            - "front": UGV forward-looking camera (5-30m range)
            - "mast": UGV elevated mast camera (3-15m range)
            - "drone": legacy behavior using the drone aerial model

        Returns the per-frame detection record plus a ``confirmed`` flag and the
        robot-survivor distance in metres.
        """
        dx_m = (robot.world_xy[0] - survivor.world_xy[0]) / self.sim_units_per_meter
        dy_m = (robot.world_xy[1] - survivor.world_xy[1]) / self.sim_units_per_meter
        distance_m = math.hypot(dx_m, dy_m)

        # Resolve camera mode
        if camera_mode == "auto":
            if distance_m <= 15.0 and self.ugv_mast_model_name:
                resolved_camera = "mast"
            elif self.ugv_front_model_name:
                resolved_camera = "front"
            else:
                resolved_camera = "drone"
        else:
            resolved_camera = camera_mode

        footprint_m = max(2.0 * float(view_radius_m), 1.0)
        altitude_m = footprint_m / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        pseudo_drone = SimDrone(
            index=-1,
            name=f"ground_confirm_survivor_{survivor.index}",
            world_xy=survivor.world_xy,
            altitude_agl=float(altitude_m) * self.sim_units_per_meter,
        )
        # Close-range: disable tiling and swap in the appropriate detector.
        prev_tiled = self.person_tiled
        prev_detector = self._person_detector
        prev_model = self.person_model_name
        self.person_tiled = False

        if resolved_camera in ("front", "mast"):
            ugv_det = self._get_ugv_detector(resolved_camera)
            if ugv_det is not None:
                self._person_detector = ugv_det
                model_name = (self.ugv_front_model_name if resolved_camera == "front"
                              else self.ugv_mast_model_name)
                self.person_model_name = model_name or self.person_model_name
            elif self.ground_person_model_name != self.person_model_name:
                if self._ground_detector is None:
                    from .person_detector import PersonDetector
                    self._ground_detector = PersonDetector(
                        model_name=self.ground_person_model_name, conf=self.person_conf,
                        iou=self.person_iou, device=self.person_device,
                    )
                self._person_detector = self._ground_detector
                self.person_model_name = self.ground_person_model_name
        elif self.ground_person_model_name != self.person_model_name:
            if self._ground_detector is None:
                from .person_detector import PersonDetector
                self._ground_detector = PersonDetector(
                    model_name=self.ground_person_model_name, conf=self.person_conf,
                    iou=self.person_iou, device=self.person_device,
                )
            self._person_detector = self._ground_detector
            self.person_model_name = self.ground_person_model_name

        try:
            record = self.render_and_detect(
                drone=pseudo_drone, survivors=[survivor],
                wildfire_state=wildfire_state, image_path=image_path,
            )
        finally:
            self.person_tiled = prev_tiled
            self._person_detector = prev_detector
            self.person_model_name = prev_model

        confirmed = any(
            d.get("matched_survivor_index") == survivor.index for d in record.get("detections", [])
        )
        confidence = max(
            (float(d["confidence"]) for d in record.get("detections", [])
             if d.get("matched_survivor_index") == survivor.index),
            default=0.0,
        )
        return {
            "agent": "ground_robot",
            "robot_index": int(robot.index),
            "survivor_index": int(survivor.index),
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 4),
            "distance_m": round(float(distance_m), 3),
            "view_radius_m": round(float(view_radius_m), 3),
            "camera_mode": resolved_camera,
            "image_path": record.get("image_path"),
            "detections": record.get("detections", []),
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
