"""Motion detection via temporal frame differencing for drone-based SAR.

Detects new objects appearing between consecutive rendered frames as the drone
moves. Since survivors are static, the drone's movement causes them to appear
in new positions or become newly visible. Frame differencing highlights these
changes against the moving background.

This module works on rendered PIL images from the SimulationCvAdapter pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class MotionDetectorConfig:
    """Configuration for the temporal motion detector."""

    # Minimum absolute pixel difference to count as "changed"
    diff_threshold: int = 30

    # Minimum blob area (pixels) to be considered a detection
    min_blob_area: int = 100

    # Maximum blob area (filter out large changes from drone movement artifacts)
    max_blob_area: int = 15000

    # Morphological operations: dilation kernel size for connecting nearby pixels
    dilation_size: int = 5

    # Base detection probability (motion detection is not perfect)
    base_detection_prob: float = 0.65

    # How much drone movement degrades detection (large movements = more noise)
    movement_penalty_scale: float = 0.3

    # Confidence assigned to motion-only detections
    motion_confidence_base: float = 0.45

    # Confidence boost for larger blobs (more likely real)
    confidence_area_boost: float = 0.15

    # Smoke degrades frame differencing (reduces contrast)
    smoke_penalty_coeff: float = 0.8


class MotionDetector:
    """Detects survivors via temporal frame differencing.

    Compares consecutive rendered drone frames and identifies regions of
    significant change that match survivor-sized blobs. Works by:
    1. Converting frames to grayscale
    2. Computing absolute pixel difference
    3. Thresholding to binary change mask
    4. Finding connected components (blobs)
    5. Filtering by size and reporting detections

    Limitations:
    - Requires the drone to be moving (static hover = no detections)
    - Cannot detect survivors that remain occluded across frames
    - Fire/smoke movement creates false positives
    - Survivors must be static (moving survivors would also trigger, but
      in wildfire SAR most are immobile)
    """

    def __init__(self, config: MotionDetectorConfig | None = None):
        self.config = config or MotionDetectorConfig()
        self._prev_frame: np.ndarray | None = None
        self._prev_drone_xy: tuple[float, float] | None = None
        self._frame_count = 0

    def reset(self):
        """Reset state between episodes."""
        self._prev_frame = None
        self._prev_drone_xy = None
        self._frame_count = 0

    def detect(
        self,
        frame: Image.Image,
        *,
        drone_xy: tuple[float, float],
        footprint_world: float,
        image_size: int,
        smoke_load: float = 0.0,
    ) -> list[dict]:
        """Run motion detection on a new frame.

        Parameters
        ----------
        frame : PIL Image (the rendered drone view)
        drone_xy : current drone world position
        footprint_world : camera footprint in world units
        image_size : frame size in pixels
        smoke_load : average smoke intensity in view (degrades detection)

        Returns
        -------
        List of detection dicts with center_px, bbox_xyxy, confidence, area.
        Returns empty list on the first frame (no previous to compare).
        """
        self._frame_count += 1
        cfg = self.config

        # Convert to grayscale numpy array
        gray = np.array(frame.convert("L"), dtype=np.float32)

        if self._prev_frame is None:
            self._prev_frame = gray
            self._prev_drone_xy = drone_xy
            return []

        # Compute frame difference
        diff = np.abs(gray - self._prev_frame)

        # Apply threshold
        binary = (diff > cfg.diff_threshold).astype(np.uint8)

        # Simple morphological dilation (connect nearby pixels)
        binary = self._dilate(binary, cfg.dilation_size)

        # Find connected components (blobs)
        blobs = self._find_blobs(binary)

        # Compute drone movement penalty
        if self._prev_drone_xy is not None:
            dx = drone_xy[0] - self._prev_drone_xy[0]
            dy = drone_xy[1] - self._prev_drone_xy[1]
            movement = math.sqrt(dx * dx + dy * dy)
            # Normalize by footprint — large relative movement = more noise
            relative_movement = movement / max(footprint_world, 1e-6)
            movement_factor = max(0.1, 1.0 - cfg.movement_penalty_scale * relative_movement * 10.0)
        else:
            movement_factor = 1.0

        # Smoke penalty
        smoke_factor = max(0.2, 1.0 - cfg.smoke_penalty_coeff * smoke_load)

        detections = []
        for blob in blobs:
            area = blob["area"]
            if area < cfg.min_blob_area or area > cfg.max_blob_area:
                continue

            # Detection probability
            det_prob = cfg.base_detection_prob * movement_factor * smoke_factor

            # Confidence: larger blobs are more likely real
            area_fraction = min(1.0, area / 2000.0)
            confidence = cfg.motion_confidence_base + cfg.confidence_area_boost * area_fraction
            confidence *= movement_factor * smoke_factor

            detections.append({
                "center_px": [round(blob["cx"], 1), round(blob["cy"], 1)],
                "bbox_xyxy": blob["bbox"],
                "confidence": round(max(0.1, min(0.95, confidence)), 4),
                "area_px": area,
                "detection_probability": round(det_prob, 4),
                "sensor": "motion",
                "movement_factor": round(movement_factor, 3),
                "smoke_factor": round(smoke_factor, 3),
            })

        # Update state for next frame
        self._prev_frame = gray
        self._prev_drone_xy = drone_xy

        return detections

    def _dilate(self, binary: np.ndarray, size: int) -> np.ndarray:
        """Simple box dilation without scipy/cv2 dependency."""
        if size <= 1:
            return binary
        h, w = binary.shape
        result = np.zeros_like(binary)
        half = size // 2
        for r in range(h):
            for c in range(w):
                if binary[r, c]:
                    r0, r1 = max(0, r - half), min(h, r + half + 1)
                    c0, c1 = max(0, c - half), min(w, c + half + 1)
                    result[r0:r1, c0:c1] = 1
        return result

    def _find_blobs(self, binary: np.ndarray) -> list[dict]:
        """Simple connected-component labeling (4-connected flood fill)."""
        h, w = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        blobs = []

        for r in range(h):
            for c in range(w):
                if binary[r, c] and not visited[r, c]:
                    # Flood fill
                    blob_pixels = []
                    stack = [(r, c)]
                    while stack:
                        pr, pc = stack.pop()
                        if pr < 0 or pr >= h or pc < 0 or pc >= w:
                            continue
                        if visited[pr, pc] or not binary[pr, pc]:
                            continue
                        visited[pr, pc] = True
                        blob_pixels.append((pr, pc))
                        stack.extend([(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)])

                    if blob_pixels:
                        rows = [p[0] for p in blob_pixels]
                        cols = [p[1] for p in blob_pixels]
                        blobs.append({
                            "area": len(blob_pixels),
                            "cx": sum(cols) / len(cols),
                            "cy": sum(rows) / len(rows),
                            "bbox": [min(cols), min(rows), max(cols), max(rows)],
                        })

        return blobs

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def has_previous_frame(self) -> bool:
        return self._prev_frame is not None


def fuse_cv_motion(
    cv_detections: list[dict],
    motion_detections: list[dict],
    *,
    fusion_mode: str = "boost",
    match_radius_px: float = 50.0,
    confidence_boost: float = 0.15,
) -> list[dict]:
    """Fuse CV detections with motion detections.

    Parameters
    ----------
    cv_detections : detections from CV pipeline
    motion_detections : detections from motion detector
    fusion_mode : 'boost' (motion boosts CV confidence),
                  'union' (either source → candidate),
                  'confirm' (motion confirms CV, suppress unconfirmed)
    match_radius_px : maximum pixel distance to match CV and motion detections
    confidence_boost : confidence increase when motion confirms CV detection

    Returns
    -------
    Fused detection list with motion confirmation metadata.
    """
    fused = []

    # Match motion blobs to CV detections by pixel proximity
    motion_matched = set()

    for cv_det in cv_detections:
        cv_cx, cv_cy = cv_det.get("center_px", [0, 0])
        best_motion = None
        best_dist = float("inf")

        for i, md in enumerate(motion_detections):
            if i in motion_matched:
                continue
            mcx, mcy = md["center_px"]
            dist = math.sqrt((cv_cx - mcx) ** 2 + (cv_cy - mcy) ** 2)
            if dist < match_radius_px and dist < best_dist:
                best_dist = dist
                best_motion = i

        motion_confirmed = best_motion is not None
        if motion_confirmed:
            motion_matched.add(best_motion)

        if fusion_mode == "boost":
            conf = cv_det.get("confidence", 0.5)
            if motion_confirmed:
                conf = min(0.99, conf + confidence_boost)
            fused.append({
                **cv_det,
                "confidence": round(conf, 4),
                "motion_confirmed": motion_confirmed,
                "fusion_source": "cv+motion" if motion_confirmed else "cv_only",
            })

        elif fusion_mode == "union":
            fused.append({
                **cv_det,
                "motion_confirmed": motion_confirmed,
                "fusion_source": "cv+motion" if motion_confirmed else "cv_only",
            })

        elif fusion_mode == "confirm":
            if motion_confirmed:
                fused.append({
                    **cv_det,
                    "motion_confirmed": True,
                    "fusion_source": "cv+motion",
                })
            # Unconfirmed CV detections dropped in confirm mode

    # In union mode, add unmatched motion detections as standalone candidates
    if fusion_mode == "union":
        for i, md in enumerate(motion_detections):
            if i not in motion_matched:
                fused.append({
                    "class_name": "person",
                    "confidence": md["confidence"],
                    "bbox_xyxy": md["bbox_xyxy"],
                    "center_px": md["center_px"],
                    "matched_survivor_index": None,
                    "motion_confirmed": True,
                    "fusion_source": "motion_only",
                    "sensor": "motion",
                })

    return fused
