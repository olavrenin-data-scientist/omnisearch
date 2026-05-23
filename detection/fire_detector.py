"""
HSV-based fire detector.

Fire pixels in RGB images sit in a narrow chromatic band: red→orange→yellow
hues with high saturation and value. Two HSV windows (one around red-orange,
one around yellow) cover most flame colors. We threshold, denoise with a
small morphological open, label connected components, and return per-region
bounding boxes + a binary mask.

This is a rule-based proxy for a learned fire classifier; it's good enough
to demo the "detect fire first, then look for survivors" pipeline and to
serve as a baseline. Swap in a trained CNN later by implementing the same
``FireDetector.detect`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from scipy import ndimage

# Type for things we can detect on
ImageLike = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class FireDetection:
    """One detected fire region."""
    box: tuple        # (x1, y1, x2, y2) in pixel coords
    area: int         # number of fire pixels in the region
    confidence: float # 0..1 — fraction of the bbox that is actually fire-coloured

    def to_xyxy(self) -> tuple:
        return self.box


@dataclass
class FireResult:
    """Per-image result from FireDetector."""
    detections: List[FireDetection] = field(default_factory=list)
    mask: np.ndarray = None              # (H, W) bool
    image_shape: tuple = (0, 0)          # (H, W)

    @property
    def has_fire(self) -> bool:
        return len(self.detections) > 0

    @property
    def total_fire_pixels(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0


class FireDetector:
    """
    HSV-threshold fire detector.

    Parameters
    ----------
    min_region_pixels :
        Discard connected components smaller than this (filters noise).
    min_value :
        HSV Value threshold (0..255). Real flames are bright.
    min_saturation :
        HSV Saturation threshold (0..255). Real flames are saturated.
    """

    def __init__(
        self,
        min_region_pixels: int = 300,
        min_value: int = 200,
        min_saturation: int = 180,
        max_hue: int = 30,
    ):
        self.min_region_pixels = min_region_pixels
        self.min_value = min_value
        self.min_saturation = min_saturation
        self.max_hue = max_hue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: ImageLike) -> FireResult:
        rgb = _load_rgb(image)
        mask = self._fire_mask(rgb)
        mask = self._denoise(mask)

        labeled, n_components = ndimage.label(mask)
        detections: List[FireDetection] = []
        for label_id in range(1, n_components + 1):
            region = labeled == label_id
            area = int(region.sum())
            if area < self.min_region_pixels:
                continue
            ys, xs = np.where(region)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox_area = max((x2 - x1 + 1) * (y2 - y1 + 1), 1)
            detections.append(FireDetection(
                box=(x1, y1, x2, y2),
                area=area,
                confidence=float(area) / bbox_area,
            ))

        # Sort by area, largest first (most salient fire region)
        detections.sort(key=lambda d: d.area, reverse=True)
        return FireResult(detections=detections, mask=mask, image_shape=rgb.shape[:2])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fire_mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = _rgb_to_hsv_uint8(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # Red-orange band (H ~ 0-25) and yellow band (H ~ 25-40) — both fire-ish.
        # OpenCV's HSV H is 0..179, so we work in that range.
        hue_fire  = h <= self.max_hue
        saturated = s >= self.min_saturation
        bright    = v >= self.min_value
        return hue_fire & saturated & bright

    def _denoise(self, mask: np.ndarray) -> np.ndarray:
        # Small morphological open (erode then dilate) kills speckle noise.
        opened = ndimage.binary_opening(mask, iterations=1)
        return ndimage.binary_closing(opened, iterations=2)


# ----------------------------------------------------------------------
# Image loading helpers (avoid hard cv2 dependency in the public API)
# ----------------------------------------------------------------------
def _load_rgb(image: ImageLike) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:    # RGBA
            return image[..., :3]
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def _rgb_to_hsv_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to OpenCV-style HSV uint8 (H: 0..179, S/V: 0..255)."""
    # Use cv2 if available (fast and matches OpenCV thresholds people are used to).
    try:
        import cv2
        bgr = rgb[..., ::-1]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    except Exception:
        # Fallback: pure-numpy conversion.
        f = rgb.astype(np.float32) / 255.0
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        mx = f.max(axis=-1)
        mn = f.min(axis=-1)
        d  = mx - mn
        h = np.zeros_like(mx)
        nonzero = d > 0
        rmax = (mx == r) & nonzero
        gmax = (mx == g) & nonzero
        bmax = (mx == b) & nonzero
        h[rmax] = ((g[rmax] - b[rmax]) / d[rmax]) % 6
        h[gmax] = ((b[gmax] - r[gmax]) / d[gmax]) + 2
        h[bmax] = ((r[bmax] - g[bmax]) / d[bmax]) + 4
        h = (h * 30).astype(np.uint8)    # match OpenCV: H scaled to 0..180
        s = np.where(mx > 0, (d / mx) * 255, 0).astype(np.uint8)
        v = (mx * 255).astype(np.uint8)
        return np.stack([h, s, v], axis=-1)
