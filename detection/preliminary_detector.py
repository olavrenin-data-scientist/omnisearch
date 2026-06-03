"""Preliminary detector helpers for simulator/CV workflow tests.

This module intentionally does not run a learned model. It turns renderer
ground-truth boxes into detector-like outputs, with optional miss probability
and pixel noise. Use it while wiring the simulation workflow, then replace it
with YOLO or another real detector later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from pathlib import Path


@dataclass(frozen=True)
class PreliminaryDetection:
    """One detector-like person result in pixel coordinates."""

    box: tuple[int, int, int, int]
    confidence: float = 1.0
    class_name: str = "person"

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5


@dataclass
class PreliminaryResult:
    """Per-frame preliminary detection result."""

    detections: list[PreliminaryDetection] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)

    @property
    def n_people(self) -> int:
        return len(self.detections)


class PreliminaryPersonDetector:
    """Detector stub backed by renderer ground-truth labels.

    Parameters
    ----------
    detection_probability:
        Probability of keeping each ground-truth person box.
    pixel_noise_std:
        Gaussian noise added independently to box coordinates. ``0`` gives
        perfect detections.
    confidence:
        Confidence attached to detections. If ``confidence_jitter`` is nonzero,
        this is the mean confidence.
    confidence_jitter:
        Uniform jitter subtracted/added around ``confidence``.
    seed:
        Random seed for repeatable workflow tests.
    """

    def __init__(
        self,
        *,
        detection_probability: float = 1.0,
        pixel_noise_std: float = 0.0,
        confidence: float = 0.95,
        confidence_jitter: float = 0.0,
        seed: int = 7,
    ):
        self.detection_probability = max(0.0, min(1.0, float(detection_probability)))
        self.pixel_noise_std = max(0.0, float(pixel_noise_std))
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.confidence_jitter = max(0.0, float(confidence_jitter))
        self.rng = random.Random(int(seed))

    def detect_boxes(
        self,
        boxes_xyxy: list[tuple[int, int, int, int]],
        *,
        image_size: int,
    ) -> PreliminaryResult:
        detections = []
        for box in boxes_xyxy:
            if self.rng.random() > self.detection_probability:
                continue
            noisy_box = self._jitter_box(box, image_size=image_size)
            conf = self._confidence()
            detections.append(PreliminaryDetection(box=noisy_box, confidence=conf))
        return PreliminaryResult(detections=detections, image_shape=(image_size, image_size))

    def detect_yolo_label_file(self, label_path: str | Path, *, image_size: int) -> PreliminaryResult:
        return self.detect_boxes(_read_yolo_boxes(label_path, image_size=image_size), image_size=image_size)

    def _jitter_box(self, box: tuple[int, int, int, int], *, image_size: int) -> tuple[int, int, int, int]:
        if self.pixel_noise_std == 0.0:
            return box
        x1, y1, x2, y2 = box
        values = [
            int(round(value + self.rng.gauss(0.0, self.pixel_noise_std)))
            for value in (x1, y1, x2, y2)
        ]
        values[0] = max(0, min(image_size - 1, values[0]))
        values[1] = max(0, min(image_size - 1, values[1]))
        values[2] = max(values[0] + 1, min(image_size, values[2]))
        values[3] = max(values[1] + 1, min(image_size, values[3]))
        return values[0], values[1], values[2], values[3]

    def _confidence(self) -> float:
        if self.confidence_jitter == 0.0:
            return self.confidence
        low = max(0.0, self.confidence - self.confidence_jitter)
        high = min(1.0, self.confidence + self.confidence_jitter)
        return self.rng.uniform(low, high)


def _read_yolo_boxes(label_path: str | Path, *, image_size: int) -> list[tuple[int, int, int, int]]:
    path = Path(label_path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            _class_id, cx, cy, width, height = (float(value) for value in parts[:5])
        except ValueError:
            continue
        x1 = int(round((cx - width * 0.5) * image_size))
        y1 = int(round((cy - height * 0.5) * image_size))
        x2 = int(round((cx + width * 0.5) * image_size))
        y2 = int(round((cy + height * 0.5) * image_size))
        boxes.append(
            (
                max(0, min(image_size - 1, x1)),
                max(0, min(image_size - 1, y1)),
                max(0, min(image_size, x2)),
                max(0, min(image_size, y2)),
            )
        )
    return boxes
