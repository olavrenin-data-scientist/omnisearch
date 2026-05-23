"""
YOLOv8 person detector.

Wraps Ultralytics YOLOv8 with ``classes=[0]`` so only the COCO "person" class
fires (https://docs.ultralytics.com/datasets/detect/coco — person is the very
first class in the 80-class list). The nano weights (``yolov8n.pt``) are the
6 MB default; pass any other ``yolov8{s,m,l,x}.pt`` for better accuracy at
the cost of speed.

Predict-mode reference: https://docs.ultralytics.com/modes/predict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

ImageLike = Union[str, Path, np.ndarray, Image.Image]

# COCO class index for "person" (https://docs.ultralytics.com/datasets/detect/coco)
PERSON_CLASS_ID = 0


@dataclass
class PersonDetection:
    """One detected person."""
    box: tuple          # (x1, y1, x2, y2) in pixel coords
    confidence: float   # YOLO confidence 0..1

    def to_xyxy(self) -> tuple:
        return self.box

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max((x2 - x1) * (y2 - y1), 0)


@dataclass
class PersonResult:
    """Per-image result from PersonDetector."""
    detections: List[PersonDetection] = field(default_factory=list)
    image_shape: tuple = (0, 0)         # (H, W)

    @property
    def n_people(self) -> int:
        return len(self.detections)


class PersonDetector:
    """
    YOLOv8 person detector. Loads the model once and reuses it across calls.

    Parameters
    ----------
    model_name :
        Ultralytics weights file. ``yolov8n.pt`` (default) auto-downloads on
        first use. Larger options: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt.
    conf :
        Confidence threshold (0..1).
    iou :
        NMS IoU threshold.
    device :
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` to let Ultralytics decide.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
    ):
        self.model = YOLO(model_name)
        # Sanity check: class 0 is 'person'. If not, the model isn't COCO-pretrained.
        if self.model.names[PERSON_CLASS_ID] != "person":
            raise ValueError(
                f"Loaded model's class 0 is {self.model.names[PERSON_CLASS_ID]!r}, "
                f"not 'person'. Use a COCO-pretrained YOLOv8 model."
            )
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image: ImageLike) -> PersonResult:
        # Ultralytics accepts PIL, ndarray, path, URL — pass through directly.
        # `classes=[PERSON_CLASS_ID]` filters to person only.
        results = self.model.predict(
            source=image,
            classes=[PERSON_CLASS_ID],
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        if not results:
            return PersonResult(detections=[], image_shape=(0, 0))

        r = results[0]
        h, w = r.orig_shape
        detections: List[PersonDetection] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()      # (N, 4)
            confs = r.boxes.conf.cpu().numpy()     # (N,)
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                detections.append(PersonDetection(
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(c),
                ))
        # Sort by confidence, descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return PersonResult(detections=detections, image_shape=(h, w))
