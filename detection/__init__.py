"""OmniSearch computer-vision package.

This package contains two pieces of the CV stack:

* detector wrappers: fire thresholding, YOLO person detection, and the
  fire/person alert pipeline;
* simulation image helpers: NAIP imagery fetchers and altitude-scaled drone
  crop rendering for synthetic CV integration tests.

Detector example:

    >>> from detection import DetectionPipeline
    >>> pipe = DetectionPipeline()
    >>> result = pipe.run("path/to/image.jpg")
    >>> result.fire_boxes         # bounding boxes of fire regions
    >>> result.person_boxes       # YOLOv8 detections (class 0 = person)
    >>> result.survivors_in_fire  # subset of person_boxes overlapping a fire region

Fire detection uses HSV color thresholding (red/orange/yellow + high value/saturation);
person detection uses Ultralytics YOLOv8 with ``classes=[0]`` per the COCO class list
(person = class 0 — see https://docs.ultralytics.com/datasets/detect/coco).
"""

from .camera_renderer import (
    CameraDetection,
    CameraRender,
    DroneCameraRenderer,
    FireCell,
    SmokeCell,
    SurvivorObject,
)
from .fire_detector import FireDetector, FireDetection
from .naip import fetch_naip_image, fetch_naip_tiled_image


def __getattr__(name: str):
    if name in {"PersonDetector", "PersonDetection"}:
        from .person_detector import PersonDetection, PersonDetector

        return {"PersonDetector": PersonDetector, "PersonDetection": PersonDetection}[name]
    if name in {"DetectionPipeline", "PipelineResult"}:
        from .pipeline import DetectionPipeline, PipelineResult

        return {"DetectionPipeline": DetectionPipeline, "PipelineResult": PipelineResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CameraDetection",
    "CameraRender",
    "DroneCameraRenderer",
    "FireCell",
    "SmokeCell",
    "SurvivorObject",
    "FireDetector",
    "FireDetection",
    "fetch_naip_image",
    "fetch_naip_tiled_image",
    "PersonDetector",
    "PersonDetection",
    "DetectionPipeline",
    "PipelineResult",
]
