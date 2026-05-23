"""
OmniSearch detection pipeline.

Runs **fire detection first, then survivor (person) detection**:

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

from .fire_detector import FireDetector, FireDetection
from .person_detector import PersonDetector, PersonDetection
from .pipeline import DetectionPipeline, PipelineResult

__all__ = [
    "FireDetector",
    "FireDetection",
    "PersonDetector",
    "PersonDetection",
    "DetectionPipeline",
    "PipelineResult",
]
