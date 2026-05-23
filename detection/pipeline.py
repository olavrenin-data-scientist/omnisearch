"""
Two-stage detection pipeline: fire first, then survivors.

The pipeline runs both detectors on an image and then computes, for each
detected person, whether they overlap a detected fire region. That subset
is the "survivors in fire" alert — what the deployed system would prioritize.

Order matters here: in production we expect to run the cheap HSV fire
detector at high frame rate on UAV feeds, and trigger the heavier YOLOv8
person detector when fire is seen. The current ``run()`` always runs both
for simplicity; ``run(triggered_only=True)`` short-circuits to "no fire =>
no person detection".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .fire_detector   import FireDetector, FireResult, ImageLike
from .person_detector import PersonDetector, PersonResult, PersonDetection


@dataclass
class PipelineResult:
    fire:    FireResult
    persons: PersonResult
    survivors_in_fire: List[PersonDetection] = field(default_factory=list)

    @property
    def n_survivors_in_fire(self) -> int:
        return len(self.survivors_in_fire)

    @property
    def alert(self) -> bool:
        """Anyone we'd ping the operator about."""
        return self.n_survivors_in_fire > 0


class DetectionPipeline:
    """
    Sequential fire→person detector.

    Parameters
    ----------
    fire_detector, person_detector :
        Pre-constructed detectors. If ``None``, defaults are built with the
        ``fire_kwargs`` / ``person_kwargs`` you provide.
    proximity_pixels :
        A person is "in fire" if their bbox is within this many pixels of
        any fire region's bbox (in addition to direct overlap). Set to 0
        to require strict box intersection.
    """

    def __init__(
        self,
        fire_detector: FireDetector = None,
        person_detector: PersonDetector = None,
        fire_kwargs: dict | None = None,
        person_kwargs: dict | None = None,
        proximity_pixels: int = 20,
    ):
        self.fire = fire_detector or FireDetector(**(fire_kwargs or {}))
        self.person = person_detector or PersonDetector(**(person_kwargs or {}))
        self.proximity_pixels = proximity_pixels

    def run(self, image: ImageLike, triggered_only: bool = False) -> PipelineResult:
        """
        Run the pipeline.

        triggered_only=True : skip the person detector entirely if no fire was
        seen — useful when running on a stream where most frames have no fire.
        """
        fire_result = self.fire.detect(image)

        if triggered_only and not fire_result.has_fire:
            return PipelineResult(
                fire=fire_result,
                persons=PersonResult(detections=[], image_shape=fire_result.image_shape),
            )

        person_result = self.person.detect(image)

        # Find which persons are in / near a fire region
        in_fire: List[PersonDetection] = []
        for p in person_result.detections:
            if self._near_any_fire(p.box, fire_result):
                in_fire.append(p)

        return PipelineResult(
            fire=fire_result,
            persons=person_result,
            survivors_in_fire=in_fire,
        )

    def _near_any_fire(self, pbox: tuple, fire: FireResult) -> bool:
        px1, py1, px2, py2 = pbox
        m = self.proximity_pixels
        for f in fire.detections:
            fx1, fy1, fx2, fy2 = f.box
            if (px2 + m >= fx1 and fx2 + m >= px1
                    and py2 + m >= fy1 and fy2 + m >= py1):
                return True
        return False
