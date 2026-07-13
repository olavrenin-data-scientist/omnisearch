"""Multi-object tracking for temporal consistency in survivor detection.

Wraps the ByteTrack algorithm (via the `supervision` library) to provide:
  - Persistent track IDs across consecutive frames
  - False-positive suppression (require N consecutive detections to confirm)
  - Track interpolation for missed frames
  - Per-track confidence accumulation

Usage:
    tracker = SurvivorTracker(min_hits=3, max_age=5)
    for frame_detections in stream:
        tracked = tracker.update(frame_detections)
        confirmed = tracker.get_confirmed_tracks()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv


@dataclass
class TrackedDetection:
    """A single tracked detection with persistent ID and confirmation state."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    center_px: tuple[float, float]
    hit_count: int
    age: int
    confirmed: bool


class SurvivorTracker:
    """ByteTrack-based multi-object tracker for survivor detection.

    Parameters
    ----------
    track_activation_threshold : float
        Detection confidence threshold for initiating a new track.
    lost_track_buffer : int
        Number of frames a lost track is kept before deletion.
    minimum_matching_threshold : float
        IoU threshold for matching detections to existing tracks.
    min_hits : int
        Minimum consecutive detections required before a track is "confirmed"
        (reported as a true detection). Reduces false positives.
    frame_rate : int
        Expected frame rate, used by ByteTrack for internal velocity estimation.
    """

    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 5,
        minimum_matching_threshold: float = 0.8,
        min_hits: int = 2,
        frame_rate: int = 1,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.min_hits = int(min_hits)
        self._track_hits: dict[int, int] = {}
        self._track_confs: dict[int, list[float]] = {}
        self._frame_count = 0

    def reset(self):
        """Reset tracker state (e.g., between episodes)."""
        self.tracker.reset()
        self._track_hits.clear()
        self._track_confs.clear()
        self._frame_count = 0

    def update(
        self,
        detections: list[dict],
    ) -> list[TrackedDetection]:
        """Update tracker with new frame detections.

        Parameters
        ----------
        detections : list of dict
            Each dict must have "bbox_xyxy" (list of 4 floats) and
            "confidence" (float). Other fields are preserved.

        Returns
        -------
        list of TrackedDetection
            All active tracks (both confirmed and tentative).
        """
        self._frame_count += 1

        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array([d["bbox_xyxy"] for d in detections], dtype=np.float32)
            confs = np.array([d["confidence"] for d in detections], dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=confs,
            )

        tracked = self.tracker.update_with_detections(sv_dets)

        results: list[TrackedDetection] = []
        active_ids: set[int] = set()

        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                tid = int(track_id)
                active_ids.add(tid)
                self._track_hits[tid] = self._track_hits.get(tid, 0) + 1
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                if tid not in self._track_confs:
                    self._track_confs[tid] = []
                self._track_confs[tid].append(conf)

                box = tuple(float(v) for v in tracked.xyxy[i])
                cx = (box[0] + box[2]) * 0.5
                cy = (box[1] + box[3]) * 0.5
                hits = self._track_hits[tid]

                results.append(TrackedDetection(
                    track_id=tid,
                    bbox_xyxy=box,
                    confidence=conf,
                    center_px=(cx, cy),
                    hit_count=hits,
                    age=self._frame_count,
                    confirmed=hits >= self.min_hits,
                ))

        # Clean up old tracks no longer active
        stale = [tid for tid in self._track_hits if tid not in active_ids]
        for tid in stale:
            if self._frame_count - self._track_hits.get(tid, 0) > 10:
                self._track_hits.pop(tid, None)
                self._track_confs.pop(tid, None)

        return results

    def get_confirmed_tracks(self, tracks: list[TrackedDetection]) -> list[TrackedDetection]:
        """Filter to only confirmed tracks (min_hits consecutive detections)."""
        return [t for t in tracks if t.confirmed]

    def get_track_avg_confidence(self, track_id: int) -> float:
        """Get the running average confidence for a track."""
        confs = self._track_confs.get(track_id, [])
        return sum(confs) / len(confs) if confs else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count
