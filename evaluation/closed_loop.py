"""
Closed-loop rollout: simulator → synthetic UAV view → detection pipeline.

Drives a VMAS WildfireSearchScenario forward for N steps, renders a
top-down UAV view every K steps, runs the fire→YOLOv8 pipeline on the
rendered image, and scores detections against ground truth (which the
simulator gives us for free).

The result is a ``ClosedLoopRun`` with per-frame records: the rendered
image, the detection result, the ground truth, and per-frame counts
(true positives, false positives, missed detections). Plot these in the
companion notebook to visualise where the perception pipeline succeeds
or fails on the abstract sim.

The policy here is **random** by default — the closed-loop demo
validates the perception pipeline, not policy quality. A trained
policy can be plugged in by passing ``action_fn`` that maps env → list
of action tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario
from detection import DetectionPipeline, PipelineResult
from .sim_renderer import GroundTruth, UAVRenderer


@dataclass
class FrameRecord:
    """One rendered + detected frame in a closed-loop run."""
    step:        int
    image:       np.ndarray
    ground_truth: GroundTruth
    pipeline:    PipelineResult

    # Scoring against ground truth (per-frame counts)
    tp_persons:    int = 0   # detected persons that overlap a gt survivor box
    fp_persons:    int = 0   # detected persons with no gt match
    fn_persons:    int = 0   # gt survivors with no detection match

    fire_detected: bool = False   # any fire detection at all
    fire_in_gt:    bool = False   # any actual burning cell

    @property
    def fire_correct(self) -> bool:
        return self.fire_detected == self.fire_in_gt


@dataclass
class ClosedLoopRun:
    frames:    List[FrameRecord] = field(default_factory=list)
    n_steps:   int = 0
    n_envs:    int = 0
    n_alerts:  int = 0

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def summary(self) -> dict:
        """Aggregate metrics across all rendered frames."""
        if not self.frames:
            return {}
        tp = sum(f.tp_persons for f in self.frames)
        fp = sum(f.fp_persons for f in self.frames)
        fn = sum(f.fn_persons for f in self.frames)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        fire_acc  = sum(f.fire_correct for f in self.frames) / len(self.frames)
        return {
            "n_frames":       len(self.frames),
            "n_alerts":       self.n_alerts,
            "person_tp":      tp,
            "person_fp":      fp,
            "person_fn":      fn,
            "person_precision": round(precision, 3),
            "person_recall":    round(recall,    3),
            "fire_presence_accuracy": round(fire_acc, 3),
        }


def _boxes_overlap(a, b, slack: int = 8) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax2 + slack >= bx1 and bx2 + slack >= ax1
            and ay2 + slack >= by1 and by2 + slack >= ay1)


def _score_frame(rec: FrameRecord) -> None:
    """Fill TP/FP/FN counts on a FrameRecord by matching detections vs ground truth."""
    detected = [p.box for p in rec.pipeline.persons.detections]
    truth    = list(rec.ground_truth.survivor_pixel_boxes)

    # Greedy IoU-ish matching: a detection matches a truth box if their
    # bboxes overlap (with slack); each truth box matches at most once.
    matched_truth = [False] * len(truth)
    for d in detected:
        for ti, t in enumerate(truth):
            if not matched_truth[ti] and _boxes_overlap(d, t):
                matched_truth[ti] = True
                rec.tp_persons += 1
                break
        else:
            rec.fp_persons += 1
    rec.fn_persons = matched_truth.count(False)

    rec.fire_detected = len(rec.pipeline.fire.detections) > 0
    rec.fire_in_gt    = len(rec.ground_truth.fire_pixel_boxes) > 0


def run_closed_loop(
    n_steps:        int = 120,
    render_every:   int = 15,
    num_envs:       int = 2,
    env_index:      int = 0,
    seed:           int = 0,
    device:         str = "cpu",
    pipeline:       Optional[DetectionPipeline] = None,
    renderer:       Optional[UAVRenderer] = None,
    action_fn:      Optional[Callable] = None,
    scenario_kwargs: Optional[dict] = None,
) -> ClosedLoopRun:
    """
    Run a closed-loop rollout-and-detect demo.

    Parameters
    ----------
    n_steps :
        Total sim steps to take.
    render_every :
        Render + detect every N sim steps.
    env_index :
        Which parallel env's state to render (we keep a small batch so
        VMAS stays happy, but render only one).
    action_fn :
        Optional ``env -> actions`` callable. Default = random actions.
        Plug in a trained policy here.
    """
    scenario_kwargs = scenario_kwargs or {}
    pipeline = pipeline or DetectionPipeline()
    renderer = renderer or UAVRenderer()

    env = WildfireSearchScenario.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        seed=seed,
        **scenario_kwargs,
    )
    env.reset()

    if action_fn is None:
        action_fn = lambda env: env.get_random_actions()

    run = ClosedLoopRun(n_steps=n_steps, n_envs=num_envs)

    for step in range(n_steps):
        env.step(action_fn(env))
        if step % render_every != 0:
            continue
        img, gt = renderer.render(env.scenario, env_index=env_index)
        det     = pipeline.run(img)
        rec = FrameRecord(step=step, image=img, ground_truth=gt, pipeline=det)
        _score_frame(rec)
        if det.alert:
            run.n_alerts += 1
        run.frames.append(rec)

    return run
