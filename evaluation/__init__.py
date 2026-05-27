"""OmniSearch evaluation utilities - metrics plus optional perception demos."""

from .mission_metrics import (
    EpisodeRecorder,
    MissionMetrics,
    evaluate_policy,
    degradation_resilience_ratio,
)

__all__ = [
    "UAVRenderer",
    "render_uav_view",
    "ClosedLoopRun",
    "run_closed_loop",
    "EpisodeRecorder",
    "MissionMetrics",
    "evaluate_policy",
    "degradation_resilience_ratio",
]


def __getattr__(name):
    """Avoid importing YOLO dependencies for metrics-only evaluation runs."""
    if name in {"UAVRenderer", "render_uav_view"}:
        from .sim_renderer import UAVRenderer, render_uav_view
        return {"UAVRenderer": UAVRenderer, "render_uav_view": render_uav_view}[name]
    if name in {"ClosedLoopRun", "run_closed_loop"}:
        from .closed_loop import ClosedLoopRun, run_closed_loop
        return {"ClosedLoopRun": ClosedLoopRun, "run_closed_loop": run_closed_loop}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
