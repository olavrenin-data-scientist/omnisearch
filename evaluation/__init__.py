"""OmniSearch evaluation utilities — rendering, closed-loop demos, metrics."""

from .sim_renderer import UAVRenderer, render_uav_view

__all__ = ["UAVRenderer", "render_uav_view"]

# `from evaluation import ClosedLoopRun, run_closed_loop` works after
# closed_loop is built; we do the import lazily to avoid hard ordering.
try:
    from .closed_loop import ClosedLoopRun, run_closed_loop  # noqa: F401
    __all__ += ["ClosedLoopRun", "run_closed_loop"]
except ImportError:
    pass
