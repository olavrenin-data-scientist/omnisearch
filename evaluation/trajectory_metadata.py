"""Metadata helpers for exported trajectory files."""

from __future__ import annotations

from typing import Any, Sequence


def trajectory_timing(frames: Sequence[dict[str, Any]], sim_step_seconds: float) -> dict[str, float | int]:
    """Describe simulated duration independently of trajectory frame sampling."""
    actual_n_steps = int(frames[-1]["step"]) if frames else 0
    step_seconds = round(float(sim_step_seconds), 4)
    return {
        "actual_n_steps": actual_n_steps,
        "recorded_frame_count": len(frames),
        "sim_step_seconds": step_seconds,
        "actual_duration_seconds": round(actual_n_steps * step_seconds, 4),
    }
