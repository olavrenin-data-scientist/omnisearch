"""Small action-bound transforms shared by HARL training and evaluation."""

from __future__ import annotations

import numpy as np


def transform_continuous_action(action: np.ndarray, transform: str = "clip") -> np.ndarray:
    """Map unbounded actor output into the VMAS ``[-1, 1]`` action range."""
    action = np.asarray(action, dtype=np.float32)
    transform = transform.replace("-", "_")
    if transform == "clip":
        return np.clip(action, -1.0, 1.0).astype(np.float32)
    if transform == "tanh":
        return np.tanh(action).astype(np.float32)
    if transform == "radial_tanh":
        norm = np.linalg.norm(action, axis=-1, keepdims=True)
        scale = np.tanh(norm) / np.maximum(norm, 1e-6)
        return (action * scale).astype(np.float32)
    raise ValueError(
        f"Unsupported action_transform={transform!r}; "
        "expected 'clip', 'tanh', or 'radial_tanh'",
    )
