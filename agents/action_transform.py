"""Small action-bound transforms shared by HARL training and evaluation."""

from __future__ import annotations

import numpy as np


def transform_continuous_action(action: np.ndarray, transform: str = "clip") -> np.ndarray:
    """Map unbounded actor output into the VMAS ``[-1, 1]`` action range."""
    action = np.asarray(action, dtype=np.float32)
    if transform == "clip":
        return np.clip(action, -1.0, 1.0).astype(np.float32)
    if transform == "tanh":
        return np.tanh(action).astype(np.float32)
    raise ValueError(f"Unsupported action_transform={transform!r}; expected 'clip' or 'tanh'")
