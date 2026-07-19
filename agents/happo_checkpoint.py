"""Project-owned metadata stored alongside HARL HAPPO checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "omnisearch_training_config.json"
MANIFEST_VERSION = 1


def checkpoint_run_dir(checkpoint_dir: str | Path) -> Path:
    """Return the HARL run directory for a models directory."""
    checkpoint_dir = Path(checkpoint_dir)
    return checkpoint_dir.parent if checkpoint_dir.name == "models" else checkpoint_dir


def load_training_manifest(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Load OmniSearch training metadata for a checkpoint, if available."""
    path = checkpoint_run_dir(checkpoint_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported HAPPO training manifest version in {path}")
    return payload


def save_training_manifest(
    runner,
    *,
    harl_args: dict[str, Any],
    algo_args: dict[str, Any],
    env_args: dict[str, Any],
) -> Path:
    """Write the configuration needed to reproduce a HARL training environment."""
    run_dir = _runner_run_dir(runner)
    path = run_dir / MANIFEST_FILENAME
    payload = {
        "version": MANIFEST_VERSION,
        "harl_args": harl_args,
        "algo_args": algo_args,
        "env_args": env_args,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def merge_training_scenario(
    export_scenario: dict[str, Any],
    manifest: dict[str, Any],
    *,
    max_steps: int | None = None,
    comms_dropout: float | None = None,
) -> dict[str, Any]:
    """Restore training settings while preserving deliberate evaluation controls."""
    training_scenario = manifest.get("env_args", {}).get("scenario_kwargs", {})
    merged = {
        **export_scenario,
        **training_scenario,
    }
    if max_steps is not None:
        merged["max_steps"] = max_steps
    if comms_dropout is not None:
        merged["comms_dropout"] = comms_dropout
    return merged


def _runner_run_dir(runner) -> Path:
    """Find the run directory across HARL versions."""
    for attribute in ("run_dir", "log_dir"):
        value = getattr(runner, attribute, None)
        if value:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            return path

    save_dir = getattr(runner, "save_dir", None)
    if save_dir:
        path = Path(save_dir)
        run_dir = path.parent if path.name == "models" else path
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    raise RuntimeError("Could not locate the HARL run directory for checkpoint metadata")
