"""Crash-safe JSON checkpoint helpers for diagnostic scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def partial_json_path(final_path: str | Path) -> Path:
    final = Path(final_path)
    if final.suffix:
        return final.with_name(f"{final.stem}.partial{final.suffix}")
    return final.with_name(f"{final.name}.partial.json")


def _payload_with_progress(
    payload: dict[str, Any],
    *,
    complete: bool,
    completed_rollouts: int,
    total_rollouts: int,
) -> dict[str, Any]:
    result = dict(payload)
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "complete": bool(complete),
            "completed_rollouts": int(completed_rollouts),
            "total_rollouts": int(total_rollouts),
        }
    )
    result["metadata"] = metadata
    return result


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=True,
            sort_keys=sort_keys,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_partial_json(
    final_path: str | Path,
    payload: dict[str, Any],
    *,
    completed_rollouts: int,
    total_rollouts: int,
    sort_keys: bool = False,
) -> Path:
    partial = partial_json_path(final_path)
    checkpoint = _payload_with_progress(
        payload,
        complete=False,
        completed_rollouts=completed_rollouts,
        total_rollouts=total_rollouts,
    )
    _write_json_atomic(partial, checkpoint, sort_keys=sort_keys)
    return partial


def write_final_json(
    final_path: str | Path,
    payload: dict[str, Any],
    *,
    completed_rollouts: int,
    total_rollouts: int,
    sort_keys: bool = False,
) -> Path:
    final = Path(final_path)
    completed = _payload_with_progress(
        payload,
        complete=True,
        completed_rollouts=completed_rollouts,
        total_rollouts=total_rollouts,
    )
    _write_json_atomic(final, completed, sort_keys=sort_keys)
    partial_json_path(final).unlink(missing_ok=True)
    return final
