#!/usr/bin/env python3
"""Backfill AUC metrics in older joint diagnostic JSON files.

This helper is meant for older ``diagnose_joint_happo.py`` or
``diagnose_joint_baseline_strategies.py`` outputs that have first scout/confirm
steps but were written before AUC fields were added.

Scout/confirm AUC can be reconstructed exactly from first event steps. Coverage
and confidence AUC require per-step or already-saved row AUC values; if an old
file only has final coverage/confidence, this script leaves those AUC fields
missing unless ``--approx-final-for-missing-continuous`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


AUC_FIELDS = ("scout_auc", "confirm_auc", "coverage_auc", "confidence_auc")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if _is_number(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [float(value) for value in values if _is_number(value)]
    if not finite:
        return float("nan")
    mean = _mean(finite)
    return float(math.sqrt(sum((value - mean) ** 2 for value in finite) / len(finite)))


def _row_steps(row: dict[str, Any], payload: dict[str, Any]) -> int:
    for key in ("episode_steps", "max_steps"):
        value = row.get(key)
        if _is_number(value) and int(value) > 0:
            return int(value)
    scenario = payload.get("scenario_kwargs") or payload.get("scenario") or {}
    value = scenario.get("max_steps") if isinstance(scenario, dict) else None
    if _is_number(value) and int(value) > 0:
        return int(value)
    metadata = payload.get("metadata", {})
    value = metadata.get("steps") if isinstance(metadata, dict) else None
    if _is_number(value) and int(value) > 0:
        return int(value)
    raise ValueError("Could not determine episode length from row or payload")


def _active_survivor_count(row: dict[str, Any], first_steps: list[Any]) -> int:
    count = row.get("active_survivors", row.get("survivors", len(first_steps)))
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = len(first_steps)
    return max(0, min(count, len(first_steps)))


def _explicit_active_survivor_indices(
    row: dict[str, Any],
    first_steps: list[Any],
) -> list[int] | None:
    explicit = row.get("active_survivor_indices")
    if isinstance(explicit, list):
        return [
            int(index)
            for index in explicit
            if isinstance(index, (int, float)) and 0 <= int(index) < len(first_steps)
        ]

    mask = row.get("active_survivor_mask")
    if isinstance(mask, list):
        return [
            index
            for index, active in enumerate(mask[: len(first_steps)])
            if bool(active)
        ]
    return None


def _auc_contribution(first_step: Any, steps: int) -> float:
    try:
        first_step = float(first_step)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(first_step):
        return 0.0
    if first_step <= 0:
        return 1.0
    if first_step <= steps:
        return float((steps - first_step + 1.0) / steps)
    return 0.0


def _event_auc_from_first_steps(
    row: dict[str, Any],
    payload: dict[str, Any],
    key: str,
) -> float | None:
    first_steps = row.get(key)
    if not isinstance(first_steps, list):
        return None

    steps = max(_row_steps(row, payload), 1)
    active_indices = _explicit_active_survivor_indices(row, first_steps)
    if active_indices is not None:
        if not active_indices:
            return 1.0
        return float(
            sum(_auc_contribution(first_steps[index], steps) for index in active_indices)
            / len(active_indices)
        )

    active_count = _active_survivor_count(row, first_steps)
    if active_count <= 0:
        return 1.0

    # Older variable-survivor JSONs may have e.g. 10 survivor observation slots
    # but only 5 active survivors and no active-slot mask. In that case we must
    # not assume that slots 0..4 were active. Non-active slots never receive event
    # steps, and active-but-unfound survivors contribute zero, so summing all
    # observed event steps and dividing by active_count reconstructs event AUC.
    observed = [
        _auc_contribution(first_step, steps)
        for first_step in first_steps
        if first_step is not None
    ]
    return float(sum(observed) / active_count)


def _continuous_auc_from_series(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        series = row.get(key)
        if isinstance(series, list):
            values = [float(value) for value in series if _is_number(value)]
            if values:
                return _mean(values)
    return None


def _time_bin_average(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    bins = row.get("time_bins")
    if not isinstance(bins, list):
        return None
    values: list[float] = []
    for bucket in bins:
        if not isinstance(bucket, dict):
            continue
        for key in keys:
            value = bucket.get(key)
            if _is_number(value):
                values.append(float(value))
                break
    return _mean(values) if values else None


def _coverage_auc(row: dict[str, Any], approximate_final: bool) -> float | None:
    value = _continuous_auc_from_series(
        row,
        (
            "coverage_fraction_by_step",
            "coverage_over_time",
            "coverage_trace",
            "coverage_values",
        ),
    )
    if value is not None:
        return value
    value = _time_bin_average(
        row,
        (
            "coverage_fraction",
            "final_coverage_fraction",
            "mean_coverage_fraction",
        ),
    )
    if value is not None:
        return value
    if approximate_final and _is_number(row.get("final_coverage_fraction")):
        return float(row["final_coverage_fraction"])
    return None


def _confidence_auc(row: dict[str, Any], approximate_final: bool) -> float | None:
    value = _continuous_auc_from_series(
        row,
        (
            "confidence_mean_by_step",
            "confidence_over_time",
            "confidence_trace",
            "confidence_values",
        ),
    )
    if value is not None:
        return value
    value = _time_bin_average(
        row,
        (
            "confidence_mean",
            "final_confidence_mean",
            "mean_confidence",
        ),
    )
    if value is not None:
        return value
    if approximate_final and _is_number(row.get("final_confidence_mean")):
        return float(row["final_confidence_mean"])
    return None


def _set_auc_value(
    row: dict[str, Any],
    key: str,
    value: float | None,
    *,
    overwrite_existing: bool,
) -> bool:
    if not overwrite_existing and key in row and _is_number(row[key]):
        return False
    if value is None or not math.isfinite(float(value)):
        return False
    old_value = row.get(key)
    if overwrite_existing and _is_number(old_value) and abs(float(old_value) - float(value)) < 1e-12:
        return False
    row[key] = float(value)
    return True


def _summary_block(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if _is_number(value)]
    return {
        "mean": _mean(finite),
        "std": _std(finite),
        "count": float(len(finite)),
    }


def _patch_summary(
    payload: dict[str, Any],
    *,
    overwrite_existing: bool,
) -> dict[str, int]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    summary = payload.setdefault("summary", {})
    fast_metrics = summary.setdefault("fast_metrics", {})
    changed: dict[str, int] = {}

    for field in AUC_FIELDS:
        values = [
            float(row[field])
            for row in rows
            if isinstance(row, dict) and _is_number(row.get(field))
        ]
        if not values:
            continue
        mean_key = f"mean_{field}"
        std_key = f"std_{field}"
        mean = _mean(values)
        std = _std(values)
        if (
            overwrite_existing
            or mean_key not in summary
            or not _is_number(summary.get(mean_key))
        ) and (
            not _is_number(summary.get(mean_key))
            or abs(float(summary.get(mean_key)) - mean) >= 1e-12
        ):
            summary[mean_key] = mean
            changed[mean_key] = 1
        if (
            overwrite_existing
            or std_key not in summary
            or not _is_number(summary.get(std_key))
        ) and (
            not _is_number(summary.get(std_key))
            or abs(float(summary.get(std_key)) - std) >= 1e-12
        ):
            summary[std_key] = std
            changed[std_key] = 1
        if (
            overwrite_existing
            or field not in fast_metrics
            or not isinstance(fast_metrics.get(field), dict)
        ):
            fast_metrics[field] = _summary_block(values)
            changed[f"fast_metrics.{field}"] = 1
    return changed


def patch_payload(
    payload: dict[str, Any],
    *,
    approximate_final: bool = False,
    overwrite_existing: bool = False,
) -> tuple[dict[str, int], list[str]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("JSON payload does not contain a rows list")

    changed: dict[str, int] = {field: 0 for field in AUC_FIELDS}
    warnings: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        changed["scout_auc"] += int(
            _set_auc_value(
                row,
                "scout_auc",
                _event_auc_from_first_steps(row, payload, "first_scout_steps"),
                overwrite_existing=overwrite_existing,
            )
        )
        changed["confirm_auc"] += int(
            _set_auc_value(
                row,
                "confirm_auc",
                _event_auc_from_first_steps(row, payload, "first_confirm_steps"),
                overwrite_existing=overwrite_existing,
            )
        )
        changed["coverage_auc"] += int(
            _set_auc_value(
                row,
                "coverage_auc",
                _coverage_auc(row, approximate_final),
                overwrite_existing=overwrite_existing,
            )
        )
        changed["confidence_auc"] += int(
            _set_auc_value(
                row,
                "confidence_auc",
                _confidence_auc(row, approximate_final),
                overwrite_existing=overwrite_existing,
            )
        )

    changed.update(_patch_summary(payload, overwrite_existing=overwrite_existing))
    row_count = max(sum(isinstance(row, dict) for row in rows), 1)
    for field in ("coverage_auc", "confidence_auc"):
        available = sum(
            1
            for row in rows
            if isinstance(row, dict) and _is_number(row.get(field))
        )
        if available < row_count:
            warnings.append(
                f"{field}: filled {available}/{row_count} rows. "
                "Old final-only JSON cannot recover this exact per-step AUC."
            )
    return {key: value for key, value in changed.items() if value}, warnings


def patch_file(
    path: Path,
    *,
    dry_run: bool,
    backup: bool,
    approximate_final: bool,
    overwrite_existing: bool,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed, warnings = patch_payload(
        payload,
        approximate_final=approximate_final,
        overwrite_existing=overwrite_existing,
    )
    if dry_run:
        print(f"{path}: would update {changed or 'nothing'}")
    else:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
        path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        print(f"{path}: updated {changed or 'nothing'}")
    for warning in warnings:
        print(f"  warning: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add missing AUC metrics to joint diagnostic JSON files."
    )
    parser.add_argument("json_files", type=Path, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not create a .json.bak copy before writing.")
    parser.add_argument(
        "--approx-final-for-missing-continuous",
        action="store_true",
        help=(
            "If coverage/confidence AUC cannot be reconstructed, use final "
            "coverage/confidence as an approximate placeholder. Scout/confirm "
            "AUC are still computed from event timing."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recalculate AUC fields even when they already exist.",
    )
    args = parser.parse_args()

    for path in args.json_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        patch_file(
            path,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            approximate_final=args.approx_final_for_missing_continuous,
            overwrite_existing=args.overwrite_existing,
        )


if __name__ == "__main__":
    main()
