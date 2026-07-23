from __future__ import annotations

import json
import types

import pytest

from agents.happo_checkpoint import MANIFEST_FILENAME
from scripts.diagnose_joint_happo import (
    _event_time_bins,
    _label_counts,
    _mean_path_by_agent,
    _recall_threshold_time_stats,
    _scenario_kwargs,
)


class MissingNoneNamespace(types.SimpleNamespace):
    def __getattr__(self, _name):
        return None


def test_event_time_bins_report_new_and_cumulative_recall() -> None:
    rows = [
        {
            "survivors": 2,
            "episode_steps": 100,
            "first_scout_steps": [10, 60],
        },
        {
            "survivors": 2,
            "episode_steps": 100,
            "first_scout_steps": [None, 20],
        },
    ]

    bins = _event_time_bins(rows, key="first_scout_steps", bins=2)

    assert bins[0]["mean_new_recall"] == pytest.approx(0.5)
    assert bins[0]["mean_cumulative_recall"] == pytest.approx(0.5)
    assert bins[1]["mean_new_recall"] == pytest.approx(0.25)
    assert bins[1]["mean_cumulative_recall"] == pytest.approx(0.75)


def test_failure_label_counts_exclude_success() -> None:
    rows = [
        {"uav_failure_labels": ["success"]},
        {"uav_failure_labels": ["partial_search", "low_coverage"]},
        {"uav_failure_labels": ["partial_search"]},
    ]

    assert _label_counts(rows, "uav_failure_labels") == {
        "partial_search": 2,
        "low_coverage": 1,
    }


def test_mean_path_by_agent_handles_variable_agent_counts() -> None:
    rows = [
        {"paths": [10.0, 20.0]},
        {"paths": [30.0]},
    ]

    assert _mean_path_by_agent(rows, "paths") == pytest.approx([20.0, 20.0])


def test_recall_threshold_time_stats_reports_seconds_and_reached_fraction() -> None:
    rows = [
        {
            "survivors": 5,
            "step_seconds": 2.0,
            "first_confirm_steps": [5, 10, 15, 20, None],
        },
        {
            "survivors": 5,
            "step_seconds": 2.0,
            "first_confirm_steps": [3, 4, None, None, None],
        },
    ]

    stats = _recall_threshold_time_stats(rows, key="first_confirm_steps", threshold=0.80)

    assert stats["reached_count"] == pytest.approx(1.0)
    assert stats["reached_fraction"] == pytest.approx(0.5)
    assert stats["mean_s"] == pytest.approx(40.0)
    assert stats["std_s"] == pytest.approx(0.0)


def test_scenario_kwargs_can_override_checkpoint_comms(tmp_path) -> None:
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "env_args": {
                    "scenario_kwargs": {
                        "n_drones": 3,
                        "n_ground": 2,
                        "n_survivors": 5,
                        "n_decoys": 0,
                        "comms_dropout": 0.1,
                        "comms_dropout_mode": "iid",
                        "comms_map_mode": "global",
                        "comms_dropout_min_steps": 5,
                        "comms_dropout_max_steps": 15,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = MissingNoneNamespace(
        steps=300,
        joint_survivor_diagnostic=False,
        joint_schema_ugv_diagnostic=False,
        joint_diagnostic_ugvs=2,
        comms_dropout=0.3,
        comms_dropout_mode="bursty",
        comms_map_mode="per-agent",
        comms_dropout_min_steps=8,
        comms_dropout_max_steps=13,
    )

    kwargs = _scenario_kwargs(models_dir, args)

    assert kwargs["comms_dropout"] == pytest.approx(0.3)
    assert kwargs["comms_dropout_mode"] == "bursty"
    assert kwargs["comms_map_mode"] == "per_agent"
    assert kwargs["comms_dropout_min_steps"] == 8
    assert kwargs["comms_dropout_max_steps"] == 13


def test_scenario_kwargs_migrate_checkpoint_global_maps(tmp_path) -> None:
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "env_args": {
                    "scenario_kwargs": {
                        "n_drones": 3,
                        "n_ground": 2,
                        "n_survivors": 5,
                        "comms_map_mode": "global",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = MissingNoneNamespace(
        steps=300,
        joint_survivor_diagnostic=False,
        joint_schema_ugv_diagnostic=False,
        joint_diagnostic_ugvs=2,
    )

    kwargs = _scenario_kwargs(models_dir, args)

    assert kwargs["comms_map_mode"] == "per_agent"
