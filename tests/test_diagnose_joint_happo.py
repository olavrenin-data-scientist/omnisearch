from __future__ import annotations

import pytest

from scripts.diagnose_joint_happo import (
    _event_time_bins,
    _label_counts,
    _mean_path_by_agent,
)


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
