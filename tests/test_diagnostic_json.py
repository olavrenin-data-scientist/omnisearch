import json
from pathlib import Path

from scripts.diagnostic_json import (
    partial_json_path,
    write_final_json,
    write_partial_json,
)


def test_partial_json_checkpoint_and_final_cleanup(tmp_path: Path) -> None:
    final = tmp_path / "diagnostics.json"
    partial = partial_json_path(final)
    payload = {
        "metadata": {"strategy": "test"},
        "rows": [{"seed": 1000}],
        "summary": {"episodes": 1},
    }

    written_partial = write_partial_json(
        final,
        payload,
        completed_rollouts=1,
        total_rollouts=2,
    )

    assert written_partial == partial
    assert not final.exists()
    partial_payload = json.loads(partial.read_text(encoding="utf-8"))
    assert partial_payload["metadata"] == {
        "strategy": "test",
        "complete": False,
        "completed_rollouts": 1,
        "total_rollouts": 2,
    }

    written_final = write_final_json(
        final,
        payload,
        completed_rollouts=2,
        total_rollouts=2,
    )

    assert written_final == final
    assert not partial.exists()
    final_payload = json.loads(final.read_text(encoding="utf-8"))
    assert final_payload["metadata"]["complete"] is True
    assert final_payload["metadata"]["completed_rollouts"] == 2
    assert final_payload["metadata"]["total_rollouts"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_path_without_json_suffix(tmp_path: Path) -> None:
    assert partial_json_path(tmp_path / "diagnostics") == (
        tmp_path / "diagnostics.partial.json"
    )
