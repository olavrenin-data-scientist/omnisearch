"""
Isaac Lab + Cosmos 3 preflight for OmniSearch.

This script does not train anything; it verifies local prerequisites and
creates a run folder for first Isaac video demos.

Run:
    python scripts/isaac_preflight.py
    python scripts/isaac_preflight.py --output-dir results/isaac_demo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _run_ok(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        return False, f"{exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, (proc.stdout or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/isaac_demo",
        help="Directory where preflight outputs and recordings will be stored.",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, dict[str, str | bool]] = {}

    checks["python"] = {
        "ok": True,
        "detail": sys.version.replace("\n", " "),
    }

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        ok, detail = _run_ok([nvidia_smi])
        checks["gpu_driver"] = {"ok": ok, "detail": detail.splitlines()[0] if detail else "OK"}
    else:
        checks["gpu_driver"] = {
            "ok": False,
            "detail": "nvidia-smi not found (Isaac Sim typically expects NVIDIA GPU drivers).",
        }

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        ok, detail = _run_ok([ffmpeg_path, "-version"])
        first_line = detail.splitlines()[0] if detail else "ffmpeg found"
        checks["ffmpeg"] = {"ok": ok, "detail": first_line}
    else:
        checks["ffmpeg"] = {
            "ok": False,
            "detail": "ffmpeg not found (needed for robust frame-to-video workflows).",
        }

    module_checks = {
        "isaaclab": _has_module("isaaclab"),
        "omni.isaac.lab": _has_module("omni.isaac.lab"),
        "omni.isaac.kit": _has_module("omni.isaac.kit"),
    }
    for module_name, ok in module_checks.items():
        checks[f"module:{module_name}"] = {
            "ok": ok,
            "detail": "importable" if ok else "not importable",
        }

    all_ok = all(item["ok"] for item in checks.values())

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(out_dir),
        "checks": checks,
        "ready_for_first_isaac_demo": all_ok,
        "next_steps": [
            "Follow docs/ISAAC_LAB_COSMOS3_QUICKSTART.md",
            "Use the 'First Video Demo' section to record viewport/camera output",
            "Export per-episode stats and map them into evaluation/mission_metrics.py format",
        ],
    }

    out_path = out_dir / "preflight_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print(" OmniSearch Isaac/Cosmos preflight")
    print("=" * 88)
    for key, value in checks.items():
        status = "OK " if value["ok"] else "ERR"
        print(f"[{status}] {key:22s} {value['detail']}")
    print("-" * 88)
    print(f"Ready for first Isaac demo: {all_ok}")
    print(f"Wrote: {out_path.relative_to(REPO_ROOT)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
