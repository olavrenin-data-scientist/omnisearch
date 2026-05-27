"""
Smoke training: HAPPO on the WildfireSearchScenario via HARL.

This script is a thin wrapper around ``agents.harl_runner.train_happo`` — all
the monkey-patching, config building, and runner wiring lives there so the
comms-dropout sweep can call the same code path.

Smoke budget: 8000 env steps × 8 parallel envs (batched in one VMAS instance)
≈ 5 seconds on CPU at ~1700 FPS.

Run from repo root:

    python scripts/train_happo_smoke.py

Prerequisite: HARL must be installed in the active venv:

    git clone https://github.com/PKU-MARL/HARL ../HARL
    pip install -e ../HARL
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.harl_runner import train_happo


def main():
    print("=" * 60)
    print(" OmniSearch — HAPPO smoke training (HARL)")
    print("=" * 60)

    t0 = time.time()
    result = train_happo(
        seed              = 1,
        num_env_steps     = 8_000,
        comms_dropout     = 0.0,
        n_rollout_threads = 8,
        exp_name          = "happo_smoke",
    )
    print("-" * 60)
    print(f" HAPPO smoke complete in {time.time() - t0:.1f}s")
    print(f"   last mean episode reward: {result['mean_episode_reward']:+.3f}")
    print(f"   last mean step reward:    {result['mean_step_reward']:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
