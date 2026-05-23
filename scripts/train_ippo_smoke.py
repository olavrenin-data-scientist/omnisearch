"""
Smoke training: IPPO on the WildfireSearchScenario.

IPPO = fully decentralized PPO (each agent group has its own actor *and*
critic, no parameter sharing across groups). This is the "decentralized
heterogeneous baseline" for the comms-dropout ablation: comms dropout
hurts IPPO worse than MAPPO because there is no centralized critic to
compensate for the missing neighbor signal.

Run from repo root:
    python scripts/train_ippo_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from benchmarl.algorithms import IppoConfig
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig

from agents.wildfire_task import make_wildfire_task
from agents.train_helpers import smoke_config


def main():
    print("=" * 60)
    print(" OmniSearch — IPPO smoke training")
    print("=" * 60)
    print(f" torch: {torch.__version__}  ({'cuda' if torch.cuda.is_available() else 'cpu'})")

    task             = make_wildfire_task(max_steps=150)
    algorithm_config = IppoConfig.get_from_yaml()
    model_config     = MlpConfig.get_from_yaml()
    experiment_cfg   = smoke_config()

    print(f" iters:       {experiment_cfg.max_n_iters}")
    print(f" frames/iter: {experiment_cfg.on_policy_collected_frames_per_batch}")
    print(f" envs/worker: {experiment_cfg.on_policy_n_envs_per_worker}")
    print("-" * 60)

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0,
        config=experiment_cfg,
    )

    t0 = time.time()
    experiment.run()
    print("-" * 60)
    print(f" IPPO smoke complete in {time.time() - t0:.1f}s")
    print(f" Output dir: {experiment.folder_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
