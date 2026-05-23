"""
Smoke training: MAPPO on the WildfireSearchScenario.

Tiny budget (~6k frames in 3 iterations) to verify the BenchMARL pipeline
runs end-to-end on our custom scenario. Not a research run — reward will
move only slightly. The successful exit is the milestone, not the score.

Run from repo root:
    python scripts/train_mappo_smoke.py

Real training: bump max_n_iters and frames_per_batch by ~50x, enable W&B in
`loggers`, switch to GPU if available, and use `MappoConfig` or `HappoConfig`
depending on the experiment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make repo-root imports work whether invoked from root or scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from agents.wildfire_task import make_wildfire_task


def build_experiment_config() -> ExperimentConfig:
    cfg = ExperimentConfig.get_from_yaml()

    # Small budget for smoke run
    cfg.max_n_iters = 3
    cfg.max_n_frames = 100_000_000        # fall-through; iters caps first
    cfg.on_policy_collected_frames_per_batch = 2_000
    cfg.on_policy_n_envs_per_worker = 8
    cfg.on_policy_n_minibatch_iters = 4
    cfg.on_policy_minibatch_size = 200

    # Skip evaluation and rendering to keep smoke run fast
    cfg.evaluation = False
    cfg.render = False

    # No external logger for smoke. Add "wandb" here to enable W&B.
    cfg.loggers = []

    # Write run artifacts under results/  (gitignored).
    cfg.save_folder = str(ROOT / "results")

    # Smaller LR / device suitable for CPU
    cfg.lr = 3e-4
    cfg.sampling_device = "cpu"
    cfg.train_device   = "cpu"
    cfg.buffer_device  = "cpu"

    return cfg


def main():
    print("=" * 60)
    print(" OmniSearch — MAPPO smoke training")
    print("=" * 60)
    print(f" torch:  {torch.__version__}  ({'cuda' if torch.cuda.is_available() else 'cpu'})")

    task             = make_wildfire_task(max_steps=150)
    algorithm_config = MappoConfig.get_from_yaml()
    model_config     = MlpConfig.get_from_yaml()
    experiment_cfg   = build_experiment_config()

    expected_frames = (
        experiment_cfg.max_n_iters
        * experiment_cfg.on_policy_collected_frames_per_batch
    )
    print(f" iters:  {experiment_cfg.max_n_iters}")
    print(f" frames/iter: {experiment_cfg.on_policy_collected_frames_per_batch}")
    print(f" total ≈ {expected_frames} frames")
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
    dt = time.time() - t0

    print("-" * 60)
    print(f" Smoke training complete in {dt:.1f}s")
    print(f" Output dir: {experiment.folder_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
