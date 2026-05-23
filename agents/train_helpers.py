"""
Shared training-config builders for OmniSearch BenchMARL runs.

Two helpers:

- ``smoke_config``  — tiny budget (~6k frames) for "does it run?" checks.
- ``research_config`` — bigger budget (~400k frames) for actual experiments.

Both write under ``results/`` (gitignored) and skip W&B by default; pass
``loggers=["wandb"]`` to opt in.

These wrap ``benchmarl.experiment.ExperimentConfig`` rather than reimplementing
it — every BenchMARL knob is still reachable on the returned object.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from benchmarl.experiment import ExperimentConfig

# Repo root (parent of this file's parent)
ROOT = Path(__file__).resolve().parent.parent


def _base_cpu_config() -> ExperimentConfig:
    cfg = ExperimentConfig.get_from_yaml()
    cfg.sampling_device = "cpu"
    cfg.train_device    = "cpu"
    cfg.buffer_device   = "cpu"
    cfg.save_folder     = str(ROOT / "results")
    return cfg


def smoke_config(
    iters: int = 3,
    frames_per_batch: int = 2_000,
    envs_per_worker: int = 8,
    loggers: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Tiny budget — meant to verify the pipeline, not produce a useful policy."""
    cfg = _base_cpu_config()
    cfg.max_n_iters = iters
    cfg.max_n_frames = 100_000_000
    cfg.on_policy_collected_frames_per_batch = frames_per_batch
    cfg.on_policy_n_envs_per_worker = envs_per_worker
    cfg.on_policy_n_minibatch_iters = 4
    cfg.on_policy_minibatch_size = max(frames_per_batch // 10, 64)
    cfg.lr = 3e-4
    cfg.evaluation = False
    cfg.render = False
    cfg.loggers = loggers or []
    return cfg


def research_config(
    iters: int = 200,
    frames_per_batch: int = 6_000,
    envs_per_worker: int = 32,
    loggers: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Default budget for a credible single-seed run (~400k frames at iters=200)."""
    cfg = _base_cpu_config()
    cfg.max_n_iters = iters
    cfg.max_n_frames = 100_000_000
    cfg.on_policy_collected_frames_per_batch = frames_per_batch
    cfg.on_policy_n_envs_per_worker = envs_per_worker
    cfg.on_policy_n_minibatch_iters = 16
    cfg.on_policy_minibatch_size = 600
    cfg.lr = 3e-4
    cfg.evaluation = True
    cfg.evaluation_interval = 30_000
    cfg.evaluation_episodes = 8
    cfg.render = False
    cfg.loggers = loggers or []
    cfg.checkpoint_interval = max(iters // 10, 1)
    cfg.checkpoint_at_end = True
    return cfg
