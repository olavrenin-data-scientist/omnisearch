"""
Smoke training: HAPPO on the WildfireSearchScenario via HARL.

HARL doesn't ship a VMAS interface, so this script:

  1. Imports our HARL-compatible adapter (agents/harl_env.py)
  2. Monkey-patches HARL's env registry to recognise env_name="wildfire"
  3. Registers a minimal logger so HARL's runner doesn't fall over
  4. Constructs algo_args + env_args at SMOKE budget (very small)
  5. Runs HARL's OnPolicyHARunner — that's the HAPPO/HATRPO runner

The smoke budget is intentionally tiny (~30 s wall on CPU): two short
episodes, one rollout thread, two ppo epochs. The success criterion is
"HAPPO trained without crashing and the actor + critic updated at least
once." Tune the values in build_args() for a real research run.

Run from repo root:

    python scripts/train_happo_smoke.py

Prerequisite: HARL must be installed in the active venv:

    git clone https://github.com/PKU-MARL/HARL  ../HARL
    pip install -e ../HARL
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# Step 1 — monkey-patch HARL's env registry to recognise "wildfire"
# ----------------------------------------------------------------------
def _register_wildfire_with_harl():
    import harl.envs as harl_envs_pkg
    import harl.utils.envs_tools as envs_tools
    import harl.utils.configs_tools as configs_tools
    from harl.common.base_logger import BaseLogger

    from agents.harl_env import WildfireHARLEnv
    from agents.harl_vec_env import make_batched_wildfire_vec_env

    _orig_train  = envs_tools.make_train_env
    _orig_eval   = envs_tools.make_eval_env
    _orig_render = envs_tools.make_render_env
    _orig_nagent = envs_tools.get_num_agents

    def make_train_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            # n_threads is the number of parallel envs we want. Build ONE
            # batched VMAS env at num_envs=n_threads — single tensor op per
            # step, no subprocess overhead.
            return make_batched_wildfire_vec_env(n_threads, seed, env_args)
        return _orig_train(env_name, seed, n_threads, env_args)

    def make_eval_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            return make_batched_wildfire_vec_env(n_threads, seed + 10_000, env_args)
        return _orig_eval(env_name, seed, n_threads, env_args)

    def make_render_env(env_name, seed, env_args):
        if env_name == "wildfire":
            env = WildfireHARLEnv({**env_args, "seed": seed})
            return env, env.n_agents, env.agents
        return _orig_render(env_name, seed, env_args)

    def get_num_agents(env_name, env_args, envs):
        if env_name == "wildfire":
            return envs.n_agents
        return _orig_nagent(env_name, env_args, envs)

    envs_tools.make_train_env  = make_train_env
    envs_tools.make_eval_env   = make_eval_env
    envs_tools.make_render_env = make_render_env
    envs_tools.get_num_agents  = get_num_agents

    # Patch task-name lookup (run-dir + tensorboard tag use this)
    _orig_task = configs_tools.get_task_name
    def get_task_name(env_name, env_args):
        if env_name == "wildfire":
            return "wildfire_search"
        return _orig_task(env_name, env_args)
    configs_tools.get_task_name = get_task_name

    # Minimal logger — HARL looks up LOGGER_REGISTRY[args["env"]]
    class WildfireLogger(BaseLogger):
        def get_task_name(self):
            return "wildfire_search"

    harl_envs_pkg.LOGGER_REGISTRY["wildfire"] = WildfireLogger


# ----------------------------------------------------------------------
# Step 2 — build smoke-budget args
# ----------------------------------------------------------------------
def build_args():
    args = {
        "algo":      "happo",
        "env":       "wildfire",
        "exp_name":  "happo_smoke",
        "load_config": "",
    }

    algo_args = {
        "seed":   {"seed_specify": True, "seed": 1},
        "device": {
            "cuda": False, "cuda_deterministic": True, "torch_threads": 4,
        },
        "train": {
            # n_rollout_threads now drives BatchedVMASVecEnv's num_envs —
            # one VMAS instance running this many envs in a single batch.
            "n_rollout_threads":     8,
            "num_env_steps":         8_000,    # smoke (8 envs × 100 steps × ~10 updates)
            "episode_length":        100,
            "log_interval":          1,
            "eval_interval":         1,
            "use_valuenorm":         True,
            "use_linear_lr_decay":   False,
            "use_proper_time_limits": True,
            "model_dir":             None,
        },
        "eval": {
            "use_eval":               False,
            "n_eval_rollout_threads": 1,
            "eval_episodes":          2,
        },
        "render": {
            "use_render":     False,
            "render_episodes": 1,
        },
        "model": {
            "hidden_sizes":  [128, 128],
            "activation_func": "relu",
            "use_feature_normalization": True,
            "initialization_method": "orthogonal_",
            "gain":          0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy":       False,
            "recurrent_n":   1,
            "data_chunk_length": 10,
            "lr":            5e-4,
            "critic_lr":     5e-4,
            "opti_eps":      1e-5,
            "weight_decay":  0,
            "std_x_coef":    1,
            "std_y_coef":    0.5,
        },
        "algo": {
            "ppo_epoch":         2,
            "critic_epoch":      2,
            "use_clipped_value_loss": True,
            "clip_param":        0.2,
            "actor_num_mini_batch":  1,
            "critic_num_mini_batch": 1,
            "entropy_coef":      0.01,
            "value_loss_coef":   1,
            "use_max_grad_norm": True,
            "max_grad_norm":     10.0,
            "use_gae":           True,
            "gamma":             0.99,
            "gae_lambda":        0.95,
            "use_huber_loss":    True,
            "use_policy_active_masks": True,
            "huber_delta":       10.0,
            "action_aggregation": "prod",
            "share_param":       False,
            "fixed_order":       False,
        },
        "logger": {
            "log_dir":  str(ROOT / "results" / "harl_runs"),
        },
    }

    env_args = {
        "max_cycles":      100,
        "scenario_kwargs": {"max_steps": 100, "n_drones": 3, "n_ground": 2},
    }

    return args, algo_args, env_args


# ----------------------------------------------------------------------
# Step 3 — drive HARL's HAPPO runner
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" OmniSearch — HAPPO smoke training (HARL)")
    print("=" * 60)

    _register_wildfire_with_harl()
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner

    args, algo_args, env_args = build_args()
    print(f" total env steps:   {algo_args['train']['num_env_steps']}")
    print(f" episode length:    {algo_args['train']['episode_length']}")
    print(f" rollout threads:   {algo_args['train']['n_rollout_threads']}")
    print(f" output dir:        {algo_args['logger']['log_dir']}")
    print("-" * 60)

    t0 = time.time()
    runner = OnPolicyHARunner(args, algo_args, env_args)
    runner.run()
    runner.close()
    print("-" * 60)
    print(f" HAPPO smoke complete in {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
