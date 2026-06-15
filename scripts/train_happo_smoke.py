"""
HAPPO training on the WildfireSearchScenario via HARL.

HARL doesn't ship a VMAS interface, so this script:

  1. Monkey-patches HARL's env registry to recognise env_name="wildfire"
  2. Registers a minimal logger so HARL's runner doesn't fall over
  3. Builds algo_args + env_args from CLI flags
  4. Runs HARL's OnPolicyHARunner

Budgets:
  smoke    (default)    ~2 000 steps, episode_length=150   ≈ 5-10 s on CPU
  research (--research)  400 000 steps, episode_length=500 ≈ tens of minutes on CPU

Run from repo root:

    python scripts/train_happo_smoke.py
    python scripts/train_happo_smoke.py --research
    python scripts/train_happo_smoke.py --research --preset tuned
    python scripts/train_happo_smoke.py --num-env-steps 20000 --comms-dropout 0.3
    python scripts/train_happo_smoke.py --entropy-coef 0.05 --seed 42

Prerequisite: HARL must be installed in the active venv:

    git clone https://github.com/PKU-MARL/HARL ../HARL
    pip install -e ../HARL
"""

from __future__ import annotations

import argparse
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
    from harl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
    from harl.common.base_logger import BaseLogger

    from agents.harl_env import WildfireHARLEnv

    def _wildfire_env_fn(rank: int, seed: int, env_args: dict):
        def init_env():
            args = copy.deepcopy(env_args)
            args["seed"] = seed + rank * 1000
            return WildfireHARLEnv(args)
        return init_env

    _orig_train  = envs_tools.make_train_env
    _orig_eval   = envs_tools.make_eval_env
    _orig_render = envs_tools.make_render_env
    _orig_nagent = envs_tools.get_num_agents

    def make_train_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            fns = [_wildfire_env_fn(i, seed, env_args) for i in range(n_threads)]
            return ShareDummyVecEnv(fns) if n_threads == 1 else ShareSubprocVecEnv(fns)
        return _orig_train(env_name, seed, n_threads, env_args)

    def make_eval_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            fns = [_wildfire_env_fn(i, seed + 10_000, env_args) for i in range(n_threads)]
            return ShareDummyVecEnv(fns) if n_threads == 1 else ShareSubprocVecEnv(fns)
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

    _orig_task = configs_tools.get_task_name
    def get_task_name(env_name, env_args):
        if env_name == "wildfire":
            return "wildfire_search"
        return _orig_task(env_name, env_args)
    configs_tools.get_task_name = get_task_name

    class WildfireLogger(BaseLogger):
        def get_task_name(self):
            return "wildfire_search"

    harl_envs_pkg.LOGGER_REGISTRY["wildfire"] = WildfireLogger


# ----------------------------------------------------------------------
# Step 2 — build HARL args from CLI parameters
# ----------------------------------------------------------------------
def build_args(
    num_env_steps:  int,
    episode_length: int,
    seed:           int,
    comms_dropout:  float,
    entropy_coef:   float,
    exp_name:       str,
    n_rollout_threads: int = 1,
    terrain_cache_path: str | None = None,
    drone_min_footprint: float = 0.0,
    ground_confirm_min: float = 0.0,
    fire_grid_size: int = 128,
    reward_search: bool = False,
    reward_confirm: bool = False,
    recurrent: bool = False,
    model_dir: str | None = None,
    drone_camera_fov_deg: float | None = None,
    drone_flight_levels_m: tuple[float, ...] | None = None,
    ground_confirmation_range_m: float | None = None,
    coverage_obs_grid: int = 0,
) -> tuple[dict, dict, dict]:
    args = {
        "algo":        "happo",
        "env":         "wildfire",
        "exp_name":    exp_name,
        "load_config": "",
    }

    algo_args = {
        "seed":   {"seed_specify": True, "seed": seed},
        "device": {
            "cuda": False, "cuda_deterministic": True, "torch_threads": 4,
        },
        "train": {
            "n_rollout_threads":      n_rollout_threads,
            "num_env_steps":          num_env_steps,
            "episode_length":         episode_length,
            "log_interval":           1,
            "eval_interval":          1,
            "use_valuenorm":          True,
            "use_linear_lr_decay":    False,
            "use_proper_time_limits": True,
            "model_dir":              model_dir,
        },
        "eval": {
            "use_eval":               False,
            "n_eval_rollout_threads": 1,
            "eval_episodes":          2,
        },
        "render": {
            "use_render":      False,
            "render_episodes": 1,
        },
        "model": {
            "hidden_sizes":               [128, 128],
            "activation_func":            "relu",
            "use_feature_normalization":  True,
            "initialization_method":      "orthogonal_",
            "gain":                       0.01,
            "use_naive_recurrent_policy": False,
            "use_recurrent_policy":       recurrent,
            "recurrent_n":                1,
            "data_chunk_length":          10,
            "lr":                         5e-4,
            "critic_lr":                  5e-4,
            "opti_eps":                   1e-5,
            "weight_decay":               0,
            "std_x_coef":                 1,
            "std_y_coef":                 1.0,  # was 0.5; higher keeps action std from
                                                # collapsing into saturated corner-camping
        },
        "algo": {
            "ppo_epoch":               2,
            "critic_epoch":            2,
            "use_clipped_value_loss":  True,
            "clip_param":              0.2,
            "actor_num_mini_batch":    1,
            "critic_num_mini_batch":   1,
            "entropy_coef":            entropy_coef,
            "value_loss_coef":         1,
            "use_max_grad_norm":       True,
            "max_grad_norm":           10.0,
            "use_gae":                 True,
            "gamma":                   0.99,
            "gae_lambda":              0.95,
            "use_huber_loss":          True,
            "use_policy_active_masks": True,
            "huber_delta":             10.0,
            "action_aggregation":      "prod",
            "share_param":             False,
            "fixed_order":             False,
        },
        "logger": {
            "log_dir": str(ROOT / "results" / "harl_runs"),
        },
    }

    scenario_kwargs = {
        "max_steps":     episode_length,
        "n_drones":      3,
        "n_ground":      2,
        "comms_dropout": comms_dropout,
        "fire_grid_size": fire_grid_size,
        "drone_min_footprint": drone_min_footprint,
        "ground_confirm_min": ground_confirm_min,
    }
    if terrain_cache_path:
        scenario_kwargs["terrain_source"] = "real"
        scenario_kwargs["terrain_cache_path"] = terrain_cache_path
    # Physical (non-floor) sensor levers. On a small terrain these give a real
    # detection footprint at floor 0: footprint ~ flight_altitude * tan(fov/2),
    # both converted to sim units via sim_units_per_meter.
    if drone_camera_fov_deg is not None:
        scenario_kwargs["drone_camera_fov_deg"] = float(drone_camera_fov_deg)
    if drone_flight_levels_m is not None:
        scenario_kwargs["drone_flight_levels_m"] = tuple(float(v) for v in drone_flight_levels_m)
    if ground_confirmation_range_m is not None:
        scenario_kwargs["ground_confirmation_range_m"] = float(ground_confirmation_range_m)
    if coverage_obs_grid and coverage_obs_grid > 0:
        scenario_kwargs["coverage_obs_grid"] = int(coverage_obs_grid)
    if reward_search:
        # Search-dominant reward: make finding/scouting survivors clearly worth
        # more than any movement/hazard cost, and strengthen potential-based
        # shaping toward survivors, so the policy is rewarded for searching
        # rather than for sitting still to avoid cost (the degenerate optimum
        # under the default cost-heavy reward).
        scenario_kwargs.update({
            "r_found_survivor":   10.0,   # was 1.0
            "r_drone_scout":       2.0,   # was 0.3
            "r_ground_confirm":    4.0,   # was 0.5
            "r_drone_shaping":     0.30,  # was 0.05  (dense pull toward survivors)
            "r_ground_shaping":    0.50,  # was 0.10  (stronger directed pull)
            "r_ground_approach":   0.10,  # dense bonus peaking ON a scouted survivor
            "ground_approach_radius": 0.4,
            "r_fire_penalty":     -0.20,  # was -1.0  (no longer dominates)
            "r_ground_travel_cost": -0.01,  # was -0.05
            "r_drone_climb_cost":  -0.005,  # was -0.02
            "r_time_penalty":     -0.0005,  # was -0.001
            "r_coverage":          5.0,    # max team bonus for covering the full map once
        })
    if reward_confirm:
        # Confirmation-dominant reward. Diagnosis: under reward_search the ground
        # robots learn to stand still (UGV travel ~2.86 vs expert ~4.8) because
        # confirming is rare/costly and idling is a safe zero. This makes ground
        # confirmation the dominant signal, removes the costs that punish moving,
        # strengthens the dense pull toward scouted survivors, and adds a per-step
        # penalty for every scouted-but-unconfirmed survivor still waiting.
        scenario_kwargs.update({
            "r_found_survivor":      15.0,
            "r_ground_confirm":      12.0,   # dominant term
            "r_drone_scout":          1.0,   # was 2.0 (don't over-reward scouting alone)
            "r_drone_shaping":        0.15,
            "r_ground_shaping":       1.5,    # strong potential-based pull
            "r_ground_approach":      1.0,    # strong dense bonus peaking ON a survivor
            "ground_approach_radius": 0.6,
            "r_fire_penalty":         0.0,    # cut: was making robots avoid moving
            "r_ground_travel_cost":   0.0,    # cut: was making standing still optimal
            "r_drone_climb_cost":     0.0,
            "r_time_penalty":        -0.001,
            "r_coverage":             0.5,    # small: encourage spread, don't farm it
            "r_pending_penalty":     -0.005,  # per scouted-unconfirmed survivor per step
            "r_ground_coverage":      3.0,    # ground robots sweep instead of waiting
            "ground_coverage_radius": 0.08,
        })
    env_args = {
        "max_cycles":      episode_length,
        "scenario_kwargs": scenario_kwargs,
    }

    return args, algo_args, env_args


# ----------------------------------------------------------------------
# Step 3 — CLI + main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research",       action="store_true",
                   help="Larger budget: 400k steps, episode_length=500 (tens of minutes on CPU).")
    p.add_argument("--num-env-steps",  type=int,   default=None,
                   help="Total env steps override (default: 2000 smoke / 400000 research).")
    p.add_argument("--episode-length", type=int,   default=None,
                   help="Episode length override (default: 150 smoke / 500 research).")
    p.add_argument("--seed",           type=int,   default=1)
    p.add_argument("--comms-dropout",  type=float, default=0.0,
                   help="Per-step prob each agent's comms are dropped (default: 0.0).")
    p.add_argument("--entropy-coef",   type=float, default=0.01,
                   help="Higher (0.05+) encourages exploration — helps break "
                        "the drones-at-corners action-saturation pathology.")
    p.add_argument("--exp-name",       default="happo_smoke")
    p.add_argument("--n-rollout-threads", type=int, default=1,
                   help="Parallel rollout envs. More threads => more diverse data per "
                        "update and faster wall-clock (e.g. 8).")
    p.add_argument("--terrain-cache-path", default=None,
                   help="Train on this cached real terrain (recommended: match what you evaluate on, "
                        "e.g. data/terrain_cache/malibu_creek_1km_128.npz). Default uses the scenario default.")
    p.add_argument("--drone-min-footprint", type=float, default=0.0,
                   help="Floor on the drone scout footprint (sim units). >0 gives RL a learnable reward "
                        "signal on large terrains by ensuring drones actually scout survivors (e.g. 0.15).")
    p.add_argument("--ground-confirm-min", type=float, default=0.0,
                   help="Floor on ground confirm range (sim units). >0 gives ground robots a learnable confirm reward (e.g. 0.12).")
    p.add_argument("--fire-grid-size", type=int, default=128)
    p.add_argument("--model-dir", default=None,
                   help="Warm-start actors from a checkpoint dir (e.g. a behaviour-cloned results/bc_happo) and RL-fine-tune.")
    p.add_argument("--recurrent", action="store_true",
                   help="Use a recurrent (GRU) policy so agents remember where they have searched.")
    p.add_argument("--reward-search", action="store_true",
                   help="Use a search-dominant reward (survivor find/scout >> movement/hazard cost) "
                        "to avoid the do-nothing degenerate policy.")
    p.add_argument("--reward-confirm", action="store_true",
                   help="Use a confirmation-dominant reward: ground confirmation is the dominant "
                        "term, movement/fire costs are cut, and idling while survivors are scouted-"
                        "but-unconfirmed is penalized. Fixes the 'ground robots don't move' optimum.")
    p.add_argument("--drone-camera-fov-deg", type=float, default=None,
                   help="Drone camera field of view (deg). Wider FOV => larger scout footprint. "
                        "Lets RL get detection signal at floor 0 on small terrains (e.g. 140).")
    p.add_argument("--drone-flight-levels-m", default=None,
                   help="Comma-separated flight altitudes in meters (>=2 values), e.g. '50,80,100'. "
                        "Higher altitude => larger scout footprint.")
    p.add_argument("--ground-confirmation-range-m", type=float, default=None,
                   help="Ground confirmation range in meters (physical, not a floor), e.g. 30.")
    p.add_argument("--coverage-obs-grid", type=int, default=0,
                   help="Add a KxK team-coverage map + global fraction to the observation so the "
                        "policy can learn systematic sweeping (e.g. 6). 0 = off.")
    p.add_argument("--preset", choices=("smoke", "tuned", "floor0-1km"), default="smoke",
                   help="Preset for defaults. 'floor0-1km' (recommended) trains on the 1km terrain "
                        "with wide-FOV/high-altitude sensors so detection works at floor 0.")
    args = p.parse_args()

    drone_flight_levels_m = None
    if args.drone_flight_levels_m:
        drone_flight_levels_m = tuple(
            float(v) for v in str(args.drone_flight_levels_m).split(",") if v.strip()
        )

    if args.preset == "floor0-1km":
        # Validated floor-0 config: small (1km) terrain restores real detection
        # geometry, wide FOV + high altitude enlarge the scout footprint, and
        # long episodes give time to confirm. Expert recall ceiling ~0.47 here
        # at floor 0 (scripts/diag_floor0_ceiling.py), so RL has signal to learn.
        if args.terrain_cache_path is None:
            args.terrain_cache_path = str(
                ROOT / "data" / "terrain_cache" / "malibu_creek_1km_128.npz"
            )
        if args.drone_camera_fov_deg is None:
            args.drone_camera_fov_deg = 140.0
        if drone_flight_levels_m is None:
            drone_flight_levels_m = (50.0, 80.0, 100.0)
        if args.ground_confirmation_range_m is None:
            args.ground_confirmation_range_m = 30.0
        args.recurrent = True
        # Confirmation-dominant reward is the default for this preset (search
        # reward let the ground robots stand still). --reward-search still
        # overrides if explicitly requested.
        if not args.reward_search:
            args.reward_confirm = True
        # Floor stays 0 by design.
        args.drone_min_footprint = 0.0
        args.ground_confirm_min = 0.0
        if args.entropy_coef == 0.01:
            args.entropy_coef = 0.02
        # Team-coverage observation: lets the policy learn systematic sweeping.
        if args.coverage_obs_grid <= 0:
            args.coverage_obs_grid = 6

    if args.research:
        num_env_steps  = args.num_env_steps  or 400_000
        episode_length = args.episode_length or (1_000 if args.preset == "floor0-1km" else 500)
    elif args.preset == "floor0-1km":
        num_env_steps  = args.num_env_steps  or 240_000
        episode_length = args.episode_length or 1_000
    else:
        num_env_steps  = args.num_env_steps  or 2_000
        episode_length = args.episode_length or 150

    if args.preset == "tuned":
        # Keep explicit user values, but upgrade defaults to convergence-oriented
        # settings that consistently beat the smoke profile.
        if args.entropy_coef == 0.01:
            args.entropy_coef = 0.02
        if args.drone_min_footprint <= 0.0:
            args.drone_min_footprint = 0.15
        if args.ground_confirm_min <= 0.0:
            args.ground_confirm_min = 0.12
        args.recurrent = True
        args.reward_search = True

    print("=" * 60)
    print(f" OmniSearch — HAPPO ({'RESEARCH' if args.research else 'SMOKE'})")
    print(f" num_env_steps:  {num_env_steps}")
    print(f" episode_length: {episode_length}")
    print(f" seed:           {args.seed}")
    print(f" comms_dropout:  {args.comms_dropout}")
    print(f" entropy_coef:   {args.entropy_coef}")
    print(f" exp_name:       {args.exp_name}")
    print("=" * 60)

    _register_wildfire_with_harl()
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner
    from agents.happo_checkpoint import save_training_manifest

    harl_args, algo_args, env_args = build_args(
        num_env_steps  = num_env_steps,
        episode_length = episode_length,
        seed           = args.seed,
        comms_dropout  = args.comms_dropout,
        entropy_coef   = args.entropy_coef,
        exp_name       = args.exp_name,
        n_rollout_threads = args.n_rollout_threads,
        terrain_cache_path = args.terrain_cache_path,
        drone_min_footprint = args.drone_min_footprint,
        ground_confirm_min = args.ground_confirm_min,
        fire_grid_size = args.fire_grid_size,
        reward_search = args.reward_search,
        reward_confirm = args.reward_confirm,
        recurrent = args.recurrent,
        model_dir = args.model_dir,
        drone_camera_fov_deg = args.drone_camera_fov_deg,
        drone_flight_levels_m = drone_flight_levels_m,
        ground_confirmation_range_m = args.ground_confirmation_range_m,
        coverage_obs_grid = args.coverage_obs_grid,
    )
    print(f" log dir: {algo_args['logger']['log_dir']}")
    print("-" * 60)

    t0 = time.time()
    runner = OnPolicyHARunner(harl_args, algo_args, env_args)
    manifest_path = save_training_manifest(
        runner,
        harl_args=harl_args,
        algo_args=algo_args,
        env_args=env_args,
    )
    print(f" training config: {manifest_path}")
    print(f" tensorboard log dir: {runner.log_dir}")
    print(f" tensorboard: tensorboard --logdir \"{runner.log_dir}\" --port 6006")
    runner.run()
    runner.close()

    print("-" * 60)
    print(f" HAPPO training complete in {time.time() - t0:.1f}s")
    print(f" Checkpoints saved to: {algo_args['logger']['log_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
