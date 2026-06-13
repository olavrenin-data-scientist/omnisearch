"""
Behaviour-clone HAPPO actors from the lawnmower baseline, then save a
HARL-compatible checkpoint that ``train_happo_smoke.py --model-dir`` can
RL-fine-tune.

Pure RL plateaus at recall ~0.20 on the wildfire search task — it learns the
sub-behaviours but not the near-optimal coordination the hand-coded lawnmower
heuristic encodes. This pre-trains each HAPPO actor to *imitate* lawnmower's
actions (max-likelihood / negative-log-prob behaviour cloning from collected
demonstrations), giving RL a strong starting point near heuristic performance.

    python scripts/train_bc_happo.py --demos 60 --epochs 40 \
        --terrain-cache-path data/terrain_cache/malibu_creek_1km_128.npz \
        --drone-min-footprint 0.15 --ground-confirm-min 0.20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmas
from gymnasium.spaces import Box

from envs.wildfire_search import WildfireSearchScenario
from agents.baselines import LawnmowerPolicy
from agents.harl_runner import default_algo_args


def collect_demos(scenario_kwargs, demos, steps, seed0):
    """Roll out lawnmower; return per-agent (obs, action) arrays."""
    obs_by_agent, act_by_agent = None, None
    for d in range(demos):
        e = vmas.make_env(scenario=WildfireSearchScenario(), num_envs=2, device="cpu",
                          continuous_actions=True, seed=seed0 + d, max_steps=steps, **scenario_kwargs)
        e.reset(); sc = e.scenario
        n_agents = len(e.agents)
        if obs_by_agent is None:
            obs_by_agent = [[] for _ in range(n_agents)]
            act_by_agent = [[] for _ in range(n_agents)]
        pol = LawnmowerPolicy(e)
        for _ in range(steps):
            actions = pol(e)
            for i, agent in enumerate(e.agents):
                obs_by_agent[i].append(sc.observation(agent).cpu().numpy())   # (B, obs)
                a = actions[i]
                act_by_agent[i].append((a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)))
            e.step(actions)
    obs = [np.concatenate(o, axis=0).astype(np.float32) for o in obs_by_agent]
    act = [np.clip(np.concatenate(a, axis=0), -1.0, 1.0).astype(np.float32) for a in act_by_agent]
    return obs, act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=int, default=60, help="Number of lawnmower rollouts to collect.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--terrain-cache-path", default="data/terrain_cache/malibu_creek_1km_128.npz")
    p.add_argument("--drone-min-footprint", type=float, default=0.15)
    p.add_argument("--ground-confirm-min", type=float, default=0.20)
    p.add_argument("--fire-grid-size", type=int, default=128)
    p.add_argument("--out", default=str(ROOT / "results" / "bc_happo"))
    args = p.parse_args()

    scenario_kwargs = dict(
        n_drones=3, n_ground=2, fire_grid_size=args.fire_grid_size,
        terrain_source="real", terrain_cache_path=args.terrain_cache_path,
        drone_min_footprint=args.drone_min_footprint, ground_confirm_min=args.ground_confirm_min,
    )
    print(f"Collecting {args.demos} lawnmower demos x {args.steps} steps ...")
    obs, act = collect_demos(scenario_kwargs, args.demos, args.steps, args.seed)
    print(f"  per-agent samples: {[o.shape[0] for o in obs]}, obs_dim={obs[0].shape[1]}, act_dim={act[0].shape[1]}")

    # Build HAPPO actors with the same architecture the fine-tune will use.
    from harl.algorithms.actors.happo import HAPPO
    algo_args = default_algo_args()
    merged = {**algo_args["model"], **algo_args["algo"], **algo_args["train"]}
    tmp = vmas.make_env(scenario=WildfireSearchScenario(), num_envs=1, device="cpu",
                        continuous_actions=True, seed=0, **scenario_kwargs)
    tmp.reset()
    obs_spaces = list(tmp.observation_space.spaces)
    act_spaces = [Box(-1.0, 1.0, (tmp.get_agent_action_size(a),), dtype=np.float32) for a in tmp.agents]

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    rnn_n, rnn_h = merged["recurrent_n"], merged["hidden_sizes"][-1]

    for i, (obs_s, act_s) in enumerate(zip(obs_spaces, act_spaces)):
        actor = HAPPO(merged, obs_s, act_s, device)
        net = actor.actor
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        X = torch.from_numpy(obs[i]); Y = torch.from_numpy(act[i])
        N = X.shape[0]
        for ep in range(args.epochs):
            perm = rng.permutation(N)
            tot = 0.0; nb = 0
            for s in range(0, N, args.batch):
                idx = perm[s:s + args.batch]
                xb, yb = X[idx], Y[idx]
                B = xb.shape[0]
                rnn = np.zeros((B, rnn_n, rnn_h), dtype=np.float32)
                masks = np.ones((B, 1), dtype=np.float32)
                # NLL behaviour cloning: maximise log-prob of the demo action.
                log_probs, _entropy, _dist = net.evaluate_actions(xb.numpy(), rnn, yb.numpy(), masks)
                loss = -log_probs.mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"  agent{i} epoch {ep:2d}/{args.epochs}  NLL={tot/max(nb,1):.4f}")
        torch.save(net.state_dict(), out_dir / f"actor_agent{i}.pt")

    # Save a config.json so the loader/fine-tune knows the architecture.
    with (out_dir / "config.json").open("w") as f:
        json.dump({"algo_args": algo_args}, f)
    print(f"\nBC checkpoint saved to: {out_dir}")
    print("Fine-tune with:  python scripts/train_happo_smoke.py --model-dir", out_dir,
          "--num-env-steps ... --reward-search --recurrent ...")


if __name__ == "__main__":
    main()
