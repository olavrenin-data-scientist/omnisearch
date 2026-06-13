"""
DAgger (Dataset Aggregation) to clone the lawnmower expert's *precision* into
HAPPO actors — beyond what plain behaviour cloning achieves.

Plain BC drifts: the cloned policy visits states the expert never demonstrated,
acts imprecisely there, and the errors compound (this is why BC scouts well but
fumbles the ground robots' A* approach-to-survivor). DAgger fixes this: roll out
the *current* policy, query the lawnmower expert for the correct action at every
state the policy actually visits, aggregate those (state, expert-action) pairs,
and retrain. Iterating tightens the clone on its own state distribution.

    python scripts/train_dagger_happo.py --init results/bc_happo \
        --iterations 8 --rollout-steps 500 --rollouts 8 --epochs 8 \
        --terrain-cache-path data/terrain_cache/malibu_creek_1km_128.npz
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from agents.happo_policy import HappoPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="results/bc_happo", help="Starting checkpoint (a BC one).")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=8, help="Policy rollouts per DAgger iteration.")
    p.add_argument("--rollout-steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=8, help="Retrain epochs per iteration.")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta0", type=float, default=0.5,
                   help="Initial prob of acting with the EXPERT (decays to 0); eases distribution shift.")
    p.add_argument("--max-samples", type=int, default=400_000, help="Cap aggregated dataset size per agent.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--terrain-cache-path", default="data/terrain_cache/malibu_creek_1km_128.npz")
    p.add_argument("--drone-min-footprint", type=float, default=0.15)
    p.add_argument("--ground-confirm-min", type=float, default=0.20)
    p.add_argument("--fire-grid-size", type=int, default=128)
    p.add_argument("--out", default="results/dagger_happo")
    args = p.parse_args()

    sk = dict(n_drones=3, n_ground=2, fire_grid_size=args.fire_grid_size,
              terrain_source="real", terrain_cache_path=args.terrain_cache_path,
              drone_min_footprint=args.drone_min_footprint, ground_confirm_min=args.ground_confirm_min)

    # Architecture + spaces from the init checkpoint's config.
    algo_args = json.load(open(Path(args.init) / "config.json"))["algo_args"]
    merged = {**algo_args["model"], **algo_args["algo"], **algo_args["train"]}
    rnn_n, rnn_h = merged["recurrent_n"], merged["hidden_sizes"][-1]
    from harl.algorithms.actors.happo import HAPPO
    tmp = vmas.make_env(scenario=WildfireSearchScenario(), num_envs=1, device="cpu",
                        continuous_actions=True, seed=0, **sk)
    tmp.reset()
    obs_spaces = list(tmp.observation_space.spaces)
    act_spaces = [Box(-1.0, 1.0, (tmp.get_agent_action_size(a),), dtype=np.float32) for a in tmp.agents]
    n_agents = len(tmp.agents)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Seed the working checkpoint from --init (actors + critic + config).
    for f in Path(args.init).glob("*"):
        shutil.copy(f, out / f.name)

    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    data_obs = [[] for _ in range(n_agents)]   # aggregated dataset
    data_act = [[] for _ in range(n_agents)]

    for it in range(args.iterations):
        beta = args.beta0 * (1.0 - it / max(args.iterations - 1, 1))   # expert mixing, decays to 0
        # ---- 1. Roll out current policy; label every visited state with the expert ----
        for r in range(args.rollouts):
            e = vmas.make_env(scenario=WildfireSearchScenario(), num_envs=2, device="cpu",
                              continuous_actions=True, seed=args.seed + it * 100 + r, max_steps=args.rollout_steps, **sk)
            e.reset(); sc = e.scenario
            policy = HappoPolicy.from_checkpoint(str(out)); policy.reset()
            expert = LawnmowerPolicy(e)
            for _ in range(args.rollout_steps):
                expert_a = expert(e)                         # label
                policy_a = policy(e)                         # current clone
                for i, agent in enumerate(e.agents):
                    data_obs[i].append(sc.observation(agent).cpu().numpy())
                    ea = expert_a[i]
                    data_act[i].append(ea.cpu().numpy() if torch.is_tensor(ea) else np.asarray(ea))
                # Act with expert w.p. beta (eases shift early), else the clone.
                step_a = expert_a if rng.random() < beta else policy_a
                e.step(step_a)
        # ---- 2. Retrain actors on the aggregated dataset ----
        nll_last = []
        for i in range(n_agents):
            X = np.concatenate(data_obs[i], 0).astype(np.float32)
            Y = np.clip(np.concatenate(data_act[i], 0), -1, 1).astype(np.float32)
            if X.shape[0] > args.max_samples:                 # keep most-recent
                X = X[-args.max_samples:]; Y = Y[-args.max_samples:]
            data_obs[i] = [X]; data_act[i] = [Y]              # compact
            actor = HAPPO(merged, obs_spaces[i], act_spaces[i], device)
            actor.actor.load_state_dict(torch.load(out / f"actor_agent{i}.pt", map_location=device, weights_only=True))
            opt = torch.optim.Adam(actor.actor.parameters(), lr=args.lr)
            N = X.shape[0]; Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
            for ep in range(args.epochs):
                perm = rng.permutation(N)
                for s in range(0, N, args.batch):
                    idx = perm[s:s + args.batch]; B = len(idx)
                    rnn = np.zeros((B, rnn_n, rnn_h), dtype=np.float32); masks = np.ones((B, 1), dtype=np.float32)
                    lp, _e, _d = actor.actor.evaluate_actions(Xt[idx].numpy(), rnn, Yt[idx].numpy(), masks)
                    loss = -lp.mean(); opt.zero_grad(); loss.backward(); opt.step()
            nll_last.append(float(loss.detach()))
            torch.save(actor.actor.state_dict(), out / f"actor_agent{i}.pt")
        print(f"iter {it+1}/{args.iterations}  beta={beta:.2f}  dataset={data_obs[0][0].shape[0]}/agent  NLL={np.mean(nll_last):.3f}")

    print(f"\nDAgger checkpoint saved to: {out}")


if __name__ == "__main__":
    main()
