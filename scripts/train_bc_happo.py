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
    """Roll out lawnmower; return per-agent list of (T, B, dim) rollout arrays.

    Sequences are kept intact (not concatenated) so recurrent BC can do
    backprop-through-time over them.
    """
    obs_rollouts, act_rollouts = None, None
    for d in range(demos):
        e = vmas.make_env(scenario=WildfireSearchScenario(), num_envs=2, device="cpu",
                          continuous_actions=True, seed=seed0 + d, max_steps=steps, **scenario_kwargs)
        e.reset(); sc = e.scenario
        n_agents = len(e.agents)
        if obs_rollouts is None:
            obs_rollouts = [[] for _ in range(n_agents)]
            act_rollouts = [[] for _ in range(n_agents)]
        pol = LawnmowerPolicy(e)
        o_steps = [[] for _ in range(n_agents)]; a_steps = [[] for _ in range(n_agents)]
        for _ in range(steps):
            actions = pol(e)
            for i, agent in enumerate(e.agents):
                o_steps[i].append(sc.observation(agent).cpu().numpy())   # (B, obs)
                a = actions[i]
                a_steps[i].append(a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a))
            e.step(actions)
        for i in range(n_agents):
            obs_rollouts[i].append(np.stack(o_steps[i], 0).astype(np.float32))               # (T, B, obs)
            act_rollouts[i].append(np.clip(np.stack(a_steps[i], 0), -1, 1).astype(np.float32))  # (T, B, act)
    return obs_rollouts, act_rollouts


def _make_sequences(rollouts, chunk):
    """Turn per-agent [(T,B,dim)...] into stacked (chunk, n_seq, dim) sequences."""
    segs = []
    for arr in rollouts:                       # arr: (T, B, dim)
        T, B, D = arr.shape
        n = (T // chunk) * chunk
        if n == 0:
            continue
        a = arr[:n].reshape(n // chunk, chunk, B, D)   # (n_chunks, L, B, D)
        a = a.transpose(0, 2, 1, 3).reshape(-1, chunk, D)  # (n_chunks*B, L, D)
        segs.append(a)
    seqs = np.concatenate(segs, 0)             # (N, L, D)
    return seqs.transpose(1, 0, 2)             # (L, N, D)  (T-major friendly)


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
    p.add_argument("--recurrent", action="store_true",
                   help="Clone into a recurrent (GRU) policy via BPTT over sequences — "
                        "captures the multi-step navigation (waypoint following) a feedforward clone misses.")
    p.add_argument("--chunk", type=int, default=50, help="BPTT sequence length for recurrent BC.")
    p.add_argument("--seq-batch", type=int, default=64, help="Sequences per recurrent BC minibatch.")
    p.add_argument("--out", default=str(ROOT / "results" / "bc_happo"))
    args = p.parse_args()

    scenario_kwargs = dict(
        n_drones=3, n_ground=2, fire_grid_size=args.fire_grid_size,
        terrain_source="real", terrain_cache_path=args.terrain_cache_path,
        drone_min_footprint=args.drone_min_footprint, ground_confirm_min=args.ground_confirm_min,
    )
    print(f"Collecting {args.demos} lawnmower demos x {args.steps} steps ({'recurrent' if args.recurrent else 'feedforward'} BC) ...")
    obs_rollouts, act_rollouts = collect_demos(scenario_kwargs, args.demos, args.steps, args.seed)

    # Build HAPPO actors with the same architecture the fine-tune will use.
    from harl.algorithms.actors.happo import HAPPO
    algo_args = default_algo_args()
    if args.recurrent:
        algo_args = {**algo_args, "model": {**algo_args["model"], "use_recurrent_policy": True}}
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

        if not args.recurrent:
            # Feedforward BC: shuffle individual (obs, action) samples.
            X = torch.from_numpy(np.concatenate([r.reshape(-1, r.shape[-1]) for r in obs_rollouts[i]], 0))
            Y = torch.from_numpy(np.concatenate([r.reshape(-1, r.shape[-1]) for r in act_rollouts[i]], 0))
            N = X.shape[0]
            for ep in range(args.epochs):
                perm = rng.permutation(N); tot = 0.0; nb = 0
                for s in range(0, N, args.batch):
                    idx = perm[s:s + args.batch]
                    B = len(idx)
                    rnn = np.zeros((B, rnn_n, rnn_h), dtype=np.float32)
                    masks = np.ones((B, 1), dtype=np.float32)
                    log_probs, _e, _d = net.evaluate_actions(X[idx].numpy(), rnn, Y[idx].numpy(), masks)
                    loss = -log_probs.mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss); nb += 1
                if ep % 10 == 0 or ep == args.epochs - 1:
                    print(f"  agent{i} epoch {ep:2d}/{args.epochs}  NLL={tot/max(nb,1):.4f}")
        else:
            # Recurrent BC: BPTT over (chunk)-length sequences. Layout is T-major
            # (L*N_seq, dim) with rnn_states (N_seq, recurrent_n, hidden), which is
            # exactly what HARL's evaluate_actions expects for sequence training.
            Oseq = _make_sequences(obs_rollouts[i], args.chunk)   # (L, Nseq, obs)
            Aseq = _make_sequences(act_rollouts[i], args.chunk)   # (L, Nseq, act)
            L, Nseq, _ = Oseq.shape
            for ep in range(args.epochs):
                perm = rng.permutation(Nseq); tot = 0.0; nb = 0
                for s in range(0, Nseq, args.seq_batch):
                    sel = perm[s:s + args.seq_batch]
                    M = len(sel)
                    ob = Oseq[:, sel, :].reshape(L * M, -1)        # T-major flatten
                    ac = Aseq[:, sel, :].reshape(L * M, -1)
                    rnn = np.zeros((M, rnn_n, rnn_h), dtype=np.float32)
                    masks = np.ones((L * M, 1), dtype=np.float32)
                    log_probs, _e, _d = net.evaluate_actions(ob, rnn, ac, masks)
                    loss = -log_probs.mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss); nb += 1
                if ep % 10 == 0 or ep == args.epochs - 1:
                    print(f"  agent{i} epoch {ep:2d}/{args.epochs}  NLL={tot/max(nb,1):.4f}  (seqs={Nseq}, L={L})")
        torch.save(net.state_dict(), out_dir / f"actor_agent{i}.pt")

    # Save a config.json so the loader/fine-tune knows the architecture.
    with (out_dir / "config.json").open("w") as f:
        json.dump({"algo_args": algo_args}, f)
    print(f"\nBC checkpoint saved to: {out_dir}")
    print("Fine-tune with:  python scripts/train_happo_smoke.py --model-dir", out_dir,
          "--num-env-steps ... --reward-search --recurrent ...")


if __name__ == "__main__":
    main()
