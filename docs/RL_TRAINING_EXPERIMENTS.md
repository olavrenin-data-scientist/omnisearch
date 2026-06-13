# OmniSearch — MARL Training Experiments (HAPPO)

A record of the reinforcement-learning training runs for the wildfire search-and-rescue
scenario: what was tried, episode budgets, training hours, results, and comparison against
the non-learning baselines.

Author: Oleksii Lavrenin (seniorQAautomationEngineer). A formatted PDF of this report lives at
[RL_TRAINING_EXPERIMENTS.pdf](RL_TRAINING_EXPERIMENTS.pdf).

---

## Goal & question

**Can a trained MARL policy (HAPPO) match or beat hand-coded coordination heuristics** on the
mission metric **survivor recall** (fraction of 5 survivors confirmed by a ground robot)? Drones
scout (broad downward camera); ground robots confirm (precise, slow). Three hand-coded baselines
provide the bar.

## Evaluation conditions (all numbers below)

- **Terrain:** real cached terrain, ~1 km bbox (`data/terrain_cache/malibu_creek_1km_128.npz`), 128×128 grid.
- **Episode:** 1000 steps; `recall = found / 5` at episode end; mean over 5–10 seeds.
- **Reward-signal floors (opt-in, default 0.0):** `drone_min_footprint = 0.15`, `ground_confirm_min = 0.20`.
  On a real-km terrain the *physical* drone footprint (~0.033) and confirm range (~0.017) shrink so far
  that neither RL nor the baselines get a usable scout/confirm reward; the floors restore it. **Both HAPPO
  and the baselines are evaluated at the same floors** — comparisons are apples-to-apples. Repo defaults
  remain 0.0 (physical). *Results are only comparable across the team at the same floor settings.*

---

## Non-learning baselines (the bar to beat)

Fixed hand-coded heuristics — no training. (Measured at floors 0.15 / 0.20, 1 km, 1000 steps.)

| Baseline | How it works | scouted | recall |
|---|---|---|---|
| random | everyone moves randomly (control) | 0.62 | 0.06 |
| nearest_candidate | random drones; UGVs → nearest scouted survivor | 0.60 | 0.62 |
| lawnmower | serpentine drone sweep; UGVs A* to nearest scouted | 0.82 | 0.92 |

---

## HAPPO training experiments (chronological)

Each run diagnosed and fixed a distinct bottleneck. Episodes × 500 steps = env steps.
`*` = short diagnostic run.

| # | Episodes | hrs | Key configuration | scouted | recall | Finding / bottleneck |
|---|---|---|---|---|---|---|
| 1 | 100 | 0.1 | default 22 km terrain, no floors | 0.2 | 0.07 | Wrong (huge) terrain + no reward signal |
| 2 | 1000 | 0.6 | 1 km, footprint floor 0.15 | 0.33 | 0.00 | Cost-dominated reward → do-nothing optimum |
| 3 | 2000 | 1.1 | +entropy 0.08, search-reward | 0.16 | 0.00 | Reward-hacks dense shaping; action saturation |
| 4 | 300* | 0.2 | +std_y 1.0, +entropy 0.25 | 0.47 | 0.00 | Action saturation **broken**; scouting recovers |
| 5 | 3000 | 2.0 | std1.0 + ent0.25 + search | 0.56 | 0.00 | Scouting up; ground robots never confirm |
| 6 | 4000 | 2.6 | +ground_confirm floor 0.12 | 0.36 | 0.16 | Recall lifts off 0 for the first time |
| 7 | 10000 | 8.6 | same (non-recurrent) | 0.33 | 0.10 | **Plateau** — more budget does NOT help |
| 8 | 6000 | 5.2 | +coverage reward +recurrent (GRU) | 0.62 | 0.12 | Coverage breakthrough; scouting now competitive |
| 9 | 300* | 0.2 | +ground approach reward, confirm 0.20 | 0.56 | 0.20 | Ground-confirm leg lifts recall to 0.20 |
| 10 | 6000 | 5.9 | all fixes (coverage+recurrent+approach) | 0.56 | 0.20 | Plateaus at 0.20 |

**Total HAPPO training ≈ 31 hours** across all runs (above, plus earlier 80k/400k experiments),
plus behaviour-cloning / DAgger / evaluation. On CPU (Apple M-series): ~150–280 env-steps/sec;
recurrent (GRU) runs ~30 % slower.

---

## Learning from demonstration (cloning the lawnmower expert)

Pure RL plateaus at recall ~0.20. To approach the heuristic, clone its behaviour, then RL-fine-tune.

| Approach | scouted | recall | Note |
|---|---|---|---|
| Behaviour cloning (feedforward), no RL | 0.74 | 0.30 | Clones the sweep well; ground-confirm precision doesn't transfer |
| BC + RL fine-tune | 0.46 | 0.36 | RL erodes the cloned sweep |
| **Recurrent BC (GRU, BPTT)** | **0.74** | **0.38** | **Best learned** — memory captures multi-step navigation |
| DAgger (clone precision) | — | *in progress* | Re-label clone's own states with expert; fixes drift |

---

## Comparison: learned vs. non-learning

| Policy | Trained? | recall |
|---|---|---|
| HAPPO — pure RL (best, all fixes) | yes | 0.20 |
| HAPPO — BC + RL fine-tune | yes | 0.36 |
| **HAPPO — recurrent BC (best learned)** | yes | **0.38** |
| random | no | 0.06 |
| nearest_candidate | no | 0.62 |
| lawnmower | no | 0.92 |

---

## Conclusion (honest)

- **HAPPO improved from a do-nothing collapse (0.00) to a functioning searcher** — best learned recall
  **0.38** (recurrent BC). Drone scouting became competitive with the baselines (0.56–0.74 vs lawnmower 0.82).
- **It does NOT beat the hand-coded baselines** (nearest 0.62, lawnmower 0.92). Pure RL plateaus at ~0.20 —
  more episodes do not help (1k and 10k both ~0.10–0.20). Behaviour cloning lifts a learned policy to 0.30–0.38.
- **Why:** six bottlenecks were diagnosed and fixed (footprint floor, action-saturation, coverage reward,
  recurrent memory, ground-confirm floor, ground-approach reward). The residual gap is the ground robots'
  A* approach-to-survivor **precision**, which neither RL nor feedforward BC reproduces. Recurrent BC and
  DAgger target exactly this.
- **Scientific takeaway:** hand-coded coordination heuristics remain superior on mission recall; learned
  MARL matches the individual *scouting* sub-behaviour but not the full *scout → confirm* coordination at
  this budget.

---

## Code & commits

Training scripts:
- `scripts/train_happo_smoke.py` — HAPPO via HARL. Flags: `--terrain-cache-path`, `--drone-min-footprint`,
  `--ground-confirm-min`, `--entropy-coef`, `--reward-search`, `--recurrent`, `--model-dir`.
- `scripts/train_bc_happo.py` — behaviour cloning (feedforward + `--recurrent` BPTT) from the lawnmower.
- `scripts/train_dagger_happo.py` — DAgger (iteratively re-label the clone's own states with the expert).
- MAPPO/IPPO smoke scripts exist (BenchMARL) but their checkpoint→eval loaders are not wired, so they are
  not in the comparison.

Key commits:
- `34f27b6` — Add behaviour cloning from lawnmower + RL fine-tune for HAPPO
- `dcd92e1` — Add ground directed-approach reward for the scout→confirm leg
- `384fa70` — Add coverage reward + recurrent policy: HAPPO learns systematic search
- `1c62f87` — Make HAPPO trainable: opt-in floors, anti-saturation, search reward
- `122acda` — Floor drone scout footprint so survivors are detectable on real terrain

> Note: all recall numbers are measured at the opt-in floors (0.15 / 0.20), not the physical defaults (0.0).
> Standardize the floor values across the team for comparable results.
