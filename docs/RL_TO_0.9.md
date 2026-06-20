# From ~0 to 0.97 — how we got HAPPO to ≥0.9

The kinds of changes that took survivor recall (fraction of 5 survivors confirmed) from a do-nothing
**0.0** to a **learned 0.97** — honestly, under realistic sensors. Config: 1 km terrain, floor 0,
line-of-sight enforced, drone EO/IR confirmation, `n_ground=2`, 1000-step episodes.

| Metric | Value |
|---|---|
| **Learned HAPPO (BC + RL)** | **0.97** |
| Learned HAPPO under 30% comms dropout | 0.97 |
| From-scratch RL (plateau) | 0.60 |
| Lawnmower expert (ceiling) | 1.00 |

**The one-line story:** three kinds of change, in order — **fix the world** (make the map small
enough that detection is physically possible), **fix the model, not the sensors** (let drones confirm
from altitude with line-of-sight — the honest route to ≥0.9), and **fix how we learn** (clone the
expert, then RL fine-tune, and keep the best checkpoint).

---

## The recall journey

| Stage | Conditions | Recall | Real? |
|---|---|---|---|
| Pure RL (early) | 1 km, basic reward shaping | ~0.20 | real but weak |
| Generous sensors | confirm range ≈ ½ map, see-through terrain | 0.90 | **artifact** |
| + Line-of-sight gate | same, but confirmation needs a clear view | 0.40 | exposes the artifact |
| From-scratch (realistic) | floor 0, LOS, drone-confirm | 0.60 | real, but below ceiling |
| Lawnmower expert | floor 0, LOS, drone-confirm | 1.00 | real (scripted) |
| **BC warm-start + RL fine-tune** | floor 0, LOS, drone-confirm | **0.97** | **real (learned)** |

*Recall = confirmed / 5 survivors, mean over 6 seeds × 1000-step episodes. "Artifact" = the number
was real but came from trivializing the task, not from coordination.*

**Honesty — the 0.90 was an artifact:** requiring line-of-sight collapses the generous-sensor 0.90 to
0.40; realistic aerial confirmation recovers it to 1.00 honestly (0.90 → 0.40 → 1.00).

**The learned-policy lever:** under the realistic config, cloning the expert then RL-fine-tuning lifts
the learned policy from 0.60 to 0.97 — close to the scripted 1.00 ceiling.

---

## What kinds of changes — and why each mattered

### 1. Fix the world geometry — *precondition*

Switched from the default ~22 km map to a **1 km terrain**. On the big map a survivor is a
~0.04 %-of-map pinpoint and **everyone scored 0.0** — the geometry, not the policy, was the blocker.

*Effect:* `sim_units_per_meter` ~20× larger → confirmation becomes physically possible. Lifts recall
off zero.

### 2. Fix the model, not the sensors — *decisive*

Added `drone_can_confirm` — drones confirm survivors from altitude with a clear top-down view (real
EO/IR aerial SAR) — and a `confirm_requires_los` line-of-sight gate so nothing confirms through a
mountain.

*Effect:* honest ceiling rises to 0.83–1.00 with realistic FOV/altitude, instead of inflating sensor
range. The LOS gate proves the recall is real.

### 3. Fix how we learn — *got us to 0.9*

**Behaviour-clone** the 1.0-recall lawnmower into all 5 HAPPO actors (`train_bc_happo.py`), then **RL
fine-tune** from that warm-start (`--model-dir`). Keep the best checkpoint by recall, not the last.

*Effect:* learned policy 0.60 → 0.97. Fine-tune starts at ~79 reward vs ~42 from scratch; snapshotting
avoids the late-run regression (final fell to 0.80).

### 4. Reward & observation shaping — *supporting*

Team `coverage_obs_grid` so the policy can learn systematic sweeping, a confirmation-dominant reward,
an idle/pending penalty, a ground-exploration reward, and a recurrent **GRU** policy.

*Effect:* necessary plumbing so the clone matches the obs space and RL can express the sweep-and-confirm
behaviour. On their own (from scratch) they reach only ~0.20–0.70.

---

## The two-step path that reached 0.9

**Step 1 · Behaviour cloning → Step 2 · RL fine-tune → Learned 0.97**

**Step 1:** roll out the lawnmower expert under the realistic config and clone its actions into the
HAPPO actors with recurrent BPTT (NLL 0.08 → −1.3). **Step 2:** load those actors and run HAPPO (critic
trains fresh) for 1M steps. Evaluate periodic snapshots and keep the best (`results/happo_realistic_best`,
0.97 recall, robust to 30% comms dropout).

**What to claim — and what not to:** the credible headline is **≥0.9 by realistic aerial confirmation**
(expert 0.83–1.00; learned 0.97). The earlier flat 0.90 from generous sensors + see-through terrain
should only ever be cited as a sensor-sensitivity upper bound, never as the operating point.
