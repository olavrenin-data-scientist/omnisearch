# OmniSearch Notebooks

Five notebooks, each verifying a distinct layer of the system. Each one is
self-contained and writes embedded outputs (plots, tables) back in place when
executed — so you can run once and then read the results in your IDE without
re-executing.

| # | Notebook | Purpose | Runtime |
|---|---|---|---|
| 01 | [01_setup_and_demo.ipynb](01_setup_and_demo.ipynb) | Environment + dependency verification | ~5 s |
| 02 | [02_detection_pipeline.ipynb](02_detection_pipeline.ipynb) | Fire → YOLOv8 person → alert pipeline | ~10 s |
| 03 | [03_sweep_results.ipynb](03_sweep_results.ipynb) | Visualise the comms-dropout sweep results | ~2 s |
| 04 | [04_closed_loop.ipynb](04_closed_loop.ipynb) | Closed loop — sim → synthetic UAV view → detection | ~12 s |
| 05 | [05_baseline_comparison.ipynb](05_baseline_comparison.ipynb) | Per-metric bars + per-metric winners across baselines | ~2 s |

Prerequisites for all five:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install ipykernel jupyter
```

Then select the `.venv` kernel in your IDE, or run `jupyter lab notebooks/`.

---

## 01 — Setup & Demo

**File:** [01_setup_and_demo.ipynb](01_setup_and_demo.ipynb)

Verifies that every component of the stack imports and works on this machine.
Run it first; the cells are organised so that if something fails, you know
exactly which dependency to fix.

What each section does:

1. **Environment check** — Python version, platform, executable path.
2. **(Optional) Install dependencies** — Commented out by default; uncomment
   to `pip install` torch / vmas / torchrl / benchmarl / ultralytics / wandb
   into the active venv.
3. **Per-component verification** — PyTorch, VMAS 1.5.2, TorchRL 0.11.1,
   BenchMARL, YOLOv8 (downloads `yolov8n.pt` weights, asserts class 0 is
   `person`), W&B. Each cell either prints a `✓` line or raises so you see
   what's broken.
4. **Hello-world VMAS demo** — 5 agents in the built-in `navigation` scenario,
   32 parallel envs, 100 random steps. Confirms VMAS itself works before any
   custom scenarios get involved.
5. **BenchMARL training preview** — Reference code (markdown) for what a
   training call looks like. Doesn't execute training.

**Expected outputs:**
- 6/7 components verify clean (SimFire fails on Python 3.11 — non-blocking,
  there's a cellular-automata fallback).
- The demo prints `100 steps completed` with an average return around −2 to −5
  (random policy, expected to be poor).

---

## 02 — Detection Pipeline

**File:** [02_detection_pipeline.ipynb](02_detection_pipeline.ipynb)

End-to-end walkthrough of the **fire-first, then YOLOv8 person** detection
pipeline defined in [`detection/`](../detection/). Five test scenarios verify
both detectors fire in isolation, neither produces false positives on
unrelated content, and the proximity-based "survivor in fire" alert behaves
correctly.

**Modules exercised:**
- [`detection/fire_detector.py`](../detection/fire_detector.py) — HSV
  thresholding (hue ≤ 30, saturation ≥ 180, value ≥ 200) +
  `scipy.ndimage.label` connected components + morphological denoising.
- [`detection/person_detector.py`](../detection/person_detector.py) — wraps
  Ultralytics YOLOv8 with `classes=[0]` so only the COCO `person` class
  fires. Asserts model class 0 is `'person'` so a wrongly-pretrained model
  fails loudly.
- [`detection/pipeline.py`](../detection/pipeline.py) — runs both detectors,
  computes which person boxes overlap (or are within 20 px of) a fire
  region, exposes `alert` flag, and supports `triggered_only=True` to skip
  the heavier YOLOv8 pass on frames with no fire.

**Test scenarios** (all visualised with matplotlib bbox overlays):

| Test | Input | Expected |
|---|---|---|
| 1 | Ultralytics `bus.jpg` (3 people, no fire) | ≥2 person boxes, 0 fire boxes, **alert = False** |
| 2 | Synthetic orange blob, no people | ≥1 fire box, 0 person boxes |
| 3 | `bus.jpg` on the left, orange stripe on the right | Both detectors fire, but **alert = False** because geometry doesn't overlap |
| 4 | A person crop pasted into a fully-orange canvas | **alert = True**, 1 survivor-in-fire |
| 5 | `bus.jpg` with `triggered_only=True` | YOLOv8 is skipped (0 persons), confirming the short-circuit works |

**Why HSV thresholds got tightened:** the initial defaults (hue ≤ 40, value
≥ 150, saturation ≥ 80) flagged bus paint and clothing on `bus.jpg` as fire
— 34 false positives. The current defaults eliminate those without
sacrificing detection on real flame colours.

**Limitation:** the fire detector is rule-based, not learned. The interface
is designed so a trained CNN can be dropped in by implementing the same
`FireDetector.detect(image) → FireResult` signature.

---

## 03 — Sweep Results Viewer

**File:** [03_sweep_results.ipynb](03_sweep_results.ipynb)

Loads the most recent
`results/comms_dropout_sweep_*.json` produced by
[`scripts/comms_dropout_sweep.py`](../scripts/comms_dropout_sweep.py) and
visualises it as a pivot table, a line plot, and a wall-time bar chart.

**Prerequisite:** at least one sweep must have been run.

```bash
python scripts/comms_dropout_sweep.py            # smoke   (~30 s total)
python scripts/comms_dropout_sweep.py --research # real    (~hours)
```

The sweep itself runs 2 algorithms (MAPPO, IPPO) × 4 comms-dropout values
(0.0, 0.2, 0.5, 0.8) and writes a JSON record per cell with the final mean
return, wall time, and BenchMARL output directory.

**What the notebook produces:**

1. **Pivot table** — mean return per (algo × dropout) cell, indexed by
   algorithm.
2. **Line plot** — mean return vs comms_dropout, one line per algorithm. The
   *expected* (post-real-training) shape: IPPO degrades faster than MAPPO as
   dropout grows, because MAPPO's centralised critic still sees global
   state while IPPO doesn't.
3. **Wall-time bar chart** — sanity check that all cells took comparable
   time. Outliers indicate config drift or thermal throttling.

**Critical caveat:** at smoke budget (3 iters × 6 000 frames per cell) all
results are below the noise floor. The shapes you see are noise, not signal.
Use `--research` for credible numbers, and add multi-seed loops in
`comms_dropout_sweep.py` for confidence bands.

The notebook re-discovers the latest sweep JSON every time you execute the
"Load the most recent sweep" cell — so re-running the sweep then re-running
the notebook is the workflow for iteration.

---

## 04 — Closed Loop: Sim → UAV View → Detection

**File:** [04_closed_loop.ipynb](04_closed_loop.ipynb)

Closes the loop between the abstract MARL simulator and the real-image
detection pipeline. VMAS renders coloured circles; YOLOv8 needs real people.
To bridge the gap, [`evaluation/sim_renderer.py`](../evaluation/sim_renderer.py)
synthesises a "UAV top-down view" of the scenario state:

| Sim state | Rendered as |
|---|---|
| Fire grid cell on at `(gy, gx)` | Orange tile in the corresponding image region |
| Survivor landmark at world `(x, y)` | Real person crop (from `bus.jpg`) pasted at projected pixel position |
| Forest (everything else) | Dark green background |

The same `DetectionPipeline` from notebook 02 then runs on this synthetic
frame — so the perception code that would process a real UAV camera feed
is what processes the simulated one.

**What the notebook produces:**

1. **Run summary** — total frames evaluated, alerts fired, and aggregated
   precision / recall / fire-presence accuracy across all rendered frames.
2. **Frame grid** — 8 rendered UAV views with detection overlays:
   - **green** boxes = person detections that match a ground-truth survivor
   - **yellow** boxes = false positive person detections
   - **red** boxes = persons flagged as "survivor in fire" (the alert subset)
   - **orange** boxes = fire regions detected
3. **Alert timeline** — three signals over sim steps:
   - Fire detected (orange) — the fire-HSV stage returned ≥1 box
   - Person↔fire alert (red) — at least one person box overlapped a fire box
   - Ground-truth fire (gray dashed) — the simulator's fire grid had ≥1
     burning cell

The **meaningful event** is the gap between the orange and red lines: fire
becomes visible from step 0, but the alert only fires at step 30 when fire
spreads close enough to threaten a survivor.

**Modules exercised:**
- [`evaluation/sim_renderer.py`](../evaluation/sim_renderer.py) — the
  `UAVRenderer` class. Caches the person sprite (LRU 1) so subsequent renders
  are fast.
- [`evaluation/closed_loop.py`](../evaluation/closed_loop.py) —
  `run_closed_loop(n_steps, render_every, action_fn, ...)`. Drives the env
  forward with a (random by default) policy, renders every K steps, runs the
  pipeline, scores each frame against ground truth (TP / FP / FN by greedy
  bbox matching with 8-pixel slack).
- [`envs/wildfire_search.py`](../envs/wildfire_search.py) — the scenario,
  unchanged from training. The notebook uses 2 parallel envs (VMAS prefers
  batches) and visualises env 0.

**Policy used:** **random**. The closed-loop demo validates the perception
pipeline, not the policy. To plug in a trained MAPPO/IPPO policy:

```python
from benchmarl.experiment import Experiment

experiment = Experiment.reload_from_file('results/.../checkpoint_xxx.pt')
actor = experiment.algorithm.get_policy_for_collection()

def policy_fn(env):
    # Convert env observations → TensorDict → actor → per-agent action list.
    # See benchmarl/experiment/experiment.py for the exact key plumbing.
    ...

run = run_closed_loop(action_fn=policy_fn, n_steps=200)
```

Smoke training doesn't save checkpoints — run with `--research` (or set
`checkpoint_at_end=True` in [`agents/train_helpers.py`](../agents/train_helpers.py))
to produce a loadable model.

**Headline metrics from the random-policy run:**
- `person_recall = 1.000` — every survivor detected in every frame
- `person_precision = 0.816` — YOLOv8 occasionally splits a sprite into two
  overlapping boxes
- `fire_presence_accuracy = 1.000` — fire detector matches ground truth on
  every frame
- 6 alerts fired out of 8 rendered frames

---

## 05 — Baseline Comparison

**File:** [05_baseline_comparison.ipynb](05_baseline_comparison.ipynb)

Loads the most recent `results/baseline_comparison_*.json` produced by
[`scripts/compare_baselines.py`](../scripts/compare_baselines.py) and shows
per-strategy results across the **six mission-level metrics** from the
project plan.

**Prerequisite:**

```bash
python scripts/compare_baselines.py --seeds 3 --steps 200
```

This runs the four hand-coded coordination strategies (`random`,
`lawnmower`, `nearest_candidate`, `highest_confidence`) on the
`WildfireSearchScenario` and records survivor_recall, time_to_verification,
false_positive_trips, hazard_exposure, ugv_travel_cost per run.

**What the notebook produces:**

1. **Mean ± std table** per strategy across all metrics.
2. **Per-metric bar charts** (one chart per metric, strategies on x-axis).
3. **Winners table** — which strategy scores best on each metric.

Trained MAPPO/IPPO/HAPPO checkpoints plug into the same harness via a policy
wrapper once checkpoint loaders are wired into `compare_baselines.py`.

---

## Running everything in order

Headless batch (useful for CI or quick "does it all still work" smoke checks):

```bash
source .venv/bin/activate

jupyter nbconvert --to notebook --execute --inplace notebooks/01_setup_and_demo.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_detection_pipeline.ipynb

# 03 requires sweep output to exist
python scripts/comms_dropout_sweep.py
jupyter nbconvert --to notebook --execute --inplace notebooks/03_sweep_results.ipynb

jupyter nbconvert --to notebook --execute --inplace notebooks/04_closed_loop.ipynb

# 05 requires baseline comparison output to exist
python scripts/compare_baselines.py --seeds 3 --steps 200
jupyter nbconvert --to notebook --execute --inplace notebooks/05_baseline_comparison.ipynb
```

Total wall time: ~60 s on a typical Mac laptop (CPU). The sweep dominates.
