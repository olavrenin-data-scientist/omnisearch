# OmniSearch Notebooks

Three notebooks, each verifying a distinct layer of the system. Each one is
self-contained and writes embedded outputs (plots, tables) back in place when
executed — so you can run once and then read the results in your IDE without
re-executing.

| # | Notebook | Purpose | Runtime |
|---|---|---|---|
| 01 | [01_setup_and_demo.ipynb](01_setup_and_demo.ipynb) | Environment + dependency verification | ~5 s |
| 03 | [03_sweep_results.ipynb](03_sweep_results.ipynb) | Visualise the comms-dropout sweep results | ~2 s |
| 05 | [05_baseline_comparison.ipynb](05_baseline_comparison.ipynb) | Per-metric bars + per-metric winners across baselines | ~2 s |

Prerequisites for all three:

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
   to `pip install` torch / vmas / torchrl / benchmarl / w&b into the
   active venv.
3. **Per-component verification** — PyTorch, VMAS 1.5.2, TorchRL 0.11.1,
   BenchMARL, W&B. Each cell either prints a `✓` line or raises so you see
   what's broken.
4. **Hello-world VMAS demo** — 5 agents in the built-in `navigation`
   scenario, 32 parallel envs, 100 random steps. Confirms VMAS itself works
   before any custom scenarios get involved.
5. **BenchMARL training preview** — Reference code (markdown) for what a
   training call looks like. Doesn't execute training.

---

## 03 — Sweep Results Viewer

**File:** [03_sweep_results.ipynb](03_sweep_results.ipynb)

Loads the most recent `results/comms_dropout_sweep_*.json` produced by
[`scripts/comms_dropout_sweep.py`](../scripts/comms_dropout_sweep.py) and
visualises it as a pivot table, a line plot, and a wall-time bar chart.

**Prerequisite:** at least one sweep must have been run.

```bash
python scripts/comms_dropout_sweep.py --seeds 3            # smoke (~7 min)
python scripts/comms_dropout_sweep.py --seeds 5            # detectable p<0.05
python scripts/comms_dropout_sweep.py --seeds 5 --research # real budget (~hours)
```

The sweep runs **3 algorithms** (MAPPO, IPPO, HAPPO) × **4 dropouts**
(0.0, 0.2, 0.5, 0.8) × **N seeds**. Each cell writes a JSON record per seed
with the final mean reward, wall time, and Mann-Whitney U significance
tests within each algorithm (dropout=0 vs each higher dropout).

**What the notebook produces:**

1. **Pivot table** — mean ± std per (algo × dropout) cell, indexed by
   algorithm.
2. **Line plot** — mean return vs comms_dropout, one line per algorithm.
3. **Wall-time bar chart** — sanity check that all cells took comparable
   time.

**Critical caveats:**
- At smoke budget the absolute values are noisy and the per-algorithm
  curves shouldn't be over-interpreted.
- BenchMARL's `mean_return` and HARL's `mean_episode_reward` are *not*
  directly comparable in absolute terms — only within an algorithm
  across dropouts.

The notebook re-discovers the latest sweep JSON every time you execute the
"Load the most recent sweep" cell — re-running the sweep then re-running
the notebook is the workflow for iteration.

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
2. **Bar chart panel** (2×2) — one bar chart per numeric metric.
3. **Winners table** — for each metric, which strategy was best across the
   seeds.

The bar charts are the headline output for "does any heuristic beat the
others?" — `nearest_candidate` tends to win on `survivor_recall` at this
scenario configuration. Trained policies (MAPPO / IPPO / HAPPO via
checkpoint) will plug into the same harness once the corresponding
loaders are wired into `compare_baselines.py`.

---

## Running everything in order

Headless batch (useful for CI or quick "does it all still work" smoke checks):

```bash
source .venv/bin/activate

jupyter nbconvert --to notebook --execute --inplace notebooks/01_setup_and_demo.ipynb

# 03 requires sweep output to exist
python scripts/comms_dropout_sweep.py --seeds 3
jupyter nbconvert --to notebook --execute --inplace notebooks/03_sweep_results.ipynb

# 05 requires baseline comparison output to exist
python scripts/compare_baselines.py --seeds 3 --steps 200
jupyter nbconvert --to notebook --execute --inplace notebooks/05_baseline_comparison.ipynb
```

Total wall time: dominated by the sweep (~7 min smoke). The notebooks themselves render in a few seconds.
