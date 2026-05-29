# OmniSearch

**Heterogeneous air-ground robotic swarms for wildfire survivor search.**

MIDS Capstone · Summer 2026 · UC Berkeley
Team: Ann-Kathrin Schütz · Oleksii Lavrenin · Jefferson-Stanley Jules

> When wildfires trap people and every second counts, our AI-trained drone and ground robot teams find survivors that human rescuers can't reach in time.

---

## Why

Wildfires routinely strand people in hazardous, hard-to-reach locations. Ground search and rescue teams cannot safely enter active burn zones; aerial drones can survey large areas but cannot verify survivors at close range. **No existing product integrates aerial drones and ground robots into an autonomous coordinated search-and-rescue system.** OmniSearch is the *coordination layer* that connects aerial eyes to ground hands.

Markets at the intersection (wildfire AI, SAR drones, ground robotics) exceed **$10B** with 13–21% CAGR. Existing players occupy one column each (Skydio, Dryad, FireSwarm, Boston Dynamics). The gap — autonomous heterogeneous air-ground coordination — is where OmniSearch sits.

## Research question

> Can heterogeneous air-ground robot teams learn cooperative survivor verification strategies that **outperform hand-coded heuristics**, and **degrade gracefully under communication dropout**?

The MVP answers this in simulation. Three sub-questions:

1. **Heterogeneity** — do drones + ground robots beat drones-only / ground-only on mission-level metrics?
2. **Coordination** — does learned MARL beat hand-coded baselines (nearest-candidate, highest-confidence, lawnmower)?
3. **Robustness** — how does each strategy degrade across 0% → 70% comms dropout?

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| Multi-agent sim | [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) | 2D, CPU-vectorized, fast |
| Fire spread | Cellular automata over a 16×16 grid | SimFire compatible (needs Python 3.9–3.10) |
| MARL training | [BenchMARL](https://github.com/facebookresearch/BenchMARL) (MAPPO, IPPO) | HAPPO not in BenchMARL — see *HAPPO note* below |
| Detection (stretch) | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) with `classes=[0]` | Person = COCO class 0 |
| Experiment tracking | [Weights & Biases](https://wandb.ai) | Optional; pass `loggers=["wandb"]` |
| Deliverable (planned) | React + Three.js viewer | Strategy comparison & replay |

**HAPPO is wired.** BenchMARL 1.x ships MAPPO/IPPO/MADDPG/MASAC/QMIX/VDN/IQL — *not* HAPPO. True HAPPO (Kuba 2022) is in the [HARL](https://github.com/PKU-MARL/HARL) library. We bridge HARL to our VMAS scenario via [agents/harl_env.py](agents/harl_env.py) (a HARL-shape adapter around `WildfireSearchScenario`) and run it via [scripts/train_happo_smoke.py](scripts/train_happo_smoke.py). The script monkey-patches HARL's env registry at runtime to recognise the `wildfire` env without modifying HARL source — clone HARL once next to this repo and `pip install -e ../HARL` is the whole install. MAPPO and IPPO via BenchMARL stay available for comparison.

---

## Setup

**Requirements:** Python 3.10 or 3.11. macOS / Linux. No GPU needed for the smoke runs.

```bash
# 1. Clone and enter
git clone <repo-url> omnisearch && cd omnisearch

# 2. Create + activate a virtualenv (3.11 recommended; 3.10 needed for SimFire)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install everything
pip install --upgrade pip
pip install torch torchvision
pip install vmas torchrl benchmarl ultralytics "pettingzoo[mpe]"
pip install wandb tensorboard tqdm hydra-core omegaconf matplotlib seaborn pandas numpy pyyaml scipy networkx
pip install pytest black ruff ipykernel jupyter nbformat nbconvert "moviepy<2.0.0"

# 4. (Optional) register the kernel for Jupyter / IDE notebooks
python -m ipykernel install --user --name omnisearch --display-name "Python (omnisearch)"

# 5. Verify everything works
jupyter notebook notebooks/01_setup_and_demo.ipynb
# Or headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_setup_and_demo.ipynb
```

**SimFire.** SimFire's PyGame dep requires Python 3.9–3.10. On 3.11 it won't install — fire spread falls back to the in-repo cellular-automata model, which is good enough for the MARL training story. Run `python3.10 -m venv .venv` instead if you need SimFire.

**W&B login** (optional, only for training with logging):
```bash
wandb login
```

---

## How to run

Each entry point is a script in `scripts/` or a notebook in `notebooks/`. All outputs land in `results/` (gitignored).

### 1. Smoke training — verify the MARL pipeline runs (~3–6 s each)

```bash
python scripts/train_mappo_smoke.py     # MAPPO — centralized critic, per-group actors (BenchMARL)
python scripts/train_ippo_smoke.py      # IPPO  — fully decentralized              (BenchMARL)
python scripts/train_happo_smoke.py     # HAPPO — sequential update + monotonic improvement (HARL)
```

Each trains a tiny budget. The successful exit is the milestone — the policy is far from converged.

For HAPPO, install HARL first (one-time):

```bash
cd .. && git clone https://github.com/PKU-MARL/HARL.git
cd HARL && pip install -e .
cd ../omnisearch
```

For a credible training run, edit the config in [agents/train_helpers.py](agents/train_helpers.py) (`research_config()`) for MAPPO/IPPO, or the `build_args()` function in `scripts/train_happo_smoke.py` for HAPPO.

### 2. Comms-dropout sweep — the headline ablation

```bash
python scripts/comms_dropout_sweep.py             # smoke   ~30 s   (2 algos × 4 dropouts × 6k frames)
python scripts/comms_dropout_sweep.py --research  # real    ~hours  (~400k frames per cell)
```

Writes per-cell `mean_return`, wall time, and BenchMARL output directory to `results/comms_dropout_sweep_*.json`. Visualise with notebook 03.

### 3. Baseline comparison — does MARL actually beat heuristics?

```bash
python scripts/compare_baselines.py                       # 3 seeds, 200 steps, all 4 baselines
python scripts/compare_baselines.py --seeds 5 --steps 250
```

Runs each strategy across multiple seeds, reports mean ± std on all six mission-level metrics (survivor recall, time-to-verification, false-positive trips, hazard exposure, UGV travel cost), writes `results/baseline_comparison_*.json`. **Trained MAPPO/IPPO policies plug in via the same harness** once a checkpoint exists — see [scripts/compare_baselines.py](scripts/compare_baselines.py) for the TODO.

### 4. Notebooks — exploratory + visualization

```bash
jupyter notebook notebooks/   # opens browser file picker
```

See [notebooks/README.md](notebooks/README.md) for the cell-by-cell walkthrough of each notebook.

| # | Notebook | Purpose |
|---|---|---|
| 01 | Setup & Demo | Environment + dependency verification |
| 02 | Detection Pipeline | Fire → YOLOv8 person → alert |
| 03 | Sweep Results | Visualise comms-dropout JSON |
| 04 | Closed Loop | Sim → synthetic UAV view → detection |
| 05 | Baseline Comparison | Bar charts + winners table per metric |

### 5. Web viewer — React + Three.js replay

```bash
# 1. Export one trajectory JSON per baseline (~1.5 s total)
python scripts/export_trajectories.py

# 2. Serve the web/ folder (browsers need HTTP for fetch)
python -m http.server -d web 8080

# 3. Open http://localhost:8080
```

Top-down replay of each strategy with playback controls + mission-metrics panel. No `npm install`, no build — single HTML file, dependencies loaded from `esm.sh`. See [web/README.md](web/README.md).

---

## Project layout

```
omnisearch/
├── envs/wildfire_search.py        # WildfireSearchScenario (VMAS)
│
├── agents/
│   ├── wildfire_task.py           # BenchMARL Task wiring (MAPPO / IPPO)
│   ├── train_helpers.py           # smoke_config() / research_config()
│   ├── baselines.py               # Hand-coded coordination strategies
│   └── harl_env.py                # HARL-shape adapter for HAPPO training
│
├── detection/                     # Fire → YOLOv8 person pipeline (real images)
│   ├── fire_detector.py           # HSV thresholding + connected components
│   ├── person_detector.py         # YOLOv8 wrapper, classes=[0]
│   └── pipeline.py                # Two-stage: fire-first, then person
│
├── evaluation/
│   ├── mission_metrics.py         # The 6 metrics from the project plan + DRR
│   ├── closed_loop.py             # Sim rollout + per-frame detection + GT scoring
│   └── sim_renderer.py            # Synthetic UAV top-down view of the scenario
│
├── scripts/
│   ├── train_mappo_smoke.py       # MAPPO smoke run (BenchMARL)
│   ├── train_ippo_smoke.py        # IPPO  smoke run (BenchMARL)
│   ├── train_happo_smoke.py       # HAPPO smoke run (HARL)
│   ├── comms_dropout_sweep.py     # MAPPO × IPPO × 4 dropouts ablation
│   ├── compare_baselines.py       # Multi-seed baseline comparison
│   └── export_trajectories.py     # → web/trajectories/*.json for the viewer
│
├── notebooks/                     # See notebooks/README.md
├── results/                       # Training artifacts (gitignored)
└── web/                           # React + Three.js strategy viewer
    ├── index.html
    └── README.md
```

---

## Mission-level metrics (the MVP success criteria)

Every coordination strategy — hand-coded baseline or trained policy — is scored on the same six metrics. Direction column shows which way improvement runs.

| Metric | Direction | Definition |
|---|---|---|
| `survivor_recall` | higher | fraction of survivors confirmed by a ground robot |
| `time_to_verification` | lower | avg steps between drone scout and ground confirmation |
| `false_positive_trips` | lower | ground-robot trips to a location with no survivor |
| `hazard_exposure` | lower | step-count ground robots spend on burning cells |
| `ugv_travel_cost` | lower | total ground-robot path length |
| `drr` (across dropouts) | higher | degradation resilience ratio under comms loss |

Definitions and implementation in [evaluation/mission_metrics.py](evaluation/mission_metrics.py).

Baselines defined in [agents/baselines.py](agents/baselines.py):
- **`random`** — both agent types take random actions (control)
- **`lawnmower`** — drones sweep a serpentine path; ground robots head to nearest scouted survivor
- **`nearest_candidate`** — drones random; ground robots go to nearest scouted-not-confirmed survivor
- **`highest_confidence`** — drones lawnmower; ground robots prioritize freshest scout

### Smoke comparison (3 seeds × 200 steps)

```
strategy                recall     ttv   haz  ugv_dist
------------------------------------------------------
random                    0.00     nan     0      1.11
lawnmower                 0.27   154.3     0      1.76
nearest_candidate         0.33   169.0     0      1.77
highest_confidence        0.00     nan     0      3.33
```

`nearest_candidate` is the strongest baseline at 33% recall. **HAPPO/MAPPO must beat 33%** to justify the training complexity.

---

## Status

| Component | Status |
|---|---|
| Environment + dependencies | ✓ |
| `WildfireSearchScenario` (heterogeneous, CA fire, comms_dropout knob) | ✓ |
| Detection pipeline (fire → YOLOv8 person → alert) | ✓ |
| BenchMARL wired (MAPPO + IPPO) | ✓ |
| **HAPPO wired (HARL adapter + smoke train passes)** | ✓ |
| Comms-dropout sweep + results notebook | ✓ |
| Closed-loop sim → UAV view → detection | ✓ |
| Mission-level metrics (6 + DRR) | ✓ |
| Hand-coded baselines + comparison harness | ✓ |
| Web deliverable (React + Three.js strategy viewer) | ✓ |
| Trained-policy rollout (load checkpoint into `compare_baselines.py`) | ✗ — needs `--research` budget run first |
| HAPPO in the comms-dropout sweep alongside MAPPO/IPPO | ✗ — sweep currently uses BenchMARL only |
| Multi-seed sweep + confidence bands | ✗ |
| Statistical significance tests (Mann-Whitney U) | ✗ |
| Probabilistic sensor model + candidate belief map | ✗ — current baselines use ground-truth scout/found |

---

## Project plan

The full project plan with problem statement, market analysis, target customer research, and detailed roadmap is the source of truth for scope and direction. Key sections relevant to this codebase:

- §17.3 — Probabilistic sensor model (current implementation uses ground-truth shortcut; full model is a TODO)
- §17.5 — Coordination strategies (all implemented in [agents/baselines.py](agents/baselines.py))
- §17.6 — Evaluation metrics (implemented in [evaluation/mission_metrics.py](evaluation/mission_metrics.py))
- §17.7 — Real-world data sources for scenario realism (NIFC, LANDFIRE, etc. — not yet integrated)

---

## Contributing

The codebase is organised by *function*, not by *team member*. Anyone can edit any file. Conventions:

- **Tests / smoke runs first.** New code should land with a smoke test demonstrating it runs.
- **One commit per logical change.** Bundle the test, the implementation, and the README update.
- **Don't commit `results/`** — it's gitignored.

---

## License

See [LICENSE](LICENSE).
