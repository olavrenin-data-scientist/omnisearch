# OmniSearch 🔥🤖

**MAPPO/HAPPO-trained drone + ground robot swarms for wildfire survivor search**

MIDS Capstone · Summer 2026 · UC Berkeley

## Research Question

Can heterogeneous air-ground robot teams (drones + ground robots) learn cooperative
survivor search strategies, and how does performance degrade under
communication dropout?

> **Note on HAPPO:** BenchMARL 1.x ships MAPPO, IPPO, MADDPG, MASAC, QMIX, VDN, IQL —
> **not HAPPO**. Real HAPPO (Kuba et al., 2022) requires the separate
> [HARL](https://github.com/PKU-MARL/HARL) library and a non-trivial integration.
> The heterogeneity itself, however, is already captured by our scenario's
> `group_map = {drone: [...], ground: [...]}` — both MAPPO and IPPO train
> separate per-group policies on top of that. The baseline matrix is:
>
> - **MAPPO** — centralized critic, decentralized per-group actors
> - **IPPO**  — fully decentralized (per-group actor *and* critic)
>
> True HAPPO is a documented stretch goal in [scripts/comms_dropout_sweep.py](scripts/comms_dropout_sweep.py).

## Stack

| Component | Tool |
|-----------|------|
| Multi-agent sim | [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) |
| Fire spread | [SimFire](https://github.com/mitrefireline/simfire) / Cellular automata |
| MARL algorithm | HAPPO via [BenchMARL](https://github.com/facebookresearch/BenchMARL) |
| Person detection | [YOLOv8-nano](https://github.com/ultralytics/ultralytics) (COCO class 0) |
| Experiment tracking | [W&B](https://wandb.ai) |
| Web deliverable | React + Three.js |

## Setup

Create a Python 3.10 or 3.11 environment, then run the setup notebook:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install ipykernel jupyter
python -m ipykernel install --user --name omnisearch --display-name "Python (omnisearch)"

jupyter notebook notebooks/01_setup_and_demo.ipynb
```

The notebook walks through dependency install, per-component verification, and a hello-world VMAS demo (5 agents, 100 random steps, 32 parallel envs).

> **Note:** SimFire requires Python <3.10. On 3.11 the project uses a cellular-automata fallback for fire spread.

## Quick Start

```bash
# 1) MAPPO smoke training on the wildfire scenario (~3s, 6k frames)
python scripts/train_mappo_smoke.py

# 2) IPPO smoke training (~3s, 6k frames)
python scripts/train_ippo_smoke.py

# 3) Comms-dropout ablation sweep (the headline experiment)
python scripts/comms_dropout_sweep.py            # smoke: 2 algos × 4 dropouts × 6k frames
python scripts/comms_dropout_sweep.py --research # real: ~400k frames per cell
```

Per-cell wall time on CPU: ~3s smoke, ~30 min research. Outputs land in `results/` (gitignored).

## Layout

| Path | Purpose |
|------|---------|
| [notebooks/01_setup_and_demo.ipynb](notebooks/01_setup_and_demo.ipynb) | Environment setup, verification, hello-world VMAS demo |
| [notebooks/02_detection_pipeline.ipynb](notebooks/02_detection_pipeline.ipynb) | Fire → YOLOv8 person detection pipeline, end-to-end |
| [envs/wildfire_search.py](envs/wildfire_search.py) | `WildfireSearchScenario` — heterogeneous drones + ground robots, CA fire spread, survivor landmarks |
| [detection/fire_detector.py](detection/fire_detector.py) | HSV-based fire detector (rule-based; swap for CNN later) |
| [detection/person_detector.py](detection/person_detector.py) | YOLOv8 wrapper with `classes=[0]` (person only) |
| [detection/pipeline.py](detection/pipeline.py) | Two-stage fire→person pipeline with proximity-based alerts |
| `agents/` `fire/` `evaluation/` | Project modules (training agents, fire spread, metrics) |
| `configs/{env,training}/` | Hydra/YAML configs |
| `scripts/` | Train/eval entry points |
| `web/` | React + Three.js deliverable |

## Detection pipeline at a glance

```python
from detection import DetectionPipeline

pipe = DetectionPipeline()
result = pipe.run("uav_frame.jpg")

result.fire.detections        # HSV-detected fire regions
result.persons.detections     # YOLOv8 person detections (class 0 only)
result.survivors_in_fire      # persons whose bbox overlaps a fire region
result.alert                  # True iff survivors_in_fire is non-empty
```

For high-FPS UAV feeds, use `pipe.run(frame, triggered_only=True)` — skips the heavy YOLOv8 pass on frames with no fire.
