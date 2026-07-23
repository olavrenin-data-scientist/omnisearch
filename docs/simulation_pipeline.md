# OmniSearch Simulation Pipeline

This guide explains the practical pipeline for running OmniSearch simulations:
building terrain caches, exporting trajectories, viewing episodes in the web
viewer, and generating diagnostic reports. It is meant as an operational
companion to the conceptual documentation:

- [Simulation overview](simulation_overview.md)
- [Perception model](perception_model.md)
- [Reinforcement learning and HAPPO](reinforcement_learning_happo.md)
- [Observation and reward system](observation_reward_system.md)
- [Baseline approaches](baseline_approaches.md)
- [Communication dropout](communication_dropout.md)

All commands below assume the repository root:

```bash
cd /Users/aschuetz/Software/capstone/omnisearch
source .venv/bin/activate
```

## 1. Pipeline Overview

```text
[1] Terrain cache        data/terrain_cache/<area>_<grid>.npz
        |
[2] Simulation export    web/trajectories/<approach>.json
        |
[3] Web viewer           http://localhost:8080
        |
[4] Diagnostics          outputs/.../*.json and outputs/.../*.png
        |
[5] EDA reports          results/eda/*.html
```

The simulator uses one physical environment for all approaches. Hand-coded
baselines and trained HAPPO actors are evaluated through the same scenario,
perception model, terrain constraints, communication model, and mission metrics.

## 2. Terrain Caches

The wildfire scenario loads terrain from a pre-built `.npz` cache. A cache
contains the rasterized simulation layers used by the environment:

- elevation and slope,
- land cover and traversability,
- roads, water, buildings, and obstacle masks,
- fuel, moisture, fire-spread inputs, and mobility costs.

Each `.npz` file is accompanied by a `.metadata.json` file with provenance and
configuration details.

Common caches currently in the repository include:

| File | Area | Grid |
|---|---|---:|
| `malibu_creek_500m_128.npz` | Malibu Creek compact diagnostic area | 128 |
| `malibu_creek_1sqkm_256.npz` | Malibu Creek 1 km x 1 km area | 256 |
| `topanga_state_park_1sqkm_256.npz` | Topanga State Park | 256 |
| `aubern_sra_1sqkm_256.npz` | Auburn SRA | 256 |
| `san_marcos_foothills_1sqkm_256.npz` | San Marcos Foothills | 256 |
| `pinnacles_1sqkm_256.npz` | Pinnacles | 256 |
| `big_sur_128.npz` | Big Sur | 128 |

Build a cache from a place name:

```bash
python scripts/build_real_terrain_cache.py \
  --place "Malibu Creek State Park, California" \
  --grid-size 256
```

Or from an explicit longitude/latitude bounding box:

```bash
python scripts/build_real_terrain_cache.py \
  --bbox -118.78 34.08 -118.74 34.12 \
  --grid-size 256
```

Optional LANDFIRE fuel data can be enabled with a free LANDFIRE account email:

```bash
python scripts/build_real_terrain_cache.py \
  --place "Malibu Creek State Park, California" \
  --grid-size 256 \
  --landfire-email your@email.com
```

Important cache-builder options:

| Flag | Description |
|---|---|
| `--place` | Place name to geocode |
| `--bbox W S E N` | Explicit longitude/latitude bounding box |
| `--grid-size` | Raster resolution used by fire, terrain, confidence, and planner grids |
| `--cache-dir` | Output directory, usually `data/terrain_cache/` |
| `--dem-resolution-m` | USGS DEM target resolution |
| `--road-width-m` | Width used when rasterizing roads/trails |
| `--building-height-m` | Default building obstacle height |
| `--landfire-email` | Enables LANDFIRE fuel products when available |

## 3. Export Trajectories

`scripts/export_trajectories.py` runs one or more approaches and records a
step-by-step JSON trajectory for the web viewer.

```bash
python scripts/export_trajectories.py
```

By default, `--approach all` exports the registered hand-coded baselines and
HAPPO if a checkpoint is available.

Current approaches:

| Approach | Description |
|---|---|
| `random_action` | independent random UAV motion; UGVs route after local target knowledge |
| `random_walk` | persistent random headings with boundary and terrain safety |
| `lawnmower` | land-aware serpentine UAV coverage with route-aware UGV confirmation |
| `highest_confidence` | lawnmower UAV coverage; UGVs prioritize strongest retained detection confidence |
| `ant_colony` | decentralized recency-map coverage; UGVs use local survivor memory |
| `matched_heuristic` | lawnmower UAVs with scenario-side UGV assignment and planner hints |
| `happo` | trained HAPPO policy loaded from a checkpoint |

Useful export options:

| Flag | Default | Description |
|---|---:|---|
| `--approach` | `all` | Export all, `happo`, or one named baseline |
| `--steps` | `500` | Episode horizon unless a checkpoint environment overrides it |
| `--frame-stride` | `1` | Record every Nth frame while still simulating every step |
| `--seed` | `0` | Random seed |
| `--grid-size` | `128` | Fire/terrain grid resolution for manual exports |
| `--terrain-cache-path` | auto | Explicit terrain cache |
| `--ignore-happo-env` | off | Use CLI settings instead of restoring checkpoint scenario settings |
| `--baseline-ugv-controller` | `native` | Keep baseline UGV logic or use `matched_heuristic` UGV control |

Communication export options:

| Flag | Default | Description |
|---|---:|---|
| `--comms-dropout` | `0.0` | Agent-level dropout fraction |
| `--comms-dropout-mode` | `bursty` | `iid` or `bursty` |
| `--comms-map-mode` | `per_agent` | Communication-gated per-agent map memories |
| `--comms-dropout-min-steps` | `5` | Minimum burst length |
| `--comms-dropout-max-steps` | `15` | Maximum burst length |

Examples:

```bash
# Export all baselines and HAPPO if available
python scripts/export_trajectories.py

# Export one baseline
python scripts/export_trajectories.py --approach lawnmower

# Export the learned policy from its saved environment
python scripts/export_trajectories.py --approach happo

# Show communication dropout in the viewer
python scripts/export_trajectories.py \
  --approach ant_colony \
  --comms-dropout 0.3 \
  --comms-dropout-mode bursty

# Run a manual terrain experiment instead of restoring checkpoint settings
python scripts/export_trajectories.py \
  --ignore-happo-env \
  --terrain-cache-path data/terrain_cache/topanga_state_park_1sqkm_256.npz \
  --grid-size 256 \
  --steps 900
```

## 4. Web Viewer

Serve the viewer locally:

```bash
python -m http.server -d web 8080
```

Then open:

```text
http://localhost:8080
```

The viewer auto-discovers JSON files in `web/trajectories/`. It displays:

- terrain, elevation, roads, obstacles, and land-cover layers,
- fire, smoke, and burned-area evolution,
- UAV footprints and detection-probability links,
- UGV paths and target-confirmation behavior,
- communication status per agent,
- mission metrics and playback controls.

The trajectory JSONs can be large. Use `--frame-stride` for long episodes when
the viewer becomes slow, while keeping the simulator step rate unchanged.

## 5. HAPPO Training Runs

Training is handled separately from trajectory export. The current RL
documentation focuses on HAPPO:

```bash
python scripts/train_happo_smoke.py ...
```

For reproducibility, trained runs save their scenario configuration in the run
directory. Export and diagnostic scripts can restore these settings so the
policy is evaluated under the same terrain, number of agents, episode length,
reward settings, observation settings, and communication configuration that were
used during training.

Reference run configurations are typically stored under:

```text
results/harl_runs/wildfire/wildfire_search/happo/<run-name>/
```

and include:

```text
omnisearch_training_config.json
models/
```

Use `--ignore-happo-env` only when the goal is a manual scenario experiment
rather than a faithful checkpoint export.

## 6. Joint Diagnostics

For paper-style evaluation, the diagnostic scripts produce aggregate JSON files
and plots over many seeds.

Typical scripts:

| Script | Purpose |
|---|---|
| `scripts/diagnose_joint_happo.py` | Evaluate trained HAPPO in the joint UAV/UGV setting |
| `scripts/diagnose_joint_baseline_strategies.py` | Evaluate one hand-coded baseline at a time |
| `scripts/diagnose_uav_happo.py` | UAV-focused confidence and coverage diagnostics |
| `scripts/diagnose_ugv_happo.py` | UGV routing, assignment, and confirmation diagnostics |
| `scripts/diagnose_ugv_baseline_strategies.py` | UGV heuristic A* baseline comparison |

The main evaluation outputs are usually written under `outputs/`, for example:

```text
outputs/baseline_comparison/
outputs/communication_dropout/
outputs/terrain_generalization/
outputs/survivors_load/
```

These JSON and PNG files are the source for result tables, ablation studies,
baseline comparisons, and slide figures.

## 7. EDA Reports

The EDA scripts generate standalone HTML reports with Plotly. They are useful
for inspecting terrain, perception, and trajectories outside the main viewer.

Terrain layers:

```bash
python scripts/terrain_eda.py \
  --terrain-cache data/terrain_cache/malibu_creek_1sqkm_256.npz \
  --out results/eda/malibu_terrain_eda.html
```

Terrain plus a trajectory overlay:

```bash
python scripts/terrain_eda.py \
  --terrain-cache data/terrain_cache/malibu_creek_1sqkm_256.npz \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/malibu_lawnmower_terrain.html
```

Perception diagnostics:

```bash
python scripts/perception_eda.py \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/perception_eda.html
```

3D terrain surface:

```bash
python scripts/terrain_3d_plot.py \
  --terrain-cache data/terrain_cache/malibu_creek_1sqkm_256.npz \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/malibu_3d.html \
  --vertical-exaggeration 3.0
```

## 8. Run a Complete Local Demo

```bash
mkdir -p results/eda

# 1. Export trajectories
python scripts/export_trajectories.py \
  --ignore-happo-env \
  --terrain-cache-path data/terrain_cache/malibu_creek_1sqkm_256.npz \
  --grid-size 256 \
  --steps 500

# 2. Build terrain diagnostics
python scripts/terrain_eda.py \
  --terrain-cache data/terrain_cache/malibu_creek_1sqkm_256.npz \
  --out results/eda/malibu_terrain_eda.html

# 3. Build perception diagnostics from one trajectory
python scripts/perception_eda.py \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/perception_eda.html

# 4. Serve the viewer
python -m http.server -d web 8080
```

## 9. Output Files

| Path | Generated by | Description |
|---|---|---|
| `data/terrain_cache/*.npz` | `build_real_terrain_cache.py` | Terrain, fire, mobility, and planner grids |
| `data/terrain_cache/*.metadata.json` | `build_real_terrain_cache.py` | Terrain-cache provenance |
| `web/trajectories/*.json` | `export_trajectories.py` | Per-frame viewer trajectories |
| `outputs/**/*.json` | diagnostic scripts | Multi-seed metrics and summaries |
| `outputs/**/*.png` | diagnostic scripts | Result plots and diagnostics |
| `results/eda/*_terrain_eda.html` | `terrain_eda.py` | Terrain layer reports |
| `results/eda/*_perception_eda.html` | `perception_eda.py` | UAV perception reports |
| `results/eda/*_3d.html` | `terrain_3d_plot.py` | 3D terrain visualizations |

## 10. Common Pitfalls

- If a HAPPO export looks different from a diagnostic run, check whether the
  exporter restored checkpoint settings or used `--ignore-happo-env`.
- If terrain and grid settings disagree, prefer the grid size stored in the
  terrain cache and checkpoint configuration.
- If communication experiments appear unexpectedly strong or weak, check
  `--comms-dropout-mode`; IID and bursty outages have different temporal
  structure even at the same dropout fraction.
- If the viewer is slow, export fewer frames with `--frame-stride`.
- If a baseline comparison needs fair UGV planner support, compare native
  baseline UGV behavior against `--baseline-ugv-controller matched_heuristic`.
