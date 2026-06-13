# OmniSearch Simulation Pipeline

End-to-end guide for building terrain, running the wildfire simulation, exporting trajectories for the web viewer, and generating EDA reports.

---

## Overview

```
[1] Build terrain cache        → data/terrain_cache/<name>_<grid>.npz
        ↓
[2] Export trajectories        → web/trajectories/<strategy>.json
        ↓
[3] View in browser            → http://localhost:8080
        ↓
[4] EDA reports (optional)     → results/eda/*.html
```

All scripts run from the **repo root** with the project venv active:

```bash
cd /Users/aschuetz/Software/capstone/omnisearch
source .venv/bin/activate
```

---

## Step 1 — Build a Terrain Cache

The simulation loads real terrain from a pre-built `.npz` cache. The cache combines:
- **USGS 3DEP** — 10m digital elevation model
- **OpenStreetMap** — roads, buildings, water bodies, land cover
- **LANDFIRE** — fuel moisture, canopy cover, fire behaviour fuel models (optional, requires email)

### Pre-built caches (already in the repo)

| File | Area | Grid |
|---|---|---|
| `malibu_creek_state_park_california_128.npz` | Malibu Creek SP, CA | 128×128 |
| `big_sur_128.npz` | Big Sur, CA | 128×128 |

### Build a new cache

**By place name (geocoded automatically):**
```bash
python scripts/build_real_terrain_cache.py \
  --place "Malibu Creek State Park, California" \
  --grid-size 128
```

**By explicit bounding box (lon/lat):**
```bash
python scripts/build_real_terrain_cache.py \
  --bbox -118.78 34.08 -118.74 34.12 \
  --grid-size 128
```

**With LANDFIRE fuel data** (requires a free LANDFIRE account email):
```bash
python scripts/build_real_terrain_cache.py \
  --place "Malibu Creek State Park, California" \
  --grid-size 128 \
  --landfire-email your@email.com
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--place` | Malibu Creek SP, CA | Place name to geocode |
| `--bbox W S E N` | — | Explicit lon/lat bounding box |
| `--grid-size` | 128 | Simulation grid resolution |
| `--cache-dir` | `data/terrain_cache/` | Output directory |
| `--dem-resolution-m` | 10 | USGS DEM resolution in meters |
| `--road-width-m` | 8.0 | Road cell width |
| `--building-height-m` | 7.0 | Default obstacle height |
| `--landfire-email` | — | Email for LANDFIRE API (enables fuel data) |

The output is saved as `data/terrain_cache/<slugified-place>_<grid>.npz` alongside a `.metadata.json` with full provenance. Building takes 1–5 minutes depending on the area and whether LANDFIRE is enabled.

---

## Step 2 — Export Trajectories

Runs the hand-coded baseline strategies through the simulation and records each step as a JSON frame for the web viewer.

```bash
python scripts/export_trajectories.py
```

This produces one JSON file per strategy in `web/trajectories/`:

| Strategy | Description |
|---|---|
| `random_action.json` | All agents take independent random actions |
| `lawnmower.json` | Drones sweep in terrain-aware lanes; ground robots head to nearest scouted survivor |
| `nearest_candidate.json` | Drones random walk; ground robots go to nearest scouted survivor |
| `highest_confidence.json` | Drones sweep; ground robots prioritise most recently scouted survivor |
| `ant_colony.json` | Drones avoid locally fresh coverage pheromone; UGVs follow locally known survivor events |
| `happo_trained.json` | Trained HAPPO policy (exported only when a checkpoint exists) |

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--steps` | 500 | Episode length |
| `--grid-size` | 128 | Fire/terrain grid resolution |
| `--comms-dropout` | 0.0 | Comms dropout rate (0 = perfect radio, 0.8 = mostly broken) |
| `--seed` | 0 | Random seed |
| `--terrain-cache-path` | auto | Override terrain cache path |
| `--ignore-happo-env` | off | Keep CLI terrain/settings instead of restoring the latest HAPPO training environment |

**Examples:**

```bash
# Default run (Malibu, 128 grid, 500 steps)
python scripts/export_trajectories.py

# Show communication dropout effects in the viewer
python scripts/export_trajectories.py --comms-dropout 0.3

# Different terrain
python scripts/export_trajectories.py \
  --terrain-cache-path data/terrain_cache/big_sur_128.npz

# Longer episode, specific seed
python scripts/export_trajectories.py --steps 800 --seed 42
```

---

## Step 3 — Web Viewer

Serve the viewer and open it in your browser:

```bash
python -m http.server -d web 8080
```

Then open **http://localhost:8080**.

The viewer auto-discovers all `*.json` files in `web/trajectories/`. Use the **Strategy** dropdown to switch between runs. Features:

- Real terrain overlay (land cover, elevation, obstacles)
- Animated fire spread with smoke and burned area accumulation
- Drone camera footprints and per-survivor detection probability links
- Ground robot trails and heading
- Comms status dot per agent (green = up, pulsing red = dropped)
- Mission metrics panel (survivors found/scouted, fire stats)
- Playback controls with variable speed

---

## Step 4 — EDA Reports

Three standalone HTML reports built with Plotly.js — open in any browser, no server needed.

### Terrain EDA

Maps of land cover, elevation, slope, moisture, fuel density, traversability, and obstacles.

```bash
python scripts/terrain_eda.py \
  --terrain-cache data/terrain_cache/malibu_creek_state_park_california_128.npz \
  --out results/eda/malibu_terrain_eda.html
```

Overlay a trajectory to compare where agents actually went vs the terrain:

```bash
python scripts/terrain_eda.py \
  --terrain-cache data/terrain_cache/malibu_creek_state_park_california_128.npz \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/malibu_lawnmower_terrain.html
```

### Perception EDA

Per-drone detection probability, camera footprint coverage, smoke/fire visibility factors, and altitude profiles across an episode.

```bash
python scripts/perception_eda.py \
  --trajectory web/trajectories/nearest_candidate.json \
  --out results/eda/perception_eda.html
```

### 3D Terrain Plot

Interactive 3D elevation surface with optional drone flight paths overlaid.

```bash
# Terrain only
python scripts/terrain_3d_plot.py \
  --terrain-cache data/terrain_cache/malibu_creek_state_park_california_128.npz \
  --out results/eda/malibu_3d.html

# With drone flight paths from a trajectory
python scripts/terrain_3d_plot.py \
  --terrain-cache data/terrain_cache/malibu_creek_state_park_california_128.npz \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/malibu_lawnmower_3d.html \
  --vertical-exaggeration 3.0
```

---

## Run Everything at Once

```bash
mkdir -p results/eda

# 1. Export all baseline trajectories
python scripts/export_trajectories.py

# 2. Generate all EDA reports
python scripts/terrain_eda.py \
  --terrain-cache data/terrain_cache/malibu_creek_state_park_california_128.npz \
  --out results/eda/malibu_terrain_eda.html

python scripts/perception_eda.py \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/perception_eda.html

python scripts/terrain_3d_plot.py \
  --terrain-cache data/terrain_cache/malibu_creek_state_park_california_128.npz \
  --trajectory web/trajectories/lawnmower.json \
  --out results/eda/malibu_3d.html

# 3. Open EDA results
open results/eda/

# 4. Serve the web viewer
python -m http.server -d web 8080
```

---

## Output Files

| Path | Generated by | Description |
|---|---|---|
| `data/terrain_cache/*.npz` | `build_real_terrain_cache.py` | Simulation terrain grid |
| `data/terrain_cache/*.metadata.json` | `build_real_terrain_cache.py` | Provenance and source metadata |
| `web/trajectories/*.json` | `export_trajectories.py` | Per-step episode recordings |
| `results/eda/*_terrain_eda.html` | `terrain_eda.py` | Terrain layer maps |
| `results/eda/*_perception_eda.html` | `perception_eda.py` | Drone perception analysis |
| `results/eda/*_3d.html` | `terrain_3d_plot.py` | 3D elevation viewer |
