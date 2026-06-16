# OmniSearch · Strategy Viewer (React + Three.js)

Minimal browser app to replay rolled-out trajectories from each
coordination strategy and inspect their final mission-level metrics.

## How to run (3D viewer, from a clean checkout)

The viewer itself is buildless (no `npm install`). The work is on the
Python side: the 3D scene needs trajectory JSONs that **contain real
terrain**, and producing those has real prerequisites. Run these from
the **repo root**, in order. Every Python command assumes the venv is
active — see step 0.

```bash
# 0. Activate the venv FIRST (every new terminal). Without this, `python`
#    may not exist on macOS — see Troubleshooting.
source .venv/bin/activate          # prompt should now show (.venv)

# 1. Install the geospatial deps the terrain builder needs (one-time).
pip install -r requirements-geo.txt   # osmnx, rasterio, geopandas, shapely, pyproj

# 2. Build the real-terrain cache (one-time per place; needs internet —
#    pulls a USGS 3DEP DEM + OpenStreetMap roads/buildings).
python scripts/build_real_terrain_cache.py \
  --place "Malibu Creek State Park, California" \
  --grid-size 128
#    ⚠ This writes a HASH-named file (e.g. 41524c3b7cb9_128.npz), but the
#    exporter's default path expects the SLUG name. Copy it so the default
#    path resolves (replace the hash with the one printed above):
cp data/terrain_cache/<hash>_128.npz \
   data/terrain_cache/malibu_creek_state_park_california_128.npz

# 3. Export one trajectory JSON per strategy (now WITH terrain).
python scripts/export_trajectories.py

# 4. Serve the web/ directory (browsers won't fetch JSON from file://).
python -m http.server -d web 8080

# 5. Open http://localhost:8080 for 3D, or /index2d.html for 2D.
#    If a tab was already open, HARD-refresh (Cmd+Shift+R) to drop cached JSON.
```

Steps 1–2 are one-time setup. After that, the day-to-day loop is just
steps 3–5 (export → serve → refresh).

> The **2D viewer** (`/index2d.html`) only needs agent `x`/`y`, so it
> works even without terrain — handy as a fallback if the 3D setup above
> isn't done yet.

### What you'll see

- A 3D replay of the search-and-rescue mission by default
  - **Blue dots** — drones (fast, wide-area)
  - **Green dots** — ground robots (slow, verifying)
  - **Red dots** — survivors not yet scouted by a drone
  - **Yellow dots** — survivors scouted by a drone but not yet confirmed
  - **Light-green dots** — survivors confirmed by a ground robot
  - **Orange tiles** — burning fire cells (cellular-automata spread)
- Strategy selector in the header — switch between `random_action` /
  `random_walk` / `lawnmower` / `nearest_candidate` /
  `highest_confidence` / `ant_colony`
- Playback controls — play / pause, scrubber, speed (1× → 8×)
- Right panel — final mission metrics for the loaded run

## Troubleshooting

Real failures hit while bringing the 3D viewer up locally, with the exact
fix for each.

| Symptom | Cause | Fix |
|---|---|---|
| `zsh: command not found: python` | venv not active; macOS has no bare `python` | `source .venv/bin/activate` (prompt shows `(.venv)`), or use `python3` |
| `Missing geospatial dependencies: osmnx, pyproj, rasterio` | terrain builder deps not installed | `pip install -r requirements-geo.txt` |
| `FileNotFoundError: No cached real terrain ... malibu_creek_..._128.npz` | terrain cache not built, or built under the hash name only | Build it (step 2), then `cp` the hash-named `.npz` to the slug name |
| **3D canvas is black, but the metrics panel, dropdown, and playback all work** | the loaded `trajectories/*.json` is **stale** — exported before terrain existed, so it has no `terrain` block and no agent altitude. The scene builder bails (`buildTerrainMesh` returns `null`) and there's nothing to draw | Re-run `python scripts/export_trajectories.py` to rewrite the JSONs **with** terrain, then hard-refresh the browser |
| Black canvas, **no errors** in DevTools | the red console lines are usually browser-extension noise (`contentscript.js`, `ObjectMultiplex`, `MaxListenersExceeded`) — not the app. Confirms it's the stale-JSON case above, not a crash | same as above |
| `happo_trained` is dark while the baselines render | its checkpoint is incompatible (`size mismatch ... [25] vs [54]` — obs space grew), so the exporter **skips** it and leaves the old terrain-less JSON | Retrain to get a matching checkpoint, then re-export (see below) |
| Export aborts on the HAPPO step | an incompatible checkpoint raised `RuntimeError` mid-export | Already handled — the exporter now catches it and skips HAPPO gracefully. Update if you're on an older revision |
| `Address already in use` on `http.server` | a server is already bound to that port | Reuse the running one, or pick another port: `... 8081` |

### Making `happo_trained` render

`happo_trained` needs a checkpoint whose observation/action shapes match
the **current** scenario. If the scenario changed since the checkpoint was
saved, retrain to produce a fresh one, then re-export:

```bash
python scripts/train_happo_smoke.py     # ~8 s; produces a current-shape checkpoint
python scripts/export_trajectories.py    # picks up the newest checkpoint → happo_trained.json
```

Note the smoke budget (~2000 timesteps) is enough to make the viewer
*render* but the policy is essentially untrained (survivor recall ≈ 0).
For a meaningful trained policy, retrain at a real budget.

## How it's built

Single-file React + Three.js app using ES module imports and an
`importmap` to fetch from `esm.sh` at runtime. No bundler, no transpiler.

- `index.html` — default 3D viewer: React app, Three.js scene, and styling.
- `index2d.html` — legacy top-down 2D viewer.
- `trajectories/*.json` — one per strategy, written by
  [scripts/export_trajectories.py](../scripts/export_trajectories.py).
  See [evaluation/trajectory_export.py](../evaluation/trajectory_export.py)
  for the JSON schema.

The Three.js scene uses an `OrthographicCamera` for a true top-down view
of the world's `[-1, 1] × [-1, 1]` coordinate space. Each frame the React
side calls `renderFrame()` which clears the agent/survivor/fire groups
and rebuilds them — cheap for ~10 objects per frame.

## Upgrading to a build system

For a production capstone deliverable (TypeScript, tree-shaking, bundle
optimisation), upgrade to Vite:

```bash
cd web
npm create vite@latest . -- --template react-ts
npm install three @react-three/fiber @react-three/drei
# Move the App + scene logic from index.html/index2d.html into src/App.tsx
```

The buildless version is enough for the demo. The Vite upgrade is the
move when you want to add routing (e.g. side-by-side strategy compare),
tests, or distribute as a static build.

## Adding a new strategy to the viewer

Add the strategy to [agents/baselines.py](../agents/baselines.py)
(implementing the `policy(env) → actions` interface), then re-run
`python scripts/export_trajectories.py`. The viewer auto-discovers
strategies via the `STRATEGY_NAMES` constant in `index.html` and
`index2d.html` — also add the new name there.

Trained MAPPO/IPPO policies will plug in the same way once a checkpoint
loader lands.
