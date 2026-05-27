# OmniSearch · Strategy Viewer (React + Three.js)

Minimal browser app to replay rolled-out trajectories from each
coordination strategy and inspect their final mission-level metrics.

## How to run

```bash
# 1. Export trajectories (one JSON per baseline strategy, ~1.5 s total)
python scripts/export_trajectories.py

# 2. Serve the web/ directory (browsers won't fetch JSON from file://)
python -m http.server -d web 8080

# 3. Open http://localhost:8080
```

That's it. No `npm install`, no build step.

### What you'll see

- A top-down replay of the search-and-rescue mission
  - **Blue dots** — drones (fast, wide-area)
  - **Green dots** — ground robots (slow, verifying)
  - **Red dots** — survivors not yet scouted by a drone
  - **Yellow dots** — survivors scouted by a drone but not yet confirmed
  - **Light-green dots** — survivors confirmed by a ground robot
  - **Orange tiles** — burning fire cells (cellular-automata spread)
- Strategy selector in the header — switch between `random` /
  `lawnmower` / `nearest_candidate` / `highest_confidence`
- Playback controls — play / pause, scrubber, speed (1× → 8×)
- Right panel — final mission metrics for the loaded run

## How it's built

Single-file React + Three.js app using ES module imports and an
`importmap` to fetch from `esm.sh` at runtime. No bundler, no transpiler.

- `index.html` — everything: the React app, the Three.js setup, the
  styling. ~300 lines.
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
# Move the App + scene logic from index.html into src/App.tsx
```

The buildless version is enough for the demo. The Vite upgrade is the
move when you want to add routing (e.g. side-by-side strategy compare),
tests, or distribute as a static build.

## Adding a new strategy to the viewer

Add the strategy to [agents/baselines.py](../agents/baselines.py)
(implementing the `policy(env) → actions` interface), then re-run
`python scripts/export_trajectories.py`. The viewer auto-discovers
strategies via the `STRATEGY_NAMES` constant in `index.html` — also add
the new name there.

Trained MAPPO/IPPO policies will plug in the same way once a checkpoint
loader lands.
