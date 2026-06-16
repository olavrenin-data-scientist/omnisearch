# OmniSearch EDA — Data Generation Pipeline

Generate ~500 simulation runs across six coordination strategies, archive each
output with a timestamp under `simulations/data/runs/`, then aggregate
everything into one CSV under `simulations/data/aggregate/` for the EDA.

## Folder layout

```
simulations/
├── configs/
│   ├── generate_run_config.py   # builds the run matrix
│   └── run_config.csv           # generated: 510 runs (6 strats x 5 dropout x 17 seeds)
├── scripts/
│   ├── run_eda_batch.py         # runs each config row, timestamps each output
│   └── aggregate_to_csv.py      # flattens all timestamped JSONs -> CSV
└── data/
    ├── runs/                    # timestamped run copies (+ _manifest.jsonl)
    └── aggregate/               # simulations_aggregate_data.csv

# Outside simulations/ (your existing project):
scripts/export_trajectories.py   # YOUR simulator
web/trajectories/                # export_trajectories.py writes <approach>.json here
```

## What gets swept

Fixed for every run (so strategies are comparable): `grid_size=128`,
`terrain_source=real`, the terrain cache, and `steps=500`.

Swept:
- **approach** — `random_action`, `random_walk`, `lawnmower`,
  `nearest_candidate`, `highest_confidence`, `ant_colony`
- **comms_dropout** — `0.0, 0.2, 0.4, 0.6, 0.8`
- **seed** — `0..16`

6 x 5 x 17 = **510 runs**.

## Workflow

```bash
# 1. Generate the run matrix -> simulations/configs/run_config.csv
python simulations/configs/generate_run_config.py

# 2. (optional) sanity-check the commands without running
python simulations/scripts/run_eda_batch.py --dry-run --limit 5

# 3. Run the full batch.
#    For each run it: calls export_trajectories.py -> waits for it to finish
#    -> reads web/trajectories/<approach>.json -> copies it to
#    simulations/data/runs/<approach>__d<d>__s<seed>__<timestamp>.json
#    -> only then starts the next run.
python simulations/scripts/run_eda_batch.py

#    Resume later if interrupted (skips runs already in the manifest):
python simulations/scripts/run_eda_batch.py --resume

# 4. Aggregate every timestamped JSON into the analysis CSV
python simulations/scripts/aggregate_to_csv.py
#    -> simulations/data/aggregate/simulations_aggregate_data.csv
```

The exact simulator call used per run:

```bash
python scripts/export_trajectories.py \
    --approach <approach> --steps 500 --grid-size 128 \
    --comms-dropout <d> --seed <s> \
    --terrain-source real \
    --terrain-cache-path data/terrain_cache/f15b8960d21b_128.npz
```

## run_config.csv

One row per run. Columns: `run_id, approach, comms_dropout, seed, steps,
grid_size, terrain_source, terrain_cache_path, ignore_happo_env`.

## simulations_aggregate_data.csv columns

One row per run.

**Run parameters:** `strategy, seed, comms_dropout, grid_size, n_steps,
actual_n_steps, n_drones, n_ground, n_survivors, sim_step_seconds,
actual_duration_seconds, terrain_source`

**Six summary metrics** (from `metadata.metrics`): `survivor_recall,
time_to_verification, false_positive_trips, hazard_exposure, ugv_travel_cost`

**Derived time-series features** (computed from frames): `n_frames,
peak_fire_cells, final_fire_cells, peak_smoke_cells, final_smoke_cells,
steps_to_first_scouted, steps_to_first_found, final_scouted, final_found,
mean_comms_uptime, total_drone_path_len`

`comms_dropout` is recovered from the archive filename (it is not stored inside
the simulator's JSON), which is why the runner encodes it into the name.

## Notes / things to adjust

- **Web folder location.** `run_eda_batch.py` assumes the simulator writes to
  `web/trajectories/<approach>.json` at the project root. If it writes
  elsewhere, set `OMNISEARCH_WEB_DIR=/path/to/dir` (or edit `WEB_DIR`).
- **Archive location.** Defaults to `simulations/data/runs/`. Override with
  `OMNISEARCH_RUNS_DIR` if desired.
- **Faster aggregation.** `aggregate_to_csv.py --no-frames` skips the per-frame
  features and emits metrics-only rows in a fraction of the time.
