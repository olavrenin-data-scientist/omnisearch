"""
simulations/configs/generate_run_config.py
===========================================
Generates the experiment configuration for the OmniSearch EDA: ~500 runs
spanning all six coordination strategies, several comms-dropout levels, and
multiple random seeds.

Writes  simulations/configs/run_config.csv  — one row per run, consumed by
simulations/scripts/run_eda_batch.py.

Design rule for comparability: every run shares the same grid size, terrain
source, terrain cache, and episode length. Only the swept axes change:
    - approach        (the strategy under test)
    - comms_dropout   (network degradation)
    - seed            (statistical replication)

Run:  python simulations/configs/generate_run_config.py
"""

import csv
import os

# ----------------------------------------------------------------------
# FIXED parameters (identical for every run, so strategies are comparable)
# ----------------------------------------------------------------------
FIXED = {
    "steps": 500,
    "grid_size": 128,
    "terrain_source": "real",
    "terrain_cache_path": "data/terrain_cache/f15b8960d21b_128.npz",
    "ignore_happo_env": False,
}

# ----------------------------------------------------------------------
# SWEPT axes
# ----------------------------------------------------------------------
APPROACHES = [
    "random_action",
    "random_walk",
    "lawnmower",
    "nearest_candidate",
    "highest_confidence",
    "ant_colony",
]

# Comms dropout: perfect radio through mostly-broken (range 0 - 0.8)
COMMS_DROPOUT = [0.0, 0.2, 0.4, 0.6, 0.8]

# Seeds for statistical replication. 6 strategies x 5 dropout x N seeds:
#   6 * 5 * 17 = 510 runs  (~500)
SEEDS = list(range(17))   # 0..16

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "run_config.csv")

COLUMNS = [
    "run_id", "approach", "comms_dropout", "seed",
    "steps", "grid_size", "terrain_source", "terrain_cache_path",
    "ignore_happo_env",
]


def build_runs():
    runs = []
    run_id = 0
    for approach in APPROACHES:
        for dropout in COMMS_DROPOUT:
            for seed in SEEDS:
                runs.append({
                    "run_id": run_id,
                    "approach": approach,
                    "comms_dropout": round(dropout, 3),
                    "seed": seed,
                    **FIXED,
                })
                run_id += 1
    return runs


def main():
    runs = build_runs()
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    print(f"Wrote {len(runs)} runs to {OUTPUT_PATH}")
    print(f"  {len(APPROACHES)} approaches x {len(COMMS_DROPOUT)} dropout "
          f"x {len(SEEDS)} seeds = {len(runs)} runs")


if __name__ == "__main__":
    main()
