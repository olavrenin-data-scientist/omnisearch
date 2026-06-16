#!/usr/bin/env python3
"""
simulations/scripts/aggregate_to_csv.py
=======================================
Reads every timestamped trajectory JSON in simulations/data/runs/ (produced by
run_eda_batch.py) and flattens each into ONE row of an analysis-ready CSV,
written to simulations/data/aggregate/simulations_aggregate_data.csv.

Each row = one simulation run, with three groups of columns:

  RUN PARAMETERS   strategy, seed, comms_dropout, grid_size, n_steps,
                   n_drones, n_ground, n_survivors, sim_step_seconds, ...
  SUMMARY METRICS  the six mission metrics from metadata.metrics:
                   survivor_recall, time_to_verification,
                   false_positive_trips, hazard_exposure, ugv_travel_cost
  DERIVED FEATURES computed from the frame time-series:
                   final/peak fire & smoke cell counts, steps-to-first-found,
                   steps-to-first-scouted, final found/scouted counts,
                   mean comms uptime, total drone path length, etc.

The comms_dropout and timestamp are recovered from the filename
(written by run_eda_batch.py); everything else comes from inside the JSON.

Usage:
    python simulations/scripts/aggregate_to_csv.py
    python simulations/scripts/aggregate_to_csv.py --runs-dir simulations/data/runs \
        --out simulations/data/aggregate/simulations_aggregate_data.csv
    python simulations/scripts/aggregate_to_csv.py --no-frames   # metrics only, fast
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../simulations
DEFAULT_RUNS_DIR = os.path.join(SIM_ROOT, "data", "runs")
DEFAULT_OUT = os.path.join(SIM_ROOT, "data", "aggregate", "simulations_aggregate_data.csv")

# Recovers approach, dropout, seed, timestamp from the archive filename:
#   ant_colony__d0p4__s7__20260615-050723-123456.json
NAME_RE = re.compile(
    r"^(?P<approach>.+?)__d(?P<dropout>[0-9p]+)__s(?P<seed>\d+)__(?P<ts>[\d\-]+)\.json$"
)


def parse_filename(path):
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "file_approach": d["approach"],
        "file_comms_dropout": float(d["dropout"].replace("p", ".")),
        "file_seed": int(d["seed"]),
        "file_timestamp": d["ts"],
    }


def safe(d, *keys, default=None):
    """Nested get: safe(meta, 'metrics', 'survivor_recall')."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def derive_frame_features(frames):
    """Compute per-run features from the frame time-series. Robust to missing
    keys so it works across strategies and simulator versions."""
    feats = {
        "n_frames": len(frames),
        "peak_fire_cells": 0,
        "final_fire_cells": 0,
        "peak_smoke_cells": 0,
        "final_smoke_cells": 0,
        "steps_to_first_scouted": "",
        "steps_to_first_found": "",
        "final_scouted": "",
        "final_found": "",
        "mean_comms_uptime": "",
        "total_drone_path_len": "",
    }
    if not frames:
        return feats

    first_scouted_step = None
    first_found_step = None
    comms_up_total = 0
    comms_obs_total = 0

    prev_pos = {}
    drone_path = 0.0
    have_path = False

    for fr in frames:
        step = fr.get("step", 0)

        fire = fr.get("fire_cells")
        if isinstance(fire, list):
            feats["peak_fire_cells"] = max(feats["peak_fire_cells"], len(fire))
            feats["final_fire_cells"] = len(fire)

        smoke = fr.get("smoke_cells")
        if isinstance(smoke, list):
            feats["peak_smoke_cells"] = max(feats["peak_smoke_cells"], len(smoke))
            feats["final_smoke_cells"] = len(smoke)

        survivors = fr.get("survivors")
        if isinstance(survivors, list):
            sc = sum(1 for s in survivors if s.get("scouted"))
            fd = sum(1 for s in survivors if s.get("found"))
            feats["final_scouted"] = sc
            feats["final_found"] = fd
            if sc > 0 and first_scouted_step is None:
                first_scouted_step = step
            if fd > 0 and first_found_step is None:
                first_found_step = step

        agents = fr.get("agents")
        if isinstance(agents, list):
            for a in agents:
                if "comms_up" in a:
                    comms_obs_total += 1
                    comms_up_total += 1 if a.get("comms_up") else 0
                if a.get("type") == "drone" and "x" in a and "y" in a:
                    name = a.get("name", "")
                    p = (a["x"], a["y"])
                    if name in prev_pos:
                        dx = p[0] - prev_pos[name][0]
                        dy = p[1] - prev_pos[name][1]
                        drone_path += math.hypot(dx, dy)
                        have_path = True
                    prev_pos[name] = p

    feats["steps_to_first_scouted"] = first_scouted_step if first_scouted_step is not None else ""
    feats["steps_to_first_found"] = first_found_step if first_found_step is not None else ""
    if comms_obs_total:
        feats["mean_comms_uptime"] = round(comms_up_total / comms_obs_total, 4)
    if have_path:
        feats["total_drone_path_len"] = round(drone_path, 4)
    return feats


def row_from_file(path, include_frames=True):
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ! skipping {os.path.basename(path)}: {e}", file=sys.stderr)
        return None

    meta = data.get("metadata", {})
    metrics = meta.get("metrics", {})
    fname = parse_filename(path)

    row = {
        # --- provenance ---
        "source_file": os.path.basename(path),
        "timestamp": fname.get("file_timestamp", ""),
        # --- run parameters (prefer JSON metadata; fall back to filename) ---
        "strategy": meta.get("strategy", fname.get("file_approach", "")),
        "seed": meta.get("seed", fname.get("file_seed", "")),
        # comms_dropout is not stored in metadata, so take it from the filename
        "comms_dropout": fname.get("file_comms_dropout", ""),
        "grid_size": meta.get("fire_grid_size", ""),
        "n_steps": meta.get("n_steps", ""),
        "actual_n_steps": meta.get("actual_n_steps", ""),
        "n_drones": meta.get("n_drones", ""),
        "n_ground": meta.get("n_ground", ""),
        "n_survivors": meta.get("n_survivors", ""),
        "sim_step_seconds": meta.get("sim_step_seconds", ""),
        "actual_duration_seconds": meta.get("actual_duration_seconds", ""),
        "terrain_source": safe(meta, "terrain", "source", default=""),
        # --- summary metrics (the six) ---
        "survivor_recall": metrics.get("survivor_recall", ""),
        "time_to_verification": metrics.get("time_to_verification", ""),
        "false_positive_trips": metrics.get("false_positive_trips", ""),
        "hazard_exposure": metrics.get("hazard_exposure", ""),
        "ugv_travel_cost": metrics.get("ugv_travel_cost", ""),
    }

    if include_frames:
        row.update(derive_frame_features(data.get("frames", [])))

    return row


# Stable column order for the CSV.
BASE_COLS = [
    "source_file", "timestamp",
    "strategy", "seed", "comms_dropout",
    "grid_size", "n_steps", "actual_n_steps",
    "n_drones", "n_ground", "n_survivors",
    "sim_step_seconds", "actual_duration_seconds", "terrain_source",
    "survivor_recall", "time_to_verification", "false_positive_trips",
    "hazard_exposure", "ugv_travel_cost",
]
FRAME_COLS = [
    "n_frames",
    "peak_fire_cells", "final_fire_cells",
    "peak_smoke_cells", "final_smoke_cells",
    "steps_to_first_scouted", "steps_to_first_found",
    "final_scouted", "final_found",
    "mean_comms_uptime", "total_drone_path_len",
]


def main():
    ap = argparse.ArgumentParser(description="Aggregate timestamped run JSONs into a CSV.")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help="Folder of timestamped *.json runs (default simulations/data/runs).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    ap.add_argument("--no-frames", action="store_true",
                    help="Skip per-frame derived features (metrics only, faster).")
    ap.add_argument("--glob", default="*.json",
                    help="Filename glob within runs-dir (default *.json).")
    args = ap.parse_args()

    include_frames = not args.no_frames
    paths = sorted(
        p for p in glob.glob(os.path.join(args.runs_dir, args.glob))
        if not os.path.basename(p).startswith("_")  # skip _manifest.jsonl etc.
    )
    if not paths:
        print(f"No run files found in {args.runs_dir}")
        return

    cols = BASE_COLS + (FRAME_COLS if include_frames else [])
    rows = []
    for i, path in enumerate(paths, 1):
        row = row_from_file(path, include_frames=include_frames)
        if row:
            rows.append(row)
        if i % 50 == 0:
            print(f"  processed {i}/{len(paths)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows x {len(cols)} cols to {args.out}")


if __name__ == "__main__":
    main()
