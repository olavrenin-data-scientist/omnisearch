#!/usr/bin/env python3
"""
simulations/scripts/run_eda_batch.py
====================================
Drives the full EDA batch. For every row in simulations/configs/run_config.csv
it:

  1. Invokes the simulator:
        python scripts/export_trajectories.py \
            --approach <approach> --steps <steps> --grid-size <grid> \
            --comms-dropout <d> --seed <s> \
            --terrain-source real --terrain-cache-path <cache> \
            [--ignore-happo-env]

  2. The simulator writes its output to web/trajectories/ as <approach>.json
     (same name as the strategy, e.g. web/trajectories/lawnmower.json,
     overwritten each run).

  3. Waits for the run to FULLY finish (subprocess.run blocks), confirms the
     output file is freshly written, then COPIES it to simulations/data/runs/,
     appending a timestamp (and the run parameters) so nothing is lost:
        <approach>__d<dropout>__s<seed>__<YYYYmmdd-HHMMSS-ffffff>.json
     Only then does the next run start.

After every run finishes, point simulations/scripts/aggregate_to_csv.py at
simulations/data/runs to build the analysis CSV.

Usage:
    python simulations/scripts/run_eda_batch.py
    python simulations/scripts/run_eda_batch.py --limit 10    # smoke test
    python simulations/scripts/run_eda_batch.py --dry-run     # print commands
    python simulations/scripts/run_eda_batch.py --resume      # skip done runs

Paths assume you run from the project root (the folder containing simulations/
and the web/ output folder).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------
# Paths. SIM_ROOT is the simulations/ folder holding configs/, scripts/, data/.
# --------------------------------------------------------------------------
SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../simulations
PROJECT_ROOT = os.path.dirname(SIM_ROOT)                                 # parent of simulations/

CONFIG_PATH = os.path.join(SIM_ROOT, "configs", "run_config.csv")
EXPORT_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "export_trajectories.py")

# Where export_trajectories.py drops <approach>.json. The simulator's --out
# defaults to web/trajectories, so a lawnmower run produces
# web/trajectories/lawnmower.json.
WEB_DIR = os.environ.get(
    "OMNISEARCH_WEB_DIR", os.path.join(PROJECT_ROOT, "web", "trajectories")
)

# Where we archive the timestamped copies: simulations/data/runs/
RUNS_DATA_DIR = os.environ.get(
    "OMNISEARCH_RUNS_DIR", os.path.join(SIM_ROOT, "data", "runs")
)

# A small manifest tracking every completed run (for --resume and provenance).
MANIFEST_PATH = os.path.join(RUNS_DATA_DIR, "_manifest.jsonl")

# Columns in run_config.csv that should be parsed as ints / bools / floats.
INT_FIELDS = {"run_id", "seed", "steps", "grid_size"}
FLOAT_FIELDS = {"comms_dropout"}
BOOL_FIELDS = {"ignore_happo_env"}


def coerce_row(row):
    """Convert CSV string values to the right Python types."""
    out = {}
    for k, v in row.items():
        if k in INT_FIELDS:
            out[k] = int(v)
        elif k in FLOAT_FIELDS:
            out[k] = float(v)
        elif k in BOOL_FIELDS:
            out[k] = str(v).strip().lower() in ("true", "1", "yes")
        else:
            out[k] = v
    return out


def load_runs(path):
    with open(path, newline="") as f:
        return [coerce_row(r) for r in csv.DictReader(f)]


def build_command(run):
    """Translate one config run into the export_trajectories.py CLI call."""
    cmd = [
        sys.executable, EXPORT_SCRIPT,
        "--approach", str(run["approach"]),
        "--steps", str(run["steps"]),
        "--grid-size", str(run["grid_size"]),
        "--comms-dropout", str(run["comms_dropout"]),
        "--seed", str(run["seed"]),
        "--terrain-source", str(run.get("terrain_source", "real")),
        "--terrain-cache-path", str(run["terrain_cache_path"]),
    ]
    if run.get("ignore_happo_env"):
        cmd.append("--ignore-happo-env")
    return cmd


def timestamped_name(run, ts):
    """Build the archive filename embedding params + a unique timestamp."""
    d = str(run["comms_dropout"]).replace(".", "p")   # 0.4 -> 0p4
    return f"{run['approach']}__d{d}__s{run['seed']}__{ts}.json"


def load_done_keys():
    """Return the set of (approach, dropout, seed) already in the manifest."""
    done = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["approach"], rec["comms_dropout"], rec["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_manifest(record):
    os.makedirs(RUNS_DATA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run the OmniSearch EDA batch.")
    ap.add_argument("--config", default=CONFIG_PATH, help="Path to run_config.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N runs (smoke test).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without executing.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip runs already recorded in the manifest.")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going if a single run fails.")
    args = ap.parse_args()

    runs = load_runs(args.config)
    if args.limit is not None:
        runs = runs[: args.limit]

    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs(RUNS_DATA_DIR, exist_ok=True)

    done = load_done_keys() if args.resume else set()

    total = len(runs)
    ok, skipped, failed = 0, 0, 0
    t_start = time.time()

    for i, run in enumerate(runs, 1):
        key = (run["approach"], run["comms_dropout"], run["seed"])
        if args.resume and key in done:
            skipped += 1
            continue

        cmd = build_command(run)
        prefix = f"[{i}/{total}] {run['approach']} d={run['comms_dropout']} s={run['seed']}"

        if args.dry_run:
            print(prefix, "->", " ".join(cmd))
            continue

        # The simulator writes web/trajectories/<approach>.json, overwriting it
        # each run. Record the current mtime (if any) so we can confirm THIS run
        # produced a fresh file before we copy it.
        web_out = os.path.join(WEB_DIR, f"{run['approach']}.json")
        mtime_before = os.path.getmtime(web_out) if os.path.exists(web_out) else -1.0

        print(prefix, "running...", flush=True)
        # subprocess.run blocks until the run fully completes (check=True raises
        # on non-zero exit). We do NOT start the next run until this returns,
        # so every run finishes and is collected before the next begins.
        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"  ! FAILED ({e.returncode})", flush=True)
            if args.continue_on_error:
                continue
            print("Stopping. Use --continue-on-error to skip failures.")
            break

        # Confirm the run actually produced a NEW output file (fresh mtime),
        # not a leftover from a previous run.
        if not os.path.exists(web_out):
            failed += 1
            print(f"  ! expected output not found: {web_out}", flush=True)
            if args.continue_on_error:
                continue
            break
        if os.path.getmtime(web_out) <= mtime_before:
            failed += 1
            print(f"  ! output not refreshed (stale): {web_out}", flush=True)
            if args.continue_on_error:
                continue
            break

        # Copy with a unique timestamp into runs/data so nothing is overwritten.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        archive_name = timestamped_name(run, ts)
        archive_path = os.path.join(RUNS_DATA_DIR, archive_name)
        shutil.copy2(web_out, archive_path)

        append_manifest({
            "run_id": run.get("run_id"),
            "approach": run["approach"],
            "comms_dropout": run["comms_dropout"],
            "seed": run["seed"],
            "steps": run["steps"],
            "grid_size": run["grid_size"],
            "timestamp": ts,
            "archive_file": archive_name,
        })
        ok += 1
        print(f"  -> saved data/runs/{archive_name}", flush=True)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done. ok={ok} skipped={skipped} failed={failed} "
          f"of {total} in {elapsed:.1f}s")
    print(f"Archived runs in: {RUNS_DATA_DIR}")
    print(f"Next: python simulations/scripts/aggregate_to_csv.py")


if __name__ == "__main__":
    main()
