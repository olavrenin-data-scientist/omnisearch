#!/usr/bin/env bash
set -euo pipefail

cd /Users/aschuetz/Software/capstone/omnisearch

RUN_TRAIN="${RUN_TRAIN:-TRUE}"
RUN_VAL="${RUN_VAL:-TRUE}"
RUN_EXPORT="${RUN_EXPORT:-FALSE}"
RUN_SERVER="${RUN_SERVER:-FALSE}"
NUM_STEPS=300000
NUM_STEPS_STR=300k

MODLABEL="uav3_${NUM_STEPS_STR}_confproposal_cov0_confLG60_conf30_moveconf010_confoverlap006_thr080_cleantarg010steps10_fh"

if [[ "$RUN_TRAIN" == "TRUE" ]]; then
  .venv/bin/python scripts/train_happo_smoke.py \
    --uav-survivor-diagnostic \
    --uav-diagnostic-drones 3 \
    --num-env-steps ${NUM_STEPS} \
    --episode-length 300 \
    --seed 1 \
    --share-param \
    --uav-frontier-source confidence \
    --uav-frontier-mode local_global \
    --uav-frontier-obs-radius-m 60 \
    --uav-coverage-reward 0 \
    --uav-move-coverage-reward 0 \
    --uav-frontier-alignment-reward 0.05 \
    --uav-overlap-penalty 0 \
    --uav-confidence-reward 30 \
    --uav-confidence-move-reward 0.1 \
    --uav-confidence-obs-grid 6 \
    --uav-outside-footprint-penalty 0.1 \
    --uav-confidence-overlap-penalty 0.02 \
    --uav-confidence-overlap-threshold 0.80 \
    --uav-cleanup-target-progress-reward 0.1 \
    --uav-cleanup-target-obs \
    --uav-cleanup-target-refresh-mode fixed-hold \
    --exp-name "$MODLABEL"
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_uav_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps 300 \
    --seeds {1000..1099} \
    --diagnostic-level fast \
    --json-output "outputs/${NUM_STEPS_STR}/${MODLABEL}_1000_1099.json" \
    --plots-output "outputs/${NUM_STEPS_STR}/${MODLABEL}_1000_1099.png"
fi

if [[ "$RUN_EXPORT" == "TRUE" ]]; then
  .venv/bin/python scripts/export_trajectories.py \
    --approach happo \
    --happo-checkpoint "$CKPT" \
    --seed 1017 \
    --steps 300 \
    --out web/trajectories

  cp web/trajectories/happo_trained.json "web/trajectories/${MODLABEL}.json"
fi

#### MODEL 2
MODLABEL="uav3_${NUM_STEPS_STR}_2frontier_baseline"

if [[ "$RUN_TRAIN" == "TRUE" ]]; then
  .venv/bin/python scripts/train_happo_smoke.py \
    --uav-survivor-diagnostic \
    --uav-diagnostic-drones 3 \
    --num-env-steps ${NUM_STEPS} \
    --episode-length 300 \
    --seed 1 \
    --share-param \
    --exp-name "$MODLABEL"
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_uav_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps 300 \
    --seeds {1000..1099} \
    --json-output "outputs/${NUM_STEPS_STR}/${MODLABEL}_1000_1099.json" \
    --plots-output "outputs/${NUM_STEPS_STR}/${MODLABEL}_1000_1099.png"
fi

# #### MODEL 3
MODLABEL="uav3_${NUM_STEPS_STR}_2frontier_baseline_cleantarget010"

if [[ "$RUN_TRAIN" == "TRUE" ]]; then
  .venv/bin/python scripts/train_happo_smoke.py \
    --uav-survivor-diagnostic \
    --uav-diagnostic-drones 3 \
    --num-env-steps ${NUM_STEPS} \
    --episode-length 300 \
    --seed 1 \
    --share-param \
    --uav-cleanup-target-progress-reward 0.1 \
    --uav-cleanup-target-obs \
    --uav-cleanup-target-refresh-mode fixed-hold \
    --exp-name "$MODLABEL"
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_uav_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps 300 \
    --seeds {1000..1099} \
    --json-output "outputs/${NUM_STEPS_STR}/${MODLABEL}_1000_1099.json" \
    --plots-output "outputs/${NUM_STEPS_STR}/${MODLABEL}_1000_1099.png"
fi


