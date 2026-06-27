#!/usr/bin/env bash
set -euo pipefail

cd /Users/aschuetz/Software/capstone/omnisearch

MODLABEL="uav3_900k_confproposal_cov0_confLG60_conf30_moveconf010"

RUN_TRAIN="${RUN_TRAIN:-TRUE}"
RUN_VAL="${RUN_VAL:-TRUE}"
RUN_EXPORT="${RUN_EXPORT:-TRUE}"
RUN_SERVER="${RUN_SERVER:-FALSE}"

.venv/bin/python scripts/train_happo_smoke.py \
  --uav-survivor-diagnostic \
  --uav-diagnostic-drones 3 \
  --num-env-steps 900000 \
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
  --exp-name "$MODLABEL"

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_uav_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps 300 \
    --seeds {1000..1099} \
    --json-output "outputs/900k/${MODLABEL}_1000_1099.json" \
    --plots-output "outputs/900k/${MODLABEL}_1000_1099.png"
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

if [[ "$RUN_SERVER" == "TRUE" ]]; then
  python -m http.server -d web
fi

#SECOND RUN

MODLABEL="uav3_900k_confproposal_cov0_overlap0_conf30_moveconf010"
.venv/bin/python scripts/train_happo_smoke.py \
  --uav-survivor-diagnostic \
  --uav-diagnostic-drones 3 \
  --num-env-steps 900000 \
  --episode-length 300 \
  --seed 1 \
  --share-param \
  --uav-frontier-source coverage \
  --uav-frontier-mode sector_topk \
  --uav-frontier-top-k 2 \
  --uav-frontier-obs-radius-m 150 \
  --uav-confidence-obs-grid 6 \
  --uav-coverage-reward 0 \
  --uav-move-coverage-reward 0 \
  --uav-overlap-penalty 0 \
  --uav-confidence-reward 30 \
  --uav-confidence-move-reward 0.10 \
  --uav-confidence-gamma 2.0 \
  --uav-confidence-eps 0.05 \
  --uav-outside-footprint-penalty 0.10 \
  --uav-frontier-alignment-reward 0.05 \
  --exp-name "$MODLABEL"

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_uav_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps 300 \
    --seeds {1000..1099} \
    --json-output "outputs/900k/${MODLABEL}_1000_1099.json" \
    --plots-output "outputs/900k/${MODLABEL}_1000_1099.png"
fi
