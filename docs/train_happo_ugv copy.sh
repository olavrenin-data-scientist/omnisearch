#!/usr/bin/env bash
set -euo pipefail

cd /Users/aschuetz/Software/capstone/omnisearch

RUN_TRAIN="${RUN_TRAIN:-FALSE}"
RUN_VAL="${RUN_VAL:-FALSE}"
RUN_EXPORT="${RUN_EXPORT:-FALSE}"
RUN_SERVER="${RUN_SERVER:-FALSE}"
RUN_TENSORBOARD="${RUN_TENSORBOARD:-FALSE}"

NSTEPS_PER_EPISODE="${NSTEPS_PER_EPISODE:-900}"
NEPOCHS="${NEPOCHS:-150}"
NSTEPS_TOTAL=$((NEPOCHS * NSTEPS_PER_EPISODE * 8))
NDRONES=4
NUGVS=3

MODLABEL="ugv${NUGVS}_1km_256_${NEPOCHS}_${NSTEPS_PER_EPISODE}_shortfall0005"

if [[ "$RUN_TRAIN" == "TRUE" ]]; then
  .venv/bin/python scripts/train_happo_smoke.py \
  --joint-schema-ugv-diagnostic \
  --hidden-sizes 128 128 \
  --n-drone $NDRONES \
  --n-ugvs $NUGVS \
  --num-env-steps "$NSTEPS_TOTAL" \
  --episode-length "$NSTEPS_PER_EPISODE" \
  --enable-fire \
  --ugv-route-progress-shortfall-penalty 0.005 \
  --ugv-target-assignment-mode greedy_sticky \
  --terrain-cache-path data/terrain_cache/malibu_creek_1km_256.npz \
  --fire-grid-size 256 \
  --exp-name "$MODLABEL"
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_ugv_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps "$NSTEPS_PER_EPISODE" \
    --seeds $(seq 1000 1099) \
    --json-output "outputs/${MODLABEL}_1000_1099.json" \
    --plots-output "outputs/${MODLABEL}_1000_1099.png"
fi

if [[ "$RUN_EXPORT" == "TRUE" ]]; then
  .venv/bin/python scripts/export_trajectories.py \
    --approach happo \
    --happo-checkpoint "$CKPT" \
    --seed 1087 \
    --steps "$NSTEPS_PER_EPISODE" \
    --out web/trajectories \
fi

if [[ "$RUN_SERVER" == "TRUE" ]]; then
  python -m http.server -d web
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/logs | head -1)

if [[ "$RUN_TENSORBOARD" == "TRUE" ]]; then
  .venv/bin/tensorboard --logdir $CKPT
fi
