#!/usr/bin/env bash
set -euo pipefail

cd /Users/aschuetz/Software/capstone/omnisearch

RUN_TRAIN="${RUN_TRAIN:-FALSE}"
RUN_VAL="${RUN_VAL:-FALSE}"
RUN_EXPORT="${RUN_EXPORT:-FALSE}"
RUN_SERVER="${RUN_SERVER:-FALSE}"
RUN_TENSORBOARD="${RUN_TENSORBOARD:-FALSE}"
RUN_BASELINE="${RUN_BASELINE:-FALSE}"

NSTEPS_PER_EPISODE="${NSTEPS_PER_EPISODE:-900}"
NEPOCHS="${NEPOCHS:-254}"
NSTEPS_TOTAL=$((NEPOCHS * NSTEPS_PER_EPISODE * 8))
NDRONES=4
NUGVS=3
NSURVIVORS_OBS=5
NSURVIVORS_MIN=5
NSURVIVORS_MAX=5

MODLABEL="uav${NDRONES}_ugv${NUGVS}_1km_256_${NEPOCHS}_${NSTEPS_PER_EPISODE}_hid128_fire_nsurv_min${NSURVIVORS_MIN}_max${NSURVIVORS_MAX}_obs${NSURVIVORS_OBS}"

if [[ "$RUN_TRAIN" == "TRUE" ]]; then
  .venv/bin/python scripts/train_happo_smoke.py \
  --joint-survivor-diagnostic \
  --hidden-sizes 128 128 \
  --n-drone $NDRONES \
  --n-ugvs $NUGVS \
  --num-env-steps "$NSTEPS_TOTAL" \
  --episode-length "$NSTEPS_PER_EPISODE" \
  --n-survivors $NSURVIVORS_OBS \
  --active-survivors-min $NSURVIVORS_MIN \
  --active-survivors-max $NSURVIVORS_MAX \
  --enable-fire \
  --uav-fire-footprint-penalty 0.05 \
  --uav-fire-penalty-threshold 0.6 \
  --ugv-route-progress-shortfall-penalty 0.0025 \
  --ugv-target-assignment-mode greedy_sticky \
  --exp-name "$MODLABEL"
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/models | head -1)

if [[ "$RUN_VAL" == "TRUE" ]]; then
  date
  .venv/bin/python scripts/diagnose_joint_happo.py \
    --checkpoint-dir "$CKPT" \
    --steps "$NSTEPS_PER_EPISODE" \
    --seeds $(seq 1000 1049) \
    --json-output "outputs/${MODLABEL}_1000_1049.json" \
    --plots-output "outputs/${MODLABEL}_1000_1049.png"
fi

if [[ "$RUN_EXPORT" == "TRUE" ]]; then
  .venv/bin/python scripts/export_trajectories.py \
    --approach all \
    --happo-checkpoint "$CKPT" \
    --seed 1022 \
    --steps "$NSTEPS_PER_EPISODE" \
    --out web/trajectories
fi

if [[ "$RUN_SERVER" == "TRUE" ]]; then
  python -m http.server -d web
fi

if [[ "$RUN_BASELINE" == "TRUE" ]]; then
  .venv/bin/python scripts/diagnose_joint_baseline_strategies.py \
      --strategy lawnmower \
      --happo-checkpoint $CKPT \
      --steps 900 \
      --seeds $(seq 1000 1049) \
      --json-output "outputs/${MODLABEL}_lawnmower_1000_1049.json" \
      --plots-output "outputs/${MODLABEL}_lawnmower_1000_1049.png"
  
  .venv/bin/python scripts/diagnose_joint_baseline_strategies.py \
      --strategy ant_colony \
      --happo-checkpoint $CKPT \
      --steps 900 \
      --seeds $(seq 1000 1049) \
      --json-output "outputs/${MODLABEL}_ant_colony_1000_1049.json" \
      --plots-output "outputs/${MODLABEL}_ant_colony_1000_1049.png"
  
#  .venv/bin/python scripts/diagnose_joint_baseline_strategies.py \
#      --strategy random_walk \
#      --happo-checkpoint $CKPT \
#      --steps 900 \
#      --seeds $(seq 1000 1049) \
#      --json-output "outputs/${MODLABEL}_random_walk_1000_1049..json" \
#      --plots-output "outputs/${MODLABEL}_random_walk_1000_1049.png"
fi

CKPT=$(ls -td "results/harl_runs/wildfire/wildfire_search/happo/${MODLABEL}"/*/logs/train_episode_rewards_by_class | head -1)

if [[ "$RUN_TENSORBOARD" == "TRUE" ]]; then
  .venv/bin/tensorboard --logdir $CKPT
fi

