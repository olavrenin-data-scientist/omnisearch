cd /Users/aschuetz/Software/capstone/omnisearch
source /Users/aschuetz/Software/capstone/omnisearch/.venv/bin/activate
.venv/bin/python scripts/train_happo_smoke.py --uav-survivor-diagnostic --uav-diagnostic-drones 3 --share-param --num-env-steps 900000 --episode-length 300 --uav-frontier-obs --uav-frontier-alignment-reward 0.10 --exp-name uav3_900k_2frontier

.venv/bin/python scripts/train_happo_smoke.py --uav-survivor-diagnostic --uav-diagnostic-drones 3 --share-param --num-env-steps 900000 --episode-length 300 --uav-coverage-reward 0.5 --uav-coverage-normalization opportunity --uav-move-coverage-reward 0.1 --uav-coverage-opportunity-cap 1.0 --uav-frontier-obs --uav-frontier-alignment-reward 0.10 --exp-name uav3_900k_oppcov_2frontier_move

.venv/bin/python scripts/train_happo_smoke.py --uav-survivor-diagnostic --uav-diagnostic-drones 3 --share-param --num-env-steps 900000 --episode-length 300 --uav-coverage-reward 0.5 --uav-coverage-normalization opportunity --uav-move-coverage-reward 0.1 --uav-coverage-opportunity-cap 1.0 --uav-frontier-obs --uav-frontier-alignment-reward 0.10 --exp-name uav3_900k_oppcov_2frontier_move_opppen
.venv/bin/python scripts/train_happo_smoke.py --uav-survivor-diagnostic --uav-diagnostic-drones 3 --share-param --num-env-steps 900000 --episode-length 300 --exp-name uav3_900k
