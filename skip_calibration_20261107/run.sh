#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
previous=/mnt/data_hdd/luoxiangyu/dit_pairs
exec /home/luoxiangyu/Wan2.1/.venv/bin/python -u skip_calibration_20261107/run.py \
  --config expansion_20261106/config_train32.json \
  --previous /home/luoxiangyu/direct_replay_synced_20261107 \
  --decisions "$previous/decision_search_20261107/samples" \
  --native-validation decision_search_20261107/results/eval_guarded/results.json \
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
  --jm /home/luoxiangyu/jm-skip-calibration-20261107 \
  --data "$previous/exp_expand32_20261106" \
  --output /home/luoxiangyu/skip_calibration_20261107 "$@"
