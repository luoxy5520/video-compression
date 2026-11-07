#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4
previous=/mnt/data_hdd/luoxiangyu/dit_pairs
exec /home/luoxiangyu/Wan2.1/.venv/bin/python -u direct_replay_20261107/run.py \
  --config expansion_20261106/config_train32.json \
  --mode expansion_20261106/results_expand32/stage_4.pt \
  --old-regressor full_state_20261107/results/regressor.pt \
  --head /mnt/data_hdd/models/dit_compression/decision_search_20261107/decision.pt \
  --decisions "$previous/decision_search_20261107/samples" \
  --baseline "$previous/full_state_pilot_20261107/benchmark" \
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
  --jm /home/luoxiangyu/jm-direct-replay-20261107 \
  --data "$previous/exp_expand32_20261106" \
  --output /home/luoxiangyu/direct_replay_20261107 \
  --weights /mnt/data_hdd/models/dit_compression/direct_replay_20261107/residual.f64 "$@"
