#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=/mnt/data_hdd/luoxiangyu/dit_pairs/closed_loop_20261107
python=/home/luoxiangyu/Wan2.1/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
for seeds in 1 2; do
  "$python" -u closed_loop_20261107/train_motion.py \
    --config expansion_20261106/config_train32.json \
    --mode expansion_20261106/results_expand32/stage_4.pt \
    --regressor full_state_20261107/results/regressor.pt \
    --train-per-prompt "$seeds" --output "$work/motion_per_prompt_$seeds"
done
