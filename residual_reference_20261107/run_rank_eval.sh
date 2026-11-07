#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python=/home/luoxiangyu/Wan2.1/.venv/bin/python
work=/mnt/data_hdd/luoxiangyu/dit_pairs/residual_reference_20261107
common=(--config expansion_20261106/config_train32.json --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
"$python" -u residual_reference_20261107/calibrate_rank.py "${common[@]}" --checkpoint "$work/rank_projection/rank.pt" --output "$work/calibrated"
"$python" -u residual_reference_20261107/benchmark_source.py "${common[@]}" --type rank --checkpoint "$work/calibrated/selected.pt"   --kit /home/luoxiangyu/senior-oracle-expand32-20261106 --jm /home/luoxiangyu/jm-residual-reference-20261107   --previous /mnt/data_hdd/luoxiangyu/dit_pairs/full_state_pilot_20261107/benchmark   --motion /mnt/data_hdd/luoxiangyu/dit_pairs/closed_loop_20261107/motion_fixed/motion.pt   --output "$work/eval_rank_projection"
