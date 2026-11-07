#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python=/home/luoxiangyu/Wan2.1/.venv/bin/python
work=/mnt/data_hdd/luoxiangyu/dit_pairs/closed_loop_20261107
kit=/home/luoxiangyu/senior-oracle-expand32-20261106
jm=/home/luoxiangyu/jm-closed-loop-20261107
data=/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106
config=expansion_20261106/config_train32.json
common=(--config "$config" --kit "$kit" --jm "$jm" --data "$data")
heads=(--mode expansion_20261106/results_expand32/stage_4.pt --regressor full_state_20261107/results/regressor.pt)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
"$python" -u closed_loop_20261107/benchmark_closed_loop.py "${common[@]}" "${heads[@]}" --output "$work/smoke_fullsearch" --smoke
"$python" -u closed_loop_20261107/benchmark_closed_loop.py "${common[@]}" "${heads[@]}" --output "$work/benchmark" --motion "$work/motion_fixed/motion.pt"
"$python" -u closed_loop_20261107/residual_control.py "${common[@]}" --output "$work/residual" --previous /mnt/data_hdd/luoxiangyu/dit_pairs/full_state_pilot_20261107/benchmark
"$python" -u closed_loop_20261107/benchmark_closed_loop.py "${common[@]}" "${heads[@]}" --output "$work/extra" --motion "$work/motion_fixed/motion.pt" --extra-only
