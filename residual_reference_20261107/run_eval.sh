#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=/mnt/data_hdd/luoxiangyu/dit_pairs/residual_reference_20261107
python=/home/luoxiangyu/Wan2.1/.venv/bin/python
common=(--config expansion_20261106/config_train32.json
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106
  --jm /home/luoxiangyu/jm-residual-reference-20261107
  --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106
  --previous /mnt/data_hdd/luoxiangyu/dit_pairs/full_state_pilot_20261107/benchmark)
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
for model in aligned_generated aligned_mixed; do
  "$python" -u residual_reference_20261107/benchmark_source.py "${common[@]}" --checkpoint "$work/$model/source.pt" --output "$work/eval_$model"
done
test -s "$work/pretrained_tail/training.json"
"$python" -u residual_reference_20261107/benchmark_source.py "${common[@]}" --type tail --checkpoint "$work/pretrained_tail/tail.pt" --output "$work/eval_pretrained_tail"
