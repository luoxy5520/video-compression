#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=/mnt/data_hdd/luoxiangyu/dit_pairs/residual_reference_20261107
python=/home/luoxiangyu/Wan2.1/.venv/bin/python
common=(--config expansion_20261106/config_train32.json
        --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106
        --traces "$work/traces" --init "$work/source_generated/source.pt" --steps 4500)
test -s "$work/traces/results.json"
test -s "$work/public/manifest.json"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$python" -u residual_reference_20261107/train_source.py "${common[@]}" --output "$work/aligned_generated" > "$work/aligned_generated.log" 2>&1 &
first=$!
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$python" -u residual_reference_20261107/train_source.py "${common[@]}" --public "$work/public/manifest.json" --output "$work/aligned_mixed" > "$work/aligned_mixed.log" 2>&1 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
exit "$status"
