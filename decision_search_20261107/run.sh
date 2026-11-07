#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=/mnt/data_hdd/luoxiangyu/dit_pairs/decision_search_20261107
previous=/mnt/data_hdd/luoxiangyu/dit_pairs/residual_reference_20261107
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4
python=/home/luoxiangyu/Wan2.1/.venv/bin/python
common=(--config expansion_20261106/config_train32.json
  --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106
  --jm "${DECISION_JM:-/home/luoxiangyu/jm-decision-mode-20261107}"
  --work "$work" --projection "$previous/calibrated/selected.pt")
"$python" -u decision_search_20261107/experiment.py "${common[@]}" "$@"
