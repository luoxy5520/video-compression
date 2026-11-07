#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
models=/mnt/data_hdd/models/dit_compression
exec /home/luoxiangyu/Wan2.1/.venv/bin/python -u learned_qpel_20261107/run.py \
 --config expansion_20261106/config_train32.json \
 --projection "$models/residual_reference_20261107/rank40_projection.pt" \
 --decision "$models/decision_search_20261107/decision.pt" \
 --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
 --jm /home/luoxiangyu/jm-learned-qpel-20261107 \
 --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106 \
 --output /home/luoxiangyu/learned_qpel_20261107 \
 --weights "$models/learned_qpel_20261107/qpel.pt" "$@"
