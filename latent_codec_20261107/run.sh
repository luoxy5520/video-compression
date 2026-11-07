#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec /home/luoxiangyu/Wan2.1/.venv/bin/python -u latent_codec_20261107/run.py \
  --config expansion_20261106/config_train32.json \
  --projection /mnt/data_hdd/models/dit_compression/residual_reference_20261107/rank40_projection.pt \
  --decision /mnt/data_hdd/models/dit_compression/decision_search_20261107/decision.pt \
  --native-validation decision_search_20261107/results/eval_guarded/results.json \
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
  --jm /home/luoxiangyu/jm-latent-codec-20261107 \
  --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106 \
  --output /home/luoxiangyu/latent_codec_20261107 "$@"
