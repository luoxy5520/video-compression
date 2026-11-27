#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=timestamp,index,pci.bus_id,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv -l 1 > gpu-monitor.csv &
monitor=$!
trap "kill $monitor 2>/dev/null || true" EXIT
py="$HOME/Wan2.1/.venv/bin/python"
for sample in sample_0000 sample_1004; do
    date -Is
    "$py" -u scripts/profile_pipeline.py "$sample" "$sample"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$py" -u scripts/profile_encoder.py "historical-$sample" "$sample"
done
date -Is
touch COMPLETE
