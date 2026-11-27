#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=timestamp,index,pci.bus_id,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv -l 1 > gpu-monitor-recovery.csv &
monitor=$!
nvidia-smi pmon -i 0 -s um -d 2 > gpu-process-recovery.log &
watcher=$!
trap "kill $monitor $watcher 2>/dev/null || true" EXIT
py="$HOME/Wan2.1/.venv/bin/python"
date -Is
"$py" -u scripts/profile_pipeline.py profile-recovery-sample_0000 sample_0000 --repeats 0 --profiles 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$py" -u scripts/profile_encoder.py historical-sample_0000 sample_0000
"$py" -u scripts/profile_pipeline.py sample_1004 sample_1004
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$py" -u scripts/profile_encoder.py historical-sample_1004 sample_1004
date -Is
touch COMPLETE
"$py" -u scripts/profile_rgb.py rgb-detail
touch RGB_COMPLETE
date -Is
