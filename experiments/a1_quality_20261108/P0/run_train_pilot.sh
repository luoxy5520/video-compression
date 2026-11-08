#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
p=experiments/a1_quality_20261108/P0
py="$HOME/Wan2.1/.venv/bin/python"
kit="$HOME/a_new48_20261107/kit"
jm="$HOME/a1_quality_20261108/P0/jm_mb_bits"
models=/mnt/data_hdd/models/dit_compression/a1_quality_20261108
out="${1:?provide fresh output directory}"
sample="${2:-sample_1004}"
mkdir "$out"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

for first in 1 5 9 13; do
    pids=()
    for ((frame=first; frame<first+4; frame++)); do
        printf -v name 'frame_%02d' "$frame"
        "$py" "$p/local_cost.py" --config "$kit/state_config.json" \
          --checkpoint "$models/locked_hidden.pt" --head "$models/frozen_coarse_oracle.pt" \
          --kit "$kit" --jm "$jm" --output "$out/$name" \
          --sample "$sample" --frame "$frame" --trace-bits >"$out/$name.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 )) || exit 1
done
"$py" -u "$p/train_replay.py" --config "$kit/state_config.json" --kit "$kit" --jm "$jm" \
    --audit "$out" --output "$out/replay" --sample "$sample"
