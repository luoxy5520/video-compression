#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
p=experiments/a1_quality_20261108/P0
py="$HOME/Wan2.1/.venv/bin/python"
run="${1:?main batch output}"
out="${2:?new output}"
mkdir "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t samples < <("$py" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["val"], sep="\n")' "$run/expected.json")
test "${#samples[@]}" -eq 16
pids=()
for sample in "${samples[@]}"; do
    while [[ ! -s "$run/val/$sample/replay/rd_curve.csv" ]]; do
        if [[ -s "$run/status.json" ]]; then exit 1; fi
        sleep 10
    done
    "$py" -u "$p/trajectory_check.py" --split val --sample "$sample" \
      --replay "$run/val/$sample/replay" --output "$out/$sample" >"$out/$sample.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} == 4 )); then
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        (( failed == 0 )) || exit 1
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
