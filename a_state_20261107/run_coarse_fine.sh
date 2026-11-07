#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
run="${1:?fresh experiment root with teacher archive}"
kit="$HOME/a_new48_20261107/kit"
deadline=$((SECONDS+3600))
while [[ ! -s "$run/teacher/results.json" ]]; do
    if (( SECONDS>deadline )); then echo "Teacher trajectories did not finish" >&2; exit 1; fi
    sleep 5
done
exec "$HOME/Wan2.1/.venv/bin/python" -u a_state_20261107/coarse_fine_mv.py \
  --config expansion_20261106/config_train32.json "$kit/state_config.json" \
  --test-config "$kit/state_config.json" --teacher "$run/teacher/labels" \
  --checkpoint a_state_20261107/results/expanded_hidden_state.pt \
  --kit "$kit" --jm "$HOME/jm-direct-replay-20261107" --output "$run/student"
