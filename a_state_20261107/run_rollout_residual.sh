#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
r="${1:?fresh output directory}"
kit="$HOME/a_new48_20261107/kit"
exec "$HOME/Wan2.1/.venv/bin/python" -u a_state_20261107/rollout_residual.py \
  --config expansion_20261106/config_train32.json "$kit/state_config.json" \
  --test-config "$kit/state_config.json" \
  --checkpoint a_state_20261107/results/expanded_hidden_state.pt \
  --kit "$kit" --jm "$HOME/jm-direct-replay-20261107" --output "$r" "${@:2}"
