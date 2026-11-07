#!/usr/bin/env bash
# Use a config with a fresh data output path and a fresh run directory.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
py="$HOME/Wan2.1/.venv/bin/python"
cfg="${1:?generation config}"
r="${2:?fresh run directory}"
mkdir "$r"
"$py" -u expansion_20261106/generate_pairs.py "$cfg" > "$r/generate.log" 2>&1 &
generation=$!
OMP_NUM_THREADS=1 "$py" -u a_state_20261107/label_expansion.py "$cfg" "$r/kit" \
    "$HOME/senior-oracle-expand32-20261106" > "$r/oracle.log" 2>&1 &
labels=$!
trap 'kill "$generation" "$labels" 2>/dev/null || true' EXIT
wait "$generation"
wait "$labels"
trap - EXIT
"$py" -u a_state_20261107/verify_expansion.py "$r/kit/state_config.json" "$cfg" "$r/dataset_audit.json"
