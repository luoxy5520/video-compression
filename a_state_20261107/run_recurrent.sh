#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
r="${1:?new output directory}"
mkdir "$r"
py="$HOME/Wan2.1/.venv/bin/python"
kit="$HOME/a_new48_20261107/kit"
new="$kit/state_config.json"
old=expansion_20261106/config_train32.json
"$py" -u a_state_20261107/state.py --kind hidden --recurrent --config "$old" "$new" --epochs 24 \
    --output "$r/model" > "$r/train.log" 2>&1
sha256sum "$r/model/state.pt" > "$r/frozen.sha256"
"$py" -u a_state_20261107/evaluate.py --config "$new" --checkpoint "$r/model/state.pt" --kit "$kit" \
    --jm "$HOME/jm-direct-replay-20261107" --output "$r/eval" --replay 20 > "$r/eval.log" 2>&1
