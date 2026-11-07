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
deadline=$((SECONDS+3600))
while [[ ! -s "$HOME/a_blind_20261107/comparison.json" ]]; do
    if (( SECONDS>deadline )); then exit 1; fi
    sleep 10
done
"$py" -u a_state_20261107/state.py --kind hidden --config "$old" "$new" --epochs 24 \
    --output "$r/model" > "$r/train.log" 2>&1
sha256sum "$r/model/state.pt" > "$r/frozen.sha256"
"$py" -u a_state_20261107/evaluate.py --config "$new" --checkpoint "$r/model/state.pt" --kit "$kit" \
    --jm "$HOME/jm-direct-replay-20261107" --output "$r/eval" --replay 20 > "$r/eval.log" 2>&1
for arm in clean hidden; do
    cp="$r/model/state.pt"
    pred="$r/eval"
    if [[ "$arm" = clean ]]; then
        cp="$HOME/a_blind_20261107/expanded/state.pt"
        pred="$HOME/a_blind_20261107/eval_expanded"
    fi
    "$py" -u a_state_20261107/calibrate.py --config "$old" "$new" --test-config "$new" \
        --checkpoint "$cp" --predictions "$pred" --output "$r/calibration_$arm.json" > "$r/calibration_$arm.log" 2>&1
done
