#!/usr/bin/env bash
# 113Server, one new run directory. Compare an unchanged pilot with an expanded fit
# on the same untouched prompt groups. Evaluation begins only after both are locked.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
py="$HOME/Wan2.1/.venv/bin/python"
r="${1:?new output directory}"
mkdir "$r"
new="$HOME/a_new48_20261107/kit/state_config.json"
old=expansion_20261106/config_train32.json
pilot="$HOME/a_state_pilot_20261107/model/state.pt"
kit="$HOME/a_new48_20261107/kit"
jm="$HOME/jm-direct-replay-20261107"
deadline=$((SECONDS+5400))
while [[ ! -s "$new" ]]; do
    if (( SECONDS > deadline )); then
        echo "Timed out waiting for accepted labels" >&2
        exit 1
    fi
    sleep 10
done
# Validate video identity before selecting any weight.
"$py" -u a_state_20261107/verify_expansion.py "$new" a_state_20261107/new48.json "$r/dataset_audit.json"
"$py" -u a_state_20261107/state.py --config "$old" "$new" --epochs 24 --output "$r/expanded" > "$r/train.log" 2>&1
sha256sum "$pilot" "$r/expanded/state.pt" "$old" "$new" > "$r/frozen.sha256"
for arm in pilot expanded; do
    cp="$pilot"
    if [[ "$arm" = expanded ]]; then cp="$r/expanded/state.pt"; fi
    "$py" -u a_state_20261107/evaluate.py --config "$new" --checkpoint "$cp" --kit "$kit" --jm "$jm" \
        --output "$r/eval_$arm" --replay 20 > "$r/eval_$arm.log" 2>&1
done
"$py" -u a_state_20261107/compare_blind.py "$r"
