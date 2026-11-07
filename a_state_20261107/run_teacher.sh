#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
py="$HOME/Wan2.1/.venv/bin/python"
r="${1:?new output directory}"
mkdir "$r"
models=/mnt/data_hdd/models/dit_compression
cfg=expansion_20261106/config_train32.json
kit="$HOME/senior-oracle-expand32-20261106"
jm="$HOME/jm-direct-replay-20261107"
"$py" -u a_state_20261107/teacher.py --config "$cfg" --teacher-jm "$HOME/jm-a-teacher-20261107" \
    --jm "$jm" --kit "$kit" --projection "$models/residual_reference_20261107/rank40_projection.pt" \
    --decision "$models/decision_search_20261107/decision.pt" --qpel "$models/learned_qpel_20261107/qpel.pt" \
    --output "$r/teacher" > "$r/trace.log" 2>&1
for arm in oracle distilled; do
    params=()
    if [[ "$arm" = distilled ]]; then params=(--teacher-labels "$r/teacher/labels"); fi
    "$py" -u a_state_20261107/state.py --config "$cfg" --final-only --epochs 24 \
        --output "$r/$arm" "${params[@]}" > "$r/train_$arm.log" 2>&1
done
sha256sum "$r/oracle/state.pt" "$r/distilled/state.pt" > "$r/frozen.sha256"
for arm in oracle distilled; do
    "$py" -u a_state_20261107/evaluate.py --config "$cfg" --steps 20 --replay 20 --kit "$kit" --jm "$jm" \
        --checkpoint "$r/$arm/state.pt" --output "$r/eval_$arm" > "$r/eval_$arm.log" 2>&1
done
