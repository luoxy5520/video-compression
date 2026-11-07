#!/usr/bin/env bash
# Durable, single-GPU validation queue. A trial never touches the regression split.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
root="${1:?output directory}"
limit="${2:-6}"  # 0 keeps scheduling until stopped or an operational guard trips
[[ "$limit" =~ ^[0-9]+$ ]]
root="$(mkdir -p "$root" && cd "$root" && pwd)"
exec 9>"$root/queue.lock"
if [[ "${A1_WAIT_FOR_QUEUE:-0}" == 1 ]]; then
    echo "Waiting for the existing queue lock"
    flock 9
else
    flock -n 9
fi
if [[ "${A1_RESUME_AFTER_STOP:-0}" == 1 && -e "$root/STOP" ]]; then
    rm -- "$root/STOP"
fi
py="$HOME/Wan2.1/.venv/bin/python"
kit="$HOME/a_new48_20261107/kit"
teacher="$HOME/a_coarse_fine_20261107/teacher"
[[ -s "$teacher/results.json" ]]
"$py" - "$teacher/results.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert len(r)==60 and all(x["bitstream_exact"] and x["encoder_matches_decoder"] for x in r)
PY
risk=(0.25 0.5 1 2 0.75 1.5)
for ((i=1;limit==0 || i<=limit;i++)); do
    if [[ -e "$root/STOP" ]]; then echo "Stopped before trial $i by STOP marker"; break; fi
    run="$root/trial_$(printf '%03d' "$i")"
    if [[ -s "$run/results.json" ]]; then continue; fi
    if [[ -e "$run" ]]; then mv "$run" "$run.incomplete.$(date +%s)"; fi
    available="$(df -Pk "$root" | awk 'END{print $4}')"
    if (( available < 104857600 )); then echo "Need at least 100 GiB free"; exit 1; fi
    resume=()
    if ((limit == 0 && i > 1)); then
        previous="$root/trial_$(printf '%03d' "$((i-1))")"
        [[ -s "$previous/results.json" && -s "$previous/oracle.pt" && -s "$previous/teacher.pt" ]]
        resume=(--resume-motion "$previous")
    fi
    echo "START trial=$i risk=${risk[(i-1)%6]} seed=$((1700+i)) resume=${resume[1]:-fresh}"
    "$py" -u a_state_20261107/coarse_fine_mv.py \
      --config expansion_20261106/config_train32.json "$kit/state_config.json" \
      --checkpoint a_state_20261107/results/expanded_hidden_state.pt \
      --teacher "$teacher/labels" --kit "$kit" --jm "$HOME/jm-direct-replay-20261107" \
      --output "$run" --validation-only --risk-weight "${risk[(i-1)%6]}" --seed "$((1700+i))" "${resume[@]}" \
      > "$run.log" 2>&1
    "$py" a_state_20261107/queue_score.py "$root"
done
