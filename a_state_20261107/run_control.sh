#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
r="${1:?new output directory}"
mkdir "$r"
py="$HOME/Wan2.1/.venv/bin/python"
old=expansion_20261106/config_train32.json
kit="$HOME/a_new48_20261107/kit"
new="$kit/state_config.json"
deadline=$((SECONDS+5400))
while [[ ! -s "$HOME/a_blind_20261107/dataset_audit.json" ]]; do
    if (( SECONDS>deadline )); then exit 1; fi
    sleep 10
done
"$py" - "$old" "$new" "$r" <<'PY'
import json,sys
from pathlib import Path
old,new,out=map(Path,sys.argv[1:])
old,new=(json.loads(p.read_text()) for p in (old,new))
updates=24*4*(len(old["splits"]["train"])+len(new["splits"]["train"]))
new["splits"]["train"]=[]
(out/"val_only_new.json").write_text(json.dumps(new))
(out/"updates").write_text(str(updates)+"\n")
PY
"$py" -u a_state_20261107/state.py --config "$old" "$r/val_only_new.json" --epochs 80 \
    --max-updates "$(cat "$r/updates")" --output "$r/model" > "$r/train.log" 2>&1
sha256sum "$r/model/state.pt" > "$r/frozen.sha256"
"$py" -u a_state_20261107/evaluate.py --config "$new" --checkpoint "$r/model/state.pt" --kit "$kit" \
    --jm "$HOME/jm-direct-replay-20261107" --output "$r/eval" --replay 20 > "$r/eval.log" 2>&1
