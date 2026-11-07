#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
r="${1:?new output directory}"
mkdir "$r"
py="$HOME/Wan2.1/.venv/bin/python"
kit="$HOME/a_new48_20261107/kit"
new="$kit/state_config.json"
old=expansion_20261106/config_train32.json
cp="$HOME/a_hidden_blind_20261107/model/state.pt"
"$py" -u a_state_20261107/evaluate.py --config "$old" "$new" --checkpoint "$cp" \
    --kit "$kit" --jm "$HOME/jm-direct-replay-20261107" --output "$r/predictions" \
    --split train val test > "$r/export.log" 2>&1
"$py" - "$r" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
out=root/"predictions"
metrics=json.loads((out/"results.json").read_text())
assert len(metrics)==320
seen=set()
manifest=[]
for row in metrics:
    sid,step,split=row["sample_id"],row["step"],row["split"]
    assert (sid,step) not in seen
    seen.add((sid,step))
    npz=out/split/sid/f"step_{step:03d}"/"prediction.npz"
    manifest.append(dict(sample_id=sid,step=step,split=split,prediction=str(npz.relative_to(root)),
                         sha256=hashlib.sha256(npz.read_bytes()).hexdigest()))
(root/"manifest.json").write_text(json.dumps(dict(prediction_count=len(manifest),records=manifest),indent=2))
PY
# NPZ files are already compressed; use an uncompressed tar for a fast handoff.
cp "$cp" "$r/state.pt"
cp a_state_20261107/README.md "$r/README.md"
tar -C "$r" -cf "$r/handoff_80x4.tar" manifest.json state.pt README.md predictions
(cd "$r" && sha256sum handoff_80x4.tar state.pt > SHA256SUMS)
