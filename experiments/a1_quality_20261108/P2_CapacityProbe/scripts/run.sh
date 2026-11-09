set -euo pipefail
cd ~/dit-p2-capacityprobe-20261109
d=experiments/a1_quality_20261108/P2_CapacityProbe
r=~/a1_quality_20261108/P2_CapacityProbe
py=~/Wan2.1/.venv/bin/python
cat > "$d/protocol_locked.yaml" <<'EOF'
target: OnPolicy O0 T0, historical M0 first eight training prompt representatives
widths: {A: 64, B: 352, C: 1504}
head: fixed 32 features per block and 16 raster frequencies, only first convolution width changes
seeds: [1701, 1702]
optimizer: AdamW 0.001, batch one video, same seeded within-epoch shuffle
budget: 3200 updates primary, 4000 common auxiliary cap
checkpoints: [400, 800, 1600, 2400, 3200, 4000]
capacity_gate: both seeds gain >=15 percentage points at 3200 relative to A
c1: same architecture chosen by mean train recall at 3200, paired ranking at 3200 on A and chosen width; if no capacity gate, A only
c1_gain: not_run_no_rd_labels
replay: full validation for A/B/C at 3200, plus rank arms if run; no validation-driven selection
accept: +3 percent each clip, positive P and last4 means both seeds, worst at least -0.30 dB
blind: sealed
EOF
until test -s "$r/audit/audit.json"; do if grep -q Traceback "$r/audit.log"; then exit 1; fi; sleep 3; done
pair() {
 local arm=$1 loss=$2 steps=$3 gpu seed
 local pids=()
 for gpu in 0 1; do
  seed=$((1701+gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$py" "$d/scripts/train_capacity.py" "$r" --arm "$arm" --seed "$seed" --loss "$loss" --steps "$steps" > "$r/$loss-$arm-$seed.log" 2>&1 &
  pids+=("$!")
 done
 local job failed=0
 for job in "${pids[@]}"; do wait "$job" || failed=1; done
 test "$failed" -eq 0
}
pair A hard 4000
pair B hard 4000
pair C hard 4000
best=$("$py" - "$r" <<'PY'
import json,sys
from pathlib import Path
from statistics import mean
r=Path(sys.argv[1])
result={a:[mean(x["recall"] for x in json.loads((r/f"c0-{a}-{s}/run.json").read_text())["snapshots"]["3200"]["train"]) for s in (1701,1702)] for a in "ABC"}
passed=[a for a in "BC" if all(b-c>=.15 for b,c in zip(result[a],result["A"],strict=True))]
best=max(passed,key=lambda a:mean(result[a])) if passed else "A"
(r/"c1_gate.json").write_text(json.dumps(dict(train_recall=result,passed=passed,chosen=best,scope="matched_loss_probe" if passed else "minimal_failure_diagnostic"),indent=2)+"\n")
print(best)
PY
)
pair A rank 3200
if [ "$best" != A ]; then pair "$best" rank 3200; fi
