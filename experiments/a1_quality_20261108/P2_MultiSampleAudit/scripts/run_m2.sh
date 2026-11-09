set -euo pipefail
cd ~/dit-p2-multisample-audit-20261108
d=experiments/a1_quality_20261108/P2_MultiSampleAudit
r=~/a1_quality_20261108/P2_MultiSampleAudit
py=~/Wan2.1/.venv/bin/python
until [ "$(find "$r" -maxdepth 2 -path "*/scale-*/run.json" | wc -l)" -eq 10 ] && [ "$(find "$r" -maxdepth 2 -path "*/duplicate-*/run.json" | wc -l)" -eq 6 ]; do
 if grep -l Traceback "$r"/scale-*.log "$r"/duplicate-*.log 2>/dev/null | grep -q .; then exit 1; fi
 sleep 8
done
n=$("$py" "$d/scripts/m2.py" "$r")
if [ "$n" -eq 0 ]; then exit 0; fi
pids=()
for gpu in 0 1; do (
 seed=$((1701+gpu))
 CUDA_VISIBLE_DEVICES=$gpu "$py" "$d/scripts/m2.py" "$r" --seed "$seed"
 for policy in bce mixed candidate_weighted blocked; do
   CUDA_VISIBLE_DEVICES=$gpu "$py" "$d/scripts/train.py" "$r" "$r/m2-$policy-$n-$seed" --n "$n" --seed "$seed"        --policy "$policy" --steps 400 --checkpoints 400 > "$r/m2-$policy-$n-$seed.log" 2>&1
 done
) & pids+=("$!")
done
for job in "${pids[@]}"; do wait "$job"; done
