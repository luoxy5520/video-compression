set -euo pipefail
cd ~/dit-p2-multisample-audit-20261108
d=experiments/a1_quality_20261108/P2_MultiSampleAudit
r=~/a1_quality_20261108/P2_MultiSampleAudit
py=~/Wan2.1/.venv/bin/python
cat > "$d/locked.json" <<'EOF'
{"target":"OnPolicy T0","order":["M1","M0","M2_if_training_collapse","M3_if_reliable_fit"],"seeds":[1701,1702],"n":[1,2,4,8,44],"fixed_updates":400,"common_exposures":80,"small_max_exposures":400,"max_updates":3520,"duplicates":[2,4,8],"collapse_recall":0.8,"position_metric":"post-quantization legal P16 top four per frame, 64 positives; video and prompt macro","m2_trigger":"first multi-video scale below 80% train recall in both seeds at its max budget","m2_comparison_updates":400,"blind":"sealed","p3":"not_started"}
EOF
until test -s "$r/m1_unit_tests.json"; do if grep -q Traceback "$r/m1.log"; then exit 1; fi; sleep 4; done
run_pair() {
 local mode=$1 n=$2 steps=$3
 shift 3
 local pids=() gpu seed
 for gpu in 0 1; do
  seed=$((1701+gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$py" "$d/scripts/train.py" "$r" "$r/$mode-$n-$seed" --n "$n" --seed "$seed"     --steps "$steps" "$@" > "$r/$mode-$n-$seed.log" 2>&1 &
  pids+=("$!")
 done
 local job failed=0
 for job in "${pids[@]}"; do wait "$job" || failed=1; done
 test "$failed" -eq 0
}
for n in 1 2 4 8 44; do
 steps=$((400*n)); if ((steps>3520)); then steps=3520; fi
 run_pair scale "$n" "$steps" --checkpoints 400 "$((80*n))" "$steps"
done
for n in 2 4 8; do
 run_pair duplicate "$n" 400 --checkpoints 400 --duplicate
done
