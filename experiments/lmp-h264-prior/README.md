# Reproduce the LMP experiments on 113Server

Use a fresh writable output directory containing these scripts. The scripts use the documented historical JM kits, cached predictions and Wan virtual environment on 113Server. Input paths and hashes appear in `audit.jsonl`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py="$HOME/Wan2.1/.venv/bin/python"
dev=(sample_0000 sample_0004 sample_0008 sample_0012 sample_0016 sample_1000 sample_1004 sample_1008)

"$py" audit.py > audit.jsonl
"$py" baseline.py sample_0000 sample_1004 > baseline.log
"$py" trace_build.py
(cd jm-trace/lencod && make -j4 CC='gcc -fcommon')
"$py" trace_run.py "${dev[@]}" > trace-dev.jsonl
"$py" probe.py train "${dev[@]}" > probe-exact.jsonl
"$py" mode_probe.py train val > mode.jsonl
bash validate.sh
"$py" summarize.py --by-video probe-exact.jsonl
"$py" summarize.py results-validation/*.json
```

The trace run verifies original H.264 bitstream hashes. Keep the baseline timing run isolated from the trace and probe jobs. Validation diagnostics may run concurrently. Preserve the original baseline and probe outputs before rerunning in an existing directory.
