#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py="$HOME/Wan2.1/.venv/bin/python"
mkdir -p results-validation
"$py" -c 'import json; print(*json.load(open("frozen.json"))["validation"],sep="\n")' |
  xargs -P 4 -n 1 bash -c '
    set -eu
    s=$1
    "$HOME/Wan2.1/.venv/bin/python" trace_run.py "$s" >"results-validation/$s.trace"
    "$HOME/Wan2.1/.venv/bin/python" probe.py val "$s" >"results-validation/$s.json"
  ' _
