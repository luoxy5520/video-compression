#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-/home/luoxiangyu/Wan2.1/.venv/bin/python}
"$PYTHON" train_mode.py --config config_train32.json --evaluate-only
