#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-/home/luoxiangyu/Wan2.1/.venv/bin/python}
"$PYTHON" utility_scripts/a0_mode.py --config configs/exp_base.yaml
