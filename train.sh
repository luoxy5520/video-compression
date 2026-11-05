#!/bin/bash
set -euo pipefail
# Dataset generation only; state-head training requires oracle labels.
cd /home/luoxiangyu/dit-paired-experiment
/home/luoxiangyu/Wan2.1/.venv/bin/python utility_scripts/generate_pairs.py exp_base.yaml
