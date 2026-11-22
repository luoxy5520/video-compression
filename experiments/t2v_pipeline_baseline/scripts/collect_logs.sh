#!/usr/bin/env bash
# Send run metadata and text logs as a tar stream; exclude all video payloads.
set -euo pipefail
cd "${1:?Path to the 113Server run directory is required}"
find . -type f \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' \
    -o -name '*.cfg' -o -name 'gpu-monitor*.csv' -o -name 'run.sh' -o -name 'recover.sh' -o -name '*COMPLETE' \) \
    -not -path './scripts/*' -not -path './configs/*' -print0 |
  tar --null --files-from=- -cf -
