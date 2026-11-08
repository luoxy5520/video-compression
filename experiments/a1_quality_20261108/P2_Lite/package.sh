#!/usr/bin/env bash
# Package this run on 113Server. Keep regenerable frozen feature caches on the server.
set -euo pipefail
run_root="${1:?P2_Lite server directory required}"
checkout="${2:?new artifacts staging directory required}"
source_dir="$(cd "$(dirname "$0")" && pwd)"
mkdir "$checkout"
checkout="$(cd "$checkout" && pwd)"
cd "$run_root"
find run01 -type f -print0 | sort -z | xargs -0 sha256sum > "$checkout/FILES.sha256"
find run01/features -type f -print0 | sort -z | xargs -0 sha256sum > "$checkout/SERVER_ONLY.sha256"
find . -maxdepth 1 -name '*.log' -type f -print0 | sort -z | xargs -0 sha256sum > "$checkout/LOGS.sha256"
cp run01/locked_config.json run01/label_manifest.json "$checkout/"
cp run01/report/{summary.json,rd.csv,rd.svg} "$checkout/"
cp -a "$source_dir" "$checkout/reproduction"
find "$checkout/reproduction" -name __pycache__ -type d -prune -exec rm -r {} +
archive="$checkout/p2-lite-run01.tar.zst.part-"
tar --exclude='run01/features' -cf - run01 ./*.log | zstd -T4 -3 -c | split -b 48m -d -a 3 - "$archive"
(
    cd "$checkout"
    sha256sum p2-lite-run01.tar.zst.part-* > ARCHIVE_PARTS.sha256
)
cat > "$checkout/README.md" <<'EOF'
# P2-Lite run01 research artifacts

Validation runs, checkpoints, training histories, label cache, actual JM bitstreams,
coefficient traces, bit counts, score files, diagnostic controls, and review MP4s
are in the split archive. Inputs are identified by the label manifest. Frozen
feature projections are regenerable and remain on 113Server. Their complete
checksums are in SERVER_ONLY.sha256. FILES.sha256 covers the full server run.

Verify the parts and unpack on a research machine with zstd installed.

```sh
sha256sum -c ARCHIVE_PARTS.sha256
cat p2-lite-run01.tar.zst.part-* | zstd -dc | tar xf -
sha256sum -c LOGS.sha256
grep -v '  run01/features/' FILES.sha256 | sha256sum -c -
```

The display MP4s are re-encoded composites. Actual scoring streams are stored
under each evaluated policy and clip as prediction.264.
EOF
