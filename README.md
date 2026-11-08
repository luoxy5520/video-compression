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

The first DC seed originally logged pooled last-four Y-PSNR. The tables at this
branch root use the P0.6 arithmetic frame mean for every run. After verifying the
original archive, apply the correction and verify its files.

```sh
sha256sum -c CORRECTION_ARCHIVE.sha256
zstd -dc dc1701-tail-correction.tar.zst | tar xf -
sha256sum -c CORRECTED_TAIL.sha256
```

The correction covers metrics only. It preserves the original bitstreams, models,
and per-frame scores.
