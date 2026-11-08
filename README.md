# P2-OnPolicy research archive

The matching code SHA is in CODE_COMMIT. Original run paths are relative to `~/a1_quality_20261108/P2_OnPolicy` on 113Server.

- Core includes the baseline audit and complete O0 paired labels, metrics, native streams, coefficient traces and run logs.
- Students includes the eight fixed-sample and four full O1 runs, training histories, head weights, pre-exported states, actual JM bitstreams, full frame and reference evidence, and diagnostic checks. O2 and P3 did not start.
- The historical sources and frozen feature caches, and reproducible audit YUV and intermediate state files stay on 113Server. SERVER_ONLY.tsv gives sizes, hashes and recovery context. Decode the archived H.264 to recover reconstructions. Reproduce the packed audit state with the repository audit command and frozen dependencies.

Verify and extract into an empty directory.

```sh
sha256sum -c CORE.sha256
sha256sum -c STUDENTS.sha256
sha256sum -c FILES.sha256
cat core.tar.zst.part-* | zstd -dc | tar -xf -
cat students.tar.zst.part-* | zstd -dc | tar -xf -
```
