# P2-CapacityProbe research archive

CODE_COMMIT records the verified code branch. On 113Server the original run root is ~/a1_quality_20261108/P2_CapacityProbe. Core contains the fresh A audit, C0/C1 checkpoints, histories, per-frame location and exposure evidence, C2 sampled neighbor rows, all validation milestones, state arrays, full per-frame metrics, and native H.264 streams. Input labels contains the exact T0/T1 NPZ inputs reused from OnPolicy. Historical original sources, frozen input features, and recreatable baseline YUV and state intermediates remain on 113Server. SERVER_ONLY.tsv gives absolute paths, sizes, SHA-256 and recovery context.

```sh
sha256sum -c CORE.sha256
sha256sum -c FILES.sha256
cat core.tar.zst.part-* | zstd -dc | tar -xf -
zstd -dc input_labels.tar.zst | tar -xf -
```
