# P2-LabelProbe research archive

The exact code and report commit is recorded in CODE_COMMIT. The original run root is `~/a1_quality_20261108/P2_LabelProbe` on 113Server.

- Core contains the fresh A audit, fixed-sample and multi-prompt L0 runs, source-assisted native label extracts, checkpoint weights, bitstreams, per-frame scores, syntax bits, and both initial retry logs.
- Labels and probes contains L1 reconstructed-reference label regeneration, full paired metrics, source-assisted replay, six duplicate-generation checks, L2 predictions and models, and the source-free export checks.
- The original Wan hidden, frozen feature caches, source YUV, historical checkpoint and source artifacts remain on 113Server. SERVER_ONLY.tsv identifies relevant omitted originals by size and SHA-256. Reproducible audit reconstruction YUV, neutral dummy YUV and packed state binaries also remain there. Dense L1 coefficient tensors were computed in memory and can be regenerated. Python plotting packages can be reinstalled from the versioned install log.

Extract into an empty directory.

```sh
sha256sum -c CORE.sha256
sha256sum -c LABELS_PROBES.sha256
sha256sum -c FILES.sha256
cat core.tar.zst.part-* | zstd -dc | tar -xf -
cat labels-probes.tar.zst.part-* | zstd -dc | tar -xf -
```
