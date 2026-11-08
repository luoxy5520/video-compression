# P2-InfoProbe research archive

The code and report commit is `f5e02b7126565f3a92b4a21ad6383491c494d603` on `codex/a1-p2-infoprobe-20261108`. Original paths are relative to `~/a1_quality_20261108/P2_InfoProbe` on 113Server.

- Core contains the completed audit, D0, D1 and repaired D0, with native streams, coefficients, input state NPZ, syntax bits, MC traces and logs.
- Probes contains G0–G4 metrics, head weights, emitted states, native streams, 60 G-label caches, eight captured block-14 features and capture manifests. G5 did not start.
- Historical locked A weights, original hidden and backbone feature caches, source YUV and prior artifacts stay on 113Server. Their identities are in the historical manifests and the per-clip manifest in `g_labels`. Reproducible core YUV reconstructions, neutral dummy input, intermediate replay state binaries and partial output from the first audit attempt also stay on 113Server. Decode archived streams for reconstructions and use archived states with the matching code to regenerate replay binaries.

Verify and extract in a separate destination directory.

```sh
sha256sum -c CORE.sha256
sha256sum -c PROBES.sha256
sha256sum -c FILES.sha256
cat core.tar.zst.part-* | zstd -dc | tar -xf -
cat probes.tar.zst.part-* | zstd -dc | tar -xf -
```
