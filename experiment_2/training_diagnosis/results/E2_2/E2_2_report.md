# E2.2 fixed-symbol temporal entropy report

Base: adapted I3 R3-identity initialization, D16 middle point10000 steps. Base transforms, hyperprior and quantization frozen. Only two-layer context probability head trained,2000steps on16 legal train videos. Evaluation:2 historical development videos; no independent generalization claim.

| Mode | Mean complete Bytes | Mean saving vs TP0 |
|---|---:|---:|
| TP0 | 306047.0 | 0.00% |
| TP1 | 289581.0 | 5.40% |
| TP2 | 284153.0 | 7.20% |

All canonical integer symbols q and reconstructed latents match TP0 exactly, with independent decoder process verification for every video/model. Integer mean k shifts CDF origin: coding internal q-k then adding k preserves q exactly; no floating-mean quantization change. Hyperlatents are decoded before probability estimation; history comes only from reconstructed previous Z (TP1) or reconstructed coding q (TP2). No future frames or untransmitted source statistics.

TP0/TP1/TP2 use the same new TPE1 video container. TP0 payload is the existing independent hyperprior; its metadata differs from five legacy LVC2 files, so savings are attributed only against this same-container TP0. All model weights are pre-shared; sizes/hashes reported separately. Actual encoder/decoder CDF computation is CPU float32 on this host, tested exactly; portable cross-device CDF consistency is NOT VERIFIED.

First-slice main/hyper payloads are unchanged; gains arise in later slices. See temporal_rate_per_slice.csv and temporal_prior_bitstream_breakdown.csv. CPU encode/decode/context/probability/rANS timing uses1warmup+5formal per video. These are CPU-path measurements, not GPU Latent-to-bitstream or quality-matched H.264 acceleration.

Fixed-symbol target5% average saving is reached on these two development videos. A small P2 joint latent RD run is a separate lossy experiment:2 slices for500steps then5 slices, target1500 steps; its metrics must be measured afresh. P3 pixel-aware joint training and P4 contextual transform are NOT RUN at this snapshot.
