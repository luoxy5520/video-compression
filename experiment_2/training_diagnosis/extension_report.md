# Extended temporal codec training results

Generated 2026-11-28T14:05:17.521141+08:00; execution complete, figure/text review PENDING.

All18 historical development videos; independent generalization NOT VERIFIED. Sealed test NOT OPENED. Original DiT/VAE frozen. Previous684-row experiment preserved;90 new actual-stream measurements appended in a separate comparison directory.

| Method | Videos | Mean Bytes | Mean PSNR-Y | Uniform matched-rate delta/dB | Comparison videos |
|---|---:|---:|---:|---:|---:|
| D42_f1_seed20261127_20000 | 18 | 305865.9 | 38.7252 | 0.8883 | 18 |
| D42_temporal5000_TP0 | 18 | 304897.9 | 38.7252 | 0.9407 | 18 |
| D42_temporal5000_TP2 | 18 | 291785.7 | 38.7252 | 1.6386 | 18 |
| D42_temporal5000_TP2Z | 18 | 306179.8 | 38.7252 | 0.8711 | 18 |
| P2 | 18 | 286282.6 | 38.4196 | 1.5817 | 18 |
| P2_long10000_P2 | 18 | 278098.1 | 38.5766 | 2.1842 | 18 |
| P3 | 18 | 287604.1 | 38.4857 | 1.5762 | 18 |
| P3_long1000_P3 | 18 | 292381.9 | 38.5706 | 1.3737 | 18 |

D42 frozen-symbol TP2 mean byte saving versus normalized TP0: 4.2687%; independent history contribution versus same-capacity zero-history head: 4.5569%. All18 reconstructions and canonical symbols exactly match their new D42 reference.

P2/P3 continuations retain optimizer, scheduler, RNG and full-video sampling position. D42 temporal heads use a different frozen base and training budget from original D16/10000 temporal heads, so that cross-family difference is not a single-factor history ablation.

Matched-rate and matched-quality comparisons interpolate log(Bytes) only within each measured Pareto support; coverage can differ. Bytes and PSNR changes together do not establish equal-rate success. H.264 comparison CSVs retain unfavorable cases. LPIPS/DISTS, BD-rate, P4, DiT joint training and independent validation NOT RUN.

Fresh GPU-start deployment timing (one warmup +five formal) is under results/performance/{P2_long10000,P3_long1000,D42_temporal5000}. Codec encoder never invokes VAE; both receiver endpoints are rawYUV420. Encoder speedup is not a two-end speedup.
