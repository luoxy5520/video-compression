# D42 joint RD: three matched-source working points

Same frozen starting D42/20000 +TP2/5000 weights;42 legal training videos;seed20261127;10000 joint updates each;lambda factors0.1/1/10;new optimizers for all three phases. No original Wan parameters trained.

| Method | Videos | Mean Bytes | Mean PSNR-Y | Uniform matched-rate delta/dB | Matched videos |
|---|---:|---:|---:|---:|---:|
| P2_D42_high10000_P2 | 18 | 286055.3 | 38.9675 | 2.112725967557844 | 18 |
| P2_D42_joint10000_P2 | 18 | 271570.2 | 38.8921 | 2.870637167390415 | 18 |
| P2_D42_low10000_P2 | 18 | 248807.8 | 38.0498 | 3.1544196323854528 | 18 |

All three models independently stream-decoded on first video and actually stream-decoded/nativeVAE evaluated on all18 historical development videos. Symbols may change; these are lossy joint models, not fixed-symbol savings. Adverse or nonmonotonic points retained. Full per-video metrics, main/hyper/header rates, probability diagnostics and measured-support-only Uniform/H264 comparisons are under results/joint_sweep_comparison. No extrapolation or validated BD-rate. Independent validation and sealed test NOT RUN. Figure/text review PENDING. Formal GPU-start profiles use one warmup +five repetitions under the GPU0 experiment lock; encoder speedup is not total two-end speedup.
