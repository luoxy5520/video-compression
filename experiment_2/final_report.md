# E2 resumed experiment report

The SSH interruption has been resolved. I2/I3 smoke and repeat passed. Eight short intra runs and fixed-symbol I1 comparison completed. This is a development-stage report, not a final independent evaluation.

## Fixed-symbol I1 versus I0

16 original train videos; two separate historical development groups. Identical quantized symbols and bit-exact latent reconstruction. One independent-process decode per bitwidth passed. Full files include metadata; shared weights reported separately.

| Bits | Mean I0 Bytes | Mean I1 Bytes | I1 byte change |
|---|---:|---:|---:|
| 4 | 137351.0 | 201071.5 | +46.39% |
| 5 | 264995.5 | 268972.0 | +1.50% |
| 6 | 338672.5 | 334289.0 | -1.29% |
| 8 | 457912.5 | 461315.0 | +0.74% |

I1 uses a simple learned per-channel zero-mean Gaussian, not a contextual model. INT6 savings are small; other tested widths regress. Spatial/temporal dependencies retained by Zstd are not represented by this independent Gaussian.

## Short intra runs

16 train videos, two group-disjoint historical development videos, 1000 steps per model, four weights per architecture, separate entropy auxiliary optimizer. Native frozen Wan VAE decoded recovered latents. No VAE at encoder.

| Video | Model | Lambda | Actual Bytes | PSNR-Y | PSNR-RGB |
|---|---|---:|---:|---:|---:|
| sample_0000 | I0_INT2 | — | 22597 | 16.849 | 14.901 |
| sample_0000 | I0_INT3 | — | 72775 | 30.109 | 27.327 |
| sample_0000 | I0_INT4 | — | 122698 | 36.767 | 34.655 |
| sample_0000 | I0_INT5 | — | 242979 | 41.375 | 39.693 |
| sample_0000 | I0_INT6 | — | 329484 | 44.889 | 43.575 |
| sample_0000 | I0_INT7 | — | 429417 | 48.894 | 47.827 |
| sample_0000 | I0_INT8 | — | 466513 | 52.812 | 51.931 |
| sample_0000 | I2 | 16 | 28488 | 25.131 | 22.765 |
| sample_0020 | I0_INT2 | — | 25028 | 15.412 | 13.338 |
| sample_0020 | I0_INT3 | — | 98217 | 19.981 | 17.886 |
| sample_0020 | I0_INT4 | — | 152004 | 24.385 | 22.850 |
| sample_0020 | I0_INT5 | — | 287012 | 28.907 | 27.631 |
| sample_0020 | I0_INT6 | — | 347861 | 33.728 | 32.612 |
| sample_0020 | I0_INT7 | — | 435441 | 39.019 | 38.008 |
| sample_0020 | I0_INT8 | — | 449312 | 44.554 | 43.647 |
| sample_0020 | I2 | 16 | 26292 | 16.872 | 15.284 |
| sample_0000 | I2 | 128 | 51851 | 25.084 | 22.965 |
| sample_0020 | I2 | 128 | 49279 | 15.983 | 14.573 |
| sample_0000 | I2 | 1024 | 58556 | 25.421 | 23.159 |
| sample_0020 | I2 | 1024 | 50815 | 16.901 | 15.299 |
| sample_0000 | I2 | 8192 | 50551 | 22.372 | 20.626 |
| sample_0020 | I2 | 8192 | 53048 | 16.679 | 15.148 |
| sample_0000 | I3 | 16 | 21001 | 25.454 | 23.102 |
| sample_0020 | I3 | 16 | 19320 | 16.753 | 15.169 |
| sample_0000 | I3 | 128 | 44195 | 25.087 | 22.747 |
| sample_0020 | I3 | 128 | 43731 | 17.003 | 15.450 |
| sample_0000 | I3 | 1024 | 53246 | 24.568 | 22.617 |
| sample_0020 | I3 | 1024 | 49257 | 16.439 | 14.861 |
| sample_0000 | I3 | 8192 | 53018 | 23.346 | 21.845 |
| sample_0020 | I3 | 8192 | 53534 | 16.499 | 14.693 |

Curves span substantially different rates. `development_comparison.json` contains per-video log-rate interpolation only within measured Uniform support; these sparse interpolations do not establish the preset matched-rate gain threshold. No extrapolation is used. Four rate weights need tuning: these short runs do not produce a monotonic RD curve.

I2/I3 real-byte main/hyper/metadata breakdown, resource use, model sizes and SHA256 are in intra_sweep.json; full training curves are retained. At least one slice per checkpoint was decoded in an independent process for the first development video. All slices were decoded from actual files and compared to forward reconstruction during training evaluation.

## Gates and remaining work

- Real-stream gate: PASS for I2/I3, repeated; odd-size/checkpoint/corruption regression PASS.
- Independent validation: NOT RUN; original validation remains development; sealed test unopened.
- I4 and P0–P4 temporal experiments: NOT RUN. Need a selected, converged intra operating point and finer matched-rate evaluation before increasing architecture complexity.
- Pixel-domain training: NOT RUN; VAE is evaluation-only in this round.
- H.264 matched-quality comparison and preset 0.5dB/10% target: NOT ESTABLISHED. Historical mean is not used as a per-video comparator.
- Formal five-repeat timings, acceleration, FLOPs, RSS, SSIM/LPIPS/DISTS: NOT RUN. VAE diagnostic times in smoke are not formal latency.
- Full GLVC training-strategy reading and ELIC source verification remain incomplete due source access failures, recorded in literature_mapping.md.

No final deployment architecture is selected. No generator, original VAE weights, denoising steps, E0/E1 code or sealed test data were changed.
