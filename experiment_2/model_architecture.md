# Minimal Intra prototype

I2: 16→32 3x3 conv, GELU, 32→32 stride-2 3x3 analysis; learned per-channel EntropyBottleneck; transpose-conv 32→32 (4x4 stride 2), GELU, conv 32→16 synthesis.
I3: same transforms; hyper analysis two stride-2 3x3 convs with GELU; entropy bottleneck on hyperlatents; two transpose convs predict positive zero-mean Gaussian scales, cropped to coding shape. GaussianConditional encodes main symbols.

60x104→30x52→hyper 8x13; scale synthesis 32x52 cropped to30x52. Synthesis crops to original size (also supports odd spatial sizes). No untransmitted input statistics, temporal conditions, RGB modules, pretrained codec weights, VAE calls or H.264 in encoder.

Smoke: one existing train/development video, all five time slices, seed 20261127, 100 Adam steps, lr .001, lambda_z 16. Rate is bits per input latent element; loss = rate + 16 MSE + .01 entropy auxiliary quantile loss. One optimizer is sufficient for this functional smoke; production training should separate main/quantile optimizers and tune lambda on development data. Not a rate-distortion sweep.

Short sweep additionally uses16 original training videos and2 disjoint historical development groups, 1000 steps, lambda16/128/1024/8192, main Adam lr.001, auxiliary Adam lr.001, clip main gradient norm1. No new independent holdout. I1 uses16 positive learned channel Gaussian scales and no transform; 200 Adam steps atlr.03 for each INT4/5/6/8, preserving E1 quantization exactly.
