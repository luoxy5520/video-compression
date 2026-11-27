# E2.1c single-video learning

PASS on one original legal train video sample_0001, not generalization. The R3 cancellation-of-downsample adaptation uses analytical identity initialization (same conv/GELU shapes), then real quantization and gradient training. This is a new experiment, not historical I3 checkpoint continuation.

O1 fixed step0.1, bounded coding range±255, STE; 1000 steps MSE0.00051822,377623 actual Bytes. Learned symbols max absolute magnitude stayed below the configured bound. No unbounded rescaling argument is used as compression evidence.

O2 initial R/D reference6910.10; initial rate gradient norm4.58562, distortion norm0.00229094 (gradient-balance reference~2001.6). Tested factors0.1/0.3/1/3/10 with same initialization, train batch and scale-head calibration. Learned scale-head initialization is in shared checkpoint parameters, never decoder-only untransmitted data.

Actual points: lambda691→222175Bytes/44.861dB;2073→267875/47.024;6910→305811/47.895;20730→328391/48.024;69101→347315/48.128. Latent MSE decreases overall; complete bytes increase. First slice of each checkpoint independently decoded, all slices hard-forward consistent. These measure only training-video fit and do not establish a development-set win.

Historical1000-step sweep did not swap loss terms. Original downsampled transform reconstruction floor, short/condition-sensitive optimization, training noise versus inference rounding and the mismatch between latent and pixel losses remain different measured factors. They do not justify claiming a unique causal explanation from lambda versus video PSNR alone.
