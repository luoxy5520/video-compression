# D42 joint RD continuation

{
  "method": "P2_D42_joint10000_P2",
  "videos": 18,
  "mean_actual_bytes": 271570.22222222225,
  "mean_psnr_y": 38.89214576373927,
  "mean_psnr_rgb": 37.792898563235845,
  "mean_SSIM_RGB": 0.9712460736433665,
  "mean_last4_psnr_y": 39.54756102150299,
  "worst_psnr_y": 30.880935743665088
}

New joint phase from frozenD42/20000 plusTP2/5000,10000 joint updates,42 legal training videos. Optimizer initialized anew, not an exact continuation of D16 or D42 latent-only optimizers. All18 historical development videos actually encoded/decoded and native-VAE evaluated. Full baseline comparison at results/d42_joint_comparison; GPU-start timing at results/performance/P2_D42_joint10000. Figure/text review PENDING. Independent validation and sealed test NOT RUN. Original DiT/VAE frozen.
