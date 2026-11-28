# Frozen-VAE-aware42-video joint fine tuning

[
  {
    "method": "P2_D42_joint10000_P2",
    "videos": 18,
    "mean_actual_bytes": 271570.22222222225,
    "mean_psnr_y": 38.89214576373927,
    "mean_psnr_rgb": 37.792898563235845,
    "mean_SSIM_RGB": 0.9712460736433665,
    "mean_last4_psnr_y": 39.54756102150299,
    "worst_psnr_y": 30.880935743665088
  },
  {
    "method": "P3_D42_1000_P3",
    "videos": 18,
    "mean_actual_bytes": 271028.44444444444,
    "mean_psnr_y": 39.08368160027125,
    "mean_psnr_rgb": 37.985681059835244,
    "mean_SSIM_RGB": 0.972106913725535,
    "mean_last4_psnr_y": 39.722278367986206,
    "worst_psnr_y": 31.198573009206058
  }
]

P3_D42 starts from the actual measured P2_D42_joint10000, trains1000 full-sequence steps over the same42 legal train videos, with new optimizer and pixel-gradient calibration. Source codec and original DiT/VAE weights preserved. All18 historical development videos actually stream decoded and nativeVAE evaluated. Full measured-support Uniform/H264 comparisons at results/d42_pixel_comparison; fresh GPU-start profiling at results/performance/P3_D42_1000. No original model training or sealed test access. Independent validation and perceptual-model metrics NOT RUN. Figure/text review PENDING.
