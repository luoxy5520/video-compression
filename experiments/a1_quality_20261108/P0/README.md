# P0 · 固定参考候选回放

已完成基线复现和一个训练视频的固定前缀审计。全量标签、A–F 整段回放、RD 曲线和阶段决策待完成。

| 12 条历史开发回归（3 个 prompt） | 结果 |
|---|---:|
| P 帧 Y-PSNR | 22.838 dB |
| 码率 | 427.707 kb/s |
| Y-SSIM | 0.67053 |
| 末帧 Y-PSNR | 20.327 dB |
| 最后 4 帧减最早 4 个 P 帧 | −7.362 dB |
| 状态头前向 | 2.34 ms |
| 压缩 NPZ 导出 | 163.01 ms |

12 条的七个 NPZ 数组、NPZ 文件、适配状态、码流及重建与历史产物逐字节一致。全部通过三方像素一致性校验。运行时间从缓存特征开始，状态头和导出单列，另有逐条适配器、编码与 FFmpeg 解码计时。历史入口未单独测 JM 解码时长，完整双解码耗时为 NA。

P0 训练 smoke 在 `sample_1000` 第 2 个 P 帧固定前序重建。候选为基线 MV、粗到细 top-4、零 MV，模式固定 P16。配套残差由源减候选 MC 构造。局部 Y SSE 从基线 MV 的 1187.31 降至逐块 min-D 的 916.39，约降 22.8%。这是分别回放整张候选帧后做的局部拼表，使用真实源构造残差。它还没有经过混合候选的真实整段回放或匹配码率检验。

## 复现

在 113Server 的 `~/dit-paired-experiment` 中执行。选择空闲 GPU，并为新运行指定空目录。已有运行在 `~/a1_quality_20261108/P0`。

```bash
p=experiments/a1_quality_20261108/P0
py="$HOME/Wan2.1/.venv/bin/python"
kit="$HOME/a_new48_20261107/kit"
models=/mnt/data_hdd/models/dit_compression/a1_quality_20261108
out="$HOME/a1_quality_p0_reproduction"
mkdir "$out"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

"$py" "$p/baseline.py" --config "$kit/state_config.json" \
  --checkpoint "$models/locked_hidden.pt" --kit "$kit" \
  --jm "$HOME/jm-direct-replay-20261107" \
  --history "$HOME/a_hidden_blind_20261107/eval" --output "$out/baseline"

cp -a "$HOME/jm-a-coeff-20261107" "$out/jm_mb_bits"
"$py" "$p/patch_mb_bits.py" "$out/jm_mb_bits"
make -C "$out/jm_mb_bits/lencod" -j4

"$py" "$p/local_cost.py" --config "$kit/state_config.json" \
  --checkpoint "$models/locked_hidden.pt" --head "$models/frozen_coarse_oracle.pt" \
  --kit "$kit" --jm "$out/jm_mb_bits" --output "$out/train_smoke_bits" --trace-bits
```

原始回放、训练候选标签、日志位于服务器运行目录。权重位于模型盘。GitHub 保存代码、哈希、紧凑指标和复现入口。

## 单训练视频全段回放

```bash
bash experiments/a1_quality_20261108/P0/run_train_pilot.sh \
  "$HOME/a1_quality_20261108/P0/train_pilot_1004_reproduction" sample_1004
```

对平移场景 `sample_1004` 的 16 个 P 帧分别固定 A 组重建前缀，采集六个候选的三平面重建和原生比特计数。随后在任何整段回放之前固定并保存 16 个 P 帧的最终状态。A 保留原 Skip/P16，A_fixed_p16 以及 B–F 固定 P16。B 在历史粗到细 top-4 中取离 Oracle 最近的 MV，并使用原 A 残差。C/D/E 使用基于固定 A 轨迹的源配套残差，F 沿用原 A 残差。D 的 λ 仅在这条训练视频扫描，未用于选验证配置。C 同时是 D 的 λ=0 对照。

整段回放中的重建参考会偏离标签采集时固定的 A 轨迹。此实验测量局部标签跨轨迹使用时的后果。它不构成可随轨迹重新计算 residual 的源可用上界。各组的宏块层及残差比特为原生计数器汇总，码率则使用实际完整码流字节。

| 单训练视频 | P 帧 Y-PSNR | kb/s | 末帧 Y-PSNR |
|---|---:|---:|---:|
| A | 21.10 | 460.25 | 19.76 |
| A_fixed_p16 | 21.10 | 470.51 | 19.75 |
| B | 21.60 | 500.16 | 20.68 |
| C | 15.92 | 5038.81 | 12.87 |
| D，λ=64（D 组最高 PSNR） | 16.26 | 4366.87 | 13.31 |
| E | 14.74 | 5028.65 | 11.73 |
| F，λ=256（F 组最高 PSNR） | 21.50 | 599.68 | 20.42 |

全部 16 组整段码流通过编码器、JM、FFmpeg 三方一致性和零 ME 时间核验。C/D/E 使用的残差在 A 轨迹上配对，一旦连续替换 P 帧便失去配对条件。F 的最高 PSNR 点较 A 高 0.40 dB，但码率增加约 30.3%。结果仅覆盖一个训练 prompt，暂不据此选择 P1 或 P2。完整扫点见 `rd_curve.csv`，逐帧及哈希见 `train_pilot_summary.json`。

## 全量 P0 与推进门槛

在验证结果开放前，负责人已锁定 `config.yaml` 中的门槛。train/val 分别为 44/16 条、11/4 个 prompt。旧开发回归和最终测试不参与该门槛或 λ 的评估。复现用命令如下。

```bash
p=experiments/a1_quality_20261108/P0
py="$HOME/Wan2.1/.venv/bin/python"
out="$HOME/a1_quality_20261108/P0"
"$py" "$p/run_p0_batch.py" --output "$out/full"
bash "$p/run_trajectory_val.sh" "$out/full" "$out/trajectory_val"
"$py" "$p/summarize_p0.py" --run "$out/full" --trajectory "$out/trajectory_val" \
  --output "$out/full_report"
```

轨迹配对诊断保留给定的 mode/MV。只在离线教师阶段使用真实源，记录 JM 原生 MC，再将 Y/U/V 源减 MC 的差值预导出到状态。独立回放关闭 live residual，使用零占位 YUV，码流和重建必须逐字节匹配教师。包含 A 的合法 Skip/P16 混合模式和全 P16 组；Skip 保留派生语义，不生成自由残差。该诊断只用于定位漂移和码率来源。

全量主矩阵完成 60 条视频、960 个固定参考帧、960 条整段回放。额外轨迹配对验证完成 64 条回放。阶段判定见 [decision.md](decision.md)，验证曲线见 [full_report/rd_curve.csv](full_report/rd_curve.csv)。
