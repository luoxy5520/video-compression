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
