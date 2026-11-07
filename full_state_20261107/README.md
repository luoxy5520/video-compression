# 完整预测状态回放 pilot

沿用扩容批次 20/4/8 条视频的 prompt 分组。模式头复用 15,042 参数的 A2 stage_4 权重。MV/残差头使用独立的 28,834 参数 Conv3D 主干，以模式头的卷积和时间嵌入初始化，分别输出 2 个 qpel 位移分量和每宏块 384 个像素残差值（Y 16×16、U/V 各 8×8）。残差分支接收预测 MV。MV 和残差只在 Oracle P16 宏块上计算 Smooth L1，训练 12 轮，以验证集两项损失之和选择权重。推理中不输入真实 P 帧标签。

## 服务器运行

在 `/home/luoxiangyu/dit-paired-experiment` 执行：

```bash
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
PYTHON=/home/luoxiangyu/Wan2.1/.venv/bin/python
RUN=/mnt/data_hdd/luoxiangyu/dit_pairs/full_state_pilot_20261107
"$PYTHON" full_state_20261107/train_state.py \
  --config expansion_20261106/config_train32.json \
  --mode expansion_20261106/results_expand32/stage_4.pt --output "$RUN"
"$PYTHON" full_state_20261107/benchmark.py \
  --config expansion_20261106/config_train32.json \
  --mode expansion_20261106/results_expand32/stage_4.pt --regressor "$RUN/regressor.pt" \
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
  --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106 \
  --output "$RUN/benchmark"
```

benchmark 输出目录需为新目录。每条测试视频实跑一次 VAE + JM 常规编码，另外从 step 5/10/15/20 各预测一次完整 P 帧状态。适配器保留 Oracle I 帧和固定结构，替换全部 P 帧模式、运动矢量和残差。两路采用相同 QP 28、分辨率、帧率和受限 JM 配置，均用 JM 和 FFmpeg 双解码确认像素一致。

码率包含 I 帧及码流头，按总字节数和 17/16 秒时长计算。质量以 step 20 的最终 raw 为参考，统计 16 个 P 帧的亮度和全 YUV PSNR。各视频的 dB 等权平均。耗时在预热后按视频串行采集，包含 latent 文件读入、模型/VAE、数据序列化、适配器和编码子进程。另计 FFmpeg 解码耗时。范围从 latent 已有至码流/解码输出，DiT 采样、权重初始化和 Oracle I 帧准备成本需单独测量。两路质量和码率不同，因此速度比是本配置的路径比较。

`benchmark/results.json` 包含各视频、各输入步的实测结果。各子目录保留实际预测 NPZ、码流、标准解码输出和日志。
