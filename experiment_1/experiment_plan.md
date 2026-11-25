# Experiment 1：DiT Final Latent 冗余分析与率失真优化

## 1. 研究背景

Experiment 0 已经验证：

- 可以直接从 Wan DiT Final Latent 构建真实的二进制压缩码流 LVC1。
- 编码端不需要调用 VAE Decoder，不需要生成完整 RGB/YUV 原帧。
- 接收端可以通过独立解码进程恢复 Latent，再调用原生 Wan VAE 恢复视频。
- 不需要训练额外模型，也不需要修改 DiT 或 VAE 权重。
- 两条开发视频中，INT4 相对于 VAE + H.264 Medium QP28 的 Post-DiT 编码阶段获得约73～77倍局部加速，但并非同画质比较。

当前主要问题：

1. 均匀低位宽量化存在明显的率失真瓶颈。
2. INT4 与 INT8 之间的精度区间尚未充分探索。
3. 不同 Latent 通道的重要性可能存在差异。
4. 简单时序差分未有效降低实际码流大小。
5. Latent 内部的时空冗余尚未被系统分析。

本阶段目标是解决上述问题，为后续 Latent 运动补偿、Skip 编码，以及 Compression-Aware DiT 提供实验依据。

---

## 2. 全局实验约束

### 2.1 实验环境

优先复用 Experiment 0 在113Server上的已有环境：

- GPU：A100 80GB PCIe。
- Wan2.1-T2V-1.3B。
- 原始 Wan VAE 权重。
- 已有 Python 虚拟环境。
- 已有 LVC1 Codec 实现。
- 已有 Final Latent、原始参考视频及 H.264 基线。
- 已有实验脚本和数据集划分。

首先定位 Experiment 0 实验目录，阅读已有 final_report.md、codec_format.md、reference.json、benchmark.json、validation_rows.json 及相关源代码。

不得假设文件路径、数据内容或实验状态。如果实际环境与报告不同，记录差异后再执行。

### 2.2 数据集

继续使用 Experiment 0 的：

- 两条开发样本：sample_0000、sample_1004。
- 原有16条开发验证样本。
- 原有封存测试集不得访问。

如果需要额外样本选择量化策略，优先使用此前未纳入验证集的合法开发或训练样本。

所有参数搜索、通道排序规则和自适应策略必须在开发集上确定，然后冻结，在16条验证样本上执行。

禁止根据验证结果反复调整策略后仍将其称为独立验证。

### 2.3 模型约束

本阶段全部采用无训练方法：

- 不重新训练 DiT。
- 不修改 VAE 权重。
- 不训练 Mode/MV/Residual Head。
- 不修改去噪步数。
- 不执行 Token Pruning。
- 不执行 DiT 内部计算优化。
- 不要求输出标准 H.264 码流。
- 编码路径不得调用 VAE Decoder。
- 不使用原始 RGB/YUV 作为实际编码器的输入。

允许在离线分析、画质评价及 Oracle 上界实验中调用 VAE Decoder，但必须与实际编码路径严格分离。

### 2.4 统一评价指标

所有可运行 Codec 配置需要报告：

**压缩性能：**

- 完整二进制码流大小，Bytes。
- Bitrate，kbps。
- 相对于原始 FP32 Latent 的压缩比。
- Header、量化参数、辅助信息及 Payload 的大小分解。

**视频质量：**

- PSNR-Y。
- PSNR-RGB。
- SSIM。
- LPIPS（环境允许时）。
- 逐帧 PSNR。
- 末四帧 PSNR。
- 最差视频与最差帧画质。

参考视频统一为原始 Final Latent 经原生 VAE Decoder 得到的视频。

不得使用 H.264 重建视频作为唯一质量参考。

**性能：**

- Quantization latency。
- Bit-packing latency。
- Entropy coding latency。
- 完整 Latent-to-bitstream latency。
- Bitstream-to-latent latency。
- VAE reconstruction latency。
- Post-DiT 两端处理时间。

代表配置采用一次预热、至少五次正式测量，报告 Mean、Median、Std、Min、Max。

GPU 操作需进行正确的同步计时。

所有实际码率必须来自真实生成的二进制文件，而不是理论位宽、浮点张量大小或符号数量估计。

---

# 3. E1.0：完整量化位宽扫描

## 3.1 实验目的

补齐 Experiment 0 中 INT4 与 INT8 之间的率失真曲线，回答：

- INT5、INT6、INT7 是否已经能够获得接近 H.264 的率失真性能？
- Latent 画质在哪个位宽区间发生明显变化？
- 当前率失真问题是否仅由不合理的位宽选择造成？
- 后续混合精度优化应该在哪个位宽区间开展？

## 3.2 实验变量

固定：

- 原始 Final Latent。
- VAE Decoder。
- 量化算法。
- Zstd 参数。
- 评价脚本。
- 数据集。
- 计时口径。

仅改变实际量化位宽。

测试：

| ID | 方法 |
|---|---|
| Q0 | FP32 无损参考 |
| Q2 | INT2 |
| Q3 | INT3 |
| Q4 | INT4 |
| Q5 | INT5 |
| Q6 | INT6 |
| Q7 | INT7 |
| Q8 | INT8 |

必须实现 INT3、INT5、INT6、INT7 的真实位打包。

不得将 INT5、INT6 等直接存储为 INT8 后，仍然按照理论位宽报告实际码率。

## 3.3 执行步骤

1. 复现 Experiment 0 的 INT2、INT4、INT8 基线。
2. 核验原始 Latent 和参考视频的哈希。
3. 扩展 LVC1 Codec，支持任意2～8 bit整数打包。
4. 对每个新增位宽执行独立编解码测试。
5. 检查符号范围、padding、负数映射和码流长度。
6. 在两个开发样本上执行全部配置。
7. 通过独立进程解码真实码流，恢复 Latent。
8. 使用原生 VAE Decoder 恢复视频。
9. 在16条验证样本上执行固定配置。
10. 绘制完整率失真曲线。
11. 与已有 H.264 Medium/Fast 多 QP 曲线进行比较。

基线复现失败时，暂停后续实验，先定位差异。

## 3.4 可选子实验：量化范围

在主实验完成后，使用开发样本比较：

- Per-tensor symmetric quantization。
- Per-channel symmetric quantization。
- Per-channel percentile clipping。

Percentile 可先测试99%、99.5%、99.9%及不裁剪。

所有量化尺度和裁剪所需参数必须计入实际码流。

如果已有 Codec 使用其中某种方式，应首先保留原配置，再增加消融对照。

## 3.5 输出结果

- uniform_quantization.csv
- quantization_range_ablation.csv
- rd_curve_per_video.png
- rd_curve_summary.png
- bitwidth_vs_psnr.png
- bitwidth_vs_latency.png
- bitstream_breakdown.png
- E1_0_report.md

## 3.6 阶段结论

明确回答：

- 哪个位宽的率失真表现最好？
- INT5/INT6 能否填补 INT4 与 INT8 之间的性能差距？
- 哪些视频对量化更加敏感？
- 在相近质量下，与 H.264 的码率差距是多少？

如果有效率失真曲线重叠，报告逐视频插值比较。

仅在满足有效重叠区间及实现正确性要求时计算 BD-Rate，不允许外推。

---

# 4. E1.1：Latent 通道量化敏感性分析

## 4.1 实验目的

验证不同 Latent Channel 对最终视频质量的影响是否存在明显差异，以及这种差异能否跨视频泛化。

Wan Final Latent 的典型结构：

[C, T, H, W] = [16, 5, 60, 104]

实际维度以读取到的数据为准。

核心假设：并非所有 Latent Channel 都需要相同的编码精度。

## 4.2 实验 A：单通道降精度扰动

首先构造全部通道均为 INT8 的参考配置。

然后，每次只降低一个通道的量化位宽，其他15个通道保持 INT8。

分别测试：

- INT8 → INT6。
- INT8 → INT4。
- INT8 → INT2。

每个视频共计：

- 1个 INT8 基线配置。
- 16 × 3 = 48个单通道扰动配置。

总计49个配置。

每次扰动后执行 VAE 重建，测量实际视频画质变化。

定义通道敏感度：

S(c,b) = D(reconstructed_video(c,b), reference_video) - D(INT8_video, reference_video)

其中 D 优先采用像素域 MSE。

同时报告：

- PSNR下降量。
- SSIM下降量。
- LPIPS增加量。
- 实际码流大小变化。
- 单位节省比特带来的失真增量。

单通道扰动可能存在非线性相互作用，不得将所有单通道失真简单相加视为联合量化的真实失真。

## 4.3 实验 B：通道重要性统计

对每个通道统计：

- Mean。
- Variance。
- Standard deviation。
- L1 norm。
- L2 energy。
- 最大绝对值。
- 分位数。
- 量化误差。
- 量化符号熵。
- 时间相关性。
- 空间相关性。

分析上述指标与实际像素域失真增量之间的关系。

报告 Pearson 和 Spearman correlation。

重点验证：

- 高能量通道是否一定更重要？
- Latent 量化误差是否能够预测最终视频画质损失？
- 是否存在低能量但高敏感度的通道？
- 哪些统计量可以在不调用 VAE Decoder 的情况下用于估计通道重要性？

## 4.4 实验 C：跨视频稳定性

在开发集上建立通道敏感度排序。

冻结该排序后，在16条验证视频上评价：

- 每条视频的通道敏感度排名。
- Spearman rank correlation。
- Top-4、Top-8 通道集合重叠率。
- 不同 Prompt、Seed、运动类型之间的敏感度差异。
- 排名的均值与方差。

验证阶段的 VAE 扰动结果仅用于分析，不得反馈到实际 Codec 的参数选择。

如果固定通道排名无法跨视频泛化，应明确报告，不要强行构造全局统一的重要通道集合。

## 4.5 输出结果

- channel_sensitivity.csv
- channel_statistics.csv
- channel_rank_stability.csv
- channel_sensitivity_heatmap.png
- channel_importance_ranking.png
- energy_vs_distortion.png
- quantization_error_vs_distortion.png
- E1_1_report.md

## 4.6 阶段结论

回答：

1. 通道之间是否存在明显的量化敏感性差异？
2. 该敏感性排序是否跨视频稳定？
3. 是否可以使用简单的 Latent 统计量预测重要性？
4. 固定通道混合精度是否具有研究价值？
5. 是否必须使用逐视频动态比特分配？

---

# 5. E1.2：自适应混合精度量化

## 5.1 实验目的

利用 E1.1 中发现的通道敏感性规律，为重要通道保留更高精度，为低敏感度通道分配更少比特。

验证能否在相同实际码率下改善最终视频画质，或在相同画质下减少实际码流大小。

## 5.2 实验配置

### M0：Uniform Baseline

使用 E1.0 中的全部统一位宽配置。

重点保留 INT4、INT5、INT6、INT7、INT8。

### M1：Random Mixed Precision

随机选择高精度通道。

其余通道保持低精度。

至少使用5个固定随机种子。

作为控制实验，判断混合精度收益是否真正来自重要性选择，而不是单纯增加比特。

### M2：Fixed Sensitivity Mixed Precision

依据开发集确定的固定通道敏感度排序，选择 Top-K 通道保留高精度。

测试：

- Top-2。
- Top-4。
- Top-8。
- Top-12。

位宽组合：

- INT4 + INT6。
- INT4 + INT8。
- INT5 + INT7。
- INT4 + INT5 + INT6。
- INT4 + INT6 + INT8。

先在开发集筛选代表配置，再冻结策略进入验证阶段。

### M3：Statistics-Guided Adaptive Precision

根据编码端能够直接获得的 Latent 统计量进行逐视频动态比特分配。

候选指标：

- Channel variance。
- Channel energy。
- Quantization MSE。
- Symbol entropy。
- E1.1 中验证有效的重要性代理指标。

实际编码时不得调用 VAE Decoder。

### M4：Oracle Mixed Precision

允许离线利用 VAE 重建质量进行比特分配搜索。

在多个目标码率预算下，寻找最优或近似最优的通道位宽组合。

M4 只用于估计混合精度方案的潜力上界。

如果仅采用贪心搜索，称为近似 Oracle，不得声称获得全局最优解。

M4 的搜索时间不得计入可部署方法的正常编码延迟，也不得将 M4 的结果作为无 VAE 编码方案的实际性能。

## 5.3 比特预算控制

比较不同方法时，必须控制实际码流大小。

例如：

8个通道使用 INT4，8个通道使用 INT6，其理论平均位宽为5 bit。

但这并不意味着其真实码率一定等于统一 INT5。

因为还存在：

- 不同通道符号分布。
- 量化尺度。
- 位宽配置表。
- 熵编码效率。
- 文件头部。
- Padding。

因此必须基于真实码流进行同码率或近似同码率比较。

建议目标码率预算：

- E1.0 的 INT4 实际码流附近。
- E1.0 的 INT5 实际码流附近。
- E1.0 的 INT6 实际码流附近。
- E1.0 的 INT7 实际码流附近。
- H.264 QP28 的实际码流附近。

所有比较应优先使用逐视频预算。

## 5.4 Rate-Distortion Optimization

建立以下研究目标：

J = D + λR

其中：

- D 为失真。
- R 为真实编码比特数。
- λ 为率失真权衡系数。

Oracle 实验可以使用真实像素域失真。

可部署 Codec 必须使用无需 VAE 的代理失真进行比特分配。

分别比较：

- 固定平均位宽分配。
- 固定真实码率预算。
- 基于 λ 的率失真权衡。

记录量化策略搜索本身的运行时间。

## 5.5 LVC1 Codec 扩展要求

实现：

- 通道级位宽表。
- 每通道量化参数。
- 可变位宽 Bit Packing。
- 独立 Decoder。
- Checksum。
- Version 标识。
- 错误码流检测。

编码器和解码器必须保持一致。

独立解码进程只能使用真实码流及公开固定参数，不能访问原始 Latent、实验标签或原视频。

## 5.6 输出结果

- mixed_precision_results.csv
- bit_allocation_map.csv
- actual_rate_matched_comparison.csv
- oracle_gap.csv
- mixed_precision_rd_curve.png
- bit_allocation_heatmap.png
- bitrate_vs_quality.png
- latency_breakdown.png
- E1_2_report.md

## 5.7 建议阶段性成功标准

以下是本实验建议门槛，不是 Experiment 0 已有结论：

**基础成功：**

- 可部署混合精度方法形成可重复的率失真收益。
- 优于相近实际码率下的 Uniform Quantization。
- 优于相近实际码率下的 Random Mixed Precision。

**进一步成功：**

- 在逐视频相近码率下，平均 PSNR-Y 提升至少0.5 dB。

或者：

- 在逐视频相近质量下，平均实际码率降低至少10%。

同时：

- 报告16条视频中获得收益的比例。
- 报告最差视频退化情况。
- 不允许只展示最佳案例。
- 报告相对于原始 LVC1 的额外编码开销。
- 检查是否仍保留 Post-DiT 编码延迟优势。

如果未达到上述门槛，保留完整结果，不强行进入下一阶段联合优化。

---

# 6. E1.3：Latent 时空冗余与状态变化分析

## 6.1 实验动机

借鉴 Video-native 视频理解方法：

视频信息可以通过参考状态、运动变化和残差更新进行表示，而不是对每一帧进行完全独立的处理。

将这个思想迁移到 Latent 压缩：

研究是否有必要完整编码每一个 Latent 时间切片，以及是否存在可通过参考状态预测恢复的空间区域。

本实验首先做可行性分析，不实现完整 Latent 运动编码器。

必须区分：

- Latent 视频时间维度 T。
- DiT 去噪 timestep。

本阶段只分析前者。

## 6.2 实验 A：时间切片相关性

对于 Final Latent 的各时间切片：

Z0、Z1、Z2、Z3、Z4。

统计：

- 相邻切片 MSE。
- 相邻切片 MAE。
- Cosine similarity。
- Pearson correlation。
- 不同时间间隔的相关性。
- 逐通道时间相关性。
- 相邻切片差分的数值分布和符号熵。

同时分别分析：

- 静态背景视频。
- 局部物体运动。
- 镜头运动。
- 大范围动态变化。

实际场景标签根据已有样本内容确定，禁止虚构样本类型。

单独报告 Z0→Z1 与后续切片之间的差异，因为 Wan Causal VAE 首个时间切片与后续时间切片的时序结构可能不同。

## 6.3 实验 B：空间块变化分析

在 Latent 空间进行 Block 划分。

测试空间 Block Size：

- 1×1。
- 2×2。
- 4×4。
- 8×8。

每个 Block 包含对应位置的全部16个通道。

对于当前时间切片与上一个时间切片，计算：

R(t,b) = Z(t,b) - Z(t-1,b)

进一步计算：

E(t,b) = mean(R(t,b)²)

记录：

- Mean residual energy。
- Maximum residual energy。
- Residual sparsity。
- Block cosine similarity。
- Block variance。
- Residual entropy。
- 各类 Block 的空间占比分布。

生成每个时间切片的残差能量热力图。

注意：直接差分只作为 Zero-motion 基线，不代表最优运动补偿结果。

## 6.4 实验 C：变化区域与最终视频质量的关系

目标：

验证低变化区域是否真的可以降低编码精度或进行参考状态复用。

按照 Block residual energy 排序，将 Block 分成：

- 最低能量25%。
- 次低能量25%。
- 次高能量25%。
- 最高能量25%。

分别对不同组执行受控扰动。

### C1：局部降精度

在相同的高精度参考配置下，只将选中区域从 INT8 降到 INT4。

比较不同能量组造成的视频失真。

### C2：参考状态复用

对于选中的 Block，使用上一重建 Latent 时间切片中相同空间位置的 Block 进行替换。

保持其他位置的参考精度不变。

测试替换比例：

- 5%。
- 10%。
- 20%。
- 30%。
- 50%。

比较：

- 低残差能量区域优先替换。
- 随机区域替换。
- 高残差能量区域替换。

首先执行单个时间切片的受控替换。

再执行整段闭环替换，使用上一重建状态而不是原始状态作为参考。

分别报告单步误差和累积误差。

### C3：实际重建质量分析

所有扰动配置均经过原生 VAE Decoder 恢复视频。

评价：

- 全视频 PSNR。
- 逐帧 PSNR。
- 末四帧 PSNR。
- 局部像素域失真分布。
- SSIM。
- LPIPS（可用时）。

分析 Latent 残差能量与最终视频失真之间的相关性。

不得默认一个 Latent Block 精确对应某个独立像素块或固定数量的视频帧，因为 VAE 的时空感受野可能使影响扩散。

## 6.5 实验 D：预测复用潜力评估

在保持其他信息编码策略固定的情况下，估计不同 Block 可复用比例下的潜在压缩收益。

需要分别报告：

- 可替换 Block 数量。
- 实际重建失真。
- 理论可省略的量化符号数量。
- Block Mask 所需的辅助信息。
- 可复用区域的空间聚集程度。

如果没有实现完整码流序列化和独立解码，此处只能报告理论潜力，不能声称实际码率已经下降。

如果需要验证实际字节收益，可以实现最小化的 Zero-motion Skip Probe：

- 固定 Block Size。
- 显式保存 Skip Mask。
- 编码非 Skip 区域。
- 使用上一重建 Latent 作为参考。
- 全部辅助信息计入实际文件长度。

该 Probe 不扩展到运动搜索、复杂模式决策或完整 Codec 优化。

## 6.6 输出结果

- temporal_similarity.csv
- channel_temporal_correlation.csv
- block_residual_statistics.csv
- block_energy_percentiles.csv
- skip_probe_results.csv
- energy_distortion_correlation.csv
- temporal_correlation_matrix.png
- latent_residual_heatmaps.png
- residual_energy_distribution.png
- skip_ratio_vs_psnr.png
- temporal_error_propagation.png
- E1_3_report.md

## 6.7 必须回答的研究问题

1. Latent 时间切片之间是否存在稳定相关性？
2. 不同视频的时空冗余分布是否明显不同？
3. 首个时间切片与后续切片是否需要区别处理？
4. 低残差能量是否能够作为安全复用区域的判断依据？
5. 低变化区域是否能够承受更低量化精度？
6. 参考状态复用是否导致误差累积？
7. 相比随机选择，基于变化程度的区域选择是否更加有效？
8. 是否有必要进入下一阶段 Latent 域运动补偿实验？

---

# 7. 实验执行顺序与阶段门槛

## Stage 0：环境检查与基线复现

- 检查 Experiment 0 文件。
- 核验数据和模型哈希。
- 复现 INT2、INT4、INT8。
- 检查原有16条验证视频。
- 保存环境及代码版本。

基线无法复现时停止后续执行。

## Stage 1：执行 E1.0

完成完整位宽扫描与率失真曲线。

确定后续混合精度实验的预算区间。

## Stage 2：执行 E1.1

完成通道敏感性分析、统计相关性及跨视频稳定性分析。

如果固定通道排序不稳定，保留固定策略作为对照，重点评估基于逐视频统计量的动态策略。

## Stage 3：执行 E1.2

比较 Uniform、Random、Fixed Sensitivity、Adaptive、Oracle。

开发集完成配置搜索，冻结方案后执行验证集。

## Stage 4：执行 E1.3

完成时间相关性、空间变化、扰动、参考复用及误差累积分析。

E1.3 不依赖 E1.2 必须成功，可以独立执行。

## Stage 5：综合分析

汇总两条路线：

- 值级冗余是否产生实际率失真收益。
- 时空冗余是否具有可利用的预测潜力。

未经本阶段报告审核，不自动启动 E1.4、E1.5、E1.6 或 DiT 联合训练。

---

# 8. 实验目录及记录要求

建议在 Experiment 0 基础上创建独立目录，不覆盖历史结果。

```text
experiment_1/
├── README.md
├── experiment_plan.md
├── environment.md
├── codec/
├── scripts/
├── tests/
├── configs/
├── logs/
├── results/
│   ├── E1_0/
│   ├── E1_1/
│   ├── E1_2/
│   └── E1_3/
├── figures/
├── bitstreams/
├── videos/
├── reports/
├── execution_log.md
├── file_manifest.md
├── code_changes.md
├── experiment_status.json
└── final_report.md
```

所有文件均需记录实际创建路径。

execution_log.md 必须包含：

- 每个实验执行时间。
- 执行命令。
- 输入数据。
- 实验配置。
- 输出文件。
- 运行结果。
- 报错信息。
- 修复过程。
- 未执行项目及原因。

code_changes.md 必须包含：

- 新增文件。
- 修改文件。
- 每个文件负责的功能。
- 核心函数及其输入输出。
- 与 Experiment 0 相比的变化。
- 对应测试方法。
- Git Commit Hash。

file_manifest.md 必须说明全部关键实验文件的位置和用途。

每个子实验结束后更新 experiment_status.json。

状态使用：

- NOT_STARTED。
- RUNNING。
- COMPLETE。
- FAILED。
- BLOCKED。

只有真实完成全部必要测试，才能标记 COMPLETE。

---

# 9. 视频可视化要求

选择代表性静态视频、局部运动视频和其他具有代表性的动态视频。

输出对比 MP4：

- 原始参考视频。
- H.264 QP28。
- Uniform INT4。
- Uniform INT5/INT6 代表配置。
- INT8。
- 最佳可部署 Mixed Precision。
- Oracle Mixed Precision（单独标注）。
- E1.3 代表性参考复用配置。

对比视频添加：

- Method。
- PSNR。
- 实际码流大小。
- Bitrate。
- 编码时间。

保证视频分辨率、帧率、帧数一致。

展示用 MP4 仅用于观察，不计入自定义码流大小，也不作为正式质量评价的数据源。

不要求将所有中间 Latent 或全部实验视频下载到本地。

---

# 10. 最终报告要求

final_report.md 必须按照以下结构组织：

1. Executive Summary。
2. Research Questions。
3. Experiment 0 Baseline。
4. E1.0 完整位宽率失真分析。
5. E1.1 通道敏感性与跨视频稳定性。
6. E1.2 混合精度率失真优化。
7. E1.3 时空冗余与预测复用潜力。
8. Latency Breakdown。
9. Actual Bitstream Breakdown。
10. Ablation Study。
11. Failure Cases。
12. Research Limitations。
13. Conclusions。
14. Next-Step Decision。
15. Reproducibility Instructions。
16. File and Code Manifest。

最终必须明确回答：

- 当前最佳可部署 Latent Codec 是什么？
- 相比 Experiment 0，真实码率和画质改善多少？
- 与 H.264 相近质量下的实际码率差距是多少？
- 额外编码计算开销是多少？
- 哪些通道具有可利用的值级冗余？
- 哪些区域具有可利用的时空冗余？
- 基于 Latent 统计量的重要性判断是否有效？
- 是否有必要进入运动补偿和 Skip Mode 研究？
- 哪些实验发现可能为未来 Compression-Aware DiT 提供研究动机？

不得将 Latent 可压缩性直接等同于 DiT 内部计算可跳过性。

所有尚未测量的指标标记 NOT RUN。

---

# 11. GitHub 与交付要求

确认正确的实验仓库及远端地址后：

- 提交全部新增代码。
- 提交实验配置。
- 提交复现说明。
- 提交实验报告。
- 提交必要的 CSV/JSON 统计结果。
- 提交代表性绘图及对比视频，遵循仓库的大文件策略。
- 大型中间 Latent、模型权重和全部原始日志保留在服务器，必要时提供清单。

不得推送到无关仓库。

如果没有配置正确的 GitHub Remote 或缺少权限，标记 BLOCKED，并说明具体原因。

不要为了追求正向结果而删除失败实验、修改评价标准或隐藏负向结果。

本阶段的最终目标是获得可信、可复现、有明确研究意义的实验结论，而不是强行证明某种压缩方法优于 H.264。