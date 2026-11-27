# Experiment 2.1 — Learned Latent Codec Training Diagnosis and Optimization

## 1. 实验背景

继续113Server上已有的 Experiment 2，不重新搭建实验环境，不重新设计完整 Codec。

当前已经完成：

- I0：Uniform Quantization + Zstd。
- I1：固定量化符号 + Learned Gaussian Entropy Model。
- I2：Learned Transform + Factorized Prior。
- I3：Learned Transform + Hyperprior。
- I2/I3 真实码流生成、独立解码及重复测试。
- 16条训练视频、两个历史开发样本。
- 每个模型1000 steps、四组率失真权重的短训练。

当前存在的问题：

1. I1 的独立高斯概率模型基本未超过 Zstd。
2. I2/I3 在低码率区间存在初步正向结果，但整体画质仍然较低。
3. I2/I3 没有形成合理的率失真曲线。
4. 训练数据量和训练步数可能不足。
5. 当前仅采用 Latent-domain Loss，没有采用 Pixel-domain Loss。
6. 尚未实现 Temporal Prior，无法充分利用时域冗余。

特别关注以下现象：

sample_0000 的 I3 模型：

| Lambda | Actual Bytes | PSNR-Y |
|---|---:|---:|
| 16 | 21001 | 25.454 |
| 128 | 44195 | 25.087 |
| 1024 | 53246 | 24.568 |
| 8192 | 53018 | 23.346 |

随着 Lambda 增大，码率与画质没有形成预期的权衡关系。

需要检查其原因，但不能在检查实际 Loss 定义、Latent MSE 和训练曲线之前，直接认定模型训练代码存在错误。

本实验目标不是盲目增加网络复杂度，而是先建立一个训练正确、具有充分重建能力、率失真性能可靠的 I3 Codec，再引入 VAE-aware Training 和 Temporal Prior。

---

# 2. 全局实验约束

## 2.1 继承已有环境

首先定位并阅读 Experiment 2 的实际目录和关键文件：

- README。
- experiment_plan.md。
- literature_mapping.md。
- intra_sweep.json。
- development_comparison.json。
- 训练日志。
- 模型定义。
- Loss 实现。
- Entropy Model。
- compress/decompress 实现。
- Checkpoints。
- 数据集清单。

必须从真实文件中确认实现细节，不得根据历史报告猜测代码结构。

优先复用：

- 113Server。
- A100 GPU。
- 已有 Wan2.1 环境。
- 原始 Wan VAE 权重。
- CompressAI 及现有熵编码扩展。
- I2/I3 已实现网络。
- E0/E1/E2 已有数据和基线。

不得覆盖历史实验结果及 checkpoint。

## 2.2 模型约束

本阶段：

- 冻结 Wan DiT。
- 冻结 Wan VAE。
- 不修改去噪步数。
- 不重新训练原始生成模型。
- 允许训练 Latent Codec。
- 编码端只能输入 Final Latent。
- 编码端禁止调用 VAE Decoder。
- 训练和评价阶段允许调用 VAE Decoder。
- 不引入额外 VideoDiT Refinement。
- 不引入 GAN。
- 不要求生成标准 H.264 码流。
- 不开展 Compression-Aware DiT 优化。

本阶段优先复用现有 I3 网络，不未经诊断直接更换成新的大型 CNN。

## 2.3 数据集约束

历史16条验证视频及两个开发样本已经参与研究决策，只能作为历史开发评价集。

要求：

- 核验已有合法训练数据。
- 训练集和开发集按视频来源及 Prompt/Seed 分组。
- 保证同一视频的不同切片不会跨集合泄漏。
- 不访问原有封存测试集。
- 如果存在未参与调参的合法样本，可以建立新的独立验证集。
- 如果没有新验证集，明确报告泛化尚未独立验证。

训练数据不足时，先复用现有数据完成诊断，再制定额外数据生成计划。

未经授权，不自动下载大型数据集或启动大规模 Wan 视频生成任务。

---

# 3. E2.1a：训练代码正确性检查

## 3.1 实验目的

回答：

I3 没有形成合理的率失真曲线，究竟是由于训练不充分，还是训练实现存在错误？

本实验优先进行代码审计和小规模测试，不启动长时间训练。

## 3.2 检查实际 Loss 定义

读取当前代码，确认实际使用的是：

方案 A：

L = R + λD

还是方案 B：

L = D + λR

明确记录：

- R_main 的计算方式。
- R_hyper 的计算方式。
- D_latent 的计算方式。
- Lambda 实际作用在哪个损失项。
- 是否使用 sum 或 mean。
- Rate 的归一化分母。
- Distortion 的归一化分母。
- 是否存在额外乘数。
- 训练和验证阶段是否采用一致的归一化口径。

同时记录：

- 未加权 Rate Loss。
- 未加权 Distortion Loss。
- 加权后的 Rate Loss。
- 加权后的 Distortion Loss。
- Total Loss。

在固定 checkpoint 和固定 batch 上分别代入 λ=16、128、1024、8192，验证损失计算是否符合数学定义。

不允许只记录 Total Loss。

## 3.3 检查梯度

对以下模块分别检查梯度：

- Analysis Transform g_a。
- Synthesis Transform g_s。
- Hyper Encoder h_a。
- Hyper Decoder h_s。
- Entropy Model。
- EntropyBottleneck quantiles。

记录：

- 参数数量。
- requires_grad 状态。
- Gradient Norm。
- 梯度为 None 的参数比例。
- 梯度为零的参数比例。
- NaN/Inf 数量。
- Optimizer Step 前后的参数变化。

重点排查：

- detach 使用错误。
- no_grad 使用错误。
- 梯度被量化操作截断。
- optimizer 未包含某些可训练参数。
- 主优化器和辅助优化器参数重复或遗漏。
- entropy auxiliary loss 未正确更新。
- 学习率异常。
- 混合精度导致梯度溢出或下溢。

分别执行 Rate-only、Distortion-only 和完整 RD Loss 的反向传播。

根据各模块的理论依赖关系判断梯度是否正确，而不是要求所有参数在每种 Loss 下都有梯度。

## 3.4 检查量化感知训练

确认当前使用的量化方法：

- Uniform Noise。
- STE。
- Hard Rounding。
- CompressAI Entropy Model 的默认训练行为。

检查 model.train() 与 model.eval() 下量化行为的差异。

测试：

Q0：关闭量化。

Q1：训练阶段噪声注入。

Q2：训练阶段 STE。

Q3：推理阶段真实舍入。

必须确保：

- 训练阶段有有效梯度。
- 推理阶段使用真实离散符号。
- 实际码流可以恢复正确符号。
- Decoder 恢复的 Latent 与真实压缩路径中的重建结果一致。

如果概率模型采用预测均值进行中心化量化，必须明确其符号定义，不能直接假设符号为 round(y)。

## 3.5 检查概率模型及真实码率

对于固定 checkpoint、固定样本，比较：

- Estimated main bits。
- Estimated hyper bits。
- Actual main bits。
- Actual hyper bits。
- Metadata bits。
- 完整文件 bits。

记录估计 Rate 与真实 Rate 的差异。

检查：

- 概率值是否异常。
- likelihood 是否存在零值、NaN 或 Inf。
- CDF 是否在正式压缩前更新。
- checkpoint 加载后熵模型状态是否正确。
- 符号范围及逃逸符号是否正确处理。
- Encoder 和 Decoder 使用的概率分布是否一致。
- 不同量化模式下的 estimated rate 是否混用。

不要求理论 Rate 与实际文件大小完全一致，但必须解释明显差异。

## 3.6 阶段通过条件

满足：

- Loss 数值及权重计算正确。
- 梯度传播符合网络结构。
- 可训练参数正常更新。
- 主优化器和辅助优化器正确运行。
- 无持续 NaN/Inf。
- 真实压缩和解码测试通过。
- 估计 Rate 与实际 Rate 的主要差异可以解释。

发现错误时：

1. 保存错误证据。
2. 修改代码。
3. 记录修改位置及原因。
4. 重跑受影响的测试。
5. 更新实验状态。

未通过本阶段，不允许启动后续大规模训练。

## 3.7 输出文件

- loss_audit.md
- gradient_audit.csv
- parameter_update_check.csv
- quantization_check.json
- estimated_vs_actual_rate.csv
- optimizer_audit.md
- E2_1a_report.md

---

# 4. E2.1b：关闭量化和 Rate Loss 的重建能力实验

## 4.1 实验目的

验证现有 g_a 和 g_s 是否具备足够的表达能力。

本实验暂时不关心压缩码率，只关心经过网络变换之后，是否能够恢复原始 Wan Latent。

如果网络在没有量化误差时仍然无法准确重建，那么继续增加熵编码模块并不能解决根本问题。

## 4.2 实验流程

原始 Final Latent Z

→ Analysis Transform g_a

→ 连续 Coding Latent y

→ 不进行量化

→ Synthesis Transform g_s

→ 重建 Latent Z_hat

训练 Loss：

L = MSE(Z, Z_hat)

完全关闭：

- 量化噪声。
- Hard Rounding。
- Rate Loss。
- Hyperprior 对编码结果的非必要影响。

保留现有 Analysis/Synthesis Transform 的主要结构。

如果现有 g_s 依赖其他必要条件，应保留合法条件路径并记录，不允许为了关闭量化而错误破坏网络。

本实验不生成真实压缩码流，因此码率标记为 NOT APPLICABLE。

## 4.3 实验配置

| ID | 网络配置 |
|---|---|
| R0 | 原始 Latent 直接输入 VAE，质量参考 |
| R1 | 当前 I3 Transform，关闭量化 |
| R2 | 相同结构，增加 Coding Latent 通道数 |
| R3 | 减少一次空间下采样或取消该下采样 |
| R4 | 在不明显改变结构的前提下增加网络容量 |

优先执行 R1。

只有 R1 出现明显重建瓶颈，且已经排除训练问题时，才执行 R2～R4。

不得一开始同时搜索大量网络结构。

## 4.4 训练方式

首先选择一条合法训练视频。

- 固定一个训练样本。
- 记录原始 Latent Shape。
- 记录 Coding Latent Shape。
- 分析网络 Bottleneck 的元素数量。
- 分别在100、500、1000、2000、5000 steps保存 checkpoint。
- 如果训练仍在明显改善，可继续至10000 steps。

上述步数为建议诊断预算，不是论文指定训练参数。

每个 checkpoint 记录：

- Latent MSE。
- Latent MAE。
- Relative L2 Error。
- 各通道误差。
- 各时间切片误差。
- VAE 解码后 PSNR-Y。
- PSNR-RGB。
- 逐帧画质。
- 训练曲线。

使用完整 Latent 序列进行 VAE 评价，不把各时间切片独立解码的视频拼接后冒充正常 Wan VAE 输出。

## 4.5 判断标准

以同一视频的 Uniform INT6/INT7/INT8 作为重建能力参考。

建议：

- 无量化模型应至少具有达到目标画质区间的能力。
- 如果目标是高质量压缩，应尝试接近 INT8 的重建质量。
- 如果无量化模型已经明显低于 INT6，且训练进入平台期，优先检查网络 Bottleneck 和下采样设计。
- 如果扩大容量后质量明显改善，说明原网络容量可能不足。
- 如果关闭量化后表现良好，而正式量化后质量严重下降，则重点排查量化和 RD Training。

不能将无量化重建能力称为压缩性能。

## 4.6 输出文件

- reconstruction_only_results.csv
- reconstruction_training_curve.png
- coding_latent_shape_analysis.json
- architecture_capacity_ablation.csv
- reconstructed_video_comparison.mp4
- E2_1b_report.md

---

# 5. E2.1c：单视频过拟合实验

## 5.1 实验目的

在确认网络可以重建 Latent 后，验证加入量化和码率限制时，网络能否学会有损压缩。

过拟合实验不是为了证明泛化，而是为了验证模型和训练流程是否具备基本学习能力。

## 5.2 实验配置

只选择一条合法训练视频。

恢复 I3 的：

- Analysis Transform。
- Synthesis Transform。
- Hyperprior。
- 可导量化。
- Learned Entropy Model。
- 真实熵编码。

采用三种训练模式：

| ID | Loss | 目的 |
|---|---|---|
| O0 | Latent MSE，无量化 | E2.1b重建基线 |
| O1 | Latent MSE，有量化，无Rate约束 | 判断量化误差影响 |
| O2 | Rate + λ × Latent MSE | 判断完整RD训练能力 |

O1 必须明确量化步长和符号范围，不能通过无限增大符号尺度规避量化误差而将其称为有效压缩。

对于 O2，首先根据 E2.1a 中的实际 Rate 和 Distortion 尺度选择 Lambda。

不直接照搬原来的16、128、1024、8192。

## 5.3 Lambda 选择

在初始模型和少量开发 batch 上测量：

- R。
- D。
- Rate Gradient Norm。
- Distortion Gradient Norm。

如果 Loss 为 R + λD，可以使用 R/D 作为损失数量级平衡的初始参考，再结合梯度和训练表现选择 λ。

建议围绕初始参考权重测试：

- 0.1倍。
- 0.3倍。
- 1倍。
- 3倍。
- 10倍。

该范围仅为初始搜索建议。

先进行短训练筛选，不需要对所有权重进行长时间训练。

## 5.4 训练与评价

使用固定样本进行1000～10000 steps的分阶段训练。

每隔固定步数保存：

- Training Rate。
- Latent MSE。
- 实际完整码流。
- VAE 解码 PSNR。
- Estimated/Actual Rate。
- Checkpoint。
- Learning Rate。

每个正式评价 checkpoint 必须执行真实压缩和解码。

分别绘制：

- Step vs Rate。
- Step vs Latent MSE。
- Step vs PSNR-Y。
- Actual Bytes vs PSNR-Y。
- Lambda vs Actual Bytes。
- Lambda vs Latent MSE。

## 5.5 判断标准

在同一训练视频上，至少观察到：

- Loss 有合理下降。
- 网络在量化条件下仍可重建有效视频。
- 不同 Lambda 可以产生不同的率失真工作点。
- 随失真权重增加，Latent 失真总体呈下降趋势。
- 实际码率和视频质量的总体权衡符合训练目标。
- 真实码流可以独立解码。

不要求每一个 checkpoint 或每个相邻 Lambda 都严格单调。

由于训练采用 Latent MSE，PSNR-Y 与 Latent MSE 不一定完全一致，必须分别分析。

如果模型连单个视频都无法有效学习，禁止通过增加大量训练数据掩盖问题。

## 5.6 输出文件

- single_video_overfit.csv
- lambda_calibration.csv
- overfit_loss_curves.png
- overfit_rd_curve.png
- overfit_checkpoints/
- E2_1c_report.md

---

# 6. E2.1d：增加训练数据及训练步数

## 6.1 实验目的

在训练正确性和单视频学习能力得到验证之后，判断当前 I3 的主要瓶颈是否来自训练不足及数据覆盖不足。

目标是获得稳定、可重复的 I3 率失真曲线。

## 6.2 数据规模实验

首先检查服务器已有的合法训练视频。

建议设置：

| ID | 训练数据 |
|---|---|
| D0 | 1条训练视频，诊断基线 |
| D1 | 16条训练视频，复现当前设置 |
| D2 | 32条训练视频，如合法数据充足 |
| D3 | 全部合法可用训练视频 |

如果实际数据不足，不得虚构样本数量。

保证各规模的数据集尽可能构成嵌套子集，并记录具体视频清单。

训练样本应尽可能覆盖：

- 静态场景。
- 局部运动。
- 镜头运动。
- 复杂动态场景。
- 不同 Prompt 和 Seed。

## 6.3 训练步数实验

| ID | Steps | 目的 |
|---|---:|---|
| T0 | 1000 | 复现历史短训练 |
| T1 | 5000 | 初步收敛检查 |
| T2 | 10000 | 中等规模训练 |
| T3 | 20000 | 检查进一步收益 |
| T4 | 40000 | 必要时继续训练 |

以上为建议的递进预算，不要求无条件运行到40000 steps。

如果模型仍在改善，可从已有 checkpoint 恢复训练。

恢复时必须保存并加载：

- Model State。
- Main Optimizer State。
- Auxiliary Optimizer State。
- LR Scheduler State。
- Global Step。
- 随机种子及可恢复的数据采样状态。

如果改变了网络结构或 Loss，标记为新的训练实验，不称为完全等价的断点续训。

## 6.4 控制实验规模

优先使用经过 E2.1c 筛选的三个代表性率失真工作点：

- 低码率。
- 中码率。
- 高画质。

不要对数据规模、训练步数、Lambda 进行完整笛卡尔积搜索。

推荐执行顺序：

1. 固定16条数据，检查延长训练的收益。
2. 固定已选训练策略，扩大训练数据。
3. 选定较稳定的网络及训练配置。
4. 再增加率失真工作点。
5. 对代表配置使用至少两个训练随机种子检查稳定性。

## 6.5 收敛判断

不能仅以达到指定 steps 作为训练完成标准。

应同时检查：

- Training Loss。
- Development Loss。
- Latent MSE。
- 实际码率。
- 最终视频 PSNR。
- 是否存在过拟合。
- 真实率失真曲线是否继续改善。

如果训练 Loss 下降，而开发集画质长期不提升，应检查数据泛化、失真目标或模型容量。

不盲目增加训练步数。

## 6.6 统一率失真比较

与 E1 Uniform INT2～INT8 进行比较。

每条视频单独绘制：

- I0 Uniform RD Curve。
- I3 历史1000-step RD Curve。
- I3 充分训练后的 RD Curve。
- H.264 多QP参考曲线。

只在真实测量的重叠区间进行插值。

禁止：

- 在曲线覆盖范围之外外推。
- 用低码率结果直接比较高码率结果。
- 用理论 Rate 代替实际文件大小。
- 使用两个开发样本的最佳结果声称整体泛化。

对于不在 Uniform 码率区间内、但同时码流更小且画质更高的点，可以报告为实测支配关系，不能虚构同码率插值结果。

建议阶段性目标：

- 相比 Uniform，在相近实际码率下平均 PSNR-Y 提升至少0.5 dB。

或者：

- 在相近 PSNR-Y 下，实际码率减少至少10%。

该门槛为实验预设目标，不保证达到。

必须报告有效比较样本数和覆盖率。

## 6.7 输出文件

- training_scale_results.csv
- training_steps_ablation.csv
- dataset_manifest.json
- convergence_analysis.png
- development_rd_curves.png
- matched_rate_comparison.csv
- matched_quality_comparison.csv
- worst_cases.csv
- E2_1d_report.md

---

# 7. E2.1e：VAE-aware 量化感知训练

## 7.1 实验目的

Experiment 1 已经发现：

Latent MSE 较小，不一定意味着最终视频失真较小。

因此本实验直接利用冻结的 Wan VAE Decoder，为 Codec 提供最终视频质量的监督信号。

目标不是训练 VAE，而是让 Codec 学会产生更加适合原始 VAE Decoder 重建的 Latent。

## 7.2 模型流程

原始 Final Latent Z

→ Analysis Transform

→ 可导量化

→ Synthesis Transform

→ 重建 Latent Z_hat

→ Frozen Wan VAE Decoder

→ 重建视频 V_hat

原始参考视频：

V_ref = Wan_VAE_Decoder(Z)

像素域失真：

D_pixel = MSE(V_hat, V_ref)

允许提前缓存 V_ref，减少重复计算。

必须保持原始参考与重建视频的颜色空间、归一化及后处理口径一致。

正式 PSNR-Y 评价继续使用 E0/E1 的统一参考。

## 7.3 Loss 消融

| ID | Loss |
|---|---|
| V0 | R + λ_z D_latent |
| V1 | R + λ_p D_pixel |
| V2 | R + λ_z D_latent + λ_p D_pixel |
| V3 | V2 + LPIPS，可选 |

其中：

R = R_main + R_hyper

D_latent = MSE(Z, Z_hat)

D_pixel = MSE(V_hat, V_ref)

V0 使用充分训练后的 I3 作为基线。

V1 和 V2 从相同的 I3 checkpoint 出发。

优先执行 V2，V1 作为 Loss 消融对照。

如果 LPIPS 依赖不可用，标记 V3 为 NOT RUN，不阻塞主实验。

不在本轮增加 GAN 或感知生成修复模块。

## 7.4 梯度正确性

这是本实验最关键的实现要求。

Wan VAE 参数设置为冻结状态，但重建视频分支必须允许梯度通过 VAE 传回 Latent Codec。

要求检查：

- VAE 参数 requires_grad=False。
- VAE 权重没有发生变化。
- Z_hat 可以获得非零有效梯度。
- Codec 的 g_a/g_s 可以通过 Pixel Loss 获得梯度。
- VAE 不加入 optimizer。
- 重建分支不得整体使用 torch.no_grad()。
- 重建分支不得错误 detach。
- 检查 Wan VAE 官方推理包装函数是否内部关闭了梯度。
- 检查时序缓存、原地操作等是否导致 autograd 不兼容。

原始参考视频分支可以使用 no_grad。

如果原始 Wan VAE 的推理接口不能用于反向传播，需要实现等价的可导调用路径，并验证其前向输出与原始解码接口一致。

不得修改 VAE 权重来规避梯度问题。

## 7.5 显存与训练开销

首先在 A100 上执行单个完整 Latent 序列的前向和反向测试。

记录：

- Peak GPU Memory。
- VAE Forward Time。
- VAE Backward Time。
- Codec Forward/Backward Time。
- 单次训练 Step Time。

如果显存不足，依次尝试：

- Batch Size=1。
- Gradient Accumulation。
- Gradient Checkpointing。
- 经过数值验证的混合精度。
- 合法的时空裁剪训练。

如果使用局部时序或空间裁剪，必须检查 Wan Causal VAE 的感受野、首切片特殊结构及裁剪边界影响。

不能直接将局部解码结果视为完整视频训练结果。

## 7.6 微调策略

从 E2.1d 选定的 I3 checkpoint 开始。

使用较小学习率。

建议：

- 先进行100 steps smoke test。
- 再进行1000 steps 小规模微调。
- 之后根据真实 RD 曲线继续至2000～5000 steps。
- 仍然改善时，再考虑延长训练。

Lambda_z 和 Lambda_p 根据实际 Loss 及梯度尺度进行开发集校准。

重点测试：

- 低码率工作点。
- 中码率工作点。
- 高画质工作点。

## 7.7 评价要求

比较：

- Latent MSE。
- PSNR-Y。
- PSNR-RGB。
- SSIM。
- LPIPS（可用时）。
- 实际码流 Bytes。
- 逐帧画质。
- 末四帧画质。
- 最差视频。
- 训练开销。
- 实际推理开销。

重点回答：

加入 Pixel Loss 后，是否能在同等实际码率下改善最终视频质量？

不得仅凭 PSNR 增加就声称成功，因为码率也可能增加。

确认训练结束后：

- 编码端没有 VAE 依赖。
- Codec 可以独立生成真实码流。
- VAE 仍为原始权重。
- 原始 VAE SHA256 未改变。

## 7.8 输出文件

- vae_gradient_check.json
- vae_training_memory.csv
- pixel_loss_ablation.csv
- vae_aware_rd_curve.png
- latent_vs_pixel_distortion.png
- pixel_loss_training_curves.png
- qualitative_video_comparison.mp4
- E2_1e_report.md

---

# 8. E2.2：Temporal Contextual Entropy Model

## 8.1 实验目的

借鉴 GNVC-VD、DCVC-RT 和 DCVC-HEM 的时域上下文概率建模思想。

Experiment 1 已经观察到：

- Z0→Z1 Pearson相关性：0.792。
- 后续相邻切片 Pearson相关性：0.883。

之前的直接参考复用造成较大视频失真。

本实验不直接删除或替换当前 Latent，而是尝试利用历史重建状态预测当前编码符号的概率分布，从而减少熵编码所需比特数。

本轮优先实现 Temporal Entropy Prior，不同时引入完整的复杂 Contextual Transform、运动搜索和 Skip Mode。

## 8.2 数据结构

典型 Wan Final Latent：

[B, C, T, H, W]

例如：

[B, 16, 5, 60, 104]

其中：

- Z0：第一个 Latent 时间切片。
- Z1～Z4：后续 Latent 时间切片。

实际尺寸以数据读取结果为准。

注意这里的 T 是视频 Latent 时间维度，不是 DiT 去噪 timestep。

## 8.3 第一阶段：固定符号时域熵编码

优先执行本实验，以便单独判断时域概率模型的码率收益。

冻结 E2.1d/E2.1e 已训练好的：

- Analysis Transform。
- Synthesis Transform。
- 原有 Hyperprior。
- 量化规则。

对同一视频产生固定的离散编码符号。

比较：

| ID | 熵编码方式 |
|---|---|
| TP0 | 当前 I3 Hyperprior，无时域上下文 |
| TP1 | Hyperprior + 上一重建 Latent |
| TP2 | Hyperprior + 上一重建 Coding Latent |
| TP3 | Hyperprior + 两步历史状态，可选 |

Temporal Prior 网络仅预测当前固定符号的概率分布。

它不得改变：

- 原始量化符号。
- 反量化结果。
- 重建 Latent。
- 最终视频画质。

特别注意：如果 CompressAI 的 GaussianConditional 使用 round(y-mean) 作为编码符号，那么改变 mean 可能改变实际量化结果。

因此固定符号实验必须显式保存原有离散符号和原有重建规则，只改变符号的概率编码方式。

可以使用适配的离散 CDF 和 rANS 接口实现。

如果改变了符号或重建结果，该实验应归类为有损联合优化，而不是固定符号熵编码消融。

## 8.4 Temporal Prior 网络

优先复用已有 DCVC-RT 代码中合适的上下文提取模块，或使用最小化的轻量卷积网络。

不直接套用原始像素域 DCVC-RT 的输入输出结构。

推荐初始结构：

- Context Extractor。
- Temporal Prior Transform。
- Hyperprior Feature Fusion。
- Probability Parameter Head。

输入：

- 上一时刻重建 Latent或重建Coding Latent。
- 当前已解码的 Hyperlatent。

输出：

- 当前符号的概率参数。
- 例如离散高斯分布的均值、尺度及对应 CDF 索引。

网络通道数、卷积层数等根据现有 I3 实现和实际 Tensor Shape 确定。

不得未经代码核验直接假定 GNVC-VD 的具体层数或网络参数。

## 8.5 编解码闭环

首个时间切片：

Z0 → I3 Intra Codec → Stream0 → Reconstructed Z0

后续时间切片：

1. 从上一重建状态提取 Temporal Context。
2. 解码当前 Hyperlatent。
3. 根据 Temporal Context 和 Hyperlatent 生成当前符号的概率分布。
4. 使用对应的离散 CDF 进行熵编码或熵解码。
5. 恢复当前 Coding Latent。
6. 使用 Synthesis Transform 重建当前 Wan Latent。
7. 更新下一时刻参考状态。

Encoder 和 Decoder 必须使用相同的已重建参考。

禁止：

- Encoder 使用原始 Z_(t-1)，Decoder 使用重建 Z_(t-1)。
- 使用未来尚未解码的状态作为解码条件。
- 使用未传输的原始视频统计量作为概率模型输入。
- 使用只有训练阶段才能获得的信息完成实际码流解码。

## 8.6 第一阶段训练 Loss

固定符号训练仅优化熵模型：

L = R_main + R_hyper

其中冻结的 Hyperprior Rate 可以作为固定成本记录。

重点优化当前离散编码符号的条件概率。

训练过程中记录：

- Cross Entropy。
- Estimated Conditional Bits。
- Actual Coded Bytes。
- Temporal Context 计算时间。
- Entropy Model 计算时间。

测试真实码流后，必须断言：

TP0 与 TP1/TP2 的离散符号及重建 Latent 完全一致。

如果使用非确定性浮点算子，必须检查 Encoder/Decoder 的 CDF 或索引一致性，避免熵解码失败。

## 8.7 第一阶段成功标准

主要评价：

- 在相同重建结果下，实际完整码流是否减少？
- Temporal Prior 的额外计算开销是多少？
- 码率收益能否覆盖所有额外辅助信息？
- 后续时间切片是否比首切片获得更多编码收益？

建议：

- 平均实际完整码流减少至少5%，作为初步正向目标。
- 达到10%或以上，作为进一步优化目标。
- 所有固定符号实验必须保持画质完全一致。

若无收益，先检查概率模型拟合程度、参考状态质量、CDF 精度和真实码流开销。

不立即加入更多网络模块。

## 8.8 第二阶段：联合率失真训练

在第一阶段获得可靠结果后，解除部分 Codec 参数冻结。

比较：

| ID | 方法 |
|---|---|
| P0 | E2.1最佳Intra Codec，所有切片独立编码 |
| P1 | 固定符号Temporal Prior |
| P2 | Temporal Prior + 联合Latent RD Training |
| P3 | Temporal Prior + VAE-aware RD Training |
| P4 | 进一步加入Context-conditioned Transform，可选后续 |

P2/P3 允许网络重新学习适合时域概率模型的 Coding Latent。

此时允许量化符号和重建结果改变，因此必须重新进行完整率失真评价。

训练序列逐步扩展：

- 两个相邻时间切片。
- 完整五个时间切片。
- 更长序列仅在合法数据具备时执行。

首切片独立编码，后续切片采用闭环历史参考。

P4 暂不作为本轮强制执行项目。

只有基础 Temporal Prior 已经获得可信收益，并且分析表明条件变换存在进一步优化空间时，才提出后续实现计划。

## 8.9 评价指标

分别报告：

- 首切片实际码率。
- 后续切片实际码率。
- 每个时间切片的 Main/Hyper/Metadata Bytes。
- Estimated Entropy。
- Actual Entropy Payload。
- 完整视频文件大小。
- 最终视频 PSNR。
- 逐帧质量。
- 末四帧质量。
- Latent 误差累积。
- Temporal Context Latency。
- Entropy Coding Latency。
- 完整 Latent-to-bitstream Latency。
- Bitstream-to-latent Latency。

报告时域概率模型究竟节省了多少实际比特，而不是仅报告相邻 Latent 相关性。

## 8.10 输出文件

- temporal_entropy_model.py
- temporal_prior_ablation.csv
- fixed_symbol_consistency.json
- temporal_rate_per_slice.csv
- temporal_rd_curves.png
- temporal_context_latency.csv
- temporal_prior_bitstream_breakdown.csv
- temporal_error_propagation.png
- E2_2_report.md

---

# 9. 正式性能评价

在选定代表性可部署模型后，补齐历史 Experiment 2 中未完成的指标。

## 9.1 对照方案

| ID | 方法 |
|---|---|
| B0 | E1 Uniform INT2～INT8 |
| B1 | 历史I3，1000-step |
| B2 | 充分训练后的I3 |
| B3 | VAE-aware I3 |
| B4 | Temporal Prior I3 |
| B5 | VAE-aware + Temporal Prior |
| B6 | VAE + H.264多QP |

## 9.2 压缩性能

- 完整二进制文件 Bytes。
- Bitrate。
- bpp，统一按原始视频总像素数计算。
- Main/Hyper/Metadata breakdown。
- PSNR-Y。
- PSNR-RGB。
- SSIM。
- LPIPS/DISTS，环境允许时。
- 逐视频 matched-rate comparison。
- 逐视频 matched-quality comparison。
- BD-Rate，有充分有效曲线重叠且实现经过核验时。

## 9.3 性能评价

代表配置至少一次预热、五次正式计时。

报告：

- Analysis Transform。
- Hyperprior。
- Temporal Context。
- Probability Estimation。
- Quantization。
- Entropy Encoding。
- D2H。
- 完整GPU起点 Latent-to-bitstream。
- Entropy Decoding。
- Synthesis Transform。
- VAE Decode。
- Post-DiT两端总时间。
- Peak GPU Memory。
- CPU RSS。
- Parameters。
- FLOPs/MACs。
- 模型权重大小。

如无法完成完整 Prompt-to-bitstream 实测，标记 NOT RUN，不拼接历史数据。

比较 H.264 性能时，优先使用相近画质工作点。

不得将不同质量的局部速度比称为同质量加速。

---

# 10. 实验阶段门槛及决策

| 阶段 | 通过依据 | 未通过时 |
|---|---|---|
| E2.1a | Loss、梯度、量化、码流正确 | 修复训练实现 |
| E2.1b | 网络具有目标画质所需的重建能力 | 检查Bottleneck和网络容量 |
| E2.1c | 单视频可学习，RD权衡合理 | 调整量化及优化策略 |
| E2.1d | 稳定收敛，形成可比较RD曲线 | 检查数据、Loss及泛化 |
| E2.1e | 完成像素域梯度及RD消融 | 分析Latent与Pixel失真差异 |
| E2.2 | 固定符号下真实码率下降 | 检查概率建模及辅助开销 |

E2.1d 如果未超过 Uniform，但模型训练正确且具有稳定的率失真曲线，仍允许执行 E2.1e 和 E2.2 的小规模诊断，因为像素域监督和时域概率模型正是可能改善性能的因素。

但是不得在基础训练明显异常时启动复杂模型的大规模训练。

所有未达到性能目标的结果必须完整保留。

---

# 11. 实验目录与记录要求

在 Experiment 2 下建立独立补充实验目录，例如：

experiment_2/training_diagnosis/

包含：

- README.md
- experiment_plan.md
- environment.md
- literature_mapping.md
- execution_log.md
- code_changes.md
- file_manifest.md
- experiment_status.json
- configs/
- checkpoints/
- scripts/
- tests/
- logs/
- results/E2_1a/
- results/E2_1b/
- results/E2_1c/
- results/E2_1d/
- results/E2_1e/
- results/E2_2/
- figures/
- reports/
- final_report.md

必须记录：

1. 每次执行的命令。
2. 每个新增和修改的文件。
3. 代码修改原因。
4. 训练配置及随机种子。
5. 数据来源与划分。
6. Git Commit。
7. Checkpoint SHA256。
8. 真实码流 SHA256。
9. GPU使用情况。
10. 训练耗时。
11. 实际编解码结果。
12. 失败实验及解决方法。
13. 未完成实验及具体原因。

由于前一轮出现过 SSH 中断，本轮训练脚本应支持：

- 安全保存 checkpoint。
- 断点恢复。
- 独立日志。
- 阶段状态记录。
- 避免重复执行已完成配置。

如果采用 tmux 等方式保持服务器任务运行，应记录会话名称、进程状态和恢复命令。

不得将仅启动的任务标记为 COMPLETE。

---

# 12. 最终报告必须回答的问题

1. 原始1000-step模型是否存在训练代码错误？

2. 当前 I3 的 Analysis/Synthesis Transform 是否具有足够的重建能力？

3. 量化操作本身造成了多大失真？

4. 单视频过拟合实验是否成功？

5. Lambda 应如何选择？历史 Lambda 扫描为什么没有形成合理的率失真曲线？

6. 增加训练步数能够带来多少改善？

7. 增加训练数据能够带来多少改善？

8. 加入冻结 VAE 的 Pixel Loss 后，是否改善同码率视频质量？

9. Temporal Prior 能否在不改变量化符号和最终视频质量的情况下减少实际码率？

10. Temporal Prior 的计算开销是多少？

11. 当前最佳可部署 Codec 的率失真性能如何？

12. 与 Uniform 和 H.264 相近画质下仍存在多少码率差距？

13. 下一阶段应该继续优化网络容量、Loss、熵模型，还是开始完整的 Contextual Transform？

最终报告必须严格区分：

- 已完成实验。
- 初步开发结果。
- 独立验证结果。
- 研究假设。
- 尚未执行的实验。

本阶段不自动启动 DiT 联合训练、Token Pruning、去噪步数优化或完整 VAE Decoder 微调。

所有结论必须来自实际测量，不以代码已经实现代替实验已经完成。