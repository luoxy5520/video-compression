# Experiment 2 — Learned Contextual Compression of DiT Final Latents

## 1. 任务背景

Experiment 0 已实现 Final Latent → LVC1 真实码流，验证生成端可以跳过 VAE Decoder 和 H.264 Encoder。

Experiment 1 已完成统一位宽扫描、通道敏感性、混合精度以及时空参考复用分析，但未获得有效的率失真改善。

主要结果：

- Uniform INT5：244720 Bytes，33.576 dB。
- Uniform INT6：320336 Bytes，37.962 dB。
- Uniform INT7：422716 Bytes，42.844 dB。
- Uniform INT8：448581 Bytes，47.932 dB。
- H.264 Medium QP28：约267357 Bytes，38.393 dB。
- 当前冻结混合精度方案全部未超过 Uniform。
- 直接复用低残差区域仍造成严重视频失真。

本阶段不再重复大规模原始 Latent 通道敏感性和手工位宽搜索实验。

转向已有成熟神经视频编码架构，构建面向 Wan Final Latent 的学习式压缩 Codec。

## 2. 必须阅读的参考论文

### R1：GNVC-VD

Generative Neural Video Compression via Video Diffusion Prior.

https://arxiv.org/html/2512.05016v2

重点阅读：

- Section 3.1。
- Section 3.2。
- Figure 3(b)。
- Section 3.4。
- Appendix B。
- Figure 8。
- References [11]、[16]、[24]。

重点提取：

- Anchor Latent 的独立编码。
- Predictive Latent 的条件编码。
- Analysis Transform。
- Synthesis Transform。
- Temporal Context Extractor。
- Hyperprior 和上下文熵模型。
- 实际压缩与重建路径。
- 训练损失和权重冻结策略。

暂不实现其 VideoDiT Flow-Matching Refinement 模块。

### R2：GLVC

Generative Latent Video Compression.

https://arxiv.org/abs/2510.09987

https://openreview.net/pdf/abab291375fa0e8e973c2107cc870afda294d966.pdf

重点阅读 Latent Video Compression Module 和 Training Strategy。

提取 Wan 连续 Latent 表示、统一 Intra/Inter Codec、Recurrent Memory、概率模型与率失真训练方法。

### R3：DCVC-RT

Towards Practical Real-Time Neural Video Compression.

https://arxiv.org/abs/2502.20762

官方代码：

https://github.com/microsoft/DCVC

重点检查 DCVC-family/DCVC-RT 中的：

- DC Block。
- Encoder/Decoder。
- Feature Extractor。
- Hyper Encoder/Decoder。
- Temporal Prior。
- Two-step Entropy Model。
- rANS / Entropy Coder。
- Quantization Parameter。
- Stream Writing/Reading。

### R4：ELIC

Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding.

https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html

用于设计 Intra Latent Codec。

### R5：DCVC-HEM

Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression.

https://arxiv.org/abs/2207.05894

用于设计时空上下文概率模型。

### R6：CompressAI

https://github.com/InterDigitalInc/CompressAI

https://interdigitalinc.github.io/CompressAI/entropy_models.html

用于复用 EntropyBottleneck、GaussianConditional、Hyperprior、rANS 及真实压缩接口。

所有借鉴的代码必须注明来源、版本、许可证、修改位置。

如果 GNVC-VD 或 GLVC 的作者代码不可获得，则按照论文描述实现简化的对应模块，不得伪称复现了完整论文模型。

## 3. 全局实验约束

继续使用113Server和已有 Wan2.1 实验数据。

- 冻结原始 Wan DiT。
- 冻结原始 Wan VAE。
- 不修改去噪步数。
- 不训练原始生成模型。
- 允许训练新增的轻量 Latent Codec。
- 编码端仅输入 DiT Final Latent。
- 编码端不调用 VAE Encoder/Decoder。
- 不调用 H.264 进行实际 Latent 编码。
- Decoder 允许恢复 Latent 并调用原生 Wan VAE。
- 暂不在解码端增加 VideoDiT refinement。
- 输出真实自定义二进制码流，不要求 H.264 兼容。

训练期间允许调用冻结的 VAE Decoder 计算像素域失真，但部署时编码器不得依赖 VAE。

冻结 VAE 参数不意味着禁止梯度经过 VAE 传递至新 Codec。

## 4. E2.0：环境配置、代码阅读及迁移

### E2.0.1 环境检查

读取现有 Experiment 0 和 Experiment 1 目录、数据清单、运行环境、Git 版本、模型 SHA256。

检查：

- Python。
- PyTorch。
- CUDA。
- GPU可用显存。
- 编译工具链。
- CompressAI 兼容性。
- DCVC-RT 熵编码扩展依赖。

使用独立实验环境或隔离的依赖配置，禁止直接升级原 Wan 环境中的 PyTorch/CUDA 等核心依赖。

### E2.0.2 安装与获取参考代码

在独立 third_party 目录获取：

- Microsoft/DCVC。
- CompressAI。
- 可核实来源的其他论文代码。

优先复用 CompressAI 的真实熵编码接口完成原型。

官方安装命令为：

`pip install compressai`

实际版本必须根据服务器 Python、PyTorch、CUDA 兼容性确定并锁定。

记录所有 Git Commit Hash、安装命令、依赖版本和编译日志。

不在未获授权的情况下下载大量模型权重或大型训练数据。

### E2.0.3 代码模块映射

生成 literature_mapping.md。

逐个说明：

- 论文中的模块对应开源代码中的哪个文件和类。
- 哪些模块可直接复用。
- 哪些模块需要修改输入输出维度。
- 哪些权重不能直接复用。
- 哪些模块暂不实现。
- 本实验属于论文复现还是受论文启发的适配实现。

DCVC-RT 的原始网络面向 RGB/YUV 图像输入，不能简单将16通道 Wan Latent 输入原始3通道模型。

需要删除或适配像素域 Patchify、Unpatchify 和重建头。

## 5. 数据集构建

### 5.1 训练输入

优先复用服务器中已经存在的：

- 原始 Wan Final Latent。
- 原始 VAE 解码参考视频。
- Prompt。
- Seed。
- 视频分辨率及帧数。
- 原始训练/验证/测试划分。

读取实际样本清单，不假定历史报告中的样本数量仍然准确。

训练不需要构造 H.264 Mode、MV、Residual 标签。

监督目标为：

- 原始 Final Latent。
- 对应原始 VAE 解码视频。
- Codec 的实际或估计编码率。

### 5.2 数据划分

已有16条验证视频已经参与 Experiment 1 的研究决策，因此将其作为历史开发验证集。

从其他合法、未参与调参的数据中建立新的独立验证或测试集。

原有封存测试集继续保持封存，除非用户明确授权最终测试。

按 Prompt/Seed 组合和视频来源进行分组，避免同一视频的切片跨训练和验证集合泄漏。

训练数据不足时：

1. 先执行小规模过拟合和真实码流测试。
2. 再使用已有合法训练数据完成轻量 Codec 原型。
3. 报告数据不足造成的泛化限制。
4. 如需新增大规模生成数据，单独提供计划。

不得将大量重叠 Latent Patch 视为大量独立视频。

## 6. E2.1：Learned Intra Latent Codec

### 6.1 研究目标

首先验证学习式变换与概率模型能否改善单个 Latent 时间切片的压缩性能。

不直接实现完整复杂时序网络。

### 6.2 输入

典型输入：

[B,16,60,104]

空间尺寸以实际数据为准。

### 6.3 模型结构

实现：

Input Latent

→ Analysis Transform g_a

→ Coding Latent y

→ Quantization

→ Learned Entropy Model

→ Real Bitstream

→ Entropy Decoder

→ Synthesis Transform g_s

→ Reconstructed Latent

采用以下递进配置：

| ID | 方法 |
|---|---|
| I0 | 原始 Uniform + Zstd |
| I1 | 原始量化符号 + 学习式熵模型 |
| I2 | Learnable Transform + Factorized Prior |
| I3 | Learnable Transform + Hyperprior |
| I4 | Learnable Transform + Hyperprior + Two-step Spatial Context |

I1 保持原始量化符号和重建 Latent 完全不变，只替换熵编码器，以分离概率建模收益。

I2/I3/I4 引入学习式变换。

### 6.4 建议网络结构

Analysis Transform：

- 输入16通道。
- 轻量卷积或 DC Block。
- 可选 stride-2 下采样。
- Coding Latent 通道数可测试32、64。
- 输出有利于量化和概率建模的表示。

Synthesis Transform：

- 对称或非对称轻量重建结构。
- 恢复原始16通道及空间尺寸。

所有结构参数均为本实验建议初始值，不是 GNVC-VD 原论文的精确网络参数。

针对60×104尺寸，正确处理下采样所需 padding 和 crop。

### 6.5 Hyperprior

实现：

h = h_a(y)

h_hat = Quantize(h)

利用重建的 h_hat 预测 y 的概率分布参数。

实际码流应包含：

- Hyperlatent stream。
- Main latent stream。
- 必要的元数据。
- Shape、Padding、Rate index、Model ID。
- Checksum。

熵编码采用真实 CompressAI/rANS 接口。

概率模型使用的所有信息必须能够在解码端获得。

禁止使用原始 Latent 的未传输统计量作为解码条件。

### 6.6 训练目标

初始使用：

L = R + λ × D_latent

其中：

R = R_main + R_hyper

D_latent = MSE(Z, Z_hat)

训练使用可微量化近似，实际压缩使用真实整数符号。

建议初步测试4个率失真权重，具体范围根据开发集的 Rate/Distortion 数值尺度选择。

不能假定其他论文中的 λ 数值直接适用。

### 6.7 评价要求

将全部时间切片独立编码，拼接恢复完整 Latent 后调用 VAE。

比较：

- Uniform + Zstd。
- 原始符号 + 学习式熵编码。
- Learned Transform。
- Learned Transform + Hyperprior。
- Spatial Context。

只有真实码流解码通过后，才能进入下一阶段。

## 7. E2.2：Contextual Inter Latent Codec

### 7.1 研究目标

借鉴 GNVC-VD Section 3.2 与 DCVC-RT，采用历史重建状态作为编码条件。

不使用手工 MV 搜索作为主方案。

### 7.2 编码流程

首个时间切片：

Z0 → Intra Codec → Stream0 → Z_hat0

后续切片：

F_(t-1) = ContextExtractor(Z_hat_(t-1))

y_t = g_a(Z_t, F_(t-1))

y_hat_t = Quantize(y_t)

Stream_t = EntropyEncode(y_hat_t, shared_context)

Z_hat_t = g_s(y_hat_t, F_(t-1))

参考状态必须来自实际重建结果。

禁止编码端使用真实 Z_(t-1) 而解码端使用 Z_hat_(t-1)。

### 7.3 模型配置

| ID | 方法 |
|---|---|
| P0 | E2.1 最佳 Intra Codec，所有切片独立编码 |
| P1 | 学习式预测残差编码 |
| P2 | Context-conditioned Analysis/Synthesis Transform |
| P3 | P2 + Temporal Entropy Prior |
| P4 | P3 + Two-step Spatial Context |

P1 用简单卷积预测器估计当前 Latent，再编码预测残差。

P2 不要求显式输出 MV 或残差，直接学习条件变换。

P3 将重建参考引入概率模型。

P4 借鉴 DCVC-HEM 的两步空间概率估计，尽量减少串行依赖。

若计算资源允许，后续可增加 GLVC-inspired Recurrent Memory，但不作为首轮必要配置。

### 7.4 训练方式

逐步增加 Latent 序列长度：

- Stage A：单时间切片。
- Stage B：两个相邻时间切片。
- Stage C：完整5个时间切片闭环训练。

允许使用开发数据上的短训练进行架构检查。

必须保存各阶段 checkpoint、训练曲线、验证曲线和实际资源消耗。

训练过程中监测后续切片的误差积累。

### 7.5 评价与消融

报告：

- 首切片码率。
- 各后续切片码率。
- Hyperprior 开销。
- Main payload。
- Temporal Context 的独立收益。
- 全视频 PSNR。
- 逐帧 PSNR。
- 末四帧 PSNR。
- 编码及解码延迟。

在相同实际码率或有效曲线重叠区间比较 P0～P4。

回答条件编码是否比独立切片编码更加有效。

## 8. E2.3：VAE-aware Rate-Distortion Training

### 8.1 研究动机

Experiment 1 表明原始 Latent 量化误差不能稳定预测最终视频失真。

因此不应仅以 Latent MSE 作为唯一训练目标。

### 8.2 两阶段优化

Stage 1：

L = R + λ_z × MSE(Z, Z_hat)

Stage 2：

L = R + λ_z × D_latent + λ_p × D_pixel

其中：

D_pixel = MSE(VAE_Decoder(Z), VAE_Decoder(Z_hat))

原始参考视频优先使用预先缓存的 VAE 解码结果。

VAE Decoder 参数冻结。

如果训练需要梯度经过 VAE，不得将重建分支整体置于 no_grad 或 detach 中。

使用短序列、梯度检查点、混合精度等方法控制显存。

### 8.3 训练消融

| ID | Loss |
|---|---|
| L0 | Rate + Latent MSE |
| L1 | Rate + Pixel MSE |
| L2 | Rate + Latent MSE + Pixel MSE |
| L3 | L2 + LPIPS |

L3 在环境具备可靠 LPIPS 依赖和模型权重时执行。

先以 PSNR 和源视频保真度为主目标。

Perceptual loss 作为独立实验，不将 LPIPS 的改善直接称为 PSNR 改善。

本阶段不引入 GAN 和扩散重建模块。

## 9. E2.4：真实码流及性能评估

### 9.1 码流生成

所有可部署配置必须实现：

- compress()。
- decompress()。
- Entropy Model CDF update。
- 实际二进制序列化。
- 独立解码进程。
- 模型版本标识。
- Shape 和 padding 恢复。
- 完整码流长度统计。
- 错误输入检测。

训练阶段估计 rate 不等于实际码率。

正式结果只采用真实二进制文件长度。

模型权重采用预共享部署，权重大小、加载时间和接收端部署要求单独报告。

### 9.2 统一对照

| Baseline | Description |
|---|---|
| B0 | Experiment 0 / LVC1 |
| B1 | Experiment 1 Uniform INT2～INT8 |
| B2 | H.264 Medium/Fast 多QP |
| B3 | Learned Intra Latent Codec |
| B4 | Contextual Inter Latent Codec |
| B5 | VAE-aware Contextual Latent Codec |

如环境允许，增加官方 DCVC-RT 作为像素域神经编码参考。

但必须分别报告传统路径所需 VAE Decode 时间。

### 9.3 评价指标

Rate：

- Bytes。
- kbps。
- bpp。
- Main/Hyper/Metadata breakdown。

Distortion：

- PSNR-Y。
- PSNR-RGB。
- SSIM/MS-SSIM。
- LPIPS。
- DISTS（可用时）。
- 每帧及末四帧质量。

Latency：

- GPU Latent-to-bitstream。
- Analysis Transform。
- Context Extraction。
- Hyperprior。
- Probability Estimation。
- Entropy Coding。
- D2H。
- Entropy Decoding。
- Synthesis Transform。
- VAE Decode。
- Post-DiT 两端总处理时间。

Complexity：

- 模型参数量。
- MACs/FLOPs，记录统计工具及口径。
- 峰值显存。
- CPU RSS。
- 模型权重大小。
- 中间张量大小。

代表配置进行预热及至少五次正式计时。

对 GPU 竞争和异常时间进行复测。

### 9.4 主要实验目标

目标一：

在相近实际码率下，相比 Experiment 1 Uniform 获得平均至少0.5 dB PSNR-Y 提升，或者在相近PSNR下减少至少10%的实际码率。

目标二：

逐步缩小与 H.264 的率失真差距。

H.264 QP28 的历史参考点约为：

267357 Bytes / 38.393 dB。

该均值仅作直观目标，正式结论必须逐视频比较实际码率和质量。

目标三：

保持生成端低延迟优势。

新的神经 Codec 不得仅为提高压缩率而引入过高的计算开销。

建议将相近画质下生成端相对 VAE + H.264 的实际加速比作为重要指标。

目标四：

确定性能瓶颈来自 Transform、Entropy Model、时序预测还是数据搬运。

这些目标均为实验预设门槛，不是已获得的结果。

如果没有达到门槛，完整保存负向结果。

## 10. 实验执行顺序

严格按阶段推进：

1. 完成 E2.0 环境和论文代码映射。
2. 完成 E2.1 小规模 smoke test。
3. 验证熵编码生成真实码流。
4. 训练简化 Intra Codec，比较 Uniform。
5. 执行 E2.2 时序上下文编码。
6. 执行 E2.3 像素域率失真微调。
7. 执行 E2.4 独立验证、实际计时和完整消融。

优先完成 I0～I3 与 P0～P3 的最小可运行版本。

不要在基础模型尚未有效时，同时训练大量复杂架构。

每阶段根据真实率失真收益决定是否继续增加模型复杂度。

## 11. 实验记录和交付

创建独立 experiment_2 目录，不覆盖 E0/E1。

必须保存：

- literature_mapping.md
- experiment_plan.md
- environment.md
- requirements-lock.txt
- dataset_manifest.json
- model_architecture.md
- codec_format.md
- execution_log.md
- code_changes.md
- file_manifest.md
- experiment_status.json
- checkpoints/
- configs/
- tests/
- logs/
- results/
- figures/
- reports/
- final_report.md

记录每次：

- 执行命令。
- 新增与修改文件。
- 代码来源。
- 模型配置。
- 训练超参数。
- 训练和验证损失。
- GPU显存与训练耗时。
- Checkpoint Hash。
- 真实码流文件及哈希。
- 编解码测试。
- 失败原因及解决方法。
- 未执行指标。

所有未真实完成的实验标记 NOT RUN 或 BLOCKED。

最终报告必须回答：

1. 学习式变换相对于原始量化有多少实际率失真收益？
2. 学习式熵模型能否减少相同量化符号的码流大小？
3. Hyperprior 的收益能否覆盖其额外码率？
4. 条件编码相对于独立切片编码改善多少？
5. Pixel-domain loss 是否改善最终重建视频？
6. 相近画质下与 H.264 的码率差距是多少？
7. 相近画质下生成端的实际加速比是多少？
8. 最佳可部署架构的计算和访存瓶颈是什么？

本轮不启动 Compression-Aware DiT 联合训练。

首先建立可靠、高效、真实可解码的 Wan Final Latent Codec。