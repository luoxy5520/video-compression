# Wan2.1 文生视频完整流水线性能基线

两条开发样本的冷启动、五次正式原生请求、五次阶段计时及历史 RAW 重编码均已完成。

## 实验环境与生成配置

113Server，NVIDIA A100 80GB PCIe（主机 2 张，本次单卡），AMD EPYC 7302 双路，Python 3.12.3，PyTorch 2.7.1+cu128，CUDA runtime 12.8，cuDNN 90701。Wan 源码 Git 提交 `9737cba9c1c3c4d04b33fcad41c111989865d315`。
UMT5-XXL torch.bfloat16，推理时搬至 GPU，随后卸载到 CPU。DiT 权重 torch.float32，autocast torch.bfloat16，历史 masked PyTorch SDPA。VAE 权重 torch.float32，输出 FP32。完整负向 prompt、参数和 SHA-256 见环境与配置文件。
Wan2.1-T2V-1.3B，832 × 480，17 帧，16 fps，UniPC 20 步，shift 8，CFG 6。CFG 在每步顺序执行 conditional 与 unconditional 两次 DiT 前向。VAE 按一个 latent 时间切片执行一次 decoder，共 5 片，复用因果缓存。

| 样本 | Prompt | Seed |
| --- | --- | ---: |
| sample_0000 | A fixed camera shows a white ceramic teacup on a wooden table in steady daylight. Nothing moves. | 101 |
| sample_1004 | A red toy car rolls left to right along a tiled floor, filmed at ground level by a fixed camera. | 701 |

JM 19.0 FRExt 原生 B1，Baseline profile，QP I/P 28，EPZS SearchMode 3，搜索范围 32，1 个参考帧，首帧 IDR 后接 16 个 P 帧，CAVLC，RDOptimization 0。完整配置随实验归档。FFmpeg 做 BT.709 limited YUV420p 转换及 H.264 流复制封装。

## 测量方法

每个样本独立启动 Python，首个请求兼作 cold-start 和 warm-up。native 请求从调用 generate 前计时，到 FFmpeg 退出且 MP4 存在为止。profile 请求使用相同入口和参数，在 Text Encoder、DiT、VAE 边界同步该卡，另用 CUDA Event 记录同一流中的前向、调度和解码。逐步事件在整个请求完成后解析。CUDA Event 区间也可含流空闲和主机提交间隔。每次请求的解码校验和哈希均在计时之后。
Text Encoder 墙钟覆盖 T5 上下卡、正负 prompt tokenization、前向和条件构建。DiT 墙钟覆盖 scheduler 设置和去噪循环。DiT 下卡及清理、请求设置、latent 噪声初始化、编码器准备等落入 Other。VAE 墙钟从收到 final latent 开始，到浮点视频张量输出结束。RGB 量化、布局、传输、YUV 转换和文件写入计入预处理。流水线未执行 VAE Encoder 或独立的 DiT Encoder。
sample_0000 的 native 与 profile 分属两个进程。首个 profile 因计时包装器绑定错误中止。修复后，阶段计时在单独预热的新进程上执行。初次 MP4 帧率校验失败的调试产物也独立保留。
表 1 占比为同一组 profile 阶段均值除以其直接实测端到端均值，采用互斥墙钟。native 端到端另列。标准差为样本标准差，P95 用线性插值。

## 表 1 各阶段耗时

### sample_0000

| Stage | Mean (s) | Median (s) | Std (s) | P95 (s) | 占端到端比例 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| End-to-End (native) | 32.2776 | 32.0405 | 1.0696 | 33.5038 | 100.00% | 5 |
| Text Encoder | 10.6879 | 10.7084 | 0.3308 | 11.0414 | 32.51% | 5 |
| DiT Denoising | 16.5000 | 16.5592 | 0.3481 | 16.8329 | 50.19% | 5 |
| VAE Decoder | 1.2031 | 1.2034 | 0.0038 | 1.2063 | 3.66% | 5 |
| RGB/YUV Preprocess | 0.1699 | 0.1680 | 0.0069 | 0.1777 | 0.52% | 5 |
| H.264 Encoding | 1.0753 | 1.0776 | 0.0062 | 1.0817 | 3.27% | 5 |
| MP4 Muxing | 0.0752 | 0.0765 | 0.0019 | 0.0767 | 0.23% | 5 |
| Other / Overhead | 3.1635 | 2.8252 | 0.9548 | 4.3833 | 9.62% | 5 |
| End-to-End | 32.8749 | 32.8974 | 0.7195 | 33.6774 | 100.00% | 5 |

Cold-start 123.448 s，模型加载 84.507 s，显式 CUDA 初始化段 0.182 s。Wan 导入也可能初始化 CUDA，包含于 cold-start 起始段。
首轮历史核验显示 final latent 逐元素相同为 True，最大绝对差 0，历史 YUV 有 0 字节不同。
Tokenizer CPU 2.719 ms，UMT5 两次前向 CUDA Event 120.014 ms。条件构建 CUDA Event 合计 0.499 ms，包含 token 张量上卡、长度计算和结果切片。模型上下卡计入 Text Encoder 墙钟。
VAE 核心 CUDA Event 1202.643 ms，5 片 decoder 合计 1201.235 ms。UniPC timestep 设置的主机侧计时为 0.168 ms。final latent 直接在 GPU 上传入 VAE。GPU RGB 量化及排布 0.714 ms，D2H 16.310 ms，CPU 字节序列化 15.840 ms，FFmpeg 转换进程和管道 131.371 ms，RAW write 5.597 ms，编码准备 0.166 ms。VAE 加预处理共 1.3730 s。
Profile 与 native 端到端均值相差 +1.85%。该差值也包含测量时段波动。
Formal profile PyTorch 峰值 allocated 11599.4 MiB，峰值 reserved 14298.0 MiB。

### sample_1004

| Stage | Mean (s) | Median (s) | Std (s) | P95 (s) | 占端到端比例 | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| End-to-End (native) | 32.8451 | 32.4761 | 1.0611 | 33.9696 | 100.00% | 5 |
| Text Encoder | 10.7978 | 10.8648 | 0.8200 | 11.6772 | 33.97% | 5 |
| DiT Denoising | 16.5480 | 16.7902 | 0.5355 | 17.0554 | 52.06% | 5 |
| VAE Decoder | 1.2037 | 1.2031 | 0.0013 | 1.2053 | 3.79% | 5 |
| RGB/YUV Preprocess | 0.1598 | 0.1541 | 0.0116 | 0.1736 | 0.50% | 5 |
| H.264 Encoding | 1.1718 | 1.1508 | 0.0972 | 1.2962 | 3.69% | 5 |
| MP4 Muxing | 0.0790 | 0.0789 | 0.0025 | 0.0818 | 0.25% | 5 |
| Other / Overhead | 1.8251 | 1.6408 | 0.3514 | 2.2276 | 5.74% | 5 |
| End-to-End | 31.7853 | 31.5784 | 0.7416 | 32.6724 | 100.00% | 5 |

Cold-start 124.003 s，模型加载 82.052 s，显式 CUDA 初始化段 0.175 s。Wan 导入也可能初始化 CUDA，包含于 cold-start 起始段。
首轮历史核验显示 final latent 逐元素相同为 True，最大绝对差 0，历史 YUV 有 0 字节不同。
Tokenizer CPU 3.956 ms，UMT5 两次前向 CUDA Event 116.246 ms。条件构建 CUDA Event 合计 0.594 ms，包含 token 张量上卡、长度计算和结果切片。模型上下卡计入 Text Encoder 墙钟。
VAE 核心 CUDA Event 1203.223 ms，5 片 decoder 合计 1201.659 ms。UniPC timestep 设置的主机侧计时为 0.173 ms。final latent 直接在 GPU 上传入 VAE。GPU RGB 量化及排布 0.723 ms，D2H 15.834 ms，CPU 字节序列化 14.176 ms，FFmpeg 转换进程和管道 124.424 ms，RAW write 4.544 ms，编码准备 0.196 ms。VAE 加预处理共 1.3635 s。
Profile 与 native 端到端均值相差 -3.23%。该差值也包含测量时段波动。
Formal profile PyTorch 峰值 allocated 11599.4 MiB，峰值 reserved 14212.0 MiB。

补充 RGB 实验复用各自历史 solver final latent，独立解码后预热一次、正式五次。以下子项来自该补充实验，主表保持请求内实测。每次输出 YUV 都与历史哈希一致。

| 样本 | RGB 量化/转 uint8 CUDA (ms) | 帧布局 CUDA (ms) | FFmpeg user CPU (ms) | FFmpeg system CPU (ms) |
| --- | ---: | ---: | ---: | ---: |
| sample_0000 | 0.621 | 0.070 | 103.423 | 96.952 |
| sample_1004 | 0.628 | 0.070 | 107.392 | 103.654 |

## 表 2 DiT 逐步 CUDA Event 耗时

逐步总时间从 conditional 前向开始计至 UniPC step 返回。区间包含 unconditional 前向、CFG 合成、调度更新，循环开头的 timestep 打包与 model.to 调用计入表 1 的 DiT 墙钟。

**sample_0000** 逐步总时间均值 747.93 ms，最小/最大 734.03/763.02 ms，P50/P95 748.03/754.50 ms。两次 DiT 前向平均 746.78 ms，每步 scheduler 平均 1.12 ms。最快/最慢的逐步均值出现在第 1 / 15 步，均值差 12.67 ms。

| Step | Timestep | Mean (ms) | Std (ms) | P95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 999 | 739.24 | 3.17 | 742.01 |
| 2 | 993 | 743.23 | 5.44 | 749.88 |
| 3 | 986 | 745.89 | 4.69 | 750.99 |
| 4 | 978 | 745.53 | 2.30 | 748.00 |
| 5 | 969 | 747.15 | 3.18 | 751.45 |
| 6 | 959 | 747.77 | 3.51 | 752.02 |
| 7 | 948 | 749.81 | 3.64 | 754.64 |
| 8 | 936 | 749.05 | 3.77 | 753.63 |
| 9 | 922 | 746.96 | 2.87 | 750.36 |
| 10 | 907 | 748.11 | 2.91 | 751.48 |
| 11 | 888 | 748.44 | 3.16 | 751.97 |
| 12 | 867 | 748.03 | 1.04 | 749.14 |
| 13 | 841 | 750.49 | 4.67 | 754.33 |
| 14 | 811 | 748.03 | 2.23 | 749.90 |
| 15 | 773 | 751.91 | 6.78 | 761.18 |
| 16 | 727 | 749.02 | 2.32 | 750.73 |
| 17 | 666 | 750.53 | 2.88 | 754.34 |
| 18 | 585 | 749.06 | 2.95 | 751.97 |
| 19 | 470 | 751.16 | 6.26 | 759.45 |
| 20 | 296 | 749.11 | 4.54 | 755.19 |

**sample_1004** 逐步总时间均值 750.27 ms，最小/最大 740.83/759.46 ms，P50/P95 749.99/755.96 ms。两次 DiT 前向平均 749.03 ms，每步 scheduler 平均 1.23 ms。最快/最慢的逐步均值出现在第 1 / 19 步，均值差 10.38 ms。

| Step | Timestep | Mean (ms) | Std (ms) | P95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 999 | 744.05 | 3.12 | 747.62 |
| 2 | 993 | 747.63 | 4.45 | 753.62 |
| 3 | 986 | 747.34 | 1.64 | 749.45 |
| 4 | 978 | 749.50 | 3.68 | 754.51 |
| 5 | 969 | 749.29 | 4.12 | 754.76 |
| 6 | 959 | 748.63 | 1.68 | 750.25 |
| 7 | 948 | 751.60 | 2.26 | 754.18 |
| 8 | 936 | 749.14 | 3.36 | 753.59 |
| 9 | 922 | 750.15 | 2.59 | 753.22 |
| 10 | 907 | 749.86 | 3.42 | 754.51 |
| 11 | 888 | 749.89 | 2.70 | 753.17 |
| 12 | 867 | 750.24 | 2.48 | 753.61 |
| 13 | 841 | 752.18 | 2.67 | 754.90 |
| 14 | 811 | 751.00 | 1.89 | 753.40 |
| 15 | 773 | 751.31 | 1.69 | 753.47 |
| 16 | 727 | 751.79 | 1.75 | 753.16 |
| 17 | 666 | 751.84 | 3.02 | 755.29 |
| 18 | 585 | 751.59 | 3.43 | 756.25 |
| 19 | 470 | 754.43 | 3.69 | 758.24 |
| 20 | 296 | 754.02 | 4.51 | 759.25 |

## 表 3 H.264 编码及解码校验

| Sample | 输入 | Encoder | Encode Mean (s) | kb/s | Bytes | P 帧 Y-PSNR (dB) | 解码 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| sample_0000 | 本次生成 | JM Native B1 | 1.0517 | 335.47 | 44555 | 39.3840 | 通过 |
| sample_0000 | 历史 RAW，本次重编码 | JM Native B1 | 0.9830 | 335.47 | 44555 | 39.3840 | 通过 |
| sample_1004 | 本次生成 | JM Native B1 | 1.2336 | 1612.15 | 214114 | 37.6352 | 通过 |
| sample_1004 | 历史 RAW，本次重编码 | JM Native B1 | 1.0837 | 1612.15 | 214114 | 37.6352 | 通过 |

P 帧 Y-PSNR 以该次编码输入为参考，先汇总 16 个 P 帧的 8-bit Y MSE，再按峰值 255 换算。全 17 帧 Y-PSNR 另存 CSV。码率按 Annex B 字节数和 17/16 s 计算。JM 裸流解码、FFmpeg MP4 解码与 JM 重建逐字节一致，并验证了 MP4 帧数、帧率和分辨率。原生 JM 的初始化、ME、模式决策、变换量化熵编码和码流单独写入为 `not_available`。

## 瓶颈与理论加速空间

Amdahl 倍率为实测 profile 总时长除以按场景扣减目标阶段后的总时长。若编码提速 k 倍，公式为 T / [T − E × (1 − 1/k)]。VAE 与 H.264 均减半时，公式为 T / [T − (V + E)/2]。T、V、E 分别为实测端到端、VAE 和 H.264 阶段均值，其余成本假设固定。

| Sample | H.264 2× | H.264 5× | H.264 时间归零 | VAE 和 H.264 均减半 |
| --- | ---: | ---: | ---: | ---: |
| sample_0000 | 1.0166× | 1.0269× | 1.0338× | 1.0359× |
| sample_1004 | 1.0188× | 1.0304× | 1.0383× | 1.0388× |

sample_0000 的最大墙钟阶段为 DiT Denoising（50.19%），其后为 Text Encoder（32.51%）。

sample_1004 的最大墙钟阶段为 DiT Denoising（52.06%），其后为 Text Encoder（33.97%）。

DiT 前向和历史 offload 下的模型搬运值得优先分析。Text Encoder 的 CUDA 前向与条件构建只占该阶段墙钟的一小部分，后续可进一步拆解 UMT5 上下卡与主机调度。本次 H.264 与 VAE 的可回收时间按实测基线计算，尚未测试相应优化。

## 历史对照

| Sample | 历史 B1-measured (s) | 历史 DiT-gated 编码 (s) | 历史阶段中位数之和 (s) | 推算特征时间 / gated 编码 |
| --- | ---: | ---: | ---: | ---: |
| sample_0000 | 1.0805 | 1.0759 | 1.1270 | 4.75% |
| sample_1004 | 1.2128 | 1.2193 | 1.2696 | 4.13% |

历史源为 113Server 的 `~/lmp-e1-e2-followup-20261121/repo/results-e1-e2-followup/summary.md`。
历史特征开销由两列历史值相减，存在舍入误差。本次标准流水线未运行 DiT 辅助特征链。历史阶段中位数之和描述的是特征处理与编码两段，历史报告未测量 prompt 到视频的总等待时间。

## 限制与复现

GPU 0 有常驻 sgl_diffusion 进程，显存约 8 GiB。进程级采样 479 条，其中 SM 利用率有数字读数 0 条。
1 秒采样的整卡 GPU 日志可识别持续负载，存在短任务漏采的可能。程序报告自身 PyTorch 峰值分配和保留显存。冷启动使用新进程，保留系统文件缓存。RAW 写入按 Python 写入返回计时，使用 OS 页缓存。RGB→YUV 计时含 FFmpeg 启停和管道复制。JM 编码时间含裸码流和重建写出，MP4 封装另计。

服务器原始产物位于 `/home/luoxiangyu/experiments/t2v_pipeline_baseline.VN08gB`。命令及更多计时字段见 [README](README.md)。
