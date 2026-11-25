# Experiment 1 — Final Latent 冗余分析与率失真优化

## 1. Executive Summary

统一位宽实测 144/144 配置；通道敏感度 882/882 配置；混合精度 560 条记录（含搜索）；参考复用/局部扰动 816/816 配置。

本轮混合精度未达到基础成功门槛：最好的冻结可部署候选在相近实际码率下，验证平均PSNR-Y仍比Uniform低 1.370 dB；本轮推荐继续使用Uniform按预算选位宽。低残差能量复用优于随机，并不表示失真已可接受。

这是按当前实测产物生成的报告；未完成项目标记 NOT RUN，不以已实现代码代替实验完成。现有结果不能证明低位宽Latent已优于H.264。

## 2. Research Questions

依次检验中间位宽、通道敏感度、固定/统计代理混合精度、时间空间残差与闭环复用。所有方法无训练、不改权重、不改去噪步数；实际Codec仅输入Final Latent。Latent可压缩性不等于DiT内部计算可跳过性。

## 3. Experiment 0 Baseline

results/baseline_gate.json 实测 PASSED：两开发样本INT2/4/8共6配置的重建Latent、PSNR、参考YUV及原码流逐字节一致。Wan VAE SHA256核验一致。开发/验证划分沿用E0，两train样本开发，16条val评价；不读取test张量。E0原结果与源码不覆盖。

## 4. E1.0 完整位宽率失真分析

验证集逐视频指标的算术平均；完整码流Bytes包含头部及量化参数。

| Bit width | n | Bytes mean | PSNR-Y mean dB |
|---|---:|---:|---:|
| INT2 | 16 | 21694 | 17.313 |
| INT3 | 16 | 80218 | 24.434 |
| INT4 | 16 | 127905 | 29.320 |
| INT5 | 16 | 244720 | 33.576 |
| INT6 | 16 | 320336 | 37.962 |
| INT7 | 16 | 422716 | 42.844 |
| INT8 | 16 | 448581 | 47.932 |

H.264 Medium：逐视频PSNR-Y匹配，log(Bytes)线性插值，只在已测量QP曲线覆盖区间内比较，禁止外推。各位宽有效视频数不同，不将不同子集的均值当成整体比较。

- INT4：3/16 有效，Latent/H.264实字节比算术平均 10.341。
- INT5：16/16 有效，Latent/H.264实字节比算术平均 6.187。
- INT6：16/16 有效，Latent/H.264实字节比算术平均 2.973。
- INT7：14/16 有效，Latent/H.264实字节比算术平均 1.439。
- INT8：5/16 有效，Latent/H.264实字节比算术平均 0.341。

INT5/INT6补齐了画质区间，但相近质量的实际码率仍明显高于H.264。不存在脱离质量/码率预算的单一最佳位宽。Q0无损，保存的348.13dB为E0数值保护上限（MSE=0），不解读为有限失真。BD-Rate NOT RUN；可选范围裁剪消融 NOT RUN。

相对E0沿用的INT4，以16val平均值计：INT5画质提高 4.256 dB，但真实Bytes增加 91.3%；这是质量/码率折中，不是同码率改进。

相对E0沿用的INT4，以16val平均值计：INT6画质提高 8.642 dB，但真实Bytes增加 150.4%；这是质量/码率折中，不是同码率改进。

16val的INT2/4/8共48个旧配置PSNR全部复现（差值<1e-6 dB）；同INT4平均字节变化 -39 Bytes，来自元数据字段差异，不记作算法收益。

## 5. E1.1 通道敏感性与跨视频稳定性

已完成扰动记录 882/882。固定排名只用两开发样本INT4单通道RGB MSE增量建立，在进入验证前冻结。单通道效应不相加推断联合失真。

开发最敏感Top4：[14, 8, 3, 15]；最低敏感四通道：[5, 12, 13, 4]。这是开发固定排序，不将验证排序反馈给Codec。

16条验证：固定排序与逐视频排序Spearman平均 0.550；Top4重叠率 0.422；Top8重叠率 0.672。详见channel_rank_stability.csv。

通道统计与实际像素域失真增量的逐视频Pearson/Spearman见statistics_distortion_correlations.csv；这些关联只作探索性描述，不证明因果或跨分布泛化。INT4/INT6实际Latent量化MSE由既有码流补充计算于channel_statistics_augmented.csv，属于事后分析，未反馈到冻结编码策略；原quantization_mse字段为INT8误差。

- variance：16val逐视频Spearman平均 -0.009，正相关比例 43.8%。
- energy：16val逐视频Spearman平均 0.262，正相关比例 75.0%。
- quantization_mse：16val逐视频Spearman平均 -0.221，正相关比例 12.5%。
- quantization_mse_int4：16val逐视频Spearman平均 -0.218，正相关比例 12.5%。
- quantization_mse_int6：16val逐视频Spearman平均 -0.218，正相关比例 12.5%。
- symbol_entropy：16val逐视频Spearman平均 0.079，正相关比例 56.2%。
- temporal_corr：16val逐视频Spearman平均 0.569，正相关比例 100.0%。

是否值得固定排序，须同时看这里的稳定性和E1.2真实码率收益；能量高不直接等价于通道重要。代理关联不等于代理分配必然改善联合失真。验证排名仅作分析，不重新排序Codec。

## 6. E1.2 混合精度率失真优化

混合精度记录 560。Random有五个固定开发种子；Fixed覆盖Top2/4/8/12及三种两级、两种三级组合；Adaptive使用variance/energy/quantization MSE/entropy。验证前仅按开发结果冻结每类两项配置。最近码率筛选不是同码率结论，最终表在真实码流区间内插值。

冻结候选中验证集观察到的最高均值（多候选比较，不是新独立测试结论）：

```json
{
  "policy": {
    "method": "M2_three",
    "levels": [
      4,
      5,
      6
    ],
    "k": 4
  },
  "n": 16,
  "mean_gain_db": -1.36955878549804,
  "worst_gain_db": -2.5062088043720294,
  "benefit_fraction": 0.0
}
```

该候选与H.264 medium在重叠质量区间内：14/16有效，实际Bytes比平均 9.205。

该候选与H.264 fast在重叠质量区间内：14/16有效，实际Bytes比平均 9.520。

这只确定当前冻结候选中的条件性最佳；需要与随机控制在相近实际码率的重叠区间比较，才能判断重要性选择本身的贡献，不以平均理论位宽或最佳单例宣称成功。

所有六种冻结可部署候选的验证收益比例均为0/16，已经未通过优于Uniform的基础门槛；因此不宣称优于随机的基础成功，也不强行进入联合优化。Random开发含五种子，验证只保留两种开发筛选配置，其实际码率覆盖有限。

近似Oracle仅在开发集、三个真实预算上运行受限贪心搜索（最多八轮、前四敏感通道的一位升级），不是全局上界保证。Oracle搜索不计为可部署编码延迟，也不用于冻结验证策略。没有达到门槛时保留负向结果。

## 7. E1.3 时空冗余与预测复用潜力

时间/空间统计覆盖18条样本。视频时间维T与DiT去噪timestep严格区分。直接差分仅为zero-motion基线，残差熵采用1/256数值分箱。1/2/4/8块包含所有16通道，边缘块按实际尺寸计算。受控重建 816/816。开发四种块尺寸、验证预先固定4×4；INT4四分位扰动，以及5/10/20/30/50%低/随机/高能区域单步t=1与闭环复用。

没有完整Skip码流序列化，实际Skip字节收益 NOT RUN。符号省略数、未压缩mask大小只能表示理论潜力；不能据此声称真实码率下降。局部Latent扰动的像素影响可能跨区域/跨帧扩散。

16val相邻切片：Z0→Z1平均Pearson 0.792、MSE 0.237；后续相邻切片平均Pearson 0.883、MSE 0.161。首切片与后续切片分别报告，不从相关性直接推断可省略。

固定4×4、16val的实际重建：

- 10% low闭环复用：PSNR-Y均值 34.938 dB；相对INT8平均下降 12.994 dB，最差下降 21.852 dB；末四帧PSNR均值 36.798 dB。
- 10% random闭环复用：PSNR-Y均值 30.727 dB；相对INT8平均下降 17.205 dB，最差下降 25.052 dB；末四帧PSNR均值 31.509 dB。
- 10% high闭环复用：PSNR-Y均值 27.471 dB；相对INT8平均下降 20.461 dB，最差下降 29.660 dB；末四帧PSNR均值 27.759 dB。
- 残差能量四分位1局部INT4：平均PSNR-Y 35.943 dB。
- 残差能量四分位4局部INT4：平均PSNR-Y 36.201 dB。

低变化是否足以作为安全依据，须看相对INT8退化而非只看是否优于随机。闭环误差的逐Latent切片MSE、逐视频帧PSNR与末四帧指标全部保留；单步仅改t=1，闭环改四个切片，二者差异同时包含扰动次数，不能全归因于传播。空间聚集程度由mask_spatial_transitions描述，局部实际像素MSE以32×32统计网格记录，不假设其对应某个独立Latent块。

## 8. Latency Breakdown

隔离代表配置测量 12/12；完整数据见latency_breakdown.json。量化/打包/熵编码均为真实函数内CPU墙钟；一次预热、五次正式测量。I/O含OS cache，无fsync。VAE操作同步CUDA，返回CPU RGB。另测GPU起点Latent-to-bitstream和D2H，输入为原始保存Latent放回GPU，未重新运行DiT；两端总耗时为GPU起点编码+独立Latent解码+原生VAE。与E0同配置/同元数据重新计时、断言字节相同，并报告额外CPU编码开销；不得直接套用E0历史加速倍数。分片期间的VAE单次耗时只作诊断。另有h264_post_dit_benchmark.json实测GPU起点VAE+YUV转换+H.264 QP28及接收耗时；QP28仅为固定参考，未将其称为同画质速度比较。

| Sample / stream | GPU encode ms | CPU quant / pack / entropy ms | Extra CPU vs E0 ms | Two ends ms | vs QP28 encoding speed |
|---|---:|---:|---:|---:|---:|
| sample_0000 / Q4 | 29.14 | 3.19 / 18.48 / 6.18 | -2.37 | 1249.06 | 182.9x |
| sample_0000 / Q5 | 33.83 | 3.04 / 21.76 / 6.85 | 0.41 | 1251.41 | 157.5x |
| sample_0000 / Q6 | 35.61 | 3.15 / 22.82 / 6.65 | -0.49 | 1258.10 | 149.6x |
| sample_0000 / Q7 | 434.15 | 3.25 / 22.33 / 4.61 | -0.43 | 1654.06 | 12.3x |
| sample_0000 / Q8 | 34.41 | 3.19 / 46.33 / 4.64 | 20.62 | 1259.53 | 154.9x |
| sample_0000 / bits_7777557777775577 | 71.67 | 5.94 / 27.80 / 30.77 | -0.24 | 1329.77 | 74.4x |
| sample_1004 / Q4 | 26.63 | 3.64 / 15.45 / 6.47 | -1.71 | 1243.83 | 61.2x |
| sample_1004 / Q5 | 28.16 | 3.31 / 16.06 / 6.72 | -4.33 | 1246.26 | 57.9x |
| sample_1004 / Q6 | 30.03 | 3.20 / 19.46 / 7.07 | 2.00 | 1253.97 | 54.3x |
| sample_1004 / Q7 | 29.20 | 3.17 / 20.44 / 4.65 | 0.58 | 1257.06 | 55.8x |
| sample_1004 / Q8 | 30.33 | 3.23 / 21.41 / 5.17 | -0.14 | 1258.31 | 53.8x |
| sample_1004 / bits_7777557777775577 | 66.49 | 6.04 / 21.83 / 31.63 | 1.50 | 1328.35 | 24.5x |

额外开销可能出现负点估计，须结合五次样本Std判断抖动，不解释为统计显著加速。上表QP28速度比不是同画质比较，完整Prompt到视频延迟 NOT RUN。

## 9. Actual Bitstream Breakdown

全部实际码率均来自真实.lvc文件。Uniform的header_bytes包含量化参数及元数据，payload为压缩后真实payload。Mixed沿用E0嵌套通道码流，顶层payload包含子码流头，不能将其当成纯熵编码负载；递归分解见actual_bitstream_breakdown.csv，量化/辅助信息JSON值与其余头部（含键名/分隔符/8字节前缀）分别计费，严格相加为完整文件长度。Checksum沿用每个叶子payload SHA256，LVC1版本标识不变。

## 10. Ablation Study

主位宽控制相同channel symmetric absmax、Zstd level3、原生VAE和评价口径。Random控制、固定/逐视频统计代理、三级分配及受限近似Oracle见实测表。可选per-tensor/percentile范围消融 NOT RUN。LPIPS NOT RUN：原环境缺少可用包/权重。

## 11. Failure Cases

完整逐视频/逐帧结果保留，不剔除低画质和退化样本。INT2可出现极低质量；INT4不能作为与高质量H.264的同画质基线。固定排序可能跨视频不稳定；闭环复用可能累积误差，须依实测表检验，不能先行宣称安全。连接/环境错误与修复见execution_log.md，worker错误保留logs与pipeline_failure.json。

例如INT6：最差视频 sample_0022，全片PSNR-Y 30.152 dB；最差帧为 sample_0022 第4帧（从0计），28.211 dB。98种验证配置的最差视频、最差帧、SSIM与末四帧指标见worst_cases.csv。

重要实现限制：验证集按视频等权的平均头部，Uniform约 1811 Bytes，嵌套Mixed约 23315 Bytes。Mixed重复保存每通道元数据，额外开销已如实计入实际码率。本轮负向结果针对当前嵌套LVC1实现及冻结分配策略，不能据此否定共享头部等其他混合精度设计。没有在观察验证结果后修改格式并重称独立验证。

## 12. Research Limitations

只有两条开发样本，验证16条且多个Prompt/Seed重复；不是宽分布独立测试。多个冻结候选的验证比较有选择偏差。无LPIPS、未重跑Prompt/DiT全过程；受限Oracle不提供全局最优保证。零运动扰动未实现实际Skip码流、复杂运动搜索或完整模式决策。所有未实际生成的表/图/视频均为NOT RUN。

## 13. Conclusions

中间位宽显著改善画质，但INT5/INT6同画质下仍有明显实际字节差距。本轮最佳可部署路线仍为Uniform，具体位宽按目标质量和码率选择；冻结Mixed全部未超过Uniform，不能用平均理论bit数判断收益。通道敏感性确有差异，但固定排序泛化有限，方差/能量代理不构成可靠重要性判据。低残差区域的复用比随机有效，然而10%闭环复用仍显著低于INT8参考，不支持将低变化直接认定为安全Skip。受限贪心Oracle未充分搜索，不能量化真正最优潜力。Latent值级/时空冗余不等于DiT计算冗余。

## 14. Next-Step Decision

不启动E1.4/E1.5/E1.6或联合训练。运动补偿/Skip研究需要在复用质量、误差累积和辅助信息开销证据上审核决定；本报告本身不构成自动启动授权。潜在Compression-Aware DiT动机仅为量化敏感度和误差传播，不推出内部Token/计算可跳过。

## 15. Reproducibility Instructions

按README顺序执行；基线失败立即停止。使用同一Wan venv和模型哈希。configs保存冻结策略；logs/pipeline_commands.jsonl记录后续实际命令、时间和GPU。源文件SHA和归档Git commit分别记录，后续归档commit不冒充此前运行版本。完整Latent/码流/块明细位于服务器，不需全部下载。

## 16. File and Code Manifest

详见file_manifest.md、code_changes.md、execution_log.md。GitHub仓库luoxy5520/video-compression，分支codex/experiment-1-latent-rd；保留E0历史，不覆盖现有远端分支。阶段状态和测量子状态见experiment_status.json，未完成交付不得标记COMPLETE。
