# Experiment 0 实验报告

## 1. Executive Summary

### 十项研究问题的回答

| 问题 | 实测回答/状态 |
|---|---|
| 1. 独立真实码流 | YES：LVC1、真实位打包、checksum、完整头部；独立进程解码测试通过。 |
| 2. 编码端VAE | YES：codec和Post-DiT latent编码路径完全不调用VAE；接收端仍需VAE。 |
| 3. 相近质量码率 | 两开发样本INT4在H.264重叠区间的log(rate)-PSNR插值码率约为medium的10.70倍和3.34倍；不是同画质直接实测，也非16val统一结论。 |
| 4. Final Latent到码流加速 | INT4相对medium/QP28为73.09倍、77.01倍；画质未匹配，不能作同质量胜出结论。 |
| 5. 完整生成端加速 | NOT RUN：不把历史Text/DiT计时拼接成本次完整请求实测。 |
| 6. 完整系统加速 | NOT RUN：Prompt级未测；本次Post-DiT至接收像素时间约1.23s，传统约1.68–1.75s，但输出格式与画质不同。 |
| 7. 峰值/搬运改善 | 提前INT8量化D2H字节由1,996,800到499,204，约减少75%；完整峰值显存、RSS改善及真实内存流量 NOT RUN。 |
| 8. 时序差分收益 | E1在开发样本和16val汇总均无收益；16val平均127,943.7→165,854.9 Bytes，PSNR完全相同。E2平均126,702.4 Bytes、29.317dB，仅微小收益。 |
| 9. 主要瓶颈 | 当前均匀低位宽量化的率失真及接收端原生VAE成本；不是已经验证的网络瓶颈。 |
| 10. 后续联合优化 | 当前证据不足以直接推进DiT去噪与压缩联合训练。先研究量化敏感性、预测、比特分配并补足缺测指标。 |

实际完成：2条开发样本、16条开发验证样本、各代表配置5次codec/性能测量、原生JM及libx264曲线、通道扰动、混合精度、SSIM代表配置、对比视频和标准绘图。LPIPS、Prompt级两种加速比、长视频、BD-Rate、GPU bit packing、真实HBM/DRAM profiling、GitHub上传均为NOT RUN。不得将本实验称为全部指标已完成。


已实际实现并测试 Final Latent → LVC1 二进制码流 → 独立重建 Latent；编码模块不导入 VAE、H.264 或像素接口。原生 VAE 仅用于参考构建及接收端重建。以下结论受实际完成范围限制。

Smoke 数据行 100；补充数据行 16；开发验证数据行 304。

状态以 results/*COMPLETE 文件和日志为准。

| 样本 | 方法 | 完整码流 Bytes | PSNR-Y dB | 编码均值 ms |
|---|---|---:|---:|---:|
| sample_0000 | x264_medium_18 | 155270 | 47.333 | 266.880 |
| sample_0000 | x264_medium_28 | 39859 | 41.900 | 199.842 |
| sample_0000 | x264_fast_28 | 38763 | 41.842 | 184.019 |
| sample_0000 | A0 | 1999376 | 348.131 | 14.074 |
| sample_0000 | A2 | 1824490 | 348.131 | 12.793 |
| sample_0000 | A3 | 1603700 | 348.131 | 16.276 |
| sample_0000 | B2 | 468604 | 52.812 | 29.761 |
| sample_0000 | C2 | 124789 | 36.767 | 33.989 |
| sample_0000 | D2 | 24688 | 16.849 | 17.049 |
| sample_0000 | E1 | 139025 | 36.767 | 13.478 |
| sample_0000 | E2_refresh4 | 107940 | 36.767 | 33.593 |
| sample_1004 | x264_medium_18 | 507564 | 46.404 | 352.648 |
| sample_1004 | x264_medium_28 | 165845 | 39.971 | 280.178 |
| sample_1004 | x264_fast_28 | 168552 | 39.991 | 280.044 |
| sample_1004 | A0 | 1999340 | 348.131 | 15.966 |
| sample_1004 | A2 | 1812308 | 348.131 | 15.021 |
| sample_1004 | A3 | 1608040 | 348.131 | 14.404 |
| sample_1004 | B2 | 460557 | 53.659 | 29.526 |
| sample_1004 | C2 | 130758 | 32.492 | 25.268 |
| sample_1004 | D2 | 29349 | 15.581 | 17.287 |
| sample_1004 | E1 | 190783 | 32.492 | 10.094 |
| sample_1004 | E2_refresh4 | 142704 | 32.479 | 27.008 |

无损项的有限超大 PSNR 数值来自 MSE 下限数值保护，表示像素完全一致，不是实际有限信噪比。

## 2. Research Question

验证真实码流、率失真、生成端及接收端负载分配，不要求优于 H.264。

## 3. Motivation

消除生成端像素物化与 VAE 解码，代价是接收端部署及运行原生 Wan VAE。

## 4. Traditional Pipeline

Final Latent → Wan VAE → RGB → BT709 limited YUV420p → JM/libx264 → H.264。

## 5. Proposed Pipeline

Final Latent → 对称量化 → 真正位打包/整数差分 → Zstd → LVC1。接收端独立解码后调用原始 VAE。

## 6. Codec Implementation

详见 codec_format.md、codec/lvc.py 和 tests/test_codec.py。量化参数、通道位宽及完整嵌套码流全部计入文件长度。未训练任何网络，未改变权重。

## 7. Experiment Setup

113Server，A100 80GB PCIe；复用 /home/luoxiangyu/Wan2.1/.venv。Wan 源码 9737cba9c1c3c4d04b33fcad41c111989865d315；模型 /mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth。原始样本配置与 SHA 见 reference.json。UniPC 20步、17帧、16fps、832×480。未重新生成 DiT 视频。

两个开发样本：sample_0000 静态，sample_1004 局部运动。验证为 expand32 的4条val + new48的12条val；不接触test。模型权重不计入单视频码率，接收端须预先部署。模型首次部署、加载成本应另看 reference.json，网络传输未测。

libx264 medium/fast，QP18/22/26/28/32/37/42，GOP17、B帧2、refs3；其他参数为对应preset默认值，日志记录实际命令。JM 原配置复用，编码汇总包含MP4封装，不能与不含封装的结果混为同口径。

## 8. Latent Statistics

各样本 statistics.json 包含 BCTHW shape、dtype、字节哈希、直方图、通道/时间均值方差与相关性。视频时间维 T 与去噪 timestep 无关。实际两条开发样本为 FP32；BF16 输入适配 NOT RUN。

## 9. Rate-Distortion Results

详见 results/rows.json、validation_rows.json、ssim.json 与 *_rd.png。展示 MP4 不参与正式码率或指标计算。PSNR基于原始RGB/YUV；SSIM仅在保存日志的配置上测量。LPIPS NOT RUN：环境无对应包与已核验权重。BD-Rate/BD-PSNR NOT RUN：尚未实现和审核有效重叠区间插值。

sample_0000：E1 相对 C2 真实文件节省 -11.41%；相同量化符号及重建张量已 assert 完全相等。

sample_0000 C2：在实测 H.264 PSNR 重叠区间作 log(rate)-PSNR 线性插值，码率约为 medium 的 10.70 倍。此为插值分析，并非同画质直接实测或BD-Rate。

sample_1004：E1 相对 C2 真实文件节省 -45.91%；相同量化符号及重建张量已 assert 完全相等。

sample_1004 C2：在实测 H.264 PSNR 重叠区间作 log(rate)-PSNR 线性插值，码率约为 medium 的 3.34 倍。此为插值分析，并非同画质直接实测或BD-Rate。

## 10. Latency Breakdown

codec编解码6次（首轮预热，正式5次），报告Mean/Median/样本Std/Min/Max及原始计时。包含文件序列化和缓存文件I/O，不保证落盘fsync。参考的 vae_timing 字段测量 decode-to-CPU-RGB 完整调用；返回元组中的VAE秒数才是同步GPU VAE解码时间。模型加载时间单列。验证集H.264每配置仅一次，用于率失真，不作稳定性能结论。

Final Latent→Bitstream 局部加速比：若benchmark.json存在则见文末本次直接测量表。Prompt→Bitstream 和 Prompt→Reconstruction：NOT RUN，历史流水线数字不拼接成本次端到端实测。接收端VAE成本没有消失。

## 11. Memory Breakdown

参考GPU allocated/reserved峰值见 reference.json；CPU峰值RSS和CPU/GPU量化D2H测试见 transfer.json。FP32→INT8使逻辑D2H字节数接近四分之一，但不是实际HBM/DRAM流量测量。编码端独立进程峰值内存对比 NOT RUN；完整系统峰值改善 NOT RUN；Nsight NOT RUN。其他GPU服务占用约8GB/卡，未停止它们。

## 12. Temporal Redundancy Analysis

E1使用int16差分存储后Zstd，可增加原始中间缓冲；收益依据完整文件。E2闭环预测引用重建上一切片，刷新2/4/8/All。每帧及末4帧指标见rows.json。17帧视频不足以验证长序列稳定性。F逐通道INT4扰动后，敏感度最强4通道保留INT8，其余INT4；此规则仅在开发样本选择，不宣称盲测。

## 13. Failure Cases

本机无numpy；复用服务器venv，不安装模型或依赖。首次本机py_compile因系统缓存路径写权限失败，后续服务器测试通过。一次工具脚本括号错误导致写入未执行，已更正。SSH在默认沙箱下被拒，随后使用已审核服务器连接执行。LPIPS不可用。所有未提供数据的指标均NOT RUN，不能以latent误差替代视频质量。

## 14. Trade-off

低位宽减少真实码流，但可能显著损失画质；时序差分是否有效依场景与真实文件长度判断。编码端消除VAE，并不等于完整系统消除VAE。

## 15. Research Limitations

未测实际网络；未测长视频；没有完全独占GPU。没有本次完整Prompt级流水线实测。GitHub推送 NOT RUN：当前实验目录未配置专用GitHub远端与大文件策略，不能推送到无关笔记仓库。

## 16. Next Steps

先补足同次实验端到端计时、LPIPS、完整内存剖析及更多运动场景，再判断是否研究联合优化；不得由Final Latent可量化推断可省略DiT去噪步骤。

## 本次直接测量的 Post-DiT 性能

| 样本 | 方法 | 编码均值 ms | Post-DiT编码加速比 | 含接收端均值 ms |
|---|---|---:|---:|---:|
| sample_0000 | x264_medium_28 | 1545.368 | 1.00 | 1679.672 |
| sample_0000 | INT8 | 26.439 | 58.45 | 1234.309 |
| sample_0000 | INT4 | 21.145 | 73.09 | 1231.852 |
| sample_1004 | x264_medium_28 | 1619.061 | 1.00 | 1751.444 |
| sample_1004 | INT8 | 28.235 | 57.34 | 1243.880 |
| sample_1004 | INT4 | 21.023 | 77.01 | 1234.167 |

以上加速比较的量化方案与QP28并非质量匹配，不能作为同等画质性能胜出结论。传统接收端输出YUV，latent方案输出CPU RGB，显示准备口径亦不同。完整Prompt级两个加速比仍为NOT RUN。原始五次计时及阶段见benchmark.json；两端峰值未分进程隔离，不能据此声称峰值显存下降。

## 16条开发验证汇总

| 方法 | 已完成样本数 | 平均码流 Bytes | 平均PSNR-Y dB |
|---|---:|---:|---:|
| INT8 | 16 | 448620.2 | 47.932 |
| INT4 | 16 | 127943.7 | 29.320 |
| INT2 | 16 | 21733.2 | 17.313 |
| E1 | 16 | 165854.9 | 29.320 |
| E2 | 16 | 126702.4 | 29.317 |
| x264_medium_28 | 16 | 267357.0 | 38.393 |
| x264_fast_28 | 16 | 263079.8 | 38.385 |

此表为逐视频指标算术均值，非所有帧合并MSE后的PSNR。完整曲线见validation_rows.json，不以单点宣称新方案更优。

归档源码提交：`0f6fea7`（运行后归档，运行时尚无本实验commit）；逐文件SHA见results/source_sha256.json。原始VAE权重507,609,880 Bytes，首次部署和存储不计入单视频码率。
