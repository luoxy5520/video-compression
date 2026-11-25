# File manifest

本地仓库：`/Users/a1/Documents/note/doc/多模态/直接对latent压缩/E0_Direct_Latent_Compression/experiment_1`。
服务器：`/home/luoxiangyu/E0_Direct_Latent_Compression/experiment_1`。

- experiment_plan.md：用户完整实验规范。
- configs/：开发网格及验证前冻结策略。
- results/baseline_gate.json：6项旧基线复现断言。
- results/environment.json、executed_source_sha256.json：实运行环境和源码摘要。
- results/E1_0/：144条真实码流评价JSON/CSV及H.264质量匹配表。
- results/E1_1/：通道统计、扰动数据、排名稳定性、统计代理相关性。
- results/E1_2/：各分片/汇总混合精度、实际码率匹配表；Oracle与可部署方法分别标注。
- results/E1_3/：时间相似性、块能量分位数、受控复用数据。
- figures/：PNG标准科研绘图；videos/：代表展示（展示不用于正式评价）。
- logs/：实运行输出；pipeline_commands.jsonl记录后续逐实验命令、时间与GPU。
- reports/、final_report.md：阶段报告与综合报告；未测量项NOT RUN。
- 服务器results下各样本的.lvc、原始参考和工作YUV、完整block_residual_statistics.csv保留服务器，不全部下载或入Git。
- experiment_status.json：真实阶段状态；*_measurements为子状态，不表示所有报告和展示已完成。

最终补充：results/E1_1/channel_statistics_augmented.csv 为既有码流INT4/6真实量化误差的事后关联分析；channel_rank_summary.csv为16val排序均值/Std，channel_ranks.csv为逐视频排名。
results/actual_bitstream_breakdown.csv递归区分量化参数值、辅助信息值、其余头部与纯压缩Payload，严格相加为实文件Bytes。
results/h264_post_dit_benchmark.json为新测QP28 GPU起点基线，不是同画质性能比较。
results/artifact_inventory.json记录服务器全部产物的路径、Bytes及SHA256；完整块CSV与码流仅保留服务器。
