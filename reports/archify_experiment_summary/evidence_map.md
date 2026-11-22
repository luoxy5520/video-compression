# 汇报图证据索引

下列路径来自当前项目。已归档的服务器提交见续跑报告 §9，桌面检出 HEAD 与服务器实验提交是不同记录。

| 图中主张 | 口径与复算 | 原始证据 |
| --- | --- | --- |
| E2 补回 7.29%，E3 补回 7.47% | 8 条历史开发视频，K=2、L∞ 半径 2 px。在 E0 漏块中，1,269 ÷ 17,419 和 1,302 ÷ 17,419。 | [候选四格表](../../experiments/lmp_continuation_20261122/results/candidate_analysis.csv) · [续跑统计](../../experiments/lmp_continuation_20261122/results/summary.json) |
| 门控搜索评估净增 3.24% | 8 条历史开发视频。199,168 次新增种子 − 40,065 次原生整数节省 + 3,100 次亚像素新增 = 162,203；除以 5,007,040 次原生评估。 | [搜索分解](../../experiments/lmp-e1-e2-followup-20261121/supplement-analysis/search_aggregate.csv) · [历史补测说明](../../experiments/lmp-e1-e2-followup-20261121/supplement-analysis/summary.md) |
| 自适应融合链路耗时增幅 6.14% | 2 条视频，各用正式重复的墙钟中位数。逐视频增幅为 6.2807%、6.0012%，再取中位数。 | [运行总表](../../experiments/lmp_continuation_20261122/results/runs.csv) · [逐次复测](../../experiments/lmp_continuation_20261122/results/repeats.jsonl) |
| 码率、画质与 14 组双解码 | Y 指标以同一缓存 VAE 源帧为评分对象。两条样本的 C2 自适应组均通过续跑质量门槛。 | [率失真表](../../experiments/lmp_continuation_20261122/results/rd_comparison.csv) · [执行记录](../../experiments/lmp_continuation_20261122/02_execution_log.md) · [汇报用原报告](../../experiments/lmp_continuation_20261122/06_final_report.md) |
| 先验 51–55 ms、匹配约 48 ms | CPU 读 latent、归一化、匹配与写候选；JM 读候选计入编码段。先验与 JM 链路用同一墙钟。 | [计时分解](../../experiments/lmp_continuation_20261122/results/timing_breakdown.csv) · [运行脚本](../../experiments/lmp_continuation_20261122/scripts/smoke.py) |
| 归一化匹配、实时像素代价、早停 | 9×9 匹配；E2 用 2×2 聚合后转换到 qpel。在线 C2 自适应只在种子改善像素与 MV 代价时放宽早停。 | [历史匹配代码](../../experiments/lmp-h264-prior/probe.py) · [自适应历史补丁](../../experiments/lmp-e1-e2-followup-20261121/jm_changes.patch) · [固定预算新补丁](../../experiments/lmp_continuation_20261122/configs/jm_fixed_search.patch) |
| E2 覆盖率 87.40% / 0.63% | 来自桶内覆盖。对应块数为 45,663 / 9,150，整体样本为 108,175 个块。换算成运动幅度分布约为 42.21% / 8.46%。 | [运动幅度与覆盖分析](../../experiments/lmp-e1-e2-followup-20261121/supplement-analysis/analysis.md) |
| 高运动样本的固定融合取舍 | C2-fixed 对同预算像素组，1,893.29 ÷ 2,045.26 − 1 ≈ −7.43%，P 帧约 +0.145 dB。对 C0 的码率增加 17.44%。链路耗时分别比较 1.171098 ÷ 1.118523 与 1.171098 ÷ 1.128034，对应 +4.70% 与 +3.82%。 | [逐视频编码](../../experiments/lmp_continuation_20261122/results/runs.csv) |
| 原生 H.264 占全请求 3.3%–3.7% | 同步 profile 分解按样本分别为 1.0753 ÷ 32.8749 和 1.1718 ÷ 31.7853。原生请求中位数是单独计时。 | [Wan 全流水线基线](../../experiments/t2v_pipeline_baseline/BASELINE_REPORT.md) |
| 历史 22→14 与 2.74→2.71 s | 独立亚像素方向实验，计时从 latent 起算。留在历史口径说明。 | [实验地图 §13](../../实验地图.html) · [逐视频原始结果](../../learned_qpel_20261107/results/results.json) |
| 旧 A、P0、P0.5、P0.6、P2-Lite | 直接预测与残差回放属于另一条研究线，P0.6 离线源辅助属于上界。 | [项目 README](../../README.md) · [实验进展总览](../../docs/experiments/DiT_H264_实验进展总览.md) |
| 候选替换与扩大门槛 | 候选替换、分层置信度预算为待测假设。扩大前对齐闭环和墙钟，沿用 10% 的净节时门槛。 | [续跑计划](../../experiments/lmp_continuation_20261122/01_experiment_plan.md) · [最终报告 §10](../../experiments/lmp_continuation_20261122/06_final_report.md) |
