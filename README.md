# E0 Direct Latent Compression

实际实验产物位于113Server `/home/luoxiangyu/E0_Direct_Latent_Compression`；本目录保存源码、日志、报告、指标和代表性展示。原始latent/YUV、全量码流和中间产物保留服务器，避免拉取大文件。

先阅读 reports/final_report.md，运行方式与码流规范见 reports/codec_format.md。不能将未执行的Prompt级端到端性能当作局部codec测试结果。

原模型权重不修改，不训练，不接触test数据。正式PSNR在展示MP4再编码前计算。

Experiment 1 的代码、冻结策略、完整实测汇总、图表与展示位于 [experiment_1](experiment_1/README.md)，综合结论见 [final_report.md](experiment_1/final_report.md)。E0历史结果保留。

Experiment 2 的训练诊断、VAE-aware优化与Temporal Prior实验见 [E2.1实验说明](experiment_2/training_diagnosis/README.md)、[完整实测报告](experiment_2/training_diagnosis/final_report.md) 和 [具体复核范围](experiment_2/training_diagnosis/review_notes.md)。已完成38配置×18条历史开发视频共684组真实码流评价；独立泛化尚未验证，封存测试集未打开。

2026-11-28继续加强训练：P2联合RD延长至10000步，P3像素联合训练延长至1000步；另在D42/20000步冻结变换上训练TP2与同容量零历史对照各5000步。新结果独立保存在 `experiment_2/training_diagnosis/results/extension_*`，状态以服务器实际队列为准，不将启动或排队视为完成。DiT和原生VAE始终冻结，只训练新增Codec模块。
