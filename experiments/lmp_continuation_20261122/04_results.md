# 结果索引

- [逐视频闭环结果](results/runs.csv)覆盖两条开发视频、七组方法、14 个独立码流。每组重复四次，结果基于三次正式重复。所有码流双解码通过，像素与编码器重建 SHA 一致。
- [计时基线](results/timing_baseline.csv)显示 C0 两条视频分别耗时 0.992 / 1.128 秒，搜索约占 16.7% / 21.5%，对应 Amdahl 理论上限 1.20× / 1.27×。[完整时间表](results/timing_breakdown.csv)记录暖机及正式重复范围。[独立阶段插桩](results/phase_detail.json)给出宏块流程、变换量化、熵编码和 RSS 峰值。
- [先验与编码的逐次时间、命令参数和哈希](results/repeats.jsonl)包含 56 个唯一运行。DiT 先验每条约 51–55 ms，匹配约占 48 ms。C2-adaptive 的直接链路时间比 C0 增加 6.28% / 6.00%。
- [率失真对比](results/rd_comparison.csv)显示自适应融合通过两个样本的质量门槛。固定融合在运动样本上较同预算像素组降低码率并提升 P 帧 PSNR，相比完整 C0 仍退化 +17.44% 码率、−0.422 dB。
- [历史候选四格表](results/candidate_analysis.csv)和[历史回退摘要](results/historical_fallback.csv)源于 11 月 21 日实验。[试点门槛结果](results/summary.json)显示六个对照方案全部未达到净加速门槛。

完整解释、复现索引和研究判断见 [最终报告](06_final_report.md)。
