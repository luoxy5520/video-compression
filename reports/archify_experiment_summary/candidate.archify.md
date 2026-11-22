# DiT 中间特征辅助 H.264 编码 {#dit-h264}

Latent 能补回部分 JM 遗漏的运动候选。两条开发视频的自适应融合保持了画质和码率，**计入先验后，编码链路反而更慢**。

@answer scope="JM 19.0 · QP28 · 2 条开发视频闭环 · 8 条历史开发视频诊断"
@meta locale=zh-CN style=swiss
@source candidates title="LMP 续跑 · 历史候选四格表" path=experiments/lmp_continuation_20261122/results/candidate_analysis.csv lines=1:17
@source search title="11 月 21 日 · 搜索调用分解" path=experiments/lmp-e1-e2-followup-20261121/supplement-analysis/search_aggregate.csv lines=1:7
@source gates title="11 月 22 日 · 连续墙钟与质量门槛" path=experiments/lmp_continuation_20261122/results/summary.json
@source runs title="11 月 22 日 · 逐视频编码及双解码记录" path=experiments/lmp_continuation_20261122/results/runs.csv lines=1:15
@source method title="LMP 续跑 · 算法、闭环与计时" path=experiments/lmp_continuation_20261122/06_final_report.md lines=31:77
@source bins title="历史运动幅度分桶 · E2 覆盖率" path=experiments/lmp-e1-e2-followup-20261121/supplement-analysis/analysis.md lines=8:20
@source t2v title="Wan 原生全流水线 · 编码占比" path=experiments/t2v_pipeline_baseline/BASELINE_REPORT.md lines=26:63

## 关键对照 {#conclusion-charts span=12}

@image id=measured-conclusion path=conclusion_chart.png alt="E2 补回 7.29% 漏块，净搜索次数增加 3.24%，自适应融合耗时增加 6.14%。高运动样本对比等预算 C-pixel 为码率 −7.43%、画质 +0.145 dB、耗时 +4.70%；对比完整 EPZS 为码率 +17.44%、画质 −0.422 dB、耗时 +3.82%。" caption="候选诊断、搜索评估与编码墙钟；高运动样本按两种基线分别计算变化。"

上排是不同实验的三项指标。下排只比较玩具车样本的固定融合。等预算下率失真改善，链路仍慢 4.70%。相对完整 EPZS，三项指标均退化。[](source:candidates)[](source:search)[](source:gates)[](source:runs)

## 候选如何进入闭环？ {#method span=6}

@workflow id=codec-flow title="VAE 源帧与 latent 候选汇入 JM"
| latent | Wan Final Latent | external |
| vae | VAE → RAW 源帧 | backend |
| match | 特征匹配 → MV | frontend |
| me | JM 像素域 ME | backend |
| bit | 闭环 → H.264 | backend |
| latent > vae | |
| latent > match | | emphasis |
| vae > me | 源帧 |
| match > me | 候选 | emphasis |
| me > bit | |

CPU 在 9×9 窗内做归一化特征匹配。种子改善初始像素与 MV 代价时，JM 放宽首道早停门槛。闭环仍用源帧与重建参考做运动补偿、残差、量化与熵编码。[](source:method)

## 闭环编码实测 {#quality span=6}

| 方案 | 茶杯链路 | 玩具车链路 |
| --- | --- | --- |
| C0 · 原生 EPZS | 0.992 s | 1.128 s |
| C-pixel · 原生早停 | 0.996 s | 1.126 s |
| C2 · 自适应融合 | 1.055 s | 1.196 s |

C2 相对 C0 的 P 帧 Y-PSNR 变化为 +0.004 / +0.018 dB，码率 +0.099% / −0.114%。14 组码流均通过双解码。[](source:runs)

## 搜索为什么没有变便宜？ {#limits span=6}

- **实测** 门控节省约 4.0 万次原生整数评估，但额外评估约 19.9 万次种子。每条先验另耗 51–55 ms。[](source:search)[](source:method)
- **实测** E2 在 ≥16 px 桶的覆盖率为 0.63%。玩具车固定融合相对 C0 的码率增加 17.44%。[](source:bins)[](source:runs)
- **待验证** 跳过冗余种子，并只在可信块替换候选，能否减少总搜索且保持 RD？

## 下一轮如何验证？ {#next span=6}

@steps
| select | 先筛选再替换 | 去重，限制总预算，仅在可信块替换候选。 |
| ablate | 做等预算闭环 | 与 EPZS 和像素对照比较净调用、RD 和含先验墙钟。 |
| gate | 达标后再扩大 | 先达到 10% 净节时与质量门槛，再做多 QP 和全链路。 |

16 条独立验证、BD-rate 与辅助版 prompt 到 MP4 尚待评测。原生 Wan 基线中，H.264 约占全请求时长的 3.3%–3.7%。[](source:t2v)
