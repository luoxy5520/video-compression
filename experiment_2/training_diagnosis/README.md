# E2.1 训练诊断与优化

在原 E2 基础上增加诊断、训练和评测，保留历史模型及实验。服务器目录为 `/home/luoxiangyu/video-compression-e2/experiment_2/training_diagnosis`。全部结果仅涉及合法训练数据和历史开发集；独立泛化尚未验证，封存测试数据未打开。

## 当前已完成与正在运行的工作

上一轮完成于 2026-11-28 02:04，新一轮于 11:25 启动（Asia/Shanghai）。下表为上一轮完成状态，当前续训见文末。

| 工作 | 实际状态 | 证据 |
|---|---|---|
| 原训练 Loss、梯度、优化器、量化及真实码率审计 | PASS | `results/E2_1a/` |
| R1/R2/R3 重建诊断、R3 同形状恒等初始化对照 | 已完成 | `results/E2_1b/` |
| 单视频 O1、五点 O2 真实码流过拟合 | 已完成 | `results/E2_1c/` |
| D16 三工作点、步数与第二随机种子对照 | 18条历史开发视频评价完成 | `results/development_pool/` |
| D16/D42 中码率各 20,000 步 | 18条历史开发视频真实码流及视频画质评测完成 | `results/development_pool/` |
| V1/V2 中码率各 1,000 步 | 18条历史开发视频评价完成 | `results/development_pool/` |
| V2 中码率 3,000 步、低/高码率各 1,000 步 | 训练及18条开发视频评价完成 | `results/queue_gpu*.json` |
| TP0/TP1/TP2、同容量零历史 TP2Z | 18条历史开发视频固定符号对照完成 | `results/development_pool/` |
| P2 联合 Latent RD 1,500 步、P3 像素联合诊断 100 步 | 已训练并独立解码 | `results/E2_2/` |
| 18 条历史开发视频完整率失真评测 | 38配置×18视频=684组完成 | `results/development_pool/` |
| P2/P3 正式部署计时与汇总报告 | 已完成，报告待最终复核 | `results/finalization_queue.json` |

实时状态以服务器 JSON 和日志为准，表中“排队”或“运行中”不表示实验完成。中码率 V2 从 D16/10,000 步起点微调，低/高码率分别从对应 D16/5,000 步起点微调；三点结果不是仅改变 Lambda 的严格单因素对照。

## 训练与恢复

上一轮由 `continue_training.py`、`finalize_queue.py` 执行，原会话已经完成退出。当前扩展训练见文末。SSH 或聊天 API 中断不会结束已启动的 tmux 作业。

P2每500步、P3每10步原子保存模型、主/辅助优化器、调度器、RNG与下一视频位置；正式检查点另存。继续同一命令会从新目录的 `latest.pt` 恢复。恢复前核验现有tmux作业，避免同一目录并发运行两次。扩展命令记录见 `logs/extension_commands_gpu*.jsonl`。

```sh
cd /home/luoxiangyu/video-compression-e2
tmux list-sessions
cat experiment_2/training_diagnosis/results/extension_queue_gpu0.json
cat experiment_2/training_diagnosis/results/extension_queue_gpu1.json
tail -n 3 experiment_2/training_diagnosis/logs/extension-P3-1000.log
.venv/bin/python -m unittest discover -s experiment_2/training_diagnosis/tests -v
```

最近一次核心回归为 5 项全部通过：默认训练前向、STE 值与梯度、动态 CDF 恢复、时序整数 CDF 平移以及极端符号逃逸编码。

## 评价口径

- 编码器仅接收最终 Latent。原 Wan DiT/VAE 始终冻结；VAE 只用于训练损失和评价。FP32 全序列 VAE 梯度重计算与原接口的前向一致性已经验证。
- 真实 rANS 文件、独立解码、完整容器 Bytes、主/Hyperprior/Metadata 分项分别记录；部署模型大小单列，不混入每视频码率。
- TP0/TP1/TP2/TP2Z 固定 canonical 整数符号和重建 Latent；P2/P3 可改变符号，分别重新评价，不能归入固定符号收益。
- 17 帧、832×480、16 fps，RGB8-bit；YUV 为 limited BT709。PSNR-Y 使用全视频 MSE，末四帧指标沿用 E1 的逐帧 PSNR 均值。
- 全池现场重编码 Uniform INT2–8 和 H.264 多 QP。只在各视频实测 Pareto 支持内插值，不外推，并记录比较覆盖率。不同配置的有效比较视频子集可能不同。
- GPU 起点正式计时一次预热、五次重复；H.264 包含编码端 VAE，Latent Codec 编码端不含 VAE；接收端统一到原始 YUV420。编码端加速与两端总耗时分别记录。

大型权重、原始 Latent 和真实码流留存 113Server；GitHub 同步代码、指标、日志及适量图像/展示视频。完整评测与汇总成功后才会生成最终实测报告，并保留图表复核状态。依赖见上级 `requirements-lock.txt`，新增依赖和许可证见 `environment.md` 与 `third_party_licenses/`。

## 11月28日扩展训练

上一轮已完成全部训练、38配置×18历史开发视频=684组真实码流评价，以及P2/P3正式部署计时。报告仍保留复核状态。TP2在重建不变时平均省字节5.00%，同容量零历史对照下的独立历史收益4.55%；P2相对Uniform同码率均值+1.582dB，但相对H.264同质量的码流比中位数约1.394，仅4/18条视频更小。

- GPU0：P2从1500步完整状态继续到10000步，然后在D42/20000步冻结变换上训练TP2及同容量TP2Z各5000步。
- GPU1：P3从100步完整状态继续到1000步。
- 当前tmux会话为 `e21-extension-gpu0`、`e21-extension-gpu1`，由 `extend_queue.py` 执行。运行状态见 `results/extension_queue_gpu*.json`。
- 新权重位于 `checkpoints/extensions/`，结果位于 `results/extensions/`；每个lane训练后实际编码并用原生VAE评价全部18条历史开发视频，结果位于 `results/extension_evaluation/`。旧模型、码流和结果不覆盖。
- 上轮全部39份实际运行脚本按原始SHA256存入 `source_snapshots/development_pool_20261128/`。归档用于来源复核；复现时需要恢复原scripts目录布局。
