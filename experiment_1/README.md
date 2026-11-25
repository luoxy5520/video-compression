# Experiment 1 — Final Latent 冗余与率失真

本目录与 E0 历史结果分离。正式数据和中间码流保留在 113Server 的
`/home/luoxiangyu/E0_Direct_Latent_Compression/experiment_1`。
只使用 E0 的两条 train 开发样本及既有 16 条 val；不读取封存测试集张量。

运行前阅读 `experiment_plan.md`、`experiment_status.json` 和 `execution_log.md`。
只有 `results/baseline_gate.json` 为 PASSED 才能执行后续实验。

```sh
cd /home/luoxiangyu/E0_Direct_Latent_Compression/experiment_1
PY=/home/luoxiangyu/Wan2.1/.venv/bin/python
$PY -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py --stage gate
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py --stage uniform
$PY scripts/analyze.py --stage statistics
CUDA_VISIBLE_DEVICES=0 $PY scripts/analyze.py --stage sensitivity
$PY scripts/pipeline.py
# 所有质量测量结束后，隔离执行代表配置的延迟测量：
CUDA_VISIBLE_DEVICES=0 $PY scripts/profile.py
PYTHONPATH=../tools_plot $PY scripts/summarize.py
```

`pipeline.py` 从已完成开发集扰动的 checkpoint 接续，冻结排名，再执行分片验证、混合精度和参考复用。
首次完整重跑可先运行 sensitivity；已运行的单进程实验可用
`--existing-sensitivity-pid PID` 在冻结排名后交接给验证分片。
该参数仅终止本实验的 sensitivity 进程，保留其已完成开发记录。
各分片从头执行；不是逐配置失败续跑。

Mixed 的随机种子、开发搜索网格和冻结策略保存在 configs。
近似 Oracle 仅用于开发集上界探测：从目标统一位宽减一开始，最多八轮，
每轮仅尝试固定敏感度前四通道的一位升级，不保证全局最优，也不代表全部 Oracle 潜力。
验证集不会用于更新策略。最近码率邻居只用于开发筛选，最终同码率比较使用测量区间内插值。

Skip/Reuse 仅为受控扰动，不是完整 Skip Codec。
`mask_uncompressed_bytes` 是理论未压缩 mask 大小，`selected_symbols` 是潜在省略数；
不得将这些数值当成实际码率收益。视频展示 MP4 不参与正式画质评价。

`experiment_status.json` 区分阶段状态与 `_measurements` 子状态；主测量结束不等于所有交付已完成。
LPIPS 不可用时保留 NOT RUN，不安装替代模型或修改原权重。
