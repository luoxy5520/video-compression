# LMP Continuation（113Server）

从 2026-11-21 已有的 JM 19.0 续跑构建与诊断继续。两条历史开发视频用于真编码、双解码和直接链路计时。路径与阶段状态见 `00_inventory.md`，最终结果见 `06_final_report.md`。

在 113Server 的全新输出目录中放置本目录的 `scripts/` 和 `configs/policy.json`，并创建 `logs/runs`、`results`、`videos`。保留旧目录 `~/lmp-h264-prior.uFe5ZV`、`~/lmp-e1-e2-followup-20261121` 与原始输入目录。编译脚本复制旧 `jm-perblock` 并应用本轮补丁。程序按历史路径读取 JM 配置、数据清单和匹配实现。

```bash
python3 scripts/record.py "编译 JM 与生成 patch" python3 scripts/build.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 24 "$HOME/Wan2.1/.venv/bin/python" -B scripts/smoke.py
```

第二条命令内生成 E2 候选，运行原生、固定、早停与融合组，双解码、计算 pooled Y-PSNR、测量耗时并汇总。使用新目录复跑以保留每轮证据。首次暖机包含进程已启动后的第一轮，缓存的物理冷启动未单独控制。

```bash
python3 scripts/record.py "汇总计时、率失真和候选证据" python3 scripts/summarize.py
```

基线的独立额外插桩运行入口，在主闭环结果成功产生后执行。这个构建单独验证开启和关闭附加探针后的 C0 哈希，并生成 `results/phase_detail.json`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 24 "$HOME/Wan2.1/.venv/bin/python" -B scripts/profile_detail.py
```
