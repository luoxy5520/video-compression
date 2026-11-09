# P2-GeneralizationProbe

[阶段判定](decision.md) · [GZ0 分层泛化](tables/gz0.md) · [GZ1 表征](tables/gz1.md) · [GZ2 目标](tables/gz2.md) · [关键图](figures/generalization.svg) · [科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-generalizationprobe-20261109)

本轮仅使用既有 44 条训练池的内部折进行 GZ0–GZ2。冻结 A 的 16 条历史验证只用于 PRE 核验。实验仓库为 113Server 的 `~/dit-p2-generalizationprobe-20261109`。运行产物在 `~/a1_quality_20261108/P2_GeneralizationProbe`。完整仓库中，`runs_manifest.json` 记录逐阶段执行命令、代码 SHA、输入权限与产物路径，`split_manifest.json` 记录 prompt×seed×video 和特征及标签哈希。启动复跑需使用独立空输出根目录，或在隔离的复现主机复用路径布局。脚本以历史模块中的统一 ROOT 常量定位实验，若更换目录应在复现副本调整该常量或本轮 common.py 的 OUT。

Mac 精简包可在空目录解包，再将更新合并到笔记工作区。已经含有本地修改的总览或地图应只合并本轮新增章节。

```sh
ssh 113Server 'cd ~/dit-p2-generalizationprobe-20261109 && d=experiments/a1_quality_20261108/P2_GeneralizationProbe && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md "$d/README.md" "$d/decision.md" "$d/tables" "$d/figures"' | tar -xf -
```
