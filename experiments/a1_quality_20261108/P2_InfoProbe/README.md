# P2-InfoProbe｜A/B 残差可学习性诊断

冻结 A 的 Mode/MV 和 Oracle I，在 832×480、16 fps、QP28、Baseline/CAVLC、单参考的真实 H.264 路径上比较残差输入。分开记录固定样本记忆、真实源辅助诊断和可部署输入学生的验证回放。验证沿用 16 条视频、4 个 prompt 等权。最终盲测保持封存。

- [环境与回放审计](environment_audit.md)
- [D0 固定样本记忆与修复](D0_overfit_report.md)
- [D1 逐臂真实参考交叉诊断](D1_position_value_report.md)
- [G0–G2 统一探针](G0_G2_comparison.md)
- [G3–G5 条件组](G3_G5_conditional_report.md)
- [阶段判定](decision.md)
- [运行清单](runs_manifest.json)

在 113Server 上从本分支工作树运行。各命令的输出目录须为空。硬件和历史路径见环境审计，训练参数见 YAML。

```sh
py="$HOME/Wan2.1/.venv/bin/python"
s=experiments/a1_quality_20261108/P2_InfoProbe/scripts
r="$HOME/a1_quality_20261108/P2_InfoProbe/reproduction"
mkdir -p "$r"
CUDA_VISIBLE_DEVICES=0 "$py" "$s/audit.py" "$r/audit"
CUDA_VISIBLE_DEVICES=0 "$py" "$s/d0.py" "$r/audit" "$r/d0"
CUDA_VISIBLE_DEVICES=0 "$py" "$s/repair.py" "$r/audit" "$r/d0" "$r/repair"
CUDA_VISIBLE_DEVICES=0 "$py" "$s/d1.py" "$r/audit" "$r/d1"
"$py" "$s/d1_summary.py" "$r/audit" "$r/d1" "$r/d1_summary.json"
"$py" "$s/g_probe.py" "$r/g_labels"
for seed in 1701 1702; do
  for group in G0 G1 G2; do
    CUDA_VISIBLE_DEVICES=0 "$py" "$s/g_probe.py" "$r/g_labels" \
      --group "$group" --seed "$seed" --out "$r/$group-$seed"
  done
done
```

源 YUV 仅用于打分或构造标签。D1 的 TT、TS、ST 在评分中使用真实源，单列为诊断。G2 使用冻结 A 轨迹的重建参考，作为离线条件对照，缺少在线参考接口及端到端成本验收。P0.6 教师码流也属于真实源辅助诊断。

[精选 MP4 与精简同步说明](../../../docs/local_review/README.md)
