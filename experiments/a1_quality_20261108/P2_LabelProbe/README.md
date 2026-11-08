# P2-LabelProbe｜残差标签稳定性

完成冻结 A 审计、L0 损失 × 频带消融、L1 参考标签稳定性，以及条件触发的三臂 L2 小试。保留冻结 A，最终盲测封存。源辅助、固定样本记忆和独立学生的结果分别报告。

[阶段判定与统一回放](decision.md) · [环境审计](environment_audit.md) · [运行清单](runs_manifest.json) · [科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-labelprobe-20261108)

在 113Server 的本分支工作树执行。复现目录应是新目录。仅在 L0/L1 阶段条件满足后执行 L2。CSV 和图表依赖 Matplotlib。

```sh
s=experiments/a1_quality_20261108/P2_LabelProbe/src
i=experiments/a1_quality_20261108/P2_InfoProbe/scripts
py="$HOME/Wan2.1/.venv/bin/python"
r="$HOME/a1_quality_20261108/P2_LabelProbe/reproduction"
mkdir -p "$r"
CUDA_VISIBLE_DEVICES=0 "$py" "$i/audit.py" "$r/audit"
"$py" "$s/l0.py" "$r/l0_labels" --audit "$r/audit"
for seed in 1701 1702; do
  for scope in 0 1 multi; do
    for arm in A B C D; do
      CUDA_VISIBLE_DEVICES=0 "$py" "$s/l0.py" "$r/l0_labels"         --out "$r/l0-$scope-$arm-$seed" --arm "$arm" --seed "$seed" --scope "$scope"
    done
  done
done
"$py" "$s/l1.py" "$r/l1"
for seed in 1701 1702; do
  for arm in Sparse Continuous Region; do
    CUDA_VISIBLE_DEVICES=0 "$py" "$s/l2.py" "$r/l1" "$r/l2-$arm-$seed" --arm "$arm" --seed "$seed"
  done
done
"$py" "$s/aggregate.py" "$r" "$r/results"
"$py" "$s/distribution.py" "$r" "$r/results"
```

单视频无标签导出使用 `src/export.py hidden.pt head.pt output.npz --arm Continuous`。精简视频与同步命令见 [本地评审](../../../docs/local_review/README.md#p2-labelprobe)。
