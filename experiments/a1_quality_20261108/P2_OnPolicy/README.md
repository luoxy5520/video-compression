# P2-OnPolicy｜残差参考轨迹对齐

[阶段判定](decision.md) · [预检](preflight.md) · [锁定协议](protocol.md) · [O0 标签稳定性](label_stability.md) · [科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-onpolicy-20261108)

在 113Server 工作树中执行，使用新运行目录。根据 O0 门槛再执行 O1。

```sh
py="$HOME/Wan2.1/.venv/bin/python"
s=experiments/a1_quality_20261108/P2_OnPolicy/scripts
i=experiments/a1_quality_20261108/P2_InfoProbe/scripts
r="$HOME/a1_quality_20261108/P2_OnPolicy/reproduction"
mkdir -p "$r"
CUDA_VISIBLE_DEVICES=0 "$py" "$i/audit.py" "$r/audit"
"$py" "$s/preflight.py" "$r"
"$py" "$s/o0.py" "$r/o0_smoke" --audit "$r/audit" --smoke
"$py" "$s/o0.py" "$r/o0" --audit "$r/audit"
"$py" "$s/o1.py" "$r/o0" --gate
for seed in 1701 1702; do
  for arm in T0 T1; do
    for index in 0 1; do
      CUDA_VISIBLE_DEVICES=0 "$py" "$s/o1.py" "$r/o0" --arm "$arm" --seed "$seed"         --fixed "$index" --out "$r/fit-$index-$arm-$seed"
    done
    CUDA_VISIBLE_DEVICES=0 "$py" "$s/o1.py" "$r/o0" --arm "$arm" --seed "$seed" --out "$r/o1-$arm-$seed"
  done
done
CUDA_VISIBLE_DEVICES=0 "$py" "$s/diagnose.py" "$r" "$r/diagnostics"
"$py" "$s/aggregate.py" "$r" "$r/results"
```

[Mac 精简同步与 MP4](../../../docs/local_review/README.md#p2-onpolicy)
