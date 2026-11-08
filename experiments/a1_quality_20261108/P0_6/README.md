# P0.6 固定 Mode/MV 的残差率失真可行性

沿用 P0.5 的 A 状态、44 条 train、16 条 val 和 prompt 隔离。保持 QP28、固定 Oracle I、Skip 零残差和单参考。最终盲测未打开。每条视频保存 `fixed_mode_mv.npz`，含原始预测 Mode/MV 和实际 JM 派生后的逐块 MV。所有严格回放逐值验证实际 Mode/MV，包括 Skip。

## 预算与闭环

离线教师在每个 P 帧开始时将上一帧的真实 JM 重建发给预算分配器。分配器用固定 A MV 做运动补偿，并以当前真实源减 MC 作为诊断残差。选定的系数预算随帧更新，所用参考由各预算策略自己的前序状态产生。逐帧交换的参考与编码结束后的重建逐像素核验。

每个候选 4×4 亮度块最多保留一个 H.264 整数变换基。对全部 16 个频率估计投影能量、幅度和比特，按局部失真收益除以估计比特排序。用该基构造像素域残差，JM 按原样量化，记录实际非零系数数目。稀疏策略中未选亮度系数和色度残差为零。A 原状态和全零 P16 残差分别保留为锚点。

逐视频的总流目标为 A 的 0%、+1%、+2%、+3%、+5% 额外比特。三种帧分配为均匀、半衰期六个 P 帧的早期倾斜，以及用 A 固定参考下离线收益估计确定帧权重。选定的帧权重固定，帧内系数与幅度仍按该策略真实参考更新。每个点最多三次码率试算，按实际总流比特选取未超预算且最接近目标的试算。没有命中时保留预算内锚点。率试算 CSV 记录命中和偏差。

严格回放关闭离线交换，将全部 P 帧残差预导出后再送给 JM。验证码流哈希与教师一致，三方重建逐像素一致，后端 ME 时间为 0.000 s。后端输入为零占位，真实源只用于离线教师和评分。RD 表中的额外比特来自实际完整码流，非零系数和残差语法比特来自 JM 的原生计数。

这是单基稀疏亮度残差家族的可行点扫描，使用真实源，不能当作 hidden 学生结果。候选排序与 CAVLC 成本是局部估计，评价以真实回放为准。频率表统计所选变换基，原生非零系数另列。单位比特收益为等权 prompt 的总 P 帧 Y SSE 改善除以额外整段码流比特，额外比特为零或负时留空。

## 复现

在 113Server 仓库运行，使用 P0.5 已验证的 JM 副本。

```bash
p=experiments/a1_quality_20261108/P0_6
py="$HOME/Wan2.1/.venv/bin/python"
r="$HOME/a1_quality_20261108"
mkdir "$r/P0_6"
cp -a "$r/P0_5/jm" "$r/P0_6/jm"
"$py" "$p/patch_teacher.py" "$r/P0_6/jm"
make -C "$r/P0_6/jm/lencod" -j8 \
  CFLAGS="-fcommon -std=gnu99 -pedantic -ffloat-store -fno-strict-aliasing -fsigned-char"
"$py" "$p/run_batch.py" --output "$r/P0_6/full"
"$py" "$p/report06.py" --run "$r/P0_6/full" --output "$r/P0_6/report"
```

结果在 `results/`。其中 `fixed_A_states.tar.gz` 打包 60 条视频的固定状态和帧权重，`per_frame.csv` 保存目标预算、估算分配、实际残差系数比特和重建指标，`rate_trials.csv` 保存所有预算试算的实际总比特。参见 [阶段判定](decision.md) 和 [真实 RD 曲线](results/rd_curve.svg)。
