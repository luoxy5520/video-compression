# 三状态消融与视频可视化

2026-11-09 · 113Server · 历史 16 条验证视频、4 个 prompt、每个 prompt 4 个 seed。先完成全量评分，再制作市场横移（1032）、固定机位（1024）、冻结 A 历史最差（1029）的预览。

## 冻结对象与运行协议

- 三个 Head 由同一冻结 hidden checkpoint 一次前向生成。权重为 `/mnt/data_hdd/models/dit_compression/a1_quality_20261108/locked_hidden.pt`，SHA-256 为 `b713fef90db9205471c7b168e1715c70d8a4c87a0f4b342f866baf04e1cd4ca5`，与原训练目录的 `model/state.pt` 一致。历史代码发布提交为 `f88ed877fb08d1c568372b0c746ae6de1a44b497`，本轮运行提交为 `67a871ba81ff89dba1a7d8f56083d2062497ca4e`。
- 历史配置为条件末块 hidden、非循环、宽度 64、594,630 参数，训练命令为 `state.py --kind hidden --config expansion_20261106/config_train32.json ~/a_new48_20261107/kit/state_config.json --epochs 24`。使用 176 个训练记录、64 个验证记录，保存第 21 轮验证损失最优的联合权重，种子 4242，AdamW，学习率 0.001。回放仅使用 UniPC step 20 的缓存 hidden `[1,7800,1536]`，未训练新权重，P3 与最终盲测保持封存。
- Mode 为 `uint8[16,30,52]`，0 是 Skip，1 是 P16×16。MV 为 `int16[16,30,52,2]`，x/y 为四分之一像素单位。残差为量化前有符号像素差，Y/U/V 分别为每块 16×16、8×8、8×8。每组从同一个 Oracle I 起步，832×480、16 fps、1I+16P、Baseline/CAVLC、QP28、单参考、历史配置关闭去块滤波。
- 每条视频各组独立滚动重建参考。A–E 的 Skip 由 JM 按邻块派生 MV，载荷适配明确置零、没有自由残差。B/C 的邻块变化可传到后续 Skip 的原生 MV，属于标准解码语义。所有 P16 的实际 MV 与交给 JM 的值逐块核对。
- A 先核对新推理的 Mode/MV/YUV 残差与历史导出逐元素一致，再核对 16 条码流逐字节一致。F 是同配置下完整 JM 源输入编码，允许搜索，重新生成的状态、码流和重建逐条复现历史 Oracle。
- A–E 完整覆盖 26,520 个宏块/视频。JM replay 分支直接应用状态并写码流，绕过常规宏块编码入口，关闭 RDOQuant，每次日志 ME 计时均为 0.000 s。F 的 ME 平均 41.924 s/条。96 个评分码流和 16 次 E 离线验证均由编码器重建、JM 解码、FFmpeg 解码三方逐字节核验，包含所有帧与 Y/U/V 三个平面。首帧各组一致。

| 组 | 实际替换及合法性适配 |
|---|---|
| A | 原始预测 Mode/MV/Residual，冻结基准。 |
| B | Oracle Mode。47,451 个预测 P16 改为 Skip，其自由 MV/残差随 Skip 语法失效。10,018 个预测 Skip 改为 P16，从同一次冻结前向的完整输出中取该块 MV 与残差。 |
| C | 预测和 Oracle 同为 P16 时替换 qpel MV，共 221,937 块，其中 212,223 块数值实际改变。预测 P16 中余下 47,451 块保留预测 MV。残差继续使用 A 的原输出。 |
| D | 同一批 221,937 块替换原始 Oracle YUV 像素残差。Oracle Skip 没有同义的 P16 残差，预测 P16 中余下 47,451 块保留预测残差。A、D 的实际 MV 逐块相同。 |
| E | 保留 A 的 Mode/MV。对每个 P16 用 `源像素 − 本组实际参考经 JM MC 得到的预测像素`，再走原生变换、量化与 CAVLC。Skip 保持零残差。导出实际 MC 并物化该残差，重新走无源输入的普通回放，16/16 码流与 YUV 完全一致。 |
| F | 完整 JM 编码参考，使用真实源自行选择 Skip/P16、MV 和残差。 |

冻结 Residual Head 内部读取同次前向的连续预测 MV，条件替换发生在导出的状态层。C 的 Residual 没有因 Oracle MV 而重新推理。E 需要真实 P 帧，只作为源辅助诊断。六组能比较单项替换及完整 JM 的联合结果，缺少 Oracle MV × 匹配源残差的交叉臂，因此无法把 E 对 F 的差距单独分配给 MV、Mode 或二者的交互项。

## 全验证集结果

P 指标按每条视频对 16 个 P 帧池化 Y-MSE 后换算 PSNR，再对 16 条等权平均。末四帧和漂移使用逐帧 PSNR 均值。码率用整条 Annex B 码流字节数（含 I 帧与头部）和 `17/16 s` 计算。时间为 113Server 实测均值，单位为秒/视频，wall 含适配、JM、双解码、评分和本组状态核对，不含共享推理和额外的 E 验证回放。

| 组 | P Y-PSNR dB | ΔA dB | kb/s | 末四帧 dB | 末帧 dB | 末四减首四 P dB | 最差样本与 P dB | JM / wall s |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A | 20.408 | 0.000 | 439.514 | 18.810 | 18.579 | −5.315 | 1029 · 13.717 | 0.894 / 3.031 |
| B | 20.672 | +0.264 | 425.463 | 19.058 | 18.818 | −5.217 | 1029 · 13.724 | 0.921 / 2.870 |
| C | 25.255 | +4.847 | 477.949 | 23.181 | 22.851 | −6.182 | 0023 · 17.026 | 0.940 / 2.876 |
| D | 20.426 | +0.018 | 2305.099 | 18.840 | 18.593 | −5.764 | 1029 · 14.112 | 0.981 / 3.784 |
| E | 35.786 | +15.377 | 3747.962 | 35.785 | 35.687 | −0.473 | 1029 · 30.264 | 1.271 / 3.189 |
| F | 37.161 | +16.752 | 2326.024 | 36.646 | 36.638 | −1.158 | 0022 · 33.533 | 43.505 / 44.781 |

全量回放的总墙钟时间为 263.083 s，四路视频任务并发，含 E 验证与输入哈希，独立于之后的封装和归档。A100 上每条共享 hidden 读取和传输平均 55.26 ms，三个 Head 与共享主干前向平均 3.46 ms。逐视频 CSV 的 A–E 推理计时引用同一次共享推理。曲线在 0 帧相交，随后 A、B、D 持续下滑。C 全部 16 条优于 A，末四帧高 4.372 dB，但从自身早期优势下降 6.182 dB。E 末四帧仅比首四 P 帧低 0.473 dB，短片中漂移大幅减轻。1029 的 E 在第 4 帧掉至 22.49 dB，第 8 帧回升至 35.34 dB，F 同两帧为 39.46、37.86 dB。

[逐视频指标](https://github.com/luoxy5520/video-compression/blob/artifacts/a1-three-state-20261109/per_video.csv) · [逐帧指标](https://github.com/luoxy5520/video-compression/blob/artifacts/a1-three-state-20261109/per_frame.csv) · [平均曲线](https://github.com/luoxy5520/video-compression/blob/artifacts/a1-three-state-20261109/psnr.svg) · [平均逐帧 CSV](https://github.com/luoxy5520/video-compression/blob/artifacts/a1-three-state-20261109/mean_curve.csv) · [实际 MC 对齐](https://github.com/luoxy5520/video-compression/blob/artifacts/a1-three-state-20261109/alignment.json) · [运行清单](https://github.com/luoxy5520/video-compression/blob/artifacts/a1-three-state-20261109/run_manifest.json)

## 画面中的差别

**Mode（A 对 B）**。更换 Skip/P16 在固定机位 1024 略稳住支架等细线并减少码率，120.8 降至 100.4 kb/s。市场横移 1032 中右侧棚柱拉扯和滑手 1029 的大块错位仍在。B 的平均收益为 0.264 dB，最差配对变化是 1031 的 −0.012 dB。固定预测 MV 与残差时，改 Mode 的影响主要限于哪些块使用显式运动和更新纹理。

**MV（A 对 C）**。市场右侧棚架的几何拉伸减轻，道路及摊位边线更接近 F。1029 的大范围拖抹减轻，柱边和遮挡区仍可见密集块错位。共同 P16 上，相对真实源的实际 MC 误差从 35.60 降到 20.56 个 Y 灰度级 RMSE，码率仅增 8.7%。由于闭环参考此前已经受损，正确方向的局部向量仍会从退化参考取样。C 的预测残差无法补回这些块中的细节，尾段仍在下降。

**原始 Residual（A 对 D）**。历史残差对应 `源 − Oracle MC`，把它加到 D 的实际 MC 后，量化前仍留下两套 MC 的差。共同 P16 上，D 的实际所需 Y 残差与写入值的平均 RMSE 为 35.45 灰度级，接近 A 的 35.60。市场棚架出现斑驳彩纹和重复边缘，1029 柱面和前景叠出亮暗错边。D 仅 7/16 条优于 A，0023 降 0.763 dB，平均码率升至 A 的 5.24 倍。系数描绘的是另一条参考轨迹上需要校正的纹理，错位时会叠加到错误位置。

**匹配 Residual（D 对 E）**。1029 中柱面刻纹和滑手轮廓重新出现，市场屋檐和灯带恢复连贯。共同 P16 上，E 物化的残差与该组需要的残差量化前完全一致，QP28 和仍然使用 Skip 的块保留了实际误差。预测 MV 可以由足量的匹配残差补偿，画质相对 D 上升 15.359 dB，码率从 2305.099 增至 3747.962 kb/s。E 的码率为 A 的 8.53 倍、F 的 1.61 倍，P PSNR 仍比 F 低 1.375 dB。E 平均写入约 340 万残差系数语法 bit/视频，D 为 189 万，A 约 366 bit。恢复依赖真实源像素及本组实时重建参考，代价主要落在系数码率及源输入权限上。F 能利用编码搜索选择更容易校正的状态，代价是更长的编码时间。

## 复现与产物

```bash
ssh 113Server
cd ~/dit-a1-three-state-20261109
d=experiments/a1_quality_20261108/A1_ThreeState
o="$HOME/a1_quality_20261108/A1_ThreeState/repro"
mkdir "$o"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  ~/Wan2.1/.venv/bin/python -u "$d/run_ablation.py" "$o/full" > "$o/full.log" 2>&1
~/Wan2.1/.venv/bin/python "$d/diagnose_alignment.py" "$o/full" "$o/alignment.json"
python3 "$d/report.py" "$o/full" "$o/report"
```

评分码流、标准解码 YUV、每组日志、NPZ、原生轨迹及重放命令在 113Server 的 `~/a1_quality_20261108/A1_ThreeState/full`。校验归档在同级 `publish`，按样本分卷。单路 MP4 只重写封装和 16 fps 时间戳，18 条精选单路解码像素与评分码流一致。六宫格和 A/E 双路需要叠字，统一 H.264 CRF 14 重编码，只作预览。Mac 同步 9 条精选单路（每条样本的 A/E/F）及一条对比合辑，完整指标留在服务器与 GitHub。

[Mac 预览与时间索引](../../../docs/local_review/README.md#a1-three-state) · [GitHub 科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/a1-three-state-20261109)
