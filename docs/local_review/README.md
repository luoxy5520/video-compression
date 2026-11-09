# 本地精简评审

- 实验结论与复现索引见 [实验进展总览](../experiments/DiT_H264_实验进展总览.md)。
- 历史阶段与 P2-Lite 结果见 [实验地图](../../实验地图.html)。
- videos/ 下为 H.264 MP4。四宫格依次是源视频、冻结 A、P0.6 离线诊断、P2-Lite。每格 832×480，17 帧，16 fps。展示采用统一重编码，原始评分码流保存在科研产物分支。

| 片段 | 对比 |
| --- | --- |
| 固定机位节拍器 | [sample_1024](videos/fixed_camera_sample_1024.mp4) |
| 市场横移 | [sample_1032](videos/camera_motion_sample_1032.mp4) |
| 最佳逐视频变化 | [sample_1028](videos/best_sample_1028.mp4) |
| 最大逐视频退化 | [sample_1029](videos/worst_sample_1029.mp4) |

新建精简评审目录。部分克隆与稀疏检出仅取本页、视频、总览和实验地图。

```sh
git clone --filter=blob:none --no-checkout --single-branch \
  --branch codex/a1-p2-lite-20261108 \
  git@github.com:luoxy5520/video-compression.git P2-Lite-review
git -C P2-Lite-review sparse-checkout set --no-cone \
  '/实验地图.html' '/docs/experiments/DiT_H264_实验进展总览.md' '/docs/local_review/**'
git -C P2-Lite-review checkout
```

[复现代码](https://github.com/luoxy5520/video-compression/tree/codex/a1-p2-lite-20261108/experiments/a1_quality_20261108/P2_Lite) · [完整科研记录](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-lite-20261108)

## P2-InfoProbe

- [d1_closest_sample_0021](videos/p2_infoprobe/d1_closest_sample_0021.mp4)
- [d1_worst_sample_1025](videos/p2_infoprobe/d1_worst_sample_1025.mp4)
- [probe_G0-1701_R_sample_1030](videos/p2_infoprobe/probe_G0-1701_R_sample_1030.mp4)

四宫格为原始源、冻结 A、D1 TT 源辅助诊断及实际学生回放。视频 17 帧、16 fps，预览统一重编码。评测使用归档原始码流。

从 113Server 取精简版，先在一个空目录运行。只导出 Markdown、地图和本轮三个 MP4。

```sh
ssh 113Server 'cd ~/dit-p2-infoprobe-20261108 && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md docs/local_review/videos/p2_infoprobe experiments/a1_quality_20261108/P2_InfoProbe/README.md experiments/a1_quality_20261108/P2_InfoProbe/decision.md experiments/a1_quality_20261108/P2_InfoProbe/environment_audit.md experiments/a1_quality_20261108/P2_InfoProbe/D0_overfit_report.md experiments/a1_quality_20261108/P2_InfoProbe/D1_position_value_report.md experiments/a1_quality_20261108/P2_InfoProbe/G0_G2_comparison.md experiments/a1_quality_20261108/P2_InfoProbe/G3_G5_conditional_report.md' | tar -xf -
```

[代码与报告](https://github.com/luoxy5520/video-compression/tree/codex/a1-p2-infoprobe-20261108/experiments/a1_quality_20261108/P2_InfoProbe) · [科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-infoprobe-20261108)

## P2-LabelProbe

- [best_sample_1032](videos/p2_labelprobe/best_sample_1032.mp4)
- [training_fit_sample_0000](videos/p2_labelprobe/training_fit_sample_0000.mp4)
- [worst_sample_1025](videos/p2_labelprobe/worst_sample_1025.mp4)

对比视频每格 832×480，17 帧，16 fps。验证示例按原型逐视频最佳和最差变化选取，四格为源、冻结 A、真实源辅助诊断和实际学生。训练示例对照同样本 L0-A/L0-B。预览经过统一重编码，原始评分码流在[科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-labelprobe-20261108)。

在 Mac 的空目录仅同步地图、Markdown 和三个 MP4。

```sh
ssh 113Server 'cd ~/dit-p2-labelprobe-20261108 && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md docs/local_review/videos/p2_labelprobe experiments/a1_quality_20261108/P2_LabelProbe/README.md experiments/a1_quality_20261108/P2_LabelProbe/decision.md' | tar -xf -
```

## P2-OnPolicy

- [best_sample_1033](videos/p2_onpolicy/best_sample_1033.mp4)
- [motion_sample_1028](videos/p2_onpolicy/motion_sample_1028.mp4)
- [worst_sample_1025](videos/p2_onpolicy/worst_sample_1025.mp4)

四宫格依次为原视频、冻结 A、T0 学生和 T1 学生。各格 832×480，17 帧，16 fps。所有图格统一重编码预览。教师上界属于源辅助诊断，见报告。

在 Mac 的空目录中执行精简同步。

```sh
ssh 113Server 'cd ~/dit-p2-onpolicy-20261108 && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md docs/local_review/videos/p2_onpolicy experiments/a1_quality_20261108/P2_OnPolicy/README.md experiments/a1_quality_20261108/P2_OnPolicy/decision.md experiments/a1_quality_20261108/P2_OnPolicy/label_stability.md' | tar -xf -
```

## P2-MultiSampleAudit

- [closest_or_best_sample_1029](videos/p2_multisample_audit/closest_or_best_sample_1029.mp4)
- [worst_sample_1025](videos/p2_multisample_audit/worst_sample_1025.mp4)

四宫格为原片、冻结 A、M2 学生和 O0 源辅助诊断。各格 832×480、17 帧、16 fps，统一重编码预览。严格评分码流见科研归档。

在 Mac 的空目录中执行精简同步。

```sh
ssh 113Server 'cd ~/dit-p2-multisample-audit-20261108 && d=experiments/a1_quality_20261108/P2_MultiSampleAudit && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md docs/local_review/videos/p2_multisample_audit "$d"/*.md' | tar -xf -
```

## P2-CapacityProbe

- [best_or_closest_sample_0021](videos/p2_capacityprobe/best_or_closest_sample_0021.mp4)
- [worst_sample_1024](videos/p2_capacityprobe/worst_sample_1024.mp4)

四宫格为原片、冻结 A、C0 学生和 O0 源辅助诊断。各格 832×480，17 帧，16 fps，统一重编码预览。严格评分码流见科研归档。

在 Mac 的空目录运行：

```sh
ssh 113Server 'cd ~/dit-p2-capacityprobe-20261109 && d=experiments/a1_quality_20261108/P2_CapacityProbe && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md docs/local_review/videos/p2_capacityprobe "$d"/*.md' | tar -xf -
```

## P2-GeneralizationProbe

本轮按开发池诊断门槛停止在冻结 A 核验，未生成新学生回放或新 MP4。代表性回放仍见上节 P2-CapacityProbe 两个视频。精简同步命令、关键图及分层数表见本轮 README。

[本轮精简交付](../../experiments/a1_quality_20261108/P2_GeneralizationProbe/README.md)

| 保留的历史 MP4 | SHA-256 |
| --- | --- |
| best_or_closest_sample_0021.mp4 | `9f670c9c23975da21be44fd8c1df2d2cf911ab30cb3ad755d198c3d561d5097b` |
| worst_sample_1024.mp4 | `9a161bf3471486e94eb43242d0b2e610fa28c5c1bb3cb9bc4d11ac4c8aaee1cc` |

## 三状态消融与视频可视化

<a id="a1-three-state"></a>

[六路与 A/E 同步合辑](videos/a1_three_state/a1_three_state_comparison.mp4) · [实验判定](../../experiments/a1_quality_20261108/A1_ThreeState/decision.md)

合辑为 16 fps、102 帧、6.375 秒，每段 17 帧。各格显示分组、原片帧号（0 为共同 I 帧）、P 帧 Y-PSNR、码率及末四帧 PSNR。

| 视频 | 六路起始 | A/E 起始 | 单路 MP4 |
|---|---:|---:|---|
| 市场横移 1032 | 0.0000 s | 1.0625 s | [A](videos/a1_three_state/sample_1032_A.mp4) · [E](videos/a1_three_state/sample_1032_E.mp4) · [F](videos/a1_three_state/sample_1032_F.mp4) |
| 固定机位 1024 | 2.1250 s | 3.1875 s | [A](videos/a1_three_state/sample_1024_A.mp4) · [E](videos/a1_three_state/sample_1024_E.mp4) · [F](videos/a1_three_state/sample_1024_F.mp4) |
| 历史最差 1029 | 4.2500 s | 5.3125 s | [A](videos/a1_three_state/sample_1029_A.mp4) · [E](videos/a1_three_state/sample_1029_E.mp4) · [F](videos/a1_three_state/sample_1029_F.mp4) |

单路为 H.264 直接封装。对比合辑重编码以叠加同步标签和指标。Mac 上用 AVFoundation 和 FFmpeg 逐帧验证了 10 个 MP4，AVFoundation 输出时间戳与 16 fps 帧序一致。全部单路、独立六路和 A/E 片段存放在 113Server 的 `~/a1_quality_20261108/A1_ThreeState/report`。

```sh
ssh 113Server 'cd ~/dit-a1-three-state-20261109 && git archive HEAD -- 实验地图.html docs/experiments/DiT_H264_实验进展总览.md docs/local_review/README.md docs/local_review/videos/a1_three_state experiments/a1_quality_20261108/A1_ThreeState/decision.md' | tar -xf -
```
