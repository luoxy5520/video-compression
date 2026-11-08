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
