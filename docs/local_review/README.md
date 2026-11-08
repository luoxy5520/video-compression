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
