# 视频归档

服务器保存目录为 `/home/luoxiangyu/lmp-continuation-20261122/videos/`。将已有 H.264 码流无损封装为 MP4，便于查看原生 C0、固定融合 C2 和自适应 C2。实际文件路径、源指纹与对应码流见 `../results/runs.csv`、`../results/repeats.jsonl`。正式指标使用独立解码的原始像素与配对的 VAE 源帧计算。

- `sample_0000_C0_vs_C2-adaptive.mp4` 为画质通过、链路变慢的自适应组。
- `sample_1004_C0_vs_C2-fixed.mp4` 为高运动样本上的固定预算退化。
- 两条并排视频左侧为原生 C0，右侧为对应方法。试点没有达到整体门槛的收益样本。
