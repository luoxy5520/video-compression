# 可运行的最终 latent 编码器

模型从冻结 Wan 主干的源代理提取 27 维块特征，并对半像素和四分之一像素方向分别排序。用 20 条视频的真实闭环教师轨迹训练，两层 64 宽隐藏层，合计 7,122 参数。4 条验证视频锁定每阶段 4 个方向。8 条独立测试视频的 P 帧亮度为 40.89 dB，码率 1289.11 kb/s，最终 latent 到码流平均 2.71 s。和全方向对照相比每块评估从 22 次降到 14 次，平均亮度少 0.031 dB，码率多 0.43%。所有 52 个训练轨迹/评测用例均通过编码器、JM 和 FFmpeg 重建一致性验证。

使用冻结 Wan 解码、秩 40 投影、整数候选头、亚像素头和 JM 闭环。运行只读最终 latent 和模型权重，I 帧由同一源代理编码。支持 832×480、17 帧、16 fps、QP 28、单参考。该版本在项目固定的生成域测试上满足质量和码率工作线。测试集包含旗帜与舞者两个 prompt，尚需扩大内容覆盖和支持其他尺寸。

服务器单文件示例：

```bash
cd /home/luoxiangyu/dit-paired-experiment
models=/mnt/data_hdd/models/dit_compression
/home/luoxiangyu/Wan2.1/.venv/bin/python latent_codec_20261107/encode.py \
  --latent /path/to/final_latent.pt \
  --projection "$models/residual_reference_20261107/rank40_projection.pt" \
  --decision "$models/decision_search_20261107/decision.pt" \
  --qpel "$models/learned_qpel_20261107/qpel.pt" --qpel-budget 4 \
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
  --jm /home/luoxiangyu/jm-learned-qpel-20261107 \
  --policy refine --exact-skip --output /home/luoxiangyu/my_latent_encode
```

输出 `encoded/prediction.264`。如需重建后端，在等价 Skip 的 JM 源树副本应用本目录的 `patch.py`，再执行 `make -C "$jm/lencod" CC="gcc -fcommon" -j8`。训练、验证及评估由 `bash learned_qpel_20261107/run.sh` 复现。运行数据和日志在 `/home/luoxiangyu/learned_qpel_20261107/`，权重在 `/mnt/data_hdd/models/dit_compression/learned_qpel_20261107/`。
