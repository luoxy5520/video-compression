# 画质不变的等价 Skip

在 P16 量化后，若 CBP 为零、且 P16 MV 与标准 Skip MV 相同，则写 Skip。解码器复现同一预测块。四条验证与八条独立测试，两条策略共 24 个用例均与转换前逐像素一致。主策略为亚像素细化加等价 Skip，8 条独立测试为 40.92 dB、1283.54 kb/s。该方案无需 Oracle I 帧或输入原视频，使用冻结 Wan 解码主干、训练的输出投影和运动候选头。

服务器端的单文件运行示例：

```bash
cd /home/luoxiangyu/dit-paired-experiment
models=/mnt/data_hdd/models/dit_compression
/home/luoxiangyu/Wan2.1/.venv/bin/python latent_codec_20261107/encode.py \
  --latent /path/to/final_latent.pt \
  --projection "$models/residual_reference_20261107/rank40_projection.pt" \
  --decision "$models/decision_search_20261107/decision.pt" \
  --kit /home/luoxiangyu/senior-oracle-expand32-20261106 \
  --jm /home/luoxiangyu/jm-exact-skip-20261107 \
  --policy refine --exact-skip --output /home/luoxiangyu/my_latent_encode
```

输出为 `encoded/prediction.264`。输入须为固定 17 帧 832×480 的最终 latent，输出 QP 28。需要重建后端时，在 `jm-latent-codec-20261107` 的源树副本应用 `patch.py`，再在 `lencod` 下执行 `make CC="gcc -fcommon" -j8`。对照代码为 `run.py`，逐样本原始结果在 `/home/luoxiangyu/exact_skip_20261107/`。
