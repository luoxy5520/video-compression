# 无 Oracle I 的最终 latent 编码

输入为最终 latent，格式为 16×5×60×104。用固定 Wan 解码主干、训练好的秩 40 投影和 MV 决策头生成源代理与 P 帧候选。JM 以代理源编码 I 帧，在实际解码参考上从有限候选中选择 P16 MV。残差使用代理与当前运动补偿的差。推理入口不加载真值 YUV 或 Oracle B1 文件。当前支持 832×480、17 帧、16 fps、QP 28、单参考、Skip/P16。

补丁在 `jm-skip-calibration-20261107` 源树上应用。验证集为 `sample_0020–0023`，独立测试为 `sample_0024–0031`。完整实验由 `bash latent_codec_20261107/run.sh` 复现，默认输出在 `/home/luoxiangyu/latent_codec_20261107`。输出路径须为新目录。

单条无评分编码由 `encode.py` 执行。它需要最终 latent、秩 40 投影、候选头、JM 运行包和二进制路径。选择 `--policy refine` 可复现独立测试的 40.92 dB / 1347.82 kb/s 对照。生成文件为 `encoded/prediction.264`，编码时延在 `encoding.json`。

```bash
jm=/home/luoxiangyu/jm-latent-codec-20261107
cp -a /home/luoxiangyu/jm-skip-calibration-20261107 "$jm"
python3 latent_codec_20261107/patch.py "$jm"
make -C "$jm/lencod" CC="gcc -fcommon" -j8
bash latent_codec_20261107/run.sh
```
