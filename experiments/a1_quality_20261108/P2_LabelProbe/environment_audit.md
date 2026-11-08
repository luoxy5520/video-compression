# 环境和历史证据复核

113Server 的 InfoProbe 工作树 clean，GitHub 代码提交为 `f5e02b7126565f3a92b4a21ad6383491c494d603`，归档为 `40040341680744b47f4781882eda2ec92ed37d6d`。P2-Lite 代码与归档分别为 `387cdb2d504e04f1810a0cd913ffccb8edbc33a3` 和 `6e875a677dff9f40da3168b8855e08aad09f21cc`。本轮从 InfoProbe 切出 `~/dit-p2-labelprobe-20261108`，数据输出独立放在 `~/a1_quality_20261108/P2_LabelProbe`。两张 A100 80GB，开工时 home 剩余约 7.2 TB。

Python 3.12、PyTorch 2.7.1+cu128、CUDA 12.8、NumPy 1.26.4。JM 19.0 FRExt，使用 `~/a1_quality_20261108/P0_6/jm` 和 `~/a_new48_20261107/kit`，FFmpeg 6.1.1。冻结 A checkpoint 位于 `/mnt/data_hdd/models/dit_compression/a1_quality_20261108/locked_hidden.pt`，SHA-256 为 `b713fef90db9205471c7b168e1715c70d8a4c87a0f4b342f866baf04e1cd4ca5`。

原始 conditional 末块 hidden 形状 [1, 7800, 1536]，冻结输入形状 [1, 67, 16, 30, 52]。历史训练/验证 44/16 条、11/4 个互斥 prompt。使用 P2-Lite 的 `run01/features` 和 `run01/labels`，通过 P0.6 源辅助码流为本轮 24 条重新抽取 16 频带 CAVLC level，三低频缓存逐项一致。YUV420 832×480，宏块为 30×52，原生 qpel x/y。4×4 块与系数均用行列光栅序，适配器从 JM 子块顺序和 zigzag run 还原。逆向映射合成整数像素基，再由 JM QP28 整数变换量化，检查实际 level、Mode/MV、Skip 零残差和三方重建。

冻结 A 16 条验证逐条复现历史码流哈希，样本 0000 的标签哈希和缓存特征逐项复现，320 个符号系数的适配 smoke 通过。L0 首次两次训练完成后，重复 budget 字段导致指标写入失败。修正后从同种子重跑，旧日志和部分产物保存在归档。

原始 A1 验收以独立状态生成后的标准码流、RD、尾帧、漂移和成本为准。实验地图 05、06、09、13 节中含真实源、Wan 解码代理或局部搜索的路径作历史辅助证据。15 参数校正曾改善浮点误差但码流不变。InfoProbe D0 的 34.276 dB / 186.519 kb/s / 30.549 dB 来自训练样本 0000 固定记忆，A 为 33.789 / 181.135 / 30.074。D1 第一字母是位置，第二字母是值，TS −0.0388 dB，ST +0.0075 dB。P0.6 +3% 与 P2-Lite 历史验证结果见 [实验总览](../../../docs/experiments/DiT_H264_实验进展总览.md)。
