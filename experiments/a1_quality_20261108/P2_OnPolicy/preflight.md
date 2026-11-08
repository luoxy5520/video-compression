# 预检与复现

113Server 原 LabelProbe 工作树 clean，从 `f274bb289f774f4b23f3274bcede428873c60dac` 创建独立工作树 `~/dit-p2-onpolicy-20261108`。P2-LabelProbe / P2-InfoProbe / P0.6 代码远端分别为 `f274bb289f774f4b23f3274bcede428873c60dac`、`f5e02b7126565f3a92b4a21ad6383491c494d603`、`a8634bd898a8d451d6dc47e5a89dc2b4a96e771b`。LabelProbe 与 InfoProbe 归档分别为 `a0ee8f56b788f4fd2706e2c7e5874073c1f5de8a`、`40040341680744b47f4781882eda2ec92ed37d6d`。

核验 LabelProbe 30 套权重与指标、24 个全频标签和 L1 标签哈希，复算中位 IoU 为 0.127775 / 0.828945，六对重复标签和码流均吻合。InfoProbe 两个 D0 头、D1 原始 JSON、12 套 G 权重与指标及 60 个 G 标签哈希通过。P0.6 固定状态、T0 原始码流和历史低频缓存哈希检查覆盖 60 条。P0.6 源辅助 +3% 报告值 20.825912 dB / 452.390 kb/s，与 O0 固定 64 点的教师参考版本分开。

冻结 A 权重位于 `/mnt/data_hdd/models/dit_compression/a1_quality_20261108/locked_hidden.pt`，SHA-256 `b713fef90db9205471c7b168e1715c70d8a4c87a0f4b342f866baf04e1cd4ca5`。GPU 为两张 A100 80 GB。PyTorch 2.7.1+cu128、CUDA 12.8、NumPy 1.26.4、JM 19.0 FRExt、FFmpeg 6.1.1。历史输入见 `P2_Lite/run01`，教师源与状态见 `P0_6/full`。JM 与 kit 沿用 `~/a1_quality_20261108/P0_6/jm` 及 `~/a_new48_20261107/kit`。开工时 home 剩余 7.2 TB。

44 train / 16 val 分属 11 / 4 个互斥 prompt，最终盲测未访问。冻结输入形状 [1, 67, 16, 30, 52]。YUV420 832×480，17 帧，QP28、Baseline/CAVLC、Oracle I、单参考。P16 块按 30×52 宏块和每块 16 个 4×4 光栅子块索引，频率为 16 个光栅位，适配器读取原生 CAVLC zigzag 并将 level 反合成为像素基。Skip 零残差和原生派生 MV。冻结 A 验证逐条重现历史码流，编码器与两个标准解码器逐帧 SHA 相同。ME 日志 0.000 s，未加入独立调用计数器。

任务书 SHA-256 为 `0c0c7c33fdefd0507d610b4fb2856b7721d633593e12260a1210b42c97526522`。实验地图 05/06/09/13 的历史高画质包含真实源、解码代理或局部搜索。本轮采用独立学生固定状态和真实闭环回放口径。
