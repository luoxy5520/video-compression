# 环境、标签与回放核验

113Server 的原工作树位于 `~/dit-paired-experiment`，原分支为 `codex/a1-p0-6-20261108`。本轮使用 `~/dit-p2-infoprobe-20261108`，从核验后的 GitHub 代码提交 `387cdb2d504e04f1810a0cd913ffccb8edbc33a3` 切出。历史科研归档远端提交为 `6e875a677dff9f40da3168b8855e08aad09f21cc`。历史 `a_state_pilot_20261107`、`jm-direct-replay-20261107`、`senior-oracle-expand32-20261106` 和 `learned_qpel_20261107` 均定位成功。Qpel 历史权重位于 `/mnt/data_hdd/models/dit_compression/learned_qpel_20261107/qpel.pt`。

| 项目 | 实测 |
| --- | --- |
| 运行库 | `~/Wan2.1/.venv`，Python 3.12，PyTorch 2.7.1+cu128，CUDA 12.8，NumPy 1.26.4 |
| 设备 | 两张 NVIDIA A100 80GB PCIe，Linux 6.17.0-29-generic |
| 编码与解码 | JM 19.0 FRExt，`~/a1_quality_20261108/P0_6/jm/bin/{lencod,ldecod}.exe`，FFmpeg 6.1.1-3ubuntu5 |
| 回放适配器 | `~/a_new48_20261107/kit/scripts/prediction_to_b1_state.py`，`linux/skip_p16.cfg`，`JM-backend/cfg/encoder_baseline.cfg` |
| 训练、验证和特征 | 44 训练、16 验证，11/4 个互斥 prompt，UniPC step 20 conditional 末块 |
| 形状 | hidden `[1,7800,1536]`，5×30×52 token 网格；冻结回放头输入 `[1,67,16,30,52]`，float32 |
| H.264 状态 | 832×480，17 帧，YUV420，P 帧 30×52 宏块，Skip=0，P16=1，MV 是 x/y 原生 qpel |
| 系数 | 有符号 QP28，4×4 频率以行列光栅编号。用 JM 子块索引和 CAVLC zigzag run 还原。Skip 无系数 |
| 权重 SHA-256 | `b713fef90db9205471c7b168e1715c70d8a4c87a0f4b342f866baf04e1cd4ca5` |
| sample_0000 原始 hidden SHA-256 | `fa8a0929022caa838dc116d0771c86dbd4eed2a325e61e56ddd59ef407b4bb45` |
| sample_0000 新生成的低频标签 SHA-256 | `88df109877e230fb7bab5501863305824cb9f98dc19920d1bb32ad2e6dc9f660` |

16 条验证 A 码流逐条与历史 SHA-256 相同。四 prompt 等权为 20.408240 dB、439.513882 kb/s、末四帧 18.809713 dB。P 帧 Y-PSNR 先按整段池化 SSE，末四帧取逐帧 Y-PSNR 算术均值。样本 `sample_0000` 的教师码流和低频标签哈希重现，冻结特征与历史缓存逐元素相等。第一次审计比较 Python 整数键与历史 JSON 字符串键，归一化序列化后重跑通过。日志保留两次记录。

适配器 smoke 逐项往返 320 个有符号系数，DC/H/V 最大幅度为 63/50/50，规避孤立幅值 1 被 JM 丢弃的边界。审计检查三方重建一致、Oracle P 读取 0、完整覆盖 26,520 个状态记录、JM 日志 ME 为 0.000 s，并核对 Mode 和原生 MV。独立 ME 调用计数器未加入。源 YUV 用作打分或离线监督，编码器输入为零填充。冻结 A 的历史权重、二进制和配置哈希见科研归档的 `historical_input_provenance.json`。

独立软件 qpel MC 对训练教师和四个验证 A 参考共 21,425,664 个 P16 Y 样本与 JM trace 逐像素一致。
