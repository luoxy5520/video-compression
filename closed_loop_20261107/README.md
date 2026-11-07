# 最终 latent 的运动预测与闭环残差

沿用原来的 20/4/8 视频划分，每条视频只读 step 20 的最终 latent。运动头比较相邻 latent 帧的 81 个位移候选的特征相关性，经 Conv3D 预测 P16 宏块的 qpel 位移。按验证集 MV 端点误差选择权重，保持原模型为对照。

`patch_jm.py` 在独立 JM 副本中加入两个可选入口：

- `JM_LATENT_MV_PRIOR` 传入当前视频的 P 帧运动候选。直接引导模式以预测 MV 为中心进行局部整数像素搜索。额外候选模式通过 `JM_LATENT_MV_EXTRA` 开启，保留原生局部窗口，只在额外候选代价更低时采用。
- `JM_LIVE_RESIDUAL` 用于 B1 诊断回放。保持原有的模式和 MV，让 JM 用当前源像素与实际参考重建的运动补偿结果计算残差。预测为 Skip 的块仍按标准 Skip 编码。

主评测采用真实 VAE 像素、JM 原生模式选择和闭环残差，在相同半径与搜索引擎下对比不同搜索起点。所有路径以同一最终 raw 为质量参考。原始后端在禁用补丁的验证中复现了逐字节相同的码流。

服务器上的源代码位于 `/home/luoxiangyu/dit-paired-experiment/closed_loop_20261107/`，独立 JM 编译目录位于 `/home/luoxiangyu/jm-closed-loop-20261107/`。完整实验输出位于 `/mnt/data_hdd/luoxiangyu/dit_pairs/closed_loop_20261107/`。可在该服务器用 `run_benchmark.sh` 和 `run_curve.sh` 复现实验，运行前为脚本中的输出设置新目录。
