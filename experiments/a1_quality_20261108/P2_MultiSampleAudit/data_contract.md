# M1 数据契约

父代码 963001586f01a4008307cf2221050f0de7b52e3d，归档 3fbc2fd28cd4563ac702cb95303e0367f2abc076。沿用 113Server Wan2.1 环境、冻结 A 和 JM 19.0 FRExt。缓存输入为 step 20 conditional hidden 的冻结特征 [1,67,16,30,52]。标签形状 [16,30,52,16,16] 依次表示 P 帧、宏块 y/x、4×4 块光栅位置、频率光栅位置。码流帧 0 为 Oracle I，标签数组第 0 项为第一 P 帧。qpel MV 为 x/y 四分之一像素，Skip 零残差。JM 适配器用整数像素基合成有符号 QP28 level，重新检查 CAVLC 原生扫描位置。坐标级清单 selected_coordinate_contract.csv 保存在归档。

44 train、16 val 对应 11/4 互斥 prompt，源哈希无跨 split 复用。召回分母始终是每视频 64 个正标签，precision 同时报实际写入数及固定 64 分母。模型输入路径与评分端源视频分离。
