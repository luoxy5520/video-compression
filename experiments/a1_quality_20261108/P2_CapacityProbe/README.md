# P2-CapacityProbe

[阶段判定与完整数表](decision.md) · [锁定协议](protocol_locked.yaml) · [科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-capacityprobe-20261109)

113Server 运行目录为 ~/a1_quality_20261108/P2_CapacityProbe，使用 Wan2.1 环境。以 InfoProbe audit.py 在新输出目录复跑冻结 A。scripts/run.sh 记录本次 C0 与 C1 顺序，复现时先换成空的输出根目录，再运行 neighbors.py 生成训练集 C2。replay_capacity.py 对两个种子分别传 --seed 1701 / 1702，按实际门槛回放所有预定主预算 checkpoint。aggregate.py 使用 Matplotlib 生成 CSV 和图。previews.py 以评分码流生成精简预览。

[Mac 精简交付](../../../docs/local_review/README.md#p2-capacityprobe)
