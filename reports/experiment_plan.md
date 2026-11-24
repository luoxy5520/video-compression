# 执行范围和阶段门槛

E0-P检查→E0-REF历史YUV字节一致→libx264多QP→A无损→B/C/D量化→E整数差分/闭环→F敏感度与混合精度→JM补充→展示/16val→G同次Post-DiT计时→归档报告。

Gate0使用断言阻止继续：有限BCTHW、原dtype不变、无损字节相同、pack/unpack一致、E1重建与C2相同、参考历史YUV一致。shape由实际数据读取，未假定16通道。

Gate1/2/3为探索判断，不能强制阳性。开发INT2严重失真，仍记录结果，不作为可部署配置。时序预测的负收益保留，不启动模型训练。验证前固定AbsMax per-channel策略；没有根据16val选择新的参数。测试集封存。

LPIPS、Prompt级两种速度比、硬件真实内存流量、长视频稳定性、GitHub上传没有完成证据则NOT RUN。报告不会用理想性能填充。
