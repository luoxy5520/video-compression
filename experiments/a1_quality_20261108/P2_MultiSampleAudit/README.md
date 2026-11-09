# P2-MultiSampleAudit

[结论与数表](decision.md) · [数据契约](data_contract.md) · [锁定协议](protocol.md) · [科研归档](https://github.com/luoxy5520/video-compression/tree/artifacts/p2-multisample-audit-20261108)

113Server 原始产物位于 ~/a1_quality_20261108/P2_MultiSampleAudit。使用历史 InfoProbe audit.py 在独立目录重跑冻结 A，然后以 m1.py 校验数据。scripts/run_m0.sh 与 scripts/run_m2.sh 记录实际分阶段入口，复现须将两脚本的输出根切换为空目录，保持历史 OnPolicy T0 标签路径。replay.py 先生成训练侧选定的回放名单，再分别使用 --seed 1701 和 --seed 1702 完成严格验证。aggregate.py 需要 Matplotlib，并生成 CSV 和图；previews.py 只重编码已有评分流。
