服务器目录：`/home/luoxiangyu/lmp-e1-e2-followup-20261121`

本地 summary.md 是中文结论，tables.csv 是逐视频表，jm_changes.patch 是编码器补丁。server-backup/ 保存远端实验脚本、输入hash、逐块候选、配置、码流、日志及诊断试跑；server-backup.tar.gz 是同内容压缩备份。为避免大量复制，未备份源视频、重建YUV、latent/hidden缓存、权重或完整JM源码；这些路径记录在manifest和input_audit中，保留在服务器个人空间。码流可从缓存重新双解码检查。

历史release保持不变，独立分支codex/lmp-e1-e2-followup；没有推送GitHub。D0–D2部分诊断未完成，D3两条小规模测试未达到收益门槛，因此没有扩大开发集或进行D4验证。
