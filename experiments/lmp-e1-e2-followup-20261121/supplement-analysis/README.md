本轮三项任务的阅读版为analysis.md。数据来自8条开发视频及sample_1028/1029两条历史验证诊断，策略未调整。

备份包含全部CSV/NPZ逐块统计、输入哈希、正确性检查、编码码流、日志、脚本、修改的两份C源文件及统计补丁。排除了源视频/重建YUV、latent/hidden张量、模型权重和完整编码器二进制。原始输入路径见input_audit.json，编译参数与二进制哈希见environment.txt、binary_hashes.json和构建脚本。远端完整YUV仍保存在 /home/luoxiangyu/lmp-e1-e2-followup-20261121/supplement-analysis。

SHA256SUMS.json已与服务器生成的所有备份文件逐一核验。详细日志运行不用于计时结论，没有执行新策略优化或训练。
