# 113Server 实验审计

执行节点为 user-AS-4124GS-TNR（113Server），Linux 6.17、AMD EPYC 7302 双路 64 逻辑核、2× NVIDIA A100 80 GB、GCC 13.3、FFmpeg 6.1.1。实验在单 CPU 上运行。桌面入口检出 codex/a0-mode @ 08dcb026363ae965d6ef651a9f1c1a8c181b36a2，存在未跟踪内容。

| 项目 | 状态 | 证据 |
|---|---|---|
| 11 月 15 日 LMP | completed / reproducible | /home/luoxiangyu/lmp-h264-prior.uFe5ZV/release，codex/lmp-h264-prior @ f080a451831e89a59948006c4299bca698fdfef6，检出干净。LMP_REPORT.md、baseline.log、probe.py、audit.jsonl。 |
| 11 月 21 日续跑 | completed / reproducible | /home/luoxiangyu/lmp-e1-e2-followup-20261121/repo，codex/lmp-e1-e2-followup @ 同一 SHA，存在未跟踪 results-e1-e2-followup。补测见同级 supplement-analysis。 |
| 工作仓库 | completed | /home/luoxiangyu/dit-paired-experiment，codex/a1-p0-6-20261108 @ a8634bd898a8d451d6dc47e5a89dc2b4a96e771b，检查时干净。 |
| 输入与权重 | completed / needs_verification | 两条开发样本的 final_raw.yuv、final_latent.pt、conditional hidden 实际存在，根目录 /mnt/data_hdd/luoxiangyu/dit_pairs。历史指纹见 supplement-analysis/input_audit.json。历史报告记录 Wan2.1 T2V 1.3B @ 9737cba 和冻结头缓存，续跑仅消费缓存 latent。 |
| JM | completed | JM 19.0，历史源码和编码器位于 /home/luoxiangyu/a_new48_20261107/kit/linux/JM。旧计时与逐块版位于 11 月 21 日目录下 jm-profile、jm-perblock。原始编解码器 SHA 见 logs/0004.stdout。 |
| 基线 A | completed / missing | 原始 B0 SearchMode=0，B1 EPZS=3。两条视频各有暖机及三次重复、哈希和双解码。已有整数与亚像素计时与评估计数。模式选择、残差变换量化与熵编码独立计时仍缺。 |
| 候选 B | completed | 8 条开发样本已复现。E0 是真实 JM P16 MVP 加零及块运动构造的序列；历史轨迹来自 B0，完整 EPZS 搜索点集合有独立记录。E2/E3 采用归一化平方距离及 9×9 匹配，2×2 latent 聚合，跨四帧位移分摊后每格映射 16 qpel。历史统计为合法 P16 块 K=2、L∞ 半径 2 px 的覆盖；验证集视频等权。融合保留 MVP，第二位置可能挤掉原生候选。共用历史重建参考的 SAD 已记录。 |
| 真实编码 C | completed / missing | 旧 D3 已在线接入原生、原生早停、DiT 单种子、DiT 门控。补测完成 8 条开发与 2 条历史验证样本的逐块计数与双解码。固定预算替换/融合及整条链路直接计时缺失。 |
| D/E | missing | 旧试点仅计算分段中位数之和；无完整验证、BD-rate 和实际 VAE→码流时间。 |

补做两条历史试点上的等预算对照和同一时钟测得的缓存 latent→码流链路，沿用旧自适应策略。仅在开发结果达到门槛后考虑冻结并扩大。两份指定标题文档在服务器已检查的两个仓库内未发现，对应方法见 LMP_REPORT.md、旧续跑 summary.md、supplement-analysis/analysis.md 和实验地图。
