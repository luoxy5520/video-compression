# Execution log

2026-11-25，Asia/Shanghai。服务器根目录：`/home/luoxiangyu/E0_Direct_Latent_Compression/experiment_1`。

- 阅读用户完整实验说明，并归档 experiment_plan.md；检查 E0 final_report、codec、reference、benchmark、validation_manifest、原脚本及 Git 历史。
- 初次 SSH 超时；用户恢复网络后重新连接成功。E0本地无remote、服务器副本不是Git仓库；用户随后指定 GitHub `luoxy5520/video-compression`，SSH认证成功。
- 本机 numpy 不可用，创建 /tmp/e1-venv 安装 numpy；本机 libzstd 不可用，完整Codec测试转到服务器同一Wan venv，未修改原环境。
- 原E0测试2项通过。E1打包覆盖2～8bit、奇数长度、padding、非法位宽/符号、独立解码和损坏检测；服务器3项通过。新增可选计时后，在/tmp/e1-code-review隔离执行4项测试通过，含计时不改变码流字节断言。
- `CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .../python -u scripts/run.py --stage gate`：两开发样本2/4/8bit共6配置通过；模型SHA、输入文件/数组SHA、历史参考YUV、重建Latent、PSNR及重编码旧码流全部一致。日志logs/gate.log，结果results/baseline_gate.json。
- 同环境 `scripts/run.py --stage uniform`：18样本×8配置=144条；独立进程只读取码流，原生WanVAE重建，PSNR-Y/RGB、SSIM、逐帧/末四帧/最差帧、真实字节、压缩比、编码/解码计时。完成记录results/E1_0/rows.json及CSV。代表开发Q4/Q6/Q8的VAE预热1+5，其余配置单次质量测量。
- `scripts/analyze.py --stage statistics`：18条样本；channel_statistics.csv、temporal_similarity.csv、block_residual_statistics.csv。完整72MB块统计留服务器；汇总进入Git。
- `scripts/analyze.py --stage sensitivity`：先两开发样本各49配置，开发排名在进入验证前保存configs/frozen_channel_ranking.json。验证分片由pipeline接续，保留开发结果。实际分片命令、GPU及带时区时间见logs/pipeline_commands.jsonl。
- `PYTHONPATH=../tools_plot .../python scripts/summarize.py`：生成率失真图、逐视频H.264有效区间内插值、块能量分位数。无外推，不计算BD-Rate。
- 本机py_compile默认缓存目录无写权限，改用PYTHONPYCACHEPREFIX=/tmp/e1-pycache；语法检查通过。
- 一次SSH会话被关闭；重连观察到服务器任务继续推进，未重跑已完成扫描。
- Git首次实现提交43f2384，已推送codex/experiment-1-latent-rd；后续结果与新增脚本见Git历史。

阶段状态以experiment_status.json为准；测量完成与全部交付完成分别记录。LPIPS NOT RUN（E0环境缺包/权重）。不触碰封存测试集，不训练/改动DiT或VAE。所有后续NOT RUN项须保留，不用正向措辞替代未完成实验。

- 验证分片首次启动失败：scripts/profile.py遮蔽标准库profile，触发PyTorch/cProfile导入异常。重命名measure_latency.py；原98条开发记录与冻结排名保留，从验证阶段接续，worker日志改为追加保留失败。

## 主测量完成

- E1.0：144/144；E1.1：882/882；E1.2：560条（开发网格、受限Oracle搜索与128条冻结验证）；E1.3：816/816。
- 性能：质量worker全部结束后执行measure_latency.py，12代表配置，量化/打包/熵编码、完整CPU/GPU起点编码、D2H、独立解码、VAE和两端时间，预热1+正式5；对同配置E0重新计时，所有流逐字节一致。
- 原生GPU起点VAE+YUV+x264 Medium QP28亦重测，明确不是同画质比较。
- 三代表视频0/8/16帧已人工检查：杯子有轻微形变/漂移、玩具车有位移、集市大范围视角变化；展示不把Prompt当实际运动标注。
- 主流程已生成29个展示MP4，ffprobe断言每条17帧、16fps，单面板832×480；Oracle仅开发样本，验证集Oracle NOT RUN。
- 最终补充Medium/Fast有效重叠区间插值、通道排名方差与INT4/6实际Latent MSE相关性、逐像素误差空间统计、完整码流递归分解。补充只用于分析，不更新任何冻结策略。
- 所有六种冻结可部署Mixed候选均未超过同码率Uniform，收益0/16；保留负向结果，不调整评价标准或重调验证集。
- 最初profile命名冲突失败日志和pipeline_failure.json保留；修复后测量与交付全部完成，不能将历史失败文件误当当前未完成状态。
