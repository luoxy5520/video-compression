# 执行记录

日期：2026-11-24，Asia/Shanghai。工作目录为本实验目录；远端镜像 `/home/luoxiangyu/E0_Direct_Latent_Compression`。完整实验子进程命令另见 logs/commands.jsonl 及 JM commands.jsonl。源代码SHA另见source_sha256.json；首次运行时实验尚无Git commit，最终归档版本见Git历史，不能将归档commit冒充此前执行commit。

| 实验 | 实际操作/命令 | 输入与输出 | 目的与状态 |
|---|---|---|---|
| E0-P | pwd、rg --files、ls、git status；ssh -o BatchMode=yes -o ConnectTimeout=15 113Server；find指定项目文件；读取已有配置及profile代码 | 原工作目录为空；发现Wan2.1、JM、已保存latent及历史延迟结果 | 环境检查完成；不修改父级笔记仓库 |
| E0-P | 读取 configs/samples.json、exp_expand32/config.json、exp_a_new48/config.json、splits32.json | 2条train开发样本及4+12条val；仅查看test划分名称，不读其张量 | 确认开发/验证界限 |
| E0-P | mkdir -p实验目录；实现 codec/lvc.py、各执行脚本及tests | 新增代码保存在本目录，再tar经SSH送独立服务器目录 | 不安装依赖、不重新生成视频 |
| Gate0 | Wan2.1/.venv/bin/python -m unittest discover -s tests -v | logs/tests.log | 2项测试通过；覆盖多个dtype、多位宽、奇数符号、校验失败及独立解码进程 |
| E0-REF | CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 Wan2.1/.venv/bin/python -u scripts/run_smoke.py | 原始FP32 latent；reference.rgb/yuv/json | 原生VAE、同步计时、历史YUV字节SHA相等断言；参考为无损原始字节 |
| E0-BASE | run_smoke.py 中ffmpeg libx264 medium/fast QP18/22/26/28/32/37/42 | reference.yuv→真实.264及解码YUV | 每开发配置预热1+正式5次；原始命令自动记录 |
| E0-A/B/C/D/E | run_smoke.py | A0/A1/A2/A3，8/4/2bit tensor/channel，P99/99.5/99.9，E1及E2刷新2/4/8/All | rows.json，完整.lvc计费；每次编解码预热1+正式5次；两条开发样本已完成 |
| E0-F/G/BASE | CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 Wan2.1/.venv/bin/python -u scripts/supplement.py | 单通道INT4扰动、敏感4通道INT8+其他INT4、CPU/GPU量化D2H；复用原生JM | supplement.json、transfer.json、JM各配置与日志；顺序在Smoke通过后执行 |
| 展示 | Wan2.1/.venv/bin/python -u scripts/present.py | 从原始重建生成MP4、SSIM日志和曲线 | 展示不计入正式码率与编码耗时 |
| 16val | Wan2.1/.venv/bin/python -u scripts/validate.py | 原始manifest中split=val的16条 | 固定channel AbsMax、INT8/4/2、E1/E2 refresh4及x264曲线；validation_rows.json |
| E0-G | Wan2.1/.venv/bin/python -u scripts/benchmark.py | 原始同一GPU latent | 本次直接测量post-DiT编码和接收像素时间；benchmark.json；不重跑DiT |
| 报告 | python3 scripts/generate_report.py | 仅已保存的实测JSON | reports/final_report.md、file_manifest.md |

## 错误与修复

- 本机系统Python缺numpy，因此所有数值实验复用服务器现有venv。
- 一次functions脚本括号错误：未写入任何文件，随后重新执行。
- 本机py_compile尝试写系统缓存被拒：不更改系统权限，改为服务器测试。
- SSH默认沙箱发生Operation not permitted：经已审核的require_escalated SSH继续。
- 缺zstandard Python包：使用系统libzstd ctypes，不安装或替换环境。
- 缺lpips包：LPIPS明确NOT RUN，不以其他指标代替。
- JM重复封装可能遇既有MP4：运行前unlink本次实验自己的展示文件，保留所有编码日志；在执行JM前修正。

不虚构未完成状态。最终是否完成依 results/COMPLETE、SUPPLEMENT_COMPLETE、PRESENT_COMPLETE、VALIDATION_COMPLETE、BENCHMARK_COMPLETE 及最后日志判定。原始码流和YUV留服务器；本地同步轻量指标、源码、日志及代表性展示。

## 最终元数据审核与绘图

- 发现验证/混合精度码流最初仅带数据集checkpoint配置，显式VAE SHA与归一化字段需补齐：运行 scripts/finalize_metadata.py，对全部正式码流重新编码、重测5次并断言重建张量与原码流逐值相等；所有文件长度与相关指标更新。既有视频质量不变。
- 原环境缺matplotlib，首次绘图NOT RUN；随后仅安装到服务器独立实验目录tools_plot，不修改Wan venv。使用scripts/plot_results.py生成标准曲线，LPIPS仍NOT RUN。pip目标目录中的pillow与gradio约束提示不影响Wan环境，因为目标目录仅用于独立绘图进程。
- 原始数据、完整码流和中间YUV均留服务器独立实验目录，部分代表性LVC同步本地供直接审查。

## 完成核验

服务器已观察到 COMPLETE、SUPPLEMENT_COMPLETE、PRESENT_COMPLETE、VALIDATION_COMPLETE、BENCHMARK_COMPLETE、METADATA_COMPLETE、FINAL_COMPLETE。16条val共304条记录，2条开发样本共100条基础记录；最终数量以JSON实际值为准。码流元数据重写后全部重新计时并验证重建不变。源码归档commit为0f6fea7；最终结果由后续归档commit保存。本地videos/提供规范命名展示，bitstreams/提供10个代表码流。最终报告十项回答明确保留未执行指标。

一次归档命令把状态文本和tar混写至/tmp临时文件，未用于解压或数据分析；随后用纯tar输出重新同步成功。
