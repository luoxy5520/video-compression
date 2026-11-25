# Code changes

- `codec/lvc.py`：E0独立副本；2～8bit位宽/符号范围/真实长度/padding校验。pack输入整数数组与位宽、输出真实bytes；unpack反向。可选profile字典记录量化、打包、熵编码与完整耗时，不改变码流字节。E0源码未修改。
- `scripts/runtime.py`：冻结复制E0定义部分，移除自动运行的主实验；保留相同VAE像素转换/BT709/PSNR计时口径。
- `scripts/run.py`：模型/数据/旧码流/质量基线门槛、统一位宽固定扫描；失败写FAILED，不越过门槛。
- `scripts/decode.py`：独立接收进程，仅依赖Codec与NumPy，输入码流、输出临时npy；不导入VAE或源数据。
- `scripts/analyze.py`：通道统计、空间块与时间相关性、49配置扰动；冻结开发排序，验证支持分片。
- `scripts/mixed.py`：Uniform之外的Random/Fixed/Statistics候选、开发限定贪心近似Oracle；实际编码不调用VAE。
- `scripts/reuse.py`：四分位局部降精度及低/随机/高能区域单步/闭环复用；明确仅理论符号和mask，不声称实际Skip码率。
- `scripts/pipeline.py`：分片执行、记录命令/时区/GPU、验证前冻结、数量门槛、失败保留日志；测量COMPLETE单独记录。
- `scripts/measure_latency.py`：所有质量任务结束后，代表配置预热1+5，且断言码流与原实验逐字节相同。
- `scripts/summarize.py`：只读实测结果生成CSV与图；有效区间内插值，不外推。
- `tests/test_codec.py`：4项Codec测试，服务器隔离目录通过；包括2～8bit及profile不改变码流。

初始实现commit：43f2384。完整位宽扫描及后续框架归档commit：6d5cf2d；最终分析/调度及结果归档commit见Git历史；实执行源码摘要位于results/executed_source_sha256.json。首次执行在代码归档前，不能将后续commit冒充当时执行版本。

- `scripts/report.py`：从实测数据生成16节综合报告及4份阶段报告，明确负向结果、实现瓶颈、未测项目与受限Oracle。
- `scripts/extra_figures.py`：递归真实码流分解、通道排序/配置图、误差传播、局部像素热图、Oracle预算比较；不选择新策略。
- `scripts/videos.py`：三种实际核验场景的统一尺寸/帧率/帧数展示；同一开发选择策略，Oracle单独标注，ffprobe核验29个MP4。
- `scripts/postprocess.py`：等待所有质量worker退出，再测试并切换字节等价的计时版Codec，隔离测量性能、生成报告与交付清单。

最终Codec：4项测试通过；12个实测代表配置再次断言计时重编码与原码流逐字节相同。统计补充仅重用既有18条合法样本和码流，无验证反馈调整。最终源码哈希及产物哈希见results/source_sha256_*和artifact_inventory.json。

- `scripts/finalize_archive.py`：核对计数、冻结验证策略、597168条块统计及所有码流分解相加恒等式，记录18条输入文件/张量/参考哈希与最终归档版本；不重跑模型。调度器接受完整882条既有扰动结果，避免重复验证；未来失败时写FAILED。
