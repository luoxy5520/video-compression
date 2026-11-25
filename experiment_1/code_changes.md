# Code changes

- `codec/lvc.py`：E0独立副本；2～8bit位宽/符号范围/真实长度/padding校验。pack输入整数数组与位宽、输出真实bytes；unpack反向。可选profile字典记录量化、打包、熵编码与完整耗时，不改变码流字节。E0源码未修改。
- `scripts/runtime.py`：冻结复制E0定义部分，移除自动运行的主实验；保留相同VAE像素转换/BT709/PSNR计时口径。
- `scripts/run.py`：模型/数据/旧码流/质量基线门槛、统一位宽固定扫描；失败写FAILED，不越过门槛。
- `scripts/decode.py`：独立接收进程，仅依赖Codec与NumPy，输入码流、输出临时npy；不导入VAE或源数据。
- `scripts/analyze.py`：通道统计、空间块与时间相关性、49配置扰动；冻结开发排序，验证支持分片。
- `scripts/mixed.py`：Uniform之外的Random/Fixed/Statistics候选、开发限定贪心近似Oracle；实际编码不调用VAE。
- `scripts/reuse.py`：四分位局部降精度及低/随机/高能区域单步/闭环复用；明确仅理论符号和mask，不声称实际Skip码率。
- `scripts/pipeline.py`：分片执行、记录命令/时区/GPU、验证前冻结、数量门槛、失败保留日志；测量COMPLETE单独记录。
- `scripts/profile.py`：所有质量任务结束后，代表配置预热1+5，且断言码流与原实验逐字节相同。
- `scripts/summarize.py`：只读实测结果生成CSV与图；有效区间内插值，不外推。
- `tests/test_codec.py`：4项Codec测试，服务器隔离目录通过；包括2～8bit及profile不改变码流。

初始实现commit：43f2384。新增分析/调度及结果的归档commit见Git历史；实执行源码摘要位于results/executed_source_sha256.json。首次执行在代码归档前，不能将后续commit冒充当时执行版本。
