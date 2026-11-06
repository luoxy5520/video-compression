# Linux 适配
原始 BAT/PowerShell 和 Windows PE 二进制保持不变。Linux 后端由官方 JM19.0 源码加包内 macroblock.c/state_io.c/state_io.h/slice.c.diff 的等价补丁构建。`linux/JM/bin/*.exe` 虽后缀为exe，实际是Linux ELF文件。

源码包从 https://iphome.hhi.de/suehring/tml/download/jm19.0.zip 下载，放在linux/jm19.0.zip，执行 `bash linux/build_backend.sh`。使用CFLAGS=-fcommon解决旧JM在现代GCC上的全局符号兼容问题。原始数据通过data软链接复用，不复制DiT权重。

运行（在包根目录）：
```bash
PY=/home/luoxiangyu/Wan2.1/.venv/bin/python
$PY linux/run_pipeline.py check
$PY linux/run_pipeline.py one
$PY linux/run_pipeline.py all
$PY linux/run_pipeline.py ideal
```

check核对依赖并实测4个step20 clean latent等于final latent、step20 raw等于final raw。one独立生成output_smoke。all四路并行生成output，不复用smoke，随后调用原包build_dataset.py。ideal对4个样本分别调用原包prediction_to_b1_state.py并用Oracle I模板回放，要求JM/FFmpeg解码像素相同，且理想回放码流与Oracle逐字节相同。

输出与原包关键目录相同，日志简化为每条目录的dump.log/replay.log/parse.log等。Python解析器/数据集构建器/预测adapter未改动。未执行真实模型预测测试；此项需要训练后的完整mode/MV/residual输出。

包内JM源码和二进制未自动上传GitHub。实验结果由experiment_report.json汇总，保留Oracle I条件，不能作为真正在线提前退出完成的证据。
