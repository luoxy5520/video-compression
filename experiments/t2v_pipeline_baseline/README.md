# Wan2.1 T2V 性能基线

[实验报告](BASELINE_REPORT.md)记录本次实测、历史对照与理论加速上限。工作负载限于历史开发样本 `sample_0000` 与 `sample_1004`。样本固定参数及输入来源见 [samples.json](configs/samples.json)。

## 复现

在 113Server 将本目录复制到一个新路径。需要已有 Wan2.1 工作树及虚拟环境、本地 checkpoint、历史 B1 的 JM kit、FFmpeg 和 FFprobe。选择空闲或空载的 GPU。历史仓库和输入保持只读，所有输出写入新目录。

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py="$HOME/Wan2.1/.venv/bin/python"
for sample in sample_0000 sample_1004; do
  "$py" -u scripts/profile_pipeline.py "run/$sample" "$sample" --repeats 5 --profiles 5
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$py" -u scripts/profile_encoder.py "run/historical-$sample" "$sample"
done
"$py" -u scripts/profile_rgb.py run/rgb-detail
```

每个样本的首轮兼作冷启动和小规模校验，校验通过才开始该样本的正式计时。用 `--repeats 0 --profiles 0` 可只执行首轮。在测量期间可用以下命令记录两张卡的整卡采样。

```bash
nvidia-smi --query-gpu=timestamp,index,pci.bus_id,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv -l 1
```

使用带 Matplotlib 的 Python 生成报告、CSV 和 PNG。第二个参数是报告和图表的目标目录。
从远端同步时可用 `scripts/collect_logs.sh` 生成仅含元数据和文本日志的 tar 流，再用 `--server-run` 保留报告中的原始服务器路径。

```bash
python scripts/summarize.py run .
```

`native` 测量未经阶段同步的请求级延迟。`profile` 测量互斥的阶段墙钟和同卡 CUDA Event。`cold_warmup` 是新进程的首个请求。`historical_raw_reencode` 是历史 RAW 在本轮重新编码。各次原始日志、命令和解码校验在服务器实验目录保留，代码及轻量测量证据可独立归档。

## 资产来源

- Wan2.1 主程序 `~/Wan2.1/wan/text2video.py`，历史 attention 路径 `~/Wan2.1/experiments/baseline17/run_remote_baseline17.py`。
- 两个生成批次的配置和缓存位于 `/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106` 和同级 `exp_a_new48_20261107`。历史生成参数源于原生成配置、服务器生成脚本与两条样本的 pairing 元数据。
- B1 原生可执行文件及配置取自 `~/a_new48_20261107/kit`，按 2026-11-21 的 `run_followup.py` 组合基础配置、`skip_p16.cfg` 与 CLI 参数。D0 历史结果用于重新编码的码流 SHA-256 核验。
- 本次只读取两个明确允许的开发样本。未进入封存盲测目录。
