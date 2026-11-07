# 最终 latent：参考对齐和源图像预测

本目录只使用求解器最终的 `final_latent.pt`，以 `final_raw.yuv` 为目标；没有使用各步的 clean prediction，也没有测提前退出。固定生成域 20 训练 / 4 验证 / 8 测试视频，按 prompt 隔离。帧格式为 17 帧、832×480、16 fps、YUV420p。详细结论见 [RESULTS.md](RESULTS.md)。

推理时有两条接口：

- 混合路径：`final_latent → 源图像预测 → JM 选择 Skip/P16 和 MV → 从真实重建参考做 MC → 量化、重建及熵编码`。推理不读取测试原视频或 Oracle I 帧。
- 模式与 MV 诊断：使用上一轮预测的模式和 MV，沿用 Oracle I 帧，P16 使用新头预测的源图像与实际 MC 之差。仅用来定位瓶颈。

低秩输出投影是学习到的源图像头，依赖冻结的 Wan VAE 解码主干。现有的直接状态接口 `mode_id / mv_qpel / residual_{y,u,v}` 在上一轮目录 `full_state_20261107/`，没有在本轮被替换。

| 文件 | 用途 |
| --- | --- |
| `prepare_public.py` | 逐条裁剪 DAVIS 2017，使用同一 Wan VAE 生成 latent / VAE 重建 raw 对；保留官方 train/val |
| `patch_trace.py`、`export_trace.py` | 从独立 JM 源树导出真实预测轨迹上的运动补偿像素，以及对齐的残差和 mask |
| `train_source.py`、`run_training.sh` | 小型 3D CNN 源图像头，以及对齐残差的两组同初始化对照 |
| `train_tail.py` | 冻结 Wan 解码前缀，训练空间上采样尾部 |
| `train_rank.py`、`calibrate_rank.py` | 训练因果输出投影，仅用生成域验证集选秩 |
| `benchmark_source.py`、`run_eval.sh`、`run_rank_eval.sh` | 真实码流、JM/FFmpeg 双解码和逐视频指标 |
| `results/calibrated/selected.pt` | 7,083 参数的输出投影；需要配套冻结的 Wan VAE |
| `results/eval_*/results.json` | 逐视频指标和汇总 |
| `results/public/manifest.json`、`results/traces/results.json` | 配对来源及逐样本的参考轨迹检查 |

从服务器仓库根目录运行，依赖上一轮的 Oracle/模式及 MV 模型产物。基础 Wan 权重在 `/mnt/data_hdd/models/Wan2.1-T2V-1.3B/`。新增 JM trace 补丁应加到上一轮 `patch_jm.py` 已处理的 JM 副本后，再重新编译。脚本输出目录需是新目录。

```bash
p=residual_reference_20261107
w=/mnt/data_hdd/luoxiangyu/dit_pairs/residual_reference_20261107
py=/home/luoxiangyu/Wan2.1/.venv/bin/python
kit=/home/luoxiangyu/senior-oracle-expand32-20261106
jm=/home/luoxiangyu/jm-residual-reference-20261107
args=(--config expansion_20261106/config_train32.json \
      --data /mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106)

"$py" "$p/prepare_public.py" --archive "$w/davis.zip.part" --output "$w/public"
"$py" "$p/patch_trace.py" "$jm"
make -C "$jm/lencod" -j8
"$py" "$p/export_trace.py" "${args[@]}" --kit "$kit" --jm "$jm" --output "$w/traces"
"$py" "$p/train_source.py" "${args[@]}" --output "$w/source_generated"
bash "$p/run_training.sh"
"$py" "$p/train_tail.py" "${args[@]}" --public "$w/public/manifest.json" --output "$w/pretrained_tail"
"$py" "$p/train_rank.py" "${args[@]}" --public "$w/public/manifest.json" --output "$w/rank_projection"
bash "$p/run_eval.sh"
bash "$p/run_rank_eval.sh"
```

服务器产物根目录为上述 `$w`。每个测试样本的 `proxy.yuv`、各路径的 `prediction.264`、编码重建和双解码核对结果在对应的 `eval_*/sample_*/`。原始轨迹与对齐的 `aligned_labels.npz` 在 `traces/sample_*/`。大文件留在数据盘。
