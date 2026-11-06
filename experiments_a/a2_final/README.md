# 中间特征预测最终压缩状态

`dataset_local/exp_2026_11_05/a2_final_pairs.json` 为配对入口：16条输入记录，4个唯一最终目标。每个样本第5/10/15/20步均指向同一 `sample_xxxx/final_raw.yuv` 和同一最终标签 `sample_xxxx/state.npz`。数据路径相对清单目录，标签路径相对 labels_root。清单包含 final raw 的 SHA256，防止误用当前步骤的标签。

已实现 clean latent 输入、共享时间条件模式头。进度条件为 step/20；模式输出 [16,30,52]，0 Skip、1 P16x16。hidden 路径保留在清单中，当前模型尚未使用 hidden。MV 和 residual 预测尚未实现。当前四个样本仅为链路 pilot。

训练分组沿用 A0：红车两个seed训练，蓝球两个seed测试，同一视频四步不跨集合。检查器同时拒绝相同prompt跨集合。正式实验需要增加视频并新增独立验证集。

## 师妹生成标签
对每个样本只编码一次 final_raw.yuv，输出 labels_root/sample_xxxx/state.npz 中的 mode，以及 metadata.json。metadata 沿用 A0 约定，并新增 raw_sha256（清单中 target_raw_sha256 的值），确保视频来源一致。

```json
{"sample_id":"sample_0000","source":"solver_final","qp":28,"pixel_format":"yuv420p","profile":"baseline","entropy":"cavlc","reference_frames":1,"backend_version":"实际版本","raw_sha256":"从清单复制对应值"}
```

## 服务器运行
在 /home/luoxiangyu/dit-paired-experiment 下：
```bash
PY=/home/luoxiangyu/Wan2.1/.venv/bin/python
$PY experiments_a/utility_scripts/a2_final_mode.py --config experiments_a/configs/exp_a2_final.yaml --check-features
# 真实标签到位，配置 labels_root 后：
$PY experiments_a/utility_scripts/a2_final_mode.py --config experiments_a/configs/exp_a2_final.yaml --check-labels
bash experiments_a/a2_final/train.sh
bash experiments_a/a2_final/test.sh
```

结果按step分别输出准确率、Macro-F1、Skip/P16x16召回率、混淆矩阵、平均置信度和多数类基线。预测文件名包含 sample_id 和 step，标签的目标始终是最终状态。

当前真实标签缺失，尚未开展真实训练，也无预测精度结论。此实验不含退出控制器；预测稳定性、置信度校准与实际后端质量需在后续独立验证。
