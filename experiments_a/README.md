# 实验 A：师姐的状态可读性实验

按初版 PDF 推进：A0.1 模式 → A0.2 MV（oracle/预测模式）→ A0.3 residual（oracle 模式/MV/重建参考）→ A1 最终状态联合输出及 H.264 replay → A2 共享时间条件中间步骤状态头。

## A0.1 当前状态
已实现独立 final latent 解码、轻量 Conv3D 模式头、真实标签校验、训练、测试、置信度输出、准确率/Macro-F1/两类召回率/混淆矩阵/训练集多数类基线。输入为 solver 最终 latent；不训练 DiT。输出 [16,30,52]，仅16个P帧，每宏块0=Skip、1=P16x16。

当前只有4个样本，按 prompt 分组：红车两 seed 训练，蓝球两 seed 测试，避免相同 prompt 跨集合。此规模仅用于链路验证。标签未到位前不会启动实际训练或报告模型精度。

## 标签交接
`labels_root/sample_0000/state.npz` 包含 `mode`，shape [16,30,52]，值仅0/1。
同目录 metadata.json 至少包含：
```json
{"sample_id":"sample_0000","source":"solver_final","qp":28,"pixel_format":"yuv420p","profile":"baseline","entropy":"cavlc","reference_frames":1,"backend_version":"填写实际版本"}
```
必须来自 `sample_xxxx/final_raw.yuv`，不能直接复用 step_020/raw.yuv 的标签。编码固定1I+16P、单参考ref_idx=0、P帧仅Skip/P16x16；这些约束还需后端配置/码流检查，不由metadata声明证明。

服务器数据目录中的 `a0_manifest.json` 列出最终 latent 与 final_raw.yuv。标签未生成，label_path 为 null。

运行：
```bash
# 在服务器实验目录内
/home/luoxiangyu/Wan2.1/.venv/bin/python experiments_a/utility_scripts/a0_mode.py --config experiments_a/configs/exp_base.yaml --check-only
bash experiments_a/train.sh
bash experiments_a/test.sh
```

后续A0.2增加 qpel 精确率、跨档率、x/y误差、运动幅度分组；A0.3需要闭环参考与真实 residual；A1需要可注入预测状态的编码后端及标准解码验证；依赖到位后分别实现和验收。现阶段不声称这些实验已经完成。
