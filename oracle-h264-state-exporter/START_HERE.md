### 这个包包含什么
- `README_DETAILED_CN.md`：各 BAT、全部目录、状态接口、Oracle I 帧和全链路测试的详细说明。
- `data/paired_data.local.json`：16 条配对记录的入口清单。
- `data/sample_0000` 至 `data/sample_0003`：清单引用的 latent、hidden、final latent 和 raw YUV。
- `JM-backend/`：当前 Oracle 状态导出所需的修改版 JM 编码器、解码器、配置、脚本和修改源码。
- 根目录 PowerShell/Python 脚本：raw YUV 到 Oracle 状态，再到训练数据集的桥接流程。
### 第一次运行
1. 解压整个 ZIP，不要单独移动 `paired_data.local.json` 或 sample 文件夹。
2. 双击 `CHECK_ENV.bat`，确认 Python、NumPy、FFmpeg 和 FFprobe 均显示 `[OK]`。
3. 双击 `RUN_ONE.bat`，先处理一条记录验证环境。
4. 单条测试成功后，双击 `RUN_ALL.bat` 处理全部 16 条记录。
5. 全量数据完成后，双击 `RUN_IDEAL_CHAIN_TEST.bat` 验证理想标签到 H.264 的完整链路。
6. 有模型输出后，运行 `RUN_PREDICTION_TO_H264.bat` 验证真实预测状态。
7. 运行被中断时，重新双击同一个 BAT；脚本会跳过已经完成且闭合验证通过的记录。
8. 如果发现旧版或不完整的 `training_dataset`，脚本会自动重命名为 `training_dataset_stale_<timestamp>` 后重建，不会覆盖旧目录。
9. 如果某条 Oracle 导出因中断只留下空目录，脚本会自动重命名为 `step_<NNN>_incomplete_<timestamp>` 后重新生成。
### 包内相对路径
- 输入清单：`data/paired_data.local.json`
- 清单路径基准：`data/`
- 输入样本：`data/sample_xxxx/...`
- JM 后端：`JM-backend/`
- 单条测试结果：`output_smoke/`
- 全量 Oracle 状态：`output/oracle_exports/<sample_id>/step_<NNN>/`
- 最终训练数据集：`output/training_dataset/`
- 总训练清单：`output/training_dataset/manifests/all.jsonl`，默认 `label_path` 指向同一 sample 的 step 20 标签。
- 训练集清单：`output/training_dataset/manifests/train.jsonl`
- 验证集清单：`output/training_dataset/manifests/val.jsonl`
- 测试集清单：`output/training_dataset/manifests/test.jsonl`
- 状态头标签：`output/training_dataset/labels/<record_id>.npz`
- 构建汇总：`output/training_dataset/dataset_summary.json`
- 当前 step 独立清单：`output/training_dataset/manifests/by_step/`
- 从最终 step 向前训练的课程清单：`output/training_dataset/manifests/curriculum/`
### 不同 step 的输入与监督
- `current_label_path` 指向当前 step 的 Oracle 状态，只用于稳定性分析。
- `target_label_path` 指向同一 sample 的 step 20 Oracle 状态，是默认训练目标。
- `target_oracle_state_path` 指向 step 20 的 I 帧模板，用于 H.264 回放。
- `input_step`、`target_step`、`step_fraction`、`remaining_steps`、`timestep`、`sigma` 显式标记输入处于哪个阶段。
- 课程顺序为 step 20，再加入 step 15、step 10、step 5。
### 当前固定的 H.264 配置
- Baseline Profile、CAVLC、1I+P。
- P 帧模式为 Skip/P16×16。
- 单参考帧、QP 28、`RDOptimization=0`。
- 输入为 8-bit planar YUV420p；当前数据为 832×480、17 帧、16 fps、BT.709 limited。
### 系统依赖
- Windows PowerShell 5.1 或 PowerShell 7。
- Python 3.9+，并安装 NumPy：`python -m pip install numpy`。
- FFmpeg 和 FFprobe 已加入系统 `PATH`。
- 项目专用的修改版 JM 已经包含在 `JM-backend/`，不需要另外配置路径。
### 注意
- 不要公开上传 `JM-backend`，除非已经确认 JM 及修改二进制的再分发许可；该包建议仅在课题组内部传递。
- 默认训练清单引用包内原始 `.pt` 文件，不重复复制大张量。
- 全量 Oracle 导出会产生较大的中间文件和状态文件，请预留足够磁盘空间。
