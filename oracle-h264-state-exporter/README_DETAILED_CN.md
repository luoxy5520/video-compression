### Oracle H.264 状态数据集工具包详细说明
本工具包用于把 DiT 不同去噪阶段保存的 latent、hidden 和 raw YUV 转换为受限 H.264 Oracle 状态，构建“早期 DiT 特征预测最终 step 20 压缩状态”的训练数据集，并验证预测状态能否生成标准 H.264 码流。
### 一、推荐运行顺序
~~~text
CHECK_ENV.bat
↓
RUN_ONE.bat
↓
RUN_ALL.bat
↓
RUN_IDEAL_CHAIN_TEST.bat
↓
训练状态头并生成 prediction.npz
↓
RUN_PREDICTION_TO_H264.bat
~~~
### 二、各个 BAT 的输入和输出
| BAT | 读取什么 | 写到哪里 | 是否依赖前一步 |
|---|---|---|---|
| `CHECK_ENV.bat` | Windows 环境、Python、NumPy、FFmpeg、FFprobe、JM 后端和输入清单 | 不生成数据 | 无 |
| `RUN_ONE.bat` | `data/paired_data.local.json`、`sample_0000/step_020` 及包内 JM 后端 | `output_smoke/` | 先通过环境检查 |
| `RUN_ALL.bat` | `data/paired_data.local.json` 和全部四个 sample 的 step 5、10、15、20 | `output/` | 建议先通过单条测试 |
| `RUN_IDEAL_CHAIN_TEST.bat` | `output/training_dataset` 中 step 20 理想标签和 `output/oracle_exports` 中相同 sample 的 step 20 Oracle I 模板 | `chain_tests/ideal/<sample_id>/<timestamp>/` | 必须先完成 `RUN_ALL.bat` |
| `RUN_PREDICTION_TO_H264.bat` | 用户指定的 `prediction.npz` 和 `output/oracle_exports` 中对应 sample 的 step 20 Oracle I 模板 | `chain_tests/inferred/<sample_id>/<timestamp>/` | 必须先完成 `RUN_ALL.bat`，并已有模型预测结果 |
### 三、BAT 详细说明
### 1. CHECK_ENV.bat
该脚本只检查环境，不修改输入数据，也不创建 Oracle 状态。检查项目包括：
- `python` 是否在 `PATH`；
- NumPy 是否可以导入；
- `ffmpeg` 和 `ffprobe` 是否在 `PATH`；
- `JM-backend/bin/lencod_state_b1.exe` 是否存在；
- `JM-backend/bin/ldecod_ref.exe` 是否存在；
- `data/paired_data.local.json` 是否存在。
缺少 NumPy 时运行：
~~~bat
python -m pip install numpy
~~~
### 2. RUN_ONE.bat
该脚本执行最小烟雾测试，只处理：
~~~text
sample_0000 / step 20
~~~
选择 step 20 是因为第一阶段训练从最终 latent 开始，监督关系为：
$$
z_{20}\rightarrow y_{20}
$$
输出位置：
~~~text
output_smoke/
├── oracle_exports/sample_0000/step_020/
└── training_dataset/
~~~
如果运行中断，再次双击即可续跑。完整结果会跳过；不完整 Oracle 目录会备份为 `step_020_incomplete_<timestamp>` 后重新生成；旧版或不完整训练数据集会备份为 `training_dataset_stale_<timestamp>` 后重建。
### 3. RUN_ALL.bat
该脚本处理全部 16 条记录：
~~~text
4 个 sample × 4 个输入 step = 16 条记录
输入 step = 5、10、15、20
最终监督 step = 20
~~~
训练关系为：
$$
z_5\rightarrow y_{20},\quad z_{10}\rightarrow y_{20},\quad z_{15}\rightarrow y_{20},\quad z_{20}\rightarrow y_{20}
$$
输出位置：
~~~text
output/
├── oracle_exports/
└── training_dataset/
~~~
`RUN_ALL.bat` 不读取或合并 `output_smoke/`；单条测试和全量构建相互独立。全量训练数据集固定包含 16 条记录，不会因为先运行 `RUN_ONE.bat` 而变成 17 条。
### 4. RUN_IDEAL_CHAIN_TEST.bat
该脚本把 step 20 的 Oracle 训练标签当作“完美状态头预测”，验证完整链路：
~~~text
理想 mode/MV/residual
+ step 20 Oracle I 帧模板
→ predicted_state.bin
→ predicted.264
→ JM 和 FFmpeg 双解码
~~~
默认测试 `sample_0000`：
~~~bat
RUN_IDEAL_CHAIN_TEST.bat
~~~
指定其他 sample：
~~~bat
RUN_IDEAL_CHAIN_TEST.bat sample_0002
~~~
脚本只读取 `output/`，不会修改 Oracle 状态和训练数据集。每次运行建立新的时间戳目录：
~~~text
chain_tests/ideal/<sample_id>/<timestamp>/
~~~
### 5. RUN_PREDICTION_TO_H264.bat
该脚本把真实模型输出的 `prediction.npz` 接入同一 H.264 后端。命令行用法：
~~~bat
RUN_PREDICTION_TO_H264.bat "D:\predictions\prediction.npz" sample_0000
~~~
双击且不提供参数时，脚本会提示输入 `prediction.npz` 的完整路径；sample 默认是 `sample_0000`。
每次运行建立：
~~~text
chain_tests/inferred/<sample_id>/<timestamp>/
~~~
它只读取 `output/` 中的 step 20 Oracle I 模板，不覆盖任何训练数据或上一次测试结果。
### 四、包内目录与文件含义
~~~text
senior-oracle-dataset-kit-20261106/
├── CHECK_ENV.bat
├── RUN_ONE.bat
├── RUN_ALL.bat
├── RUN_IDEAL_CHAIN_TEST.bat
├── RUN_PREDICTION_TO_H264.bat
├── START_HERE.md
├── README_DETAILED_CN.md
├── data/
├── JM-backend/
├── scripts/
├── output/                     # RUN_ALL 生成
├── output_smoke/               # RUN_ONE 生成
└── chain_tests/                # 两个全链路测试生成
~~~
### 五、输入数据目录 data
~~~text
data/
├── paired_data.local.json
├── sample_0000/
├── sample_0001/
├── sample_0002/
└── sample_0003/
~~~
`paired_data.local.json` 是唯一批处理入口，所有数据路径相对于它所在的 `data/` 目录解析，因此整个工具包换电脑或换盘符后仍可使用。
每个 sample 包含：
~~~text
sample_xxxx/
├── final_latent.pt
├── step_005/
│   ├── clean_latent.pt
│   ├── hidden_cond_last_block.pt
│   └── raw.yuv
├── step_010/
├── step_015/
└── step_020/
~~~
字段含义：
- `clean_latent.pt`：当前去噪阶段的 CFG clean prediction，形状 `[16,5,60,104]`；
- `hidden_cond_last_block.pt`：conditional 分支 block 29 hidden，形状 `[1,7800,1536]`；
- `raw.yuv`：由当前 clean latent 解码得到的 8-bit planar YUV420p 视频；
- `final_latent.pt`：该 sample 的最终 latent；当前 step 20 clean latent 与 final latent 一致。
视频参数：832×480、17 帧、16 fps、BT.709 limited、YUV420p。
### 六、JM-backend 目录
~~~text
JM-backend/
├── bin/
│   ├── lencod_state_b1.exe
│   └── ldecod_ref.exe
├── cfg/
├── scripts/
└── source/
~~~
- `lencod_state_b1.exe`：支持 B1 状态 dump/replay 的修改版 JM 编码器；
- `ldecod_ref.exe`：JM 参考解码器；
- `cfg/`：Baseline、Skip/P16 和解码配置；
- `scripts/run_b1_closure.ps1`：Oracle dump、状态 replay 和闭合验证；
- `source/`：与状态接口相关的修改源码。
该目录建议仅在课题组内部传递，公开发布前需要确认 JM 和修改二进制的再分发许可。
### 七、根目录 PowerShell 文件
| 文件 | 作用 |
|---|---|
| `build_from_paired_data.ps1` | 批量读取配对清单，导出 Oracle 状态并调用训练集构建器 |
| `export_oracle.ps1` | 单个 raw YUV/RGB 到 Oracle 状态、标签、H.264 和闭合报告 |
| `build_dataset.ps1` | PowerShell 训练集构建入口 |
| `encode_prediction.ps1` | `prediction.npz + Oracle I 模板 → 标准 H.264 → 双解码验证` |
| `run_ideal_chain_test.ps1` | 理想标签全链路测试的路径解析和时间戳输出包装器 |
| `run_prediction_to_h264.ps1` | 模型预测全链路测试的路径解析和时间戳输出包装器 |
| `run_windows.bat` | `RUN_ALL.bat` 调用的内部 Windows 批处理入口，通常不需要单独运行 |
| `setup_local.ps1` | 仓库开发模式下注册外部 JM 路径；完整包已经包含 JM，通常不需要运行 |
### 八、scripts 目录
| 文件 | 作用 |
|---|---|
| `parse_b1_state.py` | 把二进制 JMZV1/B1 状态解析为 NumPy Oracle 标签 |
| `build_dataset.py` | 转换 P 帧标签、连接早期输入与最终 step 20 目标、划分数据并生成课程清单 |
| `prediction_to_b1_state.py` | 把模型输出的 mode/MV/residual 写回可 replay 的 B1 状态 |
### 九、Oracle 单条输出含义
~~~text
output/oracle_exports/<sample_id>/step_<NNN>/
├── input_yuv420p.yuv
├── oracle_state.bin
├── oracle_labels.npz
├── state_metadata.json
├── metadata.json
├── reference.264
├── replayed.264
├── reference_recon.yuv
├── closure_result.json
└── logs/
~~~
- `oracle_state.bin`：完整 I 帧初始化状态与当前 step 的 P 帧 Oracle 状态；
- `oracle_labels.npz`：解析后的完整 Oracle 张量；
- `reference.264`：正常 Oracle 编码产生的标准 H.264；
- `replayed.264`：只从状态 replay 生成的 H.264；
- `closure_result.json`：码流一致、JM 解码一致和 FFmpeg 解码一致结果；
- `metadata.json`：sample、step、输入哈希、编码配置和工具版本。
完整 Oracle 导出必须满足：
~~~json
{
  "bitstream_bit_exact": true,
  "jm_decoded_pixels_exact": true,
  "ffmpeg_decoded_pixels_exact": true
}
~~~
### 十、训练数据集输出含义
~~~text
output/training_dataset/
├── labels/
├── manifests/
│   ├── all.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── by_step/
│   └── curriculum/
├── splits/
├── rejected.jsonl
└── dataset_summary.json
~~~
转换后的 `labels/<record_id>.npz` 包含：
~~~text
source_frame_index
mode_id
mode_valid
mv_qpel
mv_valid
residual_y
residual_u
residual_v
residual_valid
~~~
每条 manifest 记录显式保存：
- `latent_path`、`hidden_path`：当前输入 step 的特征；
- `input_step`、`timestep`、`sigma`、`step_fraction`、`remaining_steps`：当前输入阶段；
- `current_label_path`：当前 step 的 $y_s$，用于稳定性分析；
- `target_label_path`：相同 sample 的 step 20 标签 $y_{20}$；
- `label_path`：默认训练标签，与 `target_label_path` 相同；
- `target_oracle_state_path`：step 20 Oracle I 帧模板；
- `target_step=20`。
### 十一、从最终 step 向前训练
课程清单位于：
~~~text
output/training_dataset/manifests/curriculum/
├── stage_01_steps_020_train.jsonl
├── stage_02_steps_020_015_train.jsonl
├── stage_03_steps_020_015_010_train.jsonl
└── stage_04_steps_020_015_010_005_train.jsonl
~~~
推荐顺序：
1. 先训练 $z_{20}\rightarrow y_{20}$；
2. 加入 $z_{15}\rightarrow y_{20}$；
3. 加入 $z_{10}\rightarrow y_{20}$；
4. 最后加入 $z_5\rightarrow y_{20}$。
如果一个 batch 混合多个 step，训练代码应读取 `step_fraction`、`timestep` 或 `sigma` 作为显式时间条件。只按单 step 清单训练时，可直接通过 `manifests/by_step/` 区分。
### 十二、当前压缩中间状态的设计
当前第一版接口把状态分为三个部分：固定编码配置、Oracle I 帧初始化状态、需要模型预测的 P 帧状态。
固定编码配置包括：
- 8-bit YUV420p；
- H.264 Baseline Profile；
- CAVLC；
- `1I + P`；
- 单参考帧；
- QP 28；
- Skip/P16×16；
- 4×4 transform；
- `RDOptimization=0`。
这些字段不由状态头预测。
需要状态头预测的 P 帧状态包括：
~~~text
mode_id
mv_qpel
residual_y
residual_u
residual_v
~~~
其中 `mode_id=0` 表示 Skip，`mode_id=1` 表示 P16×16；P16 MV 使用 qpel 整数单位；residual 是与最终 qpel MV 和后端重建参考一致的闭环像素域 residual。
### 十三、为什么第一阶段需要 Oracle I 帧
后续帧的压缩不是彼此独立的。当前单参考 `IPPP` 链路中，第一个 P 帧引用 I 帧的重建结果，后续 P 帧通常继续引用上一张已经重建的参考帧，因此整条运动补偿和 residual 链最终由第一张 I 帧锚定。
后端需要的是完整的 I 帧编码状态，而不只是第一帧原始 RGB/YUV。当前 `oracle_state.bin` 中的 I 帧部分包括：
- 每个 I 宏块的 I16×16 预测模式；
- 色度帧内预测模式；
- 量化后的稀疏变换系数；
- 编码后重建的 Y、U、V 参考像素；
- 初始化后续 P 帧参考链所需的固定配置。
后续运动补偿必须引用编码器和解码器共同得到的重建帧，不能直接引用原始第一帧。对于 P16 宏块，闭环 residual 的概念为：
$$
R_t=X_t-\operatorname{MC}(\hat X_{t-1},v_t)
$$
其中 $\hat X_{t-1}$ 是后端上一帧的重建结果，$v_t$ 是最终 qpel MV。只要 I 帧或中间参考帧发生变化，后续 residual 的定义也会随之变化。
### 十四、第一阶段为什么固定 step 20 Oracle I 帧
第一阶段准备统一使用 step 20 最终视频的真实 I 帧压缩状态作为参考锚点，只让状态头学习：能否从 step 5、10、15、20 的 latent 或 hidden 中，提前预测 step 20 后续 P 帧的最终压缩状态。
关系为：
~~~text
step 5/10/15/20 的 latent 或 hidden
+
统一的 step 20 Oracle I 帧状态
↓
预测 step 20 后续 P 帧的 mode/MV/residual
↓
状态 adapter
↓
标准 H.264 码流
↓
JM 与 FFmpeg 双解码验证
~~~
对应的监督关系为：
$$
z_5\rightarrow y_{20}^{P},\quad z_{10}\rightarrow y_{20}^{P},\quad z_{15}\rightarrow y_{20}^{P},\quad z_{20}\rightarrow y_{20}^{P}
$$
其中 $y_{20}^{P}$ 只表示 step 20 的最终 P 帧模式、MV 和闭环 residual；step 20 I 帧状态由 `target_oracle_state_path` 提供，不作为第一阶段模型预测目标。
这样做的目的包括：
- 暂时排除第一帧预测误差；
- 固定所有实验的参考锚点和闭环条件；
- 单独验证早期 DiT 特征是否已经包含最终后续帧的压缩信息；
- 暂时不修改 DiT 主体，只冻结 DiT 并训练轻量状态头；
- 让 step 5、10、15 的预测结果可以在相同后端条件下公平比较。
不能把 step 5 的 Oracle I 帧与 step 20 的 P 帧状态直接拼接，因为两者的重建参考不同，step 20 residual 不再与 step 5 的参考链闭合。
### 十五、Oracle I 帧实验的结论边界
Oracle I 帧只是当前实验中的理想固定条件，不是最终系统永久依赖的输入。现阶段严谨的结论应表述为：
> 在给定最终 step 20 I 帧编码状态和重建参考的条件下，早期 DiT 特征能否预测最终视频后续 P 帧的压缩状态，并生成可被标准解码器一致解码的 H.264 码流。
现阶段不能直接声称已经完全跳过所有 VAE Decode 和传统编码，因为 step 20 第一帧仍然由 Oracle 初始化；也不能声称已经实现完全在线的 step 5 提前退出系统，因为真实推理时尚未解决如何在 step 5 获得最终 I 帧参考。
### 十六、Oracle I 帧的后续替换路线
建议按以下顺序推进：
1. 固定 step 20 Oracle I 帧，验证早期 step 预测最终后续 P 帧状态是否可行；
2. 分别用理想 Oracle P 状态和真实模型预测 P 状态跑通“预测状态 → H.264 → 标准解码”完整链路；
3. 比较 step 5、10、15 的模式、MV、residual、码率、解码质量和长参考链漂移，确定最早可退出 step；
4. 将 Oracle I 帧替换为“第一帧正常 VAE Decode + 传统帧内编码”的结果；
5. 如有必要，再增加轻量的 latent/hidden → I 帧状态预测头；
6. I 帧来源改变后，必须基于实际重建第一帧和实际后端参考链重新构造 residual 标签并进行闭环训练。
对于图生视频任务，还可以直接使用给定首帧进行正常帧内编码，作为比 I 帧状态预测更简单的工程路径。
### 十七、prediction.npz 接口
真实状态头至少输出：
| 字段 | 形状 | 含义 |
|---|---|---|
| `mode_id` | `[P,Hmb,Wmb]` | 0=Skip，1=P16×16 |
| `mv_qpel` | `[P,Hmb,Wmb,2]` | P16 qpel MV |
| `residual_y` | `[P,Hmb,Wmb,16,16]` | Y 像素域闭环 residual |
| `residual_u` | `[P,Hmb,Wmb,8,8]` | U residual |
| `residual_v` | `[P,Hmb,Wmb,8,8]` | V residual |
| `source_frame_index` | `[P]` | 可选，默认 1 到 P |
也可以用 `[P,Hmb,Wmb,2]` 的 `mode_logits` 替代 `mode_id`。Skip 的 MV 和 residual 会被忽略，Skip MV 由 H.264 邻域规则确定性推导。
### 十八、全链路测试输出
理想和真实预测测试目录均包含：
~~~text
<timestamp>/
├── predicted_state.bin
├── predicted.264
├── predicted_jm_decoded.yuv
├── predicted_ffmpeg_decoded.yuv
├── predicted_encoder_recon.yuv
├── adapter_result.json
├── prediction_replay_result.json
├── ffprobe.json
└── logs/
~~~
验收字段位于 `prediction_replay_result.json`：
~~~json
{
  "validation": {
    "jm_matches_ffmpeg": true,
    "standard_decode_valid": true
  }
}
~~~
只有 `standard_decode_valid=true` 才表示生成的码流能够被 JM 和 FFmpeg 一致解码。
### 十九、覆盖、续跑与备份规则
- 完整且闭合通过的 Oracle 导出：跳过；
- 不完整 Oracle 导出：重命名为 `step_<NNN>_incomplete_<timestamp>` 后重建；
- 当前 schema 且记录数一致的训练数据集：跳过；
- 旧 schema 或不完整训练数据集：重命名为 `training_dataset_stale_<timestamp>` 后重建；
- 重复 `sample_id + step`：拒绝并写入 `rejected.jsonl`；
- 两个全链路测试：每次新建时间戳目录，永不覆盖历史结果。
### 二十、当前研究边界
当前验证的是：
~~~text
Oracle step 20 I 帧初始化
+ 从 step 5/10/15/20 latent 或 hidden 预测最终 P 帧状态
→ 标准 H.264
~~~
当前尚未覆盖：I 帧状态预测、多参考帧、变量 P 分块、B 帧、CABAC、动态 QP、状态头在线接入 DiT 和自适应提前退出控制器。
