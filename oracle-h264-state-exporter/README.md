# Oracle H.264 State Exporter

完整 Windows 交付包及 Oracle I 帧、课程训练和全链路 BAT 说明见 `README_DETAILED_CN.md`。

把 8-bit packed raw RGB 或现成的 raw YUV420p 视频一次转换为：

- 固定配置的 YUV420p 输入；
- JM B1 Oracle 压缩中间状态；
- 训练可读的 NumPy `.npz` 标签；
- 参考 H.264 码流与重建视频；
- 完整参数、哈希和闭合验证结果。

该工具封装了已经验证的 `JM-state-to-H264-b1-continuous-residual` 工程。JM 在 `dump` 模式下执行正常的模式决策与运动估计并导出 Oracle 状态；随后在 `replay` 模式下仅依赖状态文件重新生成标准 H.264，用于验证状态是否闭合。

## 1. 环境

- Windows PowerShell 5.1 或 PowerShell 7；
- FFmpeg 和 FFprobe 已加入 `PATH`；
- Python 3.9+；
- NumPy；
- 本地已有经过修改和验证的 JM B1 工程。

由于 JM 及修改版可执行文件的再分发许可需要单独确认，GitHub 仓库不直接提交 `lencod_state_b1.exe` 和 `ldecod_ref.exe`。首次使用时只需注册本地 JM 工程路径。

## 2. 首次配置

```powershell
.\setup_local.ps1 `
  -JmPackageRoot "C:\path\to\JM-state-to-H264-b1-continuous-residual"
```

该命令生成不会被 Git 提交的 `.oracle-state.local.json`。

## 3. 一条命令导出

```powershell
.\export_oracle.ps1 `
  -InputRgb .\sample.rgb `
  -Width 256 -Height 256 -Frames 16 -Fps 30 `
  -Qp 28 -ModePreset skip-p16 -ReferenceFrames 1 `
  -SampleId sample_0001 -DiffusionStep 30 -Seed 42 `
  -Prompt "a car driving on a road" `
  -FeatureRef "features/sample_0001_step_030.npz"
```

raw RGB 输入默认为逐帧紧密排列的 `rgb24`。如果输入是 `bgr24`、`rgba` 或 `bgra`，使用 `-PixelFormat` 指定。

已有 raw YUV420p 时可直接输入，不会发生二次颜色转换：

```powershell
.\export_oracle.ps1 `
  -InputYuv .\raw.yuv `
  -Width 832 -Height 480 -Frames 17 -Fps 16 `
  -ColorMatrix bt709 -ColorRange limited `
  -Qp 28 -ModePreset skip-p16 -ReferenceFrames 1
```

`SampleId`、`DiffusionStep`、`Prompt`、`Seed` 和 `FeatureRef` 是可选的配对字段，会原样写入 `metadata.json`，用于把 Oracle 标签与师姐保存的 hidden state 或 clean latent 对齐。

也可以显式指定输出目录：

```powershell
.\export_oracle.ps1 `
  -InputRgb .\sample.rgb `
  -Width 256 -Height 256 -Frames 16 `
  -OutputDir .\runs\sample_0001
```

## 4. 输出

每次运行生成一个独立目录：

```text
runs/<sample>-<timestamp>/
├── input_yuv420p.yuv
├── oracle_state.bin
├── oracle_labels.npz
├── state_metadata.json
├── metadata.json
├── reference.264
├── reference_recon.yuv
├── replayed.264
├── closure_result.json
└── logs/
```

其中：

- `oracle_state.bin`：后端可以直接回放的 JMZV1 version 7 状态；
- `oracle_labels.npz`：状态头训练和分析使用的规则张量；
- `reference.264`：正常 Oracle 编码得到的标准码流；
- `replayed.264`：不读取原始像素、仅从状态回放得到的码流；
- `closure_result.json`：逐字节码流一致性、JM/FFmpeg 解码一致性和耗时；
- `metadata.json`：RGB 输入、颜色转换、编码配置、工具版本和文件哈希。

## 5. NPZ 标签

主要数组采用宏块网格布局，其中 `T` 为帧数，`Hmb=height/16`，`Wmb=width/16`：

| 字段 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `slice_type` | `[T,Hmb,Wmb]` | `int16` | JM slice 类型，P=0、I=2 |
| `mb_type` | `[T,Hmb,Wmb]` | `int16` | JM 宏块类型原始枚举 |
| `is_skip` | `[T,Hmb,Wmb]` | `uint8` | P-Skip 标记 |
| `qp` | `[T,Hmb,Wmb]` | `int16` | 宏块 QP |
| `mv_qpel` | `[T,Hmb,Wmb,4,4,2]` | `int16` | List0 4×4 网格最终 qpel MV |
| `ref_idx` | `[T,Hmb,Wmb,4,4]` | `int8` | List0 参考索引 |
| `mv_qpel_mb` | `[T,Hmb,Wmb,2]` | `int16` | 左上 4×4 单元的宏块级 MV |
| `mv_is_uniform` | `[T,Hmb,Wmb]` | `uint8` | 整个宏块是否共享同一个 MV |
| `is_p/is_i` | `[T,Hmb,Wmb]` | `uint8` | P/I 宏块有效掩码 |
| `residual_y` | `[T,Hmb,Wmb,16,16]` | `int16` | P 非 Skip 宏块 Y residual |
| `residual_u/v` | `[T,Hmb,Wmb,8,8]` | `int16` | P 非 Skip 宏块 U/V residual |
| `has_residual` | `[T,Hmb,Wmb]` | `uint8` | 是否存在 residual payload |
| `i_recon_y/u/v` | 宏块网格 | `uint8` | I 帧已重建参考样本 K |

`mv_qpel_mb` 仅在 `mv_is_uniform=1` 时可视为严格的单 MV 表示。变量分块实验应使用完整的 `mv_qpel` 网格。

## 6. 当前边界

- 8-bit packed raw RGB 输入；
- 转换为 8-bit YUV420p；
- H.264 Baseline Profile、CAVLC、I/P、4×4 transform；
- I 帧仅支持 I16×16；
- `ModePreset=skip-p16` 仅允许 Skip/P16×16，适合第一版状态头；
- `ModePreset=variable` 支持 Skip 和变量 P 分块；
- `ModePreset=p16` 是非 Skip P16×16 配置；
- `RDOptimization=0`；
- `.npz` 暂不展开 I 帧稀疏量化系数，但保留 I 帧重建参考样本；
- raw RGB 不携带色彩元数据，RGB→YUV 的复现依赖记录的 FFmpeg 版本与命令。

`skip-p16` 不会修改本地 JM 工程。脚本会在本次输出目录下建立隔离的临时运行副本，仅把已验证的 P16 配置中的 `PSliceSkip` 打开。

## 7. GitHub 发布前检查

仓库可以直接提交脚本、解析器和文档，但发布前仍应：

1. 确认修改版 JM 源码/补丁和二进制的许可；
2. 决定仓库使用的开源许可证；
3. 不提交 `.oracle-state.local.json`、视频、码流、状态二进制和训练数据；
4. 用 GitHub Release 或内部存储单独管理允许分发的工具链产物。

## 8. 构建训练/验证/测试数据集

Oracle 样本导出完成后，可把多个导出目录转换为 P 帧状态头直接使用的标签，并生成按 prompt 或 sample 分组的训练、验证和测试清单。

在尚未取得 DiT 特征时，可以先构建 labels-only 数据集：

```powershell
.\build_dataset.ps1 `
  -ExportsRoot .\runs `
  -OutputDir .\dataset-labels-only `
  -ExpectedModePreset skip-p16
```

取得 final latent 或 hidden state 后，使用严格模式：

```powershell
.\build_dataset.ps1 `
  -ExportsRoot .\runs `
  -FeatureRoot D:\dit-dataset `
  -OutputDir .\dataset-v1 `
  -ExpectedModePreset skip-p16 `
  -StepFilter final `
  -RequireFeatures `
  -CopyFeatures
```

构建器会拒绝以下样本：

- 三项闭合验证未全部通过；
- 缺少 Oracle 状态、标签或重建视频；
- 模式超出 Skip/P16×16；
- P16 MV 不是宏块内统一 MV；
- Skip/P16 与 residual payload 不一致；
- 严格模式下找不到对应的特征文件；
- `sample_id + step` 重复。

默认监督目标不是当前 step 自身的状态，而是同一 `sample_id` 的最终 step 状态。每条清单同时保留：

- `current_label_path`：当前 step 的 Oracle 标签，用于状态稳定性分析；
- `target_label_path`：最终 step 的 Oracle 标签，作为状态头默认训练目标；
- `target_oracle_state_path`：最终 step 的 Oracle I 帧模板，用于预测状态回放；
- `label_path`：兼容字段，默认等于 `target_label_path`；
- `input_step`、`target_step`、`step_fraction`、`remaining_steps`、`timestep` 和 `sigma`：显式区分不同扩散阶段。

构建器还会生成从最终 step 向前扩展的课程学习清单：

```text
manifests/curriculum/
├── stage_01_steps_020_train.jsonl
├── stage_02_steps_020_015_train.jsonl
├── stage_03_steps_020_015_010_train.jsonl
└── stage_04_steps_020_015_010_005_train.jsonl
```

第一阶段只训练 `step=20 -> y20`，随后依次加入 step 15、10、5；所有阶段的默认 `label_path` 均指向相同 sample 的 step 20 标签。`manifests/by_step/` 另提供各 step 独立清单。

输出目录包括：

```text
dataset-v1/
├── labels/<record_id>.npz
├── manifests/all.jsonl
├── manifests/train.jsonl
├── manifests/val.jsonl
├── manifests/test.jsonl
├── splits/train.txt
├── splits/val.txt
├── splits/test.txt
├── rejected.jsonl
└── dataset_summary.json
```

转换后的标签只保留 P 帧，并直接提供：

- `mode_id` 与 `mode_valid`；
- 宏块级 `mv_qpel` 与 `mv_valid`；
- Y/U/V residual 与 `residual_valid`；
- 对应的原视频帧号 `source_frame_index`。

## 9. 中间预测状态一键生成 H.264

状态头推理后需要输出 `.npz`，其中包含：

```text
mode_id 或 mode_logits
mv_qpel
residual_y
residual_u
residual_v
source_frame_index（可选）
```

运行：

```powershell
.\encode_prediction.ps1 `
  -PredictionNpz .\prediction.npz `
  -OracleTemplateState .\oracle_state.bin `
  -OutputDir .\prediction-run `
  -Fps 30
```

该脚本会：

1. 保留 Oracle 模板中的 I 帧初始化状态；
2. 用模型结果覆盖全部 P 帧模式、MV 和 residual；
3. 对 Skip MV 执行标准邻域推导并忽略模型给出的 Skip MV；
4. 将浮点 MV 和 residual 取整，residual 裁剪到 `[-255,255]`；
5. 生成 `predicted_state.bin`；
6. 通过 JM replay 生成 `predicted.264`；
7. 分别使用 JM 和 FFmpeg 解码；
8. 检查两个标准解码器的输出像素哈希是否一致。

输出：

```text
prediction-run/
├── predicted_state.bin
├── predicted.264
├── predicted_encoder_recon.yuv
├── predicted_jm_decoded.yuv
├── predicted_ffmpeg_decoded.yuv
├── adapter_result.json
├── prediction_replay_result.json
├── ffprobe.json
└── logs/
```

当前接口明确依赖 Oracle I 帧模板，但不会复用模板中的任何 P 帧模式、MV 或 residual。
JM replay 产生的 `ReconFile` 只保留作内部诊断，不作为通过条件；这是因为当前修改版 JM 的状态注入位置会让该文件与标准解码结果不同。实际合法性以 JM 解码与 FFmpeg 解码逐像素一致为准。

## 10. paired_data.local.json 一键桥接

师姐的数据目录中需要同时保留 `paired_data.local.json` 及其相对路径指向的 `.pt` 和 `.yuv` 文件。PowerShell 一条命令完成全部记录的 Oracle 导出和训练集构建：

```powershell
.\build_from_paired_data.ps1 `
  -PairedDataJson "D:\exp_2026_11_05\paired_data.local.json" `
  -OutputRoot "D:\exp_2026_11_05_oracle" `
  -JmPackageRoot "D:\JM-state-to-H264-b1-continuous-residual" `
  -Resume
```

也可以直接调用 Windows 批处理入口：

```bat
run_windows.bat "D:\exp_2026_11_05\paired_data.local.json" "D:\exp_2026_11_05_oracle" "D:\JM-state-to-H264-b1-continuous-residual"
```

脚本固定使用当前已闭合的 H.264 配置：Baseline、CAVLC、1I+P、Skip/P16、单参考帧、QP 28、`RDOptimization=0`。它会验证每条 raw YUV 的精确字节数，解析所有相对路径，按 `sample_id/step` 输出 Oracle 状态，并把 latent、hidden、final latent 三类特征路径写入训练清单。`-Resume` 会跳过已经通过三项闭合验证的记录。

若 `training_dataset` 是旧 schema 或上次失败留下的不完整目录，`-Resume` 会先把它重命名为 `training_dataset_stale_<timestamp>` 作为备份，再使用已完成的 Oracle 状态重新构建；不会覆盖备份，也不会重新执行已闭合的 JM Oracle 编码。

若单条 Oracle 导出只留下不完整目录，`-Resume` 会将其重命名为 `step_<NNN>_incomplete_<timestamp>` 后重新生成；闭合验证通过的 Oracle 导出仍然直接跳过。

默认输出：

```text
<OutputRoot>/
├── oracle_exports/<sample_id>/step_<NNN>/
└── training_dataset/
    ├── labels/
    ├── manifests/all.jsonl
    ├── manifests/train.jsonl
    ├── manifests/val.jsonl
    ├── manifests/test.jsonl
    ├── rejected.jsonl
    └── dataset_summary.json
```

调试时可用 `-Steps 5,10` 或 `-MaxRecords 1` 限制记录。默认清单引用原始 `.pt`，避免复制大张量；如需完全独立的数据集包，增加 `-CopyFeatures -CopyBackendArtifacts`。
