# DiT 视频压缩配对实验

## 最新：新增32视频真实训练已完成

见 [expansion_20261106/RESULTS.md](expansion_20261106/RESULTS.md)。新批次32视频/128步骤记录已全部通过Oracle闭合与特征验收，A0.1及共享时间条件模式头已完成真实训练。模型、指标、代码和索引在本分支；新批次完整raw/特征和Oracle产物分别约7.6GiB/8.7GiB，保留在服务器与数据盘备份，未再次上传普通Git。下面目录说明仍是原4样本的完整Git数据，勿混淆。


原4样本数据来自 `codex/a2-final-handoff`。包含4个生成视频、每个视频第5/10/15/20步的特征与当前视频，以及完整20步采样的最终视频。原4条Oracle标签和真实pilot训练也已完成，最新32条结果见上方报告。

## 先看哪里

| 目录/文件 | 内容 | 用途 |
|---|---|---|
| `dataset_local/exp_2026_11_05/` | 完整张量、四步骤raw视频、最终视频 | 全部原始实验数据 |
| `dataset_local/exp_2026_11_05/paired_data.local.json` | 16条记录，数据路径相对此JSON所在目录 | 读取每步latent、hidden、当前raw |
| `dataset_local/exp_2026_11_05/a2_final_pairs.json` | 16条输入 → 4份最终目标 | 中间特征预测最终压缩状态的配对关系 |
| `handoff_final_for_labels/` | 4份最终raw、final_targets.json和标签说明 | 师妹生成最终oracle标签的精简入口 |
| `experiments_a/` | A0、A2-final训练/评价代码及配置 | 标签完成后运行实验 |
| `utility_scripts/generate_pairs.py` | 最初的DiT采集脚本 | 复现数据生成，服务器路径需修改 |
| `exp_base.yaml` | 原始采集配置，JSON格式的YAML子集 | 固定模型、采样参数、prompt和seed |

## 四个样本

| sample_id | prompt | seed |
|---|---|---|
| sample_0000 | A fixed camera shows a red toy car moving steadily from left to right on a tabletop, with a static background. | 42 |
| sample_0001 | 同上红车prompt | 43 |
| sample_0002 | A fixed camera shows a blue ball rolling slowly across a wooden floor. | 42 |
| sample_0003 | 同上蓝球prompt | 43 |

Wan2.1-T2V-1.3B，UniPC完整20步，shift=8，CFG=6；832×480，17帧，16fps。权重不在此仓库。

## 数据目录：以一个样本为例

```text
dataset_local/exp_2026_11_05/
├── paired_data.local.json       # 可迁移的各步骤清单
├── a2_final_pairs.json          # 所有步骤指向最终状态的配对清单
├── a0_manifest.json             # 原服务器最终视频清单，含服务器绝对路径
├── manifest.jsonl               # 原采集记录，含服务器绝对路径
├── SHA256SUMS_FULL              # 本次上传全部数据文件的校验和
└── sample_0000/
    ├── step_005/
    │   ├── clean_latent.pt
    │   ├── hidden_cond_last_block.pt
    │   └── raw.yuv
    ├── step_010/                # 同样三份文件
    ├── step_015/                # 同样三份文件
    ├── step_020/                # 同样三份文件
    ├── final_latent.pt
    ├── final_raw.yuv
    └── complete.json
```

每步 `clean_latent.pt` 是该步CFG预测的干净latent，公式 z0=z_sigma-sigma*v_CFG；不是直接存储的noisy latent。维度 [16,5,60,104]，顺序 C,T,H,W。hidden是最后一个DiT block（索引29）的conditional分支输出，维度 [1,7800,1536]，顺序 B,L,D。

`step_005/raw.yuv` 等是对应步骤clean latent经VAE解码得到的完整17帧视频，代表当前阶段估计。`final_latent.pt` 是完整采样后sampler返回的latent；`final_raw.yuv` 是它经VAE解码得到的最终视频。保存位置与语义不同，不凭文件名断言step20与最终latent数值一定不同。

所有raw：8-bit planar YUV420p、832×480、17帧、16fps、BT709 limited，无文件头，每份10,183,680字节。YUV不是可直接用浏览器播放的MP4。可用：

```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 832x480 -framerate 16 dataset_local/exp_2026_11_05/sample_0000/final_raw.yuv
```

## 师妹如何生成最终标签

1. 读取 `handoff_final_for_labels/final_targets.json`，对其中4份 `final_raw.yuv` 分别编码一次。
2. 固定QP28、Baseline、CAVLC、1I+16P、单参考ref_idx=0、P帧仅Skip/P16x16，I帧使用oracle。需真正支持这些约束及状态导出的后端，普通FFmpeg编码并不自动提供完整标签接口。
3. 标签目录每个sample_id一份state.npz和metadata.json，具体接口见 `experiments_a/a2_final/README.md`。
4. 同一样本的step5/10/15/20共享这一套最终标签。不要给这项任务使用step_xxx/raw.yuv的标签。

完整数据目录与精简交接目录里的final_raw.yuv是同一来源的副本；无需两处重复编码。若另做“当前状态读取/状态稳定性”实验，才对各step的raw.yuv单独生成当前状态标签。

## 本地读取

张量和视频本次直接存入Git，无需Git LFS；完整克隆约1GB数据，请保留目录结构。

```bash
git clone --branch codex/a2-final-handoff --single-branch git@github.com:luoxy5520/video-compression.git
cd video-compression/dataset_local/exp_2026_11_05
sha256sum -c SHA256SUMS_FULL  # Linux；macOS使用 shasum -a 256 -c SHA256SUMS_FULL
```

```python
import json
from pathlib import Path
import torch
root = Path('dataset_local/exp_2026_11_05')
records = json.loads((root / 'paired_data.local.json').read_text())['records']
for r in records:
    latent = torch.load(root / r['latent'], map_location='cpu', weights_only=True)
    hidden = torch.load(root / r['hidden'], map_location='cpu', weights_only=True)
    current_video_path = root / r['raw_yuv']

# 最终目标配对清单的特征/视频路径同样相对root；标签路径相对独立labels_root
pairs = json.loads((root / 'a2_final_pairs.json').read_text())['records']
```

## 训练状态与边界

已完成原4条和新增32条的真实模式训练。详细权重与指标见 expansion_20261106/results_pilot4/ 和 results_expand32/。当前实现使用clean latent+step预测最终模式；hidden路径已提供，但hidden输入、MV/residual头尚未实现。

4个样本仅用于链路pilot，红车两seed训练、蓝球两seed测试；同一样本所有步骤不跨集合。不能据此得出泛化或提前退出有效性的结论。训练/评价命令及标签约定见 `experiments_a/a2_final/README.md`，执行记录见同目录 `执行与交接记录.md`。
