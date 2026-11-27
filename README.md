# LMP 与 JM 视频编码实验交接

## 接手入口

这条线研究的是**用 Wan 的 latent 给 JM 提供运动矢量候选，减少 H.264 编码中的搜索开销**。VAE 解码生成完整 17 帧 RAW/YUV，JM 用当前源帧和它自己上一帧的重建画面做运动估计。Latent 候选告诉 JM 可以先到哪里找相似块。JM 继续计算运动补偿和残差，并写出标准 H.264 码流。

直接从 latent 打包成独立码流的 E0/E1 实验由原负责人继续。LMP 早期报告另有组名 `E0`、`E1-cached`、`E2`、`E3`。在本交接中，它们分别指 JM 原生候选、缓存预测头、final latent 匹配、conditional hidden 匹配。实际接入最新 JM 编码对照的是 E2。 [最新闭环报告](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/06_final_report.md)

- **接手分支** `codex/experiment-archive-20261122`。截至 2026 年 11 月 27 日，远端已核对到 `6a0c65c672f0904c1758fc35384813d20e384bbe`。 [归档分支](https://github.com/luoxy5520/video-compression/tree/codex/experiment-archive-20261122) · [固定版本](https://github.com/luoxy5520/video-compression/tree/6a0c65c672f0904c1758fc35384813d20e384bbe)
- **第一次复跑** 从 `experiments/lmp_continuation_20261122/README.md` 和 `scripts/smoke.py` 入手。在 113Server 使用两个历史开发样本，先核验 C0 的码流 SHA-256，再读每组的编码质量和计时。 [复跑说明](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/README.md)
- **当前结论** 两条开发视频上的自适应融合通过现有画质、码率和解码核验，含先验的链路耗时中位数增加 **6.14%**。新策略先在这两条视频上验证。 [试点门槛和汇总](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/results/summary.json)

## 分支、提交与实验阶段

下表的提交都在同一个 GitHub 归档分支上。短提交号对应链接中的完整 SHA。

| 实验 | 已完成的工作 | 归档提交与代码目录 |
| --- | --- | --- |
| 11 月 15 日，LMP 原始评估 | 核对源帧、latent 和历史编码资产。对照 JM 快速全搜 B0 与原生 EPZS B1。导出 JM 候选，离线比较 final latent、hidden、原生 MV 及融合的覆盖率和 SAD。额外诊断了模式候选。 | [`7a682c5`](https://github.com/luoxy5520/video-compression/commit/7a682c5b38ff6914437cd623180ef22c34be7012) · [`lmp-h264-prior`](https://github.com/luoxy5520/video-compression/tree/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp-h264-prior) |
| 11 月 21 日，候选与搜索诊断 | 在 8 条开发视频分析补回与挤占了哪些原生候选。测试不同空间聚合、特征匹配和时间位置，补查两条历史验证样本。给 EPZS 加整数、亚像素及逐块计数。在线完成原生、原生早停、DiT 单种子、DiT 门控的两视频闭环对照。 | [`7a682c5`](https://github.com/luoxy5520/video-compression/commit/7a682c5b38ff6914437cd623180ef22c34be7012)、[`d37d13d`](https://github.com/luoxy5520/video-compression/commit/d37d13ddc19eab8385dda3076f7bd0aa5ce5fadc) · [`lmp-e1-e2-followup-20261121`](https://github.com/luoxy5520/video-compression/tree/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp-e1-e2-followup-20261121) |
| 11 月 22 日，LMP 闭环续跑 | 在静态茶杯和运动玩具车上跑 7 组方法。每组一次暖机、三次正式重复，共 56 次编码。增加三组等整数预算对照，用同一次运行的墙钟测量先验加编码。14 组码流均经 JM 和 FFmpeg 解码，解码像素与各组 JM 编码器重建一致。 | [`b3658a1`](https://github.com/luoxy5520/video-compression/commit/b3658a1896e79f40af68234a82ec95a0e9c7f451)、[`017bdab`](https://github.com/luoxy5520/video-compression/commit/017bdabdf795295ed54e2fa62349cc5c686ec331)、[`d37d13d`](https://github.com/luoxy5520/video-compression/commit/d37d13ddc19eab8385dda3076f7bd0aa5ce5fadc) · [`lmp_continuation_20261122`](https://github.com/luoxy5520/video-compression/tree/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122) |
| 11 月 22 日，Wan 到原生 JM 基准 | 单独测完整 prompt → MP4 请求及 Text Encoder、DiT、VAE、转 YUV、JM、封装阶段。也重编历史 RAW，做码流复现与解码校验。用于估算编码在完整流水线的占比。 | [`e850e8f`](https://github.com/luoxy5520/video-compression/commit/e850e8f98b2d4327c097e9ec7a5097e73c5aba52)、[`6a0c65c`](https://github.com/luoxy5520/video-compression/commit/6a0c65c672f0904c1758fc35384813d20e384bbe) · [`t2v_pipeline_baseline`](https://github.com/luoxy5520/video-compression/tree/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/t2v_pipeline_baseline) |

服务器原始历史还记录了 `codex/lmp-h264-prior` 的 `f080a45`，以及 `lmp-continuation-20261122/release` 中的 `1a74db1`（固定预算）、`14527e4`（独立阶段插桩）、`ae340bd`（服务器交付状态）。读旧报告时会看到这些 SHA。接手检出使用上表的 GitHub 归档分支。 [历史审计](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/00_inventory.md) · [续跑报告 Git 索引](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/06_final_report.md#9-git-与复现)

## 改代码从哪里进

| 要做的事 | 代码入口 | 对应证据 |
| --- | --- | --- |
| 改候选生成 | `lmp_continuation_20261122/scripts/smoke.py` 的 `prior()` 调用历史 `lmp-h264-prior/probe.py` 的 `match()`。final latent 是 `16×5×60×104`，聚成 `5×30×52×64`，做 9×9 归一化特征匹配。相邻 latent 时间位置的位移近似分配给四张 P 帧，再写成 qpel 候选文件。 | `lmp-h264-prior/LMP_REPORT.md`、`results/candidate_analysis.csv` |
| 改候选何时进入 JM | 门控和原生分支从 11 月 21 日的 `build_d3.py`、`jm_changes.patch` 追到 `me_epzs_int.c` 的 `EPZS_integer_motion_estimation`。本轮 `scripts/build.py` 复制历史 `jm-perblock`，注入 `fixed_search.c.inc` 的固定预算分支。 | `configs/policy.json`、`configs/jm_fixed_search.patch`、`logs/server_commands.jsonl` |
| 跑闭环和查质量 | `scripts/smoke.py` 生成候选，调 `lencod.exe`，用 JM `ldecod.exe` 和 FFmpeg 解码，再读取帧数、分辨率、帧率、解码哈希、亮度 PSNR 及码率。 | `results/runs.csv`、`repeats.jsonl`、`verified.jsonl`、`logs/runs/` |
| 查瓶颈和阶段耗时 | `scripts/summarize.py` 汇总搜索、码率和画质。可选 `scripts/profile_detail.py` 单独建插桩版，测宏块处理和进程 RSS，并检查探针开关的 C0 码流一致性。 | `results/timing_breakdown.csv`、`phase_detail.json` |
| 跑完整 VAE → JM 基准 | `t2v_pipeline_baseline/scripts/profile_pipeline.py` 跑 prompt 到 MP4，`profile_encoder.py` 重编历史 YUV，`profile_vae.py` 和 `profile_rgb.py` 拆解上游。 | `BASELINE_REPORT.md`、`results/stage_summary.csv`、`server-records/` |

上述路径分别相对实验目录。C0 是同构建原生 EPZS。固定预算组每个 P16 宏块用 **18 个唯一整数候选**，随后保留 JM 原生亚像素细化。`C-pixel-fixed` 用原生中心与零矢量，`C1-fixed` 用两个 E2 中心，`C2-fixed` 用原生中心加 E2 首选。`C1-seed` 尝试一个额外 E2 候选，`C2-adaptive` 在该候选降低初始代价时放宽首道停止阈值，`C-pixel-adaptive` 是无先验的阈值对照。 [冻结参数](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/configs/policy.json) · [闭环入口](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/scripts/smoke.py)

## 用来验收复跑的结果

- **离线候选。** 历史验证集在 K=2、2 px L∞ 半径下，原生覆盖率 81.18%，E2 单独 50.80%，原生与 E2 融合 80.70%。在另一组 8 条开发视频上，E2 补回原生漏掉的 1,269 / 17,419 块（7.29%）。融合挤掉 1,342 个原生命中，补回 692 个。 [候选报告](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp-h264-prior/LMP_REPORT.md) · [候选四格表](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/results/candidate_analysis.csv)
- **等预算闭环。** 玩具车上 C2-fixed 对比 C-pixel-fixed，码率 **−7.43%**、P 帧 Y-PSNR **+0.145 dB**、链路耗时 **+4.70%**。相对原生 C0，码率 **+17.44%**、P 帧 **−0.422 dB**、链路耗时 **+3.82%**。 [逐视频实测](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/results/runs.csv)
- **自适应闭环。** 静态茶杯与玩具车的 C0 链路分别为 **0.992 / 1.128 s**，C2-adaptive 为 **1.055 / 1.196 s**。两条都通过质量门槛。先验约 **51–55 ms**，计时从读取缓存 latent 开始，到 JM 编码结束。LMP 本轮在单线程 CPU 上运行。 [计时分解](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/results/timing_breakdown.csv)
- **上游全链路。** 单卡 A100 的两条原生完整请求中位数为 **32.04 / 32.48 s**。独立阶段统计中 VAE 均约 **1.20 s**，原生 JM 编码占阶段总墙钟 **3.27% / 3.69%**。辅助版 prompt→MP4、16 条验证集的完整闭环、多 QP 和 BD-rate 仍待测试。 [原生流水线报告](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/t2v_pipeline_baseline/BASELINE_REPORT.md)

## 如何拿到代码与复跑

交接包保留四个实验目录，可以直接查看。也可以按以下命令从 GitHub 只检出相关目录，并固定到交接版本。

```bash
export GIT_LFS_SKIP_SMUDGE=1
git clone --filter=blob:none --sparse --single-branch \
  --branch codex/experiment-archive-20261122 \
  https://github.com/luoxy5520/video-compression.git lmp-jm
cd lmp-jm
git sparse-checkout set \
  experiments/lmp-h264-prior \
  experiments/lmp-e1-e2-followup-20261121 \
  experiments/lmp_continuation_20261122 \
  experiments/t2v_pipeline_baseline
git switch --detach 6a0c65c672f0904c1758fc35384813d20e384bbe
```

### 113Server 所需资产

| 资产 | 已有位置与用途 |
| --- | --- |
| Python 环境 | `/home/luoxiangyu/Wan2.1/.venv/bin/python`。续跑脚本依赖 PyTorch、NumPy，编解码和封装还调用 FFmpeg、FFprobe。 |
| JM 19.0 编解码器和配置 | `/home/luoxiangyu/a_new48_20261107/kit`。C0 的配置组合是 `JM-backend/cfg/encoder_baseline.cfg` 加 `linux/skip_p16.cfg`，解码器在 `linux/JM/bin/ldecod.exe`。 |
| 续跑构建底稿及旧审计 | `/home/luoxiangyu/lmp-e1-e2-followup-20261121` 下的 `jm-perblock`、`supplement-analysis/input_audit.json`、`repo/results-e1-e2-followup`。构建用 GCC，历史一致性配置为 `make -j4 CFLAGS=-fcommon`。 |
| 历史匹配和基线 | `/home/luoxiangyu/lmp-h264-prior.uFe5ZV` 下的 `probe.py`、`baseline.log`。历史 `probe.py` 还导入 `/home/luoxiangyu/dit-paired-experiment/a_state_20261107/motion_comp.py`。 |
| 两条开发输入 | `sample_0000` 和 `sample_1004` 的 RAW/YUV、`final_latent.pt` 及 SHA-256 见 `input_audit.json`。latent 原目录在 `/mnt/data_hdd/luoxiangyu/dit_pairs/` 的两个历史批次。两条视频均为 832×480、17 帧、16 fps、YUV420。 |
| 原始续跑和视频样例 | `/home/luoxiangyu/lmp-continuation-20261122` 下的 `logs/runs/` 和 `videos/`。原生全流水线产物在 `/home/luoxiangyu/experiments/t2v_pipeline_baseline.VN08gB`。 |

在原服务器账号核对了构建底稿、解码器、虚拟环境、历史匹配脚本和输入审计文件可读。若用独立账号，先核对这些目录的访问权限。脚本按 `Path.home()` 找历史目录，可在复跑副本里把路径配置映射到有权访问的实际位置。完整 JM 底稿、RAW/YUV、latent 缓存和视频在服务器上。

从检出的仓库根目录或交接包解压根目录执行下列命令。复跑目录用 `mktemp` 新建，便于保留旧试验结果。CPU 24 是历史实测用的亲和性，换 CPU 时在记录里注明。

```bash
src="$PWD/experiments/lmp_continuation_20261122"
run_dir=$(mktemp -d "$HOME/lmp-jm-repro.XXXXXX")
mkdir -p "$run_dir"/configs "$run_dir"/logs/runs "$run_dir"/results "$run_dir"/videos
cp -R "$src/scripts" "$run_dir/"
cp "$src/configs/policy.json" "$run_dir/configs/"
cd "$run_dir"
py="$HOME/Wan2.1/.venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$py" scripts/record.py "构建 JM" "$py" scripts/build.py
taskset -c 24 "$py" -B scripts/smoke.py
"$py" scripts/record.py "汇总闭环指标" "$py" scripts/summarize.py
```

C0 期望码流 SHA-256 可在同一版本的 `results/runs.csv` 对照。率失真对照采用固定 QP28、32 px 搜索范围、1I+16P、单参考、Skip/P16×16 和同一完整 VAE 源。验收时先确认 JM、FFmpeg 解码像素与各组编码器重建的 SHA-256 一致，再读码率、P 帧 Y-PSNR、末四帧 PSNR 和整条链路墙钟。历史试点要求逐视频码率至多 +3%，平均 P 帧至多降 0.10 dB，最差视频至多降 0.30 dB，平均末四帧至多降 0.20 dB，链路中位数至少节时 10%。 [验收实现](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/scripts/summarize.py) · [参考结果](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/results/runs.csv)

## 可接续的隔帧试验

用 final latent 和同一 17 帧 YUV 比较 **全部 P 帧原生 JM、全部 P 帧启用 latent 候选、仅第 2、4、…、16 帧启用候选**。帧号从 0 开始，其余帧走原生 JM。先在 JM 注入处做逐帧开关。当前 17 帧共享 5 个 latent 时间位置，隔帧启用仍可能调用同一批 latent 匹配。记录每帧候选启用状态与实际评估次数、P 帧和末四帧画质、码率，以及含候选处理的链路时间。隔帧闭环和去噪过程中提前确认帧定稿都处于待实验状态。 [候选生成与 JM 调用](https://github.com/luoxy5520/video-compression/blob/6a0c65c672f0904c1758fc35384813d20e384bbe/experiments/lmp_continuation_20261122/scripts/smoke.py)
