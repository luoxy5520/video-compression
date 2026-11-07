# 师姐 A0/A1/A2 复现与交接

固定配置：Wan2.1 T2V 1.3B / UniPC 20 步 / 832×480 / 17 帧 / 16 fps；
JM Baseline / CAVLC / YUV420 / QP28 / 1I+16P / 单参考 / Skip 或 P16×16。
A2 默认监督第 s 步自己的 Oracle 状态 y_s，旧 A2-final 是不同实验。

## 直接状态接口

状态头是独立 producer。推理进程不用 JM、VAE、源图像或 Oracle P 标签。
后端的 Oracle I 由实验原稿允许，是后端输入，与模型输入分开。环境在 113Server：

```bash
cd ~/dit-paired-experiment
py=~/Wan2.1/.venv/bin/python
r=~/a_state_pilot_20261107
cfg=expansion_20261106/config_train32.json
kit=~/senior-oracle-expand32-20261106
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$py" a_state_20261107/state.py --latent /path/to/final_latent.pt \
  --step 20 --checkpoint "$r/model/state.pt" --output /tmp/new_prediction.npz
```

中间步输入应为对应的 `step_005/clean_latent.pt` 等，`--step` 取 5、10、15、20。
每个 NPZ 都是 16 个 P 帧，按 `[P时间, 宏块行, 宏块列, ...]`：

| 字段 | 类型和形状 | 含义 |
|---|---|---|
| mode_id | uint8 [16,30,52] | 0 Skip，1 P16 |
| mv_qpel | int16 [16,30,52,2] | 水平 x、垂直 y，1 单位 = 1/4 像素 |
| residual_y | int16 [16,30,52,16,16] | 有符号 Y 预变换像素残差 |
| residual_u / residual_v | int16 [16,30,52,8,8] | 有符号 U/V 预变换像素残差 |
| confidence | float32 [16,30,52,3] | 模式预测置信度、MV 完全匹配概率、残差块 RMSE ≤4 的概率 |
| p16_probability | float32 [16,30,52] | 模式头的 P16 概率 |

Skip 的矢量由 H.264 邻块规则派生，Skip 文件中的 mv/residual 不消费。
MV/残差置信度只在 Oracle P16 上训练；联调时在预测 P16 区域消费，需留意覆盖偏差。当前校准数据太窄，不建议在退出策略中直接把概率当风险保证。
接口文件是研究产物，不是码流；其大小不计入 H.264 码率。无自动上传或发送给他人。

批量训练、导出和最终步回放：

```bash
"$py" a_state_20261107/state.py --config "$cfg" \
  --output /tmp/state_new_run --epochs 24
"$py" a_state_20261107/evaluate.py --config "$cfg" --checkpoint "$r/model/state.pt" \
  --kit "$kit" --jm ~/jm-direct-replay-20261107 --output /tmp/new_eval \
  --split test --steps 5 10 15 20 --replay 20
```

服务器已完成的旧测试 32 份预测位于
`~/a_state_pilot_20261107/eval/test/sample_*/step_*/prediction.npz`。
后端在有对应模板时直接调用 kit 的 `scripts/prediction_to_b1_state.py`；
使用同步了 MV 缓冲区的 JM，设置 `JM_STATE_MODE=replay`、`JM_STATE_FILE`、
`JM_REPLAY_SYNC_MV=1`，输入全零占位 YUV。不得启用 live residual、源代理、候选重排、像素 Skip 门控。
验收需要完整覆盖 26,520 个宏块、ME 时间 0.000 s，编码器与 JM/FFmpeg 像素一致。

## 诊断与扩容

- `discrete_mv.py`：A0.2 最终步 qpel 离散头，报告连续与全零对照、Oracle P16 和预测 P16 交集。跨档定义为任一分量偏离对应 qpel 格点，同时报告至少 1 像素的偏差。
- `oracle_residual.py`：A0.3 将 Oracle mode/MV 和精确 Oracle 重建作为固定条件；严格核验 Oracle 重放、记录原生 MC、只替换像素残差。先在同步 JM 源树副本应用 `patch_coeff.py` 并重编译，再给脚本 `--jm` 指向该副本。该路径保留冻结 Wan 解码前缀。
- `ablate.py`：A1 的 Oracle 替换消融，仅用于诊断；`calibrate.py` 用 val 拟合概率校准并原样评估 test。
- `proxy_state.py`：不让后端重算残差的独立预导出对照；它用冻结 Wan 前缀生成近似参考，带或不带相关性候选。不能用它证明移除 VAE。独立 qpel MC 先用四条验证视频的原生 Oracle MC 做像素级核验。
- `new48.json` 固定新增 6/3/3 个 train/val/test prompt，每个 4 seed；运行现有 `expansion_20261106/generate_pairs.py`，再用 `label_expansion.py` 逐步导出、回放并双解码验证。
- `run_blind.sh` 等新标签完整验收后检查 step 20 与最终文件相等性，训练扩容头并锁定权重，再在三组新 prompt 比较旧模型与扩容模型。测试指标不参与选权重。

当前 A1 的工作线沿用原试验：平均 P 帧 Y PSNR 比完整 JM 低不超过 1 dB、
码率增幅不超过 10%、所有流通过双解码。原稿允许的 Oracle I 仍保留。
17 帧内报告首尾变化；不把它当成更长序列的验收证据。

## 隔离 Oracle I 与延伸对照

`isolate_i.py --oracle-i 原状态文件 --output 新模板.bin --frames 17` 只读开头 I 记录，
用固定配置合成 P 槽位；既有适配器可将独立模型的 NPZ 接到该模板。
单样本已证明产生相同 replay.bin。新的批量评估默认走此路径，记录 Oracle P 读取量为零。

`run_teacher.sh` 用只读 trace 版教师在 train/val 采样，必须先验证固定状态回放与教师逐字节相同。
然后训练同架构、同预算的 Oracle 标签头和教师标签头。训练完锁定两个权重再评估旧测试。
`long_chain.py` 对两条新生成的 33 帧视频进行单 I 帧测试，前端用两个 latent 窗口，
后端保持一条不中断的 P 参考链。此处是跨长度压力测试，不是受支持规格的扩大承诺。

服务器交接包：`~/a_state_pilot_20261107/handoff_state_pilot.tar.gz`；
SHA-256：`28ba99d0ffbc2ba1fa419b5f67d9be3c017c4c6af6fb6bdc95408698685b1a35`。

更大的扩容 hidden checkpoint 位于 `~/a_hidden_blind_20261107/model/state.pt`，独立
推理应传 `--hidden /path/to/hidden_cond_last_block.pt`，而不是 `--latent`。
`run_handoff.sh` 从这份锁定权重导出原 32 条与新增 48 条的 320 份预测；
清单保留 train/val/test，只有 test 可用于独立评估。每个 NPZ 都有 SHA-256。

全量交接包（已逐文件校验）：`~/a_handoff80_20261107/handoff_80x4.tar`。
80 条视频，320 份预测（train/val/test 为 176/64/80 份），包含模型与清单。
校验命令：

```bash
"$py" a_state_20261107/verify_handoff.py \
  ~/a_handoff80_20261107/handoff_80x4.tar /tmp/handoff_validation.json
```

对应新增数据复跑入口是 `run_expand.sh`；复制 `new48.json` 后先设置新的输出目录，
不要与本次归档共用原始生成路径。对照训练入口为 `run_blind.sh`、`run_control.sh`、
`run_hidden_blind.sh` 和 `run_recurrent.sh`，运行目录互相隔离。

## A1 续训：预测 MV 与残差配对

`bash a_state_20261107/run_rollout_residual.sh ~/fresh_rollout_run`。
冻结扩容 hidden 头的模式/MV、主干和置信度层。训练时生成预测状态，使用同一回放后端
记录 P16 实际 MC；只有训练样本的原始像素参与残差监督。重复采集一次更新后的轨迹，
混合两轮监督。验证按真实回放 PSNR 选权重，候选码率需不超过完整 JM 验证均值的 110%。
锁定后在此前 12 条测试上做回归对照，并核验模式、MV 数组逐项不变。
续训候选沿用旧置信度层，残差置信度不能作为已校准的输出使用。

实测 44 train / 16 val，两轮各 8 epoch，选择第 2 轮。在此前的 12 条测试回归中，
22.838 → 22.842 dB，427.71 → 439.73 kb/s，末帧 20.327 → 20.337 dB。
24 条码流均通过零搜索、零 Oracle P 读取与三方重建核验。保留原交接权重；
`results/rollout_round2.pt` 仅为实验候选。

## 跨分支迁移与后台续训

`coarse_fine_mv.py` 复用闭环实验的候选结构，直接从 hidden 输出最终 MV。
Oracle 与闭环教师监督用相同学生结构；此前 12 条盲测仅作为固定回归。
Oracle top-4 用真实标签选候选，属于归因：21.82 dB，高于自选的 21.55 dB，
仍低于原 hidden 的 22.84 dB。

113Server 队列使用持续续训模式，固定 44 train / 16 val。验证结果写入状态文件。

```bash
cd ~/dit-paired-experiment
tail -f ~/a_motion_queue_20261107/queue.log
cat ~/a_motion_queue_20261107/status.json   # 第一组结束后出现
# 参数 0 表示从已完成的组继续训练
bash a_state_20261107/run_motion_queue.sh ~/a_motion_queue_20261107 0
# 请求已运行的队列在当前组结束后停下
touch ~/a_motion_queue_20261107/STOP
```

运行日志在 `trial_*.log`。持续模式逐组继承上一组两种监督下的运动头和锚点，
以 0.0003 的学习率重建优化器。重启会跳过已完成组。若有 STOP 文件，
确认需要恢复后，先移除该文件，再以同一目录重启。目录剩余空间小于 100 GiB 时队列停止调度。
`status.json` 记录验证集对照，交接模型由单独验收决定。
