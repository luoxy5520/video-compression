# 最终 latent 的闭环模式与运动候选

这组实验使用上一轮冻结的 Wan 解码主干和秩 40 投影，将最终 latent 变成源图像代理。按半分辨率粗搜和整数像素细搜提出每块 8 个 MV 候选。在实际 JM 重建参考上记录候选代价，训练一个共享 MLP 预测 8 个相对运动代价和 Skip 概率。编码时 JM 用当前参考给入选候选重新算代价，并执行亚像素细化、运动补偿、变换、量化和熵编码。I 帧由同一源图像代理经 JM 编码。

固定使用原生成域 20 / 4 / 8 条 train / val / test 视频，按 prompt 分组。每条 17 帧、832×480、16 fps，使用 Skip/P16、单参考、QP 28。训练和验证标签都来自以源图像代理为输入的原生半径 8 编码轨迹。测试阶段只在打分时读取 `final_raw.yuv`，所有输出码流均由 JM 和 FFmpeg 分别解码。

| 文件 | 用途 |
| --- | --- |
| `patch_jm.py` | 候选接入、搜索预算、原生代价与模式标签、Skip 检查 |
| `experiment.py` | 代理图像和候选准备、闭环标签采集、训练、阈值标定与实际码流评估 |
| `run.sh` | 服务器运行入口 |
| `results/` | 训练历史、阈值、逐视频测量、训练好的 MLP |

保护分支要求模型预测 Skip 的概率达到 0.90，再用 JM 派生的 Skip MV 对当前重建参考做运动补偿，检查每块亮度均方误差是否 ≤4。通过检查后跳过该块运动搜索，仍由原生量化和模式判断决定最终写 Skip 或 P16。自适应宽窗回退阈值只看 4 条验证视频，以相对同批 native8 的平均亮度 PSNR 降幅 ≤0.15 dB、码率增幅 ≤2% 为验证门槛，在满足门槛的阈值中选跳过宽窗最多的一档。

在服务器实验仓库根目录运行。新的 JM 源树从上一轮已编译的 `jm-residual-reference-20261107` 复制。选用新的运行目录需要更新 `run.sh` 中的 `work`。已完成的特征可复用，评估使用新的 `--eval-name`。

```bash
jm=/home/luoxiangyu/jm-decision-repro
cp -a /home/luoxiangyu/jm-residual-reference-20261107 "$jm"
python3 decision_search_20261107/patch_jm.py "$jm"
make -C "$jm/lencod" clean
make -C "$jm/lencod" CC="gcc -fcommon" -j8
export DECISION_JM="$jm"
bash decision_search_20261107/run.sh --stage collect
bash decision_search_20261107/run.sh --stage fit
bash decision_search_20261107/run.sh --stage evaluate --eval-name repro_guarded
bash decision_search_20261107/run.sh --stage candidates-only --eval-name repro_candidates
```

服务器的实验产物位于 `/mnt/data_hdd/luoxiangyu/dit_pairs/decision_search_20261107/`。每个评估用例目录有 `prediction.264`、JM 和 FFmpeg 解码结果、编码及解码日志。测试集没有教师标签。指标口径及对照见 [RESULTS.md](RESULTS.md)。
