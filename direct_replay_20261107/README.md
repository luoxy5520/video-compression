# 固定残差直接回放与闭环残差校正

固定同一批 8 条独立测试视频。输入沿用历史 step 20 路径 `step_020/clean_latent.pt`，评分目标为 `final_raw.yuv`。Oracle I 帧来自原 B1 模板。测试使用 JM Baseline/CAVLC、1I+16P、单参考、Skip/P16、QP 28、832×480、16 fps。

第一组使用原始 `prediction.npz` 中的三份残差数组，分别测旧模式/MV、仅换模式、仅换 MV、同时替换。模式使用上一轮决策头的 Skip 概率 ≥0.90。P16 MV 取决策头排序首位的整数候选，乘 4 转为 qpel；Skip MV 由 H.264 回放适配器根据邻块派生。固定残差数组指输入逐字节一致。切成 Skip 的块会按语法忽略残差。回放时没有 JM 搜索、候选重排或 Skip 像素检查。原基线与历史码流逐字节比较，每条变体核对同一重建 I 帧与三份残差哈希。

历史 B1 回放只恢复写入码流使用的 MV 缓冲区。JM 的运动补偿读取另一个缓冲区。即使两个解码器一致，历史编码端重建仍可能漂移。修复版在 P16 同步两个缓冲区，在 Skip 走 JM 原生派生，再强制核对编码器重建与解码逐像素一致。固定残差四组的修复前后 PSNR 一致。

第二组在修复后的后端固定新模式/MV 和 Oracle I 帧，分别比较旧残差、`代理源像素 − 当前 MC` 和学习校正后的残差。代理来自上一轮冻结 Wan 解码主干和秩 40 投影。训练采集 20 条 train、4 条 val 的两条闭环轨迹，每条分别使用旧残差和未训练的代理残差来驱动重建。P16 真值为 `原始源像素 − 实际 MC`。15 参数校正头对 Y/U/V 分开训练，输入代理像素、由选中 MV 产生的实际 MC 差值、旧残差及绝对差。运行时每块直接读取当前重建参考上的 MC，再向原生变换/量化提供像素残差。训练仅在生成域 train 拟合，在 val 选择正则系数；测试原始 YUV 只用于评分。

从服务器实验仓库根目录执行。需要上一轮的模型、候选缓存、代理 YUV、原始回放和 Oracle 数据。

```bash
jm=/home/luoxiangyu/jm-direct-replay-20261107
cp -a /home/luoxiangyu/jm-residual-reference-20261107 "$jm"
python3 direct_replay_20261107/patch.py "$jm"
make -C "$jm/lencod" CC="gcc -fcommon" -j8
# 历史复现，保留原始 B1 的参考路径。
bash direct_replay_20261107/run.sh
# 主结果，要求编码器与解码器的重建一致。
bash direct_replay_20261107/run.sh --sync-mv \
  --output /home/luoxiangyu/direct_replay_synced_20261107 \
  --weights /mnt/data_hdd/models/dit_compression/direct_replay_20261107/residual_synced.f64
bash direct_replay_20261107/run.sh --sync-mv --diagnostics \
  --output /home/luoxiangyu/direct_replay_synced_20261107
```

`run.sh` 允许追加 `--output` 和 `--weights` 指定独立复跑路径。默认的实验数据、预测状态、码流、解码及日志位于 `/home/luoxiangyu/direct_replay_20261107/`。主实验的数据、码流和训练记录位于 `/home/luoxiangyu/direct_replay_synced_20261107/`。主权重位于 `/mnt/data_hdd/models/dit_compression/direct_replay_20261107/residual_synced.f64`。实验地图位于 `/home/luoxiangyu/实验地图.html`。
