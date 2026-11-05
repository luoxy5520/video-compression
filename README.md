# Wan 配对实验 2026-11-05
固定配置见 exp_base.yaml（JSON 是 YAML 子集）。2 prompts × 2 seeds；完整 20 步 UniPC；第 5/10/15/20 次模型评估保存 CFG clean prediction：z0 = z_sigma - sigma * v_CFG。hidden 是最后 transformer block 的 conditional 分支输出，含 padding token，使用 patch 网格裁剪。第20步 clean prediction 和 solver 最终 latent 分开保存，不混为同一对象。

运行 `bash test.sh`，再 `bash train.sh`。此 train.sh 当前执行数据生成；状态头训练需要外部 oracle 标签，尚未实现。

产物：manifest.jsonl，每行 sample_id, step, latent, hidden, raw_yuv, prompt, seed 和解码元信息。latent 为 C,T,H,W；hidden 为 B,L,D。raw 为 832x480、17帧、16fps、BT709 limited planar YUV420p，无文件头。每个 raw 应为10183680字节。complete.json 生成后样本才算完整。严禁以有损 MP4 作为标签源。

## 压缩状态接口草案 v1（需与已有后端确认）
每个样本/步骤一个 NPZ + metadata.json：
- mode uint8 [16,30,52]：0 Skip，1 P16x16。
- mv_qpel int16 [16,30,52,2]：x,y，以1/4亮度像素为单位；Skip 也保存最终推导 MV，不能默认零。
- ref_idx uint8 [16,30,52] 全零。
- residual_y int16 [16,480,832]；residual_u/v int16 [16,240,416]。原始像素减去使用闭环重建参考帧进行规范插值得到的预测。Skip residual 必须零。
- confidence_mode/mv/residual float32 [16,30,52]，范围[0,1]，oracle 可为1。
- oracle 首I帧另存；metadata 记录 QP28、Baseline、CAVLC、单参考、1I+16P、扫描顺序、色彩信息、后端版本及参考重建哈希。

转换：校验维度和离散值 → 光栅顺序遍历宏块 → 从最终MV和邻居推导MVD（Skip 按规范验证）→ residual 整数变换/量化 → CBP/扫描/CAVLC → NAL与slice封装 → 标准解码器验证。参考帧必须是后端逐帧重建结果，不能使用 raw 前一帧。

普通 ffmpeg/libx264 不提供此完整受限状态的导出/注入接口。此仓库没有将其伪装成 oracle；需要已有后端源码或 CLI 协议后接入。当前仅交付特征/raw 数据生成链路。

代码独立仓库，避免向官方 Wan remote 推送。尚无用户自有 remote，因此未 push。重要结果应在首样本验证后生成校验和并复制至独立备份位置。
