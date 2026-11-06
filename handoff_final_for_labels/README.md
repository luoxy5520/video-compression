请读取 final_targets.json，对4份final_raw.yuv分别生成最终oracle标签。YUV路径相对此目录。视频832x480、17帧、16fps、8-bit planar YUV420p、BT709 limited。
固定QP28、Baseline、CAVLC、1I+16P、单参考ref_idx=0、P帧仅Skip/P16x16，I帧使用oracle。
第一阶段state.npz至少含mode uint8 [16,30,52]，0=Skip、1=P16x16。metadata.json接口见 ../experiments_a/a2_final/README.md，raw_sha256从final_targets.json复制。
每样本只有一份最终标签，step5/10/15/20共享它；不要使用中间raw视频生成这份标签。a2_final_pairs.json中特征路径相对服务器原始数据目录，而非此交接目录。
