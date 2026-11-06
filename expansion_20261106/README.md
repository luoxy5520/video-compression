# 2026-11-06 数据扩容与真实 Oracle 模式训练

顺序：4视频真实标签训练校验 → 32条新视频 → 128步骤Oracle闭合 → 带独立验证集的A0.1/A2-final课程训练。通过后再考虑扩展400条；本轮未启动400条。

## 配置和划分
Wan2.1-T2V-1.3B、20步UniPC、shift8、CFG6、832x480、17帧、16fps，观测5/10/15/20，延续原固定配置。新数据8个prompt×4个seed(101–104)，独立目录 `exp_expand32_20261106`，不会覆盖旧4条。prompt覆盖静态、动物平移、周期运动、摄像机运动、流体、遮挡、非刚性、人类运动。
训练5个prompt/20视频，验证1个prompt/4视频，测试2个prompt/8视频，所有seed和step随prompt分组。训练与测试语义类别也不同，小数据分布偏移仍然很大，只能视为pilot。

## 真实训练
train_mode.py直接读取原包真实training_dataset/manifests/all.jsonl和mode_id，所有输入step均使用target_label_path（最终step20标签）。最终step20与final_latent/final_raw的一致性由管线check验证。
模型15042参数，Conv3D+显式step/20嵌入，输出2类P宏块模式。冻结DiT，训练只读取缓存latent。Adam lr=.001，交叉熵。课程依次20、20+15、20+15+10、20+15+10+5。32视频每阶段15epochs，每阶段按验证集平均Macro-F1保留最佳权重，训练结束才评价测试集。4视频pilot无验证集，仅作为训练管线校验。

## 运行和结果
- generate_pairs.py exp_expand32.json：GPU0生成，保存latent/hidden/当前YUV/最终latent/最终YUV。manifest按样本原子更新；不完整目录归档，完成样本可跳过。
- continue32.py GENERATOR_PID：等待32条完成，执行check、128条Oracle导出回放、明确分组清单、GPU1真实模式训练。
- train_mode.py --config config_pilot4.json：已有4视频的真实标签训练。
- train_mode.py --config config_train32.json：32视频训练。

服务器代码：`/home/luoxiangyu/dit-paired-experiment/expansion_20261106/`
新raw/特征：`/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106/`
新Oracle包与标签：`/home/luoxiangyu/senior-oracle-expand32-20261106/output/`
权重/曲线/指标：`/mnt/data_hdd/luoxiangyu/dit_pairs/training_expand32_20261106/`
进度：pipeline32_status.json；日志：generation32.log、oracle32.log、train32.log。

原包哈希分组可能出现空测试集，本轮训练强制读取config_train32.json中固定划分，另输出manifests/explicit_prompt_split/供复查。测试集不用于选权重。此轮仅模式头，不把Oracle MV/residual混合为“完整预测”，没有宣称端到端提前退出或H264预测链路完成。
