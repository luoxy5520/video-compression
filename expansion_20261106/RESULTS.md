# 扩容32条与真实模式头训练：已完成

32条全新视频 = 8个prompt × 4个seed，保存5/10/15/20步骤，共128条特征记录、32份唯一最终监督目标。旧4条数据保留，两个批次不要仅按sample_id混合，跨批次使用(dataset_id,sample_id,step)。

## 验收
- 128条 latent/hidden 维度与有限值检查通过，raw文件长度正确。
- 128条Oracle编码/状态回放码流逐字节一致，JM/FFmpeg均与参考重建一致，回放运动搜索为0。
- 32个step20 clean latent及其raw与各自solver最终输出完全相同。
- 每样本4步均关联同一最终标签；真实训练读取JM的mode_id。
- 明确prompt分组：20条训练/4条验证/8条测试（记录数80/16/32），无prompt或seed跨集合。
- 独立test.sh输出与训练结束评价、保存的逐视频预测的混淆矩阵完全一致。

## 测试结果

| 模型 | 输入step | Accuracy | Macro-F1 | Skip recall | P16 recall |
|---|---:|---:|---:|---:|---:|
| A0最终步头 | 20 | 0.8094 | 0.7586 | 0.8634 | 0.6601 |
| A2共享时间条件头 | 5 | 0.7896 | 0.7310 | 0.8553 | 0.6080 |
| A2共享时间条件头 | 10 | 0.8195 | 0.7684 | 0.8778 | 0.6584 |
| A2共享时间条件头 | 15 | 0.8230 | 0.7730 | 0.8799 | 0.6657 |
| A2共享时间条件头 | 20 | 0.8225 | 0.7737 | 0.8761 | 0.6742 |

固定全部预测Skip：Accuracy=0.7345、Macro-F1=0.4235。训练集多数类为P16，但测试集Skip占多数，故同时报告两种固定类别基线，避免只展示训练多数类基线夸大提升。

A0训练15epochs，A2课程总计60epochs，二者训练预算不同；不能把二者差异单独归因为中间特征带来的改进。旧4条pilot测试prompt与新32条也不同，不是固定测试集学习曲线。测试视频只有2个prompt、8条视频，验证只有1个prompt，结果只支持继续研究，不构成充分的泛化结论或提前退出判据。

## 代码与产物
- train_mode.py：15042参数，Conv3D+step/20条件；冻结DiT，读取缓存特征训练。A0为stage_1.pt；共享头为stage_4.pt。
- train.sh/test.sh：训练/独立只评价入口。
- results_expand32/：四阶段权重、history.json、metrics.json、evaluation.json、report.html、逐视频指标及32份测试模式预测。
- dataset_index32/：索引、原包汇总、显式prompt划分；特征与标签实际位置在服务器。原包默认分组与显式训练分组不同，训练以config_train32.json为准。
- audit32.json：完整数据审计。

服务器：
```text
特征和raw：/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106/
Oracle与标签：/home/luoxiangyu/senior-oracle-expand32-20261106/output/
训练权重/指标：/mnt/data_hdd/luoxiangyu/dit_pairs/training_expand32_20261106/
Oracle完整备份：/mnt/data_hdd/luoxiangyu/oracle_experiments/expand32_20261106/
```

新特征/raw约7.6GiB，Oracle完整产物约8.7GiB；本次Git只推送代码、配置、索引、模型和报告，没有把这两批大文件再次直接塞入普通Git。

## 结论边界及下一阶段
本轮已完成A0.1与A2-final的模式预测训练。模式预测有初步可读性，但尚未训练MV和residual，模式预测NPZ不能单独接入完整H264回放。Oracle I条件仍存在，没有在线提前退出控制器；step5的模式F1不能证明step5可安全退出。

后续按原计划进入A0.2（最终特征+oracle/预测模式→qpel MV），再A0.3（oracle模式/MV/重建参考条件下residual），最终联合后端质量与码率评价。400条规模尚未启动；应先冻结独立测试集、扩充prompt多样性，再做固定测试集学习曲线。
