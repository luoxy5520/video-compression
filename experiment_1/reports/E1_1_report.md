## 5. E1.1 通道敏感性与跨视频稳定性

已完成扰动记录 882/882。固定排名只用两开发样本INT4单通道RGB MSE增量建立，在进入验证前冻结。单通道效应不相加推断联合失真。

开发最敏感Top4：[14, 8, 3, 15]；最低敏感四通道：[5, 12, 13, 4]。这是开发固定排序，不将验证排序反馈给Codec。

16条验证：固定排序与逐视频排序Spearman平均 0.550；Top4重叠率 0.422；Top8重叠率 0.672。详见channel_rank_stability.csv。

通道统计与实际像素域失真增量的逐视频Pearson/Spearman见statistics_distortion_correlations.csv；这些关联只作探索性描述，不证明因果或跨分布泛化。INT4/INT6实际Latent量化MSE由既有码流补充计算于channel_statistics_augmented.csv，属于事后分析，未反馈到冻结编码策略；原quantization_mse字段为INT8误差。

- variance：16val逐视频Spearman平均 -0.009，正相关比例 43.8%。
- energy：16val逐视频Spearman平均 0.262，正相关比例 75.0%。
- quantization_mse：16val逐视频Spearman平均 -0.221，正相关比例 12.5%。
- quantization_mse_int4：16val逐视频Spearman平均 -0.218，正相关比例 12.5%。
- quantization_mse_int6：16val逐视频Spearman平均 -0.218，正相关比例 12.5%。
- symbol_entropy：16val逐视频Spearman平均 0.079，正相关比例 56.2%。
- temporal_corr：16val逐视频Spearman平均 0.569，正相关比例 100.0%。

是否值得固定排序，须同时看这里的稳定性和E1.2真实码率收益；能量高不直接等价于通道重要。代理关联不等于代理分配必然改善联合失真。验证排名仅作分析，不重新排序Codec。
