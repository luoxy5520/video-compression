"""Generate a transparent milestone/final report; missing data remain NOT RUN."""
import csv
import json
from pathlib import Path
import numpy as np
from run import ROOT,save

def read(path,default):
    p=ROOT/path
    if not p.exists():return default
    if p.suffix=='.csv':return list(csv.DictReader(p.open()))
    return json.loads(p.read_text())

def main():
    uniform=read('results/E1_0/rows.json',[])
    sensitivity=read('results/E1_1/sensitivity.json',[])
    mixed=read('results/E1_2/mixed_precision_results.json',[])
    reuse=read('results/E1_3/skip_probe_results.json',[])
    latency=read('results/latency_breakdown.json',[])
    matched=read('results/E1_0/quality_matched_h264.csv',[])
    stability=read('results/E1_1/channel_rank_stability.csv',[])
    comparisons=read('results/E1_2/actual_rate_matched_comparison.csv',[])
    mixed_h264=read('results/E1_2/mixed_quality_matched_h264.csv',[])
    temporal=read('results/E1_3/temporal_similarity.csv',[])
    proxy_correlations=read('results/E1_1/statistics_distortion_correlations.csv',[])
    deployable=[v for v in comparisons if v['split']=='validation' and v['gain_db'] and json.loads(v['policy'])['method'] in ('M2','M2_three','M3')]
    best_mean=None
    if deployable:
        best_mean=max(np.mean([float(v['gain_db']) for v in deployable if v['policy']==p]) for p in {v['policy'] for v in deployable})
    text=['# Experiment 1 — Final Latent 冗余分析与率失真优化','']
    def section(title,body):text.extend(['## '+title,'',body,''])
    summary=f'统一位宽实测 {len(uniform)}/144 配置；通道敏感度 {len(sensitivity)}/882 配置；混合精度 {len(mixed)} 条记录（含搜索）；参考复用/局部扰动 {len(reuse)}/816 配置。'
    if best_mean is not None and best_mean<0:
        summary+=f'\n\n本轮混合精度未达到基础成功门槛：最好的冻结可部署候选在相近实际码率下，验证平均PSNR-Y仍比Uniform低 {-best_mean:.3f} dB；本轮推荐继续使用Uniform按预算选位宽。低残差能量复用优于随机，并不表示失真已可接受。'
    section('1. Executive Summary',summary+'\n\n这是按当前实测产物生成的报告；未完成项目标记 NOT RUN，不以已实现代码代替实验完成。现有结果不能证明低位宽Latent已优于H.264。')
    section('2. Research Questions','依次检验中间位宽、通道敏感度、固定/统计代理混合精度、时间空间残差与闭环复用。所有方法无训练、不改权重、不改去噪步数；实际Codec仅输入Final Latent。Latent可压缩性不等于DiT内部计算可跳过性。')
    section('3. Experiment 0 Baseline','results/baseline_gate.json 实测 PASSED：两开发样本INT2/4/8共6配置的重建Latent、PSNR、参考YUV及原码流逐字节一致。Wan VAE SHA256核验一致。开发/验证划分沿用E0，两train样本开发，16条val评价；不读取test张量。E0原结果与源码不覆盖。')
    lines=['验证集逐视频指标的算术平均；完整码流Bytes包含头部及量化参数。','', '| Bit width | n | Bytes mean | PSNR-Y mean dB |','|---|---:|---:|---:|']
    for b in range(2,9):
        vals=[v for v in uniform if v['split']=='validation' and v['method']==f'Q{b}']
        if vals:lines.append(f"| INT{b} | {len(vals)} | {np.mean([v['total_bytes'] for v in vals]):.0f} | {np.mean([v['psnr_y'] for v in vals]):.3f} |")
    lines+=['','H.264 Medium：逐视频PSNR-Y匹配，log(Bytes)线性插值，只在已测量QP曲线覆盖区间内比较，禁止外推。各位宽有效视频数不同，不将不同子集的均值当成整体比较。']
    for b in range(4,9):
        vals=[float(v['latent_over_h264_ratio']) for v in matched if v['split']=='validation' and v.get('preset','medium')=='medium' and v['method']==f'Q{b}' and v['latent_over_h264_ratio']]
        if vals:lines.append(f'- INT{b}：{len(vals)}/16 有效，Latent/H.264实字节比算术平均 {np.mean(vals):.3f}。')
    lines+=['','INT5/INT6补齐了画质区间，但相近质量的实际码率仍明显高于H.264。不存在脱离质量/码率预算的单一最佳位宽。Q0无损，保存的348.13dB为E0数值保护上限（MSE=0），不解读为有限失真。BD-Rate NOT RUN；可选范围裁剪消融 NOT RUN。']
    if len(uniform)==144:
        base=[v for v in uniform if v['split']=='validation' and v['method']=='Q4']
        for width in (5,6):
            vals=[v for v in uniform if v['split']=='validation' and v['method']==f'Q{width}']
            gain=np.mean([v['psnr_y'] for v in vals])-np.mean([v['psnr_y'] for v in base])
            increase=np.mean([v['total_bytes'] for v in vals])/np.mean([v['total_bytes'] for v in base])-1
            lines.append(f'\n相对E0沿用的INT4，以16val平均值计：INT{width}画质提高 {gain:.3f} dB，但真实Bytes增加 {increase:.1%}；这是质量/码率折中，不是同码率改进。')
        reproduced=read('results/E1_0/e0_reproduction_validation.csv',[])
        if len(reproduced)==48:
            delta=[float(v['byte_delta']) for v in reproduced if v['method']=='Q4']
            lines.append(f'\n16val的INT2/4/8共48个旧配置PSNR全部复现（差值<1e-6 dB）；同INT4平均字节变化 {np.mean(delta):.0f} Bytes，来自元数据字段差异，不记作算法收益。')
    section('4. E1.0 完整位宽率失真分析','\n'.join(lines))
    vals=[v for v in stability if v['split']=='validation']
    body=f'已完成扰动记录 {len(sensitivity)}/882。固定排名只用两开发样本INT4单通道RGB MSE增量建立，在进入验证前冻结。单通道效应不相加推断联合失真。'
    frozen=read('configs/frozen_channel_ranking.json',{})
    if frozen:body+='\n\n开发最敏感Top4：'+str(frozen['ranking'][:4])+'；最低敏感四通道：'+str(frozen['ranking'][-4:])+'。这是开发固定排序，不将验证排序反馈给Codec。'
    if len(vals)==16:
        body+=f"\n\n16条验证：固定排序与逐视频排序Spearman平均 {np.mean([float(v['spearman']) for v in vals]):.3f}；Top4重叠率 {np.mean([float(v['top4_overlap']) for v in vals]):.3f}；Top8重叠率 {np.mean([float(v['top8_overlap']) for v in vals]):.3f}。详见channel_rank_stability.csv。"
    else:body+='\n\n跨视频稳定性完整分析 NOT RUN。'
    body+='\n\n通道统计与实际像素域失真增量的逐视频Pearson/Spearman见statistics_distortion_correlations.csv；这些关联只作探索性描述，不证明因果或跨分布泛化。INT4/INT6实际Latent量化MSE由既有码流补充计算于channel_statistics_augmented.csv，属于事后分析，未反馈到冻结编码策略；原quantization_mse字段为INT8误差。'
    for proxy in ('variance','energy','quantization_mse','quantization_mse_int4','quantization_mse_int6','symbol_entropy','temporal_corr'):
        values=[float(v['spearman']) for v in proxy_correlations if v['split']=='validation' and v['proxy']==proxy and v['spearman']]
        if values:body+=f'\n- {proxy}：16val逐视频Spearman平均 {np.mean(values):.3f}，正相关比例 {np.mean(np.array(values)>0):.1%}。'
    if len(vals)==16:
        body+='\n\n是否值得固定排序，须同时看这里的稳定性和E1.2真实码率收益；能量高不直接等价于通道重要。代理关联不等于代理分配必然改善联合失真。验证排名仅作分析，不重新排序Codec。'
    section('5. E1.1 通道敏感性与跨视频稳定性',body)
    body=f'混合精度记录 {len(mixed)}。Random有五个固定开发种子；Fixed覆盖Top2/4/8/12及三种两级、两种三级组合；Adaptive使用variance/energy/quantization MSE/entropy。验证前仅按开发结果冻结每类两项配置。最近码率筛选不是同码率结论，最终表在真实码流区间内插值。'
    valid=[v for v in comparisons if v['split']=='validation' and v['gain_db'] and json.loads(v['policy'])['method'] in ('M2','M2_three','M3')]
    if valid:
        policies=sorted({v['policy'] for v in valid});stats=[]
        for policy in policies:
            gains=[float(v['gain_db']) for v in valid if v['policy']==policy]
            stats.append(dict(policy=json.loads(policy),n=len(gains),mean_gain_db=float(np.mean(gains)),worst_gain_db=min(gains),benefit_fraction=float(np.mean(np.array(gains)>0))))
        save(ROOT/'results/E1_2/validation_policy_summary.json',stats)
        best=max(stats,key=lambda v:v['mean_gain_db'])
        body+='\n\n冻结候选中验证集观察到的最高均值（多候选比较，不是新独立测试结论）：\n\n```json\n'+json.dumps(best,indent=2)+'\n```'
        for preset in ('medium','fast'):
            values=[float(v['latent_over_h264_ratio']) for v in mixed_h264 if v['split']=='validation' and v['preset']==preset and json.loads(v['policy'])==best['policy'] and v['latent_over_h264_ratio']]
            if values:body+=f'\n\n该候选与H.264 {preset}在重叠质量区间内：{len(values)}/16有效，实际Bytes比平均 {np.mean(values):.3f}。'
        body+='\n\n这只确定当前冻结候选中的条件性最佳；需要与随机控制在相近实际码率的重叠区间比较，才能判断重要性选择本身的贡献，不以平均理论位宽或最佳单例宣称成功。'
        if best['mean_gain_db']<0:
            body+='\n\n所有六种冻结可部署候选的验证收益比例均为0/16，已经未通过优于Uniform的基础门槛；因此不宣称优于随机的基础成功，也不强行进入联合优化。Random开发含五种子，验证只保留两种开发筛选配置，其实际码率覆盖有限。'
    else:body+='\n\n完整验证收益 NOT RUN。'
    body+='\n\n近似Oracle仅在开发集、三个真实预算上运行受限贪心搜索（最多八轮、前四敏感通道的一位升级），不是全局上界保证。Oracle搜索不计为可部署编码延迟，也不用于冻结验证策略。没有达到门槛时保留负向结果。'
    section('6. E1.2 混合精度率失真优化',body)
    body=f'时间/空间统计覆盖18条样本。视频时间维T与DiT去噪timestep严格区分。直接差分仅为zero-motion基线，残差熵采用1/256数值分箱。1/2/4/8块包含所有16通道，边缘块按实际尺寸计算。受控重建 {len(reuse)}/816。开发四种块尺寸、验证预先固定4×4；INT4四分位扰动，以及5/10/20/30/50%低/随机/高能区域单步t=1与闭环复用。'
    body+='\n\n没有完整Skip码流序列化，实际Skip字节收益 NOT RUN。符号省略数、未压缩mask大小只能表示理论潜力；不能据此声称真实码率下降。局部Latent扰动的像素影响可能跨区域/跨帧扩散。'
    if temporal:
        first=[v for v in temporal if v['split']=='validation' and v['t0']=='0' and v['t1']=='1']
        later=[v for v in temporal if v['split']=='validation' and int(v['t1'])-int(v['t0'])==1 and int(v['t0'])>0]
        body+=f"\n\n16val相邻切片：Z0→Z1平均Pearson {np.mean([float(v['pearson']) for v in first]):.3f}、MSE {np.mean([float(v['mse']) for v in first]):.3f}；后续相邻切片平均Pearson {np.mean([float(v['pearson']) for v in later]):.3f}、MSE {np.mean([float(v['mse']) for v in later]):.3f}。首切片与后续切片分别报告，不从相关性直接推断可省略。"
    if len(reuse)==816:
        body+='\n\n固定4×4、16val的实际重建：'
        for strategy in ('low','random','high'):
            values=[v for v in reuse if v['split']=='validation' and v['configuration']==dict(operation='reuse',ratio=.1,strategy=strategy,mode='closed_loop')]
            loss=[next(u['psnr_y'] for u in uniform if u['sample']==v['sample'] and u['method']=='Q8')-v['psnr_y'] for v in values]
            body+=f"\n- 10% {strategy}闭环复用：PSNR-Y均值 {np.mean([v['psnr_y'] for v in values]):.3f} dB；相对INT8平均下降 {np.mean(loss):.3f} dB，最差下降 {max(loss):.3f} dB；末四帧PSNR均值 {np.mean([v['last4_psnr_y'] for v in values]):.3f} dB。"
        for quartile in (0,3):
            values=[v for v in reuse if v['split']=='validation' and v['configuration']==dict(operation='int4',quartile=quartile)]
            body+=f"\n- 残差能量四分位{quartile+1}局部INT4：平均PSNR-Y {np.mean([v['psnr_y'] for v in values]):.3f} dB。"
        body+='\n\n低变化是否足以作为安全依据，须看相对INT8退化而非只看是否优于随机。闭环误差的逐Latent切片MSE、逐视频帧PSNR与末四帧指标全部保留；单步仅改t=1，闭环改四个切片，二者差异同时包含扰动次数，不能全归因于传播。空间聚集程度由mask_spatial_transitions描述，局部实际像素MSE以32×32统计网格记录，不假设其对应某个独立Latent块。'
    section('7. E1.3 时空冗余与预测复用潜力',body)
    body=f'隔离代表配置测量 {len(latency)}/12；完整数据见latency_breakdown.json。量化/打包/熵编码均为真实函数内CPU墙钟；一次预热、五次正式测量。I/O含OS cache，无fsync。VAE操作同步CUDA，返回CPU RGB。另测GPU起点Latent-to-bitstream和D2H，输入为原始保存Latent放回GPU，未重新运行DiT；两端总耗时为GPU起点编码+独立Latent解码+原生VAE。与E0同配置/同元数据重新计时、断言字节相同，并报告额外CPU编码开销；不得直接套用E0历史加速倍数。分片期间的VAE单次耗时只作诊断。另有h264_post_dit_benchmark.json实测GPU起点VAE+YUV转换+H.264 QP28及接收耗时；QP28仅为固定参考，未将其称为同画质速度比较。'
    benchmark=read('results/h264_post_dit_benchmark.json',[])
    if len(latency)==12:
        body+='\n\n| Sample / stream | GPU encode ms | CPU quant / pack / entropy ms | Extra CPU vs E0 ms | Two ends ms | vs QP28 encoding speed |\n|---|---:|---:|---:|---:|---:|'
        for v in latency:
            b=next(x for x in benchmark if x['sample']==v['sample'])
            t=v['timings'];parts=' / '.join(f"{t[k]['mean']*1000:.2f}" for k in ('quantization_s','packing_s','entropy_s'))
            body+=f"\n| {v['sample']} / {Path(v['stream']).stem} | {v['gpu_latent_to_bitstream_s']['mean']*1000:.2f} | {parts} | {v['extra_encode_mean_s']*1000:.2f} | {v['post_dit_two_ends_mean_s']*1000:.2f} | {b['post_dit_encoding_s']['mean']/v['gpu_latent_to_bitstream_s']['mean']:.1f}x |"
        body+='\n\n额外开销可能出现负点估计，须结合五次样本Std判断抖动，不解释为统计显著加速。上表QP28速度比不是同画质比较，完整Prompt到视频延迟 NOT RUN。'
    section('8. Latency Breakdown',body)
    section('9. Actual Bitstream Breakdown','全部实际码率均来自真实.lvc文件。Uniform的header_bytes包含量化参数及元数据，payload为压缩后真实payload。Mixed沿用E0嵌套通道码流，顶层payload包含子码流头，不能将其当成纯熵编码负载；递归分解见actual_bitstream_breakdown.csv，量化/辅助信息JSON值与其余头部（含键名/分隔符/8字节前缀）分别计费，严格相加为完整文件长度。Checksum沿用每个叶子payload SHA256，LVC1版本标识不变。')
    section('10. Ablation Study','主位宽控制相同channel symmetric absmax、Zstd level3、原生VAE和评价口径。Random控制、固定/逐视频统计代理、三级分配及受限近似Oracle见实测表。可选per-tensor/percentile范围消融 NOT RUN。LPIPS NOT RUN：原环境缺少可用包/权重。')
    body='完整逐视频/逐帧结果保留，不剔除低画质和退化样本。INT2可出现极低质量；INT4不能作为与高质量H.264的同画质基线。固定排序可能跨视频不稳定；闭环复用可能累积误差，须依实测表检验，不能先行宣称安全。连接/环境错误与修复见execution_log.md，worker错误保留logs与pipeline_failure.json。'
    breakdown=read('results/actual_bitstream_breakdown.csv',[])
    if breakdown:
        val_names={v['sample'] for v in uniform if v['split']=='validation'}
        means=[]
        for family in ('E1_0','E1_2'):
            per_video=[np.mean([float(v['header_bytes_including_quant']) for v in breakdown if v['sample']==name and family in v['stream']]) for name in val_names]
            means.append(float(np.mean(per_video)))
        body+=f'\n\n重要实现限制：验证集按视频等权的平均头部，Uniform约 {means[0]:.0f} Bytes，嵌套Mixed约 {means[1]:.0f} Bytes。Mixed重复保存每通道元数据，额外开销已如实计入实际码率。本轮负向结果针对当前嵌套LVC1实现及冻结分配策略，不能据此否定共享头部等其他混合精度设计。没有在观察验证结果后修改格式并重称独立验证。'
    section('11. Failure Cases',body)
    section('12. Research Limitations','只有两条开发样本，验证16条且多个Prompt/Seed重复；不是宽分布独立测试。多个冻结候选的验证比较有选择偏差。无LPIPS、未重跑Prompt/DiT全过程；受限Oracle不提供全局最优保证。零运动扰动未实现实际Skip码流、复杂运动搜索或完整模式决策。所有未实际生成的表/图/视频均为NOT RUN。')
    section('13. Conclusions','中间位宽显著改善画质，但INT5/INT6同画质下仍有明显实际字节差距。本轮最佳可部署路线仍为Uniform，具体位宽按目标质量和码率选择；冻结Mixed全部未超过Uniform，不能用平均理论bit数判断收益。通道敏感性确有差异，但固定排序泛化有限，方差/能量代理不构成可靠重要性判据。低残差区域的复用比随机有效，然而10%闭环复用仍显著低于INT8参考，不支持将低变化直接认定为安全Skip。受限贪心Oracle未充分搜索，不能量化真正最优潜力。Latent值级/时空冗余不等于DiT计算冗余。')
    section('14. Next-Step Decision','不启动E1.4/E1.5/E1.6或联合训练。运动补偿/Skip研究需要在复用质量、误差累积和辅助信息开销证据上审核决定；本报告本身不构成自动启动授权。潜在Compression-Aware DiT动机仅为量化敏感度和误差传播，不推出内部Token/计算可跳过。')
    section('15. Reproducibility Instructions','按README顺序执行；基线失败立即停止。使用同一Wan venv和模型哈希。configs保存冻结策略；logs/pipeline_commands.jsonl记录后续实际命令、时间和GPU。源文件SHA和归档Git commit分别记录，后续归档commit不冒充此前运行版本。完整Latent/码流/块明细位于服务器，不需全部下载。')
    section('16. File and Code Manifest','详见file_manifest.md、code_changes.md、execution_log.md。GitHub仓库luoxy5520/video-compression，分支codex/experiment-1-latent-rd；保留E0历史，不覆盖现有远端分支。阶段状态和测量子状态见experiment_status.json，未完成交付不得标记COMPLETE。')
    (ROOT/'final_report.md').write_text('\n'.join(text))
    for label,index in [('E1_0',4),('E1_1',5),('E1_2',6),('E1_3',7)]:
        # Each report is derived from the same observed data, not a separate claim.
        title=[line for line in text if line.startswith(f'## {index}.')][0]
        start=text.index(title);end=next((i for i in range(start+1,len(text)) if text[i].startswith('## ')),len(text))
        path=ROOT/'reports'/f'{label}_report.md';path.parent.mkdir(exist_ok=True);path.write_text('\n'.join(text[start:end])+'\n')

if __name__=='__main__':main()
