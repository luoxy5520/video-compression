"""Consolidate completed846 measurements; never count queued training as completed."""
from pathlib import Path
import json,csv,statistics,datetime,math
R=Path(__file__).resolve().parents[1];out=R/'results/d42_pixel_comparison';rows=json.loads((out/'rows.json').read_text());data=json.loads((out/'summary.json').read_text())
assert len(rows)==846 and len({(r['method'],r['video']) for r in rows})==846
assert len({r['method'] for r in rows})==47 and len({r['video'] for r in rows})==18
assert all(r['real_decode'] and all(math.isfinite(r[k]) for k in ('actual_bytes','psnr_y','SSIM_RGB')) for r in rows)
summary={r['method']:r for r in data['summary']};coverage={r['method']:r for r in data['coverage']}
selected=['trained_I3_mid_1000','trained_I3_f1','D42_f1_seed20261127_20000','TP2','P2_long10000_P2','P2_D42_low10000_P2','P2_D42_joint10000_P2','P2_D42_high10000_P2','P3_D42_1000_P3']
lines=['# Experiment 2.1：已完成扩展实验汇总','',f"复核时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}。下表仅包含已完成真实编码、解码和原生VAE评价的结果。",'', '共47个配置×18条历史开发视频=846组评价。训练使用最多42条已核验的合法视频。封存测试集未打开；这18条视频参与研究决策，独立泛化仍未验证。原始DiT和Wan VAE始终冻结。','', '## 实测率失真','', '| 配置 | 平均完整Bytes | 平均PSNR-Y | SSIM-RGB | Uniform同码率ΔdB | Uniform同质量省字节 | 有效视频数 |','|---|---:|---:|---:|---:|---:|---:|']
for name in selected:
    r=summary[name];c=coverage[name]
    assert c['matched_rate_videos']==c['matched_quality_videos']==18
    lines.append(f"| {name} | {r['mean_actual_bytes']:.1f} | {r['mean_psnr_y']:.3f} | {r['mean_SSIM_RGB']:.4f} | {c['mean_matched_rate_delta_db']:.3f} | {100*c['mean_matched_quality_saving']:.2f}% | 18 |")
lines+=['', '低/中/高联合模型从相同D42/20000步变换及TP2/5000步模型起步，均训练10000步、使用相同42条视频和相同采样顺序，仅改变Latent RD的λ倍数0.1/1/10。平均码流与平均PSNR-Y随λ增加，但高λ的额外字节只换来较小PSNR收益。不能据此要求每条视频或Pixel PSNR严格单调。','', '中码率VAE-aware1000步相对其联合RD起点，平均PSNR-Y提高约0.192dB，平均码流减少约542Bytes。高λ联合点在均值上被该VAE-aware点支配，仍完整保留。不同数据量、起点和累计预算的跨模型差异不是单因素因果结论。','', '插值在每条视频的实测Pareto支持内按log(Bytes)计算，不外推；表中有效比较均为18条。稀疏工作点未进行经过核验的BD-Rate计算。完整JSON包含逐视频、逐帧/末四帧、逐切片Latent MSE及Main/Hyper/Metadata分项。','', '## H.264差距与不利案例','', '| 配置 | 同质量码流比中位数（Codec/H.264） | 码流比算术均值 | Codec更小的视频数 | 有效比较视频数 |','|---|---:|---:|---:|---:|']
hrows=list(csv.DictReader((out/'h264_matched_quality_comparison.csv').open()))
for name in ('P2_D42_low10000_P2','P2_D42_joint10000_P2','P3_D42_1000_P3'):
    ratios=[1-float(r['saving_fraction']) for r in hrows if r['method']==name]
    lines.append(f"| {name} | {statistics.median(ratios):.3f} | {statistics.mean(ratios):.3f} | {sum(v<1 for v in ratios)} | {len(ratios)} |")
lines+=['', '部分视频的同质量H.264码流仅约30–50KB，Latent Codec仍约260KB；中位数接近1不代表普遍超过H.264。H.264各QP均现场编码并用独立ffmpeg解码，YUV limited BT709，PSNR-Y直接比较解码Y平面。','', '## 实际GPU起点性能','', '| 配置/视频 | Codec编码秒 | H.264编码秒 | 编码端加速比 | Codec两端总秒 | H.264两端总秒 |','|---|---:|---:|---:|---:|---:|']
for name in ('P2_D42_joint10000','P3_D42_1000'):
    measured=json.loads((R/'results/performance'/name/'deployment_benchmark.json').read_text())
    for r in measured:
        c=r['codec_samples'];h=r['h264_samples'];assert len(c)==len(h)==5
        ce=statistics.mean(s['full_gpu_latent_to_bitstream_seconds'] for s in c);he=statistics.mean(s['gpu_latent_to_h264_seconds'] for s in h);ct=statistics.mean(s['post_dit_two_ends_seconds'] for s in c);ht=statistics.mean(s['two_ends_seconds'] for s in h)
        speed=f'{r["encoder_speed_ratio"]:.3f}' if r['within_quality_tolerance'] else 'NOT COMPARABLE'
        lines.append(f"| {name}/{Path(r['video']).parent.name} | {ce:.3f} | {he:.3f} | {speed} | {ct:.3f} | {ht:.3f} |")
lines+=['', '一次预热、五次正式测量；权重加载不计入稳态时延。H.264选实际画质在±0.5dB内的QP。Codec编码器不调用VAE，H.264编码路径从同一GPU Latent开始、包含原生VAE；两端接收输出均为原始YUV420。编码端约4.8–5.0倍加速不等于两端加速；当前两端总耗时仍略慢于H.264，也未测Prompt/DiT端到端时间。','', '## 正确性与来源复核','', '- 846个配置—视频键唯一、真实解码标志齐全、核心指标有限；可分解文件的Main+Hyper+Metadata等于完整文件Bytes。','- 原2587个检查点/码流SHA256复核零不一致；旧模型、旧结果保留。','- 所有已检查的V1/V2及P3像素训练阶段VAE参数哈希不变、VAE梯度为None。完整5切片FP32因果VAE重计算前向与原接口一致，未将切片各自解码后拼视频。','- TP0/TP1/TP2/TP2Z与D42固定符号组逐视频保持canonical整数符号与重建Latent相同；联合P2/P3允许改变符号，另行完整评测。','- 评价时源码SHA快照及匹配的原始源码归档位于results各source_snapshot.json与source_snapshots/。该归档证明评价代码版本，不声称跨GPU训练逐位复现。','', '## 当前后续工作','', '上次中断的CPU概率索引优化已恢复审计和完整计时，结果单独保存，不修改已完成846组模型或第三方库。新的加强训练另存目录：中码率从1000步完整优化器/RNG状态继续到3000步，保留2000/3000步；低/高λ各新增1000步VAE-aware微调。新模型未评价完成前不纳入本表。','', 'LPIPS/DISTS、P4完整条件变换、GAN、DiT联合训练、独立验证、封存测试和完整算子FLOPs均未运行。卷积名义MACs、模型文件大小及原始时间样本另有记录。']
index_path=R/'results/index_lookup_performance_comparison.json'
if index_path.exists():
    index=json.loads(index_path.read_text());lines+=['', '## 可选概率索引优化：已补完的中断操作','', '18条历史开发视频均验证新旧码流逐字节相同，原独立解码器全部通过。新增CDF边界、相邻浮点值、非连续张量及完整码流回归，七项测试通过。完整GPU起点计时使用相同检查点，两个代表视频各一次预热、五次正式测量；本轮新增训练开始前完成。','', '| 视频 | 原编码秒 | 优化后编码秒 | 编码时延减少 | 原两端秒 | 优化后两端秒 |','|---|---:|---:|---:|---:|---:|']
    for r in index['rows']:
        assert r['bitstream_identical'];lines.append(f"| {Path(r['video']).parent.name} | {r['baseline_encoder_seconds']:.4f} | {r['bucketize_encoder_seconds']:.4f} | {100*r['encoder_latency_saving_fraction']:.2f}% | {r['baseline_two_ends_seconds']:.4f} | {r['bucketize_two_ends_seconds']:.4f} |")
    lines+=['', '编码端实测减少约1.9–3.0%，两端总时延基本不变；一个样本的两端时间略增，也保留原值。该优化通过可选 `--bucketize-indexes` 开关启用，默认历史路径保留。索引单算子的微基准不能替代上述完整流程实测。所有原始时间样本与运行策略保留在 `results/performance/index_*`。']
(R/'reviewed_extension_report.md').write_text('\n'.join(lines)+'\n')
print('reviewed846-row report generated')
