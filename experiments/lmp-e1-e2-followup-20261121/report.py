from run_followup import *
import difflib
comp=list(csv.DictReader((OUT/'stage-d1/d1_complementarity.csv').open()));totals={}
for method in ('E2','E3'):
 rr=[r for r in comp if r['method']==method];keys=('N','E0_hit','DiT_hit','Both_hit','E0_only','DiT_only','Neither','lost_by_insertion','gained_by_insertion');v={k:sum(int(x[k]) for x in rr) for k in keys};v.update(rescue=v['DiT_only']/(v['DiT_only']+v['Neither']),union=(v['E0_hit']+v['DiT_only'])/v['N']);totals[method]=v
write(OUT/'stage-d1/d1_aggregate.json',totals)
examples=[]
for s in ('sample_1004','sample_0016','sample_0008'):
 a=np.load(OUT/'stage-d1'/f'{s}_blocks.npz');mv=a['mv_qpel'];hit=lambda p:(np.abs(p[:,:2]-mv[:,None]).max(axis=-1).min(axis=1)<=8)
 e=hit(a['E0']);d=hit(a['E2']);f=hit(a['E2+E0'])
 for kind,m in [('rescue',~e&d),('lost_by_insertion',e&~f),('neither',~e&~d)]:
  for i in np.flatnonzero(m)[:3]:examples.append(dict(sample=s,kind=kind,position_frame_y_x=a['positions'][i].tolist(),mv_qpel=mv[i].tolist(),native=a['E0'][i,:2].tolist(),latent=a['E2'][i,:2].tolist()))
(OUT/'stage-d1/d1_lost_gained_examples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in examples))
# Preserve patches relative to actual historical source and record complete driver scripts.
patch=[]
for p in (ROOT/'jm-d3/lencod/src').glob('*.c'):
 old=kit/'linux/JM/lencod/src'/p.name
 if old.exists() and old.read_text()!=p.read_text():patch.extend(difflib.unified_diff(old.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='a/lencod/src/'+p.name,tofile='b/lencod/src/'+p.name))
(OUT/'jm_changes.patch').write_text(''.join(patch))
for p in ROOT.glob('*.py'):shutil.copy2(p,OUT/p.name)
profile=[];op=[]
for s in (DEV[0],DEV[6]):
 b=json.loads((OUT/'stage-d0'/s/'B1/result.json').read_text());p=json.loads((OUT/'stage-d0'/s/'profile/result.json').read_text());assert b['sha256']==p['sha256']
 pr={k:statistics.median([x[0][k] for x in p['profiles'][1:]]) for k in p['profiles'][0][0]};pr.update(sample=s,bit_exact=True);profile.append(pr)
 match=json.loads((OUT/'stage-d3'/f'{s}_feature_times.json').read_text())['median_s'];best=b['JM_wall_s']-pr['integer_s']+match
 op.append(f"{s}: B1 {b['JM_wall_s']:.4f}s，整数搜索 {pr['integer_s']:.4f}s，读取并匹配 {match:.4f}s；即使完全省去整数搜索、忽略注入成本，估计仍需 {best:.4f}s。")
(OUT/'stage-d0/d0_me_profile.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in profile));(OUT/'stage-d0/d0_opportunity.md').write_text('\n\n'.join(op))
(OUT/'stage-d0/d0_timing_sanity.md').write_text('历史构建使用 make CFLAGS=-fcommon；说明书给出的 CC="gcc -fcommon" 会保留 Makefile 的其它浮点编译选项，使码流变化。已通过无插桩重新编译对照定位，并恢复历史参数；插桩版两条码流与历史 B1 一致。每组1次热身+3次正式重复。先前并行/错误构建结果仅留作诊断，不用于最终计时。计时为函数边界单调时钟；搜索次数统计失真代价函数调用，不等于像素操作数；其它模式与帧级细分尚未实现。')
(OUT/'stage-d2/d2_root_causes.md').write_text('实测支持：在固定两个候选预算下，插入latent会丢掉部分原生正确候选，见D1 lost/gained。时间位置和运动大小分桶仅支持误差相关性，不能单独证明恒速假设是主因。空间与匹配比较见CSV；1×1对照同时改变采样位置分布，不能作单因素因果结论。未测试：人工遮挡/重复纹理标注、逐块SAD、方向变化分桶和时间插值。')
results=json.loads((OUT/'stage-d3/d3_results.json').read_text());table=[];correct=[]
for r in results:
 pr={k:statistics.median([x[0][k] for x in r['profiles'][1:]]) for k in r['profiles'][0][0]}
 row={k:r[k] for k in ('sample','arm','bitrate_kbps','P_Y_PSNR','last4_Y_PSNR','JM_wall_s','feature_s','pipeline_wall_s','min_s','max_s')};row.update(integer_ME_evals=pr['integer_evals'],subpixel_ME_evals=pr['subpixel_evals'],split='dev')
 base=next(x for x in results if x['sample']==r['sample'] and x['arm']=='B1-measured');row.update(pipeline_speedup_pct=100*(1-r['pipeline_wall_s']/base['pipeline_wall_s']),bitrate_change_pct=100*(r['bitrate_kbps']/base['bitrate_kbps']-1),PSNR_change=r['P_Y_PSNR']-base['P_Y_PSNR'],last4_change=r['last4_Y_PSNR']-base['last4_Y_PSNR']);row['quality_pass']=row['bitrate_change_pct']<=3 and row['PSNR_change']>=-.10 and row['last4_change']>=-.10;table.append(row)
 correct.append(dict(sample=r['sample'],arm=r['arm'],dual_decode=r['dual_decode'],stream_sha256=r['sha256']))
csvout(OUT/'tables.csv',table);csvout(OUT/'stage-d3/d3_timing.csv',table);csvout(OUT/'stage-d3/d3_search_counts.csv',[{k:r[k] for k in ('sample','arm','integer_ME_evals','subpixel_ME_evals')} for r in table]);csvout(OUT/'stage-d3/d3_rate_distortion.csv',table)
fallback=[]
for r in results:
 folder=OUT/'stage-d3'/r['sample']/r['arm'];lines=(folder/'encode3.log').read_text().splitlines();blocks=[list(map(int,x.split()[1:])) for x in lines if x.startswith('LMP_BLOCK ')]
 assert len(blocks)==16*30*52,(r['sample'],r['arm'],len(blocks))
 arr=np.asarray(blocks);attempted=int(arr[:,4].sum());improved=int(arr[:,5].sum());fallback.append(dict(sample=r['sample'],arm=r['arm'],blocks=len(blocks),seed_attempted=attempted,seed_improved=improved,improved_fraction=improved/len(blocks),note='full native search retained unless cheap stop threshold met; full fallback reasons not instrumented'))
csvout(OUT/'stage-d3/d3_fallback.csv',fallback)
(OUT/'stage-d3/d3_correctness.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in correct))
textlines=['# 实验结论','', '本次已完成两条视频的小规模真实编码测试；是否加速应同时看匹配开销、编码时间和画质。','', '## 阶段状态','', '- D0：partial/pass（资产、两条历史基线、搜索阶段计时与码流一致性通过；其它模式/逐帧细分未完成）。','- D1：partial（8条历史覆盖全部复现，互补与候选挤占、K/r对照和逐块记录已完成；候选来源占比、完整SAD对照尚未完成）。','- D2：partial（时间位置/运动大小分桶、空间与三种匹配对照已完成；方向变化分桶、全量SAD和人工场景标注未完成）。','- D3：两条开发视频四组测试已完成，全部双解码并与重建一致。未运行其余6条开发视频或多QP测试。','- D4：not_run；未读取封存盲测，未跑16条验证集的新编解码。','', '## 候选是否能补漏','']
for m,v in totals.items():textlines.append(f"{m}: 共 {v['N']} 个合法块；原生漏掉时补回 {v['DiT_only']} 个，补救比例 {v['rescue']:.2%}；不限制总候选数的理想并集 {v['union']:.2%}。融合挤掉 {v['lost_by_insertion']} 个原生正确块，补回 {v['gained_by_insertion']} 个。")
textlines+=['','## 速度与质量','']
for r in table:textlines.append(f"{r['sample']} {r['arm']}: 编码 {r['JM_wall_s']:.4f}s，加上特征处理 {r['pipeline_wall_s']:.4f}s，相对实测原生基线提速 {r['pipeline_speedup_pct']:.2f}%；码率变化 {r['bitrate_change_pct']:+.2f}%，P帧画质变化 {r['PSNR_change']:+.4f}dB，质量线 {'通过' if r['quality_pass'] else '未通过'}。")
success=any(r['arm'].startswith('DiT') and r['quality_pass'] and r['pipeline_speedup_pct']>=10 for r in table)
textlines+=['','## 是否继续','',('出现潜在收益，需要继续其余6条开发视频后再决定验证；本报告不得称实验完成。' if success else '当前小规模结果不足以支持扩大验证。按计划停止扩大编码规模；不能把离线命中率或单独编码耗时称为端到端加速。'),'', '## 修改与限制','', '新增 run_followup.py、offline.py、ablation.py、build_profile.py、build_d3.py、real_encode.py、report.py；修改仅限独立 JM 副本的 mv_search.c 和搜索代价调用所在源文件。详见 jm_changes.patch。历史原件、模型权重均未修改。','', '1×1空间对照使用同样81次位置比较，但偏移采用非均匀细网格，因此它是空间尺度与采样网格的联合改动，不能把差异全部归因于空间聚合。候选代价使用当前编码器真实重建参考；在线不读取历史最终运动向量。','', '逐块日志只记录是否尝试/改善种子，尚未输出每块搜索次数及完整回退原因。种子额外评估在整数搜索函数内、原生预测点之后且完整搜索展开之前；并未替换原生预测列表。pipeline是测得的特征阶段中位数加编码阶段中位数，不是单一进程全流程墙钟；不含Python启动，不能据此宣称生产部署收益。']
(OUT/'summary.md').write_text('\n'.join(textlines));write(OUT/'status.json',dict(potential_success=success,D4='not_run'))
print(json.dumps(dict(totals=totals,table=table,potential_success=success)),flush=True)
