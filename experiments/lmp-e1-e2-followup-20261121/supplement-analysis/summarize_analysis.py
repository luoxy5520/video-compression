import sys,json,csv,math,difflib,hashlib
from pathlib import Path
import numpy as np
R=Path.home()/'lmp-e1-e2-followup-20261121';O=R/'supplement-analysis';sys.path.insert(0,str(R));from run_followup import DEV,csvout,write
SAMPLES=DEV+['sample_1028','sample_1029'];edges=json.loads((O/'descriptive_bins_frozen.json').read_text())['edges'];totals=list(csv.DictReader((O/'search_totals.csv').open()));rescue_rows=[];online_rows=[];deltas=[];examples=[]
joined=[]
for s in SAMPLES:
 a=dict(np.load(O/f'{s}_offline.npz'));pos=a['positions'];idx=(pos[:,0])*1560+pos[:,1]*52+pos[:,2];arms={}
 for arm in ('B1','Native-budget','DiT-seed','DiT-gated'):
  b=np.genfromtxt(O/'encodes'/s/arm/'blocks.csv',delimiter=',',names=True,dtype=None,encoding=None);key=(b['frame']-1)*1560+(b['y']//16)*52+b['x']//16;b=b[np.argsort(key)];assert np.array_equal(np.sort(key),np.arange(24960));arms[arm]=b
  if arm!='B1':
   base=arms['B1'];bn=base['integer_evals'];dn=b['integer_evals']-b['seed_evals'];native_saved=bn-dn;sub_saved=base['subpixel_evals']-b['subpixel_evals'];extra=b['seed_evals'];net=native_saved+sub_saved-extra
   assert net.sum()==(base['integer_evals']+base['subpixel_evals']-b['integer_evals']-b['subpixel_evals']).sum()
   full=dict(sample=s,split='dev' if s in DEV else 'historical_val_diagnostic',arm=arm,baseline_native_integer=int(bn.sum()),method_native_integer=int(dn.sum()),native_integer_saved=int(native_saved.sum()),subpixel_saved=int(sub_saved.sum()),seed_extra=int(extra.sum()),net_saved=int(net.sum()),net_saved_pct=float(net.sum()/(base['integer_evals']+base['subpixel_evals']).sum()*100),blocks_net_less=int((net>0).sum()),blocks_net_more=int((net<0).sum()),blocks_net_equal=int((net==0).sum()),seed_duplicates_initial=int(b['seed_duplicate_initial'].sum()),seed_extra_nonduplicate=int((b['seed_evals']*(1-b['seed_duplicate_initial'])).sum()),threshold_raised=int(b['threshold_raised'].sum()),effective_relaxed_initial_stop=int(((b['predictors_entered']==0)&(b['threshold_raised']==1)&(b['after_seed_cost']>b['initial_stop'])).sum()),early_stop_total=int(np.isin(b['exit_reason'],[1,2,5]).sum()),seed_improved=int(b['seed_improved'].sum()))
   deltas.append(full)
   for kind,m in [('all',np.ones(24960,bool)),('seed_improved',b['seed_improved']==1),('seed_not_improved',(b['seed_evals']==1)&(b['seed_improved']==0)),('duplicate_initial',b['seed_duplicate_initial']==1),('relaxed_initial_effective',(b['predictors_entered']==0)&(b['threshold_raised']==1)&(b['after_seed_cost']>b['initial_stop'])),('native_pattern_skipped',(base['pattern_entered']==1)&(b['pattern_entered']==0))]:
    if m.any():online_rows.append(dict(sample=s,arm=arm,group=kind,N=int(m.sum()),native_integer_saved=int(native_saved[m].sum()),seed_extra=int(extra[m].sum()),subpixel_saved=int(sub_saved[m].sum()),net_saved=int(net[m].sum()),fraction_blocks_net_less=float((net[m]>0).mean())))
 # Offline rescue uses full-search historical labels; this does NOT mean B1 failed.
 e=a['E0_hit'];d=a['2x2_cosine_hit'];rescue=~e&d;miss=~e;sd=a['2x2_cosine_sad']<a['E0_sad'];base=arms['B1'];bc=(base['integer_evals']+base['subpixel_evals'])[idx]
 packed=dict(a);packed['latent_top1_hit']=np.abs(a['2x2_cosine_centers'][:,0]-a['truth']).max(axis=-1)<=8;packed.update(B1_integer=base['integer_evals'][idx],B1_subpixel=base['subpixel_evals'][idx],offline_rescue=rescue,sad_better=sd)
 for arm in ('DiT-seed','DiT-gated'):
  b=arms[arm][idx];packed[arm+'_seed_improved']=b['seed_improved'];packed[arm+'_net_saved']=bc-b['integer_evals']-b['subpixel_evals'];packed[arm+'_seed_extra']=b['seed_evals'];packed[arm+'_exit_reason']=b['exit_reason']
 np.savez_compressed(O/f'{s}_joined.npz',**packed);joined.append((s,packed))
 for kind,m in [('rescued',rescue),('unrescued_miss',miss&~d),('native_hit',e)]:
  for i in np.flatnonzero(m)[np.argsort(-bc[m])[:3]]:
   examples.append(dict(sample=s,category=kind,frame=int(pos[i,0]+1),mb_y=int(pos[i,1]),mb_x=int(pos[i,2]),motion_px=float(a['magnitude'][i]),phase=int(a['phase'][i]),texture=float(a['texture'][i]),confidence=float(a['confidence'][i]),native_sad=float(a['E0_sad'][i]),latent_sad=float(a['2x2_cosine_sad'][i]),B1_evals=int(bc[i]),DiT_seed_net_saved=int(packed['DiT-seed_net_saved'][i]),DiT_gated_net_saved=int(packed['DiT-gated_net_saved'][i]),seed_improved_online=int(packed['DiT-gated_seed_improved'][i])))
# Dev pooled, each dev individually, and the two targeted historical diagnostics separately.
sets=[('dev_pooled',[p for s,p in joined if s in DEV])]+[(s,[p]) for s,p in joined]
for label,ps in sets:
 keys=['latent_top1_hit','E0_hit','2x2_cosine_hit','E0_sad','2x2_cosine_sad','magnitude','phase','texture','confidence','pixel_change','B1_integer','B1_subpixel','DiT-gated_seed_improved','DiT-gated_net_saved'];a={k:np.concatenate([p[k] for p in ps]) for k in keys};n=len(a['phase']);e=a['E0_hit'];d=a['2x2_cosine_hit'];miss=~e;rescue=miss&d;bc=a['B1_integer']+a['B1_subpixel'];sadbetter=a['2x2_cosine_sad']<a['E0_sad']
 groups=[('all','all',np.ones(n,bool))]
 for lo,hi in [(0,1),(1,4),(4,16),(16,np.inf)]:groups.append(('motion',f'[{lo},{hi})px',(a['magnitude']>=lo)&(a['magnitude']<hi)))
 for p in range(1,5):groups.append(('phase',str(p),a['phase']==p))
 for key,ee in edges.items():
  for b in range(4):groups.append((key,f'Q{b+1}',np.digitize(a[key],ee)==b))
 for lo,hi in [(0,1),(1,4),(4,16),(16,np.inf)]:
  for b in range(4):groups.append(('motion_x_confidence',f'[{lo},{hi})px_Q{b+1}',(a['magnitude']>=lo)&(a['magnitude']<hi)&(np.digitize(a['confidence'],edges['confidence'])==b)))
 for group,bin,m in groups:
  den=int((miss&m).sum());rr=rescue&m;nm=miss&~d&m;native=e&m
  if not m.any():continue
  k=int(rr.sum());p=k/den if den else 0;z=1.96;center=(p+z*z/(2*den))/(1+z*z/den) if den else 0;half=z*math.sqrt(p*(1-p)/den+z*z/(4*den*den))/(1+z*z/den) if den else 0
  stat=lambda v,q,fn:float(fn(v[q])) if q.any() else ''
  rescue_rows.append(dict(sample=label,group=group,bin=bin,N=int(m.sum()),native_misses=den,rescued=k,rescue_rate=k/den if den else '',rescue_ci95_lo=max(0,center-half) if den else '',rescue_ci95_hi=min(1,center+half) if den else '',rescued_and_sad_better=int((rr&sadbetter).sum()),sad_better_fraction_among_rescued=float(sadbetter[rr].mean()) if k else '',rescued_B1_integer_mean=stat(a['B1_integer'],rr,np.mean),rescued_B1_subpixel_mean=stat(a['B1_subpixel'],rr,np.mean),unrescued_B1_integer_mean=stat(a['B1_integer'],nm,np.mean),unrescued_B1_subpixel_mean=stat(a['B1_subpixel'],nm,np.mean),rescued_B1_evals_mean=stat(bc,rr,np.mean),rescued_B1_evals_median=stat(bc,rr,np.median),unrescued_miss_B1_evals_mean=stat(bc,nm,np.mean),unrescued_miss_B1_evals_median=stat(bc,nm,np.median),native_hit_B1_evals_mean=stat(bc,native,np.mean),rescued_sad_online_improved_net_saved=int((rr&sadbetter&(a['DiT-gated_seed_improved']==1)&(a['DiT-gated_net_saved']>0)).sum()),top1_coverage_among_rescued=stat(a['latent_top1_hit'],rr,np.mean),online_seed_improved_among_rescued=stat(a['DiT-gated_seed_improved'],rr,np.mean),gated_net_saved_mean_among_rescued=stat(a['DiT-gated_net_saved'],rr,np.mean)))
csvout(O/'rescue_conditions.csv',rescue_rows);csvout(O/'search_decomposition.csv',deltas);csvout(O/'online_groups.csv',online_rows);(O/'block_examples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in examples))
# Aggregate calculation counts only. No wall-time claim from verbose diagnostics.
aggregate=[]
for split,samples in [('dev',DEV),('targeted_historical_val',['sample_1028','sample_1029'])]:
 for arm in ('Native-budget','DiT-seed','DiT-gated'):
  rs=[r for r in deltas if r['sample'] in samples and r['arm']==arm];fields=['baseline_native_integer','method_native_integer','native_integer_saved','subpixel_saved','seed_extra','net_saved','seed_duplicates_initial','threshold_raised','effective_relaxed_initial_stop','seed_improved'];v={k:sum(r[k] for r in rs) for k in fields};v.update(split=split,arm=arm,samples=len(rs));base=sum(int(r['total_evals']) for r in totals if r['sample'] in samples and r['arm']=='B1');v['baseline_all_evals']=base;v['net_saved_pct']=100*v['net_saved']/base;aggregate.append(v)
csvout(O/'search_aggregate.csv',aggregate)
# Focused readable report, backed by CSVs and block join files.
pooled=list(csv.DictReader((O/'d2_dev_pooled.csv').open()));strat=list(csv.DictReader((O/'d2_stratified.csv').open()))
lines=['# D2、补救条件与逐块搜索补充分析','','本轮完成8条开发视频及sample_1028、sample_1029两条历史验证视频的诊断。没有生成新视频、训练模型或重新选择门控参数。','','## 任务1：候选误差','','指标口径：K=2，覆盖半径2像素，按最大横/纵坐标差判定覆盖；EPE为两个候选到历史最终向量的最小欧氏距离（像素）。SAD为两个候选在同一个历史重建参考上的最小16×16亮度绝对误差和，越小越好，范围0–65280。SAD不是最终编码结果。Skip不进入合法运动块分母。','','### 8条开发视频合并','', '| 匹配方法 | 覆盖率 | EPE中位/90分位(px) | 平均SAD |','|---|---:|---:|---:|']
for r in pooled:
 if r['group']=='all':lines.append(f"| {r['method']} | {float(r['coverage']):.2%} | {float(r['epe_median']):.3f} / {float(r['epe_p90']):.3f} | {float(r['sad_mean']):.1f} |")
for s in ['sample_1028','sample_1029']:
 lines+=['',f'### {s}（历史验证诊断，不调参）','','| 方法 | 覆盖率 | EPE中位/90分位(px) | 平均SAD |','|---|---:|---:|---:|']
 for r in strat:
  if r['sample']==s and r['group']=='all':lines.append(f"| {r['method']} | {float(r['coverage']):.2%} | {float(r['epe_median']):.3f} / {float(r['epe_p90']):.3f} | {float(r['sad_mean']):.1f} |")
lines+=['','完整运动幅度、四个时间位置、二者交叉、特征置信度、纹理和运动转向分桶：d2_stratified.csv；开发集合并表：d2_dev_pooled.csv。','', '## 任务2：补救是否有用','', '“离线补救”只指历史原生两个候选未覆盖、latent两个候选覆盖。它不等于JM完整搜索失败，也不保证上线使用的第一个种子会改善真实参考上的编码代价。特征置信度是归一化匹配距离第一/第二名的相对间隔，不是校准的成功概率。纹理是块内平均水平/垂直亮度梯度，只是粗糙程度代理，不能直接命名遮挡或重复纹理。运动幅度/转向来自历史最终向量，仅用于事后分组。','']
for r in rescue_rows:
 if r['sample'] in ['dev_pooled','sample_1028','sample_1029'] and r['group']=='all':lines.append(f"{r['sample']}：原生漏掉{r['native_misses']}块，latent补回{r['rescued']}块（{float(r['rescue_rate']):.2%}）；其中SAD也改善的占{float(r['sad_better_fraction_among_rescued']):.2%}。补救块B1平均评估{r['rescued_B1_evals_mean']:.2f}次，未补回的漏块平均{r['unrescued_miss_B1_evals_mean']:.2f}次；离线第一候选就覆盖的占{r['top1_coverage_among_rescued']:.2%}，上线第一个种子确实降低初始代价的占{r['online_seed_improved_among_rescued']:.2%}。")
lines+=['','条件分析详见rescue_conditions.csv，包含原生漏块分母、补救率及描述性95%区间。区间按块计算，块间相关，因此不能当作独立视频泛化的置信保证。比较搜索负担时应同时查看相同运动幅度分桶，不能把运动更大的块直接与静止块比较。','','## 任务3：计算是否节省','','精确核算：净节省 = 原生整数评估减少 + 亚像素评估减少 − 新增DiT种子评估。正数才是评估次数减少，负数是增加。整数与亚像素调用的实际成本不同，单次还可能提早停止像素累加，因此相加只是未加权调用次数，不能换算CPU时间。下表包含所有P块，包括最后可能选择Skip的块。','','| 分组 | 方法 | 原生整数减少 | 亚像素减少 | 新增种子评估 | 净节省 | 净节省比例 |','|---|---|---:|---:|---:|---:|---:|']
for r in aggregate:lines.append(f"| {r['split']} | {r['arm']} | {r['native_integer_saved']} | {r['subpixel_saved']} | {r['seed_extra']} | {r['net_saved']} | {r['net_saved_pct']:+.2f}% |")
lines+=['','search_decomposition.csv逐视频给出少算/多算/相等的块数、重复初始种子、提出放宽门槛和真正因放宽门槛在初始阶段停止的次数。online_groups.csv按种子是否改善/是否重复等分类。encodes/<sample>/<arm>/blocks.csv保存每块整数/亚像素次数、种子额外次数、停止分支及门控原因；reason_codes.json解释编码。','','回退含义：种子不改善时沿用原生门槛；文件不存在或超出搜索范围时不评估种子；种子改善且门控开启时仅放宽第一道停止门槛，若仍未满足继续原生候选/模式搜索。这不是“完成整套搜索后再回退”。本轮没有改变旧策略，也没有去掉重复种子，重复计算被如实计入。','','逐块跨组差值是闭环运行的实际负担变化，不能解释为同一参考画面上的严格单块反事实；不同组早先的编码选择可能使参考画面不同。全视频评估总量差仍是真实执行次数差。离线补救与B1搜索负担的关联同样不是因果证明。','','## 正确性与交付','','40个视频/组全部与未加逐块统计的冻结编码器码流/重建一致；每条B1又与原始编码器一致。40组均经JM和FFmpeg双解码并与编码器重建SHA对齐。两个旧试点的8组逐块统计另重复一次，计数文件逐字节相同。逐块次数之和与原有全局统计一致。','','本轮只报告计算次数，不从带详细日志的运行时间推断加速。没有执行新训练、生成或策略调参；没有使用封存盲测。新增脚本、统计补丁、校验和和数据全部在supplement-analysis，历史结果保持原样。']
(O/'summary.md').write_text('\n'.join(lines));write(O/'summary_data.json',dict(aggregate=aggregate,overall_rescue=[r for r in rescue_rows if r['group']=='all'],d2_overall=[r for r in pooled if r['group']=='all']))
print(json.dumps(dict(aggregate=aggregate,overall_rescue=[r for r in rescue_rows if r['sample'] in ['dev_pooled','sample_1028','sample_1029'] and r['group']=='all'])),flush=True)
