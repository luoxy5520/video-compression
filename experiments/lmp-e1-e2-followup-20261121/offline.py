from run_followup import *
import torch
# Reuse exact historical construction, stop before costly SAD evaluation.
src=(OLD/'probe.py').read_text().split("if __name__=='__main__':")[0]
src=src[:src.index('    def sad(p):')]+"    return locals()\n"
ns={};exec(compile(src,str(OLD/'probe.py'),'exec'),ns)
os.chdir(OLD)
comp=[];front=[];bins=[];rep=[]
for sample in DEV:
 t=time.perf_counter();a=ns['analyze'](sample,'train');arms=a['arms'];truth=a['mv'][a['mask']];n=len(truth)
 historical=next(json.loads(x) for x in (OLD/'probe-exact.jsonl').read_text().splitlines() if json.loads(x)['sample']==sample)
 hit=lambda p,k,r: (np.abs(p[:,:k]-truth[:,None]).max(axis=-1).min(axis=1)<=r*4)
 for name,p in arms.items():
  actual=hit(p,2,2).mean();expect=historical['stats'][name]['2']['coverage'][1];assert abs(actual-expect)<1e-10,(sample,name,actual,expect)
  rep.append(dict(sample=sample,arm=name,coverage=float(actual),historical=expect,match=True))
 native=a['native'][a['mask']]
 for name in ('E2','E3'):
  p=arms[name];e=hit(arms['E0'],2,2);d=hit(p,2,2);f=hit(arms[name+'+E0'],2,2)
  comp.append(dict(sample=sample,method=name,N=n,E0_hit=int(e.sum()),DiT_hit=int(d.sum()),Both_hit=int((e&d).sum()),E0_only=int((e&~d).sum()),DiT_only=int((~e&d).sum()),Neither=int((~e&~d).sum()),rescue=float((~e&d).sum()/max(1,(~e).sum())),union_coverage=float((e|d).mean()),lost_by_insertion=int((e&~f).sum()),gained_by_insertion=int((~e&f).sum())))
  # Both priorities preserve MVP; duplicate removal matches historical implementation.
  for order,c in [('native_first',np.concatenate([native,p],axis=1)),('dit_second',np.concatenate([native[:,:1],p[:,:1],native[:,1:],p[:,1:]],axis=1))]:
   arms[name+'_'+order]=ns['unique'](c)
 for name,p in arms.items():
  for k in (1,2,3,4):
   for r in (0,2,4,8):
    epe=np.linalg.norm(p[:,:k]-truth[:,None],axis=-1).min(axis=1)/4
    front.append(dict(sample=sample,arm=name,K=k,r_px=r,N=n,hits=int(hit(p,k,r).sum()),actual_mean_candidates=float((p[:,:k,0]!=100000).sum(axis=1).mean()),epe_median=float(np.median(epe)),epe_p90=float(np.percentile(epe,90))))
 pos=np.argwhere(a['mask']);epe=np.linalg.norm(arms['E2'][:,:2]-truth[:,None],axis=-1).min(axis=1)/4;mag=np.linalg.norm(truth,axis=-1)/4
 for phase in range(4):
  for lo,hi in ((0,1),(1,4),(4,16),(16,1e8)):
   m=(pos[:,0]%4==phase)&(mag>=lo)&(mag<hi)
   if m.any():bins.append(dict(sample=sample,phase=phase+1,motion_min_px=lo,motion_max_px=hi,N=int(m.sum()),coverage=float(hit(arms['E2'],2,2)[m].mean()),epe_median=float(np.median(epe[m])),epe_p90=float(np.percentile(epe[m],90))))
 np.savez_compressed(OUT/'stage-d1'/f'{sample}_blocks.npz',positions=pos,mv_qpel=truth,**arms)
 # Cache deployment candidates for EVERY block, without historical truth or mask.
 feature=ns['normalization'](a['z']);offsets=np.stack([ns['match'](feature[j],feature[j-1]) for j in range(1,5)])
 candidates=(16*offsets).repeat(4,axis=0).astype(np.int16)
 candidates.tofile(OUT/'stage-d1'/f'{sample}_candidates.bin')
 write(OUT/'stage-d1'/f'{sample}_timing.json',dict(match_s=a['timing'],full_offline_s=time.perf_counter()-t,feature_existing=True))
 print(sample,'reproduced',n,flush=True)
csvout(OUT/'stage-d1/d1_reproduction.csv',rep);csvout(OUT/'stage-d1/d1_complementarity.csv',comp);csvout(OUT/'stage-d1/d1_budget_frontier.csv',front);csvout(OUT/'stage-d2/d2_temporal_bins.csv',bins)
