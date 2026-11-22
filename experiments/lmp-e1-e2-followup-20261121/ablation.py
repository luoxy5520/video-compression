from run_followup import *
import torch
sys.path.insert(0,str(OLD));import probe
rows=[]
# Prespecified 81 offsets per macroblock, same physical +/-64px interval span.
# Fine grid is nonuniform to keep the number of comparisons fixed.
fine_offsets=(-8,-4,-2,-1,0,1,2,4,8)
def coarse_match(z,metric):
 if metric=='cosine':z=probe.normalization(z)
 elif metric=='correlation':z=probe.normalization(z-z.mean(axis=-1,keepdims=True))
 return (16*np.stack([probe.match(z[j],z[j-1]) for j in range(1,5)])).repeat(4,axis=0)
def fine_match(z):
 z=probe.normalization(z);offs=sorted([(x,y) for y in fine_offsets for x in fine_offsets],key=lambda a:a[0]**2+a[1]**2);ys,xs=np.meshgrid(np.arange(1,60,2),np.arange(1,104,2),indexing='ij');out=[]
 for j in range(1,5):
  costs=[]
  for x,y in offs:
   valid=(ys+y>=0)&(ys+y<60)&(xs+x>=0)&(xs+x<104)
   v=((z[j,ys,xs]-z[j-1,np.clip(ys+y,0,59),np.clip(xs+x,0,103)])**2).mean(axis=-1);v[~valid]=np.inf;costs.append(v)
  out.append(np.asarray(offs)[np.argsort(np.stack(costs,axis=-1),axis=-1,kind='stable')[...,:8]])
 return (8*np.stack(out)).repeat(4,axis=0)
for s in DEV:
 p=probe.DATA/probe.SETS[int(int(s.split('_')[1])>=1000)]/s/'final_latent.pt';latent=probe.tensor(p).numpy();z=latent.reshape(16,5,30,2,52,2).transpose(1,2,4,3,5,0).reshape(5,30,52,64);fine=latent.transpose(1,2,3,0)
 a=np.load(OUT/'stage-d1'/f'{s}_blocks.npz');pos=a['positions'];truth=a['mv_qpel'];ix=tuple(pos.T)
 for name,fn in [('2x2_cosine',lambda:coarse_match(z,'cosine')),('2x2_L2',lambda:coarse_match(z,'L2')),('2x2_correlation',lambda:coarse_match(z,'correlation')),('1x1_cosine_nonuniform81',lambda:fine_match(fine))]:
  times=[]
  for repeat in range(4):
   t=time.perf_counter();c=fn();times.append(time.perf_counter()-t)
  c=probe.unique(c[ix]);dist=np.abs(c[:,:2]-truth[:,None]).max(axis=-1).min(axis=1);epe=np.linalg.norm(c[:,:2]-truth[:,None],axis=-1).min(axis=1)/4
  rows.append(dict(sample=s,method=name,N=len(truth),K=2,mean_candidates=float((c[:,:2,0]!=100000).sum(axis=1).mean()),coverage=float((dist<=8).mean()),epe_median=float(np.median(epe)),epe_p90=float(np.percentile(epe,90)),match_s=statistics.median(times[1:]),min_s=min(times[1:]),max_s=max(times[1:]),comparisons_per_mb=81))
 print(s,flush=True)
csvout(OUT/'stage-d2/d2_matching_ablation.csv',rows);csvout(OUT/'stage-d2/d2_spatial_ablation.csv',[x for x in rows if x['method'] in ('2x2_cosine','1x1_cosine_nonuniform81')])
