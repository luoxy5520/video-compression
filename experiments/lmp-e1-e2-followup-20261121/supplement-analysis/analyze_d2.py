import sys,os,json,time,csv,hashlib
from pathlib import Path
import numpy as np
import torch
import signal,faulthandler
faulthandler.register(signal.SIGUSR1)
faulthandler.dump_traceback_later(20,repeat=True)
R=Path.home()/'lmp-e1-e2-followup-20261121';OLD=Path.home()/'lmp-h264-prior.uFe5ZV';O=R/'supplement-analysis';O.mkdir(exist_ok=True)
sys.path[:0]=[str(R),str(OLD)];from run_followup import DEV,rawold,sha,csvout,write
import probe
import fast_sad
rng=np.random.default_rng(91);testref=rng.integers(0,256,(480,832),dtype=np.uint8);testmv=rng.integers(-128,129,(30,52,2),dtype=np.int32)
assert np.array_equal(probe.mc_luma(testref,testmv),fast_sad.luma(testref,testmv,fast_sad.planes(testref)))
SAMPLES=DEV+['sample_1028','sample_1029']
# Existing spatial implementation, only function definitions, no experiment loop.
src=(R/'ablation.py').read_text();ns={};exec(src[:src.index('for s in DEV:')],ns)
# Exact historic native candidate and legal block construction; no new hidden computation.
src=(OLD/'probe.py').read_text().split("if __name__=='__main__':")[0];src=src[:src.index('    def sad(p):')]+"    return locals()\n"
src=src.replace("for name, feature in [('E2',z),('E3',h)]:","for name, feature in [('E2',z)]:")
ctx={};exec(src,ctx)
audit={json.loads(x)['sample']:json.loads(x) for x in (OLD/'audit.jsonl').read_text().splitlines()}
rows=[];checks=[];reproduced=[]
for sample in SAMPLES:
 print('loading',sample,flush=True)
 a0=audit[sample];split='dev' if sample in DEV else 'historical_val_diagnostic'
 for key in ('raw','latent','hidden','labels'):
  expected=a0['input']['sha256'] if key=='raw' else a0[key+'_sha256'];actual=sha(a0[key]);assert actual==expected;checks.append(dict(sample=sample,kind=key,sha256=actual,path=a0[key]))
 os.chdir(OLD);a=ctx['analyze'](sample,'train' if sample in DEV else 'val');pos=np.argwhere(a['mask']);ix=tuple(pos.T);truth=a['mv'][a['mask']]
 print('features loaded',sample,flush=True)
 raw=rawold(sample);source=probe.luma(raw/'input_yuv420p.yuv');recon=probe.luma(raw/'reference_recon.yuv')
 z=a['z'];fine=a['latent'].numpy().transpose(1,2,3,0)
 methods={'E0':a['arms']['E0'],'2x2_cosine':a['arms']['E2']}
 for name,fn in [('2x2_L2',lambda:ns['coarse_match'](z,'L2')),('2x2_correlation',lambda:ns['coarse_match'](z,'correlation')),('1x1_cosine_nonuniform81',lambda:ns['fine_match'](fine))]:
  print('matching',sample,name,flush=True);methods[name]=probe.unique(fn()[ix]);print('matched',sample,name,flush=True)
 if 'E1-cached' in a['arms']:methods['E1-cached']=a['arms']['E1-cached']
 hist=json.loads((OLD/'results-validation'/f'{sample}.json').read_text()) if sample not in DEV else next(json.loads(x) for x in (OLD/'probe-exact.jsonl').read_text().splitlines() if json.loads(x)['sample']==sample)
 for method,hname in [('E0','E0'),('2x2_cosine','E2')]:
  coverage=float((np.abs(methods[method][:,:2]-truth[:,None]).max(-1).min(-1)<=8).mean());expected=hist['stats'][hname]['2']['coverage'][1];assert abs(coverage-expected)<1e-10;reproduced.append(dict(sample=sample,method=method,coverage=coverage,historical=expected))
 print('cache',sample,flush=True)
 caches=[fast_sad.planes(x) for x in recon[:-1]]
 def sad(c):
  out=[]
  for k in range(min(2,c.shape[1])):
   expanded=np.zeros((16,30,52,2),np.int32);valid=c[:,k,0]!=100000;expanded[ix]=np.where(valid[:,None],c[:,k],0)
   ss=[]
   for j in range(16):
    pred=fast_sad.luma(recon[j],expanded[j],caches[j]);cost=np.abs(source[j+1].astype(np.int16)-pred).reshape(30,16,52,16).sum(axis=(1,3));ss.append(cost)
   v=np.asarray(ss)[ix].astype(float);v[~valid]=np.inf;out.append(v)
  return np.stack(out,axis=-1).min(-1)
 # Feature confidence uses best-vs-second squared-distance gap, no historic labels.
 feat=probe.normalization(z);cands=(16*np.stack([probe.match(feat[j],feat[j-1]) for j in range(1,5)])).repeat(4,0).astype(np.int16);cands.tofile(O/f'{sample}_candidates.bin')
 confidence=np.zeros((16,30,52),np.float32)
 for j in range(4):
  off=cands[j*4,:,:,:2].astype(int)//16;yy,xx=np.indices((30,52));cost=[]
  for k in range(2):
   x=xx+off[:,:,k,0];y=yy+off[:,:,k,1];v=((feat[j+1]-feat[j,np.clip(y,0,29),np.clip(x,0,51)])**2).mean(-1);v[(x<0)|(x>=52)|(y<0)|(y>=30)]=np.inf;cost.append(v)
  c1,c2=cost;conf=np.where(np.isfinite(c2),(c2-c1)/np.maximum(c2,1e-12),0);confidence[j*4:j*4+4]=conf
 print('texture source shape',source.shape,flush=True)
 tex=[];var=[]
 for frame in range(1,17):
  im=source[frame].astype(np.int16);grad=np.zeros_like(im,dtype=np.int32);grad[:,:-1]+=np.abs(im[:,1:]-im[:,:-1]);grad[:-1,:]+=np.abs(im[1:,:]-im[:-1,:]);tex.append(grad.reshape(30,16,52,16).mean((1,3))/2)
  var.append(np.abs(im-source[frame-1].astype(np.int16)).reshape(30,16,52,16).mean((1,3)))
 texture=np.asarray(tex)[ix];variation=np.asarray(var)[ix]
 packed=dict(positions=pos,truth=truth,magnitude=np.linalg.norm(truth,axis=-1)/4,phase=pos[:,0]%4+1,texture=texture,confidence=confidence[ix],pixel_change=variation)
 # Adjacent-final-MV turn angle, retrospective diagnosis only.
 prev=np.concatenate([np.zeros_like(a['mv'][:1]),a['mv'][:-1]])[ix].astype(float);cur=truth.astype(float);den=np.linalg.norm(prev,axis=-1)*np.linalg.norm(cur,axis=-1);angle=np.full(len(cur),np.nan);ok=den>0;angle[ok]=np.degrees(np.arccos(np.clip((prev[ok]*cur[ok]).sum(-1)/den[ok],-1,1)));packed['turn_degrees']=angle
 for method,c in methods.items():
  print('SAD',sample,method,flush=True)
  valid=c[:,:2,0]!=100000;delta=c[:,:2].astype(float)-truth[:,None];dist=np.linalg.norm(delta,axis=-1)/4;dist[~valid]=np.inf;maxdist=np.abs(delta).max(-1)/4;maxdist[~valid]=np.inf
  packed[method+'_hit']=(maxdist.min(-1)<=2);packed[method+'_epe']=dist.min(-1);packed[method+'_sad']=sad(c);packed[method+'_centers']=c[:,:2]
 np.savez_compressed(O/f'{sample}_offline.npz',**packed)
 print(sample,'done',len(truth),flush=True)
write(O/'input_audit.json',checks);csvout(O/'historical_reproduction.csv',reproduced)
# Freeze descriptive bins from DEV only; never use validation to select a strategy.
all_dev={k:np.concatenate([np.load(O/f'{s}_offline.npz')[k] for s in DEV]) for k in ('texture','confidence','pixel_change')};edges={k:np.quantile(v,[.25,.5,.75]).tolist() for k,v in all_dev.items()};write(O/'descriptive_bins_frozen.json',dict(dev=DEV,edges=edges,purpose='descriptive quartiles only, no policy tuning',motion_edges_px=[0,1,4,16,float('inf')],phase='P frames modulo 4; phases 1-4'))
for sample in SAMPLES:
 a=np.load(O/f'{sample}_offline.npz');split='dev' if sample in DEV else 'historical_val_diagnostic';n=len(a['phase']);methods=[k[:-4] for k in a.files if k.endswith('_hit')]
 groups=[('all','all',np.ones(n,bool))]
 for lo,hi in [(0,1),(1,4),(4,16),(16,np.inf)]:groups.append(('motion',f'[{lo},{hi})px',(a['magnitude']>=lo)&(a['magnitude']<hi)))
 for phase in range(1,5):groups.append(('phase',str(phase),a['phase']==phase))
 for k,e in edges.items():
  for b in range(4):groups.append((k,f'Q{b+1}',np.digitize(a[k],e)==b))
 for lo,hi in [(0,30),(30,90),(90,181)]:groups.append(('turn',f'[{lo},{hi})deg',(a['turn_degrees']>=lo)&(a['turn_degrees']<hi)))
 for lo,hi in [(0,1),(1,4),(4,16),(16,np.inf)]:
  for phase in range(1,5):groups.append(('motion_x_phase',f'[{lo},{hi})px_phase{phase}',(a['magnitude']>=lo)&(a['magnitude']<hi)&(a['phase']==phase)))
 for kind,label,m in groups:
  if not m.any():continue
  for method in methods:
   e=a[method+'_epe'][m];sd=a[method+'_sad'][m];rows.append(dict(sample=sample,split=split,group=kind,bin=label,method=method,N=int(m.sum()),hits=int(a[method+'_hit'][m].sum()),coverage=float(a[method+'_hit'][m].mean()),epe_mean=float(e.mean()),epe_median=float(np.median(e)),epe_p90=float(np.percentile(e,90)),sad_mean=float(sd.mean()),sad_median=float(np.median(sd)),sad_sum=float(sd.sum()),epe_sum=float(e.sum())))
csvout(O/'d2_stratified.csv',rows)
# Pooled dev metrics computed from block arrays, including proper pooled percentiles.
pooled=[]
for method in ['E0','2x2_cosine','2x2_L2','2x2_correlation','1x1_cosine_nonuniform81','E1-cached']:
 for group in ('all','motion','phase','motion_x_phase'):
  labels=sorted(set(r['bin'] for r in rows if r['group']==group))
  for label in labels:
   arrays=[]
   for s in DEV:
    a=np.load(O/f'{s}_offline.npz');rr=[r for r in rows if r['sample']==s and r['method']==method and r['group']==group and r['bin']==label]
    if not rr:continue
    if group=='all':m=np.ones(len(a['phase']),bool)
    elif group=='phase':m=a['phase']==int(label)
    else:
     part=label.split('px')[0].strip('[').strip(')');lo,hi=map(float,part.split(','));m=(a['magnitude']>=lo)&(a['magnitude']<hi)
     if group=='motion_x_phase':m &= a['phase']==int(label.split('phase')[1])
    arrays.append((a[method+'_hit'][m],a[method+'_epe'][m],a[method+'_sad'][m]))
   if arrays:
    h,e,sd=[np.concatenate([x[i] for x in arrays]) for i in range(3)];pooled.append(dict(group=group,bin=label,method=method,N=len(h),coverage=float(h.mean()),epe_mean=float(e.mean()),epe_median=float(np.median(e)),epe_p90=float(np.percentile(e,90)),sad_mean=float(sd.mean()),sad_median=float(np.median(sd))))
csvout(O/'d2_dev_pooled.csv',pooled)
