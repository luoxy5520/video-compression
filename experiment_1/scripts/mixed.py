"""Development search followed by frozen validation. Approximate greedy oracle."""
import json
import argparse
import numpy as np
import runtime as r
from run import ROOT,E0,MODEL,save,digest,independent,ssim
from analyze import table
from codec.lvc import encode

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--subset',choices=['all','development','validation'],default='all');parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    suffix='' if args.subset=='all' else f'_{args.subset}_{args.worker}'
    assert json.loads((ROOT/'experiment_status.json').read_text())['E1_0_measurements']=='COMPLETE'
    frozen=json.loads((ROOT/'configs/frozen_channel_ranking.json').read_text())
    rank=frozen['ranking']; rows=[]
    manifest=json.loads((E0/'results/validation_manifest.json').read_text())
    sources=[(s['sample'],s['latent'],'development',s) for s in r.samples]+[(s['sample_id'],s['final_latent'],'validation',s) for s in manifest]
    if args.subset!='all':sources=[s for i,s in enumerate([v for v in sources if v[2]==args.subset]) if i%args.workers==args.worker]
    r.vae=r.WanVAE(vae_pth=str(MODEL),device='cuda:0')
    policies=[]
    for lo,hi in ((4,6),(4,8),(5,7)):
        for k in (2,4,8,12):
            policies.append(dict(method='M2',lo=lo,hi=hi,k=k))
            for seed in range(5): policies.append(dict(method='M1',lo=lo,hi=hi,k=k,seed=seed))
            for proxy in ('variance','energy','quantization_mse','entropy'):
                policies.append(dict(method='M3',lo=lo,hi=hi,k=k,proxy=proxy))
    for levels in ([4,5,6],[4,6,8]):
        for k in (2,4,8,12): policies.append(dict(method='M2_three',levels=levels,k=k))
    save(ROOT/'configs/mixed_development_grid.json',policies)
    selected=json.loads((ROOT/'configs/frozen_mixed_policies.json').read_text())['policies'] if args.subset=='validation' else None
    for name,path,split,source in sources:
        if split=='validation' and selected is None:
            # Select closest-rate PSNR gain using development only. Freeze before any validation decode.
            uniform=json.loads((ROOT/'results/E1_0/rows.json').read_text())
            scores=[]
            for policy in policies:
                candidates=[v for v in rows if v['policy']==policy and v['split']=='development']
                gains=[]
                for v in candidates:
                    baseline=min([u for u in uniform if u['sample']==v['sample'] and u['method']!='Q0'],key=lambda u:abs(np.log(u['total_bytes']/v['total_bytes'])))
                    gains.append(v['psnr_y']-baseline['psnr_y'])
                scores.append(float(np.mean(gains)))
            selected=[]
            for method in ('M1','M2','M2_three','M3'):
                choices=sorted([i for i,p in enumerate(policies) if p['method']==method],key=lambda i:scores[i],reverse=True)[:2]
                selected.extend(policies[i] for i in choices)
            save(ROOT/'configs/frozen_mixed_policies.json',dict(source_split='development only',selection='top two per family by nearest actual-rate uniform PSNR gain; nearest is screening only, no matched-rate claim',policies=selected))
        z=r.torch.load(path,weights_only=True,map_location='cpu').numpy()[None]
        rgb,_,_,_=r.pixels(z); raw=r.yuv(rgb)
        assert raw==(E0/'results'/name/'reference.yuv').read_bytes()
        folder=ROOT/'results/E1_2'/name;folder.mkdir(parents=True,exist_ok=True)
        reference=folder/'reference.yuv';reference.write_bytes(raw)
        meta=dict(source,vae_sha256=digest(MODEL),width=832,height=480,frames=17,fps=16,normalization='native WanVAE scale; unchanged final latent')
        cache={}
        def evaluate(cb,policy):
            key=tuple(cb)
            if key not in cache:
                stream=folder/('bits_'+''.join(map(str,cb))+'.lvc')
                sizes,et=r.timing(lambda:encode(z,stream,dict(channel_bits=cb),dict(algorithm='zstd'),meta))
                zh=independent(stream);rec,vt,_,_=r.pixels(zh);rr=r.yuv(rec)
                reconstructed=folder/'reconstruction.yuv';reconstructed.write_bytes(rr)
                cache[key]=dict(sample=name,split=split,channel_bits=cb,**sizes,bitrate_kbps=sizes['total_bytes']*8/(17/16)/1000,encode_s=et,vae_s=vt,ssim=ssim(reference,reconstructed,17,832,480),**r.quality(rgb,rec,raw,rr))
            row=dict(cache[key],policy=policy);rows.append(row);save(ROOT/f'results/E1_2/mixed_precision_results{suffix}.json',rows)
            return row
        for policy in (policies if split=='development' else selected):
            method=policy['method'];k=policy['k']
            order=np.array(rank)
            if method=='M1': order=np.random.default_rng(policy['seed']).permutation(16)
            if method=='M3':
                x=z[0].astype('float64').reshape(16,-1);proxy=policy['proxy']
                if proxy=='variance': importance=x.var(axis=1)
                elif proxy=='energy': importance=np.mean(x*x,axis=1)
                else:
                    scale=np.max(abs(x),axis=1,keepdims=True)/((1<<(policy['lo']-1))-1);q=np.rint(x/scale)
                    if proxy=='quantization_mse': importance=np.mean((x-q*scale)**2,axis=1)
                    else:
                        from analyze import entropy
                        importance=np.array([entropy(v) for v in q])
                order=np.argsort(importance)[::-1]
            if method=='M2_three':
                low,mid,high=policy['levels'];cb=[low]*16
                for c in order[:k+(16-k)//2]:cb[int(c)]=mid
                for c in order[:k]:cb[int(c)]=high
            else:
                cb=[policy['lo']]*16
                for c in order[:k]:cb[int(c)]=policy['hi']
            evaluate(cb,policy)
        # Oracle is isolated from deployable selection and timing, development only.
        if split=='development':
            uniform=json.loads((ROOT/'results/E1_0/rows.json').read_text())
            for bits in (4,5,6):
                budget=next(u['total_bytes'] for u in uniform if u['sample']==name and u['method']==f'Q{bits}')
                cb=[bits-1]*16;current=evaluate(cb,dict(method='M4_approximate_oracle',budget=budget))
                for step in range(8):
                    choices=[]
                    for c in rank[:4]:
                        if cb[c]>=8:continue
                        candidate=cb.copy();candidate[c]+=1
                        trial=evaluate(candidate,dict(method='M4_search',budget=budget))
                        if trial['total_bytes']<=budget and trial['psnr_y']>current['psnr_y']:choices.append((trial,candidate))
                    if not choices:break
                    current,cb=max(choices,key=lambda v:v[0]['psnr_y'])
                rows.append(dict(current,policy=dict(method='M4_approximate_oracle',budget=budget,search='greedy feasible one-bit upgrades; no global optimality guarantee')))
        save(ROOT/f'results/E1_2/mixed_precision_results{suffix}.json',rows)
        reference.unlink(); (folder/'reconstruction.yuv').unlink()
        print(name,split,'DONE',flush=True)
    table(ROOT/f'results/E1_2/mixed_precision_results{suffix}.csv',[dict(v,policy=json.dumps(v['policy']),channel_bits=json.dumps(v['channel_bits']),encode_s=json.dumps(v['encode_s']),per_frame_psnr_y=json.dumps(v['per_frame_psnr_y'])) for v in rows])

if __name__=='__main__':main()
