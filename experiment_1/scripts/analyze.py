"""Channel perturbations and latent-only redundancy statistics. No sealed data."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import runtime as r
from run import ROOT, E0, MODEL, save, digest, independent, ssim
from codec.lvc import encode, decode

def table(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def corr(a,b):
    a=np.asarray(a).ravel(); b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

def entropy(x):
    _,n=np.unique(x,return_counts=True); p=n/n.sum()
    return float(-np.sum(p*np.log2(p)))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['statistics','sensitivity'],required=True); p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);args=p.parse_args()
    assert json.loads((ROOT/'results/baseline_gate.json').read_text())['status']=='PASSED'
    sources=[(s['sample'],s['latent'],'development',s) for s in r.samples]
    manifest=json.loads((E0/'results/validation_manifest.json').read_text())
    assert len(manifest)==16 and all(s['split']=='val' for s in manifest)
    sources += [(s['sample_id'],s['final_latent'],'validation',s) for s in manifest]
    if args.workers>1:
        assert (ROOT/'configs/frozen_channel_ranking.json').exists()
        sources=[s for i,s in enumerate([v for v in sources if v[2]=='validation']) if i%args.workers==args.worker]
    suffix='' if args.workers==1 else f'_worker_{args.worker}'
    channels=[]; temporal=[]; blocks=[]; sensitivities=[]
    if args.stage=='sensitivity': r.vae=r.WanVAE(vae_pth=str(MODEL),device='cuda:0')
    for name,path,split,source in sources:
        z=r.torch.load(path,weights_only=True,map_location='cpu').numpy()[None]
        assert z.shape==(1,16,5,60,104) and np.isfinite(z).all()
        if args.stage=='sensitivity' and split=='validation' and not (ROOT/'configs/frozen_channel_ranking.json').exists():
            scores=[np.mean([v['distortion_increment'] for v in sensitivities if v['channel']==c and v['bits']==4]) for c in range(16)]
            save(ROOT/'configs/frozen_channel_ranking.json',dict(source_split='development only',bits=4,scores=scores,ranking=np.argsort(scores)[::-1].tolist()))
        folder=ROOT/'results/E1_1'/name; folder.mkdir(parents=True,exist_ok=True)
        meta=dict(source,vae_sha256=digest(MODEL),normalization='native WanVAE scale; unchanged final latent',width=832,height=480,frames=17,fps=16)
        if args.stage=='statistics':
            for c in range(16):
                x=z[0,c].astype('float64'); scale=max(np.max(abs(x))/127,1e-12); q=np.rint(x/scale)
                channels.append(dict(sample=name,split=split,channel=c,mean=float(x.mean()),variance=float(x.var()),std=float(x.std()),l1=float(abs(x).sum()),energy=float((x*x).sum()),max_abs=float(abs(x).max()),p99=float(np.percentile(abs(x),99)),p995=float(np.percentile(abs(x),99.5)),p999=float(np.percentile(abs(x),99.9)),quantization_mse=float(np.mean((x-q*scale)**2)),symbol_entropy=entropy(q),temporal_corr=corr(x[:-1],x[1:]),horizontal_corr=corr(x[:,:,:-1],x[:,:,1:]),vertical_corr=corr(x[:,:-1],x[:,1:])))
            for t in range(5):
                for u in range(t+1,5):
                    a=z[:,:,t].astype('float64'); b=z[:,:,u].astype('float64'); delta=b-a
                    temporal.append(dict(sample=name,split=split,t0=t,t1=u,gap=u-t,mse=float(np.mean(delta**2)),mae=float(abs(delta).mean()),cosine=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))),pearson=corr(a,b),residual_entropy_bin_1_256=entropy(np.rint(delta*256))))
            for size in (1,2,4,8):
                for t in range(1,5):
                    for y in range(0,60,size):
                        for x in range(0,104,size):
                            a=z[0,:,t-1,y:y+size,x:x+size].astype('float64'); b=z[0,:,t,y:y+size,x:x+size].astype('float64'); d=b-a
                            blocks.append(dict(sample=name,split=split,block_size=size,t=t,y=y,x=x,height=a.shape[1],width=a.shape[2],residual_energy=float((d*d).mean()),max_residual_energy=float((d*d).max()),residual_sparsity_1e_3=float(np.mean(abs(d)<1e-3)),cosine=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))),variance=float(b.var()),residual_entropy_bin_1_256=entropy(np.rint(d*256))))
        else:
            rgb,_,_,_=r.pixels(z); raw=r.yuv(rgb)
            assert raw==(E0/'results'/name/'reference.yuv').read_bytes()
            reference=folder/'reference.yuv'; reference.write_bytes(raw)
            baseline=None
            for channel,bits in [(-1,8)]+[(c,b) for c in range(16) for b in (6,4,2)]:
                cb=[8]*16
                if channel>=0: cb[channel]=bits
                stream=folder/f'c{channel}_b{bits}.lvc'
                sizes,et=r.timing(lambda:encode(z,stream,dict(channel_bits=cb),dict(algorithm='zstd'),meta))
                zh=independent(stream); rec,vt,_,_=r.pixels(zh); rr=r.yuv(rec)
                reconstructed=folder/'reconstruction.yuv'; reconstructed.write_bytes(rr)
                metrics=r.quality(rgb,rec,raw,rr)
                mse=float(np.mean((rgb.astype('float64')-rec)**2))
                row=dict(sample=name,split=split,channel=channel,bits=bits,mse_rgb=mse,ssim=ssim(reference,reconstructed,17,832,480),encode_mean_s=et['mean'],vae_s=vt,**sizes,**metrics)
                if baseline is None: baseline=row
                row['distortion_increment']=mse-baseline['mse_rgb'];row['psnr_drop']=baseline['psnr_y']-row['psnr_y'];row['ssim_drop']=baseline['ssim']-row['ssim']
                savings=8*(baseline['total_bytes']-row['total_bytes']);row['distortion_per_saved_bit']=row['distortion_increment']/savings if savings>0 else None
                sensitivities.append(row); save(ROOT/f'results/E1_1/sensitivity{suffix}.json',sensitivities)
            reconstructed.unlink(); reference.unlink()
        print(args.stage,name,'DONE',flush=True)
    if args.stage=='statistics':
        table(ROOT/'results/E1_1/channel_statistics.csv',channels)
        table(ROOT/'results/E1_3/temporal_similarity.csv',temporal)
        table(ROOT/'results/E1_3/block_residual_statistics.csv',blocks)
    else:
        table(ROOT/f'results/E1_1/channel_sensitivity{suffix}.csv',sensitivities)

if __name__=='__main__': main()
