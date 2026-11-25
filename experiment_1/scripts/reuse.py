"""Controlled latent perturbations; skipped-symbol counts are theoretical, not rate."""
import json
import argparse
import numpy as np
import runtime as r
from run import ROOT,E0,MODEL,save,ssim
from codec.lvc import decode
from analyze import table

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    suffix='' if args.workers==1 else f'_worker_{args.worker}'
    assert json.loads((ROOT/'experiment_status.json').read_text())['E1_0_measurements']=='COMPLETE'
    manifest=json.loads((E0/'results/validation_manifest.json').read_text())
    sources=[(s['sample'],s['latent'],'development') for s in r.samples]+[(s['sample_id'],s['final_latent'],'validation') for s in manifest]
    sources=[s for i,s in enumerate(sources) if i%args.workers==args.worker]
    # Fixed before any experiments; validation uses the preregistered block size 4.
    save(ROOT/'configs/frozen_reuse.json',dict(development_block_sizes=[1,2,4,8],validation_block_sizes=[4],ratios=[.05,.1,.2,.3,.5],strategies=['low','random','high'],random_seed=20261125,single_step_t=1,baseline_bits=8,low_bits=4,entropy_bin_width=1/256))
    r.vae=r.WanVAE(vae_pth=str(MODEL),device='cuda:0');rows=[]
    for name,path,split in sources:
        z=r.torch.load(path,weights_only=True,map_location='cpu').numpy()[None]
        baseline=decode(ROOT/'results/E1_0'/name/'Q8.lvc')
        q4=decode(ROOT/'results/E1_0'/name/'Q4.lvc')
        rgb,_,_,_=r.pixels(z);raw=r.yuv(rgb)
        assert raw==(E0/'results'/name/'reference.yuv').read_bytes()
        folder=ROOT/'results/E1_3'/name;folder.mkdir(parents=True,exist_ok=True)
        ref=folder/'reference.yuv';ref.write_bytes(raw)
        for size in ([1,2,4,8] if split=='development' else [4]):
            coords=[(y,x) for y in range(0,60,size) for x in range(0,104,size)]
            energies=[]
            for t in range(1,5):
                energies.append(np.array([np.mean((z[0,:,t,y:y+size,x:x+size]-z[0,:,t-1,y:y+size,x:x+size])**2) for y,x in coords]))
            configurations=[dict(operation='int4',quartile=q) for q in range(4)]
            configurations += [dict(operation='reuse',ratio=ratio,strategy=strategy,mode=mode) for ratio in (.05,.1,.2,.3,.5) for strategy in ('low','random','high') for mode in ('single','closed_loop')]
            for cfg in configurations:
                rec=baseline.copy(); symbols=0;count=0;selected_energy=[];mask=np.zeros((4,len(coords)),dtype='uint8')
                for t in range(1,5):
                    if cfg['operation']=='reuse' and cfg['mode']=='single' and t!=1:continue
                    order=np.argsort(energies[t-1],kind='stable')
                    if cfg['operation']=='int4':chosen=np.array_split(order,4)[cfg['quartile']]
                    else:
                        if cfg['strategy']=='high':order=order[::-1]
                        if cfg['strategy']=='random':order=np.random.default_rng(20261125+t).permutation(len(coords))
                        chosen=order[:int(round(cfg['ratio']*len(coords)))]
                    for i in chosen:
                        y,x=coords[i];region=(0,slice(None),t,slice(y,y+size),slice(x,x+size))
                        value=q4[region] if cfg['operation']=='int4' else rec[0,:,t-1,y:y+size,x:x+size]
                        rec[region]=value;symbols+=value.size;count+=1;selected_energy.append(float(energies[t-1][i]));mask[t-1,i]=1
                video,vt,_,_=r.pixels(rec);rr=r.yuv(video);reconstructed=folder/'reconstruction.yuv';reconstructed.write_bytes(rr)
                row=dict(sample=name,split=split,block_size=size,configuration=cfg,selected_blocks=count,selected_symbols=symbols,mask_uncompressed_bytes=len(np.packbits(mask)),mean_selected_residual_energy=float(np.mean(selected_energy)),latent_mse=float(np.mean((z-rec)**2)),latent_slice_mse=np.mean((z-rec)**2,axis=(0,1,3,4)).tolist(),actual_bitstream_bytes='NOT RUN: perturbation only; no skip codec',ssim=ssim(ref,reconstructed,17,832,480),vae_s=vt,**r.quality(rgb,video,raw,rr))
                rows.append(row);save(ROOT/f'results/E1_3/skip_probe_results{suffix}.json',rows)
            print(name,size,'DONE',flush=True)
        ref.unlink();reconstructed.unlink()
    table(ROOT/f'results/E1_3/skip_probe_results{suffix}.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows])

if __name__=='__main__':main()
