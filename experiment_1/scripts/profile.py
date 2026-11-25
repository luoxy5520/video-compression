"""Isolated representative latency decomposition; run after quality workers finish."""
import json
import struct
import numpy as np
import runtime as r
from run import ROOT,E0,MODEL,save
from codec.lvc import encode,decode

def stats(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values,ddof=1)),min=min(values),max=max(values),samples=values)

def main():
    assert (ROOT/'results/ALL_MAIN_MEASUREMENTS_COMPLETE').exists()
    r.vae=r.WanVAE(vae_pth=str(MODEL),device='cuda:0');rows=[]
    for source in r.samples:
        name=source['sample'];z=r.torch.load(source['latent'],weights_only=True,map_location='cpu').numpy()[None]
        paths=[ROOT/'results/E1_0'/name/f'Q{bits}.lvc' for bits in (4,5,6,7,8)]
        mixed=json.loads((ROOT/'results/E1_2/mixed_precision_results.json').read_text())
        candidates=[v for v in mixed if v['sample']==name and v['policy']['method'] in ('M2','M2_three','M3')]
        best=max(candidates,key=lambda v:v['psnr_y'])
        paths.append(ROOT/'results/E1_2'/name/('bits_'+''.join(map(str,best['channel_bits']))+'.lvc'))
        for stream in paths:
            blob=stream.read_bytes();n=struct.unpack('<I',blob[4:8])[0];h=json.loads(blob[8:8+n])
            if h.get('mixed'):
                offset=8+n;size=struct.unpack('<I',blob[offset+4:offset+8])[0];child=json.loads(blob[offset+8:offset+8+size])
                cfg=dict(channel_bits=h['channel_bits']);comp=child['compression'];meta=child['video']
            else:cfg=h['quantization'];comp=h['compression'];meta=h['video']
            profiles=[];target=stream.with_suffix('.profile.lvc')
            for i in range(6):
                profile={};encode(z,target,cfg,comp,meta,profile)
                assert target.read_bytes()==blob, 'profiling changed serialized bytes'
                if i:profiles.append(profile)
            zh,dt=r.timing(lambda:decode(target))
            _,vt=r.timing(lambda:r.pixels(zh))
            timing={k:stats([v[k] for v in profiles]) for k in profiles[0]}
            rows.append(dict(sample=name,stream=str(stream.relative_to(ROOT)),timings=timing,decode_s=dt,vae_s=vt,post_dit_two_ends_mean_s=timing['total_s']['mean']+dt['mean']+vt['mean'],scope='CPU latent input; D2H not included; file I/O included, OS cache; native VAE GPU sync; other preexisting GPU services remain'))
            target.unlink()
    save(ROOT/'results/latency_breakdown.json',rows)

if __name__=='__main__':main()
