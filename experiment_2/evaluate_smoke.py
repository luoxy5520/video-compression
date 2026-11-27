"""Frozen native Wan VAE decoder evaluation, strictly development-only."""
import sys,json,time
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=ROOT]
sys.path.insert(0,str(ROOT.parent/'experiment_1/scripts'))
import runtime as r
r.vae=r.WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0')
for p in r.vae.model.parameters(): p.requires_grad_(False)
p=Path('/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106/sample_0000/final_latent.pt'); z=torch.load(p,weights_only=True).numpy()
if z.ndim==4:z=z[None]
r.pixels(z) # warmup only, no timing claim
ref,_,_,_=r.pixels(z); ya=r.yuv(ref); rows=[]
configs=[]
for bits in (4,5,6,8):
    path=ROOT/'results'/f'I0_int{bits}_0.lvc'
    configs.append((f'I0_INT{bits}',r.decode(path),path.stat().st_size))
smoke=json.loads((ROOT/'results/smoke.json').read_text())
for kind in ('I2','I3'):
    zh=torch.cat([torch.load(ROOT/'results'/f'{kind}_{i}_decoded.pt',weights_only=True) for i in range(5)],0).permute(1,0,2,3)[None].numpy()
    configs.append((kind,zh,next(x['actual_bytes'] for x in smoke if x['kind']==kind)))
for kind,zh,size in configs:
    rgb,elapsed,peak,_=r.pixels(zh); yb=r.yuv(rgb)
    row=dict(kind=kind,scope='single development video; I2/I3 trained on this same video',actual_bytes=size,latent_mse=float(((zh-z)**2).mean()),**r.quality(ref,rgb,ya,yb),vae_decode_seconds_diagnostic=elapsed,vae_peak_bytes=peak,latency_scope='one diagnostic execution, not formal timing',bpp=size*8/(832*480*17),kbps=size*8/1000/(17/16))
    rows.append(row);(ROOT/'results/smoke_pixel.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(row),flush=True)
