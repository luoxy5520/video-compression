"""Development-only actual-stream frozen VAE quality for short intra sweep."""
import sys,json,importlib.util,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=ROOT]
sys.path.insert(0,str(ROOT.parent/'experiment_1/scripts'))
import runtime as r
spec=importlib.util.spec_from_file_location('e2codec',ROOT/'codec.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
r.vae=r.WanVAE(vae_pth='/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0')
for p in r.vae.model.parameters():p.requires_grad_(False)
rows=json.loads((ROOT/'results/intra_sweep.json').read_text()); refs={};out=[]
for row in rows:
    path=row['development_path'];idx=list(dict.fromkeys(x['development_path'] for x in rows)).index(path)
    if path not in refs:
        z=torch.load(path,weights_only=True).numpy();z=z[None] if z.ndim==4 else z
        rgb,*_=r.pixels(z);refs[path]=(rgb,r.yuv(rgb))
        for bits in range(2,9):
            stream=ROOT/'results'/f'I0_int{bits}_{idx}.lvc';zh=r.decode(stream);img,*_=r.pixels(zh)
            out.append(dict(kind=f'I0_INT{bits}',development_path=path,actual_bytes=stream.stat().st_size,**r.quality(rgb,img,refs[path][1],r.yuv(img))))
    kind=row['kind'];lam=row['lambda_z'];ck=ROOT/'checkpoints'/f'{kind}_lambda{lam}.pt';m=c.load_model(ck);frames=[]
    for t in range(5):
        stream=ROOT/'results'/f'{kind}_lambda{lam}_dev{idx}_t{t}.lvc2';frames.append(c.read_stream(stream,m,c.sha(ck)))
        if idx==0 and t==0:
            dest=stream.with_suffix('.independent.pt');subprocess.run([sys.executable,str(ROOT/'codec.py'),str(ck),str(stream),str(dest)],check=True);assert torch.equal(torch.load(dest,weights_only=True),frames[-1])
    zh=torch.cat(frames).permute(1,0,2,3)[None].numpy();img,*_=r.pixels(zh);rgb,ya=refs[path]
    result=dict(kind=kind,lambda_z=lam,development_path=path,actual_bytes=row['actual_bytes'],scope='historical development, 16 separate training groups; not independent holdout',**r.quality(rgb,img,ya,r.yuv(img)),representative_independent_process_decode=idx==0)
    out.append(result);(ROOT/'results/intra_pixel.json').write_text(json.dumps(out,indent=2));print(json.dumps(result),flush=True)
