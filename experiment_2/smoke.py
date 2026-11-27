"""Development-only overfit and independent-process rANS gate, no VAE calls."""
import json,time,sys,subprocess,hashlib
from pathlib import Path
import torch
from codec import Intra,write_stream,read_stream,sha
ROOT=Path(__file__).resolve().parent
torch.manual_seed(20261127); torch.set_num_threads(4)
p=Path('/mnt/data_hdd/luoxiangyu/dit_pairs/exp_expand32_20261106/sample_0000/final_latent.pt')
z=torch.load(p,map_location='cpu',weights_only=True)
if isinstance(z,dict):
    print('latent keys',z.keys(),flush=True)
    z=z.get('latent',z.get('final_latent'))
if z.ndim==5: z=z[0]
x=z.permute(1,0,2,3).contiguous()
assert x.shape[1]==16 and torch.isfinite(x).all()
rows=[]
for kind in ('I2','I3'):
    m=Intra(kind).cuda().train(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    curve=[]; t=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    # Smoke is a single-video overfit, not an independent validation result.
    for step in range(100):
        opt.zero_grad(); rec,rate=m(x.cuda()); mse=(rec-x.cuda()).square().mean(); loss=rate+16*mse+0.01*m.eb.loss(); loss.backward(); opt.step()
        if step%10==0 or step==99: curve.append(dict(step=step,loss=loss.item(),mse=mse.item(),estimated_bits_per_element=rate.item()))
    torch.cuda.synchronize(); elapsed=time.perf_counter()-t; peak=torch.cuda.max_memory_allocated()
    m.cpu().eval(); m.update(); ck=ROOT/'checkpoints'/f'{kind}_smoke.pt'
    torch.save(dict(kind=kind,channels=32,state=m.state_dict()),ck)
    parts=[]; stats=[]
    with torch.no_grad():
        for i,frame in enumerate(x):
            stream=ROOT/'results'/f'{kind}_{i}.lvc2'; out=ROOT/'results'/f'{kind}_{i}_decoded.pt'
            stats.append(write_stream(stream,m,frame[None],sha(ck)))
            expected,_=m(frame[None]); decoded=read_stream(stream,m,sha(ck))
            assert torch.equal(expected,decoded), (expected-decoded).abs().max().item()
            subprocess.run([sys.executable,str(ROOT/'codec.py'),str(ck),str(stream),str(out)],check=True)
            external=torch.load(out,weights_only=True); assert torch.equal(external,decoded)
            parts.append(external)
        reconstructed=torch.cat(parts)
    # Corruption must be rejected before entropy decode.
    bad=ROOT/'results'/f'{kind}_corrupt.lvc2'; blob=bytearray(stream.read_bytes()); blob[-1]^=1; bad.write_bytes(blob)
    try: read_stream(bad,m,sha(ck)); raise AssertionError('corruption accepted')
    except ValueError: pass
    bad.unlink()
    row=dict(kind=kind,scope='single development video overfit, NOT generalization',steps=100,lambda_z=16,lr=1e-3,seed=20261127,train_seconds=elapsed,peak_gpu_bytes=peak,parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=sha(ck),checkpoint_bytes=ck.stat().st_size,input_sha256=sha(p),shape=list(z.shape),actual_bytes=sum(r['bytes'] for r in stats),mse=(reconstructed-x).square().mean().item(),streams=stats,curve=curve,independent_process_decode=True,forward_decode_exact=True,corruption_rejected=True)
    rows.append(row); (ROOT/'results/smoke.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(row),flush=True)
