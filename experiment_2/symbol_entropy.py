"""I1 learned channel Gaussian, identical E1 channel-max quantized symbols.
Original experiment glue; CompressAI GaussianConditional API, BSD-3-Clause-Clear.
"""
import json,struct,hashlib,sys,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from compressai.entropy_models import GaussianConditional
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
# Import E1 CPU baseline without loading its VAE runtime.
import importlib.util
spec=importlib.util.spec_from_file_location('e1_lvc',ROOT.parent/'experiment_1/codec/lvc.py'); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(x,bits):
    s=x.abs().amax(dim=(1,2,3),keepdim=True).clamp_min(1e-12)/((1<<(bits-1))-1)
    # Match E1 numpy scale floor exactly.
    s=s.clamp_min(1e-12)
    return torch.round(x/s).clamp(-((1<<(bits-1))-1),((1<<(bits-1))-1)),s

def gaussian():
    g=GaussianConditional(None); g.update_scale_table(torch.exp(torch.linspace(-2.2,5.5,256)),force=True); return g

def decode(stream,checkpoint):
    blob=Path(stream).read_bytes()
    if len(blob)<8 or blob[:4]!=b'LSY1': raise ValueError('magic')
    n=struct.unpack('<I',blob[4:8])[0]
    if n>65536 or n>len(blob)-8: raise ValueError('header')
    h=json.loads(blob[8:8+n]); payload=blob[8+n:]
    if h['model_sha256']!=sha(checkpoint) or hashlib.sha256(payload).hexdigest()!=h['sha256'] or sum(h['lengths'])!=len(payload): raise ValueError('checksum/model/length')
    c=torch.load(checkpoint,weights_only=True); g=gaussian(); shape=h['shape']; scales=c['scales'].view(1,16,1,1).expand(1,16,shape[2],shape[3]); indexes=g.build_indexes(scales)
    off=0; frames=[]
    for size in h['lengths']:
        frames.append(g.decompress([payload[off:off+size]],indexes)[0]); off+=size
    q=torch.stack(frames,dim=1); return q,q*torch.tensor(h['quant_scale']).reshape(16,1,1,1)

def run():
    torch.manual_seed(20261127)
    manifest=json.loads((ROOT/'dataset_manifest.json').read_text())['records']
    train=[r for r in manifest if r['role']=='train_candidate' and r['exists']]
    # Existing legal train groups only. Historical development evaluated separately.
    groups={r['group'] for r in train}; dev=[r for r in manifest if r['role']=='historical_development' and r['exists'] and r['group'] not in groups]
    train=train[:16]; dev=dev[:2]
    assert train and dev
    def load(r):
        x=torch.load(r['path'],weights_only=True,map_location='cpu'); return x[0] if x.ndim==5 else x
    data=[load(r) for r in train]; rows=[]
    for bits in (4,5,6,8):
        qs=[quant(x,bits)[0] for x in data]
        init=torch.stack([q.square().mean((1,2,3)) for q in qs]).mean(0).sqrt().clamp_min(.11)
        logscale=nn.Parameter(init.log()); opt=torch.optim.Adam([logscale],lr=.03); g=gaussian(); started=time.perf_counter(); curve=[]
        for step in range(200):
            opt.zero_grad(); q=qs[step%len(qs)].permute(1,0,2,3); scales=logscale.exp().view(1,16,1,1).expand_as(q)
            # likelihoods at fixed integers; only Gaussian scale is learned.
            _,p=g(q,scales,training=False); loss=-p.clamp_min(1e-9).log2().mean(); loss.backward(); opt.step()
            if step%20==0: curve.append(dict(step=step,bits_per_symbol=loss.item()))
        train_seconds=time.perf_counter()-started
        ck=ROOT/'checkpoints'/f'I1_int{bits}.pt'; torch.save(dict(scales=logscale.detach().exp(),bits=bits,train_paths=[r['path'] for r in train]),ck)
        for idx,r in enumerate(dev):
            x=load(r); q,s=quant(x,bits); ref=q*s
            path=ROOT/'results'/f'I0_int{bits}_{idx}.lvc'; base.encode(x[None].numpy(),path,dict(bits=bits,granularity='channel',percentile=100),dict(algorithm='zstd'),dict(width=832,height=480,frames=17,fps=16))
            baseline=torch.from_numpy(base.decode(path))[0]; assert torch.equal(baseline,ref)
            strings=[]; scales=logscale.detach().exp().view(1,16,1,1).expand(1,16,*x.shape[-2:]); indexes=g.build_indexes(scales)
            for frame in q.permute(1,0,2,3): strings.append(g.compress(frame[None],indexes)[0])
            p=b''.join(strings); h=dict(version=1,shape=list(x.shape),bits=bits,quant_scale=s.flatten().tolist(),lengths=list(map(len,strings)),model_sha256=sha(ck),sha256=hashlib.sha256(p).hexdigest()); header=json.dumps(h,separators=(',',':'),sort_keys=True).encode()
            stream=ROOT/'results'/f'I1_int{bits}_{idx}.lsy'; stream.write_bytes(b'LSY1'+struct.pack('<I',len(header))+header+p)
            qh,xh=decode(stream,ck); assert torch.equal(qh,q) and torch.equal(xh,baseline)
            if idx==0:
                target=stream.with_suffix('.decoded.pt'); subprocess.run([sys.executable,__file__,'decode',str(stream),str(ck),str(target)],check=True); assert torch.equal(torch.load(target,weights_only=True),baseline)
            row=dict(bits=bits,sample=r['path'],scope='historical development only',I0_bytes=path.stat().st_size,I1_bytes=stream.stat().st_size,saving_fraction=1-stream.stat().st_size/path.stat().st_size,I1_payload_bytes=len(p),I1_metadata_bytes=8+len(header),quant_symbols_exact=True,reconstruction_exact=True,independent_process_decode=idx==0,latent_mse=(xh-x).square().mean().item(),stream_sha256=sha(stream),checkpoint_sha256=sha(ck),checkpoint_bytes=ck.stat().st_size,train_video_count=len(train),train_groups=[r['group'] for r in train],train_seconds=train_seconds,curve=curve)
            rows.append(row); (ROOT/'results/symbol_entropy.json').write_text(json.dumps(rows,indent=2)); print(json.dumps({k:v for k,v in row.items() if k not in ('curve','train_groups')}),flush=True)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='decode': torch.save(decode(sys.argv[2],sys.argv[3])[1],sys.argv[4])
    else: run()
