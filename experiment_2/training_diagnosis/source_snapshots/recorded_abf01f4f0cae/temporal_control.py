"""Same-capacity zero-history head: distinguish capacity from temporal information."""
from common import *
from diagnostic_codec import load
from temporal_entropy_model import Temporal,load_temporal
import subprocess
out=ROOT/'results/E2_2/TP2Z';out.mkdir(exist_ok=True);ck=ROOT/'checkpoints/temporal/TP2Z.pt';resume=ROOT/'checkpoints/temporal/TP2Z_resume.pt';source=ROOT/'checkpoints/D16_f1_seed20261127/step10000.pt';base,c=load(source);records=json.loads((E2/'dataset_manifest.json').read_text())['records'];train=[r for r in records if r['path'] in c['train_paths']];groups={r['group'] for r in train};dev=[r for r in records if r['role']=='historical_development' and r['group'] not in groups][:2]
existing=json.loads((out/'rows.json').read_text()) if (out/'rows.json').exists() else []
if len(existing)==len(dev):print('control already complete; preserved');exit()
torch.manual_seed(20261127);m=Temporal(c['config'],'TP2Z');restore_state(m.base,c['state']);curve=[];seconds=0
if ck.exists():m,saved=load_temporal(ck);curve=saved['curve'];seconds=saved['train_seconds']
else:
    cache=[]
    for r in train:
        fixed=m.fixed_sequence(tensor(r));seq=[]
        for q,hs,zs,zshape in fixed[1:]:seq.append((q.cuda(),hs.cuda(),torch.zeros_like(q).cuda()))
        cache.append(seq)
    m=m.cuda();opt=torch.optim.Adam(m.predictor.parameters(),lr=.001);start=0
    if resume.exists():
        saved=torch.load(resume,weights_only=False,map_location='cpu');restore_state(m,saved['state']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng']);start=saved['step'];curve=saved['curve'];seconds=saved['train_seconds']
    t0=time.perf_counter()
    for step in range(start+1,2001):
        opt.zero_grad();seq=cache[(step-1)%len(cache)];loss=0
        for q,hs,zero in seq:
            mean,scale=m.params_for(hs,zero,True);loss=loss-m.base.gc._likelihood(q,scale,mean).clamp_min(1e-9).log2().mean()/len(seq)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.predictor.parameters(),1.);opt.step();assert torch.isfinite(loss)
        if step%100==0:curve.append(dict(step=step,cross_entropy_bits_per_predicted_symbol=loss.item()))
        if step%500==0:atomic_torch(resume,dict(config=c['config'],mode='TP2Z',state=m.state_dict(),optimizer=opt.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),curve=curve,train_seconds=seconds+time.perf_counter()-t0,source_sha256=sha(source)))
    seconds+=time.perf_counter()-t0;m.cpu().eval();del cache;torch.cuda.empty_cache();atomic_torch(ck,dict(config=c['config'],mode='TP2Z',state=m.state_dict(),step=2000,curve=curve,train_seconds=seconds,source_sha256=sha(source)))
rows=list(existing)
for idx,r in enumerate(dev):
    if any(row['video']==r['path'] for row in rows):continue
    x=tensor(r);path=out/f'dev{idx}.bin';stats=m.compress_video(x,path,sha(ck));q,z=m.decompress_video(path,sha(ck));fixed=m.fixed_sequence(x);assert torch.equal(q,torch.cat([f[0] for f in fixed]));assert torch.equal(z,torch.cat([m.base.gs(f[0]) for f in fixed]));target=out/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(ck),str(path),str(target)],check=True);assert torch.equal(torch.load(target,weights_only=True)['z'],z)
    rows.append(dict(method='TP2Z_zero_history',video=r['path'],actual_bytes=stats['actual_bytes'],streams=stats,canonical_symbols_exact=True,reconstruction_exact=True,independent_decode=True,train_steps=2000,train_seconds=seconds,scope='same head and training budget as TP2, history replaced by zeros; frozen original hyperprior features retained'));save(out/'rows.json',rows);print(json.dumps(rows[-1]),flush=True)

save(out/'status.json',dict(status='COMPLETE' if len(rows)==len(dev) else 'PARTIAL',videos=len(rows),target=len(dev)))
