"""Small grouped train/development I2/I3 RD sweep, no test data or VAE calls."""
import json,time
from pathlib import Path
import torch
from codec import Intra,write_stream,read_stream,sha
R=Path(__file__).resolve().parent
torch.set_num_threads(4)
records=json.loads((R/'dataset_manifest.json').read_text())['records']
train=[r for r in records if r['role']=='train_candidate' and r['exists']][:16]
groups={r['group'] for r in train}
dev=[r for r in records if r['role']=='historical_development' and r['exists'] and r['group'] not in groups][:2]
assert len(train)==16 and len(dev)==2
assert not ({r['group'] for r in dev}&groups)
def load(r):
    z=torch.load(r['path'],weights_only=True,map_location='cpu');z=z[0] if z.ndim==5 else z
    return z.permute(1,0,2,3).contiguous()
data=[load(r).cuda() for r in train]; evaluation=[load(r) for r in dev]
results=[]
for kind in ('I2','I3'):
    for lam in (16,128,1024,8192):
        torch.manual_seed(20261127);m=Intra(kind).cuda().train()
        main=[p for n,p in m.named_parameters() if not n.endswith('.quantiles')];aux=[p for n,p in m.named_parameters() if n.endswith('.quantiles')]
        opt=torch.optim.Adam(main,lr=.001);auxopt=torch.optim.Adam(aux,lr=.001)
        curve=[];t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for step in range(1000):
            x=data[step%len(data)];opt.zero_grad();auxopt.zero_grad();rec,rate=m(x);mse=(rec-x).square().mean();loss=rate+lam*mse
            loss.backward();torch.nn.utils.clip_grad_norm_(main,1.);opt.step()
            auxopt.zero_grad();auxloss=m.eb.loss();auxloss.backward();auxopt.step()
            if step%100==0 or step==999:curve.append(dict(step=step,loss=loss.item(),mse=mse.item(),estimated_bits_per_element=rate.item(),aux=auxloss.item()))
        torch.cuda.synchronize();seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()
        m.cpu().eval();m.update();ck=R/'checkpoints'/f'{kind}_lambda{lam}.pt';torch.save(dict(kind=kind,channels=32,state=m.state_dict()),ck)
        for idx,x in enumerate(evaluation):
            stats=[];decoded=[]
            for i,frame in enumerate(x):
                path=R/'results'/f'{kind}_lambda{lam}_dev{idx}_t{i}.lvc2';stats.append(write_stream(path,m,frame[None],sha(ck)));out=read_stream(path,m,sha(ck))
                with torch.no_grad(): expected,_=m(frame[None])
                assert torch.equal(expected,out);decoded.append(out)
            row=dict(kind=kind,lambda_z=lam,steps=1000,train_videos=len(train),train_groups=[r['group'] for r in train],train_paths=[r['path'] for r in train],development_path=dev[idx]['path'],development_group=dev[idx]['group'],scope='historical development; not independent holdout',actual_bytes=sum(v['bytes'] for v in stats),mse=(torch.cat(decoded)-x).square().mean().item(),streams=stats,checkpoint_sha256=sha(ck),checkpoint_bytes=ck.stat().st_size,train_seconds=seconds,peak_gpu_bytes=peak,curve=curve,forward_decode_exact=True,independent_process_decode='covered by smoke for same architecture, not separately run for this checkpoint')
            results.append(row);(R/'results/intra_sweep.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ('curve','streams','train_groups','train_paths')}),flush=True)
