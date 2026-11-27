from pixels import *
from diagnostic_codec import load
import argparse
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=5000);a=p.parse_args();v=vae();ck=ROOT/'checkpoints'/a.name/f'step{a.steps}.pt';m,c=load(ck);out=ROOT/'results/E2_1d'/a.name;out.mkdir(exist_ok=True);records=json.loads((E2/'dataset_manifest.json').read_text())['records'];train=set(c['train_paths']);groups={r['group'] for r in records if r['path'] in train};dev=[r for r in records if r['role']=='historical_development' and r['group'] not in groups and r['exists']][:2];assert len(dev)==2
rows=[];deploy=out/f'deploy_step{a.steps}.pt';atomic_torch(deploy,dict(config=c['config'],state=m.state_dict()))
for idx,record in enumerate(dev):
    x=tensor(record);stats=[];decoded=[]
    with torch.no_grad():
        ref=rgb(decode(v,x));ya=yuv(ref)
        for i,frame in enumerate(x):
            stream=out/f'step{a.steps}_dev{idx}_t{i}.lvc2';stats.append(write_stream(stream,m,frame[None],sha(deploy)));rec=read_stream(stream,m,sha(deploy));f=forward(m,frame[None],'hard');assert torch.equal(rec,f['rec']);decoded.append(rec)
            if i==0:
                external=out/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/diagnostic_codec.py'),str(deploy),str(stream),str(external)],check=True);assert torch.equal(torch.load(external,weights_only=True),rec)
        xh=torch.cat(decoded);rec=rgb(decode(v,xh));q=quality(ref,rec,ya)
    rows.append(dict(name=a.name,step=a.steps,train_video_count=len(train),seed=c['seed'],video=record['path'],scope='historical development only; no independent generalization claim',actual_bytes=sum(s['bytes'] for s in stats),latent_mse=(xh-x).square().mean().item(),**q,streams=stats,checkpoint_sha256=sha(ck),deploy_sha256=sha(deploy),independent_decode=True))
    save(out/f'development_step{a.steps}.json',rows);print(json.dumps(rows[-1]),flush=True)
