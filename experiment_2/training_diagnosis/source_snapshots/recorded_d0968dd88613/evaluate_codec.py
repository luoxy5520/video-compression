from pixels import *
from diagnostic_codec import load
import argparse
p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--stage',default='E2_1c');a=p.parse_args();v=vae();refcache={}
for name in a.names:
    ck=ROOT/'checkpoints'/name/f'step{a.steps}.pt';out=ROOT/'results'/a.stage/name;out.mkdir(exist_ok=True)
    if (out/f'evaluation_step{a.steps}.json').exists():print('already evaluated',name,a.steps);continue
    m,c=load(ck);record=c['sample'];x=tensor(record)
    key=record['path']
    if key not in refcache:
        with torch.no_grad():ref=rgb(decode(v,x));refcache[key]=(ref,yuv(ref))
    ref,ya=refcache[key];deploy=out/f'deploy_step{a.steps}.pt';atomic_torch(deploy,dict(config=c['config'],state=m.state_dict()));stats=[];decoded=[];estimated=[]
    with torch.no_grad():
        for i,frame in enumerate(x):
            stream=out/f'step{a.steps}_t{i}.lvc2';stats.append(write_stream(stream,m,frame[None],sha(deploy)));rec=read_stream(stream,m,sha(deploy));f=forward(m,frame[None],'hard');assert torch.equal(rec,f['rec']);decoded.append(rec);estimated.append(dict(slice=i,estimated_main_bits=f['main'].item(),estimated_hyper_bits=f['hyper'].item(),actual_main_bytes=stats[-1]['main_bytes'],actual_hyper_bytes=stats[-1]['hyper_bytes'],metadata_bytes=stats[-1]['metadata_bytes']))
            if i==0:
                external=out/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/diagnostic_codec.py'),str(deploy),str(stream),str(external)],check=True);assert torch.equal(torch.load(external,weights_only=True),rec)
        xh=torch.cat(decoded);img=rgb(decode(v,xh));q=quality(ref,img,ya)
    result=dict(name=name,step=a.steps,mode=c['mode'],scope='single-video overfit; not generalization',sample=record['path'],lambda_z=c['lambda_z'],latent_mse=(xh-x).square().mean().item(),actual_bytes=sum(s['bytes'] for s in stats),main_bytes=sum(s['main_bytes'] for s in stats),hyper_bytes=sum(s['hyper_bytes'] for s in stats),metadata_bytes=sum(s['metadata_bytes'] for s in stats),streams=stats,per_slice_rate=estimated,**q,checkpoint_sha256=sha(ck),deployment_sha256=sha(deploy),deployment_bytes=deploy.stat().st_size,independent_process_decode=True,quant_step=.1,symbol_limit=255)
    save(out/f'evaluation_step{a.steps}.json',result);print(json.dumps(result),flush=True)
