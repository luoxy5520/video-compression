"""Gated fixed-symbol temporal experiment; frozen base, resumable entropy-only fit."""
from common import *
from diagnostic_codec import load
from temporal_entropy_model import Temporal,load_temporal
import argparse,subprocess
p=argparse.ArgumentParser();p.add_argument('--base-name',default='D42_f1_seed20261127');p.add_argument('--base-step',type=int,default=5000);p.add_argument('--steps',type=int,default=2000);p.add_argument('--run-name');p.add_argument('--modes',nargs='+',choices=['TP0','TP1','TP2','TP2Z'],default=['TP0','TP1','TP2']);a=p.parse_args();assert a.modes[0]=='TP0'
assert json.loads((ROOT/'results/E2_1d/gate.json').read_text())['pass']
source=ROOT/'checkpoints'/a.base_name/f'step{a.base_step}.pt';base,c=load(source);records=json.loads((E2/'dataset_manifest.json').read_text())['records'];train=[r for r in records if r['path'] in c['train_paths']];groups={r['group'] for r in train};dev=[r for r in records if r['role']=='historical_development' and r['group'] not in groups][:2];assert len(dev)==2
out=ROOT/'results'/('extensions/'+a.run_name if a.run_name else 'E2_2');out.mkdir(parents=True,exist_ok=True);ckdir=ROOT/'checkpoints'/('extensions/'+a.run_name if a.run_name else 'temporal');ckdir.mkdir(parents=True,exist_ok=True);baseline={};results=[]
if all((out/f'{mode}_status.json').exists() and json.loads((out/f'{mode}_status.json').read_text()).get('status')=='COMPLETE' for mode in a.modes):
    for mode in (m for m in a.modes if m!='TP0'):
        saved=torch.load(ckdir/f'{mode}.pt',weights_only=False,map_location='cpu');assert saved['source_sha256']==sha(source) and saved['step']>=a.steps,'new config needs a new run directory'
    print('all temporal configurations already complete; preserved');exit()
for mode in a.modes:
    torch.manual_seed(20261127);m=Temporal(c['config'],mode);restore_state(m.base,c['state']);curve=[];seconds=0;ck=ckdir/f'{mode}.pt'
    if mode!='TP0':
        if ck.exists():
            m,saved=load_temporal(ck);assert saved['source_sha256']==sha(source) and saved['step']>=a.steps
            curve=saved['curve'];seconds=saved['train_seconds']
        else:
            cache=[]
            for r in train:
                fixed=m.fixed_sequence(tensor(r));prev=None;seq=[]
                for q,hs,zs,zshape in fixed:
                    if prev is not None:seq.append((q.cuda(),hs.cuda(),prev.cuda()))
                    prev=m.previous(q)
                cache.append(seq)
            m=m.cuda();m.base.eval();opt=torch.optim.Adam(m.predictor.parameters(),lr=.001);start=0;resume=ckdir/f'{mode}_resume.pt'
            if resume.exists():
                saved=torch.load(resume,weights_only=False,map_location='cpu');assert saved['source_sha256']==sha(source)
                restore_state(m,saved['state']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng']);start=saved['step'];curve=saved['curve'];seconds=saved['train_seconds']
            t0=time.perf_counter()
            for step in range(start+1,a.steps+1):
                seq=cache[(step-1)%len(cache)];opt.zero_grad();loss=0
                for q,hs,prev in seq:
                    mean,scale=m.params_for(hs,prev,True);prob=m.base.gc._likelihood(q,scale,mean);loss=loss-prob.clamp_min(1e-9).log2().mean()/len(seq)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.predictor.parameters(),1.);opt.step();assert torch.isfinite(loss)
                if step%100==0:curve.append(dict(step=step,cross_entropy_bits_per_predicted_symbol=loss.item()))
                if step%500==0 or step==a.steps:
                    atomic_torch(resume,dict(config=c['config'],mode=mode,state=m.state_dict(),optimizer=opt.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampling_state=dict(next_video=step%len(cache),policy='fixed full-video cycle'),curve=curve,train_seconds=seconds+time.perf_counter()-t0,source_sha256=sha(source)))
                    save(out/f'{mode}_status.json',dict(status='TRAINED_AWAITING_REAL_STREAM_EVALUATION' if step==a.steps else 'RUNNING',step=step,target=a.steps))
            seconds+=time.perf_counter()-t0;m.cpu().eval();del cache;torch.cuda.empty_cache()
            atomic_torch(ck,dict(config=c['config'],mode=mode,state=m.state_dict(),curve=curve,train_seconds=seconds,source_sha256=sha(source),step=a.steps,train_paths=c['train_paths']))
    else:
        if ck.exists():
            m,saved=load_temporal(ck);assert saved['source_sha256']==sha(source)
        else:atomic_torch(ck,dict(config=c['config'],mode=mode,state=m.state_dict(),curve=[],train_seconds=0,source_sha256=sha(source),step=0,train_paths=c['train_paths']))
    m.cpu().eval()
    for idx,r in enumerate(dev):
        x=tensor(r);path=out/f'{mode}_dev{idx}.bin';stats=m.compress_video(x,path,sha(ck));q,z=m.decompress_video(path,sha(ck));fixed=m.fixed_sequence(x);expectedq=torch.cat([f[0] for f in fixed]);expectedz=torch.cat([m.base.gs(f[0]) for f in fixed]);assert torch.equal(q,expectedq) and torch.equal(z,expectedz)
        if mode=='TP0':baseline[idx]=(q.clone(),z.clone(),stats['actual_bytes'])
        else:assert torch.equal(q,baseline[idx][0]) and torch.equal(z,baseline[idx][1])
        external=out/'independent.pt';subprocess.run([sys.executable,str(ROOT/'scripts/temporal_entropy_model.py'),str(ck),str(path),str(external)],check=True);d=torch.load(external,weights_only=True);assert torch.equal(d['q'],q) and torch.equal(d['z'],z)
        row=dict(mode=mode,video=r['path'],actual_bytes=stats['actual_bytes'],saving_fraction=1-stats['actual_bytes']/baseline[idx][2],metadata_bytes=stats['metadata_bytes'],slices=stats['slices'],stream_sha256=stats['sha256'],deployment_sha256=sha(ck),deployment_bytes=ck.stat().st_size,train_seconds=seconds,canonical_symbols_exact=True,reconstruction_exact=True,independent_decode=True,scope='historical development; exact reconstructed latent equality establishes identical deterministic video reconstruction',base_name=a.base_name,base_step=a.base_step)
        results.append(row);save(out/'temporal_results.json',results);print(json.dumps(row),flush=True)
    save(out/f'{mode}_status.json',dict(status='COMPLETE',development_videos=len(dev),all_symbols_equal=True,all_latents_equal=True))
csvsave(out/'temporal_prior_ablation.csv',[{k:x[k] for k in ['mode','video','actual_bytes','saving_fraction','metadata_bytes','train_seconds','canonical_symbols_exact','reconstruction_exact','independent_decode']} for x in results]);save(out/'fixed_symbol_consistency.json',dict(all_symbols_equal=True,all_latents_equal=True,integer_mean_cdf_translation=True,mean_rule='k=round(predicted_mean); canonical q unchanged; internal rANS encodes q-k then adds k exactly'))
