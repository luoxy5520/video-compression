from common import *
from diagnostic_codec import load
import argparse
p=argparse.ArgumentParser();p.add_argument('--factor',type=float,default=1.);p.add_argument('--videos',type=int,default=16);p.add_argument('--steps',type=int,default=5000);p.add_argument('--seed',type=int,default=20261127);a=p.parse_args()
assert json.loads((ROOT/'results/E2_1c/gate.json').read_text())['pass']
source=ROOT/'checkpoints'/f'O2_R3_identity_f{a.factor:g}'/'step1000.pt';m,original=load(source);m=m.cuda();lam=original['lambda_z'];opt,aux,main=optimizers(m,1e-4);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[2000,5000,8000],gamma=.5);torch.manual_seed(a.seed)
records=data()[:a.videos];assert len(records)==a.videos
xdata=[tensor(r).cuda() for r in records];name=f'D{a.videos}_f{a.factor:g}_seed{a.seed}';out=ROOT/'results/E2_1d'/name;out.mkdir(exist_ok=True);cks=ROOT/'checkpoints'/name;cks.mkdir(exist_ok=True);latest=cks/'latest.pt';curve=[];start=0;elapsed=0;order=list(range(len(xdata)))
if latest.exists():
    c=torch.load(latest,weights_only=False,map_location='cpu');m.load_state_dict(c['state']);opt.load_state_dict(c['optimizer']);aux.load_state_dict(c['aux_optimizer']);sched.load_state_dict(c['scheduler']);torch.set_rng_state(c['rng']);torch.cuda.set_rng_state(c['cuda_rng']);start=c['step'];curve=c['curve'];elapsed=c['elapsed_total'];order=c['sampling_state']['order'];assert c['train_paths']==[r['path'] for r in records]
    for o in (opt,aux):
        for state in o.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.cuda()
if start>=a.steps:print('already complete',name,start);exit()
t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
for step in range(start+1,a.steps+1):
    if (step-1)%len(xdata)==0:order=torch.randperm(len(xdata)).tolist()
    m.train();x=xdata[order[(step-1)%len(xdata)]];opt.zero_grad();aux.zero_grad();f=forward(m,x,'ste');loss=f['rate']+lam*f['dist'];loss.backward();torch.nn.utils.clip_grad_norm_(main,1.);opt.step();aux.zero_grad();al=m.eb.loss();al.backward();aux.step();sched.step();assert torch.isfinite(loss)
    if step%50==0 or step==1:curve.append(dict(step=step,rate=f['rate'].item(),mse=f['dist'].item(),loss=loss.item(),lr=opt.param_groups[0]['lr'],train_index=order[(step-1)%len(xdata)]))
    if step%1000==0 or step==a.steps:
        m.eval();m.update();c=dict(config=m.config,state=m.state_dict(),optimizer=opt.state_dict(),aux_optimizer=aux.state_dict(),scheduler=sched.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),curve=curve,lambda_z=lam,seed=a.seed,sample=records[0],train_paths=[r['path'] for r in records],sampling_state=dict(next_index=step%len(xdata),order=order,policy='seeded epoch permutation of full videos'),elapsed_total=elapsed+time.perf_counter()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated(),mode='D',source_sha256=sha(source),experiment_note='new data-scale training from O2 weights, new optimizer; not equivalent continuation of O2')
        atomic_torch(latest,c)
        if step in (1000,5000,10000,20000,40000) or step==a.steps:
            ck=cks/f'step{step}.pt';atomic_torch(ck,c);save(out/f'step{step}.json',dict(step=step,lambda_z=lam,checkpoint_sha256=sha(ck),elapsed_total=c['elapsed_total'],peak_gpu_bytes=c['peak_gpu_bytes'],train_paths=c['train_paths']));print(name,step,curve[-1],flush=True)
        save(out/'status.json',dict(status='COMPLETE' if step==a.steps else 'RUNNING',step=step,target=a.steps))
save(out/'curve.json',curve)
