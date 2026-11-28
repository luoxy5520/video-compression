from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['R1','R2','R3'],default='R1');p.add_argument('--steps',type=int,default=5000);a=p.parse_args()
assert json.loads((ROOT/'results/E2_1a/status.json').read_text())['status']=='PASS'
v=a.variant;out=ROOT/'results/E2_1b'/v;out.mkdir(exist_ok=True);ckdir=ROOT/'checkpoints'/v;ckdir.mkdir(exist_ok=True)
record=data()[0];x=tensor(record).cuda();torch.manual_seed(20261127)
m=Intra('I3',64 if v=='R2' else 32).cuda()
if v=='R3':
    m.ga=torch.nn.Sequential(torch.nn.Conv2d(16,32,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(32,32,3,padding=1)).cuda()
    m.gs=torch.nn.Sequential(torch.nn.Conv2d(32,32,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(32,16,3,padding=1)).cuda()
params=list(m.ga.parameters())+list(m.gs.parameters());opt=torch.optim.Adam(params,lr=.001);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[2000,5000,8000],gamma=.5)
start=0;curve=[];latest=ckdir/'latest.pt'
if latest.exists():
    c=torch.load(latest,weights_only=False,map_location='cpu');m.load_state_dict(c['state']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);torch.set_rng_state(c['rng']);torch.cuda.set_rng_state(c['cuda_rng']);start=c['step'];curve=c['curve']
    for state in opt.state.values():
        for k,value in state.items():
            if isinstance(value,torch.Tensor):state[k]=value.cuda()
if start>=a.steps:print('already complete',v,start);exit()
t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
for step in range(start+1,a.steps+1):
    opt.zero_grad();y=m.ga(x);rec=m.gs(y)[:,:,:x.shape[-2],:x.shape[-1]];loss=(rec-x).square().mean();loss.backward();opt.step();sched.step()
    if step%50==0 or step==1:curve.append(dict(step=step,mse=loss.item(),lr=opt.param_groups[0]['lr']))
    if step in (100,500,1000,2000,5000,10000) or step%1000==0 or step==a.steps:
        with torch.no_grad():y=m.ga(x);rec=m.gs(y)[:,:,:x.shape[-2],:x.shape[-1]];err=rec-x
        state=dict(variant=v,channels=m.channels,state=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),curve=curve,seed=20261127,sample=record,sampling_state='single fixed full video',seconds_this_invocation=time.perf_counter()-t)
        atomic_torch(latest,state)
        if step in (100,500,1000,2000,5000,10000) or step==a.steps:
            ck=ckdir/f'step{step}.pt';atomic_torch(ck,state);atomic_torch(out/f'reconstruction_step{step}.pt',rec.detach().cpu())
            result=dict(variant=v,step=step,mse=err.square().mean().item(),mae=err.abs().mean().item(),relative_l2=(err.norm()/x.norm()).item(),channel_mse=err.square().mean((0,2,3)).tolist(),time_slice_mse=err.square().mean((1,2,3)).tolist(),input_shape=list(x.shape),coding_shape=list(y.shape),bottleneck_element_ratio=y.numel()/x.numel(),rate='NOT APPLICABLE: no quantization or entropy coding',checkpoint_sha256=sha(ck),peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_this_invocation=time.perf_counter()-t)
            save(out/f'step{step}.json',result);print(json.dumps(result),flush=True)
        save(out/'status.json',dict(status='COMPLETE' if step==a.steps else'RUNNING',step=step,target_steps=a.steps,variant=v))
save(out/'curve.json',curve)
