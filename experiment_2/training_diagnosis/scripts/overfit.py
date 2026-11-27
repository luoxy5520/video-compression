from common import *
from diagnostic_codec import Codec
import argparse
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['O1','O2'],default='O2');p.add_argument('--variant',default='R3');p.add_argument('--factor',type=float,default=1.);p.add_argument('--steps',type=int,default=1000);p.add_argument('--source-step',type=int,default=10000);a=p.parse_args()
assert json.loads((ROOT/'results/E2_1a/status.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'results/E2_1b'/a.variant/'gate.json').read_text())['pass']
torch.manual_seed(20261127);x=tensor(data()[0]).cuda();m=Codec(a.variant,.1,255).cuda();m.from_reconstruction(ROOT/'checkpoints'/a.variant/f'step{a.source_step}.pt');opt,aux,main=optimizers(m,1e-4);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[2000,5000],gamma=.5)
# New shared-model scale-head initialization from this legal training video only.
# Values are stored in checkpoint parameters; never untransmitted decoder statistics.
with torch.no_grad():
    init_std=m.ga(x).std(dim=(0,2,3)).clamp_min(.2)
    m.hs[2].weight.zero_();m.hs[2].bias.copy_(torch.log(torch.expm1((init_std-.11).clamp_min(.01))))
# One pretrained model and identical full-video batch; report both loss and gradient scales.
f=forward(m,x,'ste');rgrad=torch.autograd.grad(f['rate'],main,retain_graph=True,allow_unused=True);dgrad=torch.autograd.grad(f['dist'],main,retain_graph=True,allow_unused=True)
base_lambda=f['rate'].item()/max(f['dist'].item(),1e-12);lam=base_lambda*a.factor if a.mode=='O2' else 0
cal=dict(rate=f['rate'].item(),distortion=f['dist'].item(),rate_gradient_norm=sum(g.square().sum().item() for g in rgrad if g is not None)**.5,distortion_gradient_norm=sum(g.square().sum().item() for g in dgrad if g is not None)**.5,reference_lambda=base_lambda,selected_lambda=lam,factor=a.factor,quant_step=.1,symbol_limit=255,quantization='STE centered hyper, zero-mean main; hard values in forward',scale_head_initialization='checkpoint-shared training-video channel std; other hyper modules unchanged')
name=f'{a.mode}_{a.variant}_f{a.factor:g}';out=ROOT/'results/E2_1c'/name;out.mkdir(exist_ok=True);cks=ROOT/'checkpoints'/name;cks.mkdir(exist_ok=True);latest=cks/'latest.pt';curve=[];start=0;elapsed_previous=0
if latest.exists():
    c=torch.load(latest,weights_only=False,map_location='cpu');restore_state(m,c['state']);opt.load_state_dict(c['optimizer']);aux.load_state_dict(c['aux_optimizer']);sched.load_state_dict(c['scheduler']);torch.set_rng_state(c['rng']);torch.cuda.set_rng_state(c['cuda_rng']);start=c['step'];curve=c['curve'];lam=c['lambda_z'];elapsed_previous=c['elapsed_total']
    for o in (opt,aux):
        for state in o.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.cuda()
if start>=a.steps:print('already complete',name,start);exit()
save(out/'calibration.json',cal);t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
for step in range(start+1,a.steps+1):
    m.train();opt.zero_grad();aux.zero_grad();f=forward(m,x,'ste');loss=f['dist'] if a.mode=='O1' else f['rate']+lam*f['dist'];loss.backward();torch.nn.utils.clip_grad_norm_(main,1.);opt.step();aux.zero_grad();al=m.eb.loss();al.backward();aux.step();sched.step()
    assert torch.isfinite(loss)
    if step%50==0 or step==1:curve.append(dict(step=step,rate=f['rate'].item(),mse=f['dist'].item(),loss=loss.item(),weighted_distortion=(lam*f['dist'].item() if a.mode=='O2' else f['dist'].item()),aux=al.item(),lr=opt.param_groups[0]['lr'],max_abs_symbol=f['yh'].abs().max().item()))
    if step in (100,500,1000,2000,5000,10000) or step%1000==0 or step==a.steps:
        m.eval();m.update();c=dict(config=m.config,state=m.state_dict(),optimizer=opt.state_dict(),aux_optimizer=aux.state_dict(),scheduler=sched.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),curve=curve,lambda_z=lam,seed=20261127,sample=data()[0],sampling_state='single full video',elapsed_total=elapsed_previous+time.perf_counter()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated(),mode=a.mode)
        atomic_torch(latest,c)
        if step in (100,500,1000,2000,5000,10000) or step==a.steps:
            ck=cks/f'step{step}.pt';atomic_torch(ck,c);save(out/f'step{step}.json',dict(mode=a.mode,step=step,lambda_z=lam,checkpoint_sha256=sha(ck),elapsed_total=c['elapsed_total'],peak_gpu_bytes=c['peak_gpu_bytes']));print(name,step,curve[-1],flush=True)
        save(out/'status.json',dict(status='COMPLETE' if step==a.steps else 'RUNNING',steps=step,target=a.steps))
save(out/'curve.json',curve)
