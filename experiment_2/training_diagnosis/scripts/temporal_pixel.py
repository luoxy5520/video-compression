"""P3 small VAE-aware joint fine tune from measured P2; no original weights trained."""
from pixels import *
from temporal_entropy_model import load_temporal
from vae_differentiable import decode_full
import argparse
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100);p.add_argument('--run-name');p.add_argument('--initial-state',type=Path);a=p.parse_args()
# Only after the standalone VAE-aware1000-step experiments have actual decode results.
for mode in ('V1','V2'):assert (ROOT/'results/E2_1e'/f'{mode}_D16/development_step1000.json').exists()
source=ROOT/'checkpoints/temporal_joint/P2.pt';m,c=load_temporal(source);m=m.cuda();m.base.requires_grad_(True);v=vae();v.model.requires_grad_(False);records=[r for r in data() if r['path'] in c['train_paths']];xdata=[tensor(r).cuda() for r in records];refs={};torch.manual_seed(20261127)
main=[p for n,p in m.named_parameters() if not n.endswith('.quantiles')];opt=torch.optim.Adam(main,lr=1e-5);aux=torch.optim.Adam([m.base.eb.quantiles],lr=.001);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[100],gamma=.5);lam=c['lambda_z'];lambda_p=None
assert not ({id(p) for p in v.model.parameters()} & {id(p) for p in main})
cks=ROOT/'checkpoints'/('extensions/'+a.run_name if a.run_name else 'temporal_pixel');cks.mkdir(parents=True,exist_ok=True);out=ROOT/'results'/('extensions/'+a.run_name if a.run_name else 'E2_2/P3');out.mkdir(parents=True,exist_ok=True);latest=cks/'latest.pt';curve=[];start=0;elapsed=0
resume=latest if latest.exists() else a.initial_state
if resume is not None:
    state=torch.load(resume,weights_only=False,map_location='cpu');assert state['source_sha256']==sha(source) and state['train_paths']==c['train_paths'];restore_state(m,state['state']);opt.load_state_dict(state['optimizer']);aux.load_state_dict(state['aux_optimizer']);sched.load_state_dict(state['scheduler']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng']);start=state['step'];curve=state['curve'];elapsed=state['train_seconds'];lambda_p=state['lambda_p']
if start>=a.steps:print('already complete');exit()
def fingerprint():
    h=hashlib.sha256()
    for name,p in v.model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
vae_sha=fingerprint();file_sha=sha(MODEL);assert file_sha=='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981';t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
for step in range(start+1,a.steps+1):
    m.train();x=xdata[(step-1)%len(xdata)];idx=(step-1)%len(xdata)
    if idx not in refs:
        with torch.no_grad():refs[idx]=decode(v,x).detach().cpu()
    reference=refs[idx].cuda();opt.zero_grad();aux.zero_grad();prev=None;bits=0;recs=[]
    for frame in x:
        f=forward(m.base,frame[None],'ste');hs=m.base.hs(f['zh'])[:,:,:f['y'].shape[-2],:f['y'].shape[-1]]+.11;mean,scale=m.params_for(hs,prev,True);prob=m.base.gc._likelihood(f['yh'],scale,mean);bits=bits-prob.clamp_min(1e-9).log2().sum()+f['hyper'];recs.append(f['rec']);prev=f['yh']
    zh=torch.cat(recs);rate=bits/x.numel();dz=(zh-x).square().mean();output=decode_full(v,zh.permute(1,0,2,3)[None],True)[0];dp=(output-reference).square().mean()
    if lambda_p is None:
        gp=torch.autograd.grad(dp,zh,retain_graph=True)[0];gz=2*(zh-x)/x.numel();lambda_p=lam*gz.norm().item()/max(gp.norm().item(),1e-12);save(out/'calibration.json',dict(lambda_p=lambda_p,lambda_z=lam,pixel_gradient_norm=gp.norm().item(),latent_gradient_norm=gz.norm().item()));del gp,gz
    loss=rate+lam*dz+lambda_p*dp;loss.backward();torch.nn.utils.clip_grad_norm_(main,1.);opt.step();aux.zero_grad();m.base.eb.loss().backward();aux.step();sched.step();torch.cuda.synchronize();assert torch.isfinite(loss)
    if step%10==0 or step==1:curve.append(dict(step=step,rate=rate.item(),latent_mse=dz.item(),pixel_mse=dp.item(),loss=loss.item()));print(curve[-1],flush=True)
    del output,reference,loss,dp,dz,rate,bits,zh,recs,f,prev
    if step%10==0 or step==a.steps:
        m.eval();m.base.update();state=dict(config=c['config'],mode='TP2',state=m.state_dict(),optimizer=opt.state_dict(),aux_optimizer=aux.state_dict(),scheduler=sched.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampling_state=dict(next_video=step%len(records)),curve=curve,train_seconds=elapsed+time.perf_counter()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated(),train_paths=c['train_paths'],lambda_z=lam,lambda_p=lambda_p,source_sha256=sha(source),vae_tensor_sha256=vae_sha);atomic_torch(latest,state);save(out/'status.json',dict(status='TRAINED_AWAITING_REAL_STREAM_EVALUATION' if step==a.steps else 'RUNNING',step=step,target=a.steps))
        if step in (500,1000,2000,3000,5000) or step==a.steps:atomic_torch(cks/f'step{step}.pt',state)
        if step==a.steps:atomic_torch(cks/'P3.pt',{k:state[k] for k in ['config','mode','state','step','curve','train_seconds','peak_gpu_bytes','train_paths','lambda_z','lambda_p','source_sha256','vae_tensor_sha256']})
assert fingerprint()==vae_sha and sha(MODEL)==file_sha
save(out/'vae_integrity.json',dict(unchanged=True,tensor_sha256=vae_sha,file_sha256=file_sha,vae_gradients_none=all(p.grad is None for p in v.model.parameters())))
