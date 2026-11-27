"""Small V1/V2 fine tune; frozen full native causal VAE with verified checkpoints."""
from pixels import *
from diagnostic_codec import load
from vae_differentiable import decode_full
import argparse,gc
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['V1','V2'],default='V2');p.add_argument('--steps',type=int,default=100);p.add_argument('--factor',type=float,choices=[.1,1.,10.],default=1.);p.add_argument('--source-step',type=int);a=p.parse_args()
assert json.loads((ROOT/'results/E2_1e/vae_gradient_fp32_checkpoint.json').read_text())['status']=='PASS'
source_step=a.source_step or (10000 if a.factor==1 else 5000);source=ROOT/'checkpoints'/f'D16_f{a.factor:g}_seed20261127'/f'step{source_step}.pt';m,c=load(source);m=m.cuda().train();v=vae();v.model.requires_grad_(False);records=[r for r in data() if r['path'] in c['train_paths']];assert len(records)==16
xdata=[tensor(r).cuda() for r in records];references={};torch.manual_seed(20261127);opt,aux,main=optimizers(m,1e-5);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[500,1000],gamma=.5);name=a.mode+('_D16' if a.factor==1 else '_low_D16' if a.factor==.1 else '_high_D16');out=ROOT/'results/E2_1e'/name;out.mkdir(exist_ok=True);cks=ROOT/'checkpoints'/name;cks.mkdir(exist_ok=True);latest=cks/'latest.pt';curve=[];start=0;elapsed=0;lambda_p=None;lambda_z=c['lambda_z'] if a.mode=='V2' else 0
assert not ({id(p) for p in v.model.parameters()} & {id(p) for p in main})
def fingerprint():
    h=hashlib.sha256()
    for n,p in v.model.named_parameters():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
vae_hash=fingerprint();file_hash=sha(MODEL)
if latest.exists():
    state=torch.load(latest,weights_only=False,map_location='cpu');restore_state(m,state['state']);opt.load_state_dict(state['optimizer']);aux.load_state_dict(state['aux_optimizer']);scheduler.load_state_dict(state['scheduler']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng']);start=state['step'];curve=state['curve'];elapsed=state['elapsed_total'];lambda_p=state['lambda_p']
    assert state['train_paths']==[r['path'] for r in records] and state['source_sha256']==sha(source)
if start>=a.steps:print('already complete',name,start);exit()
t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
for step in range(start+1,a.steps+1):
    idx=(step-1)%len(records);x=xdata[idx]
    if idx not in references:
        with torch.no_grad():references[idx]=decode(v,x).detach().cpu()
    reference=references[idx].cuda();opt.zero_grad();aux.zero_grad();f=forward(m,x,'ste');zh=f['rec'];before=time.perf_counter();output=decode_full(v,zh.permute(1,0,2,3)[None],True)[0];dp=(output-reference).square().mean()
    if lambda_p is None:
        gp=torch.autograd.grad(dp,zh,retain_graph=True)[0];gz=2*(zh-x)/x.numel();lambda_p=c['lambda_z']*gz.norm().item()/max(gp.norm().item(),1e-12);save(out/'calibration.json',dict(lambda_p=lambda_p,lambda_z=lambda_z,baseline_lambda_z=c['lambda_z'],pixel_mse=dp.item(),latent_mse=f['dist'].item(),rate=f['rate'].item(),pixel_gradient_norm=gp.norm().item(),latent_gradient_norm=gz.norm().item(),criterion='match weighted initial reconstruction gradient norms; same source for V1 and V2'));del gp,gz
    loss=f['rate']+lambda_z*f['dist']+lambda_p*dp;loss.backward();torch.nn.utils.clip_grad_norm_(main,1.);opt.step();aux.zero_grad();al=m.eb.loss();al.backward();aux.step();scheduler.step();torch.cuda.synchronize();assert torch.isfinite(loss)
    if step%10==0 or step==1:
        curve.append(dict(step=step,rate=f['rate'].item(),latent_mse=f['dist'].item(),pixel_mse=dp.item(),loss=loss.item(),lambda_p=lambda_p,lambda_z=lambda_z,lr=opt.param_groups[0]['lr'],sample_index=idx,vae_forward_backward_step_seconds=time.perf_counter()-before));print(name,curve[-1],flush=True)
    # Release activation graphs before checkpoint persistence / next reference inference.
    del output,dp,loss,f,zh,reference
    if step%10==0 or step==a.steps:
        m.eval();m.update();state=dict(config=m.config,state=m.state_dict(),optimizer=opt.state_dict(),aux_optimizer=aux.state_dict(),scheduler=scheduler.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampling_state=dict(next_video=step%len(records),policy='fixed full-video cycle'),curve=curve,lambda_z=lambda_z,lambda_p=lambda_p,seed=20261127,sample=records[0],train_paths=[r['path'] for r in records],elapsed_total=elapsed+time.perf_counter()-t0,peak_gpu_bytes=torch.cuda.max_memory_allocated(),mode=a.mode,source_sha256=sha(source),vae_parameter_sha256=vae_hash)
        atomic_torch(latest,state)
        if step in (100,500,1000,2000,5000) or step==a.steps:
            ck=cks/f'step{step}.pt';atomic_torch(ck,state);save(out/f'step{step}.json',dict(step=step,checkpoint_sha256=sha(ck),elapsed_total=state['elapsed_total'],peak_gpu_bytes=state['peak_gpu_bytes']))
        m.train();save(out/'status.json',dict(status='TRAINED_AWAITING_REAL_STREAM_EVALUATION' if step==a.steps else 'RUNNING',step=step,target=a.steps))
assert fingerprint()==vae_hash and sha(MODEL)==file_hash
save(out/'vae_integrity.json',dict(unchanged=True,tensor_sha256=vae_hash,file_sha256=file_hash,vae_gradients_none=all(p.grad is None for p in v.model.parameters())))
save(out/'curve.json',curve)
