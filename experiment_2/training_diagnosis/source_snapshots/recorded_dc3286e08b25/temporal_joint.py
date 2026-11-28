"""P2 small joint latent RD: reconstructed references, 2 slices then all5.
Allows symbols/reconstructions to change; this is NOT a fixed-symbol ablation.
"""
from common import *
from temporal_entropy_model import load_temporal
import argparse
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1500);p.add_argument('--run-name');p.add_argument('--initial-state',type=Path);p.add_argument('--source-checkpoint',type=Path);p.add_argument('--base-checkpoint',type=Path);p.add_argument('--lambda-factor',type=float,choices=[.1,1.,10.],default=1.);a=p.parse_args()
measurements_path=ROOT/'results/extensions'/a.source_checkpoint.parent.name/'temporal_results.json' if a.source_checkpoint else ROOT/'results/E2_2/temporal_results.json'
measurements=json.loads(measurements_path.read_text());gains=[r['saving_fraction'] for r in measurements if r['mode']=='TP2'];assert sum(gains)/len(gains)>=.05
source=a.source_checkpoint or ROOT/'checkpoints/temporal/TP2.pt';m,tc=load_temporal(source);basec=torch.load(a.base_checkpoint or ROOT/'checkpoints/D16_f1_seed20261127/step10000.pt',weights_only=False,map_location='cpu');records=[r for r in data() if r['path'] in basec['train_paths']];assert len(records)==len(basec['train_paths']) and set(tc.get('train_paths',basec['train_paths']))==set(basec['train_paths']);xdata=[tensor(r).cuda() for r in records];m=m.cuda();m.base.requires_grad_(True);torch.manual_seed(20261127)
baseparams=[p for n,p in m.base.named_parameters() if not n.endswith('.quantiles')];priorparams=list(m.predictor.parameters());main=baseparams+priorparams;opt=torch.optim.Adam([dict(params=baseparams,lr=1e-5),dict(params=priorparams,lr=1e-4)]);aux=torch.optim.Adam([m.base.eb.quantiles],lr=.001);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[1000],gamma=.5)
cks=ROOT/'checkpoints'/('extensions/'+a.run_name if a.run_name else 'temporal_joint');cks.mkdir(parents=True,exist_ok=True);out=ROOT/'results'/('extensions/'+a.run_name if a.run_name else 'E2_2/P2');out.mkdir(parents=True,exist_ok=True);latest=cks/'latest.pt';curve=[];start=0;elapsed=0;lam=basec['lambda_z']*a.lambda_factor
resume=latest if latest.exists() else a.initial_state
if resume is not None:
    c=torch.load(resume,weights_only=False,map_location='cpu');assert c['source_sha256']==sha(source) and c['train_paths']==basec['train_paths'] and c['lambda_z']==lam;restore_state(m,c['state']);opt.load_state_dict(c['optimizer']);aux.load_state_dict(c['aux_optimizer']);sched.load_state_dict(c['scheduler']);torch.set_rng_state(c['rng']);torch.cuda.set_rng_state(c['cuda_rng']);start=c['step'];curve=c['curve'];elapsed=c['train_seconds']
if start>=a.steps:print('already complete');exit()
t0=time.perf_counter()
for step in range(start+1,a.steps+1):
    m.train();x=xdata[(step-1)%len(xdata)];x=x[:2] if step<=500 else x;opt.zero_grad();aux.zero_grad();prev=None;bits=0;dist=0
    for frame in x:
        f=forward(m.base,frame[None],'ste');hs=m.base.hs(f['zh'])[:,:,:f['y'].shape[-2],:f['y'].shape[-1]]+.11;mean,scale=m.params_for(hs,prev,True);prob=m.base.gc._likelihood(f['yh'],scale,mean);bits=bits-prob.clamp_min(1e-9).log2().sum()+f['hyper'];dist=dist+(f['rec']-frame[None]).square().sum();prev=f['yh'] # reconstructed coding latent, never true previous Z
    rate=bits/x.numel();mse=dist/x.numel();loss=rate+lam*mse;loss.backward();torch.nn.utils.clip_grad_norm_(main,1.);opt.step();aux.zero_grad();m.base.eb.loss().backward();aux.step();sched.step();assert torch.isfinite(loss)
    if step%50==0:curve.append(dict(step=step,sequence_length=len(x),rate=rate.item(),latent_mse=mse.item(),loss=loss.item()));print(curve[-1],flush=True)
    if step%500==0 or step==a.steps:
        m.eval();m.base.update();state=dict(config=tc['config'],mode='TP2',state=m.state_dict(),optimizer=opt.state_dict(),aux_optimizer=aux.state_dict(),scheduler=sched.state_dict(),step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampling_state=dict(next_video=step%len(xdata)),curve=curve,train_seconds=elapsed+time.perf_counter()-t0,train_paths=basec['train_paths'],lambda_z=lam,source_sha256=sha(source));atomic_torch(latest,state);atomic_torch(cks/f'step{step}.pt',state);save(out/'status.json',dict(status='TRAINED_AWAITING_REAL_STREAM_EVALUATION' if step==a.steps else 'RUNNING',step=step,target=a.steps))
        if step==a.steps:atomic_torch(cks/'P2.pt',{k:state[k] for k in ['config','mode','state','step','curve','train_seconds','train_paths','lambda_z','source_sha256']})
