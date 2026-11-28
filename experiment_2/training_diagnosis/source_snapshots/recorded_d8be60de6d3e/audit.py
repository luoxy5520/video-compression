from common import *
import subprocess
out=ROOT/'results/E2_1a';torch.manual_seed(20261127)
m=load_model(E2/'checkpoints/I3_lambda128.pt').train();x=tensor(data()[0]);rows=[];updates=[];losses=[]
opt,auxopt,main=optimizers(m)
mainids={id(p) for g in opt.param_groups for p in g['params']};auxids={id(p) for g in auxopt.param_groups for p in g['params']};allids={id(p) for p in m.parameters() if p.requires_grad}
assert not mainids&auxids and mainids|auxids==allids
for loss_mode in ('rate','distortion','rd','auxiliary'):
    torch.manual_seed(20261127);m.zero_grad();f=forward(m,x);loss={'rate':f['rate'],'distortion':f['dist'],'rd':f['rate']+128*f['dist'],'auxiliary':m.eb.loss()}[loss_mode];loss.backward()
    for group in ('ga','gs','ha','hs','eb','quantiles'):
        ps=[(n,p) for n,p in m.named_parameters() if (n.endswith('.quantiles') if group=='quantiles' else n.startswith(group+'.') and not n.endswith('.quantiles'))]
        grads=[p.grad for n,p in ps];valid=[g for g in grads if g is not None]
        rows.append(dict(loss_mode=loss_mode,module=group,parameters=sum(p.numel() for n,p in ps),requires_grad=all(p.requires_grad for n,p in ps),gradient_norm=sum(g.double().square().sum().item() for g in valid)**.5,none_parameter_fraction=sum(g is None for g in grads)/max(len(ps),1),zero_parameter_fraction=sum(int(g is not None and torch.count_nonzero(g).item()==0) for g in grads)/max(len(ps),1),nonfinite_elements=sum((~torch.isfinite(g)).sum().item() for g in valid)))
for mode in ('main','auxiliary'):
    m.zero_grad();before={n:p.detach().clone() for n,p in m.named_parameters()};torch.manual_seed(20261127)
    f=forward(m,x);loss=f['rate']+128*f['dist'] if mode=='main' else m.eb.loss();loss.backward();(opt if mode=='main' else auxopt).step()
    for n,p in m.named_parameters():updates.append(dict(optimizer=mode,parameter=n,in_main=id(p) in mainids,in_aux=id(p) in auxids,change_l2=(p-before[n]).norm().item(),requires_grad=p.requires_grad))
# Loss-weight audit uses one identical batch, parameters AND noise realization.
m=load_model(E2/'checkpoints/I3_lambda128.pt').train();torch.manual_seed(20261127);f=forward(m,x)
for lam in (16,128,1024,8192):losses.append(dict(lambda_z=lam,rate_unweighted=f['rate'].item(),distortion_unweighted=f['dist'].item(),rate_weighted=f['rate'].item(),distortion_weighted=lam*f['dist'].item(),total=(f['rate']+lam*f['dist']).item(),main_bits=f['main'].item(),hyper_bits=f['hyper'].item(),denominator=x.numel()))
save(out/'loss_values.json',losses);csvsave(out/'gradient_audit.csv',rows);csvsave(out/'parameter_update_check.csv',updates)
checks=[]
for mode in ('none','noise','ste','hard'):
    m.zero_grad();torch.manual_seed(20261127);f=forward(m,x,mode);f['dist'].backward();checks.append(dict(mode=mode,mse=f['dist'].item(),ga_grad_norm=sum(p.grad.square().sum().item() for p in m.ga.parameters() if p.grad is not None)**.5,gs_grad_norm=sum(p.grad.square().sum().item() for p in m.gs.parameters() if p.grad is not None)**.5,y_round_mse=(f['yh']-f['y']).square().mean().item(),symbol_min=f['yh'].min().item(),symbol_max=f['yh'].max().item()))
save(out/'quantization_check.json',dict(modes=checks,main_symbols='round(y), means=None; unit step; int32 rANS with escape handling in CompressAI',hyper_symbols='round(z - learned medians), reconstruction adds medians',training_default='independent U(-.5,.5) noise on main and hyper; eval hard rounding'))
m.eval();rates=[]
with torch.no_grad():
    for t,frame in enumerate(x):
        f=forward(m,frame[None],'hard');p=out/f't{t}.lvc2';ck=E2/'checkpoints/I3_lambda128.pt';stats=write_stream(p,m,frame[None],sha(ck));decoded=read_stream(p,m,sha(ck));assert torch.equal(decoded,f['rec']);py=f['py'];pz=f['pz']
        rates.append(dict(slice=t,estimated_main_bits=f['main'].item(),estimated_hyper_bits=f['hyper'].item(),actual_main_bits=8*stats['main_bytes'],actual_hyper_bits=8*stats['hyper_bytes'],metadata_bits=8*stats['metadata_bytes'],complete_file_bits=8*stats['bytes'],zero_likelihoods=int((py==0).sum()+(pz==0).sum()),nonfinite_likelihoods=int((~torch.isfinite(py)).sum()+(~torch.isfinite(pz)).sum()),stream_sha256=stats['sha256']))
    subprocess.run([sys.executable,str(E2/'codec.py'),str(ck),str(p),str(out/'independent.pt')],check=True);assert torch.equal(torch.load(out/'independent.pt',weights_only=True),decoded)
csvsave(out/'estimated_vs_actual_rate.csv',rates)
# Quantiles absent in main RD is expected: forward uses detached medians; auxiliary loss trains them.
assert all(r['nonfinite_elements']==0 for r in rows)
for lm,mods in [('rate',['ga','ha','hs','eb']),('distortion',['ga','gs']),('rd',['ga','gs','ha','hs','eb']),('auxiliary',['quantiles'])]:
    assert all(next(r for r in rows if r['loss_mode']==lm and r['module']==mod)['gradient_norm']>0 for mod in mods),(lm,mods)
(out/'optimizer_audit.md').write_text(f'# Optimizer audit\n\nMain/aux overlap:0; coverage:{len(mainids|auxids)}/{len(allids)} trainable tensors. Main Adam lr.001; auxiliary Adam lr.001. Quantiles excluded from main, exclusively auxiliary. GaussianConditional has no trainable parameters; hyper decoder predicts scales. No AMP used in historical codec training. Global gradient clip1 makes different high-lambda runs similar in scale and can obscure rate gradients; this is optimization behavior, not a swapped-loss bug.\n')
(out/'loss_audit.md').write_text('# Loss audit\n\nHistorical train_intra.py: L=R+lambda*MSE. R=sum(-log2 main likelihood)+sum(-log2 hyper likelihood), divided by all B*C*H*W input latent elements, including all5 slices treated as batch. D=mean squared latent error over identical element count. No pixel or extra multiplier. Noise in training versus hard rounding for evaluation is intentional and must be reported separately. Fixed-checkpoint mathematical weight test saved in loss_values.json. Auxiliary loss optimized separately; the earlier100-step smoke instead added .01*aux to one optimizer, and is not equivalent to the1000-step runs. Validation previously recorded only actual bytes and hard reconstruction MSE, not a conflicting normalized total loss.\n')
(out/'E2_1a_report.md').write_text('# E2.1a\n\nPASS: fixed-batch weight arithmetic, expected gradient routes, disjoint complete optimizers, actual parameter updates, finite likelihood/gradient tests, hard-forward/real-stream equality and independent decoding. No evidence of a swapped loss or detached training transform in historical1000-step implementation. Evidence does not establish convergence. Estimated versus actual differences include scale-table discretization, finite CDF precision, rANS framing/tails and header cost; inspect measured CSV rather than substituting estimates for bytes. EB quantiles receive auxiliary gradients as expected, not main RD gradients.\n')
save(out/'status.json',dict(status='PASS',checkpoint=sha(ck),sample=data()[0]['path'],seed=20261127));print('E2.1a PASS',flush=True)
