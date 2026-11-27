"""Capacity witness, not learned compression: GELU(x)-GELU(-x)=x.
Same R3 convolution shapes; changes initialization only. Tests unseen random tensor too.
"""
from common import *
from diagnostic_codec import Codec
m=Codec('R3',.1,255)
# Each two-conv/GELU transform represents identity using +/- channel pairs.
for mod in (m.ga[0],m.gs[1]):
    for p in mod.parameters():p.data.zero_()
for i in range(16):
    m.ga[0][0].weight.data[i,i,1,1]=1;m.ga[0][0].weight.data[i+16,i,1,1]=-1
    m.ga[0][2].weight.data[i,i,1,1]=1;m.ga[0][2].weight.data[i,i+16,1,1]=-1
    m.gs[1][0].weight.data[i,i,1,1]=1;m.gs[1][0].weight.data[i+16,i,1,1]=-1
    m.gs[1][2].weight.data[i,i,1,1]=1;m.gs[1][2].weight.data[i,i+16,1,1]=-1
x=tensor(data()[0]);random=torch.randn_like(x)
with torch.no_grad():
    # Bypass quant step wrappers: pure continuous ga/gs.
    y=m.ga[0](x);rec=m.gs[1](y);random_error=(m.gs[1](m.ga[0](random))-random).abs().max().item()
assert (rec-x).abs().max()<2e-6 and random_error<2e-6
name='R3_identity';out=ROOT/'results/E2_1b'/name;out.mkdir(exist_ok=True);ckdir=ROOT/'checkpoints'/name;ckdir.mkdir(exist_ok=True)
# Reconstruction checkpoint compatible with existing from_reconstruction loader.
state={**{'ga.'+k:v for k,v in m.ga[0].state_dict().items()},**{'gs.'+k:v for k,v in m.gs[1].state_dict().items()}}
ck=ckdir/'step0.pt';atomic_torch(ck,dict(state=state,variant='R3',step=0,seed=20261127,initialization='analytic identity; not trained'))
atomic_torch(out/'reconstruction_step0.pt',rec)
err=rec-x
save(out/'step0.json',dict(variant=name,step=0,mse=err.square().mean().item(),mae=err.abs().mean().item(),relative_l2=(err.norm()/x.norm()).item(),input_shape=list(x.shape),coding_shape=list(y.shape),bottleneck_element_ratio=y.numel()/x.numel(),random_tensor_max_abs_error=random_error,checkpoint_sha256=sha(ck),rate='NOT APPLICABLE: analytic continuous reconstruction, no coding',interpretation='capacity witness and new initialization, not a trained codec or evidence of compression performance'))
save(out/'gate.json',{'pass':True,'basis':'analytical unchanged R3-shape identity, <2e-6 max error on video and random tensor; use native VAE evaluation to verify pixel equality'})
print('identity capacity witness',err.square().mean().item(),random_error)
