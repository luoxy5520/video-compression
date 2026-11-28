"""Nominal convolution MAC accounting; entropy/index/elementwise costs excluded."""
from common import *
from temporal_entropy_model import load_temporal
from torch import nn
out=ROOT/'results/performance';ck=ROOT/'checkpoints/temporal/TP2.pt';t=time.perf_counter();m,c=load_temporal(ck);load_seconds=time.perf_counter()-t;record=data()[0];x=tensor(record);current=[];handles=[]
def hook(name):
    def record(mod,inputs,output):
        inp=inputs[0];kh,kw=mod.kernel_size
        if isinstance(mod,nn.ConvTranspose2d):mac=inp.numel()*(mod.out_channels//mod.groups)*kh*kw
        else:mac=output.numel()*(mod.in_channels//mod.groups)*kh*kw
        current.append(dict(layer=name,input_shape=list(inp.shape),output_shape=list(output.shape),nominal_MACs=int(mac),output_tensor_bytes=output.numel()*output.element_size()))
    return record
for name,mod in m.named_modules():
    if isinstance(mod,(nn.Conv2d,nn.ConvTranspose2d)):handles.append(mod.register_forward_hook(hook(name)))
with torch.no_grad():
    p=out/'complexity.bin';m.compress_video(x,p,sha(ck));encode=list(current);current.clear();m.decompress_video(p,sha(ck));decode=list(current)
for h in handles:h.remove()
save(out/'complexity.json',dict(parameters=sum(p.numel() for p in m.parameters()),trainable_prior_parameters=sum(p.numel() for p in m.predictor.parameters()),deployment_bytes=ck.stat().st_size,CPU_model_load_seconds=load_seconds,load_scope='one filesystem-cache-warm CPU construction/state load; VAE and GPU upload excluded',input_shape=list(x.shape),encoder_nominal_conv_MACs=sum(r['nominal_MACs'] for r in encode),decoder_nominal_conv_MACs=sum(r['nominal_MACs'] for r in decode),maximum_single_conv_output_bytes=max(r['output_tensor_bytes'] for r in encode+decode),encoder_layers=encode,decoder_layers=decode,method='custom PyTorch forward hooks on actual encode/decode; Conv2d output-elements*Cin/groups*kernel; ConvTranspose2d input-elements*Cout/groups*kernel, nominal padded-kernel upper bound; excludes elementwise, entropy CDF/index and VAE; one MAC is not reported as one FLOP',script_sha256=sha(__file__)))
