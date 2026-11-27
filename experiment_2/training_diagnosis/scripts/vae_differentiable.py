"""Original differentiable glue for frozen native Wan decoder, unchanged weights.
Matches native causal slicing. Local immutable cache snapshots cross checkpoints;
no detached history, no local crops and no concatenation of independently decoded videos.
Wan source: commit9737cba9c1c3c4d04b33fcad41c111989865d315, Apache-2.0.
"""
import torch
from torch.utils.checkpoint import checkpoint

def decode_full(vae,z,checkpoint_chunks=True):
    model=vae.model;model.clear_cache()
    z=z/vae.scale[1].view(1,16,1,1,1)+vae.scale[0].view(1,16,1,1,1)
    x=model.conv2(z);cache=tuple([None]*model._conv_num);outputs=[]
    def chunk(current,previous):
        local=list(previous);output=model.decoder(current,feat_cache=local,feat_idx=[0]);return output,tuple(local)
    for t in range(x.shape[2]):
        if checkpoint_chunks:
            output,cache=checkpoint(chunk,x[:,:,t:t+1],cache,use_reentrant=False)
        else:output,cache=chunk(x[:,:,t:t+1],cache)
        outputs.append(output)
    return torch.cat(outputs,dim=2).float().clamp(-1,1)
