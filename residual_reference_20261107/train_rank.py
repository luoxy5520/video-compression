"""Distill a low-rank causal output projection with the pretrained decoder frozen."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_source import load_config, load_input, rows_for
from train_tail import FrozenPrefix


class RankProjection(nn.Module):
    def __init__(self, channels, rank):
        super().__init__()
        self.in_proj=nn.Conv3d(channels,rank,1,bias=False)
        self.out_proj=nn.Conv3d(rank,3,3)

    def forward(self,x):
        return self.out_proj(self.in_proj(x))

    @torch.no_grad()
    def initialize(self, layer, svd):
        u,s,v=svd
        rank=self.in_proj.out_channels
        self.in_proj.weight.copy_(u[:,:rank].T[:,:,None,None,None])
        kernel=(s[:rank,None]*v[:rank]).reshape(rank,3,3,3,3).transpose(0,1)
        self.out_proj.weight.copy_(kernel)
        self.out_proj.bias.copy_(layer.bias)


def install(vae, model):
    layer=vae.model.decoder.head[-1]
    def forward(x, cache_x=None):
        padding=list(layer._padding)
        if cache_x is not None and padding[4]>0:
            x=torch.cat((cache_x.to(x.device),x),dim=2)
            padding[4]-=cache_x.shape[2]
        return model(F.pad(x,padding))
    layer.forward=forward


def main():
    p=argparse.ArgumentParser()
    for key in ("config","data","public","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--steps",type=int,default=1000)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng=random.Random(42)
    config,_,index=load_config(a.config)
    split=rows_for(config,index,a.data)
    public=[row for row in json.loads(a.public.read_text()) if row["split"]=="train"]
    a.output.mkdir(parents=True,exist_ok=False)
    vae=FrozenPrefix().vae
    layer=vae.model.decoder.head[-1]
    groups={"train":[],"val":[]}
    current=groups["train"]

    def collect(_module,args,out):
        x=args[0]
        cache=args[1] if len(args)>1 else None
        h,w=out.shape[-2:]
        for _ in range(2):
            i,j=rng.randrange(1,h-32),rng.randrange(1,w-32)
            patch=x[:,:,:1,i-1:i+33,j-1:j+33]
            if cache is not None:
                patch=torch.cat((cache[:,:,-2:,i-1:i+33,j-1:j+33].to(x.device),patch),dim=2)
            if patch.shape[2]<3:
                patch=F.pad(patch,(0,0,0,0,3-patch.shape[2],0))
            target=out[:,:,:1,i:i+32,j:j+32]
            current.append((patch[0].to(torch.bfloat16).cpu(),target[0].to(torch.bfloat16).cpu()))

    hook=layer.register_forward_hook(collect)
    try:
        for name,rows in (("train",split["train"]+public),("val",split["val"])):
            current=groups[name]
            for row in rows:
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    vae.decode([load_input(row,"cuda")[0]])
                print(json.dumps(dict(cached=row["sample_id"],split=name,patches=len(current))),flush=True)
    finally:
        hook.remove()
    channels=layer.in_channels
    matrix=layer.weight.detach().float().permute(1,0,2,3,4).reshape(channels,-1)
    svd=torch.linalg.svd(matrix,full_matrices=False)

    def evaluate(head):
        losses=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for i in range(0,len(groups["val"]),8):
                x,y=(torch.stack(items).cuda() for items in zip(*groups["val"][i:i+8]))
                losses.append(float(F.mse_loss(head(x).float().clamp(-1,1),y.float().clamp(-1,1))))
        return -10*math.log10(max(sum(losses)/len(losses),1e-12)/4)

    choices=[]
    for rank in (48,56,64,72,80):
        model=RankProjection(channels,rank).cuda()
        model.initialize(layer,svd)
        quality=evaluate(model)
        choices.append(dict(rank=rank,validation_patch_rgb_psnr=quality))
        if quality>=52:
            break
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
    best=quality
    history=[dict(step=0,validation_patch_rgb_psnr=quality)]

    def save():
        torch.save(dict(weights=model.state_dict(),channels=channels,rank=rank),a.output/"rank.pt")
    save()
    for step in range(1,a.steps+1):
        x,y=(torch.stack(items).cuda() for items in zip(*rng.sample(groups["train"],8)))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(x)
        F.mse_loss(output.float(),y.float()).backward()
        opt.step()
        if step%200==0 or step==a.steps:
            quality=evaluate(model)
            record=dict(step=step,validation_patch_rgb_psnr=quality)
            history.append(record)
            print(json.dumps(record),flush=True)
            if quality>best:
                best=quality
                save()
    (a.output/"training.json").write_text(json.dumps(dict(
        rank=rank,original_parameters=sum(x.numel() for x in layer.parameters()),
        trainable_parameters=sum(x.numel() for x in model.parameters()),rank_selection=choices,
        history=history,best=best,patches={k:len(v) for k,v in groups.items()}),indent=2))


if __name__=="__main__":
    main()
