"""Distill a small source head from Wan's frozen half-resolution decode features."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_source import load_config, load_input, read_planes, rows_for


class Captured(Exception):
    pass


class FrozenPrefix:
    def __init__(self):
        sys.path.insert(0,"/home/luoxiangyu/Wan2.1")
        from wan.modules.vae import WanVAE
        self.vae=WanVAE(vae_pth="/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",device="cuda")
        self.last_resample=[m for m in self.vae.model.decoder.upsamples if hasattr(m,"resample")][-1]

    @torch.no_grad()
    def __call__(self,latent):
        model=self.vae.model
        def stop(_module, args):
            raise Captured(args[0])
        hook=self.last_resample.resample.register_forward_pre_hook(stop)
        model.clear_cache()
        outputs=[]
        try:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                scale=self.vae.scale
                x=model.conv2(latent/scale[1].view(1,16,1,1,1)+scale[0].view(1,16,1,1,1))
                for t in range(x.shape[2]):
                    model._conv_idx=[0]
                    try:
                        model.decoder(x[:,:,t:t+1],feat_cache=model._feat_map,feat_idx=model._conv_idx)
                    except Captured as capture:
                        outputs.append(capture.args[0].to(torch.bfloat16))
                    else:
                        raise RuntimeError("Frozen decoder did not reach tap")
            result=torch.cat(outputs,dim=0)
            if (result.shape[0],*result.shape[-2:])!=(17,240,416):
                raise ValueError(f"Unexpected prefix shape {result.shape}")
            return result
        finally:
            hook.remove()
            model.clear_cache()


class TailHead(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.first=nn.Conv2d(channels,96,1)
        self.blocks=nn.ModuleList([
            nn.Sequential(nn.Conv2d(96,96,3,padding=1),nn.SiLU(),nn.Conv2d(96,96,3,padding=1))
            for _ in range(4)])
        self.out=nn.Conv2d(96,6,3,padding=1)
        self.skip=nn.Conv2d(channels,6,1)

    def forward(self,x):
        f=F.silu(self.first(x))
        for block in self.blocks:
            f=f+0.2*block(f)
        yuv=self.out(F.silu(f))+self.skip(x)
        return F.pixel_shuffle(yuv[:,:4],2),yuv[:,4:]


def main():
    p=argparse.ArgumentParser()
    for key in ("config","data","public","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--steps",type=int,default=4000)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng=random.Random(42)
    config,_,index=load_config(a.config)
    splits=rows_for(config,index,a.data)
    public=[row for row in json.loads(a.public.read_text()) if row["split"]=="train"]
    a.output.mkdir(parents=True,exist_ok=False)
    prefix=FrozenPrefix()
    patches={"train":[],"public":[],"val":[]}
    for split,rows in (("train",splits["train"]),("public",public),("val",splits["val"])):
        for row in rows:
            features=prefix(load_input(row,"cuda"))
            y,uv=read_planes(row["raw_path"])
            for _ in range(12 if split!="val" else 24):
                t=rng.randrange(17)
                i,j=rng.randrange(177),rng.randrange(353)
                x=features[t,:,i:i+64,j:j+64].contiguous().cpu()
                yt=y[0,t,:,2*i:2*i+128,2*j:2*j+128].contiguous()
                ut=uv[0,t,:,i:i+64,j:j+64].contiguous()
                patches[split].append((x,yt,ut))
            print(json.dumps(dict(cached=row["sample_id"],split=split,patches=len(patches[split]))),flush=True)
            del features
    channels=patches["train"][0][0].shape[0]
    model=TailHead(channels).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)

    def loss(batch):
        x,y,uv=(torch.stack(items).cuda() for items in zip(*batch))
        with torch.autocast("cuda",dtype=torch.bfloat16):
            yp,up=model(x)
        ym=F.mse_loss(yp.float(),y.float()/127.5-1)
        return ym+0.5*F.mse_loss(up.float(),uv.float()/127.5-1),ym

    best=math.inf
    history=[]
    for step in range(1,a.steps+1):
        model.train()
        pool=patches["train" if step%2 else "public"]
        opt.zero_grad(set_to_none=True)
        cost,_=loss(rng.sample(pool,8))
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if step%250==0 or step==a.steps:
            model.eval()
            with torch.inference_mode():
                values=[float(loss(patches["val"][i:i+8])[1]) for i in range(0,len(patches["val"]),8)]
            score=float(np.mean(values))
            row=dict(step=step,validation_patch_y_psnr=-10*math.log10(score/4))
            print(json.dumps(row),flush=True)
            history.append(row)
            if score<best:
                best=score
                torch.save(dict(channels=channels,weights=model.state_dict()),a.output/"tail.pt")
    (a.output/"training.json").write_text(json.dumps(dict(channels=channels,
        trainable_parameters=sum(x.numel() for x in model.parameters()),steps=a.steps,
        patches={k:len(v) for k,v in patches.items()},history=history),indent=2))


if __name__=="__main__":
    main()
