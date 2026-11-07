"""Predict source YUV from final latent. JM computes residual as source minus live MC."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_state_20261107"))
from train_state import load_config

W, H, T = 832, 480, 17
FRAME = W*H*3//2


class SourceHead(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.first = nn.Conv3d(16, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv3d(width,width,3,padding=1), nn.SiLU(),
                          nn.Conv3d(width,width,3,padding=1))
            for _ in range(4)
        ])
        self.last = nn.Conv3d(width, 384, 1)
        self.skip = nn.Conv3d(16, 384, 1)

    @staticmethod
    def unpack(packed, channels, scale):
        b, _, t, h, w = packed.shape
        arranged = packed.reshape(b,4,channels*scale**2,t,h,w).permute(0,3,1,2,4,5)
        arranged = torch.cat((arranged[:,0,:1], arranged[:,1:].flatten(1,2)), dim=1)
        return F.pixel_shuffle(arranged.flatten(0,1), scale).reshape(b,T,channels,h*scale,w*scale)

    def forward(self, latent):
        x = F.silu(self.first(latent))
        for block in self.blocks:
            x = x + block(x)*0.2
        packed = self.last(F.silu(x))+self.skip(latent)
        return self.unpack(packed[:,:256],1,8), self.unpack(packed[:,256:],2,4)


def read_planes(path):
    raw = np.fromfile(path,dtype=np.uint8)
    if raw.size != T*FRAME:
        raise ValueError(f"Invalid raw: {path}")
    frames=raw.reshape(T,FRAME)
    y=torch.from_numpy(frames[:,:W*H].copy()).reshape(1,T,1,H,W)
    uv=torch.from_numpy(frames[:,W*H:].copy()).reshape(1,T,2,H//2,W//2)
    return y,uv


def load_input(row, device):
    return torch.load(row["latent_path"], map_location="cpu", weights_only=True).unsqueeze(0).to(device).float()


def write_raw(output, planes):
    y, uv = [(x[0].float().clamp(-1,1)*127.5+127.5).round().byte().cpu().numpy() for x in planes]
    np.concatenate((y.reshape(T,-1),uv.reshape(T,-1)),axis=1).tofile(output)


def rows_for(config, index, root):
    return {split: [dict(sample_id=sid,latent_path=index[sid,20]["latent_path"],
                              raw_path=str(root/sid/"final_raw.yuv"))
                    for sid in ids] for split,ids in config["splits"].items()}


def main():
    p=argparse.ArgumentParser()
    for name in ("config","data","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--public",type=Path)
    p.add_argument("--traces",type=Path)
    p.add_argument("--init",type=Path)
    p.add_argument("--steps",type=int,default=1500)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng=random.Random(42)
    torch.backends.cudnn.allow_tf32=True
    config,_,index=load_config(a.config)
    rows=rows_for(config,index,a.data)
    public=[] if a.public is None else [r for r in json.loads(a.public.read_text()) if r["split"]=="train"]
    a.output.mkdir(parents=True,exist_ok=False)
    cache={}
    for row in rows["train"]+rows["val"]+public:
        cache[row["sample_id"]]=(load_input(row,"cpu"),read_planes(row["raw_path"]))
    model=SourceHead().cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init,weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4 if a.init else 3e-4)
    aligned={}
    if a.traces:
        for row in rows["train"]:
            sid=row["sample_id"]
            with np.load(a.traces/sid/"aligned_labels.npz",allow_pickle=False) as f:
                aligned[sid]=tuple(
                    tuple(torch.from_numpy(f[f"{key}_{plane}"]).unsqueeze(0).unsqueeze(2) if plane=="y"
                          else torch.from_numpy(f[f"{key}_{plane}"]).unsqueeze(0)
                          for key in ("mc","residual","mask")) for plane in ("y","uv"))
    def loss(row):
        latent, targets=cache[row["sample_id"]]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            predicted=model(latent.cuda())
        errors=[F.mse_loss(x.float(),target.cuda().float()/127.5-1) for x,target in zip(predicted,targets)]
        total=errors[0]+errors[1]*.5
        if model.training and row["sample_id"] in aligned:
            for plane,x in zip(aligned[row["sample_id"]],predicted):
                mc,residual,mask=(item.cuda() for item in plane)
                if mask.any():
                    estimated=x.float()*127.5+127.5-mc
                    total=total+0.25*F.mse_loss(estimated[mask]/127.5,residual[mask].float()/127.5)
        return total, errors[0]

    best=float("inf")
    history=[]
    for step in range(1,a.steps+1):
        model.train()
        pool=public if public and step%2==0 else rows["train"]
        opt.zero_grad(set_to_none=True)
        total,_=loss(rng.choice(pool))
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if step%100==0 or step==a.steps:
            model.eval()
            with torch.inference_mode():
                val=[float(loss(row)[1]) for row in rows["val"]]
            score=float(np.mean(val))
            stat=dict(step=step,validation_y_psnr=float(-10*np.log10(score/4)))
            history.append(stat)
            print(json.dumps(stat),flush=True)
            if score<best:
                best=score
                torch.save(model.state_dict(),a.output/"source.pt")
    (a.output/"training.json").write_text(json.dumps(dict(
        parameters=sum(p.numel() for p in model.parameters()), train_generated=len(rows["train"]),
        train_public=len(public),aligned_train=len(aligned),steps=a.steps,history=history,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),indent=2))


if __name__=="__main__":
    main()
