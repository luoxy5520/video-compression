"""A0.2: factorized discrete qpel head with overflow classes and fixed controls."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate import calibration
from state import latent, load_model, records

LIMIT=64
CLASSES=2*LIMIT+3


def data(rows):
    for row in rows:
        if row["step"]!=20:
            continue
        with np.load(row["root"]/row["current_label_path"],allow_pickle=False) as a:
            yield row,latent(row["final_latent_path"],"cpu"),torch.from_numpy(a["mode_id"]==1),torch.from_numpy(a["mv_qpel"]).long()


def metrics(pred,truth):
    n=len(truth)
    if n==0:
        return dict(count=0)
    d=pred-truth
    exact=(d==0).all(-1)
    mag=np.linalg.norm(truth,axis=-1)
    regions=[]
    for lo,hi in [(0,1),(1,4),(4,16),(16,np.inf)]:
        mask=(mag>=lo)&(mag<hi)
        regions.append(dict(lo=lo,hi=hi if np.isfinite(hi) else None,count=int(mask.sum()),
                           exact=float(exact[mask].mean()) if mask.any() else None))
    return dict(count=n,exact=float(exact.mean()),qpel_cross_bin=float((~exact).mean()),
                one_pixel_or_more=float((np.abs(d)>=4).any(-1).mean()),
                mae_xy=np.abs(d).mean(0).tolist(),bias_xy=d.mean(0).tolist(),
                epe=float(np.linalg.norm(d,axis=-1).mean()),by_motion=regions)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    for k in ("checkpoint","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--epochs",type=int,default=16)
    a=p.parse_args()
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(90)
    random.seed(90)
    torch.backends.cudnn.allow_tf32=True
    index=records(a.config)
    base=load_model(a.checkpoint).requires_grad_(False)
    head=nn.Sequential(nn.Conv3d(base.mode.in_channels,64,1),nn.SiLU(),nn.Conv3d(64,2*CLASSES,1)).cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=.002)
    cached={}
    for split in ("train","val","test"):
        group=[]
        for row,z,m,v in data(index[split]):
            with torch.no_grad():
                x=base.encode_features(z.cuda(),z.new_tensor([1.]).cuda())
                mode=base.mode(x)[0].argmax(0).cpu().bool()
                old=base.motion(x)[0].movedim(0,-1).mul(16).round().cpu().long()
                group.append((row["sample_id"],x[0].half().cpu(),m,v,mode,old))
        cached[split]=group
    best=-1
    history=[]
    for epoch in range(1,a.epochs+1):
        random.shuffle(cached["train"])
        scores={}
        for split in ("train","val"):
            train=split=="train"
            losses,exacts=[],[]
            head.train(train)
            with torch.set_grad_enabled(train):
                for _,x,m,v,_,old in cached[split]:
                    if not m.any():
                        continue
                    mask=m.cuda()
                    logits=head(x[None].cuda().float())[0].movedim(0,-1).reshape(*m.shape,2,CLASSES)[mask]
                    label=v[m].cuda()
                    bins=(label+LIMIT+1).clamp(0,CLASSES-1)
                    loss=F.cross_entropy(logits.flatten(0,1),bins.flatten())
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    losses.append(loss.item())
                    choice=logits.argmax(-1)
                    pred=choice-LIMIT-1
                    fallback=old[m].cuda()
                    pred=torch.where(choice==0,fallback.clamp(max=-LIMIT-1),pred)
                    pred=torch.where(choice==CLASSES-1,fallback.clamp(min=LIMIT+1),pred)
                    exacts.append((pred==label).all(-1).float().mean().item())
            scores[split]=dict(ce=float(np.mean(losses)),exact=float(np.mean(exacts)))
        history.append(dict(epoch=epoch,**scores))
        print(json.dumps(history[-1]),flush=True)
        if scores["val"]["exact"]>best:
            best=scores["val"]["exact"]
            torch.save(dict(epoch=epoch,limit=LIMIT,weights=head.state_dict()),a.output/"mv.pt")
    head.load_state_dict(torch.load(a.output/"mv.pt",weights_only=True)["weights"])
    head.eval()
    result=[]
    with torch.inference_mode():
        for sid,x,m,v,mode,old in cached["test"]:
            logits=head(x[None].cuda().float())[0].movedim(0,-1).reshape(*m.shape,2,CLASSES)
            conf,choice=logits.softmax(-1).max(-1)
            choice=choice.cpu()
            pred=choice-LIMIT-1
            pred=torch.where(choice==0,old.clamp(max=-LIMIT-1),pred)
            pred=torch.where(choice==CLASSES-1,old.clamp(min=LIMIT+1),pred)
            masks={"oracle_p16":m,"joint_p16":m&mode}
            values={}
            for name,mask in masks.items():
                pv,truth=pred[mask].numpy(),v[mask].numpy()
                values[name]=dict(coverage=int(mask.sum()),discrete=metrics(pv,truth),
                                  continuous=metrics(old[mask].numpy(),truth),
                                  zero=metrics(np.zeros_like(pv),truth),
                                  confidence=calibration(conf.prod(-1).cpu()[mask].numpy(),(pv==truth).all(-1)))
            np.savez_compressed(a.output/f"{sid}.npz",mv_qpel=pred.short().numpy(),
                                confidence=conf.prod(-1).cpu().numpy())
            result.append(dict(sample_id=sid,**values))
    (a.output/"results.json").write_text(json.dumps(dict(history=history,per_video=result),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
