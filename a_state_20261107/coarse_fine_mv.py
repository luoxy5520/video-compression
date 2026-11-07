"""2-D candidate anchors + integer refinement + joint qpel phase, from hidden only."""
import argparse
import copy
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluate import load_backend
from rollout_residual import load_rollout, replay_item, save
from state import input_tensor, load_model, predict, records

N=24


class Motion(nn.Module):
    def __init__(self,width,anchors):
        super().__init__()
        self.register_buffer("anchors",anchors)
        self.conv=nn.Sequential(nn.Conv3d(width,64,3,padding=1),nn.SiLU(),
                                nn.Conv3d(64,48,3,padding=1),nn.SiLU())
        self.out=nn.Conv3d(48,len(anchors)*19,1)

    def forward(self,x):
        return self.out(self.conv(x))[0].movedim(0,-1).reshape(16,30,52,len(self.anchors),19)

    def decode(self, logits):
        idx=logits[...,0].argmax(-1)
        chosen=logits.gather(-2,idx[...,None,None].expand(*idx.shape,1,19)).squeeze(-2)
        integer=self.anchors[idx]+chosen[...,1:3].mul(4).round()
        phase=chosen[...,3:].argmax(-1)
        mv=integer*4+torch.stack((phase%4,phase//4),-1)
        return mv.clamp(-2048,2047).short()


def label(row,teacher=None):
    path=teacher/f"{row['sample_id']}.npz" if teacher else row["root"]/row["current_label_path"]
    with np.load(path,allow_pickle=False) as z:
        return torch.from_numpy(z["mode_id"]==1),torch.from_numpy(z["mv_qpel"].astype("i8"))


def codebook(group, seed=1701):
    # Sample equally per training clip; reserve anchors for distinct nonzero directions.
    rng=np.random.default_rng(seed)
    pool=[]
    for keep,v in group:
        x=(v[keep]//4).numpy()
        x=x[np.any(x!=0,axis=-1)]
        if len(x):
            pool.append(x[rng.choice(len(x),min(len(x),4000),replace=False)])
    x=np.concatenate(pool).astype("f4")
    freq=np.abs(x).max(-1)
    weights=np.sqrt(np.clip(freq,1,24))
    a=[np.zeros(2,dtype="f4")]
    for _ in range(N-1):
        d=((x[:,None]-np.stack(a)[None])**2).sum(-1).min(-1)
        p=np.sqrt(d)*weights
        a.append(x[rng.choice(len(x),p=p/p.sum())])
    a=np.stack(a)
    for _ in range(10):
        ix=((x[:,None]-a[None])**2).sum(-1).argmin(-1)
        for i in range(1,len(a)):
            keep=ix==i
            if keep.any(): a[i]=np.average(x[keep],axis=0,weights=weights[keep])
    return torch.from_numpy(np.unique(np.rint(a),axis=0).astype("f4"))


def loss_fn(head,raw,keep,v,risk_weight=0.):
    if not keep.any(): return None
    logits=raw[keep]
    v=v[keep]
    integer=v//4
    phase=v[:,0]%4+(v[:,1]%4)*4
    dist=(integer[:,None]-head.anchors[None]).square().sum(-1)
    idx=dist.argmin(-1)
    # Soft coarse targets make nearby anchors interchangeable.
    s=logits[...,0].log_softmax(-1)
    soft=(-dist/4).softmax(-1)
    rank=-(soft*s).sum(-1)
    chosen=logits[torch.arange(len(idx),device=idx.device),idx]
    target=(integer-head.anchors[idx])/4
    coarse=F.smooth_l1_loss(chosen[:,1:3],target,reduction="none",beta=.25).sum(-1)
    fine=F.cross_entropy(chosen[:,3:],phase,reduction="none")
    magnitude=v.float().norm(dim=-1)
    weights=torch.where(magnitude>=4,1.35,.65)
    loss=((rank+2*coarse+.25*fine)*weights).mean()
    if risk_weight:
        phases=torch.arange(16,device=v.device)
        phase_xy=torch.stack((phases%4,phases//4),-1).float()
        frac=logits[...,3:].softmax(-1) @ phase_xy
        candidates=(head.anchors[None]+4*logits[...,1:3])*4+frac
        error=(candidates-v[:,None]).square().sum(-1).add(1).sqrt()/16
        moving=torch.where(magnitude>=4,2.,.35)
        loss=loss+risk_weight*((s.exp()*error).sum(-1)*moving).mean()
    return loss


def score(pred,v,mask):
    v=v[mask].float()
    d=(pred[mask].float()-v).norm(dim=-1)
    large=v.norm(dim=-1)>=4
    if not len(d):
        return {}
    return dict(epe=float(d.mean()),exact=float((d==0).float().mean()),
                nonzero_epe=float(d[large].mean()) if large.any() else None,
                nonzero_count=int(large.sum()))


def saved(head,path):
    torch.save(dict(weights=head.state_dict(),anchors=head.anchors.cpu()),path)


def fit(a,base,data,train,val,arm):
    torch.manual_seed(a.seed)
    rng=random.Random(a.seed)
    teacher=a.teacher if arm=="teacher" else None
    truth={r["sample_id"]:label(r,teacher) for r in train}
    if a.resume_motion:
        previous = torch.load(a.resume_motion / f"{arm}.pt", map_location="cpu", weights_only=True)
        head = Motion(base.mode.in_channels, previous["anchors"]).cuda()
        head.load_state_dict(previous["weights"])
    else:
        anchors = codebook(list(truth.values()), a.seed).cuda()
        head = Motion(base.mode.in_channels, anchors).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=.0003 if a.resume_motion else .001)
    path=a.output/f"{arm}.pt"
    history=[]
    best=float("inf")
    order=list(truth)
    for e in range(a.epochs):
        head.train()
        rng.shuffle(order)
        vals=[]
        for sid in order:
            keep,v=(t.cuda() for t in truth[sid])
            loss=loss_fn(head,head(data[sid].cuda().float()),keep,v,a.risk_weight)
            if loss is None: continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),5)
            opt.step()
            vals.append(loss.item())
        head.eval()
        with torch.no_grad():
            validation=[]
            for r in val:
                p=head.decode(head(data[r["sample_id"]].cuda().float()))
                m,v=(x.cuda() for x in label(r))
                validation.append(score(p,v,m))
        overall=statistics.mean(x["epe"] for x in validation if x)
        moving=statistics.mean(x["nonzero_epe"] for x in validation if x.get("nonzero_epe") is not None)
        key=(overall+moving)/2
        row=dict(epoch=e+1,train_loss=statistics.mean(vals),val_epe=overall,val_nonzero_epe=moving)
        print(arm,json.dumps(row),flush=True)
        history.append(row)
        if key<best:
            best=key
            saved(head,path)
    head.load_state_dict(torch.load(path,weights_only=True)["weights"])
    return head,history


@torch.inference_mode()
def infer(base,head,row):
    z=input_tensor(row,base.kind)
    arrays,timing=predict(base,z,20)
    if head is None:
        return arrays,timing
    x=base.encode_features(z,z.new_tensor([1.]))
    v=head.decode(head(x))
    # Residual head is conditioned on the selected model MV, even before adaptation.
    r=base.residual(torch.cat((x,v.movedim(-1,0)[None].float()/16),dim=1))[0].movedim(0,-1)
    arrays["mv_qpel"]=v.cpu().numpy()
    r=r.mul(16).round().clamp(-255,255).short().cpu().numpy()
    for p,k,l,u in (("y",16,0,256),("u",8,256,320),("v",8,320,384)):
        arrays["residual_"+p]=r[...,l:u].reshape(16,30,52,k,k)
    # Old MV/residual probabilities do not apply to this new path.
    del arrays["confidence"]
    return arrays,timing


def assess(a,backend,base,heads,rows,stage,trace=False):
    from concurrent.futures import ThreadPoolExecutor
    output={key:[] for key in heads}
    jobs=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for row in rows:
            m,v=label(row)
            for key,head in heads.items():
                stem,motion = head if isinstance(head,tuple) else (base,head)
                arrays,_=infer(stem,motion,row)
                mv=torch.from_numpy(arrays["mv_qpel"])
                stat=score(mv,v,m)
                stat["zero_control"]=score(torch.zeros_like(v),v,m)
                folder=a.output/stage/key/row["sample_id"]
                job=pool.submit(replay_item,a,backend,folder,row,arrays,trace)
                jobs.append((key,stat,job))
        for key,stat,job in jobs:
            result=job.result()
            result["motion"]=stat
            output[key].append(result)
            print(stage,key,result["sample_id"],round(result["psnr_y_p_db"],3),flush=True)
    return output


def adapt(a,backend,base,head,train,data,arm):
    """Freeze the winning motion head and supervise residuals on its reached references."""
    assess(a,backend,base,{arm:head},train,"adapt_trace",trace=True)
    target_dir=a.output/"adapt_trace"/arm
    model=copy.deepcopy(base).eval().requires_grad_(False)
    model.residual.requires_grad_(True)
    opt=torch.optim.AdamW(model.residual.parameters(),lr=.0005)
    items={}
    for row in train:
        sid=row["sample_id"]
        x=data[sid].cuda().float()
        with torch.no_grad():
            mv=head.decode(head(x))
            inp=torch.cat((x,mv.movedim(-1,0)[None].float()/16),1).half().cpu()
        items[sid]=(inp,*load_rollout(target_dir,sid))
    rng=random.Random(1702)
    order=list(items)
    for epoch in range(1,2 if a.smoke else 9):
        rng.shuffle(order)
        losses=[]
        for sid in order:
            x,keep,r=items[sid]
            m=keep.cuda()
            if not m.any(): continue
            y=model.residual(x.cuda().float())[0].movedim(0,-1)
            loss=F.smooth_l1_loss(y[m],r.cuda()[m].float()/16,beta=.25)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual.parameters(),5)
            opt.step()
            losses.append(loss.item())
        print("ADAPT",epoch,statistics.mean(losses),flush=True)
    save(model,a.output/"adapted_residual.pt")
    return model


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+")
    p.add_argument("--test-config",nargs="+")
    for key in ("teacher","checkpoint","kit","jm","output","input-hidden","head","resume-motion"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--validation-only",action="store_true")
    p.add_argument("--risk-weight",type=float,default=0.)
    p.add_argument("--seed",type=int,default=1701)
    a=p.parse_args()
    if not a.checkpoint or not a.output or (not a.input_hidden and not all((a.config,a.teacher,a.kit,a.jm))):
        p.error("Need --checkpoint/--output; training also needs configs, teacher, kit and JM")
    if not a.input_hidden and not a.validation_only and not a.test_config:
        p.error("A regression evaluation requires --test-config")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    base=load_model(a.checkpoint).requires_grad_(False)
    assert base.kind=="hidden" and not base.recurrent
    if a.input_hidden:
        if not a.head or a.output.suffix!=".npz" or a.output.exists():
            p.error("Export requires --head and a fresh .npz")
        checkpoint=torch.load(a.head,weights_only=True,map_location="cpu")
        head=Motion(base.mode.in_channels,checkpoint["anchors"]).cuda().eval()
        head.load_state_dict(checkpoint["weights"])
        arrays,_=infer(base,head,dict(hidden_path=a.input_hidden,step=20))
        np.savez_compressed(a.output,**arrays)
        return
    a.output.mkdir(parents=True)
    index=records(a.config)
    train,val=([r for r in index[s] if r["step"]==20] for s in ("train","val"))
    test=[] if a.validation_only else [r for r in records(a.test_config)["test"] if r["step"]==20]
    assert {r["prompt"] for r in train+val}.isdisjoint(r["prompt"] for r in test)
    if a.smoke:
        train,val,test=train[:2],val[:1],test[:1]
        a.epochs=1
    data={}
    with torch.no_grad():
        for r in train+val:
            z=input_tensor(r,base.kind)
            data[r["sample_id"]]=base.encode_features(z,z.new_tensor([1.])).half().cpu()
    arms={}
    history={}
    for arm in ("oracle","teacher"):
        arms[arm],history[arm]=fit(a,base,data,train,val,arm)
    backend=load_backend(a.kit)
    validation=assess(a,backend,base,dict(base=None,**arms),val,"validation")
    native=statistics.mean(Path(r["current_oracle_state_path"]).with_name("reference.264")
                           .stat().st_size*128/17000 for r in val)
    eligible=[k for k in arms if statistics.mean(r["bitrate_kbps"] for r in validation[k])<=1.10*native]
    chosen=max(eligible,key=lambda k:statistics.mean(r["psnr_y_p_db"] for r in validation[k])) if eligible else None
    selected={} if chosen is None else {chosen:arms[chosen]}
    if chosen is not None:
        adapted=adapt(a,backend,base,arms[chosen],train,data,chosen)
        val_adapt=assess(a,backend,base,{"adapted":(adapted,arms[chosen])},val,"validation")
        validation.update(val_adapt)
        selected["adapted"]=(adapted,arms[chosen])
    testing={} if a.validation_only else assess(a,backend,base,dict(base=None,**selected),test,"regression")
    result = dict(selected=chosen, history=history, validation=validation, regression=testing,
                  eligible=eligible, smoke=a.smoke, risk_weight=a.risk_weight, seed=a.seed,
                  validation_only=a.validation_only,
                  resumed_from=a.resume_motion.name if a.resume_motion else None)
    pending = a.output / "results.tmp"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False))
    pending.replace(a.output / "results.json")


if __name__=="__main__":
    main()
