"""Matched capacity and position-loss probes on the audited eight T0 clips."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"P2_MultiSampleAudit/scripts"))
from m1 import ON,p
from train import sample_rows,training,measure
from o1 import data_for,infer,level_bounds
from l0 import embed,decode,value_logits

WIDTHS={"A":64,"B":352,"C":1504}
HIST=p.ROOT/"P2_MultiSampleAudit"


def build(width):
    old=p.CFG["hidden_width"]
    try:
        p.CFG["hidden_width"]=width
        return p.SparseHead(67,16).cuda()
    finally:
        p.CFG["hidden_width"]=old


def pairwise(net,sample,rng):
    x,mask,q=sample
    h=embed(net,x)[mask]
    logits=net.select(h)
    i,b,f=(q!=0).nonzero(as_tuple=True)
    pos=logits[i,b,f]
    negatives=logits.masked_fill(q!=0,-torch.inf).flatten()
    hard=negatives.topk(len(i)//2).values
    count=len(i)-len(hard)
    idx=torch.randint(negatives.numel(),(count*2,),device=x.device,generator=rng)
    easy=negatives[idx]
    easy=easy[torch.isfinite(easy)][:count]
    assert len(easy)==count
    neg=torch.cat((hard,easy))
    loc=F.softplus(neg[None,:]-pos[:,None]).mean()
    v=value_logits(net,h[i,b],f)
    target=q[i,b,f]
    value=F.binary_cross_entropy_with_logits(v[:,0],(target>0).float())+F.cross_entropy(v[:,1:],target.abs()-1)
    return loc+value,dict(pos=float(loc.detach()),neg=0.,value=float(value.detach()))


@torch.inference_mode()
def full_diagnostics(net,sample):
    x,mask,q=sample
    maxima=level_bounds("cuda")
    out,h,logits,_=infer(net,x,mask,maxima)
    local=out[mask.cpu().numpy()]
    truth=q.cpu().numpy()
    t=mask.nonzero()[:,0]
    targets=[]
    ranks=[]
    filtered=[]
    for f in range(16):
        use=t==f
        scores,freq=logits[use].max(-1)
        k=scores.flatten().argsort(descending=True,stable=True)[:4]
        b,freq_hat=k%16,freq.flatten()[k]
        indices=use.nonzero().flatten()[k//16]
        targets.append(int(torch.count_nonzero(q[indices,b,freq_hat])))
        filtered.append(int(np.count_nonzero((local[use.cpu()]!=0)&(truth[use.cpu()]!=0))))
        s=logits[use].flatten()
        teacher=q[use].flatten()!=0
        selected=s.topk(16).indices
        ranks.append([int(teacher[selected[:k]].sum()) for k in (1,4,8,16)])
    pos=q!=0
    a=logits[pos]
    hard=logits.masked_fill(pos,-torch.inf).flatten().topk(64).values
    return dict(frame_hits_pre_filter=targets,frame_hits_written=filtered,frame_unconstrained_rank_hits=ranks,
                pos_logit_quantiles=torch.quantile(a,torch.tensor([0,.5,1.],device=x.device)).tolist(),
                hard_logit_quantiles=torch.quantile(hard,torch.tensor([0,.5,1.],device=x.device)).tolist(),
                hard_higher_than_median_positive=int((hard>a.median()).sum()))


def go(root,arm,seed,loss,steps):
    torch.set_num_threads(2)
    audit=json.loads((root/"audit/audit.json").read_text())
    assert len(audit["clips"])==16
    rows=p.rows_by_split()
    tr=sample_rows(rows["train"],8)
    torch.manual_seed(seed)
    order=torch.Generator().manual_seed(seed)
    neg_rng=torch.Generator(device="cuda").manual_seed(seed+1000)
    train=[data_for(ON/"o0",r,"T0") for r in tr]
    val=[data_for(ON/"o0",r,"T0") for r in rows["val"]]
    net=build(WIDTHS[arm])
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    name=f"c0-{arm}-{seed}" if loss=="hard" else f"c1-rank-{arm}-{seed}"
    dest=root/name
    dest.mkdir(exist_ok=False)
    checkpoints=sorted({400,800,1600,2400,3200,steps} & set(range(1,steps+1)))
    exposure=Counter()
    snap={}
    hist=[]
    start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for update in range(1,steps+1):
        if (update-1)%8==0:
            perm=torch.randperm(8,generator=order).tolist()
        i=perm[(update-1)%8]
        exposure[tr[i]["sample_id"]]+=1
        opt.zero_grad(set_to_none=True)
        log=update==1 or update%50==0
        value,detail=training(net,train[i],"hard",collect=log) if loss=="hard" else pairwise(net,train[i],neg_rng)
        value.backward()
        opt.step()
        if log:
            hist.append(dict(step=update,sample=tr[i]["sample_id"],lr=.001,loss=float(value.detach()),**detail))
        if update in checkpoints:
            entry={}
            for k,clips,source in (("train",tr,train),("val",rows["val"],val)):
                entry[k]=[]
                for clip,data in zip(clips,source,strict=True):
                    one=measure(net,data,clip["sample_id"])
                    if update in (3200,steps):
                        one.update(full_diagnostics(net,data))
                    one["prompt"]=clip["prompt"]
                    entry[k].append(one)
            ck=dest/f"head-{update}.pt"
            torch.save(net.state_dict(),ck)
            entry.update(exposures=dict(exposure),head_sha256=p.digest(ck))
            snap[update]=entry
            print(update,*(mean(r["recall"] for r in entry[k]) for k in ("train","val")),flush=True)
            p.save_json(dest/"progress.json",dict(snapshots=snap,history=hist))
            if arm=="A" and loss=="hard" and update==3200:
                hist_path=HIST/f"scale-8-{seed}"
                prev=torch.load(hist_path/"head-3200.pt",weights_only=True)
                matched=all(torch.equal(prev[k],v) for k,v in net.state_dict().items())
                old=json.loads((hist_path/"run.json").read_text())["snapshots"]["3200"]["train"]
                assert [r["hits"] for r in entry["train"]]==[r["hits"] for r in old]
                assert matched
                p.save_json(dest/"historical_match.json",dict(parameter_tensor_exact=matched,train_hits=[r["hits"] for r in old]))
    p.save_json(dest/"run.json",dict(arm=arm,width=WIDTHS[arm],loss=loss,seed=seed,params=sum(v.numel() for v in net.parameters()),
        steps=steps,elapsed_s=time.monotonic()-start,snapshots=snap,history=hist))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("--arm",choices=tuple(WIDTHS),required=True)
    cli.add_argument("--seed",type=int,required=True)
    cli.add_argument("--loss",choices=("hard","rank"),default="hard")
    cli.add_argument("--steps",type=int,default=4000)
    a=cli.parse_args()
    go(a.root,a.arm,a.seed,a.loss,a.steps)


if __name__=="__main__":
    main()
