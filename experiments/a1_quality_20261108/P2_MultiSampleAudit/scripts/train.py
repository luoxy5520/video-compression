"""Scale, duplicate, and orthogonal optimization diagnostics."""
import argparse
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from m1 import ON,p,chosen_rows
from o1 import infer,level_bounds,data_for,score
from l0 import decode,embed,value_logits

SIZES=(1,2,4,8,44)


def training(net, sample, policy, generator=None, collect=False):
    x,mode,q=sample
    h=embed(net,x)[mode]
    logits=net.select(h)
    i,b,f=(q!=0).nonzero(as_tuple=True)
    pos=q!=0
    n=len(i)
    z=logits[i,b,f]
    negatives=logits.masked_fill(pos,-torch.inf).flatten()
    hard=negatives.topk(n).values
    if policy=="bce":
        location=F.binary_cross_entropy_with_logits(logits,pos.float(),pos_weight=logits.new_tensor(24))
        lp=F.softplus(-z).sum()*24/logits.numel()
        ln=location-lp
    else:
        if policy=="mixed":
            sampled=torch.randint(negatives.numel(),(n,),device=logits.device,generator=generator)
            easy=negatives[sampled]
            # A sampled positive is excluded by the same legality mask.
            hard=torch.cat((hard[:n//2],easy[:n-n//2]))
        lp=F.softplus(-z).mean()
        ln=F.softplus(hard).mean()
        location=lp+ln
    v=value_logits(net,h[i,b],f)
    truth=q[i,b,f]
    value=F.binary_cross_entropy_with_logits(v[:,0],(truth>0).float())+F.cross_entropy(v[:,1:],truth.abs()-1)
    if collect:
        if policy=="bce":
            pos_g=24*(1-z.sigmoid()).norm()/logits.numel()
            neg_g=logits[~pos].sigmoid().norm()/logits.numel()
        else:
            pos_g=(1-z.sigmoid()).norm()/n
            neg_g=hard.sigmoid().norm()/len(hard)
        ds=dict(pos=float(lp.detach()),neg=float(ln.detach()),value=float(value.detach()),
           positive_logit_grad_norm=float(pos_g.detach()),negative_logit_grad_norm=float(neg_g.detach()),
           candidates=logits.numel(),positives=n,negatives_used=logits.numel()-n if policy=="bce" else n,
           hard_above_median_positive=float((hard>z.median()).float().mean()),
           mean_logit=float(logits.mean().detach()),top_logit=float(logits.max().detach()))
    else:
        ds={}
    return location+value,ds


@torch.inference_mode()
def measure(net, sample, sid):
    x,mask,q=sample
    maxima=level_bounds("cuda")
    out,h,logits,cost=infer(net,x,mask,maxima)
    local=out[mask.cpu().numpy()]
    truth=q.cpu().numpy()
    hits=(local!=0)&(truth!=0)
    tp=int(hits.sum())
    i,b,f=(q!=0).nonzero(as_tuple=True)
    values=decode(net,h[i,b],f,maxima)
    y=q[i,b,f]
    return dict(sample=sid,positives=len(i),candidates=q.numel(),hits=tp,
                recall=tp/len(i),precision=tp/max(1,cost["nonzero"]),precision_at_64=tp/64,
                iou=tp/(len(i)+cost["nonzero"]-tp),
                joint_exact=int((hits&(local==truth)).sum()),
                value_exact=float((values==y).float().mean()),
                sign_correct=float((values.sign()==y.sign()).float().mean()),
                value_mae=float((values-y).abs().float().mean()),
                logit_quantiles=torch.quantile(logits.flatten()[::max(1,logits.numel()//4096)],
                        torch.tensor([0,.5,.9,.99,1.],device=logits.device)).tolist(),**cost)


def sample_rows(rows,n):
    representatives=chosen_rows(rows)
    return representatives[:n] if n<44 else rows


def grad_norm(group):
    return float(torch.stack([x.grad.detach().square().sum() for x in group if x.grad is not None]).sum().sqrt())


def train(root,out,n,seed,duplicate,policy,steps,checkpoints):
    torch.manual_seed(seed)
    order_rng=torch.Generator().manual_seed(seed)
    negatives=torch.Generator(device="cuda").manual_seed(seed+1000)
    rows=p.rows_by_split()
    train_rows=sample_rows(rows["train"],n)
    if duplicate:
        train_rows=[chosen_rows(rows["train"])[0]]*n
    samples=[data_for(ON/"o0",row,"T0") for row in train_rows]
    if duplicate:
        assert all(all(torch.equal(a,b) for a,b in zip(samples[0],s,strict=True)) for s in samples)
    vals=[data_for(ON/"o0",row,"T0") for row in rows["val"]]
    net=p.SparseHead(67,16).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    candidate_mean=np.mean([q.numel() for _,_,q in samples])
    out.mkdir(exist_ok=False)
    initial=time.time()
    exposure=Counter()
    snapshots={}
    stats=[]
    for i in range(1,steps+1):
        if (i-1)%n==0:
            perm=torch.randperm(n,generator=order_rng).tolist()
        if policy=="blocked":
            full_blocks=(steps//(4*n))*(4*n)
            idx=((i-1)//4)%n if i<=full_blocks else (i-1-full_blocks)%n
        else:
            idx=perm[(i-1)%n]
        exposure[idx]+=1
        optimizer.zero_grad(set_to_none=True)
        log=i==1 or policy!="hard" or i%50==0
        loss,diagnostics=training(net,samples[idx],policy,negatives,log)
        if policy=="candidate_weighted":
            loss=loss*(samples[idx][2].numel()/candidate_mean)
        loss.backward()
        if log:
            stats.append(dict(update=i,sample=train_rows[idx]["sample_id"],replica=idx,
                 exposure=exposure[idx],lr=.001,batch=1,weight_decay=.01,total=float(loss.detach()),
                 select_grad=grad_norm(net.select.parameters()),value_grad=grad_norm(net.value.parameters()),
                 shared_grad=grad_norm(net.body.parameters()),**diagnostics))
        optimizer.step()
        if i in checkpoints:
            train_points=[measure(net,s,r["sample_id"]) for s,r in zip(samples,train_rows,strict=True)]
            val_points=[measure(net,s,r["sample_id"]) for s,r in zip(vals,rows["val"],strict=True)]
            checkpoints_path=out/f"head-{i}.pt"
            torch.save(net.state_dict(),checkpoints_path)
            snapshots[i]=dict(train=train_points,val=val_points,exposure=dict(exposure),head_sha256=p.digest(checkpoints_path))
            print(i,float(np.mean([r["recall"] for r in train_points])),
                  float(np.mean([r["recall"] for r in val_points])),flush=True)
    p.save_json(out/"run.json",dict(n=n,seed=seed,policy=policy,duplicate=duplicate,steps=steps,
                snapshots=snapshots,history=stats,started_epoch_s=initial,elapsed_s=time.time()-initial))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("root",type=Path)
    a.add_argument("out",type=Path)
    a.add_argument("--n",type=int,choices=SIZES,required=True)
    a.add_argument("--seed",type=int,required=True)
    a.add_argument("--duplicate",action="store_true")
    a.add_argument("--policy",choices=("hard","bce","mixed","candidate_weighted","blocked"),default="hard")
    a.add_argument("--steps",type=int,required=True)
    a.add_argument("--checkpoints",type=int,nargs="+",required=True)
    args=a.parse_args()
    assert json.loads((args.root/"m1_unit_tests.json").read_text())["pass_m1"]
    torch.set_num_threads(2)
    train(args.root,args.out,args.n,args.seed,args.duplicate,args.policy,args.steps,set(args.checkpoints))


if __name__=="__main__":
    main()
