"""Three supervision representations with identical input, network and replay budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from l0 import HIST, UPDATES, chosen_rows, decode, embed, p, rank, replay, value_logits

FREQ=[0,1,4]
ARMS=("Sparse","Continuous","Region")
K=64


def sample(root,row):
    sid=row["sample_id"]
    x=torch.load(HIST/"features"/f"{sid}.pt",weights_only=True).cuda()
    data=p.p05.load_npz(root/sid/"A.npz")
    mode=torch.from_numpy(p.p05.load_npz(HIST/"labels"/f"{sid}.npz")["mode"]==1).cuda()
    sparse=torch.from_numpy(data["sparse"][...,FREQ]).cuda().long()[mode]
    scale=torch.tensor(p.SCALE.flatten()[FREQ] / (1<<19),device=x.device,dtype=torch.float32)
    continuous=torch.from_numpy(data["continuous"]).cuda().float()[mode]*scale
    return x,mode,sparse,continuous


def ranking(scores,active):
    pos=scores[active]
    neg=scores.masked_fill(active,-torch.inf).flatten().topk(len(pos)).values
    return F.softplus(-pos).mean()+F.softplus(neg).mean()


def real_level(logits,max_q):
    magnitudes=torch.arange(1,65,device=logits.device)
    masked=logits[:,1:].masked_fill(magnitudes>max_q[:,None],-torch.inf)
    return (masked.softmax(1)*magnitudes).sum(1)*logits[:,0].tanh()


def train_step(net, data, arm, generator, max_q):
    x,mode,q,c=data
    h=embed(net,x)[mode]
    s=net.select(h)
    active=q!=0
    i,b,f=active.nonzero(as_tuple=True)
    select=ranking(s,active)
    if arm=="Region":
        region=s.flatten(1).max(1).values
        select=(select+ranking(region,active.flatten(1).any(1)))/2
    if arm=="Continuous":
        chosen=torch.randint(c.numel(),(8192,),generator=generator,device=x.device)
        loc, fi=chosen//3,chosen%3
        si,sb=loc//16,loc%16
        i,b,f=(torch.cat((left,right)) for left,right in ((i,si),(b,sb),(f,fi)))
        z=value_logits(net,h[i,b],f)
        value=F.smooth_l1_loss(real_level(z,max_q[f])/8,c[i,b,f]/8)
    else:
        z=value_logits(net,h[i,b],f)
        y=q[i,b,f]
        value=F.binary_cross_entropy_with_logits(z[:,0],(y>0).float())+F.cross_entropy(z[:,1:],y.abs()-1)
    return select,value


@torch.inference_mode()
def infer(net,data,arm,max_q,diagnostics=True):
    x,mode,q,c=data
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before=torch.cuda.memory_allocated()
    start=time.perf_counter()
    h=embed(net,x)[mode]
    s=net.select(h)
    best_score,best_f=s.max(-1)
    if arm=="Region":
        region,b=best_score.max(-1)
        i=region.argsort(descending=True,stable=True)[:K]
        b=b[i]
    else:
        chosen=best_score.flatten().argsort(descending=True,stable=True)[:K]
        i,b=chosen//16,chosen%16
    f=best_f[i,b]
    if arm=="Continuous":
        signed=real_level(value_logits(net,h[i,b],f),max_q[f])
        y=(signed.sign()*torch.floor(signed.abs()+342/2048)).long()
        quantization_error=float((signed-y).abs().mean())
    else:
        y=decode(net,h[i,b],f,max_q)
        quantization_error=None
    local=np.zeros((*h.shape[:2],16),np.int16)
    local[i.cpu(),b.cpu(),np.asarray(FREQ)[f.cpu()]]=(y*(y.abs()>=2)).cpu()
    output=np.zeros((*p.SHAPE,16),np.int16)
    output[mode.cpu().numpy()]=local
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()-before
    if not diagnostics:
        return output, {}
    target=q!=0
    support=local[...,FREQ]!=0
    common=support&target.cpu().numpy()
    ti,tb,tf=target.nonzero(as_tuple=True)
    if arm=="Continuous":
        real=real_level(value_logits(net,h[ti,tb],tf),max_q[tf])
        value=(real.sign()*torch.floor(real.abs()+342/2048)).long()
    else:
        value=decode(net,h[ti,tb],tf,max_q)
    target_values=q[ti,tb,tf]
    return output, dict(budget=K,nonzero=int(support.sum()),positive=int(target.sum()),overlap=int(common.sum()),
            precision=float(common.sum()/max(1,support.sum())),recall=float(common.sum()/max(1,int(target.sum()))),
            dc=rank(s[...,0].cpu().numpy().ravel(),target[...,0].cpu().numpy().ravel(),int(target[...,0].sum())),
            value_mae=float((value-target_values).abs().float().mean()),
            value_exact=float((value==target_values).float().mean()),
            sign_correct=float((value.sign()==target_values.sign()).float().mean()),
            quantization_error=quantization_error,head_seconds=elapsed,head_peak_extra_bytes=peak)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("labels",type=Path)
    cli.add_argument("out",type=Path)
    cli.add_argument("--arm",choices=ARMS,required=True)
    cli.add_argument("--seed",type=int,required=True)
    a=cli.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    a.out.mkdir(exist_ok=False)
    rows=p.rows_by_split()
    train=chosen_rows(rows["train"])
    data=[sample(a.labels,row) for row in train]
    net=p.SparseHead(67,3).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=0.001)
    shuffle=torch.Generator().manual_seed(a.seed)
    position=torch.Generator(device="cuda").manual_seed(114+int(a.seed))
    max_q=torch.tensor([p.max_level(f) for f in FREQ],device="cuda")
    history=[]
    for i in range(UPDATES):
        if i%len(data)==0:
            order=torch.randperm(len(data),generator=shuffle).tolist()
        opt.zero_grad(set_to_none=True)
        select,value=train_step(net,data[order[i%len(data)]],a.arm,position,max_q)
        (select+value).backward()
        opt.step()
        if i==0 or (i+1)%50==0:
            history.append(dict(update=i+1,loc=float(select.detach()),value=float(value.detach())))
    torch.save(net.state_dict(),a.out/"head.pt")
    p.save_json(a.out/"train_history.json",history)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[]
        for row in rows["val"]:
            levels,diag=infer(net.eval(),sample(a.labels,row),a.arm,max_q)
            jobs.append(pool.submit(replay,a.out,"val",row,levels,diag))
        points=[job.result() for job in jobs]
    p.save_json(a.out/"run.json",dict(arm=a.arm,seed=a.seed,scope="L2",points=points,
                 params=sum(v.numel() for v in net.parameters()),summary=p.summarize(points)))
    print(a.arm,a.seed,p.summarize(points),flush=True)


if __name__=="__main__":
    main()
