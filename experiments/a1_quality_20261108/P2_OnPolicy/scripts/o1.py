"""Identical sparse heads trained on different reference-conditioned targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean
import tempfile
import time

import numpy as np
import torch

from o0 import ARMS, FRAME_K, decode as decode_reference, historical, p, persist, strict
from l0 import HIST, chosen_rows, decode, embed, rank, step
from repair import to_pixels

FREQS=list(range(16))


def labels_for(root,row,arm):
    return torch.from_numpy(p.p05.load_npz(root/row["sample_id"]/f"{arm}.npz")["q"])


def data_for(root,row,arm):
    sid=row["sample_id"]
    feature=torch.load(HIST/"features"/f"{sid}.pt",weights_only=True).cuda()
    old=p.p05.load_npz(HIST/"labels"/f"{sid}.npz")
    mask=torch.from_numpy(old["mode"]==1).cuda()
    q=labels_for(root,row,arm).cuda()[mask].long()
    return feature,mask,q


def level_bounds(device):
    return torch.tensor([p.max_level(f) for f in FREQS],device=device)


@torch.inference_mode()
def infer(net,x,mask,maxima):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial=torch.cuda.memory_allocated()
    start=time.perf_counter()
    h=embed(net,x)[mask]
    logits=net.select(h)
    score,frequency=logits.max(-1)
    t=mask.nonzero()[:,0]
    selected=[]
    for frame in range(16):
        indices=(t==frame).nonzero().flatten()
        ranked=score[indices].flatten().argsort(descending=True,stable=True)[:FRAME_K]
        selected.append(torch.stack((indices[ranked//16],ranked%16)))
    i,b=torch.cat(selected,dim=1)
    f=frequency[i,b]
    value=decode(net,h[i,b],f,maxima)
    local=np.zeros(h.shape[:-1]+(16,),np.int16)
    local[i.cpu(),b.cpu(),f.cpu()]=(value*(value.abs()>=2)).cpu()
    out=np.zeros((*p.SHAPE,16),np.int16)
    out[mask.cpu().numpy()]=local
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()-initial
    return out,h,logits,dict(head_seconds=elapsed,head_peak_extra_bytes=peak,nonzero=int(np.count_nonzero(out)),
                            frequency=np.count_nonzero(out,axis=(0,1,2,3)).tolist(),
                            per_frame_nonzero=np.count_nonzero(out,axis=(1,2,3,4)).tolist())


@torch.inference_mode()
def diagnose(net,h,logits,q,out,mask,maxima):
    target=q!=0
    valid_count=int(target.numel())
    positive=int(target.sum())
    output=out[mask.cpu().numpy()]
    actual=output!=0
    matched=actual&target.cpu().numpy()
    i,b,f=target.nonzero(as_tuple=True)
    y=decode(net,h[i,b],f,maxima)
    truth=q[i,b,f]
    rank_diag=rank(logits.cpu().numpy().ravel(),target.cpu().numpy().ravel(),positive)
    sign=float((y.sign()==truth.sign()).float().mean())
    correct=float((y==truth).float().mean())
    mae=float((y-truth).abs().float().mean())
    return dict(candidates=valid_count,positive=positive,positive_fraction=positive/valid_count,
                recall=rank_diag["recall"],precision=rank_diag["precision"],ap=rank_diag["ap"],
                realized_recall=int(matched.sum())/positive,
                realized_precision=int(matched.sum())/max(1,int(actual.sum())),
                realized_support_overlap=int(matched.sum()),realized_joint_exact=int((matched&(output==q.cpu().numpy())).sum()),
                sign_correct=sign,value_exact=correct,value_mae=mae)


def score(dest,split,row,q,diag):
    sid=row["sample_id"]
    root,source,base,fixed,refs=historical(split,row)
    anchor=refs["T1"][1]
    a_ref=decode_reference(refs["T1"][0]/"prediction.264")
    with tempfile.TemporaryDirectory(dir="/dev/shm",prefix="onpolicy-o1-") as tmp:
        path=Path(tmp)/"jm"
        start=time.perf_counter()
        state=dict(base,residual_y=to_pixels(q))
        adapter=time.perf_counter()-start
        quality,enc,frame_hash,bits=strict(path,row,state,source,fixed,q)
        stream=p.digest(path/"prediction.264")
        persist(path,dest/sid)
    drift=np.square(enc[:-1].astype(np.float32)-a_ref.astype(np.float32)).mean(axis=(1,2))
    result=dict(sample=sid,prompt=row["prompt"],split=split,policy="frame",budget=FRAME_K,
                stream_sha256=stream,frame_yuv_sha256=frame_hash,ref_vs_A_mse_by_frame=drift.tolist(),
                level_to_pixels_s=adapter,coeff_bits=int(bits[...,6:].sum()),mv_bits=int(bits[...,5].sum()),
                per_frame_coeff_bits=bits[...,6:].sum(axis=(1,2)).tolist(),
                delta_db=quality["psnr_y_p_db"]-anchor["psnr_y_p_db"],
                last4_db=p.last4(quality),delta_last4_db=p.last4(quality)-p.last4(anchor),delta_offline_3_db=0,
                rate_pct=100*(quality["bitstream_bytes"]/anchor["bitstream_bytes"]-1),
                per_frame_delta_db=[a-b for a,b in zip(quality["psnr_y_by_frame"],anchor["psnr_y_by_frame"],strict=True)],
                **quality,**diag)
    p.save_json(dest/sid/"metrics.json",result)
    return result


def train(root,out,arm,seed,fixed,locked):
    rows=p.rows_by_split()
    torch.manual_seed(seed)
    if fixed is not None:
        train_rows=[chosen_rows(rows["train"])[fixed]]
        val_rows=train_rows
        updates=locked["fixed_fit_steps"]
    else:
        train_rows=rows["train"]
        val_rows=rows["val"]
        updates=locked["train_epochs"]*len(train_rows)
    out.mkdir(exist_ok=False)
    data=[data_for(root,r,arm) for r in train_rows]
    net=p.SparseHead(67,16).cuda()
    maxima=level_bounds("cuda")
    optimizer=torch.optim.AdamW(net.parameters(),lr=0.001)
    order=torch.Generator().manual_seed(seed)
    history=[]
    losses=[]
    for i in range(updates):
        if i%len(data)==0:
            perm=torch.randperm(len(data),generator=order).tolist()
        optimizer.zero_grad(set_to_none=True)
        sample=data[perm[i%len(data)]]
        select,value,_=step(net,sample,True,False)
        (select+value).backward()
        optimizer.step()
        losses.append((float(select.detach()),float(value.detach())))
        interval=len(data) if fixed is None else 50
        if (i+1)%interval==0:
            history.append(dict(update=i+1,loss_select=mean(r[0] for r in losses),loss_level=mean(r[1] for r in losses)))
            losses.clear()
            print(arm,seed,fixed,history[-1],flush=True)
    torch.save(net.state_dict(),out/"head.pt")
    p.save_json(out/"train_history.json",history)

    @torch.inference_mode()
    def measured(sample):
        x,mask,q=sample
        levels,h,logits,sysinfo=infer(net,x,mask,maxima)
        loc,val,_=step(net,sample,True,False)
        diagnostics=diagnose(net,h,logits,q,levels,mask,maxima)
        return levels,dict(loss_select=float(loc),loss_level=float(val),**sysinfo,**diagnostics)

    net.eval()
    val_split="train" if fixed is not None else "val"
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[]
        for row in val_rows:
            levels,stats=measured(data_for(root,row,arm))
            jobs.append(pool.submit(score,out,val_split,row,levels,stats))
        points=[job.result() for job in jobs]
    train=[dict(sample=row["sample_id"],prompt=row["prompt"],**measured(sample)[1])
           for row,sample in zip(train_rows,data,strict=True)] if fixed is None else []
    result=dict(arm=arm,seed=seed,scope="full" if fixed is None else "fixed",updates=updates,
                params=sum(x.numel() for x in net.parameters()),head_sha256=p.digest(out/"head.pt"),
                train=train,points=points,summary=p.summarize(points))
    p.save_json(out/"run.json",result)
    print(result["summary"],flush=True)


def gate(root):
    doc=json.loads((root/"metrics.json").read_text())
    rows=[r for r in doc if r["split"]=="val"]
    assert len(doc)==60 and len(rows)==16
    status={}
    for arm in ARMS:
        points=[dict(r["arms"][arm],prompt=r["prompt"],sample=r["sample"],policy="frame",budget=FRAME_K,
                     delta_offline_3_db=0) for r in rows]
        status[arm]=p.summarize(points)
    t1=status["T1"]
    passed=t1["primary_per_clip_cap"] and t1["delta_db"]>0 and t1["delta_last4_db"]>0
    p.save_json(root.parent/"o0_gate.json",dict(proceed=passed,teacher_rd=status,reference_policy="fixed-prefix oracle"))
    assert passed, "T1 fixed-prefix labels did not pass O0. Stop O1."


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("labels",type=Path)
    cli.add_argument("--out",type=Path)
    cli.add_argument("--arm",choices=ARMS)
    cli.add_argument("--seed",type=int,choices=(1701,1702))
    cli.add_argument("--fixed",type=int,choices=(0,1))
    cli.add_argument("--gate",action="store_true")
    a=cli.parse_args()
    torch.set_num_threads(2)
    if a.gate:
        gate(a.labels)
    else:
        locked=json.loads((Path(__file__).resolve().parents[1]/"configs/locked.json").read_text())
        train(a.labels,a.out,a.arm,a.seed,a.fixed,locked)


if __name__=="__main__":
    main()
