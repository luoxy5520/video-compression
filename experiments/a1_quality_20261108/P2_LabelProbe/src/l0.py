"""Matched loss × frequency ablation with strict JM replay."""
import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2_InfoProbe/scripts"))
from d0 import p
from repair import embed, to_pixels

HIST = p.ROOT / "P2_Lite/run01"
ARMS = dict(A=(False, [0,1,4]), B=(True, [0,1,4]), C=(False, list(range(16))), D=(True, list(range(16))))
UPDATES = 400


def chosen_rows(rows):
    unique = {}
    for row in rows:
        unique.setdefault(row["prompt"], row)
    return list(unique.values())[:8]


def prep(root, audit):
    assert (audit / "audit.json").exists()
    root.mkdir(exist_ok=False)
    rows = p.rows_by_split()
    manifest = []
    for split, group in (("train", chosen_rows(rows["train"])), ("val", rows["val"])):
        for row in group:
            folder, source = p.paths(split,row)
            meta = {r["arm"]:r for r in json.loads((folder/"summary.json").read_text())}
            teacher = folder / p.CFG["teacher"]
            if not teacher.exists():
                teacher = folder / meta[p.CFG["teacher"]]["fallback"]
            state = p.p05.load_npz(teacher/"prediction.npz")
            sid = row["sample_id"]
            fixed = p.p05.load_npz(folder/"fixed_mode_mv.npz")
            with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="l0label-") as tmp:
                case = Path(tmp)/"replay"
                quality, levels, mv, bits = p.strict(case, row, state, source)
                assert p.digest(case/"prediction.264") == meta[p.CFG["teacher"]]["stream_sha256"]
                np.testing.assert_array_equal(mv, fixed["native_mv_qpel"].reshape(16,1560,2))
                np.testing.assert_array_equal(bits[...,2], fixed["mode_id"].reshape(16,1560))
            old = p.p05.load_npz(HIST/"labels"/f"{sid}.npz")
            np.testing.assert_array_equal(old["levels"],levels[..., [0,1,4]])
            np.testing.assert_array_equal(old["mode"],state["mode_id"])
            path = root / f"{sid}.npz"
            np.savez_compressed(path, q=levels, mode=state["mode_id"])
            manifest.append(dict(sample=sid, split=split, prompt=row["prompt"],
                label_sha256=p.digest(path), cached_feature_sha256=p.digest(HIST/"features"/f"{sid}.pt"),
                teacher_stream_sha256=meta[p.CFG["teacher"]]["stream_sha256"]))
            print(sid, np.count_nonzero(levels), flush=True)
    p.save_json(root/"manifest.json",manifest)


def load(root,row,freq):
    sid = row["sample_id"]
    s = p.p05.load_npz(root/f"{sid}.npz")
    x = torch.load(HIST/"features"/f"{sid}.pt",weights_only=True).cuda()
    mode = torch.from_numpy(s["mode"]==1).cuda()
    q = torch.from_numpy(s["q"][...,freq][s["mode"]==1]).long().cuda()
    return x, mode, q


def value_logits(net, h, f):
    w = net.value.weight.reshape(net.frequencies,65,32)[f]
    bias = net.value.bias.reshape(net.frequencies,65)[f]
    return torch.bmm(w,h.unsqueeze(-1)).squeeze(-1)+bias


def decode(net,h,f,max_q):
    z = value_logits(net,h,f)
    bins = torch.arange(64,device=h.device)
    value = z[:,1:].masked_fill(bins>=max_q[f,None],-torch.inf).argmax(1)+1
    return value*torch.where(z[:,0]>0,1,-1)


def step(net, sample, hard, grad):
    x, mode, q = sample
    h = embed(net,x)[mode]
    logits = net.select(h)
    active = q != 0
    i,b,f = active.nonzero(as_tuple=True)
    target = q[i,b,f]
    if hard:
        neg = logits.masked_fill(active,-torch.inf).flatten().topk(len(i)).values
        pos = logits[active]
        loc = F.softplus(-pos).mean()+F.softplus(neg).mean()
        grads = [float(v.detach()) for v in ((1-pos.sigmoid()).norm()/len(i), neg.sigmoid().norm()/len(i))] if grad else []
    else:
        loc = F.binary_cross_entropy_with_logits(logits, active.float(),pos_weight=logits.new_tensor(24))
        grads = [float(v.detach()) for v in (24*(1-logits[active].sigmoid()).norm()/logits.numel(),
                 logits[~active].sigmoid().norm()/logits.numel())] if grad else []
    val = value_logits(net,h[i,b],f)
    value = F.binary_cross_entropy_with_logits(val[:,0],(target>0).float()) + F.cross_entropy(val[:,1:],target.abs()-1)
    return loc,value,grads


def rank(score,target,k):
    order = np.argsort(-score,kind="stable")
    hits = target[order]
    n = int(target.sum())
    tp = int(hits[:k].sum())
    ap = float((np.cumsum(hits)[hits]/(np.flatnonzero(hits)+1)).mean()) if n else None
    return dict(candidates=len(target), positives=n,k=k,hits=tp, precision=tp/k if k else None,
                recall=tp/n if n else None,ap=ap,threshold=float(score[order[k-1]]) if k else None)


@torch.inference_mode()
def predict(net,sample,freq,budget,diagnostics=True):
    x,mask,q = sample
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    initial=torch.cuda.memory_allocated()
    h = embed(net,x)[mask]
    logits = net.select(h)
    max_q=torch.tensor([p.max_level(f) for f in freq],device=x.device)
    score, best = logits.max(-1)
    order = score.flatten().argsort(descending=True,stable=True)[:budget]
    i,b = order//16,order%16
    f = best[i,b]
    value=decode(net,h[i,b],f,max_q)
    out=np.zeros((*p.SHAPE,16),np.int16)
    local=np.zeros(q.shape[:-1]+(16,),np.int16)
    valid=value.abs()>=2
    local[i.cpu(),b.cpu(),np.asarray(freq)[f.cpu()]]=(value*valid).cpu()
    out[mask.cpu().numpy()]=local
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()-initial
    if not diagnostics:
        return out, {}
    truth=q.cpu().numpy()
    s=logits.cpu().numpy()
    i,b,f=(q!=0).nonzero(as_tuple=True)
    known=q[i,b,f].cpu().numpy()
    y=decode(net,h[i,b],f,max_q).cpu().numpy()
    dc=f.cpu().numpy()==0
    dc_rank=rank(s[...,0].ravel(),truth[...,0].ravel()!=0, int(np.count_nonzero(truth[...,0])))
    per_freq={str(fr):rank(s[...,j].ravel(),truth[...,j].ravel()!=0,int(np.count_nonzero(truth[...,j])))
              for j,fr in enumerate(freq)}
    all_rank=rank(s.ravel(),truth.ravel()!=0,int(np.count_nonzero(truth)))
    report=dict(dc=dc_rank,per_freq=per_freq,all=all_rank,
        sign_correct=float((np.sign(y)==np.sign(known)).mean()),value_exact=float((y==known).mean()),
        value_mae=float(np.abs(y-known).mean()),dc_sign_correct=float((np.sign(y[dc])==np.sign(known[dc])).mean()),
        dc_value_exact=float((y[dc]==known[dc]).mean()),dc_mae=float(np.abs(y[dc]-known[dc]).mean()),
        nonzero=int(np.count_nonzero(out)),logit_quantiles=np.quantile(s,[0,.01,.5,.99,1]).tolist(),
        emitted_level_hist={str(k):int(v) for k,v in zip(*np.unique(out[out!=0],return_counts=True))},
        head_seconds=elapsed, head_peak_extra_bytes=peak)
    return out, report


def replay(dest, split, row, levels, extra):
    root, source = p.paths(split,row)
    sid=row["sample_id"]
    base = p.p05.load_npz(root/"A/prediction.npz")
    fixed=p.p05.load_npz(root/"fixed_mode_mv.npz")
    anchor=next(m for m in json.loads((root/"summary.json").read_text()) if m["arm"]=="A")
    with tempfile.TemporaryDirectory(dir="/dev/shm",prefix="labelreplay-") as tmp:
        folder=Path(tmp)/"jm"
        quality,actual,mv,bits=p.strict(folder,row,dict(base,residual_y=to_pixels(levels)),source)
        np.testing.assert_array_equal(actual,levels)
        np.testing.assert_array_equal(mv,fixed["native_mv_qpel"].reshape(16,1560,2))
        np.testing.assert_array_equal(bits[...,2],fixed["mode_id"].reshape(16,1560))
        case=dest/sid
        p.p05.preserve(folder,case)
        for f in ("bits.csv","coeff.bin"):
            (case/f).write_bytes((folder/f).read_bytes())
    out=dict(sample=sid,prompt=row["prompt"],split=split,policy="video",
             delta_db=quality["psnr_y_p_db"]-anchor["psnr_y_p_db"],last4_db=p.last4(quality),
             delta_last4_db=p.last4(quality)-p.last4(anchor),delta_offline_3_db=0,
             rate_pct=100*(quality["bitstream_bytes"]/anchor["bitstream_bytes"]-1),
             coeff_bits=int(bits[...,6:].sum()),stream_sha256=p.digest(case/"prediction.264"),**quality,**extra)
    p.save_json(case/"metrics.json",out)
    return out


def fit(root,out,arm,seed,scope):
    torch.manual_seed(seed)
    hard,freq=ARMS[arm]
    rows=p.rows_by_split()
    group=chosen_rows(rows["train"])
    fit_rows=group[:8] if scope=="multi" else [group[int(scope)]]
    eval_rows=rows["val"] if scope=="multi" else fit_rows
    samples=[load(root,row,freq) for row in fit_rows]
    net=p.SparseHead(samples[0][0].shape[1],len(freq)).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=0.001)
    order=torch.Generator().manual_seed(seed)
    out.mkdir(exist_ok=False)
    hist=[]
    for i in range(UPDATES):
        if i%len(samples)==0:
            permutation=torch.randperm(len(samples),generator=order).tolist()
        sample=samples[permutation[i%len(samples)]]
        opt.zero_grad(set_to_none=True)
        logged=i==0 or (i+1)%50==0
        loc,value,g=step(net,sample,hard,logged)
        (loc+value).backward()
        opt.step()
        if logged:
            hist.append(dict(update=i+1,loc=float(loc.detach()),value=float(value.detach()),logit_grad_pos_neg=g))
        if i==199 and scope!="multi":
            _,snapshot=predict(net,sample,freq,64)
            p.save_json(out/"step200.json",snapshot)
    torch.save(net.state_dict(),out/"head.pt")
    p.save_json(out/"train_history.json",hist)
    train_diag=[dict(sample=row["sample_id"],**predict(net,sample,freq,64)[1])
                for row,sample in zip(fit_rows,samples,strict=True)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[]
        split="val" if scope=="multi" else "train"
        for row in eval_rows:
            sample=load(root,row,freq)
            # A common position budget across all arms. Fixed memory replay
            # uses the full teacher support count, including for low-band arms.
            budget=64 if scope=="multi" else int(np.count_nonzero(p.p05.load_npz(root/f"{row['sample_id']}.npz")["q"]))
            levels,diag=predict(net,sample,freq,budget)
            jobs.append(pool.submit(replay,out,split,row,levels,dict(diag,budget=budget)))
        points=[j.result() for j in jobs]
    p.save_json(out/"run.json",dict(arm=arm,seed=seed,scope=scope,train=train_diag,points=points,
                summary=p.summarize(points),params=sum(t.numel() for t in net.parameters())))
    print(arm,seed,scope,p.summarize(points),flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("--audit",type=Path)
    cli.add_argument("--out",type=Path)
    cli.add_argument("--arm",choices=tuple(ARMS))
    cli.add_argument("--seed",type=int)
    cli.add_argument("--scope",choices=("0","1","multi"))
    a=cli.parse_args()
    torch.set_num_threads(2)
    if a.audit:
        prep(a.root,a.audit)
    else:
        fit(a.root,a.out,a.arm,a.seed,a.scope)


if __name__=="__main__":
    main()
