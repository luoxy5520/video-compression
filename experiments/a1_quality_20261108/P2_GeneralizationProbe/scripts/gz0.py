"""Run the locked support/query comparison using the existing sparse heads."""
import argparse
import time
from statistics import mean

from common import *
from o1 import infer, level_bounds
from l0 import rank

def make_prior(samples):
    shape = (*p.SHAPE,16)
    positives = torch.zeros(shape, device="cuda")
    opportunities = torch.zeros(p.SHAPE[:3], device="cuda")
    for _,mask,q in samples:
        positives[mask] += q != 0
        opportunities += mask
    return (positives + .001) / (opportunities[...,None,None] + 1)

def frame_hits(score,q,t):
    counts = []
    for frame in range(16):
        use = t == frame
        score4,freq = score[use].max(-1)
        idx = score4.flatten().argsort(descending=True,stable=True)[:4]
        hit = q[use][idx//16,idx%16,freq.flatten()[idx]] != 0
        counts.append(int(hit.sum()))
    return counts

@torch.inference_mode()
def evaluate(net, sid, sample, prior):
    x,mask,q = sample
    out,_,score,cost = infer(net,x,mask,level_bounds("cuda"))
    truth = q != 0
    actual = torch.from_numpy(out).cuda()[mask]
    hit = (actual != 0) & truth
    t = mask.nonzero()[:,0]
    q_flat = truth.flatten().cpu().numpy()
    model = rank(score.flatten().cpu().numpy(),q_flat,64)
    prior_scores = prior[mask]
    base = rank(prior_scores.flatten().cpu().numpy(),q_flat,64)
    n = len(q_flat)
    harmonic = np.log(n) + np.euler_gamma + 0.5/n
    random_ap = harmonic/n + (63/(n-1))*(1-harmonic/n)
    positive = int(hit.sum())
    return dict(video_id=sid, candidates=n, positives=64, chance=64/n,
        random_ap=random_ap, random_recall=mean(4/(int((t==f).sum())*256) for f in range(16)),
        ap=model["ap"], prior_ap=base["ap"], top64=model["recall"],
        hits=positive, recall=positive/64, precision=positive/max(cost["nonzero"],1),
        pre_filter_hits=frame_hits(score,q,t), prior_hits=frame_hits(prior_scores,q,t),
        frame_hits=[int(hit[t==i].sum()) for i in range(16)],
        frequency_hits=hit.sum((0,1)).tolist(), frequency_positives=truth.sum((0,1)).tolist(),
        position_hits=hit.sum((0,2)).tolist(), position_positives=truth.sum((0,2)).tolist(),
        support_iou=positive/(64+cost["nonzero"]-positive),
        joint_sign_agreement=float((actual[hit].sign()==q[hit].sign()).float().mean()) if positive else None,
        joint_level_mae=float((actual[hit]-q[hit]).abs().float().mean()) if positive else None, **cost)

def summarize(rows):
    def avg(k):
        return mean(r[k] for r in rows)
    return dict(n=len(rows), **{k:avg(k) for k in
        ("recall","precision","ap","prior_ap","random_ap","random_recall","chance","head_seconds","head_peak_extra_bytes")},
        prior_recall=mean(sum(r["prior_hits"])/64 for r in rows),
        recall_ci=bootstrap([r["recall"] for r in rows]),
        recall_delta_prior_ci=bootstrap([r["recall"]-sum(r["prior_hits"])/64 for r in rows]),
        recall_delta_random_ci=bootstrap([r["recall"]-r["random_recall"] for r in rows]),
        ap_delta_prior_ci=bootstrap([r["ap"]-r["prior_ap"] for r in rows]),
        ap_delta_random_ci=bootstrap([r["ap"]-r["random_ap"] for r in rows]))

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("seed",type=int,choices=CFG["seeds"])
    args = cli.parse_args()
    torch.set_num_threads(2)
    split = read(OUT/"split_manifest.json")
    history = read(OUT/"pre_history.json")
    assert len(history["a_streams"]) == 16 and all(history["m1_units"].values())
    rows = {r["sample_id"]:r for r in p.rows_by_split()["train"]}
    samples = {sid:data_for(ON/"o0",r,"T0") for sid,r in rows.items()}
    seed = args.seed
    for fold,groups in enumerate(split["folds"]):
        support = [samples[sid] for sid in groups["A"]]
        prior = make_prior(support)
        for arm,width in (("A",64),("C1",1504)):
            torch.manual_seed(seed)
            order = torch.Generator().manual_seed(seed)
            neg = torch.Generator(device="cuda").manual_seed(seed+1000)
            net = build(width)
            optimizer = torch.optim.AdamW(net.parameters(),lr=.001)
            folder = OUT/f"gz0-{fold}-{arm}-{seed}"
            folder.mkdir()
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(CFG["epochs"]):
                for i in torch.randperm(len(support),generator=order):
                    optimizer.zero_grad(set_to_none=True)
                    loss,_ = training(net,support[i],"hard",collect=False) if arm=="A" else pairwise(net,support[i],neg)
                    loss.backward()
                    optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-start
            peak = torch.cuda.max_memory_allocated()
            ck = folder/"head.pt"
            torch.save(net.state_dict(),ck)
            net.eval()
            points = {sid:evaluate(net,sid,samples[sid],prior) for sid in rows}
            groups_summary = {g:summarize([points[s] for s in ids]) for g,ids in groups.items() if ids}
            p.save_json(folder/"metrics.json",dict(seed=seed,fold=fold,arm=arm,elapsed_s=elapsed,
                training_peak_bytes=peak,checkpoint_sha256=p.digest(ck),params=sum(x.numel() for x in net.parameters()),
                groups=groups_summary,points=points))
            print(fold,arm,seed,{g:(v["recall"],v["ap"]) for g,v in groups_summary.items()},flush=True)

if __name__=="__main__":
    main()
