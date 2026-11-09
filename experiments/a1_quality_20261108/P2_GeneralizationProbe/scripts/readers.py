"""Capacity-matched representation and continuous-target diagnostics."""
import argparse
import time

from common import *
from l0 import rank

OPT = read(HERE/"configs/readers.json")
SEED = OPT["diagnostic_seed"]

def pack(row, continuous):
    sid = row["sample_id"]
    root,_ = p.paths("train",row)
    base = p.p05.load_npz(root/"A/prediction.npz")
    legal = base["mode_id"]==1
    mv = base["mv_qpel"]
    motion = np.digitize(np.linalg.norm(mv.astype(np.float32),axis=-1),OPT["motion_qpel_cutoffs"])
    rng = np.random.default_rng(SEED + int(sid.split("_")[-1]))
    region = np.argwhere(legal)
    region = region[rng.choice(len(region),OPT["regions_per_video"],replace=False)]
    metadata = {}
    candidates = {}
    for arm in (("T1",) if continuous else ("T0",)):
        q = p.p05.load_npz(ON/"o0"/sid/f"{arm}.npz")["q"]
        items = set()
        for t,y,x,b,f in np.argwhere(q!=0):
            items.add((t,y,x,b,f))
            eligible = np.argwhere(legal & (motion==motion[t,y,x]) & (q[...,b,f]==0))
            for u,v,w in eligible[rng.choice(len(eligible),min(len(eligible),OPT["matched_negatives_per_positive"]),replace=False)]:
                items.add((u,v,w,b,f))
        items = np.asarray(sorted(items))
        loc = tuple(items[:,:3].T)
        b,f = items[:,3:].T
        candidates[arm] = items[:,:3]
        metadata[arm] = dict(q=q[(*loc,b,f)], group=(b*16+f)*4+motion[loc],
            coords=items, support=q[loc]!=0)
    allcoords = np.concatenate([region,*candidates.values()])
    t,y,x = allcoords.T
    t4 = t//4+1
    hidden = p.input_tensor(row,"hidden",device="cpu")
    h = hidden.reshape(5,30,52,1536)[t4,y,x].float().numpy()
    latent = p.input_tensor(row,device="cpu")[0].numpy()
    l = latent.reshape(16,5,30,2,52,2).transpose(1,2,4,0,3,5)[t4,y,x].reshape(len(t),64)
    f0 = torch.load(CACHE/"features"/f"{sid}.pt",weights_only=True)[0,:64].movedim(0,-1).numpy()[t,y,x]
    side = np.column_stack([mv[t,y,x]/32,(t%4)/3]).astype(np.float32)
    n = len(region)
    record = dict(features={"F0":f0,"F1":h,"F2":l},side=side,region_n=n,region_motion=motion[tuple(region.T)],
                  candidate=metadata,region_coords=region)
    if continuous:
        with np.load(OUT/"continuous"/f"{sid}.npz") as z:
            index = tuple(region.T)
            record.update(energy=z["energy"][index],low=z["low"][index],quantized=z["quantized"][index])
    return record

def design(raw, train_ids, arm):
    n = raw[train_ids[0]]["region_n"]
    fit = np.concatenate([raw[s]["features"][arm][:n] for s in train_ids])
    center = fit.mean(0)
    fit -= center
    cov = fit.T @ fit
    _,vec = np.linalg.eigh(cov)
    rotation = vec[:,-OPT["pca_components"]:]
    scale = (fit@rotation).std(0).clip(1e-5)
    return {s:np.column_stack([((r["features"][arm]-center)@rotation)/scale,r["side"]])
            for s,r in raw.items()}

def ridge(x,y,w):
    x = np.column_stack([x,np.ones(len(x))])
    gram = x.T@(x*w[:,None])
    gram += OPT["ridge_penalty"]*np.eye(gram.shape[0])
    return np.linalg.solve(gram,x.T@(y*w.reshape((-1,)+(1,)*(y.ndim-1))))

def predict(x,beta):
    return x@beta[:-1]+beta[-1]

def normalize(x):
    return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8)

def cat(raw,ids,key):
    return np.concatenate([raw[s][key] for s in ids])

def candidate_fit(raw,z,ids,arm):
    keys = ("q","group","support")
    train = {k:np.concatenate([raw[s]["candidate"][arm][k] for s in ids]) for k in keys}
    train["x"] = np.concatenate([z[s][raw[s]["region_n"]:] for s in ids])
    x,y,g = train["x"],train["q"]!=0,train["group"]
    positives = np.bincount(g[y],minlength=1024)
    counts = np.bincount(g,minlength=1024)
    prior = (positives+1)/(counts+2)
    x = np.column_stack([x,prior[g]])
    weights = np.where(y,.5/y.mean(),.5/(~y).mean())
    train.update(prior=prior,beta=ridge(x,y,weights),ids=np.concatenate([
        np.repeat(s,len(raw[s]["candidate"][arm]["q"])) for s in ids]))
    return train

def nullable_mean(x):
    return float(np.mean(x)) if len(x) else None

def sparse_score(raw,z,s,arm,tr,rng):
    val = raw[s]["candidate"][arm]
    x = z[s][raw[s]["region_n"]:]
    prior = tr["prior"][val["group"]]
    linear = predict(np.column_stack([x,prior]),tr["beta"])
    pred = np.zeros(len(x))
    top = np.full(len(x),-1)
    rand = top.copy()
    sim = np.zeros(len(x))
    for g in np.unique(val["group"]):
        idx = np.flatnonzero(val["group"]==g)
        ref = np.flatnonzero(tr["group"]==g)
        if not len(ref):
            pred[idx]=prior[idx]
            continue
        cos = normalize(x[idx]) @ normalize(tr["x"][ref]).T
        best = np.argsort(-cos,axis=1,kind="stable")[:,:5]
        nearest = ref[best]
        pred[idx] = (tr["q"][nearest]!=0).mean(1)
        top[idx] = nearest[:,0]
        sim[idx] = cos[np.arange(len(idx)),best[:,0]]
        rand[idx] = rng.choice(ref,len(idx))
    y = val["q"]!=0
    coverage = top>=0
    base = dict(n=len(x),positive=int(y.sum()),sampled_opportunity=float(y.mean()),coverage=float(coverage.mean()))
    for tag,score in (("knn",pred),("linear",linear),("prior",prior)):
        base[tag] = rank(score,y,int(y.sum()))
    details = []
    for tag,nn in (("knn",top),("random",rand)):
        sel = nn>=0
        q,n = val["q"][sel],tr["q"][nn[sel]]
        both = (q!=0)&(n!=0)
        mask1,mask2 = val["support"][sel],tr["support"][nn[sel]]
        union = (mask1|mask2).sum((1,2))
        base[tag+"_pair"] = dict(agreement=nullable_mean((q!=0)==(n!=0)),
            pos_recall=nullable_mean(n[q!=0]!=0), precision=nullable_mean(q[n!=0]!=0),
            sign=nullable_mean(np.sign(q[both])==np.sign(n[both])),
            level_mae=nullable_mean(np.abs(q[both]-n[both])), both_positive=int(both.sum()),
            support_iou=nullable_mean((mask1&mask2).sum((1,2))[union>0]/union[union>0]))
        for i,j in zip(np.flatnonzero(sel),nn[sel],strict=True):
            details.append(dict(kind=tag,coords=val["coords"][i].tolist(), neighbor_video=str(tr["ids"][j]),
                score=float(sim[i]) if tag=="knn" else None, q=int(val["q"][i]), neighbor_q=int(tr["q"][j])))
    return base, details

def corr(x,y):
    if np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(x,y)[0,1])

def ranks(x):
    values,inverse,counts = np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2)[inverse]

def continuous_fit(raw,z,ids):
    x = np.concatenate([z[s][:raw[s]["region_n"]] for s in ids])
    y = np.log1p(cat(raw,ids,"energy"))
    q = cat(raw,ids,"low").reshape(len(x),48).astype(float)
    motion = cat(raw,ids,"region_motion")
    w = np.ones(len(x))
    return dict(x=x, y=y, q=q, motion=motion, y_fit=ridge(x,y,w), q_fit=ridge(x,q,w),
        ids=np.concatenate([np.repeat(s,raw[s]["region_n"]) for s in ids]))

def continuous_score(raw,z,s,tr,rng):
    n = raw[s]["region_n"]
    x = z[s][:n]
    energy = raw[s]["energy"]
    low = raw[s]["low"].reshape(n,48).astype(float)
    true_q = raw[s]["quantized"].reshape(n,48)
    motion = raw[s]["region_motion"]
    match = np.zeros(n,dtype=int)
    random = match.copy()
    sim = np.zeros(n)
    energy_prior = np.zeros(n)
    q_prior = np.zeros_like(low)
    for m in np.unique(motion):
        idx = np.flatnonzero(motion==m)
        ref = np.flatnonzero(tr["motion"]==m)
        if not len(ref):
            ref = np.arange(len(tr["x"]))
        cosine = normalize(x[idx])@normalize(tr["x"][ref]).T
        near = cosine.argmax(1)
        match[idx] = ref[near]
        sim[idx] = cosine[np.arange(len(idx)),near]
        random[idx] = rng.choice(ref,len(idx))
        energy_prior[idx] = np.median(tr["y"][ref])
        q_prior[idx] = np.median(tr["q"][ref],axis=0)
    result = {}
    for arm,y,q in (("linear",predict(x,tr["y_fit"]),predict(x,tr["q_fit"])),
                   ("knn",tr["y"][match],tr["q"][match]),
                   ("random",tr["y"][random],tr["q"][random]),
                   ("prior",energy_prior,q_prior),("zero",np.zeros(n),np.zeros_like(low))):
        pred = np.expm1(y.clip(0,np.log1p(255**2)))
        k = int(n*OPT["top_region_fraction"])
        best = np.argsort(-energy,kind="stable")[:k]
        chosen = np.argsort(-pred,kind="stable")[:k]
        # Identical QP28 rounding from p.quantize, on fractional estimates rounded to integer.
        scale = np.asarray(p.SCALE).reshape(-1)[[0,1,4]]
        qr = np.rint(q).astype(np.int64).reshape(n,16,3)
        level = (((np.abs(qr)*scale+(342<<8))>>19)*np.sign(qr)).reshape(n,48)
        nz = np.abs(low)>=np.tile(OPT["near_zero_transform_abs"],16)
        result[arm] = dict(energy_mae=float(np.abs(energy-pred).mean()),pearson=corr(energy,pred),
            spearman=corr(ranks(energy),ranks(pred)),top_q_recall=len(np.intersect1d(best,chosen))/k,
            log_energy_mae=float(np.abs(np.log1p(energy)-y).mean()),
            low_mae_by_frequency=np.abs(low-q).reshape(n,16,3).mean((0,1)).tolist(),
            sign_outside_near_zero=nullable_mean(np.sign(low[nz])==np.sign(q[nz])),
            near_zero_fraction=float((~nz).mean()),quantized_nonzero_disagree=float(((level!=0)!=(true_q!=0)).mean()),
            quantized_nonzero_recall=nullable_mean(level[true_q!=0]!=0),
            quantized_threshold_sensitivity=float(np.mean(
                ((((np.abs(qr)+1)*scale+(342<<8))>>19)>0)!=
                ((((np.abs(qr)-1).clip(0)*scale+(342<<8))>>19)>0))))
    energy_delta = np.abs(np.log1p(energy)-tr["y"][match])
    random_delta = np.abs(np.log1p(energy)-tr["y"][random])
    result["distance_bins"] = [dict(lo=lo,hi=hi,n=int(sel.sum()),
        energy_log_gap=nullable_mean(energy_delta[sel]),random_log_gap=nullable_mean(random_delta[sel]))
        for lo,hi in ((-1,.25),(.25,.5),(.5,.75),(.75,.9),(.9,1.01))
        for sel in [(sim>=lo)&(sim<hi)]]
    worst = np.argsort(-energy_delta)[:5]
    result["examples"] = [dict(coords=raw[s]["region_coords"][i].tolist(),cosine=float(sim[i]),
        neighbor=str(tr["ids"][match[i]]),energy=float(energy[i]),neighbor_energy=float(np.expm1(tr["y"][match[i]]))) for i in worst]
    return result

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("gz1","gz2"))
    args = cli.parse_args()
    torch.set_num_threads(2)
    stage = args.stage
    continuous = stage=="gz2"
    arm = "T1" if continuous else "T0"
    assert len(list(OUT.glob("gz0-*-*-*/metrics.json"))) == 8
    if continuous:
        assert (OUT/"gz1.json").is_file() and (OUT/"gz2_labels.json").is_file()
    manifest = read(OUT/"split_manifest.json")
    start = time.monotonic()
    raw = {r["sample_id"]:pack(r,continuous) for r in p.rows_by_split()["train"]}
    result = []
    for fi,fold in enumerate(manifest["folds"]):
        ids = fold["A"]
        for feature in ("F0","F1","F2"):
            z = design(raw,ids,feature)
            tr = candidate_fit(raw,z,ids,arm)
            reg = continuous_fit(raw,z,ids) if continuous else None
            scores = {}
            folder = OUT/f"{stage}_pairs"
            folder.mkdir(exist_ok=True)
            for index,s in enumerate(fold["B"]+fold["D"]):
                rng = np.random.default_rng(SEED+index)
                sparse,details = sparse_score(raw,z,s,arm,tr,rng)
                p.save_json(folder/f"{fi}-{feature}-{s}.json",details)
                scores[s] = dict(sparse=sparse)
                if continuous:
                    scores[s]["continuous"] = continuous_score(raw,z,s,reg,rng)
            result.append(dict(fold=fi,feature=feature,aligned_dimension=raw[ids[0]]["features"][feature].shape[1],
                common_dimension=z[ids[0]].shape[1],scores=scores))
            print(stage,fi,feature,round(time.monotonic()-start,2),flush=True)
    p.save_json(OUT/f"{stage}.json",dict(seconds=time.monotonic()-start,results=result))
if __name__=="__main__":
    main()
