"""Train-only, leave-video-out matched neighbor label diagnostics."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from train_capacity import HIST,ON,p,sample_rows
from m1 import HIST as CACHE


def motion_bins(mv):
    return np.digitize(np.linalg.norm(mv.astype(np.float32),axis=-1),[1,8,32])


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    a=cli.parse_args()
    rng=np.random.default_rng(61109)
    rows=p.rows_by_split()
    first=sample_rows(rows["train"],8)
    seen={r["sample_id"] for r in first}
    extras=[r for r in rows["train"] if r["sample_id"] not in seen][:4]
    selected=first+extras
    o0={x["sample"]:x for x in json.loads((ON/"o0/metrics.json").read_text())}
    g={x["sample"]:x for x in json.loads((p.ROOT/"P2_InfoProbe/g_labels/manifest.json").read_text())}
    hist=json.loads((HIST/"scale-8-1701/run.json").read_text())
    assert [r["sample_id"] for r in first]==[r["sample"] for r in hist["snapshots"]["3200"]["train"]]
    contract=[]
    data=[]
    vectors=[]
    supports=[]
    for row in rows["train"]+rows["val"]:
        sid=row["sample_id"]
        item=o0[sid]
        feature_path=CACHE/"features"/f"{sid}.pt"
        label_path=ON/"o0"/sid/"T0.npz"
        assert p.digest(feature_path)==g[sid]["feature_sha256"]
        assert p.digest(label_path)==item["arms"]["T0"]["label_sha256"]
        contract.append(dict(sample=sid,split=item["split"],prompt=row["prompt"],video_seed=row["seed"],
            selected_C0=sid in seen,feature_sha256=g[sid]["feature_sha256"],label_sha256=item["arms"]["T0"]["label_sha256"],
            reference_sha256=item["arms"]["T0"]["source_reference_stream_sha256"]))
        if row not in selected:
            continue
        root,source=p.paths("train",row)
        base=p.p05.load_npz(root/"A/prediction.npz")
        q=p.p05.load_npz(label_path)["q"]
        mode=base["mode_id"]==1
        mv=motion_bins(base["mv_qpel"])
        y=p.p05.read_yuv(source)[0][1:].reshape(16,30,16,52,16).transpose(0,1,3,2,4)
        texture=y.var(axis=(-1,-2))
        x=torch.load(feature_path,weights_only=True)[0].movedim(0,-1).numpy()
        for t,j,k,b,f in np.argwhere(q!=0):
            locations=np.argwhere(mode & (mv==mv[t,j,k]) & (q[...,b,f]==0))
            if len(locations)<2:
                locations=np.argwhere(mode & (q[...,b,f]==0))
            idx=rng.choice(len(locations),size=2,replace=False)
            for ti,ji,ki in [(t,j,k),*locations[idx]]:
                data.append(dict(sample=sid,motion=int(mv[ti,ji,ki]),block=int(b),frequency=int(f),
                    level=int(q[ti,ji,ki,b,f]),texture=float(texture[ti,ji,ki]),frame=int(ti+1),
                    mb_y=int(ji),mb_x=int(ki),extra=sid not in seen))
                vectors.append(x[ti,ji,ki])
                supports.append(q[ti,ji,ki]!=0)
    Z=np.stack(vectors).astype(np.float32)
    Z/=np.linalg.norm(Z,axis=1,keepdims=True).clip(1e-8)
    sim=Z@Z.T
    supports=np.stack(supports)
    samples=np.array([r["sample"] for r in data])
    f=np.array([r["frequency"] for r in data])
    b=np.array([r["block"] for r in data])
    m=np.array([r["motion"] for r in data])
    l=np.array([r["level"] for r in data])
    records=[]
    for i,row in enumerate(data):
        exact=(f==f[i])&(b==b[i])&(m==m[i])
        local=(samples==samples[i])&exact
        local &= ~np.array([(r["frame"],r["mb_y"],r["mb_x"])==(row["frame"],row["mb_y"],row["mb_x"]) for r in data])
        other=(samples!=samples[i])&exact
        fallback=False
        if not other.any():
            other=(samples!=samples[i])&(f==f[i])&(b==b[i])
            fallback=True
        if not other.any():
            continue
        index=np.flatnonzero(other)
        best=index[np.argsort(-sim[i,index],kind="stable")[:5]]
        random=rng.choice(index,size=len(best),replace=False)
        # Permute the binary labels and levels within the same matching stratum.
        shuffled=rng.permutation(l[index])
        shuffled_lookup=dict(zip(index,shuffled,strict=True))
        controls={"cross_knn":best,"matched_random":random,"shuffled_labels":best}
        if local.any():
            ind=np.flatnonzero(local)
            controls["same_video"]=ind[np.argsort(-sim[i,ind],kind="stable")[:5]]
        for tag,neighbors in controls.items():
            for rank,j in enumerate(neighbors,1):
                level=int(shuffled_lookup[j]) if tag=="shuffled_labels" else int(l[j])
                union=(supports[i]|supports[j]).sum()
                both=l[i]!=0 and level!=0
                records.append(dict(query=i,neighbor=int(j),rank=rank,kind=tag,similarity=float(sim[i,j]),
                    matched_motion=not fallback,source_video=samples[i],target_video=samples[j],extra=row["extra"],
                    frequency=row["frequency"],block=row["block"],motion=row["motion"],texture=row["texture"],
                    target_texture=data[j]["texture"],qpos=int(l[i]!=0),npos=int(level!=0),agreement=int((l[i]!=0)==(level!=0)),
                    sign_equal=int(np.sign(l[i])==np.sign(level)) if both else None,
                    level_mae=int(abs(l[i]-level)) if both else None,
                    mb_support_iou=float((supports[i]&supports[j]).sum()/union) if union and tag!="shuffled_labels" else None))
    result={}
    for tag in ("cross_knn","matched_random","shuffled_labels","same_video"):
        rr=[r for r in records if r["kind"]==tag and r["rank"]==1 and r["matched_motion"]]
        result[tag]=dict(n=len(rr),agreement=mean(r["agreement"] for r in rr),
                         pos_retrieval=mean(r["npos"] for r in rr if r["qpos"]),
                         mean_cosine=mean(r["similarity"] for r in rr))
    pairs={}
    for r in records:
        if r["matched_motion"] and r["rank"]==1 and r["kind"] in ("cross_knn","matched_random"):
            pairs.setdefault(r["query"],{})[r["kind"]]=r
    delta=[]
    for sid in dict.fromkeys(samples):
        diff=[v["cross_knn"]["agreement"]-v["matched_random"]["agreement"] for v in pairs.values()
              if v["cross_knn"]["source_video"]==sid]
        if diff: delta.append(mean(diff))
    ci=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
    confusion=[r for r in records if r["kind"]=="cross_knn" and r["rank"]==1 and r["matched_motion"] and r["similarity"]>=.99 and not r["agreement"]]
    mismatch={str(motion):dict(n=sum(r["motion"]==motion for r in confusion),
             median_texture_gap=float(np.median([abs(r["texture"]-r["target_texture"]) for r in confusion if r["motion"]==motion])) if any(r["motion"]==motion for r in confusion) else None)
             for motion in range(4)}
    data=dict(videos=[r["sample_id"] for r in selected],features="frozen 67-channel macroblock input",
              candidates=len(samples),exact_top1=result,paired_video_macro_diff=mean(delta),bootstrap95=ci,
              high_similarity_mismatch=mismatch)
    p.save_json(a.root/"c2_summary.json",data)
    p.save_json(a.root/"feature_label_contract.json",contract)
    with (a.root/"c2_neighbors.csv").open("w") as file:
        w=csv.DictWriter(file,fieldnames=list(records[0]),lineterminator="\n")
        w.writeheader(); w.writerows(records)
    print(data,flush=True)


if __name__=="__main__":
    main()
