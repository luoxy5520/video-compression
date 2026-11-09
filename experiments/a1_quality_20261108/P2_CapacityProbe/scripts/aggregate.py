"""Aggregate capacity runs, matched neighbors and all prespecified strict replays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean,median

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"]="p2-capacityprobe"
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def write(path,rows,sep=","):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter=sep,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def plot(fig,path):
    fig.savefig(path,metadata={"Date":None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("out",type=Path)
    a=cli.parse_args()
    root,out=a.root,a.out
    out.mkdir(exist_ok=True)
    figs=out/"figures"
    figs.mkdir(exist_ok=True)
    baseline=read(root/"audit/audit.json")
    hashes=[]
    for p in baseline["clips"]:
        sid=p["sample"]
        path=root/"audit/A"/sid/"prediction.264"
        hashes.append(dict(sample=sid,stream_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
          encoder_matches_decoder=p["encoder_matches_decoder"],decoder_agreement=p["decoder_agreement"],
          me_log_s=p["motion_search_s_reported"],psnr=p["psnr_y_p_db"],kbps=p["bitrate_kbps"],last4=p["last4_db"]))
    write(out/"baseline_replay_hashes.tsv",hashes,"\t")
    curves=[]
    rows=[]
    histories=[]
    runs={}
    for pat in ("c0-*/run.json","c1-*/run.json"):
        for p in sorted(root.glob(pat)):
            run=read(p)
            name=p.parent.name
            runs[name]=run
            for i,entry in run["snapshots"].items():
                for split in ("train","val"):
                    points=entry[split]
                    keys=("recall","precision","precision_at_64","iou","value_exact","sign_correct","value_mae","nonzero")
                    desc=dict(run=name,arm=run["arm"],loss=run["loss"],seed=run["seed"],params=run["params"],
                              update=int(i),split=split,train_wall_s=run["elapsed_s"])
                    curves.append(dict(**desc,**{key:mean(x[key] for x in points) for key in keys},
                         micro_recall=sum(r["hits"] for r in points)/sum(r["positives"] for r in points),
                         joint_exact=sum(r["joint_exact"] for r in points)/sum(r["positives"] for r in points)))
                    for r in points:
                        rows.append(dict(**desc,**{key:r[key] for key in ("sample","prompt","hits","positives","candidates",*keys)},
                              prefilter_hits=sum(r["frame_hits_pre_filter"]) if "frame_hits_pre_filter" in r else None))
            histories.extend(dict(run=name,**x) for x in run["history"])
    write(out/"c0_train_recall.csv",[x for x in rows if x["split"]=="train" and x["loss"]=="hard"])
    write(out/"c0_validation_recall.csv",[x for x in rows if x["split"]=="val" and x["loss"]=="hard"])
    write(out/"c0_training_curves.csv",[x for x in curves if x["loss"]=="hard"])
    write(out/"c1_loss_ablation.csv",[x for x in curves if x["update"]==3200])
    write(root/"training_history.csv",histories)
    replay=[]; points=[]
    for p in sorted(root.glob("replay-*/run.json")):
        r=read(p)
        pts=r["points"]
        replay.append(dict(run=r["name"],**r["summary"],head_ms=median(x["head_seconds"] for x in pts)*1000,
            peak_mib=max(x["head_peak_extra_bytes"] for x in pts)/2**20,coeff_bits=mean(x["coeff_bits"] for x in pts),
            nonzero=mean(x["nonzero"] for x in pts),last_frame=mean(x["last_frame_psnr_y"] for x in pts),
            train_wall_s=runs[r["name"]]["elapsed_s"]))
        for q in pts:
            points.append(dict(run=r["name"],**{k:q[k] for k in ("sample","prompt","psnr_y_p_db","last4_db","last_frame_psnr_y",
                "ssim_y_p","bitrate_kbps","delta_db","delta_last4_db","coeff_bits","nonzero","head_seconds","head_peak_extra_bytes")}))
    assert len(replay)==len(runs)
    write(out/"strict_replay_per_video.csv",points)
    write(out/"strict_replay_summary.csv",replay)
    write(out/"c0_runtime_memory.csv",[{k:r[k] for k in ("run","head_ms","peak_mib","train_wall_s","nonzero","coeff_bits")} for r in replay])
    c2=read(root/"c2_summary.json")
    neighbors=[]
    for r in csv.DictReader((root/"c2_neighbors.csv").open()):
        if r["rank"]=="1" and r["matched_motion"]=="True":
            neighbors.append(r)
    bins=[-1,.3,.6,.8,.9,.95,.99,1.001]
    binned=[]
    for arm in ("cross_knn","matched_random","shuffled_labels","same_video"):
        for lo,hi in zip(bins,bins[1:]):
            rr=[r for r in neighbors if r["kind"]==arm and lo<=float(r["similarity"])<hi]
            if rr:
                binned.append(dict(arm=arm,lo=lo,hi=hi,n=len(rr),agreement=mean(int(r["agreement"]) for r in rr)))
    write(out/"c2_knn_label_agreement.csv",binned)
    (out/"feature_label_contract.json").write_bytes((root/"feature_label_contract.json").read_bytes())
    gate=read(root/"c1_gate.json")
    (out/"summary.json").write_text(json.dumps(dict(curves=curves,replay=replay,c2=c2,gate=gate,baseline=baseline["baseline"]),indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6.5,3.5),layout="constrained")
    for seed in (1701,1702):
        q=sorted((r for r in curves if r["split"]=="train" and r["loss"]=="hard" and r["update"]==3200 and r["seed"]==seed),key=lambda r:r["params"])
        ax.plot([r["params"] for r in q],[100*r["recall"] for r in q],marker="o",label=str(seed))
    ax.set(xscale="log",xlabel="Trainable head parameters",ylabel="Training realized recall (%)")
    ax.legend()
    plot(fig,figs/"train_recall_vs_capacity.svg")
    fig,ax=plt.subplots(figsize=(7,3.6),layout="constrained")
    for name in runs:
        q=sorted((r for r in curves if r["run"]==name and r["split"]=="train"),key=lambda r:r["update"])
        ax.plot([r["update"]/8 for r in q],[100*r["recall"] for r in q],marker=".",label=name)
    ax.set(xlabel="Exposure per video",ylabel="Training realized recall (%)")
    ax.legend(fontsize=7,ncol=2)
    plot(fig,figs/"training_curves.svg")
    fig,ax=plt.subplots(figsize=(7,3.6),layout="constrained")
    for name in runs:
        q=[r for r in histories if r["run"]==name]
        chunks=[q[i:i+8] for i in range(0,len(q),8)]
        ax.plot([mean(r["step"] for r in c) for c in chunks],[mean(r["loss"] for r in c) for c in chunks],label=name)
    ax.set(xlabel="Update",ylabel="Logged loss (8-point block mean)")
    ax.legend(fontsize=7,ncol=2)
    plot(fig,figs/"loss_curves.svg")
    fig,ax=plt.subplots(figsize=(7,3.6),layout="constrained")
    for arm in ("cross_knn","matched_random","shuffled_labels","same_video"):
        q=[r for r in binned if r["arm"]==arm]
        ax.plot([(r["lo"]+r["hi"])/2 for r in q],[100*r["agreement"] for r in q],marker="o",label=arm)
    ax.set(xlabel="Cosine similarity bin midpoint",ylabel="Label agreement (%)",title="Matched candidate strata; balanced diagnostic sampling")
    ax.legend(fontsize=7)
    plot(fig,figs/"knn_similarity_vs_label_agreement.svg")
    fig,ax=plt.subplots(figsize=(7,3.6),layout="constrained")
    ax.scatter(baseline["baseline"]["bitrate_kbps"],0,label="Frozen A")
    for r in replay:
        ax.scatter(r["bitrate_kbps"],r["delta_db"],label=r["run"])
    ax.set(xlabel="Actual bitrate (kb/s)",ylabel="P Y-PSNR difference (dB)")
    ax.legend(fontsize=7,ncol=2)
    plot(fig,figs/"rd_curve.svg")


if __name__=="__main__":
    main()
