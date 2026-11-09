"""Aggregate fitted scale, ablations and strict validation replays."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean,median

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"]="p2-multisample"
import matplotlib.pyplot as plt
import numpy as np
import torch


def read(p):
    return json.loads(p.read_text())


def write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def plot(fig,path):
    fig.savefig(path,metadata={"Date":None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument("root",type=Path)
    c.add_argument("out",type=Path)
    a=c.parse_args()
    root,out=a.root,a.out
    out.mkdir(exist_ok=True)
    figs=out/"figures"
    figs.mkdir(exist_ok=True)
    cols=("recall","precision","precision_at_64","iou","value_exact","value_mae","sign_correct","nonzero")
    groups={}
    curves=[]
    raw=[]
    hist=[]
    for pattern in ("scale-*/run.json","duplicate-*/run.json","m2-*/run.json"):
        for p in sorted(root.glob(pattern)):
            name=p.parent.name
            r=read(p)
            groups[name]=r
            for update,m in r["snapshots"].items():
                for split in ("train","val"):
                    points=m[split]
                    key=dict(run=name,n=r["n"],seed=r["seed"],policy=r["policy"],duplicate=r["duplicate"],
                             split=split,update=int(update),mean_exposures=int(update)/r["n"],
                             min_exposure=min(m["exposure"].values()),max_exposure=max(m["exposure"].values()))
                    curves.append(dict(**key,**{k:mean(v[k] for v in points) for k in cols},
                        joint_exact=sum(v["joint_exact"] for v in points)/(64*len(points))))
                    raw.extend(dict(**key,sample=v["sample"],**{k:v[k] for k in cols},hits=v["hits"],
                         joint_exact=v["joint_exact"],positives=v["positives"],candidates=v["candidates"]) for v in points)
            hist.extend(dict(run=name,**x) for x in r["history"])
    assert sum(name.startswith("scale") for name in groups)==10
    assert sum(name.startswith("duplicate") for name in groups)==6
    duplicate_checks={}
    for seed in (1701,1702):
        single=torch.load(root/f"scale-1-{seed}/head-400.pt",map_location="cpu",weights_only=True)
        for n in (2,4,8):
            path=root/f"duplicate-{n}-{seed}/head-400.pt"
            model=torch.load(path,map_location="cpu",weights_only=True)
            equal=all(torch.equal(single[k],model[k]) for k in single)
            duplicate_checks[path.parent.name]=equal
            assert equal
    write(out/"m0_scale_train.csv",[x for x in raw if x["run"].startswith("scale") and x["split"]=="train"])
    write(out/"m0_scale_validation.csv",[x for x in raw if x["run"].startswith("scale") and x["split"]=="val"])
    write(out/"m0_duplicate_samples.csv",[x for x in curves if x["duplicate"]])
    write(out/"m2_ablations.csv",[x for x in curves if x["run"].startswith("m2-")])
    write(root/"m2_batch_statistics.csv",hist)
    gradients=[x for f in sorted(root.glob("m2-gradient-*.json")) for x in read(f)]
    write(out/"m2_gradient_cosines.csv",gradients)
    replay=[]
    per_video=[]
    for p in sorted(root.glob("replay-*/run.json")):
        a=read(p)
        pts=a["points"]
        replay.append(dict(run=a["name"],update=a["step"],**a["summary"],
                      head_ms=median(x["head_seconds"] for x in pts)*1000,
                      peak_mib=max(x["head_peak_extra_bytes"] for x in pts)/2**20,
                      negative_clips=sum(x["delta_db"]<0 for x in pts),
                      coeff_bits=mean(x["coeff_bits"] for x in pts),nonzero=mean(x["nonzero"] for x in pts)))
        for x in pts:
            keys=("sample","prompt","psnr_y_p_db","bitrate_kbps","last4_db","delta_db","delta_last4_db","coeff_bits",
                  "nonzero","head_seconds","head_peak_extra_bytes","ssim_y_p")
            per_video.append(dict(run=a["name"],update=a["step"],**{k:x[k] for k in keys}))
    assert len(replay)==6
    write(out/"strict_replay_summary.csv",replay)
    write(out/"per_video_metrics.csv",per_video)
    write(out/"runtime_memory.csv",[{k:r[k] for k in ("run","update","head_ms","peak_mib","nonzero","coeff_bits")} for r in replay])
    (out/"summary.json").write_text(json.dumps(dict(curves=curves,replay=replay,duplicate_tensor_equality=duplicate_checks,
            gradients=gradients,gate=read(root/"m2_gate.json"),selection=read(root/"replay_selection.json")),indent=2)+"\n")

    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout="constrained")
    for ax,split in zip(axes,("train","val"),strict=True):
        for method,condition in (("400 updates",lambda x:x["update"]==400),("80 exposures",lambda x:x["mean_exposures"]==80),
             ("Max budget",lambda x:x["update"]==min(400*x["n"],3520))):
            for seed in (1701,1702):
                q=sorted((x for x in curves if x["run"].startswith("scale") and x["seed"]==seed and x["split"]==split and condition(x)),key=lambda x:x["n"])
                ax.plot([x["n"] for x in q],[100*x["recall"] for x in q],marker="o",linestyle="-" if seed==1701 else "--",
                        label=f"{method}, {seed}")
        ax.set(xscale="log",xticks=[1,2,4,8,44],xticklabels=[1,2,4,8,44],xlabel="Distinct training videos",ylabel="Realized Recall (%)",title=split)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=6)
    plot(fig,figs/"scale_vs_recall.svg")
    fig,ax=plt.subplots(figsize=(6.4,3.6),layout="constrained")
    for name in sorted(groups):
        if not name.startswith("scale") or not name.endswith("1701"): continue
        q=sorted((x for x in curves if x["run"]==name and x["split"]=="train"),key=lambda x:x["update"])
        ax.plot([x["mean_exposures"] for x in q],[100*x["recall"] for x in q],marker="o",label=name)
    ax.set(xlabel="Mean updates per video",ylabel="Training recall (%)")
    ax.legend(fontsize=8)
    plot(fig,figs/"recall_vs_video_exposure.svg")
    fig,ax=plt.subplots(figsize=(6.4,3.6),layout="constrained")
    counts=[64/x["candidates"] for x in raw if x["run"]=="scale-44-1701" and x["update"]==3520 and x["split"]=="train"]
    ax.hist(np.asarray(counts)*100,bins=10)
    ax.set(xlabel="Positive label fraction (%)",ylabel="Training clips",title="64 positive levels per clip")
    plot(fig,figs/"training_positive_ratio.svg")
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout="constrained")
    for name in sorted(groups):
        if not name.startswith("m2-") or not name.endswith("1701"): continue
        q=groups[name]["history"]
        axes[0].plot([x["update"] for x in q],[x["total"] for x in q],alpha=.7,label=name)
        axes[1].plot([x["update"] for x in q],[x["select_grad"] for x in q],alpha=.7)
    axes[0].set(xlabel="Update",ylabel="Loss")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Update",ylabel="Selection head gradient norm",yscale="log")
    plot(fig,figs/"loss_gradient_diagnostics.svg")
    fig,ax=plt.subplots(figsize=(6.4,3.6),layout="constrained")
    ax.scatter(439.513882,0,label="Frozen A")
    for r in replay:
        ax.scatter(r["bitrate_kbps"],r["delta_db"],label=r["run"])
    ax.set(xlabel="Actual bitrate (kb/s)",ylabel="Y-PSNR change (dB)")
    ax.legend(fontsize=7)
    plot(fig,figs/"rd_comparison.svg")


if __name__=="__main__":
    main()
