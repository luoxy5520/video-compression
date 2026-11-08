"""Aggregate LabelProbe per-video data and create publication plots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(run, data):
    output=[]
    for row in data["points"]:
        keys=("sample","prompt","split","psnr_y_p_db","bitrate_kbps","ssim_y_p","last4_db","delta_db",
              "delta_last4_db","rate_pct","coeff_bits","nonzero","head_seconds","head_peak_extra_bytes",
              "precision","recall","overlap","sign_correct","value_exact","value_mae","quantization_error")
        output.append(dict(run=run,**{k:row.get(k) for k in keys},dc_recall=row["dc"]["recall"],dc_ap=row["dc"]["ap"],
                           dc_value_exact=row.get("dc_value_exact"),cap_3_pct=row["rate_pct"]<=3,cap_5_pct=row["rate_pct"]<=5))
    return output


def groups(rows):
    by=defaultdict(list)
    for row in rows:
        by[row["prompt"]].append(row)
    return by


def average(rows,k):
    return mean(mean(r[k] for r in p) for p in groups(rows).values())


def rd(run, data):
    rows=data["points"]
    h=data["summary"]
    return dict(run=run,scope=data["scope"],arm=data["arm"],seed=data["seed"],parameters=data["params"],
                psnr_y=h["psnr_y_p_db"],kbps=h["bitrate_kbps"],last4=h["last4_db"],delta_db=h["delta_db"],
                delta_last4=h["delta_last4_db"],worst=h["worst_clip"],worst_delta=h["worst_delta_db"],
                failed_3_pct=sum(r["rate_pct"]>3 for r in rows),failed_5_pct=sum(r["rate_pct"]>5 for r in rows),
                negative_mean_clips=sum(r["delta_db"]<0 for r in rows),
                dc_ap=average(rows,"dc_ap") if "dc_ap" in rows[0] else mean(r["dc"]["ap"] for r in rows),
                dc_recall=mean(r["dc"]["recall"] for r in rows),ssim=average(rows,"ssim_y_p"),
                ms=median(r["head_seconds"] for r in rows)*1000,
                mib=max(r["head_peak_extra_bytes"] for r in rows)/2**20)


def draw(out, label_data, summary):
    val=[r for r in label_data if r["split"]=="val"]
    pairs=list(val[0]["pairs"])
    fig, ax=plt.subplots(figsize=(7.0,3.4),layout="constrained")
    ax.boxplot([[r["pairs"][k]["iou"] for r in val] for k in pairs],tick_labels=pairs)
    ax.set(ylim=(0,1.03),ylabel="Sparse support IoU (16 matched clips)",title="Same clip, different reconstructed reference")
    ax.grid(axis="y",alpha=.3)
    fig.savefig(out/"label_support.svg")
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(7.0,3.6),layout="constrained")
    for pair in pairs:
        hist=defaultdict(int)
        for row in val:
            for k,n in row["pairs"][pair]["dense_delta_hist"].items():
                hist[int(k)]+=n
        total=sum(hist.values())
        x=sorted(hist)
        ax.step(x,[hist[k]/total for k in x],where="mid",label=pair)
    ax.set(yscale="log",xlabel="Dense coefficient difference (QP28 levels)",ylabel="Probability",
           title="Dense level shifts across reconstructed references")
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(out/"level_shifts.svg")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.0,4.0),layout="constrained")
    runs=[r for r in summary if r["scope"] in ("L2","multi")]
    for arm in dict.fromkeys(("L0" if r["scope"]=="multi" else "L2")+" "+r["arm"] for r in runs):
        points=[r for r in runs if ("L0" if r["scope"]=="multi" else "L2")+" "+r["arm"]==arm]
        ax.scatter([r["kbps"] for r in points],[r["delta_db"] for r in points],label=arm)
    ax.axhline(0,color="black",linewidth=.8)
    ax.axvline(439.5138823529412,color="black",linewidth=.8,linestyle=":")
    ax.set(xlabel="Actual H.264 bitrate (kb/s)",ylabel="P-frame Y-PSNR difference vs A (dB)",
           title="Matched validation replay")
    ax.legend(ncol=2)
    ax.grid(alpha=.25)
    fig.savefig(out/"validation_rd.svg")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    runs={f.parent.name:read(f) for f in sorted(a.root.glob("l[02]-*/run.json"))}
    l1=read(a.root/"l1/metrics.json")
    assert len([k for k in runs if k.startswith("l0-")])==24
    summary=[rd(k,d) for k,d in runs.items()]
    paired=[r for run,d in runs.items() for r in metrics(run,d)]
    prompts=[]
    for k,d in runs.items():
        if d["scope"] not in ("multi","L2"):
            continue
        for prompt,rows in groups(d["points"]).items():
            prompts.append(dict(run=k,prompt=prompt,clips=len(rows),**{
                field:mean(r[field] for r in rows) for field in ("delta_db","delta_last4_db","bitrate_kbps","psnr_y_p_db")}))
    stability=[]
    distributions=[]
    cross=[]
    for r in l1:
        for k,v in r["pairs"].items():
            omit=("dense_delta_hist","iou_by_frame")
            stability.append(dict(sample=r["sample"],prompt=r["prompt"],pair=k,**{m:n for m,n in v.items() if m not in omit}))
        for pair,value in r["cross_cost"].items():
            cross.append(dict(sample=r["sample"],pair=pair,delta_dense_sse_from_zero=value))
        for k,v in r["trajectories"].items():
            frequencies=v["frequency"]
            dense=v["dense_frequency"]
            distributions.append(dict(sample=r["sample"],split=r["split"],prompt=r["prompt"],trajectory=k,
                p16_blocks=r["p16_blocks"],selected=sum(frequencies),selected_dc=frequencies[0],
                selected_low=sum(frequencies[j] for j in (0,1,4)),dense_dc=dense[0],dense_low=sum(dense[j] for j in (0,1,4)),
                level_abs_mean=v["level_abs_mean"],positive_sign=v["positive_sign"],
                mc_mse_mean=mean(v["mc_mse_by_frame"]),mc_mse_last4=mean(v["mc_mse_by_frame"][-4:]),
                verified_repeat="repeat_sha256" in v))
    a.out.mkdir(parents=True,exist_ok=True)
    figs=a.out/"figures"
    figs.mkdir(exist_ok=True)
    write(a.out/"l0_ablation.csv",[r for r in summary if r["scope"]!="L2"])
    l2=[r for r in summary if r["scope"]=="L2"]
    if l2:
        write(a.out/"l2_probe_rd.csv",l2)
    write(a.out/"per_video.csv",paired)
    write(a.out/"per_prompt.csv",prompts)
    write(a.out/"l1_reference_stability.csv",stability)
    write(a.out/"l1_label_distribution.csv",distributions)
    write(a.out/"l1_cross_cost.csv",cross)
    draw(figs,l1,summary)


if __name__=="__main__":
    main()
