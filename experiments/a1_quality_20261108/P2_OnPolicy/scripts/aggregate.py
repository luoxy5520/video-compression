"""Compute paired video statistics, reproducible intervals, and concise artifacts."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"]="p2-onpolicy"
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def by_prompt(points,key):
    return {p:mean(r[key] for r in points if r["prompt"]==p) for p in dict.fromkeys(r["prompt"] for r in points)}


def avg(points,key):
    return mean(by_prompt(points,key).values())


def interval(points,key,seed):
    rng=np.random.default_rng(seed)
    values=[]
    for p in dict.fromkeys(r["prompt"] for r in points):
        a=np.asarray([r[key] for r in points if r["prompt"]==p])
        values.append(rng.choice(a,(10000,len(a)),replace=True).mean(axis=1))
    return np.quantile(np.mean(values,axis=0),[.025,.975]).tolist()


def clean_figure(fig,path):
    fig.savefig(path,metadata={"Date":None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("root",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    root,out=args.root,args.out
    out.mkdir(parents=True,exist_ok=True)
    figures=out/"figures"
    figures.mkdir(exist_ok=True)
    o0=read(root/"o0/metrics.json")
    assert len(o0)==60
    val=[r for r in o0 if r["split"]=="val"]
    records=[]; labels=[]; frames=[]; teachers=[]
    for row in o0:
        key={k:row[k] for k in ("sample","split","prompt")}
        info=row["stability"]
        records.append(dict(**key,**{k:v for k,v in info.items() if not isinstance(v,(dict,list))}))
        frames.extend(dict(**key,**x) for x in row["frame_frequency"])
        for arm,stat in row["arms"].items():
            labels.append(dict(**key,arm=arm,candidates=row["candidates"],selected=stat["selected"],
                label_sha256=stat["label_sha256"],source_sha256=row["source_sha256"],
                mode_mv_sha256=stat["mode_mv_sha256"],reference_stream_sha256=stat["source_reference_stream_sha256"],
                frame_frequency_counts=" ".join(map(str,stat["counts_by_frequency"]))))
            teachers.append(dict(**key,arm=arm,authority="offline fixed-prefix diagnostic",**{
              k:stat[k] for k in ("psnr_y_p_db","last4_db","bitrate_kbps","delta_db","delta_last4_db","rate_pct","coeff_bits","mv_bits",
                                  "reference_consistent","local_dense_sse_delta_vs_zero")},same_label_frames=sum(stat["equal_label_by_frame"]),
                mean_ref_mse=mean(stat["actual_vs_prefix_ref_mse_by_frame"])))
    write(out/"label_pair_manifest.csv",labels)
    write(out/"trajectory_iou.csv",records)
    write(root/"o0_frame_frequency.csv",frames)
    write(out/"teacher_rd_comparison.csv",teachers)
    runs={f.parent.name:read(f) for f in sorted(root.glob("o1-*/run.json"))}
    assert len(runs)==4
    student=[]; student_frames=[]; paired=[]; by_seed={}; summary={}
    metrics=("psnr_y_p_db","bitrate_kbps","last4_db","ssim_y_p","delta_db","delta_last4_db","rate_pct","coeff_bits","mv_bits",
             "nonzero","ap","precision","recall","realized_precision","realized_recall","value_exact","sign_correct","value_mae","loss_select","loss_level")
    for name,run in runs.items():
        pts=run["points"]
        assert len(run["train"])==44 and len(pts)==16
        summary[name]=dict(run["summary"],params=run["params"],updates=run["updates"],
          negative_clips=sum(p["delta_db"]<0 for p in pts),overcap_clips=sum(p["rate_pct"]>3 for p in pts),
          head_median_ms=median(p["head_seconds"] for p in pts)*1000,
          head_peak_mib=max(p["head_peak_extra_bytes"] for p in pts)/2**20,
          jm_median_s=median(p["jm_replay_s"] for p in pts),
          ffmpeg_median_s=median(p["ffmpeg_decode_s"] for p in pts),
          last_frame_db=avg(pts,"last_frame_psnr_y"),
          last4_delta_by_prompt=by_prompt(pts,"delta_last4_db"),
          delta_interval=interval(pts,"delta_db",run["seed"]),
          last4_interval=interval(pts,"delta_last4_db",run["seed"]),
          val={k:avg(pts,k) for k in metrics},
          train={k:avg(run["train"],k) for k in ("ap","realized_precision","realized_recall","value_exact","sign_correct","loss_select","loss_level")})
        for p in pts:
            student.append(dict(run=name,sample=p["sample"],prompt=p["prompt"],seed=run["seed"],**{k:p[k] for k in metrics}))
            for i in range(17):
                student_frames.append(dict(run=name,sample=p["sample"],frame=i,psnr_y=p["psnr_y_by_frame"][i],
                   delta_psnr_y=p["per_frame_delta_db"][i],ssim_y=p["ssim_y_by_frame"][i],
                   ref_vs_A_mse=None if i==0 else p["ref_vs_A_mse_by_frame"][i-1],
                   coeff_bits=None if i==0 else p["per_frame_coeff_bits"][i-1]))
    for seed in (1701,1702):
        left=runs[f"o1-T0-{seed}"]["points"]
        right=runs[f"o1-T1-{seed}"]["points"]
        items=[]
        for t0,t1 in zip(left,right,strict=True):
            assert t0["sample"]==t1["sample"]
            items.append(dict(seed=seed,sample=t0["sample"],prompt=t0["prompt"],
                **{k:t1[k]-t0[k] for k in ("psnr_y_p_db","last4_db","bitrate_kbps","ap","value_exact")}))
        paired.extend(items)
        by_seed[str(seed)]={k:dict(mean=avg(items,k),bootstrap_95=interval(items,k,seed),prompts=by_prompt(items,k))
                          for k in ("psnr_y_p_db","last4_db","bitrate_kbps")}
    go=all(summary[f"o1-T1-{s}"]["passed"] and by_seed[str(s)]["psnr_y_p_db"]["bootstrap_95"][0]>0
           for s in (1701,1702))
    write(out/"t0_t1_per_video.csv",student)
    write(out/"paired_differences.csv",paired)
    write(root/"t0_t1_per_frame.csv",student_frames)
    own=read(root/"diagnostics/own_reference_readout.json")
    diagnosis={f"{arm}-{seed}":{key:mean(r[key] for r in own if r["arm"]==arm and r["seed"]==seed)
               for key in ("block_energy_logit_correlation","realized_support_overlap","value_exact","sign_correct","actual_signed_level_mae")}
              for arm in ("T0","T1") for seed in (1701,1702)}
    stability=dict(median_val_iou=median(r["stability"]["iou"] for r in val),
                  median_train_iou=median(r["stability"]["iou"] for r in o0 if r["split"]=="train"),
                  median_val_overlap=median(r["stability"]["overlap"] for r in val),
                  median_sign_agreement=1-median(r["stability"]["joint_sign_flip"] for r in val if r["stability"]["joint_sign_flip"] is not None),
                  median_joint_level_mae=median(r["stability"]["joint_level_mae"] for r in val if r["stability"]["joint_level_mae"] is not None),
                  median_dense_exact=median(r["stability"]["dense_exact"] for r in val),
                  median_to_zero_T0=median(r["stability"]["left_selected_dense_zero"] for r in val),
                  median_to_zero_T1=median(r["stability"]["t1_selected_dense_zero"] for r in val),
                  self_consistent_clips={arm:sum(r["arms"][arm]["reference_consistent"] for r in val) for arm in ("T0","T1")},
                  mean_prefix_ref_mse={arm:mean(mean(r["arms"][arm]["actual_vs_prefix_ref_mse_by_frame"]) for r in val) for arm in ("T0","T1")})
    data=dict(stability=stability,o1=summary,paired_by_seed=by_seed,own_reference=diagnosis,o2_gate=go)
    (out/"metric_summary.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.6),layout="constrained")
    for ax,arm in zip(axes,("T0","T1"),strict=True):
        n=np.zeros((16,16))
        for r in frames:
            if r["split"]=="val" and r["frequency"]!="all":
                n[r["frequency"],r["frame"]-1]+=r["n0" if arm=="T0" else "n1"]
        image=ax.imshow(n,origin="lower",aspect="auto",vmin=0,vmax=64,extent=(.5,16.5,-.5,15.5))
        ax.set(title=arm,xlabel="P frame",ylabel="Raster frequency")
    fig.colorbar(image,ax=axes,label="Offline source-assisted selections")
    clean_figure(fig,figures/"label_frequency.svg")
    fig,ax=plt.subplots(figsize=(6.5,3.4),layout="constrained")
    for seed in (1701,1702):
        items=[p for p in paired if p["seed"]==seed]
        ax.plot(range(1,17),[p["psnr_y_p_db"] for p in items],marker="o",label=f"Seed {seed}")
    ax.axhline(0,color="black",linewidth=.8)
    ax.set(xlabel="Matched validation clip",ylabel="T1 minus T0 Y-PSNR (dB)")
    ax.legend()
    clean_figure(fig,figures/"paired_rd.svg")


if __name__=="__main__":
    main()
