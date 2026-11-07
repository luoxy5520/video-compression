"""Compute content-group diagnostics from locked final-step evaluations."""
import json
import statistics as stats
from pathlib import Path

d=Path(__file__).resolve().parent/"results"
names=("blind_pilot","blind_control","blind_expanded","blind_hidden","blind_recurrent")
groups={}
for name in names:
    file=d/f"{name}.json"
    if not file.exists(): continue
    for row in json.loads(file.read_text()):
        if row["step"]==20:
            groups.setdefault(row["prompt"],{}).setdefault(name,[]).append(row)

def mean(xs):
    v=[x for x in xs if x is not None]
    return stats.mean(v) if v else None

report={}
for prompt,arms in groups.items():
    report[prompt]={}
    for name,rows in arms.items():
        oracle=[r["state"]["oracle_p16"] for r in rows]
        counts=[m["count"] for m in oracle]
        motion=[m for m in oracle if m["count"]]
        zero=mean(m["motion_bins"][0]["count"]/m["count"] for m in motion)
        values=dict(clips=len(rows),oracle_p16_fraction=stats.mean(counts)/24960,
            oracle_mv_zero_baseline=zero,mv_exact=mean(m["exact"] for m in motion),
            mv_mae_xy=[mean(m["mv_mae_xy"][i] for m in motion) for i in (0,1)],
            mode_macro_f1=mean(r["state"]["macro_f1"] for r in rows),
            bitrate_kbps=mean(r["replay"]["bitrate_kbps"] for r in rows),
            psnr_y_p_db=mean(r["replay"]["psnr_y_p_db"] for r in rows),
            oracle_psnr_y_p_db=mean(r["oracle"].get("psnr_y_p_pooled") for r in rows),
            last_frame_psnr=mean(r["replay"]["last_frame_psnr_y"] for r in rows))
        report[prompt][name.removeprefix("blind_")]=values
(d/"blind_content_groups.json").write_text(json.dumps(report,indent=2,allow_nan=False))
for p,arms in report.items():
    print(p)
    for key,r in arms.items():
        print(" ",key,"f1",round(r["mode_macro_f1"],3),"exact",round(r["mv_exact"],3),
              "zero",round(r["oracle_mv_zero_baseline"],3),"PSNR",round(r["psnr_y_p_db"],3))
