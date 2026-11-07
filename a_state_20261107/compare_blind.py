"""Paired prompt-group comparison. No tuning or gate uses these results."""
import json
import statistics
import sys
from pathlib import Path

p=Path(sys.argv[1])
arms={key:{(r["sample_id"],r["step"]):r for r in json.loads((p/f"eval_{key}/results.json").read_text())}
      for key in ("pilot","expanded")}
assert arms["pilot"].keys()==arms["expanded"].keys()
def measures(r):
    s=r["state"]
    out=dict(mode_f1=s["macro_f1"],mv_exact=s["oracle_p16"].get("exact"),
             y_res_rmse=s["oracle_p16"].get("residual_rmse",{}).get("y"))
    if "replay" in r:
        out.update({k:r["replay"][k] for k in ("psnr_y_p_db","bitrate_kbps","last_frame_psnr_y","ssim_y_p")})
        if "psnr_y_p_pooled" in r["oracle"]:
            out.update(jm_psnr_y_p_db=r["oracle"]["psnr_y_p_pooled"],jm_bitrate_kbps=r["oracle"]["bitrate_kbps"])
    return out

def mean(values):
    v=[x for x in values if x is not None]
    return statistics.mean(v) if v else None

report={}
for step in (5,10,15,20):
    prompts={}
    for idx,a in arms["pilot"].items():
        if idx[1]!=step: continue
        b=arms["expanded"][idx]
        assert a["prompt"]==b["prompt"]
        prompts.setdefault(a["prompt"],[]).append((measures(a),measures(b)))
    metric=list(next(iter(prompts.values()))[0][0])
    group=[]
    for prompt, rows in prompts.items():
        a={k:mean(x[k] for x,y in rows) for k in metric}
        b={k:mean(y[k] for x,y in rows) for k in metric}
        group.append(dict(prompt=prompt,videos=len(rows),pilot=a,expanded=b,
                          delta={k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in metric}))
    report[step]=dict(groups=group,mean={arm:{k:mean(g[arm][k] for g in group) for k in metric}
                                         for arm in ("pilot","expanded","delta")})
if "jm_psnr_y_p_db" in report[20]["mean"]["pilot"]:
    verdict={}
    for arm in ("pilot","expanded"):
        v=report[20]["mean"][arm]
        gap=v["jm_psnr_y_p_db"]-v["psnr_y_p_db"]
        increase=v["bitrate_kbps"]/v["jm_bitrate_kbps"]-1
        verdict[arm]=dict(mean_y_loss_db=gap,rate_increase_fraction=increase,
                          pass_project_quality_rate_line=gap<=1 and increase<=.10)
    report["final_replay_gate"]=verdict
(p/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
