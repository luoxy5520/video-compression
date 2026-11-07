"""Summarize held-out regression quality without using stale residual confidence."""
import json
import statistics
import sys
from pathlib import Path

p=Path(sys.argv[1])
res=json.loads(p.read_text())
reference=json.loads(Path(sys.argv[2]).read_text())
reference={r["sample_id"]:r["oracle"] for r in reference if r["step"]==20}
keys=("psnr_y_p_db","bitrate_kbps","last_frame_psnr_y","late_minus_early_psnr","ssim_y_p")

def mean(rows):
    return {k:statistics.mean(r[k] for r in rows) for k in keys}

def checked(rows):
    assert all(r["encoder_matches_decoder"] and r["decoder_agreement"]
               and r["motion_search_s_reported"]==0 and r["oracle_p_bytes_read"]==0 for r in rows)

test=res["test"]
assert test.keys()=={"base","selected"}
ids={r["sample_id"] for r in test["base"]}
assert ids=={r["sample_id"] for r in test["selected"]}==reference.keys()
for rows in test.values():
    checked(rows)
for rows in res["validation"].values():
    checked(rows)
base=mean(test["base"])
selected=mean(test["selected"])
oracle=dict(psnr_y_p_db=statistics.mean(r["psnr_y_p_pooled"] for r in reference.values()),
            bitrate_kbps=statistics.mean(r["bitrate_kbps"] for r in reference.values()))
gap=oracle["psnr_y_p_db"]-selected["psnr_y_p_db"]
rate=selected["bitrate_kbps"]/oracle["bitrate_kbps"]-1
groups={}
for prompt in dict.fromkeys(r["prompt"] for r in test["base"]):
    groups[prompt]={k:mean([r for r in v if r["prompt"]==prompt]) for k,v in test.items()}
out=dict(selected=res["selected"],validation={k:mean(v) for k,v in res["validation"].items()},
         test=dict(videos=len(ids),base=base,selected=selected,oracle=oracle,
                   delta={k:selected[k]-base[k] for k in keys},groups=groups,
                   loss_db_vs_full_jm=gap,rate_increase_vs_full_jm=rate,passed=gap<=1 and rate<=.10))
Path(sys.argv[3]).write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps(out["test"]["delta"]))
