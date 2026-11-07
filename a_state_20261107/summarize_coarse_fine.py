"""Report motion and codec outcomes, keeping offline top-K oracles separate."""
import json
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1])
def read(name):
    return json.loads((root/f"{name}.json").read_text())
def mean(x):
    v=[a for a in x if a is not None]
    return statistics.mean(v) if v else None
def summarize(rows):
    assert all(x["decoder_agreement"] and x["encoder_matches_decoder"] and
               x["motion_search_s_reported"]==0 and x["oracle_p_bytes_read"]==0 for x in rows)
    a={k:mean(x[k] for x in rows) for k in ("psnr_y_p_db","last_frame_psnr_y","bitrate_kbps")}
    a["motion"]={k:mean(x["motion"].get(k) for x in rows) for k in ("exact","epe","nonzero_epe")}
    a["zero_control"]={k:mean(x["motion"]["zero_control"].get(k) for x in rows)
                       for k in ("exact","epe","nonzero_epe")}
    a["videos"]=len(rows)
    return a

x=read("coarse_fine")
stats=dict(selected=x["selected"],
           validation={arm:summarize(rows) for arm,rows in x["validation"].items()},
           regression={arm:summarize(rows) for arm,rows in x["regression"].items()})
stats["offline_candidate_diagnostic"]={}
for arm in ("oracle","teacher"):
    rows=read(f"coarse_rank_{arm}")
    stats["offline_candidate_diagnostic"][arm]={k:mean(x["moving"][k] for x in rows if "moving" in x)
        for k in ("top1_epe","best4_epe","top1_exact","top4_exact")}
(root/"coarse_fine_summary.json").write_text(json.dumps(stats,indent=2,allow_nan=False))
print(json.dumps(stats,ensure_ascii=False))
