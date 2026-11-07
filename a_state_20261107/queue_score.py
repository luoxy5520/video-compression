"""Atomically publish validation-only progress; never select on a regression split."""
import json
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1])
scores=[]
for f in sorted(root.glob("trial_[0-9]*/results.json")):
    if ".incomplete." in f.parent.name:
        continue
    x=json.loads(f.read_text())
    assert not x["regression"] and x["validation_only"]
    def mean(arm,k):
        return statistics.mean(r[k] for r in x["validation"][arm])
    baseline=mean("base","psnr_y_p_db")
    for arm, rows in x["validation"].items():
        assert len(rows)==16
        assert all(r["decoder_agreement"] and r["encoder_matches_decoder"] and
                   r["motion_search_s_reported"]==0 and r["oracle_p_bytes_read"]==0 for r in rows)
        if arm=="base":
            continue
        scores.append(dict(trial=f.parent.name,arm=arm,risk=x["risk_weight"],seed=x["seed"],
            psnr=mean(arm,"psnr_y_p_db"),rate=mean(arm,"bitrate_kbps"),
            gain_db=mean(arm,"psnr_y_p_db")-baseline,
            moving_epe=statistics.mean(r["motion"]["nonzero_epe"] for r in rows if r["motion"].get("nonzero_epe") is not None)))
scores.sort(key=lambda x:x["gain_db"],reverse=True)
tmp=root/"status.tmp"
tmp.write_text(json.dumps(dict(trials=len({x["trial"] for x in scores}),best=scores[0] if scores else None,
                               candidates=scores),indent=2,allow_nan=False))
tmp.replace(root/"status.json")
