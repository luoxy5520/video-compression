"""Check persisted A results for coverage and consistency without rerunning the models."""
import json
from pathlib import Path

d=Path(__file__).resolve().parent/"results"
def read(key):
    return json.loads((d/f"{key}.json").read_text())

def coverage(rows, n, steps, expected_replays):
    ids={r["sample_id"] for r in rows}
    pairs={(r["sample_id"],r["step"]) for r in rows}
    assert len(ids)==n and len(pairs)==len(rows)==n*len(steps)
    assert pairs=={(sid,s) for sid in ids for s in steps}
    replay=[r["replay"] for r in rows if "replay" in r]
    assert len(replay)==expected_replays
    for r in replay:
        assert r["encoder_matches_decoder"] and r["decoder_agreement"]
        assert r["motion_search_s_reported"] in (None,0)
        assert r.get("oracle_p_bytes_read",0)==0
    return dict(videos=n,predictions=len(rows),replays=len(replay))

report={}
report["pilot"]=coverage(read("pilot"),8,(5,10,15,20),8)
report["hidden_pilot"]=coverage(read("hidden_pilot"),8,(5,10,15,20),0)
for arm in ("pilot","control","expanded","hidden","recurrent"):
    key=f"blind_{arm}"
    if (d/f"{key}.json").exists():
        report[key]=coverage(read(key),12,(5,10,15,20),12)
for k,n in (("oracle_residual",8),("oracle_residual_blind",12)):
    rows=read(k)
    assert len(rows)==n and all(r["replay"]["encoder_matches_decoder"] for r in rows)
    report[k]=n
x=read("teacher_trace")
assert len(x)==24 and all(r["bitstream_exact"] and r["encoder_matches_decoder"] for r in x)
report["teacher_exact"]=len(x)
x=read("long33")
assert len(x)==2 and all(r["frames"]==33 and len(r["direct"]["psnr_y_by_frame"])==33 and
                        r["direct"]["motion_search_s_reported"]==0 and r["direct"]["encoder_matches_decoder"] for r in x)
report["stress33"]=len(x)
x=read("frozen_codec_blind")
assert x["passed"] and len(x["per_video"])==12 and all(r["encoder_matches_decoder"] for r in x["per_video"])
report["frozen_codec_blind"]=len(x["per_video"])
assert not any(r["pass_project_quality_rate_line"] for r in read("blind_comparison")["final_replay_gate"].values())
x=read("handoff_validation")
assert x["shapes_dtypes_and_hashes_passed"] and x["predictions"]==320
report["handoff_predictions"]=320
print(json.dumps(report,indent=2))
