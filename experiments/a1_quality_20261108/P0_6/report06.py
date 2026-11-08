"""Summarize measured P0.6 sweeps and apply the existing prompt-level gate."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

from sweep import CONFIGS, FRAMES, HERE, JM, P05, PCTS, POLICIES, digest, records
from diagnose import csv_write


def average(rows, key):
    groups=defaultdict(list)
    for row in rows:
        groups[row["prompt_id"]].append(row[key])
    return mean(mean(r) for r in groups.values())


def svg(path, curves):
    keys=["A","zero",*(f"{p}_{n}" for p in POLICIES for n in PCTS)]
    points={r["arm"]:r for r in curves if r["arm"] in keys}
    xmin=5*math.floor(min(p["bitrate_kbps"] for p in points.values())/5)-2
    xmax=5*math.ceil(max(p["bitrate_kbps"] for p in points.values())/5)+2
    ymin=math.floor(2*min(p["psnr_y_db"] for p in points.values()))/2-.1
    ymax=math.ceil(2*max(p["psnr_y_db"] for p in points.values()))/2+.1
    def xy(rate, psnr):
        return 82+710*(rate-xmin)/(xmax-xmin), 402-305*(psnr-ymin)/(ymax-ymin)
    data=['<svg xmlns="http://www.w3.org/2000/svg" width="870" height="480" viewBox="0 0 870 480">',
          '<rect width="870" height="480" fill="#fff"/><g font-family="system-ui,sans-serif" fill="#172943">',
          '<text x="34" y="37" font-size="23" font-weight="700">Fixed Mode/MV residual RD · JM QP28</text>',
          '<text x="34" y="60" font-size="13">Validation · 4 equally weighted prompts · strict replay</text>',
          '<text x="33" y="88" font-size="12">Y-PSNR (dB)</text>']
    for y in range(math.ceil(ymin*2),math.floor(ymax*2)+1):
        _,py=xy(xmin,y/2)
        data.append(f'<path d="M82 {py:.1f}h710" stroke="#dbe4ec"/><text x="74" y="{py+4:.1f}" text-anchor="end" font-size="12">{y/2:g}</text>')
    for rate in range(5*math.ceil(xmin/5),5*math.floor(xmax/5)+1,5):
        x,_=xy(rate,ymin)
        data.append(f'<path d="M{x:.1f} 97v305" stroke="#dbe4ec"/><text x="{x:.1f}" y="420" text-anchor="middle" font-size="12">{rate}</text>')
    colors=["#3266aa","#0b877b","#d15d42"]
    for i,(policy,color) in enumerate(zip(POLICIES,colors)):
        rows=[points[f"{policy}_{pct}"] for pct in sorted(PCTS)]
        coords=[xy(r["bitrate_kbps"],r["psnr_y_db"]) for r in rows]
        coord_string=" ".join(f"{x:.2f},{y:.2f}" for x,y in coords)
        data.append(f'<polyline points="{coord_string}" fill="none" stroke="{color}" stroke-width="2"/>')
        for (x,y),r in zip(coords,rows):
            data.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{r["arm"]}: {r["psnr_y_db"]:.3f} dB, {r["bitrate_kbps"]:.3f} kb/s</title></circle>')
        x=90+200*i
        data.append(f'<circle cx="{x}" cy="462" r="4" fill="{color}"/><text x="{x+10}" y="466" font-size="13">{policy}</text>')
    r=points["A"]
    x,y=xy(r["bitrate_kbps"],r["psnr_y_db"])
    data.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="#26384d"/><text x="{x+8:.1f}" y="{y-8:.1f}" font-size="12">A</text>')
    data.append('<text x="435" y="439" text-anchor="middle" font-size="13">Measured bitrate (kb/s)</text></g></svg>')
    path.write_text("\n".join(data))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    assert json.loads((a.run/"status.json").read_text())==dict(total=60,failed=[])
    gate=yaml.safe_load((HERE.parent/"P0/config.yaml").read_text())["gate"]
    groups=records(CONFIGS)
    expected=json.loads((a.run/"expected.json").read_text())
    manifest, video, frame, trials=[],[],[],[]
    for split in ("train","val"):
        rows=[r for r in groups[split] if r["step"]==20]
        assert [r["sample_id"] for r in rows]==expected[split]
        for row in rows:
            sample=row["sample_id"]
            root=a.run/split/sample
            prompt=hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
            base=dict(split=split,prompt_id=prompt,video_id=sample,seed=row["seed"])
            state=root/"fixed_mode_mv.npz"
            weights=root/"frame_weights.json"
            manifest.append(dict(**base,mode_mv_path=str(state),mode_mv_sha256=digest(state),
                                 weights_path=str(weights),weights_sha256=digest(weights)))
            with (root/"per_frame.csv").open() as f:
                fr=[{**base,**r} for r in csv.DictReader(f)]
            assert len(fr)==17*FRAMES
            frame.extend(fr)
            if (root/"trials.csv").exists():
                with (root/"trials.csv").open() as f:
                    trials.extend({**base,**r} for r in csv.DictReader(f))
            values=json.loads((root/"summary.json").read_text())
            assert len(values)==17 and values[0]["arm"]=="A"
            a_bits=values[0]["bitstream_bits"]
            for v in values:
                assert v["decoder_agreement"] and v["encoder_matches_decoder"]
                assert v["motion_search_s_reported"]==0 and v["oracle_p_bytes_read"]==0
                if v["allocation"]!="anchor":
                    assert v["under_cap"]
                these=[r for r in fr if r["arm"]==v["arm"]]
                assert len(these)==16
                video.append(dict(**base,**{k:v[k] for k in ("arm","allocation","target_pct","target_total_bits",
                             "bitstream_bits","under_cap","paired_offline","diagnostic_oracle","fallback",
                             "bitrate_kbps","ssim_y_p","stream_sha256","state_sha256")},
                             psnr_y_db=v["psnr_y_p_db"],last4_psnr_y_db=mean(v["psnr_y_by_frame"][-4:]),
                             delta_total_bits=v["bitstream_bits"]-a_bits,
                             residual_coeff_bits=sum(int(r["residual_coeff_bits"]) for r in these),
                             luma_nonzero_coeffs=sum(int(r["luma_nonzero_coeffs"]) for r in these),
                             total_y_sse=65025*10**(-v["psnr_y_p_db"]/10)*832*480*FRAMES,
                             three_way_equal=True,me_time_s=0,oracle_p_bytes_read=0))
    val=[r for r in video if r["split"]=="val"]
    base={r["video_id"]:r for r in val if r["arm"]=="A"}
    curves=[]
    for name in [v["arm"] for v in values]:
        rows=[r for r in val if r["arm"]==name]
        for row in rows:
            for key in ("psnr_y_db","last4_psnr_y_db","total_y_sse"):
                row["delta_"+key]=row[key]-base[row["video_id"]][key]
        group=defaultdict(list)
        for r in rows:
            group[r["prompt_id"]].append(r["delta_psnr_y_db"])
        psnr=average(rows,"psnr_y_db")
        gain=average(rows,"delta_psnr_y_db")
        rate=average(rows,"bitrate_kbps")
        rate_gain=100*(rate/average(list(base.values()),"bitrate_kbps")-1)
        last4=average(rows,"last4_psnr_y_db")
        tail=average(rows,"delta_last4_psnr_y_db")
        positive=mean(mean(v)>0 for v in group.values())
        delta_bits=average(rows,"delta_total_bits")
        passing=(gain>=gate["min_psnr_gain_db_at_controlled_rate"] and
                 rate_gain<=gate["max_rate_increase_pct"] and tail>=gate["min_last4_delta_db"] and
                 positive>=gate["min_positive_prompt_fraction"])
        curves.append(dict(arm=name,target_pct=rows[0]["target_pct"],bitrate_kbps=rate,psnr_y_db=psnr,
            delta_rate_pct=rate_gain,delta_psnr_db=gain,last4_psnr_y_db=last4,delta_last4_db=tail,
            positive_prompt_fraction=positive,mean_added_total_bits=delta_bits,
            mean_residual_coeff_bits=average(rows,"residual_coeff_bits"),
            mean_luma_nonzero_coeffs=average(rows,"luma_nonzero_coeffs"),
            y_sse_reduction_per_added_bit=-average(rows,"delta_total_y_sse")/delta_bits if delta_bits>0 else None,
            source_paired_clips=sum(r["paired_offline"] for r in rows),passes=passing))
    cap=average(list(base.values()),"bitrate_kbps")*1.05
    best=max((r for r in curves if r["bitrate_kbps"]<=cap),key=lambda r:r["psnr_y_db"])
    a.output.mkdir(parents=True,exist_ok=False)
    for name,data in (("per_video",video),("per_frame",frame),("rd_curve",curves),("rate_trials",trials)):
        if name=="per_video":
            data=[{k:v for k,v in r.items() if k not in
                   ("delta_psnr_y_db","delta_last4_psnr_y_db","delta_total_y_sse")} for r in data]
        csv_write(a.output/f"{name}.csv",data)
    (a.output/"fixed_state_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in manifest))
    (a.output/"summary.json").write_text(json.dumps(dict(
        validation_clips=16,validation_prompts=4,rate_cap_kbps=cap,best_under_cap=best,
        passing_arms=[r["arm"] for r in curves if r["passes"]],curves=curves,
        p06_encoder_sha256=digest(JM/"bin/lencod.exe"),
        p05_manifest_sha256=digest(P05/"report/manifest.json")),indent=2,allow_nan=False))
    svg(a.output/"rd_curve.svg",curves)


if __name__=="__main__":
    main()
