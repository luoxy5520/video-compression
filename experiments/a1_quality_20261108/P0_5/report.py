"""Publish the P0.5 bit ledger, measured RD, and within-video error relationships."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from diagnose import CONFIGS, FRAMES, HERE, LAMBDAS, digest, csv_write, records

BIT_FIELDS = ("mode_run_bits", "mv_bits", "luma_coeff_bits", "chroma_coeff_bits",
              "other_mb_bits", "i_and_container_bits")
METRICS = ("p16_fraction", "luma_nonzero_coeffs", "chroma_nonzero_coeffs",
           "recon_mse_y", "previous_recon_error_mse_y", "mc_reference_error_mse_y",
           "motion_error_clean_ref_mse_y", "reference_motion_cross_y", "ideal_residual_mse_y",
           "ideal_residual_mse_y_p16", "residual_error_mse_y", "residual_error_mse_y_p16")


def frame_bits(path):
    import re
    data = path.read_bytes()
    starts = list(re.finditer(rb"\x00\x00(?:\x00)?\x01", data))
    p = []
    for i, start in enumerate(starts):
        if data[start.end()] & 31 == 1:
            end = starts[i+1].start() if i+1 < len(starts) else len(data)
            p.append(8*(end-start.start()))
    assert len(p) == FRAMES, "Expected one P slice per frame"
    return p


def cohort(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r["prompt_id"]].append(float(r[key]))
    return mean(mean(r) for r in groups.values())


def aligned(rows, arm):
    return {r["video_id"]:r for r in rows if r["arm"] == arm}


def diff_groups(group, base, key):
    groups = defaultdict(list)
    for row in group:
        groups[row["prompt_id"]].append(row[key] - base[row["video_id"]][key])
    return [mean(x) for x in groups.values()]


def association(frames, arm, x, y):
    videos = defaultdict(list)
    for row in frames:
        if row["arm"] == arm and row[x] is not None and row[y] is not None:
            videos[row["video_id"]].append([row[x],row[y]])
    centered = []
    per_clip = []
    for series in videos.values():
        a = np.log1p(np.array(series))
        a -= a.mean(0)
        if len(a) >= 2 and np.min(np.linalg.norm(a,axis=0)) > 1e-12:
            per_clip.append(float(np.corrcoef(a.T)[0,1]))
            centered.append(a)
    if not centered:
        return dict(within_video_pearson=None, median_video_pearson=None, videos=0)
    return dict(within_video_pearson=float(np.corrcoef(np.concatenate(centered).T)[0,1]),
                median_video_pearson=float(np.median(per_clip)), videos=len(per_clip))


def bar_svg(path, data):
    colors = ("#3266aa","#13a69a","#df674b","#e9a649","#8266a7","#abb7c6")
    names = ("Mode + Skip run","MV syntax","Luma coefficients","Chroma coefficients",
             "Other MB syntax","I frame + framing")
    top = math.ceil(max(sum(x.values()) for x in data.values()) / 1000) * 1000
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="410" viewBox="0 0 900 410">',
           '<rect width="900" height="410" fill="#fff"/>',
           '<g font-family="system-ui, sans-serif" fill="#182941">',
           '<text x="36" y="43" font-size="24" font-weight="700">A/F/E bit allocation</text>',
           '<text x="36" y="68" font-size="14">Validation · 16 videos, 4 equally weighted prompts · kb/s</text>']
    for i in range(5):
        px=100+630*i/4
        svg.append(f'<path d="M{px} 101v214" stroke="#dae2eb"/><text x="{px}" y="334" text-anchor="middle" font-size="12">{top*i/4:.0f}</text>')
    for i,(arm,bits) in enumerate(data.items()):
        y=121+64*i
        x=100
        svg.append(f'<text x="64" y="{y+24}" font-size="19" font-weight="700">{arm}</text>')
        for color, field in zip(colors,BIT_FIELDS):
            width=630*bits[field]/top
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="35" fill="{color}"/>')
            x+=width
        svg.append(f'<text x="{x+9:.3f}" y="{y+23}" font-size="14">{sum(bits.values()):,.1f}</text>')
    for i,(name,color) in enumerate(zip(names,colors)):
        x=38+280*(i%3); y=358+25*(i//3)
        svg.append(f'<rect x="{x}" y="{y-10}" width="13" height="13" fill="{color}"/><text x="{x+20}" y="{y+1}" font-size="13">{name}</text>')
    svg.append("</g></svg>")
    path.write_text("\n".join(svg))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    status=json.loads((args.run/"status.json").read_text())
    assert status==dict(total=60,failed=[])
    gate=yaml.safe_load((HERE.parent/"P0/config.yaml").read_text())["gate"]
    groups=records(CONFIGS)
    all_frames, videos, rd, rd_frames, local, manifest = [], [], [], [], [], []
    for split in ("train","val"):
        for row in (r for r in groups[split] if r["step"]==20):
            sample=row["sample_id"]
            folder=args.run/split/sample
            prompt=hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
            native=json.loads((folder/"summary.json").read_text())
            assert len(native)==21
            base=dict(split=split,prompt_id=prompt,video_id=sample,seed=row["seed"])
            labels=folder/"local_candidates.npz"
            manifest.append(dict(**base,diagnostic_oracle=True,labels=str(labels),sha256=digest(labels)))
            for arm, values in native.items():
                assert values["decoder_agreement"] and values["encoder_matches_decoder"]
                assert values["motion_search_s_reported"]==0 and values["oracle_p_bytes_read"]==0
                p_bits=frame_bits(folder/arm/"prediction.264")
                rd.append(dict(**base,arm=arm,psnr_y_db=values["psnr_y_p_db"],
                               bitrate_kbps=values["bitrate_kbps"],last4=mean(values["psnr_y_by_frame"][-4:]),
                               p16_fraction=values["p16_fraction"],diagnostic_oracle=values["diagnostic_oracle"],
                               three_way_pixel_equal=True,me_time_s=0,oracle_p_bytes_read=0,
                               stream_sha256=values["stream_sha256"],state_sha256=values["state_sha256"]))
                rd_frames.extend(dict(**base,arm=arm,frame=i+1,psnr_y_db=y,p_nal_bits=b)
                                 for i,(y,b) in enumerate(zip(values["psnr_y_by_frame"][1:],p_bits)))
            raw=list(csv.DictReader((folder/"per_frame.csv").open()))
            for arm in ("A","F","E"):
                series=[]
                p_bits=frame_bits(folder/arm/"prediction.264")
                for old in (r for r in raw if r["arm"]==arm):
                    f={k:float(v) if v else None for k,v in old.items()
                       if k not in ("arm","split","video_id")}
                    f["frame"]=int(f["frame"])
                    f["p_nal_bits"]=p_bits[f["frame"]-1]
                    f["p_framing_bits"]=f["p_nal_bits"]-f["mb_layer_bits"]
                    assert f["p_framing_bits"] >= 0
                    if f["frame"]>1:
                        identity=f["motion_error_clean_ref_mse_y"]+f["mc_reference_error_mse_y"]-2*f["reference_motion_cross_y"]
                        assert abs(identity-f["ideal_residual_mse_y"])<1e-8
                    series.append(dict(**base,arm=arm,**f))
                assert len(series)==16
                all_frames.extend(series)
                count=lambda k: int(sum(f[k] for f in series))
                avg=lambda k: mean(v for f in series if (v:=f[k]) is not None)
                total=8*native[arm]["bitstream_bytes"]
                bits={k:count(k) for k in BIT_FIELDS[:-1]}
                bits["i_and_container_bits"]=total-sum(bits.values())
                videos.append(dict(**base,arm=arm,bitrate_kbps=native[arm]["bitrate_kbps"],
                                   pframe_psnr_y_db=native[arm]["psnr_y_p_db"],
                                   **bits,**{k:avg(k) for k in METRICS},stream_bits=total))
            local.extend({**base, **r} for r in csv.DictReader((folder/"local_per_frame.csv").open()))
    val_rd=[r for r in rd if r["split"]=="val"]
    base=aligned(val_rd,"A")
    curves=[]
    for arm in native:
        group=[r for r in val_rd if r["arm"]==arm]
        q=cohort(group,"psnr_y_db")
        rate=cohort(group,"bitrate_kbps")
        q_gain=q-cohort(list(base.values()),"psnr_y_db")
        rate_gain=100*(rate/cohort(list(base.values()),"bitrate_kbps")-1)
        tail_gain=mean(diff_groups(group,base,"last4"))
        positive=mean(x>0 for x in diff_groups(group,base,"psnr_y_db"))
        eligible=(q_gain>=gate["min_psnr_gain_db_at_controlled_rate"] and
                  rate_gain<=gate["max_rate_increase_pct"] and tail_gain>=gate["min_last4_delta_db"] and
                  positive>=gate["min_positive_prompt_fraction"])
        curves.append(dict(arm=arm,psnr_y_db=q,bitrate_kbps=rate,delta_psnr_db=q_gain,
                           delta_rate_pct=rate_gain,delta_last4_db=tail_gain,
                           positive_prompt_fraction=positive,p16_fraction=cohort(group,"p16_fraction"),
                           diagnostic_oracle=group[0]["diagnostic_oracle"],passes=eligible))
    val=[r for r in videos if r["split"]=="val"]
    val_frames=[r for r in all_frames if r["split"]=="val" and r["frame"]>=2]
    bit_summary={arm:{k:cohort([r for r in val if r["arm"]==arm],k)*16/17000 for k in BIT_FIELDS}
                 for arm in ("A","F","E")}
    errors={arm:{k:cohort([r for r in val if r["arm"]==arm],k) for k in METRICS} for arm in bit_summary}
    associations={arm:{
        f"{x}_vs_{y}":association(val_frames,arm,x,y)
        for x,y in (("mc_reference_error_mse_y","residual_error_mse_y_p16"),
                    ("mc_reference_error_mse_y","ideal_residual_mse_y_p16"),
                    ("motion_error_clean_ref_mse_y","ideal_residual_mse_y_p16"),
                    ("ideal_residual_mse_y_p16","residual_coeff_bits"))} for arm in bit_summary}
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in (("per_frame",all_frames),("per_video",videos),("rd_curve",curves),
                      ("rd_per_video",rd),("rd_per_frame",rd_frames),("local_per_frame",local)):
        csv_write(args.output/f"{name}.csv",data)
    (args.output/"candidate_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in manifest))
    bar_svg(args.output/"bit_allocation.svg",bit_summary)
    gain=sum(bit_summary["E"].values())-sum(bit_summary["A"].values())
    residual_delta=sum(bit_summary["E"][k]-bit_summary["A"][k] for k in BIT_FIELDS[2:4])
    (args.output/"summary.json").write_text(json.dumps(dict(
        train_videos=44,val_videos=16,val_prompts=4,curves=curves,bit_kbps=bit_summary,
        residual_share_of_e_minus_a=residual_delta/gain,errors=errors,associations=associations,
        passing_joint=[r["arm"] for r in curves if r["arm"].startswith("joint") and r["passes"]]
    ),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
