"""Generate compact P2-Lite tables, an RD figure, and aligned review MP4s."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, median

import torch

from p2lite import CFG, P06, ROOT, paths, rows_by_split, save_json


def read(path):
    return json.loads(path.read_text())


def mean_prompt(group, key):
    prompts = {r["prompt"] for r in group}
    return mean(mean(r[key] for r in group if r["prompt"] == p) for p in prompts)


def compare(output, review, row, model, policy, budget, name):
    root, source = paths("val", row)
    teacher = root / CFG["teacher"] / "prediction.264"
    if not teacher.exists():
        meta = next(r for r in read(root / "summary.json") if r["arm"] == CFG["teacher"])
        teacher = root / meta["fallback"] / "prediction.264"
    anchor = root / "A" / "prediction.264"
    student = output / model / f"{policy}_{budget}" / row["sample_id"] / "prediction.264"
    filters = ";".join(
        f"[{i}:v]setpts=N/(16*TB),setsar=1,drawtext=text='{label}':x=14:y=14:"
        f"fontsize=24:fontcolor=white:box=1:boxcolor=black@0.8:boxborderw=6[v{i}]"
        for i, label in enumerate(("Source", "A frozen hidden", "P0.6 offline diagnostic", f"P2-Lite {model}"))
    )
    filters += ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex_threads", "1",
                    "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "832x480",
                    "-framerate", "16", "-i", str(source),
                    *[arg for path in (anchor,teacher,student) for arg in ("-r","16","-i",str(path))],
                    "-filter_complex",filters,"-map","[out]","-frames:v","17","-r","16","-an",
                    "-c:v","libx264","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",
                    str(review / f"{name}_{row['sample_id']}.mp4")], check=True)


def figure(dest, points, anchors):
    xmin, xmax = 436, 465
    visible = [r for r in points if r["bitrate_kbps"] <= xmax]
    offscreen = len(points) - len(visible)
    ymin, ymax = min(19.6, min(r["psnr_y_p_db"] for r in visible) - .1), 21.25

    def xy(rate, db):
        return 86 + 710*(rate-xmin)/(xmax-xmin), 365 - 270*(db-ymin)/(ymax-ymin)

    content = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="450" viewBox="0 0 850 450">',
               '<rect width="850" height="450" fill="white"/>',
               '<g font-family="sans-serif" fill="#142338">',
               '<text x="45" y="42" font-size="24" font-weight="bold">P2-Lite · actual H.264 rate–distortion</text>',
               '<text x="45" y="67" font-size="13">16 validation clips · 4 equal prompt weights · JM QP28</text>']
    for tick in range(440, int(xmax), 5):
        x, _ = xy(tick, ymin)
        content.append(f'<path d="M{x} 95V365" stroke="#dfe7ef"/><text x="{x}" y="385" text-anchor="middle">{tick}</text>')
    for tick in (19,19.5,20,20.5,21):
        if tick < ymin:
            continue
        _, y = xy(xmin, tick)
        content.append(f'<path d="M86 {y}H796" stroke="#dfe7ef"/><text x="76" y="{y+4}" text-anchor="end">{tick}</text>')
    for rate in (anchors["A"]["bitrate_kbps"]*1.03,anchors["A"]["bitrate_kbps"]*1.05):
        x, _ = xy(rate, ymin)
        content.append(f'<path d="M{x} 95V365" stroke="#8a5362" stroke-dasharray="4 4"/>')
    colors = dict(dc="#3266aa", low="#0b877b")
    for row in visible:
        x,y = xy(row["bitrate_kbps"], row["psnr_y_p_db"])
        color = colors[row["run"].split("-")[0]]
        fill = color if row["primary_per_clip_cap"] else "white"
        content.append(f'<circle cx="{x}" cy="{y}" r="5" stroke="{color}" fill="{fill}"><title>'
                       f'{row["run"]} {row["policy"]} {row["budget"]}: {row["delta_db"]:+.3f} dB</title></circle>')
    for label, row in anchors.items():
        x,y=xy(row["bitrate_kbps"],row["psnr_y_db"])
        content.append(f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="#492a48"/>'
                       f'<text x="{x+8}" y="{y-6}" font-size="12">{label}</text>')
    content += ['<text x="450" y="410" text-anchor="middle" font-size="14">Actual bitrate (kb/s)</text>',
                '<text x="20" y="220" font-size="14" transform="rotate(-90 20 220)">P-frame Y-PSNR (dB)</text>',
                '<text x="88" y="435" font-size="12"><tspan fill="#3266aa">DC</tspan> · '
                '<tspan fill="#0b877b">DC + low</tspan> · hollow = over +3% on at least one clip</text>',
                f'<text x="796" y="435" text-anchor="end" font-size="12">{offscreen} high-rate points in RD table</text></g></svg>']
    dest.write_text("\n".join(content))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--videos", action="store_true")
    a=p.parse_args()
    report=a.output / "report"
    report.mkdir(exist_ok=True)
    points, best, frames = [], [], []
    for path in sorted(a.output.glob("*/rd.json")):
        run=path.parent.name
        metrics=read(path.with_name("metrics.json"))
        rows=[dict(r,run=run) for r in read(path)]
        points.extend(rows)
        for r in metrics:
            for frame, (psnr, nnz, bits) in enumerate(zip(r["psnr_y_by_frame"][1:],r["frame_luma_nonzero"],r["frame_coeff_bits"]),1):
                frames.append(dict(run=run, sample=r["sample"], policy=r["policy"], budget=r["budget"],
                                   frame=frame, psnr_y_db=psnr, luma_nonzero=nnz, residual_coeff_bits=bits))
        eligible=[r for r in rows if r["primary_per_clip_cap"]]
        nonzero=[r for r in eligible if any(m["luma_nonzero"] for m in metrics
                                           if (m["policy"],m["budget"])==(r["policy"],r["budget"]))]
        winner=max(nonzero or eligible,key=lambda r:r["delta_db"])
        group=[r for r in metrics if (r["policy"],r["budget"])==(winner["policy"],winner["budget"])]
        params=torch.load(path.with_name("model.pt"),weights_only=True)["params"]
        best.append(dict(winner, params=params,
                         coeff_bits=mean_prompt(group,"coeff_bits"), luma_nonzero=mean_prompt(group,"luma_nonzero"),
                         last_frame_delta_db=mean_prompt([dict(r,last_delta=r["last_frame_psnr_y"]-
                             next(m for m in read(P06/"val"/r["sample"]/"summary.json") if m["arm"]=="A")["last_frame_psnr_y"])
                             for r in group],"last_delta"),
                         head_ms_median=1000*median(r["student_head_seconds"] for r in group),
                         head_mib_peak=max(r["student_peak_extra_bytes"] for r in group)/2**20,
                         selected_true_positive=sum(r["selected_true_positive"] for r in group),
                         total_selected=sum(r["luma_nonzero"] for r in group),
                         teacher_positive=sum(r["teacher_positive"] for r in group),
                         teacher_pos_sign_correct=sum(r["teacher_pos_sign_correct"] for r in group),
                         teacher_pos_magnitude_correct=sum(r["teacher_pos_magnitude_correct"] for r in group),
                         video=[dict(sample=r["sample"],delta_db=r["delta_db"],delta_last4_db=r["delta_last4_db"],
                                     rate_pct=r["rate_pct"]) for r in group]))
    with (report / "per_frame.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames)
    with (report / "rd.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(dict(r,prompt_delta_db=json.dumps(r["prompt_delta_db"])) for r in points)
    with (ROOT/"P0_6/report/rd_curve.csv").open() as f:
        reference={r["arm"]:{k:float(r[k]) for k in ("bitrate_kbps","psnr_y_db")} for r in csv.DictReader(f)}
    figure(report/"rd.svg",points,{"A":reference["A"],"B +3% offline":reference["offline_gain_3"],
                                  "B +5% offline":reference["early_5"]})
    save_json(report/"summary.json",dict(best_nonzero=best, replay_count=len(frames)//16,
              passed=[dict(r) for r in points if r["passed"]], reference=reference))
    if a.videos:
        review=report/"videos"
        review.mkdir(exist_ok=True)
        winner=max(best,key=lambda r:r["delta_db"])
        ranked=sorted(winner["video"],key=lambda r:r["delta_db"])
        rows={row["sample_id"]:row for row in rows_by_split()["val"]}
        samples={"worst":ranked[0]["sample"],"best":ranked[-1]["sample"]}
        for label, phrase in (("occlusion","skater"),("fixed_camera","metronome"),("camera_motion","market")):
            row=next(row for row in rows.values() if phrase in row["prompt"])
            if row["sample_id"] not in samples.values():
                samples[label]=row["sample_id"]
        for name,sample in samples.items():
            compare(a.output,review,rows[sample],winner["run"],winner["policy"],winner["budget"],name)


if __name__ == "__main__":
    main()
