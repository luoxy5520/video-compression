"""Render labeled four-panel previews from actual scored H.264 replays."""
import json
import subprocess
import sys
from pathlib import Path

from d0 import p

root, out = [Path(arg).resolve() for arg in sys.argv[1:]]
out.mkdir(parents=True, exist_ok=True)
clips = json.loads((root / "d1/d1.json").read_text())
rows = {r["sample_id"]: r for r in p.rows_by_split()["val"]}
raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", "832x480", "-framerate", "16"]
delta = lambda r: r["arms"]["SS"]["psnr_y_p_db"] - r["A"]["psnr_y_p_db"]


def render(path, inputs, captions):
    args = ["ffmpeg", "-nostdin", "-loglevel", "error", "-n"]
    for source in inputs:
        args += (raw if source.suffix == ".yuv" else ["-framerate", "16"]) + ["-i", str(source)]
    panels = [f"[{i}:v]setpts=N/(16*TB),drawtext=text='{name}':x=12:y=12:"
              f"fontsize=24:fontcolor=white:box=1:boxcolor=black@0.68[p{i}]"
              for i, name in enumerate(captions)]
    layout = "[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0"
    args += ["-filter_complex", ";".join([*panels, layout]), "-frames:v", "17",
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-movflags", "+faststart", str(path)]
    subprocess.run(args, check=True)


def inputs(clip):
    sid = clip["sample"]
    _, source = p.paths("val", rows[sid])
    base = root / "run02/A" / sid / "prediction.264"
    tt = root / "d1" / sid / str(clip["budgets"]["TT"]) / "TT/prediction.264"
    return source, base, tt


for name, clip in (("closest", min(clips, key=lambda r: abs(delta(r)))), ("worst", min(clips, key=delta))):
    sid = clip["sample"]
    ss = root / "d1" / sid / str(clip["budgets"]["SS"]) / "SS/prediction.264"
    render(out / f"d1_{name}_{sid}.mp4", (*inputs(clip), ss),
           ("Source", "Frozen A", "D1 TT - source-assisted", "D1 SS - student"))

experiments = [json.loads(f.read_text()) for f in sorted(root.glob("G[01]-*/metrics.json"))]
if experiments:
    rd, probe, run = max((d["rd"][k]["delta_db"], k, f'{d["group"]}-{d["seed"]}')
                         for d in experiments for k in ("C","R"))
    best = next(d for d in experiments if f'{d["group"]}-{d["seed"]}' == run)
    nearest = min((v[probe] for v in best["val"] if v[probe]["luma_nonzero"]),
                   key=lambda r: abs(r["delta_db"]))
    sid = nearest["sample"]
    clip = next(c for c in clips if c["sample"] == sid)
    student = root / run / sid / f"{probe}_64" / sid / "prediction.264"
    render(out / f"probe_{run}_{probe}_{sid}.mp4", (*inputs(clip), student),
           ("Source", "Frozen A", "D1 TT - source-assisted", f"{run} Probe {probe} - student"))
