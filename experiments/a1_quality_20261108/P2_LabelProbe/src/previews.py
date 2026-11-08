"""Render aligned four-panel previews of measured streams."""
import argparse
import json
from pathlib import Path
import subprocess

from l0 import p

RAW=["-f","rawvideo","-pix_fmt","yuv420p","-video_size","832x480","-framerate","16"]


def render(dest,inputs,captions):
    command=["ffmpeg","-nostdin","-v","error","-n"]
    for path in inputs:
        command+=(RAW if path.suffix==".yuv" else ["-framerate","16"])+["-i",str(path)]
    panels=[f"[{i}:v]setpts=N/(16*TB),drawtext=text='{name}':x=12:y=12:fontsize=24:"
            f"fontcolor=white:box=1:boxcolor=black@0.68[p{i}]" for i,name in enumerate(captions)]
    panels.append("[p0][p1][p2][p3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0")
    subprocess.run(command+["-filter_complex",";".join(panels),"-frames:v","17","-an","-c:v","libx264",
        "-preset","fast","-crf","18","-movflags","+faststart",str(dest)],check=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("run",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    scored=[(f.parent.name,point) for f in sorted(args.run.glob("l2-*/run.json"))
            for point in json.loads(f.read_text())["points"] if point["rate_pct"]<=5]
    rows=p.rows_by_split()
    val={r["sample_id"]:r for r in rows["val"]}
    for name,key in (("best",max),("worst",min)):
        run,point=key(scored,key=lambda pair:pair[1]["delta_db"])
        sid=point["sample"]
        root,source=p.paths("val",val[sid])
        teacher=args.run/"l1"/sid/"A"/sid/"prediction.264"
        student=args.run/run/sid/"prediction.264"
        render(args.out/f"{name}_{sid}.mp4",(source,root/"A/prediction.264",teacher,student),
               ("Source","Frozen A","L1 offline source-assisted",f"{run} student"))
    row=rows["train"][0]
    sid=row["sample_id"]
    root,source=p.paths("train",row)
    render(args.out/f"training_fit_{sid}.mp4",(source,root/"A/prediction.264",
           args.run/"l0-0-A-1701"/sid/"prediction.264",args.run/"l0-0-B-1701"/sid/"prediction.264"),
           ("Source","Frozen A","L0-A same-video fit","L0-B same-video fit"))


if __name__=="__main__":
    main()
