"""Render browser-ready comparison grids from the scored YUV and decoded outputs."""
import argparse
import json
from pathlib import Path
import subprocess


def render(output, panels):
    command = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-y","-filter_complex_threads","1"]
    for label, path in panels:
        if path.stat().st_size != 17*832*480*3//2:
            raise ValueError(f"Expected 17 frames: {path}")
        command += ["-f","rawvideo","-pix_fmt","yuv420p","-video_size","832x480",
                    "-framerate","16","-i",str(path)]
    graph = []
    for i,(label,_) in enumerate(panels):
        if not all(c.isalnum() or c in " .+-/" for c in label):
            raise ValueError(f"Unsupported preview label: {label}")
        graph.append(f"[{i}:v]scale=416:240:flags=lanczos,pad=416:272:0:32:color=0x111827,"
                     f"drawtext=font=DejaVu Sans:text='{label}':fontcolor=white:fontsize=16:x=12:y=8[p{i}]")
    cols = 3 if len(panels) in (3,6) else 2
    layout = "|".join(f"{i%cols*416}_{i//cols*272}" for i in range(len(panels)))
    graph.append("".join(f"[p{i}]" for i in range(len(panels)))+
                 f"xstack=inputs={len(panels)}:layout={layout},setpts=3*PTS,fps=16[v]")
    temp = output.with_suffix(".tmp.mp4")
    subprocess.run(command+["-filter_complex",";".join(graph),"-map","[v]","-an","-c:v","libx264",
                            "-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",
                            str(temp)],check=True)
    temp.replace(output)


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    report = json.loads((a.run/"results.json").read_text())
    records = { (r["sample_id"],r["variant"]):r for r in report["per_video"] if r.get("split","test")=="test" }
    if a.experiment=="qpel":
        selected = report["calibration"]["selected"]
        cases = [("Full local directions","teacher")]
        if selected:
            cases.append((f"Learned top {selected}",f"top{selected}"))
    elif a.experiment=="exact":
        if not a.previous:
            raise ValueError("Exact Skip previews require --previous")
        for sid in sorted({sid for sid,_ in records}):
            r = records[sid,"refine"]
            panels = [("Source YUV",a.data/sid/"final_raw.yuv"),
                      (f"Before  {r['old_bitrate_kbps']:.1f} kbps",
                       a.previous/"samples"/sid/"refine/decoded_ffmpeg.yuv"),
                      (f"Exact Skip  {r['bitrate_kbps']:.1f} kbps",
                       a.run/"samples"/sid/"refine/decoded_ffmpeg.yuv")]
            render(a.output/f"{sid}.mp4",panels)
        return
    elif a.experiment=="latent":
        tag = report["calibration"]["selected"]
        cases = [("Single MV","top1"),("Local qpel refine","refine")]
        if tag not in ("top1","refine"):
            cases.append(("Calibrated Skip",tag))
    elif a.experiment=="skip":
        selected = report["calibration"]["selected"]
        tag = f"mse_{selected}" if isinstance(selected,int) else selected
        cases = [("Fixed Skip","fixed")]
        if tag != "all_p16":
            cases.append(("Calibrated gate",tag))
        cases.append(("All P16","all_p16"))
    else:
        cases = [("Old replay","baseline"),("New Skip only","mode_only"),("New MV only","mv_only"),
                 ("New mode and MV","mode_mv"),("Live residual","learned_residual")]
    for sid in sorted({sid for sid,_ in records}):
        panels = [("Source YUV",a.data/sid/"final_raw.yuv")]
        for label,tag in cases:
            score = records[sid,tag]["psnr_y_p_db"]
            panels.append((f"{label}  {score:.2f} dB",a.run/"samples"/sid/tag/"decoded_ffmpeg.yuv"))
        render(a.output/f"{sid}.mp4",panels)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("run","data","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--experiment",choices=("replay","skip","latent","exact","qpel"),required=True)
    main(parser.parse_args())
