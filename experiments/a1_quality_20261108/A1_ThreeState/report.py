"""Summarize verified replays and package macOS-compatible views."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean

ARMS = "ABCDEF"
LABELS = {
    "A": "Pred Mode + MV + Residual",
    "B": "Oracle Mode",
    "C": "Oracle MV on common P16",
    "D": "Oracle Residual on common P16",
    "E": "Source - live MC",
    "F": "Full JM reference",
}
# Historical camera-motion example, fixed-camera metronome and frozen A worst example.
SELECTED = ("sample_1032", "sample_1024", "sample_1029")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(manifest, output):
    clips = manifest["videos"]
    assert len(clips) == 16
    rows, frames = [], []
    for clip in clips:
        for arm, quality in clip["qualities"].items():
            curve = quality["psnr_y_by_frame"]
            assert len(curve) == 17
            assert quality["encoder_matches_decoder"] and quality["decoder_agreement"]
            rows.append(dict(sample=clip["sample"], seed=clip["seed"], prompt=clip["prompt"], arm=arm,
                             p_y_psnr_db=quality["psnr_y_p_db"], bitrate_kbps=quality["bitrate_kbps"],
                             last4_db=mean(curve[-4:]), last_frame_db=curve[-1],
                             last4_minus_first4_p_db=mean(curve[-4:]) - mean(curve[1:5]),
                             stream_bytes=quality["bitstream_bytes"],
                             hidden_load_s=quality.get("load_latent_s"),
                             heads_s=quality.get("head_seconds"), adapter_s=quality.get("adapt_s"),
                             jm_s=quality.get("jm_replay_s", quality.get("jm_encode_s")),
                             motion_search_s=quality["motion_search_s_reported"],
                             ffmpeg_decode_s=quality["ffmpeg_decode_s"], wall_s=quality["arm_wall_s"],
                             residual_syntax_bits=quality.get("residual_syntax_bits"),
                             bitstream_sha256=clip["stream_sha256"][arm], **clip["adaptation"]))
            frames.extend(dict(sample=clip["sample"], arm=arm, frame=idx, kind="P" if idx else "I",
                               pts_s=idx/16, y_psnr_db=db,
                               yuv420_frame_sha256=quality["decoded_frame_sha256"][idx])
                          for idx, db in enumerate(curve))
    write_csv(output / "per_video.csv", rows)
    write_csv(output / "per_frame.csv", frames)
    selected = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    averages = {}
    for arm, subset in selected.items():
        worst = min(subset, key=lambda r: r["p_y_psnr_db"])
        largest_drop = min(zip(subset, selected["A"]),
                           key=lambda pair: pair[0]["p_y_psnr_db"] - pair[1]["p_y_psnr_db"])
        averages[arm] = {key: mean(r[key] for r in subset) for key in
                         ("p_y_psnr_db", "bitrate_kbps", "last4_db", "last_frame_db",
                          "last4_minus_first4_p_db", "jm_s", "wall_s", "motion_search_s")}
        averages[arm].update(worst=worst["sample"], worst_psnr=worst["p_y_psnr_db"],
                             max_drop_sample=largest_drop[0]["sample"],
                             max_drop_db=largest_drop[0]["p_y_psnr_db"]-largest_drop[1]["p_y_psnr_db"])
    curves = {a: [mean(r["y_psnr_db"] for r in frames if r["arm"] == a and r["frame"] == i)
                  for i in range(17)] for a in ARMS}
    write_csv(output / "mean_curve.csv", [dict(frame=i, pts_s=i/16, **{a: c[i] for a, c in curves.items()})
                                         for i in range(17)])
    lo = 4 * int(min(map(min, curves.values())) // 4)
    hi = 4 * (1 + int(max(map(max, curves.values())) // 4))
    xx = lambda i: 72 + i * 48
    yy = lambda db: 365 - (db-lo) * 310 / (hi-lo)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="425" viewBox="0 0 900 425">',
           '<rect width="900" height="425" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#172033">',
           '<text x="72" y="24" font-size="18">Validation mean frame Y-PSNR (dB)</text>']
    for y in range(lo, hi + 1, 4):
        svg.append(f'<path d="M72 {yy(y)}H840" stroke="#ddd"/><text x="41" y="{yy(y)+4}">{y}</text>')
    for i in range(17):
        svg.append(f'<text x="{xx(i)}" y="387" text-anchor="middle">{i}</text>')
    svg.append('<text x="455" y="410" text-anchor="middle">Frame (0 = shared I)</text>')
    for i, (arm, color) in enumerate(zip(ARMS, ("#30343b", "#936215", "#0963be", "#9b3881", "#04845d", "#d14828"))):
        points = " ".join(f"{xx(j)},{yy(db)}" for j, db in enumerate(curves[arm]))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{520+i*54}" y="27" fill="{color}">{arm}</text>')
    (output / "psnr.svg").write_text("\n".join([*svg, "</g></svg>"]))
    return rows, averages, curves


def media(root, output, rows):
    commands = []

    def ff(args):
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *map(str, args)]
        commands.append(cmd)
        subprocess.run(cmd, check=True)

    def probe(path, width, height, frames):
        cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_streams", "-of", "json", str(path)]
        commands.append(cmd)
        info, = json.loads(subprocess.check_output(cmd))["streams"]
        assert (info["codec_name"], info["codec_tag_string"], info["pix_fmt"]) == ("h264", "avc1", "yuv420p")
        assert (info["width"], info["height"], info["avg_frame_rate"], int(info["nb_read_frames"])) == (
            width, height, "16/1", frames), info
        assert abs(float(info["duration"]) - frames/16) < 1e-4

    parts = []
    clips = {(r["sample"], r["arm"]): r for r in rows}
    for sample in SELECTED:
        mp4 = {}
        for arm in ARMS:
            bit = root / sample / arm / "prediction.264"
            mp4[arm] = output / f"{sample}_{arm}.mp4"
            ff(["-r", 16, "-i", bit, "-map", "0:v:0", "-an", "-c:v", "copy",
                "-bsf:v", "setts=pts=N*1000:dts=N*1000:duration=1000:time_base=1/16000",
                "-video_track_timescale", 16000, "-tag:v", "avc1", "-movflags", "+faststart", mp4[arm]])
            probe(mp4[arm], 832, 480, 17)
            decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(mp4[arm]),
                                                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
            with (bit.parent / "decoded_ffmpeg.yuv").open("rb") as expected:
                assert hashlib.sha256(decoded).hexdigest() == hashlib.file_digest(expected, "sha256").hexdigest()

        for group, title in [(ARMS, "six"), ("AE", "ae")]:
            filters = []
            for idx, arm in enumerate(group):
                r = clips[sample, arm]
                label = f"{sample}  {arm}  {LABELS[arm]}  frame %{{n}}"
                stat = (f'P Y-PSNR {r["p_y_psnr_db"]:.2f} dB  |  {r["bitrate_kbps"]:.1f} kb/s  |  '
                        f'last 4 {r["last4_db"]:.2f} dB')
                filters.append(
                    f"[{idx}:v]setpts=PTS-STARTPTS,pad=832:528:0:48:black,"
                    f"drawtext=text='{label}':x=12:y=4:fontsize=20:fontcolor=white,"
                    f"drawtext=text='{stat}':x=12:y=28:fontsize=16:fontcolor=white[v{idx}]")
            layout = "|".join(f"{i%3*832}_{i//3*528}" for i in range(len(group)))
            filters.append("".join(f"[v{i}]" for i in range(len(group))) +
                           f"xstack=inputs={len(group)}:layout={layout}:shortest=1[stack]")
            x = 0 if title == "six" else 416
            filters.append(f"[stack]pad=2496:1056:{x}:(oh-ih)/2:black[out]")
            target = output / f"{sample}_{title}.mp4"
            ff([*(arg for a in group for arg in ("-i", mp4[a])), "-filter_complex_threads", 2,
                "-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
                "-threads", 4, "-bf", 0, "-preset", "fast", "-crf", 14, "-pix_fmt", "yuv420p",
                "-video_track_timescale", 16000, "-movflags", "+faststart", target])
            probe(target, 2496, 1056, 17)
            parts.append(target)
    concat = output / "concat.txt"
    concat.write_text("".join(f"file '{p.name}'\nduration 1.0625\n" for p in parts))
    reel = output / "a1_three_state_comparison.mp4"
    ff(["-f", "concat", "-safe", 1, "-i", concat, "-c", "copy",
        "-video_track_timescale", 16000, "-movflags", "+faststart", reel])
    probe(reel, 2496, 1056, 102)
    (output / "media_commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    manifest = json.loads((args.full / "run_manifest.json").read_text())
    rows, averages, curves = metrics(manifest, out)
    (out / "summary.json").write_text(json.dumps(dict(averages=averages, curves=curves,
                                                     wall_s=manifest["wall_s"]), indent=2) + "\n")
    media(args.full.resolve(), out, rows)


if __name__ == "__main__":
    main()
