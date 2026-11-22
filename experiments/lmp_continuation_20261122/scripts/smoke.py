"""Run the frozen two-clip development comparison on 113Server."""
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from fractions import Fraction

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HIST = Path.home() / "lmp-e1-e2-followup-20261121"
PRIOR = Path.home() / "lmp-h264-prior.uFe5ZV"
KIT = Path.home() / "a_new48_20261107/kit"
CODEC = ROOT / "jm/bin/lencod.exe"
DEC = KIT / "linux/JM/bin/ldecod.exe"
BASE_CFG = KIT / "JM-backend/cfg/encoder_baseline.cfg"
IP_CFG = KIT / "linux/skip_p16.cfg"
DEC_CFG = KIT / "JM-backend/cfg/decoder.cfg"
POLICY = json.loads((ROOT / "configs/policy.json").read_text())
ARMS = {
    "C0": 0, "C-pixel-fixed": 4, "C1-fixed": 5, "C2-fixed": 6,
    "C-pixel-adaptive": 1, "C1-seed": 2, "C2-adaptive": 3,
}
ENV = {key: val for key, val in os.environ.items()
       if not key.startswith(("JM_", "LMP_"))}
ENV.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
torch.set_num_threads(1)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append(path, row):
    with path.open("a") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args, folder, name, env=None, outputs=()):
    args = [str(arg) for arg in args]
    log_stdout, log_stderr = (folder / f"{name}.{stream}" for stream in ("stdout", "stderr"))
    begin = time.perf_counter()
    stamp = datetime.now().astimezone().isoformat()
    with log_stdout.open("wb") as out, log_stderr.open("wb") as err:
        proc = subprocess.run(args, cwd=folder, stdout=out, stderr=err, env=env or ENV)
    wall = time.perf_counter() - begin
    return (proc.returncode, wall, {
        "timestamp": stamp, "purpose": name, "cwd": str(folder), "command": args,
        "exit_code": proc.returncode, "stdout": str(log_stdout), "stderr": str(log_stderr),
        "duration_s": wall, "produced": [str(path) for path in outputs if path.exists()],
        "status": "success" if proc.returncode == 0 else "failed",
        "experiment_environment": {k: v for k, v in (env or ENV).items() if k.startswith("LMP_")},
    })


def record(result):
    journal = ROOT / "logs/commands.jsonl"
    status, wall, row = result
    with journal.open() as file:
        row["sequence"] = sum(1 for _ in file) + 1
    append(journal, row)
    if status:
        raise RuntimeError(f"{row['purpose']} failed, see {row['stderr']}")
    return wall


def prior(latent, target, probe):
    start = time.perf_counter()
    z = probe.tensor(latent).numpy()
    assert z.shape == (16, 5, 60, 104)
    loaded = time.perf_counter()
    z = z.reshape(16, 5, 30, 2, 52, 2).transpose(1, 2, 4, 3, 5, 0).reshape(5, 30, 52, 64)
    z = probe.normalization(z)
    normalized = time.perf_counter()
    offsets = np.stack([probe.match(z[j], z[j - 1]) for j in range(1, 5)])
    matched = time.perf_counter()
    (16 * offsets).repeat(4, axis=0).astype("<i2").tofile(target)
    end = time.perf_counter()
    return dict(prior_s=end-start, prior_read_s=loaded-start,
                prior_normalize_s=normalized-loaded, prior_match_s=matched-normalized,
                prior_write_s=end-matched)


def verify(folder, raw, bit, recon, media):
    jm, ff = folder / "jm.yuv", folder / "ffmpeg.yuv"
    record(run([DEC, "-d", DEC_CFG, "-p", f"InputFile = {bit}",
                "-p", f"OutputFile = {jm}", "-p", 'RefFile = ""'],
               folder, "decode-jm-verified", outputs=[jm]))
    record(run(["ffmpeg", "-v", "error", "-y", "-i", bit,
                "-pix_fmt", "yuv420p", "-f", "rawvideo", ff],
               folder, "decode-ffmpeg-verified", outputs=[ff]))
    muxed = media or folder / "verification.mp4"
    record(run(["ffmpeg", "-v", "error", "-y", "-framerate", "16", "-i", bit,
                "-c", "copy", muxed], folder, "remux", outputs=[muxed]))
    record(run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,nb_read_frames,avg_frame_rate",
                "-of", "json", muxed], folder, "probe-mp4"))
    stream = json.loads((folder / "probe-mp4.stdout").read_text())["streams"][0]
    assert stream["codec_name"] == "h264" and Fraction(stream["avg_frame_rate"]) == 16, stream
    assert (stream["width"], stream["height"], int(stream["nb_read_frames"])) == (832, 480, 17)
    digest = sha(ff)
    assert digest == sha(recon) == sha(jm)
    luma = lambda path: np.fromfile(path, np.uint8).reshape(17, -1)[:, :832*480]
    mse = np.square(luma(raw).astype(np.float64) - luma(ff)).mean(axis=1)
    return dict(decoded_sha256=digest, decoded_frames=17, fps=16, width=832, height=480,
                dual_decode=True, bitrate_kbps=bit.stat().st_size * 8 * 16 / 17 / 1000,
                **{key: float(10*np.log10(255**2/mse[window].mean())) for key, window in {
                    "all_psnr": slice(None), "p_psnr": slice(1, None),
                    "last4_psnr": slice(-4, None)}.items()})


def main():
    import sys
    sys.path.insert(0, str(PRIOR))
    import probe

    audit = json.loads((HIST / "supplement-analysis/input_audit.json").read_text())
    legacy = {row["sample"]: row["sha256"] for row in map(
        json.loads, (PRIOR / "baseline.log").read_text().splitlines()) if row["mode"] == 3}
    cfg_hashes = {str(path): sha(path) for path in (BASE_CFG, IP_CFG, DEC_CFG)}
    binary_sha = sha(CODEC)
    write_json(ROOT / "logs/environment.json", dict(
        hostname=platform.node(), kernel=platform.release(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        gpu_used=False, encoder_sha256=binary_sha, decoder_sha256=sha(DEC),
        configs=cfg_hashes, historical_base="f080a451831e89a59948006c4299bca698fdfef6",
        jm_patch_sha256=sha(ROOT / "configs/jm_fixed_search.patch"),
        runner_sha256=sha(Path(__file__)), prior_generator_sha256=sha(PRIOR / "probe.py"),
        prior_storage_bytes=16*30*52*8*2*2, encoder_threads=1))
    summaries = []
    repeats_path = ROOT / "results/repeats.jsonl"
    finished = {}
    if repeats_path.exists():
        finished = {row["run_id"]: row for row in map(json.loads, repeats_path.read_text().splitlines())}
    for sample in POLICY["samples"]:
        files = {row["kind"]: row for row in audit if row["sample"] == sample}
        raw, latent = (Path(files[k]["path"]) for k in ("raw", "latent"))
        meta = json.loads((raw.parent / "metadata.json").read_text())
        assert sha(raw) == sha(latent.parent / "final_raw.yuv") == files["raw"]["sha256"]
        assert sha(latent) == files["latent"]["sha256"]
        rows = {arm: [] for arm in ARMS}
        sample_folder = ROOT / "logs/runs" / sample
        sample_folder.mkdir(exist_ok=True)
        for repeat in range(POLICY["repeats_after_warmup"] + 1):
            # Rotate ordering to reduce correlation with drift and keep timings serial.
            arms = list(ARMS)
            for arm in arms[repeat:] + arms[:repeat]:
                mode = ARMS[arm]
                folder = sample_folder / arm
                folder.mkdir(exist_ok=True)
                bit, recon, candidates = (folder / name for name in
                                           ("stream.264", "recon.yuv", "prior.bin"))
                params = dict(InputFile=raw, OutputFile=bit, ReconFile=recon,
                              SourceWidth=832, SourceHeight=480, FramesToBeEncoded=17,
                              FrameRate=16, QPISlice=POLICY["qp"], QPPSlice=POLICY["qp"],
                              RDOptimization=0, NumberReferenceFrames=1,
                              PList0References=1, SearchMode=3, SearchRange=POLICY["range_px"])
                args = [CODEC, "-d", BASE_CFG, "-f", IP_CFG]
                for k, v in params.items():
                    args.extend(["-p", f"{k} = {v}"])
                env = dict(ENV, LMP_MODE=str(mode))
                use_prior = mode in (2, 3, 5, 6)
                if use_prior:
                    env["LMP_CANDIDATES"] = str(candidates)
                name = f"encode-{repeat}"
                run_id = f"lmp-continuation-20261122-{sample}-{arm}-{repeat}"
                if run_id in finished:
                    existing = finished[run_id]
                    assert existing["encoder_sha256"] == binary_sha
                    assert existing["config_sha256"] == cfg_hashes
                    assert existing["source_sha256"] == files["raw"]["sha256"]
                    assert existing["bitstream_sha256"] == sha(bit)
                    rows[arm].append(existing)
                    continue
                start = time.perf_counter()
                phase = prior(latent, candidates, probe) if use_prior else dict(prior_s=0)
                measured = run(args, folder, name, env, [bit, recon])
                pipeline_s = time.perf_counter() - start
                encoder_s = record(measured)
                profile = re.search(r"^LMP_PROFILE (\{.*\})$", (folder / f"{name}.stderr").read_text(), re.M)
                assert profile, folder
                detail = json.loads(profile.group(1))
                if mode in (4, 5, 6):
                    assert detail["integer_evals"] == 18*16*30*52, detail
                row = dict(run_id=run_id, experiment_id="lmp-continuation-20261122",
                           sample=sample, arm=arm, repeat=repeat, warmup=repeat == 0,
                           dataset="historical_dev", seed=meta["pairing"]["seed"],
                           source_sha256=files["raw"]["sha256"], source_path=str(raw),
                           latent_sha256=files["latent"]["sha256"] if use_prior else None,
                           candidate_sha256=sha(candidates) if use_prior else None,
                           config_sha256=cfg_hashes, encoder_sha256=binary_sha,
                           params={k: str(v) for k, v in params.items()}, mode=mode,
                           pipeline_s=pipeline_s, encoder_s=encoder_s,
                           bitstream_sha256=sha(bit), bitstream_path=str(bit), **phase, **detail)
                if arm == "C0":
                    assert row["bitstream_sha256"] == legacy[sample], row
                rows[arm].append(row)
                append(repeats_path, row)
        for arm, arm_rows in rows.items():
            assert len({r["bitstream_sha256"] for r in arm_rows}) == 1
            folder = sample_folder / arm
            media = ROOT / "videos" / f"{sample}_{arm}.mp4" if arm in ("C0", "C2-fixed", "C2-adaptive") else None
            quality = verify(folder, raw, folder / "stream.264", folder / "recon.yuv", media)
            summary = {k: arm_rows[-1][k] for k in
                       ("sample", "arm", "integer_evals", "subpixel_evals",
                        "bitstream_sha256", "bitstream_path")}
            for k in ("prior_s", "encoder_s", "pipeline_s", "integer_s", "subpixel_s"):
                data = [r[k] for r in arm_rows[1:]]
                summary.update({k: statistics.median(data), f"{k}_min": min(data),
                                f"{k}_max": max(data), f"{k}_warmup": arm_rows[0][k]})
            middle = sorted(arm_rows[1:], key=lambda r: r["pipeline_s"])[1]
            summary["me_fraction"] = (middle["integer_s"] + middle["subpixel_s"]) / middle["encoder_s"]
            summary["amdahl_me_limit"] = 1 / (1 - summary["me_fraction"])
            summary.update(quality, mp4=str(media) if media else "")
            summaries.append(summary)
            append(ROOT / "results/verified.jsonl", summary)
            print(sample, arm, "verified", flush=True)
    for row in summaries:
        control = next(r for r in summaries if r["sample"] == row["sample"] and r["arm"] == "C0")
        row.update(speedup=control["pipeline_s"] / row["pipeline_s"],
                   pipeline_reduction_pct=100*(1-row["pipeline_s"]/control["pipeline_s"]),
                   bitrate_change_pct=100*(row["bitrate_kbps"]/control["bitrate_kbps"]-1),
                   p_psnr_change_db=row["p_psnr"]-control["p_psnr"],
                   last4_change_db=row["last4_psnr"]-control["last4_psnr"])
    with (ROOT / "results/runs.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
