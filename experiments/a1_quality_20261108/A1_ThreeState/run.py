"""Replay the six frozen A1 state interventions on the historical 16-video validation set."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "P2_Lite"))
import p2lite as p
from baseline import LOCKED
from benchmark import collect_quality
from evaluate import replay, video_details
from state import PLANES, targets

ARMS = "ABCDEF"
RESIDUALS = [f"residual_{plane}" for plane, *_ in PLANES]


def trace_backend(folder, source=None):
    backend = p.p05.load_backend(p.KIT)
    run = backend.run

    def traced(cmd, cwd, log, env=None):
        if Path(cmd[0]).name == "lencod.exe":
            env = dict(env, JM_P05_BITS=str(folder / "bits.csv"),
                       JM_P05_MC=str(folder / "trace.bin"))
            if source is not None:
                env["JM_ALIGNED_PROXY"] = str(source)
        with (cwd / "commands.jsonl").open("a") as f:
            f.write(json.dumps(dict(argv=list(map(str, cmd)),
                                    jm_env={k: v for k, v in (env or {}).items() if k.startswith("JM_")})) + "\n")
        return run(cmd, cwd, log, env)

    backend.run = traced
    return backend


def intervene(pred, oracle):
    mode, mv, residual = (x.numpy() for x in oracle)
    pred_p16 = pred["mode_id"] == 1
    oracle_p16 = mode == 1
    common = pred_p16 & oracle_p16
    states = {arm: {k: pred[k].copy() for k in ("mode_id", "mv_qpel", *RESIDUALS)}
              for arm in "ABCDE"}
    states["B"]["mode_id"] = mode.astype(np.uint8)
    states["C"]["mv_qpel"][common] = mv[common]
    for plane, size, start, stop in PLANES:
        key = f"residual_{plane}"
        states["D"][key][common] = residual[..., start:stop].reshape(*mode.shape, size, size)[common]
        states["E"][key].fill(0)
    stats = dict(pred_p16=int(pred_p16.sum()), oracle_p16=int(oracle_p16.sum()),
                 common_p16=int(common.sum()),
                 b_skip_to_p16=int((~pred_p16 & oracle_p16).sum()),
                 b_p16_to_skip=int((pred_p16 & ~oracle_p16).sum()),
                 c_mv_changed=int(np.any(pred["mv_qpel"][common] != mv[common], axis=-1).sum()),
                 d_residual_changed=int(sum(np.any(states["D"][k][common] != pred[k][common],
                                                 axis=(-2, -1)).sum() for k in RESIDUALS)))
    for state in states.values():
        skip = state["mode_id"] == 0
        state["mv_qpel"][skip] = 0
        for key in RESIDUALS:
            state[key][skip] = 0
    return states, stats


def checked_replay(folder, row, state, source, timing, aligned=False):
    folder.mkdir()
    backend = trace_backend(folder, source if aligned else None)
    quality = replay(backend, p.JM, folder, row, state, source, timing)
    trace, bits = p.p05.read_trace(folder, False)
    mode = state["mode_id"].reshape(16, 1560)
    np.testing.assert_array_equal(bits[..., 2], mode)
    native = trace["fields"][..., :2]
    np.testing.assert_array_equal(native[mode == 1], state["mv_qpel"].reshape(16, 1560, 2)[mode == 1])
    assert not bits[mode == 0, 6:].any()
    quality["residual_syntax_bits"] = int(bits[..., 6:].sum())
    quality["decoded_frame_sha256"] = [
        hashlib.sha256(frame).hexdigest() for frame in
        np.fromfile(folder / "decoded_ffmpeg.yuv", np.uint8).reshape(17, -1)]
    return quality, trace


def full_jm(folder, row, source):
    folder.mkdir()
    backend = p.p05.load_backend(p.KIT)
    start = time.perf_counter()
    bit = folder / "prediction.264"
    recon = folder / "encoder_recon.yuv"
    backend.encoder(folder, "dump", folder / "oracle_state.bin", source, bit, recon)
    elapsed = time.perf_counter() - start
    historical = Path(row["current_oracle_state_path"])
    assert p.digest(bit) == p.digest(historical.with_name("reference.264"))
    assert p.digest(recon) == p.digest(row["current_reference_recon_path"])
    assert p.digest(folder / "oracle_state.bin") == p.digest(historical)
    quality = collect_quality(backend, folder, bit, source,
                              dict(jm_encode_s=elapsed, latent_to_bitstream_s=elapsed))
    decoded = folder / "decoded_ffmpeg.yuv"
    assert p.digest(recon) == p.digest(decoded)
    search = re.search(r"Total ME time for sequence\s*:\s*(\d+\.\d+)\s*sec",
                       (folder / "dump.log").read_text(), re.I)
    assert search
    return dict(**quality, **video_details(backend, folder, source, decoded, "full_jm"),
                encoder_matches_decoder=True, motion_search_s_reported=float(search[1]),
                decoded_frame_sha256=[hashlib.sha256(frame).hexdigest() for frame in
                                      np.fromfile(decoded, np.uint8).reshape(17, -1)])


def experiment(folder, row, pred, timing):
    start = time.perf_counter()
    sid = row["sample_id"]
    historical, source = p.paths("val", row)
    with np.load(historical / "A/prediction.npz", allow_pickle=False) as baseline:
        for key in ("mode_id", "mv_qpel", *RESIDUALS):
            np.testing.assert_array_equal(pred[key], baseline[key], err_msg=f"{sid} {key}")
    np.savez_compressed(folder / "frozen_head_output.npz", **pred)
    states, adaptation = intervene(pred, targets(row))
    qualities = {}
    traces = {}
    for arm in "ABCDE":
        arm_start = time.perf_counter()
        quality, traces[arm] = checked_replay(folder / arm, row, states[arm], source, timing, arm == "E")
        quality["arm_wall_s"] = time.perf_counter() - arm_start
        qualities[arm] = quality
    assert p.digest(folder / "A/prediction.264") == p.digest(historical / "A/prediction.264")
    for arm in "DE":
        np.testing.assert_array_equal(traces["A"]["fields"][..., :2], traces[arm]["fields"][..., :2])
    # Verify E by materializing the signed source-minus-native-MC residual
    # and passing it through the ordinary replay backend with no source input.
    mask = states["E"]["mode_id"].reshape(16, 1560) == 1
    raw = p.p05.frame_blocks(p.p05.read_yuv(source))[1:].astype(np.int16)
    aligned = raw - traces["E"]["mc"].astype(np.int16)
    aligned[~mask] = 0
    p.p05.from_blocks(states["E"], aligned)
    check, _ = checked_replay(folder / "E_static_verification", row, states["E"], source, {})
    assert p.digest(folder / "E/prediction.264") == p.digest(folder / "E_static_verification/prediction.264")
    assert check["decoded_frame_sha256"] == qualities["E"]["decoded_frame_sha256"]
    adaptation["e_materialized_roundtrip_exact"] = True
    adaptation["e_p16_mc_rmse_y"] = float(np.sqrt(np.mean(aligned[mask, :256].astype(float)**2)))
    arm_start = time.perf_counter()
    qualities["F"] = full_jm(folder / "F", row, source)
    qualities["F"]["arm_wall_s"] = time.perf_counter() - arm_start
    assert all(q["decoded_frame_sha256"][0] == qualities["A"]["decoded_frame_sha256"][0] for q in qualities.values())
    hashes = {a: p.digest(folder / a / "prediction.264") for a in ARMS}
    result = dict(sample=sid, prompt=row["prompt"], seed=row["seed"], input=dict(
        source=str(source), source_sha256=p.digest(source),
        oracle_state=str(row["current_oracle_state_path"]), oracle_state_sha256=p.digest(row["current_oracle_state_path"]),
        oracle_labels=str(row["root"] / row["current_label_path"]),
        oracle_labels_sha256=p.digest(row["root"] / row["current_label_path"]),
        hidden=str(row["hidden_path"]), hidden_sha256=p.digest(row["hidden_path"])),
        adaptation=adaptation, stream_sha256=hashes, qualities=qualities,
        video_wall_s=time.perf_counter() - start)
    p.save_json(folder / "result.json", result)
    print(sid, {a: round(q["psnr_y_p_db"], 3) for a, q in qualities.items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    start = time.perf_counter()
    assert not any(k.startswith("JM_") for k in os.environ)
    assert p.digest(p.CKPT) == LOCKED
    rows = [row for row in p.records(p.CONFIGS)["val"] if row["step"] == 20]
    assert len(rows) == 16 and len({row["prompt"] for row in rows}) == 4
    if args.smoke:
        rows = rows[:1]
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    model = p.load_model(p.CKPT).requires_grad_(False)
    with torch.inference_mode():
        p.predict(model, p.input_tensor(rows[0], "hidden"), 20)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for row in rows:
            folder = out / row["sample_id"]
            folder.mkdir()
            with torch.inference_mode():
                load_start = time.perf_counter()
                hidden = p.input_tensor(row, "hidden")
                torch.cuda.synchronize()
                load_s = time.perf_counter() - load_start
                pred, timing = p.predict(model, hidden, 20)
            futures.append(pool.submit(experiment, folder, row, pred, dict(timing, load_latent_s=load_s)))
        results = [f.result() for f in futures]
    backend = p.p05.load_backend(p.KIT)
    tools = [p.JM / "bin/lencod.exe", Path(backend.ENC), Path(backend.DEC),
             p.JM / "lencod/src/state_io.c", p.KIT / "linux/run_pipeline.py",
             p.KIT / "scripts/prediction_to_b1_state.py", p.KIT / "linux/skip_p16.cfg",
             p.KIT / "JM-backend/cfg/encoder_baseline.cfg",
             p.KIT / "JM-backend/cfg/decoder.cfg", *p.CONFIGS]
    p.save_json(out / "run_manifest.json", dict(
        code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        command=[sys.executable, *sys.argv], output=str(out), wall_s=time.perf_counter() - start,
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        checkpoint=str(p.CKPT), checkpoint_sha256=LOCKED, split="historical_validation_step20",
        tools_sha256={str(tool): p.digest(tool) for tool in tools},
        videos=results))


if __name__ == "__main__":
    main()
