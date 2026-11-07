"""Fixed-state replay ablation and closed-loop residual correction."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/name) for name in
                ("decision_search_20261107", "residual_reference_20261107", "full_state_20261107")]
from benchmark import FRAME_BYTES, collect_quality, load_backend
from experiment import DecisionHead
from export_trace import parse_trace
from train_source import read_planes
from train_state import load_config, load_latent, networks

SHAPE = (16, 30, 52)
RESIDUAL = ("residual_y", "residual_u", "residual_v")


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def get_old(a, row, sid, split, regressor, mode):
    if split == "test":
        source = a.baseline/sid/"prediction_020/prediction.npz"
        with np.load(source, allow_pickle=False) as result:
            return dict(result)
    with torch.inference_mode():
        latent = load_latent(row, "cuda")
        step = torch.ones(1, device="cuda")
        modes = mode(latent, step)[0].argmax(0).byte().cpu().numpy()
        mv, res = regressor(latent, step)
        mv = mv[0].mul(16).round().cpu().numpy().astype("<i2")
        res = res[0].mul(16).round().clamp(-255, 255).cpu().numpy().astype("<i2")
    return dict(mode_id=modes, mv_qpel=mv,
                residual_y=res[...,:256].reshape(*SHAPE,16,16),
                residual_u=res[...,256:320].reshape(*SHAPE,8,8),
                residual_v=res[...,320:].reshape(*SHAPE,8,8))


def get_new(a, sid, head):
    with np.load(a.decisions/sid/"features.npz", allow_pickle=False) as source:
        features = torch.from_numpy(source["features"]).cuda()
        vectors = source["vectors"]
        prep_s = float(source["vae_s"]+source["proposal_s"])
    start = time.perf_counter()
    with torch.inference_mode():
        ranks, mode = head(features)
        modes = (mode.sigmoid() < .90).cpu().numpy().astype("uint8")
        best = ranks.argmin(1).cpu().numpy()
    return modes.reshape(SHAPE), (4*vectors[np.arange(len(best)),best]).reshape(*SHAPE,2), prep_s+time.perf_counter()-start


def encode(a, backend, row, sid, tag, arrays, *, proxy=None, weights=None, trace=False,
           decision_s=None, extra_env=None):
    folder = a.output/"samples"/sid/tag
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    state = folder/"predicted.bin"
    np.savez(folder/"prediction.npz", **arrays)
    backend.run([sys.executable, a.kit/"scripts/prediction_to_b1_state.py",
                 "--prediction", folder/"prediction.npz", "--oracle-template", row["target_oracle_state_path"],
                 "--output-state", state, "--result-json", folder/"adapter.json"], folder, folder/"adapter.log")
    adapted = time.perf_counter()
    env = os.environ.copy()
    for name in ("JM_LIVE_RESIDUAL", "JM_LIVE_MC_TRACE", "JM_ALIGNED_PROXY", "JM_ALIGNED_WEIGHTS",
                 "JM_REPLAY_SYNC_MV"):
        env.pop(name, None)
    env.update(JM_STATE_MODE="replay", JM_STATE_FILE=str(state))
    if a.sync_mv:
        env["JM_REPLAY_SYNC_MV"] = "1"
    if proxy:
        env["JM_ALIGNED_PROXY"] = str(proxy)
    if weights:
        env["JM_ALIGNED_WEIGHTS"] = str(weights)
    if trace:
        env["JM_LIVE_MC_TRACE"] = str(folder/"mc.bin")
    if extra_env:
        env.update(extra_env)
    bitstream = folder/"prediction.264"
    settings = dict(InputFile=a.output/"dummy.yuv", OutputFile=bitstream,
                    ReconFile=folder/"recon.yuv", SourceWidth=832, SourceHeight=480,
                    FramesToBeEncoded=17, FrameRate=16, QPISlice=28, QPPSlice=28,
                    RDOptimization=0, NumberReferenceFrames=1, PList0References=1)
    command = [a.jm/"bin/lencod.exe", "-d", a.kit/"JM-backend/cfg/encoder_baseline.cfg",
               "-f", a.kit/"linux/skip_p16.cfg"]
    for key, value in settings.items():
        command += ["-p", f"{key} = {value}"]
    backend.run(command, folder, folder/"encode.log", env)
    elapsed = time.perf_counter()-start
    result = collect_quality(backend, folder, bitstream, a.data/sid/"final_raw.yuv",
                             dict(adapter_s=adapted-start, replay_s=elapsed-(adapted-start),
                                  cached_state_to_bitstream_s=elapsed,
                                  latent_to_bitstream_s=elapsed+(decision_s or 0)))
    encoder_matches_decoder = backend.sha(folder/"recon.yuv") == backend.sha(folder/"decoded_ffmpeg.yuv")
    if a.sync_mv and not encoder_matches_decoder:
        raise ValueError(f"Encoder and decoder reconstruction differ: {sid}, {tag}")
    result.update(sample_id=sid, variant=tag, decision_s=decision_s,
                  encoder_matches_decoder=encoder_matches_decoder,
                  i_frame_sha256=digest(np.fromfile(folder/"decoded_ffmpeg.yuv", dtype=np.uint8,
                                                   count=FRAME_BYTES)),
                  bitstream_sha256=backend.sha(bitstream), p16_fraction=float(arrays["mode_id"].mean()),
                  mode_sha256=digest(arrays["mode_id"]), mv_sha256=digest(arrays["mv_qpel"]),
                  residual_sha256=[digest(arrays[name]) for name in RESIDUAL])
    (folder/"metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result, folder


def planes(path):
    y, uv = read_planes(path)
    return y[0,:,0].numpy(), uv[0,:,0].numpy(), uv[0,:,1].numpy()


def unpack_blocks(blocks):
    size = blocks.shape[-1]
    image = blocks.transpose(0,1,3,2,4).reshape(16,30*size,52*size)
    return np.pad(image, ((1,0),(0,0),(0,0)))


def samples(a, sid, arrays, folder, rng):
    mc, masks = parse_trace(folder/"mc.bin")
    proxies = planes(a.decisions/sid/"proxy.yuv")
    target = planes(a.data/sid/"final_raw.yuv")
    block_count = int(arrays["mode_id"].sum())
    for i, (ref, mask, proxy, truth, key) in enumerate(zip(mc,masks,proxies,target,RESIDUAL)):
        size = arrays[key].shape[-1]
        pixels = np.flatnonzero(mask)
        if pixels.size != block_count*size**2:
            raise ValueError(f"Incomplete actual MC trace: {sid}, {key}")
        ids = rng.choice(pixels, size=min(50000,pixels.size), replace=False)
        p, m, y, old = [x.reshape(-1)[ids].astype(np.float64)
                        for x in (proxy, ref, truth, unpack_blocks(arrays[key]))]
        base = p-m
        features = np.column_stack((np.ones(len(ids)), (p-128)/128, base/128, old/128, np.abs(base)/128))
        yield i, features, (y-m)-base


def fit(a, rows):
    rng = np.random.default_rng(42)
    fitted = {}
    for split in ("train","val"):
        grouped = [[],[],[]]
        for sid in rows[split]:
            for tag in ("teacher_old", "teacher_proxy"):
                folder = a.output/"samples"/sid/tag
                with np.load(folder/"prediction.npz", allow_pickle=False) as saved:
                    arrays = dict(saved)
                for plane, x, y in samples(a, sid, arrays, folder, rng):
                    if len(y):
                        grouped[plane].append((x,y))
        fitted[split] = [(np.concatenate(x),np.concatenate(y)) for x,y in (zip(*p) for p in grouped)]
    params, report = [], []
    for (x,y),(vx,vy) in zip(fitted["train"], fitted["val"]):
        matrix, vector = x.T@x/len(y), x.T@y/len(y)
        options = []
        for regularization in (0.0001, 0.001, 0.01):
            w = np.linalg.solve(matrix+regularization*np.diag([0,1,1,1,1]),vector)
            error = np.mean((vy-vx@w)**2)
            options.append((error,regularization,w))
        error, regularization, w = min(options,key=lambda t:t[0])
        params.append(w)
        report.append(dict(train_pixels=len(y), val_pixels=len(vy), ridge=regularization,
                           val_proxy_residual_rmse=float(np.sqrt(np.mean(vy**2))),
                           val_learned_residual_rmse=float(np.sqrt(error))))
    np.asarray(params,dtype="<f8").tofile(a.weights)
    (a.output/"training.json").write_text(json.dumps(report,indent=2))


def main(a):
    torch.set_num_threads(4)
    config, _, rows = load_config(a.config)
    splits = config["splits"]
    backend = load_backend(a.kit)
    a.output.mkdir(parents=True,exist_ok=False)
    a.weights.parent.mkdir(parents=True,exist_ok=True)
    with (a.output/"dummy.yuv").open("wb") as raw:
        raw.truncate(17*FRAME_BYTES)
    head = DecisionHead(27).cuda().eval()
    head.load_state_dict(torch.load(a.head, weights_only=True))
    mode, reg = networks(a.mode, "cuda")
    reg.load_state_dict(torch.load(a.old_regressor,weights_only=True))
    reg.eval()
    results = []
    manifest = {k:[dict(sample_id=sid,prompt=rows[sid,20]["prompt"],
                        seed=rows[sid,20].get("seed"),
                        latent_path=rows[sid,20]["latent_path"],
                        reference=str(a.data/sid/"final_raw.yuv")) for sid in group]
                for k,group in splits.items()}
    (a.output/"dataset.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    # Test the requested fixed-residual comparison before training the correction.
    for sid in splits["test"]:
        row = rows[sid,20]
        old = get_old(a,row,sid,"test",reg,mode)
        modes,mv,decision_s = get_new(a,sid,head)
        reference_hash = None
        for tag, arrays in (("baseline",old),("mode_only",dict(old,mode_id=modes)),
                            ("mv_only",dict(old,mv_qpel=mv)),
                            ("mode_mv",dict(old,mode_id=modes,mv_qpel=mv))):
            result,folder = encode(a,backend,row,sid,tag,arrays,
                                   decision_s=decision_s if tag!="baseline" else None)
            if tag == "baseline":
                saved = a.baseline/sid/"prediction_020/prediction.264"
                if not a.sync_mv and backend.sha(saved)!=result["bitstream_sha256"]:
                    raise ValueError(f"Step 20 baseline is not byte-identical: {sid}")
                reference_hash = result["i_frame_sha256"]
            if result["i_frame_sha256"]!=reference_hash or result["residual_sha256"]!=[digest(old[k]) for k in RESIDUAL]:
                raise ValueError(f"Fixed control changed: {sid}, {tag}")
            results.append(result)
    (a.output/"phase1.json").write_text(json.dumps(results,indent=2))
    for split in ("train","val"):
        for sid in splits[split]:
            row = rows[sid,20]
            old = get_old(a,row,sid,split,reg,mode)
            modes,mv,_ = get_new(a,sid,head)
            arrays = dict(old,mode_id=modes,mv_qpel=mv)
            for tag in ("teacher_old","teacher_proxy"):
                proxy = a.decisions/sid/"proxy.yuv" if tag=="teacher_proxy" else None
                encode(a,backend,row,sid,tag,arrays,proxy=proxy,trace=True)
    fit(a,splits)
    for sid in splits["test"]:
        row = rows[sid,20]
        fixed = next(r for r in results if r["sample_id"]==sid and r["variant"]=="mode_mv")
        with np.load(a.output/"samples"/sid/"mode_mv/prediction.npz",allow_pickle=False) as saved:
            arrays = dict(saved)
        for tag in ("proxy_residual","learned_residual"):
            result,_ = encode(a,backend,row,sid,tag,arrays,proxy=a.decisions/sid/"proxy.yuv",
                              weights=a.weights if tag=="learned_residual" else None,
                              decision_s=fixed["decision_s"])
            if any(result[k]!=fixed[k] for k in ("mode_sha256","mv_sha256","i_frame_sha256","residual_sha256")):
                raise ValueError(f"Residual ablation changed fixed state: {sid}, {tag}")
            results.append(result)
    means = {tag:{key:float(np.mean([r[key] for r in results if r["variant"]==tag]))
                  for key in ("psnr_y_p_db","psnr_yuv_p_db","bitrate_kbps","cached_state_to_bitstream_s",
                              "latent_to_bitstream_s","p16_fraction")}
             for tag in sorted({r["variant"] for r in results})}
    (a.output/"results.json").write_text(json.dumps(dict(means=means,per_video=results),indent=2))


def diagnose(a):
    """Isolate the source proxy error and test sensitivity to fixed Skip decisions."""
    _, _, rows = load_config(a.config)
    backend = load_backend(a.kit)
    comparisons = json.loads((a.output/"phase1.json").read_text())
    results = []
    for r in comparisons:
        if r["variant"] != "mode_mv":
            continue
        sid = r["sample_id"]
        with np.load(a.output/"samples"/sid/"mode_mv/prediction.npz",allow_pickle=False) as saved:
            fixed = dict(saved)
        cases = (
            ("oracle_source_diagnostic", fixed, a.data/sid/"final_raw.yuv"),
            ("all_p16_diagnostic", dict(fixed,mode_id=np.ones_like(fixed["mode_id"])),
             a.decisions/sid/"proxy.yuv"),
        )
        for tag, arrays, proxy in cases:
            result,_ = encode(a,backend,rows[sid,20],sid,tag,arrays,proxy=proxy)
            if result["i_frame_sha256"] != r["i_frame_sha256"] or result["mv_sha256"] != r["mv_sha256"]:
                raise ValueError(f"Unexpected I frame or MV array change: {sid}, {tag}")
            results.append(result)
    (a.output/"diagnostics.json").write_text(json.dumps(results,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("config","mode","old-regressor","head","decisions","baseline","kit","jm","data","output","weights"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--diagnostics",action="store_true")
    parser.add_argument("--sync-mv",action="store_true")
    args = parser.parse_args()
    (diagnose if args.diagnostics else main)(args)
