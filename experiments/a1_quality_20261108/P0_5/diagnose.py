"""P0.5 offline Skip/P16 selection and bitrate attribution through fixed JM replay."""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "P0"))
from baseline import digest
from evaluate import load_backend, replay
from local_cost import as_blocks, read_yuv
from motion_comp import chroma, luma
from run_p0_batch import CONFIGS, KIT
from state import PLANES, records

ROOT = Path.home() / "a1_quality_20261108"
OLD = ROOT / "P0"
JM = ROOT / "P0_5/jm"
E = "E_baseline_mv_with_ideal_residual"
LAMBDAS = (0, 1, 4, 16, 64, 256)
P16_COUNT = 1560
FRAMES = 16


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_trace(folder, probe):
    dtype = [("ids", "<u4", 4), ("fields", "<i4", 4), ("mc", "<u2", 384)]
    if probe:
        dtype += [("skip_mv", "<i4", 2), ("skip_mc", "<u2", 384)]
    data = np.fromfile(folder / "trace.bin", dtype=dtype)
    data = data.reshape(FRAMES, P16_COUNT)
    bits = np.loadtxt(folder / "bits.csv", delimiter=",", dtype=np.int32).reshape(FRAMES, P16_COUNT, 8)
    assert np.array_equal(data["ids"][..., :3], bits[..., :3])
    assert np.array_equal(bits[..., 0], np.broadcast_to(np.arange(1, 17)[:, None], bits.shape[:2]))
    assert np.array_equal(bits[..., 1], np.broadcast_to(np.arange(P16_COUNT), bits.shape[:2]))
    assert np.isin(bits[..., 2], [0, 1]).all()
    return data, bits


def frame_blocks(planes):
    return np.concatenate([np.stack([as_blocks(f, size) for f in p]).reshape(len(p), P16_COUNT, -1)
                           for p, (_, size, _, _) in zip(planes, PLANES)], axis=-1)


def array_blocks(arrays):
    return np.concatenate([arrays[f"residual_{p}"].reshape(FRAMES, P16_COUNT, -1)
                           for p, *_ in PLANES], axis=-1)


def from_blocks(state, residual):
    for p, size, start, stop in PLANES:
        state[f"residual_{p}"] = residual[..., start:stop].reshape(FRAMES, 30, 52, size, size).astype(np.int16)


def mc_for(planes, vectors):
    mv = vectors.reshape(FRAMES, 30, 52, 2)
    return frame_blocks([np.stack([fn(frame, v) for frame, v in zip(p[:-1], mv)])
                         for p, fn in zip(planes, (luma, chroma, chroma))])


def execute(backend, folder, row, arrays, source, *, probe=False, live=False, trace=True):
    folder.mkdir()
    previous = backend.run

    def run(cmd, cwd, log, env=None):
        if Path(cmd[0]).name == "lencod.exe":
            env = dict(env)
            if trace:
                env.update(JM_P05_BITS=str(folder / "bits.csv"), JM_P05_MC=str(folder / "trace.bin"))
            if probe:
                env["JM_P05_SKIP_PROBE"] = "1"
            if live:
                env["JM_LIVE_RESIDUAL"] = "1"
                cmd = [f"InputFile = {source}" if str(x).startswith("InputFile = ") else x for x in cmd]
        return previous(cmd, cwd, log, env)

    np.savez_compressed(folder / "prediction.npz", **arrays)
    before = digest(folder / "prediction.npz")
    backend.run = run
    try:
        quality = replay(backend, JM, folder, row, arrays, source, {})
    finally:
        backend.run = previous
    assert before == digest(folder / "prediction.npz")
    assert quality["encoder_matches_decoder"] and quality["decoder_agreement"]
    assert quality["motion_search_s_reported"] == 0 and quality["oracle_p_bytes_read"] == 0
    return dict(**quality, npz_sha256=before, state_sha256=digest(folder / "replay.bin"),
                stream_sha256=digest(folder / "prediction.264"))


def preserve(folder, out):
    out.mkdir()
    for pattern in ("*.log", "adapter.json", "prediction.264", "prediction.npz"):
        for path in folder.glob(pattern):
            shutil.copyfile(path, out / path.name)


def paired(backend, work, row, arrays, source, original):
    teacher = work.with_name(work.name + "_teacher")
    measured = execute(backend, teacher, row, arrays, source, live=True)
    native, _ = read_trace(teacher, False)
    keep = native["ids"][..., 2] == 1
    residual = np.where(keep[..., None], original[1:].astype(np.int16) - native["mc"], 0)
    new = dict(arrays)
    from_blocks(new, residual)
    quality = execute(backend, work, row, new, source)
    assert quality["stream_sha256"] == measured["stream_sha256"]
    assert digest(work / "encoder_recon.yuv") == digest(teacher / "encoder_recon.yuv")
    shutil.rmtree(teacher)
    return new, quality


def per_frame(arm, row, split, arrays, folder, quality, original, source_planes, probe):
    native, bits = read_trace(folder, probe)
    recon_planes = read_yuv(folder / "encoder_recon.yuv")
    recon = frame_blocks(recon_planes)
    keep = native["ids"][..., 2] == 1
    assert np.array_equal(keep, arrays["mode_id"].reshape(keep.shape) == 1)
    actual = native["mc"]
    vectors = native["fields"][..., :2]
    assert np.array_equal(actual, mc_for(recon_planes, vectors)), (arm, "native actual MC mismatch")
    assert not native["fields"][..., 2:][~keep].any()
    assert not bits[..., 5:][~keep].any()
    assert np.array_equal(vectors[keep], arrays["mv_qpel"].reshape(*keep.shape, 2)[keep])
    clean_mc = mc_for(source_planes, vectors)
    source = original[1:].astype(np.int16)
    motion = source - clean_mc
    reference = actual.astype(np.int16) - clean_mc
    ideal = source - actual
    residual = np.where(keep[..., None], array_blocks(arrays), 0)
    error = residual - ideal
    delta = recon[1:].astype(np.int16) - source
    reference_pixels = recon[:-1].astype(np.int16) - original[:-1]

    def mse(array, frame, sel=slice(None)):
        pixels = array[frame, sel, :256].astype(np.float64)
        return float(np.square(pixels).mean()) if pixels.size else None

    def inner(x, y, frame):
        return float(np.mean(x[frame, :, :256].astype(np.float64) * y[frame, :, :256]))

    stats = []
    for t in range(FRAMES):
        flags = keep[t]
        stats.append(dict(
            split=split, video_id=row["sample_id"], frame=t+1, arm=arm,
            p16_fraction=float(flags.mean()), native_skip_nonzero_mv_count=int(
                (np.any(vectors[t] != 0, axis=-1) & ~flags).sum()),
            mb_layer_bits=int(bits[t,:,3].sum()), mode_run_bits=int(bits[t,:,4].sum()),
            mv_bits=int(bits[t,:,5].sum()), luma_coeff_bits=int(bits[t,:,6].sum()),
            chroma_coeff_bits=int(bits[t,:,7].sum()),
            residual_coeff_bits=int(bits[t,:,6:].sum()),
            other_mb_bits=int((bits[t,:,3]-bits[t,:,4:].sum(-1)).sum()),
            luma_nonzero_coeffs=int(native["fields"][t,:,2].sum()),
            chroma_nonzero_coeffs=int(native["fields"][t,:,3].sum()),
            recon_mse_y=mse(delta,t), previous_recon_error_mse_y=mse(reference_pixels,t),
            mc_reference_error_mse_y=mse(reference,t), motion_error_clean_ref_mse_y=mse(motion,t),
            reference_motion_cross_y=inner(reference,motion,t), ideal_residual_mse_y=mse(ideal,t),
            ideal_residual_mse_y_p16=mse(ideal,t,flags), residual_error_mse_y=mse(error,t),
            residual_error_mse_y_p16=mse(error,t,flags), psnr_y_db=quality["psnr_y_by_frame"][t+1]))
    return stats, native, bits


def ue_length(value):
    return 2 * np.floor(np.log2(np.asarray(value) + 1)).astype(np.int32) + 1


def skip_runs(modes):
    left = np.zeros_like(modes, dtype=np.int32)
    right = np.zeros_like(modes, dtype=np.int32)
    for i in range(1, modes.shape[1]):
        left[:, i] = np.where(modes[:, i-1] == 0, left[:, i-1]+1, 0)
    for i in range(modes.shape[1]-2, -1, -1):
        right[:, i] = np.where(modes[:, i+1] == 0, right[:, i+1]+1, 0)
    return left, right


def joint_candidates(p0, native, bits, original, a_recon, arrays):
    old = [load_npz(p0 / f"frame_{t:02d}/cost_labels.npz") for t in range(1, 17)]
    shape = (FRAMES, P16_COUNT)
    valid = np.stack([r["valid"] for r in old]).reshape(*shape, 6)
    mv = np.stack([r["mv_qpel"] for r in old]).reshape(*shape, 6, 2)
    sse = np.stack([r["sse_yuv"][..., 0] for r in old]).reshape(*shape, 6)
    rates = np.stack([r["mb_layer_bits"] for r in old]).reshape(*shape, 6)
    src = original[1:, :, :256].astype(np.int32)
    skip_sse = np.square(src-native["skip_mc"][..., :256]).sum(-1)
    a_sse = np.square(src-a_recon[1:, :, :256]).sum(-1)
    modes = arrays["mode_id"].reshape(shape)
    left, right = skip_runs(modes)
    # Compare isolated split/merge of the CAVLC run in the frozen A raster context.
    # P0 forced-P16 rate includes ue(0) plus syntax. A P16 includes its actual preceding run.
    run_p16 = ue_length(left) + ue_length(right)
    skip_rate = ue_length(left + right + 1)
    a_p16_rate = bits[..., 3] - ue_length(left) + run_p16
    return dict(
        valid=np.concatenate((valid, (modes==1)[..., None], np.ones((*shape,1),bool)), -1),
        mv=np.concatenate((mv, arrays["mv_qpel"].reshape(*shape,1,2),
                           native["skip_mv"][...,None,:]), -2),
        sse=np.concatenate((sse, a_sse[...,None], skip_sse[...,None]), -1),
        rate=np.concatenate((rates - 1 + run_p16[...,None], a_p16_rate[...,None],
                             skip_rate[...,None]), -1))


def joint_state(arrays, choices, candidates, p0, cache):
    state = dict(arrays)
    keep = choices != 7
    state["mode_id"] = keep.reshape(FRAMES, 30, 52).astype(np.uint8)
    mv = np.take_along_axis(candidates["mv"], choices[...,None,None], axis=-2).squeeze(-2)
    state["mv_qpel"] = np.where(keep[...,None], mv, 0).reshape(FRAMES,30,52,2).astype(np.int16)
    residual = array_blocks(arrays).copy()
    for t in range(FRAMES):
        for k in np.unique(choices[t]):
            if k >= 6:
                continue
            if (t,k) not in cache:
                candidate = load_npz(p0 / f"frame_{t+1:02d}/candidate_{k}/prediction.npz")
                cache[t,k] = np.concatenate([candidate[f"residual_{p}"][-1].reshape(P16_COUNT,-1)
                                             for p, *_ in PLANES], -1)
            blocks = cache[t,k]
            select = choices[t] == k
            residual[t, select] = blocks[select]
    from_blocks(state, np.where(keep[...,None], residual, 0))
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=("train","val"))
    p.add_argument("--sample", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    row, = (r for r in records(CONFIGS)[args.split] if r["step"] == 20 and r["sample_id"] == args.sample)
    p0 = OLD / "full" / args.split / args.sample
    source = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
    source_planes = read_yuv(source)
    original = frame_blocks(source_planes)
    backend = load_backend(KIT)
    args.output.mkdir(parents=True)
    frames, summaries, local = [], {}, []
    with tempfile.TemporaryDirectory(prefix="a1-p05-", dir="/dev/shm") as temp:
        work = Path(temp)
        def save(arm, path, quality, arrays):
            summaries[arm] = dict(**quality, p16_fraction=float(arrays["mode_id"].mean()),
                                  diagnostic_oracle=arm != "A")
            preserve(path, args.output / arm)

        a = load_npz(p0 / "replay/A/prediction.npz")
        a_path = work / "A"
        quality = execute(backend, a_path, row, a, source, probe=True)
        assert quality["stream_sha256"] == digest(p0 / "replay/A/prediction.264")
        stats, native, bits = per_frame("A", row, args.split, a, a_path, quality, original, source_planes, True)
        frames.extend(stats)
        save("A", a_path, quality, a)
        predicted_skip = mc_for(read_yuv(a_path / "encoder_recon.yuv"), native["skip_mv"])
        if not np.array_equal(native["skip_mc"], predicted_skip):
            bad = native["skip_mc"] != predicted_skip
            print("Skip MC mismatch by plane", [int(bad[...,a:b].sum()) for _,_,a,b in PLANES],
                  "on native Skip", int(bad[native["ids"][...,2]==0].sum()), flush=True)
            first = tuple(np.argwhere(bad)[0])
            print(first, native["skip_mv"][first[:2]], native["skip_mc"][first], predicted_skip[first], flush=True)
            raise ValueError("Skip probe did not match independent MC")

        existing_skip = native["ids"][...,2] == 0
        assert np.array_equal(native["skip_mc"][existing_skip], native["mc"][existing_skip])
        candidates = joint_candidates(p0, native, bits, original,
                                      frame_blocks(read_yuv(a_path/"encoder_recon.yuv")), a)
        np.savez_compressed(args.output / "local_candidates.npz", **candidates)
        del native, bits
        shutil.rmtree(a_path)

        for arm, old_name in (("F","F_lambda_256"), ("E", E)):
            arrays = load_npz(p0 / "replay" / old_name / "prediction.npz")
            path = work / arm
            if arm == "E":
                arrays, quality = paired(backend, path, row, arrays, source, original)
                if args.split == "val":
                    historical = OLD / "trajectory_val" / args.sample / E / "fixed_replay/prediction.264"
                    assert quality["stream_sha256"] == digest(historical)
            else:
                quality = execute(backend, path, row, arrays, source)
                assert quality["stream_sha256"] == digest(p0 / "replay" / old_name / "prediction.264")
            stats, _, _ = per_frame(arm, row, args.split, arrays, path, quality, original, source_planes, False)
            frames.extend(stats)
            save(arm,path,quality,arrays)
            shutil.rmtree(path)

        cache = {}
        for weight in ((256,) if args.smoke else LAMBDAS):
            cost = candidates["sse"] + weight*candidates["rate"]
            masked = np.where(candidates["valid"],cost,np.inf)
            selected = masked.argmin(-1)
            for t in range(FRAMES):
                best = masked[t].min(-1)
                p16 = masked[t,:,:7].min(-1)
                local.append(dict(split=args.split,video_id=args.sample,frame=t+1,lambda_rd=weight,
                                  skip_selected=int((selected[t]==7).sum()),
                                  native_skip_nonzero=int((candidates["mv"][t,:,7]!=0).any(-1).sum()),
                                  p16_only_cost=float(p16.sum()), joint_cost=float(best.sum())))
            joint = joint_state(a,selected,candidates,p0,cache)
            for policy in ("fixed","predicted","paired"):
                name = f"joint_{policy}_{weight}"
                path = work / name
                state = dict(joint)
                if policy == "predicted":
                    res = np.where(state["mode_id"].reshape(FRAMES,P16_COUNT,1),
                                   array_blocks(a),0)
                    from_blocks(state,res)
                if policy == "paired":
                    state, quality = paired(backend,path,row,state,source,original)
                else:
                    quality = execute(backend,path,row,state,source,trace=False)
                save(name,path,quality,state)
                shutil.rmtree(path)
    csv_write(args.output / "per_frame.csv",frames)
    csv_write(args.output / "local_per_frame.csv",local)
    (args.output/"summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
    print("DONE",args.split,args.sample,flush=True)


if __name__ == "__main__":
    main()
