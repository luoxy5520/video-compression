"""Offline check of source residuals paired with the actual reconstruction chain."""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline import digest
from evaluate import load_backend, replay
from local_cost import as_blocks, read_yuv
from oracle_residual import parse_trace
from run_p0_batch import CONFIGS, JM, KIT
from state import PLANES, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--sample", default="sample_1004")
    p.add_argument("--replay", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    row, = (r for r in records(CONFIGS)[args.split] if r["step"] == 20 and r["sample_id"] == args.sample)
    ref = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
    source = read_yuv(ref)
    backend = load_backend(KIT)
    run = backend.run
    arms = ("A", "E_baseline_mv_with_ideal_residual", "C_min_distortion", "D_lambda_64")
    args.output.mkdir(parents=True)
    results = []
    for arm in arms:
        # Retain modes and MV. Residuals are overwritten in the offline teacher.
        with np.load(args.replay / arm / "prediction.npz") as data:
            state = dict(data)
        keep = state["mode_id"] == 1
        teacher = args.output / arm / "offline_teacher"
        fixed = args.output / arm / "fixed_replay"
        teacher.mkdir(parents=True)
        fixed.mkdir()

        def trace_run(cmd, folder, log, env=None):
            if Path(cmd[0]).name == "lencod.exe":
                env = dict(env, JM_P0_MB_BITS=str(folder / "mb_bits.csv"))
                if folder == teacher:
                    cmd = [f"InputFile = {ref}" if str(a).startswith("InputFile = ") else a for a in cmd]
                    env["JM_LIVE_RESIDUAL"] = "1"
            return run(cmd, folder, log, env)

        backend.run = trace_run
        replay(backend, JM, teacher, row, state, ref, {}, mc_trace=True)
        traces, masks = parse_trace(teacher / "mc.bin")
        for (plane, size, _, _), src, mc, mask in zip(PLANES, source, traces, masks):
            assert np.array_equal(np.stack([as_blocks(m, size) for m in mask[1:]]),
                                  np.broadcast_to(keep[..., None, None], (*keep.shape, size, size)))
            difference = np.stack([as_blocks(a.astype(np.int16) - b, size)
                                   for a, b in zip(src[1:], mc[1:])])
            state[f"residual_{plane}"] = np.where(keep[..., None, None], difference, 0)
        prediction = fixed / "prediction.npz"
        np.savez_compressed(prediction, **state)
        before = digest(prediction)
        quality = replay(backend, JM, fixed, row, state, ref, {})
        assert before == digest(prediction)
        for name in ("prediction.264", "encoder_recon.yuv"):
            assert digest(teacher / name) == digest(fixed / name), name
        trace = np.loadtxt(fixed / "mb_bits.csv", dtype=np.int32, delimiter=",")
        results.append(dict(arm=arm, diagnostic_oracle=True, split=args.split, sample_id=args.sample,
                            offline_teacher_exact=True, state_sha256=digest(fixed / "replay.bin"),
                            bitstream_sha256=digest(fixed / "prediction.264"),
                            p_residual_bits=int(trace[:, 3].sum()), skip_fraction=float((~keep).mean()), **quality))
        print(arm, quality["psnr_y_p_db"], quality["bitrate_kbps"], flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
