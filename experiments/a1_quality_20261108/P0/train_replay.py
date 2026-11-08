"""Combine a training clip's per-frame candidate audits and replay A–F as full streams."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from baseline import digest
from evaluate import load_backend, replay
from local_cost import read_yuv
from state import PLANES, records

LAMBDAS = (0, 1, 4, 16, 64, 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "kit", "jm", "audit", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()
    row, = (r for r in records([args.config])["train"]
            if r["step"] == 20 and r["sample_id"] == args.sample)
    frames = [args.audit / f"frame_{t:02d}" for t in range(1, 17)]
    with np.load(frames[-1] / "baseline_prefix/prediction.npz") as data:
        baseline = dict(data)
    local, ideal = [], []
    for folder in frames:
        with np.load(folder / "cost_labels.npz") as data:
            assert bool(data["diagnostic_oracle"])
            local.append(dict(data))
        options = {p: [] for p, *_ in PLANES}
        for idx in range(local[-1]["valid"].shape[-1]):
            with np.load(folder / f"candidate_{idx}/prediction.npz") as data:
                for p in options:
                    options[p].append(data[f"residual_{p}"][-1])
        ideal.append({p: np.stack(r, axis=2) for p, r in options.items()})
    mv = np.stack([x["mv_qpel"] for x in local])
    valid = np.stack([x["valid"] for x in local])
    distortion = np.stack([x["sse_yuv"][..., 0] for x in local])
    bits = np.stack([x["mb_layer_bits"] for x in local])
    oracle_mv = np.stack([x["oracle_mv_qpel"] for x in local])
    nearest = 1 + np.square(mv[..., 1:5, :].astype(np.int32) - oracle_mv[..., None, :]).sum(-1).argmin(-1)
    min_d = np.where(valid, distortion, np.inf).argmin(-1)
    zeros = np.zeros_like(min_d)

    backend = load_backend(args.kit)
    run = backend.run

    def trace_run(cmd, folder, log, env=None):
        if Path(cmd[0]).name == "lencod.exe":
            env = dict(env, JM_P0_MB_BITS=str(folder / "mb_bits.csv"))
        return run(cmd, folder, log, env)

    backend.run = trace_run
    ref = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
    assert read_yuv(ref)[0].shape == (17, 480, 832)
    args.output.mkdir(parents=True)
    indices = np.indices(min_d.shape)
    results = []

    def measure(name, selected=None, use_ideal=False, force_p16=True):
        arrays = {key: val.copy() for key, val in baseline.items()}
        if force_p16:
            arrays["mode_id"].fill(1)
        if selected is not None:
            arrays["mv_qpel"] = mv[(*indices, selected)]
        if use_ideal:
            y, x = indices[1:, 0]
            for p, *_ in PLANES:
                arrays[f"residual_{p}"] = np.stack(
                    [r[p][y, x, sel] for r, sel in zip(ideal, selected)])
        folder = args.output / name
        folder.mkdir()
        prediction = folder / "prediction.npz"
        np.savez_compressed(prediction, **arrays)
        before = digest(prediction)
        start = time.perf_counter()
        quality = replay(backend, args.jm, folder, row, arrays, ref, {})
        assert before == digest(prediction)
        trace = np.loadtxt(folder / "mb_bits.csv", dtype=np.int32, delimiter=",")
        assert trace.shape == (16 * 1560, 4)
        results.append(dict(arm=name, sample_id=args.sample, split="train",
                            diagnostic_oracle=name not in ("A", "A_fixed_p16"),
                            skip_fraction=float((arrays["mode_id"] == 0).mean()),
                            p_mb_layer_bits=int(trace[:, 2].sum()),
                            p_residual_bits=int(trace[:, 3].sum()),
                            state_sha256=digest(folder / "replay.bin"),
                            bitstream_sha256=digest(folder / "prediction.264"),
                            full_replay_and_scoring_s=time.perf_counter() - start, **quality))
        print(name, quality["psnr_y_p_db"], quality["bitrate_kbps"], flush=True)

    measure("A", force_p16=False)
    measure("A_fixed_p16")
    measure("B_nearest_with_fixed_residual", nearest)
    measure("C_min_distortion", min_d, True)
    measure("F_lambda_0", min_d)
    measure("E_baseline_mv_with_ideal_residual", zeros, True)
    for weight in LAMBDAS[1:]:
        min_j = np.where(valid, distortion + weight * bits, np.inf).argmin(-1)
        measure(f"D_lambda_{weight}", min_j, True)
        measure(f"F_lambda_{weight}", min_j)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    columns = ["arm", "sample_id", "split", "diagnostic_oracle", "psnr_y_p_db", "psnr_yuv_p_db",
               "ssim_y_p", "bitrate_kbps", "p_mb_layer_bits", "p_residual_bits",
               "last_frame_psnr_y", "late_minus_early_psnr", "skip_fraction"]
    with (args.output / "rd_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
