"""Audit native JM candidate reconstruction on a fixed prefix of one training clip."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline import LOCKED, digest
from evaluate import load_backend, replay
from state import PLANES, input_tensor, load_model, predict, records, targets
from coarse_fine_mv import Motion
from motion_comp import chroma, luma
from oracle_residual import coeff, parse_trace

FRAME_BYTES = 832 * 480 * 3 // 2


def read_yuv(path):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, FRAME_BYTES)
    return (raw[:, :399360].reshape(-1, 480, 832),
            raw[:, 399360:499200].reshape(-1, 240, 416),
            raw[:, 499200:].reshape(-1, 240, 416))


def as_blocks(plane, size):
    return plane.reshape(30, size, 52, size).transpose(0, 2, 1, 3)


@torch.inference_mode()
def candidate_mvs(base, proposal, row):
    hidden = input_tensor(row, base.kind)
    pred, _ = predict(base, hidden, 20)
    feat = base.encode_features(hidden, hidden.new_tensor([1.0]))
    logits = proposal(feat)
    ids = logits[..., 0].topk(4, dim=-1).indices
    selected = logits.gather(-2, ids[..., None].expand(*ids.shape, 19))
    integer = proposal.anchors[ids] + selected[..., 1:3].mul(4).round()
    phase = selected[..., 3:].argmax(-1)
    coarse = (integer * 4 + torch.stack((phase % 4, phase // 4), -1)).clamp(-2048, 2047)
    return pred, coarse.short().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "head", "kit", "jm", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--trace-bits", action="store_true")
    parser.add_argument("--sample", default="sample_1000")
    parser.add_argument("--frame", type=int, default=2, choices=range(1, 17))
    args = parser.parse_args()
    start = time.perf_counter()
    assert digest(args.checkpoint) == LOCKED
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    row, = (r for r in records([args.config])["train"]
            if r["step"] == 20 and r["sample_id"] == args.sample)
    base = load_model(args.checkpoint)
    saved = torch.load(args.head, map_location="cpu", weights_only=True)
    proposal = Motion(base.mode.in_channels, saved["anchors"]).cuda().eval()
    proposal.load_state_dict(saved["weights"])
    pred, coarse = candidate_mvs(base, proposal, row)
    backend = load_backend(args.kit)
    if args.trace_bits:
        run = backend.run

        def trace_run(cmd, folder, log, env=None):
            if Path(cmd[0]).name == "lencod.exe":
                env = dict(env, JM_P0_MB_BITS=str(folder / "mb_bits.csv"))
            return run(cmd, folder, log, env)

        backend.run = trace_run
    args.output.mkdir(parents=True)
    t = args.frame
    source = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
    raw = args.output / "source_prefix.yuv"
    with source.open("rb") as stream:
        raw.write_bytes(stream.read((t + 1) * FRAME_BYTES))
    assert raw.stat().st_size == (t + 1) * FRAME_BYTES
    prefix = {key: val[:t].copy() for key, val in pred.items()}
    original = args.output / "baseline_prefix"
    original.mkdir()
    replay(backend, args.jm, original, row, prefix, raw, {})
    baseline = read_yuv(original / "encoder_recon.yuv")
    source_planes = read_yuv(raw)
    mv = np.concatenate([prefix["mv_qpel"][-1, ..., None, :], coarse[t-1],
                         np.zeros((30, 52, 1, 2), dtype=np.int16)], axis=-2)
    valid = np.ones(mv.shape[:-1], dtype=bool)
    for k in range(1, mv.shape[-2]):
        valid[..., k] = ~(mv[..., k:k+1, :] == mv[..., :k, :]).all(-1).any(-1)
    costs, nonzeros, arms, mb_bits, residual_bits = [], [], [], [], []
    for k in range(mv.shape[-2]):
        name = f"candidate_{k}"
        folder = args.output / name
        folder.mkdir()
        arrays = {key: val.copy() for key, val in prefix.items()}
        arrays["mode_id"][-1] = 1
        arrays["mv_qpel"][-1] = mv[..., k, :]
        mc = [fn(ref[t-1], mv[..., k, :]) for fn, ref in zip((luma, chroma, chroma), baseline)]
        for (plane, size, _, _), src, ref in zip(PLANES, source_planes, mc):
            arrays[f"residual_{plane}"][-1] = as_blocks(src[t].astype(np.int16) - ref, size)
        np.savez_compressed(folder / "prediction.npz", **arrays)
        before = digest(folder / "prediction.npz")
        quality = replay(backend, args.jm, folder, row, arrays, raw, {}, coeff_trace=True, mc_trace=True)
        assert before == digest(folder / "prediction.npz")
        traced, masks = parse_trace(folder / "mc.bin")
        for got, expected, mask in zip(traced, mc, masks):
            assert mask[t].all() and np.array_equal(got[t], expected), (name, "MC mismatch")
        reconstructed = read_yuv(folder / "encoder_recon.yuv")
        assert all(np.array_equal(a[:t], b[:t]) for a, b in zip(reconstructed, baseline))
        costs.append(np.stack([np.square(as_blocks(src[t].astype(np.int32) - rec[t], size))
                               .sum(axis=(-2, -1)) for (_, size, _, _), src, rec in
                               zip(PLANES, source_planes, reconstructed)], axis=-1))
        levels = coeff(folder / "coeff.bin")
        nonzeros.append(np.array([np.count_nonzero(levels[t, mb]) for mb in range(1560)]).reshape(30, 52))
        if args.trace_bits:
            trace = np.loadtxt(folder / "mb_bits.csv", dtype=np.int32, delimiter=",")
            assert trace.shape == (t * 1560, 4)
            current = trace[trace[:, 0] == t]
            current = current[np.argsort(current[:, 1])]
            assert np.array_equal(current[:, 1], np.arange(1560))
            assert (current[:, 2] >= current[:, 3]).all()
            mb_bits.append(current[:, 2].reshape(30, 52))
            residual_bits.append(current[:, 3].reshape(30, 52))
        arms.append(dict(id=k, state_sha256=digest(folder / "replay.bin"),
                         bitstream_sha256=digest(folder / "prediction.264"), **quality))
        print(name, "three-plane native MC exact", flush=True)

    sse = np.stack(costs, axis=-2)
    nonzero = np.stack(nonzeros, axis=-1)
    best = np.where(valid, sse[..., 0], np.inf).argmin(-1)
    mode, oracle_mv, _ = (v.numpy()[t-1] for v in targets(row))
    bit_labels = {}
    if args.trace_bits:
        bit_labels = dict(mb_layer_bits=np.stack(mb_bits, axis=-1),
                          residual_bits=np.stack(residual_bits, axis=-1))
    np.savez_compressed(args.output / "cost_labels.npz", mv_qpel=mv, valid=valid,
                        sse_yuv=sse, residual_nonzero_proxy=nonzero, min_d_id=best,
                        oracle_mode=mode, oracle_mv_qpel=oracle_mv, diagnostic_oracle=True, **bit_labels)
    summary = dict(sample_id=args.sample, split="train", frame=t, diagnostic_oracle=True,
                   per_candidate=arms, mc_matches_jm_yuv=True, fixed_prefix_equal=True,
                   candidate_slots=["baseline_mv", "coarse1", "coarse2", "coarse3", "coarse4", "zero"],
                   distinct_candidates_mean=float(valid.sum(-1).mean()),
                   best_differs_from_baseline_fraction=float((best != 0).mean()),
                   baseline_mv_mean_sse_y=float(sse[..., 0, 0].mean()),
                   min_d_mean_sse_y=float(np.take_along_axis(sse[..., 0], best[..., None], axis=-1).mean()),
                   context="Each trial replaces the entire final P frame; fixed temporal prefix, "
                           "candidate-specific spatial neighbors and deblocking.",
                   rate_measure=("Native CAVLC macroblock-layer counters under each candidate frame's "
                                 "spatial context. Excludes slice/NAL overhead and is not mixed-frame RD."
                                 if args.trace_bits else "Nonzero coefficient proxy only."),
                   elapsed_seconds=time.perf_counter() - start)
    (args.output / "local_cost_audit.json").write_text(json.dumps(summary, indent=2))
    print("TRAIN SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
