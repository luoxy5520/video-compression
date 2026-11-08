"""Train a sparse integer-level head and evaluate it through strict JM replay."""
import argparse
import csv
import json
import hashlib
import struct
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "P0_6")]
from sweep import BASIS, CONFIGS, FRAMES, JM, KIT, ROOT, digest, p05, records
from state import input_tensor, load_model, predict
from shadow_luma import SCALE, SCAN, fwd
from evaluate import replay

CFG = json.loads((HERE / "config.json").read_text())
CKPT = Path("/mnt/data_hdd/models/dit_compression/a1_quality_20261108/locked_hidden.pt")
P06 = ROOT / "P0_6/full"
SHAPE = (16, 30, 52, 16)
MAGNITUDES = 64


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def rows_by_split():
    group = records(CONFIGS)
    result = {s: [r for r in group[s] if r["step"] == 20] for s in ("train", "val")}
    assert [len(result[s]) for s in result] == [44, 16]
    return result


def paths(split, row):
    root = P06 / split / row["sample_id"]
    source = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
    return root, source


def native_levels(path, mode):
    """Read JM's P16 luma CAVLC levels in raster 4x4 block/frequency order."""
    out = np.zeros((*SHAPE, 16), np.int16)
    with path.open("rb") as f:
        for t, y, x in zip(*np.where(mode == 1)):
            assert struct.unpack("<II", f.read(8)) == (t + 1, y * 52 + x)
            for block in range(27):
                count, = f.read(1)
                scan = -1
                for level, run in struct.iter_unpack("<iB", f.read(5 * count)):
                    scan += run + 1
                    if block < 16:
                        b8, b4 = divmod(block, 4)
                        raster = (2 * (b8 // 2) + b4 // 2) * 4 + 2 * (b8 % 2) + b4 % 2
                        out[t, y, x, raster, SCAN[scan]] = level
        assert f.read(1) == b""
    return out


def transform(pixels):
    return fwd(fwd(pixels).swapaxes(-1, -2)).swapaxes(-1, -2)


def quantize(pixels):
    coef = transform(pixels)
    return ((np.abs(coef) * SCALE + (342 << 8)) >> 19) * np.sign(coef)


def mb_to_blocks(y):
    return y.reshape(*SHAPE[:3], 4, 4, 4, 4).swapaxes(-3, -2).reshape(*SHAPE, 4, 4)


def to_pixels(levels):
    """Synthesize a single-frequency pixel block at each selected 4x4 location."""
    assert (np.count_nonzero(levels, axis=-1) <= 1).all()
    blocks = np.zeros((*SHAPE, 4, 4), np.int16)
    for freq in CFG["frequencies"]["low"]:
        basis = BASIS[freq].reshape(4, 4)
        amps = np.arange(256 // int(abs(basis).max()) + (abs(basis).max() > 1))
        mapped = quantize(amps[:, None, None] * basis)[:, freq // 4, freq % 4]
        table = np.searchsorted(mapped, np.arange(MAGNITUDES + 1))
        q = levels[..., freq]
        use = q != 0
        index = table[np.abs(q[use])]
        assert np.all(index < len(amps)) and np.array_equal(mapped[index], np.abs(q[use]))
        blocks[use] = (amps[index] * np.sign(q[use]))[:, None, None] * basis
    assert np.array_equal(quantize(blocks).reshape(levels.shape), levels)
    return blocks.reshape(*SHAPE[:3], 4, 4, 4, 4).swapaxes(-3, -2).reshape(*SHAPE[:3], 16, 16)


def strict(folder, row, arrays, source):
    backend = p05.load_backend(KIT)
    run = backend.run

    def traced(cmd, cwd, log, env=None):
        if Path(cmd[0]).name == "lencod.exe":
            env = dict(env, JM_P05_BITS=str(folder / "bits.csv"), JM_P05_MC=str(folder / "trace.bin"))
        return run(cmd, cwd, log, env)

    backend.run = traced
    folder.mkdir()
    quality = replay(backend, JM, folder, row, arrays, source, {}, coeff_trace=True)
    trace, bits = p05.read_trace(folder, False)
    q = native_levels(folder / "coeff.bin", arrays["mode_id"])
    assert quality["encoder_matches_decoder"] and quality["decoder_agreement"]
    assert quality["oracle_p_bytes_read"] == 0 and quality["motion_search_s_reported"] == 0
    assert (np.count_nonzero(q, axis=(3, 4)) == trace["fields"][..., 2].reshape(16, 30, 52)).all()
    return quality, q, trace["fields"][..., :2].copy(), bits


def prep(split, row, output):
    root, source = paths(split, row)
    sample = row["sample_id"]
    fixed = p05.load_npz(root / "fixed_mode_mv.npz")
    teacher = root / CFG["teacher"]
    anchor = root / "A"
    meta = {r["arm"]: r for r in json.loads((root / "summary.json").read_text())}
    base = p05.load_npz(anchor / "prediction.npz")
    mode = base["mode_id"]
    for key in ("mode_id", "mv_qpel"):
        assert np.array_equal(base[key], fixed[key])
    state = p05.load_npz(teacher / "prediction.npz") if teacher.exists() else p05.load_npz(root / meta[CFG["teacher"]]["fallback"] / "prediction.npz")
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="p2label-") as tmp:
        work = Path(tmp) / "trace"
        quality, q, native, bits = strict(work, row, state, source)
        assert digest(work / "prediction.264") == meta[CFG["teacher"]]["stream_sha256"]
        assert np.array_equal(native, fixed["native_mv_qpel"].reshape(16, 1560, 2))
        assert np.array_equal(mode.reshape(16, 1560), bits[..., 2])
    assert not q[mode == 0].any()
    pre = quantize(mb_to_blocks(state["residual_y"])).reshape(q.shape)
    assert np.array_equal(q[q != 0], pre[q != 0])
    freq = CFG["frequencies"]["low"]
    file = output / "labels" / f"{sample}.npz"
    np.savez_compressed(file, levels=q[..., freq], mode=mode)
    eligible = int((mode == 1).sum()) * 16
    motion = np.linalg.norm(base["mv_qpel"].astype(np.float32), axis=-1)
    motion_bins = np.digitize(motion, [1, 8, 32])
    motion_stats = []
    for b in range(4):
        mask = (mode == 1) & (motion_bins == b)
        motion_stats.append([int(mask.sum()) * 16, int(np.count_nonzero(q[mask]))])
    return dict(split=split, sample=sample, prompt_id=hashlib.sha256(row["prompt"].encode()).hexdigest()[:16],
                hidden_path=row["hidden_path"], hidden_sha256=digest(row["hidden_path"]),
                teacher_stream_sha256=meta[CFG["teacher"]]["stream_sha256"],
                fixed_state_sha256=digest(root / "fixed_mode_mv.npz"), label_sha256=digest(file),
                eligible_blocks=eligible, frame_counts=np.count_nonzero(q, axis=(1, 2, 3, 4)).tolist(),
                frequency_counts=np.count_nonzero(q, axis=(0, 1, 2, 3)).tolist(),
                signed_levels=[dict(Counter(map(int, q[..., f][q[..., f] != 0]))) for f in freq],
                motion_bins=motion_stats)


@torch.inference_mode()
def feature(model, z, base):
    x = model.encode_features(z, z.new_ones(1))
    modes = torch.from_numpy(base["mode_id"]).to(x.device)
    vectors = torch.from_numpy(base["mv_qpel"]).to(x.device).float()
    return torch.cat((x, vectors.movedim(-1, 0)[None] / 32,
                      modes[None, None].float()), dim=1)


def prepare(output):
    output.mkdir(exist_ok=False)
    (output / "labels").mkdir()
    (output / "features").mkdir()
    assert digest(CKPT) == CFG["checkpoint_sha256"]
    (output / "locked_config.json").write_bytes((HERE / "config.json").read_bytes())
    rows = rows_by_split()
    model = load_model(CKPT).requires_grad_(False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = []
        for split, group in rows.items():
            for row in group:
                jobs.append(pool.submit(prep, split, row, output))
        for split, group in rows.items():
            for row in group:
                root, _ = paths(split, row)
                base = p05.load_npz(root / "A/prediction.npz")
                torch.save(feature(model, input_tensor(row, "hidden"), base).cpu(), output / "features" / (row["sample_id"] + ".pt"))
        report = [job.result() for job in jobs]
    save_json(output / "label_manifest.json", report)
    print("DIAGNOSTICS COMPLETE", len(report), flush=True)


class SparseHead(nn.Module):
    def __init__(self, channels, frequencies):
        super().__init__()
        self.frequencies = frequencies
        self.body = nn.Sequential(nn.Conv3d(channels, CFG["hidden_width"], (1, 3, 3), padding=(0, 1, 1)),
                                  nn.SiLU(), nn.Conv3d(CFG["hidden_width"], 32 * 16, 1), nn.SiLU())
        self.select = nn.Linear(32, frequencies)
        self.value = nn.Linear(32, frequencies * (MAGNITUDES + 1))

    def forward(self, x):
        x = self.body(x).movedim(1, -1).reshape(*x.shape[:1], *x.shape[2:], 16, 32)
        value = self.value(x).reshape(*x.shape[:-1], self.frequencies, MAGNITUDES + 1)
        return self.select(x), value[..., 0], value[..., 1:]


def data_for(output, rows, freqs):
    for row in rows:
        sid = row["sample_id"]
        state = p05.load_npz(output / "labels" / f"{sid}.npz")
        x = torch.load(output / "features" / f"{sid}.pt", weights_only=True)
        q = state["levels"][..., [CFG["frequencies"]["low"].index(f) for f in freqs]]
        yield x, torch.from_numpy(state["mode"] == 1), torch.from_numpy(q)


def losses(pred, valid, q):
    p, sign, mag = (v[0][valid] for v in pred)
    q = q[valid].long()
    active = q != 0
    selection = F.binary_cross_entropy_with_logits(p, active.float(), pos_weight=p.new_tensor(CFG["select_pos_weight"]))
    if active.any():
        value = F.binary_cross_entropy_with_logits(sign[active], (q[active] > 0).float())
        value = value + F.cross_entropy(mag[active], q[active].abs() - 1)
    else:
        value = selection.new_zeros(())
    return selection, value


def train(output, variant, seed):
    assert (output / "locked_config.json").read_bytes() == (HERE / "config.json").read_bytes()
    freqs = CFG["frequencies"][variant]
    rows = rows_by_split()
    samples = list(data_for(output, rows["train"], freqs))
    torch.manual_seed(seed)
    model = SparseHead(samples[0][0].shape[1], len(freqs)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    history = []
    for epoch in range(CFG["epochs"]):
        train_losses = []
        for index in torch.randperm(len(samples)).tolist():
            x, keep, q = (tensor.cuda() for tensor in samples[index])
            optimizer.zero_grad(set_to_none=True)
            select, value = losses(model(x), keep, q)
            (select + CFG["value_loss_weight"] * value).backward()
            optimizer.step()
            train_losses.append([select.item(), value.item()])
        history.append(dict(epoch=epoch+1, loss=np.mean(train_losses, axis=0).tolist()))
        print(variant, seed, history[-1], flush=True)
    dest = output / f"{variant}-{seed}"
    dest.mkdir()
    torch.save(dict(weights=model.state_dict(), frequencies=freqs, channels=samples[0][0].shape[1],
                    seed=seed, params=sum(p.numel() for p in model.parameters())), dest / "model.pt")
    save_json(dest / "history.json", history)


def max_level(freq):
    basis = BASIS[freq].reshape(4, 4)
    return int(quantize(basis * (255 // abs(basis).max())).ravel()[freq])


def predict_scores(model, x, freqs):
    probability, sign, mag = (item[0] for item in model(x))
    magnitude = torch.stack([mag[..., i, :max_level(f)].argmax(-1) + 1
                             for i, f in enumerate(freqs)], dim=-1)
    value = magnitude * torch.where(sign > 0, 1, -1)
    # Native JM may discard isolated level-one blocks. Levels of magnitude >=2
    # always exceed its block discard threshold under this QP28 profile.
    return probability.sigmoid().cpu().numpy(), value.cpu().numpy()


def choose_levels(prob, value, mode, freqs, policy, budget):
    frequency = prob.argmax(-1, keepdims=True)
    scores = np.take_along_axis(prob, frequency, axis=-1)[..., 0]
    q = np.take_along_axis(value, frequency, axis=-1)[..., 0]
    scores = np.where((mode == 1)[..., None] & (np.abs(q) >= 2), scores, -np.inf)
    chosen = np.zeros(scores.size, dtype=bool)
    if policy == "threshold":
        chosen[:] = scores.ravel() >= budget
    else:
        groups = scores.reshape(16 if policy == "frame" else 1, -1)
        order = np.argsort(-groups, axis=1, kind="stable")[:, :budget]
        np.put_along_axis(chosen.reshape(groups.shape), order, np.isfinite(np.take_along_axis(groups, order, 1)), 1)
    keep = chosen.reshape(SHAPE)
    levels = np.zeros((*SHAPE, 16), np.int16)
    np.put_along_axis(levels, np.asarray(freqs)[frequency], (q * keep)[..., None], axis=-1)
    return levels


def last4(quality):
    return mean(quality["psnr_y_by_frame"][-4:])


def evaluate_clip(job):
    dest, row, policy, budget, levels, timing, diag = job
    root, source = paths("val", row)
    base = p05.load_npz(root / "A/prediction.npz")
    fixed = p05.load_npz(root / "fixed_mode_mv.npz")
    meta = {r["arm"]: r for r in json.loads((root / "summary.json").read_text())}
    mode = base["mode_id"]
    assert np.array_equal(fixed["mode_id"], mode)
    assert np.array_equal(fixed["mv_qpel"], base["mv_qpel"])
    assert not levels[mode == 0].any()
    arrays = dict(base, residual_y=to_pixels(levels))
    case = dest / f"{policy}_{budget}" / row["sample_id"]
    case.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="p2eval-") as tmp:
        work = Path(tmp) / "jm"
        quality, actual, native, bits = strict(work, row, arrays, source)
        assert np.array_equal(actual, levels), "JM level mismatch"
        assert np.array_equal(native, fixed["native_mv_qpel"].reshape(16, 1560, 2))
        assert np.array_equal(bits[..., 2], mode.reshape(16, 1560))
        p05.preserve(work, case)
        for name in ("coeff.bin", "bits.csv"):
            (case / name).write_bytes((work / name).read_bytes())
    base_bits = meta["A"]["bitstream_bytes"] * 8
    point = dict(policy=policy, budget=budget, sample=row["sample_id"],
                 prompt=row["prompt"], rate_pct=100*(quality["bitstream_bytes"]*8/base_bits - 1),
                 delta_db=quality["psnr_y_p_db"]-meta["A"]["psnr_y_p_db"],
                 last4_db=last4(quality), delta_last4_db=last4(quality)-last4(meta["A"]),
                 delta_offline_3_db=quality["psnr_y_p_db"]-meta[CFG["teacher"]]["psnr_y_p_db"],
                 stream_sha256=digest(case / "prediction.264"),
                 coeff_bits=int(bits[..., 6:].sum()), luma_nonzero=int(np.count_nonzero(levels)),
                 frame_luma_nonzero=np.count_nonzero(levels, axis=(1, 2, 3, 4)).tolist(),
                 frame_coeff_bits=bits[..., 6:].sum(axis=(1, 2)).tolist(),
                 **quality, **timing, **diag)
    save_json(case / "metrics.json", point)
    return point


def summarize(points):
    groups = defaultdict(list)
    for point in points:
        groups[point["prompt"]].append(point)
    keys = ("delta_db", "delta_last4_db", "rate_pct", "bitrate_kbps", "psnr_y_p_db", "last4_db", "delta_offline_3_db")
    means = {key: mean(mean(r[key] for r in group) for group in groups.values()) for key in keys}
    worst = min(points, key=lambda r:r["delta_db"])
    primary = all(r["rate_pct"] <= CFG["primary_rate_pct"] + 1e-8 for r in points)
    maximum = all(r["rate_pct"] <= CFG["absolute_rate_pct"] + 1e-8 for r in points)
    passed = (primary and means["delta_db"] > CFG["min_mean_delta_db_exclusive"]
              and means["delta_last4_db"] > CFG["min_last4_delta_db_exclusive"]
              and worst["delta_db"] >= CFG["severe_clip_drop_db"])
    return dict(policy=points[0]["policy"], budget=points[0]["budget"], **means,
                primary_per_clip_cap=primary, maximum_per_clip_cap=maximum, passed=passed,
                worst_clip=worst["sample"], worst_delta_db=worst["delta_db"],
                prompt_delta_db={key:mean(r["delta_db"] for r in group) for key,group in groups.items()})


@torch.inference_mode()
def evaluate(output, variant, seed):
    dest = output / f"{variant}-{seed}"
    saved = torch.load(dest / "model.pt", weights_only=True)
    freqs = saved["frequencies"]
    model = SparseHead(saved["channels"], len(freqs)).cuda().eval()
    model.load_state_dict(saved["weights"])
    rows = rows_by_split()["val"]
    policies = [("video", k) for k in CFG["video_budgets"]]
    if variant == "low":
        policies += [("frame", k) for k in CFG["frame_budgets"]]
        policies += [("threshold", k) for k in CFG["thresholds"]]
    jobs = []
    for row, (x, keep, q) in zip(rows, data_for(output, rows, freqs)):
        x = x.cuda()
        predict_scores(model, x, freqs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        before = torch.cuda.memory_allocated()
        probability, value = predict_scores(model, x, freqs)
        timing = dict(student_head_seconds=time.perf_counter()-start,
                      student_peak_extra_bytes=torch.cuda.max_memory_allocated()-before)
        q = q.numpy()
        positive = q != 0
        diag = dict(teacher_positive=int(positive.sum()), eligible_positions=int(keep.sum())*16*len(freqs),
                    teacher_pos_sign_correct=int((np.sign(value[positive]) == np.sign(q[positive])).sum()),
                    teacher_pos_magnitude_correct=int((np.abs(value[positive]) == np.abs(q[positive])).sum()))
        for policy, k in policies:
            levels = choose_levels(probability, value, keep.numpy(), freqs, policy, k)
            chosen = levels[..., freqs] != 0
            jobs.append((dest,row,policy,k,levels,timing,
                         dict(diag, selected_true_positive=int((chosen & positive).sum()))))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate_clip, jobs))
    save_json(dest / "metrics.json", results)
    fields = [k for k, v in results[0].items() if isinstance(v, (str, int, float, bool))]
    with (dest / "per_video.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    save_json(dest / "rd.json", [summarize([r for r in results if (r["policy"],r["budget"])==(p,k)])
                                 for p,k in policies])
    print("EVALUATION COMPLETE", variant, seed, len(results), flush=True)


@torch.inference_mode()
def export(checkpoint, hidden_path, output, policy, budget):
    """Deployment entry point. The scorer and source assets are outside this path."""
    if digest(CKPT) != CFG["checkpoint_sha256"]:
        raise ValueError("Frozen A checkpoint mismatch")
    frozen = load_model(CKPT).requires_grad_(False)
    z = input_tensor({"hidden_path": hidden_path}, "hidden")
    base, _ = predict(frozen, z, 20)
    saved = torch.load(checkpoint, weights_only=True)
    freqs = saved["frequencies"]
    head = SparseHead(saved["channels"], len(freqs)).cuda().eval()
    head.load_state_dict(saved["weights"])
    probability, value = predict_scores(head, feature(frozen, z, base), freqs)
    q = choose_levels(probability, value, base["mode_id"], freqs, policy, budget)
    base["residual_y"] = to_pixels(q)
    np.savez_compressed(output, **base)


def smoke(output):
    row = rows_by_split()["train"][0]
    root, source = paths("train", row)
    base = p05.load_npz(root / "A/prediction.npz")
    mode = base["mode_id"]
    q = np.zeros((*SHAPE, 16), np.int16)
    locations = np.argwhere(mode == 1)
    for j, freq in enumerate(CFG["frequencies"]["low"]):
        max_q = max_level(freq)
        for i, value in enumerate([*range(-max_q, -1), *range(2, max_q + 1)]):
            q[tuple(locations[j * 128 + i]) + (i % 16, freq)] = value
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="p2smoke-") as tmp:
        folder = Path(tmp) / "jm"
        quality, actual, native, bits = strict(folder, row, dict(base, residual_y=to_pixels(q)), source)
        assert np.array_equal(actual, q)
        fixed = p05.load_npz(root / "fixed_mode_mv.npz")
        assert np.array_equal(native, fixed["native_mv_qpel"].reshape(16, 1560, 2))
        assert np.array_equal(bits[..., 2], mode.reshape(16, 1560))
        proof = output / "adapter_smoke"
        p05.preserve(folder, proof)
        (proof / "coeff.bin").write_bytes((folder / "coeff.bin").read_bytes())
        save_json(proof / "verification.json", dict(**quality, exact_levels=int(np.count_nonzero(q)),
                                                   max_levels=[max_level(f) for f in CFG["frequencies"]["low"]]))
    print("ADAPTER SMOKE PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "train", "evaluate", "export"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=tuple(CFG["frequencies"]), default="dc")
    parser.add_argument("--seed", type=int, choices=CFG["seeds"], default=CFG["seeds"][0])
    parser.add_argument("--student", type=Path)
    parser.add_argument("--hidden", type=Path)
    parser.add_argument("--policy", choices=("video", "frame", "threshold"), default="video")
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "smoke":
        smoke(args.output)
    elif args.phase == "train":
        train(args.output, args.variant, args.seed)
    elif args.phase == "evaluate":
        evaluate(args.output, args.variant, args.seed)
    else:
        if args.student is None or args.hidden is None or args.budget is None:
            parser.error("export requires --student, --hidden and --budget")
        budget = args.budget if args.policy == "threshold" else int(args.budget)
        export(args.student, args.hidden, args.output, args.policy, budget)


if __name__ == "__main__":
    main()
