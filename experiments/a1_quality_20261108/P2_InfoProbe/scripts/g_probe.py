"""Matched residual probes with frozen A references and strict native replay."""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from d0 import p
from state import latent
from sweep import blocks, luma

HIST = p.ROOT / "P2_Lite/run01"
FREQS = [0, 1, 4]
MAXIMA = np.array([p.max_level(f) for f in FREQS])
BUDGET = 64
FIXED_COUNT = 8192


def raster(a):
    return a.reshape(30, 4, 52, 4, *a.shape[1:]).swapaxes(1, 2).reshape(30, 52, 16, *a.shape[1:])


def prepare_item(split, row, out):
    root, source = p.paths(split, row)
    sid = row["sample_id"]
    a = p.p05.load_npz(root / "A/prediction.npz")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
                                   "-i", str(root / "A/prediction.264"), "-threads", "1",
                                   "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
    refs = np.frombuffer(raw, np.uint8).reshape(17, -1)[:-1, :832*480].reshape(16, 480, 832)
    truth = p.p05.read_yuv(source)[0]
    levels, dc = [], []
    for t, ref in enumerate(refs):
        residual = blocks(truth[t+1].astype(np.int16) - luma(ref, a["mv_qpel"][t]).astype(np.int16))
        dc.append(raster(residual.mean(1)))
        q = p.quantize(residual.reshape(-1, 4, 4)).reshape(-1, 16)
        levels.append(raster(q[:, FREQS]))
    q = np.stack(levels).astype(np.int16)
    valid = a["mode_id"] == 1
    q[~valid] = 0
    rgb = refs.reshape(16, 30, 4, 4, 52, 4, 4).mean(axis=(3, 6))
    rgb = rgb.transpose(0, 1, 3, 2, 4).reshape(16, 30, 52, 16)
    name = out / f"{sid}.npz"
    np.savez_compressed(name, q=q, dc=np.stack(dc).astype(np.float16), valid=valid,
                        ref=rgb.astype(np.float16))
    return dict(split=split, sample=sid, labels_sha256=p.digest(name),
                source_sha256=p.digest(source), a_stream_sha256=p.digest(root / "A/prediction.264"),
                feature_sha256=p.digest(HIST / "features" / f"{sid}.pt"))


def prepare(out):
    out.mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(prepare_item, split, row, out)
                for split, rows in p.rows_by_split().items() for row in rows]
        manifest = [job.result() for job in jobs]
    p.save_json(out / "manifest.json", manifest)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = p.SparseHead(83, 3).body
        self.output = nn.Linear(32, 7)

    def forward(self, x):
        return self.output(self.body(x).movedim(1, -1).reshape(*p.SHAPE, 32))


def subset(rows):
    unique = {}
    for row in rows:
        unique.setdefault(row["prompt"], row)
    return list(unique.values())[:4]


@torch.inference_mode()
def alternative(row, group, frozen, middle):
    if group == "G3":
        z = latent(row["latent_path"])
        b, c, t, h, w = z.shape
        x = z.reshape(b,c,t,h//2,2,w//2,2).permute(0,1,4,6,2,3,5).reshape(b,c*4,t,h//2,w//2)
        x = F.silu(frozen.stem(x) + frozen.step(x.new_ones((1, 1)))[:, :, None, None, None])
        for conv in frozen.body:
            x = x + F.silu(conv(x))
        x = frozen.temporal(x[:, :, 1:]).reshape(b,4,-1,4,h//2,w//2)
        return x.permute(0,2,3,1,4,5).flatten(2,3).cpu()
    z = torch.load(middle / f"{row['sample_id']}.pt", weights_only=True).cuda().float()
    return frozen.encode_features(z, z.new_ones(1)).cpu()


def data(labels, row, group, alternative_features=None):
    sid = row["sample_id"]
    y = p.p05.load_npz(labels / f"{sid}.npz")
    cached = torch.load(HIST / "features" / f"{sid}.pt", weights_only=True)
    if alternative_features is not None:
        cached[:, :64] = alternative_features[sid]
    if group == "G0":
        cached[:, 64:] = 0
    ref = torch.from_numpy(y["ref"]).float().movedim(-1, 0)[None] / 127.5 - 1
    if group != "G2":
        ref.zero_()
    x = torch.cat((cached, ref), 1).cuda()
    valid = torch.from_numpy(y["valid"]).cuda()
    q = torch.from_numpy(y["q"]).cuda().float()
    dc = torch.from_numpy(y["dc"]).cuda().float()
    rng = np.random.default_rng(52020)
    indices = rng.choice(int(valid.sum()) * 16, FIXED_COUNT, replace=False)
    return x, valid, q, dc, torch.from_numpy(indices).cuda()


def loss(net, sample):
    x, mask, q, dc, idx = sample
    pred = net(x)[mask].reshape(-1, 7)
    q = q[mask].reshape(-1, 3)
    dc = dc[mask].flatten()
    max_q, freq = q.abs().max(-1)
    important = max_q.topk(BUDGET).indices
    positive = important * 3 + freq[important]
    ranked = pred[:, :3].flatten()
    negative = ranked.clone()
    negative[positive] = -torch.inf
    select = F.softplus(-ranked[positive]).mean() + F.softplus(negative.topk(BUDGET).values).mean()
    support = torch.cat((idx, important))
    c = F.smooth_l1_loss(pred[support, 3:6], q[support] / 8)
    r = F.smooth_l1_loss(pred[support, 6], dc[support] / 32)
    return select, c, r


def positions(score, q, mode):
    prob = np.where(np.abs(q) >= 2, score, -np.inf)
    return p.choose_levels(prob, q, mode, FREQS if q.shape[-1] == 3 else [0], "video", BUDGET)


@torch.inference_mode()
def infer(net, sample):
    x, mask, q, dc, idx = sample
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    initial = torch.cuda.memory_allocated()
    pred = net(x)
    c = pred[..., 3:6] * 8
    r = pred[..., 6] * 32
    mode = mask.cpu().numpy()
    score = pred[..., :3].cpu().numpy()
    c_dense = c.cpu().numpy()
    r_dense = r.cpu().numpy()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() - initial
    q_int = np.clip(np.rint(c_dense), -MAXIMA, MAXIMA).astype(np.int16)
    r_int = (np.sign(r_dense) * ((np.abs(r_dense) * (16 * p.SCALE[0, 0]) + (342 << 8)) // (1 << 19)))
    r_int = np.clip(r_int, -MAXIMA[0], MAXIMA[0]).astype(np.int16)
    c_fixed = c[mask].reshape(-1, 3)[idx].round().cpu().numpy()
    y = q[mask].reshape(-1, 3)[idx].cpu().numpy()
    rmse = (r[mask].flatten()[idx] - dc[mask].flatten()[idx]).square().mean().sqrt().item()
    sign = mean(float((np.sign(c_fixed[y * s > 0]) == s).mean()) for s in (-1, 1))
    c_levels = positions(score, q_int, mode)
    r_levels = positions(score[..., :1], r_int[..., None], mode)
    return dict(mae=float(np.abs(c_fixed-y).mean()), exact=float((c_fixed == y).mean()),
                sign_balanced_accuracy=sign, lowfreq_residual_rmse=rmse,
                student_head_seconds=elapsed, student_peak_extra_bytes=peak,
                c_nonzero=int(np.count_nonzero(c_levels)), r_nonzero=int(np.count_nonzero(r_levels))), c_levels, r_levels


def measure(dest, row, metrics, c, r):
    dest = dest / row["sample_id"]
    dest.mkdir()
    return {probe: p.evaluate_clip((dest, row, probe, BUDGET, levels, {}, metrics))
            for probe, levels in (("C", c), ("R", r))}


def fit(labels, out, group, seed, small=False, middle=None):
    out.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    rows = p.rows_by_split()
    if small:
        rows = {split: subset(group_rows) for split, group_rows in rows.items()}
    features = None
    if group in ("G3", "G4"):
        frozen = p.load_model(p.CKPT).eval().requires_grad_(False)
        features = {r["sample_id"]: alternative(r, group, frozen, middle)
                    for group_rows in rows.values() for r in group_rows}
    net = Head().cuda()
    optim = torch.optim.AdamW(net.parameters(), lr=0.001)
    history = []
    for epoch in range(12):
        losses = []
        for idx in torch.randperm(len(rows["train"])).tolist():
            sample = data(labels, rows["train"][idx], group, features)
            optim.zero_grad(set_to_none=True)
            parts = loss(net, sample)
            sum(parts).backward()
            optim.step()
            losses.append([v.item() for v in parts])
        history.append(np.mean(losses, axis=0).tolist())
        print(group, seed, epoch+1, history[-1], flush=True)
    torch.save(net.state_dict(), out / "head.pt")
    val = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = []
        for row in rows["val"]:
            metrics, c, r = infer(net.eval(), data(labels, row, group, features))
            jobs.append(pool.submit(measure, out, row, metrics, c, r))
        val = [job.result() for job in jobs]
    train = [infer(net, data(labels, row, group, features))[0] for row in rows["train"]]
    result = dict(group=group, seed=seed, params=sum(t.numel() for t in net.parameters()),
                  history=history, train_probe=train, head_sha256=p.digest(out / "head.pt"), val=val,
                  rd={key: p.summarize([v[key] for v in val]) for key in ("C", "R")})
    p.save_json(out / "metrics.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--group", choices=("G0", "G1", "G2", "G3", "G4"))
    parser.add_argument("--seed", type=int, choices=(1701, 1702))
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--middle", type=Path)
    a = parser.parse_args()
    torch.set_num_threads(2)
    if a.group:
        fit(a.labels, a.out, a.group, a.seed, a.small, a.middle)
    else:
        prepare(a.labels)


if __name__ == "__main__":
    main()
