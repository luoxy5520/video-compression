"""Fit and evaluate a final-latent shortlist and search-budget policy."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "residual_reference_20261107"), str(ROOT / "full_state_20261107")]
from benchmark import FPS, FRAMES, H, W, decode_check, load_backend, psnr
from train_rank import RankProjection, install
from train_source import load_config, load_input, rows_for
from train_tail import FrozenPrefix

BLOCKS = 16 * 30 * 52
CANDIDATES = 8


def proxy(row, output, vae):
    start = time.perf_counter()
    latent = load_input(row, "cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        rgb = ((vae.decode([latent[0]])[0] + 1) * 127.5).round().byte()
        packed = rgb.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
                    "-vf", "scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                    "-pix_fmt", "yuv420p", "-f", "rawvideo", str(output)],
                   input=packed.tobytes(), check=True)
    return time.perf_counter() - start


@torch.inference_mode()
def shortlist(path):
    # Search half-resolution luma, then evaluate integer-pel shifts at full resolution.
    y = np.fromfile(path, dtype=np.uint8).reshape(FRAMES, H*W*3//2)[:, :H*W].copy()
    y = torch.from_numpy(y).cuda().float().reshape(FRAMES, 1, H, W)
    half = F.avg_pool2d(y, 2).squeeze(1)
    reference = F.pad(half[:-1, None], (12, 12, 12, 12), mode="replicate").squeeze(1)
    offsets = [(x,y) for y in range(-12,13,2) for x in range(-12,13,2)]
    coarse = []
    for chunk in range(0, len(offsets), 16):
        shifted = torch.stack([reference[:, 12+dy:12+dy+H//2, 12+dx:12+dx+W//2]
                               for dx,dy in offsets[chunk:chunk+16]])
        errors = (shifted - half[1:]).abs().reshape(-1, 1, H//2, W//2)
        coarse.append(F.avg_pool2d(errors, 8).reshape(-1, 16, 30, 52))
    costs = torch.cat(coarse)
    grid = torch.tensor(offsets, device="cuda")
    costs = costs + .035 * grid.abs().sum(1)[:, None, None, None]
    base = 2 * grid[costs.topk(3, dim=0, largest=False).indices].reshape(3, BLOCKS, 2)
    current = F.unfold(y[1:], 16, stride=16).transpose(1, 2).reshape(BLOCKS, 256)
    previous = F.pad(y[:-1], (24,24,24,24), mode="replicate").flatten()
    t, row, col = torch.meshgrid(torch.arange(16, device="cuda"), torch.arange(30, device="cuda"),
                                 torch.arange(52, device="cuda"), indexing="ij")
    ry, rx = torch.meshgrid(torch.arange(16, device="cuda"), torch.arange(16, device="cuda"), indexing="ij")
    stride = W + 48
    origin = t.reshape(-1, 1) * (H + 48) * stride + (row.reshape(-1,1)*16 + 24 + ry.flatten()) * stride
    origin = origin + col.reshape(-1,1)*16 + 24 + rx.flatten()
    all_costs, all_vectors = [], []
    for centers in base:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                vec = centers + torch.tensor((dx,dy), device="cuda")
                valid = vec.abs().amax(dim=1) <= 24
                shift = vec.clamp(-24,24)
                error = (previous[origin + (shift[:,1] * stride + shift[:,0])[:,None]] - current).abs().mean(1)
                all_costs.append((error + .025 * shift.abs().sum(1)).masked_fill(~valid, torch.inf))
                all_vectors.append(vec)
    errors = torch.stack(all_costs).T
    vectors = torch.stack(all_vectors, dim=1)
    # Near-duplicate coarse centers can propose the same integer position.
    ids = (vectors[...,1]+26)*53 + vectors[...,0]+26
    unique_order = ids.argsort(dim=1, stable=True)
    ordered = ids.gather(1, unique_order)
    repeated = F.pad(ordered[:,1:] == ordered[:,:-1], (1,0), value=False)
    errors = errors.masked_fill(torch.zeros_like(repeated).scatter_(1, unique_order, repeated), torch.inf)
    best, index = errors.topk(CANDIDATES, largest=False)
    if not torch.isfinite(best).all():
        raise ValueError("Too few valid motion candidates")
    vectors = vectors.gather(1, index[...,None].expand(-1,-1,2))
    zero = (previous[origin] - current).abs().mean(1, keepdim=True)
    stats = torch.cat((zero/16, current.std(1,keepdim=True)/32, current.mean(1,keepdim=True)/255,
                       best/16, vectors.flatten(1)/24), dim=1)
    return vectors.cpu().numpy().astype(np.int16), stats.cpu().numpy()


def encode(a, backend, folder, raw, vectors, narrow, collect=False, skip=False):
    folder.mkdir()
    options = os.environ.copy()
    for key in ("JM_LATENT_MV_PRIOR","JM_LATENT_MV_EXTRA","JM_DECISION_SHORTLIST",
                "JM_DECISION_TRACE","JM_DECISION_MODE_TRACE"):
        options.pop(key,None)
    options["JM_STATE_MODE"] = "off"
    if vectors is not None:
        payload = np.zeros((BLOCKS, 2 + 2*CANDIDATES), dtype="<i2")
        payload[:,0] = np.asarray(narrow, dtype=np.int16) + 2*np.asarray(skip, dtype=np.int16)
        payload[:,1] = vectors.shape[1] if collect else (np.asarray(narrow)!=0)*vectors.shape[1]
        payload[:,2:2+2*vectors.shape[1]] = vectors.reshape(BLOCKS,-1)
        path = folder/"shortlist.bin"
        payload.tofile(path)
        options["JM_DECISION_SHORTLIST"] = str(path)
    if collect:
        options.update(JM_DECISION_TRACE=str(folder/"costs.bin"),
                       JM_DECISION_MODE_TRACE=str(folder/"modes.bin"))
    bitstream = folder/"prediction.264"
    settings = dict(InputFile=raw, OutputFile=bitstream, ReconFile=folder/"recon.yuv",
                    SourceWidth=W, SourceHeight=H, FramesToBeEncoded=FRAMES, FrameRate=FPS,
                    QPISlice=28, QPPSlice=28, RDOptimization=0, NumberReferenceFrames=1,
                    PList0References=1, SearchMode=-1, SearchRange=8)
    command = [a.jm/"bin/lencod.exe", "-d", a.kit/"JM-backend/cfg/encoder_baseline.cfg",
               "-f", a.kit/"linux/skip_p16.cfg"]
    for key, value in settings.items():
        command += ["-p", f"{key} = {value}"]
    start = time.perf_counter()
    backend.run(command,folder,folder/"encode.log",options)
    elapsed = time.perf_counter() - start
    decoded, decode_s = decode_check(backend,folder,bitstream)
    counts = re.search(r"Decision budget: narrow=(\d+) wide=(\d+) shortlist=(\d+)(?: skip=(\d+))?",
                       (folder/"encode.log").read_text())
    n, w, calls, early = (int(v or 0) for v in counts.groups()) if counts else (0, BLOCKS, 0, 0)
    if vectors is not None and (n+w+early != BLOCKS or calls+early != BLOCKS):
        raise ValueError(f"Missing native block decisions in {folder}: {n}, {w}, {calls}, {early}")
    return decoded, dict(encoder_s=elapsed, decode_s=decode_s, narrow_blocks=n, wide_blocks=w,
                         guarded_skip_blocks=early,
                         bitrate_kbps=bitstream.stat().st_size*8*FPS/FRAMES/1000,
                         decoder_agreement=True)


def read_labels(folder):
    costs = np.fromfile(folder/"costs.bin", dtype="<i8").reshape(-1, 7+CANDIDATES)
    modes = np.fromfile(folder/"modes.bin", dtype="<i2").reshape(-1,3)
    def align(records):
        ids = (records[:,0].astype(np.int64)-1)*1560 + records[:,1]
        order = ids.argsort()
        if not np.array_equal(ids[order], np.arange(BLOCKS)):
            raise ValueError(f"Incomplete native trace at {folder}")
        return records[order,2:]
    costs, modes = align(costs), align(modes)[:,0]
    if not np.isin(modes,[0,1]).all():
        raise ValueError("Unexpected macroblock mode")
    return costs, modes


def collect(a, rows, backend):
    saved = torch.load(a.projection, weights_only=True)
    vae = FrozenPrefix().vae
    projection = RankProjection(saved["channels"], saved["rank"]).cuda().eval()
    projection.load_state_dict(saved["weights"])
    install(vae, projection)
    for split in a.split:
        for row in rows[split]:
            sample = a.work/"samples"/row["sample_id"]
            if (sample/"features.npz").exists():
                continue
            staging = sample.with_name(sample.name + f".{os.getpid()}.partial")
            staging.mkdir(parents=True, exist_ok=False)
            output = sample
            sample = staging
            start = time.perf_counter()
            vae_s = proxy(row, sample/"proxy.yuv", vae)
            vectors, features = shortlist(sample/"proxy.yuv")
            proposal_s = time.perf_counter()-start-vae_s
            if split != "test":
                folder = sample/"teacher"
                encode(a, backend, folder, sample/"proxy.yuv", vectors, 0, collect=True)
                costs, modes = read_labels(folder)
                np.savez_compressed(sample/"labels.npz", costs=costs, modes=modes)
            np.savez_compressed(sample/"features.npz", vectors=vectors, features=features,
                                vae_s=vae_s, proposal_s=proposal_s)
            sample.rename(output)
            print(json.dumps(dict(sample=row["sample_id"], split=split, vae_s=vae_s,
                                  proposal_s=proposal_s)), flush=True)


class DecisionHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(features, 64), nn.SiLU(),
                                  nn.Linear(64, 64), nn.SiLU())
        self.rank = nn.Linear(64, CANDIDATES)
        self.skip = nn.Linear(64, 1)

    def forward(self, features):
        hidden = self.body(features)
        return self.rank(hidden), self.skip(hidden).squeeze(-1)


def features_at(a, row):
    sample = a.work/"samples"/row["sample_id"]
    with np.load(sample/"features.npz", allow_pickle=False) as saved:
        return sample, dict(saved)


def fit(a, rows):
    torch.manual_seed(42)
    cache = {}
    for split in ("train", "val"):
        data = []
        for row in rows[split]:
            sample, saved = features_at(a, row)
            with np.load(sample/"labels.npz", allow_pickle=False) as label:
                costs = label["costs"].astype(np.float64)
                # Ratio relative to JM's wide-window winner on the same reconstructed reference.
                regret = np.log((costs[:,5:]+1) / (costs[:,:1]+1)).clip(-2, 3)
                data.append((saved["features"], regret.astype(np.float32), label["modes"]==0))
        cache[split] = tuple(torch.from_numpy(np.concatenate(x)).cuda() for x in zip(*data))
    features, target, skip = cache["train"]
    val_features, val_target, val_skip = cache["val"]
    model = DecisionHead(features.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    best = float("inf")
    history = []
    for epoch in range(20):
        model.train()
        for ids in torch.randperm(len(features), device="cuda").split(4096):
            rank, mode = model(features[ids])
            loss = F.smooth_l1_loss(rank, target[ids]) + 0.2 * F.binary_cross_entropy_with_logits(
                mode, skip[ids].float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            rank, mode = model(val_features)
            score = float(F.smooth_l1_loss(rank, val_target) +
                          0.2 * F.binary_cross_entropy_with_logits(mode, val_skip.float()))
            selected = rank.topk(4, largest=False).indices
            stats = dict(epoch=epoch+1, validation_loss=score,
                         top4_mean_positive_log_regret=float(val_target.gather(1, selected).amin(1).clamp_min(0).mean()),
                         proxy_top4_mean_positive_log_regret=float(val_target[:,:4].amin(1).clamp_min(0).mean()),
                         mode_accuracy=float(((mode>=0)==val_skip).float().mean()),
                         mode_majority_accuracy=float(max(val_skip.float().mean(), (~val_skip).float().mean())))
        if score < best:
            best = score
            torch.save(model.state_dict(), a.work/"decision.pt")
        history.append(stats)
        print(json.dumps(stats), flush=True)
    (a.work/"training.json").write_text(json.dumps(history, indent=2))


def evaluate(a, rows, backend, candidates_only=False):
    output = a.work/a.eval_name
    output.mkdir(exist_ok=False)
    results = []
    model = None
    thresholds = None
    for split in ("val", "test"):
        for row in rows[split]:
            sample, saved = features_at(a, row)
            if model is None:
                model = DecisionHead(saved["features"].shape[1]).cuda().eval()
                model.load_state_dict(torch.load(a.work/"decision.pt", weights_only=True))
            with torch.inference_mode():
                start = time.perf_counter()
                ranks, mode = model(torch.from_numpy(saved["features"]).cuda())
                top = ranks.topk(4, largest=False).indices.cpu().numpy()
                risk = ranks.amin(1).cpu().numpy()
                skip = (mode.sigmoid()>=.90).cpu().numpy()
                head_s = time.perf_counter()-start
            if thresholds is None:
                # Candidate thresholds are proposed once from the first validation clip.
                thresholds = {str(p): float(np.quantile(risk,p)) for p in (.25,.5,.75,.9)}
            ranked = np.take_along_axis(saved["vectors"], top[...,None], axis=1)
            cases = dict(native8=(None,0,0.0,False),
                         local1=(saved["vectors"][:,:0],1,0.0,False),
                         proxy4=(saved["vectors"][:,:4],1,0.0,False),
                         learned4=(ranked,1,head_s,False),
                         skipguard=(ranked,0,head_s,skip))
            if candidates_only:
                cases = dict(proxy_only4=(saved["vectors"][:,:4],4,0.0,False),
                             learned_only1=(ranked[:,:1],4,head_s,False),
                             learned_only4=(ranked,4,head_s,False),
                             learned_only4_skip=(ranked,4,head_s,skip),
                             learned_local4_skip=(ranked,1,head_s,skip))
            levels = {} if candidates_only else (thresholds if split=="val" else selected)
            for name, limit in levels.items():
                cases[f"gate_{name}"] = (ranked,risk<=limit,head_s,False)
                cases[f"joint_{name}"] = (ranked,risk<=limit,head_s,skip)
            for name, (vectors, narrow, policy_s, skip_mask) in cases.items():
                folder = output/f"{row['sample_id']}_{name}"
                decoded, stats = encode(a, backend, folder, sample/"proxy.yuv", vectors, narrow, skip=skip_mask)
                item = dict(split=split, sample_id=row["sample_id"], variant=name,
                            **stats, **psnr(row["raw_path"], decoded),
                            vae_s=float(saved["vae_s"]), policy_s=policy_s,
                            proposal_s=float(saved["proposal_s"]) if vectors is not None and vectors.shape[1] else 0)
                item["latent_to_bitstream_s"] = item["vae_s"]+item["proposal_s"]+policy_s+item["encoder_s"]
                results.append(item)
                print(json.dumps(item), flush=True)
        if split == "val" and not candidates_only:
            baseline = means(results, split, "native8")
            valid = []
            for name, threshold in thresholds.items():
                candidate = means(results, split, f"joint_{name}")
                if (candidate["psnr_y_p_db"] >= baseline["psnr_y_p_db"]-.15 and
                    candidate["bitrate_kbps"] <= baseline["bitrate_kbps"]*1.02):
                    valid.append((candidate["narrow_blocks"]+candidate["guarded_skip_blocks"], name, threshold))
            selected = {}
            if valid:
                _, name, threshold = max(valid)
                selected[name] = threshold
            (output/"calibration.json").write_text(json.dumps(dict(
                thresholds=thresholds, selected=selected,
                max_val_psnr_drop=.15, max_val_rate_ratio=1.02), indent=2))
    summary = {split: {name:means(results,split,name) for name in sorted(
        {r["variant"] for r in results if r["split"]==split})} for split in ("val","test")}
    (output/"results.json").write_text(json.dumps(dict(means=summary, per_video=results),indent=2))


def means(results, split, name):
    records = [r for r in results if r["split"]==split and r["variant"]==name]
    keys = ("psnr_y_p_db","psnr_yuv_p_db","bitrate_kbps","latent_to_bitstream_s","encoder_s",
            "narrow_blocks","guarded_skip_blocks")
    return {key:float(np.mean([r[key] for r in records])) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    for key in ("config","data","kit","jm","work","projection"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--split", nargs="+", default=["train","val","test"],
                        choices=["train","val","test"])
    parser.add_argument("--stage", choices=["collect","fit","evaluate","candidates-only"], default="collect")
    parser.add_argument("--eval-name", default="eval_guarded")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, _, index = load_config(args.config)
    rows = rows_for(config, index, args.data)
    if args.stage == "fit":
        fit(args, rows)
    elif args.stage in ("evaluate","candidates-only"):
        evaluate(args, rows, load_backend(args.kit), candidates_only=args.stage=="candidates-only")
    else:
        collect(args, rows, load_backend(args.kit))


if __name__ == "__main__":
    main()
