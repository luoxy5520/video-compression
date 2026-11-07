"""Train fixed-mode/MV residuals on references reached by the student's own replay."""
import argparse
import copy
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate import load_backend, replay, state_metrics
from oracle_residual import blocks, parse_trace, yuv
from state import PLANES, input_tensor, load_model, predict, records, targets


def replay_item(a, backend, folder, row, arrays, trace):
    folder.mkdir(parents=True)
    raw = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
    metrics = replay(backend, a.jm, folder, row, arrays, raw, {}, mc_trace=trace)
    if trace:
        mc, masks = parse_trace(folder / "mc.bin")
        keep = arrays["mode_id"] == 1
        residuals = []
        for i, ((plane, size, _, _), source) in enumerate(zip(PLANES, yuv(raw))):
            mask = blocks(masks[i], size)
            if not np.array_equal(mask, np.broadcast_to(keep[..., None, None], mask.shape)):
                raise ValueError(f"Incomplete on-policy MC coverage: {folder} {plane}")
            diff = blocks(source.astype(np.int16) - mc[i].astype(np.int16), size)
            diff = np.where(mask, diff, 0)
            residuals.append(diff.reshape(*keep.shape, size*size))
        np.savez_compressed(folder / "rollout_target.npz", mask=keep,
                            residual=np.concatenate(residuals, axis=-1))
    metrics.update(sample_id=row["sample_id"], prompt=row["prompt"], step=20)
    return metrics


def replay_rows(a, backend, model, rows, folder, trace=False, score_state=False):
    reports = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = []
        for row in rows:
            arrays, timing = predict(model, input_tensor(row, model.kind), 20)
            case = folder / row["sample_id"]
            job = pool.submit(replay_item, a, backend, case, row, arrays, trace)
            score = state_metrics(arrays, targets(row)) if score_state else None
            jobs.append((job, score, timing))
        for job, score, timing in jobs:
            result = job.result()
            result.update(timing)
            if score_state:
                result["state"] = score
            reports.append(result)
            print("REPLAY", folder.name, result["sample_id"], round(result["psnr_y_p_db"], 3), flush=True)
    return reports


@torch.inference_mode()
def features(model, rows):
    cache = {}
    for row in rows:
        z = input_tensor(row, model.kind)
        x = model.encode_features(z, z.new_tensor([1.]))
        motion = model.motion(x)
        cache[row["sample_id"]] = torch.cat((x, motion), dim=1).half().cpu()
    return cache


def load_rollout(folder, sid):
    with np.load(folder/sid/"rollout_target.npz", allow_pickle=False) as data:
        return torch.from_numpy(data["mask"]), torch.from_numpy(data["residual"])


def save(model, path):
    torch.save(dict(weights=model.state_dict(), kind=model.kind, recurrent=False,
                    width=model.mode.in_channels), path)


def select(validation, baseline):
    def avg(rows, k):
        return statistics.mean(r[k] for r in rows)
    # Constrain bitrate against the existing project threshold on full-JM validation.
    eligible = [k for k, v in validation.items()
                if avg(v, "bitrate_kbps") <= 1.10 * baseline]
    if not eligible:
        raise ValueError("No eligible validation candidates")
    return max(eligible, key=lambda k: avg(validation[k], "psnr_y_p_db"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--test-config", nargs="+", required=True)
    for key in ("checkpoint", "kit", "jm", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(19)
    rng = random.Random(19)
    a.output.mkdir(parents=True, exist_ok=False)
    index = records(a.config)
    train, val = ([r for r in index[s] if r["step"] == 20] for s in ("train", "val"))
    if a.smoke:
        train, val = train[:1], val[:1]
        a.rounds = a.epochs = 1
    test = [r for r in records(a.test_config)["test"] if r["step"] == 20]
    used = {r["prompt"] for r in train + val}
    assert used.isdisjoint(r["prompt"] for r in test)
    base = load_model(a.checkpoint)
    assert not base.recurrent
    model = copy.deepcopy(base).eval().requires_grad_(False)
    model.residual.requires_grad_(True)
    backend = load_backend(a.kit)
    cache = features(base, train)
    validation = {"base": replay_rows(a, backend, base, val, a.output/"val_base")}
    native_kbps = statistics.mean(Path(r["current_oracle_state_path"]).with_name("reference.264")
                                  .stat().st_size*128/17000 for r in val)
    history = []
    checkpoints = {"base": a.checkpoint}
    for n in range(a.rounds):
        tr = a.output/f"rollout_{n}"
        replay_rows(a, backend, model, train, tr, trace=True)
        data = {sid:load_rollout(tr, sid) for sid in cache}
        if n == 0:
            initial = data
        opt = torch.optim.AdamW(model.residual.parameters(), lr=.0005)
        order = list(data)
        for e in range(a.epochs):
            rng.shuffle(order)
            losses = []
            for sid in order:
                x = cache[sid].cuda().float()
                y = model.residual(x)[0].movedim(0, -1)
                # Mix the current and initial trajectories to reduce target chasing.
                refs = (data[sid],) if n == 0 else (data[sid], initial[sid])
                parts = []
                for mask, residual in refs:
                    m = mask.cuda()
                    if m.any():
                        target = residual.cuda()[m].float()/16
                        parts.append(F.smooth_l1_loss(y[m], target, beta=.25))
                if not parts:
                    continue
                loss = torch.stack(parts).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 5.)
                opt.step()
                losses.append(loss.item())
            if not losses:
                raise ValueError("No P16 supervision")
            entry = dict(round=n+1, epoch=e+1, train_loss=statistics.mean(losses))
            history.append(entry)
            print(json.dumps(entry), flush=True)
        tag = f"round{n+1}"
        cp = a.output/f"{tag}.pt"
        save(model, cp)
        checkpoints[tag] = cp
        validation[tag] = replay_rows(a, backend, model, val, a.output/f"val_{tag}")
    chosen = select(validation, native_kbps)
    assert all(torch.equal(base.state_dict()[k], model.state_dict()[k])
               for k in base.state_dict() if not k.startswith("residual."))
    final = load_model(checkpoints[chosen])
    test_results = {}
    if not a.smoke:
        for tag, arm in (("base", base), ("selected", final)):
            test_results[tag] = replay_rows(a, backend, arm, test, a.output/f"test_{tag}", score_state=True)
        for r in test:
            sid = r["sample_id"]
            with np.load(a.output/"test_base"/sid/"prediction.npz") as x, np.load(a.output/"test_selected"/sid/"prediction.npz") as y:
                assert np.array_equal(x["mode_id"],y["mode_id"]) and np.array_equal(x["mv_qpel"],y["mv_qpel"])
    (a.output/"results.json").write_text(json.dumps(dict(selected=chosen, validation=validation,
        test=test_results, history=history, validation_native_kbps=native_kbps, smoke=a.smoke,
        note="Only residual layers update; MV and mode frozen; prior holdout is a regression set."),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
