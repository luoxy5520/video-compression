"""Fit P-frame motion and pixel residuals on the existing grouped pilot."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "expansion_20261106"))
from train_mode import Head


class StateRegressor(nn.Module):
    """Separate regression trunk initialized from the existing mode head."""

    def __init__(self, mode_head):
        super().__init__()
        import copy
        self.conv = copy.deepcopy(mode_head.conv)
        self.time = copy.deepcopy(mode_head.t)
        self.motion = nn.Conv3d(32, 2, 1)
        self.residual = nn.Sequential(nn.Conv3d(34, 32, 1), nn.SiLU(), nn.Conv3d(32, 384, 1))

    def forward(self, latent, step):
        features = F.silu(self.conv(latent) + self.time(step.reshape(-1, 1))[:, :, None, None, None])
        features = F.interpolate(features, size=(17, 30, 52), mode="trilinear", align_corners=False)[:, :, 1:]
        motion = self.motion(features)
        residual = self.residual(torch.cat((features, motion), dim=1))
        return motion.movedim(1, -1), residual.movedim(1, -1)


def load_config(path):
    config = json.loads(Path(path).read_text())
    dataset = Path(config["dataset"])
    rows = [json.loads(line) for line in (dataset / "manifests/all.jsonl").read_text().splitlines()]
    split_sets = [set(ids) for ids in config["splits"].values()]
    for i, ids in enumerate(split_sets):
        prompts = {r["prompt"] for r in rows if r["sample_id"] in ids}
        for other in split_sets[i+1:]:
            if ids & other or prompts & {r["prompt"] for r in rows if r["sample_id"] in other}:
                raise ValueError("Sample or prompt crosses splits")
    index = {(r["sample_id"], int(r["input_step"])): r for r in rows}
    return config, dataset, index


def networks(checkpoint, device):
    mode = Head().to(device)
    mode.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    mode.eval().requires_grad_(False)
    return mode, StateRegressor(mode).to(device)


def read_target(path):
    with np.load(path, allow_pickle=False) as data:
        mask = torch.from_numpy(data["mode_id"] == 1)
        motion = torch.from_numpy(data["mv_qpel"].astype(np.float32)) / 16
        residual = torch.from_numpy(np.concatenate([
            data["residual_y"].reshape(16, 30, 52, 256),
            data["residual_u"].reshape(16, 30, 52, 64),
            data["residual_v"].reshape(16, 30, 52, 64),
        ], axis=-1))
    return mask, motion, residual


def load_latent(row, device):
    return torch.load(row["latent_path"], map_location="cpu", weights_only=True).float().unsqueeze(0).to(device)


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    config, dataset, index = load_config(args.config)
    device = "cuda"
    _, model = networks(args.mode, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    steps = [5, 10, 15, 20]
    keys = {split: [(sid, step) for sid in config["splits"][split] for step in steps] for split in ("train", "val")}
    latents = {key: load_latent(index[key], "cpu") for group in keys.values() for key in group}
    targets = {
        sid: read_target(dataset / index[sid, 20]["target_label_path"])
        for split in ("train", "val") for sid in config["splits"][split]
    }
    history = []
    best = float("inf")
    for epoch in range(args.epochs):
        random.shuffle(keys["train"])
        scores = {}
        for split, group in keys.items():
            training = split == "train"
            model.train(training)
            values = []
            with torch.set_grad_enabled(training):
                for sid, step in group:
                    mask, target_mv, target_res = targets[sid]
                    mask = mask.to(device)
                    if not mask.any():
                        continue
                    mv, res = model(latents[sid, step].to(device), torch.tensor([step/20], device=device))
                    loss_mv = F.smooth_l1_loss(mv[0][mask], target_mv.to(device)[mask], beta=.25)
                    loss_res = F.smooth_l1_loss(res[0][mask], target_res.to(device)[mask].float()/16, beta=.25)
                    loss = loss_mv + loss_res
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    values.append([loss_mv.item(), loss_res.item()])
            scores[split] = np.mean(values, axis=0).tolist()
        record = {"epoch": epoch+1, "train_mv_res_loss": scores["train"], "val_mv_res_loss": scores["val"]}
        print(json.dumps(record), flush=True)
        history.append(record)
        score = sum(scores["val"])
        if score < best:
            best = score
            torch.save(model.state_dict(), output / "regressor.pt")
    if not (output / "regressor.pt").exists():
        raise RuntimeError("No finite validation checkpoint")
    (output / "training.json").write_text(json.dumps({
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs, "seed": 42, "history": history,
        "normalization": 16, "loss": "P16-masked Smooth L1 for qpel MV and signed pixel residual",
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    train(parser.parse_args())
