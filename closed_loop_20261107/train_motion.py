"""Final-latent motion proposals from spatial correlation across latent frames."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_state_20261107"))
from train_state import load_config, load_latent, networks


class MotionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.radius = 4
        self.project = nn.Conv3d(113, 64, 1)
        self.refine = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.SiLU(),
        )
        self.output = nn.Conv3d(64, 2, 1)

    def forward(self, latent):
        cur, ref = latent[:, :, 1:], latent[:, :, :-1]
        query = F.normalize(cur, dim=1)
        r = self.radius
        padded = F.pad(F.normalize(ref, dim=1), (r, r, r, r, 0, 0), mode="replicate")
        h, w = cur.shape[-2:]
        costs = [
            (query * padded[..., y:y+h, x:x+w]).sum(1, keepdim=True)
            for y in range(2*r+1) for x in range(2*r+1)
        ]
        features = F.avg_pool3d(torch.cat([cur, cur-ref, *costs], 1), (1, 2, 2))
        features = F.silu(self.project(features))
        motion = self.output(features + self.refine(features))
        return F.interpolate(motion, (16, 30, 52), mode="trilinear", align_corners=False).movedim(1, -1)


def main():
    parser = argparse.ArgumentParser()
    for key in ("config", "mode", "regressor", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-per-prompt", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    config, dataset, index = load_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.allow_tf32 = True
    inputs, truth = {}, {}
    for ids in config["splits"].values():
        for sid in ids:
            row = index[sid, 20]
            inputs[sid] = load_latent(row, "cuda")
            with np.load(dataset / row["target_label_path"], allow_pickle=False) as target:
                truth[sid] = (
                    torch.from_numpy(target["mode_id"] == 1).cuda(),
                    torch.from_numpy(target["mv_qpel"].astype(np.float32)).cuda(),
                )
    all_train = config["splits"]["train"]
    train_ids = all_train
    if args.train_per_prompt is not None:
        if args.train_per_prompt < 1:
            raise ValueError("train-per-prompt must be positive")
        groups = {}
        for sid in all_train:
            groups.setdefault(index[sid, 20]["prompt"], []).append(sid)
        train_ids = [sid for group in groups.values() for sid in group[:args.train_per_prompt]]
    model = MotionHead().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = random.sample(train_ids, len(train_ids))
        while len(order) < len(all_train):
            order.extend(random.sample(train_ids, len(train_ids)))
        order = order[:len(all_train)]
        for sid in order:
            mask, mv = truth[sid]
            if not mask.any():
                continue
            opt.zero_grad(set_to_none=True)
            pred = model(inputs[sid])[0]
            loss = F.smooth_l1_loss(pred[mask], mv[mask]/16, beta=0.25)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            errors = []
            for sid in config["splits"]["val"]:
                mask, mv = truth[sid]
                if mask.any():
                    diff = model(inputs[sid])[0][mask]*16-mv[mask]
                    errors.append(float(torch.linalg.vector_norm(diff, dim=-1).mean()))
        score = float(np.mean(errors))
        if not np.isfinite(score):
            raise ValueError("No finite validation EPE")
        if score < best:
            best = score
            torch.save(model.state_dict(), out/"motion.pt")
        history.append(score)
        print(json.dumps(dict(epoch=epoch+1,val_epe_qpel=score)), flush=True)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    model.load_state_dict(torch.load(out/"motion.pt", weights_only=True))
    _, previous = networks(args.mode, "cuda")
    previous.load_state_dict(torch.load(args.regressor, weights_only=True))
    previous.eval()
    rows = []
    with torch.inference_mode():
        for sid in config["splits"]["test"]:
            mask, target = truth[sid]
            old, _ = previous(inputs[sid], torch.ones(1, device="cuda"))
            new = model(inputs[sid])
            for name, value in (("prior", old), ("correlation", new)):
                qpel = (value[0]*16).round()
                if not torch.isfinite(qpel).all():
                    raise ValueError("Nonfinite test proposal")
                arr = qpel.clamp(-32768,32767).cpu().numpy().astype("<i2")
                arr.tofile(out/f"{sid}_{name}.mv")
                diff = qpel[mask]-target[mask]
                rows.append(dict(sample_id=sid,head=name,epe_qpel=float(torch.linalg.vector_norm(diff,dim=-1).mean()),
                                 exact=float((diff==0).all(-1).float().mean())))
    (out/"motion_metrics.json").write_text(json.dumps(dict(
        parameters=sum(p.numel() for p in model.parameters()),epochs=args.epochs,
        best_val_epe_qpel=best,validation_history=history,test=rows,
        train_videos=len(train_ids), steps_per_epoch=len(all_train), peak_allocated_gib=peak_gib),indent=2))


if __name__ == "__main__":
    main()
