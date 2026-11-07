"""Shared time-conditioned latent -> complete P state. No JM, VAE or pixels at inference."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STEPS = (5, 10, 15, 20)
PLANES = (("y", 16, 0, 256), ("u", 8, 256, 320), ("v", 8, 320, 384))


class StateHead(nn.Module):
    def __init__(self, width=64, kind="latent", recurrent=False):
        super().__init__()
        self.kind=kind
        self.recurrent=recurrent
        if recurrent:
            self.state_embed=nn.Conv2d(2+2+6,16,1)
            self.gate=nn.Conv2d(2*width+16,2*width,3,padding=1)
            self.update=nn.Conv2d(2*width+16,width,3,padding=1)
        if kind=="hidden":
            self.hidden_proj=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,64))
        self.stem = nn.Conv3d(64, width, 3, padding=1)
        self.step = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.body = nn.ModuleList(nn.Conv3d(width, width, 3, padding=1) for _ in range(3))
        self.temporal = nn.Conv3d(width, width * 4, 1)
        self.mode = nn.Conv3d(width, 2, 1)
        self.motion = nn.Conv3d(width, 2, 1)
        self.residual = nn.Sequential(nn.Conv3d(width+2, width, 1), nn.SiLU(), nn.Conv3d(width, 384, 1))
        self.confidence = nn.Conv3d(width, 2, 1)

    def encode_features(self, z, step):
        if self.kind=="hidden":
            b=z.shape[0]
            h,w=60,104
            x=self.hidden_proj(z).reshape(b,5,30,52,64).movedim(-1,1)
        else:
            b, c, t, h, w = z.shape
            x = z.reshape(b,c,t,h//2,2,w//2,2).permute(0,1,4,6,2,3,5).reshape(b,c*4,t,h//2,w//2)
        x = F.silu(self.stem(x) + self.step(step[:, None])[:, :, None, None, None])
        for conv in self.body:
            x = x + F.silu(conv(x))
        x = self.temporal(x[:, :, 1:]).reshape(b,4,-1,4,h//2,w//2)
        return x.permute(0,2,3,1,4,5).flatten(2,3)

    def output_layers(self,x):
        mv = self.motion(x)
        return (self.mode(x), mv.movedim(1,-1),
                self.residual(torch.cat((x,mv),dim=1)).movedim(1,-1),
                self.confidence(x).movedim(1,-1))

    def forward(self, z, step):
        x = self.encode_features(z,step)
        if not self.recurrent:
            return self.output_layers(x)
        b,c,_,h,w=x.shape
        memory=x.new_zeros((b,c,h,w))
        prior=x.new_zeros((b,16,h,w))
        out=[]
        for xt in x.unbind(dim=2):
            reset,carry=self.gate(torch.cat((xt,memory,prior),dim=1)).sigmoid().chunk(2,dim=1)
            update=self.update(torch.cat((xt,memory*reset,prior),dim=1)).tanh()
            memory=carry*memory+(1-carry)*update
            result=self.output_layers((xt+memory).unsqueeze(2))
            out.append(result)
            mode,mv,res,_=result
            components=[mode.squeeze(2).softmax(1),mv.squeeze(1).movedim(-1,1)]
            for p,k,l,u in PLANES:
                r=res[:,0,...,l:u]
                components.extend((r.mean(-1)[:,None],r.abs().mean(-1)[:,None]))
            prior=self.state_embed(torch.cat(components,dim=1)).tanh()
        return tuple(torch.cat([row[i] for row in out],dim=2 if i==0 else 1) for i in range(4))


def records(configs):
    result = {name: [] for name in ("train", "val", "test")}
    prompts = {}
    seen = set()
    for path in configs:
        cfg = json.loads(Path(path).read_text())
        root = Path(cfg["dataset"])
        all_rows = {}
        for line in (root/"manifests/all.jsonl").read_text().splitlines():
            r = json.loads(line)
            key = r["sample_id"], int(r["input_step"])
            if key in all_rows:
                raise ValueError(f"Duplicate manifest record {key}")
            all_rows[key] = r
        for split, ids in cfg["splits"].items():
            for sid in ids:
                for step in STEPS:
                    r = all_rows[sid,step]
                    if (sid,step) in seen or prompts.setdefault(r["prompt"],split) != split:
                        raise ValueError(f"Split overlap {sid}: {r['prompt']}")
                    seen.add((sid,step))
                    result[split].append(dict(r, step=step, root=root))
    return result


def latent(path, device="cuda"):
    z = torch.load(path, map_location="cpu", weights_only=True)
    if z.shape != (16,5,60,104):
        raise ValueError(f"Unexpected latent shape {z.shape}: {path}")
    return z[None].float().to(device)


def input_tensor(row, kind="latent", device="cuda"):
    if kind=="hidden":
        return hidden(row["hidden_path"],device)
    return latent(row["final_latent_path"] if row["step"]==20 else row["latent_path"],device)


def hidden(path,device="cuda"):
    z=torch.load(path,weights_only=True,map_location="cpu")
    if z.shape!=(1,7800,1536):
        raise ValueError(f"Unexpected hidden token shape {z.shape}: {path}")
    # Preserve on-disk precision in host cache; projection runs in fp32.
    return z if device=="cpu" else z.to(device,torch.float32)


def targets(row, teacher=None):
    path=teacher/f"{row['sample_id']}.npz" if teacher else row["root"]/row["current_label_path"]
    with np.load(path, allow_pickle=False) as d:
        m = torch.from_numpy(d["mode_id"].astype(np.int64))
        if not ((m==0)|(m==1)).all():
            raise ValueError("Non Skip/P16 mode")
        v = torch.from_numpy(d["mv_qpel"])
        r = torch.from_numpy(np.concatenate([d[f"residual_{p}"].reshape(*m.shape,k*k)
                                            for p,k,_,_ in PLANES], axis=-1))
    return m, v, r


def loss_fn(pred, m, v, r):
    mode, mv, res, conf = pred
    mode_loss = F.cross_entropy(mode, m[None], weight=mode.new_tensor([1.,2.]))
    keep = m==1
    if not keep.any():
        return mode_loss
    mv, res, conf = mv[0][keep],res[0][keep],conf[0][keep]
    v, r = v[keep].float(),r[keep].float()
    mv_loss = F.smooth_l1_loss(mv,v/16,beta=.25)
    res_loss = F.smooth_l1_loss(res,r/16,beta=.20)
    with torch.no_grad():
        rounded_mv = mv.mul(16).round()
        rounded_r = res.mul(16).round().clamp(-255,255)
        labels = torch.stack([(rounded_mv==v).all(-1), (rounded_r-r).square().mean(-1)<=16],-1).float()
    return mode_loss + 2*mv_loss + 2*res_loss + .20*F.binary_cross_entropy_with_logits(conf,labels)


def load_model(checkpoint):
    saved = torch.load(checkpoint,weights_only=True,map_location="cpu")
    head = StateHead(saved["width"],saved.get("kind","latent"),saved.get("recurrent",False)).cuda().eval()
    head.load_state_dict(saved["weights"])
    return head


@torch.inference_mode()
def predict(head, z, step):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    m,v,r,c = head(z,z.new_tensor([step/20]))
    pm = m[0].softmax(0).movedim(0,-1)
    mode = pm.argmax(-1)
    v = v[0].mul(16).round().clamp(-2048,2047)
    r = r[0].mul(16).round().clamp(-255,255)
    conf = torch.cat([pm.max(-1).values[...,None],c[0].sigmoid()],dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    if not all(torch.isfinite(x).all() for x in (pm,v,r,conf)):
        raise ValueError("Nonfinite prediction")
    res = r.short().cpu().numpy()
    arrays = dict(mode_id=mode.byte().cpu().numpy(),mv_qpel=v.short().cpu().numpy(),
                  confidence=conf.cpu().numpy(), p16_probability=pm[...,1].cpu().numpy())
    for p,k,l,u in PLANES:
        arrays["residual_"+p]=res[...,l:u].reshape(16,30,52,k,k)
    return arrays, dict(head_seconds=elapsed,head_peak_allocated_bytes=peak,head_peak_delta_bytes=peak-base)


def train(a):
    torch.manual_seed(4242)
    random.seed(4242)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    index = records(a.config)
    splits = {s:[r for r in index[s] if not a.final_only or r["step"]==20] for s in ("train","val")}
    assert all(splits.values())
    a.output.mkdir(parents=True, exist_ok=False)
    head = StateHead(a.width,a.kind,a.recurrent).cuda()
    opt = torch.optim.AdamW(head.parameters(),lr=a.lr)
    cache = {s: [(r, input_tensor(r,a.kind,"cpu"), targets(r,a.teacher_labels))
                 for r in rows] for s,rows in splits.items()}
    best = math.inf
    history = []
    updates=0
    for e in range(1,a.epochs+1):
        scores = {}
        random.shuffle(cache["train"])
        for split, rows in cache.items():
            training = split=="train"
            head.train(training)
            values = []
            with torch.set_grad_enabled(training):
                for row,z,truth in rows:
                    if training and a.max_updates is not None and updates>=a.max_updates:
                        break
                    z = z.to("cuda",torch.float32)
                    m,v,r = (t.cuda() for t in truth)
                    pred = head(z,z.new_tensor([row["step"]/20]))
                    loss = loss_fn(pred,m,v,r)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    if training:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(),5)
                        opt.step()
                        updates+=1
                    values.append(loss.item())
            scores[split] = float(np.mean(values))
        history.append(dict(epoch=e,updates=updates,**scores))
        print(json.dumps(history[-1]),flush=True)
        if scores["val"] < best:
            best = scores["val"]
            torch.save(dict(weights=head.state_dict(),width=a.width,epoch=e,final_only=a.final_only,kind=a.kind,recurrent=a.recurrent),a.output/"state.pt")
        (a.output/"training.json").write_text(json.dumps(dict(parameters=sum(p.numel() for p in head.parameters()),
            train_records=len(cache["train"]),val_records=len(cache["val"]),kind=a.kind,recurrent=a.recurrent,history=history),indent=2))
        if a.max_updates is not None and updates>=a.max_updates:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",nargs="+")
    group.add_argument("--latent",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--step",type=int,choices=STEPS,default=20)
    group.add_argument("--hidden",type=Path)
    p.add_argument("--kind",choices=("latent","hidden"),default="latent")
    p.add_argument("--teacher-labels",type=Path)
    p.add_argument("--final-only",action="store_true")
    p.add_argument("--recurrent",action="store_true")
    p.add_argument("--max-updates",type=int)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--lr",type=float,default=.001)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    if a.max_updates is not None and a.max_updates<=0:
        p.error("--max-updates must be positive")
    if a.config:
        if a.teacher_labels and not a.final_only:
            p.error("Teacher traces only supervise the final step")
        train(a)
    else:
        if not a.checkpoint or a.output.suffix!=".npz" or a.output.exists():
            p.error("Inference requires --checkpoint and a new .npz output")
        model=load_model(a.checkpoint)
        if (a.hidden is None)!=(model.kind=="latent"):
            p.error("Use the input flag that matches the checkpoint's feature type")
        arrays,timing=predict(model,hidden(a.hidden) if a.hidden else latent(a.latent),a.step)
        np.savez_compressed(a.output,**arrays)
        print(json.dumps(timing))


if __name__ == "__main__":
    main()
