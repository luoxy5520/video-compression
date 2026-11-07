"""Distill subpixel directions from the exact-Skip closed-loop teacher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from qpel_head import model, predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"latent_codec_20261107"))
from run import BLOCKS, load_backend, load_config, load_models, prepare_one, run_case, summary


def trace_labels(path):
    data = np.fromfile(path,dtype="<u2").reshape(-1,4)
    ids = (data[:,0].astype(np.int64)-1)*1560+data[:,1]
    order = ids.argsort()
    if not np.array_equal(ids[order],np.arange(BLOCKS)) or (data[:,2:]>8).any():
        raise ValueError(f"Invalid or incomplete subpixel trace: {path}")
    return data[order,2:].astype(np.int64)


def fit(a, splits):
    torch.manual_seed(42)
    network = model().cuda()
    optimizer = torch.optim.AdamW(network.parameters(),lr=.002)
    cache = {}
    for split in ("train","val"):
        folders = [a.output/"samples"/sid for sid in splits[split]]
        x = np.concatenate([np.load(p/"features.npy",allow_pickle=False) for p in folders])
        y = np.concatenate([trace_labels(p/"teacher/teacher.bin") for p in folders])
        cache[split] = torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda()
    x,y = cache["train"]
    vx,vy = cache["val"]
    best = float("inf")
    history = []
    weights = torch.tensor([1.]*8+[.4],device="cuda")
    a.weights.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(20):
        network.train()
        for idx in torch.randperm(len(x),device="cuda").split(4096):
            logits = network(x[idx]).reshape(-1,9)
            loss = F.cross_entropy(logits,y[idx].reshape(-1),weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            logits = network(vx).reshape(-1,2,9)
            loss = float(F.cross_entropy(logits.reshape(-1,9),vy.reshape(-1),weight=weights))
            directional = vy<8
            top4 = (logits[:,:,:8].topk(4).indices==vy[...,None]).any(-1)
            stat = dict(epoch=epoch+1,val_loss=loss,val_top4_direction_recall=float(top4[directional].float().mean()))
        if loss<best:
            best = loss
            torch.save(network.state_dict(),a.weights)
        history.append(stat)
    (a.output/"training.json").write_text(json.dumps(history,indent=2))
    network.load_state_dict(torch.load(a.weights,weights_only=True))
    return network


def main(a):
    config,_,_ = load_config(a.config)
    splits = config["splits"]
    backend = load_backend(a.kit)
    a.output.mkdir(parents=True,exist_ok=False)
    vae, head = load_models(a)
    warm = a.output/"warm"
    prepare_one(a.data/splits["val"][0]/"final_latent.pt",warm,vae,head)
    for split, sids in splits.items():
        for sid in sids:
            folder = a.output/"samples"/sid
            times = prepare_one(a.data/sid/"final_latent.pt",folder,vae,head,keep_features=True)
            (folder/"prepared.json").write_text(json.dumps(times,indent=2))
            print(json.dumps(dict(sample_id=sid,split=split,**times)),flush=True)
    del vae,head

    def teacher(item):
        split,sid = item
        folder = a.output/"samples"/sid
        return run_case(a,backend,(split,sid,"refine"),name="teacher",
                        extra_env={"JM_REPLAY_EXACT_SKIP":"1",
                                   "JM_QPEL_TRACE":str(folder/"teacher/teacher.bin")})

    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        teachers = list(executor.map(teacher,[(s,sid) for s in ("train","val","test") for sid in splits[s]]))
    network = fit(a,splits)
    timing = {sid:predict(network,a.output/"samples"/sid) for s in ("val","test") for sid in splits[s]}

    def evaluate(item):
        split,sid,budget = item
        folder = a.output/"samples"/sid
        name = f"top{budget}"
        result = run_case(a,backend,(split,sid,"refine"),name=name,
                          extra_env={"JM_REPLAY_EXACT_SKIP":"1","JM_QPEL_BUDGET":str(budget),
                                     "JM_QPEL_PRIOR":str(folder/"qpel_prior.bin")})
        result["head_s"] = timing[sid]
        result["latent_to_bitstream_s"] += timing[sid]
        (folder/name/"metrics.json").write_text(json.dumps(result,indent=2))
        return result

    val_teacher = [r for r in teachers if r["split"]=="val"]
    exact = {r["sample_id"]:r for r in val_teacher}
    baseline = summary(val_teacher)["teacher"]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        validation = list(executor.map(evaluate,[("val",sid,k) for k in (2,4,6) for sid in splits["val"]]))
    eligible = []
    for name, row in summary(validation).items():
        worst = max(exact[r["sample_id"]]["psnr_y_p_db"]-r["psnr_y_p_db"]
                    for r in validation if r["variant"]==name)
        if (row["psnr_y_p_db"]>=baseline["psnr_y_p_db"]-.10
            and row["bitrate_kbps"]<=baseline["bitrate_kbps"]*1.02 and worst<=.35):
            eligible.append(int(name[3:]))
    selected = min(eligible) if eligible else None
    calibration = dict(selected=selected,eligible=eligible,mean_drop=.10,worst_drop=.35,rate_ratio=1.02)
    (a.output/"calibration.json").write_text(json.dumps(calibration,indent=2))
    test = []
    if selected:
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            test = list(executor.map(evaluate,[("test",sid,selected) for sid in splits["test"]]))
    records = teachers+validation+test
    (a.output/"results.json").write_text(json.dumps(dict(calibration=calibration,per_video=records,
         means={s:summary([r for r in records if r["split"]==s]) for s in splits}),indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    for key in ("config","projection","decision","kit","jm","data","output","weights"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    main(parser.parse_args())
