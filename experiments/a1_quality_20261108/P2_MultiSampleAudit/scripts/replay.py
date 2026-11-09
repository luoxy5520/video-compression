"""Run full validation replays for preselected training-only milestones."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from m1 import ON,p
from o1 import data_for,infer,score,level_bounds
from repair import to_pixels


def read(path):
    return json.loads(path.read_text())


def select(root):
    gate=read(root/"m2_gate.json")
    n=gate["first_persistent_collapse"]
    targets={f"scale-44-{s}":3520 for s in (1701,1702)}
    chosen=None
    if n:
        policy_scores={}
        for policy in ("bce","mixed","candidate_weighted","blocked"):
            policy_scores[policy]=mean(mean(r["recall"] for r in read(root/f"m2-{policy}-{n}-{s}/run.json")["snapshots"]["400"]["train"]) for s in (1701,1702))
        chosen=max(policy_scores,key=policy_scores.get)
        for seed in (1701,1702):
            targets[f"scale-{n}-{seed}"]=400
            targets[f"m2-{chosen}-{n}-{seed}"]=400
    return dict(selected_m2_policy=chosen,targets=targets)


@torch.inference_mode()
def run(root,out,name,step):
    model=p.SparseHead(67,16).cuda().eval()
    model.load_state_dict(torch.load(root/name/f"head-{step}.pt",weights_only=True))
    maxima=level_bounds("cuda")
    out.mkdir(exist_ok=False)
    val=p.rows_by_split()["val"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[]
        for row in val:
            x,mask,_=data_for(ON/"o0",row,"T0")
            q,_,_,diag=infer(model,x,mask,maxima)
            jobs.append(pool.submit(score,out,"val",row,q,diag))
        results=[j.result() for j in jobs]
    row=val[0]
    base_model=p.load_model(p.CKPT).eval().requires_grad_(False)
    h=p.input_tensor(row,"hidden")
    base,_=p.predict(base_model,h,20)
    q,*_=infer(model,p.feature(base_model,h,base),torch.from_numpy(base["mode_id"]==1).cuda(),maxima)
    state=dict(base,residual_y=to_pixels(q))
    expected=p.p05.load_npz(out/row["sample_id"]/"prediction.npz")
    checks={k:bool(np.array_equal(state[k],expected[k])) for k in ("mode_id","mv_qpel","residual_y","residual_u","residual_v")}
    assert all(checks.values())
    p.save_json(out/"run.json",dict(name=name,step=step,points=results,export_fields=checks,summary=p.summarize(results)))
    print(name,step,p.summarize(results),flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("--seed",type=int)
    a=cli.parse_args()
    torch.set_num_threads(2)
    selection=select(a.root)
    if a.seed is None:
        p.save_json(a.root/"replay_selection.json",selection)
    else:
        for name,step in selection["targets"].items():
            if name.endswith(f"-{a.seed}"):
                run(a.root,a.root/f"replay-{name}-{step}",name,step)


if __name__=="__main__":
    main()
