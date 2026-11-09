"""Validate every prespecified main-budget state through native JM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from train_capacity import ON,p,build
from o1 import infer,level_bounds,score,data_for
from repair import to_pixels


@torch.inference_mode()
def one(root,run):
    config=json.loads((run/"run.json").read_text())
    net=build(config["width"]).eval()
    net.load_state_dict(torch.load(run/"head-3200.pt",weights_only=True))
    rows=p.rows_by_split()["val"]
    out=root/("replay-"+run.name)
    out.mkdir(exist_ok=False)
    bounds=level_bounds("cuda")
    jobs=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in rows:
            x,mask,_=data_for(ON/"o0",row,"T0")
            q,_,_,diag=infer(net,x,mask,bounds)
            jobs.append(pool.submit(score,out,"val",row,q,diag))
        pts=[j.result() for j in jobs]
    row=rows[0]
    frozen=p.load_model(p.CKPT).eval().requires_grad_(False)
    h=p.input_tensor(row,"hidden")
    base,_=p.predict(frozen,h,20)
    levels,*_=infer(net,p.feature(frozen,h,base),torch.from_numpy(base["mode_id"]==1).cuda(),bounds)
    base["residual_y"]=to_pixels(levels)
    scored=p.p05.load_npz(out/row["sample_id"]/"prediction.npz")
    checks={k:bool(np.array_equal(scored[k],base[k])) for k in ("mode_id","mv_qpel","residual_y","residual_u","residual_v")}
    assert all(checks.values())
    p.save_json(out/"run.json",dict(name=run.name,head_sha256=p.digest(run/"head-3200.pt"),points=pts,
                                   summary=p.summarize(pts),export_smoke=checks))
    print(run.name,p.summarize(pts),flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("root",type=Path)
    a.add_argument("--seed",type=int,required=True)
    args=a.parse_args()
    torch.set_num_threads(2)
    runs=[*args.root.glob(f"c0-*-{args.seed}"),*args.root.glob(f"c1-rank-*-{args.seed}")]
    for run in sorted(runs):
        one(args.root,run)


if __name__=="__main__":
    main()
