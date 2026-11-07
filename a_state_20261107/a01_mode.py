"""A0.1 dedicated final-step mode checkpoint: accuracy, recalls and reliability."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate import calibration
from state import latent, records
from calibrate import fit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"expansion_20261106"))
from train_mode import Head, metric


@torch.inference_mode()
def read(model,row):
    z=latent(row["final_latent_path"])
    probabilities=model(z,z.new_tensor([1.]))[0].softmax(0)
    c,p=probabilities.max(0)
    with np.load(row["root"]/row["current_label_path"],allow_pickle=False) as d:
        y=torch.from_numpy(d["mode_id"].astype("i8"))
    return c.cpu(),p.cpu(),y


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    model=Head().cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,weights_only=True))
    rows=records(a.config)
    val=[read(model,r) for r in rows["val"] if r["step"]==20]
    slope,offset=fit(np.concatenate([c.numpy().ravel() for c,_,_ in val]),
                     np.concatenate([(p==y).numpy().ravel() for _,p,y in val]))
    cases=[]
    for r in rows["test"]:
        if r["step"]!=20: continue
        c,p,y=read(model,r)
        cases.append((r["sample_id"],c.flatten(),p.flatten(),y.flatten()))
    all_cases=[("all_pooled",*(torch.cat([r[i] for r in cases]) for i in range(1,4))),*cases]
    result={}
    for key,c,p,y in all_cases:
        c=c.double().clamp(1e-7,1-1e-7)
        result[key]=dict(**metric(p,y),raw_confidence=calibration(c,p==y),
                         val_affine_confidence=calibration((slope*c.logit()+offset).sigmoid(),p==y))
    a.output.write_text(json.dumps(dict(val_fit=[slope,offset],test=result),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
