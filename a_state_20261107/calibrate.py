"""Fit confidence temperatures only on val, report per-video metrics on test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import calibration
from state import PLANES, STEPS, input_tensor, load_model, predict, records, targets


def get(a, model, row):
    step=row["step"]
    if row["split"]=="val":
        pred,_=predict(model,input_tensor(row,model.kind),step)
    else:
        with np.load(a.predictions/"test"/row["sample_id"]/f"step_{step:03d}/prediction.npz",allow_pickle=False) as data:
            pred=dict(data)
    m,v,r=(x.numpy() for x in targets(row))
    keep=m==1
    pred_r=np.concatenate([pred[f"residual_{p}"].reshape(*m.shape,k*k) for p,k,*_ in PLANES],-1)
    conf=pred["confidence"]
    return [(conf[...,0].ravel(),(pred["mode_id"]==m).ravel()),
            (conf[keep,1],(pred["mv_qpel"][keep]==v[keep]).all(-1)),
            (conf[keep,2],np.mean((pred_r[keep].astype("f4")-r[keep])**2,axis=-1)<=16)]


def fit(p,y):
    # Fit an affine logit map to account for both temperature and class prevalence.
    p=torch.from_numpy(p).double().clamp(1e-7,1-1e-7).logit()
    y=torch.from_numpy(y).double()
    params=torch.tensor([1.,0.],dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([params],max_iter=30,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(p*params[0]+params[1],y)+.0001*((params[0]-1)**2+params[1]**2)
        loss.backward()
        return loss
    opt.step(closure)
    return params.detach().tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    p.add_argument("--test-config",nargs="+")
    for k in ("checkpoint","predictions","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    rows=records(a.config)
    test_rows=records(a.test_config)["test"] if a.test_config else rows["test"]
    model=load_model(a.checkpoint)
    result={}
    for step in STEPS:
        vals=[get(a,model,dict(row,split="val")) for row in rows["val"] if row["step"]==step]
        fitted={key:fit(np.concatenate([v[i][0] for v in vals]),np.concatenate([v[i][1] for v in vals]))
                for i,key in enumerate(("mode","mv_exact","residual_rmse_le4"))}
        tests={}
        for row in test_rows:
            if row["step"]!=step:
                continue
            tests[row["sample_id"]]={}
            for (key,(slope,offset)),(prob,hits) in zip(fitted.items(),get(a,model,dict(row,split="test"))):
                p=np.clip(prob.astype("f8"),1e-7,1-1e-7)
                calibrated=1/(1+np.exp(-np.clip(slope*np.log(p/(1-p))+offset,-700,700)))
                tests[row["sample_id"]][key]=dict(raw=calibration(prob,hits),calibrated=calibration(calibrated,hits))
        result[step]=dict(fitted_val=fitted,test=tests)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
