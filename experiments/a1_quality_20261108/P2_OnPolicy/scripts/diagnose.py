"""Check label-free state export and evaluate readout on own decoded references."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from o0 import ARMS, decode, historical, labels, p
from o1 import diagnose, infer, level_bounds
from l0 import HIST
from repair import to_pixels


def load(path):
    model=p.SparseHead(67,16).cuda().eval()
    model.load_state_dict(torch.load(path,weights_only=True))
    return model


@torch.inference_mode()
def export(hidden,head,out):
    assert p.digest(p.CKPT)==p.CFG["checkpoint_sha256"]
    frozen=p.load_model(p.CKPT).eval().requires_grad_(False)
    h=p.input_tensor({"hidden_path":str(hidden)},"hidden")
    base,_=p.predict(frozen,h,20)
    mask=torch.from_numpy(base["mode_id"]==1).cuda()
    q,*_=infer(head,p.feature(frozen,h,base),mask,level_bounds("cuda"))
    base["residual_y"]=to_pixels(q)
    np.savez_compressed(out,**base)


@torch.inference_mode()
def examine(root,out):
    out.mkdir(exist_ok=False)
    rows=p.rows_by_split()["val"]
    selected={}
    for row in rows:
        selected.setdefault(row["prompt"],row)
    smoke={}
    result=[]
    for seed in (1701,1702):
        for arm in ARMS:
            directory=root/f"o1-{arm}-{seed}"
            net=load(directory/"head.pt")
            for row in selected.values():
                sid=row["sample_id"]
                _,source,base,_,_=historical("val",row)
                feature=torch.load(HIST/"features"/f"{sid}.pt",weights_only=True).cuda()
                mask=torch.from_numpy(base["mode_id"]==1).cuda()
                q,h,logits,_=infer(net,feature,mask,level_bounds("cuda"))
                own=decode(directory/sid/"prediction.264")
                target=labels(own,p.p05.read_yuv(source)[0],base)
                truth=torch.from_numpy(target["sparse"]).cuda()[mask].long()
                stats=diagnose(net,h,logits,truth,q,mask,level_bounds("cuda"))
                energy=np.log1p(np.square(target["dense"][base["mode_id"]==1].astype(np.float32)).sum(-1))
                confidence=logits.max(-1).values.cpu().numpy()
                corr=float(np.corrcoef(confidence.ravel(),energy.ravel())[0,1])
                picked=q!=0
                current=target["dense"][picked].astype(np.float64)
                emitted=q[picked].astype(np.float64)
                result.append(dict(sample=sid,prompt=row["prompt"],arm=arm,seed=seed,
                    block_energy_logit_correlation=corr,dense_nonzero_at_predicted_support=float((current!=0).mean()),
                    actual_signed_level_mae=float(np.abs(emitted-current).mean()),**stats))
            row=rows[0]
            sid=row["sample_id"]
            destination=out/f"export-{arm}-{seed}.npz"
            export(row["hidden_path"],net,destination)
            candidate=p.p05.load_npz(destination)
            scored=p.p05.load_npz(directory/sid/"prediction.npz")
            equal={key:bool(np.array_equal(candidate[key],scored[key])) for key in
                   ("mode_id","mv_qpel","residual_y","residual_u","residual_v")}
            assert all(equal.values()),equal
            smoke[f"{arm}-{seed}"]=dict(sample=sid,state_sha256=p.digest(destination),fields=equal)
    p.save_json(out/"own_reference_readout.json",result)
    p.save_json(out/"export_verification.json",smoke)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run",type=Path)
    cli.add_argument("out",type=Path)
    a=cli.parse_args()
    torch.set_num_threads(2)
    examine(a.run,a.out)


if __name__=="__main__":
    main()
