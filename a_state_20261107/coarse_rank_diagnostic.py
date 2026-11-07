"""Distinguish missing MV candidates from a wrong top-1 decision."""
import argparse
import json
from pathlib import Path

import torch

from coarse_fine_mv import Motion,label
from state import input_tensor,load_model,records


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    for k in ("checkpoint","head","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    base=load_model(a.checkpoint)
    saved=torch.load(a.head,weights_only=True,map_location="cpu")
    h=Motion(base.mode.in_channels,saved["anchors"]).cuda().eval()
    h.load_state_dict(saved["weights"])
    out=[]
    for row in records(a.config)["test"]:
        if row["step"]!=20: continue
        z=input_tensor(row,base.kind)
        x=h(base.encode_features(z,z.new_tensor([1.])))
        m,v=(k.cuda() for k in label(row))
        if not m.any(): continue
        x=x[m]
        v=v[m]
        ids=x[...,0].topk(min(4,len(h.anchors)),dim=-1).indices
        selected=x.gather(-2,ids[...,None].expand(*ids.shape,19))
        integer=h.anchors[ids]+selected[...,1:3].mul(4).round()
        phase=selected[...,3:].argmax(-1)
        pred=(integer*4+torch.stack((phase%4,phase//4),-1)).clamp(-2048,2047)
        errors=(pred-v[:,None]).norm(dim=-1)
        moving=v.float().norm(dim=-1)>=4
        pairs={"all":errors,"moving":errors[moving]}
        result=dict(sample_id=row["sample_id"],prompt=row["prompt"])
        for key,d in pairs.items():
            if len(d):
                result[key]=dict(top1_epe=float(d[:,0].mean()),best4_epe=float(d.min(-1).values.mean()),
                                 top1_exact=float((d[:,0]==0).float().mean()),
                                 top4_exact=float((d==0).any(-1).float().mean()))
        out.append(result)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
