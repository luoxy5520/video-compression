"""Diagnostic only: pick nearest to Oracle MV from four model hypotheses, then replay."""
import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from coarse_fine_mv import Motion,label,infer
from evaluate import load_backend
from rollout_residual import replay_item
from state import PLANES,input_tensor,load_model,records


@torch.inference_mode()
def pick(base,h,row):
    pred,_=infer(base,h,row)
    z=input_tensor(row,base.kind)
    feat=base.encode_features(z,z.new_tensor([1.]))
    x=h(feat)
    ids=x[...,0].topk(min(4,len(h.anchors)),dim=-1).indices
    selected=x.gather(-2,ids[...,None].expand(*ids.shape,19))
    integer=h.anchors[ids]+selected[...,1:3].mul(4).round()
    phase=selected[...,3:].argmax(-1)
    candidates=(integer*4+torch.stack((phase%4,phase//4),-1)).clamp(-2048,2047)
    mask,truth=(v.cuda() for v in label(row))
    idx=(candidates-truth[...,None,:]).square().sum(-1).argmin(-1)
    use=mask & torch.as_tensor(pred["mode_id"]==1,device=mask.device)
    idx=torch.where(use,idx,torch.zeros_like(idx))
    v=candidates.gather(-2,idx[...,None,None].expand(*idx.shape,1,2)).squeeze(-2).short()
    r=base.residual(torch.cat((feat,v.movedim(-1,0)[None].float()/16),dim=1))[0].movedim(0,-1)
    r=r.mul(16).round().clamp(-255,255).short().cpu().numpy()
    pred["mv_qpel"]=v.cpu().numpy()
    for p,k,l,u in PLANES:
        pred["residual_"+p]=r[...,l:u].reshape(16,30,52,k,k)
    return pred


def main():
    p=argparse.ArgumentParser()
    for key in ("config","checkpoint","head","kit","jm","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    a.output.mkdir(parents=True)
    base=load_model(a.checkpoint)
    s=torch.load(a.head,map_location="cpu",weights_only=True)
    h=Motion(base.mode.in_channels,s["anchors"]).cuda().eval()
    h.load_state_dict(s["weights"])
    backend=load_backend(a.kit)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[]
        for row in records([a.config])["test"]:
            if row["step"]!=20: continue
            pred=pick(base,h,row)
            jobs.append(pool.submit(replay_item,a,backend,a.output/row["sample_id"],row,pred,False))
        rows=[j.result() for j in jobs]
    assert len(rows)==12
    result=dict(diagnostic="Oracle nearest top4 on Oracle/predicted P16 intersection; not deployable",
                per_video=rows,mean={k:statistics.mean(x[k] for x in rows)
                    for k in ("psnr_y_p_db","last_frame_psnr_y","bitrate_kbps")})
    (a.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(result["mean"])


if __name__=="__main__":
    main()
