"""Select the smallest rank that preserves full-video validation source fidelity."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from benchmark_source import FrozenPrefix, RankProjection, install, load_config, load_input, psnr, rows_for


def main():
    p=argparse.ArgumentParser()
    for key in ("config","data","checkpoint","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    cfg,_,index=load_config(a.config)
    rows=rows_for(cfg,index,a.data)["val"]
    saved=torch.load(a.checkpoint,weights_only=True)
    trained=RankProjection(saved["channels"],saved["rank"]).cuda().eval()
    trained.load_state_dict(saved["weights"])
    prefix=FrozenPrefix()
    original=prefix.vae.model.decoder.head[-1]
    source_weight=trained.out_proj.weight.detach().transpose(0,1).reshape(saved["rank"],-1)
    matrix=trained.in_proj.weight.detach().reshape(saved["rank"],-1).T @ source_weight
    svd=torch.linalg.svd(matrix,full_matrices=False)
    choices=[]
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp, torch.inference_mode():
        proxy=Path(tmp)/"proxy.yuv"
        for rank in [x for x in (16,24,32,40) if x<saved["rank"]]+[saved["rank"]]:
            if rank==saved["rank"]:
                model=trained
            else:
                model=RankProjection(saved["channels"],rank).cuda().eval()
                model.initialize(original,svd)
                model.out_proj.bias.copy_(trained.out_proj.bias)
            install(prefix.vae,model)
            scores=[]
            for row in rows:
                latent=load_input(row,"cuda")
                video=prefix.vae.decode([latent[0]])[0]
                rgb=((video+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy()
                subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","832x480",
                                "-r","16","-i","pipe:0","-vf","scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                                "-pix_fmt","yuv420p","-f","rawvideo",str(proxy)],input=rgb.tobytes(),check=True)
                score=psnr(row["raw_path"],proxy)["psnr_y_p_db"]
                scores.append(100 if score is None else score)
            choices.append(dict(rank=rank,validation_y_psnr=scores))
            print(json.dumps(choices[-1]),flush=True)
            if min(scores)>=52 or rank==saved["rank"]:
                break
    a.output.mkdir(exist_ok=False)
    torch.save(dict(channels=saved["channels"],rank=rank,weights=model.state_dict()),a.output/"selected.pt")
    (a.output/"calibration.json").write_text(json.dumps(dict(
        choices=choices,selected_rank=rank,threshold_min_y_psnr=52,
        projection_parameters=sum(x.numel() for x in model.parameters()),
        original_parameters=sum(x.numel() for x in original.parameters())),indent=2))


if __name__=="__main__":
    main()
