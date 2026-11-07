"""Run a learned source proxy through JM's real reference loop."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"full_state_20261107"))
from benchmark import FPS,FRAMES,decode_check,load_backend,psnr
from train_source import SourceHead, load_config, load_input, rows_for, write_raw
from train_tail import FrozenPrefix, TailHead
from train_rank import RankProjection, install


def main():
    p=argparse.ArgumentParser()
    for key in ("config","kit","jm","data","previous","output","checkpoint"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--type",choices=["direct","tail","rank"],default="direct")
    p.add_argument("--motion",type=Path)
    p.add_argument("--variant", action="append",
                   choices=("native8","fixed_model_mode_mv","native1","assisted1","assisted8"))
    a=p.parse_args()
    if a.variant and len(set(a.variant)) != len(a.variant):
        p.error("--variant may only be specified once per case")
    if any(v.startswith("assisted") for v in (a.variant or ())) and not a.motion:
        p.error("assisted variants require --motion")
    torch.set_num_threads(4)
    config,_,index=load_config(a.config)
    rows=rows_for(config,index,a.data)
    backend=load_backend(a.kit)
    checkpoint=torch.load(a.checkpoint,weights_only=True)
    if a.type=="rank":
        prefix=FrozenPrefix()
        model=RankProjection(checkpoint["channels"],checkpoint["rank"]).cuda().eval()
        model.load_state_dict(checkpoint["weights"])
        install(prefix.vae,model)
    elif a.type=="tail":
        model=TailHead(checkpoint["channels"]).cuda().eval()
        model.load_state_dict(checkpoint["weights"])
        prefix=FrozenPrefix()
    else:
        model=SourceHead().cuda().eval()
        model.load_state_dict(checkpoint)

    motion_head=None
    if a.motion:
        sys.path.insert(0,str(ROOT/"closed_loop_20261107"))
        from train_motion import MotionHead
        motion_head=MotionHead().cuda().eval()
        motion_head.load_state_dict(torch.load(a.motion,weights_only=True))

    def predict(latent):
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            if a.type=="direct":
                return model(latent)
            if a.type=="rank":
                video=prefix.vae.decode([latent[0]])[0]
                return ((video+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy()
            feature=prefix(latent)
            chunks=[model(x) for x in feature.split(2)]
            return tuple(torch.cat(x).unsqueeze(0) for x in zip(*chunks))

    predict(load_input(rows["val"][0],"cuda"))
    torch.cuda.synchronize()
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    variants=["native8","fixed_model_mode_mv"]
    if motion_head is not None:
        variants.extend(("native1","assisted1","assisted8"))
    variants=a.variant or variants
    for row in rows["test"]:
        sid=row["sample_id"]
        folder=a.output/sid
        folder.mkdir()
        proxy=folder/"proxy.yuv"
        start=time.perf_counter()
        latent=load_input(row,"cuda")
        planes=predict(latent)
        if a.type=="rank":
            subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
                            "-s","832x480","-r","16","-i","pipe:0",
                            "-vf","scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                            "-pix_fmt","yuv420p","-f","rawvideo",str(proxy)],input=planes.tobytes(),check=True)
        else:
            write_raw(proxy,planes)
        del planes
        head_s=time.perf_counter()-start
        proxy_quality=psnr(row["raw_path"],proxy)
        for variant in variants:
            case=folder/variant
            case.mkdir()
            bit=case/"prediction.264"
            settings=dict(InputFile=proxy,OutputFile=bit,ReconFile=case/"encoder_recon.yuv",
                        SourceWidth=832,SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,
                        QPISlice=28,QPPSlice=28,RDOptimization=0,NumberReferenceFrames=1,
                        PList0References=1,SearchMode=-1,SearchRange=1 if variant.endswith("1") else 8)
            command=[a.jm/"bin/lencod.exe","-d",a.kit/"JM-backend/cfg/encoder_baseline.cfg",
                     "-f",a.kit/"linux/skip_p16.cfg"]
            for key,value in settings.items():
                command+=["-p",f"{key} = {value}"]
            env=os.environ.copy()
            for key in ("JM_LIVE_RESIDUAL","JM_LIVE_MC_TRACE","JM_LATENT_MV_PRIOR","JM_LATENT_MV_EXTRA"):
                env.pop(key,None)
            env["JM_STATE_MODE"]="off"
            if variant=="fixed_model_mode_mv":
                env.update(JM_STATE_MODE="replay", JM_LIVE_RESIDUAL="1",
                           JM_STATE_FILE=str(a.previous/sid/"prediction_020/predicted.bin"))
            motion_s=0.0
            if variant.startswith("assisted"):
                start=time.perf_counter()
                with torch.inference_mode():
                    values=(motion_head(latent)[0]*16).round().clamp(-32768,32767).cpu().numpy()
                prior=case/"motion_prior.mv"
                values.astype("<i2").tofile(prior)
                env.update(JM_LATENT_MV_PRIOR=str(prior),JM_LATENT_MV_EXTRA="1")
                motion_s=time.perf_counter()-start
            start=time.perf_counter()
            backend.run(command,case,case/"encode.log",env)
            jm_s=time.perf_counter()-start
            accepted=0
            if variant.startswith("assisted"):
                import re
                match=re.search(r"Latent MV extra candidate accepted: (\d+)",(case/"encode.log").read_text())
                if not match:
                    raise ValueError("JM did not read learned MV candidates")
                accepted=int(match[1])
            decoded,_=decode_check(backend,case,bit)
            item=dict(sample_id=sid,variant=variant,proxy_y_psnr=proxy_quality["psnr_y_p_db"],
                      bitrate_kbps=bit.stat().st_size*8*FPS/FRAMES/1000,
                      head_s=head_s,motion_s=motion_s,jm_s=jm_s,latent_to_bitstream_s=head_s+motion_s+jm_s,
                      extra_candidates_accepted=accepted,
                      decoder_agreement=True,**psnr(row["raw_path"],decoded))
            results.append(item)
            print(json.dumps(item),flush=True)
    means={}
    for variant in variants:
        group=[r for r in results if r["variant"]==variant]
        means[variant]={key:float(np.mean([r[key] for r in group])) for key in
                        ("proxy_y_psnr","psnr_y_p_db","psnr_yuv_p_db","bitrate_kbps","head_s","jm_s","latent_to_bitstream_s")}
    (a.output/"results.json").write_text(json.dumps(dict(means=means,per_video=results),indent=2))


if __name__=="__main__":
    main()
