"""A1 stress test: 33 generated frames, a single Oracle I and one P reference chain."""
import argparse
import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from evaluate import load_backend,replay,video_details
from state import load_model,predict

PROMPTS=(
  "A small red delivery van drives from left to right behind a wire fence; a fixed camera films it without cuts.",
  "A fixed camera looks across a crowded ice rink as two skaters cross in opposite directions without cuts.",
)


def generate(root,weights):
    wan=Path.home()/"Wan2.1"
    sys.path[:0]=[str(wan),str(wan/"experiments/baseline17")]
    from run_remote_baseline17 import sdpa
    import wan.modules.model as wm
    from wan.text2video import WanT2V,FlowUniPCMultistepScheduler
    from wan.configs import WAN_CONFIGS
    wm.flash_attention=sdpa
    pipe=WanT2V(WAN_CONFIGS["t2v-1.3B"],str(weights),t5_cpu=False)
    call=FlowUniPCMultistepScheduler.step
    for i,prompt in enumerate(PROMPTS):
        folder=root/f"stress_{i}"
        folder.mkdir()
        state={}
        def capture(self,*args,**kwargs):
            r=call(self,*args,**kwargs)
            state["last"]=r[0].detach()
            return r
        FlowUniPCMultistepScheduler.step=capture
        try:
            video=pipe.generate(prompt,size=(832,480),frame_num=33,shift=8,
                                sample_solver="unipc",sampling_steps=20,guide_scale=6,seed=819+i,offload_model=True)
        finally:
            FlowUniPCMultistepScheduler.step=call
        z=state["last"].squeeze(0).cpu()
        assert z.shape==(16,9,60,104),z.shape
        torch.save(z,folder/"latent.pt")
        rgb=((video.float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy()
        subprocess.run(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","832x480","-r","16",
                        "-i","pipe:0","-vf","scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                        "-pix_fmt","yuv420p","-f","rawvideo",str(folder/"source.yuv")],input=rgb.tobytes(),check=True)
        print("GENERATED",folder.name,flush=True)


def native(a,backend,folder):
    bit,state,recon=map(lambda s:folder/s,("reference.264","oracle.bin","reference.yuv"))
    env={k:v for k,v in os.environ.items() if not k.startswith("JM_")}
    env.update(JM_STATE_MODE="dump",JM_STATE_FILE=str(state))
    cmd=[backend.ENC,"-d",backend.ROOT/"JM-backend/cfg/encoder_baseline.cfg","-f",backend.ROOT/"linux/skip_p16.cfg"]
    for k,v in dict(InputFile=folder/"source.yuv",OutputFile=bit,ReconFile=recon,SourceWidth=832,
        SourceHeight=480,FramesToBeEncoded=33,FrameRate=16,QPISlice=28,QPPSlice=28,
        RDOptimization=0,NumberReferenceFrames=1,PList0References=1,IntraPeriod=0).items():
        cmd+=["-p",f"{k} = {v}"]
    backend.run(cmd,folder,folder/"oracle.log",env)
    print("ORACLE",folder.name,flush=True)
    return state,bit,recon


def main():
    p=argparse.ArgumentParser()
    for k in ("checkpoint","output","kit","jm","weights"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    generate(a.output,a.weights)
    head=load_model(a.checkpoint)
    backend=load_backend(a.kit)
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        folders=sorted(a.output.glob("stress_*"))
        jobs=[pool.submit(native,a,backend,f) for f in folders]
        for i,(folder,job) in enumerate(zip(folders,jobs)):
            z=torch.load(folder/"latent.pt",weights_only=True).float().cuda()
            out=[predict(head,z[:,offset:offset+5].unsqueeze(0).contiguous(),20) for offset in (0,4)]
            arrays={k:np.concatenate([x[0][k] for x in out]) for k in out[0][0]}
            state,bit,recon=job.result()
            sub=folder/"direct"
            sub.mkdir()
            ref=folder/"source.yuv"
            record=replay(backend,a.jm,sub,{"current_oracle_state_path":state},arrays,ref,
                          dict(head_seconds=sum(x[1]["head_seconds"] for x in out)))
            oracle=video_details(backend,folder,ref,recon,"oracle",33)
            oracle["bitrate_kbps"]=bit.stat().st_size*128/33000
            result=dict(sample_id=folder.name,prompt=PROMPTS[i],frames=33,oracle=oracle,direct=record)
            results.append(result)
            print(folder.name,record["psnr_y_p_pooled"],record["last_frame_psnr_y"],flush=True)
    (a.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
