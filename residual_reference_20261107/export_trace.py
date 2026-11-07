"""Generate closed-loop residual labels with frozen model modes and motion."""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"full_state_20261107"))
from benchmark import decode_check, load_backend
from train_state import load_config, load_latent, networks
from train_source import read_planes

HEADER=struct.Struct("<7H")


def parse_trace(path):
    mc = [np.zeros((17,480,832),np.uint8),np.zeros((17,240,416),np.uint8),np.zeros((17,240,416),np.uint8)]
    masks = [np.zeros_like(x,dtype=bool) for x in mc]
    with path.open("rb") as f:
        while chunk:=f.read(HEADER.size):
            t, mb, plane, x, y, w, h=HEADER.unpack(chunk)
            if not (1<=t<17 and mb<1560 and plane<3 and w>0 and h>0):
                raise ValueError("Invalid native MC record")
            size=16 if plane==0 else 8
            if x+w>size or y+h>size:
                raise ValueError("MC subblock out of bounds")
            row,col=divmod(mb,52)
            region=(t,slice(row*size+y,row*size+y+h),slice(col*size+x,col*size+x+w))
            if masks[plane][region].any():
                raise ValueError("Duplicate native MC samples")
            mc[plane][region]=np.frombuffer(f.read(w*h),np.uint8).reshape(h,w)
            masks[plane][region]=True
    return mc,masks


def main():
    parser=argparse.ArgumentParser()
    for key in ("config","kit","jm","data","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    config,_,index=load_config(args.config)
    backend=load_backend(args.kit)
    mode,reg=networks(ROOT/"expansion_20261106/results_expand32/stage_4.pt","cpu")
    reg.load_state_dict(torch.load(ROOT/"full_state_20261107/results/regressor.pt",map_location="cpu",weights_only=True))
    reg.eval()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for sid in config["splits"]["train"]+config["splits"]["val"]:
        folder=args.output/sid
        folder.mkdir()
        row=index[sid,20]
        latent=load_latent(row,"cpu")
        with torch.inference_mode():
            progress=torch.ones(1)
            modes=mode(latent,progress)[0].argmax(0).byte().numpy()
            mv=reg(latent,progress)[0][0].mul(16).round().numpy().astype("<i2")
        np.savez(folder/"proposal.npz",mode_id=modes,mv_qpel=mv,
                 residual_y=np.zeros((16,30,52,16,16),dtype=np.int16),
                 residual_u=np.zeros((16,30,52,8,8),dtype=np.int16),
                 residual_v=np.zeros((16,30,52,8,8),dtype=np.int16))
        state=folder/"proposal.bin"
        backend.run([sys.executable,args.kit/"scripts/prediction_to_b1_state.py",
                     "--prediction",folder/"proposal.npz","--oracle-template",row["target_oracle_state_path"],
                     "--output-state",state,"--result-json",folder/"adapter.json"],folder,folder/"adapter.log")
        raw=args.data/sid/"final_raw.yuv"
        bit=folder/"live.264"
        trace=folder/"mc.bin"
        settings=dict(InputFile=raw,OutputFile=bit,ReconFile=folder/"encoder_recon.yuv",SourceWidth=832,
                      SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,QPISlice=28,QPPSlice=28,
                      RDOptimization=0,NumberReferenceFrames=1,PList0References=1)
        command=[args.jm/"bin/lencod.exe","-d",args.kit/"JM-backend/cfg/encoder_baseline.cfg",
                 "-f",args.kit/"linux/skip_p16.cfg"]
        for key,value in settings.items():
            command+=["-p",f"{key} = {value}"]
        env=os.environ.copy()
        env.update(JM_STATE_MODE="replay",JM_STATE_FILE=str(state),JM_LIVE_RESIDUAL="1",JM_LIVE_MC_TRACE=str(trace))
        backend.run(command,folder,folder/"encode.log",env)
        decode_check(backend,folder,bit)
        mc,masks=parse_trace(trace)
        y,uv=read_planes(raw)
        planes=[y[0,:,0].numpy(),uv[0,:,0].numpy(),uv[0,:,1].numpy()]
        residuals=[np.where(mask,src.astype(np.int16)-pred.astype(np.int16),0)
                   for src,pred,mask in zip(planes,mc,masks)]
        count=int((modes==1).sum())
        if [int(m.sum()) for m in masks] != [count*256,count*64,count*64]:
            raise ValueError(f"{sid} MC coverage disagrees with P16 state")
        np.savez_compressed(folder/"aligned_labels.npz",mc_y=mc[0],mc_uv=np.stack(mc[1:],axis=1),
                            residual_y=residuals[0],residual_uv=np.stack(residuals[1:],axis=1),
                            mask_y=masks[0],mask_uv=np.stack(masks[1:],axis=1))
        result=dict(sample_id=sid,p16_blocks=count,decoder_agreement=True)
        results.append(result)
        print(json.dumps(result),flush=True)
    (args.output/"results.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
