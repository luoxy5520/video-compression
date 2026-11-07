"""Trace only train/val; require fixed-state replay to reproduce the teacher stream."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from evaluate import load_backend,replay
from state import PLANES,records

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),*[str(ROOT/p) for p in ("learned_qpel_20261107","latent_codec_20261107")]]
from latent_codec_20261107.run import load_models,prepare_one
from qpel_head import model,predict

DT=np.dtype([("frame","<i2"),("addr","<i2"),("mode","<i2"),("mv","<i2",(2,)),("r","<i2",(384,))])


def decode_trace(path):
    t=np.fromfile(path,dtype=DT)
    assert len(t)==24960 and np.array_equal(t["frame"],np.repeat(np.arange(1,17),1560))
    assert np.array_equal(t["addr"],np.tile(np.arange(1560),16))
    m=t["mode"]
    assert np.isin(m,(0,1)).all()
    arrays=dict(mode_id=m.reshape(16,30,52).astype("u1"),mv_qpel=t["mv"].reshape(16,30,52,2))
    for p,k,l,u in PLANES:
        arrays[f"residual_{p}"]=t["r"][:,l:u].reshape(16,30,52,k,k).copy()
    return arrays


def run(a,backend,folder,row):
    mv=np.fromfile(folder/"shortlist.bin","<i2").reshape(16,30,52,4,2)
    inp=folder/"input.npz"
    zeros={f"residual_{p}":np.zeros((16,30,52,k,k),"i2") for p,k,*_ in PLANES}
    np.savez(inp,mode_id=np.ones((16,30,52),"u1"),mv_qpel=mv[...,0,:],**zeros)
    state=folder/"seed.bin"
    backend.run([sys.executable,a.kit/"scripts/prediction_to_b1_state.py","--prediction",inp,
                 "--oracle-template",row["current_oracle_state_path"],"--output-state",state,
                 "--result-json",folder/"seed_adapter.json"],folder,folder/"seed.log")
    env={k:v for k,v in os.environ.items() if not k.startswith("JM_")}
    env.update(JM_STATE_MODE="replay",JM_STATE_FILE=str(state),JM_REPLAY_SYNC_MV="1",
        JM_ALIGNED_PROXY=str(folder/"proxy.yuv"),JM_LATENT_SHORTLIST=str(folder/"shortlist.bin"),
        JM_LATENT_REFINE="1",JM_REPLAY_EXACT_SKIP="1",JM_QPEL_PRIOR=str(folder/"qpel_prior.bin"),
        JM_QPEL_BUDGET="4",JM_A_TEACHER_TRACE=str(folder/"teacher.bin"))
    bit=folder/"teacher.264"
    cmd=[a.teacher_jm/"bin/lencod.exe","-d",a.kit/"JM-backend/cfg/encoder_baseline.cfg",
         "-f",a.kit/"linux/skip_p16.cfg"]
    for key,value in dict(InputFile=folder/"proxy.yuv",OutputFile=bit,ReconFile=folder/"teacher_recon.yuv",
        SourceWidth=832,SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,QPISlice=28,QPPSlice=28,
        RDOptimization=0,NumberReferenceFrames=1,PList0References=1).items():
        cmd+=["-p",f"{key} = {value}"]
    backend.run(cmd,folder,folder/"teacher.log",env)
    arrays=decode_trace(folder/"teacher.bin")
    out=folder/"verify"
    out.mkdir()
    ref=Path(row["source_export_dir"])/"input_yuv420p.yuv"
    result=replay(backend,a.jm,out,row,arrays,ref,{})
    if backend.sha(bit)!=backend.sha(out/"prediction.264") or backend.sha(folder/"teacher_recon.yuv")!=backend.sha(out/"encoder_recon.yuv"):
        raise ValueError(f"Teacher trace failed byte-exact replay: {folder}")
    np.savez_compressed(a.output/"labels"/f"{row['sample_id']}.npz",**arrays)
    print(row["sample_id"],result["psnr_y_p_db"],result["bitrate_kbps"],flush=True)
    return dict(sample_id=row["sample_id"],bitstream_exact=True,**result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    p.add_argument("--limit",type=int)
    for k in ("teacher-jm","jm","kit","output","projection","decision","qpel"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"labels").mkdir()
    idx=records(a.config)
    rows=[r for split in ("train","val") for r in idx[split] if r["step"]==20]
    if a.limit:
        rows=rows[:a.limit]
    backend=load_backend(a.kit)
    vae,head=load_models(a)
    qpel=model().cuda().eval()
    saved=torch.load(a.qpel,weights_only=True)
    qpel.load_state_dict(saved.get("weights",saved))
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[]
        for row in rows:
            folder=a.output/row["sample_id"]
            prepare_one(row["final_latent_path"],folder,vae,head,keep_features=True)
            predict(qpel,folder)
            jobs.append(pool.submit(run,a,backend,folder,row))
        results=[j.result() for j in jobs]
    (a.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
