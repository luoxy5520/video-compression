"""Score current-state predictions and optionally replay the full P state."""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isolate_i import isolate
from state import PLANES, STEPS, input_tensor, load_model, predict, records, targets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"full_state_20261107"))
from benchmark import FRAME_BYTES, W, H, collect_quality, decode_check, load_backend


def calibration(prob, hits):
    if not len(hits):
        return None
    p = np.asarray(prob,dtype=np.float64)
    h = np.asarray(hits,dtype=np.float64)
    bins = np.minimum((p*10).astype(int),9)
    ece = 0.
    for b in range(10):
        use = bins == b
        if use.any():
            ece += use.mean()*abs(p[use].mean()-h[use].mean())
    return dict(mean=float(p.mean()),frequency=float(h.mean()),ece10=float(ece),
                brier=float(np.mean((p-h)**2)), count=int(len(h)))


def state_metrics(pred, truth):
    mode,mv,res = (x.numpy() for x in truth)
    m = pred["mode_id"]
    acc = m == mode
    conf = pred["confidence"]
    cm = np.bincount((2*mode+m).ravel(),minlength=4).reshape(2,2)
    denom = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*cm.diagonal(),denom,out=np.zeros(2),where=denom>0)
    masks = {"oracle_p16":mode==1,"oracle_and_predicted_p16":(mode==1)&(m==1)}
    result = dict(mode_accuracy=float(acc.mean()),macro_f1=float(f1.mean()),
                  recall=[float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None for i in range(2)],
                  mode_confusion=cm.tolist(), mode_calibration=calibration(conf[...,0].ravel(),acc.ravel()))
    pr = np.concatenate([pred[f"residual_{p}"].reshape(*m.shape,k*k) for p,k,_,_ in PLANES],axis=-1)
    for name, keep in masks.items():
        if not keep.any():
            result[name] = {"count":0}
            continue
        v = mv[keep].astype(np.int32)
        pv = pred["mv_qpel"][keep].astype(np.int32)
        delta = pv-v
        exact = (delta==0).all(-1)
        err = pr[keep].astype(np.float32)-res[keep]
        mag = np.linalg.norm(v,axis=-1)
        reg = []
        for lo,hi in [(0,1),(1,4),(4,16),(16,np.inf)]:
            group = (mag>=lo)&(mag<hi)
            reg.append(dict(low_qpel=lo, high_qpel=hi if np.isfinite(hi) else None,
                            count=int(group.sum()),exact=float(exact[group].mean()) if group.any() else None))
        result[name] = dict(count=int(keep.sum()),exact=float(exact.mean()),
            qpel_cross_bin=float((~exact).mean()),at_least_one_pixel=float((np.abs(delta)>=4).any(-1).mean()),
            mv_mae_xy=np.abs(delta).mean(0).tolist(),mv_bias_xy=delta.mean(0).tolist(),
            mv_epe=float(np.linalg.norm(delta,axis=-1).mean()),motion_bins=reg,
            residual_rmse={p:float(np.sqrt(np.mean(err[:,l:u]**2))) for p,k,l,u in PLANES},
            zero_residual_rmse={p:float(np.sqrt(np.mean(res[keep,l:u].astype(np.float64)**2))) for p,k,l,u in PLANES},
            mv_calibration=calibration(conf[keep,1],exact),
            residual_calibration=calibration(conf[keep,2],np.mean(err**2,axis=-1)<=16))
    return result


def video_details(backend, folder, ref, dec, prefix, frames_count=17):
    ref, dec = Path(ref),Path(dec)
    frames = np.fromfile(ref,dtype="u1").reshape(frames_count,FRAME_BYTES)[:,:W*H]
    got = np.fromfile(dec,dtype="u1").reshape(frames_count,FRAME_BYTES)[:,:W*H]
    mse = np.square(frames.astype("f4")-got).mean(1)
    db = lambda x: float(10*math.log10(255**2/x)) if x else None
    psnr = [db(x) for x in mse]
    log = folder/f"{prefix}_ssim.log"
    raw = ["-f","rawvideo","-pixel_format","yuv420p","-video_size",f"{W}x{H}","-framerate","16"]
    stats = f"{prefix}_ssim.txt"
    backend.run(["ffmpeg","-nostdin","-hide_banner",*raw,"-i",ref,*raw,"-i",dec,
                 "-filter_complex",f"[0:v][1:v]ssim=stats_file={stats}","-an","-f","null","-"],folder,log)
    ssim = [float(re.search(r"\bY:([\d.]+)",s)[1]) for s in (folder/stats).read_text().splitlines()]
    if len(ssim)!=frames_count:
        raise ValueError("SSIM frame count mismatch")
    early=[x for x in psnr[1:5] if x is not None]
    late=[x for x in psnr[-4:] if x is not None]
    return dict(psnr_y_p_pooled=db(mse[1:].mean()),psnr_y_by_frame=psnr,last_frame_psnr_y=psnr[-1],ssim_y_by_frame=ssim,
                ssim_y_p=float(np.mean(ssim[1:])),last_frame_ssim_y=ssim[-1],
                late_minus_early_psnr=float(np.mean(late)-np.mean(early)) if len(early)==len(late)==4 else None)


def replay(backend, jm, folder, row, arrays, reference, timings, *, coeff_trace=False, mc_trace=False):
    # This adapter reads the Oracle template solely for I records and stream structure.
    # Every P mode/MV/residual comes from the prediction NPZ.
    prediction, state = folder/"prediction.npz",folder/"replay.bin"
    if not prediction.exists():
        np.savez_compressed(prediction,**arrays)
    start = time.perf_counter()
    frames_count=arrays["mode_id"].shape[0]+1
    i_template=folder/"i_and_neutral_p.bin"
    i_bytes=isolate(row["current_oracle_state_path"],i_template,frames_count)
    backend.run([sys.executable,backend.ROOT/"scripts/prediction_to_b1_state.py",
                 "--prediction",prediction,"--oracle-template",i_template,
                 "--output-state",state,"--result-json",folder/"adapter.json"],folder,folder/"adapter.log")
    adapt = time.perf_counter()-start
    dummy = folder/"dummy.yuv"
    with dummy.open("wb") as f:
        f.truncate(frames_count*FRAME_BYTES)
    env = {k:v for k,v in os.environ.items() if not k.startswith("JM_")}
    env.update(JM_STATE_MODE="replay",JM_STATE_FILE=str(state),JM_REPLAY_SYNC_MV="1")
    if mc_trace:
        env["JM_LIVE_MC_TRACE"] = str(folder/"mc.bin")
    if coeff_trace:
        env["JM_A_COEFF_TRACE"] = str(folder/"coeff.bin")
    bit, recon = folder/"prediction.264",folder/"encoder_recon.yuv"
    cmd = [jm/"bin/lencod.exe","-d",backend.ROOT/"JM-backend/cfg/encoder_baseline.cfg",
           "-f",backend.ROOT/"linux/skip_p16.cfg"]
    for k,v in dict(InputFile=dummy,OutputFile=bit,ReconFile=recon,SourceWidth=W,SourceHeight=H,
                    FramesToBeEncoded=frames_count,FrameRate=16,QPISlice=28,QPPSlice=28,RDOptimization=0,
                    NumberReferenceFrames=1,PList0References=1).items():
        cmd += ["-p",f"{k} = {v}"]
    start = time.perf_counter()
    backend.run(cmd,folder,folder/"encode.log",env)
    encode = time.perf_counter()-start
    log = (folder/"encode.log").read_text()
    search = re.search(r"Total ME time for sequence\s*:\s*(\d+\.\d+)\s*sec",log,re.I)
    if not search or float(search[1]) != 0 or f"JM state coverage (replay): records={frames_count*1560}" not in log:
        raise ValueError(f"Expected complete replay with zero JM search: {folder}")
    timing=dict(**timings,adapt_s=adapt,jm_replay_s=encode,
                latent_to_bitstream_s=timings.get("head_seconds",0)+timings.get("load_latent_s",0)
                    +timings.get("export_s",0)+adapt+encode)
    if frames_count==17:
        quality=collect_quality(backend,folder,bit,reference,timing)
    else:
        dec,decode_s=decode_check(backend,folder,bit)
        size=bit.stat().st_size
        quality=dict(**timing,ffmpeg_decode_s=decode_s,decoder_agreement=True,bitstream_bytes=size,
                     bitrate_kbps=size*8*16/frames_count/1000)
    dec = folder/"decoded_ffmpeg.yuv"
    if backend.sha(recon)!=backend.sha(dec):
        raise ValueError(f"Encoder reconstruction != standard decoders: {folder}")
    quality.update(oracle_i_bytes_read=i_bytes,oracle_p_bytes_read=0,encoder_matches_decoder=True, motion_search_s_reported=float(search[1]) if search else None,
                   **video_details(backend,folder,reference,dec,"prediction",frames_count))
    return quality


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    for k in ("checkpoint","output","kit","jm"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--split",nargs="+",choices=["train","val","test"],default=["test"])
    p.add_argument("--steps",type=int,nargs="+",choices=STEPS,default=STEPS)
    p.add_argument("--replay",type=int,nargs="*",choices=STEPS,default=[])
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    a.output.mkdir(parents=True,exist_ok=False)
    model=load_model(a.checkpoint)
    backend=load_backend(a.kit)
    rows=records(a.config)
    warm=rows["val"][0]
    predict(model,input_tensor(warm,model.kind),warm["step"])
    results=[]
    for split in a.split:
        for r in rows[split]:
            step=r["step"]
            if step not in a.steps:
                continue
            folder=a.output/split/r["sample_id"]/f"step_{step:03d}"
            folder.mkdir(parents=True)
            start=time.perf_counter()
            z=input_tensor(r,model.kind)
            torch.cuda.synchronize()
            load_s=time.perf_counter()-start
            pred,timing=predict(model,z,step)
            start=time.perf_counter()
            np.savez_compressed(folder/"prediction.npz",**pred)
            timing.update(load_latent_s=load_s,export_s=time.perf_counter()-start)
            result=dict(split=split,sample_id=r["sample_id"],prompt=r["prompt"],step=step,**timing)
            result["state"]=state_metrics(pred,targets(r))
            if step in a.replay:
                ref=Path(r["source_export_dir"])/"input_yuv420p.yuv"
                result["replay"]=replay(backend,a.jm,folder,r,pred,ref,timing)
                recon=r["current_reference_recon_path"]
                result["oracle"]=dict(**video_details(backend,folder,ref,recon,"oracle"),
                  bitrate_kbps=Path(r["current_oracle_state_path"]).with_name("reference.264").stat().st_size*8*16/17/1000)
            (folder/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False))
            results.append(result)
            print(r["sample_id"],step,result["state"]["macro_f1"],result.get("replay",{}).get("psnr_y_p_db"),flush=True)
    (a.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
