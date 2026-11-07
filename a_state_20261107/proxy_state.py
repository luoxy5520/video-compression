"""Export fixed P state with reference-consistent residuals computed before replay."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from discrete_mv import CLASSES,LIMIT
from evaluate import replay, load_backend
from motion_comp import chroma,luma
from oracle_residual import blocks,yuv,parse_trace,load_models,proxy
from experiment import shortlist
from shadow_luma import feedback
from state import PLANES,latent,load_model,predict,records


def load(a):
    state=load_model(a.checkpoint)
    head=torch.nn.Sequential(torch.nn.Conv3d(state.mode.in_channels,64,1),torch.nn.SiLU(),
                             torch.nn.Conv3d(64,2*CLASSES,1)).cuda().eval()
    saved=torch.load(a.mv,weights_only=True)
    if saved["limit"]!=LIMIT:
        raise ValueError("MV qpel bins disagree")
    head.load_state_dict(saved["weights"])
    vae,decision=load_models(a)
    return state,head,vae,decision


@torch.inference_mode()
def prepare(a,models,path,folder):
    state,head,vae,decision=models
    folder.mkdir(parents=True)
    start=time.perf_counter()
    z=latent(path)
    pred,_=predict(state,z,20)
    feature=state.encode_features(z,z.new_tensor([1.]))
    logits=head(feature)[0].movedim(0,-1).reshape(16,30,52,2,CLASSES)
    confidence,bins=logits.softmax(-1).max(-1)
    bins=bins.cpu().numpy()
    continuous=pred["mv_qpel"]
    mv=np.where(bins==0,np.minimum(continuous,-LIMIT-1),bins-LIMIT-1)
    mv=np.where(bins==CLASSES-1,np.maximum(continuous,LIMIT+1),mv).astype(np.int16)
    proxy({"latent_path":str(path)},folder/"proxy.yuv",vae)
    if a.correlation:
        candidates,features=shortlist(folder/"proxy.yuv")
        cost,_=decision(torch.from_numpy(features).cuda())
        best=cost.argmin(-1).cpu().numpy()
        pred["proxy_mv"]=(candidates[np.arange(len(best)),best]*4).reshape(16,30,52,2)
    pred["discrete_confidence"]=confidence.prod(-1).cpu().numpy()
    return pred,mv,yuv(folder/"proxy.yuv"),time.perf_counter()-start


def pack(pred,mv,source,variant):
    if variant=="all_p16_zero":
        mv=np.zeros_like(mv)
    if variant=="all_p16_continuous":
        mv=pred["mv_qpel"]
    if variant.startswith(("all_p16_proxy","all_p16_feedback_proxy")):
        mv=pred["proxy_mv"]
    mode=np.ones_like(pred["mode_id"]) if variant.startswith("all_p16") else pred["mode_id"]
    out=dict(mode_id=mode,mv_qpel=mv)
    if variant=="predicted":
        out.update(mode_confidence=pred["confidence"][...,0],
                   mv_confidence=pred["discrete_confidence"])
    for plane, (p,k,_,_) in enumerate(PLANES):
        fn=luma if k==16 else chroma
        pixels=source[plane]
        if k==16 and "_feedback" in variant:
            ref=pixels[0]
            diff=[]
            for frame in range(1,17):
                mc=fn(ref,mv[frame-1])
                r=pixels[frame].astype(np.int16)-mc
                diff.append(r)
                ref=feedback(mc,r)
        else:
            diff=[pixels[frame].astype(np.int16)-fn(pixels[frame-1],mv[frame-1]) for frame in range(1,17)]
        diff=np.stack(diff).reshape(16,30,k,52,k).transpose(0,1,3,2,4)
        out[f"residual_{p}"]=np.where(mode[...,None,None]==1,diff,0).astype(np.int16)
    if "_skip" in variant:
        # Select Skip only where the H.264 zero predictor is guaranteed. The
        # residual threshold is an open-loop experiment and does not ensure
        # quantized reconstruction equivalence.
        zero=(mv==0).all(-1)
        left=np.pad(zero[:,:,:-1],((0,0),(0,0),(1,0)),constant_values=True)
        above=np.pad(zero[:,:-1,:],((0,0),(1,0),(0,0)),constant_values=True)
        threshold=float(variant.rsplit("skip",1)[-1])
        err=np.zeros_like(mode,dtype=np.float64)
        for plane in "yuv":
            err+=np.mean(out[f"residual_{plane}"].astype(np.float64)**2,axis=(-2,-1))
        use=zero&(left|above)&(err<=threshold)
        out["mode_id"][use]=0
        for plane in "yuv":
            out[f"residual_{plane}"][use]=0
    return out


def verify(a,row):
    folder=a.oracle_mc/row["sample_id"]/"oracle"
    mc,masks=parse_trace(folder/"mc.bin")
    recon=yuv(row["current_reference_recon_path"])
    with np.load(folder/"prediction.npz",allow_pickle=False) as d:
        mv=d["mv_qpel"]
    report=[]
    for i,plane in enumerate("yuv"):
        fn=luma if i==0 else chroma
        got=np.stack([fn(recon[i][j],mv[j]) for j in range(16)])
        mask=masks[i][1:]
        delta=got.astype(np.int16)-mc[i][1:]
        dif=int(np.count_nonzero(delta[mask]))
        report.append(dict(plane=plane,pixels=int(mask.sum()),differences=dif,
                           max_abs=int(np.max(np.abs(delta[mask]),initial=0))))
    if a.feedback:
        with np.load(folder/"prediction.npz",allow_pickle=False) as d:
            res=d["residual_y"].transpose(0,1,3,2,4).reshape(16,480,832)
        got=np.stack([feedback(mc[0][j+1],res[j]) for j in range(16)])
        mask=masks[0][1:]
        diff=np.abs(got.astype(np.int16)-recon[0][1:])
        report.append(dict(plane="luma_recon",pixels=int(mask.sum()),
                           differences=int(np.count_nonzero(diff[mask])),max_abs=int(diff[mask].max(initial=0))))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+")
    p.add_argument("--latent",type=Path)
    options=("predicted","all_p16_discrete","all_p16_zero","all_p16_continuous")
    correlated=("all_p16_proxy","all_p16_proxy_skip4","all_p16_proxy_skip16")
    p.add_argument("--variant",choices=options+correlated+("all_p16_feedback_discrete","all_p16_feedback_proxy"))
    feedback_arms=("all_p16_feedback_discrete","all_p16_feedback_proxy")
    p.add_argument("--correlation",action="store_true")
    p.add_argument("--feedback",action="store_true")
    for k in ("checkpoint","mv","output","projection","decision","kit","jm","oracle-mc"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    a.output.mkdir(parents=True,exist_ok=False)
    if a.latent:
        if a.variant in correlated or a.variant=="all_p16_feedback_proxy":
            a.correlation=True
        pred,mv,src,_=prepare(a,load(a),a.latent,a.output/"input")
        np.savez_compressed(a.output/"prediction.npz",**pack(pred,mv,src,a.variant or "predicted"))
        return
    if a.feedback: a.correlation=True
    rows=records(a.config)
    if a.oracle_mc:
        checks={r["sample_id"]:verify(a,r) for r in rows["val"] if r["step"]==20}
        (a.output/"mc_check.json").write_text(json.dumps(checks,indent=2))
        if any(p["differences"] for ps in checks.values() for p in ps):
            raise ValueError("Standalone MC disagrees with JM on the Oracle validation trajectory")
    models=load(a)
    backend=load_backend(a.kit)
    results=[]
    variants=("all_p16_discrete",)+correlated if a.correlation else options
    if a.feedback: variants=("all_p16_discrete","all_p16_proxy")+feedback_arms
    selected=None
    for split in ("val","test"):
        split_results=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[]
            for row in rows[split]:
                if row["step"]!=20:
                    continue
                sid=row["sample_id"]
                root=a.output/sid
                pred,mv,src,t=prepare(a,models,row["final_latent_path"],root)
                for tag in variants if split=="val" else (selected,):
                    folder=root/tag
                    folder.mkdir()
                    start=time.perf_counter()
                    out=pack(pred,mv,src,tag)
                    ref=Path(row["source_export_dir"])/"input_yuv420p.yuv"
                    run=pool.submit(replay,backend,a.jm,folder,row,out,ref,{"head_seconds":t+time.perf_counter()-start})
                    jobs.append((sid,tag,run))
            for sid,tag,job in jobs:
                row=dict(sample_id=sid,variant=tag,split=split,**job.result())
                split_results.append(row)
                row["oracle_kbps"]=Path(next(r["current_oracle_state_path"] for r in rows[split]
                    if r["sample_id"]==sid and r["step"]==20)).with_name("reference.264").stat().st_size*128/17000
                print(sid,tag,row["psnr_y_p_db"],row["bitrate_kbps"],flush=True)
        results.extend(split_results)
        if split=="val":
            by_tag={tag:[r for r in split_results if r["variant"]==tag] for tag in variants}
            score={tag:np.mean([r["psnr_y_p_db"] for r in rs]) for tag,rs in by_tag.items()}
            rate={tag:np.mean([r["bitrate_kbps"] for r in rs]) for tag,rs in by_tag.items()}
            max_quality=max(score.values())
            allowed=[tag for tag in variants if score[tag]>=max_quality-.25]
            selected=min(allowed,key=rate.get) if a.correlation else max(variants,key=score.get)
            criterion="lowest validation bitrate within 0.25 dB of best quality" if a.correlation else "max validation mean P Y PSNR"
            (a.output/"selection.json").write_text(json.dumps(dict(selected=selected,criterion=criterion)))
    (a.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
