"""A0.3: isolate residual prediction with Oracle mode/MV/reference."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from evaluate import load_backend, replay
from state import PLANES, latent, load_model, predict, records, targets

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"residual_reference_20261107"))
sys.path.append(str(ROOT/"latent_codec_20261107"))
from export_trace import parse_trace
from run import load_models, proxy


def blocks(x,k):
    return x[1:].reshape(16,30,k,52,k).transpose(0,1,3,2,4)


def yuv(path):
    x=np.fromfile(path,np.uint8).reshape(17,832*480*3//2)
    return [x[:,:399360].reshape(17,480,832),x[:,399360:499200].reshape(17,240,416),
            x[:,499200:].reshape(17,240,416)]


def coeff(path):
    """Densify by scan position; do not confuse shifted run lists with level changes."""
    buf=memoryview(path.read_bytes())
    i=0
    result={}
    while i<len(buf):
        frame,mb=np.frombuffer(buf[i:i+8],dtype="<u4")
        i+=8
        key=int(frame),int(mb)
        if key in result:
            raise ValueError(f"Duplicate trace {key}")
        v=np.zeros((27,64),dtype=np.int32)
        for b in range(27):
            n=buf[i]
            i+=1
            index=-1
            for _ in range(n):
                level=int.from_bytes(buf[i:i+4],"little",signed=True)
                index+=1+buf[i+4]
                i+=5
                v[b,index]=level
        result[key]=v
    return result


def compare_coeff(a,b):
    x,y=coeff(a),coeff(b)
    if x.keys()!=y.keys():
        raise ValueError("Oracle and predicted P16 coefficient blocks differ")
    a=np.stack(list(x.values()))
    b=np.stack([y[k] for k in x])
    active=(a!=0)|(b!=0)
    return dict(p16_mbs=len(a),oracle_nonzero=int((a!=0).sum()),predicted_nonzero=int((b!=0).sum()),
                changed_levels=int((a!=b).sum()),active_union=int(active.sum()),
                mae_on_active=float(np.abs(a-b)[active].mean()) if active.any() else None)


def setup(a,backend,root,row):
    folder=root/row["sample_id"]
    m,v,r=(x.numpy() for x in targets(row))
    arrays=dict(mode_id=m.astype("u1"),mv_qpel=v)
    for p,k,l,u in PLANES:
        arrays[f"residual_{p}"]=r[...,l:u].reshape(*m.shape,k,k)
    ref=Path(row["source_export_dir"])/"input_yuv420p.yuv"
    o=folder/"oracle"
    o.mkdir()
    replay(backend,a.jm,o,row,arrays,ref,{},mc_trace=True,coeff_trace=True)
    if backend.sha(o/"encoder_recon.yuv")!=backend.sha(row["current_reference_recon_path"]):
        raise ValueError("MC trace did not come from the exact Oracle reference trajectory")
    if backend.sha(o/"prediction.264")!=backend.sha(Path(row["current_oracle_state_path"]).with_name("reference.264")):
        raise ValueError("Oracle trajectory did not reproduce the reference stream")
    return row,folder,arrays,ref


def features(folder,arrays):
    mc,mask=parse_trace(folder/"oracle/mc.bin")
    proxies=yuv(folder/"proxy.yuv")
    with np.load(folder/"direct.npz",allow_pickle=False) as f:
        direct={p:f[f"residual_{p}"] for p,k,l,u in PLANES}
    keep=arrays["mode_id"]==1
    for i,(p,k,_,_) in enumerate(PLANES):
        if not np.array_equal(blocks(mask[i],k),np.broadcast_to(keep[...,None,None],(*keep.shape,k,k))):
            raise ValueError(f"Incomplete oracle MC coverage {folder}, {p}")
        source=blocks(proxies[i],k)[keep].astype(np.float32)
        ref=blocks(mc[i],k)[keep].astype(np.float32)
        prior=direct[p][keep].astype(np.float32)
        base=source-ref
        x=np.stack([np.ones_like(base),(source-128)/128,base/32,prior/32,np.abs(base)/32],-1)
        yield p,k,keep,x,base,arrays[f"residual_{p}"][keep]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    for k in ("checkpoint","kit","jm","output","projection","decision"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--locked-fit",type=Path)
    a=p.parse_args()
    torch.set_num_threads(4)
    backend=load_backend(a.kit)
    a.output.mkdir(parents=True,exist_ok=False)
    rows={s:[r for r in rs if r["step"]==20] for s,rs in records(a.config).items()}
    head=load_model(a.checkpoint)
    vae,_=load_models(a)
    rng=np.random.default_rng(77)

    def prepare(split):
        group=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=[]
            for row in rows[split]:
                folder=a.output/row["sample_id"]
                folder.mkdir()
                jobs.append(pool.submit(setup,a,backend,a.output,row))
                proxy({"latent_path":row["final_latent_path"]},folder/"proxy.yuv",vae)
                pred,_=predict(head,latent(row["final_latent_path"]),20)
                np.savez_compressed(folder/"direct.npz",**pred)
            for job in jobs:
                group.append(job.result())
        return group

    if a.locked_fit:
        fit=json.loads(a.locked_fit.read_text())
        if [r["plane"] for r in fit]!=list("yuv"):
            raise ValueError("Expected three locked YUV weights")
        weights=[np.asarray(r["weights"],dtype=np.float64) for r in fit]
    else:
        stats={}
        for split in ("train","val"):
            matrices=np.zeros((3,5,5),dtype=np.float64)
            vectors=np.zeros((3,5),dtype=np.float64)
            sums=np.zeros(3)
            counts=np.zeros(3)
            for row,folder,arrays,ref in prepare(split):
                for i,(p,k,keep,x,b,y) in enumerate(features(folder,arrays)):
                    if not y.size:
                        continue
                    take=rng.choice(y.size,min(80000,y.size),replace=False)
                    X=x.reshape(-1,5)[take].astype(np.float64)
                    Y=(y-b).ravel()[take].astype(np.float64)
                    matrices[i]+=X.T@X
                    vectors[i]+=X.T@Y
                    sums[i]+=Y@Y
                    counts[i]+=len(Y)
            if (counts==0).any():
                raise ValueError("Empty Oracle P16 split")
            stats[split]=matrices/counts[:,None,None],vectors/counts[:,None],sums/counts
        weights=[]
        fit=[]
        for i,(p,k,_,_) in enumerate(PLANES):
            tr,tv,_=stats["train"]
            vr,vv,base=stats["val"]
            candidates=[]
            for lam in (.00001,.0001,.001,.01):
                w=np.linalg.solve(tr[i]+lam*np.diag([0,1,1,1,1]),tv[i])
                mse=base[i]-2*(vv[i]@w)+w@vr[i]@w
                candidates.append((mse,lam,w))
            mse,lam,w=min(candidates,key=lambda t:t[0])
            weights.append(w)
            fit.append(dict(plane=p,ridge=lam,weights=w.tolist(),val_rmse=float(np.sqrt(mse)),
                            val_proxy_minus_oracle_mc_rmse=float(np.sqrt(base[i]))))
    (a.output/"fit.json").write_text(json.dumps(fit,indent=2))
    results=[]
    for row,folder,arrays,ref in prepare("test"):
        output=arrays.copy()
        rmse={}
        for (p,k,keep,x,b,y),w in zip(features(folder,arrays),weights):
            pred=np.rint(b+(x@w)).clip(-255,255).astype(np.int16)
            output[f"residual_{p}"]=np.zeros_like(arrays[f"residual_{p}"])
            output[f"residual_{p}"][keep]=pred
            rmse[p]={key:float(np.sqrt(np.mean((v-y.astype(np.float64))**2)))
                     for key,v in [("learned",pred),("proxy_minus_oracle_mc",b),("zero",0),("direct",x[...,3]*32)]}
        out=folder/"conditioned"
        out.mkdir()
        quality=replay(backend,a.jm,out,row,output,ref,{},coeff_trace=True)
        baseline=Path(row["current_oracle_state_path"]).with_name("reference.264").stat().st_size*128/17000
        result=dict(sample_id=row["sample_id"],rmse=rmse,oracle_bitrate_kbps=baseline,replay=quality,
                    coefficients=compare_coeff(folder/"oracle/coeff.bin",out/"coeff.bin"))
        results.append(result)
        print(json.dumps(result),flush=True)
    (a.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
