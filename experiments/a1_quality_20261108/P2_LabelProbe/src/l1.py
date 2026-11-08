"""Regenerate paired residual labels on three actual reconstructed trajectories."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import subprocess

import numpy as np

from l0 import chosen_rows, p, replay
from sweep import BASIS, NORM, SCAN, blocks, luma
from g_probe import raster

ARMS=("A","teacher","student")
FRAME_K=4
LOW=[0,1,4]


def decode(path):
    raw=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-threads","1","-i",str(path),
                                 "-threads","1","-pix_fmt","yuv420p","-f","rawvideo","-"])
    return np.frombuffer(raw,np.uint8).reshape(17,-1)[:-1,:832*480].reshape(16,480,832)


def labels(ref,source,base):
    sparse,dense,continuous,mc_mse=[],[],[],[]
    for t in range(16):
        mc=luma(ref[t],base["mv_qpel"][t])
        valid=(base["mode_id"][t]==1).repeat(4,axis=0).repeat(4,axis=1).ravel()
        r=blocks(source[t+1].astype(np.int32)-mc.astype(np.int32))
        transform=p.transform(r.reshape(-1,4,4)).reshape(-1,16)
        q=p.quantize(r.reshape(-1,4,4)).reshape(-1,16)
        q[~valid]=0
        transform[~valid]=0
        dot=r@BASIS.T
        limit=255//np.abs(BASIS).max(1)
        amp=np.clip(np.rint(dot/NORM).astype(np.int32),-limit,limit)
        # Quantize the candidate basis, rather than the unprojected dense
        # residual, before selecting a level that can survive JM.
        lookup=np.stack([p.quantize(np.arange(256)[:,None,None]*BASIS[f].reshape(4,4))[:,f//4,f%4]
                         for f in range(16)],axis=1)
        effective=lookup[np.abs(amp),np.arange(16)]*np.sign(amp)
        gain=2*dot*amp-NORM*amp**2
        rate=9+2*np.floor(np.log2(np.abs(amp)+1)).astype(np.int32)+SCAN//3
        score=np.where(valid[:,None] & (np.abs(effective)>=2) & (gain>0),gain/rate,-np.inf)
        f=score.argmax(-1)
        b=np.arange(len(valid))
        best=score[b,f]
        order=np.argsort(-best,kind="stable")[:FRAME_K]
        order=order[np.isfinite(best[order])]
        sparse_t=np.zeros_like(q,dtype=np.int16)
        sparse_t[order,f[order]]=effective[order,f[order]]
        sparse.append(raster(sparse_t))
        dense.append(raster(q.astype(np.int16)))
        continuous.append(raster(transform[:,LOW].astype(np.int16)))
        mc_mse.append(float(np.square(r[valid]).mean()))
    return dict(sparse=np.stack(sparse),dense=np.stack(dense),continuous=np.stack(continuous),mc_mse=np.asarray(mc_mse))


def pair(a,b,mask):
    u,v=a["sparse"],b["sparse"]
    x,y=u!=0,v!=0
    both=x&y
    compare=a["dense"][x]
    changed=b["dense"][x]
    offset=a["dense"].astype(np.int32)-b["dense"].astype(np.int32)
    delta=offset[mask]
    count=int(both.sum())
    return dict(size_left=int(x.sum()),size_right=int(y.sum()),overlap=count,
                iou=count/int((x|y).sum()),precision=count/int(y.sum()),recall=count/int(x.sum()),
                joint_sign_flip=float((np.sign(u[both])!=np.sign(v[both])).mean()) if count else None,
                joint_level_mae=float(np.abs(u[both]-v[both]).mean()) if count else None,
                joint_exact=float((u[both]==v[both]).mean()) if count else None,
                dense_mae=float(np.abs(delta).mean()),dense_exact=float((delta==0).mean()),
                dense_delta_hist={str(k):int(n) for k,n in zip(*np.unique(delta,return_counts=True))},
                left_selected_dense_mae=float(np.abs(compare-changed).mean()),
                left_selected_dense_zero=float((changed==0).mean()),
                left_selected_dense_flip=float(((compare*changed)<0).mean()),
                iou_by_frame=[float(i.sum()/j.sum()) if j.any() else None for i,j in zip(both,x|y,strict=True)])


def cross_cost(a,b,mask):
    truth=b["dense"][mask].astype(np.float64)
    q=a["sparse"][mask].astype(np.float64)
    # Coefficient-space squared-error delta against zero on this target trajectory.
    return float(np.sum(q*q-2*q*truth))


def clip(out,row,split,info):
    root,src=p.paths(split,row)
    sid=row["sample_id"]
    case=out/sid
    case.mkdir()
    base=p.p05.load_npz(root/"A/prediction.npz")
    source=p.p05.read_yuv(src)[0]
    paths={"A":root/"A/prediction.264"}
    if split=="val":
        paths.update(teacher=root/"offline_gain_3/prediction.264",
                     student=info/"d1"/sid/"64/SS/prediction.264")
        if not paths["teacher"].is_file():
            meta=next(m for m in json.loads((root/"summary.json").read_text()) if m["arm"]=="offline_gain_3")
            paths["teacher"]=root/meta["fallback"]/"prediction.264"
    data,details={},{}
    for name,path in paths.items():
        ref=decode(path)
        label=labels(ref,source,base)
        file=case/f"{name}.npz"
        # The sparse target and continuous low bands are retained. Dense
        # 16-frequency targets are regenerable from the same stream/source.
        np.savez_compressed(file,sparse=label["sparse"],continuous=label["continuous"])
        q=label["sparse"]
        dense=label["dense"]
        bins,counts=np.unique(q[q!=0],return_counts=True)
        details[name]=dict(path=str(path),reference_stream_sha256=p.digest(path),label_sha256=p.digest(file),
                           frequency=np.count_nonzero(q,axis=(0,1,2,3)).tolist(),
                           dense_frequency=np.count_nonzero(dense,axis=(0,1,2,3)).tolist(),
                           level_hist={str(i):int(n) for i,n in zip(bins,counts)},
                           frames=np.count_nonzero(q,axis=(1,2,3,4)).tolist(),
                           level_abs_mean=float(np.abs(q[q!=0]).mean()),
                           positive_sign=float((q[q!=0]>0).mean()),mc_mse_by_frame=label["mc_mse"].tolist())
        if split=="val":
            data[name]=label
            rep=case/name
            rep.mkdir()
            details[name]["replay"]=replay(rep,split,row,q,{"budget":FRAME_K*16})
        if sid in ("sample_0020","sample_1024"):
            second=labels(decode(path),source,base)
            for key in label:
                np.testing.assert_array_equal(label[key],second[key])
            file2=case/f"{name}-repeat.npz"
            np.savez_compressed(file2,sparse=second["sparse"],continuous=second["continuous"])
            assert p.digest(file2)==p.digest(file)
            rep2=case/f"{name}-repeat"
            rep2.mkdir()
            check=replay(rep2,split,row,second["sparse"],{"budget":FRAME_K*16})
            assert check["stream_sha256"]==details[name]["replay"]["stream_sha256"]
            details[name]["repeat_sha256"]=p.digest(file2)
            details[name]["repeat_stream_sha256"]=check["stream_sha256"]
    mode=base["mode_id"]==1
    pairs={f"{a}-{b}":pair(data[a],data[b],mode) for a,b in combinations(data,2)}
    costs={f"{a}-{b}":cross_cost(data[a],data[b],mode) for a in data for b in data}
    result=dict(sample=sid,split=split,prompt=row["prompt"],p16_blocks=int(mode.sum())*16,
                source_sha256=p.digest(src),trajectories=details,pairs=pairs,cross_cost=costs)
    p.save_json(case/"clip.json",result)
    print(sid, {k:round(v["iou"],3) for k,v in pairs.items()},flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("out",type=Path)
    a.add_argument("--info",type=Path,default=p.ROOT/"P2_InfoProbe")
    args=a.parse_args()
    args.out.mkdir(exist_ok=False)
    rows=p.rows_by_split()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(clip,args.out,row,split,args.info) for split,group in
              (("val",rows["val"]),("train",chosen_rows(rows["train"]))) for row in group]
        points=[j.result() for j in jobs]
    p.save_json(args.out/"metrics.json",points)


if __name__=="__main__":
    main()
