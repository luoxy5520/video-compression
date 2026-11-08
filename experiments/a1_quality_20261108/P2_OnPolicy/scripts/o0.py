"""Generate two matched sparse label sets and verify native fixed-prefix replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"P2_LabelProbe/src"))
from l1 import decode, labels, pair, p
from repair import to_pixels
from sweep import luma

ARMS=("T0","T1")
FRAME_K=4


def sha_frames(frames):
    return [hashlib.sha256(x).hexdigest() for x in frames]


def historical(split,row):
    root,source=p.paths(split,row)
    frozen=p.p05.load_npz(root/"A/prediction.npz")
    fixed=p.p05.load_npz(root/"fixed_mode_mv.npz")
    meta={r["arm"]:r for r in json.loads((root/"summary.json").read_text())}
    for k in ("mode_id","mv_qpel"):
        np.testing.assert_array_equal(frozen[k],fixed[k])
    paths={}
    for arm,tag in zip(ARMS,("offline_gain_3","A"),strict=True):
        case=root/tag
        if not (case/"prediction.264").exists():
            case=root/meta[tag]["fallback"]
        stream=case/"prediction.264"
        assert p.digest(stream)==meta[tag]["stream_sha256"],stream
        paths[arm]=(case,meta[tag])
    return root,source,frozen,fixed,paths


def validate_mc(folder,recon,frozen,trace):
    used=frozen["mode_id"].reshape(16,1560)==1
    mc=np.stack([luma(ref,mv).reshape(30,16,52,16).transpose(0,2,1,3).reshape(1560,256)
                 for ref,mv in zip(recon[:-1],frozen["mv_qpel"],strict=True)])
    np.testing.assert_array_equal(mc[used],trace["mc"][...,:256][used],err_msg=str(folder))


def strict(folder,row,arrays,source,fixed,expected=None,mc_check=False):
    quality,actual,mv,bits=p.strict(folder,row,arrays,source)
    np.testing.assert_array_equal(mv,fixed["native_mv_qpel"].reshape(16,1560,2))
    np.testing.assert_array_equal(bits[...,2],fixed["mode_id"].reshape(16,1560))
    assert not actual[fixed["mode_id"]==0].any()
    if expected is not None:
        np.testing.assert_array_equal(actual,expected)
    enc=p.p05.read_yuv(folder/"encoder_recon.yuv")[0]
    if mc_check:
        trace,_=p.p05.read_trace(folder,False)
        validate_mc(folder,enc,arrays,trace)
    frame=p.FRAMES+1
    files=("encoder_recon.yuv","decoded_jm.yuv","decoded_ffmpeg.yuv")
    # Use the exact output paths selected by evaluate.decode_check.
    paths=[folder/f for f in files]
    assert all(f.is_file() for f in paths),str(list(folder.glob("*.yuv")))
    blocks=[sha_frames(np.fromfile(f,np.uint8).reshape(frame,-1)) for f in paths]
    assert blocks[0]==blocks[1]==blocks[2]
    return quality,enc,blocks[0],bits


def compare_frame_freq(a,b):
    qa,qb=a["sparse"],b["sparse"]
    x,y=qa!=0,qb!=0
    output=[]
    for t in range(16):
        for freq in [None,*range(16)]:
            mask=(t,...,slice(None) if freq is None else freq)
            left,right=x[mask],y[mask]
            both=left&right
            n=int(both.sum())
            nl,nr=int(left.sum()),int(right.sum())
            v1,v2=qa[mask][both],qb[mask][both]
            output.append(dict(frame=t+1,frequency="all" if freq is None else freq,
                n0=nl,n1=nr,overlap=n,iou=n/(nl+nr-n) if nl+nr-n else None,
                precision=n/nr if nr else None,recall=n/nl if nl else None,
                sign_agreement=float((np.sign(v1)==np.sign(v2)).mean()) if n else None,
                level_mae=float(np.abs(v1-v2).mean()) if n else None,
                t0_selected_to_zero=float((b["dense"][mask][left]==0).mean()) if nl else None,
                t1_selected_to_zero=float((a["dense"][mask][right]==0).mean()) if nr else None))
    return output


def persist(folder,case):
    p.p05.preserve(folder,case)
    for name in ("bits.csv","coeff.bin"):
        (case/name).write_bytes((folder/name).read_bytes())


def clip(out,split,row,smoke):
    sid=row["sample_id"]
    case=out/sid
    case.mkdir()
    root,src,base,fixed,refs=historical(split,row)
    src_y=p.p05.read_yuv(src)[0]
    valid=base["mode_id"]==1
    traces={}
    labels_by_arm={}
    a_ref=decode(refs["T1"][0]/"prediction.264")
    with tempfile.TemporaryDirectory(dir="/dev/shm",prefix="onpolicy-") as tmp:
        work=Path(tmp)
        # Verify teacher's original native Mode/MV and actual decoder references
        # before using them. The frozen A stream is checked against the locked SHA.
        t0_state=p.p05.load_npz(refs["T0"][0]/"prediction.npz")
        for k in ("mode_id","mv_qpel"):
            np.testing.assert_array_equal(t0_state[k],base[k])
        teacher,teacher_y,_,_=strict(work/"original_teacher",row,t0_state,src,fixed,mc_check=smoke)
        assert p.digest(work/"original_teacher/prediction.264")==refs["T0"][1]["stream_sha256"]
        refs_y={"T0":teacher_y[:-1],"T1":a_ref}
        for arm in ARMS:
            prefix=refs_y[arm]
            generated=labels(prefix,src_y,base)
            q=generated["sparse"]
            assert np.all(np.count_nonzero(q,axis=(1,2,3,4))==FRAME_K)
            assert not q[~valid].any()
            path=case/f"{arm}.npz"
            np.savez_compressed(path,q=q)
            if smoke:
                other=case/f"{arm}-repeat.npz"
                np.savez_compressed(other,q=labels(prefix,src_y,base)["sparse"])
                assert p.digest(path)==p.digest(other)
            folder=work/arm
            arrays=dict(base,residual_y=to_pixels(q))
            quality,enc,frame_sha,bits=strict(folder,row,arrays,src,fixed,q,mc_check=smoke)
            actual=enc[:-1]
            drift=np.square(actual.astype(np.float32)-prefix.astype(np.float32)).mean(axis=(1,2))
            fresh=labels(actual,src_y,base)["sparse"]
            match=(fresh==q).all(axis=(1,2,3,4))
            anchor=refs["T1"][1]
            traces[arm]=dict(
                source_reference_stream_sha256=refs[arm][1]["stream_sha256"],
                prefix_ref_y_sha256=sha_frames(prefix),label_sha256=p.digest(path),
                frame_yuv_sha256=frame_sha,reference_consistent=bool(match.all()),equal_label_by_frame=match.tolist(),
                actual_vs_prefix_ref_mse_by_frame=drift.tolist(),
                source_ref_vs_A_mse_by_frame=np.square(prefix.astype(np.float32)-a_ref.astype(np.float32)).mean(axis=(1,2)).tolist(),
                diag_output_vs_A_mse_by_frame=np.square(actual.astype(np.float32)-a_ref.astype(np.float32)).mean(axis=(1,2)).tolist(),
                mode_mv_sha256=p.digest(root/"fixed_mode_mv.npz"),motion_mse_by_frame=generated["mc_mse"].tolist(),
                local_dense_sse_delta_vs_zero=float(np.sum(q.astype(np.float64)**2-2*q.astype(np.float64)*generated["dense"])),
                counts_by_frequency=np.count_nonzero(q,axis=(0,1,2,3)).tolist(),
                selected=int(np.count_nonzero(q)),coeff_bits=int(bits[...,6:].sum()),mv_bits=int(bits[...,5].sum()),
                delta_db=quality["psnr_y_p_db"]-anchor["psnr_y_p_db"],
                delta_last4_db=p.last4(quality)-p.last4(anchor),last4_db=p.last4(quality),
                rate_pct=100*(quality["bitstream_bytes"]/anchor["bitstream_bytes"]-1),
                stream_sha256=p.digest(folder/"prediction.264"),**quality)
            persist(folder,case/arm)
            labels_by_arm[arm]=generated
    left,right=(labels_by_arm[x] for x in ARMS)
    stability=pair(left,right,valid)
    stability.update(t1_selected_dense_zero=float((left["dense"][right["sparse"]!=0]==0).mean()))
    result=dict(sample=sid,prompt=row["prompt"],split=split,candidates=int(valid.sum())*16*16,
                source_sha256=p.digest(src),stability=stability,frame_frequency=compare_frame_freq(left,right),
                arms=traces)
    p.save_json(case/"metrics.json",result)
    print(split,sid,round(stability["iou"],3),{a:round(traces[a]["delta_db"],4) for a in ARMS},flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    assert (args.audit/"audit.json").is_file()
    rows=p.rows_by_split()
    if args.smoke:
        unique={}
        for row in rows["train"]:
            unique.setdefault(row["prompt"],row)
        jobs=[("train",row) for row in list(unique.values())[:2]]
    else:
        jobs=[(split,row) for split,group in rows.items() for row in group]
    args.out.mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks=[pool.submit(clip,args.out,split,row,args.smoke) for split,row in jobs]
        results=[task.result() for task in tasks]
    p.save_json(args.out/"metrics.json",results)


if __name__=="__main__":
    main()
