"""Allocate sparse native-transform residuals against the evolving fixed-MV JM reference."""
import argparse
import json
import math
import shutil
import socket
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "P0_5"))
import diagnose as p05
from diagnose import CONFIGS, FRAMES, KIT, P16_COUNT, digest, execute, load_backend
from diagnose import load_npz, preserve, read_trace, read_yuv, records
from motion_comp import luma
sys.path.insert(0, str(HERE.parent))
from P0_5.report import frame_bits

ROOT = p05.ROOT
P05 = ROOT / "P0_5"
JM = ROOT / "P0_6/jm"
PCTS = (5, 3, 2, 1, 0)
POLICIES = ("uniform", "early", "offline_gain")
TRANSFORM = np.array([[1,1,1,1],[2,1,-1,-2],[1,-1,-1,1],[1,-2,2,-1]], dtype=np.int32)
BASIS = np.einsum("ay,bx->abyx",TRANSFORM,TRANSFORM).reshape(16,16)
NORM = (BASIS**2).sum(1)
AMPLITUDE_CAP = 255//np.abs(BASIS).max(1)
# CAVLC heuristic. Admission is determined by the measured complete bitstream.
SCAN = np.array([0,1,5,6,2,4,7,12,3,8,11,13,9,10,14,15])


def blocks(y):
    return y.reshape(120,4,208,4).transpose(0,2,1,3).reshape(-1,16)


def from_blocks(samples):
    return (samples.reshape(120,208,4,4).transpose(0,2,1,3).reshape(30,16,52,16)
            .transpose(0,2,1,3).reshape(P16_COUNT,16,16))


def candidates(target, ref, mv, valid):
    mc = luma(ref, mv).astype(np.int32)
    delta = blocks(target.astype(np.int32)-mc)
    dot = delta @ BASIS.T
    amp = np.clip(np.rint(dot / NORM).astype(np.int32), -AMPLITUDE_CAP, AMPLITUDE_CAP)
    gain = 2*dot*amp - NORM*amp**2
    # Rank conservatively against small AC amplitudes near QP28's dead zone.
    gain = np.where((np.abs(amp)>=3) & valid[:,None], gain, 0)
    rate = 9 + 2*np.floor(np.log2(np.abs(amp)+1)).astype(np.int32) + SCAN//3
    idx = (gain/rate).argmax(1)
    ind = np.arange(len(idx))
    return idx, amp[ind,idx], gain[ind,idx], rate[ind,idx]


def choose(target, ref, mv, valid, budget):
    idx, amp, gain, rate = candidates(target,ref,mv,valid)
    order = np.argsort(-gain/rate,kind="stable")
    order = order[(np.cumsum(rate[order])<=budget) & (gain[order]>0)]
    sparse = np.zeros((len(valid),16),dtype=np.int16)
    sparse[order] = amp[order,None]*BASIS[idx[order]]
    return from_blocks(sparse), len(order), int(rate[order].sum())


def recv(conn, n):
    buf=bytearray(n)
    data=memoryview(buf)
    while data:
        count=conn.recv_into(data)
        if not count:
            raise RuntimeError("JM closed the reference exchange early")
        data=data[count:]
    return buf


def teacher(backend, work, row, zero, source, source_y, vectors, valid, budgets):
    path = work / "frame.sock"
    residual, selected, estimated, refs = [], [], [], []
    old = backend.run
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen(1)
        listener.settimeout(30)
        def exchange():
            conn,_=listener.accept()
            with conn:
                conn.settimeout(30)
                for t in range(FRAMES):
                    frame,=struct.unpack("<I",recv(conn,4))
                    assert frame==t+1
                    ref=np.frombuffer(recv(conn,832*480*2),dtype="<u2").reshape(480,832).astype(np.uint8)
                    refs.append(ref)
                    selected_y,n,b=choose(source_y[t+1],ref,vectors[t],valid[t],budgets[t])
                    residual.append(selected_y.reshape(30,52,16,16))
                    selected.append(n)
                    estimated.append(b)
                    conn.sendall(selected_y.astype("<i2",copy=False).tobytes())
        def run(cmd, cwd, log, env=None):
            if Path(cmd[0]).name == "lencod.exe":
                env=dict(env,JM_P06_SOCKET=str(path))
            return old(cmd,cwd,log,env)
        backend.run=run
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                job=pool.submit(exchange)
                try:
                    metric=execute(backend,work/"teacher",row,zero,source,trace=False)
                finally:
                    job.result()
        finally:
            backend.run=old
            path.unlink()
    recon=read_yuv(work/"teacher/encoder_recon.yuv")[0]
    assert np.array_equal(np.stack(refs),recon[:-1]),"JM exchanged the wrong reference"
    arrays=dict(zero,residual_y=np.stack(residual))
    return arrays,metric,selected,estimated


def frame_record(arm, folder, quality, requested, estimated, selected, anchor_vectors, anchor_mode):
    trace,bits=read_trace(folder,False)
    assert np.array_equal(trace["fields"][...,:2],anchor_vectors)
    assert np.array_equal(bits[...,2],anchor_mode)
    assert not trace["fields"][...,2:][anchor_mode==0].any()
    coeff=trace["fields"][...,2:].sum(1)
    if arm not in ("A","zero"):
        assert (coeff[:,0] <= np.asarray(selected)).all() and not coeff[:,1].any()
    actual=frame_bits(folder/"prediction.264")
    return [dict(arm=arm,frame=t+1,proxy_budget_bits=int(requested[t]),proxy_allocated_bits=int(estimated[t]),
                 selected_transform_blocks=int(selected[t]),luma_nonzero_coeffs=int(coeff[t,0]),
                 chroma_nonzero_coeffs=int(coeff[t,1]),residual_coeff_bits=int(bits[t,:,6:].sum()),
                 mv_bits=int(bits[t,:,5].sum()),p_nal_bits=actual[t],psnr_y_db=quality["psnr_y_by_frame"][t+1])
            for t in range(FRAMES)]


def allocation(source_y, ref_y, vectors, valid):
    gain=[]
    for t in range(FRAMES):
        _,_,s,b=candidates(source_y[t+1],ref_y[t],vectors[t],valid[t])
        top=np.sort(s/b)[-128:]
        gain.append(float(np.sqrt(top.mean()+1)))
    weights=dict(uniform=np.ones(FRAMES),early=np.exp2(-np.arange(FRAMES)/6),offline_gain=np.array(gain))
    return {k:v/v.sum() for k,v in weights.items()}


def budgets_for(total, weights):
    endpoints=np.floor(np.r_[0,np.cumsum(weights)]*total+1e-8).astype(int)
    endpoints[-1]=total
    return np.diff(endpoints)


def trials(backend, work, row, zero, source, source_y, vectors, valid, weights,
           target, zero_bits, scale):
    available=target-zero_bits
    if available<=0:
        return None, [], scale
    initial=max(1,round(available*scale))
    budget=initial
    lower,upper=0,None
    attempts=[]
    best=None
    for i in range(3):
        frame_budgets=budgets_for(budget,weights)
        arrays,quality,selected,estimated=teacher(backend,work,row,zero,source,source_y,
                                                   vectors,valid,frame_budgets)
        shutil.rmtree(work/"teacher")
        size=quality["bitstream_bytes"]*8
        attempts.append(dict(proxy_budget_bits=budget,actual_total_bits=size,
                             selected_transform_blocks=sum(selected),under_cap=size<=target))
        if size<=target:
            lower=budget
            if best is None or size>best[0]["bitstream_bytes"]*8:
                best=(quality,arrays,frame_budgets,selected,estimated)
        else:
            upper=budget if upper is None else min(upper,budget)
        spent=size-zero_bits
        if spent>0:
            scale=np.clip(budget/spent,0.15,5.0)
        if size<=target and target-size<=max(64,available*.08):
            break
        if upper is not None and lower+1>=upper:
            break
        guess=round(available*float(scale)*.96) if spent>0 else budget*2
        budget=max(lower+1,guess)
        if upper is not None:
            budget=min(upper-1,budget)
        if any(a["proxy_budget_bits"]==budget for a in attempts):
            break
    return best,attempts,float(scale)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",required=True,choices=("train","val"))
    p.add_argument("--sample",required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    row,=(r for r in records(CONFIGS)[a.split] if r["step"]==20 and r["sample_id"]==a.sample)
    original=P05/"full"/a.split/a.sample/"A"
    baseline=load_npz(original/"prediction.npz")
    zero={**baseline,**{f"residual_{k}":np.zeros_like(baseline[f"residual_{k}"]) for k in "yuv"}}
    source=Path(row["source_export_dir"])/"input_yuv420p.yuv"
    source_y=read_yuv(source)[0]
    vectors=baseline["mv_qpel"]
    valid=(baseline["mode_id"]==1).repeat(4,axis=1).repeat(4,axis=2).reshape(FRAMES,-1)
    p05.JM=JM
    backend=load_backend(KIT)
    a.output.mkdir(parents=True)
    frames,curve,trial_rows=[],[],[]
    z=[0]*FRAMES
    with tempfile.TemporaryDirectory(dir="/dev/shm",prefix="a1-p06-") as tmp:
        work=Path(tmp)
        path=work/"A"
        base=execute(backend,path,row,baseline,source)
        assert base["stream_sha256"]==digest(original/"prediction.264")
        trace,bits=read_trace(path,False)
        anchor_vectors=trace["fields"][...,:2].copy()
        anchor_mode=bits[...,2].copy()
        assert np.array_equal(anchor_mode,baseline["mode_id"].reshape(FRAMES,P16_COUNT))
        assert np.array_equal(anchor_vectors[anchor_mode==1],vectors.reshape(FRAMES,P16_COUNT,2)[anchor_mode==1])
        np.savez_compressed(a.output/"fixed_mode_mv.npz",mode_id=baseline["mode_id"],mv_qpel=baseline["mv_qpel"],
                            native_mv_qpel=anchor_vectors.reshape(FRAMES,30,52,2))
        alloc=allocation(source_y,read_yuv(path/"encoder_recon.yuv")[0],vectors,valid)
        (a.output/"frame_weights.json").write_text(json.dumps({k:v.tolist() for k,v in alloc.items()}))
        base_bits=base["bitstream_bytes"]*8
        def record(arm, policy, pct, cap, result, use, frame_budget, selected, estimated, paired):
            frames.extend(frame_record(arm,use,result,frame_budget,estimated,selected,anchor_vectors,anchor_mode))
            curve.append(dict(arm=arm,allocation=policy,target_pct=pct,target_total_bits=cap,
                              bitstream_bits=result["bitstream_bytes"]*8,
                              under_cap=result["bitstream_bytes"]*8<=cap,paired_offline=paired,
                              diagnostic_oracle=paired,**result))
        record("A","anchor",0,base_bits,base,path,z,z,z,False)
        preserve(path,a.output/"A")
        shutil.rmtree(path)
        path=work/"zero"
        q0=execute(backend,path,row,zero,source)
        zbits=q0["bitstream_bytes"]*8
        record("zero","anchor",0,base_bits,q0,path,z,z,z,False)
        preserve(path,a.output/"zero")
        zero_frames=frames[-16:].copy()
        shutil.rmtree(path)
        for policy in (("uniform",) if a.smoke else POLICIES):
            scale=1.
            for pct in ((5,1,0) if a.smoke else PCTS):
                cap=math.floor(base_bits*(100+pct)/100)
                best,attempts,scale=trials(backend,work,row,zero,source,source_y,vectors,valid,
                                           alloc[policy],cap,zbits,scale)
                arm=f"{policy}_{pct}"
                for i,d in enumerate(attempts,1):
                    trial_rows.append(dict(arm=arm,attempt=i,target_total_bits=cap,**d))
                if best is None:
                    # A's measured stream is always valid at every target.
                    q,anchor=(q0,"zero") if zbits<=cap else (base,"A")
                    curve.append(dict(arm=arm,allocation=policy,target_pct=pct,target_total_bits=cap,
                                      bitstream_bits=q["bitstream_bytes"]*8,under_cap=True,
                                      paired_offline=False,diagnostic_oracle=False,fallback=anchor,**q))
                    orig=zero_frames if anchor=="zero" else frames[:16]
                    frames.extend(dict(f,arm=arm) for f in orig)
                else:
                    measured,arrays,frame_budget,selected,estimated=best
                    assert np.array_equal(arrays["mode_id"],baseline["mode_id"])
                    assert np.array_equal(arrays["mv_qpel"],vectors)
                    assert not arrays["residual_y"][baseline["mode_id"]==0].any()
                    path=work/arm
                    strict=execute(backend,path,row,arrays,source)
                    assert strict["stream_sha256"]==measured["stream_sha256"]
                    record(arm,policy,pct,cap,strict,path,frame_budget,selected,estimated,True)
                    curve[-1]["fallback"]=""
                    preserve(path,a.output/arm)
                    shutil.rmtree(path)
        for point in curve:
            point.setdefault("fallback","")
    p05.csv_write(a.output/"per_frame.csv",frames)
    if trial_rows:
        p05.csv_write(a.output/"trials.csv",trial_rows)
    (a.output/"summary.json").write_text(json.dumps(curve,indent=2,allow_nan=False))
    print("DONE",a.split,a.sample,flush=True)


if __name__=="__main__":
    main()
