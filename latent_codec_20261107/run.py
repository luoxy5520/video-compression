"""Run an Oracle-free, bounded-search H.264 encoder from final latent."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/name) for name in ("decision_search_20261107","residual_reference_20261107",
                                            "full_state_20261107")]
from benchmark import collect_quality, load_backend
from experiment import DecisionHead, proxy, shortlist
from train_rank import RankProjection, install
from train_state import load_config
from train_tail import FrozenPrefix

BLOCKS = 16*30*52
HEADER = struct.Struct("<8s14I")
RECORD = np.dtype([("frame","<u4"),("addr","<u4"),("slice","<i2"),("mode","<i2"),("qp","<i2"),
                   ("intra","<i2"),("chroma","<i2"),("blocks","i1",4),("mv","<i2",(16,2)),
                   ("ref","i1",16),("residual","<i2",384)])
OPTIONS = ("top1","top4","refine","skip4","skip9","skip16","skip25","skip36","skip64","skip100")


def write_state(path, mv):
    """Make version-7 P records directly from the proposal array."""
    if mv.shape != (BLOCKS,2) or not np.issubdtype(mv.dtype,np.integer):
        raise ValueError("Expected one qpel vector for every P block")
    data = np.zeros(BLOCKS,dtype=RECORD)
    data["frame"] = np.repeat(np.arange(1,17),1560)
    data["addr"] = np.tile(np.arange(1560),16)
    data["mode"] = 1
    data["qp"] = 28
    data["blocks"] = 1
    data["mv"] = mv[:,None,:]
    with path.open("wb") as f:
        f.write(HEADER.pack(b"JMZV1\0\0\0",7,832,480,1560,6,1,0,66,40,1,3,BLOCKS,
                            RECORD.itemsize-768,0))
        data.tofile(f)


@torch.inference_mode()
def load_models(a):
    torch.set_num_threads(4)
    vae = FrozenPrefix().vae
    saved = torch.load(a.projection,weights_only=True)
    decoder = RankProjection(saved["channels"],saved["rank"]).cuda().eval()
    decoder.load_state_dict(saved["weights"])
    install(vae,decoder)
    head = DecisionHead(27).cuda().eval()
    head.load_state_dict(torch.load(a.decision,weights_only=True))
    return vae, head


@torch.inference_mode()
def prepare_one(latent, folder, vae, head):
    folder.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    vae_s = proxy({"latent_path":str(latent)},folder/"proxy.yuv",vae)
    candidates, features = shortlist(folder/"proxy.yuv")
    costs,_ = head(torch.from_numpy(features).cuda())
    top = costs.topk(4,largest=False).indices.cpu().numpy()
    qpel = 4*np.take_along_axis(candidates,top[...,None],axis=1)
    qpel.astype("<i2").tofile(folder/"shortlist.bin")
    write_state(folder/"predicted_p.bin",qpel[:,0])
    return dict(preparation_s=time.perf_counter()-start,vae_s=vae_s)


def prepare(a, samples):
    vae, head = load_models(a)
    for i,sid in enumerate([samples[0],*samples]):
        folder = a.output/"samples"/sid
        timing = prepare_one(a.data/sid/"final_latent.pt",folder,vae,head)
        if i:
            (folder/"prepared.json").write_text(json.dumps(timing,indent=2))
            print(json.dumps(dict(sample_id=sid,**timing)),flush=True)


def encode_prepared(a, backend, sample, case, tag):
    case.mkdir()
    bitstream = case/"prediction.264"
    proxy_path = sample/"proxy.yuv"
    options = os.environ.copy()
    for key in ("JM_SKIP_GATE_MSE","JM_SKIP_GATE_TRACE","JM_LATENT_SHORTLIST","JM_LATENT_REFINE",
                "JM_LIVE_RESIDUAL","JM_LIVE_MC_TRACE","JM_ALIGNED_WEIGHTS"):
        options.pop(key,None)
    options.update(JM_STATE_MODE="replay",JM_STATE_FILE=str(sample/"predicted_p.bin"),
                   JM_REPLAY_SYNC_MV="1",JM_REPLAY_NATIVE_I="1",JM_ALIGNED_PROXY=str(proxy_path))
    if tag != "top1":
        options["JM_LATENT_SHORTLIST"] = str(sample/"shortlist.bin")
    if tag not in ("top1","top4"):
        options["JM_LATENT_REFINE"] = "1"
    if tag.startswith("skip"):
        options["JM_SKIP_GATE_MSE"] = tag.removeprefix("skip")
    settings = dict(InputFile=proxy_path,OutputFile=bitstream,ReconFile=case/"recon.yuv",
                    SourceWidth=832,SourceHeight=480,FramesToBeEncoded=17,FrameRate=16,
                    QPISlice=28,QPPSlice=28,RDOptimization=0,NumberReferenceFrames=1,PList0References=1)
    command = [a.jm/"bin/lencod.exe","-d",a.kit/"JM-backend/cfg/encoder_baseline.cfg",
               "-f",a.kit/"linux/skip_p16.cfg"]
    for key,value in settings.items():
        command += ["-p",f"{key} = {value}"]
    start = time.perf_counter()
    backend.run(command,case,case/"encode.log",options)
    return bitstream, time.perf_counter()-start


def run_case(a, backend, item):
    split,sid,tag = item
    sample = a.output/"samples"/sid
    case = sample/tag
    bitstream, encode_s = encode_prepared(a,backend,sample,case,tag)
    prepared = json.loads((sample/"prepared.json").read_text())
    result = collect_quality(backend,case,bitstream,a.data/sid/"final_raw.yuv",
                             dict(encoder_s=encode_s,**prepared,
                                  latent_to_bitstream_s=encode_s+prepared["preparation_s"]))
    if backend.sha(case/"recon.yuv") != backend.sha(case/"decoded_ffmpeg.yuv"):
        raise ValueError(f"Reconstructed reference mismatch: {sid}, {tag}")
    log = (case/"encode.log").read_text()
    skip = re.search(r"B1 Skip gate: accepted=(\d+) total=(\d+)",log)
    motion = re.search(r"B1 latent motion: blocks=(\d+) candidates=(\d+)",log)
    skip_count = int(skip[1]) if skip else 0
    if tag.startswith("skip") and (not skip or int(skip[2])!=BLOCKS):
        raise ValueError("Incomplete Skip gate")
    if tag != "top1" and (not motion or int(motion[1])+skip_count!=BLOCKS):
        raise ValueError("Incomplete shortlist selection")
    result.update(sample_id=sid,split=split,variant=tag,encoder_matches_decoder=True,
                  skip_fraction=skip_count/BLOCKS,candidates_per_mb=int(motion[2])/BLOCKS if motion else 1)
    (case/"metrics.json").write_text(json.dumps(result,indent=2))
    return result


def summary(records):
    return {tag:{key:float(np.mean([r[key] for r in records if r["variant"]==tag]))
                 for key in ("psnr_y_p_db","psnr_yuv_p_db","bitrate_kbps","skip_fraction",
                             "latent_to_bitstream_s","encoder_s","candidates_per_mb")}
            for tag in dict.fromkeys(r["variant"] for r in records)}


def main(a):
    config,_,_ = load_config(a.config)
    val, test = (config["splits"][key] for key in ("val","test"))
    a.output.mkdir(parents=True,exist_ok=False)
    prepare(a,val+test)
    backend = load_backend(a.kit)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        validation = list(executor.map(lambda x:run_case(a,backend,x),
                          [("val",sid,tag) for tag in OPTIONS for sid in val]))
    val_stats = summary(validation)
    native = json.loads(a.native_validation.read_text())
    reference = {r["sample_id"]:r for r in native["per_video"]
                 if r["variant"]=="native8" and r["split"]=="val"}
    if set(reference)!=set(val):
        raise ValueError("Validation split mismatch")
    native_psnr = float(np.mean([r["psnr_y_p_db"] for r in reference.values()]))
    eligible = []
    for tag,row in val_stats.items():
        worst = max(reference[r["sample_id"]]["psnr_y_p_db"]-r["psnr_y_p_db"]
                    for r in validation if r["variant"]==tag)
        if row["psnr_y_p_db"]>=native_psnr-.25 and worst<=.75:
            eligible.append(tag)
    selected = min(eligible,key=lambda tag:val_stats[tag]["bitrate_kbps"]) if eligible else max(
        val_stats,key=lambda tag:val_stats[tag]["psnr_y_p_db"])
    calibration = dict(selected=selected,eligible=eligible,native_validation_psnr=native_psnr,
                       mean_drop_limit=.25,worst_drop_limit=.75,
                       fallback_max_validation_psnr=not bool(eligible))
    print(json.dumps(calibration),flush=True)
    (a.output/"calibration.json").write_text(json.dumps(calibration,indent=2))
    test_variants = list(dict.fromkeys(["top1","top4","refine",selected]))
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        testing = list(executor.map(lambda x:run_case(a,backend,x),
                       [("test",sid,tag) for tag in test_variants for sid in test]))
    (a.output/"results.json").write_text(json.dumps(dict(calibration=calibration,
        means=dict(val=val_stats,test=summary(testing)),per_video=validation+testing),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("config","projection","decision","native-validation","kit","jm","data","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    main(parser.parse_args())
