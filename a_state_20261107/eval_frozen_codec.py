"""External check of the already frozen model-assisted closed-loop codec."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
from state import records
from evaluate import video_details

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"learned_qpel_20261107")]
from latent_codec_20261107.run import load_backend,load_models,prepare_one,run_case
from qpel_head import model,predict


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    for key in ("kit","jm","data","output","projection","decision","qpel"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    idx=records(a.config)
    a.output.mkdir(parents=True)
    backend=load_backend(a.kit)
    vae,head=load_models(a)
    q=model().cuda().eval()
    q.load_state_dict(torch.load(a.qpel,weights_only=True))
    warm=next(x for x in idx["val"] if x["step"]==20)
    prepare_one(warm["final_latent_path"],a.output/"warm",vae,head)
    checksums={key:hashlib.sha256(getattr(a,key).read_bytes()).hexdigest()
               for key in ("projection","decision","qpel")}
    results=[]
    for r in idx["test"]:
        if r["step"]!=20: continue
        sid=r["sample_id"]
        folder=a.output/"samples"/sid
        times=prepare_one(r["final_latent_path"],folder,vae,head,keep_features=True)
        elapsed=predict(q,folder)
        times["preparation_s"]+=elapsed
        (folder/"prepared.json").write_text(json.dumps(times))
        res=run_case(a,backend,("test",sid,"refine"),name="frozen_qpel4",
                     extra_env={"JM_REPLAY_EXACT_SKIP":"1","JM_QPEL_BUDGET":"4",
                                "JM_QPEL_PRIOR":str(folder/"qpel_prior.bin")})
        reference=Path(r["source_export_dir"])/"input_yuv420p.yuv"
        oracle=video_details(backend,folder,reference,r["current_reference_recon_path"],"oracle")
        bit=Path(r["current_oracle_state_path"]).with_name("reference.264")
        res.update(prompt=r["prompt"],oracle_psnr_y_p_db=oracle["psnr_y_p_pooled"],
                   oracle_bitrate_kbps=bit.stat().st_size*128/17000)
        results.append(res)
        print(sid,res["psnr_y_p_db"],res["bitrate_kbps"],flush=True)
    quality=statistics.mean(r["oracle_psnr_y_p_db"]-r["psnr_y_p_db"] for r in results)
    rate=statistics.mean(r["bitrate_kbps"] for r in results)/statistics.mean(r["oracle_bitrate_kbps"] for r in results)-1
    (a.output/"results.json").write_text(json.dumps(dict(weights_sha256=checksums,per_video=results,
        mean_psnr_loss_db=quality,rate_increase_fraction=rate,passed=quality<=1 and rate<=.10),indent=2))
if __name__=="__main__":
    main()
