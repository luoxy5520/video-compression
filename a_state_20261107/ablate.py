"""Step-20 replay: factorial Oracle substitution, diagnostic only."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from evaluate import load_backend, replay
from state import PLANES, records, targets


def run(a, backend, row, tag, keys):
    folder=a.output/row["sample_id"]/tag
    folder.mkdir(parents=True)
    with np.load(a.predictions/"test"/row["sample_id"]/"step_020/prediction.npz",allow_pickle=False) as saved:
        out=dict(saved)
    m,v,r=(x.numpy() for x in targets(row))
    truth=dict(mode_id=m.astype("u1"),mv_qpel=v)
    for p,k,l,u in PLANES:
        truth[f"residual_{p}"]=r[...,l:u].reshape(*m.shape,k,k)
    for key in keys:
        out[key]=truth[key]
    ref=Path(row["source_export_dir"])/"input_yuv420p.yuv"
    result=replay(backend,a.jm,folder,row,out,ref,{})
    result.update(sample_id=row["sample_id"],variant=tag,oracle_keys=keys,
                  p16_fraction=float(out["mode_id"].mean()))
    print(row["sample_id"],tag,result["psnr_y_p_db"],flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",nargs="+",required=True)
    for k in ("kit","jm","predictions","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True)
    backend=load_backend(a.kit)
    rows=[r for r in records(a.config)["test"] if r["step"]==20]
    residual=[f"residual_{p}" for p,*_ in PLANES]
    cases=dict(oracle_mode=["mode_id"],oracle_mv=["mv_qpel"],
               oracle_mode_mv=["mode_id","mv_qpel"],
               oracle_residual=residual, oracle_mode_residual=["mode_id",*residual],
               oracle_mv_residual=["mv_qpel",*residual])
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run,a,backend,row,tag,keys) for row in rows for tag,keys in cases.items()]
        result=[j.result() for j in jobs]
    (a.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
