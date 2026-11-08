"""Summarize native teacher allocation and motion-conditioned sparsity by content."""
import argparse
from pathlib import Path

import numpy as np

from aggregate import read, write
from l0 import p


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("results",type=Path)
    a=cli.parse_args()
    rows={r["sample_id"]:r for group in p.rows_by_split().values() for r in group}
    label_data=read(a.root/"l1/metrics.json")
    output=[]
    for clip in label_data:
        sid=clip["sample"]
        row=rows[sid]
        root,_=p.paths(clip["split"],row)
        base=p.p05.load_npz(root/"A/prediction.npz")
        mode=base["mode_id"]==1
        motion=np.linalg.norm(base["mv_qpel"].astype(np.float32),axis=-1)
        category=row.get("category","unspecified")
        variants={"native_teacher":p.p05.load_npz(a.root/"l0_labels"/f"{sid}.npz")["q"]}
        variants.update({k:p.p05.load_npz(a.root/"l1"/sid/f"{k}.npz")["sparse"] for k in clip["trajectories"]})
        for key,q in variants.items():
            picked=q!=0
            for name,mask in (("all",mode),("qpel_lt1",mode & (motion<1)),("qpel_1_to_8",mode & (motion>=1) & (motion<8)),
                              ("qpel_8_to_32",mode & (motion>=8) & (motion<32)),("qpel_ge32",mode & (motion>=32))):
                values=q[mask]
                native=values[values!=0]
                frequencies=np.count_nonzero(values,axis=(0,1))
                output.append(dict(sample=sid,split=clip["split"],prompt=clip["prompt"],category=category,
                                   trajectory=key,motion=name,candidates=int(mask.sum())*16*16,
                                   selected=len(native),dc=int(frequencies[0]),low=int(frequencies[[0,1,4]].sum()),
                                   sign_positive=int((native>0).sum()),mean_abs_level=float(np.abs(native).mean()) if len(native) else None,
                                   last4_selected=int(picked[-4:][mask[-4:]].sum()),
                                   frame_count=" ".join(str(int(i)) for i in np.count_nonzero(picked & mask[...,None,None],axis=(1,2,3,4)))))
    write(a.results/"l1_content_motion.csv",output)
    records=[]
    for f in sorted(a.root.glob("l0-*/run.json")):
        d=read(f)
        for r in d["points"]:
            for frequency,stats in r["per_freq"].items():
                records.append(dict(run=f.parent.name,sample=r["sample"],frequency=frequency,**stats))
    write(a.results/"l0_per_frequency.csv",records)


if __name__=="__main__":
    main()
