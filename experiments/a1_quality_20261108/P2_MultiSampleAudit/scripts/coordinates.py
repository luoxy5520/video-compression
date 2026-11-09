"""Expand the audited sparse label files into the coordinate contract."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from m1 import ON,p

out=Path(sys.argv[1])
manifest={x["sample"]:x for x in json.loads((ON/"o0/metrics.json").read_text())}
with (out/"selected_coordinate_contract.csv").open("w") as f:
    w=csv.writer(f,lineterminator="\n")
    w.writerow("sample_id prompt_id seed video_id frame_idx dit_step reference_trajectory_id mb_x mb_y block4x4_x block4x4_y frequency_index is_skip is_p16 qpel_mv_x qpel_mv_y target_quant_level selected_oracle_label label_hash feature_hash".split())
    for split,rows in p.rows_by_split().items():
        for row in rows:
            sid=row["sample_id"]
            root,_=p.paths(split,row)
            base=p.p05.load_npz(root/"A/prediction.npz")
            prompt=hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
            feature=p.digest(p.ROOT/"P2_Lite/run01/features"/f"{sid}.pt")
            for arm in ("T0","T1"):
                q=p.p05.load_npz(ON/"o0"/sid/f"{arm}.npz")["q"]
                item=manifest[sid]["arms"][arm]
                for t,y,x,b,k in np.argwhere(q!=0):
                    mv=base["mv_qpel"][t,y,x]
                    assert base["mode_id"][t,y,x]==1
                    w.writerow((sid,prompt,row["seed"],sid,t+1,row["step"],item["source_reference_stream_sha256"],x,y,b%4,b//4,k,
                              0,1,*mv,q[t,y,x,b,k],1,item["label_sha256"],feature))
