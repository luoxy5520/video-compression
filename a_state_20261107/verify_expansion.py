"""Check complete coverage, exact step-20 identity and unchanged prompt split."""
import hashlib
import json
import sys
from pathlib import Path

import torch

from state import STEPS, records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


cfg, plan, out=map(Path,sys.argv[1:])
actual=records([cfg])
planned=json.loads(plan.read_text())
spec={x["sample_id"]:x for x in planned["samples"]}
raw=Path(planned["output"])
report={}
for split,rows in actual.items():
    assert len({r["prompt"] for r in rows})==dict(train=6,val=3,test=3)[split]
    assert len(rows)==sum(x["split"]==split for x in spec.values())*len(STEPS)
    for r in rows:
        item=spec[r["sample_id"]]
        assert (split,r["prompt"],r["seed"])==(item["split"],item["prompt"],item["seed"])
        assert Path(r["latent_path"]).exists() and Path(r["hidden_path"]).exists()
        if r["step"]==20:
            x=torch.load(r["latent_path"],weights_only=True,map_location="cpu")
            y=torch.load(r["final_latent_path"],weights_only=True,map_location="cpu")
            assert torch.equal(x,y),r["sample_id"]
            folder=raw/r["sample_id"]
            assert sha(folder/"final_raw.yuv")==sha(folder/"step_020/raw.yuv"),r["sample_id"]
    report[split]=dict(videos=len(rows)//4,prompts=len({r["prompt"] for r in rows}),records=len(rows))
out.write_text(json.dumps(dict(step20_bit_exact=True,splits=report),indent=2))
print(report,flush=True)
