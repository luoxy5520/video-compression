"""Validate the delivered 80×4 archive, including all NPZ fields and hashes."""
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

BASE=(16,30,52)
EXPECTED=dict(mode_id=(BASE,"u1"),mv_qpel=(BASE+(2,),"i2"),
    residual_y=(BASE+(16,16),"i2"),residual_u=(BASE+(8,8),"i2"),residual_v=(BASE+(8,8),"i2"),
    confidence=(BASE+(3,),"f4"),p16_probability=(BASE,"f4"))
archive,output=map(Path,sys.argv[1:3])
with tarfile.open(archive,"r") as tar:
    keys={x.name for x in tar if x.isfile()}
    assert {"manifest.json","state.pt","README.md"}<=keys
    manifest=json.load(tar.extractfile("manifest.json"))
    assert manifest["prediction_count"]==len(manifest["records"])==320
    counts=Counter()
    steps=defaultdict(set)
    split_by_id={}
    for row in manifest["records"]:
        sid,step,split=row["sample_id"],row["step"],row["split"]
        assert step in (5,10,15,20) and step not in steps[sid]
        assert split_by_id.setdefault(sid,split)==split
        steps[sid].add(step)
        name=f"predictions/{split}/{sid}/step_{step:03d}/prediction.npz"
        assert row["prediction"]==name and name in keys
        with tar.extractfile(name) as f:
            data=f.read()
        assert hashlib.sha256(data).hexdigest()==row["sha256"],name
        with np.load(io.BytesIO(data),allow_pickle=False) as npz:
            assert set(npz.files)==set(EXPECTED),name
            for field,(shape,dtype) in EXPECTED.items():
                x=npz[field]
                assert x.shape==shape and x.dtype==np.dtype(dtype), (name,field)
                if field.startswith("residual"):
                    assert np.min(x)>=-255 and np.max(x)<=255
                elif field=="mode_id":
                    assert np.max(x)<=1
                elif x.dtype.kind=="f":
                    assert np.isfinite(x).all() and np.min(x)>=0 and np.max(x)<=1
        counts[split]+=1
    assert counts==dict(train=176,val=64,test=80)
    assert len(steps)==80 and all(len(v)==4 for v in steps.values())
    ckpt_hash=hashlib.sha256(tar.extractfile("state.pt").read()).hexdigest()
with archive.open("rb") as f:
    sha=hashlib.sha256()
    for data in iter(lambda:f.read(1<<20),b""):
        sha.update(data)
output.write_text(json.dumps(dict(predictions=320,videos=80,predictions_by_split=counts,
    archive_sha256=sha.hexdigest(),checkpoint_sha256=ckpt_hash,shapes_dtypes_and_hashes_passed=True),indent=2))
