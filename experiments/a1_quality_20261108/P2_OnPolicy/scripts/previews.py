"""Render aligned validation examples without changing the scored bitstreams."""
import argparse
import importlib.util
import json
from pathlib import Path

from o0 import p

previous=Path(__file__).resolve().parents[2]/"P2_LabelProbe/src/previews.py"
spec=importlib.util.spec_from_file_location("label_previews",previous)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("out",type=Path)
    a=cli.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    rows={r["sample_id"]:r for r in p.rows_by_split()["val"]}
    seed=1701
    doc=json.loads((a.root/f"o1-T1-{seed}/run.json").read_text())
    pts=doc["points"]
    best=max(pts,key=lambda r:r["delta_db"])
    worst=min(pts,key=lambda r:r["delta_db"])
    used={best["sample"],worst["sample"]}
    motion=next(r for r in pts if r["sample"] not in used and ("skater" in r["prompt"] or "pan" in r["prompt"]))
    for name,point in (("best",best),("worst",worst),("motion",motion)):
        sid=point["sample"]
        root,src=p.paths("val",rows[sid])
        streams=[a.root/f"o1-{arm}-{seed}"/sid/"prediction.264" for arm in ("T0","T1")]
        module.render(a.out/f"{name}_{sid}.mp4",(src,root/"A/prediction.264",*streams),
                      ("Source","Frozen A","T0 student, seed 1701","T1 student, seed 1701"))


if __name__=="__main__":
    main()
