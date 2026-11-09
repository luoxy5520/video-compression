"""Create labeled H.264 previews of the measured student and offline diagnostic."""
import argparse
import importlib.util
import json
from pathlib import Path

from train_capacity import ON,p

path=Path(__file__).resolve().parents[2]/"P2_LabelProbe/src/previews.py"
spec=importlib.util.spec_from_file_location("label_previews",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("root",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    args.out.mkdir(exist_ok=False,parents=True)
    choice=json.loads((args.root/"c1_gate.json").read_text())["chosen"]
    path=args.root/f"replay-c0-{choice}-1701"
    pts=json.loads((path/"run.json").read_text())["points"]
    rows={r["sample_id"]:r for r in p.rows_by_split()["val"]}
    for label,fn in (("best_or_closest",max),("worst",min)):
        sid=fn(pts,key=lambda r:r["delta_db"])["sample"]
        anchor,src=p.paths("val",rows[sid])
        module.render(args.out/f"{label}_{sid}.mp4",
            (src,anchor/"A/prediction.264",path/sid/"prediction.264",ON/"o0"/sid/"T0/prediction.264"),
            ("Source","Frozen A",f"C0-{choice} student, seed 1701","O0 source-assisted diagnostic"))


if __name__=="__main__":
    main()
