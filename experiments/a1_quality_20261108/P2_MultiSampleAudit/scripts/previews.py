"""Preview measured replay examples, including a separate offline oracle."""
import argparse
import importlib.util
import json
from pathlib import Path

from m1 import ON,p

path=Path(__file__).resolve().parents[2]/"P2_LabelProbe/src/previews.py"
spec=importlib.util.spec_from_file_location("label_previews",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("root",type=Path)
    a.add_argument("out",type=Path)
    v=a.parse_args()
    v.out.mkdir(parents=True,exist_ok=False)
    rows={r["sample_id"]:r for r in p.rows_by_split()["val"]}
    choice=json.loads((v.root/"replay_selection.json").read_text())["selected_m2_policy"]
    file,=v.root.glob(f"replay-m2-{choice}-*-1701-400/run.json")
    points=json.loads(file.read_text())["points"]
    for label,fun in (("closest_or_best",max),("worst",min)):
        r=fun(points,key=lambda r:r["delta_db"])
        sid=r["sample"]
        root,src=p.paths("val",rows[sid])
        module.render(v.out/f"{label}_{sid}.mp4",
            (src,root/"A/prediction.264",file.parent/sid/"prediction.264",ON/"o0"/sid/"T0/prediction.264"),
            ("Source","Frozen A",f"M2 {choice}, seed 1701","O0 source-assisted diagnostic"))


if __name__=="__main__":
    main()
