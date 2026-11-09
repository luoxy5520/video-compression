"""Build tables, plot and provenance from the immutable run outputs."""
import platform
import shutil
from statistics import mean

from common import *

REPO = "https://github.com/luoxy5520/video-compression"
ART = REPO+"/tree/artifacts/p2-generalizationprobe-20261109"

def pct(v):
    return f"{100*v:.3f}%" if v is not None else "—"

def display(v):
    return f"{v:.3f}" if v is not None else "—"

def main():
    data = read(HERE/"metrics/summary.json")
    folds = read(OUT/"split_manifest.json")["folds"]
    for name in ("pre_history.json","split_manifest.json","gz2_labels.json"):
        shutil.copy2(OUT/name, HERE/("metrics" if name!="split_manifest.json" else "")/name)
    header = "| 折 | 头 | 种子 | 组 | n | 写入召回 | Precision | AP | 先验召回 / AP | 随机召回 / AP | 召回 95% CI |"
    rows = [header,"|"+"---|"*11]
    for run in data["gz0"]:
        for group,v in run["groups"].items():
            rows.append(f"| {run['fold']} | {run['arm']} | {run['seed']} | {group} | {v['n']} | {pct(v['recall'])} | "
                        f"{pct(v['precision'])} | {pct(v['ap'])} | {pct(v['prior_recall'])} / {pct(v['prior_ap'])} | "
                        f"{pct(v['random_recall'])} / {pct(v['random_ap'])} | {' – '.join(map(pct,v['recall_ci']))} |")
    (HERE/"tables/gz0.md").write_text("# GZ0｜逐视频等权\n\n"+"\n".join(rows)+"\n")
    rows = ["| 折 | 组 | 臂 | kNN AP / Recall | 线性 AP / Recall | 先验 AP / Recall | Top-1 正类召回 / 随机 | 覆盖 |",
            "|"+"---|"*8]
    for r in data["gz1"]["records"]:
        for group,v in r["groups"].items():
            pairs = " | ".join(" / ".join(pct(v[m][k]) for k in ("ap","recall")) for m in ("knn","linear","prior"))
            rows.append(f"| {r['fold']} | {group} | {r['feature']} | {pairs} | "
                        f"{pct(v['knn_pair']['pos_recall'])} / {pct(v['random_pair']['pos_recall'])} | {pct(v['coverage'])} |")
    rows.extend(["", "## 原生宽度敏感性", "", rows[0], rows[1]])
    for r in data["gz1_native"]["records"]:
        for group,v in r["groups"].items():
            pairs = " | ".join(" / ".join(pct(v[m][k]) for k in ("ap","recall")) for m in ("knn","linear","prior"))
            rows.append(f"| {r['fold']} | {group} | {r['feature']} | {pairs} | "
                        f"{pct(v['knn_pair']['pos_recall'])} / {pct(v['random_pair']['pos_recall'])} | {pct(v['coverage'])} |")
    (HERE/"tables/gz1.md").write_text("# GZ1｜匹配负例采样；AP 仅在同一采样协议内可比\n\n"+"\n".join(rows)+"\n")
    rows = ["| 折 | 组 | 臂 | 方法 | Z1 AP / Recall | Z2 Spearman / Pearson | Z2 MAE | Top 10% 区域召回 | Z3 MAE 0/1/4 | Z3 符号 / 非零召回 |",
            "|"+"---|"*10]
    for r in data["gz2"]["records"]:
        for group,v in r["groups"].items():
            for method in ("linear","knn","prior","random","zero"):
                z = v["continuous"][method]
                sparse = " / ".join(pct(v[method][k]) for k in ("ap","recall")) if method in v and "ap" in v[method] else "—"
                rows.append(f"| {r['fold']} | {group} | {r['feature']} | {method} | {sparse} | "
                    f"{display(z['spearman'])} / {display(z['pearson'])} | {display(z['energy_mae'])} | {pct(z['top_q_recall'])} | "
                    f"{' / '.join(map(display,z['low_mae_by_frequency']))} | "
                    f"{pct(z['sign_outside_near_zero'])} / {pct(z['quantized_nonzero_recall'])} |")
    (HERE/"tables/gz2.md").write_text("# GZ2｜冻结 A 参考的同折诊断\n\n"+"\n".join(rows)+"\n")

    import xml.etree.ElementTree as ET
    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="1050", height="330",
                     viewBox="0 0 1050 330", role="img")
    ET.SubElement(svg,"title").text = "Cross-video residual diagnosis"
    def element(tag,**attrs):
        return ET.SubElement(svg,tag,{k.replace("_","-"):str(v) for k,v in attrs.items()})
    def label(x,y,value,size=12,anchor="start"):
        element("text",x=x,y=y,font_family="system-ui, sans-serif",font_size=size,fill="#192532",
                text_anchor=anchor).text = value
    element("rect",width=1050,height=330,fill="white")
    panels = []
    for group,title,top in (("A","Held-in position recall",100),("B","Same prompt, new seed",0.5)):
        series = {arm:[r["groups"][group]["recall"]*100 for r in data["gz0"] if r["arm"]==arm]
                  for arm in ("A","C1")}
        panels.append((title,top,series,["0/1701","0/1702","1/1701","1/1702"]))
    series = {arm:[v["continuous"]["linear"]["top_q_recall"]*100
        for r in data["gz2"]["records"] if r["feature"]==arm for v in r["groups"].values()]
        for arm in ("F0","F1","F2")}
    series["Chance"] = [10]*4
    panels.append(("Top-decile energy region recall",50,series,["0/B","0/D","1/B","1/D"]))
    for i,(title,top,series,xlabels) in enumerate(panels):
        left = 48+i*350
        label(left,32,title,16)
        x = [left+36+j*73 for j in range(4)]
        for tick in range(6):
            y = 245-37*tick
            element("line",x1=left,y1=y,x2=left+285,y2=y,stroke="#d8e2eb")
            label(left-7,y+4,f"{top*tick/5:g}%",10,"end")
        for j,name in enumerate(xlabels):
            label(x[j],268,name,11,"middle")
        for j,(name,values) in enumerate(series.items()):
            color = ("#176cb0","#ad4a16","#007b63","#687583")[j]
            points = [(a,245-185*b/top) for a,b in zip(x,values,strict=True)]
            element("polyline",points=" ".join(f"{a},{b:.2f}" for a,b in points),
                    fill="none",stroke=color,stroke_width=2)
            for a,b in points:
                element("circle",cx=a,cy=f"{b:.2f}",r=4,fill=color)
            element("circle",cx=left+j*69,cy=294,r=4,fill=color)
            label(left+8+j*69,298,name)
    ET.ElementTree(svg).write(HERE/"figures/generalization.svg",encoding="unicode")

    path = str(HERE.relative_to(HERE.parents[2]))
    python = "~/Wan2.1/.venv/bin/python"
    pre = read(OUT/"pre_history.json")
    commands = []
    def run(name,commit,cmd,output,permissions,**extra):
        commands.append(dict(name=name,code_sha=commit,command=cmd,output=output,permissions=permissions,status="complete",**extra))
    def sha(name):
        return (OUT/name).read_text().strip()
    run("PRE-A","f29a63f5bc6d16940ffa8e3a0f18643e0e4360e6",
        f"{python} {path.replace('P2_GeneralizationProbe','P2_InfoProbe')}/scripts/audit.py {OUT}/pre_a",
        str(OUT/"pre_a"),"16 historical dev scoring sources and first train clip for label/adapter smoke")
    run("PRE-identity",sha("pre-code"),f"{python} {path}/scripts/pre.py",str(OUT/"pre_history.json"),
        "44 train identity and cache hashes; read-only 16 historical dev for PRE checkpoint verification")
    for seed in CFG["seeds"]:
        run("GZ0",sha("gz0-code"),f"CUDA_VISIBLE_DEVICES={seed-1701} {python} {path}/scripts/gz0.py {seed}",
            str(OUT/f"gz0-*-*-{seed}"),"F0 plus fixed Mode/MV as inputs; T0 offline supervision and evaluation only",
            seed=seed,folds=folds,epochs=400,arms={"A":106784,"C1":1713824},label="T0")
    run("GZ2-offline-labels",sha("labels-code"),f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {python} {path}/scripts/labels.py",
        str(OUT/"continuous"),"44 train sources for offline frozen-A supervision only",label="T1")
    for stage in ("gz1","gz1_native","gz2"):
        run(stage,sha("native-code" if stage=="gz1_native" else "readers-code"),f"OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 {python} {path}/scripts/readers.py {stage}",
            str(OUT/f"{stage}.json"),"44 train-pool caches only at readout; support-only fit; source-assisted labels used as targets",
            folds=folds,seed=61109,label="T1" if stage=="gz2" else "T0")
    commands.append(dict(name="New JM student replay",status="not_run",reason="No independent student passed locked gate"))
    p.save_json(HERE/"runs_manifest.json",dict(repo=str(HERE.parents[2]),python=platform.python_version(),
        environment={k:pre[k] for k in ("torch","numpy","cuda","gpu")},kernel=platform.release(),
        input_hashes="split_manifest.json",continuous_target_hashes="metrics/gz2_labels.json",
        replay_hashes="metrics/pre_history.json",configs=["configs/locked.json","configs/readers.json"],runs=commands))

if __name__=="__main__":
    main()
