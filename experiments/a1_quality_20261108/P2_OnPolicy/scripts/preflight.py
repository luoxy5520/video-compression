"""Verify the archived training inputs and predecessor metrics before O0."""
import argparse
import json
from pathlib import Path
import subprocess
from statistics import median

import numpy as np

from o0 import historical, p, sha_frames


def read(path):
    return json.loads(path.read_text())


def check_label_probe(parent, label):
    data=read(parent/"P2_LabelProbe/runs_manifest.json")
    for run,hashes in data["runs"].items():
        assert p.digest(label/run/"run.json")==hashes["metrics_sha256"]
        assert p.digest(label/run/"head.pt")==hashes["head_sha256"]
    assert p.digest(label/"l0_labels/manifest.json")==data["inputs_sha256"]["full_frequency_manifest"]
    assert p.digest(label/"l1/metrics.json")==data["inputs_sha256"]["l1_trajectory_manifest"]
    for entry in read(label/"l0_labels/manifest.json"):
        assert p.digest(label/"l0_labels"/f"{entry['sample']}.npz")==entry["label_sha256"]
    m=read(label/"l1/metrics.json")
    count=0
    for row in m:
        for arm,value in row["trajectories"].items():
            path=label/"l1"/row["sample"]
            assert p.digest(path/f"{arm}.npz")==value["label_sha256"]
            if "repeat_sha256" in value:
                assert p.digest(path/f"{arm}-repeat.npz")==value["repeat_sha256"]==value["label_sha256"]
                for tag,expected in ((arm,value["replay"]["stream_sha256"]),(arm+"-repeat",value["repeat_stream_sha256"])):
                    assert p.digest(path/tag/row["sample"]/"prediction.264")==expected
                count+=1
    assert count==6
    val=[r for r in m if r["split"]=="val"]
    iou={k:median(r["pairs"][k]["iou"] for r in val) for k in ("A-teacher","A-student")}
    assert round(iou["A-teacher"],3)==.128 and round(iou["A-student"],3)==.829
    return dict(heads_and_metrics=len(data["runs"]),label_repeat_pairs=count,reference_iou=iou)


def check_info_probe(parent,info):
    data=read(parent/"P2_InfoProbe/runs_manifest.json")
    assert p.digest(info/"d0/head.pt")==data["d0_original_sha256"]
    assert p.digest(info/"repair/head.pt")==data["d0_repair_sha256"]
    assert p.digest(info/"d1/d1.json")==data["d1_sha256"]
    for row in data["g_runs"]:
        run=info/row["run"]
        assert p.digest(run/"head.pt")==row["head_sha256"]
        assert p.digest(run/"metrics.json")==row["metrics_sha256"]
    for row in read(info/"g_labels/manifest.json"):
        assert p.digest(info/"g_labels"/f"{row['sample']}.npz")==row["labels_sha256"]
    return dict(g_runs=len(data["g_runs"]),d1_sha256=data["d1_sha256"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    r=args.out
    parent=Path(__file__).resolve().parents[2]
    rows=p.rows_by_split()
    hist=p.ROOT/"P2_Lite/run01"
    raw=read(hist/"label_manifest.json")
    assert len(raw)==60
    assert p.digest(p.CKPT)==p.CFG["checkpoint_sha256"]
    previous=read(hist/"input_provenance.json")
    for path,sha in previous["sha256"].items():
        assert p.digest(path)==sha
    preexisting={x["sample"]:x for x in raw}
    for split,group in rows.items():
        for row in group:
            sid=row["sample_id"]
            root,source,base,fixed,refs=historical(split,row)
            m=preexisting[sid]
            assert m["split"]==split
            assert p.digest(root/"fixed_mode_mv.npz")==m["fixed_state_sha256"]
            assert p.digest(hist/"labels"/f"{sid}.npz")==m["label_sha256"]
            assert refs["T0"][1]["stream_sha256"]==m["teacher_stream_sha256"]
    metrics=read(p.ROOT/"P0_6/report/summary.json")
    anchor=next(r for r in metrics["curves"] if r["arm"]=="A")
    teacher=next(r for r in metrics["curves"] if r["arm"]=="offline_gain_3")
    audit=read(r/"audit/audit.json")
    expected={"psnr_y_p_db":anchor["psnr_y_db"],"bitrate_kbps":anchor["bitrate_kbps"],"last4_db":anchor["last4_psnr_y_db"]}
    for k,v in expected.items():
        assert abs(audit["baseline"][k]-v)<1e-10
    evidence={}
    for row in rows["val"]:
        sid=row["sample_id"]
        replay=r/"audit/A"/sid
        paths=[replay/f for f in ("encoder_recon.yuv","decoded_jm.yuv","decoded_ffmpeg.yuv")]
        frames=[sha_frames(np.fromfile(p,np.uint8).reshape(17,-1)) for p in paths]
        assert frames[0]==frames[1]==frames[2]
        evidence[sid]=dict(frame_yuv_sha256=frames[0],stream_sha256=p.digest(replay/"prediction.264"))
    out=dict(label_probe=check_label_probe(parent,p.ROOT/"P2_LabelProbe"),
             info_probe=check_info_probe(parent,p.ROOT/"P2_InfoProbe"),baseline=audit["baseline"],
             p06_teacher=teacher,baseline_frame_evidence=evidence,verified_historical_p06_label_hashes=len(raw),
             frozen_checkpoint_sha256=p.digest(p.CKPT),torch=audit["torch"],cuda=audit["cuda"],
             gpu=audit["gpu"],numpy=audit["numpy"],feature_shape=audit["feature_shape"],
             source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    p.save_json(r/"preflight.json",out)
    print(out["baseline"],out["label_probe"],out["info_probe"],flush=True)


if __name__=="__main__":
    main()
