"""Calibrate decoder-consistent Skip thresholds, then run one held-out policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"direct_replay_20261107"))
from run import encode, load_backend, load_config


def means(records):
    keys = ("psnr_y_p_db", "psnr_yuv_p_db", "bitrate_kbps", "cached_state_to_bitstream_s",
            "latent_to_bitstream_s", "skip_fraction")
    return {key:float(np.mean([r[key] for r in records])) for key in keys}


def main(a):
    config, _, index = load_config(a.config)
    backend = load_backend(a.kit)
    a.sync_mv = True
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/"dummy.yuv").open("wb") as output:
        output.truncate(17*832*480*3//2)
    previous = json.loads((a.previous/"results.json").read_text())
    before = {(r["sample_id"],r["variant"]):r for r in previous["per_video"]}
    native = json.loads(a.native_validation.read_text())
    native = {r["sample_id"]:r for r in native["per_video"]
              if r["variant"]=="native8" and r["split"]=="val"}
    if set(native) != set(config["splits"]["val"]):
        raise ValueError("Native validation split differs")
    arrays = {}
    for split in ("val","test"):
        for sid in config["splits"][split]:
            name = "mode_mv" if split=="test" else "teacher_proxy"
            with np.load(a.previous/"samples"/sid/name/"prediction.npz", allow_pickle=False) as archive:
                arrays[sid] = dict(archive)
    def evaluate(item):
        split,sid,limit = item
        state = arrays[sid]
        if limit != "fixed":
            state = dict(state,mode_id=np.ones_like(state["mode_id"]))
        env = {}
        tag = f"mse_{limit}" if isinstance(limit,int) else limit
        if isinstance(limit,int):
            env["JM_SKIP_GATE_MSE"] = str(limit)
            if a.trace:
                env["JM_SKIP_GATE_TRACE"] = str(a.output/"samples"/sid/tag/"skip_trace.bin")
        with np.load(a.decisions/sid/"features.npz", allow_pickle=False) as cache:
            prep_s = float(cache["vae_s"]+cache["proposal_s"])
        result,folder = encode(a,backend,index[sid,20],sid,tag,state,
                               proxy=a.decisions/sid/"proxy.yuv",
                               decision_s=prep_s,extra_env=env)
        if isinstance(limit,int):
            count = re.search(r"B1 Skip gate: accepted=(\d+) total=(\d+)",
                              (folder/"encode.log").read_text())
            if not count or int(count[2]) != 24960:
                raise ValueError(f"Incomplete Skip trace: {sid} {tag}")
            skip = int(count[1])/int(count[2])
        else:
            skip = 1-float(state["mode_id"].mean())
        if split=="test":
            fixed = before[sid,"proxy_residual"]
            if result["i_frame_sha256"]!=fixed["i_frame_sha256"] or result["mv_sha256"]!=fixed["mv_sha256"]:
                raise ValueError(f"Unexpected I frame or MV input change: {sid}")
            if limit=="fixed" and result["bitstream_sha256"]!=fixed["bitstream_sha256"]:
                raise ValueError(f"Fixed control changed: {sid}")
        result.update(split=split,skip_fraction=skip)
        (folder/"metrics.json").write_text(json.dumps(result,indent=2))
        return result

    options = ["fixed","all_p16",*a.thresholds]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        validation = list(executor.map(evaluate, [( "val",sid,limit)
                           for limit in options for sid in config["splits"]["val"]]))
    native_mean = float(np.mean([r["psnr_y_p_db"] for r in native.values()]))
    by_limit = {limit:[r for r in validation if r["variant"]==f"mse_{limit}"] for limit in a.thresholds}
    eligible = []
    for limit, records in by_limit.items():
        quality = means(records)
        worst = max(native[r["sample_id"]]["psnr_y_p_db"]-r["psnr_y_p_db"] for r in records)
        if quality["psnr_y_p_db"]>=native_mean-.25 and worst<=.75:
            eligible.append((quality["bitrate_kbps"],limit))
    selected = min(eligible)[1] if eligible else "all_p16"
    calibration = dict(selected=selected,native8_mean_psnr=native_mean,
                       criteria=dict(mean_drop_db=.25,worst_drop_db=.75),
                       eligible_thresholds=[limit for _,limit in sorted(eligible)])
    (a.output/"calibration.json").write_text(json.dumps(calibration,indent=2))
    print(json.dumps(calibration), flush=True)
    test_options = list(dict.fromkeys(["fixed","all_p16",selected]))
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        test = list(executor.map(evaluate,[("test",sid,limit)
                    for limit in test_options for sid in config["splits"]["test"]]))
    records = validation+test
    summary = {split:{tag:means([r for r in records if r["variant"]==tag and r["split"]==split])
                      for tag in sorted({r["variant"] for r in records if r["split"]==split})}
               for split in ("val","test")}
    (a.output/"results.json").write_text(json.dumps(dict(calibration=calibration,means=summary,
                                                        per_video=records),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("config","previous","decisions","native-validation","kit","jm","data","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--thresholds",nargs="+",type=int,default=[1,2,4,6,9,16,25,36,49,64])
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--trace",action="store_true")
    main(parser.parse_args())
