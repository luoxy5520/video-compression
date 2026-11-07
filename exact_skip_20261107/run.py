"""Evaluate the exact Skip conversion on identical prepared latents."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"latent_codec_20261107"))
from run import load_backend, run_case, summary


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"samples").mkdir()
    previous = json.loads((a.previous/"results.json").read_text())
    reference = {(r["split"],r["sample_id"],r["variant"]):r for r in previous["per_video"]}
    tags = ("refine","skip25")
    items = [key for key in reference if key[2] in tags]
    for sid in sorted({item[1] for item in items}):
        folder = a.output/"samples"/sid
        folder.mkdir()
        for name in ("proxy.yuv","shortlist.bin","predicted_p.bin","prepared.json"):
            (folder/name).symlink_to(a.previous/"samples"/sid/name)
    os.environ["JM_REPLAY_EXACT_SKIP"] = "1"
    backend = load_backend(a.kit)
    def evaluate(item):
        result = run_case(a,backend,item)
        split,sid,tag = item
        old = reference[item]
        folder = a.output/"samples"/sid/tag
        before = a.previous/"samples"/sid/tag/"decoded_ffmpeg.yuv"
        if backend.sha(before)!=backend.sha(folder/"decoded_ffmpeg.yuv"):
            raise ValueError(f"Exact Skip changed reconstruction: {sid}, {tag}")
        match = re.search(r"B1 exact Skip: converted=(\d+)",(folder/"encode.log").read_text())
        if not match:
            raise ValueError(f"Missing exact Skip count: {sid}, {tag}")
        result.update(converted=int(match[1]),old_bitrate_kbps=old["bitrate_kbps"],
                      identical_reconstruction=True)
        return result
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        results = list(executor.map(evaluate,items))
    (a.output/"results.json").write_text(json.dumps(dict(
        means={s:summary([r for r in results if r["split"]==s]) for s in ("val","test")},
        per_video=results),indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    for key in ("previous","kit","jm","data","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    main(parser.parse_args())
