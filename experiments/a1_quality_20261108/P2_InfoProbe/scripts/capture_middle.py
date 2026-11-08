"""Recreate the original step-20 conditioning and verify last-block equality."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from d0 import p

WAN = Path.home() / "Wan2.1"
sys.path[:0] = [str(WAN), str(WAN / "experiments/baseline17")]

from run_remote_baseline17 import sdpa
import wan.modules.model as model
from wan.configs import WAN_CONFIGS
from wan.text2video import WanT2V

model.flash_attention = sdpa


def subset(rows):
    unique = {}
    for row in rows:
        unique.setdefault(row["prompt"], row)
    return list(unique.values())[:4]


class Captured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--part", type=int, choices=(0, 1), required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = "/mnt/data_hdd/models/Wan2.1-T2V-1.3B"
    pipe = WanT2V(WAN_CONFIGS["t2v-1.3B"], checkpoint, t5_cpu=False)
    state = dict(call=0)

    def middle(_, args, output):
        if state["call"] == 38:
            state["middle"] = output.detach().cpu().clone()

    def last(_, args, output):
        if state["call"] == 38:
            state["last"] = output.detach().cpu().clone()
            raise Captured
        state["call"] += 1

    h1 = pipe.model.blocks[14].register_forward_hook(middle)
    h2 = pipe.model.blocks[-1].register_forward_hook(last)
    rows = p.rows_by_split()
    manifest = []
    for row in (subset(rows["train"]) + subset(rows["val"]))[a.part::2]:
        sid = row["sample_id"]
        start = time.monotonic()
        state.clear()
        state["call"] = 0
        try:
            pipe.generate(row["prompt"], size=(832,480), frame_num=17, shift=8,
                          sample_solver="unipc", sampling_steps=20, guide_scale=6,
                          seed=int(row["seed"]), offload_model=True)
        except Captured:
            pass
        cached = torch.load(row["hidden_path"], weights_only=True)
        assert cached.shape == state["middle"].shape == state["last"].shape == (1,7800,1536)
        torch.testing.assert_close(state["last"], cached, rtol=0, atol=0)
        path = a.out / f"{sid}.pt"
        torch.save(state["middle"], path)
        item = dict(sample=sid, block=14, step=20, branch="conditional",
                    last_matches_exact=True, sha256=p.digest(path), seconds=time.monotonic()-start)
        manifest.append(item)
        print(json.dumps(item), flush=True)
    h1.remove()
    h2.remove()
    p.save_json(a.out / f"manifest-{a.part}.json", manifest)


if __name__ == "__main__":
    main()
