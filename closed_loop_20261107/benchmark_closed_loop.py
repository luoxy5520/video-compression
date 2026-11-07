"""Measure final-latent MV hints with native JM closed-loop reconstruction."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train_motion import MotionHead
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_state_20261107"))
from benchmark import conventional, load_backend
from train_state import load_config, load_latent, networks


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "kit", "jm", "mode", "regressor", "data", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--extra-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    config, _, index = load_config(args.config)
    backend = load_backend(args.kit)
    _, old = networks(args.mode, "cuda")
    old.load_state_dict(torch.load(args.regressor, weights_only=True))
    old.eval()
    new = MotionHead().cuda().eval()
    if not args.smoke:
        new.load_state_dict(torch.load(args.motion, weights_only=True))
    sys.path.insert(0, "/home/luoxiangyu/Wan2.1")
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth="/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", device="cuda")
    warm = load_latent(index[config["splits"]["val"][0], 20], "cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vae.decode([warm[0]])
    with torch.inference_mode():
        old(warm, torch.ones(1, device="cuda"))
        new(warm)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=False)
    ids = config["splits"]["val"][:1] if args.smoke else config["splits"]["test"]
    cases = [(4,"native"), (4,"prior")] if args.smoke else [
        (radius, head) for radius in (4, 8) for head in ("native", "prior", "correlation")
    ]
    if args.extra_only:
        cases = [(4, "extra"), (8, "extra")]
    records = []
    for sid in ids:
        row = index[sid, 20]
        reference = args.data / sid / "final_raw.yuv"
        for radius, head in cases:
            folder = args.output / sid / f"{head}_{radius}"
            folder.mkdir(parents=True)
            head_seconds = 0.0

            def encode(work, _mode, _state, raw, bitstream, recon):
                nonlocal head_seconds
                env = os.environ.copy()
                env["JM_STATE_MODE"] = "off"
                env.pop("JM_LATENT_MV_PRIOR", None)
                env.pop("JM_LATENT_MV_EXTRA", None)
                if head == "extra":
                    env["JM_LATENT_MV_EXTRA"] = "1"
                if head != "native":
                    start = time.perf_counter()
                    latent = load_latent(row, "cuda")
                    with torch.inference_mode():
                        motion = old(latent, torch.ones(1, device="cuda"))[0] if head == "prior" else new(latent)
                        motion = (motion[0]*16).round().clamp(-32768,32767).cpu().numpy()
                    if not np.isfinite(motion).all():
                        raise ValueError("Nonfinite motion")
                    prior = folder / "latent_prior.mv"
                    motion.astype("<i2").tofile(prior)
                    env["JM_LATENT_MV_PRIOR"] = str(prior)
                    head_seconds = time.perf_counter() - start
                settings = dict(InputFile=raw, OutputFile=bitstream, ReconFile=recon, SourceWidth=832,
                                SourceHeight=480, FramesToBeEncoded=17, FrameRate=16, QPISlice=28,
                                QPPSlice=28, RDOptimization=0, NumberReferenceFrames=1,
                                PList0References=1, SearchRange=radius, SearchMode=-1)
                command = [args.jm/"bin/lencod.exe", "-d", args.kit/"JM-backend/cfg/encoder_baseline.cfg",
                           "-f", args.kit/"linux/skip_p16.cfg"]
                for key, value in settings.items():
                    command += ["-p", f"{key} = {value}"]
                backend.run(command, work, work/"encode.log", env)

            backend.encoder = encode
            result = conventional(backend, folder, row, reference, vae)
            if not result["raw_bit_exact_to_reference"]:
                raise ValueError("VAE raw differs from label source")
            log = (folder/"encode.log").read_text()
            guided = re.search(r"Latent-guided P16 search calls: (\d+)", log)
            count = int(guided[1]) if guided else 0
            if (head != "native") != (count > 0):
                raise ValueError("Unexpected learned center usage")
            if radius == 32 and not result["bitstream_matches_oracle_encoder"]:
                raise ValueError("Disabled patch changed original bitstream")
            accepted = re.search(r"Latent MV extra candidate accepted: (\d+)", log)
            result["extra_candidates_accepted"] = int(accepted[1]) if accepted else 0
            result["jm_encode_s"] -= head_seconds
            result.update(sample_id=sid, radius=radius, head=head,
                          proposal_s=head_seconds, guided_search_calls=count)
            records.append(result)
            (folder/"metrics.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    means = {}
    for radius, head in cases:
        selected = [r for r in records if (r["radius"],r["head"]) == (radius,head)]
        keys = ("bitrate_kbps", "psnr_y_p_db", "psnr_yuv_p_db", "latent_to_bitstream_s",
                "latent_to_decoded_s", "jm_encode_s", "proposal_s")
        means[f"{head}_{radius}"] = {key:float(np.mean([r[key] for r in selected])) for key in keys}
    (args.output/"results.json").write_text(json.dumps(dict(means=means,per_video=records),indent=2))


if __name__ == "__main__":
    main()
