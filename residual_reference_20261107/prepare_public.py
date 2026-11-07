"""Pair DAVIS frames with final Wan latents and their VAE reconstruction."""
import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

W, H, T = 832, 480, 17


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=90)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, "/home/luoxiangyu/Wan2.1")
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth="/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", device="cuda")
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(a.archive) as archive, torch.inference_mode():
        splits = {}
        for split in ("train", "val"):
            name = next(n for n in archive.namelist() if n.endswith(f"ImageSets/2017/{split}.txt"))
            splits[split] = archive.read(name).decode().split()
        sequences = {}
        for name in archive.namelist():
            parts = name.split("/")
            if "JPEGImages" in parts and "480p" in parts and name.endswith(".jpg"):
                sequences.setdefault(parts[-2], []).append(name)
        for split, names in splits.items():
            for seq in names:
                if len(rows) >= a.limit:
                    break
                frames = sorted(sequences.get(seq, []))
                if len(frames) < T:
                    continue
                start = (len(frames) - T) // 2
                chosen = frames[start:start+T]
                folder = a.output / seq
                folder.mkdir(exist_ok=True)
                latent_file = folder / "final_latent.pt"
                raw_file = folder / "final_raw.yuv"
                if not (latent_file.exists() and raw_file.exists() and raw_file.stat().st_size == W*H*T*3//2):
                    images = []
                    for name in chosen:
                        with Image.open(io.BytesIO(archive.read(name))) as im:
                            images.append(np.asarray(ImageOps.fit(im.convert("RGB"), (W,H), method=Image.Resampling.BICUBIC)))
                    rgb = torch.from_numpy(np.stack(images)).permute(3,0,1,2).contiguous().cuda().float()/127.5-1
                    latent = vae.encode([rgb])[0]
                    if tuple(latent.shape) != (16,5,60,104) or not torch.isfinite(latent).all():
                        raise ValueError(f"Invalid latent for {seq}")
                    raw = ((vae.decode([latent])[0]+1)*127.5).round().byte().permute(1,2,3,0).contiguous().cpu().numpy()
                    temp = folder / "raw.partial"
                    subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}",
                                    "-r","16","-i","pipe:0","-vf","scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                                    "-pix_fmt","yuv420p","-f","rawvideo",str(temp)],input=raw.tobytes(),check=True)
                    if temp.stat().st_size != W*H*T*3//2:
                        raise ValueError(f"Wrong output size for {seq}")
                    torch.save(latent.cpu(), latent_file)
                    temp.replace(raw_file)
                rows.append(dict(sample_id=f"davis_{seq}", split=split, sequence=seq,
                                 frames=chosen, latent_path=str(latent_file), raw_path=str(raw_file)))
                print(json.dumps(dict(sequence=seq,split=split,complete=len(rows))),flush=True)
    (a.output/"manifest.json").write_text(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
