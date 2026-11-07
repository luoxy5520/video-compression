"""Hold P-mode/MV fixed and recompute the residual inside native JM."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_state_20261107"))
from benchmark import FPS, FRAMES, decode_check, load_backend, psnr
from train_state import load_config


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "kit", "jm", "data", "previous", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    config, _, _ = load_config(args.config)
    backend = load_backend(args.kit)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for sid in config["splits"]["test"]:
        folder = args.output / sid
        folder.mkdir()
        raw = args.data / sid / "final_raw.yuv"
        state = args.previous / sid / "prediction_020/predicted.bin"
        bitstream = folder / "live_residual.264"
        settings = dict(InputFile=raw, OutputFile=bitstream, ReconFile=folder/"encoder_recon.yuv",
                        SourceWidth=832, SourceHeight=480, FramesToBeEncoded=17, FrameRate=16,
                        QPISlice=28, QPPSlice=28, RDOptimization=0, NumberReferenceFrames=1,
                        PList0References=1)
        command = [args.jm/"bin/lencod.exe", "-d", args.kit/"JM-backend/cfg/encoder_baseline.cfg",
                   "-f", args.kit/"linux/skip_p16.cfg"]
        for key, value in settings.items():
            command += ["-p", f"{key} = {value}"]
        env = os.environ.copy()
        env.pop("JM_LATENT_MV_PRIOR", None)
        env.update(JM_STATE_MODE="replay", JM_STATE_FILE=str(state), JM_LIVE_RESIDUAL="1")
        backend.run(command, folder, folder/"encode.log", env)
        decoded, _ = decode_check(backend, folder, bitstream)
        metric = dict(sample_id=sid, decoder_agreement=True,
                      bitrate_kbps=bitstream.stat().st_size * 8 * FPS / FRAMES / 1000,
                      **psnr(raw, decoded))
        results.append(metric)
        print(json.dumps(metric), flush=True)
    (args.output/"results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
