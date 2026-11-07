"""Replay step-20 predictions with selected Oracle P fields for diagnosis."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmark import FPS, FRAME_BYTES, FRAMES, decode_check, load_backend, psnr
from train_state import load_config


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "kit", "data", "benchmark", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config, dataset, index = load_config(args.config)
    backend = load_backend(args.kit)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    variants = {
        "oracle_mode": ("mode_id",),
        "only_residual_predicted": ("mode_id", "mv_qpel"),
        "only_mv_predicted": ("mode_id", "residual_y", "residual_u", "residual_v"),
    }
    for sid in config["splits"]["test"]:
        row = index[sid, 20]
        target = dataset / row["target_label_path"]
        prediction = args.benchmark / sid / "prediction_020/prediction.npz"
        reference = args.data / sid / "final_raw.yuv"
        with np.load(target, allow_pickle=False) as oracle, np.load(prediction, allow_pickle=False) as predicted:
            for variant, oracle_keys in variants.items():
                folder = args.output / sid / variant
                folder.mkdir(parents=True)
                npz = folder / "mixed.npz"
                np.savez(npz, **{key: (oracle[key] if key in oracle_keys else predicted[key]) for key in predicted.files})
                state = folder / "mixed.bin"
                backend.run([sys.executable, backend.ROOT / "scripts/prediction_to_b1_state.py",
                             "--prediction", npz, "--oracle-template", row["target_oracle_state_path"],
                             "--output-state", state, "--result-json", folder / "adapter.json"],
                            folder, folder / "adapter.log")
                dummy = folder / "dummy.yuv"
                with dummy.open("wb") as handle:
                    handle.truncate(FRAMES * FRAME_BYTES)
                bitstream = folder / "mixed.264"
                backend.encoder(folder, "replay", state, dummy, bitstream, folder / "encoder_recon.yuv")
                decoded, _ = decode_check(backend, folder, bitstream)
                metrics = dict(sample_id=sid, variant=variant, decoder_agreement=True,
                               bitrate_kbps=bitstream.stat().st_size * 8 * FPS / FRAMES / 1000,
                               **psnr(reference, decoded))
                results.append(metrics)
                print(json.dumps(metrics), flush=True)
    means = {
        variant: {key: float(np.mean([r[key] for r in results if r["variant"] == variant]))
                  for key in ("bitrate_kbps", "psnr_y_p_db", "psnr_yuv_p_db")}
        for variant in variants
    }
    (args.output / "results.json").write_text(json.dumps({
        "means": means, "per_video": results,
        "note": "Diagnostic hybrid states only; residual predictions were produced with predicted MV conditioning. Oracle residual targets were derived under Oracle motion and reconstruction, so hybrid effects are not additive.",
    }, indent=2))


if __name__ == "__main__":
    main()
