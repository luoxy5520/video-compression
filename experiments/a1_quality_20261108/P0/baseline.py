"""Reproduce the locked hidden model on its historical development regression."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "a_state_20261107"))
from evaluate import load_backend, replay, state_metrics
from state import input_tensor, load_model, predict, records, targets

LOCKED = "b713fef90db9205471c7b168e1715c70d8a4c87a0f4b342f866baf04e1cd4ca5"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "kit", "jm", "history", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.checkpoint) != LOCKED:
        raise ValueError("Wrong locked checkpoint")
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.checkpoint)
    assert model.kind == "hidden" and not model.recurrent
    data = records([args.config])
    rows = [row for row in data["test"] if row["step"] == 20]
    assert len(rows) == 12
    backend = load_backend(args.kit)
    warm = data["val"][0]
    predict(model, input_tensor(warm, model.kind), warm["step"])
    args.output.mkdir(parents=True)
    results = []
    for row in rows:
        folder = args.output / row["sample_id"]
        folder.mkdir()
        start = time.perf_counter()
        hidden = input_tensor(row, model.kind)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - start
        pred, timing = predict(model, hidden, row["step"])
        prediction = folder / "prediction.npz"
        start = time.perf_counter()
        np.savez_compressed(prediction, **pred)
        timing.update(load_latent_s=load_s, export_s=time.perf_counter() - start)
        before = digest(prediction)
        old = args.history / "test" / row["sample_id"] / "step_020"
        with np.load(old / prediction.name) as historic:
            assert set(pred) == set(historic.files)
            array_equal = {key: bool(np.array_equal(val, historic[key])) for key, val in pred.items()}
        assert all(array_equal[key] for key in pred if pred[key].dtype.kind in "iu")
        reference = Path(row["source_export_dir"]) / "input_yuv420p.yuv"
        quality = replay(backend, args.jm, folder, row, pred, reference, timing)
        assert before == digest(prediction)
        files = {name: digest(folder / name) for name in (
            "prediction.npz", "replay.bin", "prediction.264", "encoder_recon.yuv", "decoded_ffmpeg.yuv"
        )}
        historic_hash_equal = {name: files[name] == digest(old / name)
                               for name in files if (old / name).is_file()}
        history = json.loads((old / "metrics.json").read_text())["replay"]
        for key in ("psnr_y_p_db", "bitrate_kbps", "last_frame_psnr_y"):
            assert abs(quality[key] - history[key]) < 1e-6, key
        result = dict(sample_id=row["sample_id"], prompt=row["prompt"], seed=row.get("seed"),
                      split="regression_development", diagnostic_oracle=False, fixed_oracle_i=True,
                      checkpoint_sha256=LOCKED, array_equal=array_equal,
                      sha256=files, historic_hash_equal=historic_hash_equal,
                      state=state_metrics(pred, targets(row)), replay=quality)
        (folder / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        results.append(result)
        print(row["sample_id"], quality["psnr_y_p_db"], quality["bitrate_kbps"], flush=True)
    mean = {key: statistics.mean(row["replay"][key] for row in results)
            for key in ("psnr_y_p_db", "bitrate_kbps", "last_frame_psnr_y", "ssim_y_p",
                        "late_minus_early_psnr", "head_seconds", "export_s")}
    (args.output / "summary.json").write_text(json.dumps(dict(mean=mean, per_video=results), indent=2))
    print(mean, flush=True)


if __name__ == "__main__":
    main()
