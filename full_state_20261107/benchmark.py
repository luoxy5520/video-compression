"""Compare actual model replay and conventional JM on the held-out clips."""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train_state import load_config, load_latent, networks, read_target

W, H, FRAMES, FPS = 832, 480, 17, 16
FRAME_BYTES = W * H * 3 // 2
STEPS = (5, 10, 15, 20)


def load_backend(kit):
    spec = importlib.util.spec_from_file_location("oracle_backend", kit / "linux/run_pipeline.py")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def psnr(reference, reconstruction):
    expected = FRAMES * FRAME_BYTES
    x = np.fromfile(reference, dtype=np.uint8)
    y = np.fromfile(reconstruction, dtype=np.uint8)
    if x.size != expected or y.size != expected:
        raise ValueError(f"Unexpected YUV size: {x.size}, {y.size}")
    error = (x.astype(np.float32) - y.astype(np.float32)).reshape(FRAMES, FRAME_BYTES)[1:]
    def db(mse):
        return None if mse == 0 else 10 * math.log10(255**2 / mse)
    return {
        "psnr_y_p_db": db(np.mean(error[:, :W*H] ** 2, dtype=np.float64)),
        "psnr_yuv_p_db": db(np.mean(error ** 2, dtype=np.float64)),
    }


def decode_check(backend, folder, bitstream):
    ffmpeg_yuv = folder / "decoded_ffmpeg.yuv"
    jm_yuv = folder / "decoded_jm.yuv"
    start = time.perf_counter()
    backend.run(["ffmpeg", "-v", "error", "-xerror", "-y", "-i", bitstream,
                 "-pix_fmt", "yuv420p", "-f", "rawvideo", ffmpeg_yuv],
                folder, folder / "ffmpeg_decode.log")
    elapsed = time.perf_counter() - start
    backend.run([backend.DEC, "-d", backend.ROOT / "JM-backend/cfg/decoder.cfg",
                 "-p", f"InputFile = {bitstream}", "-p", f"OutputFile = {jm_yuv}",
                 "-p", 'RefFile = ""'], folder, folder / "jm_decode.log")
    if backend.sha(ffmpeg_yuv) != backend.sha(jm_yuv):
        raise ValueError(f"Independent decoders disagree: {folder}")
    return ffmpeg_yuv, elapsed


def collect_quality(backend, folder, bitstream, reference, timings):
    decoded, decode_s = decode_check(backend, folder, bitstream)
    bitrate = bitstream.stat().st_size * 8 * FPS / FRAMES / 1000
    return dict(
        bitstream_bytes=bitstream.stat().st_size,
        bitrate_kbps=bitrate,
        decoder_agreement=True,
        **psnr(reference, decoded),
        **timings,
        ffmpeg_decode_s=decode_s,
        latent_to_decoded_s=timings["latent_to_bitstream_s"] + decode_s,
    )


def conventional(backend, folder, row, reference, vae):
    start = time.perf_counter()
    latent = load_latent(row, "cuda")
    loaded = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        video = vae.decode([latent[0]])[0]
    rgb = ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0)
    rgb_bytes = rgb.contiguous().cpu().numpy().tobytes()
    raw = folder / "baseline_raw.yuv"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
                    "-vf", "scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                    "-pix_fmt", "yuv420p", "-f", "rawvideo", str(raw)], input=rgb_bytes, check=True)
    raw_ready = time.perf_counter()
    bitstream = folder / "conventional.264"
    backend.encoder(folder, "off", folder / "unused.bin", raw, bitstream, folder / "encoder_recon.yuv")
    encoded = time.perf_counter()
    result = collect_quality(backend, folder, bitstream, reference, {
        "load_latent_s": loaded-start, "vae_rgb_to_yuv_s": raw_ready-loaded,
        "jm_encode_s": encoded-raw_ready, "latent_to_bitstream_s": encoded-start,
    })
    result["raw_bit_exact_to_reference"] = backend.sha(raw) == backend.sha(reference)
    oracle_reference = Path(row["target_oracle_state_path"]).with_name("reference.264")
    result["bitstream_matches_oracle_encoder"] = backend.sha(bitstream) == backend.sha(oracle_reference)
    return result


def predict(backend, folder, row, step, reference, mode, regressor, target):
    start = time.perf_counter()
    latent = load_latent(row, "cuda")
    loaded = time.perf_counter()
    with torch.inference_mode():
        progress = torch.tensor([step/20], device="cuda")
        modes = mode(latent, progress)[0].argmax(dim=0).byte().cpu().numpy()
        motion, residual = regressor(latent, progress)
        motion = motion[0].mul(16).round().cpu().numpy()
        residual = residual[0].mul(16).round().clamp(-255, 255).cpu().numpy()
    if not np.isfinite(motion).all() or not np.isfinite(residual).all():
        raise ValueError("Non-finite prediction")
    inferred = time.perf_counter()
    prediction = folder / "prediction.npz"
    if np.max(np.abs(motion)) > 32767:
        raise ValueError("Predicted qpel displacement outside int16")
    np.savez(prediction, mode_id=modes, mv_qpel=motion.astype(np.int16),
             residual_y=residual[..., :256].astype(np.int16).reshape(16, 30, 52, 16, 16),
             residual_u=residual[..., 256:320].astype(np.int16).reshape(16, 30, 52, 8, 8),
             residual_v=residual[..., 320:].astype(np.int16).reshape(16, 30, 52, 8, 8))
    state = folder / "predicted.bin"
    backend.run([sys.executable, backend.ROOT / "scripts/prediction_to_b1_state.py",
                 "--prediction", prediction, "--oracle-template", row["target_oracle_state_path"],
                 "--output-state", state, "--result-json", folder / "adapter.json"],
                folder, folder / "adapter.log")
    adapted = time.perf_counter()
    dummy = folder / "dummy.yuv"
    with dummy.open("wb") as handle:
        handle.truncate(FRAMES * FRAME_BYTES)
    bitstream = folder / "prediction.264"
    backend.encoder(folder, "replay", state, dummy, bitstream, folder / "encoder_recon.yuv")
    encoded = time.perf_counter()
    result = collect_quality(backend, folder, bitstream, reference, {
        "load_latent_s": loaded-start, "head_infer_s": inferred-loaded,
        "serialize_and_adapter_s": adapted-inferred, "jm_replay_s": encoded-adapted,
        "latent_to_bitstream_s": encoded-start,
    })
    mask, mv, res = target
    if mask.any():
        selected = mask.numpy()
        delta = motion[selected] - mv.numpy()[selected] * 16
        result["oracle_p16_mv_epe_qpel"] = float(np.linalg.norm(delta, axis=-1).mean())
        result["oracle_p16_mv_exact"] = float(np.all(delta == 0, axis=-1).mean())
        result["oracle_p16_residual_mae"] = float(np.abs(residual[selected] - res.numpy()[selected]).mean())
    result["predicted_p16_fraction"] = float(modes.mean())
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ("config", "mode", "regressor", "kit", "data", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--wan", type=Path, default=Path("/home/luoxiangyu/Wan2.1"))
    p.add_argument("--vae", type=Path, default=Path("/mnt/data_hdd/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    args = p.parse_args()
    torch.set_num_threads(4)
    config, dataset, index = load_config(args.config)
    backend = load_backend(args.kit)
    mode, regressor = networks(args.mode, "cuda")
    regressor.load_state_dict(torch.load(args.regressor, weights_only=True))
    regressor.eval()
    sys.path.insert(0, str(args.wan))
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth=str(args.vae), device="cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    # Warm CUDA kernels with validation input. No validation bitstreams enter the timing aggregates.
    warm = load_latent(index[config["splits"]["val"][0], 20], "cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vae.decode([warm[0]])
    with torch.inference_mode():
        progress = torch.tensor([1.], device="cuda")
        mode(warm, progress)
        regressor(warm, progress)
    torch.cuda.synchronize()
    del warm

    results = []
    for sid in config["splits"]["test"]:
        reference = args.data / sid / "final_raw.yuv"
        row = index[sid, 20]
        target = read_target(dataset / row["target_label_path"])
        cases = [("conventional", 20)] + [("prediction", step) for step in STEPS]
        for method, step in cases:
            folder = args.output / sid / f"{method}_{step:03d}"
            folder.mkdir(parents=True)
            if method == "conventional":
                result = conventional(backend, folder, row, reference, vae)
            else:
                result = predict(backend, folder, index[sid, step], step, reference, mode, regressor, target)
            result.update(sample_id=sid, step=step, method=method)
            results.append(result)
            (folder / "metrics.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    means = {}
    for method, step in cases:
        records = [r for r in results if r["method"] == method and r["step"] == step]
        means[f"{method}_{step}"] = {
            key: float(np.mean([r[key] for r in records if r.get(key) is not None]))
            for key in records[0]
            if isinstance(records[0][key], (int, float)) and key not in {"step", "decoder_agreement",
                "raw_bit_exact_to_reference", "bitstream_matches_oracle_encoder"}
        }
    (args.output / "results.json").write_text(json.dumps(dict(
        test_videos=len(config["splits"]["test"]), parameters={
            "mode": sum(p.numel() for p in mode.parameters()),
            "mv_residual": sum(p.numel() for p in regressor.parameters()),
        }, means=means, per_video=results,
        quality_reference="Original final VAE raw, P frames only, 8-bit peak=255",
        timing_scope="Warm model, latent file to bitstream or FFmpeg reconstruction. Includes IO, adapter and process startup. DiT, weights load, and Oracle I preparation excluded.",
        oracle_dependency="First I-frame state and fixed template; all P-mode/MV/residual predicted",
    ), indent=2))
    print("BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
