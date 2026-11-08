"""Publish compact provenance and measurements, leaving raw replay artifacts on 113Server."""
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from baseline import ROOT, digest
from evaluate import load_backend
from state import records

HOME = Path.home()
RUN = HOME / "a1_quality_20261108/P0"
MODELS = Path("/mnt/data_hdd/models/dit_compression/a1_quality_20261108")
KIT = HOME / "a_new48_20261107/kit"
CONFIGS = [ROOT / "expansion_20261106/config_train32.json", KIT / "state_config.json"]


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    rows = {split: [r for r in group if r["step"] == 20]
            for split, group in records(CONFIGS).items()}
    manifests = [Path(json.loads(p.read_text())["dataset"]) / "manifests/all.jsonl" for p in CONFIGS]
    baselines = json.loads((RUN / "baseline/summary.json").read_text())
    smoke = json.loads((RUN / "train_smoke_bits/local_cost_audit.json").read_text())
    for arm in smoke["per_candidate"]:
        previous = RUN / "train_smoke" / f"candidate_{arm['id']}"
        assert arm["bitstream_sha256"] == digest(previous / "prediction.264")
        assert arm["state_sha256"] == digest(previous / "replay.bin")
    payloads = [*CONFIGS, *manifests, *sorted(MODELS.glob("*.pt")),
                Path("/mnt/data_hdd/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
                HOME / "jm-direct-replay-20261107/bin/lencod.exe",
                Path(load_backend(KIT).DEC),
                RUN / "jm_mb_bits/bin/lencod.exe", KIT / "scripts/prediction_to_b1_state.py",
                KIT / "JM-backend/cfg/encoder_baseline.cfg", KIT / "linux/skip_p16.cfg"]
    write_json(output / "manifest.json", dict(
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        historical_result_commit="f88ed877fb08d1c568372b0c746ae6de1a44b497",
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), sha256={str(p): digest(p) for p in payloads},
        splits={split: dict(videos=len(group), prompts=len({r["prompt"] for r in group}),
                            sample_ids=[r["sample_id"] for r in group])
                for split, group in rows.items()},
        final_blind_set="Unallocated; all 20 historical test videos have prior exposure",
        runtime=str(RUN)))
    replay = [row["replay"] for row in baselines["per_video"]]
    write_json(output / "summary.json", dict(
        baseline=dict(**baselines["mean"], videos=12, independent_prompts=3,
                      bit_exact_historical_all=all(all(r["historic_hash_equal"].values())
                                                   for r in baselines["per_video"]),
                      all_three_recon_equal=all(r["decoder_agreement"] and r["encoder_matches_decoder"]
                                                for r in replay),
                      yuv_psnr_db=statistics.mean(r["psnr_yuv_p_db"] for r in replay),
                      min_pframe_y_psnr_db=min(r["psnr_y_p_db"] for r in replay)),
        training_smoke=dict(bit_trace_nonperturbation=True, **smoke),
        status="P0 smoke complete; validation gate thresholds pending"))
    with (output / "metrics_per_video.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow("experiment_id split prompt_id seed video_id source_checkpoint_hash state_hash replay_bitstream_hash mode_macro_f1 skip_recall p16_recall mv_qpel_exact mv_epe_all mv_epe_nonzero pframe_psnr_y_db pframe_psnr_yuv_db pframe_ssim_y bitrate_kbps pframe_bits residual_bits last_frame_psnr_y_db first4_psnr_y_db last4_psnr_y_db drift_last4_minus_first4_db head_inference_ms export_ms replay_ms peak_cuda_mib jm_me_time_s p_oracle_reads jm_ffmpeg_pixel_equal notes".split())
        for row in baselines["per_video"]:
            q, s, h = row["replay"], row["state"], row["sha256"]
            psnr = q["psnr_y_by_frame"]
            mv = s["oracle_p16"]
            writer.writerow([
                "A1-locked", row["split"], hashlib.sha256(row["prompt"].encode()).hexdigest()[:16],
                row["seed"], row["sample_id"], row["checkpoint_sha256"], h["replay.bin"],
                h["prediction.264"], s["macro_f1"], *s["recall"], mv.get("exact", "NA"),
                mv.get("mv_epe", "NA"), "NA", q["psnr_y_p_db"], q["psnr_yuv_p_db"], q["ssim_y_p"],
                q["bitrate_kbps"], "NA", "NA", psnr[-1], statistics.mean(psnr[1:5]),
                statistics.mean(psnr[-4:]), q["late_minus_early_psnr"], q["head_seconds"] * 1000,
                q["export_s"] * 1000, q["jm_replay_s"] * 1000, q["head_peak_delta_bytes"] / 2**20,
                q["motion_search_s_reported"], q["oracle_p_bytes_read"], q["decoder_agreement"],
                "Fixed Oracle I. MV metrics on Oracle P16. Baseline logs lack per-P/residual bits and nonzero EPE."])
    with (output / "candidate_manifest.jsonl").open("w") as stream:
        for arm in smoke["per_candidate"]:
            stream.write(json.dumps(dict(
                sample_id=smoke["sample_id"], split="train", frame=smoke["frame"],
                candidate_id=arm["id"], candidate=smoke["candidate_slots"][arm["id"]],
                diagnostic_oracle=True, state_sha256=arm["state_sha256"],
                bitstream_sha256=arm["bitstream_sha256"],
                labels=str(RUN / "train_smoke_bits/cost_labels.npz"))) + "\n")
    write_json(output / "local_cost_audit.json", smoke)


if __name__ == "__main__":
    main()
