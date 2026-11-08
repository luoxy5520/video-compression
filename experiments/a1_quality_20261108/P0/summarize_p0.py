"""Evaluate the pre-registered gate from completed P0 train/validation replays."""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from baseline import LOCKED, digest
from evaluate import state_metrics
from run_p0_batch import CONFIGS
from state import records, targets

A = "A"
E = "E_baseline_mv_with_ideal_residual"
C = "C_min_distortion"


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def psnr(row):
    return row["psnr_y_p_db"]


def rate(row):
    return row["bitrate_kbps"]


def last4(row):
    return mean(row["psnr_y_by_frame"][-4:])


def p_nal_bits(path):
    stream = path.read_bytes()
    starts = list(re.finditer(rb"\x00\x00(?:\x00)?\x01", stream))
    p_slices = [i for i, start in enumerate(starts) if stream[start.end()] & 31 == 1]
    assert len(p_slices) >= 16
    return 8 * sum((starts[i + 1].start() if i + 1 < len(starts) else len(stream))
                   - starts[i].start() for i in p_slices)


def add_rows(output, raw, row, split, arms, labels, source_bits):
    mode, mv, _ = labels
    moving = (mv != 0).any(dim=-1).numpy() & (mode.numpy() == 1)
    for arm in arms:
        assert arm["decoder_agreement"] and arm["encoder_matches_decoder"]
        assert arm["oracle_p_bytes_read"] == 0 and arm["motion_search_s_reported"] == 0
        case = raw / "replay" / arm["arm"]
        with np.load(case / "prediction.npz") as source:
            arrays = dict(source)
        stats = state_metrics(arrays, labels)
        error = np.linalg.norm(arrays["mv_qpel"].astype(np.int32) - mv.numpy(), axis=-1)
        motion = stats["oracle_p16"]
        psnr_by_frame = arm["psnr_y_by_frame"]
        output.append(dict(
            experiment_id=arm["arm"], split=split,
            prompt_id=hashlib.sha256(row["prompt"].encode()).hexdigest()[:16], seed=row["seed"],
            video_id=row["sample_id"], source_checkpoint_hash=source_bits,
            state_checkpoint_hash=LOCKED, state_hash=arm["state_sha256"],
            replay_bitstream_hash=arm["bitstream_sha256"], mode_macro_f1=stats["macro_f1"],
            skip_recall=stats["recall"][0], p16_recall=stats["recall"][1],
            mv_qpel_exact=motion.get("exact", "NA"), mv_epe_all=motion.get("mv_epe", "NA"),
            mv_epe_nonzero=float(error[moving].mean()) if moving.any() else "NA",
            pframe_psnr_y_db=psnr(arm), pframe_psnr_yuv_db=arm["psnr_yuv_p_db"],
            pframe_ssim_y=arm["ssim_y_p"], bitrate_kbps=rate(arm),
            pframe_bits=p_nal_bits(case / "prediction.264"), p_mb_layer_bits=arm["p_mb_layer_bits"], residual_bits=arm["p_residual_bits"],
            last_frame_psnr_y_db=psnr_by_frame[-1], first4_psnr_y_db=mean(psnr_by_frame[1:5]),
            last4_psnr_y_db=last4(arm), drift_last4_minus_first4_db=arm["late_minus_early_psnr"],
            head_inference_ms="NA", export_ms="NA", replay_ms=arm["jm_replay_s"] * 1000,
            peak_cuda_mib="NA", jm_me_time_s=arm["motion_search_s_reported"],
            p_oracle_reads=arm["oracle_p_bytes_read"], jm_ffmpeg_pixel_equal=arm["decoder_agreement"],
            diagnostic_oracle=arm["diagnostic_oracle"],
            notes="Fixed Oracle I; MV metrics on Oracle P16; P bits include Annex B framing; head costs not collected in batch."))


def local_summary(raw, weights):
    counts = defaultdict(float)
    for frame in range(1, 17):
        path = raw / f"frame_{frame:02d}/cost_labels.npz"
        with np.load(path) as f:
            valid, sse = f["valid"], f["sse_yuv"][..., 0]
            mv, bits = f["mv_qpel"], f["mb_layer_bits"]
            oracle_moving = (f["oracle_mode"] == 1) & (f["oracle_mv_qpel"] != 0).any(-1)
            counts["moving_blocks"] += int(oracle_moving.sum())
            counts["blocks"] += valid.shape[0] * valid.shape[1]
            counts["distinct_sum"] += int(valid.sum())
            for lam in weights:
                cost = sse + lam * bits
                best = np.where(valid, cost, np.inf).min(-1)
                # Treat a duplicate vector as covered by any slot with that vector.
                best_idx = np.where(valid, cost, np.inf).argmin(-1)
                chosen = np.take_along_axis(mv, best_idx[..., None, None], axis=-2)
                covered = (mv[..., 1:5, :] == chosen).all(-1).any(-1)
                counts[f"lambda_{lam}_coarse4_covered"] += int(covered.sum())
                counts[f"lambda_{lam}_local_regret_sum"] += int((cost[..., 0] - best).sum())
                counts[f"lambda_{lam}_moving_local_regret_sum"] += int((cost[..., 0] - best)[oracle_moving].sum())
    return dict(counts)


def gate_for_arm(groups, arm, thresholds):
    deltas = []
    for prompt, videos in groups.items():
        arms = [v[arm] for v in videos]
        baseline = [v[A] for v in videos]
        deltas.append(dict(
            prompt_id=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            psnr=mean(map(psnr, arms)) - mean(map(psnr, baseline)),
            rate=mean(map(rate, arms)), baseline_rate=mean(map(rate, baseline)),
            last4=mean(map(last4, arms)) - mean(map(last4, baseline))))
    gain = mean(d["psnr"] for d in deltas)
    rate_delta = 100 * (mean(d["rate"] for d in deltas) / mean(d["baseline_rate"] for d in deltas) - 1)
    tail = mean(d["last4"] for d in deltas)
    positive = sum(d["psnr"] > thresholds["positive_prompt_gain_strictly_above_db"] for d in deltas) / len(deltas)
    rng = np.random.default_rng(20261108)
    bootstrap = rng.choice([d["psnr"] for d in deltas], (2000, len(deltas)), replace=True).mean(1)
    avg = lambda measure: mean(mean(measure(v[arm]) for v in videos) for videos in groups.values())
    worst = min((v[arm] for videos in groups.values() for v in videos), key=psnr)
    return dict(psnr_y_db=avg(psnr), bitrate_kbps=avg(rate), last4_psnr_y_db=avg(last4),
                worst_video=worst["sample_id"], worst_video_psnr_y_db=psnr(worst),
                per_prompt=deltas, delta_psnr_db=gain, delta_rate_pct=rate_delta, delta_last4_db=tail,
                positive_prompt_fraction=positive, prompt_bootstrap_95pct_db=np.quantile(bootstrap, [.025, .975]).tolist(),
                passes=(gain >= thresholds["min_psnr_gain_db_at_controlled_rate"] and
                        rate_delta <= thresholds["max_rate_increase_pct"] and
                        tail >= thresholds["min_last4_delta_db"] and
                        positive >= thresholds["min_positive_prompt_fraction"]))


def decide(groups, arms, gate):
    scores = {arm: gate_for_arm(groups, arm, gate) for arm in arms}
    e, pairs = scores[E], {}
    for weight in gate["lambda_y_sse_per_mb_bit"]:
        d_name = f"D_lambda_{weight}" if weight else C
        f_name = f"F_lambda_{weight}"
        d, f = scores[d_name], scores[f_name]
        d_e = d["delta_psnr_db"] - e["delta_psnr_db"]
        rate_d_e = 100 * ((100 + d["delta_rate_pct"]) / (100 + e["delta_rate_pct"]) - 1)
        pairs[weight] = dict(d=d_name, f=f_name, d_vs_e_psnr_db=d_e, d_vs_e_rate_pct=rate_d_e,
                             eligible=(d["passes"] and f["passes"] and
                                       d_e >= gate["min_d_vs_e_gain_db_at_controlled_rate"] and
                                       rate_d_e <= gate["max_d_rate_increase_vs_e_pct"]))
    eligible = [w for w, d in pairs.items() if d["eligible"]]
    if eligible:
        chosen = max(eligible, key=lambda w: (scores[pairs[w]["f"]]["delta_psnr_db"], -w))
        decision = "proceed_P1"
    elif e["passes"] or any(scores[d["d"]]["passes"] for d in pairs.values()):
        decision, chosen = "proceed_P2", None
    else:
        decision, chosen = "stop_and_rethink_candidates", None
    return dict(decision=decision, chosen_lambda=chosen, scores=scores, d_f_by_lambda=pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trajectory", type=Path)
    args = p.parse_args()
    gate = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())["gate"]
    assert gate["status"] == "locked_by_owner_before_validation"
    expected = json.loads((args.run / "expected.json").read_text())
    status = json.loads((args.run / "status.json").read_text())
    assert not status["failed"] and status["total"] == sum(map(len, expected.values()))
    rows = records(CONFIGS)
    checkpoint = Path("/mnt/data_hdd/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    source_sha = digest(checkpoint)
    metrics, local, groups, manifest = [], [], defaultdict(list), []
    arm_names = None
    for split in ("train", "val"):
        active = [row for row in rows[split] if row["step"] == 20]
        assert expected[split] == [r["sample_id"] for r in active]
        for row in active:
            sample = row["sample_id"]
            raw = args.run / split / sample
            runs = json.loads((raw / "replay/summary.json").read_text())
            names = [r["arm"] for r in runs]
            arm_names = arm_names or names
            assert names == arm_names and len(runs) == 16 and all(r["split"] == split for r in runs)
            add_rows(metrics, raw, row, split, runs, targets(row), source_sha)
            local.append(dict(split=split, video_id=sample,
                              **local_summary(raw, gate["lambda_y_sse_per_mb_bit"])))
            manifest.extend(dict(split=split, video_id=sample, frame=t, diagnostic_oracle=True,
                                 path=str(path), sha256=digest(path))
                            for t in range(1, 17) for path in [raw / f"frame_{t:02d}/cost_labels.npz"])
            if split == "val":
                groups[row["prompt"]].append({r["arm"]: r for r in runs})
    summary = decide(groups, arm_names, gate)
    trajectory = []
    if args.trajectory:
        trajectory_groups = defaultdict(list)
        for row in rows["val"]:
            if row["step"] != 20:
                continue
            sample = row["sample_id"]
            paired = json.loads((args.trajectory / sample / "summary.json").read_text())
            assert len(paired) == 4 and all(r["offline_teacher_exact"] and r["split"] == "val" and
                                            r["sample_id"] == sample for r in paired)
            baseline = next(v[A] for v in groups[row["prompt"]] if v[A]["sample_id"] == sample)
            combined = {A: baseline, **{f"paired_{r['arm']}": r for r in paired}}
            trajectory_groups[row["prompt"]].append(combined)
            trajectory.extend(paired)
        summary["trajectory_diagnostic"] = {
            arm: gate_for_arm(trajectory_groups, arm, gate)
            for arm in next(iter(trajectory_groups.values()))[0] if arm != A
        }
    args.output.mkdir(parents=True, exist_ok=False)
    csv_file(args.output / "metrics_per_video.csv", metrics)
    csv_file(args.output / "local_cost_per_video.csv", local)
    with (args.output / "candidate_manifest.jsonl").open("w") as stream:
        for row in manifest:
            stream.write(json.dumps(row) + "\n")
    score_rows = [dict(arm=arm, **{k: v for k, v in values.items() if k not in ("prompt_bootstrap_95pct_db", "per_prompt")})
                  for arm, values in summary["scores"].items()]
    csv_file(args.output / "rd_curve.csv", score_rows)
    if trajectory:
        fields = ("sample_id", "arm", "diagnostic_oracle", "psnr_y_p_db", "bitrate_kbps",
                  "last_frame_psnr_y", "late_minus_early_psnr", "skip_fraction", "p_residual_bits",
                  "offline_teacher_exact", "state_sha256", "bitstream_sha256")
        csv_file(args.output / "trajectory_per_video.csv", [{k: r[k] for k in fields} for r in trajectory])
    (args.output / "summary.json").write_text(json.dumps(
        dict(validation_videos=len(expected["val"]), validation_prompts=len(groups), **summary), indent=2))
    print(summary["decision"], flush=True)


if __name__ == "__main__":
    main()
