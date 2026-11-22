import csv
import json
import shutil
from pathlib import Path
from statistics import mean, median

root = Path(__file__).resolve().parents[1]
result = root / "results"
historical = Path.home() / "lmp-e1-e2-followup-20261121"
stage = historical / "repo/results-e1-e2-followup"
for source, target in (
    (stage / "stage-d1/d1_complementarity.csv", "candidate_analysis.csv"),
    (stage / "stage-d3/d3_fallback.csv", "historical_fallback.csv"),
    (historical / "supplement-analysis/search_aggregate.csv", "historical_search_aggregate.csv"),
):
    shutil.copyfile(source, result / target)


def read(name):
    with (result / name).open() as file:
        return list(csv.DictReader(file))


def subset(rows, names, filename):
    with (result / filename).open("w") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows({key: row[key] for key in names} for row in rows)


rows = read("runs.csv")
timing = ["sample", "arm", "me_fraction", "amdahl_me_limit"]
timing += [f"{key}{suffix}" for key in
           ("prior_s", "integer_s", "subpixel_s", "encoder_s", "pipeline_s")
           for suffix in ("", "_min", "_max", "_warmup")]
timing += ["integer_evals", "subpixel_evals"]
subset(rows, timing, "timing_breakdown.csv")
subset([r for r in rows if r["arm"] == "C0"], timing, "timing_baseline.csv")
subset(rows, ["sample", "arm", "bitrate_kbps", "all_psnr", "p_psnr", "last4_psnr",
              "bitrate_change_pct", "p_psnr_change_db", "last4_change_db",
              "pipeline_reduction_pct", "speedup", "dual_decode"], "rd_comparison.csv")

candidate = {}
for arm in ("E2", "E3"):
    matched = [r for r in read("candidate_analysis.csv") if r["method"] == arm]
    totals = {k: sum(int(row[k]) for row in matched) for k in (
        "N", "Both_hit", "E0_only", "DiT_only", "Neither", "lost_by_insertion", "gained_by_insertion")}
    totals["P_hit_given_E0_miss"] = totals["DiT_only"] / (totals["DiT_only"] + totals["Neither"])
    candidate[arm] = totals

gates = {}
for arm in dict.fromkeys(r["arm"] for r in rows if r["arm"] != "C0"):
    matched = [r for r in rows if r["arm"] == arm]
    value = lambda key: [float(row[key]) for row in matched]
    stats = dict(median_reduction_pct=median(value("pipeline_reduction_pct")),
                 median_speedup=median(value("speedup")),
                 max_bitrate_change_pct=max(value("bitrate_change_pct")),
                 mean_p_psnr_change_db=mean(value("p_psnr_change_db")),
                 worst_p_psnr_change_db=min(value("p_psnr_change_db")),
                 mean_last4_change_db=mean(value("last4_change_db")),
                 dual_decode=all(row["dual_decode"] == "True" for row in matched))
    stats["passes_quality"] = (
        stats["max_bitrate_change_pct"] <= 3
        and stats["mean_p_psnr_change_db"] >= -.10
        and stats["worst_p_psnr_change_db"] >= -.30
        and stats["mean_last4_change_db"] >= -.20
        and stats["dual_decode"]
    )
    stats["passes_speed"] = stats["median_reduction_pct"] >= 10
    stats["passes_all"] = stats["passes_quality"] and stats["passes_speed"]
    gates[arm] = stats

(root / "results/summary.json").write_text(json.dumps(dict(
    samples=sorted({r["sample"] for r in rows}), group_video_count=len(rows),
    closed_loop_validation="development_only", arm_metrics=gates,
    historical_candidate_analysis=candidate, historical_source=str(stage),
    v2b_wall="not_run", bd_rate="not_run",
), indent=2, ensure_ascii=False) + "\n")
