"""Write compact matched-probe tables and paired clip CSV from server metrics."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def grouped(items, field):
    prompts = {r["prompt"] for r in items}
    return mean(mean(r[field] for r in items if r["prompt"] == prompt) for prompt in prompts)


def table(runs):
    rows = [
        "| 组 / 种子 | Probe | P Y-PSNR dB | kb/s | 末四帧 dB | ΔP dB | Δ末四帧 dB | Y-SSIM | 最差样本 ΔP dB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    probe_rows = [
        "| 组 / 种子 | C MAE | C 整数精确率 | C 符号平衡准确率 | R RMSE | Top-64 命中 C / R | 头部 ms / MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d, support in runs:
        run = d["run"]
        for probe, rd in d["rd"].items():
            items = [v[probe] for v in d["val"]]
            rows.append(f'| {run} | {probe} | {rd["psnr_y_p_db"]:.6f} | {rd["bitrate_kbps"]:.3f} | '
                        f'{rd["last4_db"]:.6f} | {rd["delta_db"]:+.6f} | {rd["delta_last4_db"]:+.6f} | '
                        f'{grouped(items,"ssim_y_p"):.6f} | {rd["worst_clip"]} {rd["worst_delta_db"]:+.6f} |')
        c = [v["C"] for v in d["val"]]
        overlap = " / ".join(str(sum(r[k]["overlap"] for r in support)) for k in ("C","R"))
        timing = median(r["student_head_seconds"] for r in c) * 1000
        peak = max(r["student_peak_extra_bytes"] for r in c) / 2**20
        probe_rows.append(f'| {run} | {grouped(c,"mae"):.4f} | {grouped(c,"exact"):.2%} | '
                          f'{grouped(c,"sign_balanced_accuracy"):.2%} | {grouped(c,"lowfreq_residual_rmse"):.4f} | '
                          f'{overlap} | {timing:.2f} / {peak:.1f} |')
    return "\n".join(rows) + "\n\n" + "\n".join(probe_rows) + "\n"


def read(root):
    runs = []
    for path in sorted(root.glob("G*/metrics.json")):
        d = json.loads(path.read_text())
        d["run"] = path.parent.name
        support = json.loads(path.with_name("support.json").read_text())
        runs.append((d, support))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    runs = read(a.root)
    full = [(d,s) for d,s in runs if "-small-" not in d["run"]]
    conditional = [(d,s) for d,s in runs if "-small-" in d["run"]]
    d1 = json.loads((a.root / "d1_summary.json").read_text())
    a.out.mkdir(exist_ok=True)
    (a.out / "full_tables.md").write_text(table(full))
    (a.out / "conditional_tables.md").write_text(table(conditional))
    keys = ("sample", "prompt", "psnr_y_p_db", "bitrate_kbps", "last4_db",
            "delta_db", "delta_last4_db", "rate_pct", "ssim_y_p", "coeff_bits")
    with (a.out / "paired_validation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "probe", *keys], extrasaction="ignore", lineterminator=chr(10))
        writer.writeheader()
        for clip in d1["clips"]:
            for arm, row in clip.items():
                writer.writerow(dict(row, run=f"D1-{arm}", probe="DC"))
        for d, _ in runs:
            for v in d["val"]:
                for probe, row in v.items():
                    writer.writerow(dict(row, run=d["run"], probe=probe))


if __name__ == "__main__":
    main()
