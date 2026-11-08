"""Recompute the first seed's tail metric using the historical P0.6 convention."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from p2lite import P06, save_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("overlay", type=Path)
    args = parser.parse_args()
    original = args.run / "dc-1701"
    corrected = args.overlay / "run01/dc-1701"
    corrected.mkdir(parents=True)
    results = json.loads((original / "metrics.json").read_text())
    anchors = {}
    for item in results:
        sid = item["sample"]
        if sid not in anchors:
            history = json.loads((P06 / "val" / sid / "summary.json").read_text())
            anchors[sid] = mean(next(p for p in history if p["arm"] == "A")["psnr_y_by_frame"][-4:])
        item["last4_db"] = mean(item["psnr_y_by_frame"][-4:])
        item["delta_last4_db"] = item["last4_db"] - anchors[sid]
        case = corrected / f'{item["policy"]}_{item["budget"]}' / sid
        case.mkdir(parents=True)
        save_json(case / "metrics.json", item)
    save_json(corrected / "metrics.json", results)
    with (original / "per_video.csv").open() as f:
        fields = csv.DictReader(f).fieldnames
    with (corrected / "per_video.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    points = json.loads((original / "rd.json").read_text())
    updates = [summarize([item for item in results if (item["policy"], item["budget"]) ==
                          (p["policy"], p["budget"])]) for p in points]
    save_json(corrected / "rd.json", updates)
    original_report = args.run / "report"
    report = args.overlay / "run01/report"
    report.mkdir()
    summary = json.loads((original_report / "summary.json").read_text())
    for row in summary["best_nonzero"]:
        if row["run"] == "dc-1701":
            point = next(p for p in updates if (p["policy"], p["budget"]) ==
                         (row["policy"], row["budget"]))
            row.update(last4_db=point["last4_db"], delta_last4_db=point["delta_last4_db"])
            by_sample = {item["sample"]:item["delta_last4_db"] for item in results
                         if (item["policy"], item["budget"]) == (row["policy"], row["budget"])}
            for video in row["video"]:
                video["delta_last4_db"] = by_sample[video["sample"]]
    save_json(report / "summary.json", summary)
    with (original_report / "rd.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    for row in rows:
        if row["run"] == "dc-1701":
            point = next(p for p in updates if (p["policy"], p["budget"]) ==
                         (row["policy"], int(row["budget"])))
            row.update(last4_db=point["last4_db"], delta_last4_db=point["delta_last4_db"])
    with (report / "rd.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
