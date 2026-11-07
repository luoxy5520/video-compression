"""Check prediction replay completeness and baseline provenance."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def audit(benchmark, ablation, prior_modes):
    report = json.loads((benchmark / "results.json").read_text())
    records = report["per_video"]
    predictions = [r for r in records if r["method"] == "prediction"]
    baseline = [r for r in records if r["method"] == "conventional"]
    assert len(predictions) == 32 and len(baseline) == 8
    assert len({(r["sample_id"], r["method"], r["step"]) for r in records}) == 40
    assert all(r["decoder_agreement"] for r in records)
    assert all(r["raw_bit_exact_to_reference"] and r["bitstream_matches_oracle_encoder"] for r in baseline)
    for r in predictions:
        sid, step = r["sample_id"], r["step"]
        folder = benchmark / sid / f"prediction_{step:03d}"
        with np.load(folder / "prediction.npz", allow_pickle=False) as predicted:
            with np.load(prior_modes / f"{sid}_step_{step:03d}_mode.npz", allow_pickle=False) as prior:
                assert np.array_equal(predicted["mode_id"], prior["mode_id"])
        log = (folder / "replay.log").read_text()
        match = re.search(r"Total ME time for sequence\s*:\s*([0-9.]+)", log)
        assert match and float(match[1]) == 0
        expected_rate = (folder / "prediction.264").stat().st_size * 8 * 16 / 17 / 1000
        assert math.isclose(expected_rate, r["bitrate_kbps"])
    variants = json.loads((ablation / "results.json").read_text())["per_video"]
    assert len(variants) == 24 and all(r["decoder_agreement"] for r in variants)
    return dict(
        baseline_clips=8, predicted_clips=32, hybrid_clips=24,
        dual_decoder_agreement="64/64",
        raw_and_bitstream_baseline_matches_oracle="8/8",
        mode_predictions_match_prior_checkpoint="32/32",
        prediction_replays_without_motion_search="32/32",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("benchmark", "ablation", "prior-modes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.benchmark, args.ablation, args.prior_modes)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
