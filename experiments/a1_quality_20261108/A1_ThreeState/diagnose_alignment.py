"""Measure how well the residual carried by each P16 arm fits its actual MC."""
import json
import sys
from pathlib import Path

import numpy as np

from run_ablation import p


def main(root, output):
    manifest = json.loads((root / "run_manifest.json").read_text())
    stats = []
    for clip in manifest["videos"]:
        directory = root / clip["sample"]
        source = p.p05.frame_blocks(p.p05.read_yuv(clip["input"]["source"]))[1:, :, :256].astype(np.int16)
        with np.load(clip["input"]["oracle_labels"], allow_pickle=False) as oracle:
            oracle_p16 = oracle["mode_id"].reshape(16, 1560) == 1
        row = dict(sample=clip["sample"])
        for arm in "ACDE":
            folder = directory / (arm if arm != "E" else "E_static_verification")
            state = p.p05.load_npz(folder / "prediction.npz")
            mask = oracle_p16 & (state["mode_id"].reshape(16, 1560) == 1)
            trace, _ = p.p05.read_trace(folder, False)
            mc = trace["mc"][mask, :256].astype(np.int16)
            true_residual = source[mask] - mc
            supplied = state["residual_y"].reshape(16, 1560, 256)[mask]
            row[arm] = dict(
                p16_count=int(mask.sum()),
                mc_rmse=float(np.sqrt(np.mean(true_residual.astype(float)**2))),
                residual_mismatch_rmse=float(np.sqrt(np.mean((true_residual-supplied).astype(float)**2))))
        assert row["E"]["residual_mismatch_rmse"] == 0
        stats.append(row)
    output.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
