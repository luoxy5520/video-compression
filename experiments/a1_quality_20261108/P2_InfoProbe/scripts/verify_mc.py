"""Verify the standalone P16 qpel sampler against the native JM trace."""
import argparse
from pathlib import Path

import numpy as np

from d0 import p
from sweep import luma


def verify(path, row, split):
    root, _ = p.paths(split, row)
    state = p.p05.load_npz(root / "A/prediction.npz")
    refs = p.p05.read_yuv(path / "encoder_recon.yuv")[0][:-1]
    trace, _ = p.p05.read_trace(path, False)
    use = state["mode_id"].reshape(16, 1560) == 1
    predicted = np.stack([luma(ref, mv).reshape(30, 16, 52, 16).transpose(0, 2, 1, 3).reshape(1560, 256)
                          for ref, mv in zip(refs, state["mv_qpel"], strict=True)])
    actual = trace["mc"][..., :256]
    np.testing.assert_array_equal(predicted[use], actual[use])
    return dict(sample=row["sample_id"], p16_pixels_compared=int(use.sum()) * 256,
                differences=0)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root", type=Path)
    args = a.parse_args()
    rows = p.rows_by_split()
    results = [verify(args.root / "d0/teacher", rows["train"][0], "train")]
    for row in rows["val"][::4]:
        results.append(verify(args.root / "run02/A" / row["sample_id"], row, "val"))
    p.save_json(args.root / "mc_alignment.json", results)


if __name__ == "__main__":
    main()
