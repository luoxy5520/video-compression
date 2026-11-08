"""Score emitted coefficient support against frozen-A large-residual support."""
import argparse
import json
from pathlib import Path

import numpy as np

from g_probe import BUDGET, FIXED_COUNT, FREQS, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    a = parser.parse_args()
    for f in sorted(a.root.glob("G*/metrics.json")):
        dest = f.parent / "support.json"
        if dest.exists():
            continue
        doc = json.loads(f.read_text())
        support = []
        for row in doc["val"]:
            sid = row["C"]["sample"]
            y = p.p05.load_npz(a.root / "g_labels" / f"{sid}.npz")
            q, mask = y["q"], y["valid"]
            vals = np.abs(q).max(-1)
            k = np.argsort(-vals.ravel(), kind="stable")[:BUDGET]
            target = np.zeros(q.shape, bool)
            target.reshape(-1, 3)[k, np.abs(q).argmax(-1).ravel()[k]] = True
            item = dict(sample=sid, prompt=row["C"]["prompt"])
            fixed = np.random.default_rng(52020).choice(int(mask.sum())*16, FIXED_COUNT, replace=False)
            target_values = q[mask].reshape(-1, 3)[fixed]
            item["zero_baseline_exact"] = float((target_values == 0).mean())
            item["zero_baseline_mae"] = float(np.abs(target_values).mean())
            for probe in ("C", "R"):
                coeff = f.parent / sid / f"{probe}_{BUDGET}" / sid / "coeff.bin"
                output = p.native_levels(coeff, mask.astype(np.uint8))[..., FREQS]
                chosen = output != 0
                tp = int((chosen & target).sum())
                selected = int(chosen.sum())
                item[probe] = dict(emitted=selected, overlap=tp,
                                   precision=tp/selected if selected else None,
                                   recall=tp/BUDGET, f1=2*tp/(selected+BUDGET),
                                   teacher_pos_joint_exact=int((chosen & target & (output == q)).sum()))
            support.append(item)
        p.save_json(dest, support)


if __name__ == "__main__":
    main()
