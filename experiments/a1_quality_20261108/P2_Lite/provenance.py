"""Hash the existing training/validation assets referenced by this experiment."""
import argparse
from pathlib import Path

import numpy as np
import torch

from p2lite import CFG, CKPT, CONFIGS, JM, KIT, digest, paths, rows_by_split, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    records = []
    for split, rows in rows_by_split().items():
        for row in rows:
            root, source = paths(split, row)
            anchor = root / "A"
            teacher = root / CFG["teacher"] / "prediction.npz"
            assets = [source, Path(row["current_oracle_state_path"]), anchor / "prediction.npz",
                      anchor / "prediction.264", root / "summary.json"]
            if teacher.is_file():
                assets.append(teacher)
            records.append(dict(sample=row["sample_id"], split=split,
                                sha256={str(path):digest(path) for path in assets}))
    binaries = [CKPT, *CONFIGS, JM/"bin/lencod.exe", JM/"bin/ldecod.exe",
                JM/"lencod/src/state_io.c", JM/"lencod/src/slice.c",
                KIT/"scripts/prediction_to_b1_state.py", KIT/"linux/skip_p16.cfg",
                KIT/"linux/run_pipeline.py", KIT/"JM-backend/cfg/encoder_baseline.cfg"]
    save_json(a.output, dict(torch=torch.__version__, numpy=np.__version__,
                             gpu=torch.cuda.get_device_name(), records=records,
                             sha256={str(path):digest(path) for path in binaries}))


if __name__ == "__main__":
    main()
