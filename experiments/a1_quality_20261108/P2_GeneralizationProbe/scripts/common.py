import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "P2_CapacityProbe/scripts"))
from train_capacity import ON, build, data_for, measure, p, pairwise, sample_rows, training
from m1 import HIST as CACHE, units

OUT = p.ROOT / "P2_GeneralizationProbe"
HISTORY = p.ROOT / "P2_CapacityProbe"
CFG = json.loads((HERE / "configs/locked.json").read_text())

def read(path):
    return json.loads(path.read_text())

def keyed(rows):
    return {r["sample"]: r for r in rows}

def digest_frames(path):
    frames = np.fromfile(path, np.uint8).reshape(17, -1)
    return [hashlib.sha256(f).hexdigest() for f in frames]

def bootstrap(delta, seed=61109):
    rng = np.random.default_rng(seed)
    return np.quantile(rng.choice(delta, (10000, len(delta)), replace=True).mean(1), [.025, .975]).tolist()
