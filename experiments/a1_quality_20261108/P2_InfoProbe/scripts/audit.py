"""Verify the frozen A and native teacher labels using fresh strict JM replay."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2_Lite"))
import p2lite as p

OUT = Path(sys.argv[1]).resolve()
HIST = p.ROOT / "P2_Lite/run01"


def verify(row):
    root, source = p.paths("val", row)
    sid = row["sample_id"]
    base = p.p05.load_npz(root / "A/prediction.npz")
    fixed = p.p05.load_npz(root / "fixed_mode_mv.npz")
    meta = {r["arm"]: r for r in json.loads((root / "summary.json").read_text())}
    folder = OUT / "A" / sid
    quality, _, mv, bits = p.strict(folder, row, base, source)
    assert p.digest(folder / "prediction.264") == meta["A"]["stream_sha256"]
    np.testing.assert_array_equal(mv, fixed["native_mv_qpel"].reshape(16, 1560, 2))
    np.testing.assert_array_equal(bits[..., 2], fixed["mode_id"].reshape(16, 1560))
    return dict(sample=sid, prompt=row["prompt"], last4_db=p.last4(quality), **quality)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    for sub in ("A", "labels", "features"):
        (OUT / sub).mkdir()
    rows = p.rows_by_split()
    assert len({r["prompt"] for r in rows["train"]}) == 11
    assert len({r["prompt"] for r in rows["val"]}) == 4
    assert not {r["prompt"] for r in rows["train"]} & {r["prompt"] for r in rows["val"]}
    train = rows["train"][0]
    sid = train["sample_id"]
    original = next(r for r in json.loads((HIST / "label_manifest.json").read_text()) if r["sample"] == sid)
    with ThreadPoolExecutor(max_workers=4) as pool:
        baseline = pool.map(verify, rows["val"])
        fresh = p.prep("train", train, OUT)
        fresh = json.loads(json.dumps(fresh))
        assert fresh == original, {k: (fresh[k], original[k]) for k in fresh if fresh[k] != original[k]}
        root, _ = p.paths("train", train)
        frozen = p.load_model(p.CKPT).eval().requires_grad_(False)
        assert p.digest(p.CKPT) == p.CFG["checkpoint_sha256"]
        z = p.input_tensor(train, "hidden")
        x = p.feature(frozen, z, p.p05.load_npz(root / "A/prediction.npz")).cpu()
        historical = torch.load(HIST / "features" / f"{sid}.pt", weights_only=True)
        torch.testing.assert_close(x, historical, rtol=0, atol=0)
        torch.save(x, OUT / "features" / f"{sid}.pt")
        p.smoke(OUT)
        points = list(baseline)
    means = {key: mean(mean(r[key] for r in points if r["prompt"] == prompt)
                       for prompt in {r["prompt"] for r in points})
             for key in ("psnr_y_p_db", "bitrate_kbps", "last4_db")}
    p.save_json(OUT / "audit.json", dict(baseline=means, clips=points, label=fresh,
                input_shape=list(z.shape), feature_shape=list(x.shape), feature_dtype=str(x.dtype),
                feature_sha256=p.digest(OUT / "features" / f"{sid}.pt"),
                torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                gpu=torch.cuda.get_device_name()))
    print(means, flush=True)


if __name__ == "__main__":
    main()
