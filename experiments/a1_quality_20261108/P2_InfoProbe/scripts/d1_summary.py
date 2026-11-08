"""Aggregate D1 and measure coefficient and reference changes from preserved streams."""
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

from d1 import ARMS, local_truth, p, synth_table


def grouped(items, key):
    return mean(mean(row[key] for row in items if row["prompt"] == prompt)
                for prompt in {row["prompt"] for row in items})


def clip(row, result, root, audit):
    sid = result["sample"]
    history, source = p.paths("val", row)
    base = p.p05.load_npz(history / "A/prediction.npz")
    mode = base["mode_id"]
    source = p.p05.read_yuv(source)[0]
    anchor_ref = p.p05.read_yuv(audit / "A" / sid / "encoder_recon.yuv")[0][:-1]
    amp = synth_table()
    outcomes = {}
    levels = {}
    for arm in ARMS:
        path = root / sid / str(result["budgets"][arm]) / arm
        ref = p.p05.read_yuv(path / "encoder_recon.yuv")[0][:-1]
        q = p.native_levels(path / "coeff.bin", mode)[..., 0]
        levels[arm] = q
        field = q.reshape(16, 30, 52, 4, 4).transpose(0, 1, 3, 2, 4).reshape(16, -1)
        truths = []
        for t in range(16):
            valid = (mode[t] == 1).repeat(4, axis=0).repeat(4, axis=1).ravel()
            truth, _ = local_truth(source[t+1], ref[t], base["mv_qpel"][t], valid, amp)
            truths.append(truth)
        truth = np.stack(truths)
        chosen = field != 0
        signed = chosen & (truth != 0)
        dif = ref.astype(np.float32) - anchor_ref.astype(np.float32)
        metrics = result["arms"][arm]
        outcomes[arm] = dict(sample=sid, prompt=result["prompt"],
            delta_db=metrics["psnr_y_p_db"]-result["A"]["psnr_y_p_db"],
            delta_last4_db=metrics["last4_db"]-result["A"]["last4_db"],
            rate_pct=(metrics["bitstream_bytes"]/result["A"]["bitstream_bytes"]-1)*100,
            **{k: metrics[k] for k in ("psnr_y_p_db","last4_db","bitrate_kbps","ssim_y_p","coeff_bits","nonzero")},
            frame_delta_db=[float(a-b) for a,b in zip(metrics["psnr_y_by_frame"],result["A"]["psnr_y_by_frame"])],
            reference_rmse_by_frame=np.sqrt(np.square(dif).mean((1,2))).tolist(),
            selected_value_mae=float(np.abs(field[chosen]-truth[chosen]).mean()) if chosen.any() else None,
            selected_sign_agreement=float((np.sign(field[signed])==np.sign(truth[signed])).mean()) if signed.any() else None)
    for arm in ARMS:
        outcomes[arm]["tt_overlap"] = int(((levels[arm] != 0) & (levels["TT"] != 0)).sum())
    return outcomes


def main():
    audit, root, out = [Path(arg) for arg in sys.argv[1:]]
    rows = {row["sample_id"]: row for row in p.rows_by_split()["val"]}
    data = json.loads((root / "d1.json").read_text())
    results = [clip(rows[r["sample"]], r, root, audit) for r in data]
    rd = {}
    for arm in ARMS:
        values = [r[arm] for r in results]
        keys = ("delta_db", "delta_last4_db", "rate_pct", "psnr_y_p_db", "last4_db",
                "bitrate_kbps", "ssim_y_p", "coeff_bits", "nonzero")
        rd[arm] = {k: grouped(values, k) for k in keys}
        rd[arm]["worst"] = min(values, key=lambda r:r["delta_db"])["sample"]
        rd[arm]["prompt_delta_db"] = {prompt: mean(r["delta_db"] for r in values if r["prompt"]==prompt)
                                      for prompt in {r["prompt"] for r in values}}
        rd[arm]["tt_overlap_total"] = sum(r["tt_overlap"] for r in values)
    p.save_json(out, dict(rd=rd, clips=results))


if __name__ == "__main__":
    main()
