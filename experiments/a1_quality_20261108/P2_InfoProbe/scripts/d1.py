"""Cross positions and signed DC levels on each intervention's live JM trajectory."""
import argparse
import json
import socket
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "P2_Lite"))
import p2lite as p
from sweep import blocks, from_blocks, luma, recv

HIST = p.ROOT / "P2_Lite/run01"
ARMS = ("TT", "TS", "ST", "SS")


def synth_table():
    amp = np.arange(256)
    q = p.quantize(np.broadcast_to(amp[:, None, None], (256, 4, 4)))[:, 0, 0]
    return np.searchsorted(q, np.arange(q.max() + 1)).astype(np.int16)


def local_truth(target, ref, mv, valid, amp):
    residual = blocks(target.astype(np.int32) - luma(ref, mv).astype(np.int32))
    # QP28 DC quantization, directly on the source minus this arm's MC.
    q = p.quantize(residual.reshape(-1, 4, 4))[:, 0, 0]
    q = np.where(valid & (np.abs(q) >= 2), q, 0)
    pixels = amp[np.abs(q)] * np.sign(q)
    gain = 2 * residual.sum(1) * pixels - 16 * pixels**2
    rank = np.argsort(-np.where(gain > 0, gain, 0), kind="stable")
    return q, rank[gain[rank] > 0]


def replay(row, arm, out, base, fixed, source, predicted, selected, cap):
    out.mkdir(parents=True)
    backend = p.p05.load_backend(p.KIT)
    run = backend.run
    amp = synth_table()
    qframes, refs = [], []
    state = dict(base, residual_y=np.zeros_like(base["residual_y"]))
    if arm == "SS":
        state["residual_y"] = p.to_pixels(selected)
    source_y = p.p05.read_yuv(source)[0] if arm != "SS" else None

    def exchange(listener):
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(30)
            for t in range(16):
                assert struct.unpack("<I", recv(conn, 4)) == (t + 1,)
                ref = np.frombuffer(recv(conn, 832 * 480 * 2), "<u2").reshape(480, 832).astype(np.uint8)
                refs.append(ref)
                valid = (base["mode_id"][t] == 1).repeat(4, axis=0).repeat(4, axis=1).ravel()
                truth, teacher = local_truth(source_y[t + 1], ref, base["mv_qpel"][t], valid, amp)
                students = selected[t, ..., 0].reshape(30, 52, 4, 4).transpose(0, 2, 1, 3).ravel()
                q = np.zeros_like(truth)
                locations = np.flatnonzero(students)
                if arm == "ST":
                    q[locations] = truth[locations]
                else:
                    teachers = teacher[:len(locations)]
                    values = predicted[t, ..., 0].reshape(30, 52, 4, 4).transpose(0, 2, 1, 3).ravel()
                    q[teachers] = truth[teachers] if arm == "TT" else values[teachers]
                    q[np.abs(q) < 2] = 0
                qframes.append(q.reshape(30, 4, 52, 4).transpose(0, 2, 1, 3).reshape(30, 52, 16))
                pixels = amp[np.abs(q)] * np.sign(q)
                data = from_blocks(np.broadcast_to(pixels[:, None], (pixels.size, 16)))
                conn.sendall(data.astype("<i2", copy=False).tobytes())

    def wrapped(cmd, cwd, log, env=None):
        if Path(cmd[0]).name == "lencod.exe":
            env = dict(env, JM_P05_BITS=str(out / "bits.csv"), JM_P05_MC=str(out / "trace.bin"))
            if arm != "SS":
                env["JM_P06_SOCKET"] = str(socket_path)
        return run(cmd, cwd, log, env)

    backend.run = wrapped
    if arm == "SS":
        quality = p.replay(backend, p.JM, out, row, state, source, {}, coeff_trace=True)
        expected = selected
    else:
        import tempfile
        with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="d1-") as tmp:
            socket_path = Path(tmp) / "ref.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(socket_path))
                listener.listen(1)
                listener.settimeout(30)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    job = pool.submit(exchange, listener)
                    quality = p.replay(backend, p.JM, out, row, state, source, {}, coeff_trace=True)
                    job.result()
        np.testing.assert_array_equal(refs, p.p05.read_yuv(out / "encoder_recon.yuv")[0][:-1])
        expected = np.zeros_like(selected)
        expected[..., 0] = qframes
        state["residual_y"] = p.to_pixels(expected)
        np.savez_compressed(out / "effective_state.npz", **state)
    actual = p.native_levels(out / "coeff.bin", base["mode_id"])
    np.testing.assert_array_equal(actual, expected)
    trace, bits = p.p05.read_trace(out, False)
    np.testing.assert_array_equal(trace["fields"][..., :2], fixed["native_mv_qpel"].reshape(16, 1560, 2))
    np.testing.assert_array_equal(bits[..., 2], base["mode_id"].reshape(16, 1560))
    assert quality["oracle_p_bytes_read"] == 0 and quality["motion_search_s_reported"] == 0
    return dict(**quality, last4_db=p.last4(quality), coeff_bits=int(bits[..., 6:].sum()),
                nonzero=int(np.count_nonzero(actual)), under_cap=quality["bitstream_bytes"] <= cap,
                stream_sha256=p.digest(out / "prediction.264"))


def clip(row, dest, probability, values, base_metric):
    root, source = p.paths("val", row)
    sid = row["sample_id"]
    base = p.p05.load_npz(root / "A/prediction.npz")
    fixed = p.p05.load_npz(root / "fixed_mode_mv.npz")
    results = {}
    cap = base_metric["bitstream_bytes"] * 103 // 100
    for arm in ARMS:
        results[arm] = {}
        for budget in (64, 32, 16, 8, 4, 2, 1, 0):
            levels = p.choose_levels(probability, values, base["mode_id"], [0], "video", budget)
            result = replay(row, arm, dest / sid / str(budget) / arm, base, fixed, source,
                            values, levels, cap)
            results[arm][budget] = result
            if result["under_cap"]:
                break
        assert result["under_cap"]
    budgets = {arm: list(trials)[-1] for arm, trials in results.items()}
    points = {arm: trials[budgets[arm]] for arm, trials in results.items()}
    summary = dict(sample=sid, prompt=row["prompt"], budgets=budgets, cap_bytes=cap,
                   A=base_metric, arms=points, attempts=results)
    p.save_json(dest / sid / "metrics.json", summary)
    print(sid, budgets, {a: round(r["psnr_y_p_db"] - base_metric["psnr_y_p_db"], 4)
                        for a, r in points.items()}, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    points = {r["sample"]: r for r in json.loads((args.audit / "audit.json").read_text())["clips"]}
    rows = p.rows_by_split()["val"]
    saved = torch.load(HIST / "dc-1701/model.pt", weights_only=True)
    head = p.SparseHead(saved["channels"], 1).cuda().eval()
    head.load_state_dict(saved["weights"])
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        jobs = []
        for row, (x, _, _) in zip(rows, p.data_for(HIST, rows, [0]), strict=True):
            probability, values = p.predict_scores(head, x.cuda(), [0])
            jobs.append(pool.submit(clip, row, out, probability, values, points[row["sample_id"]]))
        clips = [job.result() for job in jobs]
    p.save_json(out / "d1.json", clips)


if __name__ == "__main__":
    main()
