"""Offline-only location, integer value and reference-trajectory interventions."""
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from p2lite import (
    CFG, SHAPE, SparseHead, data_for, evaluate_clip, max_level, mb_to_blocks,
    p05, paths, predict_scores, quantize, rows_by_split, save_json, strict, summarize,
)


def top_mask(teacher, count):
    ordered = np.argsort(-np.abs(teacher).ravel(), kind="stable")[:count]
    mask = np.zeros(teacher.size, bool)
    mask[ordered] = np.abs(teacher.ravel()[ordered]) >= 2
    return mask.reshape(teacher.shape)


def audit(job):
    output, row, scored, student_case, values, freqs = job
    root, source = paths("val", row)
    student = p05.load_npz(student_case / "prediction.npz")
    baseline = p05.load_npz(root / "A/prediction.npz")
    teacher = p05.load_npz(output / "labels" / (row["sample_id"] + ".npz"))["levels"]
    teacher = teacher[..., [CFG["frequencies"]["low"].index(f) for f in freqs]]
    mode = student["mode_id"]
    source_mb = p05.frame_blocks(p05.read_yuv(source))[1:, :, :256].reshape(*SHAPE[:3], 16, 16)
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="p2diag-") as tmp:
        work = Path(tmp)
        _, student_q, _, _ = strict(work / "student", row, student, source)
        target_path = root / CFG["teacher"] / "prediction.npz"
        teacher_state = p05.load_npz(target_path) if target_path.exists() else baseline
        strict(work / "teacher", row, teacher_state, source)
        targets = []
        for arm in ("student", "teacher"):
            trace, _ = p05.read_trace(work / arm, False)
            mc = trace["mc"][..., :256].reshape(*SHAPE[:3], 16, 16).astype(np.int32)
            targets.append(quantize(mb_to_blocks(source_mb.astype(np.int32) - mc)).reshape(*SHAPE, 16)[..., freqs])
        on_student, on_teacher = targets
    q = student_q[..., freqs]
    original_mask = q != 0
    teacher_mask = top_mask(teacher, int(original_mask.sum()))
    target_mask = np.abs(teacher) >= 2
    delta = (on_student - on_teacher)[target_mask]
    changes = dict(teacher_reference_positions=int(target_mask.sum()),
                   reference_shift_mean_abs_levels=float(np.abs(delta).mean()) if delta.size else 0.,
                   reference_shift_sign_changes=int((np.sign(on_student[target_mask]) != np.sign(on_teacher[target_mask])).sum()),
                   original_student_nonzero=int(original_mask.sum()))
    supported = np.array([max_level(f) for f in freqs])
    oracle_value = np.clip(on_student, -supported, supported)
    arms = {
        "oracle_position_student_value": np.where(teacher_mask, values, 0),
        "student_position_oracle_value": np.where(original_mask, oracle_value, 0),
        "oracle_position_oracle_value": np.where(teacher_mask, teacher, 0),
    }
    results = []
    for arm, sparse in arms.items():
        sparse[np.abs(sparse) < 2] = 0
        levels = np.zeros((*SHAPE, 16), np.int16)
        levels[..., freqs] = sparse
        results.append(evaluate_clip((scored,row,arm,0,levels,{},
                                      dict(oracle_assisted=True, **changes))))
    return results


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("run")
    p.add_argument("policy")
    p.add_argument("budget")
    a = p.parse_args()
    root = a.output / a.run
    ckpt = torch.load(root / "model.pt", weights_only=True)
    freqs = ckpt["frequencies"]
    model = SparseHead(ckpt["channels"], len(freqs)).cuda().eval()
    model.load_state_dict(ckpt["weights"])
    rows = rows_by_split()["val"]
    scored = root / "offline_interventions"
    scored.mkdir()
    jobs = []
    for row, (x, _, _) in zip(rows, data_for(a.output, rows, freqs)):
        _, values = predict_scores(model, x.cuda(), freqs)
        jobs.append((a.output,row,scored,root / f"{a.policy}_{a.budget}" / row["sample_id"],values,freqs))
    with ThreadPoolExecutor(max_workers=8) as pool:
        points = [point for result in pool.map(audit, jobs) for point in result]
    save_json(scored / "metrics.json", points)
    save_json(scored / "rd.json", [dict(summarize([r for r in points if r["policy"] == arm]), oracle_assisted=True)
                                   for arm in sorted({r["policy"] for r in points})])


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
