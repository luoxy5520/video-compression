"""Fixed-video memorization diagnostics with exact native JM targets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "P2_Lite"))
import p2lite as p


def stats(pred, val, q, valid, k):
    order = np.argsort(-pred.ravel(), kind="stable")[:k]
    chosen = np.zeros(q.size, bool)
    chosen[order] = True
    chosen = chosen.reshape(q.shape) & valid
    positive = (q != 0) & valid
    common = positive & chosen
    tp, pp, ap = int(common.sum()), int(chosen.sum()), int(positive.sum())
    signs = [float((np.sign(val[q * s > 0]) == s).mean()) for s in (-1, 1) if (q * s > 0).any()]
    return dict(tp=tp, selected=pp, positives=ap, precision=tp / pp, recall=tp / ap,
                f1=2 * tp / (pp + ap), random_expected_overlap=pp * ap / int(valid.sum()),
                positive_fraction=ap / int(valid.sum()), sign_balanced_accuracy=float(np.mean(signs)),
                value_mae=float(np.abs(val[positive] - q[positive]).mean()),
                value_exact=float((val[positive] == q[positive]).mean()),
                magnitude_exact=float((np.abs(val[positive]) == np.abs(q[positive])).mean()),
                joint=int((common & (val == q)).sum()))


def model_fit(x, q, mode, name, updates, seed):
    torch.manual_seed(seed)
    net = p.SparseHead(x.shape[1], q.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=p.CFG["lr"])
    valid = torch.from_numpy(mode).cuda()
    target = torch.from_numpy(q).cuda()
    history = []
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        selection, value = p.losses(net(x), valid, target)
        (selection + value).backward()
        if step == 0:
            norms = {name: float(weight.grad.norm()) for name, weight in net.named_parameters()}
            assert all(np.isfinite(v) and v > 0 for v in norms.values())
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == updates:
            history.append(dict(step=step + 1, selection=float(selection.detach()), value=float(value.detach())))
    with torch.inference_mode():
        prob, pred = p.predict_scores(net.eval(), x, p.CFG["frequencies"]["low"])
    support = np.broadcast_to(mode[..., None, None], q.shape)
    prob = np.where(support, prob, -np.inf)
    result = stats(prob, pred, q, support, int(np.count_nonzero(q)))
    print(name, history[-1], result, flush=True)
    return net, prob, pred, dict(stats=result, history=history, first_gradient_norms=norms)


def memorize(q):
    # An independent trainable cell per signed coefficient. For squared error
    # summed over cells, one SGD update at lr 1/2 reaches the exact target.
    table = nn.Parameter(torch.zeros(q.shape))
    target = torch.from_numpy(q).float()
    optimizer = torch.optim.SGD([table], lr=0.5)
    (table - target).square().sum().backward()
    optimizer.step()
    result = table.detach().numpy().astype(np.int16)
    np.testing.assert_array_equal(result, q)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit = json.loads((args.audit / "audit.json").read_text())
    row = p.rows_by_split()["train"][0]
    sid = row["sample_id"]
    assert sid == audit["label"]["sample"] == "sample_0000"
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    root, source = p.paths("train", row)
    meta = {r["arm"]: r for r in json.loads((root / "summary.json").read_text())}
    teacher = p.p05.load_npz(root / p.CFG["teacher"] / "prediction.npz")
    base = p.p05.load_npz(root / "A/prediction.npz")
    fixed = p.p05.load_npz(root / "fixed_mode_mv.npz")
    freqs = p.CFG["frequencies"]["low"]
    mode = base["mode_id"] == 1

    def replay(name, state, expected=None):
        folder = out / name
        quality, q, mv, bits = p.strict(folder, row, state, source)
        np.testing.assert_array_equal(mv, fixed["native_mv_qpel"].reshape(16, 1560, 2))
        np.testing.assert_array_equal(bits[..., 2], base["mode_id"].reshape(16, 1560))
        if expected is not None:
            np.testing.assert_array_equal(q, expected)
        return dict(**quality, last4_db=p.last4(quality), stream_sha256=p.digest(folder / "prediction.264"),
                    nonzero=int(np.count_nonzero(q)), coeff_bits=int(bits[..., 6:].sum())), q

    baseline, _ = replay("A", base)
    assert baseline["stream_sha256"] == meta["A"]["stream_sha256"]
    teacher_metric, full = replay("teacher", teacher)
    assert teacher_metric["stream_sha256"] == meta[p.CFG["teacher"]]["stream_sha256"]
    assert (np.count_nonzero(full, axis=-1) <= 1).all()
    q = full[..., freqs]
    np.testing.assert_array_equal(q, p.p05.load_npz(args.audit / "labels" / f"{sid}.npz")["levels"])
    first = tuple(np.argwhere(q != 0)[0][:4])
    x = torch.load(args.audit / "features" / f"{sid}.pt", weights_only=True).cuda()
    results = dict(sample=sid, anchor=baseline, teacher=teacher_metric, tiers={},
                   frequencies=np.count_nonzero(full, axis=(0, 1, 2, 3)).tolist(),
                   teacher_chroma_matches_A=all(np.array_equal(base[k], teacher[k]) for k in ("residual_u", "residual_v")))
    levels = np.zeros_like(full)
    for name in ("block", "frame", "video"):
        selected = np.zeros_like(full)
        target = np.zeros_like(q)
        mask = mode.copy()
        if name == "block":
            selected[first] = full[first]
            target[first] = q[first]
            mask[:] = False
            mask[first[:3]] = True
        elif name == "frame":
            t = first[0]
            selected[t] = full[t]
            target[t] = q[t]
            mask[np.arange(len(mask)) != t] = False
        else:
            selected[:] = full
            target[:] = q
        remembered = memorize(selected)
        # The P0.6 basis amplitudes already encode every native frequency.
        # Select that immutable adapter template by the remembered block support.
        support = np.any(remembered != 0, axis=-1)[..., None, None]
        blocks = np.where(support, p.mb_to_blocks(teacher["residual_y"]), 0)
        pixels = blocks.reshape(*p.SHAPE[:3], 4, 4, 4, 4).swapaxes(-3, -2).reshape(*p.SHAPE[:3], 16, 16)
        lookup, _ = replay("lookup_" + name, dict(teacher, residual_y=pixels), remembered)
        net, prob, pred, fit = model_fit(x, target, mask, name, 200, 1701)
        results["tiers"][name] = dict(lookup=lookup, original_head=fit)
        if name == "video":
            count = int(np.count_nonzero(q))
            levels = p.choose_levels(prob, pred, mode, freqs, "video", count)
            student, _ = replay("memorized_head", dict(base, residual_y=p.to_pixels(levels)), levels)
            results["memorized_head"] = student
            results["head_sha256"] = None
            torch.save(net.state_dict(), out / "head.pt")
            results["head_sha256"] = p.digest(out / "head.pt")
        p.save_json(out / "d0.json", results)
    assert results["tiers"]["video"]["lookup"]["stream_sha256"] == teacher_metric["stream_sha256"]


if __name__ == "__main__":
    main()
