"""Test balanced hard-negative ranking and remove the low-frequency output restriction."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from d0 import p, stats


def to_pixels(levels):
    blocks = np.zeros((*p.SHAPE, 4, 4), np.int16)
    assert (np.count_nonzero(levels, axis=-1) <= 1).all()
    for f in range(16):
        q = levels[..., f]
        use = q != 0
        if not use.any():
            continue
        basis = p.BASIS[f].reshape(4, 4)
        amps = np.arange(255 // abs(basis).max() + 1)
        mapped = p.quantize(amps[:, None, None] * basis)[:, f // 4, f % 4]
        idx = np.searchsorted(mapped, np.abs(q[use]))
        np.testing.assert_array_equal(mapped[idx], np.abs(q[use]))
        blocks[use] = (amps[idx] * np.sign(q[use]))[:, None, None] * basis
    np.testing.assert_array_equal(p.quantize(blocks).reshape(levels.shape), levels)
    return blocks.reshape(*p.SHAPE[:3], 4, 4, 4, 4).swapaxes(-3, -2).reshape(*p.SHAPE[:3], 16, 16)


def values(net, x, f):
    weights = net.value.weight.reshape(16, 65, 32)[f]
    bias = net.value.bias.reshape(16, 65)[f]
    return torch.bmm(weights, x.unsqueeze(-1)).squeeze(-1) + bias


def decode(logits, f, maxima):
    columns = torch.arange(64, device=logits.device)
    magnitudes = logits[:, 1:].masked_fill(columns >= maxima[f, None], -torch.inf).argmax(1) + 1
    return magnitudes * torch.where(logits[:, 0] > 0, 1, -1)


def embed(net, x):
    features = net.body(x)
    return features.movedim(1, -1).reshape(*p.SHAPE, 32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("d0", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    out = a.out.resolve()
    out.mkdir(exist_ok=False)
    row = p.rows_by_split()["train"][0]
    root, source = p.paths("train", row)
    base = p.p05.load_npz(root / "A/prediction.npz")
    fixed = p.p05.load_npz(root / "fixed_mode_mv.npz")
    mode = base["mode_id"]
    q = p.native_levels(a.d0 / "teacher/coeff.bin", mode)
    x = torch.load(a.audit / "features/sample_0000.pt", weights_only=True).cuda()
    eligible = torch.from_numpy(mode == 1).cuda()
    y = torch.from_numpy(q[mode == 1]).cuda().long()
    pos = y != 0
    i, b, f = pos.nonzero(as_tuple=True)
    target = y[i, b, f]
    k = len(i)
    net = p.SparseHead(x.shape[1], 16).cuda()
    optim = torch.optim.AdamW(net.parameters(), lr=0.001)
    maxima = torch.tensor([p.max_level(f) for f in range(16)], device=x.device)

    def score(name, levels):
        folder = out / name
        quality, actual, native, bits = p.strict(folder, row, dict(base, residual_y=to_pixels(levels)), source)
        np.testing.assert_array_equal(actual, levels)
        np.testing.assert_array_equal(native, fixed["native_mv_qpel"].reshape(16, 1560, 2))
        np.testing.assert_array_equal(bits[..., 2], mode.reshape(16, 1560))
        return dict(**quality, last4_db=p.last4(quality), coeff_bits=int(bits[..., 6:].sum()),
                    stream_sha256=p.digest(folder / "prediction.264"))

    lookup = score("lookup_same_chroma", q)
    history = []
    for step in range(400):
        optim.zero_grad(set_to_none=True)
        h = embed(net, x)[eligible]
        logits = net.select(h)
        positive = logits[i, b, f]
        hard = logits.masked_fill(pos, -torch.inf).flatten().topk(k).values
        selection = F.softplus(-positive).mean() + F.softplus(hard).mean()
        v = values(net, h[i, b], f)
        value = (F.binary_cross_entropy_with_logits(v[:, 0], (target > 0).float())
                 + F.cross_entropy(v[:, 1:], target.abs() - 1))
        (selection + value).backward()
        optim.step()
        if (step + 1) % 50 == 0:
            point = dict(step=step+1, position=float(selection.detach()), value=float(value.detach()))
            history.append(point)
            print(point, flush=True)
    with torch.inference_mode():
        h = embed(net.eval(), x)[eligible]
        logits = net.select(h)
        teacher_values = decode(values(net, h[i, b], f), f, maxima)
        _, inds = logits.max(-1).values.flatten().topk(k)
        mb, block = inds // 16, inds % 16
        freq = logits[mb, block].argmax(-1)
        predicted = decode(values(net, h[mb, block], freq), freq, maxima)
        output = torch.zeros_like(y)
        output[mb, block, freq] = predicted * (predicted.abs() >= 2)
        actual = np.zeros_like(q)
        actual[mode == 1] = output.cpu().numpy()
        probabilities = np.full(q.shape, -np.inf, np.float32)
        ranked = np.full(y.shape, -np.inf, np.float32)
        ranked[mb.cpu(), block.cpu(), freq.cpu()] = logits[mb, block, freq].cpu().numpy()
        probabilities[mode == 1] = ranked
        prediction_at_teacher = np.zeros_like(q)
        prediction_at_teacher[q != 0] = teacher_values.cpu().numpy()
    quality = score("repair_head", actual)
    torch.save(net.state_dict(), out / "head.pt")
    p.save_json(out / "repair.json", dict(lookup_same_chroma=lookup, head=quality, history=history,
        parameter_count=sum(t.numel() for t in net.parameters()),
        head_sha256=p.digest(out / "head.pt"), emitted_coefficients=int(np.count_nonzero(actual)),
        position_value_metrics=stats(probabilities, prediction_at_teacher, q,
                                    np.broadcast_to(mode[..., None, None] == 1, q.shape), k)))


if __name__ == "__main__":
    main()
