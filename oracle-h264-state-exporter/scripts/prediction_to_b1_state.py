#!/usr/bin/env python3
"""Replace all P-frame records in a B1 state template with model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<8s14I")
MB_RECORD = struct.Struct("<IIhhhhh4b32h16b")
EXPECTED_MAGIC = b"JMZV1\x00\x00\x00"
EXPECTED_VERSION = 7
P_SLICE = 0
I_SLICE = 2
P_SKIP = 0
P16X16 = 1


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected EOF: requested {size} bytes, got {len(data)}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_i_payload(handle, num_blk8x8: int) -> bytes:
    parts: list[bytes] = []
    for _ in range(num_blk8x8 * 4 + 3):
        count_raw = read_exact(handle, 1)
        count = count_raw[0]
        parts.append(count_raw)
        parts.append(read_exact(handle, count * 5))
    parts.append(read_exact(handle, 384))
    return b"".join(parts)


def read_template(path: Path) -> tuple[bytes, dict[str, int], list[tuple[tuple, bytes]]]:
    entries: list[tuple[tuple, bytes]] = []
    with path.open("rb") as handle:
        header_raw = read_exact(handle, HEADER.size)
        unpacked = HEADER.unpack(header_raw)
        magic = unpacked[0]
        keys = (
            "version",
            "width",
            "height",
            "frame_size_mbs",
            "num_blk8x8",
            "chroma_format_idc",
            "symbol_mode",
            "profile_idc",
            "level_idc",
            "num_ref_frames",
            "p_mode_mask",
            "records",
            "mb_record_bytes",
            "coeff_record_bytes",
        )
        header = dict(zip(keys, unpacked[1:]))
        if magic != EXPECTED_MAGIC:
            raise ValueError(f"Unsupported template magic: {magic!r}")
        if header["version"] != EXPECTED_VERSION:
            raise ValueError(f"Unsupported template version: {header['version']}")
        if header["mb_record_bytes"] != MB_RECORD.size:
            raise ValueError("Template MB record size does not match version-7 B1")
        if header["chroma_format_idc"] != 1:
            raise ValueError("Only YUV420 templates are supported")
        if header["symbol_mode"] != 0 or header["profile_idc"] != 66:
            raise ValueError("Current backend requires Baseline Profile and CAVLC")
        if header["num_ref_frames"] != 1:
            raise ValueError("Current backend requires a single reference frame")

        for _ in range(header["records"]):
            fields = MB_RECORD.unpack(read_exact(handle, MB_RECORD.size))
            slice_type = fields[2]
            mb_type = fields[3]
            if slice_type == I_SLICE:
                payload = read_i_payload(handle, header["num_blk8x8"])
            elif slice_type == P_SLICE:
                payload = b"" if mb_type == P_SKIP else read_exact(handle, 768)
            else:
                raise ValueError(f"Unsupported slice type in template: {slice_type}")
            entries.append((fields, payload))
        trailing = handle.read()
        if trailing:
            raise ValueError(f"Template contains {len(trailing)} trailing bytes")
    return header_raw, header, entries


def finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def quantize_int16(value: np.ndarray, name: str) -> tuple[np.ndarray, int]:
    array = finite_array(value, name)
    rounded = np.rint(array.astype(np.float64))
    clipped = np.count_nonzero((rounded < -32768) | (rounded > 32767))
    if clipped:
        raise ValueError(f"{name} has {clipped} values outside int16 range")
    return rounded.astype(np.int16), 0


def quantize_residual(value: np.ndarray, name: str) -> tuple[np.ndarray, int]:
    array = finite_array(value, name)
    rounded = np.rint(array.astype(np.float64))
    clipped_mask = (rounded < -255) | (rounded > 255)
    clipped = int(np.count_nonzero(clipped_mask))
    rounded = np.clip(rounded, -255, 255)
    return rounded.astype(np.int16), clipped


def load_prediction(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with np.load(path, allow_pickle=False) as prediction:
        if "mode_id" in prediction:
            mode_id = finite_array(prediction["mode_id"], "mode_id")
            if not np.all(mode_id == np.rint(mode_id)):
                raise ValueError("mode_id must contain integer class IDs")
            mode_id = mode_id.astype(np.int64)
        elif "mode_logits" in prediction:
            logits = finite_array(prediction["mode_logits"], "mode_logits")
            if logits.ndim != 4 or logits.shape[-1] != 2:
                raise ValueError("mode_logits must have shape [P,Hmb,Wmb,2]")
            mode_id = np.argmax(logits, axis=-1).astype(np.int64)
        else:
            raise ValueError("prediction must contain mode_id or mode_logits")
        if mode_id.ndim != 3:
            raise ValueError("mode_id must have shape [P,Hmb,Wmb]")
        if not set(np.unique(mode_id).tolist()).issubset({0, 1}):
            raise ValueError("mode_id supports only 0=Skip and 1=P16x16")

        required = ("mv_qpel", "residual_y", "residual_u", "residual_v")
        missing = [key for key in required if key not in prediction]
        if missing:
            raise ValueError(f"prediction is missing arrays: {missing}")
        mv, _ = quantize_int16(prediction["mv_qpel"], "mv_qpel")
        residual_y, clipped_y = quantize_residual(prediction["residual_y"], "residual_y")
        residual_u, clipped_u = quantize_residual(prediction["residual_u"], "residual_u")
        residual_v, clipped_v = quantize_residual(prediction["residual_v"], "residual_v")

        p_frames, height_mbs, width_mbs = mode_id.shape
        expected = {
            "mv_qpel": (p_frames, height_mbs, width_mbs, 2),
            "residual_y": (p_frames, height_mbs, width_mbs, 16, 16),
            "residual_u": (p_frames, height_mbs, width_mbs, 8, 8),
            "residual_v": (p_frames, height_mbs, width_mbs, 8, 8),
        }
        actual = {
            "mv_qpel": mv.shape,
            "residual_y": residual_y.shape,
            "residual_u": residual_u.shape,
            "residual_v": residual_v.shape,
        }
        for key, expected_shape in expected.items():
            if actual[key] != expected_shape:
                raise ValueError(f"{key} shape must be {expected_shape}, got {actual[key]}")

        if "source_frame_index" in prediction:
            source_frames = np.asarray(prediction["source_frame_index"], dtype=np.int64)
            if source_frames.shape != (p_frames,):
                raise ValueError("source_frame_index must have shape [P]")
        else:
            source_frames = np.arange(1, p_frames + 1, dtype=np.int64)

    return (
        {
            "mode_id": mode_id,
            "mv_qpel": mv,
            "residual_y": residual_y,
            "residual_u": residual_u,
            "residual_v": residual_v,
            "source_frame_index": source_frames,
        },
        {
            "residual_y_clipped": clipped_y,
            "residual_u_clipped": clipped_u,
            "residual_v_clipped": clipped_v,
        },
    )


def median3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.median(np.stack((a, b, c), axis=0), axis=0).astype(np.int16)


def derive_skip_mv(motion: np.ndarray, row: int, col: int) -> np.ndarray:
    if row == 0 or col == 0:
        return np.zeros(2, dtype=np.int16)
    left = motion[row, col - 1]
    top = motion[row - 1, col]
    if np.all(left == 0) or np.all(top == 0):
        return np.zeros(2, dtype=np.int16)
    if col + 1 < motion.shape[1]:
        diagonal = motion[row - 1, col + 1]
    else:
        diagonal = motion[row - 1, col - 1]
    return median3(left, top, diagonal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--oracle-template", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    header_raw, header, entries = read_template(args.oracle_template)
    prediction, quantization = load_prediction(args.prediction)
    width_mbs = header["width"] // 16
    height_mbs = header["height"] // 16
    if prediction["mode_id"].shape[1:] != (height_mbs, width_mbs):
        raise ValueError(
            "prediction macroblock grid does not match template: "
            f"prediction={prediction['mode_id'].shape[1:]}, template={(height_mbs, width_mbs)}"
        )

    p_frame_numbers = sorted({fields[0] for fields, _payload in entries if fields[2] == P_SLICE})
    source_frames = prediction["source_frame_index"].tolist()
    if source_frames != p_frame_numbers:
        raise ValueError(
            f"source_frame_index does not match template P frames: {source_frames} vs {p_frame_numbers}"
        )
    frame_to_index = {frame_no: index for index, frame_no in enumerate(source_frames)}

    qp_values = {int(fields[4]) for fields, _payload in entries}
    if len(qp_values) != 1:
        raise ValueError(f"current backend requires one fixed QP, got {sorted(qp_values)}")
    qp = next(iter(qp_values))

    motion_by_frame: dict[int, np.ndarray] = {
        frame_no: np.zeros((height_mbs, width_mbs, 2), dtype=np.int16)
        for frame_no in p_frame_numbers
    }
    output_parts = [header_raw]
    skip_count = 0
    p16_count = 0
    skip_mv_ignored_nonzero = 0

    for fields, payload in entries:
        frame_no, mb_addr, slice_type = int(fields[0]), int(fields[1]), int(fields[2])
        if slice_type == I_SLICE:
            output_parts.append(MB_RECORD.pack(*fields))
            output_parts.append(payload)
            continue

        pred_index = frame_to_index[frame_no]
        row, col = divmod(mb_addr, width_mbs)
        mode = int(prediction["mode_id"][pred_index, row, col])
        predicted_mv = prediction["mv_qpel"][pred_index, row, col]
        if mode == 0:
            skip_count += 1
            if np.any(predicted_mv != 0):
                skip_mv_ignored_nonzero += 1
            final_mv = derive_skip_mv(motion_by_frame[frame_no], row, col)
            mb_type = P_SKIP
            b8_modes = [0, 0, 0, 0]
            new_payload = b""
        else:
            p16_count += 1
            final_mv = predicted_mv.astype(np.int16)
            mb_type = P16X16
            b8_modes = [1, 1, 1, 1]
            residual = np.concatenate(
                (
                    prediction["residual_y"][pred_index, row, col].reshape(-1),
                    prediction["residual_u"][pred_index, row, col].reshape(-1),
                    prediction["residual_v"][pred_index, row, col].reshape(-1),
                )
            ).astype("<i2", copy=False)
            new_payload = residual.tobytes(order="C")

        motion_by_frame[frame_no][row, col] = final_mv
        mv_grid = np.tile(final_mv, (4, 4, 1)).reshape(-1).tolist()
        ref_grid = [0] * 16
        new_fields = (
            frame_no,
            mb_addr,
            P_SLICE,
            mb_type,
            qp,
            0,
            0,
            *b8_modes,
            *mv_grid,
            *ref_grid,
        )
        output_parts.append(MB_RECORD.pack(*new_fields))
        output_parts.append(new_payload)

    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_state.write_bytes(b"".join(output_parts))
    template_hash = sha256(args.oracle_template)
    output_hash = sha256(args.output_state)
    result: dict[str, Any] = {
        "schema": "oracle-h264-state-exporter/prediction-adapter-v1",
        "prediction": str(args.prediction.resolve()),
        "oracle_template": str(args.oracle_template.resolve()),
        "output_state": str(args.output_state.resolve()),
        "configuration": {
            "width": header["width"],
            "height": header["height"],
            "frames": len(p_frame_numbers) + 1,
            "p_frames": len(p_frame_numbers),
            "qp": qp,
            "profile_idc": header["profile_idc"],
            "symbol_mode": header["symbol_mode"],
            "num_ref_frames": header["num_ref_frames"],
            "mode_preset": "skip-p16",
        },
        "counts": {
            "skip_macroblocks": skip_count,
            "p16_macroblocks": p16_count,
            "skip_predictions_with_nonzero_mv_ignored": skip_mv_ignored_nonzero,
        },
        "quantization": quantization,
        "hashes": {
            "oracle_template_sha256": template_hash,
            "predicted_state_sha256": output_hash,
            "state_matches_oracle_template": output_hash == template_hash,
        },
        "contract": {
            "skip_mv": "ignored and deterministically derived from reconstructed neighbor motion",
            "skip_residual": "ignored and omitted",
            "p16_ref_idx": 0,
            "p16_mv_unit": "qpel integer",
            "residual_range": "rounded and clipped to [-255,255]",
            "oracle_dependency": "I-frame state only; all P-frame mode/MV/residual records are replaced",
        },
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
