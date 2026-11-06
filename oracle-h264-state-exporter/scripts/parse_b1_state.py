#!/usr/bin/env python3
"""Parse JMZV1 version-7 B1 state into dense, training-friendly arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8s14I")
MB_RECORD = struct.Struct("<IIhhhhh4b32h16b")
EXPECTED_MAGIC = b"JMZV1\x00\x00\x00"
EXPECTED_VERSION = 7
P_SLICE = 0
I_SLICE = 2
P_SKIP = 0


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected EOF: requested {size} bytes, got {len(data)}")
    return data


def skip_sparse_block(handle) -> int:
    count = read_exact(handle, 1)[0]
    read_exact(handle, count * 5)
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_state(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    records: list[dict] = []
    sparse_value_count = 0

    with path.open("rb") as handle:
        header_values = HEADER.unpack(read_exact(handle, HEADER.size))
        magic = header_values[0]
        values = header_values[1:]
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
        header = dict(zip(keys, values))

        if magic != EXPECTED_MAGIC:
            raise ValueError(f"Unsupported magic: {magic!r}")
        if header["version"] != EXPECTED_VERSION:
            raise ValueError(f"Unsupported state version: {header['version']}")
        if header["mb_record_bytes"] != MB_RECORD.size:
            raise ValueError(
                f"MB record size mismatch: header={header['mb_record_bytes']}, "
                f"parser={MB_RECORD.size}"
            )
        if header["chroma_format_idc"] != 1:
            raise ValueError("Only YUV420 state is supported")

        width_mbs = header["width"] // 16
        height_mbs = header["height"] // 16
        if width_mbs * height_mbs != header["frame_size_mbs"]:
            raise ValueError("Header macroblock geometry is inconsistent")

        for _ in range(header["records"]):
            raw = MB_RECORD.unpack(read_exact(handle, MB_RECORD.size))
            frame_no, mb_addr = raw[0], raw[1]
            slice_type, mb_type, qp, i16mode, c_ipred_mode = raw[2:7]
            b8x8_mode = np.asarray(raw[7:11], dtype=np.int8)
            mv = np.asarray(raw[11:43], dtype=np.int16).reshape(4, 4, 2)
            ref_idx = np.asarray(raw[43:59], dtype=np.int8).reshape(4, 4)

            residual_y = np.zeros((16, 16), dtype=np.int16)
            residual_u = np.zeros((8, 8), dtype=np.int16)
            residual_v = np.zeros((8, 8), dtype=np.int16)
            i_recon_y = np.zeros((16, 16), dtype=np.uint8)
            i_recon_u = np.zeros((8, 8), dtype=np.uint8)
            i_recon_v = np.zeros((8, 8), dtype=np.uint8)
            has_residual = False
            has_i_recon = False

            if slice_type == I_SLICE:
                block_count = header["num_blk8x8"] * 4 + 3
                for _block in range(block_count):
                    sparse_value_count += skip_sparse_block(handle)
                i_recon_y = np.frombuffer(read_exact(handle, 256), dtype=np.uint8).copy().reshape(16, 16)
                i_recon_u = np.frombuffer(read_exact(handle, 64), dtype=np.uint8).copy().reshape(8, 8)
                i_recon_v = np.frombuffer(read_exact(handle, 64), dtype=np.uint8).copy().reshape(8, 8)
                has_i_recon = True
            elif slice_type == P_SLICE:
                if mb_type != P_SKIP:
                    residual_y = np.frombuffer(read_exact(handle, 512), dtype="<i2").copy().reshape(16, 16)
                    residual_u = np.frombuffer(read_exact(handle, 128), dtype="<i2").copy().reshape(8, 8)
                    residual_v = np.frombuffer(read_exact(handle, 128), dtype="<i2").copy().reshape(8, 8)
                    has_residual = True
            else:
                raise ValueError(f"Unsupported slice type {slice_type} at frame={frame_no}, mb={mb_addr}")

            records.append(
                {
                    "frame_no": frame_no,
                    "mb_addr": mb_addr,
                    "slice_type": slice_type,
                    "mb_type": mb_type,
                    "qp": qp,
                    "i16mode": i16mode,
                    "c_ipred_mode": c_ipred_mode,
                    "b8x8_mode": b8x8_mode,
                    "mv": mv,
                    "ref_idx": ref_idx,
                    "residual_y": residual_y,
                    "residual_u": residual_u,
                    "residual_v": residual_v,
                    "i_recon_y": i_recon_y,
                    "i_recon_u": i_recon_u,
                    "i_recon_v": i_recon_v,
                    "has_residual": has_residual,
                    "has_i_recon": has_i_recon,
                }
            )

        trailing = handle.read()
        if trailing:
            raise ValueError(f"State contains {len(trailing)} unparsed trailing bytes")

    if not records:
        raise ValueError("State contains no records")

    frames = max(record["frame_no"] for record in records) + 1
    shape = (frames, height_mbs, width_mbs)

    arrays: dict[str, np.ndarray] = {
        "slice_type": np.full(shape, -1, dtype=np.int16),
        "mb_type": np.full(shape, -1, dtype=np.int16),
        "qp": np.full(shape, -1, dtype=np.int16),
        "i16mode": np.full(shape, -1, dtype=np.int16),
        "chroma_intra_mode": np.full(shape, -1, dtype=np.int16),
        "b8x8_mode": np.full(shape + (4,), -1, dtype=np.int8),
        "mv_qpel": np.zeros(shape + (4, 4, 2), dtype=np.int16),
        "ref_idx": np.full(shape + (4, 4), -1, dtype=np.int8),
        "mv_qpel_mb": np.zeros(shape + (2,), dtype=np.int16),
        "ref_idx_mb": np.full(shape, -1, dtype=np.int8),
        "mv_is_uniform": np.zeros(shape, dtype=np.uint8),
        "ref_is_uniform": np.zeros(shape, dtype=np.uint8),
        "is_p": np.zeros(shape, dtype=np.uint8),
        "is_i": np.zeros(shape, dtype=np.uint8),
        "is_skip": np.zeros(shape, dtype=np.uint8),
        "has_residual": np.zeros(shape, dtype=np.uint8),
        "residual_y": np.zeros(shape + (16, 16), dtype=np.int16),
        "residual_u": np.zeros(shape + (8, 8), dtype=np.int16),
        "residual_v": np.zeros(shape + (8, 8), dtype=np.int16),
        "has_i_recon": np.zeros(shape, dtype=np.uint8),
        "i_recon_y": np.zeros(shape + (16, 16), dtype=np.uint8),
        "i_recon_u": np.zeros(shape + (8, 8), dtype=np.uint8),
        "i_recon_v": np.zeros(shape + (8, 8), dtype=np.uint8),
    }
    seen = np.zeros(shape, dtype=np.uint8)

    for record in records:
        frame = record["frame_no"]
        mb_addr = record["mb_addr"]
        row, col = divmod(mb_addr, width_mbs)
        if frame >= frames or row >= height_mbs:
            raise ValueError(f"Record index outside tensor geometry: frame={frame}, mb={mb_addr}")
        index = (frame, row, col)
        if seen[index]:
            raise ValueError(f"Duplicate state record at frame={frame}, mb={mb_addr}")
        seen[index] = 1

        arrays["slice_type"][index] = record["slice_type"]
        arrays["mb_type"][index] = record["mb_type"]
        arrays["qp"][index] = record["qp"]
        arrays["i16mode"][index] = record["i16mode"]
        arrays["chroma_intra_mode"][index] = record["c_ipred_mode"]
        arrays["b8x8_mode"][index] = record["b8x8_mode"]
        arrays["mv_qpel"][index] = record["mv"]
        arrays["ref_idx"][index] = record["ref_idx"]
        arrays["mv_qpel_mb"][index] = record["mv"][0, 0]
        arrays["ref_idx_mb"][index] = record["ref_idx"][0, 0]
        is_p = record["slice_type"] == P_SLICE
        arrays["mv_is_uniform"][index] = int(is_p and np.all(record["mv"] == record["mv"][0, 0]))
        arrays["ref_is_uniform"][index] = int(is_p and np.all(record["ref_idx"] == record["ref_idx"][0, 0]))
        arrays["is_p"][index] = int(is_p)
        arrays["is_i"][index] = int(record["slice_type"] == I_SLICE)
        arrays["is_skip"][index] = int(record["slice_type"] == P_SLICE and record["mb_type"] == P_SKIP)
        arrays["has_residual"][index] = int(record["has_residual"])
        arrays["residual_y"][index] = record["residual_y"]
        arrays["residual_u"][index] = record["residual_u"]
        arrays["residual_v"][index] = record["residual_v"]
        arrays["has_i_recon"][index] = int(record["has_i_recon"])
        arrays["i_recon_y"][index] = record["i_recon_y"]
        arrays["i_recon_u"][index] = record["i_recon_u"]
        arrays["i_recon_v"][index] = record["i_recon_v"]

    if not np.all(seen):
        missing = int(seen.size - seen.sum())
        raise ValueError(f"Dense frame/macroblock tensor has {missing} missing records")

    metadata = {
        "schema": "oracle-h264-state-exporter/npz-v1",
        "source_state_schema": "JMZV1/version-7-B1-KMR",
        "source_sha256": sha256(path),
        "header": {
            **header,
            "magic": magic.rstrip(b"\x00").decode("ascii"),
        },
        "tensor_geometry": {
            "frames": frames,
            "height_mbs": height_mbs,
            "width_mbs": width_mbs,
        },
        "enum_contract": {
            "P_SLICE": P_SLICE,
            "I_SLICE": I_SLICE,
            "P_SKIP": P_SKIP,
            "mb_type": "raw JM enum; do not remap without recording the mapping",
        },
        "units": {
            "mv_qpel": "quarter-pel integer units",
            "residual": "signed pixel-domain YUV residual before transform/quantization",
        },
        "payload": {
            "i_sparse_coefficient_values_skipped": sparse_value_count,
            "i_sparse_coefficients_exported": False,
            "i_reconstructed_reference_samples_exported": True,
        },
        "counts": {
            "records": len(records),
            "skip_macroblocks": int(arrays["is_skip"].sum()),
            "residual_macroblocks": int(arrays["has_residual"].sum()),
            "i_reference_macroblocks": int(arrays["has_i_recon"].sum()),
            "uniform_mv_p_macroblocks": int(arrays["mv_is_uniform"].sum()),
        },
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    args = parser.parse_args()

    arrays, metadata = parse_state(args.input)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_npz": str(args.output_npz),
                "metadata_json": str(args.metadata_json),
                "tensor_geometry": metadata["tensor_geometry"],
                "counts": metadata["counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
