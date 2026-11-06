#!/usr/bin/env python3
"""Build train/validation/test manifests and P-state targets from Oracle exports."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_LABEL_ARRAYS = (
    "is_p",
    "is_i",
    "is_skip",
    "mb_type",
    "mv_qpel_mb",
    "mv_is_uniform",
    "has_residual",
    "residual_y",
    "residual_u",
    "residual_v",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    return value.strip("-._") or "unknown"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_feature(feature_ref: Any, feature_root: Path | None) -> tuple[str | None, bool]:
    if feature_ref is None or not str(feature_ref).strip():
        return None, False
    raw = Path(str(feature_ref))
    if raw.is_absolute():
        resolved = raw.resolve()
    elif feature_root is not None:
        resolved = (feature_root / raw).resolve()
    else:
        return raw.as_posix(), False
    return str(resolved), resolved.is_file()


def validate_closure(closure: dict[str, Any]) -> list[str]:
    problems = []
    for key in (
        "bitstream_bit_exact",
        "jm_decoded_pixels_exact",
        "ffmpeg_decoded_pixels_exact",
    ):
        if closure.get(key) is not True:
            problems.append(f"closure check is not true: {key}")
    return problems


def convert_p_targets(source: Path, destination: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as labels:
        missing = [key for key in REQUIRED_LABEL_ARRAYS if key not in labels]
        if missing:
            raise ValueError(f"missing label arrays: {missing}")

        is_p = labels["is_p"].astype(bool)
        is_i = labels["is_i"].astype(bool)
        if is_p.ndim != 3:
            raise ValueError(f"is_p must have shape [T,Hmb,Wmb], got {is_p.shape}")

        p_frame_mask = np.all(is_p.reshape(is_p.shape[0], -1), axis=1)
        i_frame_mask = np.all(is_i.reshape(is_i.shape[0], -1), axis=1)
        mixed_frame_mask = ~(p_frame_mask | i_frame_mask)
        if np.any(mixed_frame_mask):
            frames = np.flatnonzero(mixed_frame_mask).tolist()
            raise ValueError(f"mixed or unsupported slice types in frames: {frames}")
        p_frames = np.flatnonzero(p_frame_mask)
        i_frames = np.flatnonzero(i_frame_mask)
        if len(i_frames) != 1 or i_frames[0] != 0:
            raise ValueError(f"expected exactly one leading I frame, got {i_frames.tolist()}")
        if len(p_frames) != is_p.shape[0] - 1:
            raise ValueError("expected all frames after frame 0 to be P frames")

        mb_type = labels["mb_type"][p_frames].astype(np.int16)
        allowed = set(np.unique(mb_type).tolist())
        if not allowed.issubset({0, 1}):
            raise ValueError(f"P macroblock types are outside skip-p16: {sorted(allowed)}")

        is_skip = labels["is_skip"][p_frames].astype(bool)
        expected_skip = mb_type == 0
        if not np.array_equal(is_skip, expected_skip):
            raise ValueError("is_skip is inconsistent with mb_type")

        mode_id = np.where(is_skip, 0, 1).astype(np.uint8)
        mode_valid = np.ones_like(mode_id, dtype=np.uint8)
        mv_valid = (~is_skip).astype(np.uint8)
        residual_valid = labels["has_residual"][p_frames].astype(np.uint8)
        if not np.array_equal(mv_valid, residual_valid):
            raise ValueError("P16 mask and residual payload mask are inconsistent")

        uniform = labels["mv_is_uniform"][p_frames].astype(bool)
        if not np.all(uniform):
            bad = int(uniform.size - uniform.sum())
            raise ValueError(f"skip-p16 contains {bad} non-uniform MV macroblocks")

        arrays = {
            "source_frame_index": p_frames.astype(np.int32),
            "mode_id": mode_id,
            "mode_valid": mode_valid,
            "mv_qpel": labels["mv_qpel_mb"][p_frames].astype(np.int16),
            "mv_valid": mv_valid,
            "residual_y": labels["residual_y"][p_frames].astype(np.int16),
            "residual_u": labels["residual_u"][p_frames].astype(np.int16),
            "residual_v": labels["residual_v"][p_frames].astype(np.int16),
            "residual_valid": residual_valid,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)

    return {
        "frames_total": int(is_p.shape[0]),
        "p_frames": int(len(p_frames)),
        "height_mbs": int(is_p.shape[1]),
        "width_mbs": int(is_p.shape[2]),
        "p_macroblocks": int(mode_id.size),
        "skip_macroblocks": int((mode_id == 0).sum()),
        "p16_macroblocks": int((mode_id == 1).sum()),
        "target_arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }


def assign_splits(
    rows: list[dict[str, Any]],
    group_by: str,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        group_key = prompt if group_by == "prompt" and prompt else row["sample_id"]
        groups[group_key].append(row)

    randomizer = random.Random(seed)
    grouped = list(groups.items())
    randomizer.shuffle(grouped)
    grouped.sort(key=lambda item: len(item[1]), reverse=True)

    target = {name: len(rows) * ratio for name, ratio in ratios.items()}
    result = {"train": [], "val": [], "test": []}
    for _group_key, members in grouped:
        deficits = {
            name: target[name] - len(result[name])
            for name in ("train", "val", "test")
        }
        chosen = max(deficits, key=lambda name: (deficits[name], ratios[name]))
        result[chosen].extend(members)

    for split_name, split_rows in result.items():
        split_rows.sort(key=lambda row: row["record_id"])
        for row in split_rows:
            row["split"] = split_name
    return result


def parse_step(value: Any) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"step must be a non-negative integer, got {value!r}")
    return int(text)


def link_final_targets(
    rows: list[dict[str, Any]], target_step: str = "final"
) -> list[int]:
    """Link every input step to the final-step labels/state of the same sample."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["input_step"] = parse_step(row["step"])
        grouped[row["sample_id"]].append(row)

    observed_steps: set[int] = set()
    for sample_id, members in grouped.items():
        by_step = {row["input_step"]: row for row in members}
        if len(by_step) != len(members):
            raise ValueError(f"duplicate numeric step in sample {sample_id}")
        observed_steps.update(by_step)
        if target_step == "final":
            declared = {
                parse_step(row["total_steps"])
                for row in members
                if row.get("total_steps") not in (None, "")
            }
            if len(declared) > 1:
                raise ValueError(
                    f"inconsistent total_steps for sample {sample_id}: {sorted(declared)}"
                )
            final_step = next(iter(declared)) if declared else max(by_step)
        else:
            final_step = parse_step(target_step)
        if final_step not in by_step:
            raise ValueError(
                f"target step {final_step} is missing for sample {sample_id}; "
                f"available steps are {sorted(by_step)}"
            )
        target = by_step[final_step]
        geometry_keys = ("width", "height", "frames", "fps", "qp", "mode_preset")
        for row in members:
            mismatched = [key for key in geometry_keys if row.get(key) != target.get(key)]
            if mismatched:
                raise ValueError(
                    f"input/target configuration mismatch for {row['record_id']}: {mismatched}"
                )
            row["target_step"] = final_step
            row["step_fraction"] = row["input_step"] / final_step if final_step else 1.0
            row["remaining_steps"] = final_step - row["input_step"]
            row["is_final_step"] = row["input_step"] == final_step
            row["target_record_id"] = target["record_id"]
            row["target_label_path"] = target["current_label_path"]
            row["target_oracle_state_path"] = target["current_oracle_state_path"]
            row["target_reference_recon_path"] = target[
                "current_reference_recon_path"
            ]
            row["label_path"] = row["target_label_path"]
            row["oracle_state_path"] = row["target_oracle_state_path"]
            row["reference_recon_path"] = row["target_reference_recon_path"]
            row["target_summary"] = target["current_target_summary"]

    descending = sorted(observed_steps, reverse=True)
    rank = {step: index + 1 for index, step in enumerate(descending)}
    for row in rows:
        row["curriculum_rank"] = rank[row["input_step"]]
    return descending


def write_step_and_curriculum_manifests(
    manifests_dir: Path,
    rows: list[dict[str, Any]],
    descending_steps: list[int],
) -> None:
    by_step_dir = manifests_dir / "by_step"
    curriculum_dir = manifests_dir / "curriculum"
    included: set[int] = set()
    for stage, step in enumerate(descending_steps, start=1):
        step_rows = [row for row in rows if row["input_step"] == step]
        write_jsonl(by_step_dir / f"step_{step:03d}_all.jsonl", step_rows)
        for split in ("train", "val", "test"):
            write_jsonl(
                by_step_dir / f"step_{step:03d}_{split}.jsonl",
                [row for row in step_rows if row.get("split") == split],
            )
        included.add(step)
        stage_name = f"stage_{stage:02d}_steps_" + "_".join(
            f"{value:03d}" for value in sorted(included, reverse=True)
        )
        stage_rows = [row for row in rows if row["input_step"] in included]
        write_jsonl(curriculum_dir / f"{stage_name}_all.jsonl", stage_rows)
        for split in ("train", "val", "test"):
            write_jsonl(
                curriculum_dir / f"{stage_name}_{split}.jsonl",
                [row for row in stage_rows if row.get("split") == split],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--group-by", choices=("prompt", "sample"), default="prompt")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20261105)
    parser.add_argument("--step-filter")
    parser.add_argument(
        "--target-step",
        default="final",
        help="Final supervision step. Default: declared total_steps, otherwise max step.",
    )
    parser.add_argument(
        "--feature-kind",
        choices=("latent", "hidden", "both", "legacy"),
        default="both",
    )
    parser.add_argument(
        "--expected-mode-preset",
        choices=("skip-p16", "p16", "variable", "any"),
        default="skip-p16",
    )
    parser.add_argument("--require-features", action="store_true")
    parser.add_argument("--copy-features", action="store_true")
    parser.add_argument("--copy-backend-artifacts", action="store_true")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise SystemExit(f"train/val/test ratios must sum to 1, got {ratio_sum}")
    if min(args.train_ratio, args.val_ratio, args.test_ratio) < 0:
        raise SystemExit("split ratios must be non-negative")

    exports_root = args.exports_root.resolve()
    output_dir = args.output_dir.resolve()
    feature_root = args.feature_root.resolve() if args.feature_root else None
    if not exports_root.is_dir():
        raise SystemExit(f"exports root does not exist: {exports_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = [
        path for path in exports_root.rglob("metadata.json")
        if not is_relative_to(path.resolve(), output_dir)
    ]
    metadata_paths.sort()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    record_ids: set[str] = set()

    for metadata_path in metadata_paths:
        export_dir = metadata_path.parent
        reasons: list[str] = []
        try:
            metadata = read_json(metadata_path)
            pairing = metadata.get("pairing") or {}
            sample_id = str(pairing.get("sample_id") or export_dir.name)
            step = str(pairing.get("diffusion_step") or "unknown")
            if args.step_filter is not None and step != args.step_filter:
                continue

            mode_preset = str((metadata.get("h264") or {}).get("mode_preset") or "unknown")
            if args.expected_mode_preset != "any" and mode_preset != args.expected_mode_preset:
                reasons.append(
                    f"mode preset mismatch: expected {args.expected_mode_preset}, got {mode_preset}"
                )

            closure_path = export_dir / "closure_result.json"
            labels_path = export_dir / "oracle_labels.npz"
            state_metadata_path = export_dir / "state_metadata.json"
            oracle_state_path = export_dir / "oracle_state.bin"
            recon_path = export_dir / "reference_recon.yuv"
            for required in (
                closure_path,
                labels_path,
                state_metadata_path,
                oracle_state_path,
                recon_path,
            ):
                if not required.is_file():
                    reasons.append(f"missing artifact: {required.name}")

            if closure_path.is_file():
                reasons.extend(validate_closure(read_json(closure_path)))

            feature_refs = {
                "latent": pairing.get("latent_ref"),
                "hidden": pairing.get("hidden_ref"),
                "final_latent": pairing.get("final_latent_ref"),
                "legacy": pairing.get("feature_ref"),
            }
            resolved_features = {
                name: resolve_feature(value, feature_root)
                for name, value in feature_refs.items()
            }
            required_feature_names = {
                "latent": ("latent",),
                "hidden": ("hidden",),
                "both": ("latent", "hidden"),
                "legacy": ("legacy",),
            }[args.feature_kind]
            if args.require_features or args.copy_features:
                for feature_name in required_feature_names:
                    feature_path, feature_exists = resolved_features[feature_name]
                    if not feature_exists:
                        reasons.append(
                            f"{feature_name} feature is required but missing: {feature_path}"
                        )

            record_id = slug(sample_id)
            if step not in ("", "unknown", "final"):
                record_id += "__step-" + slug(step)
            if record_id in record_ids:
                reasons.append(f"duplicate record_id: {record_id}")

            if reasons:
                rejected.append(
                    {
                        "export_dir": str(export_dir),
                        "sample_id": sample_id,
                        "step": step,
                        "reasons": reasons,
                    }
                )
                continue

            target_relative = Path("labels") / f"{record_id}.npz"
            target_path = output_dir / target_relative
            target_summary = convert_p_targets(labels_path, target_path)

            manifest_features = {
                name: {"path": path, "exists": exists}
                for name, (path, exists) in resolved_features.items()
            }
            if args.copy_features:
                for feature_name in required_feature_names:
                    feature_path, feature_exists = resolved_features[feature_name]
                    if not feature_exists or feature_path is None:
                        raise ValueError(
                            f"cannot copy missing {feature_name} feature for record {record_id}: {feature_path}"
                        )
                    source_feature = Path(feature_path)
                    feature_relative = (
                        Path("features")
                        / feature_name
                        / f"{record_id}{source_feature.suffix}"
                    )
                    copied_feature = output_dir / feature_relative
                    copied_feature.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_feature, copied_feature)
                    manifest_features[feature_name] = {
                        "path": feature_relative.as_posix(),
                        "exists": True,
                    }

            manifest_state_path = str(oracle_state_path.resolve())
            manifest_recon_path = str(recon_path.resolve())
            if args.copy_backend_artifacts:
                backend_dir = output_dir / "backend" / record_id
                backend_dir.mkdir(parents=True, exist_ok=True)
                copied_state = backend_dir / "oracle_state.bin"
                copied_recon = backend_dir / "reference_recon.yuv"
                shutil.copy2(oracle_state_path, copied_state)
                shutil.copy2(recon_path, copied_recon)
                manifest_state_path = copied_state.relative_to(output_dir).as_posix()
                manifest_recon_path = copied_recon.relative_to(output_dir).as_posix()

            input_meta = metadata.get("input") or {}
            row = {
                "record_id": record_id,
                "sample_id": sample_id,
                "step": step,
                "prompt": pairing.get("prompt"),
                "seed": pairing.get("seed"),
                "latent_path": manifest_features["latent"]["path"],
                "latent_exists": manifest_features["latent"]["exists"],
                "hidden_path": manifest_features["hidden"]["path"],
                "hidden_exists": manifest_features["hidden"]["exists"],
                "final_latent_path": manifest_features["final_latent"]["path"],
                "final_latent_exists": manifest_features["final_latent"]["exists"],
                "feature_path": manifest_features["legacy"]["path"],
                "feature_exists": manifest_features["legacy"]["exists"],
                "current_label_path": target_relative.as_posix(),
                "current_oracle_state_path": manifest_state_path,
                "current_reference_recon_path": manifest_recon_path,
                "source_export_dir": str(export_dir.resolve()),
                "width": input_meta.get("width"),
                "height": input_meta.get("height"),
                "frames": input_meta.get("frames"),
                "fps": input_meta.get("fps"),
                "qp": (metadata.get("h264") or {}).get("qp"),
                "mode_preset": mode_preset,
                "timestep": pairing.get("timestep"),
                "sigma": pairing.get("sigma"),
                "hidden_branch": pairing.get("hidden_branch"),
                "hidden_block": pairing.get("hidden_block"),
                "sampler": pairing.get("sampler"),
                "total_steps": pairing.get("total_steps"),
                "current_target_summary": target_summary,
            }
            accepted.append(row)
            record_ids.add(record_id)
        except Exception as exc:  # keep a complete rejection report for batch jobs
            rejected.append(
                {
                    "export_dir": str(export_dir),
                    "reasons": [f"{type(exc).__name__}: {exc}"],
                }
            )

    if not accepted:
        write_jsonl(output_dir / "rejected.jsonl", rejected)
        raise SystemExit("No valid samples were accepted; inspect rejected.jsonl")

    try:
        curriculum_steps = link_final_targets(accepted, args.target_step)
    except ValueError as exc:
        raise SystemExit(f"Failed to link final-step targets: {exc}") from exc

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    split_rows = assign_splits(accepted, args.group_by, ratios, args.split_seed)
    all_rows = sorted(accepted, key=lambda row: row["record_id"])

    manifests_dir = output_dir / "manifests"
    write_jsonl(manifests_dir / "all.jsonl", all_rows)
    for split_name, rows in split_rows.items():
        write_jsonl(manifests_dir / f"{split_name}.jsonl", rows)
        (output_dir / "splits").mkdir(parents=True, exist_ok=True)
        (output_dir / "splits" / f"{split_name}.txt").write_text(
            "".join(row["record_id"] + "\n" for row in rows),
            encoding="utf-8",
        )
    write_step_and_curriculum_manifests(manifests_dir, all_rows, curriculum_steps)
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    split_counts = {name: len(rows) for name, rows in split_rows.items()}
    skip_count = sum(row["target_summary"]["skip_macroblocks"] for row in accepted)
    p16_count = sum(row["target_summary"]["p16_macroblocks"] for row in accepted)
    feature_counts = {
        name: dict(Counter(
            "present" if row[f"{name}_exists"] else "missing" for row in accepted
        ))
        for name in ("latent", "hidden", "final_latent")
    }
    summary = {
        "schema": "oracle-h264-state-exporter/dataset-v2-final-target",
        "exports_root": str(exports_root),
        "output_dir": str(output_dir),
        "configuration": {
            "expected_mode_preset": args.expected_mode_preset,
            "step_filter": args.step_filter,
            "group_by": args.group_by,
            "split_seed": args.split_seed,
            "ratios": ratios,
            "require_features": args.require_features,
            "copy_features": args.copy_features,
            "copy_backend_artifacts": args.copy_backend_artifacts,
            "feature_kind": args.feature_kind,
            "target_step": args.target_step,
            "curriculum_order": curriculum_steps,
        },
        "counts": {
            "metadata_files_discovered": len(metadata_paths),
            "accepted_records": len(accepted),
            "rejected_records": len(rejected),
            "splits": split_counts,
            "features": feature_counts,
            "skip_macroblocks": skip_count,
            "p16_macroblocks": p16_count,
        },
    }
    write_json(output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
