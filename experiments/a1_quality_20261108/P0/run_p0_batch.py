"""Run the frozen P0 matrix on train and validation only, with isolated clip outputs."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from baseline import ROOT
from state import records

HERE = Path(__file__).resolve().parent
HOME = Path.home()
KIT = HOME / "a_new48_20261107/kit"
JM = HOME / "a1_quality_20261108/P0/jm_mb_bits"
MODELS = Path("/mnt/data_hdd/models/dit_compression/a1_quality_20261108")
CONFIGS = [ROOT / "expansion_20261106/config_train32.json", KIT / "state_config.json"]


def run_clip(row, split, output, slot, cores, frames_in_parallel):
    sample = row["sample_id"]
    case = output / split / sample
    case.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot % 2), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    common = ["--config", *map(str, CONFIGS), "--kit", str(KIT), "--jm", str(JM),
              "--sample", sample, "--split", split]
    cmd = ["taskset", "-c", ",".join(map(str, cores)), sys.executable, "-u"]

    def run_frame(frame):
        name = f"frame_{frame:02d}"
        with (case / f"{name}.log").open("w") as log:
            subprocess.run([*cmd, str(HERE / "local_cost.py"), *common,
                            "--checkpoint", str(MODELS / "locked_hidden.pt"),
                            "--head", str(MODELS / "frozen_coarse_oracle.pt"),
                            "--output", str(case / name), "--frame", str(frame), "--trace-bits"],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

    with ThreadPoolExecutor(max_workers=frames_in_parallel) as pool:
        list(pool.map(run_frame, range(1, 17)))
    with (case / "replay.log").open("w") as log:
        subprocess.run([*cmd, str(HERE / "train_replay.py"), *common,
                        "--audit", str(case), "--output", str(case / "replay")],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--clips-in-parallel", default=8, type=int)
    p.add_argument("--frames-in-parallel", default=3, type=int)
    a = p.parse_args()
    gate = (HERE / "config.yaml").read_text()
    if "status: locked_by_owner_before_validation" not in gate:
        raise RuntimeError("Owner approval must be recorded before validation")
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed = sorted(os.sched_getaffinity(0))
    if a.clips_in_parallel < 1 or len(allowed) // a.clips_in_parallel < 2:
        raise ValueError("Choose a clip concurrency with at least two CPUs per slot")
    slots = Queue()
    for i in range(a.clips_in_parallel):
        slots.put((i, allowed[i::a.clips_in_parallel]))
    rows = records(CONFIGS)
    jobs = [(split, row) for split in ("train", "val") for row in rows[split] if row["step"] == 20]
    (output / "expected.json").write_text(json.dumps(
        {split: [r["sample_id"] for s, r in jobs if s == split] for split in ("train", "val")}, indent=2))

    def task(job):
        split, row = job
        sample = row["sample_id"]
        case = output / split / sample
        if (case / "replay/rd_curve.csv").is_file():
            print("VERIFIED EXISTING", split, sample, flush=True)
            return None
        slot, cores = slots.get()
        try:
            if case.exists():
                case.rename(case.with_name(f"{sample}.partial-{os.getpid()}"))
            run_clip(row, split, output, slot, cores, a.frames_in_parallel)
            print("DONE", split, sample, flush=True)
            return None
        except (OSError, subprocess.CalledProcessError) as exc:
            print("FAILED", split, sample, str(exc), flush=True)
            return (split, sample)
        finally:
            slots.put((slot, cores))

    with ThreadPoolExecutor(max_workers=a.clips_in_parallel) as pool:
        failed = [job for job in pool.map(task, jobs) if job is not None]
    (output / "status.json").write_text(json.dumps(dict(total=len(jobs), failed=failed), indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
