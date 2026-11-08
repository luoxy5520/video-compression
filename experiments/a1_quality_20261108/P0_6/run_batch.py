"""Run the P0.6 matrix on the existing prompt-isolated train and validation clips."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from pathlib import Path

from sweep import CONFIGS, HERE, P05, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", default=12, type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((P05 / "full/expected.json").read_text())
    data = records(CONFIGS)
    jobs = [(split, [r["sample_id"] for r in data[split] if r["step"]==20])
            for split in ("train","val")]
    assert dict(jobs) == expected and tuple(map(len,expected.values())) == (44,16)
    (args.output / "expected.json").write_text(json.dumps(expected, indent=2))
    cores = sorted(os.sched_getaffinity(0))
    slots = Queue()
    for i in range(args.workers):
        slots.put(",".join(map(str,cores[i::args.workers])))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    def run(job):
        split,sample=job
        folder=args.output/split/sample
        folder.parent.mkdir(exist_ok=True)
        if (folder/"summary.json").is_file():
            return None
        if folder.exists():
            folder.rename(folder.with_name(f"{sample}.partial-{os.getpid()}"))
        cpu=slots.get()
        try:
            with (folder.with_suffix(".log")).open("w") as log:
                subprocess.run(["taskset","-c",cpu,sys.executable,"-u",str(HERE/"sweep.py"),
                                "--split",split,"--sample",sample,"--output",str(folder)],
                               check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            print("DONE",split,sample,flush=True)
            return None
        except subprocess.CalledProcessError:
            print("FAILED",split,sample,flush=True)
            return job
        finally:
            slots.put(cpu)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed=[r for r in pool.map(run,[(s,k) for s,ids in jobs for k in ids]) if r]
    (args.output/"status.json").write_text(json.dumps(dict(total=60, failed=failed),indent=2))
    if failed:
        sys.exit(1)


if __name__=="__main__":
    main()
