"""Export and validate a new dataset without touching the original 32 clips."""
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
data = Path(cfg["output"])
kit = Path(sys.argv[2]).resolve()
old = Path(sys.argv[3]).resolve()
kit.mkdir(parents=True, exist_ok=True)
(kit / "linux").mkdir(exist_ok=True)
for relative, source in [("data", data), ("linux/JM", old / "linux/JM"),
                         ("JM-backend", old / "JM-backend"), ("scripts", old / "scripts")]:
    dst = kit / relative
    if dst.is_symlink():
        if dst.resolve() != source.resolve():
            raise ValueError(dst)
    else:
        dst.symlink_to(source, target_is_directory=True)
for name in ("run_pipeline.py", "skip_p16.cfg"):
    shutil.copy2(old / "linux" / name, kit / "linux" / name)
spec = importlib.util.spec_from_file_location("oracle", kit / "linux/run_pipeline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
out = kit / "output"
out.mkdir(exist_ok=True)
expected = {(x["sample_id"], step) for x in cfg["samples"] for step in cfg["observe"]}
jobs = {}
last_progress = time.monotonic()
with ThreadPoolExecutor(max_workers=8) as pool:
    while jobs.keys() != expected:
        manifest = data / "manifest.jsonl"
        for line in manifest.read_text().splitlines() if manifest.exists() else ():
            row = json.loads(line)
            key = row["sample_id"], row["step"]
            if key not in expected:
                raise ValueError(key)
            if key in jobs or not (data / key[0] / "complete.json").exists():
                continue
            for field in ("latent", "hidden", "raw_yuv", "final_latent"):
                row[field] = str(Path(row[field]).relative_to(data))
            jobs[key] = pool.submit(module.export, row, out)
            last_progress = time.monotonic()
        for job in jobs.values():
            if job.done():
                job.result()
        if jobs.keys() != expected:
            if time.monotonic() - last_progress > 900:
                raise TimeoutError("No generated samples in 15 minutes")
            time.sleep(3)
    for job in jobs.values():
        job.result()
module.build(out)
dataset = out / "training_dataset"
counts = json.loads((dataset / "dataset_summary.json").read_text())["counts"]
assert counts["accepted_records"] == len(expected) and counts["rejected_records"] == 0, counts
splits = {split: [x["sample_id"] for x in cfg["samples"] if x["split"] == split]
          for split in ("train", "val", "test")}
(kit / "state_config.json").write_text(json.dumps(dict(dataset=str(dataset), splits=splits), indent=2))
print("EXPANSION PASS", counts, flush=True)
