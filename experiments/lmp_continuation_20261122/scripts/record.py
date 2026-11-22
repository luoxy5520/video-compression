import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("purpose")
parser.add_argument("command", nargs="+")
parser.add_argument("--produces", action="append", default=[])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
logs = root / "logs"
logs.mkdir(exist_ok=True)
with (logs / "commands.jsonl").open("a+") as journal:
    fcntl.flock(journal, fcntl.LOCK_EX)
    journal.seek(0)
    sequence = sum(1 for _ in journal) + 1
    stdout = logs / f"{sequence:04d}.stdout"
    stderr = logs / f"{sequence:04d}.stderr"
    stamp = datetime.now().astimezone().isoformat()
    start = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            status = subprocess.run(args.command, stdout=out, stderr=err).returncode
        except OSError as exc:
            err.write(str(exc).encode())
            status = 127
    journal.write(json.dumps({
        "sequence": sequence, "timestamp": stamp, "purpose": args.purpose,
        "cwd": os.getcwd(), "command": args.command, "exit_code": status,
        "stdout": str(stdout.relative_to(root)), "stderr": str(stderr.relative_to(root)),
        "duration_s": time.monotonic() - start, "produced": args.produces,
        "status": "success" if status == 0 else "failed",
    }, ensure_ascii=False) + "\n")
print(stdout.read_text(errors="replace"), end="")
print(stderr.read_text(errors="replace"), end="", file=__import__("sys").stderr)
raise SystemExit(status)
