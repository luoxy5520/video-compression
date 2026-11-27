# Execution log — 2026-11-27 Asia/Shanghai

- Read user experiment specification; confirmed GitHub E1 HEAD6276d6ae1dd8befa4228de73b821894b08064836.
- SSH113Server read existing E0/E1 environment, input manifest, original split metadata, GPU availability and Wan model hash. No sealed test tensors opened.
- Local clone E1 branch, `git switch -c codex/experiment-2-learned-codec`.
- Created server E2 isolated venv; see environment.md and logs/install*.log. Initial ensurepip and latest SciPy failures recorded; corrected without modifying Wan environment.
- Read official paper pages and cloned DCVC/CompressAI source. Full GLVC PDF and ELIC source remain unverified due access failures; mapping states these gaps.
- `CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment_2/smoke.py > experiment_2/logs/smoke.log 2>&1`.
- First independent decoder failed because dynamic CDF buffers had empty shapes on construction; fixed checkpoint loader to resize buffers and preserve checkpoint CDFs. Original failure log retained.
- Reran smoke; independent decoding, actual bytes and checksum checks pending completion. No formal RD conclusion.
- `.venv/bin/python experiment_2/inventory.py`: metadata-only manifest audit, excludes test split and marks old validation historical_development. Independent holdout not certified.

- Server SSH then stopped responding; multiple fresh connections with ConnectTimeout8/10 timed out. Existing sessions stopped producing output. Final smoke outcome, effective lock and generated manifest cannot be verified/retrieved. No further training stages launched. Local py_compile passed for all three Python scripts. First GitHub push succeeded at a4abf30.
