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

## Resume 2026-11-27 18:54 Asia/Shanghai
- SSH restored. Prior smoke had completed successfully; retrieved original results, logs, effective lock and real manifest. Repeated smoke with `CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment_2/smoke.py`; both architectures passed again.
- `symbol_entropy.py`: learned per-channel Gaussian scales from 16 original train groups, evaluated exact fixed symbols on two separate historical development groups at INT4/5/6/8. Independent decoding for one video per width; reconstruction matches I0 exactly.
- Corrected I1 timing field to measure training only and reran.
- `evaluate_smoke.py`: initial namespace import conflict fixed; frozen VAE decode on GPU1, native scale normalization. No encoder VAE use.
- `train_intra.py`: eight short runs, I2/I3 × lambda16/128/1024/8192, 1000 steps each, 16 training videos, 2 separate historical development groups, separate auxiliary optimizer, gradient clipping.
- `baseline.py`: full INT2–INT8 actual-file baseline, no large manual channel search.

- All eight short-training configurations completed; evaluate_intra.py completed 30 pixel rows (14 Uniform, 16 learned). Representative first slice of every learned checkpoint decoded in a separate process; all passed.
- `python -m unittest discover -s experiment_2/tests -v`: PASS including both I2/I3 at61x105, checkpoint reloading, model mismatch, truncation and bit corruption.
- summarize.py: initial plotting dependency absent; installed fixed plotting dependencies only into E2. Final summary PASS, 30 pixel rows, 8 training runs, 14 within-support diagnostic interpolations.
- Source snapshots synchronized and verified: DCVC cbdae87a5445114cdc7f48816da63ea80bdeac40; CompressAI 79e51871e972c517bc528b689ccf15137fcc7785. Corrected CompressAI license name to BSD-3-Clause-Clear from its pyproject.toml.
- Stop at intra decision gate for this resumed run: short-model curves are not monotonic and no converged matched-rate operating point is selected; P-stage complexity and VAE-aware training remain NOT RUN. All launched work completed; no background training left running.
