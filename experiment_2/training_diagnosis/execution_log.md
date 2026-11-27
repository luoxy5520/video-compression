# Execution log —2026-11-27 Asia/Shanghai

- Read actual E2 README, plan, source mapping, model, loss, sweep/diagnostic metrics, logs, data manifest, checkpoints and Wan decode source. Local E2 HEADbfeff6b.
- `python scripts/audit.py`: E2.1a PASS. Detailed loss/gradient/updates/quantization/estimated-vs-real evidence saved. Fixed numeric CSV formatting and reran.
- tmux e21-r1: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment_2/training_diagnosis/scripts/reconstruction.py --variant R1 --steps 5000 > .../logs/R1.log 2>&1`. Finished5000 steps, continuous MSE.0033196; still improving.
- tmux e21-r1-extend: same command with --steps10000, restores latest.pt including optimizer/scheduler/RNG and fixed sampling state; then reconstruction_eval.py --variant R1. In progress.
- Recover: `tmux attach -t e21-r1-extend`; if process exits, rerun reconstruction.py (completed configs skip), then reconstruction_eval.py. No startup command marks completion; stage JSON only updates at saved checkpoints.
