# Execution log —2026-11-27 Asia/Shanghai

- Read actual E2 README, plan, source mapping, model, loss, sweep/diagnostic metrics, logs, data manifest, checkpoints and Wan decode source. Local E2 HEADbfeff6b.
- `python scripts/audit.py`: E2.1a PASS. Detailed loss/gradient/updates/quantization/estimated-vs-real evidence saved. Fixed numeric CSV formatting and reran.
- tmux e21-r1: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment_2/training_diagnosis/scripts/reconstruction.py --variant R1 --steps 5000 > .../logs/R1.log 2>&1`. Finished5000 steps, continuous MSE.0033196; still improving.
- tmux e21-r1-extend: same command with --steps10000, restores latest.pt including optimizer/scheduler/RNG and fixed sampling state; then reconstruction_eval.py --variant R1. In progress.
- Recover: `tmux attach -t e21-r1-extend`; if process exits, rerun reconstruction.py (completed configs skip), then reconstruction_eval.py. No startup command marks completion; stage JSON only updates at saved checkpoints.

- Fixed diagnostic helper noise ordering: the first regression detected that direct torch.rand z-order differed from CompressAI channel-major sampling. Changed helper to call original eb/gc training forwards; exact fixed-seed equality now passes. This was a diagnostic-helper defect, not a historical-training defect. Original failure log retained. Audit rerun PASS.
- R1 at10000 MSE.00235337,PSNR-Y41.107; INT6 reference48.191. R2=64codingchannels5000 MSE.00225682/41.197dB; R3 no downsample5000 MSE.00178029/42.015dB.
- identity_control.py: same R3 shapes, analytic +/-GELU initialization; max random-tensor error2.384e-7; full-sequence native VAE73.879dB. Capacity witness only. No original model weights changed.
- tmux e21-overfit: O1 then5 O2 factors at1000steps, fixedstep.1,range±255; all complete. Scale-head bias initialized from legal training-video std and stored as shared model parameters.
- evaluate_codec.py: all five O2 points and O1 measured with full5slices, trueLVC2 streams, independent decoder and native VAE; full results saved. overfit_gate.py formatting duplication fixed; actual trend gate PASS (MSE/loglambda correlation-.808,bytes/loglambda+.981).
- tmux e21-d16-mid and e21-d16-mid-seed2: train_scale.py --videos16 --factor1 --steps5000 --seed20261127/20261128 onGPU0/GPU1 respectively; then evaluate_development.py at1000/5000(primaryseed),5000(secondseed). New data-scale experiments initialized from O2 weights, not equivalent O2 optimizer continuation. Epoch permutations provide actual seed variation; order/RNG saved for exact resume.
