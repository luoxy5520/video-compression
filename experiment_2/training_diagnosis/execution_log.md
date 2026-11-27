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

## Resume after usage limit
- Previous tool action was not executed because automatic approval review could not run after usage limit; no unsafe action determination was made. Source files written before that point were preserved.
- Initial SSH retries timed out; user confirmed VPN restored. Reconnected21:18 Asia/Shanghai, verified tmux queues ended and retrieved completed D42/5000 and D16/10000 results.
- D16 continuation initially failed on dynamic EB CDF shape32x38 versus32x27; added restore_state buffer resizing, retained failure log, resumed from untouched5000 checkpoint with optimizer/scheduler/RNG and sampler order.
- D16 primary:1000→5000→10000 development PSNR-Y43.591→44.475→44.873 and32.823→34.080→34.421; bytes323755→303983→298603 and334059→318763→315427.
- D16 secondseed5000:303643/44.319 and318455/34.133. D42/5000:306971/43.993 and315747/33.728; no consistent data-scale improvement.
- New tests PASS: original forward equality, STE values/gradient, dynamicCDF restore, TP1/TP2 integer-mean canonical symbol/reconstruction equality.
- tmux e21-d16-low/high:16videos, factors.1/10,5000steps plus separate full development evaluation.
- temporal_entropy_model.py and temporal_run.py added but training NOT RUN; frozen base requires E2.1d gate. Integer CDF-origin translation leaves canonical symbols q and dequantization fixed.

- Complete low/middle/high development curves:14/14 measured points within Uniform support,2 unique historical development videos. summarize_scale.py gate PASS after last low-point video completed; earlier incomplete13-row snapshot correctly returnedFalse.
- Native full-sequenceFP32 VAE gradient run OOM at peak74.55GB allocated (other service~8GB). Batch already1; gradient accumulation cannot lower per-sequence activations. Added pure local causal-cache checkpoint function, no detach/crop/independent-slice video concatenation. Full checkpointed FP32 run PASS: exact native forward parity,52.18GB peak, forward1.182s/backward3.265s diagnostic. VAE parameter/file hashes unchanged, no VAE optimizer entries or parameter gradients.
- V1_D16 and V2_D16 from the same D16/10000 checkpoint, fixed common16 train videos, lr1e-5, initial pixel/latent reconstruction-gradient norm calibration, target100step smoke onGPU1/GPU0; model/optimizer/scheduler/RNG checkpoints every10 steps. Native frozen VAE FP32 throughout; no mixed precision needed.
- TP0/TP1/TP2 frozen-symbol actual files and every independent decoder passed; TP1 gains6.124%/4.676%,TP2 gains8.723%/5.668%. New identical TPE1 container normalizes metadata for all methods; original fiveLVC2 files have a different header total and are not used as TP0 denominator.
- temporal_profile.py:CPU path only,1warmup+5formal per video; GPU endpoint timing NOT RUN.
- P2 temporal_joint.py small1500-step joint latent RD starts only after averageTP2>5%; two slices first500steps, then5. It is a separate lossy experiment; no fixed-symbol claims. Concurrent with pixel smoke onGPU0, so its training time is not isolated benchmark latency.
