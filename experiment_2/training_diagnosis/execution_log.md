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

## Increased training budget — user authorized2026-11-27
- Resumed at22:24 Asia/Shanghai. V2 at680,V1 at650; both continued normally throughout API quota interruption. No original Wan parameters trained.
- User explicitly requested stronger training and continued work until usage exhaustion. Started tmux e21-d16-20000 and e21-d42-20000, exact state continuation with preserved earlier checkpoints. D16 reached20000; actual video quality at20000 remains pending evaluation.
- Started tmux e21-enhanced-gpu0/gpu1 with continue_training.py. Both wait until V1/V2 actual1000-step evaluation contains both development videos and VAE integrity passes. GPU0:P3 full-sequence pixel-aware joint100+actual evaluation, then V2middle3000+actual evaluation. GPU1:V2low1000+evaluation, V2high1000+evaluation, then full18-video frozen history pool. Commands/status are journaled in logs/queue_gpu*_commands.jsonl and results/queue_gpu*.json. Failure is recorded asFAILED, neverCOMPLETE; waiting timeout24h.
- Original native VAE remains FP32 frozen. Step cost~4–5s; queued runs are bounded training tasks, not recurring automations. No new data or model weights downloaded, no sealed test tensors read.
- Same-capacity TP2Z zero-history head trained2000steps:actual bytes300656/304424 on the two development videos, versus real TP2 271671/296635. This separates added head capacity from historical information.
- TP2 hybrid GPU analysis + CPU probability/rANS benchmark completed1warmup+5formal on2videos before1000-step continuation. Actual codec271683Bytes/44.873dB vs x264QP23 77174Bytes/44.607dB;296643/34.421 vs x264QP29 382708/34.323. Both within declared0.5dB tolerance, encoder speed ratios4.965x/5.010x. These do not imply two-end speedups or whole-pool performance. GPU main canonical symbols matched CPU on both tested videos; actual bytes differ slightly.
- Performance report initially failed NumPy bool serialization; retained failure log, fixed bool conversion and reran formal measurements. Raw timing samples persisted separately.
- Five regression tests PASS including extreme±255 symbols/CDFescape, checkpoint buffer resizing, train-forward equality and fixed-symbol integral-CDF translation. Later TPE1 parser now validates version/shapes/lengths and supports only the selected R3 layout.
- Analytical identity control weights are deterministic. The original auxiliary random-tensor witness did not explicitly seed its random draw; future reruns now seed it and skip completed evidence rather than overwrite it. This does not change the fixed identity transform or any seeded training experiment.

## Continued execution — 2026-11-27 23:08 Asia/Shanghai
- V1 and V2 each completed1000 steps and independently decoded both historical diagnostic videos. V1:293719Bytes/44.9517dB,321703/34.8290; V2:300835/44.7787,323895/34.7907. Frozen VAE integrity checks passed. These are two development cases, not full-pool conclusions.
- D16 and D42 middle-rate exact continuations both reached20000 steps; actual full-pool quality for those checkpoints is still pending.
- P3 completed its bounded100-step pixel-aware joint diagnosis and independent decoding:238015Bytes/44.8687dB,293315/34.5045. P3 changes canonical symbols; no fixed-symbol savings claim applies.
- GPU0 now continues V2 middle from1000 to3000 steps; GPU1 trains V2 low1000, followed by high1000. Both GPUs report100% utilization. Existing unrelated services remain running.
- Full18-video pool and final P2/P3 deployment profiling are queued, NOT RUN as of this update. Queue journals/status JSON distinguish active, waiting, failed and completed tasks. Server tmux execution continues through SSH/API interruptions.
- Added fresh multi-QP H.264 pool references with explicit BT709, measured-support-only H.264 comparisons and coverage, plus unique per-video figure filenames across dataset sources. Dataset integrity audit passed42 train/18 historical development tensors with no exact hash or Prompt/Seed overlap. Sealed test files were not read.

- CPU-only extended checkpoint audit now PASS:both20k checkpoints ×18 declared development videos=36 real-stream roundtrips; first slice of first video percheckpoint independently decoded. D16 mean311214.3Bytes/MSE0.000589926;D42 mean305865.9Bytes/MSE0.000548192. Pixel quality NOT MEASURED by this audit.

## Resumed2026-11-28 11:22 Asia/Shanghai
- Previous queues completed overnight:full18 pool01:59,profiling/report02:04;684 unique method-video rows,38 configurations,all real decoded.
- Fixed-symbol quality/latent equality verified over18 videos;TP2 savings5.00%,history-only gain4.55%.P2 versus Uniform matched-rate+1.582dB over18;versusH264 same-quality median byte ratio1.394,wins4/18.
- Recovered all39 exact source files against actual-run hashes into source_snapshots/development_pool_20261128.
- Started isolated P2 1500-to10000 and P3 100-to1000 at11:25 with optimizer/scheduler/RNG continuation. GPU0 then trains frozen D42/20000 TP2 and TP2Z5000. Eachlane finishes with all18 actual-stream/nativeVAE evaluation. Original artifacts preserved;core5 regressions PASS.

- At11:41 GPU0 completed72 new real-stream/nativeVAE development evaluations. P2long10000:278098.1 meanBytes/38.5766dB,Uniform matched-rate+2.1842dB across18 (oldP2+1.5817). D42fixedTP2:291785.7 meanBytes/38.7252dB;same-container savings4.2687%,history-vs-zero4.5569%.
- D42joint10000 new phase started11:41:59 from frozenD42/20000+TP2/5000. Source/checkpoint paths isolated,42 legal train videos,new optimizer intentionally initialized. GPU0 file lock serializes our training and future formal profiles. P3 continues onGPU1.
