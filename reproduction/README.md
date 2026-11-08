# P2-Lite reproduction

The prototype uses the step-20 conditional DiT hidden cache and frozen A backbone. Only the sparse Y residual head trains. The unit of selection is a P16 4×4 block and one frequency. DC is (0,0), the horizontal first frequency is (0,1), and the vertical first frequency is (1,0), in row/column order. The head emits retain and sign logits and categorical integer magnitude logits. The adapter synthesizes a bounded integer pixel basis amplitude that quantizes to the requested level. The profile suppresses predicted levels of magnitude one to avoid JM coefficient discard. Every measured replay checks the exact signed levels, fixed modes and native qpel MVs.

The existing P0/P0.5/P0.6 scripts and kit supply the manifests and strict backend. Use the 113Server Wan2.1 Python environment. Scoring reads source YUV, while JM receives zero-filled input. The export command reads frozen weights, hidden features and the trained head. Oracle I is still required to build the downstream replay state.

```sh
d=experiments/a1_quality_20261108/P2_Lite
run="$HOME/a1_quality_20261108/P2_Lite/run01"
python "$d/p2lite.py" prepare "$run"
python "$d/p2lite.py" smoke "$run"
for variant in dc low; do
    for seed in 1701 1702; do
        python "$d/p2lite.py" train "$run" --variant "$variant" --seed "$seed"
        python "$d/p2lite.py" evaluate "$run" --variant "$variant" --seed "$seed"
    done
done
python "$d/diagnose_student.py" "$run" dc-1701 video 64
python "$d/report.py" "$run" --videos
```

Preparation replays P0.6 offline_gain_3 and checks bitstream hashes. It extracts native CAVLC levels and caches frozen features. The teacher labels reflect a source-assisted reference chain. Training uses per-video AdamW updates and positive-weighted BCE, sign BCE, and magnitude cross-entropy. The offline interventions use validation source. The final blind split remains sealed.

```sh
python "$d/p2lite.py" export student.npz --student "$run/dc-1701/model.pt" \
    --hidden /path/to/step_020/hidden_cond_last_block.pt --policy video --budget 64
```

The example checkpoint and budget describe a failed validation point. See the experiment overview for the measured decision.
