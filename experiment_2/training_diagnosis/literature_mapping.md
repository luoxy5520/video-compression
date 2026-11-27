# Sources

Inherit ../literature_mapping.md. Existing I3 and CompressAI1.2.8 reused. Original diagnostic scripts use credited CompressAI APIs (BSD-3-Clause-Clear), no copied upstream network replaced. Wan native decoder inspected at Wan2.1 commit9737cba9c1c3c4d04b33fcad41c111989865d315; original code not modified.

Verified Wan origin https://github.com/Wan-Video/Wan2.1.git, commit9737cba9c1c3c4d04b33fcad41c111989865d315, Apache-2.0. vae_differentiable.py adapts only the native decode scale/conv2/causal-slice loop into local immutable cache snapshots for PyTorch checkpointing; original Wan source/weights unchanged. Forward equality measured on full sequence. Temporal probability model is an original minimal2-conv fusion/head inspired by the earlier cited papers, not an upstream model reproduction.
