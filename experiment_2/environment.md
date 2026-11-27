# 113Server E2 environment

Server experiment root: `/home/luoxiangyu/video-compression-e2`; original E0/E1 preserved under `/home/luoxiangyu/E0_Direct_Latent_Compression`.
Python 3.12.3, torch 2.7.1+cu128, runtime CUDA12.8; driver580.82.07; 2x A10080GB. Initial use ~8GB/GPU by existing sgl_diffusion services; do not terminate them. GCC13.3, nvcc12.0; no core upgrades.
Wan git9737cba9c1c3c4d04b33fcad41c111989865d315; VAE SHA25638071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 (recomputed).

Isolation: `python3 -m venv --without-pip .venv`; `.venv/lib/python3.12/site-packages/wan-readonly.pth` contains `/home/luoxiangyu/Wan2.1/.venv/lib/python3.12/site-packages`. New packages live in E2 only; use `~/Wan2.1/.venv/bin/python -m pip --python ~/video-compression-e2/.venv/bin/python install --no-deps ...`. This reuses core binaries, not a portable standalone environment. No package writes into Wan environment.

Pinned new dependencies: compressai1.2.8, scipy1.14.1, scikit-learn1.5.2, torch-geometric2.6.1, pytorch-msssim1.0.0; full effective freeze in requirements-lock.txt after installation. Initial latest scipy1.18.1 failed against inherited numpy1.26.4; compatibility pin fixed import. ensurepip unavailable, avoided system apt/sudo.

Official sources obtained via local SSH GitHub clones and synced separately, because server HTTPS clone stalled. No models or datasets downloaded. DCVC extensions not built; CompressAI precompiled rANS backend import succeeded.
