# Artifact locations

Git tracks code, configs, source mapping, plans, environment lock, execution/failure logs, lightweight JSON metrics and reports. Large shared model checkpoints, per-slice LVC2 streams and decoded tensors remain on 113Server under `~/video-compression-e2/experiment_2/{checkpoints,results}`; metrics carry their SHA256 and complete size. Official reference repositories are separate under `~/video-compression-e2/third_party` and excluded from this Git repository.

New outputs: results/{smoke,symbol_entropy,uniform,intra_sweep,intra_pixel,development_comparison,artifact_hashes}.json; figures/intra_development_rd.png; reports/resumed_report.md. requirements-lock.txt and dataset_manifest.json now contain actual retrieved content, replacing interruption placeholders. Executable regression: tests/test_stream.py.
