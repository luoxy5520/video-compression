# Changes

New original code: codec.py (I2/I3 transforms, CompressAI entropy API, LVC2 serialization/checkpoint loading); smoke.py (100-step development overfit and independent decoder checks); inventory.py (metadata-only train/val manifest audit).
No E0/E1 behavior changed. External code is API dependency only. Source commits/licenses and exact source-to-module mappings in literature_mapping.md. No upstream file modified.

Resume: added symbol_entropy.py (I1 learned Gaussian fixed-symbol ablation), baseline.py (I0 comparison curve), train_intra.py (group-isolated short sweep), evaluate_smoke.py/evaluate_intra.py (frozen VAE native pixel evaluation). All original glue using credited upstream APIs. No E0/E1 or upstream source modifications.
