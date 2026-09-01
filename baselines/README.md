# Baselines

`simple.py` provides the integrated BPR, LightGCN, and SGL runners. `external` contains the source required for GC-MC, CKE, KTUP, KGCN, KGAT, KGIN, KGCL, KGRec, and KRDN.

All models are launched through `scripts/run_baseline.sh`, which gives them a common dataset root and evaluation interface while preserving model-specific hyperparameters.
