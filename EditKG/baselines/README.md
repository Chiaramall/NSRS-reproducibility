# Baselines

| Implementation | Launcher names |
| --- | --- |
| `simple.py` | `bpr`, `lightgcn`, `sgl` |
| `external/recbole` | `gcmc`, `cke`, `ktup`, `kgcn` |
| Native code under `external/` | `kgat`, `kgin`, `kgcl`, `kgrec`, `krdn` |

From `EditKG/`, run `bash scripts/run_baseline.sh MODEL DATASET [extra arguments]`. The same launcher also accepts `editkg`.

The READMEs under `external/` describe the original projects and their environments. For this study, use the launcher and environment described in the [EditKG README](../README.md).

Pass overrides from [configs/paper_hyperparameters.tsv](../configs/paper_hyperparameters.tsv) explicitly; the launcher does **not** load them automatically. Example for LightGCN:

```bash
bash scripts/run_baseline.sh lightgcn last-fm \
  --dim 64 --lr 0.001 --batch_size 1024 --layers 3 --l2 0.0001 \
  --epochs 300 --output results/runs/paper_hparams
```

`configs/recbole_lastfm.yaml` is shared by all three datasets.

## Validation search and final refit

Use the [complete benchmark](../README.md#complete-search-and-final-evaluation)
for automated search, same-epoch validation selection, and final multi-seed
refitting. It also includes the three ItemKNN variants. From the repository root:

```bash
bash scripts/reproduce_editkg.sh all
```

The commands above document the legacy fixed-reference launcher; the benchmark
uses controlled runners with isolated outputs and does not select on final test
metrics.
