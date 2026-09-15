# EditKG Reproducibility Study

Code, dataset artifacts, and experiment configurations for evaluating EditKG and its recommendation baselines on Last-FM, Amazon-Book, and Yelp2018.

## Repository layout

```text
.
├── main.py                 # EditKG training entry point
├── modules/                # EditKG model components
├── utils/                  # Data loading, ranking, and metric functions
├── baselines/              # Integrated runners and vendored implementations
├── configs/                # Baseline hyperparameters and RecBole configuration
├── data/                   # Filtered datasets and validation views
└── scripts/                # Data preparation and model launchers
```

## Environment

Run the commands below from the repository root. The modern environment uses Python 3.12 and PyTorch; the recorded software versions are in [environment-modern.txt](environment-modern.txt).

```bash
python3 -m venv .envs/editkg
source .envs/editkg/bin/activate
pip install -r requirements.txt
```

PyTorch and `torch-scatter` must use compatible builds for the target CUDA installation.

KGAT requires a separate Python 3.7 / TensorFlow 1.15 environment at `.envs/kgat-legacy`, or an interpreter supplied through `KGAT_PYTHON`. See [environment-legacy.txt](environment-legacy.txt). The launcher defaults KGAT to CPU because its TensorFlow/CUDA combination is incompatible with Ampere GPUs.

## Data

The prepared datasets are included in `data/editkg-filtered/`. [data/README.md](data/README.md) documents the reconstruction rules, source-file requirements, and validation file layout.

To regenerate validation views from the filtered training data:

```bash
python scripts/prepare_validation_splits.py
```

## Running experiments

Use the common launcher for EditKG and the baselines:

```bash
GPU_ID=0 bash scripts/run_baseline.sh editkg last-fm
GPU_ID=0 bash scripts/run_baseline.sh lightgcn last-fm
GPU_ID=0 bash scripts/run_baseline.sh kgcn amazon-book
```

The launcher defaults to `data/editkg-filtered`. Set `EDITKG_PYTHON` to use another modern Python interpreter. Supported baseline names and configuration overrides are documented in [baselines/README.md](baselines/README.md).

For validation with a native runner, point it to the reduced training view:

```bash
EDITKG_DATA_ROOT="$PWD/data/editkg-validation" \
  GPU_ID=0 bash scripts/run_baseline.sh editkg last-fm
```

## Evaluation protocol and metrics

Report **Recall@20** and **NDCG@20**, averaged over users, with full-catalog ranking and training interactions excluded.

- **EditKG and native baselines** evaluate `test.txt` from the selected data root: validation interactions in `editkg-validation`, held-out test interactions in `editkg-filtered`. Early stopping uses this split.
- **RecBole baselines** divide that same `test.txt` into framework validation and test files; users with one held-out item appear in both. With the default data root, model selection therefore uses held-out test interactions. With `editkg-validation`, neither file is the final test set.

Choose hyperparameters and training duration on validation, then retrain on the full filtered training set and evaluate the untouched test set at the fixed duration. The launcher does not automate this workflow.

EditKG's `best_metric` reports independent maxima, which may come from different epochs. Report Recall and NDCG from the same selected epoch.

## Citation

```bibtex
@inproceedings{tang2024editkg,
  title={EditKG: Editing Knowledge Graph for Recommendation},
  author={Tang, Gu and Gan, Xiaoying and Wang, Jinghe and Lu, Bin and Wu, Lyuwen and Fu, Luoyi and Zhou, Chenghu},
  booktitle={Proceedings of the 47th ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={112--122},
  year={2024}
}
```
