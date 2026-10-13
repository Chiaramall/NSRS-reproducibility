# EditKG Reproducibility Study

Code, datasets and configurations for evaluating EditKG and recommendation baselines on Last-FM, Amazon-Book and Yelp2018.

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

Run the commands below from `EditKG/`. The PyTorch environment uses Python 3.12; the recorded software versions are in [environment-modern.txt](environment-modern.txt).

```bash
conda create -y -n nsrs-editkg-3-12 python=3.12 pip
conda activate nsrs-editkg-3-12
bash scripts/install_environment.sh cu126
export EDITKG_PYTHON="$(command -v python)"
```

The installer pins Torch 2.7.0 with CUDA 12.6, installs the matching
`torch-scatter==2.1.2+pt27cu126` wheel, then installs `requirements.txt`.
`torch-scatter` is deliberately outside that file: pip's isolated source build
cannot import Torch just because Torch appears earlier in the same requirements
file. Wheel-only installation fails clearly when the platform is unsupported.
The installer finishes with `pip check` and a scatter operation/import check.
RecBole runs from the included `baselines/external/recbole` tree. Its runtime
packages are listed explicitly: the PyPI `recbole==1.2.1` dependency would pull
Ray <=2.6.3, which has no Python 3.12 wheel. The optional upstream Ray tuning
entry point imports Ray only when called; the local launchers do not need it.

For CPU installation/import checks, use `bash scripts/install_environment.sh cpu`.
The original native GPU runners may still require CUDA. With an existing Python
3.12 interpreter, a venv is also supported:

```bash
python3.12 -m venv .envs/editkg
source .envs/editkg/bin/activate
bash scripts/install_environment.sh cu126
export EDITKG_PYTHON="$(command -v python)"
```

The CUDA installation procedure follows the [PyG wheel instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

KGAT requires a separate Python 3.7 / TensorFlow 1.15 environment at `.envs/kgat-legacy`, or an interpreter supplied through `KGAT_PYTHON`. See [environment-legacy.txt](environment-legacy.txt). The launcher defaults KGAT to CPU because its TensorFlow/CUDA combination is incompatible with Ampere GPUs.

Create the CPU KGAT environment from `EditKG/`:

```bash
conda create -y --prefix "$PWD/.envs/kgat-legacy" python=3.7 pip
conda run --prefix "$PWD/.envs/kgat-legacy" python -m pip install \
  'tensorflow==1.15.0' 'numpy==1.18.5' 'scipy==1.4.1' \
  'scikit-learn==0.22.1' 'protobuf==3.20.3'
```

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

These commands use `data/editkg-filtered` and evaluate on the test split. Set `EDITKG_PYTHON` to use another Python interpreter. Supported baseline names and configuration overrides are documented in [baselines/README.md](baselines/README.md).

For validation with a native runner, point it to the reduced training view:

```bash
EDITKG_DATA_ROOT="$PWD/data/editkg-validation" \
  GPU_ID=0 bash scripts/run_baseline.sh editkg last-fm
```

## Complete search and final evaluation

From the repository root, activate the modern environment and install the KGAT
legacy environment as described above:

```bash
bash EditKG/reproduce.sh plan
GPU_ID=0 bash EditKG/reproduce.sh all
```

This covers EditKG, BPR, LightGCN, SGL, GCMC, CKE, KTUP, KGCN, KGAT, KGIN,
KGCL, KGRec, KRDN, ItemKNN-CF, ItemKNN-CBF and ItemKNN-CFCBF on all three
included datasets. Append dataset keys to restrict a shell command. The stages
`search`, `final`, and `report` can also be run separately with identical settings.
The old fixed-reference commands remain available as `validation` and
`historical-test` and are separate from the new benchmark.

For a model subset or a smaller budget, run from `EditKG/`:

```bash
python scripts/benchmark.py --stage all --datasets last-fm \
  --models editkg bpr lightgcn --trials 5 --epochs 30 \
  --final-seeds 2023 2024 --output results/benchmark_small
```

Options include `--gpu`, `--cpu`, `--python`, `--kgat-python`, `--data-root`,
`--validation-root`, `--search-seed` and `--training-seed`. `--stage plan` prints
model/dataset fit counts without creating files or training. The shell wrapper
accepts `TRIALS` (30), `EPOCHS` (300), `FINAL_SEEDS` (`2020 2021 2022 2023 2024`),
`OUTPUT_DIR`, `GPU_ID`, and `PYTHON`; `DRY_RUN=1` only prints its invocation.
Full default searches can take several days.

### Search

The driver reads `hyperparameters/EditKG.json`. For categorical spaces it tries
the recorded explicit baseline screens first, then seeded random configurations;
recorded author-confirmed values are included in the candidate space. ItemKNN
uses the recorded six-row joint grid. This is a newly executable, deterministic
search protocol; it does not claim to reconstruct historical Hyperopt/Optuna
trial order or reproduce their exact scores.

Every candidate trains on `editkg-validation/<dataset>/train.txt` and evaluates
on that view's full `test.txt` (the validation interactions). RecBole and simple
baselines do not subdivide this file in the controlled workflow. Validation
Recall@20 selects the configuration and epoch; ties use NDCG@20, then the earlier
epoch/trial. A candidate is evaluated each epoch up to the requested budget.

### Final refit and metrics

The selected configuration is freshly initialized for each final seed, trained
on `editkg-filtered/<dataset>/train.txt` for exactly the selected number of epochs,
and evaluated once on the full untouched test interactions. No validation loader,
test-based early stopping, or best-test-epoch checkpoint selection is used during
final refitting. Native and RecBole runners share this rule.

Report macro-averaged **Recall@20** and **NDCG@20**, using full-catalog ranking with
training items excluded. Both metrics come from the same selected validation
epoch or final fitted model. Final summaries contain per-seed metrics, their mean
and sample standard deviation. ItemKNN is deterministic and is fitted once.

The ItemKNN runners reuse the sparse weighting/similarity implementation from
CAFE. CF profiles use training interactions; CBF profiles use binary outgoing
`(relation, tail)` and incoming `(relation, head)` KG features; CFCBF concatenates
weighted interaction and content profiles. They rebuild profiles on full
training for final evaluation. This explicit feature definition replaces the
previously missing study runner and is recorded here for reproducibility.

### Outputs and resumption

Under `results/benchmark/<dataset>/<model>/`:

- `protocol.json`: search settings, data/source hashes, and final seeds;
- `search/trial_*/`: parameters, command, stdout log, and same-epoch metrics;
- `selection.json`: validation-selected configuration and one-based epoch count;
- `final/seed_*/`: fresh refit command, log, and one final test measurement;
- `summary.json`: all final seeds and aggregate metrics.

The output root also contains CSV reports scoped to the requested model/dataset
subset. Completed runs resume only when identities match. Failed runs are retried;
partial metrics are not treated as completed results. A changed dataset,
configuration, source, or seed list requires a new output directory. A lock
prevents concurrent processes from writing the same benchmark directory.

## Smoke test

From `EditKG/`, with the appropriate modern and KGAT legacy environments:

```bash
python scripts/smoke_benchmark.py --output /tmp/editkg-smoke \
  --kgat-python "$PWD/.envs/kgat-legacy/bin/python"
```

This generates synthetic interactions/KG, runs two candidates and up to two
epochs per model, then performs a fresh final fit. Its scores are checks of the
workflow, not experimental results. For a fast CPU-only subset, append
`--models bpr lightgcn sgl itemknn-cf --cpu`. Use a new output directory after
source changes; otherwise the same smoke command resumes completed results.

## Evaluation protocol and metrics

Use the complete driver above for validation-selected results. The standalone
`run_baseline.sh` keeps the native historical behavior: evaluation/early stopping
uses the selected root's `test.txt`, and the original simple/RecBole paths divide
that file into framework validation and test portions. Those legacy scores are
not the controlled final benchmark. The new driver bypasses that subdivision
and records its authoritative metrics in each run's `metrics.json`.

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
