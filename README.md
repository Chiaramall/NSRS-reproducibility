# The Promise of Neuro-Symbolic Recommender Systems: Reproducibility and the Challenge of Demonstrating Progress

Code and configurations for four reproducibility studies on neuro-symbolic recommender systems. Each project has its own dependencies, datasets and evaluation protocol.

| Project | Datasets | Models | Documentation |
| --- | --- | --- | --- |
| **NCR** | MovieLens 100K, Amazon Movies & TV, Amazon Electronics | NCR, NCR-E, NCR-E-WoLR, BPR-MF, SVD++, DMF, NeuMF, GRU4Rec, GRU4RecHidasi, STAMP, SASRec, BERT4Rec | [Local README](NCR/README.md) |
| **LTNrec** | MindReader, MovieLens 100K and 1M | MF / MF-MSE, BPR-MF, LTN, LTN-genres, EASE, LightGCN, MostPopular, ItemKNN, UserKNN | [Local README](LTN/README.md) |
| **CAFE** | Amazon Beauty, Cell Phones, Clothing, CDs & Vinyl | CAFE, BPR, BPR-HFT, DeepCoNN, CKE, RippleNet, KGAT, HeteroEmbed, PGPR, ItemKNN-CF, ItemKNN-CBF, ItemKNN-CFCBF | [Local README](CAFE/README.md) |
| **EditKG** | Last-FM, Amazon-Book, Yelp2018 | EditKG, BPR, LightGCN, SGL, GCMC, CKE, KTUP, KGCN, KGAT, KGIN, KGCL, KGRec, KRDN, ItemKNN-CF, ItemKNN-CBF, ItemKNN-CFCBF | [Local README](EditKG/README.md) |

The table lists executable models included in each project. LTN model coverage differs by dataset; see its local README.

## Hyperparameter configurations

Search spaces and recorded selected hyperparameters are available for [NCR](hyperparameters/NCR.json), [LTN](hyperparameters/LTN.json), [CAFE](hyperparameters/CAFE.json) and [EditKG](hyperparameters/EditKG.json). These files are records; a launcher only loads them when explicitly documented below.

## Getting started

Download and extract the supplied repository archive, then open a terminal in its top-level directory (the directory containing this README). The commands below assume Bash on Linux and an installed Conda distribution.

Create one environment per project, even when they use the same Python version: the NumPy and PyTorch requirements differ.

```bash
conda create -y -n nsrs-ncr-3-7 python=3.7 pip
conda create -y -n nsrs-ltn-3-12 python=3.12 pip
conda create -y -n nsrs-cafe-3-12 python=3.12 pip
conda create -y -n nsrs-editkg-3-12 python=3.12 pip
```

Each section starts from the repository root. The scripts can also be invoked by absolute path from any directory. Set `DRY_RUN=1` to print their commands without starting experiments. Full searches can take several days. Training and evaluation are combined in the existing runners where noted.

### NCR

Python 3.7 is required for the pinned legacy dependencies. Processed splits for all three datasets are included.

```bash
conda activate nsrs-ncr-3-7
python -m pip install -r NCR/requirements-hyperopt.txt
# Nine baselines: validation hyperparameter search on all three datasets.
GPU_ID=0 bash NCR/reproduce.sh tune
# Retrain selected baseline configurations over five seeds and evaluate.
GPU_ID=0 bash NCR/reproduce.sh evaluate
# NCR, NCR-E and NCR-E-WoLR: recorded configurations, five seeds, train + evaluate.
GPU_ID=0 bash NCR/reproduce.sh variants
```

Alternatively, `bash NCR/reproduce.sh all` performs all three steps. Dataset keys are `ml100k01-1-5`, `5MoviesTV01-1-5`, and `Electronics01-1-5`; append one or more keys to restrict a run:

```bash
GPU_ID=-1 bash NCR/reproduce.sh all ml100k01-1-5
```

`GPU_ID=-1` selects CPU. `MAX_EVALS` defaults to 30 and `SEEDS` to `2019,2020,2021,2022,2023`. Baseline outputs go to `NCR/tuning/<dataset>` and `<dataset>_seeds`; variant metrics and summaries go to `NCR/tuning/variants/`. The variants load `hyperparameters/NCR.json`; this is a rerun of the recorded configurations, not a new variant search. Epoch selection uses validation NDCG@5.

`main.py --model_name all` is **not** supported. The `--models all` option belongs to the baseline tuning/evaluation scripts and excludes the three NCR variants. See [individual NCR commands](NCR/README.md).

### LTNrec

Use the separate Python 3.12 environment. The MindReader snapshot is included; MovieLens is downloaded by the pipeline.

```bash
conda activate nsrs-ltn-3-12
python -m pip install -r LTN/requirements.txt
# Search, training and evaluation for every included MindReader model,
# including the large-embedding study.
bash LTN/reproduce.sh mindreader
# Download MovieLens 100K/1M; run full search, training and evaluation.
bash LTN/reproduce.sh movielens
```

`bash LTN/reproduce.sh all` runs both studies. Set `DEVICE=cpu` for the MovieLens CPU pipeline. The full MovieLens profile specifies 5,580 validation fits and 1,800 final fits. Outputs are under `LTN/results/` and `LTN/benchmark_outputs/`. For separate stages, smoke runs and resumption details, see [LTNrec instructions](LTN/README.md).

### CAFE

Dataset keys are `beauty`, `cell`, `clothing`, and `cd`. Beauty's graph and initialization are included. **Before running the experiments**, download/extract the public PGPR archive and create the data links in the [data guide](CAFE/baselines/DATASETS.md); the four-domain baseline comparison needs this archive, including for Beauty.

```bash
conda activate nsrs-cafe-3-12
python -m pip install --upgrade pip
# Compatible Linux / CUDA 12.4 graph stack, including KGAT's DGL dependency.
python -m pip install 'torch==2.4.0+cu124' 'torchdata==0.8.0' \
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip install 'dgl==2.4.0+cu124' \
  -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
python -m pip install -r CAFE/requirements.txt
python -m pip check
# Prepare all four domains and missing CAFE initializations.
GPU_ID=0 bash CAFE/reproduce.sh prepare
# CAFE paper parameters: five-seed training, inference and evaluation.
GPU_ID=0 bash CAFE/reproduce.sh cafe
# Baselines: validation search, final retraining/evaluation; PGPR uses fixed settings.
GPU_ID=0 bash CAFE/reproduce.sh baselines
```

`bash CAFE/reproduce.sh all` performs those three stages. Append dataset keys to restrict any stage, e.g. `bash CAFE/reproduce.sh all beauty`. This setup requires a GPU/driver compatible with CUDA 12.4. Results are written under `CAFE/results/`.

The original `scripts/reproducibility.py all` runs the **CAFE-only SGD reproduction and grid**, not every model. The Adam counterpart is `cafe_paper_reproducibility.py`. Their grids use test-scored executions and are historical sensitivity analyses, not validation-selected final results. They remain available through `bash CAFE/reproduce.sh legacy-grid`; they are excluded from the default `all` workflow. The [baseline instructions](CAFE/baselines/README.md) describe the validation-based baseline searches and metrics.

### EditKG

Use Python 3.12 for the modern stack. Install PyTorch first and then a matching prebuilt `torch-scatter` wheel; installing everything together from an unpinned requirements file can fail with `ModuleNotFoundError: No module named 'torch'` during isolated compilation.

```bash
conda activate nsrs-editkg-3-12
# Installs Torch 2.7.0 / CUDA 12.6, matching scatter, and the remaining dependencies.
bash EditKG/scripts/install_environment.sh cu126
# One-model validation run using the activated Conda environment.
EDITKG_PYTHON="$(command -v python)" \
  EDITKG_DATA_ROOT="$PWD/EditKG/data/editkg-validation" \
  GPU_ID=0 bash EditKG/scripts/run_baseline.sh editkg last-fm
```

The installer also accepts `cpu` for CPU installation/import checks; this does not make all native GPU runners CPU-compatible. See [installation details](EditKG/README.md#environment).

KGAT needs its **own** Python 3.7 / TensorFlow environment. Create it before running the all-model script (KGAT defaults to CPU):

```bash
conda create -y --prefix "$PWD/EditKG/.envs/kgat-legacy" python=3.7 pip
conda run --prefix "$PWD/EditKG/.envs/kgat-legacy" python -m pip install \
  'tensorflow==1.15.0' 'numpy==1.18.5' 'scipy==1.4.1' \
  'scikit-learn==0.22.1' 'protobuf==3.20.3'
# All 16 models, all three datasets: validation search, final refit and evaluation.
GPU_ID=0 bash EditKG/reproduce.sh all
```

Dataset keys are `last-fm`, `amazon-book`, and `yelp2018`; append keys to restrict a run. The complete driver uses the recorded search spaces in `hyperparameters/EditKG.json`, selects parameters and a one-based epoch count on **validation Recall@20** (ties: NDCG@20), retrains from scratch on the full training split for exactly that duration, and evaluates the full held-out test split **once**, at the end. Recall and NDCG always come from the same epoch.

Run stages separately or preview the workload:

```bash
bash EditKG/reproduce.sh plan
bash EditKG/reproduce.sh search
bash EditKG/reproduce.sh final
bash EditKG/reproduce.sh report
```

Defaults: up to 30 candidate configurations per model/dataset, 300 training epochs per candidate, training seed 2024, and final seeds 2020–2024. ItemKNN uses its six recorded joint configurations and one deterministic final fit. The driver tries the recorded explicit baseline candidates first, then uses reproducible categorical random search; it starts a new search rather than recreating unavailable historical TPE trial states. Set `TRIALS`, `EPOCHS`, `FINAL_SEEDS` (space separated), and `OUTPUT_DIR` to change these defaults; keep settings identical across separate stages. See [the driver guide](EditKG/BENCHMARK.md#complete-search-and-final-evaluation) for model subsets and direct CLI options.

Per-trial commands, logs, selections, per-seed results, mean/sample-standard-deviation summaries and CSV reports are saved under `EditKG/results/benchmark/`. Compatible completed runs are resumed; changes to data, code or protocol require a fresh output directory. Concurrent runs cannot share an output directory.

The previous fixed-parameter runs remain available as `validation` and `historical-test`. They apply `configs/paper_hyperparameters.tsv` to the twelve original baselines and launcher defaults to EditKG. `historical-test` preserves the original test-scored stopping behavior for comparison; use the new `all` / `search` / `final` pipeline for validation-selected results.

## Reproducibility notes

- `DRY_RUN=1 bash NCR/reproduce.sh all` (and the corresponding scripts for the other studies) previews commands without training or downloading datasets.
- Activate the project's environment before running its script; `PYTHON` can override the interpreter.
- Use fresh output directories when changing settings; some original runners resume by filename.
- Follow each study's validation/test protocol. Historical test-scored grids are not validation hyperparameter searches.

## References

- Hanxiong Chen, Shaoyun Shi, Yunqi Li, and Yongfeng Zhang. *Neural Collaborative Reasoning*. The Web Conference, 2021.
- Tommaso Carraro. *Logic Tensor Networks for Top-N Recommendation*.
- Yikun Xian et al. *CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation*. CIKM, 2020.
- Gu Tang et al. *EditKG: Editing Knowledge Graph for Recommendation*. SIGIR, 2024.
