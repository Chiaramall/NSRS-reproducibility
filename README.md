# CAFE Reproducibility and Extended Evaluation

This repository provides a cleaned, reproducible research snapshot of
**CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable
Recommendation** (CIKM 2020), together with the local reproducibility study
and extended baseline evaluation on the four Amazon domains used in the
paper.

## Included models

- Original CAFE neural-symbolic recommender.
- BPR, CKE, RippleNet and KGAT through the declared RecBole protocol.
- BPR-HFT and DeepCoNN common-protocol review-aware implementations.
- HeteroEmbed, implemented as a PyTorch port of the authors' released
  CFKG/KBE4ER objective.
- Official PGPR launcher.
- ItemKNN-CF, ItemKNN-CBF and ItemKNN-CFCBF.

BPR-HFT and DeepCoNN are reported as common-protocol reimplementations
because the CAFE paper did not release their exact top-N code and hyperparameters.

## Repository layout

- `data/Beauty/`: original public CAFE Beauty data and KG initialization.
- `data/<domain>/preprocessing_metadata.json`: validated metadata for the
  additional domains. Their generated raw files are intentionally excluded.
- `scripts/`: preprocessing, evaluation and hyperparameter-selection tools.
- `baselines/`: protocol documentation, dependencies and every baseline
  implementation under `baselines/models/`; RecBole support code is under
  `baselines/recbole_core/`.
- `results/`: compact CSV/JSON results and selected configurations. Training
  logs, caches and checkpoints are not included.
- `tests/`: regression tests for the added baseline implementations.

## Environment

Create a clean environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for the local CUDA version. KGAT also
requires a compatible DGL build; the tested versions are documented in
`requirements.txt`.

Every evaluated baseline implementation is centralized in `baselines/models/`,
including PGPR, HeteroEmbed, the review-aware models and ItemKNN extensions.
The supporting RecBole engine and its upstream license are kept separately in
`baselines/recbole_core/`.

## Data preparation

Beauty is distributed with the original CAFE repository. The other three
domains are reconstructed from the public PGPR Amazon archive:

```bash
python scripts/prepare_all_amazon_datasets.py \
  --datasets cell clothing cd --threads 4
```

The converter preserves the official split, prevents train/test overlap,
uses training review text only and records hashes and counts in the generated
metadata. See [`baselines/DATASETS.md`](baselines/DATASETS.md).

## Running CAFE

```bash
python preprocess.py --dataset beauty
python train_neural_symbol.py --dataset beauty --name cafe_beauty
python execute_neural_symbol.py \
  --dataset beauty --name cafe_beauty --do_infer true
python execute_neural_symbol.py \
  --dataset beauty --name cafe_beauty --do_execute true
```

For five-seed reproducibility and compact grid evaluation:

```bash
python scripts/reproducibility.py all --dataset beauty --python python
```

## Baselines and tuning

Examples:

```bash
python scripts/run_paper_recbole.py \
  --dataset beauty --models BPR CKE RippleNet KGAT

python scripts/run_review_baselines.py \
  --dataset beauty --models BPR-HFT DeepCoNN

python scripts/run_knn_hybrids.py \
  --datasets beauty cell clothing cd

python scripts/run_paper_pgpr.py \
  --dataset beauty --seeds 123 124 125 126 127 --gpu 0

python scripts/tune_paper_baselines.py \
  --datasets beauty cell clothing cd
```

Hyperparameters are selected using an inner validation split derived only
from public training interactions. The outer test set is used only after
selection. The full protocol is described in `baselines/README.md` and stored
beside the result summaries.

## Results

Local outputs and published comparison values are stored under `results/`.
The consolidated tables are:

- `results/extended_evaluation/baseline_comparison.csv`;
- `results/extended_evaluation/cafe_results.csv`;
- `results/extended_evaluation/itemknn_comparison.csv`.

The provenance file stored with the consolidated tables distinguishes results
verified against repository artifacts from values imported from the supplied
study summary.

Metrics are reported as percentages for NDCG@10, Recall@10, HR@10 and
Precision@10.

## References

1. Yikun Xian et al. *CAFE: Coarse-to-Fine Neural Symbolic Reasoning for
   Explainable Recommendation*. CIKM 2020.
2. Qingyao Ai et al. *Learning Heterogeneous Knowledge Base Embeddings for
   Explainable Recommendation*. Algorithms 2018.
