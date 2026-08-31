# CAFE paper baselines on four Amazon datasets

This directory separates two different artifacts:

`paper_table2_all_datasets.csv` contains the Table 2 reference percentages for
all four datasets; local runs remain separate under `results/paper_baselines/`
and use the exact CAFE/PGPR split with full-ranking top-10 evaluation.

The paper compares CAFE with eight baselines:

- MF/review: BPR, BPR-HFT, DeepCoNN.
- KG embedding: CKE, RippleNet, KGAT.
- Path reasoning: HeteroEmbed and PGPR.

## Data protocol

The CAFE labels and the review-bearing PGPR data have exactly the same
149,844 train pairs and 48,658 test pairs for 22,363 users. The converter
preserves those pairs and creates RecBole atomic files:

```bash
.venv/bin/python scripts/prepare_paper_baselines.py
```

The generated `valid` file is intentionally empty. The runner trains for a
fixed number of epochs without early stopping and evaluates only once on the
untouched 30% test split. This avoids using test data for model selection.

The three non-Beauty datasets can be verified and converted from the official
PGPR files with:

```bash
.venv-baselines/bin/python scripts/prepare_all_amazon_datasets.py \
  --datasets cell clothing cd
```

Beauty remains the public CAFE dataset. The converter creates both CAFE-format
folders (`data/Cellphones_Accessories`, `data/Clothing`, `data/CDs_Vinyl`)
and RecBole tokens (`cell_paper`, `clothing_paper`, `cds_paper`). It uses only
training review text, applies the official TF-IDF and frequency filters, writes
inverse edges explicitly, and records source/split hashes in each metadata file. Reference results for
all datasets are in `paper_table2_all_datasets.csv`.

## Runnable common-protocol models

The complete implementation index is `baselines/models/README.md`; scripts in
`scripts/` are launchers and experiment utilities.


The RecBole 1.2.1 implementations used for BPR, CKE, RippleNet and KGAT are
exposed directly under `baselines/models/`. The supporting RecBole engine is
bundled separately under `baselines/recbole_core/`. Install its runtime
dependencies, then run:

```bash
python scripts/run_paper_recbole.py \
  --models BPR CKE RippleNet KGAT \
  --seeds 123 124 125 126 127
```

The runner imports the four files in `baselines/models/` through the supporting
engine. The only model-source change is a DGL 2.x API compatibility update in
KGAT; equations and objectives are otherwise those of RecBole 1.2.1. The MIT
license is retained in `baselines/recbole_core/LICENSE`.

HeteroEmbed uses the authors released objective via the modern PyTorch port:

```bash
.venv-baselines/bin/python scripts/run_paper_heteroembed.py \
  --datasets beauty \
  --seeds 123 124 125 126 127 \
  --epochs 20 \
  --learning-rate 0.5 \
  --negative-samples 5
```

## Review-aware common-protocol models

The CAFE paper does not publish the code, hyperparameters, or top-N
adaptations used for BPR-HFT and DeepCoNN. This repository therefore provides
explicit common-protocol reimplementations rather than claiming exact
reproduction of those two paper rows:

```bash
.venv-baselines/bin/python scripts/run_review_baselines.py \
  --dataset beauty \
  --models BPR-HFT DeepCoNN \
  --seeds 123 124 125 126 127
```

Both models consume only the tokenized training reviews in the PGPR files and
use the same untouched 30% test split and full-ranking evaluator as the other
local baselines. BPR-HFT combines pairwise BPR with an HFT-style topic loss.
DeepCoNN uses separate CNN encoders for user and item review documents and a
pairwise top-N objective. Exact architectural and protocol choices are stored
beside every result in `protocol.json`; paper reference numbers remain
separate.

PGPR is included under `baselines/models/pgpr/` and uses the official
implementation at commit `df6aac19c3adfb056f42abb9e5af44fc3fdaa0d5`
through the modern launcher:

```bash
python scripts/run_paper_pgpr.py --dataset beauty \
  --seeds 123 124 125 126 127 --gpu 0
```

The launcher runs preprocessing once, then TransE (30 epochs), the RL agent
(50 epochs), and top-10 evaluation sequentially for each seed. It resumes from
completed artifacts and writes per-seed values plus mean and sample standard
deviation under `results/paper_baselines/<dataset>/pgpr/`. `--add-products` is enabled
so every user receives ten recommendations; use `--no-add-products` only to
replicate the legacy evaluator that drops users with fewer than ten paths.
