# CAFE paper baselines on four Amazon datasets

The paper baselines are BPR, BPR-HFT, DeepCoNN, CKE, RippleNet, KGAT,
HeteroEmbed and PGPR. ItemKNN variants extend this comparison.

## Evaluation protocol

First follow [Data preparation](DATASETS.md). All commands below run from the
repository root with the environment in the main README activated.

Fixed-parameter baseline runners use the public training interactions and
evaluate on the outer test split. The generated outer `valid` file is empty;
these runs do not perform validation-based hyperparameter selection.

The tuning commands instead create an inner validation split from the public
training set, select hyperparameters there, and refit on the full training
set before final outer-test evaluation. Never select configurations by their
outer-test scores.

Local baseline evaluation ranks unseen candidate items and reports NDCG@10,
Recall@10, HR@10 and Precision@10 as percentages. PGPR obtains candidates from
paths and optionally fills missing positions, as described below.

## Runnable common-protocol models

See [Model implementations](models/README.md) for the source files. Install the
RecBole runtime dependencies, then run:

```bash
python scripts/run_paper_recbole.py \
  --models BPR CKE RippleNet KGAT \
  --seeds 123 124 125 126 127
```

Run HeteroEmbed with:

```bash
python scripts/run_paper_heteroembed.py \
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
python scripts/run_review_baselines.py \
  --dataset beauty \
  --models BPR-HFT DeepCoNN \
  --seeds 123 124 125 126 127
```

Both models use training reviews and the shared full-ranking evaluator.

Run PGPR with:

```bash
python scripts/run_paper_pgpr.py --dataset beauty \
  --seeds 123 124 125 126 127 --gpu 0
```

The launcher runs preprocessing once, then TransE (30 epochs), the RL agent
(50 epochs), and top-10 evaluation sequentially for each seed. It resumes from
completed artifacts and writes per-seed values plus mean and sample standard
deviation under `results/paper_baselines/<dataset>/pgpr/`. `--add-products` is enabled
to fill missing positions with scored unseen products; use `--no-add-products` only to
replicate the legacy evaluator that drops users with fewer than ten paths.

## ItemKNN extensions

```bash
python scripts/run_knn_hybrids.py --datasets beauty cell clothing cd
```

See [ItemKNN methodology](ITEMKNN_HYBRIDS.md) for features, similarity and
validation isolation. [HeteroEmbed commands](HETEROEMBED.md) cover execution and tuning.

## Hyperparameter selection

```bash
python scripts/tune_paper_baselines.py --datasets beauty cell clothing cd
python scripts/tune_paper_heteroembed.py --datasets beauty cell clothing cd
python scripts/tune_knn_hybrids.py --datasets beauty cell clothing cd
```

Use `--help` to inspect budgets and defaults. The first command covers the
RecBole and review-aware models; HeteroEmbed and ItemKNN have dedicated tuners.
PGPR uses the fixed settings documented above.

## Generated outputs

Runners write local metrics and protocol records below `results/`; tuners also
write selected configurations there.
