# CAFE Reproducibility and Extended Evaluation

Code for reproducing **CAFE: Coarse-to-Fine Neural Symbolic Reasoning for
Explainable Recommendation** (CIKM 2020) and comparing it with recommendation
baselines on Beauty, Cell Phones, Clothing and CDs.

## Included models

- Original CAFE neural-symbolic recommender.
- BPR, CKE, RippleNet and KGAT using RecBole.
- BPR-HFT and DeepCoNN using review text.
- HeteroEmbed, implemented as a PyTorch port of the authors' released
  CFKG/KBE4ER objective.
- Official PGPR launcher.
- ItemKNN-CF, ItemKNN-CBF and ItemKNN-CFCBF.

BPR-HFT and DeepCoNN are reimplementations evaluated with the same protocol
as the other local baselines. The CAFE paper did not release their exact
top-N code and hyperparameters.

## Repository layout

- `data/Beauty/`: original public CAFE Beauty data and KG initialization.
- `data/<domain>/preprocessing_metadata.json`: preprocessing metadata for the
  additional domains. Generate their data files with the conversion scripts.
- `scripts/`: preprocessing, evaluation and hyperparameter-selection tools.
- `baselines/`: protocol documentation, dependencies and every baseline
  implementation under `baselines/models/`; RecBole support code is under
  `baselines/recbole_core/`.
- `results/`: experiment outputs, generated locally and ignored by Git.
- `tests/`: regression tests for the added baseline implementations.

## Environment

Run commands from `CAFE/`. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for the local CUDA version. KGAT also
requires a compatible DGL build; the tested versions are documented in
`requirements.txt`.

## Data preparation

Beauty's CAFE graph and initialization are included. Baselines also require
the public PGPR review data; the other domains require conversion. Follow
[Data preparation](baselines/DATASETS.md) first, including archive extraction
and PGPR data links. The converters do not download the source archive.

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

See [Baseline protocol and commands](baselines/README.md) for all model
launchers, fixed-parameter runs, validation-based selection and output formats.
The [implementation index](baselines/models/README.md) lists the source files.

Precomputed scores are not distributed. Runners create `results/` as needed,
including per-run metrics, selected configurations and protocol records.
Regenerate selections with the tuning commands before evaluating a selected
configuration.

## References

1. Yikun Xian et al. *CAFE: Coarse-to-Fine Neural Symbolic Reasoning for
   Explainable Recommendation*. CIKM 2020.

