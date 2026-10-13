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
conda create -y -n nsrs-cafe-3-12 python=3.12 pip
conda activate nsrs-cafe-3-12
python -m pip install --upgrade pip
python -m pip install 'torch==2.4.0+cu124' 'torchdata==0.8.0' \
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip install 'dgl==2.4.0+cu124' \
  -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
python -m pip install -r requirements.txt
python -m pip check
```

These commands install the Python 3.12 graph stack for a GPU/driver compatible
with CUDA 12.4, including the DGL dependency used by KGAT. Keep this environment
separate from LTN and EditKG.

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

For five-seed evaluation at the fixed paper Adam settings:

```bash
python scripts/cafe_paper_reproducibility.py baseline --dataset beauty --python python
```

`reproducibility.py all` runs the CAFE-only SGD reproduction and grid;
`cafe_paper_reproducibility.py all` runs its Adam counterpart. Neither runs the
recommendation baselines. Both grids select using test-scored executions and
are historical sensitivity analyses, not validation-based model selection.

## Baselines and tuning

See [Baseline protocol and commands](baselines/README.md) for all model
launchers, fixed-parameter runs, validation-based selection and output formats.
The [implementation index](baselines/models/README.md) lists the source files.

Precomputed scores are not distributed. Runners create `results/` as needed,
including per-run metrics, selected configurations and protocol records.
Regenerate selections with the tuning commands before evaluating a selected
configuration.

## All models and datasets

After the data preparation prerequisites, run from the repository root with
`nsrs-cafe-3-12` activated:

```bash
bash CAFE/reproduce.sh prepare
bash CAFE/reproduce.sh cafe
bash CAFE/reproduce.sh baselines
# Or perform all three stages:
bash CAFE/reproduce.sh all
```

The defaults cover `beauty cell clothing cd`; append keys to restrict a stage.
`GPU_ID` selects the physical GPU and `DRY_RUN=1` previews commands. The baseline
stage runs the three validation-based tuners (including their final refits and
evaluations) plus fixed-parameter PGPR. The CAFE stage uses fixed paper settings.
The historical SGD/Adam grids are available separately with `legacy-grid`.

## References

- Yikun Xian et al. [*CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation*](https://arxiv.org/abs/2010.15620). ACM International Conference on Information and Knowledge Management (CIKM), 2020.
