# The Promise of Neuro-Symbolic Recommender Systems:  Reproducibility and the Challenge of Demonstrating Progress

This repository brings together four reproducibility studies on neuro-symbolic recommender systems. Each directory is an independent project with its own environment, datasets, evaluation protocol, and documentation.

| Project | Datasets | Scope | Documentation |
| --- | --- | --- | --- |
| **NCR** — Neural Collaborative Reasoning | MovieLens 100K, Amazon Movies & TV, Amazon Electronics | NCR variants and nine recommendation baselines | [Local README](NCR/README.md) · [Standalone repository](https://github.com/Chiaramall/NCR-repro) |
| **LTNrec** — Logic Tensor Networks for recommendation | MindReader, MovieLens 100K and 1M | LTN variants and classical or neural baselines | [Local README](LTN/README.md) · [Standalone repository](https://github.com/Chiaramall/LTN-repro) |
| **CAFE** — Coarse-to-Fine Neural Symbolic Reasoning | Four Amazon product domains | CAFE reproduction and extended baseline evaluation | [Local README](CAFE/README.md) · [Standalone repository](https://github.com/Chiaramall/CAFE-repro) |
| **EditKG** — Editing Knowledge Graphs for recommendation | Last-FM, Amazon-Book, Yelp2018 | EditKG and knowledge-aware baselines | [Local README](EditKG/README.md) · [Standalone repository](https://github.com/Chiaramall/EDITKG-repro) |

## Hyperparameter configurations

Search spaces, optimization settings and selected parameters are grouped by model and dataset in one JSON per paper: [NCR](hyperparameters/NCR.json), [LTN](hyperparameters/LTN.json), [CAFE](hyperparameters/CAFE.json), [EditKG](hyperparameters/EditKG.json). Each model uses the same structure: optimization method and search ranges (`optimization`), followed by selected parameters (`best_hyperparameters`) grouped by dataset and, where applicable, training proportion or search procedure.

## Getting started

```bash
git clone https://github.com/Chiaramall/NSRS-reproducibility.git
cd NSRS-reproducibility
```

Do not install all requirements into a single environment. Enter the project you want to reproduce and follow its README.

### NCR

NCR uses a legacy environment tested with Python 3.7.12. Processed splits for all three datasets are included.

```bash
cd NCR
python -m pip install -r requirements.txt
cd src
python main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 \
  --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 \
  --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 \
  --random_seed 2022 --gpu -1
```

Install `requirements-hyperopt.txt` for baseline tuning and multi-seed evaluation. See the [NCR instructions](NCR/README.md) for the available models and datasets.

### LTNrec

LTNrec requires Python 3.10 or later. The MindReader snapshot is included; MovieLens is downloaded by the benchmark pipeline.

```bash
cd LTN
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

`main.py` runs the MindReader study. The separate MovieLens pipeline provides smoke and full configurations; see the [LTNrec instructions](LTN/README.md).

### CAFE

The original Beauty data and KG initialization are included. The other Amazon domains and some baselines require public source archives and additional data preparation.

```bash
cd CAFE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python preprocess.py --dataset beauty
python scripts/reproducibility.py all --dataset beauty --python python
```

Follow the [CAFE data guide](CAFE/baselines/DATASETS.md) before running the four-domain comparison. KGAT requires a compatible PyTorch/DGL installation.

### EditKG

The modern EditKG environment uses Python 3.12. Prepared filtered datasets and validation views are included for all three domains.

```bash
cd EditKG
python3 -m venv .envs/editkg
source .envs/editkg/bin/activate
python -m pip install -r requirements.txt
GPU_ID=0 bash scripts/run_baseline.sh editkg last-fm
```

PyTorch and `torch-scatter` must use compatible builds. KGAT requires the separate Python 3.7 / TensorFlow 1.15 environment described in the [EditKG instructions](EditKG/README.md).

## Reproducibility notes

- Run commands from the relevant project directory.
- Generated checkpoints, logs, downloaded data, and most result files are ignored by Git.
- Full studies may require multiple seeds, large searches, and several days of computation. Use the smoke or single-run configurations first when provided.
- Validation and test protocols differ across projects. Follow the selection procedure documented for each study to avoid test-set leakage.

## References

- Hanxiong Chen, Shaoyun Shi, Yunqi Li, and Yongfeng Zhang. *Neural Collaborative Reasoning*. The Web Conference, 2021.
- Tommaso Carraro. *Logic Tensor Networks for Top-N Recommendation*.
- Yikun Xian et al. *CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation*. CIKM, 2020.
- Gu Tang et al. *EditKG: Editing Knowledge Graph for Recommendation*. SIGIR, 2024.

BibTeX entries and upstream-code attribution are provided in the individual project READMEs.
