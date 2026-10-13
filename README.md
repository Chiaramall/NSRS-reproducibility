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

Download and extract the repository archive. Use Bash on Linux and a separate
Conda environment for each study: NCR uses Python 3.7; LTNrec, CAFE and EditKG
use Python 3.12, with different dependencies.

Complete the linked installation and data preparation steps before running a
study. All commands below start from the repository root and run the complete
workflow for the models and datasets listed above. Full experiments can take
several days.

### NCR

Follow the [NCR setup](NCR/README.md#setup), including
`requirements-hyperopt.txt`. The processed datasets are included.

```bash
conda activate nsrs-ncr-3-7
bash NCR/reproduce.sh all
```

This runs baseline hyperparameter search and evaluation, plus the three NCR
variants with their recorded configurations. See the [NCR README](NCR/README.md)
for individual runs, evaluation details and outputs.

### LTNrec

Follow the [LTNrec setup](LTN/README.md#setup). The MindReader snapshot is
included; the pipeline downloads MovieLens 100K and 1M.

```bash
conda activate nsrs-ltn-3-12
bash LTN/reproduce.sh all
```

This runs the MindReader and MovieLens studies, including model selection,
training and evaluation. See the [LTNrec README](LTN/README.md) for individual
runs, evaluation details and outputs.

### CAFE

Follow the [CAFE setup](CAFE/README.md#environment) and
[data preparation guide](CAFE/baselines/DATASETS.md). The complete comparison
requires the public PGPR archive and a GPU compatible with the CUDA 12.4 stack.

```bash
conda activate nsrs-cafe-3-12
bash CAFE/reproduce.sh all
```

This prepares all four domains, evaluates CAFE and runs the baseline searches
and evaluations. See the [CAFE README](CAFE/README.md) for individual runs,
evaluation details and outputs.

### EditKG

Follow the [EditKG setup](EditKG/README.md#environment), including the separate
Python 3.7 / TensorFlow environment for KGAT. Prepared datasets and validation
splits are included; the full workflow uses an NVIDIA GPU.

```bash
conda activate nsrs-editkg-3-12
bash EditKG/reproduce.sh all
```

This runs validation-based hyperparameter search, final retraining and test
evaluation for all 16 models. See the [EditKG README](EditKG/README.md) for
individual runs, evaluation details and outputs.

## References

- Hanxiong Chen et al. [*Neural Collaborative Reasoning*](https://doi.org/10.1145/3442381.3449973). The Web Conference (WWW), 2021.
- Tommaso Carraro et al. [*Logic Tensor Networks for Top-N Recommendation*](https://ceur-ws.org/Vol-3212/paper1.pdf). International Workshop on Neural-Symbolic Learning and Reasoning (NeSy), 2022.
- Yikun Xian et al. [*CAFE: Coarse-to-Fine Neural Symbolic Reasoning for Explainable Recommendation*](https://arxiv.org/abs/2010.15620). ACM International Conference on Information and Knowledge Management (CIKM), 2020.
- Gu Tang et al. [*EditKG: Editing Knowledge Graph for Recommendation*](https://doi.org/10.1145/3626772.3657723). International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2024.
