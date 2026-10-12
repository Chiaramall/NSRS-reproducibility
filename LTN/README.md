# LTNrec

Experiments on Logic Tensor Networks for top-N recommendation with MindReader
and MovieLens. The code builds on the original LTNrec implementation by
Tommaso Carraro (`tommasocarraro/LTNrec`).

## Setup

Use Python 3.10 or later on Linux. Run commands from `LTN/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency versions are taken from the development environment.

## Experiments

| Dataset | Models | Evaluation |
| --- | --- | --- |
| MindReader | MF, LTN, LTN-genres, EASE, LightGCN, MostPopular, ItemKNN, UserKNN | Supplied split, one positive and 100 negatives |
| MovieLens 100K and 1M | MF-MSE, BPR-MF, EASE, LightGCN, LTN, LTN-genres | Temporal split, one positive and 100 fixed negatives |

Both studies use training proportions of 100%, 80%, 60%, 40% and 20%,
with hit@10 and ndcg@10. Their data splits and genre signals differ:
MindReader contains direct genre feedback; MovieLens genre preferences are
inferred from training ratings. See [MovieLens methods](movielens_paper_benchmark/METHODS.md).

### MindReader

The required `mr_ntp_all_entities` snapshot is included under `dataset/`.
The movie-only alternative is not used by these experiments.

```bash
python main.py
python mindreader_large_embeddings.py --stage all
python mindreader_baselines.py --stage all
python mindreader_classic_baselines.py --stage all
```

`main.py` performs grid search with `config/models_config_grid_search.json`
and evaluates MF, LTN and LTN-genres over 30 seeds. Options include
`--models`, `--proportions`, `--runs`, `--seed`, `--output` and `--model-dir`.
Use a new model directory when changing the search configuration: the original
runner reuses checkpoints by filename and does not validate configuration changes.

The large-embedding study compares each selected control configuration against
64, 128 and 256 dimensions, using three validation seeds and 30 final seeds.
Controls are stored once in `config/mindreader_controls.json`. When the control
wins, the script reuses its recorded final metrics from [results](results/README.md).
Other selected configurations are trained again.

The two baseline scripts cover EASE/LightGCN and MostPopular/ItemKNN/UserKNN,
respectively. Each script exposes its options with `--help`.

### MovieLens

```bash
python -m movielens_paper_benchmark.download --datasets ml-100k ml-1m --root data/movielens
python -m movielens_paper_benchmark --config movielens_paper_benchmark/configs/smoke.json --stage all --device auto
python -m movielens_paper_benchmark --config movielens_paper_benchmark/configs/full.json --stage plan --details
python -m movielens_paper_benchmark --config movielens_paper_benchmark/configs/full.json --stage all --device auto
```

The smoke profile checks the pipeline with short training runs. The full profile
specifies 5,580 validation fits and 1,800 final fits and can take several days.
Use `--stage status --details` to inspect progress and `--stage report` to
regenerate reports. Repeating the same command resumes compatible saved state.
Outputs are written under `benchmark_outputs/` and excluded from Git.

MovieLens evaluation uses 100 sampled negatives per positive item.

## Layout

- `ltnrec/`: MindReader data loaders, models, training and metrics.
- `movielens_paper_benchmark/`: MovieLens pipeline and shared EASE/LightGCN models.
- `config/`: MindReader search space and selected control configurations.
- `dataset/`: MindReader snapshot used by the experiments.
- `results/`: recorded MindReader control metrics required by the embedding study.
- `tests/`: additional model tests; MovieLens tests live with the pipeline.

## Tests

```bash
python -m pytest -q
```
