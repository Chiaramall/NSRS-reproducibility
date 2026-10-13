# NCR

Research code for experiments with Neural Collaborative Reasoning (NCR), explicit-feedback variants and recommendation baselines. Based on the NCR implementation by Hanxiong Chen, Shaoyun Shi, Yunqi Li and Yongfeng Zhang (WWW 2021).

## Setup

The code was checked with Python 3.7.12 and the versions in `requirements.txt`. Run the setup commands from `NCR/`.

```bash
conda create -y -n nsrs-ncr-3-7 python=3.7 pip
conda activate nsrs-ncr-3-7
python -m pip install -r requirements.txt
```

For hyperparameter search and evaluation across seeds:

```bash
python -m pip install -r requirements-hyperopt.txt
```

This is a legacy environment. Compatibility with newer Python versions has not been verified.

## Layout

- `src/`: models, data processing, training and ranking metrics.
- `dataset/`: processed MovieLens 100K, Amazon Movies & TV and Electronics splits.
- `scripts/tune.py`: shared Hyperopt search for nine baselines.
- `scripts/evaluate.py`: evaluate selected configurations across seeds.
- `scripts/*_config.py`: model-specific search spaces and arguments.
- `scripts/common.py`: shared command construction and metric parsing.

Training creates `log/`, `model/` and `result/`. Search outputs are written to `tuning/`. These directories and regenerated dataset metadata are ignored by Git.

## Train NCR

Run training from `src/` so that the dataset and output paths resolve correctly.

```bash
cd src
python main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 \
  --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 \
  --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 \
  --random_seed 2022 --gpu 0
```

Use `--gpu -1` for CPU. Dataset names are `ml100k01-1-5`, `5MoviesTV01-1-5` and `Electronics01-1-5`.

Model identifiers include `NCR`, `NCR-E`, `NCR-E-WoLR`, `BPR-MF`, `SVD++`, `DMF`, `NeuMF`, `GRU4Rec`, `GRU4RecHidasi`, `STAMP`, `SASRec` and `BERT4Rec`. NCR-E uses explicit feedback; NCR-E-WoLR disables its logic regularizer. Historical command-line aliases are retained.

## Baseline experiments

From `NCR/`:

```bash
python scripts/tune.py --models all --dataset ml100k01-1-5 \
  --max_evals 30 --gpu 0 --output_dir tuning/ml100k
python scripts/evaluate.py --models all --dataset ml100k01-1-5 \
  --best_dir tuning/ml100k --output_dir tuning/ml100k_seeds \
  --seeds 2019,2020,2021,2022,2023 --gpu 0
```

`all` includes the nine baselines, excluding the NCR variants. Use a separate output directory for each dataset and protocol. Evaluation resumes completed runs from their saved metrics; use a new directory when changing settings.

Both the training epoch and the hyperparameter configuration are selected by validation NDCG@5. Metrics are stored in the order NDCG@5, NDCG@10, HR@5, HR@10; test scores do not determine the selected configuration. Evaluation reports the mean and sample standard deviation across seeds.

## All datasets and models

From the repository root, after activating `nsrs-ncr-3-7` and installing
`NCR/requirements-hyperopt.txt`:

```bash
bash NCR/reproduce.sh tune
bash NCR/reproduce.sh evaluate
bash NCR/reproduce.sh variants
# Or perform all three stages:
bash NCR/reproduce.sh all
```

Append dataset keys to restrict the run. `GPU_ID=-1` selects CPU;
`DRY_RUN=1` previews commands. The first two stages use the nine baseline
runners above. The `variants` stage trains and evaluates NCR, NCR-E and
NCR-E-WoLR over five seeds using `hyperparameters/NCR.json`, filling omitted
fields with the explicit common settings in `scripts/run_variants.py`.
It writes per-seed metrics and mean/sample-standard-deviation summaries under
`tuning/variants/<dataset>/<model>/`. Hyperparameters are fixed to the recorded
values; epochs are selected on validation. To inspect individual variant
commands without training, run from `NCR/`:

```bash
python scripts/run_variants.py --dataset ml100k01-1-5 --dry-run
```

`main.py --model_name all` is not supported. Use the wrapper to cover the
baseline and NCR model families together.

## Reference

```bibtex
@inproceedings{chen2021neural,
  title={Neural Collaborative Reasoning},
  author={Chen, Hanxiong and Shi, Shaoyun and Li, Yunqi and Zhang, Yongfeng},
  booktitle={Proceedings of the Web Conference 2021},
  pages={1516--1527},
  year={2021}
}
```

