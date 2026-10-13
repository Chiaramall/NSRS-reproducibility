# EditKG Reproducibility Study

Code and prepared datasets for EditKG and 15 baselines on Last-FM, Amazon-Book
and Yelp2018. Run all commands below from `EditKG/`.

## Environment

Create and activate the Python 3.12 environment:

```bash
conda create -y -n nsrs-editkg-3-12 python=3.12 pip
conda activate nsrs-editkg-3-12
bash scripts/install_environment.sh cu126
```

The installer installs PyTorch and the matching `torch-scatter` wheel, then
checks the dependencies. The full workflow uses an NVIDIA GPU.

KGAT requires a separate Python 3.7 / TensorFlow environment and runs on CPU.
Create it once; keep `nsrs-editkg-3-12` active for the commands that follow:

```bash
conda create -y --prefix "$PWD/.envs/kgat-legacy" python=3.7 pip
conda run --prefix "$PWD/.envs/kgat-legacy" python -m pip install \
  'tensorflow==1.15.0' 'numpy==1.18.5' 'scipy==1.4.1' \
  'scikit-learn==0.22.1' 'protobuf==3.20.3'
```

## All datasets and models

The prepared datasets and validation splits are included. Run:

```bash
GPU_ID=0 bash reproduce.sh all
```

This searches hyperparameters on validation, retrains the selected models on
full training data, and evaluates them on the test set. It covers all 16 models
on all three datasets. Defaults are up to 30 configurations, 300 epochs per
configuration, and five final seeds; ItemKNN uses six configurations and one
final fit. A complete run can take several days.

To run only one dataset:

```bash
GPU_ID=0 bash reproduce.sh all last-fm
```

## Results

Results are saved under `results/benchmark/`, including per-model
`summary.json` files and CSV reports with mean and sample standard deviation
of Recall@20 and NDCG@20. Configuration and epoch selection use validation
Recall@20; each final model is evaluated once on test.

Repeating the command resumes completed runs. If you change data, code or
settings, choose a new output directory with `OUTPUT_DIR`.

This driver implements a reproducible search from the recorded hyperparameter
spaces; it does not reconstruct the historical trial order or guarantee the
original scores. See the [detailed guide](BENCHMARK.md) for protocol details,
individual models, smaller runs and installation alternatives, and the
[data documentation](data/README.md) for dataset preparation.

## Citation

```bibtex
@inproceedings{tang2024editkg,
  title={EditKG: Editing Knowledge Graph for Recommendation},
  author={Tang, Gu and Gan, Xiaoying and Wang, Jinghe and Lu, Bin and Wu, Lyuwen and Fu, Luoyi and Zhou, Chenghu},
  booktitle={Proceedings of the 47th ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={112--122},
  year={2024}
}
```
