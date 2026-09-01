# EditKG Reproducibility Study

This repository contains a reproducible evaluation of EditKG and its recommendation baselines on Last-FM, Amazon-Book, and Yelp2018. It keeps the original model implementation separate from baseline code, dataset artifacts, experiment configuration, and reported metrics.

## Repository layout

```text
.
├── main.py                 # EditKG training entry point
├── modules/                # EditKG model components
├── utils/                  # EditKG loading and evaluation utilities
├── baselines/
│   ├── simple.py           # BPR, LightGCN, and SGL
│   └── external/           # KG-aware baseline implementations
├── configs/                # Paper and tuning configurations
├── data/
│   ├── editkg-filtered/    # Reconstructed paper-compatible datasets
│   └── editkg-validation/  # Leakage-free validation views
├── scripts/                # Preparation, training, and tuning commands
├── results/                # Compact metrics and experiment outputs
└── docs/                   # Paper reference and reproducibility report
```

## Environment

The main environment uses Python 3.12 and PyTorch. KGAT requires its legacy Python 3.7 and TensorFlow 1.15 environment and defaults to CPU because that TensorFlow/CUDA combination is not compatible with Ampere GPUs.

```bash
python3 -m venv .envs/editkg
source .envs/editkg/bin/activate
pip install -r requirements.txt
```

For exact versions, see `environment-modern.txt` and `environment-legacy.txt`.

## Running EditKG

```bash
source .envs/editkg/bin/activate
python main.py \
  --dataset last-fm \
  --data_path data/editkg-filtered/ \
  --epoch 300 \
  --batch_size 4096 \
  --gpu_id 0 \
  --Ks '[20]'
```

## Running a baseline

```bash
GPU_ID=0 scripts/run_baseline.sh lightgcn last-fm
GPU_ID=0 scripts/run_baseline.sh kgcn amazon-book
```

Supported models are BPR, GC-MC, LightGCN, SGL, CKE, KTUP, KGCN, KGAT, KGIN, KGCL, KGRec, KRDN, and EditKG.

## Validation and tuning

Validation interactions are removed from both the optimization data and the collaborative graph. The held-out test set is not used for hyperparameter selection.

```bash
python scripts/prepare_validation_splits.py
GPU_ID=0 python scripts/run_hybrid_tpe.py
```

The hybrid protocol uses six initial configurations and up to four TPE proposals per model and dataset; KGAT uses eight observations overall. Screening is limited to 25 epochs with patience 4. The selected configuration is retrained for at most 150 epochs with patience 10 and evaluated on the test set once.

## Results

Machine-readable metrics are stored under `results/metrics`. The complete experimental discussion and comparison with the paper are in `docs/reproducibility-report.md`. Large raw logs and checkpoints are intentionally excluded.

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
