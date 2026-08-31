#!/usr/bin/env python3
"""Run common CAFE-paper baselines with one full-ranking evaluator.

This is a modern common-protocol reimplementation. Paper-reported reference
numbers remain separate in baselines/paper_table2_all_datasets.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECBOLE_ROOT = ROOT / "baselines" / "recbole_core"
ATOMIC_ROOT = ROOT / "baselines" / "data"
DATASET_TOKENS = {
    "beauty": "beauty_paper",
    "cell": "cell_paper",
    "clothing": "clothing_paper",
    "cd": "cds_paper",
}
SUPPORTED_MODELS = ("BPR", "CKE", "RippleNet", "KGAT")
DEFAULT_MODELS = ("BPR", "CKE", "RippleNet", "KGAT")
PAPER_NAMES: dict[str, str] = {}
METRIC_KEYS = {
    "ndcg": "ndcg@10",
    "recall": "recall@10",
    "hr": "hit@10",
    "precision": "precision@10",
}


def import_recbole() -> None:
    if not RECBOLE_ROOT.is_dir():
        raise RuntimeError(
            "RecBole engine missing at baselines/recbole_core"
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(RECBOLE_ROOT))


def ensure_data(args: argparse.Namespace) -> None:
    data_dir = ATOMIC_ROOT / args.dataset_token
    token = args.dataset_token
    required = (
        data_dir / f"{token}.train.inter",
        data_dir / f"{token}.valid.inter",
        data_dir / f"{token}.test.inter",
        data_dir / f"{token}.link",
        data_dir / f"{token}.kg",
        data_dir / "metadata.json",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Run the paper-dataset preparation script first; missing: "
            + ", ".join(missing)
        )


def config_for(args: argparse.Namespace, model: str, seed: int) -> dict:
    checkpoint_dir = args.results_dir / "checkpoints" / model / f"seed_{seed}"
    load_col = {"inter": ["user_id", "item_id"]}
    if model != "BPR":
        load_col.update(
            {
                "link": ["item_id", "entity_id"],
                "kg": ["head_id", "relation_id", "tail_id"],
            }
        )
    return {
        "data_path": str(ATOMIC_ROOT),
        "benchmark_filename": ["train", "valid", "test"],
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "HEAD_ENTITY_ID_FIELD": "head_id",
        "TAIL_ENTITY_ID_FIELD": "tail_id",
        "RELATION_ID_FIELD": "relation_id",
        "ENTITY_ID_FIELD": "entity_id",
        "load_col": load_col,
        "kg_reverse_r": False,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learner": "adam",
        "learning_rate": args.learning_rate,
        "embedding_size": args.embedding_size,
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        },
        "eval_args": {
            "group_by": "user",
            "order": "RO",
            "split": None,
            "mode": "full",
        },
        "metrics": ["NDCG", "Recall", "Hit", "Precision"],
        "topk": [10],
        "valid_metric": "NDCG@10",
        "eval_step": 0,
        "seed": seed,
        "reproducibility": True,
        "use_gpu": not args.cpu,
        "gpu_id": args.gpu,
        "checkpoint_dir": str(checkpoint_dir),
        "show_progress": True,
        "save_dataset": False,
        "save_dataloaders": False,
    }


def run_once(args: argparse.Namespace, model_name: str, seed: int) -> dict:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.transform import construct_transform
    from recbole.trainer import Trainer
    from recbole.utils import (
        get_model,
        get_trainer,
        init_logger,
        init_seed,
    )

    config = Config(
        model=model_name,
        dataset=args.dataset_token,
        config_dict=config_for(args, model_name, seed),
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    logger.info("Paper baseline model=%s seed=%s", model_name, seed)
    dataset = create_dataset(config)
    train_data, _empty_valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(
        config["device"]
    )
    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer: Trainer = trainer_class(config, model)
    trainer.fit(
        train_data,
        valid_data=None,
        saved=True,
        show_progress=config["show_progress"],
    )
    result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )
    row: dict[str, object] = {
        "dataset": args.dataset,
        "model": PAPER_NAMES.get(model_name, model_name),
        "runner_model": model_name,
        "seed": seed,
    }
    for output_name, recbole_name in METRIC_KEYS.items():
        row[output_name] = 100.0 * float(result[recbole_name])
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        summary: dict[str, object] = {
            "model": model,
            "seeds": len(model_rows),
        }
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in model_rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_TOKENS), default="beauty"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[123, 124, 125, 126, 127]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory; use a separate path for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_token = DATASET_TOKENS[args.dataset]
    if args.results_dir is None:
        args.results_dir = (
            ROOT / "results" / "paper_baselines" / args.dataset / "recbole"
        )
    args.results_dir = args.results_dir.resolve()
    ensure_data(args)
    import_recbole()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.results_dir / "per_seed.csv"
    rows: list[dict[str, object]] = []
    if raw_path.exists():
        with raw_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    completed = {(str(row["runner_model"]), int(row["seed"])) for row in rows}
    for model in args.models:
        for seed in args.seeds:
            if (model, seed) in completed:
                print(f"Skipping completed {model} seed={seed}", flush=True)
                continue
            row = run_once(args, model, seed)
            rows.append(row)
            write_csv(raw_path, rows)
            summary = summarize(rows)
            write_csv(args.results_dir / "summary.csv", summary)
            (args.results_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
