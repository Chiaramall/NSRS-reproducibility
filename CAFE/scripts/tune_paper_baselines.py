#!/usr/bin/env python3
"""Small, resumable TPE tuning for implemented CAFE-paper baselines.

Protocol:
1. create a deterministic 90/10 inner split from the public training set;
2. select hyperparameters by inner-validation NDCG@10 only;
3. retrain the selected configuration on the complete public training set;
4. report five-seed metrics on the untouched public test set.

PGPR is intentionally not included because its official released runner has no
separate validation path. HeteroEmbed uses a dedicated authors-code port and is tuned separately from the RecBole models.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import logging
import os
import pickle
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECBOLE_ROOT = ROOT / "baselines" / "recbole_core"
ATOMIC_ROOT = ROOT / "baselines" / "data"
OUTPUT_ROOT = ROOT / "results" / "baseline_hyperopt"
LOCK = OUTPUT_ROOT / "pipeline.lock"
STATUS = OUTPUT_ROOT / "status.json"
PROTOCOL_VERSION = 2
DATASET_TOKENS = {
    "beauty": "beauty_paper",
    "cell": "cell_paper",
    "clothing": "clothing_paper",
    "cd": "cds_paper",
}
RECBOLE_MODELS = ("BPR", "CKE", "RippleNet", "KGAT")
REVIEW_MODELS = ("BPR-HFT", "DeepCoNN")
MODELS = RECBOLE_MODELS + REVIEW_MODELS
METRICS = ("ndcg", "recall", "hr", "precision")
RECBOLE_METRICS = {
    "ndcg": "ndcg@10",
    "recall": "recall@10",
    "hr": "hit@10",
    "precision": "precision@10",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_status(**values: object) -> None:
    state: dict[str, object] = {}
    if STATUS.exists():
        state = json.loads(STATUS.read_text(encoding="utf-8"))
    state.update(values)
    state["updated_at_epoch"] = time.time()
    atomic_json(STATUS, state)


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def slug(model: str) -> str:
    return model.lower().replace("-", "_")


def configuration_digest(value: object, length: int = 12) -> str:
    serialized = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:length]


def selection_fingerprint(
    args: argparse.Namespace, dataset: str, model_name: str
) -> tuple[str, dict[str, object]]:
    token = f"{DATASET_TOKENS[dataset]}_tune"
    metadata = json.loads(
        (ATOMIC_ROOT / token / "metadata.json").read_text(encoding="utf-8")
    )
    tuning_epochs = (
        args.kgat_tuning_epochs
        if model_name == "KGAT"
        else args.tuning_review_epochs
        if model_name in REVIEW_MODELS
        else args.tuning_epochs
    )
    identity: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "model": model_name,
        "inner_train_sha256": metadata["inner_train_sha256"],
        "inner_valid_sha256": metadata["inner_valid_sha256"],
        "split_seed": metadata["split_seed"],
        "tuning_seed": args.tuning_seed,
        "tuning_epochs": tuning_epochs,
        "eval_step": args.eval_step,
        "stopping_step": args.stopping_step,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }
    return configuration_digest(identity, length=16), identity


def recbole_config(
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    token: str,
    seed: int,
    params: dict[str, Any],
    result_dir: Path,
    *,
    selection: bool,
) -> dict[str, Any]:
    load_col = {"inter": ["user_id", "item_id"]}
    if model_name != "BPR":
        load_col.update(
            {
                "link": ["item_id", "entity_id"],
                "kg": ["head_id", "relation_id", "tail_id"],
            }
        )
    epochs = args.kgat_tuning_epochs if (
        selection and model_name == "KGAT"
    ) else args.tuning_epochs if selection else args.final_recbole_epochs
    config: dict[str, Any] = {
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
        "epochs": epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learner": "adam",
        "learning_rate": float(params["learning_rate"]),
        "embedding_size": int(params["embedding_size"]),
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
        "eval_step": args.eval_step if selection else 0,
        "stopping_step": args.stopping_step,
        "seed": seed,
        "reproducibility": True,
        "use_gpu": not args.cpu,
        "gpu_id": args.gpu,
        "checkpoint_dir": str(result_dir / "checkpoints" / f"seed_{seed}"),
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
    }
    config.update(params.get("model_config", {}))
    return config


def run_recbole(
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    token: str,
    seed: int,
    params: dict[str, Any],
    result_dir: Path,
    *,
    selection: bool,
) -> dict[str, object]:
    if str(RECBOLE_ROOT) not in sys.path:
        sys.path.insert(0, str(RECBOLE_ROOT))
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import get_trainer, init_logger, init_seed
    from recbole.utils import get_model

    config = Config(
        model=model_name,
        dataset=token,
        config_dict=recbole_config(
            args,
            dataset,
            model_name,
            token,
            seed,
            params,
            result_dir,
            selection=selection,
        ),
    )
    init_seed(config["seed"], config["reproducibility"])
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    init_logger(config)
    dataset_object = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_object)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(
        config["device"]
    )
    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer: Trainer = trainer_class(config, model)
    best_score, best_result = trainer.fit(
        train_data,
        valid_data=valid_data if selection else None,
        saved=True,
        show_progress=False,
    )
    if selection:
        if not best_result:
            raise RuntimeError("RecBole validation returned no metrics")
        result = best_result
        split = "inner_validation"
    else:
        result = trainer.evaluate(
            test_data, load_best_model=True, show_progress=False
        )
        split = "outer_test"
    row: dict[str, object] = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "evaluation_split": split,
        "best_valid_score": float(best_score) if selection else "",
    }
    for output_name, recbole_name in RECBOLE_METRICS.items():
        row[output_name] = 100.0 * float(result[recbole_name])
    del trainer, model, train_data, valid_data, test_data, dataset_object
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def read_pairs(path: Path) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            uid, pid = line.rstrip("\n").split("\t")
            rows.append((int(uid), int(pid)))
    return np.asarray(rows, dtype=np.int64)


def review_args(
    args: argparse.Namespace,
    dataset: str,
    params: dict[str, Any],
    result_dir: Path,
    *,
    selection: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        results_dir=result_dir,
        epochs=args.tuning_review_epochs if selection else args.final_review_epochs,
        batch_size=args.train_batch_size,
        deepconn_batch_size=256,
        eval_batch_size=256,
        encode_batch_size=512,
        embedding_size=int(params.get("embedding_size", 64)),
        word_embedding_size=int(params.get("word_embedding_size", 64)),
        cnn_filters=int(params.get("cnn_filters", 64)),
        cnn_kernel_size=int(params.get("cnn_kernel_size", 3)),
        max_user_words=256,
        max_item_words=256,
        review_words=8,
        word_negatives=4,
        review_weight=float(params.get("review_weight", 0.1)),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        gpu=args.gpu,
        cpu=args.cpu,
        max_train_batches=0,
        max_eval_users=0,
        smoke_test=False,
    )


def build_inner_review_data(
    dataset: str, token: str, run_args: argparse.Namespace
) -> Any:
    import scripts.run_review_baselines as review

    source_name, _outer_token = review.DATASETS[dataset]
    source = review.SOURCE_ROOT / source_name
    atomic = ATOMIC_ROOT / token
    outer_metadata = json.loads(
        (ATOMIC_ROOT / DATASET_TOKENS[dataset] / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    split_metadata = json.loads(
        (atomic / "metadata.json").read_text(encoding="utf-8")
    )
    train_pairs = read_pairs(atomic / f"{token}.train.inter")
    valid_pairs = read_pairs(atomic / f"{token}.valid.inter")
    allowed = {(int(uid), int(pid)) for uid, pid in train_pairs}
    n_users = review.count_gzip_lines(source / "users.txt.gz")
    n_words = review.count_gzip_lines(source / "vocab.txt.gz")
    n_items = int(outer_metadata["items"])
    split_digest = configuration_digest(
        {
            "protocol_version": PROTOCOL_VERSION,
            "inner_train_sha256": split_metadata["inner_train_sha256"],
            "inner_valid_sha256": split_metadata["inner_valid_sha256"],
        }
    )
    cache = (
        OUTPUT_ROOT
        / "review_cache"
        / dataset
        / token
        / split_digest
        / (
            f"u{run_args.max_user_words}_i{run_args.max_item_words}"
            f"_r{run_args.review_words}"
        )
    )
    files = {
        "train": cache / "train_pairs.npy",
        "words": cache / "review_words.npy",
        "users": cache / "user_docs.npy",
        "items": cache / "item_docs.npy",
        "meta": cache / "metadata.json",
    }
    if not all(path.exists() for path in files.values()):
        cache.mkdir(parents=True, exist_ok=True)
        train_mem = np.lib.format.open_memmap(
            files["train"], mode="w+", dtype=np.int64,
            shape=(len(train_pairs), 2),
        )
        words_mem = np.lib.format.open_memmap(
            files["words"], mode="w+", dtype=np.int32,
            shape=(len(train_pairs), run_args.review_words),
        )
        user_mem = np.lib.format.open_memmap(
            files["users"], mode="w+", dtype=np.int32,
            shape=(n_users, run_args.max_user_words),
        )
        item_mem = np.lib.format.open_memmap(
            files["items"], mode="w+", dtype=np.int32,
            shape=(n_items, run_args.max_item_words),
        )
        train_mem[:] = 0
        words_mem[:] = 0
        user_mem[:] = 0
        item_mem[:] = 0
        user_cursor = np.zeros(n_users, dtype=np.int32)
        item_cursor = np.zeros(n_items, dtype=np.int32)
        written = 0
        with gzip.open(
            source / "train.txt.gz", "rt", encoding="utf-8"
        ) as handle:
            for line in handle:
                cells = line.split()
                uid, pid = int(cells[0]), int(cells[1])
                if (uid, pid) not in allowed:
                    continue
                review_words = [int(word) for word in cells[2:]]
                train_mem[written] = (uid, pid)
                words_mem[written] = review.select_review_words(
                    review_words, run_args.review_words
                )
                review.append_document_words(
                    user_mem, user_cursor, uid, review_words
                )
                review.append_document_words(
                    item_mem, item_cursor, pid, review_words
                )
                written += 1
        if written != len(train_pairs):
            raise RuntimeError(
                f"Expected {len(train_pairs)} inner reviews, found {written}"
            )
        for array in (train_mem, words_mem, user_mem, item_mem):
            array.flush()
        atomic_json(
            files["meta"],
            {
                "dataset": dataset,
                "token": token,
                "n_users": n_users,
                "n_items": n_items,
                "n_words": n_words,
                "n_reviews": written,
                "protocol_version": PROTOCOL_VERSION,
                "inner_train_sha256": split_metadata["inner_train_sha256"],
                "inner_valid_sha256": split_metadata["inner_valid_sha256"],
                "training_reviews_only": True,
                "validation_reviews_excluded_from_documents": True,
            },
        )
    cache_meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    cached_train = np.load(files["train"], mmap_mode="r")
    return review.ReviewData(
        dataset=dataset,
        n_users=int(cache_meta["n_users"]),
        n_items=int(cache_meta["n_items"]),
        n_words=int(cache_meta["n_words"]),
        train_pairs=cached_train,
        review_words=np.load(files["words"], mmap_mode="r"),
        user_docs=np.load(files["users"], mmap_mode="r"),
        item_docs=np.load(files["items"], mmap_mode="r"),
        train_items=review.group_items(cached_train, n_users, unique=True),
        test_items=review.group_items(valid_pairs, n_users, unique=False),
    )


def run_review(
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    token: str,
    seed: int,
    params: dict[str, Any],
    result_dir: Path,
    *,
    selection: bool,
) -> dict[str, object]:
    import scripts.run_review_baselines as review

    run_args = review_args(
        args, dataset, params, result_dir, selection=selection
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    )
    if selection:
        data = build_inner_review_data(dataset, token, run_args)
    else:
        data = review.load_review_data(run_args)
    row = review.train_and_evaluate(
        model_name, seed, data, run_args, device
    )
    row["evaluation_split"] = (
        "inner_validation" if selection else "outer_test"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def search_space(model_name: str) -> dict[str, Any]:
    from hyperopt import hp

    prefix = slug(model_name)
    common: dict[str, Any] = {
        "learning_rate": hp.choice(
            f"{prefix}_learning_rate", [1e-4, 3e-4, 1e-3, 3e-3]
        ),
        "embedding_size": hp.choice(
            f"{prefix}_embedding_size", [64, 100]
        ),
    }
    if model_name == "BPR":
        return common
    if model_name == "CKE":
        common["regularization"] = hp.choice(
            f"{prefix}_regularization", [1e-4, 1e-3, 1e-2]
        )
        return common
    if model_name == "RippleNet":
        common.update(
            {
                "n_hop": hp.choice(f"{prefix}_n_hop", [1, 2, 3]),
                "n_memory": hp.choice(
                    f"{prefix}_n_memory", [8, 16, 32]
                ),
                "kg_weight": hp.choice(
                    f"{prefix}_kg_weight", [1e-3, 1e-2, 1e-1]
                ),
                "reg_weight": hp.choice(
                    f"{prefix}_reg_weight", [1e-8, 1e-7, 1e-6]
                ),
            }
        )
        return common
    if model_name == "KGAT":
        common.update(
            {
                "mess_dropout": hp.choice(
                    f"{prefix}_mess_dropout", [0.0, 0.1, 0.2]
                ),
                "reg_weight": hp.choice(
                    f"{prefix}_reg_weight", [1e-6, 1e-5, 1e-4]
                ),
                "aggregator_type": hp.choice(
                    f"{prefix}_aggregator_type",
                    ["gcn", "graphsage", "bi"],
                ),
            }
        )
        return common
    if model_name == "BPR-HFT":
        common.update(
            {
                "review_weight": hp.choice(
                    f"{prefix}_review_weight", [0.01, 0.1, 1.0]
                ),
                "weight_decay": hp.choice(
                    f"{prefix}_weight_decay", [0.0, 1e-6, 1e-5, 1e-4]
                ),
            }
        )
        return common
    if model_name == "DeepCoNN":
        return {
            "learning_rate": common["learning_rate"],
            "word_embedding_size": hp.choice(
                f"{prefix}_word_embedding_size", [64, 100]
            ),
            "cnn_filters": hp.choice(
                f"{prefix}_cnn_filters", [32, 64, 100]
            ),
            "cnn_kernel_size": hp.choice(
                f"{prefix}_cnn_kernel_size", [3, 5]
            ),
            "weight_decay": hp.choice(
                f"{prefix}_weight_decay", [0.0, 1e-6, 1e-5, 1e-4]
            ),
        }
    raise ValueError(model_name)


def prepare_model_params(
    model_name: str, sampled: dict[str, Any]
) -> dict[str, Any]:
    params = normalize(sampled)
    embedding = int(params.get("embedding_size", 64))
    model_config: dict[str, Any] = {}
    if model_name == "CKE":
        regularization = float(params.pop("regularization"))
        model_config = {
            "kg_embedding_size": embedding,
            "reg_weights": [regularization, regularization],
        }
    elif model_name == "RippleNet":
        model_config = {
            key: params.pop(key)
            for key in ("n_hop", "n_memory", "kg_weight", "reg_weight")
        }
    elif model_name == "KGAT":
        model_config = {
            key: params.pop(key)
            for key in ("mess_dropout", "reg_weight", "aggregator_type")
        }
        model_config.update(
            {"kg_embedding_size": embedding, "layers": [embedding]}
        )
    if model_config:
        params["model_config"] = model_config
    return params


def trial_directory(
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    params: dict[str, Any],
) -> Path:
    fingerprint, identity = selection_fingerprint(args, dataset, model_name)
    digest = configuration_digest(
        {"selection_protocol": identity, "params": params}
    )
    return (
        OUTPUT_ROOT
        / dataset
        / slug(model_name)
        / "trials"
        / fingerprint
        / digest
    )


def run_trial(
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    sampled: dict[str, Any],
) -> dict[str, object]:
    params = prepare_model_params(model_name, sampled)
    fingerprint, identity = selection_fingerprint(args, dataset, model_name)
    result_dir = trial_directory(args, dataset, model_name, params)
    result_path = result_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        return {"loss": float(existing["loss"]), "status": "ok"}
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(result_dir / "params.json", params)
    token = f"{DATASET_TOKENS[dataset]}_tune"
    update_status(
        stage="tuning",
        dataset=dataset,
        model=model_name,
        params=params,
    )
    print(
        f"TPE trial dataset={dataset} model={model_name} params={params}",
        flush=True,
    )
    if model_name in RECBOLE_MODELS:
        row = run_recbole(
            args,
            dataset,
            model_name,
            token,
            args.tuning_seed,
            params,
            result_dir,
            selection=True,
        )
    else:
        row = run_review(
            args,
            dataset,
            model_name,
            token,
            args.tuning_seed,
            params,
            result_dir,
            selection=True,
        )
    loss = -float(row["ndcg"])
    output = {
        "loss": loss,
        "status": "ok",
        "params": params,
        "metrics": row,
        "selection_fingerprint": fingerprint,
        "selection_protocol": identity,
    }
    atomic_json(result_path, output)
    print(
        f"Trial complete: validation NDCG@10={float(row['ndcg']):.6f}",
        flush=True,
    )
    return {"loss": loss, "status": "ok"}


def tune_one(
    args: argparse.Namespace, dataset: str, model_name: str
) -> dict[str, Any]:
    from hyperopt import Trials, fmin, tpe

    model_root = OUTPUT_ROOT / dataset / slug(model_name)
    model_root.mkdir(parents=True, exist_ok=True)
    fingerprint, identity = selection_fingerprint(args, dataset, model_name)
    state_path = model_root / f"hyperopt_trials_{fingerprint}.pkl"
    if state_path.exists():
        with state_path.open("rb") as handle:
            trials = pickle.load(handle)
    else:
        trials = Trials()
    budget = args.kgat_trials if model_name == "KGAT" else args.trials
    objective: Callable[[dict[str, Any]], dict[str, object]] = partial(
        run_trial, args, dataset, model_name
    )
    algorithm = partial(
        tpe.suggest, n_startup_jobs=min(args.tpe_startup_trials, budget)
    )
    for target in range(len(trials.trials) + 1, budget + 1):
        update_status(
            stage="tuning",
            dataset=dataset,
            model=model_name,
            trial=target,
            trials=budget,
        )
        fmin(
            fn=objective,
            space=search_space(model_name),
            algo=algorithm,
            max_evals=target,
            trials=trials,
            rstate=np.random.default_rng(args.search_seed + target),
            show_progressbar=False,
            verbose=False,
        )
        temporary = state_path.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(trials, handle)
        os.replace(temporary, state_path)

    candidates = []
    trial_root = model_root / "trials" / fingerprint
    for result_path in trial_root.glob("*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("selection_fingerprint") == fingerprint:
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"No completed trials for {dataset}/{model_name}")
    best = min(candidates, key=lambda result: float(result["loss"]))
    selection = {
        "dataset": dataset,
        "model": model_name,
        "objective": "maximize inner-validation NDCG@10",
        "trials_requested": budget,
        "trials_completed": len(candidates),
        "best_validation_ndcg": -float(best["loss"]),
        "params": best["params"],
        "selection_fingerprint": fingerprint,
        "selection_protocol": identity,
        "outer_test_used_for_selection": False,
    }
    atomic_json(model_root / f"best_config_{fingerprint}.json", selection)
    atomic_json(model_root / "best_config.json", selection)
    return selection


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {
        "seeds": len(rows),
        "standard_deviation": "sample standard deviation across seeds",
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        output[f"{metric}_mean"] = statistics.fmean(values)
        output[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
    return output


def final_evaluation(
    args: argparse.Namespace,
    dataset: str,
    model_name: str,
    selection: dict[str, Any],
) -> None:
    final_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "model": model_name,
        "params": selection["params"],
        "selection_fingerprint": selection["selection_fingerprint"],
        "final_recbole_epochs": args.final_recbole_epochs,
        "final_review_epochs": args.final_review_epochs,
        "final_seeds": args.final_seeds,
    }
    final_fingerprint = configuration_digest(final_identity, length=16)
    final_root = OUTPUT_ROOT / dataset / slug(model_name) / "final"
    final_dir = final_root / final_fingerprint
    atomic_json(
        final_root / "selected_run.json",
        {
            "final_fingerprint": final_fingerprint,
            "directory": str(final_dir.relative_to(ROOT)),
            **final_identity,
        },
    )
    raw_path = final_dir / "per_seed.csv"
    rows: list[dict[str, object]] = list(read_csv(raw_path))
    completed = {int(row["seed"]) for row in rows}
    params = selection["params"]
    for seed in args.final_seeds:
        if seed in completed:
            print(
                f"Skipping final {dataset}/{model_name} seed={seed}",
                flush=True,
            )
            continue
        update_status(
            stage="final_five_seed",
            dataset=dataset,
            model=model_name,
            seed=seed,
        )
        if model_name in RECBOLE_MODELS:
            row = run_recbole(
                args,
                dataset,
                model_name,
                DATASET_TOKENS[dataset],
                seed,
                params,
                final_dir,
                selection=False,
            )
        else:
            row = run_review(
                args,
                dataset,
                model_name,
                DATASET_TOKENS[dataset],
                seed,
                params,
                final_dir,
                selection=False,
            )
        row["selected_params"] = json.dumps(
            params, sort_keys=True, separators=(",", ":")
        )
        rows.append(row)
        write_csv(raw_path, rows)
        atomic_json(final_dir / "summary.json", summarize(rows))
        print(json.dumps(row, indent=2), flush=True)


def write_protocol(args: argparse.Namespace) -> None:
    atomic_json(
        OUTPUT_ROOT / "protocol.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "search": "Hyperopt TPE",
            "datasets": args.datasets,
            "models": args.models,
            "selection_metric": "inner-validation NDCG@10",
            "inner_split": {
                "source": "outer/public training interactions only",
                "fraction": args.validation_fraction,
                "seed": args.split_seed,
                "preserves_every_user_and_item_in_inner_train": True,
            },
            "test_policy": (
                "The public outer test is untouched during search and used "
                "only after the best configuration is selected."
            ),
            "selection_scope": (
                "One fixed tuning seed is used for the compact search. "
                "Five seeds are used only for the final outer-test report."
            ),
            "multi_fidelity_note": (
                "The compact search uses fewer epochs than final retraining, "
                "especially for KGAT; this is a documented cost-saving "
                "approximation."
            ),
            "budget": {
                "trials_per_model_dataset": args.trials,
                "kgat_trials_per_dataset": args.kgat_trials,
                "tpe_startup_trials": args.tpe_startup_trials,
                "tuning_seed": args.tuning_seed,
                "final_seeds": args.final_seeds,
                "tuning_epochs": args.tuning_epochs,
                "kgat_tuning_epochs": args.kgat_tuning_epochs,
                "review_tuning_epochs": args.tuning_review_epochs,
            },
            "excluded": {
                "PGPR": (
                    "official released runner has no independent validation "
                    "path; kept at official settings"
                ),
                "HeteroEmbed": (
                    "handled by scripts/run_paper_heteroembed.py; not mixed "
                    "with the RecBole search space"
                ),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_TOKENS),
        default=list(DATASET_TOKENS),
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--kgat-trials", type=int, default=4)
    parser.add_argument("--tpe-startup-trials", type=int, default=3)
    parser.add_argument("--tuning-seed", type=int, default=123)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2020)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--final-seeds",
        nargs="+",
        type=int,
        default=[123, 124, 125, 126, 127],
    )
    parser.add_argument("--tuning-epochs", type=int, default=20)
    parser.add_argument("--kgat-tuning-epochs", type=int, default=10)
    parser.add_argument("--tuning-review-epochs", type=int, default=10)
    parser.add_argument("--final-recbole-epochs", type=int, default=100)
    parser.add_argument("--final-review-epochs", type=int, default=20)
    parser.add_argument("--eval-step", type=int, default=2)
    parser.add_argument("--stopping-step", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.kgat_trials < 1:
        raise ValueError("Trial budgets must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("The baseline Hyperopt pipeline is already running") from error
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    write_protocol(args)
    if args.plan_only:
        print((OUTPUT_ROOT / "protocol.json").read_text(encoding="utf-8"))
        return
    from scripts.prepare_baseline_tuning_splits import create_split

    for dataset in args.datasets:
        metadata = create_split(
            dataset, args.validation_fraction, args.split_seed
        )
        print(
            f"Inner split ready for {dataset}: "
            f"{metadata['train_rows']} train / {metadata['valid_rows']} valid",
            flush=True,
        )
    for dataset in args.datasets:
        for model_name in args.models:
            selection = tune_one(args, dataset, model_name)
            if not args.tune_only:
                final_evaluation(args, dataset, model_name, selection)
    update_status(stage="complete", completed=True)
    print("Baseline Hyperopt pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
