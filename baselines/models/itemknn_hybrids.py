#!/usr/bin/env python3
"""Run the ItemKNN baselines from Ferrari Dacrema et al. in CAFE.

The RecSys 2019 paper defines three distinct methods:

* ItemKNN-CF: item vectors contain implicit user interactions;
* ItemKNN-CBF: item vectors contain item content features;
* ItemKNN-CFCBF: item vectors concatenate ``[r_i, w f_i]``.

This runner uses the public CAFE/PGPR interaction splits and constructs the
item-content matrix from training-review words, brands, and categories.  It
does not use co-purchase/co-view relations as content features.  Similarities
and prediction scores stay sparse and are evaluated in user batches, which is
required for the CDs & Vinyl dataset.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
ATOMIC_ROOT = ROOT / "baselines" / "data"
SOURCE_ROOT = ROOT / "baselines" / "source" / "PGPR-datasets"
CACHE_ROOT = ROOT / "baselines" / "cache" / "knn_hybrids"
RESULT_ROOT = ROOT / "results" / "paper_baselines"
PROTOCOL_VERSION = 1

DATASETS = {
    "beauty": ("beauty_paper", "Amazon_Beauty"),
    "cell": ("cell_paper", "Amazon_Cellphones"),
    "clothing": ("clothing_paper", "Amazon_Clothing"),
    "cd": ("cds_paper", "Amazon_CDs"),
}
MODELS = ("ItemKNN-CF", "ItemKNN-CBF", "ItemKNN-CFCBF")
CONTENT_FEATURES = ("words", "brand", "category")
METRICS = ("ndcg", "recall", "hr", "precision")


@dataclass
class DatasetBundle:
    dataset: str
    split: str
    token: str
    train_path: Path
    evaluation_path: Path
    train: sp.csr_matrix
    targets: list[np.ndarray]
    n_users: int
    n_items: int


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_sparse(path: Path, matrix: sp.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    sp.save_npz(temporary, matrix, compressed=True)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(value: object, length: int = 16) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def read_pairs(path: Path) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            uid, iid = line.rstrip("\n").split("\t")
            rows.append((int(uid), int(iid)))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def group_targets(pairs: np.ndarray, n_users: int) -> list[np.ndarray]:
    grouped: list[list[int]] = [[] for _ in range(n_users)]
    for uid, iid in pairs:
        grouped[int(uid)].append(int(iid))
    return [np.asarray(items, dtype=np.int64) for items in grouped]


def load_dataset(dataset: str, split: str) -> DatasetBundle:
    outer_token, _source_name = DATASETS[dataset]
    token = outer_token if split == "outer" else f"{outer_token}_tune"
    directory = ATOMIC_ROOT / token
    train_path = directory / f"{token}.train.inter"
    evaluation_suffix = "test" if split == "outer" else "valid"
    evaluation_path = directory / f"{token}.{evaluation_suffix}.inter"
    metadata_path = directory / "metadata.json"
    for path in (train_path, evaluation_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_items = int(metadata["items"])
    train_pairs = read_pairs(train_path)
    evaluation_pairs = read_pairs(evaluation_path)
    if not len(train_pairs):
        raise RuntimeError(f"Training split is empty: {train_path}")
    max_user = int(train_pairs[:, 0].max())
    if len(evaluation_pairs):
        max_user = max(max_user, int(evaluation_pairs[:, 0].max()))
    n_users = max_user + 1
    values = np.ones(len(train_pairs), dtype=np.float32)
    train = sp.csr_matrix(
        (values, (train_pairs[:, 0], train_pairs[:, 1])),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    train.sum_duplicates()
    train.data[:] = 1.0
    return DatasetBundle(
        dataset=dataset,
        split=split,
        token=token,
        train_path=train_path,
        evaluation_path=evaluation_path,
        train=train,
        targets=group_targets(evaluation_pairs, n_users),
        n_users=n_users,
        n_items=n_items,
    )


def retained_words(
    words: list[int],
    documents: int,
    frequency: Counter[int],
    document_frequency: Counter[int],
) -> list[int]:
    counts = Counter(words)
    weighted = {
        word: count
        * (
            math.log(
                (1.0 + documents) / (1.0 + document_frequency[word])
            )
            + 1.0
        )
        for word, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0.0:
        return []
    return [
        word
        for word, value in weighted.items()
        if value / norm >= 0.1 and frequency[word] <= 5000
    ]


def iter_reviews(
    path: Path, allowed: set[tuple[int, int]]
) -> Iterable[tuple[int, int, list[int]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            uid, iid = int(cells[0]), int(cells[1])
            if (uid, iid) not in allowed:
                continue
            words = [int(value) for value in cells[2].split()] if cells[2] else []
            yield uid, iid, words


def content_cache_identity(
    bundle: DatasetBundle, selected_features: tuple[str, ...]
) -> dict[str, object]:
    _token, source_name = DATASETS[bundle.dataset]
    source = SOURCE_ROOT / source_name
    inputs = {}
    for filename in ("train.txt.gz", "brand_p_b.txt.gz", "category_p_c.txt.gz"):
        path = source / filename
        stat = path.stat()
        inputs[filename] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": bundle.dataset,
        "split": bundle.split,
        "train_sha256": file_sha256(bundle.train_path),
        "n_items": bundle.n_items,
        "features": selected_features,
        "inputs": inputs,
        "review_filter": {
            "normalized_tfidf_min": 0.1,
            "global_frequency_max": 5000,
            "fit_interactions_only": True,
        },
    }


def build_content_matrix(
    bundle: DatasetBundle, selected_features: tuple[str, ...]
) -> tuple[sp.csr_matrix, dict[str, object]]:
    if not selected_features:
        raise ValueError("At least one content feature family is required")
    identity = content_cache_identity(bundle, selected_features)
    digest = config_digest(identity)
    cache_dir = CACHE_ROOT / bundle.dataset / bundle.split / digest
    matrix_path = cache_dir / "icm.npz"
    metadata_path = cache_dir / "metadata.json"
    if matrix_path.exists() and metadata_path.exists():
        matrix = sp.load_npz(matrix_path).tocsr().astype(np.float32)
        if matrix.shape[0] != bundle.n_items:
            raise RuntimeError("Cached ICM item dimension does not match dataset")
        return matrix, json.loads(metadata_path.read_text(encoding="utf-8"))

    _token, source_name = DATASETS[bundle.dataset]
    source = SOURCE_ROOT / source_name
    train_coo = bundle.train.tocoo()
    allowed = set(zip(train_coo.row.tolist(), train_coo.col.tolist()))
    feature_to_column: dict[tuple[str, int], int] = {}
    row_buffer = array("I")
    column_buffer = array("I")

    def append_feature(item: int, family: str, raw_feature: int) -> None:
        if not 0 <= item < bundle.n_items:
            raise ValueError(f"Out-of-range product id {item}")
        token = (family, raw_feature)
        column = feature_to_column.get(token)
        if column is None:
            column = len(feature_to_column)
            feature_to_column[token] = column
        row_buffer.append(item)
        column_buffer.append(column)

    review_documents = 0
    retained_review_count = 0
    if "words" in selected_features:
        review_path = source / "train.txt.gz"
        frequency: Counter[int] = Counter()
        document_frequency: Counter[int] = Counter()
        for _uid, _iid, words in iter_reviews(review_path, allowed):
            frequency.update(words)
            document_frequency.update(set(words))
            review_documents += 1
        for _uid, iid, words in iter_reviews(review_path, allowed):
            kept = retained_words(
                words, review_documents, frequency, document_frequency
            )
            if kept:
                retained_review_count += 1
            for word in kept:
                append_feature(iid, "word", word)

    for family, filename in (
        ("brand", "brand_p_b.txt.gz"),
        ("category", "category_p_c.txt.gz"),
    ):
        if family not in selected_features:
            continue
        with gzip.open(source / filename, "rt", encoding="utf-8") as handle:
            for iid, line in enumerate(handle):
                for raw_feature in line.split():
                    append_feature(iid, family, int(raw_feature))

    rows = np.frombuffer(row_buffer, dtype=np.uint32).astype(np.int64)
    columns = np.frombuffer(column_buffer, dtype=np.uint32).astype(np.int64)
    matrix = sp.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (rows, columns),
        ),
        shape=(bundle.n_items, len(feature_to_column)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    metadata: dict[str, object] = {
        **identity,
        "cache_digest": digest,
        "shape": list(matrix.shape),
        "nonzero_features": int(matrix.nnz),
        "feature_columns": len(feature_to_column),
        "review_documents": review_documents,
        "retained_reviews": retained_review_count,
        "content_semantics": (
            "Training-review words plus static brand/category attributes; "
            "behavioral also_bought/viewed/together relations excluded."
        ),
    }
    atomic_sparse(matrix_path, matrix)
    atomic_json(metadata_path, metadata)
    return matrix, metadata


def apply_feature_weighting(
    matrix: sp.csr_matrix, weighting: str
) -> sp.csr_matrix:
    weighted = matrix.astype(np.float32, copy=True).tocsr()
    if weighting == "none":
        return weighted
    document_count = weighted.shape[0]
    document_frequency = np.bincount(
        weighted.indices, minlength=weighted.shape[1]
    ).astype(np.float32)
    if weighting == "tfidf":
        idf = np.zeros_like(document_frequency)
        mask = document_frequency > 0
        idf[mask] = np.log(document_count / document_frequency[mask])
        weighted.data *= idf[weighted.indices]
    elif weighting == "bm25":
        k1, b = 1.2, 0.75
        idf = np.log(
            document_count / (1.0 + document_frequency)
        ).astype(np.float32)
        row_length = np.asarray(weighted.sum(axis=1)).ravel()
        average_length = float(row_length.mean()) if len(row_length) else 1.0
        average_length = max(average_length, 1e-6)
        row_ids = np.repeat(
            np.arange(weighted.shape[0]), np.diff(weighted.indptr)
        )
        length_norm = (1.0 - b) + b * row_length[row_ids] / average_length
        values = weighted.data
        weighted.data = (
            values
            * (k1 + 1.0)
            / (values + k1 * length_norm)
            * idf[weighted.indices]
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature weighting: {weighting}")
    weighted.eliminate_zeros()
    return weighted


def build_item_profiles(
    model: str,
    train: sp.csr_matrix,
    content: sp.csr_matrix | None,
    interaction_weighting: str,
    content_weighting: str,
    content_weight: float,
) -> sp.csr_matrix:
    collaborative = apply_feature_weighting(
        train.transpose().tocsr(), interaction_weighting
    )
    if model == "ItemKNN-CF":
        return collaborative
    if content is None:
        raise ValueError(f"{model} requires an item-content matrix")
    content_profile = apply_feature_weighting(content, content_weighting)
    if model == "ItemKNN-CBF":
        return content_profile
    if model == "ItemKNN-CFCBF":
        return sp.hstack(
            [collaborative, content_profile * float(content_weight)],
            format="csr",
            dtype=np.float32,
        )
    raise ValueError(model)


def topk_cosine_similarity(
    profiles: sp.csr_matrix,
    neighbors: int,
    shrink: float,
    normalize: bool,
    block_size: int,
) -> sp.csr_matrix:
    profiles = profiles.astype(np.float32).tocsr()
    item_count = profiles.shape[0]
    if item_count < 2:
        raise ValueError("At least two items are required")
    neighbors = min(max(1, neighbors), item_count - 1)
    norms = np.sqrt(np.asarray(profiles.power(2).sum(axis=1)).ravel())
    output_rows = array("I")
    output_columns = array("I")
    output_values = array("f")

    for start in range(0, item_count, block_size):
        end = min(start + block_size, item_count)
        products = (profiles[start:end] @ profiles.transpose()).tocsr()
        for local_index, item in enumerate(range(start, end)):
            row_start, row_end = products.indptr[local_index : local_index + 2]
            columns = products.indices[row_start:row_end]
            values = products.data[row_start:row_end].astype(
                np.float32, copy=True
            )
            valid = columns != item
            columns, values = columns[valid], values[valid]
            if normalize and len(values):
                denominator = norms[item] * norms[columns] + float(shrink)
                values = np.divide(
                    values,
                    denominator,
                    out=np.zeros_like(values),
                    where=denominator > 0,
                )
            positive = np.isfinite(values) & (values > 0)
            columns, values = columns[positive], values[positive]
            if len(values) > neighbors:
                selected = np.argpartition(-values, neighbors - 1)[:neighbors]
                columns, values = columns[selected], values[selected]
            if len(values):
                order = np.lexsort((columns, -values))
                columns, values = columns[order], values[order]
                output_rows.extend([item] * len(values))
                output_columns.extend(columns.astype(np.uint32).tolist())
                output_values.extend(values.astype(np.float32).tolist())
        print(
            f"Similarity rows {end}/{item_count}",
            flush=True,
        )

    rows = np.frombuffer(output_rows, dtype=np.uint32).astype(np.int64)
    columns = np.frombuffer(output_columns, dtype=np.uint32).astype(np.int64)
    values = np.frombuffer(output_values, dtype=np.float32)
    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(item_count, item_count),
        dtype=np.float32,
    )


def evaluate(
    train: sp.csr_matrix,
    similarity: sp.csr_matrix,
    targets: list[np.ndarray],
    topk: int,
    batch_size: int,
    max_users: int,
) -> dict[str, float]:
    users = np.asarray(
        [uid for uid, items in enumerate(targets) if len(items)],
        dtype=np.int64,
    )
    if max_users:
        users = users[:max_users]
    if len(users) == 0:
        raise ValueError("No users with evaluation targets")
    if topk < 1:
        raise ValueError("topk must be positive")
    metric_values = {metric: [] for metric in METRICS}
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        scores = (train[batch_users] @ similarity).toarray()
        for row, uid in enumerate(batch_users):
            seen_start, seen_end = train.indptr[uid : uid + 2]
            scores[row, train.indices[seen_start:seen_end]] = -np.inf
        cutoff = min(topk, scores.shape[1])
        kth = scores.shape[1] - cutoff
        candidates = np.argpartition(scores, kth, axis=1)[:, kth:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        recommendations = np.take_along_axis(candidates, order, axis=1)
        for uid, predicted in zip(batch_users, recommendations):
            relevant = set(targets[int(uid)].tolist())
            hit_positions = [
                rank
                for rank, item in enumerate(predicted)
                if int(item) in relevant
            ]
            hits = len(hit_positions)
            dcg = sum(1.0 / math.log2(position + 2) for position in hit_positions)
            ideal = min(len(relevant), cutoff)
            idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal))
            metric_values["ndcg"].append(dcg / idcg if idcg else 0.0)
            metric_values["recall"].append(hits / len(relevant))
            metric_values["hr"].append(float(hits > 0))
            metric_values["precision"].append(hits / float(cutoff))
        print(f"Evaluated users {min(start + batch_size, len(users))}/{len(users)}", flush=True)
    return {
        metric: 100.0 * statistics.fmean(values)
        for metric, values in metric_values.items()
    }


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


def run_model(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    model: str,
    content: sp.csr_matrix | None,
    content_metadata: dict[str, object] | None,
) -> dict[str, object]:
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": bundle.dataset,
        "split": bundle.split,
        "model": model,
        "neighbors": args.neighbors,
        "shrink": args.shrink,
        "normalize": args.normalize,
        "interaction_weighting": args.interaction_weighting,
        "content_weighting": args.content_weighting,
        "content_weight": args.content_weight,
        "content_features": sorted(args.content_features),
        "topk": args.topk,
        "max_eval_users": args.max_eval_users,
        "train_sha256": file_sha256(bundle.train_path),
        "evaluation_sha256": file_sha256(bundle.evaluation_path),
    }
    digest = config_digest(config)
    result_dir = args.results_root / bundle.dataset / "knn_hybrids"
    similarity_dir = CACHE_ROOT / bundle.dataset / bundle.split / "similarities"
    similarity_path = similarity_dir / f"{model.lower()}_{digest}.npz"
    if similarity_path.exists():
        similarity = sp.load_npz(similarity_path).tocsr()
    else:
        profiles = build_item_profiles(
            model,
            bundle.train,
            content,
            args.interaction_weighting,
            args.content_weighting,
            args.content_weight,
        )
        similarity = topk_cosine_similarity(
            profiles,
            args.neighbors,
            args.shrink,
            args.normalize,
            args.similarity_block_size,
        )
        atomic_sparse(similarity_path, similarity)
    started = time.monotonic()
    metrics = evaluate(
        bundle.train,
        similarity,
        bundle.targets,
        args.topk,
        args.eval_batch_size,
        args.max_eval_users,
    )
    row: dict[str, object] = {
        "dataset": bundle.dataset,
        "model": model,
        "split": bundle.split,
        "deterministic": True,
        "config_digest": digest,
        "neighbors": args.neighbors,
        "shrink": args.shrink,
        "normalize": args.normalize,
        "interaction_weighting": args.interaction_weighting,
        "content_weighting": args.content_weighting,
        "content_weight": args.content_weight if model == "ItemKNN-CFCBF" else "",
        "ndcg": metrics["ndcg"],
        "recall": metrics["recall"],
        "hr": metrics["hr"],
        "precision": metrics["precision"],
        "seconds_evaluation": time.monotonic() - started,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    existing = read_csv(result_dir / "per_run.csv")
    existing = [item for item in existing if item["config_digest"] != digest]
    write_csv(result_dir / "per_run.csv", [*existing, row])
    atomic_json(
        result_dir / f"result_{digest}.json",
        {
            "config": config,
            "metrics": metrics,
            "content": content_metadata,
            "similarity_nonzeros": int(similarity.nnz),
            "deterministic": True,
            "paper_adaptation": (
                "CAFE Amazon features replace the original CiteULike "
                "title/abstract features."
            ),
        },
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=["beauty"]
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--split", choices=("outer", "inner"), default="outer")
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--interaction-weighting",
        choices=("none", "tfidf", "bm25"),
        default="none",
    )
    parser.add_argument(
        "--content-weighting",
        choices=("none", "tfidf", "bm25"),
        default="none",
    )
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument(
        "--content-features",
        nargs="+",
        choices=CONTENT_FEATURES,
        default=list(CONTENT_FEATURES),
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-eval-users", type=int, default=0)
    parser.add_argument("--results-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root = args.results_root.resolve()
    if args.neighbors < 1 or args.shrink < 0 or args.content_weight < 0:
        raise ValueError("neighbors must be positive; shrink/weight non-negative")
    plan = {
        "paper": (
            "Are We Really Making Much Progress? A Worrying Analysis of "
            "Recent Neural Recommendation Approaches (RecSys 2019)"
        ),
        "datasets": args.datasets,
        "models": args.models,
        "split": args.split,
        "metrics": ["NDCG@10", "Recall@10", "HR@10", "Precision@10"],
        "deterministic": True,
        "content_features": args.content_features,
        "behavioral_kg_relations_excluded": True,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    for dataset in args.datasets:
        bundle = load_dataset(dataset, args.split)
        needs_content = any(model != "ItemKNN-CF" for model in args.models)
        if needs_content:
            selected = tuple(sorted(set(args.content_features)))
            content, content_metadata = build_content_matrix(bundle, selected)
        else:
            content, content_metadata = None, None
        result_dir = args.results_root / dataset / "knn_hybrids"
        atomic_json(result_dir / "protocol.json", plan)
        for model in args.models:
            row = run_model(
                args, bundle, model, content, content_metadata
            )
            print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
