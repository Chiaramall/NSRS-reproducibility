#!/usr/bin/env python3
"""MostPopular, ItemKNN and UserKNN on the original MindReader protocol."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ltnrec.candidates import candidate_arrays
from ltnrec.data import MindReaderDataset
from ltnrec.metrics import compute_metric
from movielens_paper_benchmark.search import atomic_json_dump


MODELS = ("most-popular", "itemknn", "userknn")
PROPORTIONS = (1.0, 0.8, 0.6, 0.4, 0.2)
K_VALUES = (20, 50, 100, 200)
SHRINKAGES = (0.0, 10.0, 100.0)
FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def p_label(value: float) -> str:
    return f"{float(value):.1f}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sparse_fold(data: MindReaderDataset, proportion: float, seed: int):
    np.random.seed(int(seed))
    return data.get_user_movie_ratings(keep=float(proportion), seed=int(seed))


def positive_pairs(fold: Iterable[Sequence[float]]) -> np.ndarray:
    values = [(int(u), int(i)) for u, i, rating in fold if float(rating) > 0]
    result = np.asarray(values, dtype=np.int64).reshape(-1, 2)
    if not len(result):
        raise ValueError("the fold contains no positive interactions")
    return np.unique(result, axis=0)


def evaluate(model: Any, rows: Sequence[np.ndarray]) -> dict[str, float]:
    users, items = candidate_arrays(rows)
    scores = np.asarray(model.score_candidates(users, items), dtype=np.float64)
    if scores.shape != items.shape:
        raise ValueError(f"score shape {scores.shape} differs from {items.shape}")
    truth = np.zeros_like(scores)
    truth[:, -1] = 1.0
    return {
        "hit@10": float(np.mean(compute_metric("hit@10", scores, truth))),
        "ndcg@10": float(np.mean(compute_metric("ndcg@10", scores, truth))),
    }


def profile_lists(
    pairs: np.ndarray, n_users: int, n_items: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    by_user: list[list[int]] = [[] for _ in range(n_users)]
    by_item: list[list[int]] = [[] for _ in range(n_items)]
    for user, item in pairs.tolist():
        by_user[user].append(item)
        by_item[item].append(user)
    return (
        [np.asarray(sorted(set(values)), dtype=np.int64) for values in by_user],
        [np.asarray(sorted(set(values)), dtype=np.int64) for values in by_item],
    )


def top_cosine_neighbors(
    groups: Sequence[np.ndarray],
    n_nodes: int,
    k: int,
    shrinkage: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    degrees = np.zeros(n_nodes, dtype=np.int64)
    cooccurrence: list[dict[int, int]] = [defaultdict(int) for _ in range(n_nodes)]
    for raw_group in groups:
        group = np.unique(np.asarray(raw_group, dtype=np.int64))
        if not len(group):
            continue
        degrees[group] += 1
        for source in group.tolist():
            target_counts = cooccurrence[source]
            for target in group.tolist():
                if target != source:
                    target_counts[target] += 1

    neighbor_ids: list[np.ndarray] = []
    neighbor_weights: list[np.ndarray] = []
    for source, counts in enumerate(cooccurrence):
        candidates = []
        source_degree = int(degrees[source])
        if source_degree:
            for target, overlap in counts.items():
                denominator = np.sqrt(float(source_degree * degrees[target]))
                if denominator == 0:
                    continue
                cosine = float(overlap) / denominator
                significance = (
                    1.0
                    if shrinkage == 0
                    else float(overlap) / (float(overlap) + float(shrinkage))
                )
                candidates.append((cosine * significance, int(target)))
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        selected = candidates[: min(int(k), len(candidates))]
        neighbor_weights.append(
            np.asarray([value for value, _ in selected], dtype=np.float32)
        )
        neighbor_ids.append(
            np.asarray([target for _, target in selected], dtype=np.int64)
        )
    return neighbor_ids, neighbor_weights


class MostPopular:
    def __init__(self, pairs: np.ndarray, n_items: int) -> None:
        self.popularity = np.bincount(pairs[:, 1], minlength=n_items).astype(np.float32)

    def score_candidates(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        del users
        return self.popularity[items]


class ItemKNN:
    def __init__(
        self,
        pairs: np.ndarray,
        n_users: int,
        n_items: int,
        k: int,
        shrinkage: float,
    ) -> None:
        self.user_items, item_users = profile_lists(pairs, n_users, n_items)
        self.neighbors, self.weights = top_cosine_neighbors(
            self.user_items, n_items, k, shrinkage
        )

    def score_candidates(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        result = np.zeros(items.shape, dtype=np.float32)
        for row, raw_user in enumerate(users):
            profile = set(self.user_items[int(raw_user)].tolist())
            if not profile:
                continue
            for column, raw_item in enumerate(items[row]):
                neighbors = self.neighbors[int(raw_item)]
                weights = self.weights[int(raw_item)]
                if not len(neighbors):
                    continue
                mask = np.fromiter(
                    (int(neighbor) in profile for neighbor in neighbors),
                    dtype=bool,
                    count=len(neighbors),
                )
                denominator = float(weights.sum())
                if denominator > 0:
                    result[row, column] = float(weights[mask].sum()) / denominator
        return result


class UserKNN:
    def __init__(
        self,
        pairs: np.ndarray,
        n_users: int,
        n_items: int,
        k: int,
        shrinkage: float,
    ) -> None:
        self.user_items, item_users = profile_lists(pairs, n_users, n_items)
        self.item_user_sets = [set(values.tolist()) for values in item_users]
        self.neighbors, self.weights = top_cosine_neighbors(
            item_users, n_users, k, shrinkage
        )

    def score_candidates(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        result = np.zeros(items.shape, dtype=np.float32)
        for row, raw_user in enumerate(users):
            neighbors = self.neighbors[int(raw_user)]
            weights = self.weights[int(raw_user)]
            denominator = float(weights.sum())
            if denominator == 0:
                continue
            for column, raw_item in enumerate(items[row]):
                positive_users = self.item_user_sets[int(raw_item)]
                mask = np.fromiter(
                    (int(neighbor) in positive_users for neighbor in neighbors),
                    dtype=bool,
                    count=len(neighbors),
                )
                result[row, column] = float(weights[mask].sum()) / denominator
        return result


def create_model(
    name: str,
    parameters: Mapping[str, Any],
    pairs: np.ndarray,
    data: MindReaderDataset,
):
    if name == "most-popular":
        return MostPopular(pairs, data.n_movies)
    constructor = ItemKNN if name == "itemknn" else UserKNN
    return constructor(
        pairs,
        data.n_users,
        data.n_movies,
        int(parameters["k"]),
        float(parameters["shrinkage"]),
    )


def configurations(name: str) -> list[dict[str, Any]]:
    if name == "most-popular":
        return [{}]
    return [
        {"k": int(k), "shrinkage": float(shrinkage)}
        for k, shrinkage in itertools.product(K_VALUES, SHRINKAGES)
    ]


def search_path(root: Path, model: str, proportion: float) -> Path:
    return root / model / "search" / f"{p_label(proportion)}.json"


def run_path(root: Path, model: str, proportion: float, seed_index: int) -> Path:
    return root / model / "runs" / p_label(proportion) / f"seed-{seed_index}.json"


def new_search_state(
    model: str, proportion: float, validation_seeds: int
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "model": model,
        "proportion": float(proportion),
        "validation_seeds": int(validation_seeds),
        "objective": "hit@10",
        "records": [
            {"index": index, "parameters": config, "validation": []}
            for index, config in enumerate(configurations(model))
        ],
        "best_parameters": None,
        "best_validation_hit@10": None,
        "completed_fits": 0,
        "planned_fits": len(configurations(model)) * int(validation_seeds),
        "updated_at": utc_now(),
    }


def refresh_search(state: dict[str, Any]) -> None:
    required = int(state["validation_seeds"])
    completed = []
    for record in state["records"]:
        if len(record["validation"]) == required:
            record["mean_validation_hit@10"] = float(
                np.mean([entry["hit@10"] for entry in record["validation"]])
            )
            completed.append(record)
    best = None
    for record in completed:
        if best is None or record["mean_validation_hit@10"] > best[
            "mean_validation_hit@10"
        ]:
            best = record
    state["completed_fits"] = sum(
        len(record["validation"]) for record in state["records"]
    )
    state["best_parameters"] = None if best is None else best["parameters"]
    state["best_validation_hit@10"] = (
        None if best is None else best["mean_validation_hit@10"]
    )
    state["updated_at"] = utc_now()


def run_search(args, data: MindReaderDataset, model_name: str) -> None:
    expected_configs = configurations(model_name)
    for proportion in args.proportions:
        path = search_path(args.output, model_name, proportion)
        state = (
            load_json(path)
            if path.exists()
            else new_search_state(model_name, proportion, args.validation_seeds)
        )
        if (
            state.get("format_version") != FORMAT_VERSION
            or state["validation_seeds"] != args.validation_seeds
            or [record["parameters"] for record in state["records"]] != expected_configs
        ):
            raise RuntimeError(f"incompatible search state: {path}")
        for seed_index in range(args.validation_seeds):
            seed = args.search_seed + seed_index
            fold = sparse_fold(data, proportion, seed)
            pairs = positive_pairs(fold)
            for record in state["records"]:
                if len(record["validation"]) > seed_index:
                    continue
                print(
                    f"[{utc_now()}] search {model_name} p={p_label(proportion)} "
                    f"config {record['index'] + 1}/{len(state['records'])} "
                    f"seed {seed_index + 1}/{args.validation_seeds}",
                    flush=True,
                )
                started = time.perf_counter()
                fitted = create_model(model_name, record["parameters"], pairs, data)
                metrics = evaluate(fitted, data.validation)
                record["validation"].append(
                    {
                        "seed_index": seed_index,
                        "seed": seed,
                        **metrics,
                        "seconds": time.perf_counter() - started,
                    }
                )
                refresh_search(state)
                atomic_json_dump(state, path)


def run_finals(args, data: MindReaderDataset, model_name: str) -> None:
    for proportion in args.proportions:
        state = load_json(search_path(args.output, model_name, proportion))
        if state.get("completed_fits") != state.get("planned_fits"):
            raise RuntimeError(f"incomplete search for {model_name}, p={proportion}")
        parameters = state["best_parameters"]
        for seed_index in range(args.final_seeds):
            path = run_path(args.output, model_name, proportion, seed_index)
            if path.exists():
                continue
            seed = args.final_seed_start + seed_index
            print(
                f"[{utc_now()}] final {model_name} p={p_label(proportion)} "
                f"seed {seed_index + 1}/{args.final_seeds}",
                flush=True,
            )
            fold = sparse_fold(data, proportion, seed)
            pairs = positive_pairs(fold)
            started = time.perf_counter()
            fitted = create_model(model_name, parameters, pairs, data)
            metrics = evaluate(fitted, data.test)
            atomic_json_dump(
                {
                    "format_version": FORMAT_VERSION,
                    "status": "complete",
                    "model": model_name,
                    "proportion": float(proportion),
                    "seed_index": seed_index,
                    "seed": seed,
                    "parameters": parameters,
                    "metrics": metrics,
                    "seconds": time.perf_counter() - started,
                    "created_at": utc_now(),
                },
                path,
            )


def counts(args, model_name: str) -> tuple[int, int, int, int]:
    search_done = 0
    search_total = len(args.proportions) * len(configurations(model_name)) * args.validation_seeds
    final_done = 0
    final_total = len(args.proportions) * args.final_seeds
    for proportion in args.proportions:
        path = search_path(args.output, model_name, proportion)
        if path.exists():
            search_done += int(load_json(path).get("completed_fits", 0))
        final_done += sum(
            run_path(args.output, model_name, proportion, seed).exists()
            for seed in range(args.final_seeds)
        )
    return search_done, search_total, final_done, final_total


def write_report(args, model_name: str) -> None:
    summary: dict[str, Any] = {"model": model_name, "proportions": {}}
    lines = [
        f"# MindReader {model_name} results",
        "",
        "Values are test mean ± population standard deviation.",
        "",
        "| Training ratings | Hit@10 | NDCG@10 | Selected parameters |",
        "|---:|---:|---:|---|",
    ]
    for proportion in args.proportions:
        records = [
            load_json(run_path(args.output, model_name, proportion, seed))
            for seed in range(args.final_seeds)
            if run_path(args.output, model_name, proportion, seed).exists()
        ]
        if not records:
            continue
        search = load_json(search_path(args.output, model_name, proportion))
        entry = {
            "n": len(records),
            "selected_parameters": search["best_parameters"],
        }
        display = {}
        for metric in ("hit@10", "ndcg@10"):
            values = np.asarray([record["metrics"][metric] for record in records])
            entry[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n": len(values),
            }
            display[metric] = (
                f"{values.mean():.4f} ± {values.std(ddof=0):.4f} (n={len(values)})"
            )
        summary["proportions"][p_label(proportion)] = entry
        parameters = ", ".join(
            f"{key}={value}" for key, value in search["best_parameters"].items()
        ) or "none"
        lines.append(
            f"| {proportion * 100:.0f}% | {display['hit@10']} | "
            f"{display['ndcg@10']} | {parameters} |"
        )
    directory = args.output / model_name / "reports"
    atomic_json_dump(summary, directory / "summary.json")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_statistics(args, data: MindReaderDataset) -> None:
    training = data.get_user_movie_ratings()
    positives = [row for row in training if row[2] > 0]
    negatives = [row for row in training if row[2] < 0]
    possible = data.n_users * data.n_movies
    user_counts = np.bincount(
        [row[0] for row in training], minlength=data.n_users
    )
    positive_user_counts = np.bincount(
        [row[0] for row in positives], minlength=data.n_users
    )
    item_counts = np.bincount(
        [row[1] for row in training], minlength=data.n_movies
    )
    base = {
        "n_users": data.n_users,
        "n_movies": data.n_movies,
        "n_genres": data.n_genres,
        "possible_user_movie_pairs": possible,
        "training_movie_ratings": len(training),
        "positive_training_ratings": len(positives),
        "negative_training_ratings": len(negatives),
        "density": len(training) / possible,
        "sparsity": 1.0 - len(training) / possible,
        "active_training_movies": int(np.count_nonzero(item_counts)),
        "validation_users": len(data.validation),
        "test_users": len(data.test),
        "movie_genre_links": sum(len(values) for values in data.movie_to_genres.values()),
        "movies_with_genres": len(data.movie_to_genres),
        "ratings_per_user": {
            "min": int(user_counts.min()),
            "median": float(np.median(user_counts)),
            "mean": float(user_counts.mean()),
            "max": int(user_counts.max()),
        },
        "positive_ratings_per_user": {
            "min": int(positive_user_counts.min()),
            "median": float(np.median(positive_user_counts)),
            "mean": float(positive_user_counts.mean()),
            "max": int(positive_user_counts.max()),
        },
    }
    folds = {}
    for proportion in args.proportions:
        records = []
        for seed in range(args.final_seeds):
            fold = sparse_fold(data, proportion, seed)
            positive = sum(row[2] > 0 for row in fold)
            negative = sum(row[2] < 0 for row in fold)
            active_items = len({row[1] for row in fold})
            records.append(
                {
                    "ratings": len(fold),
                    "positive": positive,
                    "negative": negative,
                    "density": len(fold) / possible,
                    "active_items": active_items,
                }
            )
        folds[p_label(proportion)] = {
            key: {
                "mean": float(np.mean([record[key] for record in records])),
                "std": float(np.std([record[key] for record in records], ddof=0)),
            }
            for key in records[0]
        }
    payload = {"dataset": "MindReader", "base": base, "folds_over_30_seeds": folds}
    directory = args.output / "statistics"
    atomic_json_dump(payload, directory / "statistics.json")
    lines = [
        "# MindReader dataset statistics",
        "",
        "## Complete training split",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Users | {base['n_users']} |",
        f"| Movies | {base['n_movies']} |",
        f"| Genres | {base['n_genres']} |",
        f"| Training movie ratings | {base['training_movie_ratings']} |",
        f"| Positive ratings | {base['positive_training_ratings']} |",
        f"| Negative ratings | {base['negative_training_ratings']} |",
        f"| Density | {100 * base['density']:.4f}% |",
        f"| Sparsity | {100 * base['sparsity']:.4f}% |",
        f"| Validation users | {base['validation_users']} |",
        f"| Test users | {base['test_users']} |",
        f"| Movie-genre links | {base['movie_genre_links']} |",
        "",
        "## Sparsified training folds over 30 seeds",
        "",
        "| Nominal proportion | Ratings | Positive | Negative | Density | Active movies |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for proportion in args.proportions:
        record = folds[p_label(proportion)]
        lines.append(
            f"| {100 * proportion:.0f}% | "
            f"{record['ratings']['mean']:.1f} ± {record['ratings']['std']:.1f} | "
            f"{record['positive']['mean']:.1f} ± {record['positive']['std']:.1f} | "
            f"{record['negative']['mean']:.1f} ± {record['negative']['std']:.1f} | "
            f"{100 * record['density']['mean']:.4f}% | "
            f"{record['active_items']['mean']:.1f} ± {record['active_items']['std']:.1f} |"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def acquire_lock(output: Path, model_name: str) -> Path:
    directory = output / model_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".writer.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "search", "final", "report", "statistics", "status"),
        default="all",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data", type=Path, default=Path("dataset/mr_ntp_all_entities"))
    parser.add_argument(
        "--output", type=Path, default=Path("mindreader_classic_baselines_outputs")
    )
    parser.add_argument("--proportions", nargs="+", type=float, default=list(PROPORTIONS))
    parser.add_argument("--validation-seeds", type=int, default=3)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--final-seeds", type=int, default=30)
    parser.add_argument("--final-seed-start", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    if args.stage == "status":
        for model in args.models:
            search_done, search_total, final_done, final_total = counts(args, model)
            print(
                f"{model}: search {search_done}/{search_total}; "
                f"final {final_done}/{final_total}"
            )
        return 0
    data = MindReaderDataset(str(args.data))
    if args.stage == "statistics":
        write_statistics(args, data)
        return 0
    if args.stage == "report":
        for model in args.models:
            write_report(args, model)
        return 0
    write_statistics(args, data)
    for model in args.models:
        lock = acquire_lock(args.output, model)
        try:
            if args.stage in ("all", "search"):
                run_search(args, data, model)
            if args.stage in ("all", "final"):
                run_finals(args, data, model)
            write_report(args, model)
        finally:
            lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
