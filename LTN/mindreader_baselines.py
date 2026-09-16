"""Resumable EASE and LightGCN benchmark on the original MindReader protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from ltnrec.candidates import candidate_arrays
from ltnrec.data import MindReaderDataset
from ltnrec.metrics import compute_metric
from movielens_paper_benchmark.models.ease import EASE
from movielens_paper_benchmark.models.lightgcn import LightGCN
from movielens_paper_benchmark.search import atomic_json_dump, sample_configurations


PROTOCOL_VERSION = 1
DEFAULT_PROPORTIONS = (1.0, 0.8, 0.6, 0.4, 0.2)
SPACES = {
    "ease": {
        "regularization": [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
    },
    "lightgcn": {
        "embedding_dim": [64, 128, 256],
        "num_layers": [1, 2, 3, 4],
        "learning_rate": [0.0001, 0.0005, 0.001],
        "l2": [0.00001, 0.0001, 0.001],
        "batch_size": [512, 1024, 2048],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def p_label(proportion: float) -> str:
    return f"{float(proportion):.1f}"


def positive_pairs(fold: Iterable[Sequence[float]]) -> np.ndarray:
    pairs = [(int(user), int(item)) for user, item, rating in fold if float(rating) > 0]
    result = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(result):
        raise ValueError("the sparse fold contains no positive movie ratings")
    return result


def positive_exclusions(pairs: np.ndarray, n_users: int) -> Dict[int, set[int]]:
    result = {user: set() for user in range(int(n_users))}
    for user, item in pairs.tolist():
        result[int(user)].add(int(item))
    return result


def sparse_fold(data: MindReaderDataset, proportion: float, seed: int):
    # get_user_movie_ratings only calls np.random.seed for truthy seeds, so seed
    # the global NumPy RNG explicitly to preserve seed zero as a valid run.
    np.random.seed(int(seed))
    return data.get_user_movie_ratings(keep=float(proportion), seed=int(seed))


def evaluate(model: Any, rows: Sequence[np.ndarray]) -> Dict[str, float]:
    users, items = candidate_arrays(rows)
    scores = np.asarray(model.score_users(users, items), dtype=np.float64)
    if scores.shape != items.shape:
        raise ValueError(f"score shape {scores.shape} does not match candidates {items.shape}")
    ground_truth = np.zeros_like(scores)
    ground_truth[:, -1] = 1.0
    return {
        "hit@10": float(np.mean(compute_metric("hit@10", scores, ground_truth))),
        "ndcg@10": float(np.mean(compute_metric("ndcg@10", scores, ground_truth))),
    }


def create_and_fit(
    model_name: str,
    parameters: Mapping[str, Any],
    data: MindReaderDataset,
    fold: Sequence[Sequence[float]],
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
):
    pairs = positive_pairs(fold)
    if model_name == "ease":
        model = EASE(
            regularization=float(parameters["regularization"]),
            n_users=data.n_users,
            n_items=data.n_movies,
            dtype=np.float64,
        )
        model.fit(pairs)
        return model
    if model_name != "lightgcn":
        raise ValueError(f"unsupported model {model_name!r}")
    model = LightGCN(
        n_users=data.n_users,
        n_items=data.n_movies,
        embedding_dim=int(parameters["embedding_dim"]),
        num_layers=int(parameters["num_layers"]),
        learning_rate=float(parameters["learning_rate"]),
        l2_reg=float(parameters["l2"]),
        batch_size=int(parameters["batch_size"]),
        epochs=int(max_epochs),
        seed=int(seed),
        device=device,
    )
    model.fit(
        pairs,
        forbidden_items_by_user=positive_exclusions(pairs, data.n_users),
        validation_callback=lambda fitted: evaluate(fitted, data.validation)["hit@10"],
        patience=int(patience),
        seed=int(seed),
    )
    return model


def configurations(model: str, budget: int, search_seed: int) -> list[Dict[str, Any]]:
    space = SPACES[model]
    effective_budget = min(int(budget), 6 if model == "ease" else int(budget))
    return sample_configurations(space, effective_budget, int(search_seed))


def search_path(output: Path, model: str, proportion: float) -> Path:
    return output / model / "search" / f"{p_label(proportion)}.json"


def run_path(output: Path, model: str, proportion: float, seed_index: int) -> Path:
    return output / model / "runs" / p_label(proportion) / f"seed-{seed_index}.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def initial_search_state(
    model: str,
    proportion: float,
    configs: Sequence[Mapping[str, Any]],
    validation_seeds: int,
    search_seed: int,
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": model,
        "proportion": float(proportion),
        "validation_seeds": int(validation_seeds),
        "search_seed": int(search_seed),
        "objective": "hit@10",
        "records": [
            {"index": index, "parameters": dict(config), "validation": []}
            for index, config in enumerate(configs)
        ],
        "best_parameters": None,
        "best_validation_hit@10": None,
        "updated_at": utc_now(),
    }


def refresh_search(state: Dict[str, Any]) -> None:
    required = int(state["validation_seeds"])
    complete = []
    for record in state["records"]:
        scores = record["validation"]
        if len(scores) == required:
            record["mean_validation_hit@10"] = float(
                np.mean([entry["hit@10"] for entry in scores])
            )
            complete.append(record)
    best = None
    for record in complete:
        if best is None or record["mean_validation_hit@10"] > best["mean_validation_hit@10"]:
            best = record
    state["completed_fits"] = sum(len(record["validation"]) for record in state["records"])
    state["planned_fits"] = len(state["records"]) * required
    state["best_parameters"] = None if best is None else dict(best["parameters"])
    state["best_validation_hit@10"] = (
        None if best is None else float(best["mean_validation_hit@10"])
    )
    state["updated_at"] = utc_now()


def run_search(args, data: MindReaderDataset, model: str, device: str) -> None:
    configs = configurations(model, args.search_budget, args.search_seed)
    for proportion in args.proportions:
        path = search_path(args.output, model, proportion)
        if path.exists():
            state = load_json(path)
            expected = [dict(item) for item in configs]
            actual = [record["parameters"] for record in state["records"]]
            if (
                state.get("protocol_version") != PROTOCOL_VERSION
                or actual != expected
                or int(state["validation_seeds"]) != args.validation_seeds
            ):
                raise RuntimeError(f"incompatible existing search state: {path}")
        else:
            state = initial_search_state(
                model, proportion, configs, args.validation_seeds, args.search_seed
            )
            atomic_json_dump(state, path)
        for record in state["records"]:
            for seed_index in range(len(record["validation"]), args.validation_seeds):
                seed = args.search_seed + seed_index
                print(
                    f"[{utc_now()}] search {model} p={p_label(proportion)} "
                    f"config {record['index'] + 1}/{len(configs)} "
                    f"validation seed {seed_index + 1}/{args.validation_seeds}",
                    flush=True,
                )
                fold = sparse_fold(data, proportion, seed)
                started = time.perf_counter()
                fitted = create_and_fit(
                    model,
                    record["parameters"],
                    data,
                    fold,
                    seed,
                    device,
                    args.max_epochs,
                    args.patience,
                )
                metrics = evaluate(fitted, data.validation)
                record["validation"].append(
                    {
                        "seed_index": seed_index,
                        "seed": seed,
                        "hit@10": metrics["hit@10"],
                        "ndcg@10": metrics["ndcg@10"],
                        "training_seconds": time.perf_counter() - started,
                    }
                )
                refresh_search(state)
                atomic_json_dump(state, path)
                del fitted
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        refresh_search(state)
        atomic_json_dump(state, path)


def run_finals(args, data: MindReaderDataset, model: str, device: str) -> None:
    for proportion in args.proportions:
        state_file = search_path(args.output, model, proportion)
        if not state_file.exists():
            raise RuntimeError(f"search is missing: {state_file}")
        state = load_json(state_file)
        parameters = state.get("best_parameters")
        if parameters is None:
            raise RuntimeError(f"search is incomplete: {state_file}")
        for seed_index in range(args.final_seeds):
            path = run_path(args.output, model, proportion, seed_index)
            if path.exists():
                continue
            seed = args.final_seed_start + seed_index
            print(
                f"[{utc_now()}] final {model} p={p_label(proportion)} "
                f"seed {seed_index + 1}/{args.final_seeds}",
                flush=True,
            )
            fold = sparse_fold(data, proportion, seed)
            started = time.perf_counter()
            fitted = create_and_fit(
                model,
                parameters,
                data,
                fold,
                seed,
                device,
                args.max_epochs,
                args.patience,
            )
            metrics = evaluate(fitted, data.test)
            atomic_json_dump(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "model": model,
                    "proportion": float(proportion),
                    "seed_index": seed_index,
                    "seed": seed,
                    "parameters": parameters,
                    "metrics": metrics,
                    "training_seconds": time.perf_counter() - started,
                    "created_at": utc_now(),
                },
                path,
            )
            del fitted
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def model_counts(args, model: str) -> tuple[int, int, int, int]:
    configs = configurations(model, args.search_budget, args.search_seed)
    planned_search = len(args.proportions) * len(configs) * args.validation_seeds
    done_search = 0
    done_finals = 0
    for proportion in args.proportions:
        path = search_path(args.output, model, proportion)
        if path.exists():
            done_search += int(load_json(path).get("completed_fits", 0))
        done_finals += sum(
            run_path(args.output, model, proportion, index).exists()
            for index in range(args.final_seeds)
        )
    return done_search, planned_search, done_finals, len(args.proportions) * args.final_seeds


def write_report(args, model: str) -> None:
    summary: Dict[str, Any] = {"model": model, "proportions": {}}
    lines = [
        f"# MindReader {model} baseline",
        "",
        "Test values are mean ± population standard deviation across final seeds.",
        "",
        "| Training ratings | Hit@10 | NDCG@10 | Selected parameters |",
        "|---:|---:|---:|---|",
    ]
    for proportion in args.proportions:
        records = [
            load_json(run_path(args.output, model, proportion, index))
            for index in range(args.final_seeds)
            if run_path(args.output, model, proportion, index).exists()
        ]
        if not records:
            continue
        search = load_json(search_path(args.output, model, proportion))
        entry: Dict[str, Any] = {
            "n": len(records),
            "selected_parameters": search["best_parameters"],
        }
        displays = {}
        for metric in ("hit@10", "ndcg@10"):
            values = np.asarray([record["metrics"][metric] for record in records])
            entry[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n": int(values.size),
            }
            displays[metric] = f"{values.mean():.4f} ± {values.std(ddof=0):.4f} (n={len(values)})"
        summary["proportions"][p_label(proportion)] = entry
        parameters = ", ".join(
            f"{key}={value}" for key, value in search["best_parameters"].items()
        )
        lines.append(
            f"| {int(round(proportion * 100))}% | {displays['hit@10']} | "
            f"{displays['ndcg@10']} | {parameters} |"
        )
    reports = args.output / model / "reports"
    atomic_json_dump(summary, reports / "summary.json")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_device(requested: str, model: str) -> str:
    if model == "ease":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = str(torch.device(requested))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def acquire_lock(output: Path, model: str):
    directory = output / model
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another {model} writer is active: {path}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "search", "final", "report", "status"), default="all")
    parser.add_argument("--models", nargs="+", choices=("ease", "lightgcn"), default=["ease", "lightgcn"])
    parser.add_argument("--data", type=Path, default=Path("dataset/mr_ntp_all_entities"))
    parser.add_argument("--output", type=Path, default=Path("mindreader_baselines_outputs"))
    parser.add_argument("--proportions", nargs="+", type=float, default=list(DEFAULT_PROPORTIONS))
    parser.add_argument("--validation-seeds", type=int, default=3)
    parser.add_argument("--final-seeds", type=int, default=30)
    parser.add_argument("--final-seed-start", type=int, default=0)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--search-budget", type=int, default=36)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.validation_seeds, args.final_seeds, args.search_budget, args.torch_threads) <= 0:
        parser.error("seed counts, search budget, and thread count must be positive")
    if any(not 0 < value <= 1 for value in args.proportions):
        parser.error("proportions must be in (0,1]")
    return args


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    args.output = args.output.resolve()
    if args.stage == "status":
        for model in args.models:
            search_done, search_total, final_done, final_total = model_counts(args, model)
            print(
                f"{model}: search {search_done}/{search_total} "
                f"({100 * search_done / search_total:.1f}%); "
                f"final {final_done}/{final_total} "
                f"({100 * final_done / final_total:.1f}%)"
            )
        return 0
    if args.stage == "report":
        for model in args.models:
            write_report(args, model)
        return 0

    data = MindReaderDataset(str(args.data))
    for model in args.models:
        device = resolve_device(args.device, model)
        print(f"{model}: device={device}, output={args.output / model}", flush=True)
        lock = acquire_lock(args.output, model)
        try:
            if args.stage in {"all", "search"}:
                run_search(args, data, model, device)
            if args.stage in {"all", "final"}:
                run_finals(args, data, model, device)
            write_report(args, model)
        finally:
            lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed JSON states can be resumed.", file=sys.stderr)
        raise
