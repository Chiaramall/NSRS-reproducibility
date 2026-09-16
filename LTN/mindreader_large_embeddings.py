"""Targeted MindReader study for embedding sizes 64, 128 and 256.

The previously selected extended-grid configuration is the control for each
model/proportion.  Search varies only ``emb_size`` and compares that control
against 64, 128 and 256 over paired validation seeds.  The selected
configuration is then evaluated over 30 paired final seeds.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from movielens_paper_benchmark.search import atomic_json_dump

from ltnrec.data import MindReaderDataset
from ltnrec.loaders import (
    TrainingDataLoader,
    TrainingDataLoaderLTN,
    TrainingDataLoaderLTNGenres,
    ValDataLoader,
)
from ltnrec.models import (
    LTNTrainerMF,
    LTNTrainerMFGenres,
    MFTrainer,
    MatrixFactorization,
)


MODELS = ("mf", "ltn", "ltn-genres")
PROPORTIONS = (1.0, 0.8, 0.6, 0.4, 0.2)
LARGE_DIMENSIONS = (64, 128, 256)
PROTOCOL_VERSION = 1
REUSABLE_RESULTS = {
    "mf": Path("results/results_mf_genres_extended_30_runs.json"),
    "ltn": Path("results/results_ltn_extended_grid_30_runs.json"),
    "ltn-genres": Path("results/results_mf_genres_extended_30_runs.json"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def base_config(model: str, proportion: float) -> Dict[str, Any]:
    path = Path("config/mindreader_controls.json")
    value = json.loads(path.read_text(encoding="utf-8"))[model][str(float(proportion))]
    if not isinstance(value, dict):
        raise ValueError(f"invalid base configuration: {path}")
    return value


def candidate_dimensions(
    configuration: Mapping[str, Any],
    large_dimensions: Sequence[int],
) -> Tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(configuration["emb_size"]),
                *(int(value) for value in large_dimensions),
            }
        )
    )


def protocol_fingerprint(
    models: Sequence[str],
    proportions: Sequence[float],
    large_dimensions: Sequence[int],
    validation_seeds: int,
    final_seeds: int,
    max_epochs: int,
    patience: int,
) -> str:
    payload: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "models": list(models),
        "proportions": [float(value) for value in proportions],
        "large_dimensions": [int(value) for value in large_dimensions],
        "validation_seeds": int(validation_seeds),
        "final_seeds": int(final_seeds),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "selection_objective": "validation hit@10",
        "evaluation": "original MindReader split: 1 positive + 100 negatives",
        "base_configurations": {
            f"{model}/{proportion:.1f}": base_config(model, proportion)
            for model in models
            for proportion in proportions
        },
        "implementation": file_digest(Path(__file__).resolve()),
        "dataset": {
            name: file_digest(
                Path("dataset/mr_ntp_all_entities") / name
            )
            for name in ("meta.json", "split.json")
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_trainer(
    model_name: str,
    data: MindReaderDataset,
    fold: Any,
    configuration: Mapping[str, Any],
    work_path: Path,
):
    model = MatrixFactorization(
        data.n_users,
        data.n_movies,
        int(configuration["emb_size"]),
        biased=bool(configuration["mf_bias"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(configuration["lr"]),
        weight_decay=float(configuration["reg_coeff"]),
    )
    batch_size = int(configuration["batch_size"])
    if model_name == "mf":
        loader = TrainingDataLoader(fold, batch_size)
        trainer = MFTrainer(model, optimizer)
    elif model_name == "ltn":
        loader = TrainingDataLoaderLTN(fold, batch_size)
        trainer = LTNTrainerMF(
            model, optimizer, float(configuration["alpha"])
        )
    elif model_name == "ltn-genres":
        loader = TrainingDataLoaderLTNGenres(
            fold,
            data.get_user_genre_ratings_dict(),
            data.movie_to_genres,
            batch_size,
        )
        trainer = LTNTrainerMFGenres(
            model,
            optimizer,
            float(configuration["alpha"]),
            int(configuration["p"]),
        )
    else:
        raise ValueError(f"unsupported model: {model_name}")
    work_path.parent.mkdir(parents=True, exist_ok=True)
    return trainer, loader


def train_once(
    model_name: str,
    data: MindReaderDataset,
    proportion: float,
    configuration: Mapping[str, Any],
    seed: int,
    validation_loader: ValDataLoader,
    work_path: Path,
    max_epochs: int,
    patience: int,
) -> Tuple[Any, float]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    fold = data.get_user_movie_ratings(keep=float(proportion))
    trainer, train_loader = make_trainer(
        model_name, data, fold, configuration, work_path
    )
    work_path.unlink(missing_ok=True)
    trainer.train(
        train_loader,
        validation_loader,
        "hit@10",
        n_epochs=int(max_epochs),
        early=int(patience),
        verbose=20,
        save_path=str(work_path),
    )
    if not work_path.is_file():
        raise RuntimeError(f"training produced no checkpoint: {work_path}")
    trainer.load_model(str(work_path))
    score = float(trainer.test(validation_loader, "hit@10")["hit@10"])
    return trainer, score


def search_path(output: Path, model: str, proportion: float) -> Path:
    return output / "search" / model / f"{proportion:.1f}.json"


def run_path(
    output: Path, model: str, proportion: float, seed: int
) -> Path:
    return output / "runs" / model / f"{proportion:.1f}" / f"seed-{seed}.json"


def load_search_state(
    path: Path,
    model: str,
    proportion: float,
    base: Mapping[str, Any],
    dimensions: Sequence[int],
    validation_seeds: int,
    fingerprint: str,
) -> Dict[str, Any]:
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "model": model,
        "proportion": float(proportion),
        "base_configuration": dict(base),
        "candidate_dimensions": [int(value) for value in dimensions],
        "validation_seeds": int(validation_seeds),
        "records": [],
        "selected_configuration": None,
        "created_at": utc_now(),
    }
    if not path.exists():
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "protocol_version",
        "protocol_fingerprint",
        "model",
        "proportion",
        "base_configuration",
        "candidate_dimensions",
        "validation_seeds",
    )
    changed = [key for key in keys if state.get(key) != expected.get(key)]
    if changed:
        raise RuntimeError(
            f"incompatible search state {path}; changed fields: {changed}"
        )
    return state


def refresh_search_state(state: Dict[str, Any]) -> None:
    required = int(state["validation_seeds"])
    complete = [
        record
        for record in state["records"]
        if len(record.get("validation_scores", [])) == required
    ]
    for record in complete:
        values = np.asarray(record["validation_scores"], dtype=float)
        record["mean_validation_hit@10"] = float(values.mean())
        record["std_validation_hit@10"] = float(values.std(ddof=0))
    if complete:
        best = max(
            complete,
            key=lambda row: (
                float(row["mean_validation_hit@10"]),
                -int(row["configuration"]["emb_size"]),
            ),
        )
        state["selected_configuration"] = dict(best["configuration"])
        state["best_validation_hit@10"] = float(
            best["mean_validation_hit@10"]
        )
    state["completed_fits"] = sum(
        len(record.get("validation_scores", []))
        for record in state["records"]
    )
    state["planned_fits"] = (
        len(state["candidate_dimensions"]) * required
    )
    state["updated_at"] = utc_now()


def search(
    data: MindReaderDataset,
    output: Path,
    models: Sequence[str],
    proportions: Sequence[float],
    large_dimensions: Sequence[int],
    validation_seeds: int,
    max_epochs: int,
    patience: int,
    fingerprint: str,
) -> None:
    validation_loader = ValDataLoader(data.validation, 256)
    for model_name in models:
        for proportion in proportions:
            base = base_config(model_name, proportion)
            dimensions = candidate_dimensions(base, large_dimensions)
            path = search_path(output, model_name, proportion)
            state = load_search_state(
                path,
                model_name,
                proportion,
                base,
                dimensions,
                validation_seeds,
                fingerprint,
            )
            by_dimension = {
                int(record["configuration"]["emb_size"]): record
                for record in state["records"]
            }
            for dimension in dimensions:
                configuration = dict(base)
                configuration["emb_size"] = int(dimension)
                record = by_dimension.get(int(dimension))
                if record is None:
                    record = {
                        "configuration": configuration,
                        "validation_seed_values": [],
                        "validation_scores": [],
                        "training_seconds": [],
                    }
                    state["records"].append(record)
                    by_dimension[int(dimension)] = record
                    refresh_search_state(state)
                    atomic_json_dump(state, path)
                for seed_index in range(
                    len(record["validation_scores"]), validation_seeds
                ):
                    seed = int(seed_index)
                    print(
                        f"[{utc_now()}] search {model_name} p={proportion:.1f} "
                        f"dim={dimension} validation seed "
                        f"{seed_index + 1}/{validation_seeds}",
                        flush=True,
                    )
                    work_path = (
                        output
                        / "work"
                        / f"search-{model_name}-{proportion:.1f}-"
                        f"dim-{dimension}-seed-{seed}.pth"
                    )
                    started = time.perf_counter()
                    trainer, score = train_once(
                        model_name,
                        data,
                        proportion,
                        configuration,
                        seed,
                        validation_loader,
                        work_path,
                        max_epochs,
                        patience,
                    )
                    del trainer
                    work_path.unlink(missing_ok=True)
                    record["validation_seed_values"].append(seed)
                    record["validation_scores"].append(score)
                    record["training_seconds"].append(
                        time.perf_counter() - started
                    )
                    refresh_search_state(state)
                    atomic_json_dump(state, path)
            refresh_search_state(state)
            atomic_json_dump(state, path)
            print(
                f"[{utc_now()}] selected {model_name} p={proportion:.1f}: "
                f"{state['selected_configuration']} "
                f"(validation hit@10={state['best_validation_hit@10']:.6f})",
                flush=True,
            )


def reusable_results() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for model, path in REUSABLE_RESULTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[model] = payload[model]
    return result


def final(
    data: MindReaderDataset,
    output: Path,
    models: Sequence[str],
    proportions: Sequence[float],
    final_seeds: int,
    max_epochs: int,
    patience: int,
    fingerprint: str,
) -> None:
    validation_loader = ValDataLoader(data.validation, 256)
    test_loader = ValDataLoader(data.test, 256)
    reusable = reusable_results()
    for model_name in models:
        for proportion in proportions:
            state_file = search_path(output, model_name, proportion)
            if not state_file.exists():
                raise RuntimeError(f"missing search state: {state_file}")
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("protocol_fingerprint") != fingerprint:
                raise RuntimeError(f"incompatible search state: {state_file}")
            configuration = state.get("selected_configuration")
            if not isinstance(configuration, dict):
                raise RuntimeError(f"incomplete search state: {state_file}")
            base = base_config(model_name, proportion)
            can_reuse = configuration == base
            for seed in range(final_seeds):
                path = run_path(output, model_name, proportion, seed)
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        payload.get("protocol_fingerprint") == fingerprint
                        and payload.get("configuration") == configuration
                    ):
                        continue
                    raise RuntimeError(f"incompatible final result: {path}")
                print(
                    f"[{utc_now()}] final {model_name} p={proportion:.1f} "
                    f"seed {seed + 1}/{final_seeds}",
                    flush=True,
                )
                if can_reuse:
                    metrics = reusable[model_name][str(seed)][
                        str(float(proportion))
                    ]
                    seconds = 0.0
                    source = "reused exact extended-search final run"
                else:
                    work_path = (
                        output
                        / "work"
                        / f"final-{model_name}-{proportion:.1f}-seed-{seed}.pth"
                    )
                    started = time.perf_counter()
                    trainer, _ = train_once(
                        model_name,
                        data,
                        proportion,
                        configuration,
                        seed,
                        validation_loader,
                        work_path,
                        max_epochs,
                        patience,
                    )
                    metrics = trainer.test(
                        test_loader, ["hit@10", "ndcg@10"]
                    )
                    seconds = time.perf_counter() - started
                    del trainer
                    work_path.unlink(missing_ok=True)
                    source = "new targeted-large-embedding run"
                atomic_json_dump(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "protocol_fingerprint": fingerprint,
                        "model": model_name,
                        "proportion": float(proportion),
                        "seed": int(seed),
                        "configuration": configuration,
                        "metrics": {
                            name: float(value)
                            for name, value in metrics.items()
                        },
                        "training_seconds": float(seconds),
                        "source": source,
                        "completed_at": utc_now(),
                    },
                    path,
                )


def status(
    output: Path,
    models: Sequence[str],
    proportions: Sequence[float],
    large_dimensions: Sequence[int],
    validation_seeds: int,
    final_seeds: int,
) -> Dict[str, int]:
    planned_search = sum(
        len(candidate_dimensions(base_config(model, proportion), large_dimensions))
        * validation_seeds
        for model in models
        for proportion in proportions
    )
    completed_search = 0
    for model in models:
        for proportion in proportions:
            path = search_path(output, model, proportion)
            if path.exists():
                try:
                    completed_search += int(
                        json.loads(path.read_text(encoding="utf-8")).get(
                            "completed_fits", 0
                        )
                    )
                except (OSError, ValueError, TypeError):
                    pass
    planned_final = len(models) * len(proportions) * final_seeds
    completed_final = sum(
        1
        for model in models
        for proportion in proportions
        for seed in range(final_seeds)
        if run_path(output, model, proportion, seed).is_file()
    )
    return {
        "completed_search": completed_search,
        "planned_search": planned_search,
        "completed_final": completed_final,
        "planned_final": planned_final,
    }


def write_reports(
    output: Path,
    models: Sequence[str],
    proportions: Sequence[float],
    final_seeds: int,
) -> None:
    summary: Dict[str, Any] = {}
    selected: Dict[str, Any] = {}
    for proportion in proportions:
        label = f"{proportion:.1f}"
        summary[label] = {}
        selected[label] = {}
        for model in models:
            state = json.loads(
                search_path(output, model, proportion).read_text(
                    encoding="utf-8"
                )
            )
            selected[label][model] = state["selected_configuration"]
            records = [
                json.loads(
                    run_path(output, model, proportion, seed).read_text(
                        encoding="utf-8"
                    )
                )
                for seed in range(final_seeds)
                if run_path(output, model, proportion, seed).is_file()
            ]
            metrics: Dict[str, Any] = {}
            for name in ("hit@10", "ndcg@10"):
                values = np.asarray(
                    [record["metrics"][name] for record in records],
                    dtype=float,
                )
                if len(values):
                    metrics[name] = {
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "n": int(len(values)),
                    }
            summary[label][model] = metrics
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(summary, reports / "summary.json")
    atomic_json_dump(selected, reports / "selected_hyperparameters.json")
    lines = [
        "# Targeted large-embedding MindReader results",
        "",
        "Values are test mean ± population standard deviation. The targeted "
        "search compares the previous selected dimension with 64, 128 and 256 "
        "while holding all other hyperparameters fixed.",
        "",
        "| Training ratings | Metric | "
        + " | ".join(models)
        + " |",
        "|---|---|" + "---:|" * len(models),
    ]
    for proportion in proportions:
        label = f"{proportion:.1f}"
        for metric in ("hit@10", "ndcg@10"):
            cells = []
            for model in models:
                value = summary[label][model].get(metric)
                cells.append(
                    "—"
                    if value is None
                    else (
                        f"{value['mean']:.4f} ± {value['std']:.4f} "
                        f"(n={value['n']})"
                    )
                )
            lines.append(
                f"| {proportion * 100:.0f}% | {metric} | "
                + " | ".join(cells)
                + " |"
            )
    (reports / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def handle_sigterm(signum: int, frame: Any) -> None:
    del signum, frame
    raise KeyboardInterrupt


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".lock"
    pid_path = output / "benchmark.pid"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another process owns {output}") from error
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_sigterm)
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    try:
        yield
    finally:
        try:
            if pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass
        signal.signal(signal.SIGTERM, previous)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("plan", "status", "search", "final", "report", "all"),
        default="all",
    )
    parser.add_argument("--output", default="mindreader_large_embeddings_outputs")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--proportions", nargs="+", type=float, default=list(PROPORTIONS)
    )
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=list(LARGE_DIMENSIONS)
    )
    parser.add_argument("--validation-seeds", type=int, default=3)
    parser.add_argument("--final-seeds", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = tuple(args.models)
    proportions = tuple(float(value) for value in args.proportions)
    dimensions = tuple(int(value) for value in args.dimensions)
    if any(value not in PROPORTIONS for value in proportions):
        raise ValueError("unsupported training proportion")
    if any(value <= 0 for value in dimensions):
        raise ValueError("embedding dimensions must be positive")
    if min(args.validation_seeds, args.final_seeds, args.max_epochs, args.patience) <= 0:
        raise ValueError("seed counts, epochs and patience must be positive")
    torch.set_num_threads(int(args.torch_threads))
    output = Path(args.output)
    fingerprint = protocol_fingerprint(
        models,
        proportions,
        dimensions,
        args.validation_seeds,
        args.final_seeds,
        args.max_epochs,
        args.patience,
    )
    plan = status(
        output,
        models,
        proportions,
        dimensions,
        args.validation_seeds,
        args.final_seeds,
    )
    print(
        f"Search: {plan['completed_search']}/{plan['planned_search']}; "
        f"final: {plan['completed_final']}/{plan['planned_final']}",
        flush=True,
    )
    if args.stage in {"plan", "status"}:
        return 0
    data = MindReaderDataset(data_path="./dataset/mr_ntp_all_entities")
    with output_lock(output):
        try:
            if args.stage in {"search", "all"}:
                search(
                    data,
                    output,
                    models,
                    proportions,
                    dimensions,
                    args.validation_seeds,
                    args.max_epochs,
                    args.patience,
                    fingerprint,
                )
            if args.stage in {"final", "all"}:
                final(
                    data,
                    output,
                    models,
                    proportions,
                    args.final_seeds,
                    args.max_epochs,
                    args.patience,
                    fingerprint,
                )
            if args.stage in {"report", "final", "all"}:
                write_reports(output, models, proportions, args.final_seeds)
        except KeyboardInterrupt:
            print(
                "Interrupted safely; rerun the same command to resume.",
                flush=True,
            )
            return 130
    plan = status(
        output,
        models,
        proportions,
        dimensions,
        args.validation_seeds,
        args.final_seeds,
    )
    print(
        f"Complete. Search: {plan['completed_search']}/"
        f"{plan['planned_search']}; final: "
        f"{plan['completed_final']}/{plan['planned_final']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
