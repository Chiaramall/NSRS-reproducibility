#!/usr/bin/env python3
"""Run and summarize CAFE reproducibility and hyperparameter experiments.

For any released dataset, run the default SGD hyperparameters with seeds
123..127 and the upstream ten stochastic program executions per trained model.
The best seed is selected by mean NDCG. The grid reuses that seed and searches
lr x rank_weight.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_NAMES = {
    "beauty": "Beauty",
    "cell": "Cellphones_Accessories",
    "clothing": "Clothing",
    "cd": "CDs_Vinyl",
}
DATASET = "beauty"
RESULTS_DIR = ROOT / "results" / "reproducibility"
RUN_LOG_DIR = RESULTS_DIR / "logs"
TMP_DATASET = ROOT / "tmp" / TMP_NAMES[DATASET]
BASELINE_SEEDS = (123, 124, 125, 126, 127)
GRID_LRS = (0.05, 0.1, 0.2)
GRID_RANK_WEIGHTS = (5.0, 10.0, 20.0)
METRIC_RE = re.compile(
    r"NDCG=(?P<ndcg>[0-9.]+)\s*\|\s*"
    r"Recall=(?P<recall>[0-9.]+)\s*\|\s*"
    r"HR=(?P<hr>[0-9.]+)\s*\|\s*"
    r"Precision=(?P<precision>[0-9.]+)"
)
METRICS = ("ndcg", "recall", "hr", "precision")


def run_logged(command: list[str], log_path: Path) -> None:
    """Run a subprocess while streaming and persisting combined output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print(f"\n>>> {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n>>> {printable}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def common_args(name: str, seed: int, lr: float, rank_weight: float) -> list[str]:
    return [
        "--dataset", DATASET, "--name", name, "--seed", str(seed),
        "--epochs", "20", "--batch_size", "128", "--lr", str(lr),
        "--resume", "true",
        "--rank_weight", str(rank_weight), "--topk_candidates", "20",
        "--sample_size", "15", "--gpu", "0",
    ]


def ensure_preprocessed(python: str) -> None:
    expected = (
        TMP_DATASET / "embed.pkl",
        TMP_DATASET / "kg.pkl",
        TMP_DATASET / "user_products_pos.npy",
        TMP_DATASET / "path_count.pkl",
    )
    if all(path.exists() for path in expected):
        print(">>> Preprocessing already complete; skipping.", flush=True)
        return
    TMP_DATASET.mkdir(parents=True, exist_ok=True)
    run_logged(
        [python, "-u", "preprocess.py", "--dataset", DATASET, "--seed", "123"],
        RUN_LOG_DIR / "preprocess.log",
    )
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Preprocessing finished without expected files: {missing}")


def parse_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    for match in METRIC_RE.finditer(path.read_text(encoding="utf-8")):
        rows.append({metric: float(match.group(metric)) for metric in METRICS})
    return rows


def run_experiment(
    python: str,
    *,
    name: str,
    seed: int,
    lr: float = 0.1,
    rank_weight: float = 10.0,
) -> list[dict[str, float]]:
    model_dir = TMP_DATASET / name
    checkpoint = model_dir / "symbolic_model_epoch20.ckpt"
    inferred_paths = model_dir / "infer_path_data.pkl"
    execution_log = model_dir / "program_exe_heuristic_ss15.txt"
    args = common_args(name, seed, lr, rank_weight)

    if not checkpoint.exists():
        run_logged(
            [python, "-u", "train_neural_symbol.py", *args],
            RUN_LOG_DIR / f"{name}_train.log",
        )
    else:
        print(f">>> Found {checkpoint.relative_to(ROOT)}; skipping training.", flush=True)

    if not inferred_paths.exists():
        run_logged(
            [python, "-u", "execute_neural_symbol.py", *args, "--do_infer", "true"],
            RUN_LOG_DIR / f"{name}_infer.log",
        )
    else:
        print(f">>> Found {inferred_paths.relative_to(ROOT)}; skipping inference.", flush=True)

    rows = parse_metrics(execution_log)
    if len(rows) < 10:
        run_logged(
            [python, "-u", "execute_neural_symbol.py", *args, "--do_execute", "true"],
            RUN_LOG_DIR / f"{name}_execute.log",
        )
        rows = parse_metrics(execution_log)
    else:
        print(f">>> Found 10 execution rows for {name}; skipping execution.", flush=True)

    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 metric rows in {execution_log}, found {len(rows)}")
    return rows


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(row[metric] for row in rows)
        for metric in METRICS
    }


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(python: str) -> dict[str, object]:
    ensure_preprocessed(python)
    per_seed = []
    raw = {}
    for seed in BASELINE_SEEDS:
        name = (
            f"repro_baseline_seed_{seed}"
            if DATASET == "beauty"
            else f"repro_sgd_{DATASET}_seed_{seed}"
        )
        rows = run_experiment(python, name=name, seed=seed)
        raw[str(seed)] = rows
        per_seed.append({"seed": seed, **mean_metrics(rows)})
        write_csv(RESULTS_DIR / "baseline_per_seed.csv", per_seed)

    aggregate = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in per_seed]
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "std": sample_std(values),
        }
    best = max(per_seed, key=lambda row: (float(row["ndcg"]), -int(row["seed"])))
    summary: dict[str, object] = {
        "protocol": {
            "dataset": DATASET,
            "seeds": list(BASELINE_SEEDS),
            "executions_per_seed": 10,
            "std": "sample std across five per-seed means (ddof=1)",
            "selection_metric": "mean NDCG",
            "hyperparameters": {
                "epochs": 20, "batch_size": 128, "lr": 0.1,
                "rank_weight": 10.0, "topk_candidates": 20, "sample_size": 15,
            },
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "best_seed": int(best["seed"]),
        "raw_executions": raw,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def load_or_run_baseline(python: str) -> dict[str, object]:
    path = RESULTS_DIR / "baseline_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return run_baseline(python)


def grid_name(seed: int, lr: float, rank_weight: float) -> str:
    lr_token = str(lr).replace(".", "p")
    rank_token = str(rank_weight).replace(".", "p")
    prefix = "grid" if DATASET == "beauty" else f"grid_sgd_{DATASET}"
    return f"{prefix}_seed_{seed}_lr_{lr_token}_rw_{rank_token}"


def run_grid(python: str) -> dict[str, object]:
    baseline = load_or_run_baseline(python)
    seed = int(baseline["best_seed"])
    baseline_by_seed = {
        int(row["seed"]): row for row in baseline["per_seed"]
    }
    grid_rows: list[dict[str, object]] = []
    for lr, rank_weight in itertools.product(GRID_LRS, GRID_RANK_WEIGHTS):
        if lr == 0.1 and rank_weight == 10.0:
            metrics = {
                metric: float(baseline_by_seed[seed][metric]) for metric in METRICS
            }
            source = "reused_baseline"
        else:
            rows = run_experiment(
                python, name=grid_name(seed, lr, rank_weight), seed=seed,
                lr=lr, rank_weight=rank_weight,
            )
            metrics = mean_metrics(rows)
            source = "grid_run"
        grid_rows.append({
            "seed": seed, "lr": lr, "rank_weight": rank_weight,
            **metrics, "source": source,
        })
        write_csv(RESULTS_DIR / "grid_results.csv", grid_rows)

    best = max(
        grid_rows,
        key=lambda row: (
            float(row["ndcg"]), float(row["recall"]),
            -float(row["lr"]), -float(row["rank_weight"]),
        ),
    )
    summary: dict[str, object] = {
        "protocol": {
            "dataset": DATASET, "seed": seed,
            "selection_metric": "mean NDCG over 10 program executions",
            "lr": list(GRID_LRS), "rank_weight": list(GRID_RANK_WEIGHTS),
            "fixed": {
                "epochs": 20, "batch_size": 128,
                "topk_candidates": 20, "sample_size": 15,
            },
        },
        "results": grid_rows,
        "best": best,
    }
    (RESULTS_DIR / "grid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("preprocess", "baseline", "grid", "all"),
        help="Pipeline stage; all runs baseline followed by the grid.",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used for child processes.",
    )
    parser.add_argument(
        "--dataset", choices=tuple(TMP_NAMES), default="beauty",
        help="Released CAFE dataset (default: beauty).",
    )
    return parser.parse_args()


def configure(dataset: str) -> None:
    global DATASET, RESULTS_DIR, RUN_LOG_DIR, TMP_DATASET
    DATASET = dataset
    if dataset == "beauty":
        # Preserve the original paths so ongoing and completed Beauty work is
        # discovered and resumed without moving or duplicating artifacts.
        RESULTS_DIR = ROOT / "results" / "reproducibility"
    else:
        RESULTS_DIR = ROOT / "results" / "reproducibility_sgd" / dataset
    RUN_LOG_DIR = RESULTS_DIR / "logs"
    TMP_DATASET = ROOT / "tmp" / TMP_NAMES[dataset]


def main() -> None:
    args = parse_args()
    configure(args.dataset)
    if args.stage == "preprocess":
        ensure_preprocessed(args.python)
    elif args.stage == "baseline":
        run_baseline(args.python)
    elif args.stage == "grid":
        run_grid(args.python)
    else:
        run_baseline(args.python)
        run_grid(args.python)


if __name__ == "__main__":
    main()
