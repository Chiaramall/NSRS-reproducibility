#!/usr/bin/env python3
"""Run the official PGPR Amazon pipeline for multiple seeds.

PGPR stores the active TransE embedding in one shared file, so seeds are
deliberately run sequentially. A per-seed copy is retained for safe resume.
Optionally, one TransE embedding can be shared across policy-agent seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PGPR_ROOT = ROOT / "baselines" / "models" / "pgpr"
DATASETS = {
    "beauty": ("beauty", "Amazon_Beauty"),
    "cell": ("cell", "Amazon_Cellphones"),
    "clothing": ("cloth", "Amazon_Clothing"),
    "cd": ("cd", "Amazon_CDs"),
}
METRIC_PATTERN = re.compile(
    r"NDCG=(?P<ndcg>[\d.]+)\s*\|\s*"
    r"Recall=(?P<recall>[\d.]+)\s*\|\s*"
    r"HR=(?P<hr>[\d.]+)\s*\|\s*"
    r"Precision=(?P<precision>[\d.]+)\s*\|\s*"
    r"Invalid users=(?P<invalid_users>\d+)"
)
METRICS = ("ndcg", "recall", "hr", "precision")


def run_command(command: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PGPR_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            captured.append(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return "".join(captured)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: dict[str, object] = {"model": "PGPR", "seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = statistics.fmean(values)
        summary[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
    invalid = [int(row["invalid_users"]) for row in rows]
    summary["invalid_users_mean"] = statistics.fmean(invalid)
    return [summary]


def ensure_preprocessed(args: argparse.Namespace) -> None:
    preprocess_files = (
        args.pgpr_tmp / "dataset.pkl",
        args.pgpr_tmp / "kg.pkl",
        args.pgpr_tmp / "train_label.pkl",
        args.pgpr_tmp / "test_label.pkl",
    )
    if args.force_preprocess or any(not path.exists() for path in preprocess_files):
        run_command(
            [str(args.python), "preprocess.py", "--dataset", args.pgpr_dataset],
            args.results_dir / "logs" / "preprocess.log",
        )
    missing = [path for path in preprocess_files if not path.exists()]
    if missing:
        raise RuntimeError(f"PGPR preprocessing did not create: {missing}")


def train_transe(args: argparse.Namespace, seed: int, artifact_dir: Path) -> None:
    seed_embedding = artifact_dir / "transe_embed.pkl"
    primary_seed = args.seeds[0]
    if args.shared_transe and seed != primary_seed:
        primary_embedding = (
            args.results_dir
            / "artifacts"
            / f"seed_{primary_seed}"
            / "transe_embed.pkl"
        )
        if not primary_embedding.exists():
            raise RuntimeError(
                "Shared TransE embedding is missing for primary "
                f"seed={primary_seed}: {primary_embedding}"
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary_embedding, seed_embedding)
        shutil.copy2(primary_embedding, args.pgpr_tmp / "transe_embed.pkl")
        print(
            f"Reusing TransE embedding from seed={primary_seed} "
            f"for policy seed={seed}",
            flush=True,
        )
        return

    if seed_embedding.exists():
        shutil.copy2(seed_embedding, args.pgpr_tmp / "transe_embed.pkl")
        return

    name = f"paper_transe_e{args.transe_epochs}_seed_{seed}"
    command = [
        str(args.python),
        "train_transe_model.py",
        "--dataset",
        args.pgpr_dataset,
        "--name",
        name,
        "--seed",
        str(seed),
        "--gpu",
        args.gpu,
        "--epochs",
        str(args.transe_epochs),
        "--batch_size",
        str(args.transe_batch_size),
    ]
    run_command(command, args.results_dir / "logs" / f"seed_{seed}_transe.log")
    shared_embedding = args.pgpr_tmp / "transe_embed.pkl"
    if not shared_embedding.exists():
        raise RuntimeError("PGPR TransE training did not create transe_embed.pkl")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_embedding, seed_embedding)


def train_agent(args: argparse.Namespace, seed: int) -> str:
    name = f"paper_pgpr_e{args.agent_epochs}_seed_{seed}"
    policy = args.pgpr_tmp / name / f"policy_model_epoch_{args.agent_epochs}.ckpt"
    if not policy.exists():
        command = [
            str(args.python),
            "train_agent.py",
            "--dataset",
            args.pgpr_dataset,
            "--name",
            name,
            "--seed",
            str(seed),
            "--gpu",
            args.gpu,
            "--epochs",
            str(args.agent_epochs),
        ]
        run_command(command, args.results_dir / "logs" / f"seed_{seed}_agent.log")
    if not policy.exists():
        raise RuntimeError(f"PGPR agent training did not create {policy}")
    return name


def evaluate(args: argparse.Namespace, seed: int, name: str) -> dict[str, object]:
    path_file = args.pgpr_tmp / name / f"policy_paths_epoch{args.agent_epochs}.pkl"
    command = [
        str(args.python),
        "test_agent.py",
        "--dataset",
        args.pgpr_dataset,
        "--name",
        name,
        "--seed",
        str(seed),
        "--gpu",
        args.gpu,
        "--epochs",
        str(args.agent_epochs),
        "--run_path",
        str(not path_file.exists()),
        "--run_eval",
        "True",
        "--add_products",
        str(args.add_products),
    ]
    output = run_command(command, args.results_dir / "logs" / f"seed_{seed}_test.log")
    matches = list(METRIC_PATTERN.finditer(output))
    if not matches:
        raise RuntimeError("Could not parse PGPR top-10 metrics from test output")
    parsed = matches[-1].groupdict()
    return {
        "dataset": args.dataset,
        "model": "PGPR",
        "seed": seed,
        "transe_epochs": args.transe_epochs,
        "agent_epochs": args.agent_epochs,
        "add_products": args.add_products,
        "ndcg": float(parsed["ndcg"]),
        "recall": float(parsed["recall"]),
        "hr": float(parsed["hr"]),
        "precision": float(parsed["precision"]),
        "invalid_users": int(parsed["invalid_users"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="beauty")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[123, 124, 125, 126, 127]
    )
    parser.add_argument("--transe-epochs", type=int, default=30)
    parser.add_argument("--transe-batch-size", type=int, default=64)
    parser.add_argument("--agent-epochs", type=int, default=50)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--add-products",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill fewer-than-10 reachable paths with TransE-ranked products.",
    )
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument(
        "--shared-transe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train TransE once using the first seed and reuse it for all agents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pgpr_dataset, pgpr_directory = DATASETS[args.dataset]
    args.pgpr_tmp = PGPR_ROOT / "tmp" / pgpr_directory
    if args.results_dir is None:
        args.results_dir = (
            ROOT / "results" / "paper_baselines" / args.dataset / "pgpr"
        )
    if not args.python.is_absolute():
        args.python = ROOT / args.python
    args.python = args.python.absolute()
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.python.exists():
        raise RuntimeError(f"Python executable not found: {args.python}")
    if args.transe_batch_size < 1:
        raise ValueError("--transe-batch-size must be positive")
    protocol = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "transe_epochs": args.transe_epochs,
        "transe_batch_size": args.transe_batch_size,
        "agent_epochs": args.agent_epochs,
        "shared_transe": args.shared_transe,
        "shared_transe_seed": args.seeds[0] if args.shared_transe else None,
        "add_products": args.add_products,
    }
    (args.results_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    ensure_preprocessed(args)
    if args.preprocess_only:
        return

    raw_path = args.results_dir / "per_seed.csv"
    rows: list[dict[str, object]] = []
    if raw_path.exists():
        with raw_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    completed = {
        (
            int(row["seed"]),
            int(row["transe_epochs"]),
            int(row["agent_epochs"]),
            str(row["add_products"]).lower() == "true",
        )
        for row in rows
    }

    for seed in args.seeds:
        key = (seed, args.transe_epochs, args.agent_epochs, args.add_products)
        if key in completed:
            print(f"Skipping completed PGPR seed={seed}", flush=True)
            continue
        artifact_dir = args.results_dir / "artifacts" / f"seed_{seed}"
        train_transe(args, seed, artifact_dir)
        name = train_agent(args, seed)
        row = evaluate(args, seed, name)
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
