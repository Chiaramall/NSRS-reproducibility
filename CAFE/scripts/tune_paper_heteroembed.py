#!/usr/bin/env python3
"""Resumable compact dimension selection and final HeteroEmbed evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
OUTPUT = ROOT / "results" / "heteroembed_selection"
FINAL_ROOT = ROOT / "results" / "paper_baselines"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_single_row(path: Path, seed: int) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if int(row["seed"]) == seed
            and row["partial_run"] == "False"
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one completed row in {path}")
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("beauty", "cell", "clothing", "cd"),
        default=["beauty", "cell", "clothing", "cd"],
    )
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=[50, 100, 200, 300]
    )
    parser.add_argument("--tuning-epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--tuning-seed", type=int, default=123)
    parser.add_argument(
        "--final-seeds",
        nargs="+",
        type=int,
        default=[123, 124, 125, 126, 127],
    )
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUTPUT / "protocol.json",
        {
            "selection": "compact multi-fidelity paper dimension search",
            "metric": "inner-validation NDCG@10",
            "dimensions": args.dimensions,
            "tuning_epochs": args.tuning_epochs,
            "final_epochs": args.final_epochs,
            "tuning_seed": args.tuning_seed,
            "final_seeds": args.final_seeds,
            "outer_test_policy": "used only after dimension selection",
        },
    )
    for dataset in args.datasets:
        candidates: list[dict[str, object]] = []
        for dimension in args.dimensions:
            trial_root = OUTPUT / "trials" / dataset / f"dim_{dimension}"
            result_path = (
                trial_root
                / dataset
                / "heteroembed"
                / "per_seed.csv"
            )
            if not result_path.exists():
                run(
                    [
                        str(PYTHON),
                        "-u",
                        "scripts/run_paper_heteroembed.py",
                        "--datasets",
                        dataset,
                        "--split",
                        "inner",
                        "--seeds",
                        str(args.tuning_seed),
                        "--epochs",
                        str(args.tuning_epochs),
                        "--embedding-size",
                        str(dimension),
                        "--gpu",
                        args.gpu,
                        "--results-root",
                        str(trial_root),
                    ]
                )
            row = read_single_row(result_path, args.tuning_seed)
            candidates.append(
                {
                    "embedding_size": dimension,
                    "ndcg": float(row["ndcg"]),
                    "recall": float(row["recall"]),
                    "hr": float(row["hr"]),
                    "precision": float(row["precision"]),
                }
            )
            atomic_json(
                OUTPUT / dataset / "candidates.json", candidates
            )
        best = max(
            candidates,
            key=lambda row: (float(row["ndcg"]), -int(row["embedding_size"])),
        )
        atomic_json(OUTPUT / dataset / "best.json", best)
        run(
            [
                str(PYTHON),
                "-u",
                "scripts/run_paper_heteroembed.py",
                "--datasets",
                dataset,
                "--split",
                "outer",
                "--seeds",
                *[str(seed) for seed in args.final_seeds],
                "--epochs",
                str(args.final_epochs),
                "--embedding-size",
                str(best["embedding_size"]),
                "--gpu",
                args.gpu,
                "--results-root",
                str(FINAL_ROOT),
            ]
        )
        atomic_json(
            OUTPUT / dataset / "complete.json",
            {"selected": best, "final_complete": True},
        )


if __name__ == "__main__":
    main()
