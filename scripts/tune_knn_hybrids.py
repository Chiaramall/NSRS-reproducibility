#!/usr/bin/env python3
"""Resumable compact inner-validation search for the three ItemKNN models."""

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
OUTPUT = ROOT / "results" / "knn_hybrid_selection"
FINAL_ROOT = ROOT / "results" / "paper_baselines"
MODELS = ("ItemKNN-CF", "ItemKNN-CBF", "ItemKNN-CFCBF")

CONFIGS = (
    (50, 0, True, "none", "none", 1.0),
    (100, 10, True, "tfidf", "tfidf", 0.5),
    (200, 100, True, "bm25", "bm25", 1.0),
    (400, 500, True, "none", "tfidf", 2.0),
    (800, 1000, True, "tfidf", "bm25", 10.0),
    (100, 0, False, "none", "none", 0.1),
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def command_for(
    dataset: str,
    model: str,
    split: str,
    config: tuple[int, int, bool, str, str, float],
    results_root: Path,
) -> list[str]:
    neighbors, shrink, normalize, interaction, content, weight = config
    return [
        str(PYTHON),
        "-u",
        "scripts/run_knn_hybrids.py",
        "--datasets",
        dataset,
        "--models",
        model,
        "--split",
        split,
        "--neighbors",
        str(neighbors),
        "--shrink",
        str(shrink),
        "--normalize" if normalize else "--no-normalize",
        "--interaction-weighting",
        interaction,
        "--content-weighting",
        content,
        "--content-weight",
        str(weight),
        "--results-root",
        str(results_root),
    ]


def read_result(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No result rows in {path}")
    return rows[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("beauty", "cell", "clothing", "cd"),
        default=["beauty", "cell", "clothing", "cd"],
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_json(
        OUTPUT / "protocol.json",
        {
            "selection_metric": "inner-validation NDCG@10",
            "configurations": CONFIGS,
            "models": args.models,
            "datasets": args.datasets,
            "deterministic_models": True,
            "outer_test_policy": "used once after selection",
        },
    )
    for dataset in args.datasets:
        for model in args.models:
            candidates: list[dict[str, object]] = []
            model_slug = model.lower()
            for index, config in enumerate(CONFIGS, start=1):
                trial_root = (
                    OUTPUT / "trials" / dataset / model_slug / f"trial_{index}"
                )
                result_path = (
                    trial_root
                    / dataset
                    / "knn_hybrids"
                    / "per_run.csv"
                )
                if not result_path.exists():
                    run(
                        command_for(
                            dataset,
                            model,
                            "inner",
                            config,
                            trial_root,
                        )
                    )
                row = read_result(result_path)
                candidate = {
                    "trial": index,
                    "neighbors": config[0],
                    "shrink": config[1],
                    "normalize": config[2],
                    "interaction_weighting": config[3],
                    "content_weighting": config[4],
                    "content_weight": config[5],
                    "ndcg": float(row["ndcg"]),
                    "recall": float(row["recall"]),
                    "hr": float(row["hr"]),
                    "precision": float(row["precision"]),
                }
                candidates.append(candidate)
                atomic_json(
                    OUTPUT / dataset / model_slug / "candidates.json",
                    candidates,
                )
            best = max(
                candidates,
                key=lambda row: (
                    float(row["ndcg"]),
                    -int(row["neighbors"]),
                ),
            )
            atomic_json(
                OUTPUT / dataset / model_slug / "best.json", best
            )
            selected = (
                int(best["neighbors"]),
                int(best["shrink"]),
                bool(best["normalize"]),
                str(best["interaction_weighting"]),
                str(best["content_weighting"]),
                float(best["content_weight"]),
            )
            run(
                command_for(
                    dataset,
                    model,
                    "outer",
                    selected,
                    FINAL_ROOT,
                )
            )
            atomic_json(
                OUTPUT / dataset / model_slug / "complete.json",
                {"selected": best, "final_complete": True},
            )


if __name__ == "__main__":
    main()
