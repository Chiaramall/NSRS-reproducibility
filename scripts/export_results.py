#!/usr/bin/env python3
"""Export compact validation metrics from experiment logs."""

import argparse
import csv
import json
import re
from pathlib import Path


NAME = re.compile(r"tune_(last-fm|amazon-book|yelp2018)_([a-z0-9]+)_([a-z0-9]+)\.log")


def read_result(path):
    # Logs can be several gigabytes; inspect only a bounded tail.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 262144))
        lines = stream.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        if "VALIDATION_RESULT=" in line:
            return json.loads(line.split("VALIDATION_RESULT=", 1)[1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="paper_hparams_tuning directory")
    parser.add_argument("output", type=Path, help="output CSV")
    args = parser.parse_args()
    rows = []
    for path in sorted((args.source / "tuning").glob("tune_*.log")):
        match = NAME.fullmatch(path.name)
        result = read_result(path)
        if not match or not result:
            continue
        dataset, model, configuration = match.groups()
        rows.append({
            "dataset": dataset,
            "model": model,
            "configuration": configuration,
            "validation_recall_at_20": result.get("recall@20", result.get("valid_recall")),
            "validation_ndcg_at_20": result.get("ndcg@20", result.get("valid_ndcg")),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else (
            "dataset", "model", "configuration", "validation_recall_at_20", "validation_ndcg_at_20"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
