#!/usr/bin/env python3
"""Create deterministic inner train/validation splits for baseline tuning.

The public CAFE/PGPR test split is never read to choose hyperparameters.  The
validation set is carved only from the public training interactions, per user,
while retaining at least one training interaction for every user and item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATOMIC_ROOT = ROOT / "baselines" / "data"
DATASET_TOKENS = {
    "beauty": "beauty_paper",
    "cell": "cell_paper",
    "clothing": "clothing_paper",
    "cd": "cds_paper",
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def stable_key(seed: int, uid: int, pid: int) -> bytes:
    return hashlib.sha256(f"{seed}:{uid}:{pid}".encode()).digest()


def read_rows(path: Path) -> tuple[str, list[tuple[int, int]]]:
    with path.open(encoding="utf-8") as handle:
        header = next(handle)
        rows = []
        for line in handle:
            uid, pid = line.rstrip("\n").split("\t")
            rows.append((int(uid), int(pid)))
    return header, rows


def render_rows(header: str, rows: list[tuple[int, int]]) -> str:
    body = "".join(f"{uid}\t{pid}\n" for uid, pid in rows)
    return header + body


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_filtered_link(
    source: Path, destination: Path, allowed_items: set[str]
) -> int:
    with source.open(encoding="utf-8") as handle:
        header = next(handle)
        selected: list[str] = []
        linked_items: set[str] = set()
        for line in handle:
            if not line.strip():
                continue
            item = line.split("\t", 1)[0]
            if item in allowed_items:
                selected.append(line)
                linked_items.add(item)
    missing = allowed_items - linked_items
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise RuntimeError(f"Training items missing from KG link: {examples}")
    atomic_text(destination, header + "".join(selected))
    return len(selected)


def create_split(dataset: str, fraction: float, seed: int) -> dict[str, object]:
    source_token = DATASET_TOKENS[dataset]
    tuning_token = f"{source_token}_tune"
    source_dir = ATOMIC_ROOT / source_token
    tuning_dir = ATOMIC_ROOT / tuning_token
    source_train = source_dir / f"{source_token}.train.inter"
    source_test = source_dir / f"{source_token}.test.inter"
    source_metadata = source_dir / "metadata.json"
    for path in (source_train, source_test, source_metadata):
        if not path.exists():
            raise FileNotFoundError(path)

    header, rows = read_rows(source_train)
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        by_user[row[0]].append(row)

    validation: set[tuple[int, int]] = set()
    for uid, user_rows in by_user.items():
        if len(user_rows) < 2:
            continue
        count = min(len(user_rows) - 1, max(1, round(len(user_rows) * fraction)))
        ordered = sorted(user_rows, key=lambda row: stable_key(seed, uid, row[1]))
        validation.update(ordered[:count])

    inner_train = [row for row in rows if row not in validation]
    item_counts = Counter(pid for _uid, pid in inner_train)
    missing_items = {
        pid for _uid, pid in rows if item_counts[pid] == 0
    }
    if missing_items:
        for row in sorted(validation):
            if row[1] in missing_items:
                validation.remove(row)
                inner_train.append(row)
                missing_items.remove(row[1])
                if not missing_items:
                    break
    inner_valid = [row for row in rows if row in validation]
    inner_train.sort()
    inner_valid.sort()

    if set(inner_train) & set(inner_valid):
        raise RuntimeError("Inner train/validation overlap")
    if len(inner_train) + len(inner_valid) != len(rows):
        raise RuntimeError("Inner split does not reconstruct outer training set")
    if {uid for uid, _pid in rows} != {uid for uid, _pid in inner_train}:
        raise RuntimeError("An inner-training user was lost")
    if {pid for _uid, pid in rows} != {pid for _uid, pid in inner_train}:
        raise RuntimeError("An inner-training item was lost")

    tuning_dir.mkdir(parents=True, exist_ok=True)
    train_path = tuning_dir / f"{tuning_token}.train.inter"
    valid_path = tuning_dir / f"{tuning_token}.valid.inter"
    test_path = tuning_dir / f"{tuning_token}.test.inter"
    atomic_text(train_path, render_rows(header, inner_train))
    atomic_text(valid_path, render_rows(header, inner_valid))
    # Some common-protocol runners expose only a test evaluator. This file is
    # deliberately an alias of inner validation, never the public outer test.
    atomic_text(test_path, render_rows(header, inner_valid))

    # The outer link file also contains products that occur only in the public
    # outer test. RecBole requires every linked item to exist in the current
    # interaction split, so retaining those rows both breaks KG models and
    # unnecessarily exposes outer-test catalogue membership during tuning.
    link_path = tuning_dir / f"{tuning_token}.link"
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    linked_items = write_filtered_link(
        source_dir / f"{source_token}.link",
        link_path,
        {str(pid) for _uid, pid in rows},
    )

    source_kg = source_dir / f"{source_token}.kg"
    kg_path = tuning_dir / f"{tuning_token}.kg"
    if kg_path.is_symlink() or kg_path.exists():
        kg_path.unlink()
    kg_path.symlink_to(os.path.relpath(source_kg, tuning_dir))

    original = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata = {
        "dataset": dataset,
        "dataset_token": tuning_token,
        "purpose": "inner validation for hyperparameter selection",
        "outer_training_token": source_token,
        "split_seed": seed,
        "validation_fraction_requested": fraction,
        "train_rows": len(inner_train),
        "valid_rows": len(inner_valid),
        "test_rows": len(inner_valid),
        "test_alias_of_validation": True,
        "outer_test_never_used_for_selection": True,
        "items": len({pid for _uid, pid in rows}),
        "linked_items": linked_items,
        "inner_link_sha256": sha256(link_path),
        "inner_train_sha256": sha256(train_path),
        "inner_valid_sha256": sha256(valid_path),
        "outer_train_sha256": sha256(source_train),
        "outer_test_sha256_recorded_only": sha256(source_test),
    }
    atomic_text(
        tuning_dir / "metadata.json",
        json.dumps(metadata, indent=2) + "\n",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_TOKENS),
        default=list(DATASET_TOKENS),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    summaries = [
        create_split(dataset, args.validation_fraction, args.seed)
        for dataset in args.datasets
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
