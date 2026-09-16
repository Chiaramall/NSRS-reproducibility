#!/usr/bin/env python3
"""Create deterministic, leakage-free validation views of EditKG datasets.

The official baseline loaders treat ``test.txt`` as their evaluation split and
also build the collaborative graph from ``train.txt``.  This utility creates a
separate data root where the original training interactions are split into
train/validation, while the original test set is retained as ``final_test.txt``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


DATASETS = ("last-fm", "amazon-book", "yelp2018")


def read_users(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        values = [int(value) for value in line.split()]
        if values:
            rows.append((values[0], values[1:]))
    return rows


def stable_order(dataset: str, user: int, item: int, seed: int):
    key = f"{seed}:{dataset}:{user}:{item}".encode()
    return hashlib.sha256(key).digest()


def write_users(path: Path, rows):
    with path.open("w") as stream:
        for user, items in rows:
            if items:
                stream.write(" ".join(map(str, (user, *items))) + "\n")


def create_split(source: Path, destination: Path, dataset: str, ratio: float, seed: int):
    destination.mkdir(parents=True, exist_ok=True)
    train_rows, valid_rows = [], []
    n_train = n_valid = 0
    for user, items in read_users(source / "train.txt"):
        ordered = sorted(set(items), key=lambda item: stable_order(dataset, user, item, seed))
        # Keep at least one interaction in training. Users with one interaction
        # cannot contribute a validation positive without becoming cold-start.
        count = min(len(ordered) - 1, max(1, round(len(ordered) * ratio))) if len(ordered) > 1 else 0
        valid = ordered[:count]
        train = ordered[count:]
        train_rows.append((user, train))
        if valid:
            valid_rows.append((user, valid))
        n_train += len(train)
        n_valid += len(valid)

    write_users(destination / "train.txt", train_rows)
    write_users(destination / "test.txt", valid_rows)
    # Keep immutable references to the true test and metadata. Symlinks avoid
    # duplicating the large filtered KG and are replaced atomically if needed.
    links = {
        "final_test.txt": source / "test.txt",
        "kg_final.txt": source / "kg_final.txt",
        "item_list.txt": source / "item_list.txt",
        "entity_list.txt": source / "entity_list.txt",
        "relation_list.txt": source / "relation_list.txt",
        "user_list.txt": source / "user_list.txt",
    }
    for name, target in links.items():
        link = destination / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
    manifest = {
        "dataset": dataset,
        "seed": seed,
        "validation_ratio": ratio,
        "strategy": "per-user deterministic holdout; at least one training item retained",
        "train_interactions": n_train,
        "validation_interactions": n_valid,
        "validation_users": len(valid_rows),
        "source": str(source.resolve()),
        "test_used_during_tuning": False,
    }
    (destination / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/editkg-filtered"))
    parser.add_argument("--output", type=Path, default=Path("data/editkg-validation"))
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be between 0 and 1")
    manifests = []
    for dataset in DATASETS:
        manifests.append(create_split(args.source / dataset, args.output / dataset,
                                      dataset, args.ratio, args.seed))
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
