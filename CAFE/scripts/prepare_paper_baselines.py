#!/usr/bin/env python3
"""Prepare the exact CAFE Beauty split for common baseline runners."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAFE_DATA = ROOT / "data" / "Beauty"
PGPR_DATA = ROOT / "baselines" / "models" / "pgpr" / "data" / "Amazon_Beauty"
OUTPUT_DIR = ROOT / "baselines" / "data" / "beauty_paper"
DATASET_TOKEN = "beauty_paper"
ENTITY_RE = re.compile(r"^(?P<type>.+)_(?P<local_id>\d+)$")


def load_cafe_pairs(path: Path) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            user = int(cells[0])
            pairs.update((user, int(item)) for item in cells[1:])
    return pairs


def load_pgpr_pairs(path: Path) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            user, item, *_ = line.rstrip("\n").split("\t")
            pairs.add((int(user), int(item)))
    return pairs


def pair_digest(pairs: set[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for user, item in sorted(pairs):
        digest.update(f"{user}\t{item}\n".encode())
    return digest.hexdigest()


def verify_split() -> dict[str, object]:
    cafe_train = load_cafe_pairs(CAFE_DATA / "train_labels.txt.gz")
    cafe_test = load_cafe_pairs(CAFE_DATA / "test_labels.txt.gz")
    pgpr_train = load_pgpr_pairs(PGPR_DATA / "train.txt.gz")
    pgpr_test = load_pgpr_pairs(PGPR_DATA / "test.txt.gz")
    if cafe_train != pgpr_train:
        raise RuntimeError("CAFE and PGPR Beauty train pairs differ")
    if cafe_test != pgpr_test:
        raise RuntimeError("CAFE and PGPR Beauty test pairs differ")
    overlap = cafe_train & cafe_test
    if overlap:
        raise RuntimeError(f"Train/test overlap contains {len(overlap)} pairs")
    return {
        "train_pairs": len(cafe_train),
        "test_pairs": len(cafe_test),
        "train_sha256": pair_digest(cafe_train),
        "test_sha256": pair_digest(cafe_test),
    }


def load_entities() -> dict[int, tuple[str, str, int]]:
    entities: dict[int, tuple[str, str, int]] = {}
    with gzip.open(
        CAFE_DATA / "kg_entities.txt.gz", "rt", encoding="utf-8"
    ) as handle:
        for line in handle:
            global_id_raw, entity_name, *_ = line.rstrip("\n").split("\t")
            match = ENTITY_RE.match(entity_name)
            if match is None:
                raise ValueError(f"Unrecognized entity token: {entity_name}")
            entities[int(global_id_raw)] = (
                entity_name,
                match.group("type"),
                int(match.group("local_id")),
            )
    return entities


def load_relations() -> dict[int, str]:
    relations: dict[int, str] = {}
    with gzip.open(
        CAFE_DATA / "kg_relations.txt.gz", "rt", encoding="utf-8"
    ) as handle:
        for line in handle:
            relation_id, relation_name = line.rstrip("\n").split("\t")
            relations[int(relation_id)] = relation_name
    return relations


def write_interactions(suffix: str, source: Path) -> int:
    target = OUTPUT_DIR / f"{DATASET_TOKEN}.{suffix}.inter"
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["user_id:token", "item_id:token"])
        with gzip.open(source, "rt", encoding="utf-8") as labels:
            for line in labels:
                cells = line.rstrip("\n").split("\t")
                for item in cells[1:]:
                    writer.writerow([cells[0], item])
                    count += 1
    return count


def write_empty_validation() -> None:
    target = OUTPUT_DIR / f"{DATASET_TOKEN}.valid.inter"
    target.write_text("user_id:token\titem_id:token\n", encoding="utf-8")


def write_link(entities: dict[int, tuple[str, str, int]]) -> int:
    target = OUTPUT_DIR / f"{DATASET_TOKEN}.link"
    products = sorted(
        (local_id, name)
        for name, entity_type, local_id in entities.values()
        if entity_type == "product"
    )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["item_id:token", "entity_id:token"])
        writer.writerows(products)
    return len(products)


def write_kg(
    entities: dict[int, tuple[str, str, int]],
    relations: dict[int, str],
) -> int:
    target = OUTPUT_DIR / f"{DATASET_TOKEN}.kg"
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["head_id:token", "relation_id:token", "tail_id:token"]
        )
        with gzip.open(
            CAFE_DATA / "kg_triples.txt.gz", "rt", encoding="utf-8"
        ) as triples:
            for line in triples:
                head_raw, relation_raw, tail_raw = line.rstrip("\n").split("\t")
                head = entities[int(head_raw)]
                tail = entities[int(tail_raw)]
                relation = relations[int(relation_raw)]
                if head[1] == "user" or tail[1] == "user":
                    continue
                if relation == "purchase":
                    continue
                writer.writerow([head[0], relation, tail[0]])
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that CAFE and PGPR use identical pair splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = verify_split()
    print(json.dumps({"split_verified": split}, indent=2))
    if args.verify_only:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entities = load_entities()
    relations = load_relations()
    train_count = write_interactions(
        "train", CAFE_DATA / "train_labels.txt.gz"
    )
    write_empty_validation()
    test_count = write_interactions(
        "test", CAFE_DATA / "test_labels.txt.gz"
    )
    item_count = write_link(entities)
    kg_count = write_kg(entities, relations)
    metadata = {
        "dataset": "Amazon Beauty",
        "dataset_token": DATASET_TOKEN,
        "split": split,
        "train_rows": train_count,
        "valid_rows": 0,
        "test_rows": test_count,
        "items": item_count,
        "kg_rows_without_users_or_purchase": kg_count,
        "kg_reverse_r": False,
        "review_source": str(PGPR_DATA.relative_to(ROOT)),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
