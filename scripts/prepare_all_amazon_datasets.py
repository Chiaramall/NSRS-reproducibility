#!/usr/bin/env python3
"""Prepare the non-Beauty Amazon datasets used by the CAFE paper.

The source is the public PGPR 70/30 split. The conversion is streaming and
reproduces PGPR's review-word filtering:

* per-review normalized TF-IDF >= 0.1;
* global training-word frequency <= 5000;
* duplicate graph triples removed.

For each dataset the script writes both CAFE-compatible raw graph files and
RecBole atomic files. It never overwrites an existing completed dataset unless
``--force`` is given.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "baselines" / "source" / "PGPR-datasets"
ATOMIC_ROOT = ROOT / "baselines" / "data"
WORK_ROOT = ROOT / "baselines" / "work"
BEAUTY_RULES = ROOT / "data" / "Beauty" / "kg_rules.txt.gz"
SOURCE_ARCHIVE = ROOT / "baselines" / "source" / "pgpr_amazon_datasets.download"

RELATIONS = (
    "purchase",
    "mentions",
    "described_by",
    "produced_by",
    "belongs_to",
    "also_bought",
    "also_viewed",
    "bought_together",
    "rev_purchase",
    "rev_mentions",
    "rev_described_by",
    "rev_produced_by",
    "rev_belongs_to",
    "rev_also_bought",
    "rev_also_viewed",
    "rev_bought_together",
)
PRODUCT_RELATIONS = (
    ("brand_p_b.txt.gz", "brand", 3),
    ("category_p_c.txt.gz", "category", 4),
    ("also_bought_p_p.txt.gz", "related_product", 5),
    ("also_viewed_p_p.txt.gz", "related_product", 6),
    ("bought_together_p_p.txt.gz", "related_product", 7),
)
ENTITY_FILES = (
    ("users.txt.gz", "user"),
    ("product.txt.gz", "product"),
    ("vocab.txt.gz", "word"),
    ("brand.txt.gz", "brand"),
    ("category.txt.gz", "category"),
    ("related_product.txt.gz", "related_product"),
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source_name: str
    cafe_name: str
    atomic_token: str
    train_label_name: str
    test_label_name: str
    paper_users: int
    paper_items: int
    source_interactions: int
    paper_interactions: str
    paper_entities: int
    paper_triples: int


DATASETS = {
    "cell": DatasetSpec(
        "cell",
        "Amazon_Cellphones",
        "Cellphones_Accessories",
        "cell_paper",
        "Cellphones_Accessories_train_label.pkl",
        "Cellphones_Accessories_test_label.pkl",
        27_879,
        10_429,
        194_439,
        "194.32K",
        163_255,
        37_010_000,
    ),
    "clothing": DatasetSpec(
        "clothing",
        "Amazon_Clothing",
        "Clothing",
        "clothing_paper",
        "Clothing_train_label.pkl",
        "Clothing_test_label.pkl",
        39_387,
        23_033,
        278_677,
        "278.86K",
        425_534,
        36_370_000,
    ),
    "cd": DatasetSpec(
        "cd",
        "Amazon_CDs",
        "CDs_Vinyl",
        "cds_paper",
        "CDs_Vinyl_train_label.pkl",
        "CDs_Vinyl_test_label.pkl",
        75_258,
        64_443,
        1_097_591,
        "1.10M",
        581_105,
        387_430_000,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_line_count(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        return sum(1 for _ in handle)


def read_interactions(path: Path) -> tuple[dict[int, list[int]], str]:
    labels: dict[int, list[int]] = defaultdict(list)
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            user_raw, item_raw, *_ = line.rstrip("\n").split("\t")
            user, item = int(user_raw), int(item_raw)
            labels[user].append(item)
            digest.update(f"{user}\t{item}\n".encode())
    return dict(labels), digest.hexdigest()


def write_labels(path: Path, labels: dict[int, list[int]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for user in sorted(labels):
            items = labels[user]
            handle.write("\t".join([str(user), *(str(item) for item in items)]))
            handle.write("\n")
            count += len(items)
    return count


def write_inter_file(path: Path, source: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["user_id:token", "item_id:token"])
        with gzip.open(source, "rt", encoding="utf-8") as interactions:
            for line in interactions:
                user, item, *_ = line.rstrip("\n").split("\t")
                writer.writerow([user, item])
                count += 1
    return count


def entity_counts(source_dir: Path) -> dict[str, int]:
    return {
        entity_type: gzip_line_count(source_dir / filename)
        for filename, entity_type in ENTITY_FILES
    }


def entity_offsets(counts: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for _filename, entity_type in ENTITY_FILES:
        offsets[entity_type] = cursor
        cursor += counts[entity_type]
    return offsets


def entity_token(
    global_id: int, counts: dict[str, int], offsets: dict[str, int]
) -> str:
    for _filename, entity_type in reversed(ENTITY_FILES):
        if global_id >= offsets[entity_type]:
            local_id = global_id - offsets[entity_type]
            if local_id >= counts[entity_type]:
                raise ValueError(f"Global entity id out of range: {global_id}")
            return f"{entity_type}_{local_id}"
    raise ValueError(f"Global entity id out of range: {global_id}")


def write_entities(
    target: Path,
    source_dir: Path,
    counts: dict[str, int],
    offsets: dict[str, int],
) -> int:
    total = 0
    with gzip.open(target, "wt", encoding="utf-8") as output:
        for filename, entity_type in ENTITY_FILES:
            expected = counts[entity_type]
            seen = 0
            with gzip.open(
                source_dir / filename, "rt", encoding="utf-8"
            ) as source:
                for local_id, line in enumerate(source):
                    value = line.rstrip("\n")
                    output.write(
                        f"{offsets[entity_type] + local_id}\t"
                        f"{entity_type}_{local_id}\t{value}\n"
                    )
                    seen += 1
            if seen != expected:
                raise RuntimeError(f"Entity count changed for {entity_type}")
            total += seen
    return total


def write_relations(path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for relation_id, relation in enumerate(RELATIONS):
            handle.write(f"{relation_id}\t{relation}\n")


def compute_word_statistics(
    train_path: Path,
) -> tuple[int, Counter[int], Counter[int]]:
    documents = 0
    frequency: Counter[int] = Counter()
    document_frequency: Counter[int] = Counter()
    with gzip.open(train_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            words = [int(value) for value in cells[2].split()] if cells[2] else []
            frequency.update(words)
            document_frequency.update(set(words))
            documents += 1
    return documents, frequency, document_frequency


def retained_words(
    words: list[int],
    documents: int,
    frequency: Counter[int],
    document_frequency: Counter[int],
) -> list[int]:
    counts = Counter(words)
    weighted = {
        word: count
        * (math.log((1.0 + documents) / (1.0 + document_frequency[word])) + 1.0)
        for word, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return []
    return sorted(
        word
        for word, value in weighted.items()
        if value / norm >= 0.1 and frequency[word] <= 5000
    )


def add_triple(handle, head: int, relation: int, tail: int) -> None:
    handle.write(f"{head}\t{relation}\t{tail}\n")


def write_unsorted_triples(
    path: Path,
    source_dir: Path,
    counts: dict[str, int],
    offsets: dict[str, int],
) -> dict[str, int]:
    train_path = source_dir / "train.txt.gz"
    documents, frequency, document_frequency = compute_word_statistics(train_path)
    emitted = 0
    retained_reviews = 0
    retained_word_links = 0

    with path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        with gzip.open(train_path, "rt", encoding="utf-8") as interactions:
            for line in interactions:
                user_raw, product_raw, words_raw = line.rstrip("\n").split("\t")
                user, product = int(user_raw), int(product_raw)
                words = [int(value) for value in words_raw.split()] if words_raw else []
                kept = retained_words(
                    words, documents, frequency, document_frequency
                )
                if not kept:
                    continue
                retained_reviews += 1
                user_gid = offsets["user"] + user
                product_gid = offsets["product"] + product
                add_triple(output, user_gid, 0, product_gid)
                add_triple(output, product_gid, 8, user_gid)
                emitted += 2
                for word in kept:
                    word_gid = offsets["word"] + word
                    add_triple(output, user_gid, 1, word_gid)
                    add_triple(output, word_gid, 9, user_gid)
                    add_triple(output, product_gid, 2, word_gid)
                    add_triple(output, word_gid, 10, product_gid)
                    emitted += 4
                    retained_word_links += 1

        for filename, tail_type, relation in PRODUCT_RELATIONS:
            with gzip.open(
                source_dir / filename, "rt", encoding="utf-8"
            ) as relation_file:
                for product, line in enumerate(relation_file):
                    tails = line.strip().split()
                    product_gid = offsets["product"] + product
                    for tail_raw in tails:
                        tail_gid = offsets[tail_type] + int(tail_raw)
                        add_triple(output, product_gid, relation, tail_gid)
                        add_triple(output, tail_gid, relation + 8, product_gid)
                        emitted += 2

    return {
        "documents": documents,
        "retained_reviews": retained_reviews,
        "retained_word_links_before_dedup": retained_word_links,
        "emitted_triples_before_dedup": emitted,
    }


def sort_unique(source: Path, target: Path, threads: int) -> None:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(
        [
            "sort",
            "-u",
            "--parallel",
            str(threads),
            "-S",
            "25%",
            "-o",
            str(target),
            str(source),
        ],
        check=True,
        env=env,
    )


def gzip_copy(source: Path, target: Path) -> int:
    count = 0
    with source.open("rt", encoding="utf-8") as input_handle, gzip.open(
        target, "wt", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            output_handle.write(line)
            count += 1
    return count


def write_atomic_kg(
    source: Path,
    target: Path,
    counts: dict[str, int],
    offsets: dict[str, int],
) -> int:
    user_end = offsets["user"] + counts["user"]
    count = 0
    with source.open("rt", encoding="utf-8") as triples, target.open(
        "w", encoding="utf-8"
    ) as output:
        output.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for line in triples:
            head_raw, relation_raw, tail_raw = line.rstrip("\n").split("\t")
            head, relation, tail = int(head_raw), int(relation_raw), int(tail_raw)
            if head < user_end or tail < user_end:
                continue
            output.write(
                f"{entity_token(head, counts, offsets)}\t"
                f"{RELATIONS[relation]}\t"
                f"{entity_token(tail, counts, offsets)}\n"
            )
            count += 1
    return count


def write_atomic_link(path: Path, item_count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("item_id:token\tentity_id:token\n")
        for item in range(item_count):
            handle.write(f"{item}\tproduct_{item}\n")


def validate_expected(
    spec: DatasetSpec,
    counts: dict[str, int],
    train_count: int,
    test_count: int,
) -> None:
    if counts["user"] != spec.paper_users:
        raise RuntimeError(
            f"{spec.key}: users {counts['user']} != paper {spec.paper_users}"
        )
    if counts["product"] != spec.paper_items:
        raise RuntimeError(
            f"{spec.key}: items {counts['product']} != paper {spec.paper_items}"
        )
    if train_count + test_count != spec.source_interactions:
        raise RuntimeError(
            f"{spec.key}: interactions {train_count + test_count} "
            f"!= official source {spec.source_interactions}"
        )


def prepare_dataset(spec: DatasetSpec, force: bool, threads: int) -> dict[str, object]:
    source_dir = SOURCE_ROOT / spec.source_name
    cafe_dir = ROOT / "data" / spec.cafe_name
    atomic_dir = ATOMIC_ROOT / spec.atomic_token
    metadata_path = atomic_dir / "metadata.json"
    if metadata_path.exists() and not force:
        print(f"Skipping completed {spec.key}: {metadata_path}", flush=True)
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if not source_dir.is_dir():
        raise RuntimeError(f"Missing source directory: {source_dir}")

    cafe_dir.mkdir(parents=True, exist_ok=True)
    atomic_dir.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    counts = entity_counts(source_dir)
    offsets = entity_offsets(counts)
    train_labels, train_sha = read_interactions(source_dir / "train.txt.gz")
    test_labels, test_sha = read_interactions(source_dir / "test.txt.gz")
    overlap = sum(
        len(set(items) & set(test_labels.get(user, ())))
        for user, items in train_labels.items()
    )
    if overlap:
        raise RuntimeError(f"{spec.key}: train/test overlap contains {overlap} pairs")
    train_count = sum(map(len, train_labels.values()))
    test_count = sum(map(len, test_labels.values()))
    validate_expected(spec, counts, train_count, test_count)

    print(f"[{spec.key}] writing entities, labels and relations", flush=True)
    entity_count = write_entities(
        cafe_dir / "kg_entities.txt.gz", source_dir, counts, offsets
    )
    write_relations(cafe_dir / "kg_relations.txt.gz")
    shutil.copy2(BEAUTY_RULES, cafe_dir / "kg_rules.txt.gz")
    write_labels(cafe_dir / spec.train_label_name, train_labels)
    write_labels(cafe_dir / spec.test_label_name, test_labels)

    with tempfile.TemporaryDirectory(
        prefix=f"{spec.key}_", dir=WORK_ROOT
    ) as temp_raw:
        temp = Path(temp_raw)
        unsorted = temp / "triples.unsorted"
        sorted_unique = temp / "triples.sorted"
        print(f"[{spec.key}] pass 1/2: TF-IDF statistics and triples", flush=True)
        graph_stats = write_unsorted_triples(
            unsorted, source_dir, counts, offsets
        )
        print(f"[{spec.key}] sorting and removing duplicate triples", flush=True)
        sort_unique(unsorted, sorted_unique, threads)
        triple_count = gzip_copy(
            sorted_unique, cafe_dir / "kg_triples.txt.gz"
        )

        print(f"[{spec.key}] writing RecBole atomic files", flush=True)
        prefix = atomic_dir / spec.atomic_token
        atomic_train_count = write_inter_file(
            prefix.with_suffix(".train.inter"), source_dir / "train.txt.gz"
        )
        prefix.with_suffix(".valid.inter").write_text(
            "user_id:token\titem_id:token\n", encoding="utf-8"
        )
        atomic_test_count = write_inter_file(
            prefix.with_suffix(".test.inter"), source_dir / "test.txt.gz"
        )
        write_atomic_link(prefix.with_suffix(".link"), counts["product"])
        atomic_kg_count = write_atomic_kg(
            sorted_unique, prefix.with_suffix(".kg"), counts, offsets
        )

    metadata: dict[str, object] = {
        "dataset": spec.key,
        "source": str(source_dir.relative_to(ROOT)),
        "source_archive": str(SOURCE_ARCHIVE.relative_to(ROOT)),
        "source_archive_sha256": sha256(SOURCE_ARCHIVE),
        "split": "official PGPR 70/30",
        "train_rows": train_count,
        "test_rows": test_count,
        "train_ordered_sha256": train_sha,
        "test_ordered_sha256": test_sha,
        "train_test_overlap": overlap,
        "entities_by_type": counts,
        "entities_total": entity_count,
        "relations": len(RELATIONS),
        "cafe_triples_after_filter_and_dedup": triple_count,
        "recbole_item_kg_rows": atomic_kg_count,
        "recbole_kg_reverse_r": False,
        "review_filter": {
            "tfidf_min": 0.1,
            "global_word_frequency_max": 5000,
            **graph_stats,
        },
        "paper_table_1": {
            "users": spec.paper_users,
            "items": spec.paper_items,
            "interactions_reported": spec.paper_interactions,
            "entities": spec.paper_entities,
            "triples": spec.paper_triples,
        },
        "atomic_counts_verified": {
            "train": atomic_train_count == train_count,
            "test": atomic_test_count == test_count,
        },
        "cafe_embedding": "not included in the public PGPR archive; train TransE before running CAFE preprocess.py",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (cafe_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="The non-Beauty CAFE datasets to prepare.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    for dataset in args.datasets:
        prepare_dataset(DATASETS[dataset], args.force, args.threads)


if __name__ == "__main__":
    main()
