"""Prepare EditKG datasets for the integrated baseline runners."""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path(os.environ.get("EDITKG_DATA_ROOT", ROOT / "data")).resolve()
DATASETS = ("last-fm", "amazon-book", "yelp2018")


def read_users(path):
    with path.open() as stream:
        for line in stream:
            values = line.split()
            if len(values) > 1:
                # Match EditKG read_cf(): duplicate user-item pairs in a row
                # count once, while retaining deterministic order.
                yield values[0], list(dict.fromkeys(values[1:]))


def write_inter(path, rows):
    with path.open("w") as stream:
        stream.write("user_id:token\titem_id:token\trating:float\n")
        for user, items in rows:
            for item in items:
                stream.write(f"{user}\t{item}\t1\n")


def prepare_recbole(dataset):
    source = SOURCE_ROOT / dataset
    target = ROOT / "baseline_data" / dataset
    target.mkdir(parents=True, exist_ok=True)
    train = list(read_users(source / "train.txt"))
    valid, test = [], []
    for user, items in read_users(source / "test.txt"):
        cut = max(1, len(items) // 2)
        valid.append((user, items[:cut]))
        test.append((user, items[cut:] or items[-1:]))
    write_inter(target / f"{dataset}.train.inter", train)
    write_inter(target / f"{dataset}.valid.inter", valid)
    write_inter(target / f"{dataset}.test.inter", test)
    with (target / f"{dataset}.kg").open("w") as out, (source / "kg_final.txt").open() as inp:
        out.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for line in inp:
            head, relation, tail = line.split()
            out.write(f"e{head}\tr{relation}\te{tail}\n")
    with (target / f"{dataset}.link").open("w") as stream:
        stream.write("item_id:token\tentity_id:token\n")
        all_rows = train + valid + test
        item_count = max(int(item) for _, items in all_rows for item in items) + 1
        for item in range(item_count):
            stream.write(f"{item}\te{item}\n")


def ensure_native_links(dataset):
    source = SOURCE_ROOT / dataset
    links = {
        ROOT / "baselines" / "external" / project / "data" / dataset: source
        for project in ("kgrec", "kgcl", "krdn")
    }
    for link, source in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
    kgcl_kg = ROOT / "baselines" / "external" / "kgcl" / "data" / dataset / "kg.txt"
    if not kgcl_kg.exists():
        kgcl_kg.symlink_to(source / "kg_final.txt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=DATASETS, default=["last-fm"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for dataset in args.datasets:
        prepare_recbole(dataset)
        ensure_native_links(dataset)
        print(f"{dataset}: baseline data prepared")
