"""Reconstruct the dataset-specific primary-KG filtering used by EditKG.

The public EditKG repository only ships LastFM.  Inspection of the complete
KGAT benchmark dumps and the counts in EditKG Table 1 yields these rules:

* LastFM: retain the published KG unchanged.
* Yelp2018: retain outgoing item-attribute triples (exactly 869,603 triples).
* Amazon-Book: retain item-incident triples, excluding Freebase relation 12
  (``freebase.valuenotation.is_reviewed``).  This yields 686,517 triples,
  one more than the paper's 686,516, which is recorded in the manifest.

Outputs are written separately and never overwrite the complete KG dumps.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET_ROOT = ROOT / "data" / "editkg-filtered"
PAPER_TRIPLES = {"last-fm": 464_567, "amazon-book": 686_516, "yelp2018": 869_603}


def interaction_dimensions(source):
    max_user = max_item = -1
    for split in ("train.txt", "test.txt"):
        with (source / split).open() as stream:
            for line in stream:
                values = [int(value) for value in line.split()]
                if not values:
                    continue
                max_user = max(max_user, values[0])
                if len(values) > 1:
                    max_item = max(max_item, max(values[1:]))
    return max_user + 1, max_item + 1


def keep_triple(dataset, n_items, head, relation, tail):
    if dataset == "last-fm":
        return True
    if dataset == "yelp2018":
        return head < n_items
    if dataset == "amazon-book":
        return (head < n_items or tail < n_items) and relation != 12
    raise ValueError(dataset)


def prepare(dataset):
    source = ROOT / "data" / dataset
    target = TARGET_ROOT / dataset
    target.mkdir(parents=True, exist_ok=True)
    n_users, n_items = interaction_dimensions(source)
    retained = []
    entities, relations = set(), set()
    with (source / "kg_final.txt").open() as inp:
        for line in inp:
            head, relation, tail = map(int, line.split())
            if keep_triple(dataset, n_items, head, relation, tail):
                retained.append((head, relation, tail))
                entities.update((head, tail))
                relations.add(relation)

    # Item IDs are shared between the interaction graph and the KG, so they
    # must remain untouched. Compact only retained non-item entities.
    if dataset == "last-fm":
        entity_remap = {entity: entity for entity in entities}
    else:
        external = sorted(entity for entity in entities if entity >= n_items)
        entity_remap = {item: item for item in range(n_items)}
        entity_remap.update({
            old_id: n_items + offset for offset, old_id in enumerate(external)
        })

    with (target / "kg_final.txt").open("w") as out:
        for head, relation, tail in retained:
            out.write(f"{entity_remap[head]} {relation} {entity_remap[tail]}\n")

    with (target / "entity_id_map.tsv").open("w") as out:
        out.write("source_id\tfiltered_id\n")
        for old_id, new_id in sorted(entity_remap.items(), key=lambda pair: pair[1]):
            out.write(f"{old_id}\t{new_id}\n")

    source_entity_names = {}
    with (source / "entity_list.txt").open() as inp:
        header = inp.readline().strip()
        for line in inp:
            values = line.rsplit(maxsplit=1)
            if len(values) == 2:
                source_entity_names[int(values[1])] = values[0]
    with (target / "entity_list.txt").open("w") as out:
        out.write((header or "org_id remap_id") + "\n")
        for old_id, new_id in sorted(entity_remap.items(), key=lambda pair: pair[1]):
            out.write(f"{source_entity_names.get(old_id, old_id)} {new_id}\n")

    for filename in ("train.txt", "test.txt", "user_list.txt", "item_list.txt",
                     "relation_list.txt"):
        destination = target / filename
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to((source / filename).resolve())
    manifest = {
        "dataset": dataset,
        "source": str((source / "kg_final.txt").resolve()),
        "n_users": n_users,
        "n_items": n_items,
        "retained_triplets": len(retained),
        "paper_triplets": PAPER_TRIPLES[dataset],
        "triplet_delta": len(retained) - PAPER_TRIPLES[dataset],
        "entities_used": len(entities),
        "source_max_entity_id_plus_one": max(entities) + 1,
        "max_entity_id_plus_one": max(entity_remap.values()) + 1,
        "entity_ids_compacted": dataset != "last-fm",
        "item_ids_preserved": True,
        "relations_used": len(relations),
        "rule": {
            "last-fm": "unchanged public EditKG KG",
            "yelp2018": "head is an item",
            "amazon-book": "head or tail is an item; exclude relation 12 (is_reviewed)",
        }[dataset],
    }
    (target / "filter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("last-fm", "amazon-book", "yelp2018"):
        print(json.dumps(prepare(name), sort_keys=True))
