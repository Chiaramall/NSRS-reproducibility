# Data

## Filtered datasets

`editkg-filtered/{last-fm,amazon-book,yelp2018}` contains `train.txt`, `test.txt`, `kg_final.txt`, identifier lists, and `filter_manifest.json`.

The prepared artifacts can be used directly. To reconstruct them, first place the original benchmark files (`train.txt`, `test.txt`, `kg_final.txt`, `entity_list.txt`, `user_list.txt`, `item_list.txt`, and `relation_list.txt`) in `data/<dataset>/`, then run from the repository root:

```bash
python scripts/prepare_editkg_filtered.py
```

Raw benchmark dumps are not included in those source directories.

## Validation views

`editkg-validation` contains deterministic per-user holdouts from filtered training interactions (default ratio 10%, seed 2024), with at least one interaction retained for training per user:

| File | Meaning |
| --- | --- |
| `train.txt` | Reduced training interactions used for optimization and graph construction |
| `test.txt` | Validation interactions, named for compatibility with native loaders |
| `final_test.txt` | Original held-out test interactions |
| `validation_manifest.json` | Split settings and interaction counts |

Regenerate from the repository root:

```bash
python scripts/prepare_validation_splits.py
```
