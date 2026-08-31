# CAFE paper datasets

The paper evaluates four Amazon domains using the public PGPR 70/30 split:

| CLI key | Paper domain | Train | Test | CAFE raw directory | RecBole token |
|---|---|---:|---:|---|---|
| `beauty` | Beauty | 149,844 | 48,658 | `data/Beauty` | `beauty_paper` |
| `cell` | Cell Phones | 150,048 | 44,391 | `data/Cellphones_Accessories` | `cell_paper` |
| `clothing` | Clothing | 214,696 | 63,981 | `data/Clothing` | `clothing_paper` |
| `cd` | CDs & Vinyl | 804,090 | 293,501 | `data/CDs_Vinyl` | `cds_paper` |

The official archive is stored at
`baselines/source/pgpr_amazon_datasets.download` and has SHA-256:

```text
2269d25338843b355115d0a131c415795e658ef5c0d6abd001198617e401a5b2
```

## Validated preprocessing

Run the non-Beauty conversion with:

```bash
.venv-baselines/bin/python scripts/prepare_all_amazon_datasets.py \
  --datasets cell clothing cd --threads 4
```

The converter:

1. preserves the official train/test pairs exactly;
2. verifies zero train/test overlap;
3. uses training reviews only;
4. applies the original PGPR filter (normalized per-review TF-IDF at least
   `0.1`, global training-word frequency at most `5000`);
5. writes explicit inverse relations and removes duplicate triples;
6. creates both CAFE-compatible compressed graph files and RecBole atomic
   files.

On Beauty, the generated graph was compared as a set against the official
CAFE graph: both contain 7,832,720 triples and the difference is empty.

Per-dataset counts, checksums and filtering statistics are stored in
`preprocessing_metadata.json` under each CAFE data directory and in
`metadata.json` under each RecBole data directory.

## Running baselines

The common-protocol models accept any of the four dataset keys:

```bash
.venv-baselines/bin/python scripts/run_paper_recbole.py \
  --dataset cell \
  --models BPR CKE RippleNet KGAT \
  --seeds 123 124 125 126 127
```

The official PGPR launcher uses the same keys:

```bash
.venv-baselines/bin/python scripts/run_paper_pgpr.py \
  --dataset cell \
  --seeds 123 124 125 126 127
```

## CAFE embedding boundary

The public PGPR archive contains raw entities, reviews and product relations,
but not the pretrained CAFE/HeteroEmbed checkpoints for Cell Phones,
Clothing or CDs & Vinyl. The raw CAFE graphs and labels are prepared, but
CAFE's original `preprocess.py` cannot finish until a compatible 100-dimensional
TransE checkpoint is trained or obtained. This is intentionally recorded in
the metadata rather than silently substituting an unrelated embedding.
