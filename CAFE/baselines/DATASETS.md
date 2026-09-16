# CAFE paper datasets

The paper evaluates four Amazon domains using the public PGPR 70/30 split:

| CLI key | Paper domain | Train | Test | CAFE raw directory | RecBole token |
|---|---|---:|---:|---|---|
| `beauty` | Beauty | 149,844 | 48,658 | `data/Beauty` | `beauty_paper` |
| `cell` | Cell Phones | 150,048 | 44,391 | `data/Cellphones_Accessories` | `cell_paper` |
| `clothing` | Clothing | 214,696 | 63,981 | `data/Clothing` | `clothing_paper` |
| `cd` | CDs & Vinyl | 804,090 | 293,501 | `data/CDs_Vinyl` | `cds_paper` |

Download the public PGPR ZIP archive from the dataset link in the
[upstream PGPR README](https://github.com/orcax/PGPR). Save it at
`baselines/source/pgpr_amazon_datasets.download`.

## Source setup

Extract the archive and prepare the data:

```bash
mkdir -p baselines/source baselines/models/pgpr/data
python -m zipfile -e baselines/source/pgpr_amazon_datasets.download baselines/source
for domain in Amazon_Beauty Amazon_Cellphones Amazon_Clothing Amazon_CDs; do
  ln -s ../../../source/PGPR-datasets/$domain baselines/models/pgpr/data/$domain
done
python scripts/prepare_paper_baselines.py
```

The links expose the same raw data to the official PGPR code and the Beauty
converter; review-aware models and ItemKNN read the extracted source directly.

## Validated preprocessing

Run the non-Beauty conversion with:

```bash
python scripts/prepare_all_amazon_datasets.py \
  --datasets cell clothing cd --threads 4
```

The converter preserves the public train/test split, uses training reviews only,
and generates CAFE graph files and RecBole atomic files.

## CAFE initialization for the additional domains

The archive contains no pretrained CAFE initialization for the three additional
domains. After conversion, train PGPR with seed 123 and install its completed
100-dimensional, epoch-30 TransE checkpoint:

```bash
python scripts/run_paper_pgpr.py --dataset cell --seeds 123 --gpu 0
python scripts/install_cafe_kg_embedding.py --dataset cell
python preprocess.py --dataset cell
```

Repeat with `clothing` and `cd`. The PGPR command also trains and evaluates its
policy. These are regenerated initializations; the original paper checkpoints are
unavailable.

Continue with the training and execution commands in the main README,
substituting the dataset key and a distinct run name.
