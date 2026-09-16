# ItemKNN hybrids in CAFE

`baselines/models/itemknn_hybrids.py` implements the three deterministic ItemKNN
variants described by Ferrari Dacrema et al. (RecSys 2019):

- `ItemKNN-CF`: cosine similarity over item-user interaction vectors.
- `ItemKNN-CBF`: cosine similarity over item-content vectors.
- `ItemKNN-CFCBF`: cosine similarity over the concatenated vector
  `[interaction, content_weight * content]`.

The implementation keeps both the similarity matrix and recommendation
scoring sparse. Seen training items are masked before full-ranking top-10
evaluation.

## CAFE adaptation

The source paper evaluates content-based KNN on CiteULike title and abstract
features. For the four CAFE Amazon datasets, the content matrix instead uses:

- words from reviews that belong to the current fit split;
- static product brands;
- static product categories.

Behavioral relations (`also_bought`, `also_viewed`, and `bought_together`) are
excluded from the content matrix. For inner validation, review features are
rebuilt from the inner training interactions, so validation review text never
enters the fitted representation.

## Configuration

Use the launcher and tuner in [Baseline protocol and commands](README.md).
Supported similarity options are `--neighbors`, `--shrink`,
`--normalize/--no-normalize`, `--interaction-weighting`, `--content-weighting`
and `--content-weight`. Weighting accepts `none`, `tfidf`, or `bm25`.
Use `--split inner` for validation runs.
