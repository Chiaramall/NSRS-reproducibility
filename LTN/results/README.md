# MindReader control results

These are the original per-seed metrics consumed by
`mindreader_large_embeddings.py` when validation selects the control:

- `results_mf_genres_extended_30_runs.json`: MF and LTN-genres.
- `results_ltn_extended_grid_30_runs.json`: LTN.

The corresponding configurations are in `config/mindreader_controls.json`.
The files are preserved from the working repository; they were not regenerated
during repository cleanup. They contain 30 seeds for each model and five
training proportions. Derived summaries are not stored separately.

These files do not constitute the complete results of both datasets.
New experiment outputs are ignored by Git.
