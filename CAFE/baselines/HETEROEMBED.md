# HeteroEmbed

HeteroEmbed is implemented in `baselines/models/heteroembed.py`.

## Run

```bash
python scripts/run_paper_heteroembed.py --datasets beauty --seeds 123 124 125 126 127
```

## Hyperparameter selection

```bash
python scripts/tune_paper_heteroembed.py --datasets beauty cell clothing cd
```

Use `--split inner` for validation runs and reserve the outer test for final
evaluation. See [Baseline commands](README.md) for the shared protocol.
