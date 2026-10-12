# Baseline model implementations

This directory contains the baseline models. Launchers, preprocessing,
tuning and evaluation scripts are under `CAFE/scripts/`.

| Baseline | Implementation |
|---|---|
| BPR | `bpr.py` |
| BPR-HFT | `bpr_hft_deepconn.py` |
| DeepCoNN | `bpr_hft_deepconn.py` |
| CKE | `cke.py` |
| RippleNet | `ripplenet.py` |
| KGAT | `kgat.py` |
| HeteroEmbed | `heteroembed.py` |
| PGPR | `pgpr/` |
| ItemKNN-CF | `itemknn_hybrids.py` |
| ItemKNN-CBF | `itemknn_hybrids.py` |
| ItemKNN-CFCBF | `itemknn_hybrids.py` |

The supporting RecBole 1.2.1 engine is stored separately in
`baselines/recbole_core/`.
