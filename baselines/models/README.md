# Baseline model implementations

All model implementations evaluated in the CAFE study are centralized in this
directory. Files under `scripts/` are launchers, preprocessing tools, tuners or
evaluators; they do not contain a second implementation.

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
`baselines/recbole_core/`. Its model-resolution shims import BPR, CKE,
RippleNet and KGAT from this directory, so these are the files used at runtime.
