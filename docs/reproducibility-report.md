# Part IV — EditKG Reproducibility and Extended Evaluation

## 28. Study Overview

This study investigates the strict reproducibility of EditKG and of the recommendation baselines reported in the paper. The primary comparison uses the hyperparameters from each baseline’s original paper or official implementation. A separate common-configuration experiment was also completed, but is treated only as an exploratory extension and not as evidence for strict paper reproduction.

The experiments use the three datasets considered by the original study:

- Last-FM;
- Amazon-Book;
- Yelp2018.

The study comprises four primary stages:

1. reconstruction of the datasets and knowledge graphs;
2. reproduction of EditKG with the published configuration;
3. evaluation using model-specific original-paper or official hyperparameters;
4. repeated-seed evaluation of the definitive configurations for mean and standard deviation.

Common-configuration evaluation and local hyperparameter tuning are secondary extensions. They are not required for strict reproduction and must not be confused with the principal paper comparison.

EditKG is the target model. The twelve additional baselines integrated locally are BPR, GC-MC, LightGCN, SGL, CKE, KTUP, KGCN, KGAT, KGIN, KGCL, KGRec, and KRDN.

## 29. Dataset and Knowledge-Graph Reconstruction

| Dataset | Users | Items | Interactions | Density | KG entities | Relations | KG triplets |
|---|---:|---:|---:|---:|---:|---:|---:|
| Last-FM | 23,566 | 48,123 | 3,034,796 | 0.27% | 58,266 | 9 | 464,567 |
| Amazon-Book | 70,679 | 24,915 | 846,434 | 0.048% | 29,714 | 39 | 686,516 |
| Yelp2018 | 45,919 | 45,538 | 1,183,610 | 0.057% | 47,472 | 42 | 869,603 |

The public EditKG repository provides the required Last-FM graph but not the exact filtered Amazon-Book and Yelp2018 graphs used in the paper. The missing graphs were reconstructed from the public KGAT resources.

The reconstruction rules were:

- Last-FM: preserve the released EditKG graph unchanged;
- Amazon-Book: retain triples when either endpoint is an item, excluding the `is_reviewed` relation;
- Yelp2018: retain triples whose head entity is an item.

The reconstructed Last-FM and Yelp2018 graphs exactly match the published triplet counts. Amazon-Book differs from the paper by one triplet. Entity identifiers were compacted while preserving item identifiers.

## 30. Experimental Methodology

Performance is measured using Recall@20 and NDCG@20. Interactions are divided into training, validation, and test subsets using an 80%/10%/10% split. Evaluation uses full ranking over all eligible items for every test user.

### 30.1 EditKG configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Batch size | 4,096 |
| Learning rate | 0.0005 |
| GNN layers/context hops | 2 |
| Node dropout | 0.3 |
| Negative samples | 600 |
| Margin | 0.7 |
| Maximum epochs | 300 |
| Test batch size | 2,048 |
| Optimizer | Adam |

### 30.2 Primary baseline protocol

The EditKG paper does not provide a complete model-by-model table containing every baseline hyperparameter. Therefore, strict-reproduction configurations were reconstructed from the original paper of each baseline, its official implementation, and official RecBole defaults when a parameter was not fully specified by the original publication.

**Primary original-paper/official configuration.** Each model uses the hyperparameters reported in its original paper or official implementation, with a common maximum of 300 epochs. Evaluation is performed every ten epochs where supported. These are the results used for the principal local-versus-paper comparison.

**Compact model-specific local tuning.** A secondary extension is prepared for BPR, LightGCN, SGL, GC-MC, CKE, KTUP, and KGCN on all three datasets. Each model has six candidates: its original-paper/official configuration, the common-protocol configuration, and four targeted variants of model-specific parameters. Depending on the model, these variants cover regularization, propagation depth, graph or contrastive dropout, contrastive temperature and weight, GCN output size, KG alignment weights, margin, neighborhood size, or aggregation depth.

The resulting screen contains 126 runs: 6 candidates × 7 models × 3 datasets. Each candidate is trained for at most 25 epochs with early stopping. The winner is selected exclusively using validation Recall@20, with validation NDCG@20 used only as a tie-breaker. The complete winning configuration, rather than only its embedding dimension and learning rate, is then retrained for up to 150 epochs with an early-stopping patience of 10 evaluations and evaluated on the test set. Models still improving at 150 epochs can subsequently be extended to 300 epochs. This tuned evaluation is a local optimization extension and remains separate from strict original-paper reproduction.

**Exploratory common configuration.** A preliminary experiment used common hyperparameters across models to measure sensitivity to implementation and optimization choices. Because this setting does not reproduce the original baseline configurations, its results are reported only as an extended analysis.

All local values currently reported correspond to one run per configuration. They are not means across random seeds and no standard deviation is currently available. A statistically complete final evaluation requires at least three independent seeds for each definitive configuration.

The evaluation batch size for RecBole models was increased from 2,048 to 8,192. This changes computational throughput but not full-ranking candidate construction, Recall@20, NDCG@20, checkpoint selection, or final test evaluation.

### 30.3 Execution reliability

The complete experiment pipeline is sequential and resumable. Completed jobs are recorded and skipped after interruption. A user-level `systemd` service automatically restarts the pipeline after a crash, logout, or server reboot. A file lock prevents duplicate EditKG pipelines from using the GPU concurrently.

## 31. EditKG Reproduction

| Dataset | Metric | Local | Paper | Absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|
| Last-FM | Recall@20 | 0.1243 | 0.1212 | +0.0031 | +2.53% |
| Last-FM | NDCG@20 | 0.1121 | 0.1095 | +0.0026 | +2.39% |
| Amazon-Book | Recall@20 | 0.1777 | 0.1770 | +0.0007 | +0.37% |
| Amazon-Book | NDCG@20 | 0.1016 | 0.1003 | +0.0013 | +1.33% |
| Yelp2018 | Recall@20 | 0.0816 | 0.0854 | −0.0038 | −4.47% |
| Yelp2018 | NDCG@20 | 0.0537 | 0.0564 | −0.0027 | −4.71% |

EditKG is reproduced closely on all three datasets. Last-FM and Amazon-Book slightly exceed the published scores, while Yelp2018 remains approximately 4–5% below them. All relative differences remain below 5%, supporting the reproducibility of the target model.

## 32. Exploratory Extension: Baselines under a Common Configuration

These preliminary results evaluate the integrated models under a reconstructed common protocol. They are not a strict reproduction of the baseline results in the EditKG paper and are retained only as an exploratory sensitivity analysis. The principal reproduction results are reported in Section 33.

An asterisk denotes an approximate result from an official loop that periodically evaluates the held-out set rather than using the same separated validation/test procedure as the unified implementations.

### 32.1 Last-FM

| Model | Recall@20 local/paper | NDCG@20 local/paper |
|---|---:|---:|
| BPR | 0.0893 / 0.0690 | 0.0772 / 0.0585 |
| GC-MC | 0.1647 / 0.0709 | 0.1281 / 0.0631 |
| LightGCN | 0.0856 / 0.0738 | 0.0758 / 0.0647 |
| SGL | 0.0987 / 0.0879 | 0.0882 / 0.0775 |
| CKE | 0.0696 / 0.0845 | 0.0560 / 0.0718 |
| KTUP | 0.2638 / 0.0865 | 0.2275 / 0.0671 |
| KGCN | 0.2029 / 0.0879 | 0.1465 / 0.0694 |
| KGIN* | 0.1091 / 0.0978 | 0.0949 / 0.0848 |
| KGCL* | 0.1002 / 0.0905 | 0.0885 / 0.0769 |
| KGRec* | 0.1098 / 0.0943 | 0.0956 / 0.0810 |
| EditKG | 0.1243 / 0.1212 | 0.1121 / 0.1095 |

GC-MC, KTUP, and KGCN substantially exceed the paper values, while CKE remains below them. The magnitude of these deviations indicates that implementation and evaluation differences remain important under the common configuration.

### 32.2 Amazon-Book

| Model | Recall@20 local/paper | NDCG@20 local/paper |
|---|---:|---:|
| BPR | 0.1365 / 0.1244 | 0.0731 / 0.0658 |
| LightGCN | 0.1431 / 0.1398 | 0.0751 / 0.0736 |
| SGL | 0.1720 / 0.1445 | 0.0941 / 0.0766 |
| CKE | 0.1200 / 0.1375 | 0.0559 / 0.0685 |
| KTUP | 0.1007 / 0.1401 | 0.0464 / 0.0692 |
| KGCN | 0.0757 / 0.1111 | 0.0319 / 0.0569 |
| KGIN* | 0.1726 / 0.1687 | 0.0941 / 0.0915 |
| KGCL* | 0.1777 / 0.1496 | 0.0960 / 0.0793 |
| KGRec* | 0.1635 / 0.1443 | 0.0876 / 0.0782 |
| EditKG | 0.1777 / 0.1770 | 0.1016 / 0.1003 |

BPR, LightGCN, SGL, KGIN, KGCL, and KGRec meet or exceed their paper values. CKE, KTUP, and KGCN remain below them. EditKG obtains the strongest consolidated NDCG@20 and is effectively tied with KGCL on Recall@20 after rounding.

GC-MC is excluded from this table because its interrupted common-configuration run did not produce a consolidated final test result.

### 32.3 Yelp2018

| Model | Recall@20 local/paper | NDCG@20 local/paper |
|---|---:|---:|
| BPR | 0.0571 / 0.0568 | 0.0300 / 0.0462 |
| GC-MC | 0.0516 / 0.0688 | 0.0266 / 0.0453 |
| LightGCN | 0.0610 / 0.0632 | 0.0324 / 0.0519 |
| SGL | 0.0760 / 0.0788 | 0.0405 / 0.0518 |
| CKE | 0.0603 / 0.0653 | 0.0315 / 0.0423 |
| KTUP | 0.0490 / 0.0672 | 0.0251 / 0.0411 |
| KGCN | 0.0430 / 0.0532 | 0.0220 / 0.0338 |
| KGIN* | 0.0706 / 0.0698 | 0.0456 / 0.0451 |
| KGCL* | 0.0822 / 0.0748 | 0.0538 / 0.0491 |
| KGRec* | 0.0678 / 0.0745 | 0.0440 / 0.0478 |
| EditKG | 0.0816 / 0.0854 | 0.0537 / 0.0564 |

Yelp2018 is the most difficult dataset to reproduce across baselines. KGIN is close to the paper on both metrics. Most other models remain below their published values. KGCL marginally exceeds the local EditKG result, subject to the official-loop evaluation caveat.

## 33. Primary Reproduction: Baselines with Original-Paper or Official Hyperparameters

This stage uses hyperparameters reconstructed from each baseline’s original paper, official implementation, or official framework defaults when the publication does not fully specify a parameter. This is the principal baseline comparison for the strict reproduction study. Only completed results are included, and all current local values are single-run results rather than mean ± standard deviation.

### 33.1 Last-FM

| Model | Recall@20 local/paper | NDCG@20 local/paper |
|---|---:|---:|
| BPR | 0.0281 / 0.0690 | 0.0181 / 0.0585 |
| GC-MC | 0.0517 / 0.0709 | 0.0340 / 0.0631 |
| LightGCN | 0.0646 / 0.0738 | 0.0428 / 0.0647 |
| SGL | 0.0673 / 0.0879 | 0.0443 / 0.0775 |
| CKE | 0.2088 / 0.0845 | 0.1633 / 0.0718 |
| KTUP | 0.1798 / 0.0865 | 0.1398 / 0.0671 |
| KGCN | 0.1633 / 0.0879 | 0.1146 / 0.0694 |
| KGIN* | 0.0964 / 0.0978 | 0.0842 / 0.0848 |
| KGCL* | 0.0854 / 0.0905 | 0.0751 / 0.0769 |
| KGRec* | 0.0923 / 0.0943 | 0.0797 / 0.0810 |
| KRDN* | 0.1008 / 0.1023 | 0.0933 / 0.0946 |

KGIN, KGCL, KGRec, and KRDN closely reproduce their published Last-FM values. Their absolute Recall@20 differences are 0.0014, 0.0051, 0.0020, and 0.0015, respectively.

BPR, GC-MC, LightGCN, and SGL remain below the paper. CKE, KTUP, and KGCN substantially exceed the published scores. Model-specific hyperparameters therefore improve methodological fidelity but do not independently guarantee numerical reproduction.

KGAT did not yet produce a final recommendation result and is excluded from the numerical table. Its two compatibility defects were subsequently corrected: TensorFlow 1.15 now uses a CPU fallback on the Ampere server, and the reconstructed-KG relation table is sized from the maximum emitted relation identifier. A complete recovery run remains queued.

### 33.2 Amazon-Book

| Model | Recall@20 local/paper | NDCG@20 local/paper |
|---|---:|---:|
| BPR | 0.0643 / 0.1244 | 0.0269 / 0.0658 |
| GC-MC | 0.0468 / 0.1283 | 0.0199 / 0.0670 |
| LightGCN | 0.1197 / 0.1398 | 0.0538 / 0.0736 |
| SGL | 0.1429 / 0.1445 | 0.0652 / 0.0766 |
| CKE | 0.1062 / 0.1375 | 0.0481 / 0.0685 |
| KTUP | 0.0880 / 0.1401 | 0.0400 / 0.0692 |
| KGCN | 0.0702 / 0.1111 | 0.0297 / 0.0569 |
| KGIN* | 0.1654 / 0.1687 | 0.0890 / 0.0915 |
| KGCL* | 0.1627 / 0.1496 | 0.0862 / 0.0793 |
| KGRec* | 0.1443 / 0.1443 | 0.0750 / 0.0782 |

SGL provides the closest completed Recall@20 reproduction on Amazon-Book, differing from the paper by −0.0016. KGIN is also close to the paper, with differences of −0.0033 Recall@20 and −0.0025 NDCG@20. KGCL exceeds its published result by +0.0131 Recall@20 and +0.0069 NDCG@20. BPR, GC-MC, CKE, KTUP, and KGCN show larger negative deviations.

KGIN, KGCL, and KGRec are complete. KGRec reproduces the published Recall@20 after rounding and differs by −0.0032 on NDCG@20. KRDN is currently running. KGAT has passed a full training epoch after the compatibility corrections and its complete CPU recovery run remains queued.

### 33.3 Yelp2018

Original-paper/official Yelp2018 runs have not produced completed final metrics. They have been postponed in favor of the compact model-specific tuning stage, which still includes Yelp2018.

## 34. Reproducibility Assessment

The study produces three distinct conclusions.

First, EditKG itself is reproducible: its local metrics remain within 5% of the values reported in the paper on all datasets.

Second, baseline reproducibility is substantially less uniform. Some official implementations closely match the paper, particularly KGIN, KGCL, KGRec, and KRDN on Last-FM. Other models show large positive or negative deviations even when their published hyperparameters are used.

Third, hyperparameters are not the only source of discrepancy. Exact reproduction also depends on:

- dataset filtering and identifier mapping;
- train, validation, and test construction;
- negative sampling;
- candidate generation;
- validation and checkpoint selection;
- test-set isolation;
- implementation-specific evaluation code;
- random initialization and seed handling.

For these reasons, the original-paper/official results in Section 33 provide the relevant baseline reproduction evidence. The common-configuration results in Section 32 must be interpreted only as an exploratory re-evaluation. The EditKG executions provide the strongest strict-reproduction evidence.

The current analysis remains preliminary because each local value is obtained from a single run. Mean and standard deviation must not be inferred from these tables. The definitive statistical comparison will require repeated runs with at least three independent random seeds.

## 35. Current Experimental Coverage

The common-configuration pipeline completed 30 of 36 baseline dataset-model jobs and produced 29 consolidated metric pairs. KGAT and KRDN remain without consolidated results on the three common datasets, and the common GC-MC Amazon-Book run lacks a consolidated final test result.

The original-paper/official pipeline has completed:

- eleven Last-FM models, with KGAT excluded;
- ten Amazon-Book models: BPR, GC-MC, LightGCN, SGL, CKE, KTUP, KGCN, KGIN, KGCL, and KGRec.

KRDN Amazon-Book was interrupted before completion to prioritize tuning. The broad model-specific tuning is now running on Last-FM, Amazon-Book, and Yelp2018. The original-paper/official Yelp2018 stage was stopped after BPR completed; its GCMC run was interrupted.

A broad but fast model-specific tuning extension consisting of 126 validation-screening runs and up to 21 selected final runs is now running. It includes all three datasets and retains the common-protocol configuration as one candidate for every model. This extension is reported separately from the strict reproduction evidence. After screening, the selected configurations will be retrained and can subsequently be evaluated with repeated seeds. All completed jobs are recorded and skipped automatically after interruption or server reboot.

## 36. Main Findings

The current evidence supports the following conclusions:

- EditKG is successfully reproduced on all three datasets.
- EditKG differs from the paper by less than 5% on Recall@20 and NDCG@20.
- Dataset and KG reconstruction closely matches the published statistics.
- Baseline performance is considerably more sensitive than EditKG to implementation and evaluation details.
- The common-configuration experiment is exploratory and must not be presented as strict paper reproduction.
- Under original-paper or official hyperparameters, KGIN, KGCL, KGRec, and KRDN closely reproduce Last-FM.
- SGL currently provides the closest completed original-paper/official Amazon-Book Recall@20 result.
- Original-paper or official hyperparameters improve experimental fidelity but do not eliminate discrepancies caused by data and evaluation choices.
- Current values are single-run measurements; mean and standard deviation remain to be computed using repeated seeds.
- KGAT compatibility has been corrected and verified through one complete Amazon-Book training epoch; full CPU recovery metrics remain queued.
- The resumable `systemd` pipeline and larger evaluation batch reduce downtime and evaluation cost without altering the ranking protocol or reported metrics.

The final conclusions will be updated after the remaining Amazon-Book and Yelp2018 original-paper/official runs, KGAT recovery, and repeated-seed evaluations are complete.
