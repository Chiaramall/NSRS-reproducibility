# Methods text for the MovieLens extension

## Experimental objective

After reproducing the released MindReader experiment, we conducted a separate
benchmark to test whether the behavior of Logic Tensor Networks (LTNs) under
increasing sparsity generalizes to standard recommendation datasets. The
extension evaluates MovieLens 100K and MovieLens 1M and adds stronger
collaborative-filtering baselines. It also expands the latent dimension beyond
the original 5--20 range to 64, 128, and 256.

The MovieLens study is a new benchmark and must not be described as another
reproduction of the original paper. Its data, split, candidate protocol, and
interpretation of genre information differ from MindReader.

## Data and feedback

We used the official stable MovieLens 100K and MovieLens 1M releases. The
catalog contains items occurring in the ratings data, with their official movie
genres. Ratings of at least 4 were treated as positive feedback, ratings of at
most 2 as explicit negative feedback, and ratings equal to 3 as neutral.

We applied a strict per-user temporal leave-two-positive-out split. The
penultimate positive interaction was used for validation and the final positive
interaction for testing. Only events strictly before validation were available
for training. Users retained at least one positive training event. To evaluate
a warm-start recommendation setting, users were removed iteratively until both
held-out target items had positive support in the remaining global training
data.

## Sparsity protocol

Training histories were evaluated at 100%, 80%, 60%, 40%, and 20%. For each
seed, every user retained one positive anchor and a prefix of a single
deterministically shuffled training history. Consequently, lower proportions
were nested subsets of higher proportions. The warm-start condition was
defined before this intervention; sparsification was not target-aware and
could remove the final training support for a held-out item. We therefore
stored, for every fit, the number of validation and test targets retaining
positive support after sparsification. Validation/test targets and sampled candidate sets were unchanged across
proportions. For each user and stage, exactly 100 negatives were sampled without
replacement from items that the user never rated anywhere in the dataset. The
held-out positive was then appended, yielding 101 candidates. This prevents a
known rated item from being treated as an evaluation negative while keeping the
candidate set independent of sparsification, model, configuration, and training
seed.

## Models

The study compared six recommenders:

1. pointwise matrix factorization trained by mean squared error (MF-MSE);
2. matrix factorization trained by Bayesian Personalized Ranking (BPR-MF);
3. EASE;
4. LightGCN;
5. LTN with an MF scoring function;
6. LTN with the same scorer plus genre constraints (LTN-genres).

MF-MSE and the LTN variants shared the same seeded MF scorer, initialization,
and user/item bias parameterization. They were trained on the same observed
positive and explicit-negative binary targets, without additional sampled
negatives. BPR-MF, EASE, and LightGCN used positive implicit interactions
according to their standard objectives. For pairwise negative sampling, only
positive items in the current sparse training set were excluded; validation,
test, and removed labels were not inspected.

## MovieLens genre adaptation

MovieLens provides movie genres but not direct user preferences for genre
entities. User-genre preferences were therefore derived from non-neutral movie
ratings in the current sparse training set. The logical rule states that a
movie belonging to a disliked genre should receive a low preference score. A
positive training item and an item also matching a liked genre were excluded
from this negative rule to avoid contradictory supervision.

The data quantifier used p-mean error with p=2, while the genre-rule
quantifier's p was selected from 2, 5, and 10. The two closed formulas were
combined by a weighted p-mean-error aggregation with p=2. At genre-rule weight
1, this corresponds to the two-formula SatAgg aggregation used in the original
repository. The rule is catalog-wide and
transductive because it may constrain an unobserved movie from its static
genres. No validation or test rating is used to derive the rule. Likewise, target
identities are not used to preserve interactions during sparsification.

This adaptation does not provide independent background-knowledge labels as in
MindReader: the user-genre evidence is inferred from the sparse movie ratings
and decreases with sparsity. Results for LTN-genres must be interpreted with
this limitation.

## Model selection

Hyperparameters were selected independently for every dataset, training
proportion, and model by validation-only deterministic stratified random
search. For MF-MSE, BPR-MF, LTN, and LTN-genres, the candidate dimensions were
5, 10, 20, 64, 128, and 256, with exactly six sampled configurations per
dimension. LightGCN used dimensions 64, 128, and 256, with twelve sampled
configurations per dimension. All six EASE regularization values were evaluated.

Each learned-model search contained 36 configurations, while EASE contained 6.
Every configuration was evaluated with three validation seeds. Training used a
maximum of 100 epochs and early-stopping patience of 10. The selected
configuration was then retrained and tested with 30 new paired seeds. In total,
the protocol contains 5,580 validation fits and 1,800 final fits.

## Evaluation and reporting

For each eligible user, models ranked one held-out positive among 100 fixed
sampled unrated negatives (101 candidates), matching the candidate-set size used
by the original MindReader repository. We report hit@10 and ndcg@10. This makes
the evaluation protocol structurally comparable, but MovieLens data, temporal
splits, feedback construction, and genre semantics still differ from MindReader.
For each dataset, proportion, and model, the final table contains the arithmetic
mean and population standard deviation across the 30 runs. Seed indices are
paired across models and proportions.

The generated reports also retain all validation configurations, selected
hyperparameters, training/inference times, post-sparsification target-support
rates, and per-run CUDA peak allocated memory where applicable. CPU high-water memory is omitted because it is
cumulative within a process and is not a fair per-model measurement.

## Reproducibility controls

All splits, sparsification samples, initializations, and negative samples are
deterministic functions of stored seeds. CUDA deterministic algorithms and a
fixed cuBLAS workspace configuration are enabled. Intermediate search and final
results are atomically persisted. Fingerprints include the source code, dataset,
runtime, resolved protocol, sampled configurations, selected parameters, and
checkpoint policy. An exclusive output lock prevents concurrent jobs from
writing to the same experiment directory.

## Results checklist

After the full run, use the generated files rather than copying values from the
console:

- `reports/report.md` and `reports/report.tex` for final tables;
- `reports/metrics_long.csv` for mean, standard deviation, and run count;
- `reports/selected_hyperparameters.json` for the chosen configuration;
- `reports/search_configurations.csv` for every validation trial;
- `reports/embedding_validation_summary.csv` for the 5/10/20 versus
  64/128/256 analysis.

Before submitting the report, verify that every final cell has `n=30`, that
the status command reports 5,580/5,580 search fits and 1,800/1,800 final runs,
that post-sparsification target-support rates are discussed alongside the
ranking metrics, and that no selected learned model consistently reaches the
100-epoch cap. If
many selected configurations reach the cap, repeat those affected searches
with a larger epoch budget and a new output directory.
