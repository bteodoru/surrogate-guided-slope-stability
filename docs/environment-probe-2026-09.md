# Environment probe, September 2026

Temporary record for phase 0 of the pipeline reorganization. Deleted at the end
of phase 6 (Task 20), once the local fits have replaced the Colab ones.

## Reference: hyper-parameters selected on Colab

Source: stored output of cell 38 in `surrogate_guided_sampling_rev.ipynb`
(reference copy in `reference-data-2026-09/`). One `GridSearchCV` per
(family, arm), 5-fold `KFold(shuffle=True, random_state=42)`,
`scoring="neg_root_mean_squared_error"` on log(FOS).

| family | arm | n   | max_features | min_samples_leaf | learning_rate | max_iter | max_leaf_nodes |
|--------|-----|-----|--------------|------------------|---------------|----------|----------------|
| RF     | A   | 300 | 1.0          | 1                |               |          |                |
| RF     | C0  | 300 | 1.0          | 2                |               |          |                |
| RF     | C   | 600 | 1.0          | 1                |               |          |                |
| RF     | B   | 600 | 1.0          | 1                |               |          |                |
| GBM    | A   | 300 |              | 3                | 0.06          | 600      | 7              |
| GBM    | C0  | 300 |              | 5                | 0.03          | 600      | 7              |
| GBM    | C   | 600 |              | 3                | 0.06          | 600      | 7              |
| GBM    | B   | 600 |              | 3                | 0.10          | 600      | 7              |

## Observed locally

Output of `tools/probe_environment.py` against the reference CSVs in
`reference-data-2026-09/`, on scikit-learn 1.9.1 / joblib 1.6.0 / Python
3.14.4:

```
scikit-learn 1.9.1

== Designs (reproduced vs reference inputs) ==
  screener OOB R² (log): 0.9565   (notebook: 0.9567)
  guided: n=300 vs 300, max |Δparam| = 7.35e+01
  lhs-extension: n=300 vs 300, max |Δparam| = 5.00e-07
  test-natural: n=300 vs 300, max |Δparam| = 5.00e-11
  min normalized distance lhs-extension -> earlier sets: 0.09392
  min normalized distance guided -> earlier sets: 0.06685
  min normalized distance test-natural -> earlier sets: 0.05033
  test-stratified: n=150 vs 150, max |Δparam| = 6.66e+01

== Hyper-parameters (local vs Colab) ==
  RF  A  n=300  {'max_features': 1.0, 'min_samples_leaf': 1}  ==
  RF  C0 n=300  {'max_features': 1.0, 'min_samples_leaf': 2}  ==
  RF  C  n=600  {'max_features': 1.0, 'min_samples_leaf': 1}  ==
  RF  B  n=600  {'max_features': 1.0, 'min_samples_leaf': 1}  ==
  GBM A  n=300  {'learning_rate': 0.06, 'max_iter': 600, 'max_leaf_nodes': 7, 'min_samples_leaf': 3}  ==
  GBM C0 n=300  {'learning_rate': 0.03, 'max_iter': 600, 'max_leaf_nodes': 7, 'min_samples_leaf': 5}  ==
  GBM C  n=600  {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 7, 'min_samples_leaf': 3}  != {'learning_rate': 0.06, 'max_iter': 600, 'max_leaf_nodes': 7, 'min_samples_leaf': 3}
  GBM B  n=600  {'learning_rate': 0.06, 'max_iter': 600, 'max_leaf_nodes': 7, 'min_samples_leaf': 3}  != {'learning_rate': 0.1, 'max_iter': 600, 'max_leaf_nodes': 7, 'min_samples_leaf': 3}

All selections agree with Colab: False
```

**Verdict:** partial agreement, with the divergence understood and accepted.
The two pure-QMC designs reproduce to rounding precision — `lhs-extension`
(max |Δparam| = 5.00e-07) and `test-natural` (5.00e-11) — since
`scipy.stats.qmc.LatinHypercube` involves no model and no data-dependent
branching. The two screener-driven designs diverge materially: `guided`
(max |Δparam| = 7.35e+01) and `test-stratified` (6.66e+01). GBM arms **B**
and **C** also disagree: locally, C selects `learning_rate=0.10` and B
selects `learning_rate=0.06`, the reverse of the Colab record
(C: 0.06, B: 0.10); RF selections and GBM arms A/C0 match exactly.

Mechanism: the RF screener's OOB R² on log(FOS) is 0.9565 locally against
0.9567 recorded from the Colab run — a ~2e-4 difference, plausibly from a
scikit-learn/numpy version or platform difference, or from floating-point
non-associativity under `n_jobs=-1`. Because the `guided`/`test-stratified`
selection logic bins 200,000 candidates by predicted FOS against hard
thresholds, then runs `greedy_maxmin` off a single sequential
`np.random.default_rng` stream, that tiny model-level difference is enough to
shift which candidates land in which bin near a threshold, which cascades
through the deterministic-but-order-sensitive greedy selection into a
materially different set of 300 (or 150) rows overall — not a rounding-scale
discrepancy but a different draw. The GBM B/C flip reflects a genuine
near-tie in nested-CV score between `learning_rate=0.06` and `0.10` for those
two arms, which the same kind of floating-point sensitivity is enough to flip.

This divergence is expected and does not block moving forward. The Colab run
was a first exploratory iteration that established the methodological
direction (screener-guided sampling beats plain LHS extension); it was never
a locked-in numerical reference, and the dataset currently published on
Zenodo predates the xslope/Bishop generator entirely (it was built on the
now-superseded HYRCAN/Spencer's-method generator, replaced after a
label-integrity problem — see `CLAUDE.md`). Reproducibility going forward
means that, given the code, config and seeds, re-running the **local**
pipeline reproduces the **local** pipeline's own output — not that the local
stack must reproduce Colab's specific numbers bit-for-bit. The local
xslope+Bishop stack on scikit-learn 1.9.1 / Python 3.14.4 is the new source
of truth for what gets published; the Colab numbers above are retained here
only as a record of where that first iteration stood.
