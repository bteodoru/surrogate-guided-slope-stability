# Declared predictions and their outcomes

Predictions are declared in `sgs/config.py` (the comment above `COMPARISONS`)
before any solver run for the design they concern. Outcomes are recorded here
once, as observed, including the predictions that failed. One realization per
design; the paired bootstrap covers test-sampling variability only.

## C-seq (declared in ca86071, 2026-09-19; batches generated in 84a78ab)

Training data only (checked before any test evaluation):

| prediction | outcome | held? |
|---|---|---|
| the three batches put <= 10 points below FOS 0.5 (guided-log: 45) | 32 (per batch 22 / 6 / 4); the first batch, whose screener saw only the seed, behaves like guided-log | **no** |
| seed + batches deviate less from the target per-bin counts than seed + guided-log | sum of absolute deviations 66 vs 92 | yes |

Test sets, log-RMSE against B (outputs of e8eafe3):

| prediction | outcome | held? |
|---|---|---|
| C-log's advantage on test points with FOS < 0.5 shrinks for C-seq | shrinks in 3 of 4 family × test cells; grows under RF on test-natural | mostly |
| C-seq may gain in [0.5, 1) | gains in 3 of 4 cells, ties in 1 (GBM, test-stratified) | yes |
| C-seq ties or loses in [1, 2) | loses in 4 of 4 cells | yes |
| no systematic global gain | RF: C-seq below B on both tests; GBM: above B on both tests | yes |

On the decision range [0.5, 2) (the primary criterion), every C-seq vs B
bootstrap interval of ΔRMSE on the log scale contains 0.
