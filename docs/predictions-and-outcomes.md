# Declared predictions and their outcomes

Predictions are declared in `sgs/config.py` (the comment above `COMPARISONS`)
before any solver run for the design they concern. Outcomes are recorded here
once, as observed, including the predictions that failed. Outside the
replications there is one realization per design, and the paired bootstrap
covers test-sampling variability only.

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
| C-seq may gain in [0.5, 1) | gains in 3 of 4 cells; slightly worse in the fourth (GBM, test-stratified: 0.054 vs 0.052, n = 12) | mostly |
| C-seq ties or loses in [1, 2) | loses in 4 of 4 cells | yes |
| no systematic global gain | RF: C-seq below B on both tests; GBM: above B on both tests | yes |

On the decision range [0.5, 2) (the primary criterion), every C-seq vs B
bootstrap interval of ΔRMSE on the log scale contains 0.

## Replications (declared 2026-09-19 in the `REP_*` block of `config.py`; recorded here 2026-09-20, before any replication is solved)

Five new realizations each of the LHS extension (B), guided-log (C-log) and the
three sequential batches (C-seq), on the fixed seed and the fixed test sets.
Plan: `docs/superpowers/plans/2026-09-19-replications.md`.

**Read the levels below with care.** They come from the **tuned** fits of the
single existing realization, which is what shaped the predictions. The
replications, and the refit of this realization as r0, use **common
hyper-parameters fixed from the seed** (`LC_PARAMS`), so the levels will move.
What is predicted is the **sign and the consistency of the differences across
realizations**, never these values.

Starting point, log-RMSE by zone against B, one realization, tuned fits:

| test | family | zone | B | C-log | C-seq |
|---|---|---|---|---|---|
| natural | GBM | 0.5–1 | 0.0636 | 0.0520 (−18%) | 0.0383 (−40%) |
| natural | RF | 0.5–1 | 0.1359 | 0.1625 (**+20%**) | 0.0813 (−40%) |
| stratified | GBM | 0.5–1 | 0.0520 | 0.0255 (−51%) | 0.0542 (**+4%**) |
| stratified | RF | 0.5–1 | 0.1433 | 0.1428 (−0%) | 0.1087 (−24%) |
| natural | GBM | 1–2 | 0.0437 | 0.0520 (+19%) | 0.0590 (+35%) |
| natural | RF | 1–2 | 0.0804 | 0.1000 (+24%) | 0.0994 (+24%) |
| stratified | GBM | 1–2 | 0.0415 | 0.0437 (+5%) | 0.0576 (+39%) |
| stratified | RF | 1–2 | 0.0639 | 0.1125 (+76%) | 0.0972 (+52%) |

So on this one draw the **cost** side of the relocation is already consistent —
both guided arms are worse than B in 1–2 in 4 of 4 cells — while the **gain**
side is not: C-log gains in 2 of 4 cells and C-seq in 3 of 4. The declared zone
prediction therefore already fails in one cell here (stratified GBM, +4%), and
is at genuine risk.

Decision range [0.5, 2), the primary criterion, same fits:

| test | family | B | C-log | C-seq |
|---|---|---|---|---|
| natural | GBM | 0.0497 | 0.0520 | 0.0544 |
| natural | RF | 0.0979 | 0.1194 | 0.0950 |
| stratified | GBM | 0.0453 | 0.0386 | 0.0565 |
| stratified | RF | 0.0978 | 0.1235 | 0.1012 |

| prediction | outcome | held? |
|---|---|---|
| **Decision range:** for neither C-log nor C-seq is the Welch interval against B entirely below 0 in all four cells | | |
| **Zones:** the mean C-seq − B difference is negative in 0.5–1 and positive in 1–2, in all four cells | | |
| **C-seq vs C-log:** under GBM the interval contains 0 on both tests; under RF the mean difference is negative on both tests | | |

What the replications can and cannot settle: they measure the variability
**between designs** on a fixed seed, with fixed hyper-parameters. They do not
measure seed variability, and they do not replicate the tuned comparison. With
n = 5 the Welch degrees of freedom are at most 8, so intervals will be wide and
"inconclusive" is the expected outcome for small effects; P̂(first < second) over
the 25 pairs is reported beside each interval as a descriptive companion, and an
interval containing 0 is never written up as a tie.
