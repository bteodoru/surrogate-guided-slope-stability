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

Outcomes are the mean of the first arm − the mean of the second, on log-RMSE
over r1–r5, with the 95% interval; the four cells are given in the order
GBM/natural, GBM/stratified, RF/natural, RF/stratified. r0 enters no interval.

| prediction | outcome | held? |
|---|---|---|
| **Decision range:** for neither C-log nor C-seq is the Welch interval against B entirely below 0 in all four cells | no interval against B is entirely below 0 in any cell. C-log − B: +0.009 [+0.002, +0.016] (entirely *above* 0), −0.004 [−0.019, +0.010], −0.009 [−0.020, +0.002], −0.006 [−0.017, +0.006]. C-seq − B: +0.004 [−0.002, +0.011], −0.004 [−0.018, +0.010], +0.002 [−0.017, +0.020], −0.002 [−0.017, +0.012] | yes |
| **Zones:** the mean C-seq − B difference is negative in 0.5–1 and positive in 1–2, in all four cells | 8 of 8 signs as declared. In 0.5–1: −0.014 [−0.024, −0.005], −0.011 [−0.036, +0.013], −0.031 [−0.068, +0.006], −0.031 [−0.059, −0.003]. In 1–2: +0.011 [+0.003, +0.019], +0.001 [−0.013, +0.015], +0.018 [+0.006, +0.030], +0.018 [+0.008, +0.029] | yes |
| **C-seq vs C-log:** under GBM the interval contains 0 on both tests; under RF the mean difference is negative on both tests | the GBM half held on the decision range: −0.005 [−0.013, +0.003] and +0.001 [−0.011, +0.012]; on the whole test set it fails on test-stratified, +0.004 [+0.000, +0.008]. The RF half failed: the mean difference is **positive** on both tests, +0.011 [−0.007, +0.029] and +0.003 [−0.011, +0.018] | **no** |

What the replications can and cannot settle: they measure the variability
**between designs** on a fixed seed, with fixed hyper-parameters. They do not
measure seed variability, and they do not replicate the tuned comparison. With
n = 5 the Welch degrees of freedom are at most 8, so intervals will be wide and
"inconclusive" is the expected outcome for small effects; P̂(first < second) over
the 25 pairs is reported beside each interval as a descriptive companion, and an
interval containing 0 is never written up as a tie.

**Observed, 2026-09-20** (`outputs/metrics/replications.json`; tables
`replications__*`). On every row that decides one of the three predictions the
relative scale, log(log-RMSE), agrees in sign and in which intervals exclude 0,
so nothing below turns on the choice of scale.

The zone prediction is the one that survives intact. The relocation is
repeatable in sign in 8 of 8 cells, and the counter-example of the single tuned
realization — stratified GBM, where C-seq was 4% worse than B in 0.5–1 — does
not recur across the five new realizations under the common fixed
hyper-parameters (the mean difference there is −0.011, with an inconclusive
interval). The intervals are weaker than the signs: in 0.5–1 the C-seq − B
interval is entirely below 0 in 2 of 4 cells and inconclusive in the other two;
in 1–2 it is entirely above 0 in 3 of 4 and inconclusive on stratified GBM. The
cost side is better resolved than the gain side, as it was on the single
realization. C-log relocates the same way: entirely below 0 in 0.5–1 under RF on
both tests (−0.052 [−0.070, −0.034] natural, −0.036 [−0.056, −0.017]
stratified) and entirely above 0 in 1–2 in 3 of 4 cells.

On the decision range, the primary criterion, no guided arm has an interval
against B entirely below 0 in any of the four cells, and the only interval there
that excludes 0 puts C-log *above* B (natural, GBM). Every other cell is
inconclusive, which is what a five-realization Welch interval (df ≤ 8) buys —
not evidence that the arms are equivalent.

The ablations separate the seed from the added points, and do so in every cell.
On the decision range C₀-log and C₀-seq are above B in 4 of 4 cells with
intervals excluding 0 (C₀-log +0.032 to +0.068, C₀-seq +0.065 to +0.112), and
the paired intervals C-log − C₀-log and C-seq − C₀-seq are entirely below 0 in
4 of 4 cells (−0.036 to −0.073 and −0.066 to −0.114). Neither guided design
replaces the seed.

The interaction intervals — log(log-RMSE_rf) − log(log-RMSE_gbm), Welch between
the two arms — exclude 0 only on test-natural and only for C-log against B:
−0.249 [−0.350, −0.148] on the decision range, and likewise on all, on the
selection range, in 0.5–1 and in 1–2. Every C-seq-against-B interaction
interval, and every interaction interval on test-stratified, contains 0. The one
interaction that can be claimed is therefore narrow: on test-natural the
RF-to-GBM gap is smaller under C-log than under B. These are two tuned
tree-ensemble families and support nothing about learners in general.

r0, the existing sets, lies outside the r1–r5 range on the decision range in 6
of the 20 family × test × arm cells, and for C-log under RF it is worse than
every one of the five new realizations on both tests (0.119 against a replicated
mean of 0.095 on test-natural, 0.124 against 0.096 on test-stratified). That is
why it is reported apart: it is the draw that shaped the variants and the
predictions.
