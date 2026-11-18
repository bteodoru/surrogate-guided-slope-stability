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
| **Zones:** the mean C-seq − B difference is negative in 0.5–1 and positive in 1–2, in all four cells | 8 of 8 signs as declared. In 0.5–1: −0.014 [−0.024, −0.005], −0.011 [−0.036, +0.013], −0.031 [−0.068, +0.006], −0.031 [−0.059, −0.003]. In 1–2: +0.011 [+0.003, +0.019], +0.001 [−0.012, +0.014], +0.018 [+0.006, +0.030], +0.018 [+0.008, +0.029] | yes |
| **C-seq vs C-log:** under GBM the interval contains 0 on both tests; under RF the mean difference is negative on both tests | the declaration names no measure, so reading it on the decision range is post hoc. There the GBM half held: −0.005 [−0.013, +0.003] and +0.001 [−0.011, +0.012]; on the whole test set it fails on test-stratified, +0.0041 [+0.0005, +0.0078]. The RF half failed on the decision range, on the whole test set and on the selection range alike: the mean difference is **positive** on both tests, +0.011 [−0.007, +0.028] and +0.003 [−0.011, +0.018] | **no** |

What the replications can and cannot settle: they measure the variability
**between designs** on a fixed seed, with fixed hyper-parameters. They do not
measure seed variability, and they do not replicate the tuned comparison. With
n = 5 the Welch degrees of freedom are at most 8, so intervals will be wide and
"inconclusive" is the expected outcome for small effects; P̂(first < second) is
reported beside each interval as a descriptive companion — over the 25
realization pairs for the Welch (independent-arm) comparisons and, for paired
comparisons, over the 5 within-realization pairs, ties counting ½ in both — and
an interval containing 0 is never written up as a tie.

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
both tests (−0.052 [−0.070, −0.034] natural, −0.036 [−0.055, −0.017]
stratified) and entirely above 0 in 1–2 in 3 of 4 cells.

On the decision range, the primary criterion, neither C-log nor C-seq has an
interval against B entirely below 0 in any of the four cells, and the only C-log
or C-seq interval against B that excludes 0 there puts C-log *above* B (natural,
GBM). Every other one of those eight cells is inconclusive, which is what a
five-realization Welch interval (df ≤ 8) buys — not evidence that the arms are
equivalent.

The ablations separate the seed from the added points, and do so in every cell.
On the decision range C₀-log and C₀-seq are above B in 4 of 4 cells with
intervals excluding 0 (C₀-log +0.032 to +0.068, C₀-seq +0.065 to +0.112), and
the paired intervals C-log − C₀-log and C-seq − C₀-seq are entirely below 0 in
4 of 4 cells (−0.036 to −0.073 and −0.066 to −0.114). Neither guided design
replaces the seed.

The interaction intervals — log(log-RMSE_rf) − log(log-RMSE_gbm), Welch between
the two arms — exclude 0 in 8 of the 32 stored rows (2 tests × 8 measures × 2
contrasts). Of the five measures the reported table carries, the only ones that
exclude 0 are C-log against B on test-natural, and there in all five: −0.249
[−0.350, −0.148] on the decision range, −0.195 [−0.293, −0.097] on all, −0.122
[−0.193, −0.051] on the selection range, −0.234 [−0.434, −0.033] in 0.5–1 and
−0.172 [−0.285, −0.059] in 1–2. The other three sit in FOS ≥ 6, outside the
selection range: C-log against B −0.654 [−1.002, −0.306] and C-seq against B
−0.558 [−0.877, −0.238] on test-natural, and C-log against B −0.318 [−0.567,
−0.069] on test-stratified. Every other row contains 0: every C-seq row but the
one named, and every test-stratified row but the one named. What can be claimed
is therefore narrow: on test-natural the RF-to-GBM gap is smaller under C-log
than under B across the reported measures, and on that same test both guided
arms narrow it in FOS ≥ 6. These are two tree-ensemble families fitted with
hyper-parameters fixed from the seed and support nothing about learners in
general.

r0, the existing sets, lies outside the r1–r5 range on the decision range in 6
of the 20 family × test × arm cells, and for C-log under RF it is worse than
every one of the five new realizations on both tests (0.119 against a replicated
mean of 0.095 on test-natural, 0.123 against 0.096 on test-stratified). That is
why it is reported apart: it is the draw that shaped the variants and the
predictions.

## Structural analysis — a procedure committed before the structural code, with no hypothesis attached

This procedure was committed in c1681ce (2026-09-21 10:57), which changes only
this document and CLAUDE.md. At that commit, neither `sgs/structural.py` nor
`outputs/metrics/structural.json` had been committed. The structural code was
committed between 2 and 5 minutes later (d5023b3 to 919fd95, the artifact in
919fd95). It had been drafted before, in an implementation plan that git does
not track. Git shows the order of these commits, and nothing about when the
code was written or first run. The design is dated 2026-09-20 in a file git
does not track either.

The procedure was fixed **on replication data already computed**. The
bin-level replication results of every arm had existed since 075edb7
(2026-09-20 19:44): in `outputs/metrics/replications.json`, and drawn in
`outputs/figures/replications__*.pdf`. **The analysis is not preregistered
before those results were obtained**, and it must not be presented as if it
were. It takes no `outcome` and no `held?` column, because it is not a
prediction. What it records is the procedure of the structural analysis: the
weighting, the bin set, the thresholds and the summary statistic. The limit of
what that protects is stated below.

*Corrected 2026-09-21, after the reviews of the complete structural work.* The
first version of this section said that the procedure was committed "before
the analysis was implemented and before any of its output existed". A second
version kept "before the structural analysis was implemented". Git supports
neither claim. It shows only that no structural code or artifact had been
committed, and the bin-level results the analysis reorganizes had been
computed and drawn the day before. The same correction is applied to
Amendment 1 below.

- **Bins.** The 15 geometric bins of the selection range, `config.N_BINS`. The
  two bins outside it have no defined intended density; they appear in the
  supplementary rows and document out-of-selection-range spillover and tail
  behaviour, never under the name *truncation*.
- **Realizations.** r1–r5 confirmatory; r0 reported apart and never averaged
  into a confirmatory mean.
- **Thin bins.** Bins with fewer than `config.STRUCTURAL_MIN_TEST_N = 10` test
  points are marked and never dropped; no statement in the text rests on one.
- **No pooled weighted statistic.** The figure quotes no aggregate, so the
  choice between equal and test-count weighting does not arise. The single
  descriptive number is the count of bins in which the change in density and
  the change in log-RMSE have opposite signs, `Δdensity × Δlog-RMSE < 0`.
  Because no sentence may rest on a marked bin, that count is reported over the
  **unmarked** bins; the count over all 15 is reported beside it as a
  descriptive sensitivity, together with how many bins were thin. Bins where
  either difference is exactly zero leave the denominator and are reported.
- **Status.** No new hypothesis test or inferential claim is introduced. Welch
  intervals are reused to visualize between-design uncertainty at bin level.
  Inference remains with the pre-declared zone contrasts.
- **What `achieved − intended` is.** The net distortion of the requested
  bin-count distribution induced by screening error — not the screener's
  prediction error. Two points swapping bins leave it untouched. The
  supplementary displacement rows measure the individual predictions instead,
  and then only on the points the screener selected, which are conditioned on
  having been predicted inside the range and inside a deficit bin.
- **Binning conventions.** Everything read from `replications.json` — the
  figure's densities and errors, and both sides of the outside table — uses the
  half-open reporting zones `[lo, hi)`, whose outer bin is FOS ≥ 6. The
  allocation and displacement rows follow the generator's `(lo, hi]`, whose outer
  bin is FOS > 6, because they are compared against the designs' own bins. The
  two agree except on an exact bin edge.
- **What none of it does.** It does not separate the allocation rule from the
  screener's error. That needs a reference without screener error; the oracle
  keeps that job.
- **Blinding, and its limit.** The blinding is partial.
  - **What existed.** The bin-level errors and mean training counts of every arm
    were computed and committed before this procedure: `replications.json` and
    `replications__{test}__{family}.pdf` (075edb7, 2026-09-20). Those figures
    draw, on the same 17 bins, the mean log-RMSE over r1–r5 of B, C-log, C-seq,
    C₀-log and C₀-seq.
  - **What was read.** The zone-level replication results of the guided arms,
    recorded earlier in this document (for instance C-log − B in 0.5–1 across
    the four cells). One bin-level row of `replications.json` for a non-guided
    arm: arm A, GBM, test-natural, r0. The design columns of
    `data/replications/designs/`, including the per-bin selected counts of
    `guided_log_r1`, seen while checking that selected equals intended.
  - **What is stated as not inspected.** The author states that the bin-level
    errors of the guided arms were not inspected before this procedure was
    fixed. That statement cannot be verified from the repository, because
    those results were available. A file existing does not show that it was
    inspected, and the fact that it was available does not show that it was
    not.
  - **What the procedure therefore protects.** At most, the choice of the
    bins, thresholds, weighting and summary statistic above, to the extent the
    statement holds. It does not protect the direction of the zone-level
    pattern: knowing that 0.5–1 gains and 1–2 loses already implies, roughly,
    what the bins inside those zones will show. This is one reason the
    analysis is descriptive.

### Amendment 1 (2026-09-21): the C₀ ablations — an extension decided after the review of Tasks 0–4

**This is not part of the original declaration above.** The independent review
of the implemented Tasks 0–4 found that the procedure, the design and the
module covered only C-log and C-seq against B, leaving out C₀-log and C₀-seq
contrary to the repository's rule that every C₀ variant appears in every
analysis. The omission was in the declaration itself, not only in the code.
The amendment was committed in fe77b9a (2026-09-21 11:24), which changes only
this document. The code and artifact rows for C₀ were committed 2 minutes 18
seconds later, in 9ece938. Git shows that order and nothing about when that
code was written or run. The bin-level C₀ results it reorganizes were not
new. The bin-level
errors and training counts of C₀-log and C₀-seq had been computed and drawn in
`replications__*.pdf` since 075edb7, like those of every other arm. The author
states that they were not inspected before this amendment was fixed. As with
the declaration above, the repository cannot verify that statement. What was
known for certain is the zone- and decision-range-level C₀ replication result
recorded earlier in this document. From the main pipeline, it was also known
that the original C₀ under-predicts every FOS ≥ 6 test point.

- **Unchanged.** The main figure and the concordance count stay C-log and
  C-seq against B, the arms with the same budget (600). No C₀ row enters
  either.
- **Supplementary ablation rows, error.** For C₀-log and C₀-seq, for each family ×
  test cell and each of the 17 reporting bins (the 15 in-range bins and the
  two outside ones, all `[lo, hi)` as read from `replications.json`), the
  difference in mean log-RMSE over r1–r5:
  - **against B, by a Welch interval** (`config.REP_WELCH`). Every such row is
    labelled as a comparison with **different budgets**: C₀ trains on the
    300 guided points alone, B on 600. The difference mixes the budget with
    the training distribution and is not a like-for-like comparison.
  - **against the corresponding C, by a paired interval** (`config.REP_PAIRED`:
    C-log − C₀-log, C-seq − C₀-seq, in that order). Within a realization, C₀
    trains on exactly the guided points that C adds to the seed, so the
    realization index is a genuine pairing factor.
  Thin bins are marked by the same `STRUCTURAL_MIN_TEST_N` rule and never
  dropped. r0 does not enter these rows. The intervals visualize
  between-design variability; they are no test, and an interval containing 0
  is inconclusive, never a tie.
- **Supplementary ablation rows, training distribution.** Where the C₀ points
  are set against B, the quantity is the fraction `n_bin / N_total` over the
  17 bins, never a difference of raw counts, which would carry the budget
  difference. It is shown per bin as C₀ − B with a Welch interval, over
  r1–r5. These fractions do not enter the concordance count or any other
  summary.
- **Allocation and displacement.** C₀-log and C₀-seq use the same guided
  points as C-log and C-seq in each realization. Their allocation, batch and
  displacement rows are those of C, shared rather than duplicated, and the
  artifact says so.
- **The original C₀** (arm `C0`, role `guided`) exists only as r0 in the main
  pipeline. It was never replicated, so it has no replication interval. It
  also predates the log-deficit rule and stores no allocation provenance, so
  it has no allocation row. The artifact names it in a note and computes
  nothing for it.

### Amendment 2 (2026-09-23 or the date of this commit): the pool arms in the structural analysis

C-oracle and C-logP join `ARMS_VS_B`, and C0-oracle and C0-logP join the ablations, under the same
procedure as C-logG: descriptive, Welch against B, paired against the corresponding C. There is no
r0 for any of them. C-oracle's allocation and displacement rows are exact by construction; the
table shows them as a check, not as a finding. The overlap between C-oracle and C-logP is counted
by shared pool row. Recorded after the stage-1 note and the replication metrics of the pool arms
existed, as the git history shows; nothing here is a preregistration.

## Oracle (spec 2026-09-23): outcome

Declared text: `docs/oracle-declaration.md` §9 (tag `oracle-declared`, ab43902),
quoted below; `config.py` carries its summary. The pool stopped when its rule
was met, at chunk 22 (c022, 22 000 rows). Stage 1 is recorded in
`docs/oracle-stage1/RESULTS.md` (409c18a), before any fit of C-oracle or
C-logP; its binding values are ρ_L = 1.227 and ρ_H = 1.088. At the stop
(chunk 22) the predicted side was binding: sub-pool r3, bin 0 held exactly 38
predicted rows against a[0] = 38 (solved: 63), contrary to the expectation
declared in `docs/oracle-declaration.md` §4.4 ("The predicted side does not
bind"). C-logP therefore took all 38 of them in r3, the mirror of §4.4's "at
the stop" note on C-oracle. Stage 2 is read
from the `oracle` block of `outputs/metrics/replications.json` (fits and
metrics of bbca97d), over r1–r5 only: there is no r0 for the pool arms. The
four cells are given in the order GBM/natural, GBM/stratified, RF/natural,
RF/stratified; *m* = ln 1.10 = 0.095. L = [0.5, 1) and H = [1, 2), log-RMSE
against B, medians over r1–r5.

**Step P: the gate for C-logP passes.** Median (L, H) pairs: (−0.417, +0.281),
(−0.010, −0.156), (−0.242, +0.078), (−0.218, +0.215); median L < 0 and median
H > 0 in 3 of 4 cells, GBM/stratified being the exception. The restriction
that §9.3 places on the readings when this gate fails does not apply.

**Group A (the gate passes for C-oracle): row A5.** C-oracle's median (L, H)
pairs: (−0.256, +0.155), (−0.206, −0.274), (−0.487, +0.096), (−0.431, +0.181);
the gate holds in 3 of 4 cells, GBM/stratified again the exception. With
C-logP's gate passing, the reading of §9.3 is:

> **Mixed / inconclusive** on dR_O, with the four medians and intervals shown

No earlier row matched. The median dR_O is ≥ m in one cell (A1 needs 3) and
≤ −m in one cell (A2 needs 3); no paired interval lies inside (−m, m) (A3);
|median dR_O| < m in two cells (A4 needs 3). `attribution` is false, so no
attribution sentence is written (§9.4). The implied direction declared in §9.2,
dR_O > 0, appears in 2 of the 4 medians.

| cell | median dR_O | mean dR_O, paired 95% interval |
|---|---|---|
| GBM / natural | −0.242 | −0.184 [−0.439, +0.072] |
| GBM / stratified | −0.063 | −0.019 [−0.570, +0.531] |
| RF / natural | +0.027 | +0.199 [−0.221, +0.619] |
| RF / stratified | +0.108 | +0.118 [−0.094, +0.330] |

Every paired dR_O interval contains 0: inconclusive in 4 of 4 cells, as §9.5
expected at n = 5 (declared half-widths 0.14–0.42; observed 0.21–0.55). The
interval is centred on the mean over r1–r5, the reading on the median, hence
the two columns.

**Group C (decision range, Welch against B): row C3.** C-oracle reaches
level 2, C-logP level 1. The reading of §9.3:

> "No demonstrated gain" stands, now also with known bin membership, in this design

- C-oracle − B, mean [95% Welch interval]: +0.0014 [−0.0036, +0.0064],
  −0.0082 [−0.0218, +0.0054], −0.0145 [−0.0265, −0.0026], −0.0131 [−0.0237,
  −0.0026]. Entirely below 0 in the two RF cells (level 2), not in all four
  (level 3 not reached).
- C-logP − B: +0.0035 [−0.0025, +0.0096], −0.0077 [−0.0214, +0.0059], −0.0016
  [−0.0163, +0.0130], +0.0035 [−0.0104, +0.0174]. The mean is below 0 in two
  cells (level 1); no interval is entirely below 0.
- C-oracle − C-logP, paired: −0.0021 [−0.0095, +0.0052], −0.0005 [−0.0084,
  +0.0074], −0.0129 [−0.0303, +0.0046], −0.0166 [−0.0361, +0.0029]. Every
  interval contains 0, so C1's second condition fails as well.

C-oracle against B is not read as a statement about the screener (§9.4; only
C1 would allow that). C-oracle's points are drawn from a solved pool, so its
level against B is not a budget result.

**Density prediction (§9.2), at its two levels.**

- **L: ρ_L = 1.227 ≥ 1.10, predicted dL_O < 0.** Medians +0.047, −0.116,
  −0.262, −0.304: the predicted sign in 3 of 4 cells, **direction met**;
  |median| ≥ m in the same 3, **magnitude met**. As §9.2 declares, direction
  alone is weak (P = 5/16 under a coin flip, and the cells share training
  sets). The paired intervals of dL_O all contain 0: +0.109 [−0.098, +0.315],
  −0.003 [−0.424, +0.417], −0.260 [−0.644, +0.123], −0.247 [−0.509, +0.016]
  (mean, 95%).
- **H: ρ_H = 1.088, inside (1/1.10, 1.10).** No directional prediction was
  made for H, so neither level applies. Medians −0.008, −0.028, −0.046, −0.161;
  paired intervals −0.075 [−0.239, +0.089], −0.023 [−0.238, +0.192], −0.061
  [−0.247, +0.126], −0.128 [−0.336, +0.079], all containing 0.

**Background prediction: held.** C-oracle does not reach level 3 against B; it
reaches level 2.

**The C₀ ablations (decision range).** C₀-oracle and C₀-logP train on the 300
pool points alone, B on 600, so the comparison with B mixes the budget with the
training distribution. C₀-oracle − B is above 0 in 4 of 4 cells with Welch
intervals excluding 0 (+0.038 to +0.062), and so is C₀-logP − B (+0.035 to
+0.062). The paired intervals C-oracle − C₀-oracle (−0.044 to −0.076) and
C-logP − C₀-logP (−0.043 to −0.058) are entirely below 0 in 4 of 4 cells.
Neither pool design replaces the seed.

## Revision (declared in 1e75b97, tag `revision-declared`, 2026-11-18)

### Deviation 1 (2026-11-18, before any pool-backed revision set was drawn or fitted): capacity of C-seqF

§4 of `revision-declaration.md` draws C-seqF and C-seqP from sub-pool *r*, three batches of
100, with no rule for a bin that runs out of points. Building the designs, before anything
was written or fitted and with only pool counts read, showed that C-seqF's third batch asks
for more points in the lowest selection bin, (0.500, 0.590] on predicted FOS, than its
sub-pool has left:

- r1: 9 requested, 6 left, a shortfall of 3;
- r3: 13 requested, 1 left, a shortfall of 12.

No other batch of C-seqF, no batch of C-seqP and no D set is short. With the screener frozen,
the deficit in the lowest bin persisted from batch to batch, and the predicted-bin rows of the
4,400-row sub-pool were used up; C-seq draws a new 200,000-candidate pool for each batch and
cannot run out.

**Rule adopted (the user's decision, 2026-11-18):** a short bin keeps every point it has, and
the shortfall is water-filled, as the deficit is, over the bins that still have points, so
every batch keeps 100 points and every arm 300. The rule is applied to C-seqF and C-seqP alike
(`revision_pool.capped_allocation`); it fires only in the two batches above, moving 15 of the
3,000 points of the two arms. The other option, keeping the short bins short, was rejected
because it would make the C-seqP − C-seqF contrast compare unequal budgets. Each batch records
its requested and its drawn allocation and its shortfall. The shortfall is reported with
stage 1 as a finding of the frozen-screener control.

The draw streams were also fixed at this point, before any draw: C-seqF and C-seqP share
theirs, so their first batches, screened by the same seed-trained RF against the same
deficit, are the same rows (`revision_pool.STREAM`).

### Clarification 1 (2026-11-18, before any arm was evaluated on the large sets): the bins of T-stratL

§6 asks for "30 points per selection bin … 15 bins" and does not name the bin type. The user
chose the 15 geometric selection bins of the selection range, the bins on which C-log allocates,
as "selection bins" is used in §8.4 (`revision_large`, commit d822f60). The alternative, the 15
equal-width bins of the fixed T-strat, was set aside. Only the source's counts per bin were read
before the choice (geometric: 92 to 570 per bin; equal-width: 76 to 745), so no bin was short.

### Outcome (recorded 2026-11-19)

Results are in `outputs/metrics/revision.json`. The commits are:
- the density exponent, 763cc4a;
- stage 1 and the density prediction, 7d5aa4a;
- the fixed-set fits, a4089a1;
- the seed study, a9612c9;
- the robustness checks, b4878c3;
- the large test sets, d822f60;
- the bootstrap, d6435df.

§8.1, the independent retuning, and §9, the GP, have not been run yet. Their outcome will be
added here.

**§3, the density exponent** (fixed before any D point was drawn): a = 0.583 [0.534, 0.640]
pooled, 0.559 for RF and 0.607 for GBM.

**§2, the D arms against B on the decision range** (fixed test sets, Welch, r1..r5):

| Arm | Level | Gate | Cells below 0 |
|---|---|---|---|
| D-P | **2** | 2/4 | rf/natural −0.0195, rf/stratified −0.0230 |
| D-oracle | 2 | 2/4 | rf/natural −0.0224, rf/stratified −0.0220 |
| D₀-P | 0 | 4/4 | every interval above 0 |
| D₀-oracle | 0 | 3/4 | every interval above 0 |

- **Declared reading.** D-P is below level 3. Under §2, "no demonstrated gain" therefore
  extends to a target placed on the decision range, in this design. The title is not
  revisited on this ground.
- **The price, as predicted.** Against B, D-P has:
  - a lower error on [0.5, 1) in all four cells;
  - an error on [1, 2) that is unchanged, with the interval containing 0 in three cells and
    above 0 in GBM/natural;
  - a higher error on [2, 6) and ≥ 6 in all four cells.
- **D-P against C-logP** (paired): −0.0178 [−0.0330, −0.0027] for RF/natural and −0.0265
  [−0.0415, −0.0116] for RF/stratified. Both GBM intervals contain 0.

**§3, the density prediction for D-P.**

| Zone | Direction | Calibration |
|---|---|---|
| [0.5, 1) | met | met |
| [1, 2) | not met | met |
| [2, 6) | met | not met |
| < 0.5 | met | not met |
| ≥ 6 | met | not met |
| Decision range | met | met |

For [1, 2) the prediction is −0.007, essentially 0, so its sign carries little. The
calibration fails where the error rises: B's error there grows by more than the exponent
predicts. For D-oracle, the direction is met in 4 of 5 zones (not for FOS < 0.5) and the
calibration in 2 of 5.

**§4, the frozen-screener control.** Levels against B: C-seqF 1 (gate 4/4) and C-seqP 1
(gate 3/4). Paired contrasts on the decision range:
- C-seqF − C-logP: every interval contains 0. The deficit update shows no effect.
- C-seqP − C-seqF: RF/natural −0.0035 [−0.0064, −0.0006]; the other three cells contain 0.
- C-seq − C-seqP: every interval contains 0.

The shortfall is the one recorded in Deviation 1.

**§5, the seed study** (five seeds, paired):
- **Prediction met.** C-logP_s passes the relocation gate in 3/4 cells. Level 1.
- **D-P_s:** level 2 (rf/natural −0.0142 [−0.0237, −0.0047]; gbm/stratified −0.0034
  [−0.0063, −0.0005]); gate 4/4.
- **C₀-logP_s and D₀-P_s:** level 0.

**§6, the large test sets.** T-natL has 5,000 points, 1,608 of them on the decision range.
T-stratL has 450, with 254 on the decision range and none outside [0.5, 6]. The refits
reproduce the stored predictions on the fixed sets exactly, with a maximum absolute difference
of 0. Levels, fixed → large:

| Arm | Fixed → large |
|---|---|
| D-oracle | 2 → **3** |
| D-P, C-oracle | 2 → 2 |
| C-log, C-logG, C-logP, C-seq, C-seqF, C-seqP | 1 → 2 |
| Every ablation except D₀-oracle | 0 → 0 |
| D₀-oracle | 0 → 1 |

- **Where the verdicts differ.** Most differences are in the gbm/T-stratL cell, where every
  budget arm's interval lies below 0. T-stratL is uniform in log(FOS) over the selection
  range, which is the C arms' target.
- **The gate on the large sets.** It passes in 3–4/4 cells for every C arm and in 1/4 for
  D-oracle.
- **Reading.** Under §6, a verdict that differs between the fixed and the large sets is
  reported as such, with neither preferred. The fixed sets stay the primary ones. D-P does
  not reach level 3 on the large sets either.

**§7, the two-level bootstrap** (10,000 replicates, beside the Welch interval).
- **Fixed sets.** The bootstrap interval is a median 1.27 times as wide as the Welch interval
  (range 0.84 to 3.63). Both RF cells of C-oracle then contain 0. Every interval of a C arm
  contains 0 or lies above it. The two RF cells of D-P and of D-oracle stay below 0.
- **Large sets.** The bootstrap interval is a median 0.91 times as wide as the Welch interval.
  Its verdicts agree with Welch's in every cell but one, C-log's rf/T-stratL cell, which goes
  below 0.
- **Limit.** With five realizations, the realization level is coarse. This is stated with the
  interval.

**§8.2–8.6, the robustness checks** (fixed sets):
- **General.** No arm reaches level 3 under any check.
- **8.2, three features:** 4/72 verdicts differ. C-log and C-logG go to level 2. The gate
  differs in 5 cells.
- **8.3, crest entries dropped from training** (about 8% of each set): 5/72 verdicts differ.
  C-log goes to level 2. C-oracle's rf/natural cell goes from below 0 to containing 0. The
  gate differs in 4 cells.
- **8.4, reweighting.**
  - B-wC and B-wD against B: all eight intervals contain 0.
  - B-wD against D-P: above 0 in 3/4 cells, so D-P has the lower decision-range log-RMSE.
  - B-wC against C-logP: every interval contains 0.
- **8.5, size-matched ablations:** no ablation has a lower log-RMSE than A. 25 of 36 intervals
  lie above 0, and the rest contain 0.
- **8.6, alternative ranges:** every level is the same as the baseline, except D₀-P and
  D₀-oracle, which go from 0 to 1 on [2/3, 1.5).

**Correction: crest entries.** With the check's criterion, |Center_Y − H| < 0.01 H, the crest
entries are 2,523 of the 30,250 unique analyses, 8.3%. The "about 12%" in
`docs/crest-entry-check/RESULTS.md`, and the "one in eight" in the submitted paper, were
wrong. The record is left as it is, and the paper now says "one in twelve".
