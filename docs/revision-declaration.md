# Revision analyses: declared text

Committed 2026-11-18 with the tag `revision-declared`. From that commit it is
frozen; the outcome goes into `docs/predictions-and-outcomes.md`, never here. It declares the
analyses added in answer to the first round of peer review **before any of
them is run**: before any new point is drawn from the reference pool, before any
new set is solved and before any new fit, as the git history shows. Two quantities are fixed later, in their own commits
before the analyses that use them: the density exponent (§3) and the stage-1
counts (§4.1). Both are computed from data that already exist or from training
data only.

Everything here is conditional on the fixed seed, the fixed test sets and
`LC_PARAMS`, as in the replications, except where a section says otherwise (§5
varies the seed, §6 the test set, §8.1 the hyper-parameters).

## 1. Principles

- **Additive only.** No existing set, fit, metric or declaration changes. The
  published metrics and tables (`outputs/metrics/*.json`, `outputs/tables/`) are
  reproduced byte for byte from the revision branch, and a test checks this.
  Figures are compared by their underlying data, since image files carry
  rendering metadata. New sets go under `data/revision/`, with their provenance
  in `data/revision/run_metadata.json`. `data/run_metadata.json` is not touched.
- **New metrics, new file.** Every new result goes to `outputs/metrics/revision.json`,
  with its own fingerprint. `config.REP_STRATEGIES` and the replication
  fingerprint do not change.
- **Ablations everywhere.** Every new arm has its C₀-type ablation (the added
  points without the seed, `budget=None, ablation=True`), reported in every
  analysis.
- **Unchanged rules.** No classification metrics. Seeds only in `config.SEEDS`,
  under new keys. `qmc.LatinHypercube(..., seed=...)`. xslope `--workers` ≤ 2.
- **Intervals.** Welch for arms that share no realization-varying draw; paired
  for arms that share the realization's sub-pool or points; `REP_CI = 0.95`. An
  interval containing 0 is inconclusive.
- **Disjointness.** Every new set is checked against every earlier set (the
  pool, the replications and the test sets included) by normalized distance,
  `config.MIN_DISTANCE`. A new point within that distance of an earlier one is
  dropped, not replaced, and the count is reported. The expected count is 0.
- **Cells and margins.** A *cell* is learner × test set, so there are four. *m* = ln 1.10.
  "In ≥ 3/4" means in at least three of the four cells. The relocation gate and
  the levels against B are those of `oracle-declaration.md` §9.2.

## 2. Decision-range target (D arms)

**Question.** The tested target of response-balanced sampling is log-uniform on
the selection range [0.5, 6], and it puts about 44% of the points in range
into [2, 6). Does a target placed on the decision range itself change the
verdict?

**Allocation `a_D`.** The seed's solved FOS is water-filled to 300 points over
**4** geometric bins on [0.5, 2), with edges 0.5, 0.707, 1, 1.414 and 2, binned
as the generator bins, `(lo, hi]`. The seed counts are `[6, 23, 35, 66]`, and
`a_D = [102, 85, 72, 41]`, so seed plus added holds `[108, 108, 107, 107]`.

Why four bins, not eight:
- eight bins of the original width do not fit the predicted counts of three
  sub-pools in the lowest bin;
- four bins fit every sub-pool with a margin of 9 or more, both solved and
  predicted.

This was checked on pool counts only. No error was read.

**Arms.** These are pool arms, drawn as C-oracle and C-logP are (`replicate._pool_design`).
- **D-oracle** bins on the solved FOS.
- **D-P** bins on the prediction of the seed-trained RF screener.

They use the same sub-pool *r* for realization *r* and a random draw within each
bin, on new streams. Their ablations are **D₀-oracle** and **D₀-P**. The D arms
are alternatives to C-oracle and C-logP: they are not forced to avoid their
rows, and the overlap is reported.

**Intervals.**
- Welch: (D-oracle, B), (D-P, B), (D₀-oracle, B) and (D₀-P, B).
- Paired: (D-oracle, D-P), (D-P, C-logP), (D-oracle, C-oracle), (D-oracle, D₀-oracle)
  and (D-P, D₀-P).

**Readings.**
- **Decision range.** Levels 1–3 against B, as for C-oracle (`oracle-declaration.md` §9.2). D-P is the
  realistic arm, and D-oracle the reference.
- **Zones.** For each of FOS < 0.5, [0.5, 1), [1, 2), [2, 6) and ≥ 6, the
  per-zone Welch interval against B.
- **The price.** The zones outside [0.5, 2) receive no added points, so the
  error there is expected to rise. This is a prediction (§3), not a finding to
  discover.
- **If D-P reaches level 3,** the paper states it as a result about *where* the
  target is placed:
  - targeted at the decision range, the allocation lowers the decision-range
    error, at a price outside it;
  - "no demonstrated gain" is kept for the target over the selection range only;
  - the title and the conclusions are revisited.
- **If D-P is below level 3,** "no demonstrated gain" extends to a target placed
  on the decision range, in this design.

## 3. Density model (exponent fixed before the D arms are drawn)

**Model.** Per cell, `ln E_Z = α_{Z,cell} − a · ln n_Z`, where:
- E_Z is the log-RMSE on zone Z;
- n_Z is the number of training points in Z, using the evaluation zones `[lo, hi)`;
- the slope `a` is common to all zones and cells.

**Data for the fit.** The fit is ordinary least squares. It uses only results
that already exist:
- the arms are A, B, C-log, C-seq, C-logG, C-oracle and C-logP, with their
  ablations;
- the realizations are r1..r5;
- the zones are [0.5, 1), [1, 2) and [2, 6).

FOS < 0.5 and ≥ 6 are excluded, because they have few test points. The value of
`a` is reported per learner as well as pooled, together with a 95% interval
from a bootstrap over realizations.

**Commit order.** `a` is committed, with the code that computes it, before any
D point is drawn.

**Prediction.** For each D arm, realization and zone, the predicted log ratio
against B is `−a · ln(n_Z(D) / n̄_Z(B))`, where n̄_Z(B) is B's mean count over
r1..r5. It uses the training counts of stage 1 (§4.1). The observed value it
is compared with is `ln(E_Z(D, r) / Ē_Z(B))`, with Ē the mean of B's zone
error over r1..r5. The decision-range prediction combines the zones' predicted
squared errors, weighted by the test set's share of points in each zone.

**Two readings:**
- **direction:** the sign of the predicted ratio is met in ≥ 3/4 of the cells;
- **calibration:** the observed median lies inside the prediction ± m in
  ≥ 3/4 of the cells.

The calibration reading is what makes "accuracy follows the achieved density"
predictive. It is reported for every zone, including the zones that lose
points.

**Expected direction.** It was computed from the seed and the solved B
extension only, with no error read:
- D's training counts against B's are about 3.5 times as many on [0.5, 1) and
  1.1 times as many on [1, 2);
- they are about 0.5 times as many in the other three zones.

The predictions follow from these ratios:
- [0.5, 1): lower error than B, by a large margin;
- [1, 2): about the same as B;
- the other three zones: higher error than B;
- the decision range as a whole: lower error than B.

## 4. Frozen-screener sequential control (C-seq decomposition)

C-seq differs from C-log in three ways:
- the screener is retrained between batches;
- the deficit is updated on the solved FOS of the earlier batches;
- each batch draws a new candidate pool.

**Arms.** Two pool arms, each drawing three batches of 100 from sub-pool *r*
against C-log's allocation:
- **C-seqF:** the screener stays trained on the seed alone, and the deficit is
  updated after each batch on the solved FOS of the seed and the earlier
  batches;
- **C-seqP:** the screener is also retrained on the seed and the earlier
  batches, before each batch.

Their ablations are C₀-seqF and C₀-seqP.

**Paired contrasts, per cell:**
- C-seqF − C-logP isolates the deficit update;
- C-seqP − C-seqF isolates the retraining.

Candidate-pool renewal remains confounded in C-seq − C-seqP, and this is stated.

**Prediction.** None, beyond the declared expectation that every interval is
likely to contain 0, with the half-widths of §9.5 of the oracle declaration.

### 4.1 Stage 1 for §2 and §4

Training data only, before any fit:
- the achieved counts per bin and zone;
- Σ|achieved − intended|;
- the overlap with C-oracle and C-logP.

These are committed before any fit of the new arms.

## 5. Seed study

**Seeds.** Four new LHS seeds of 300 points, each with its own continued LHS
of 300 points. There are 8 new seeds in `config.SEEDS`, and 2,400 new analyses
with xslope. Each set is disjoint from every existing set, the pool and the test
sets included, by normalized distance.

**Arms per seed s.** There is one realization per seed. The fifth seed is
seed 42, with its existing r1 sets: B r1, and C-logP r1 and D-P r1, drawn from
sub-pool 1. For each new seed the draw is from the whole pool, which is larger
than one sub-pool. The draw is random within each predicted bin, so pool size
is not expected to change the outcome; it is stated as a difference.
- **B_s:** the seed plus its LHS extension.
- **C-logP_s:** C-log's procedure with the screener and the deficit
  recomputed on seed s, drawn on predicted FOS from the **whole** reference
  pool (22,000 rows), with the pool rows within `MIN_DISTANCE` of seed s
  excluded. The seed study needs one draw per seed, not one per sub-pool, so
  it uses the whole pool.
- **D-P_s:** the same procedure with the allocation of §2, recomputed on
  seed s.

Their ablations are C₀-logP_s and D₀-P_s. The allocation for each seed is
recomputed from that seed's deficit.

**Hyper-parameters.** Fixed as `LC_PARAMS` (chosen on seed 42). This is the
declared choice, made to keep the learner fixed. §8.1 covers retuning.

**Capacity.** Each allocation is checked against the pool's counts before any
draw. A bin short of points takes every point it has; the shortfall is reported
and is not redistributed.

**Intervals.** Paired over the five seeds, (C-logP_s, B_s) and (D-P_s, B_s): a
seed's arms share that seed.

**Readings.** The same as for the replications, over seeds instead of
realizations:
- the relocation gate;
- the levels against B on the decision range.

**Prediction.** The direction of relocation for C-logP_s holds in ≥ 3/4 of the
cells, because the domain fixes the LHS distribution of the FOS, not the seed.
The size of the relocation varies with the seed.

## 6. Large test sets

**Natural set (T-natL).** A new LHS of 5,000 points, disjoint from every set
and from the pool.

**Stratified set (T-stratL).** It is drawn from a second, independent LHS of
6,000 points:
- 30 points per selection bin, 450 in all;
- drawn at random within each bin of the **solved** FOS, `(lo, hi]`, 15 bins;
- a bin with fewer than 30 points takes all of them, and this is reported.

**Cost.** 11,000 analyses with xslope.

**What is evaluated.** Every replicated arm, old and new, r1..r5, is
re-evaluated with its stored fits on T-natL and T-stratL. The fixed test sets
stay the primary ones, because the declarations above refer to them.
On the large sets, the results reported are:
- the levels against B;
- the relocation gate;
- the zone intervals.

A verdict that differs between the fixed and the large sets is reported as
such, with neither preferred.

## 7. Uncertainty: two-level bootstrap

For each contrast against B on the decision range and its halves, the bootstrap
does two things:
- resample the realizations, separately for each arm;
- resample the test points, stratified by zone and common to the two arms of the
  contrast within a replicate.

It uses 10,000 replicates and a percentile 95% interval, on new
`config.SEEDS` keys. It is reported beside the Welch interval and does not
replace the declared readings. With five realizations, the realization level
of the bootstrap is coarse, and this is stated with the interval.

## 8. Robustness checks

None of these changes a declared reading. Each is reported as "the reading
holds" or "differs", per cell, with all ablations.

1. **Independent retuning.** B, C-log and C-seq, with their ablations, r1..r5,
   each retuned by the protocol's grid search on its own training data
   (`config.GRIDS`, `CV_FOLDS`, log-rmse).
2. **Three governing variables.** The features are angle, φ and γH/c only, with
   `LC_PARAMS`, on every replicated arm.
3. **Crest entries in training.** The rows whose critical circle enters the
   crest are dropped from every training set before refitting. The criterion is
   that of `docs/crest-entry-check/check.py`.
4. **Density reweighting.** B's training sets r1..r5, refitted with sample
   weights that bring B's binned training distribution to two targets:
   - C-log's intended one, over the 15 selection bins plus the two regions
     outside;
   - D's.

   The weight is target count over achieved count per bin. A bin with a
   target but no B point cannot be weighted, and it is reported. The two models are
   **B-wC** and **B-wD**. They are compared with B (paired) and with C-logP and
   D-P (Welch).
5. **Size-matched ablations.** A (seed only, 300 points) against every
   ablation (300 points each), Welch, from fits that already exist.
6. **Alternative decision ranges.** Two ranges, both log-symmetric around 1:
   [2/3, 1.5) and [0.4, 2.5). The levels of every arm against B are reported
   on each.

## 9. A smooth learner

**Model.** A Gaussian process, `GaussianProcessRegressor` with this kernel:
```
ConstantKernel × Matérn(ν = 2.5, one length scale per feature) + WhiteKernel
```
- The inputs are the five parameters and ln(γH/c), standardized on each
  training set. γH/c spans more than three orders of magnitude, and a
  stationary kernel on its raw value would see almost all the points in a
  small part of its range. Trees are invariant to that transform; the GP is
  not.
- The target is log FOS, with `normalize_y=True`.
- The hyper-parameters are chosen by marginal likelihood on each arm's training
  data, with 5 optimizer restarts and a new `config.SEEDS` key.

This is the same procedure for every arm, so it is symmetric.

**Arms.** Every replicated arm, old and new, with its ablations, r1..r5.

**Readings.** The relocation gate and the levels, for the GP alone, on both
test sets.

**Prediction.** The relocation gate passes for C-log under the GP in at least 1
of 2 test sets. This direction is expected because the GP is also a local
interpolator in this sense, but less sharply: it shares information across the
domain through the length scales.

**Wording.** "Consistent across two tree-ensemble families and a Gaussian
process", never model-agnostic.
