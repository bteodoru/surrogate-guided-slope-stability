# The oracle on a reference pool: declared text

Copied verbatim from the design spec of 2026-09-23 (local, not tracked): §4.4 (the stopping
rule), §9 (predictions and readings) and §10 (intervals). Committed with the tag
`oracle-declared`, before any pool chunk was solved, as the git history shows. Frozen from
that commit; the stage-2 reading quotes this file. `config.py` carries a summary of it.

### 4.4 Stopping rule (declared)

Allocation `a` is C-log's: the seed's solved FOS water-filled to 300 points over the 15 log bins of the selection range. The seed is fixed, so `a` is the same in every realization: `[38, 35, 31, 28, 22, 18, 1, 12, 16, 9, 11, 12, 18, 17, 32]`.

After each completed chunk k, over the eligible rows of chunks 1..k, the rule is met when, for every sub-pool r and every bin b:

```
solved_count[r, b]    >= a[b]     (C-oracle can draw)
predicted_count[r, b] >= a[b]     (C-logP can draw; RF screener trained on the seed)
```

Binning is the generator's: `(lo, hi]`, the first bin closed (`design.bin_counts`).

- At the first chunk that meets the rule, solving stops for good. The pool is chunks 1..k.
- If the rule is not met at the cap, that is an error. The rule is not relaxed, and a new decision is needed.
- The rule reads only pool FOS counts. It reads no test set, no fit and no error, so it cannot favour an arm.
- **Expected stop.**
  - Bin 0 has p = 0.96% under uniform sampling: 23 of 2400 LHS points, with a 95% Clopper–Pearson interval of 0.61–1.43%. No LHS row is excluded.
  - About 5.8k rows per sub-pool give a **per-sub-pool** probability of ≈ 99.5% of reaching 38 points. The rule needs all five sub-pools, so the **joint** probability at that size is ≈ 97.5%.
  - Simulated on the empirical bin rates, with 200 rows per sub-pool per chunk, the stop falls at a median of 24 chunks (p95 29). At the CP lower bound for bin 0 it falls at a median of 37 (p95 44), and P(cap of 50 reached) ≈ 0.35%. "About 30" is therefore a cautious planning figure.
- **The predicted side does not bind.** On a uniform pool of 200k, the RF screener puts 1.16–1.17% of candidates in bin 0, against a true rate of 0.96%.
  - The most demanding predicted bin is bin 0: 38 / 1.17% ≈ 3.2k rows per sub-pool in expectation. Next comes bin 1: 35 / 1.52% ≈ 2.3k.
  - Both are below the solved side's bin 0, 38 / 0.96% ≈ 4.0k.
  - The predicted side is kept in the rule anyway, so that a C-logP draw can never fail.
- **At the stop,** the binding sub-pool's bin 0 may hold exactly 38 points, and C-oracle then takes all of them. The rule stops on the *joint* counts (true bin × predicted bin). Conditional on those counts at the stop, the rows are i.i.d. within each (true bin, predicted bin) cell. This is stated, not corrected; its practical effect is negligible.

## 9. Declared predictions and readings

These are declared **before the first chunk is solved**, in `config.py` beside the C-logG block. They cover r1..r5 only.

**"In this design"** means:
- the RF screener trained on the seed;
- C-log's allocation `a`;
- a random draw within each bin;
- this reference pool, solved on darwin arm64;
- the fixed seed, test sets and `LC_PARAMS`.

Every attribution below carries that qualifier.

**Terms.**
- A *cell* is learner × test (four cells).
- A *median* is over r1..r5 within a cell.
- *m* = ln 1.10.
- *Paired interval* is `evaluate.paired_interval`: a 95% t-interval over the five per-realization values, n − 1 degrees of freedom.
- "In ≥ 3/4" always means in at least 3 of the 4 cells.

### 9.1 Stage 1 — training data only, before any fit

| quantity | expectation | type |
|---|---|---|
| C-oracle: Σ\|achieved − intended\|, points < 0.5, points > 6, share staying in bin | 0, 0, 0, 100% | construction check (asserted) |
| C-logP − C-log: Σ\|achieved − intended\| | Welch 95% interval contains 0 **and** \|mean Δ\| ≤ 15 points (C-log: 76–100) | sanity check on pool size |
| `ρ_Z` = median over r of `n_Z(C-oracle) / n_Z(C-logP)`, training set = seed + selection, Z ∈ {L = [0.5, 1), H = [1, 2)} | expected ρ_L ≈ 1.27, ρ_H ≈ 1.06 (below) | **fixes the direction of §9.2 before any fit** |
| C-oracle ∩ C-logP rows, per realization | reported | descriptive |

- **Binning.** The first two rows bin as the generator does, `(lo, hi]`. `ρ` uses the evaluation zones `[lo, hi)`, because §9.2 compares it with zone errors.
- **Expected ρ, declared now.** It was estimated on existing training data only, with no test set read:
  - C-log r1..r5 stands in for C-logP;
  - for C-oracle, a random within-true-bin draw from the solved LHS rows;
  - per realization, ρ_L = 1.29, 1.22, 1.32, 1.25, 1.30 and ρ_H = 1.07, 1.05, 1.08, 1.09, 1.02.

  The oracle recovers in range the ~45 points C-log loses below 0.5, mostly into L. The binding values of ρ are the stage-1 ones, not these.
- **If the sanity check fails,** the paired contrast C-oracle vs C-logP is unaffected, since the two share the pool. Only statements comparing with C-log carry "differs also in the candidate pool".

### 9.2 Stage 2 — test: quantities and predictions

**Quantities**, as in `evaluate.relocation_reading`:
- `L(X) = ln(E_X,L / E_B,L)` and `H(X) = ln(E_X,H / E_B,H)`, where E is the log-RMSE on the zone;
- `R(X) = H(X) − L(X)`;
- `dR_O(r) = R(C-oracle, r) − R(C-logP, r)`, in which B cancels exactly;
- `dZ_O(r) = ln(E_oracle,Z / E_logP,Z)` for Z ∈ {L, H}.

Each is read as its median per cell. The per-realization values also go into a paired interval per cell.

**Relocation gate, applied to C-oracle and to C-logP separately:** median L < 0 **and** median H > 0, in ≥ 3/4.

**Density prediction** (the testable form of "follows the achieved density"), for each zone Z:
- `ρ_Z ≥ 1.10` predicts `dZ_O < 0`: the oracle is more accurate where it has more data;
- `ρ_Z ≤ 1/1.10` predicts `dZ_O > 0`;
- otherwise no directional prediction is made for Z; the values are reported.

It is read at two levels:
- **direction met:** the predicted sign in ≥ 3/4;
- **magnitude met:** additionally, `|median dZ_O| ≥ m` in ≥ 3/4.

Both levels are reported with the four medians. Direction alone is weak: under a coin flip, P = 5/16, and the cells share training sets, so they are not independent.

**Implied prediction for dR_O.** With the expected ρ (L above the margin, H inside it), density-following predicts `dR_O > 0`. This is the direction declared for §9.3.

**Levels, C-oracle against B, on the decision range.** The quantity is the per-realization difference in decision-range log-RMSE, C-oracle − B, under a Welch interval (the arms do not share realizations):
1. **point improvement:** the mean difference is below 0 in one cell;
2. **sustained improvement:** the Welch 95% interval is entirely below 0 in one cell;
3. **systematic gain:** the interval is entirely below 0 in all four cells.

The same levels are computed for C-logP.

**Background prediction:** C-oracle does **not** reach level 3. The reason: halving the distortion with the GBM screener (Σ\|achieved − intended\| 76–100 → 43–55) produced no gain against B.

### 9.3 Readings fixed in advance

`ρ` is the stage-1 value. Within groups A and B, the first matching row applies, in the order listed. Group C is read independently of A and B, and is also ordered.

**Step P, preliminary: the gate for C-logP.** It is reported first, as passes or fails, with its four median pairs (L, H). Every A and B reading below names C-logP's gate state. If the gate fails for C-logP, no reading of dR_O is described as a change *in* relocation: it is a difference between an arm that relocates and one that does not.

**Group A: the gate passes for C-oracle.**

| # | condition | reading |
|---|---|---|
| A1 | median dR_O ≥ m in ≥ 3/4 | With ρ_L ≥ 1.10, as density-following predicts: relocation is larger with known membership, where the density in L is higher. With both ρ inside the band, relocation is larger **without** a density difference; this is against density-following for this contrast, and is stated. Attribution follows §9.4: "C-logP's smaller relocation is attributable to the screener's error, in this design" only if the paired dR_O interval excludes 0 in ≥ 3/4. Otherwise "consistent with", with no attribution |
| A2 | median dR_O ≤ −m in ≥ 3/4 | Against density-following for this contrast, whose declared direction was > 0. If the paired interval excludes 0 in ≥ 3/4: "C-logP's additional relocation is attributable to the screener's error, in this design". "Follows the achieved density" is kept only for the zones whose density prediction is met |
| A3 | the paired dR_O interval lies entirely inside (−m, m) in ≥ 3/4 | Relocation is **practically unchanged** with known membership; this is the only equivalence reading. With ρ_L ≥ 1.10 it is **evidence against** density-following for this contrast: a density difference of (ρ_L − 1) in L, from stage 1, without a corresponding relocation difference. With both ρ inside the band it is consistent with the thesis. Expected to be unreachable (§9.5) |
| A4 | \|median dR_O\| < m in ≥ 3/4, and A3 does not hold | **No difference detected.** Neither "attributable" nor "not attributable". With ρ_L ≥ 1.10, the density prediction for dR is **not borne out**, and this is stated |
| A5 | otherwise (e.g. the C-logG pattern: dR +0.12, −0.51, +0.14, −0.01) | **Mixed / inconclusive** on dR_O, with the four medians and intervals shown |

**Group B: the gate fails for C-oracle.**

| # | condition | reading |
|---|---|---|
| B1 | the gate fails for C-logP too | Relocation is not detected on this pool for **either** arm. This is reported before anything else. The relocation of C-log on the 200k pool is not contradicted, but it is not reproduced here; dR_O is reported descriptively, with no relocation label |
| B2 | the paired interval of R(C-oracle) lies entirely inside (−m, m) in ≥ 3/4 | **Relocation vanishes with known membership**: the only "vanishes" reading. Attribution follows §9.4: "C-logP's relocation is attributable to the screener's error, in this design" only if the paired dR_O interval excludes 0 in ≥ 3/4 |
| B3 | median R(C-oracle) ≤ −m in ≥ 3/4 | **Relocation reverses with known membership.** The same attribution rule as B2 |
| B4 | \|median R(C-oracle)\| < m in ≥ 3/4, and B2 does not hold | **Relocation not detected with known membership.** No attribution and no equivalence |
| B5 | otherwise | **Mixed / inconclusive**: per-zone medians shown, no relocation label |

**Group C: the decision range. Ordered; C1 is checked before C2.**

| # | condition | reading |
|---|---|---|
| C1 | C-oracle at level 3 against B, **and** the paired interval C-oracle − C-logP on decision-range log-RMSE entirely below 0 in all four cells, **and** C-logP below level 3 | "C-oracle has lower decision-range log-RMSE than B in all four cells, in this design", plus "the difference in decision-range accuracy between C-oracle and C-logP is attributable to the screener's error, in this design". "No demonstrated gain" is then qualified as holding for the screened arms; the title and headline are revisited |
| C2 | C-oracle at level 3 against B, and C1 does not hold | "C-oracle has lower decision-range log-RMSE than B in all four cells, in this design", with no statement about the screener. The result costs the pool; it is not a budget result |
| C3 | C-oracle below level 3 | "No demonstrated gain" stands, now also with known bin membership, in this design |

### 9.4 Wording about the screener

- **The only permitted attribution.** "Attributable to the screener's error, in this design" is used only for a paired C-oracle vs C-logP quantity whose paired interval excludes 0 in ≥ 3/4 cells. Where it contains 0, neither "attributable" nor "not attributable" is written.
- **The interval must exclude 0 on the side of the reading.**
  - A1 needs it entirely above 0.
  - A2, B2 and B3 need it entirely below 0. B2 and B3 are reached only after step P passed, so C-logP relocates, and the attributable direction is C-oracle relocating less.
  - An interval that excludes 0 on the other side never supports the attribution.

  This rule was added on 2026-09-23, after the plan audit and before the declaration commit.
- **Equivalence.** "Practically unchanged" and similar wording need the paired interval to lie entirely inside (−m, m), as in A3. Medians alone never support it.
- **The departure is not an empirical attribution.** It is 100% misbinning by construction (§1), and is never presented as such an attribution.
- **Unchanged rules:**
  - "controls", "causes" and "explains" stay forbidden for every comparison;
  - `achieved − intended` stays net distortion, not prediction error;
  - the structural analysis is never cited as evidence about the screener;
  - C-oracle vs B is never read as a statement about the screener except through C1.
- **CLAUDE.md** is updated to this effect **after** stage 2 exists, not before. The update says that the departure's split is definitional and that its *effect* on accuracy and relocation is what the oracle measures.
- **Relation to C-logG.** The C-logG reading of 2026-09-23 used a median-only "similar" label. That rule was declared and is not revised post hoc. It did not fire: the outcome was mixed. For the oracle the equivalence reading needs the interval, as above.

### 9.5 Expected power, declared now

The analogue already solved is C-logG − C-log: the same pool and the same stream (`REP_SEED_STREAM`), a different screener, 0–4 of 300 rows shared, five realizations. The oracle's two streams are independent; the analogue is structural, not exact. Its per-realization dR gives:
- standard deviations of 0.11 (rf/natural), 0.32 (rf/stratified), 0.19 (gbm/natural) and 0.34 (gbm/stratified);
- 95% paired-interval half-widths of 0.14–0.42 at n = 5 (t = 2.78), against m = 0.095.

If dR_O varies as much, the consequences are these:
- **A3 and B2 are practically unreachable.** An interval inside (−m, m) needs a per-realization SD below about 0.07.
- **Attribution in ≥ 3/4 cells is unlikely** unless \|dR_O\| exceeds about 0.4 in at least one stratified cell: every set of three cells includes one, and their half-widths are 0.39 and 0.42.
- **The realistic outcomes are A4, A5 and B4.** "No difference detected" is therefore the **expected** reading of a null or small effect. It is declared here, before any solve, so that it cannot later read as a post hoc failure. The oracle then contributes the stage-1 ρ, the point estimates with their intervals, and the group-C levels.

Five realizations are kept on purpose, for the same resolution as every other replicated arm in the paper (B, C-log, C-seq, C-logG). Ten or twenty realizations were considered and rejected (2026-09-23), for three reasons:
- the oracle is meant to strengthen the explanation, not to carry a result the rest of the paper cannot;
- equivalence stays out of reach even at twenty realizations;
- the noisiest cells are limited by test points per zone, not by realizations.

A larger test set, such as a held-out sub-pool or chunks solved after the stop, was weighed on the same day and rejected on the same resolution argument. It would give the oracle a test no other arm is read on (§12).

## 10. Intervals

- **Paired:** (C-oracle, C-logP), (C-oracle, C0-oracle), (C-logP, C0-logP).
- **Welch:** (C-oracle, B), (C-logP, B), (C0-oracle, B), (C0-logP, B), and (C-logP, C-log), the sanity check.
- **Learner interaction:** `Interaction("C-oracle", "C-logP", method="paired")`.
- **New, for §9.2–9.3:** per cell, the paired interval over r1..r5 of `dR_O` and of `dZ_O` for Z ∈ {L, H}. Each is computed with `evaluate.paired_interval` on the per-realization values. They sit beside the medians that `relocation_reading` already reads.
- No entries in `config.COMPARISONS`: there is no r0, so there is no Wilcoxon. Every conclusion comes from the replication intervals, and an interval containing 0 is inconclusive.
