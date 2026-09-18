# Pipeline reorganization — design

Date: 2026-09-18
Status: approved, pending implementation plan

## 1. Purpose

The study currently exists as one notebook that mixes sampling design, solver
runs, model fitting and evaluation, plus two standalone generation scripts and
about twenty CSV files whose names do not state their role. The notebook is no
longer navigable, computation and narrative are interleaved, and the analysis
runs in a different environment from the data generation.

This design replaces that arrangement with an installable package, `sgs`, whose
stages are explicit, whose artifacts carry their own provenance, and whose
preconditions are checked before expensive work starts rather than after it
fails.

Scope is the whole pipeline: sampling design, solver runs, boundary repair,
fitting, evaluation and reporting.

## 2. Decisions

| Decision | Rationale |
|---|---|
| Restructure the existing repository | `work/` and `repo/` are copies at the same commit with the same remote; the published repository keeps its history and its citations |
| One CLI over a library core | A single place where stage order is visible and preconditions are enforced; the core stays importable from the narrative notebook |
| No workflow engine (Snakemake, DVC) | Six stages with near-linear dependencies; hash manifests written by hand are sufficient, a workflow engine is a second system to learn |
| No `Makefile` | Stage order lives in `sgs run-all`; timestamp invalidation is strictly weaker than content hashing |
| No scientific parameters on the command line | Two sources of truth for one value is the failure that produced an invalid arm B once already; changing a seed must be visible in git, not in a shell history that does not exist |
| No frozen reference CSVs in the repository | Reference copies live outside the repository and are checked descriptively by the author; determinism is tested by generating twice within one run |
| Full grid search retained for GBM | Procedure and implementation do not change in the same step; `scoring` and search kind remain configuration axes for a later sensitivity check |
| Python pinned at 3.14.4 | Every current CSV was generated on it, so it is empirically validated; 3.12 is a fallback only if a future dependency refuses 3.14 |

### Out of scope

Changes to the experimental design itself: the arms, the selection range, the
test sets and the tuning protocol are carried over unchanged. The one exception
is the boundary repair of section 10, phase 3, which corrects known-wrong
labels rather than changing the design.

## 3. Repository layout

```
surrogate-guided-slope-stability/
├── pyproject.toml          installable package, pinned dependencies
├── requirements.lock       pip freeze of the reference environment
├── CLAUDE.md               protocol only: not history, not results
├── README.md
├── sgs/
│   ├── config.py           the only source of scientific truth
│   ├── manifest.py         hashing, manifest read/write, staleness detection
│   ├── data.py             loading, schema contract, derived features
│   ├── checks.py           QC, disjointness, receipt
│   ├── screener.py         preliminary RF: fit once, load thereafter
│   ├── design.py           LHS and screener-driven selection
│   ├── solver.py           xslope/Bishop wrapper
│   ├── repair.py           boundary repair
│   ├── protocol.py         searcher(family, scoring), fit_arm(...)
│   ├── fitting.py          the expensive loop, checkpointed per unit
│   ├── learning_curve.py   the only analysis that refits in a loop
│   ├── evaluate.py         metrics, pure functions
│   ├── report.py           booktabs tables and figures
│   └── cli.py              subcommands
├── data/
│   ├── designs/            parameter sets, before the solver
│   ├── results/            solved sets, immutable
│   ├── final/              repaired sets, consumed by the analysis
│   └── checks/receipt.json
├── artifacts/
│   ├── screener.joblib
│   ├── fits/               {family}__{arm}__{scoring}.joblib
│   └── learning_curve/
├── outputs/                metrics, tables, figures
├── notebooks/              narrative, imports from sgs
├── tests/
└── docs/superpowers/specs/
```

CSV payloads under `data/` and model artifacts under `artifacts/` are not
tracked: they are regenerable, they are published on Zenodo, and the repository
is not a data store. Their `*.manifest.json` sidecars **are** tracked, because
they are the provenance record — a manifest states which configuration and
which inputs produced a payload, and a tracked manifest makes that record
reviewable in git without putting megabytes of CSV in the history.

`hyrcan/` leaves the working tree. The pending modifications to
`hyrcan/run_analyses.py` and `hyrcan/run_repair.py`, together with the
untracked `hyrcan_arrays_corrected.py` and `hyrcan_test_natural.py`, are
committed first as the final state of the abandoned generator; that commit is
tagged `hyrcan-final`; the directory is then deleted from `main`. History keeps
everything and the working tree stops offering old patterns to copy. `README.md`
carries a one-line pointer to the tag. `CLAUDE.md` does not mention HYRCAN at
all.

### File naming

Role-named files replace the current names, which actively mislead —
`results_uniform_bishop.csv` holds the *guided* set and `results_test_bishop.csv`
holds the *natural* test.

| current | new | role |
|---|---|---|
| `initial_lhs_300_bishop.csv` | `seed_lhs_300.csv` | seed, arm A |
| `results_lhs_bishop.csv` | `lhs_extension_300.csv` | LHS extension, arm B |
| `results_uniform_bishop.csv` | `guided_300.csv` | guided selection, C and C₀ |
| `results_test_bishop.csv` | `test_natural_300.csv` | natural-300 test |
| `results_test_150_bishop.csv` | `test_stratified_150.csv` | target-stratified-150 test |

`data/README.md` records this mapping, because the old names are the ones under
which files were deposited on Zenodo and the DOIs must stay interpretable.
`*_part1/2` files are resume fragments and are not carried over; `*_backup`,
`* copy` and `initial_lhs_300_bishop_.csv` (old schema, no `Surface_*` columns)
are residue.

## 4. Stage graph

Stages are parameterized by set role. `--role` selects an entry in `config.py`
and introduces no new value; it is the only exception to the ban on scientific
flags, and it exists so that `run-all` contains no hand-written branching.

```
design --role seed ──► solve ──► repair ──► final/seed_lhs_300.csv
                                                  │
                                    fit-screener ◄┘
                                          │
                              artifacts/screener.joblib
                                          │
        design --role guided ◄────────────┤
        design --role test-stratified ◄───┘
        design --role lhs-extension        (independent of the screener)
        design --role test-natural         (independent of the screener)
                    │
        each: solve ──► results/<role>.csv ──► repair ──► final/<role>.csv
                    │
                 check ──► data/checks/receipt.json
                    │
                  fit ──► artifacts/fits/{family}__{arm}__{scoring}.joblib
                    │
          learning-curve ──► artifacts/learning_curve/
                    │
               evaluate ──► outputs/metrics/
                    │
                 report ──► outputs/tables/, outputs/figures/
```

`data/results/` is immutable raw solver output. `repair` reads from it and
writes to `data/final/`, adding `repair_variant` and `excluded`. A stage that
rewrote its own input would make its hash ambiguous: the payload would no longer
match the manifest that recorded it, so either every later check reports stale
or the earlier manifest is rewritten and the guarantee is lost. For sets with no
boundary rows `repair` copies and records that fact, so its position in the
chain never depends on the data.

### Artifact and manifest contract

Every artifact has a `*.manifest.json` sidecar recording: content hash, hashes
of the relevant `config.py` sections, hashes of input artifacts, library
versions, Python version, and wall time.

Configuration is hashed **by section**, not globally. Sections are `domain`,
`design`, `screener`, `protocol`, `evaluation`, and each stage declares which it
depends on. A global hash would let an added GBM grid value invalidate the
datasets, and a staleness warning that fires for unrelated reasons is a warning
one learns to ignore.

A stage that finds changed inputs **fails and names what changed**. `--force` is
the only way past it.

### Imported artifacts

The existing CSV files and the existing screener were not produced by this
pipeline, so they have no manifest, while the contract above requires a sidecar
for every artifact. They receive manifests of kind `imported`, which record the
content hash and the library and interpreter versions believed to have produced
the payload, and which state that the provenance is **declared, not verified**.
Any stage consuming an imported artifact says so in its own manifest, so the
distinction propagates instead of being laundered at the first hop. Phase 9
replaces every imported manifest with a generated one; until then, an imported
manifest is what lets `check` bind a file to something at all.

### Receipt

`sgs check` is a stage, not a function called from inside other stages, so that
validation appears in the log as a step that passed. `receipt.json` has two
sections: per-set checks keyed on the hash of the file in `data/final/`, and
pairwise checks — distance-based disjointness — keyed on the sorted pair of
hashes. `fit` requires a valid entry for every set it touches and for every
relevant pair. Regenerating one set invalidates its entry and the pairs
containing it, and nothing else.

### Execution flags

Execution control only: `--dry-run`, `--force`, `--resume`, `--workers`,
`--only-family`, `--only-arm`, `--only-scoring`. The `--only-*` filters select
subsets of what configuration already declares.

## 5. `config.py`

### Seeds

One role is one complete design procedure, not a sub-step. This reconciles the
"one seed per role" discipline with reproduction fidelity: the guided selection
consumes a single RNG stream for pool generation and then for bin selection, and
splitting that into two independent seeds would change the draws.

```python
SEEDS = {
    "seed_lhs_design":            42,
    "guided_stream":              42,   # pool AND selection, one stream, in that order
    "lhs_extension_design":       20260917,
    "test_stratified_pool":       20260918,
    "test_stratified_selection":  20260919,
    "test_natural_design":        20260920,
    "bootstrap_resampling":       20260921,
    "learning_curve_subsampling": 20260922,  # base for SeedSequence.spawn
}
```

The screener's own seeds live in the screener registry below, not here: a role
names its screener, so what that name resolves to — training role, family,
hyper-parameters, seeds — must be defined in one place rather than read from
globals. Those seeds share the value 42 with `guided_stream` but are independent
of that stream, so naming them separately changes no result.

The order contract for `guided_stream` is stated in a comment **and verified**:
a test generates the role twice within one run and compares hashes, so any
future reordering of the calls fails.

`bootstrap_resampling` and `learning_curve_subsampling` are evaluation-side and
therefore affect no existing number. Learning-curve repetitions draw per-repeat
seeds from `SeedSequence(base).spawn()` keyed on `(n, repeat)`, so two
repetitions at different `n` cannot share a subsample.

### Screeners

```python
SCREENERS = {
  "primary": Screener(trained_on="seed", family="rf",
                      params={"n_estimators": 500},
                      seeds={"fit": 42, "cv": 42}),
}
```

The preliminary surrogate is fitted with fixed hyper-parameters, not by a tuned
search: it is a screening instrument for the candidate pool, not one of the
models under comparison.

### Roles

```python
ROLES = {
  "seed":            SetRole(n=300, design="lhs",               seeds=("seed_lhs_design",),
                             screener=None),
  "lhs-extension":   SetRole(n=300, design="lhs",               seeds=("lhs_extension_design",),
                             screener=None),
  "guided":          SetRole(n=300, design="guided-maxmin",     seeds=("guided_stream",),
                             screener="primary"),
  "test-natural":    SetRole(n=300, design="lhs",               seeds=("test_natural_design",),
                             screener=None),
  "test-stratified": SetRole(n=150, design="stratified-random", seeds=("test_stratified_pool",
                                                                      "test_stratified_selection"),
                             screener="primary"),
}
```

`screener` names the artifact rather than being a boolean, so that stratifying
the test set with a screener independent of the one that produced the training
selection becomes a configuration change, not a code change.

Three design procedures, not five. `test-stratified` uses random selection
within each predicted-FOS bin, not greedy max-min: a test set must be
representative of the target distribution, not diversified, and max-min would
push it toward the edges of the domain and make it the same kind of set as the
guided training data — which is the objection this separation closes. The
distinction stays visible in the code.

### Arms

```python
ARMS = {
  "A":  Arm(roles=("seed",),                  budget=300,  ablation=False),
  "B":  Arm(roles=("seed", "lhs-extension"),  budget=600,  ablation=False),
  "C":  Arm(roles=("seed", "guided"),         budget=600,  ablation=False),
  "C0": Arm(roles=("guided",),                budget=None, ablation=True),
}
```

Arms are compositions of roles, so no per-arm loading code exists. C₀ carries
`budget=None` and `ablation=True`, so the budget table in `report.py` cannot
include it. A rule written in a document is forgotten; one encoded in a data
structure is not.

### Zones

Edges `(1.0, 1.3, 3.0, 6.0)`. The justification recorded in the config comment
is that 1.3 is the customary normative threshold for permanent drained
conditions, so the 1.0–1.3 zone is the region where an error has design
consequence. Minimax is the RMSE of the worst zone.

The notebook currently carries two zone definitions — `1.0–1.5/1.5–3.0` in a
diagnostic and `1.0–1.3/1.3–3.0` in the evaluation. Only one enters the config.

### Comparisons

```python
COMPARISONS = (
    Comparison("C",  "B",  alternative="less",      hypothesised=True),
    Comparison("C",  "A",  alternative="less",      hypothesised=True),
    Comparison("C",  "C0", alternative="two-sided", hypothesised=False),
    Comparison("C0", "B",  alternative="two-sided", hypothesised=False),
)
```

The alternative is per pair, not global. The direction was hypothesised a priori
for C against B and against A; it was not for the comparisons involving C₀, and
C₀ wins on one of the test sets. A single global alternative would make the
choice post-hoc for half the pairs. `evaluate` refuses a comparison absent from
the registry, so the alternative cannot be chosen at reporting time.

### Declared study parameters

`N_TEST_POOL = 200_000` and `MIN_DISTANCE = 1e-3` move from notebook cells into
the `design` section. They become declared parameters of the study and must
appear in Methods — the pool size in particular, because it is the mechanism by
which the method finds candidates inside the band, and a reviewer will ask how
much of the result depends on it.

## 6. Module responsibilities

`data.py` builds the derived feature `gamma_H_c` in exactly one place. Two
definitions of a derived feature can diverge silently; it is the duplicated-seed
failure on another axis.

`protocol.py` exposes `searcher(family, scoring)` and
`fit_arm(family, arm, scoring) -> FitArtifact`. `fit_arm` runs one search on the
arm's data for the final model and one `cross_val_predict` that receives **the
search object**, not a model with parameters already chosen; that nesting is
what makes the OOF residuals usable for intervals.

`fitting.py` loops over (family × arm × scoring) — 48 searches on the implicit
scoring axis, 96 with the log-versus-FOS sensitivity enabled — and writes a
checkpoint immediately after each unit. None of this touches a test set, which
is why evaluation on a new test set afterwards costs seconds rather than hours.

A fit artifact holds the serialized model, the chosen hyper-parameters, the OOF
predictions and the **raw per-fold OOF residuals** with their fold assignment.
Raw means that α, the quantile convention and the zone labels stay reporting
decisions: changing α re-runs `evaluate`, not `fit`.

The target is `log(FOS)` throughout; exponentiation happens only at the
evaluation boundary. The artifact records the convention and `evaluate` reads
it rather than re-deciding it.

The estimator gets `n_jobs=1` and the search `n_jobs=-1`. The reverse
oversubscribes cores and costs several times the runtime for nothing.

`evaluate.py` is pure:

```python
def evaluate(fits: Mapping[str, FitArtifact], test: TestSet, *, alpha, zones) -> EvalResult
```

Nothing is read from module state. In the notebook, the prediction dictionary is
a module-level variable filled by the evaluation on natural-300 and again by the
one on stratified-150, and the significance-testing block then reads the actual
values from one set and the predictions from the other, with the winner
determined by cell execution order. Explicit arguments make that contamination
impossible to express.

`report.py` reads only `outputs/metrics/` and recomputes no metric, so no number
can reach a table without having passed through `evaluate`.

## 7. Corrections to the analysis

The corrections are written into the new code directly. Porting a known-wrong
implementation in order to fix it afterwards would mean writing it twice and
risking that the fix is forgotten.

**Wilcoxon.** Built explicitly on `d = |e₁| − |e₂|`, reporting median `d`,
`P(d<0)` and the per-pair alternative from the registry. Results are phrased as
*smaller absolute errors*, never *smaller RMSE*. The current implementation
computes the right statistic but takes the direction from the mean of absolute
errors, which is not a quantity the test returns.

**Conformal quantile.** An order statistic, not an interpolated `np.quantile`:
`k = min(ceil((n+1)(1−α)), n)`, then the k-th smallest value.

**Paired bootstrap** for ΔRMSE between arms, with the limitation stated in the
output: it covers test-sampling variability only, not seed or selection
variability.

**False-stable block.** Not ported. Not commented out, not behind a flag — it
does not exist in the new code. This is a regression study; a binary count
penalises a 0.05 error like a 0.5 one.

**Calibration mismatch.** The current conformal block recomputes its OOF
residuals with an **untuned** model while the test predictions come from the
tuned one, so the intervals are calibrated on a different estimator than the one
they accompany. In the new design the residuals come from the fit artifact —
nested, same family, same arm — which removes the discrepancy by construction.
The procedure stays *cross-conformal-style*, reported empirically, with no claim
of a formal guarantee.

## 8. Testing

Tests are grouped by the failure they prevent.

**Discipline.** An AST test over every module except `config.py` fails on an
integer literal assigned to or passed as a name containing `seed` or
`random_state`, **and** on an integer literal passed positionally to
`LatinHypercube`, `default_rng`, `RandomState` or `check_random_state` —
`LatinHypercube(42)` is the shortcut a hurried author writes and a name-based
rule misses. Its counterpart fails on the vocabulary forbidden by the
no-threshold-misclassification discipline: `false_stable`, `false stable`,
classification metrics imported from `sklearn.metrics`, and any comparison of a
prediction against a fixed FOS threshold.

**Data contract.** Schema, row counts, domain ranges, no missing `FOS`, no
untreated `At_Boundary`, distance-based disjointness across all pairs of sets.
These run on small fixtures in the tests and on the real data inside `check`.

**Determinism.** Each design role is generated twice within one run and the
hashes compared. There is no comparison against a stored file: reference copies
of the current data live outside the repository and are checked descriptively by
the author (n, min, max, median FOS, bin occupancy). What the tests guarantee is
that the procedure is deterministic, not that it reproduces a file produced
before the boundary repair.

**Statistical correctness.** The order-statistic quantile on a vector where
interpolation differs; Wilcoxon invariant under permutation of the test rows and
refusing misaligned pairings — mismatched lengths or identifiers, which is where
the contamination described above actually bites; seeded bootstrap
reproducibility; minimax equal to the worst zone RMSE on constructed data.

**Notebook purity.** A test parses the `.ipynb` and fails on `sklearn` imports
or `fit` calls.

## 9. Notebook and publication

The notebook lives in `notebooks/`, imports from `sgs`, reads
`outputs/metrics/`, and produces figures and narrative. It contains no
computation. For Colab, the first cell installs the package and downloads the
data from Zenodo, which makes Colab a consumer of the pipeline rather than a
second execution environment.

Where `fits/` lives at publication remains undecided; it is regenerable, and the
decision does not block this work.

## 10. Implementation phases

Each phase begins with its tests.

**Phase 0 — environment probe.** Install scikit-learn into the local
environment and check that it selects the same hyper-parameters as the earlier
Colab runs. Python 3.14.4 is pinned rather than tested: every existing CSV was
generated on it. The environment risk is in the scikit-learn stack, not in
xslope.

Three details, because the phase runs before the package exists:

- *What code runs*: the existing local code — the tuning cells lifted out of the
  notebook — not `sgs fit`, which does not exist yet.
- *On what data*: the current CSV files, before the boundary repair, since the
  point is to compare against numbers that were produced on exactly those.
- *Where the reference hyper-parameters go*: their own file,
  `docs/environment-probe-2026-09.md`, extracted from the notebook's stored
  outputs before phase 8 rewrites it. Not `CLAUDE.md`, which is stripped of
  history and results in phase 1. The file is deleted at the end of phase 5,
  once it has served its comparison.

**Phase 1 — repository restructure.** Commit the pending HYRCAN modifications,
tag `hyrcan-final`, delete the directory, move the contents of `xslope/` to
their new locations at the repository root and remove that directory together
with its stray `git init`, add `pyproject.toml`, `config.py`, `manifest.py` and the CLI
skeleton, rename the data files and write `data/README.md`.

The renamed files land directly in `data/final/`, as imported state with
imported manifests, because that is the directory the analysis consumes and the
current files are already post-repair for the one set that needed it.
`data/results/` stays empty until phase 3, and `data/designs/` until phase 4.

`CLAUDE.md` is cleaned in this phase: the results section is **deleted**, not
marked stale; the HYRCAN material goes — both the justification for the switch
and the solver-agreement figures; the seed rule stays as a rule, without the
arm-B anecdote. The file describes the protocol, not the past and not the
results.

**Phase 2 — `check`.** Per-hash receipt and disjointness assertions. It will
fail on the eleven boundary rows, which is the correct outcome.

**Phase 3 — `solve` and `repair`.** Rebuilds the chain from its start rather
than patching in place: the affected sets are re-solved into `data/results/`,
`repair` writes `data/final/`, and the imported manifests for those sets are
replaced by generated ones. The receipt turns green.

Published numbers change here. Eleven rows are boundary-truncated, and their FOS
values are therefore not true minima. Their distribution is verified, not
assumed: three in the LHS extension, four in the guided set, four in
natural-300, **none in the seed set** (whose two boundary rows were repaired
previously) and none in target-stratified-150.

That distribution is what keeps phase 4 independent of phase 3. The screener
trains on the seed set, which phase 3 does not touch, so the screener is
unchanged and the guided *design* — the parameter combinations — is unchanged
with it. What changes in the guided set is four labels, not which points were
selected.

*Corollary.* After the repair, the hyper-parameters selected by the search may
legitimately differ from the Colab ones, because the data differ. From this
point the acceptance target is no longer agreement with Colab but determinism of
the local run.

**Phase 4 — `fit-screener` and `design`.** Delivers the screener artifact and
deterministic regeneration of every design role.

**Phase 5 — `protocol` and `fitting`.** The 48 searches; artifacts with raw
residuals. Runs once, takes hours.

**Phase 6 — `evaluate`.** Metrics as JSON, with the five corrections of
section 7. Cheap: it reads fit artifacts and a test set and runs in seconds.

**Phase 7 — `learning-curve`.** Its own phase, because it is not cheap. It
refits across sizes x repetitions x arms x families, which is hours of work
comparable to phase 5, and grouping it with evaluation would misrepresent when
the budget is spent. Its cache is separate from `artifacts/fits/` so that
invalidating one does not touch the other.

**Phase 8 — `report` and notebook.** Booktabs tables, figures, README.

**Phase 9 — full regeneration.** `run-all` from scratch before submission. Every
imported manifest is replaced by a generated one; from this point the pipeline
describes its data without a declared-provenance gap.

## 11. Open items

- `N_TEST_POOL` and `MIN_DISTANCE` become declared parameters and need a
  sentence each in Methods.
- The corrected data from phase 3 shifts every published figure; the results
  section of `CLAUDE.md` is deleted in phase 1 precisely so that no stale number
  survives the interval.
