# Pipeline reorganization — design

Date: 2026-09-18
Status: revised (simplified), pending implementation plan

The first version of this design (git history of this file) built a general
provenance framework: a manifest sidecar per artifact, configuration hashed by
section, a generic guarded producer shared by every stage, a validation receipt
and nine CLI stages. That is the right shape for a repository maintained for
years. It is too much for a research code base whose purpose is that a reviewer
can regenerate the published numbers. This revision keeps the scientific
guarantees and drops the framework.

## 1. Purpose

The study currently exists as one notebook that mixes sampling design, solver
runs, model fitting and evaluation, plus two standalone generation scripts and
about twenty CSV files whose names do not state their role. This design replaces
that with a small installable package, `sgs`, and four commands. A reviewer who
clones the repository and runs them in the pinned environment regenerates every
data set, fit, table and figure.

Acceptance target: determinism of the local run. The earlier Colab run was an
exploratory first iteration, and the dataset currently on Zenodo was produced
with a different generator (HYRCAN/Spencer); neither is a reference the new
pipeline must reproduce. The phase-0 environment probe
(`docs/environment-probe-2026-09.md`) recorded how the local stack differs from
Colab: the pure LHS designs reproduce to rounding, the screener-driven designs
and two GBM selections do not, because a 2·10⁻⁴ difference in the screener
cascades through hard bin edges and greedy max-min selection.

## 2. What is kept, and why

| Kept | Why |
|---|---|
| All scientific values in `sgs/config.py`, seeds named by role | The invalid arm B came from one seed written in two places |
| Stages design → solve → repair → fit → evaluate → report | Expensive and cheap work separated; evaluation on a new test set costs seconds |
| `data/results/` immutable; repair writes `data/final/` | What the solver returned stays visible next to what the analysis used |
| Fit cache per (family, arm, scoring) that fails loudly when its data or its grid changed | The 48 searches are the only work that costs hours |
| Distance-based disjointness asserted in code before fitting and evaluating | Required by the protocol; a failed assertion stops the run |
| Seeds, library versions and chosen hyper-parameters recorded | Needed in Methods and by a reviewer |
| Tests for formulas, sampling, metrics and reproducibility | These are the failures that change numbers silently |
| One test forbidding integer seed literals outside `config.py` | About 30 lines; it guards against the failure that already happened once |

## 3. What is dropped

| Dropped | Replaced by |
|---|---|
| `*.manifest.json` per artifact, `manifest.py` | One tracked `data/run_metadata.json`, plus the fingerprint stored inside each fit file |
| Configuration hashed by section | A fingerprint of exactly what a fit depends on (grid, fixed parameters, CV, seeds, scoring, data hashes) |
| Generic `stage.produce`, `--dry-run`, `--resume`, `--only-*` | Plain functions; a command skips outputs that exist and `--force` regenerates them |
| Resumable batched solver with part files | One process pool that writes the CSV at the end. A set of 300 takes under 15 minutes with two workers; an interrupted set is simply rerun |
| `sgs check` stage and `receipt.json` | `data.validate_sets()` called at the start of `fit` and `evaluate` |
| Screener artifact with its own staleness check | The screener is refitted in memory each time a screener-driven design is built (seconds, deterministic) |
| Lexical tests for vocabulary and classification metrics; notebook purity test | Rules stated in `CLAUDE.md`, enforced in review |
| Tuned learning curve for both families | Optional command, RF only, fixed hyper-parameters (§8) |

## 4. Repository layout

```
surrogate-guided-slope-stability/
├── pyproject.toml, requirements.lock, .python-version
├── CLAUDE.md                protocol only: not history, not results
├── README.md
├── sgs/
│   ├── config.py            the only source of scientific values; paths
│   ├── data.py              schema, derived feature, loading, validation, run metadata
│   ├── design.py            LHS, screener, guided max-min, stratified random
│   ├── solver.py            xslope/Bishop wrapper, parallel set runner, repair
│   ├── models.py            tuned searches, nested OOF residuals, fit cache, learning curve
│   ├── evaluate.py          metrics, Wilcoxon, bootstrap, conformal
│   ├── report.py            booktabs tables and figures
│   └── cli.py               sgs generate | fit | evaluate | report | learning-curve | all
├── data/
│   ├── designs/  results/  final/      CSV, not tracked
│   └── run_metadata.json               tracked
├── artifacts/fits/, artifacts/learning_curve/, artifacts/solver/   not tracked
├── outputs/metrics/, outputs/tables/, outputs/figures/            tracked
├── notebooks/               narrative only, reads outputs/
└── tests/
```

`data/` starts empty and everything in it is produced by `sgs generate`. The
current CSV files live in a reference directory outside the repository; the
author compares the regenerated sets with them descriptively. Nothing in the
pipeline reads that directory.

File names replace the misleading current ones; `data/README.md` records the
mapping, because the old names are those under which files were deposited on
Zenodo.

| current | new | role |
|---|---|---|
| `initial_lhs_300_bishop.csv` | `seed_lhs_300.csv` | seed, arm A |
| `results_lhs_bishop.csv` | `lhs_extension_300.csv` | LHS extension, arm B |
| `results_uniform_bishop.csv` | `guided_300.csv` | guided selection, C and C₀ |
| `results_test_bishop.csv` | `test_natural_300.csv` | natural-300 test |
| `results_test_150_bishop.csv` | `test_stratified_150.csv` | target-stratified-150 test |

## 5. Commands

No scientific value is accepted on the command line. Flags are `--role` (selects
a config entry), `--workers` and `--force`.

```
sgs generate [--role R] [--workers K] [--force]
    for each role in ROLE_ORDER (or the one given):
      design   → data/designs/<stem>.csv
      solve    → data/results/<stem>.csv      (immutable)
      repair   → data/final/<stem>.csv
sgs fit [--force]                → artifacts/fits/{family}__{arm}__{scoring}.joblib
sgs evaluate                     → outputs/metrics/{test}__{scoring}.json
sgs report                       → outputs/tables/*.tex, outputs/figures/*.pdf
sgs learning-curve [--force]     optional → artifacts/learning_curve/, outputs/metrics/learning_curve.json
sgs all [--workers K]            generate, fit, evaluate, report
```

Behaviour without `--force`: an output that exists is reused, with three
checks where reuse could silently mix states:

1. **Design.** The design is always recomputed (it is cheap and
   deterministic). If a design file exists and differs from the recomputed one,
   the command stops: the configuration or the seed data changed.
2. **Results.** `run_metadata.json` records the hash of the design each results
   file was solved from. Existing results solved from a different design stop
   the command.
3. **Fits.** Each fit file stores a fingerprint of its protocol configuration
   and the hashes of the final CSVs it was trained on. `load_fit` compares both
   with the current state and fails naming what changed; `sgs fit` refits
   missing units and refuses stale ones without `--force`.

`repair` is rerun whenever `final/` is missing; it only reads `results/`.

## 6. `config.py`

Carried over unchanged from the first version: `PARAMS`, `DOMAIN` (order
matters: pools are drawn column by column), `FEATURES`, `TARGET = "log"`,
`SOLVER` settings and template checksum, `REPAIR_VARIANTS`, `SELECTION_RANGE`,
`N_BINS`, pool sizes, per-bin counts, `MIN_DISTANCE`, `SEEDS`, `ROLES`,
`ROLE_ORDER`, `SCREENER`, `FAMILIES`, `ESTIMATOR_FIXED`, `GRIDS`, `CV_FOLDS`,
`SCORINGS`, `ACTIVE_SCORINGS`, `ARMS`, `TEST_ROLES`, `ZONE_EDGES`, `ALPHA`,
`N_BOOTSTRAP`, `BOOTSTRAP_CI`, `COMPARISONS`, learning-curve budgets and
repeats, and the path helpers.

Seeds:

```python
SEEDS = {
    "seed_lhs_design":            42,
    "lhs_extension_design":       20260917,
    "guided_stream":              42,   # pool AND selection, one stream, in that order
    "test_stratified_pool":       20260918,
    "test_stratified_selection":  20260919,
    "test_natural_design":        20260920,
    "screener_fit":               42,
    "screener_cv":                42,
    "protocol_cv_split":          42,
    "protocol_estimator":         42,
    "bootstrap_resampling":       20260921,
    "learning_curve_subsampling": 20260922,
}
```

The guided role consumes one RNG stream for pool generation and then for bin
selection; splitting it would change the draws. A test pins the order.

Roles, arms and comparisons are as in the first version: `test-stratified`
uses random selection inside each predicted-FOS bin (a test set must be
representative, not diversified); C₀ carries `budget=None, ablation=True`; the
Wilcoxon alternative is fixed per pair in `COMPARISONS`, one-sided only where
the direction was hypothesised a priori (C against B and A).

Zones `(1.0, 1.3, 3.0, 6.0)`, half-open. `N_GUIDED_POOL`, `N_TEST_POOL` and
`MIN_DISTANCE` are declared study parameters and appear in Methods.

`config.fit_fingerprint(family, scoring)` returns a SHA-256 over the family's
grid and fixed parameters, `CV_FOLDS`, the two protocol seeds, `TARGET`,
`FEATURES` and the scoring. Nothing else is hashed.

## 7. Module responsibilities

`data.py` builds `gamma_H_c` in exactly one place. It loads a role's usable
rows (excluded rows dropped, row ids `"<role>:<index in file>"`), composes arms
from roles, and returns test sets as `TestSet(name, X, y, ids)`.
`validate_sets(roles)` checks schema, row count, domain, a positive FOS and no
untreated boundary row among kept rows, no internal duplicates, and pairwise
normalized distance > `MIN_DISTANCE`; it raises on the first failure with the
role or pair named. `record(section, key, entry)` updates
`run_metadata.json`; `environment()` returns Python and library versions.

`design.py` holds the three procedures. `fit_screener(seed_frame)` fits the
fixed-parameter RF on log FOS and returns the model with its OOB and OOF
diagnostics; `build(role)` refits it when the role needs it and asserts that
the new design stays away from the designs listed in `disjoint_from`.

`solver.py` is the port of `run_analyses.py` and `run_repair.py`:
`calculate_fos`, `analysis_row`, `run_set(combos, workers)` (a `spawn` pool,
`imap` in design order, CSV written once at the end) and `repair_rows(rows)`.

`models.py`: `searcher(family, scoring)` (estimator `n_jobs=1`, search
`n_jobs=-1`), `fit_arm(family, arm, scoring) -> FitArtifact` (one search for the
final model, one `cross_val_predict` that receives the search object, raw signed
OOF residuals with fold ids), `save_fit`/`load_fit` with the fingerprint check,
`fit_all(force)`, and the learning curve.

`evaluate.py` is pure: `evaluate(fits, test, *, alpha, zones, comparisons, …)`
takes everything as arguments. Predictions carry test name and row ids, so
errors from one test set cannot be paired with observations from another.

`report.py` reads only `outputs/metrics/` and recomputes no metric. Every
results table includes C₀, labelled as the ablation; the budget table excludes
it.

## 8. Learning curve

Optional (`sgs learning-curve`), not part of `sgs all`. RF only, fitted with
fixed hyper-parameters: `ESTIMATOR_FIXED["rf"]` plus scikit-learn defaults
(`max_features=1.0`, `min_samples_leaf=1`). This is declared in Methods. It is
defensible for RF because the tuned searches select these values or nearly so
on every arm, and tuning barely moves RF. It would not be defensible for GBM:
library defaults penalise small budgets, which is the range the curve covers.

Budgets `(100, …, 600)`, ten repeats, subsample generators from
`SeedSequence(learning_curve_subsampling, spawn_key=(budget, repeat))`. Budgets
up to the seed size are shared between the two strategies and fitted once.

## 9. Corrections to the analysis

Written directly into the new code; the old implementations are not ported.

- **Wilcoxon** on `d = |e₁| − |e₂|`, reporting median d, P(d<0) and the
  registered alternative. Results are phrased as *smaller absolute errors*.
- **Conformal quantile** is an order statistic: `k = min(ceil((n+1)(1−α)), n)`.
  Residuals come from the fit file (nested, same family, same arm), which
  removes the old mismatch between an untuned calibration model and a tuned
  prediction model. The procedure is *cross-conformal-style*, coverage is
  reported empirically, no formal guarantee is claimed.
- **Paired bootstrap** of ΔRMSE, with the limitation written into the output:
  test-sampling variability only, not seed or selection variability.
- **False-stable block**: not ported.

## 10. Testing

- **Formulas and schema**: derived feature, `model_dims`, boundary flags, row
  schema, repair outcomes.
- **Sampling**: the first row of the seed LHS pinned (guards `seed=` against
  `rng=`), LHS stratification, guided and stratified designs fill every bin, the
  guided pool-first order, determinism and seed sensitivity, disjointness
  assertions.
- **Metrics**: order-statistic quantile where interpolation differs, zone
  bounds and minimax, Wilcoxon on constructed data, refusal of misaligned
  pairs, seeded bootstrap reproducibility, every arm including C₀ evaluated.
- **Reproducibility**: fit determinism on small data, `load_fit` failing on a
  changed grid and on changed training data, `generate` refusing a changed
  design.
- **Seeds**: one AST test over `sgs/*.py` except `config.py` that fails on an
  integer literal assigned to or passed as `seed`/`random_state`, or passed
  positionally to `LatinHypercube`, `default_rng` or `RandomState`.
- One slow test runs a real xslope analysis; it is excluded by default.

## 11. Implementation order

0. Environment probe — done.
1. Repository restructure, `CLAUDE.md`, `README.md`, `data/README.md`.
2. Package skeleton, `config.py`, the seed test.
3. `data.py`.
4. `design.py`.
5. `solver.py`.
6. `sgs generate`, then the real run for the five roles (~1 h of solver time).
7. `models.py` and `sgs fit`.
8. The 8 fits, 48 grid searches with the nested CV (hours); retire the environment probe.
9. `evaluate.py` and `sgs evaluate`.
10. `report.py`, `sgs report`, `sgs all`, the narrative notebook.
11. Learning curve (optional).

## 12. Open items

- `N_GUIDED_POOL`, `N_TEST_POOL`, `MIN_DISTANCE`, the log-scale selection
  criterion, the cross-conformal-style procedure and the fixed-parameter RF
  learning curve each need a sentence in Methods.
- Every published figure is regenerated; numbers in circulation are retired.
- Where `artifacts/fits/` lives at publication is undecided; nothing blocks on
  it.
