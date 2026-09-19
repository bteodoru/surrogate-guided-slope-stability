# CLAUDE.md

Context for working on this repository. Read fully before changing code.

## What this project is

Research code for a journal paper in applied machine learning for geotechnical
engineering. The study is **data-centric**: for a surrogate model of slope
stability, it asks how a fixed budget of expensive physics-based analyses
should be allocated across the **response space** (Factor of Safety, FOS).

A preliminary Random Forest (the screener), trained on an initial Latin
hypercube seed, screens a large pool of candidate parameter combinations; the
remaining analysis budget is spent on candidates chosen to even out the FOS
distribution of the training set over the selection range. This is compared,
budget-fair, against continued Latin hypercube sampling.

Working thesis (reframed 2026-09-19, after the tuned-GBM and C-log results
refuted "guided beats LHS" and "allocation matters more than the learner"):
at a fixed budget, uniformizing the response space does not systematically
improve accuracy over uniform sampling of the parameter space; it relocates
accuracy along the FOS axis, following the training density actually
achieved, and the screener controls how far that departs from the intended
one.

**The model is a measuring instrument, not an object of study.** The relevant
comparison is always vertical (same learner, different sampling arm), never
horizontal (RF vs. GBM on the same arm).

## Data

Generator: **xslope 0.5.2, Bishop simplified method**. Homogeneous, drained, no
water table. Parameter domain, features, the selection range, zone edges, seeds
and grids are defined in `sgs/config.py` and nowhere else.

Features: `H, angle, phi, c, gamma, gamma_H_c` where `gamma_H_c = gamma*H/c`,
built only in `sgs/data.py:add_features`. Target: **FOS, trained on log(FOS)**.

**Selection range**: FOS ∈ [0.5, 6.0]. Always call it _selection range_, never
"near failure", "band" or "critical band" — it is the interval targeted by the
guided selection, nothing more.

**Decision range**: FOS ∈ [0.5, 2), log-symmetric around 1. Log-RMSE on it,
against B, is the primary criterion; both halves are reported. Reporting zones
are FOS < 0.5 | 0.5–1 | 1–2 | 2–6 | ≥ 6 (`config.ZONE_EDGES`). 1.3 is not an
edge: it would read as a normative threshold. Log scale is primary, FOS scale
secondary.

Arms (A, B, C, C₀, C-log, C₀-log, C-seq, C₀-seq) and set roles (seed,
lhs-extension, guided, test-natural, test-stratified, guided-log,
guided-seq-1..3) are declared in `config.ARMS` and `config.ROLES`. Every C₀
variant is an ablation, not a budget arm (`budget=None, ablation=True`). A
screener role learns from the solved roles in `SetRole.trained_on` (default:
the seed); the sequential batches list the seed and every earlier batch.

The two test sets are two deployment scenarios, not two tests of the same
thing. Both are disjoint from every other set by normalized distance;
`data.validate_sets` asserts it before every fit and evaluation.

## Non-negotiable disciplines

1. **Every C₀ variant (C₀, C₀-log, C₀-seq) appears in every analysis**,
   including robustness checks. Omitting it is selective reporting.
2. **No threshold misclassification analysis anywhere** — no false-stable
   counts, no FOS = 1.0 crossing metrics, no classification metrics. This is a
   regression study; an engineer treats FOS 0.9 and 1.1 as effectively the
   same, and a binary count penalises a 0.05 error like a 0.5 one. Zone RMSE,
   minimax and conformal coverage/width in the low-FOS zone carry the
   engineering argument instead. If you find such code, delete it.
3. **Never duplicate a design call.** Every seed lives in `config.SEEDS`, named
   by role, and is referenced nowhere else as a literal; `tests/test_seeds.py`
   enforces it. Always call `qmc.LatinHypercube(..., seed=...)`, never `rng=`:
   in SciPy 1.18 they produce different designs.
4. **No claim of model-agnosticism.** Two tuned tree ensembles support
   "consistent across two independently tuned tree-ensemble families" and
   nothing stronger.
5. **No formal conformal guarantee.** Residuals are nested-CV out-of-fold
   residuals from several fold models; test predictions come from a refit
   model. Call it _cross-conformal-style_ and report coverage empirically. A
   CV+ implementation (Barber et al. 2023) would earn a citable guarantee.
6. **No scientific parameter on the command line.** `--role` selects a config
   entry; `--workers` and `--force` are execution control.
7. **The notebook computes nothing.** It reads `outputs/` and holds narrative
   and figures; no `sklearn` import, no `fit`.

## Tuning protocol (must stay symmetric)

One fixed grid per family (`config.GRIDS`), identical for every arm;
hyper-parameters selected by K-fold CV on the arm's training data only; the
test set is never touched during selection; chosen hyper-parameters are
reported. Library defaults are **not** neutral here — `min_samples_leaf=20` is
a far stronger constraint at n=300 than at n=600.
The GBM grid was extended, using training-fold CV only, until the selected
values lay inside it or on a plateau (`max_iter`); the old-grid state is tagged
`gbm-grid-v1`. Do not narrow it back to library-default neighbourhoods. One
declared exception: on C₀ under the FOS-scale criterion, `min_samples_leaf=10`
(outside the grid) scores ~5% better in CV but only in 2 of 5 folds, below the
fold spread; the grid was not extended for it (checked on training CV only).

Conformal residuals come from a **nested** CV: `cross_val_predict` receives the
search object, not a model with parameters already chosen.

Selection criterion is RMSE on the **log target**, consistent with the trained
response and the multiplicative intervals; this is declared in Methods.
`scoring` is a configuration axis (`config.ACTIVE_SCORINGS`) so the FOS-scale
sensitivity check is a partial rerun.

The learning curve is the one exception: fixed hyper-parameters, not tuned per
point, declared as such (spec §8). RF uses library defaults (what the grid
selects on every arm anyway); GBM uses the grid point selected by CV on the seed
alone (`config.LC_PARAMS`), since the seed is shared by both curves.

## Statistics

- Wilcoxon on `d = |e₁| − |e₂|`, reporting median d, P(d<0) and the per-pair
  alternative from `config.COMPARISONS`. Phrase as _smaller absolute errors_,
  never _smaller RMSE_. Comparisons absent from the registry are refused.
- Paired comparisons run on both scales (`evaluate.SCALES`: log, FOS) and on
  three subsets (all, selection range, decision range). Only the hypothesised
  entries (C and C-log against A and against B) are one-sided; every other
  entry, all C-seq ones included, is two-sided.
- Conformal quantile is an order statistic: `k = min(ceil((n+1)(1−α)), n)`.
- Paired bootstrap for ΔRMSE covers test-sampling variability only, not seed
  or selection variability; the limitation is written into the output.

## Pipeline

`sgs generate | fit | evaluate | report` (see `README.md`). `data/results/` is
raw solver output; `repair` writes `data/final/`. Fit files store a fingerprint
of their protocol configuration and training data and fail loudly when either
changed. `evaluate` is pure and reads fits and a test set passed as arguments.
`report` reads only `outputs/metrics/`. Provenance lives in
`data/run_metadata.json`.

## Performance notes

Estimators get `n_jobs=1`; the search gets `n_jobs=-1`. The reverse
oversubscribes cores. The searches of `sgs fit` are test-independent, so
evaluating on a new test set costs seconds.

## Publishing

Dataset on Zenodo, concept DOI 10.5281/zenodo.20706202. Old files stay so DOIs
resolve; corrected data goes out as a new version under the same concept DOI
with a note on the old one and the name mapping in `data/README.md`. Where
`artifacts/fits/` lives at publication is undecided.
