# Pipeline Reorganization Implementation Plan (simplified)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic notebook and the standalone xslope scripts with a small installable package `sgs` and four commands (`generate`, `fit`, `evaluate`, `report`) that regenerate every data set, fit, table and figure of the study deterministically.

**Architecture:** Eight modules under `sgs/`, one responsibility each. `config.py` is the only source of scientific values. Provenance is one tracked `data/run_metadata.json` plus a fingerprint stored inside each fit file. Commands reuse outputs that exist and refuse to mix states in three places: a changed design, results solved from another design, and stale fits. `--force` regenerates.

**Tech Stack:** Python 3.14.4, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, shapely 2.1.2, openpyxl 3.1.5, xslope 0.5.2, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-09-18-pipeline-reorganization-design.md` — read it before starting any task. `CLAUDE.md` (rewritten in Task 2) holds the protocol disciplines.

## Global Constraints

- Python reference environment: **3.14.4** (`.venv/bin/python`). Every command below uses `.venv/bin/...`.
- No scientific value is defined outside `sgs/config.py`: seeds, ranges, grids, pool sizes, thresholds, zone edges, α. The command line accepts only `--role` (selects a config entry), `--workers` and `--force`.
- Modules read configuration as `from sgs import config` and use `config.X` at call time, never `from sgs.config import X`; tests monkeypatch `config`.
- `data/` starts empty; every file in it is produced by `sgs generate`. CSV files under `data/` and everything under `artifacts/` are git-ignored. `data/run_metadata.json` and `outputs/` are tracked.
- `data/results/` is raw solver output and is never rewritten by `repair`; `repair` writes `data/final/`.
- Target is `log(FOS)` everywhere; exponentiation only at the evaluation boundary.
- Estimators inside a search get `n_jobs=1`; the search gets `n_jobs=-1`.
- C₀ appears in every analysis. No threshold-misclassification code anywhere (no false-stable counts, no classification metrics, no comparison of a prediction against a fixed FOS).
- Conformal quantile is an order statistic `k = min(ceil((n+1)(1-α)), n)`; call the procedure *cross-conformal-style*, never claim a guarantee.
- Wilcoxon is on `d = |e₁| − |e₂|` with the per-pair alternative from `config.COMPARISONS`; phrase results as *smaller absolute errors*.
- The FOS interval [0.5, 6.0] is the **selection range** in code and prose — never "band", "near failure" or "critical band".
- `qmc.LatinHypercube` is always called with the `seed=` keyword. In SciPy 1.18 `seed=42` and `rng=42` produce different designs.
- Reference copies of the old data live outside the repo in `REF=/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09`. Nothing in `sgs` reads from there.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## File structure

```
pyproject.toml, requirements.lock, .python-version, .gitignore
CLAUDE.md, README.md, data/README.md
sgs/__init__.py     empty docstring
sgs/config.py       scientific values, fingerprints, path helpers
sgs/data.py         schema, derived feature, loading, validate_sets, run metadata
sgs/design.py       lhs, screener, guided max-min, stratified random, build(role)
sgs/solver.py       xslope/Bishop wrapper, run_set, repair_rows, CSV row I/O
sgs/models.py       searcher, fit_arm, fit cache, fit_all, learning curve
sgs/evaluate.py     metric primitives, paired comparisons, evaluate(), run()
sgs/report.py       booktabs tables and figures from outputs/metrics
sgs/cli.py          generate() and the sgs subcommands
tests/conftest.py   `root` and `tiny_protocol` fixtures
tests/helpers.py    synthetic data, stub models and fits
tests/test_*.py     one file per module, plus test_seeds.py
notebooks/surrogate_guided_sampling.ipynb   narrative only
```

---

## Task 1: Probe the local scikit-learn environment against the Colab results — DONE

Completed in commit `91191b1` on branch `pipeline-reorganization`. It added `tools/probe_environment.py` and `docs/environment-probe-2026-09.md`, and copied `xslope/` and the published notebook to `$REF`.

Outcome recorded in the probe document: the two pure LHS designs reproduce to rounding (max |Δparam| 5e-7 and 5e-11); the screener-driven designs (`guided`, `test-stratified`) and the GBM B/C `learning_rate` selections differ from Colab. The screener OOB R² is 0.9565 locally against 0.9567 on Colab, and that difference cascades through hard bin edges and greedy max-min. The author ruled that this does not block: the Colab run was exploratory and the acceptance target is determinism of the local run. The probe files are deleted in Task 10.

---

## Task 2: Restructure the tree, rewrite CLAUDE.md and README, add data/README.md

**Files:**
- Delete: `xslope/` (untracked), `surrogate_guided_sampling.ipynb` (tracked)
- Create: `data/README.md`, `.python-version`, `artifacts/solver/xslope_simple1_v0.5.2.xlsx` (untracked cache)
- Modify: `.gitignore`, `README.md`
- Create (currently untracked, rewritten): `CLAUDE.md`

**Interfaces:**
- Produces: the directory layout of spec §4 and the ignore rules later tasks rely on.

- [ ] **Step 1: Confirm the reference copy is complete before deleting anything**

```bash
REF=/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09
diff <(cd xslope && ls | grep -v __pycache__ | sort) <(ls "$REF" | grep -v published | sort)
git tag -l hyrcan-final
```
Expected: no output from `diff`; `hyrcan-final` printed by `git tag`.

- [ ] **Step 2: Remove the old material and create the empty layout**

```bash
mkdir -p artifacts/solver
cp xslope/xslope_simple1_v0.5.2.xlsx artifacts/solver/
rm -rf xslope
git rm -q surrogate_guided_sampling.ipynb
mkdir -p sgs tests notebooks data outputs/metrics outputs/tables outputs/figures
echo "3.14.4" > .python-version
```
The `.xlsx` copy is only a cache. `solver.ensure_template()` (Task 6) checks it against the pinned SHA-256 and downloads it again if it is missing or wrong.

- [ ] **Step 3: Replace `.gitignore`**

```gitignore
.DS_Store
__pycache__/
*.pyc
*.egg-info/
.venv/
.pytest_cache/
.superpowers/

# Regenerable payloads. data/run_metadata.json and outputs/ stay tracked.
data/**/*.csv
artifacts/

# In-flight writes.
**/.*.tmp
```

- [ ] **Step 4: Write `data/README.md`**

```markdown
# data/

Everything in this directory is produced by `sgs generate`. CSV files are not
tracked in git; `run_metadata.json` is. For every set it records the seeds,
the SHA-256 of the design, results and final files, the repair summary, the
screener diagnostics where a screener was used, and the Python and library
versions.

| directory   | content                                              |
|-------------|------------------------------------------------------|
| `designs/`  | parameter sets, before the solver                    |
| `results/`  | raw xslope/Bishop output, never modified by repair   |
| `final/`    | boundary rows repaired; consumed by the analysis     |

## File names and the Zenodo deposit

The first Zenodo versions (concept DOI 10.5281/zenodo.20706202) used the old
names below. Two of them are misleading: `results_uniform_bishop.csv` holds
the *guided* set, and `results_test_bishop.csv` holds the *natural* test.

| old name (Zenodo ≤ previous version) | new name                   | role                        |
|--------------------------------------|----------------------------|-----------------------------|
| `initial_lhs_300_bishop.csv`         | `seed_lhs_300.csv`         | seed, arm A                 |
| `results_lhs_bishop.csv`             | `lhs_continued_300.csv`    | LHS extension, arm B        |
| `results_uniform_bishop.csv`         | `surrogate_guided_300.csv` | guided selection, C and C₀  |
| `results_test_bishop.csv`            | `test_natural_300.csv`     | natural-300 test            |
| `results_test_150_bishop.csv`        | `test_stratified_150.csv`  | target-stratified-150 test  |

Rows with a non-empty `excluded` column (`solver_error`, `unresolved`) are kept
in `final/` for transparency and dropped at load time.
```

- [ ] **Step 5: Rewrite `README.md`**

````markdown
# Surrogate-guided sampling for slope-stability surrogates

Research code for a study of how a fixed budget of physics-based slope-stability
analyses should be allocated. A preliminary random forest, trained on a Latin
hypercube seed, screens a large candidate pool; the remaining budget goes to
candidates chosen for uniform coverage of the factor-of-safety (FOS) selection
range [0.5, 6.0]. The claim is tested in a budget-fair comparison against
continued Latin hypercube sampling.

Data are generated with xslope 0.5.2 (Bishop's simplified method), homogeneous
drained slopes without a water table.

| Arm | Training data                  | Solver budget       |
|-----|--------------------------------|---------------------|
| A   | seed LHS-300                   | 300                 |
| B   | seed LHS-300 + LHS-300         | 600                 |
| C   | seed LHS-300 + guided-300      | 600                 |
| C₀  | guided-300 only                | ablation, no budget |

Two test sets represent two deployment scenarios: `test-natural` (input-space
LHS, 300) and `test-stratified` (random selection within predicted-FOS bins,
150), each disjoint from every other set.

## Running the pipeline

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/sgs generate --workers 2   # five data sets, about an hour of solver time
.venv/bin/sgs fit                    # 8 fits, 48 grid searches with the nested CV; hours
.venv/bin/sgs evaluate               # seconds
.venv/bin/sgs report                 # tables and figures
```

`sgs all --workers 2` runs the four in order. `sgs learning-curve` is optional.
A command reuses outputs that already exist and stops, naming the cause, if a
design, a results file or a fit no longer matches the current configuration or
data; `--force` regenerates. All scientific parameters live in
`sgs/config.py`; none are accepted on the command line.

## Data

Dataset: [Zenodo, concept DOI 10.5281/zenodo.20706202](https://doi.org/10.5281/zenodo.20706202).
See `data/README.md` for the mapping between file names in the deposit and in
the pipeline.

The HYRCAN generator used in an earlier version of this study is preserved at
the git tag `hyrcan-final`.
````

- [ ] **Step 6: Rewrite `CLAUDE.md`** (protocol only: no results, no HYRCAN, no arm-B anecdote)

````markdown
# CLAUDE.md

Context for working on this repository. Read fully before changing code. The
design of the pipeline is in `docs/superpowers/specs/2026-09-18-pipeline-reorganization-design.md`.

## What this project is

Research code for a journal paper in applied machine learning for geotechnical
engineering. The thesis is **data-centric**: for a surrogate model of slope
stability, how a fixed budget of expensive physics-based analyses is allocated
across the **response space** (Factor of Safety, FOS) matters more than the
choice of learner.

A preliminary Random Forest, trained on an initial Latin hypercube seed,
screens a large pool of candidate parameter combinations; the remaining
analysis budget is spent on candidates chosen for uniform coverage of the FOS
range of engineering interest. The claim is validated in a budget-fair
comparison against continued Latin hypercube sampling.

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

Arms (A, B, C, C₀) and set roles (seed, lhs-extension, guided, test-natural,
test-stratified) are declared in `config.ARMS` and `config.ROLES`. C₀ is an
ablation, not a budget arm (`budget=None, ablation=True`).

The two test sets are two deployment scenarios, not two tests of the same
thing. Both are disjoint from every other set by normalized distance;
`data.validate_sets` asserts it before every fit and evaluation.

## Non-negotiable disciplines

1. **C₀ appears in every analysis**, including robustness checks. Omitting it
   is selective reporting.
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

Conformal residuals come from a **nested** CV: `cross_val_predict` receives the
search object, not a model with parameters already chosen.

Selection criterion is RMSE on the **log target**, consistent with the trained
response and the multiplicative intervals; this is declared in Methods.
`scoring` is a configuration axis (`config.ACTIVE_SCORINGS`) so the FOS-scale
sensitivity check is a partial rerun.

The learning curve is the one exception: RF only, fixed hyper-parameters,
declared as such (spec §8).

## Statistics

- Wilcoxon on `d = |e₁| − |e₂|`, reporting median d, P(d<0) and the per-pair
  alternative from `config.COMPARISONS`. Phrase as _smaller absolute errors_,
  never _smaller RMSE_. Comparisons absent from the registry are refused.
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
````

- [ ] **Step 7: Verify and commit**

```bash
git status --short
```
Expected: `D surrogate_guided_sampling.ipynb`, modified `.gitignore` and `README.md`, untracked `CLAUDE.md`, `.python-version`, `data/README.md`; nothing under `artifacts/`.

```bash
git add -A .gitignore README.md CLAUDE.md .python-version data/README.md surrogate_guided_sampling.ipynb
git commit -m "Restructure the repository around the sgs pipeline

The standalone xslope scripts, the old CSV files and the notebooks move to an
external reference directory; data/ starts empty. CLAUDE.md now describes the
protocol only.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Package skeleton, `config.py` and the seed test

**Files:**
- Create: `pyproject.toml`, `requirements.lock`, `sgs/__init__.py`, `sgs/config.py`
- Create: `tests/__init__.py` (empty), `tests/conftest.py`, `tests/test_config.py`, `tests/test_seeds.py`

**Interfaces:**
- Produces (used by every later task): every name in `config.py` below. In particular `config.ROOT` (monkeypatched by tests), `config.SCREENER` (dict), `SetRole(n, design, seeds, stem, screener=False, disjoint_from=())`, `Arm(roles, budget, ablation)`, `Comparison(first, second, alternative, hypothesised)`, `fit_fingerprint(family, scoring) -> str`, `learning_curve_fingerprint() -> str`, and the path helpers `design_path(role)`, `results_path(role)`, `final_path(role)`, `metadata_path()`, `solver_template_path()`, `fit_path(family, arm, scoring)`, `learning_curve_path(key, budget)`, `metrics_path(test, scoring)`, `learning_curve_metrics_path()`, `tables_dir()`, `figures_dir()`.
- `tests/conftest.py` produces the `root` fixture.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "sgs"
version = "0.1.0"
description = "Surrogate-guided sampling for slope-stability surrogate models"
readme = "README.md"
license = "MIT"
# The reference environment is Python 3.14.4 (.python-version, requirements.lock);
# the floor is looser so the Colab notebook can install the package.
requires-python = ">=3.12"
dependencies = [
    "numpy==2.5.3",
    "pandas==3.0.5",
    "scipy==1.18.1",
    "scikit-learn==1.9.1",
    "joblib==1.6.0",
    "matplotlib==3.11.2",
    "shapely==2.1.2",
    "openpyxl==3.1.5",
    "xslope==0.5.2",
]

[project.optional-dependencies]
dev = ["pytest==9.1.1"]

[project.scripts]
sgs = "sgs.cli:main"

[tool.setuptools]
packages = ["sgs"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["slow: runs the real xslope solver (seconds per case)"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Install editable and freeze**

```bash
.venv/bin/pip install --no-deps -e .
.venv/bin/pip freeze --exclude-editable > requirements.lock
grep -E "^(numpy|pandas|scipy|scikit-learn|joblib|xslope|pytest)==" requirements.lock
```
Expected: the seven pinned versions listed in Tech Stack.

- [ ] **Step 3: Write `sgs/__init__.py`**

```python
"""Surrogate-guided sampling for slope-stability surrogates."""
```

- [ ] **Step 4: Write `tests/__init__.py` (empty) and `tests/conftest.py`**

```python
import pytest

from sgs import config


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Redirect every pipeline path into a temporary directory."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path
```

- [ ] **Step 5: Write the failing tests `tests/test_config.py`**

```python
import dataclasses

import pytest

from sgs import config


def test_domain_order_matches_params():
    # Pools are drawn one column at a time in this order.
    assert tuple(config.DOMAIN) == config.PARAMS


def test_features_end_with_the_derived_feature():
    assert config.FEATURES[:-1] == ("H", "angle", "phi", "c", "gamma")
    assert config.FEATURES[-1] == "gamma_H_c"


def test_every_role_seed_is_declared():
    for role in config.ROLES.values():
        for name in role.seeds:
            assert name in config.SEEDS


def test_every_declared_seed_has_a_user():
    used = {s for role in config.ROLES.values() for s in role.seeds}
    used |= {"screener_fit", "screener_cv", "protocol_cv_split", "protocol_estimator",
             "bootstrap_resampling", "learning_curve_subsampling"}
    assert used == set(config.SEEDS)


def test_role_order_covers_roles_and_respects_dependencies():
    assert set(config.ROLE_ORDER) == set(config.ROLES)
    position = {role: i for i, role in enumerate(config.ROLE_ORDER)}
    for role, spec in config.ROLES.items():
        for earlier in spec.disjoint_from:
            assert position[earlier] < position[role]
        if spec.screener:
            assert position[config.SCREENER["trained_on"]] < position[role]


def test_design_sizes_match_role_sizes():
    assert config.ROLES["guided"].n == config.N_BINS * config.GUIDED_PER_BIN
    assert config.ROLES["test-stratified"].n == config.N_BINS * config.TEST_PER_BIN
    assert config.GUIDED_DRAWN_PER_BIN >= config.GUIDED_PER_BIN


def test_arms_are_compositions_of_training_roles():
    for arm in config.ARMS.values():
        assert set(arm.roles) <= set(config.ROLES)
        assert not set(arm.roles) & set(config.TEST_ROLES)


def test_ablation_arms_carry_no_budget():
    for arm in config.ARMS.values():
        assert (arm.budget is None) == arm.ablation
    assert config.ARMS["C0"].ablation


def test_comparisons_are_registered_consistently():
    for comp in config.COMPARISONS:
        assert comp.first in config.ARMS and comp.second in config.ARMS
        assert comp.alternative in ("less", "greater", "two-sided")
        assert comp.hypothesised == (comp.alternative != "two-sided")
    assert any("C0" in (c.first, c.second) for c in config.COMPARISONS)


def test_scorings_and_test_roles_are_declared():
    assert set(config.ACTIVE_SCORINGS) <= set(config.SCORINGS)
    assert set(config.TEST_ROLES) <= set(config.ROLES)


def test_zone_edges_are_increasing():
    edges = config.ZONE_EDGES
    assert list(edges) == sorted(edges) and len(set(edges)) == len(edges)


def test_learning_curve_strategies_extend_the_base_role():
    assert config.LC_BASE_ROLE in config.ROLES
    assert set(config.LC_STRATEGIES.values()) <= set(config.ROLES)


def test_fit_fingerprint_is_stable_and_per_family(monkeypatch):
    rf, gbm = config.fit_fingerprint("rf", "log-rmse"), config.fit_fingerprint("gbm", "log-rmse")
    assert len(rf) == 64 and rf == config.fit_fingerprint("rf", "log-rmse")
    assert rf != config.fit_fingerprint("rf", "fos-rmse")
    monkeypatch.setitem(config.GRIDS["gbm"], "max_iter", [300, 600, 900])
    assert config.fit_fingerprint("gbm", "log-rmse") != gbm
    assert config.fit_fingerprint("rf", "log-rmse") == rf


def test_evaluation_settings_do_not_enter_the_fit_fingerprint(monkeypatch):
    before = config.fit_fingerprint("rf", "log-rmse")
    monkeypatch.setattr(config, "ALPHA", 0.2)
    monkeypatch.setattr(config, "ZONE_EDGES", (1.0, 2.0))
    assert config.fit_fingerprint("rf", "log-rmse") == before


def test_learning_curve_fingerprint_follows_its_settings(monkeypatch):
    before = config.learning_curve_fingerprint()
    monkeypatch.setattr(config, "LC_REPEATS", 3)
    assert config.learning_curve_fingerprint() != before


def test_paths_follow_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert config.final_path("guided") == tmp_path / "data" / "final" / "surrogate_guided_300.csv"
    assert config.fit_path("rf", "C0", "log-rmse") == tmp_path / "artifacts" / "fits" / "rf__C0__log-rmse.joblib"
    assert config.metadata_path() == tmp_path / "data" / "run_metadata.json"


def test_roles_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ROLES["seed"].n = 1
```

- [ ] **Step 6: Write `tests/test_seeds.py`**

```python
"""No integer seed literal outside sgs/config.py."""

import ast
import re
from pathlib import Path

import pytest

import sgs

SGS_DIR = Path(sgs.__file__).parent
MODULES = sorted(p for p in SGS_DIR.glob("*.py") if p.name != "config.py")
SEED_NAME = re.compile(r"seed|random_state", re.IGNORECASE)
RNG_CALLS = {"LatinHypercube", "default_rng", "RandomState", "check_random_state"}


def _int_literal(node):
    return (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool))


def _target_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return str(node.slice.value)
    return None


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def seed_violations(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None and _int_literal(node.value):
                for target in targets:
                    name = _target_name(target)
                    if name and SEED_NAME.search(name):
                        found.append(f"line {node.lineno}: {name} = literal")
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and SEED_NAME.search(kw.arg) and _int_literal(kw.value):
                    found.append(f"line {node.lineno}: {kw.arg}=literal")
            if _call_name(node) in RNG_CALLS and any(_int_literal(a) for a in node.args):
                found.append(f"line {node.lineno}: {_call_name(node)}(literal)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs
            defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                        + node.args.defaults + node.args.kw_defaults)
            for arg, default in zip(args, defaults):
                if default is not None and SEED_NAME.search(arg.arg) and _int_literal(default):
                    found.append(f"line {node.lineno}: default {arg.arg}=literal")
    return found


@pytest.mark.parametrize("snippet", [
    "seed = 42",
    "RANDOM_STATE: int = 0",
    "f(random_state=0)",
    "qmc.LatinHypercube(42)",
    "np.random.default_rng(7)",
    "def f(seed=1): pass",
    "cfg['seed'] = 3",
])
def test_checker_catches(snippet):
    assert seed_violations(snippet)


@pytest.mark.parametrize("snippet", [
    "seed = config.SEEDS['seed_lhs_design']",
    "f(random_state=config.SEEDS['protocol_estimator'], n_jobs=1)",
    "qmc.LatinHypercube(d=len(config.PARAMS), seed=seed)",
    "np.random.default_rng(config.SEEDS['guided_stream'])",
    "n_seed = len(frame)",
])
def test_checker_allows(snippet):
    assert not seed_violations(snippet)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_seed_literal_outside_config(path):
    assert seed_violations(path.read_text()) == []
```

- [ ] **Step 7: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: collection error `ImportError: cannot import name 'config' from 'sgs'`.

- [ ] **Step 8: Write `sgs/config.py`**

```python
"""The only source of scientific values for the study.

Every seed, range, grid and threshold is defined here, named by role, and
referenced from nowhere else as a literal. Modules read this file as
``from sgs import config`` and use ``config.X`` at call time, so tests can
redirect ``ROOT`` and shrink sizes with monkeypatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --- the physical problem ---------------------------------------------------------

PARAMS = ("angle", "phi", "c", "gamma", "H")

# Order matters: candidate pools are drawn one column at a time in this order,
# so reordering the keys changes every screener-driven design.
DOMAIN = {
    "angle": (10.0, 65.0),   # degrees
    "phi": (5.0, 40.0),      # degrees
    "c": (1.0, 75.0),        # kPa
    "gamma": (14.0, 22.0),   # kN/m3
    "H": (1.0, 25.0),        # m
}

FEATURES = ("H", "angle", "phi", "c", "gamma", "gamma_H_c")

# Models are trained on log(FOS); exponentiation happens only at evaluation.
TARGET = "log"


@dataclass(frozen=True)
class SolverSettings:
    package_version: str = "0.5.2"
    method: str = "bishop"
    num_slices: int = 30
    # xslope's starting strategy for the circular search: a coarse global
    # sweep before adaptive refinement. Not a random seed.
    search_start: str = "grid"
    search_tol: float = 0.01
    search_fs_tol: float = 0.0005
    search_max_iter: int = 50
    search_min_grid_frac: float = 0.03
    # Adaptive model extent, with L = H / tan(angle):
    #   extension = max(extension_h_factor * H, extension_l_factor * L)
    #   depth     = max(H, depth_l_factor * L)
    extension_h_factor: float = 1.5
    extension_l_factor: float = 0.5
    depth_l_factor: float = 0.5
    # A critical surface within this fraction of H of a domain edge is
    # treated as truncated by the model extent.
    boundary_tol_h_frac: float = 0.01
    # The workbook only supplies a valid xslope input schema; every model is
    # built in memory.
    template_url: str = (
        "https://raw.githubusercontent.com/njones61/xslope/"
        "v0.5.2/docs/inputs/slope/xslope_simple1.xlsx"
    )
    template_sha256: str = "82b2c69293b2afc5f2f2f0026948a860c5032fe01eb029b64297bbb42d1c5b37"


SOLVER = SolverSettings()

# (extension factor, depth factor), tried in order until the critical surface
# clears the domain. The original extent is not a variant: the search is
# deterministic and would reproduce the same truncated surface.
REPAIR_VARIANTS = (
    (2.0, None),
    (None, 2.0),
    (1.5, 1.5),
    (2.0, 2.0),
    (3.0, 3.0),
)


# --- sampling designs -------------------------------------------------------------

SELECTION_RANGE = (0.5, 6.0)
N_BINS = 15

# Candidate pool screened for the guided training selection. Declared study
# parameter (Methods): it is how the method finds candidates in each bin.
N_GUIDED_POOL = 200_000
GUIDED_PER_BIN = 20
# Greedy max-min draws 30 candidates per bin and a random 20 are kept. The
# other 10 were the share of an earlier, non-independent test set; they are
# still drawn so the RNG stream, and with it the training selection, follows
# the published procedure, and are then discarded.
GUIDED_DRAWN_PER_BIN = 30

# Independent pool for the target-stratified test set (declared in Methods).
N_TEST_POOL = 200_000
TEST_PER_BIN = 10

# Normalized-domain distance at or below which two points count as the same
# point (declared in Methods).
MIN_DISTANCE = 1e-3

SEEDS = {
    "seed_lhs_design": 42,
    "lhs_extension_design": 20260917,
    # One stream for the guided role: the candidate pool is drawn first and
    # the bin selection continues on the same generator. Splitting it into two
    # seeds would change the draws; tests/test_design.py pins the order.
    "guided_stream": 42,
    "test_stratified_pool": 20260918,
    "test_stratified_selection": 20260919,
    "test_natural_design": 20260920,
    # The screener's seeds share the value 42 with guided_stream but are
    # independent of that stream.
    "screener_fit": 42,
    "screener_cv": 42,
    "protocol_cv_split": 42,       # inner and outer KFold shuffling
    "protocol_estimator": 42,      # random_state of every tuned estimator
    "bootstrap_resampling": 20260921,
    # Entropy of a SeedSequence keyed on (budget, repeat).
    "learning_curve_subsampling": 20260922,
}


@dataclass(frozen=True)
class SetRole:
    n: int
    design: str                           # "lhs" | "guided-maxmin" | "stratified-random"
    seeds: tuple[str, ...]                # keys of SEEDS, in the order the procedure uses them
    stem: str                             # file name without extension
    screener: bool = False                # built from screener predictions
    disjoint_from: tuple[str, ...] = ()   # roles whose designs this one must avoid


# A test set uses random selection within each predicted-FOS bin, not greedy
# max-min: it must be representative of the target distribution, not
# diversified toward the edges of the domain like the guided training set.
ROLES = {
    "seed": SetRole(n=300, design="lhs", seeds=("seed_lhs_design",), stem="seed_lhs_300"),
    "lhs-extension": SetRole(n=300, design="lhs", seeds=("lhs_extension_design",),
                             stem="lhs_continued_300", disjoint_from=("seed",)),
    "guided": SetRole(n=300, design="guided-maxmin", seeds=("guided_stream",),
                      stem="surrogate_guided_300", screener=True, disjoint_from=("seed",)),
    "test-natural": SetRole(n=300, design="lhs", seeds=("test_natural_design",),
                            stem="test_natural_300",
                            disjoint_from=("seed", "lhs-extension", "guided")),
    "test-stratified": SetRole(n=150, design="stratified-random",
                               seeds=("test_stratified_pool", "test_stratified_selection"),
                               stem="test_stratified_150", screener=True,
                               disjoint_from=("seed", "lhs-extension", "guided", "test-natural")),
}

# Generation order: each role follows the roles it depends on.
ROLE_ORDER = ("seed", "lhs-extension", "guided", "test-natural", "test-stratified")

# The preliminary surrogate. Fixed hyper-parameters, not a tuned search: it
# screens the candidate pools and is not one of the models under comparison.
SCREENER = {"trained_on": "seed", "n_estimators": 500, "cv_folds": 5}


# --- the tuned learners and the arms ------------------------------------------------

FAMILIES = ("rf", "gbm")

ESTIMATOR_FIXED = {
    "rf": {"n_estimators": 500},
    "gbm": {"early_stopping": False},
}

# One fixed grid per family, identical for every arm.
GRIDS = {
    "rf": {
        "max_features": [0.3, 0.5, 1.0],
        "min_samples_leaf": [1, 2, 5],
    },
    "gbm": {
        "learning_rate": [0.03, 0.06, 0.10],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [3, 5, 10, 20],
        "max_iter": [300, 600],
    },
}

CV_FOLDS = 5

# Selection criterion. "log-rmse" is RMSE on the trained (log) target, the
# declared choice; "fos-rmse" is the sensitivity check. Each scoring has its
# own fit files.
SCORINGS = ("log-rmse", "fos-rmse")
ACTIVE_SCORINGS = ("log-rmse",)


@dataclass(frozen=True)
class Arm:
    roles: tuple[str, ...]
    budget: int | None
    ablation: bool


ARMS = {
    "A": Arm(roles=("seed",), budget=300, ablation=False),
    "B": Arm(roles=("seed", "lhs-extension"), budget=600, ablation=False),
    "C": Arm(roles=("seed", "guided"), budget=600, ablation=False),
    "C0": Arm(roles=("guided",), budget=None, ablation=True),
}


# --- evaluation -------------------------------------------------------------------

TEST_ROLES = ("test-natural", "test-stratified")

# Zones are half-open [lo, hi): FOS < 1.0, 1.0–1.3, 1.3–3.0, 3.0–6.0, ≥ 6.0.
# 1.3 is the customary normative threshold for permanent drained conditions,
# so 1.0–1.3 is the zone where an error has design consequence. Minimax is the
# RMSE of the worst zone.
ZONE_EDGES = (1.0, 1.3, 3.0, 6.0)

ALPHA = 0.10
N_BOOTSTRAP = 10_000
BOOTSTRAP_CI = 0.95


@dataclass(frozen=True)
class Comparison:
    first: str
    second: str
    alternative: str     # scipy.stats.wilcoxon alternative on d = |e_first| - |e_second|
    hypothesised: bool   # direction fixed a priori


# The alternative is per pair: the direction was hypothesised a priori for C
# against B and A, not for the comparisons involving C0.
COMPARISONS = (
    Comparison("C", "B", alternative="less", hypothesised=True),
    Comparison("C", "A", alternative="less", hypothesised=True),
    Comparison("C", "C0", alternative="two-sided", hypothesised=False),
    Comparison("C0", "B", alternative="two-sided", hypothesised=False),
)


# --- learning curve (optional; RF with fixed hyper-parameters) ----------------------

LC_BASE_ROLE = "seed"
LC_STRATEGIES = {"lhs": "lhs-extension", "guided": "guided"}
LC_BUDGETS = (100, 200, 300, 400, 500, 600)
LC_REPEATS = 10


# --- fingerprints -------------------------------------------------------------------

def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def fit_fingerprint(family: str, scoring: str) -> str:
    """Everything a tuned fit depends on, apart from its training data."""
    return _digest({
        "family": family, "scoring": scoring,
        "grid": GRIDS[family], "fixed": ESTIMATOR_FIXED[family], "cv_folds": CV_FOLDS,
        "seeds": {k: SEEDS[k] for k in ("protocol_cv_split", "protocol_estimator")},
        "target": TARGET, "features": FEATURES,
    })


def learning_curve_fingerprint() -> str:
    return _digest({
        "base": LC_BASE_ROLE, "strategies": LC_STRATEGIES, "budgets": LC_BUDGETS,
        "repeats": LC_REPEATS, "fixed": ESTIMATOR_FIXED["rf"],
        "seeds": {k: SEEDS[k] for k in ("protocol_estimator", "learning_curve_subsampling")},
        "target": TARGET, "features": FEATURES,
    })


# --- paths (functions, so tests can redirect ROOT) -------------------------------------

def design_path(role: str) -> Path:
    return ROOT / "data" / "designs" / f"{ROLES[role].stem}.csv"


def results_path(role: str) -> Path:
    return ROOT / "data" / "results" / f"{ROLES[role].stem}.csv"


def final_path(role: str) -> Path:
    return ROOT / "data" / "final" / f"{ROLES[role].stem}.csv"


def metadata_path() -> Path:
    return ROOT / "data" / "run_metadata.json"


def solver_template_path() -> Path:
    return ROOT / "artifacts" / "solver" / f"xslope_simple1_v{SOLVER.package_version}.xlsx"


def fit_path(family: str, arm: str, scoring: str) -> Path:
    return ROOT / "artifacts" / "fits" / f"{family}__{arm}__{scoring}.joblib"


def learning_curve_path(key: str, budget: int) -> Path:
    return ROOT / "artifacts" / "learning_curve" / f"rf__{key}__n{budget}.joblib"


def metrics_path(test: str, scoring: str) -> Path:
    return ROOT / "outputs" / "metrics" / f"{test}__{scoring}.json"


def learning_curve_metrics_path() -> Path:
    return ROOT / "outputs" / "metrics" / "learning_curve.json"


def tables_dir() -> Path:
    return ROOT / "outputs" / "tables"


def figures_dir() -> Path:
    return ROOT / "outputs" / "figures"
```

- [ ] **Step 9: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass. `test_seeds.py` scans `sgs/__init__.py` only at this point.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml requirements.lock sgs/__init__.py sgs/config.py tests/__init__.py tests/conftest.py tests/test_config.py tests/test_seeds.py
git commit -m "Add the sgs package, its configuration and the seed-literal test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: `data.py` — schema, derived feature, loading, validation, run metadata

**Files:**
- Create: `sgs/data.py`, `tests/helpers.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `config`.
- Produces:
  - `RESULT_COLUMNS: list[str]`, `META_COLUMNS = ["repair_variant", "excluded"]`, `FINAL_COLUMNS`, `EXCLUSION_REASONS = ("solver_error", "unresolved")`
  - `add_features(frame) -> DataFrame` (copy, adds `gamma_H_c`) — the **only** definition of the derived feature
  - `read_table(path) -> DataFrame` (text columns as strings, empty = `""`), `is_true(series)`, `usable(frame)`, `normalize(frame) -> ndarray`
  - `load_role(role) -> DataFrame` (usable rows of `final/<role>`, `row_id = "<role>:<index in file>"`, features), `load_arm(arm)`, `TestSet(name, X, y, ids)`, `load_test(role) -> TestSet`
  - `DataError(RuntimeError)`, `set_problems(frame, role) -> list[str]` (each entry starts with the check name), `validate_sets(roles) -> None`
  - `sha256_file(path) -> str`, `environment() -> dict`, `read_metadata() -> dict`, `record(section, key, entry) -> None`
- `tests/helpers.py` produces `synthetic_fos(frame)`, `FormulaModel`, `synthetic_final(n, seed)`, `write_final(role, frame)`, `shrink_roles(monkeypatch, sizes: dict)`.

- [ ] **Step 1: Write `tests/helpers.py`**

```python
"""Synthetic fixtures. The formula is not a slope-stability model; it only
gives a smooth, positive, FOS-like response over the real domain."""

import dataclasses

import numpy as np
import pandas as pd

from sgs import config, data


def synthetic_fos(frame):
    return (0.3
            + np.tan(np.radians(frame["phi"])) / np.tan(np.radians(frame["angle"]))
            + 4.0 * frame["c"] / (frame["gamma"] * frame["H"])).to_numpy(float)


class FormulaModel:
    """Stands in for a fitted regressor: predicts log(FOS) of the formula."""

    def predict(self, X):
        return np.log(synthetic_fos(X))


_DEFAULTS = {"At_Boundary": "False", "Search_Converged": "True", "Warnings": "",
             "Error": "", "repair_variant": "", "excluded": ""}


def synthetic_final(n, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({p: rng.uniform(lo, hi, n) for p, (lo, hi) in config.DOMAIN.items()})
    frame["FOS"] = synthetic_fos(frame)
    for column in data.FINAL_COLUMNS:
        if column not in frame:
            frame[column] = _DEFAULTS.get(column, 0.0)
    return frame[data.FINAL_COLUMNS]


def write_final(role, frame):
    path = config.final_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def shrink_roles(monkeypatch, sizes):
    roles = dict(config.ROLES)
    for role, n in sizes.items():
        roles[role] = dataclasses.replace(roles[role], n=n)
    monkeypatch.setattr(config, "ROLES", roles)
```

- [ ] **Step 2: Write the failing tests `tests/test_data.py`**

```python
import json

import numpy as np
import pandas as pd
import pytest

from sgs import config, data
from tests.helpers import shrink_roles, synthetic_final, write_final


def test_schema_order():
    assert data.RESULT_COLUMNS[:6] == ["angle", "phi", "c", "gamma", "H", "FOS"]
    assert data.FINAL_COLUMNS == data.RESULT_COLUMNS + ["repair_variant", "excluded"]


def test_derived_feature():
    frame = synthetic_final(3, 0)
    out = data.add_features(frame)
    np.testing.assert_allclose(out["gamma_H_c"], frame["gamma"] * frame["H"] / frame["c"])
    assert "gamma_H_c" not in frame


def test_load_role_drops_excluded_and_keeps_file_row_ids(root):
    frame = synthetic_final(5, 1)
    frame.loc[1, "excluded"] = "unresolved"
    frame.loc[3, "FOS"] = np.nan
    frame.loc[3, "excluded"] = "solver_error"
    write_final("seed", frame)
    loaded = data.load_role("seed")
    assert list(loaded["row_id"]) == ["seed:0", "seed:2", "seed:4"]
    assert loaded.columns[-1] == "gamma_H_c"


def test_load_role_names_the_missing_step(root):
    with pytest.raises(FileNotFoundError, match="sgs generate"):
        data.load_role("seed")


def test_load_arm_concatenates_its_roles(root):
    write_final("seed", synthetic_final(4, 1))
    write_final("guided", synthetic_final(3, 2))
    arm = data.load_arm("C")
    assert len(arm) == 7
    assert arm["row_id"].iloc[-1] == "guided:2"


def test_load_test(root):
    write_final("test-natural", synthetic_final(4, 3))
    test = data.load_test("test-natural")
    assert list(test.X.columns) == list(config.FEATURES)
    assert test.ids.tolist() == [f"test-natural:{i}" for i in range(4)]
    assert test.y.shape == (4,)


def test_load_test_refuses_training_roles(root):
    with pytest.raises(ValueError):
        data.load_test("seed")


def test_is_true_reads_csv_booleans():
    assert data.is_true(pd.Series(["True", "false", "", " TRUE "])).tolist() == [True, False, False, True]


def test_normalize_maps_domain_to_unit_cube():
    corners = pd.DataFrame([{p: lo for p, (lo, _) in config.DOMAIN.items()},
                            {p: hi for p, (_, hi) in config.DOMAIN.items()}])
    np.testing.assert_allclose(data.normalize(corners), [[0] * 5, [1] * 5])


def _problem_names(frame, role="seed"):
    return {p.split(":")[0] for p in data.set_problems(frame, role)}


def test_clean_set_has_no_problems():
    assert data.set_problems(synthetic_final(300, 1), "seed") == []


@pytest.mark.parametrize("name, corrupt", [
    ("schema", lambda f: f.drop(columns="Error")),
    ("row_count", lambda f: f.iloc[:-1]),
    ("domain", lambda f: f.assign(angle=np.r_[70.0, f["angle"].to_numpy()[1:]])),
    ("exclusion_reasons", lambda f: f.assign(excluded=["odd"] + [""] * (len(f) - 1))),
    ("fos_present", lambda f: f.assign(FOS=np.r_[np.nan, f["FOS"].to_numpy()[1:]])),
    ("no_untreated_boundary", lambda f: f.assign(At_Boundary=["True"] + ["False"] * (len(f) - 1))),
    ("no_internal_duplicates", lambda f: f.assign(**{p: np.r_[f[p].iloc[1], f[p].to_numpy()[1:]]
                                                     for p in config.PARAMS})),
])
def test_each_violation_is_named(name, corrupt):
    assert name in _problem_names(corrupt(synthetic_final(300, 1)))


def test_excluded_rows_are_not_held_to_fos_or_boundary_rules():
    frame = synthetic_final(300, 1)
    frame.loc[0, ["FOS", "excluded"]] = [np.nan, "solver_error"]
    frame.loc[1, ["At_Boundary", "excluded"]] = ["True", "unresolved"]
    assert data.set_problems(frame, "seed") == []


def test_validate_sets_passes_and_catches_overlap(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 50, "lhs-extension": 50})
    write_final("seed", synthetic_final(50, 1))
    write_final("lhs-extension", synthetic_final(50, 2))
    data.validate_sets(["seed", "lhs-extension"])
    write_final("lhs-extension", synthetic_final(50, 1))
    with pytest.raises(data.DataError, match="lhs-extension/seed: not disjoint"):
        data.validate_sets(["seed", "lhs-extension"])


def test_validate_sets_names_the_failing_role(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 50})
    frame = synthetic_final(50, 1)
    frame.loc[0, "At_Boundary"] = "True"
    write_final("seed", frame)
    with pytest.raises(data.DataError, match="seed: no_untreated_boundary"):
        data.validate_sets(["seed"])


def test_record_merges_entries_and_serializes_numpy(root):
    data.record("datasets", "seed", {"n": np.int64(3), "x": np.array([1.0, 2.0])})
    data.record("fits", "rf__A__log-rmse", {"best_params": {"min_samples_leaf": 1}})
    meta = json.loads(config.metadata_path().read_text())
    assert meta["datasets"]["seed"] == {"n": 3, "x": [1.0, 2.0]}
    assert data.read_metadata()["fits"]["rf__A__log-rmse"]["best_params"] == {"min_samples_leaf": 1}


def test_environment_lists_the_pinned_packages():
    env = data.environment()
    assert env["python"] and set(env["packages"]) >= {"numpy", "scikit-learn", "xslope"}
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_data.py -q`
Expected: `ImportError: cannot import name 'data'`.

- [ ] **Step 4: Write `sgs/data.py`**

```python
"""Schema, the one definition of the derived feature, loading, validation of
the final sets, and the run metadata file."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
from dataclasses import dataclass
from importlib import metadata as _metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sgs import config

RESULT_COLUMNS = [
    *config.PARAMS,
    "FOS",
    "Surface_L", "Surface_R", "Surface_Depth",
    "Center_X", "Center_Y", "Radius",
    "At_Boundary",
    "slope_length", "extension", "depth",
    "Search_Converged", "N_Circles_Tested", "N_Valid_Surfaces", "N_Unsolved",
    "Runtime_s", "Warnings", "Error",
]
META_COLUMNS = ["repair_variant", "excluded"]
FINAL_COLUMNS = RESULT_COLUMNS + META_COLUMNS
EXCLUSION_REASONS = ("solver_error", "unresolved")

_TEXT_COLUMNS = ("At_Boundary", "Search_Converged", "Warnings", "Error", "repair_variant", "excluded")
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "shapely", "xslope")


class DataError(RuntimeError):
    """A final set, or a pair of them, violates the data contract."""


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``gamma_H_c = gamma * H / c``. Defined only here."""
    out = frame.copy()
    out["gamma_H_c"] = out["gamma"] * out["H"] / out["c"]
    return out


def read_table(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={c: "string" for c in _TEXT_COLUMNS})
    for column in _TEXT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].fillna("")
    return frame


def is_true(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower() == "true"


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[(frame["excluded"] == "") & frame["FOS"].notna()]


def normalize(frame: pd.DataFrame) -> np.ndarray:
    lo = np.array([config.DOMAIN[p][0] for p in config.PARAMS], dtype=float)
    hi = np.array([config.DOMAIN[p][1] for p in config.PARAMS], dtype=float)
    return (frame[list(config.PARAMS)].to_numpy(float) - lo) / (hi - lo)


def _final_table(role: str) -> pd.DataFrame:
    path = config.final_path(role)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs generate --role {role}`")
    return read_table(path)


def load_role(role: str) -> pd.DataFrame:
    """Usable rows of a final set, with row identifiers and features."""
    frame = _final_table(role)
    frame.insert(0, "row_id", [f"{role}:{i}" for i in range(len(frame))])
    return add_features(usable(frame)).reset_index(drop=True)


def load_arm(arm: str) -> pd.DataFrame:
    return pd.concat([load_role(role) for role in config.ARMS[arm].roles], ignore_index=True)


@dataclass(frozen=True)
class TestSet:
    __test__ = False       # not a pytest class

    name: str
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray


def load_test(role: str) -> TestSet:
    if role not in config.TEST_ROLES:
        raise ValueError(f"{role!r} is not a test role ({', '.join(config.TEST_ROLES)})")
    frame = load_role(role)
    return TestSet(name=role,
                   X=frame[list(config.FEATURES)].reset_index(drop=True),
                   y=frame["FOS"].to_numpy(float),
                   ids=frame["row_id"].to_numpy(str))


# --- validation -----------------------------------------------------------------

def set_problems(frame: pd.DataFrame, role: str) -> list[str]:
    """Violations of the data contract by one final set; empty when clean."""
    if list(frame.columns) != FINAL_COLUMNS:
        return [f"schema: columns {list(frame.columns)}"]
    problems = []
    expected = config.ROLES[role].n
    if len(frame) != expected:
        problems.append(f"row_count: {len(frame)} rows, expected {expected}")
    outside = {p: int((~frame[p].between(lo, hi)).sum()) for p, (lo, hi) in config.DOMAIN.items()}
    if any(outside.values()):
        problems.append(f"domain: rows outside the domain {outside}")
    unknown = sorted(set(frame["excluded"]) - {"", *EXCLUSION_REASONS})
    if unknown:
        problems.append(f"exclusion_reasons: unknown {unknown}")
    kept = frame["excluded"] == ""
    bad_fos = int((kept & ~(frame["FOS"] > 0)).sum())
    if bad_fos:
        problems.append(f"fos_present: {bad_fos} kept rows without a positive FOS")
    boundary = int((kept & is_true(frame["At_Boundary"])).sum())
    if boundary:
        problems.append(f"no_untreated_boundary: {boundary} kept rows at the domain boundary")
    unit = normalize(frame)
    closest = float(cKDTree(unit).query(unit, k=2)[0][:, 1].min())
    if closest <= config.MIN_DISTANCE:
        problems.append(f"no_internal_duplicates: closest pair {closest:.2e}")
    return problems


def validate_sets(roles) -> None:
    """Fail unless every set is clean and every pair is disjoint."""
    frames = {role: _final_table(role) for role in roles}
    problems = [f"{role}: {p}" for role, frame in frames.items() for p in set_problems(frame, role)]
    for a, b in itertools.combinations(sorted(frames), 2):
        closest = float(cKDTree(normalize(frames[b])).query(normalize(frames[a]), k=1)[0].min())
        if closest <= config.MIN_DISTANCE:
            problems.append(f"{a}/{b}: not disjoint (closest cross pair {closest:.2e})")
    if problems:
        raise DataError("; ".join(problems))


# --- provenance -----------------------------------------------------------------

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def _json_default(value):
    if hasattr(value, "item") and np.ndim(value) == 0:   # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):                          # numpy array
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_metadata() -> dict:
    path = config.metadata_path()
    return json.loads(path.read_text()) if path.exists() else {}


def record(section: str, key: str, entry: dict) -> None:
    """Set ``run_metadata.json[section][key] = entry`` and write the file."""
    meta = read_metadata()
    meta.setdefault(section, {})[key] = entry
    path = config.metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sgs/data.py tests/helpers.py tests/test_data.py
git commit -m "Add data loading, validation of the final sets and run metadata

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: `design.py` — LHS, screener, guided max-min, stratified random

**Files:**
- Create: `sgs/design.py`, `tests/test_design.py`

**Interfaces:**
- Consumes: `config`, `data.normalize`, `data.add_features`, `data.load_role`, `data.sha256_file`.
- Produces:
  - `DesignError(RuntimeError)`
  - `lhs(n, seed) -> (unit ndarray, frame)`, `lhs_strata_complete(unit) -> bool`
  - `nearest_distance(points, reference) -> ndarray`, `assert_disjoint(frame, references: Mapping[str, DataFrame], min_distance) -> None`
  - `screener_model(*, oob: bool) -> RandomForestRegressor`, `fit_screener(frame) -> (model, diagnostics: dict)`
  - `uniform_pool(rng, n)`, `predict_fos(model, frame)`, `greedy_maxmin(Xs, k, rng)`, `bin_edges(n_bins, selection_range)`
  - `guided_maxmin(model, *, rng, n_pool, n_bins, per_bin, drawn_per_bin, selection_range) -> DataFrame`
  - `stratified_random(model, *, pool_rng, selection_rng, n_pool, n_bins, per_bin, selection_range, references, min_distance) -> DataFrame`
  - `read_design(role) -> DataFrame`, `build(role) -> (frame, info: dict)`. `info` has `"seeds"` and, for screener roles, `"screener"` (diagnostics plus `trained_on_sha256`). Design columns: `config.PARAMS`, plus `FOS_pred` and `bin` for screener roles.

- [ ] **Step 1: Write the failing tests `tests/test_design.py`**

```python
import hashlib

import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design
from tests.helpers import FormulaModel, shrink_roles, synthetic_final, write_final


def _hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _write_design(role, frame):
    path = config.design_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# --- LHS and disjointness -------------------------------------------------------------

def test_lhs_uses_the_seed_keyword_semantics():
    # First row of the published seed design. Pins SciPy's `seed=` stream; a
    # change of keyword (`rng=`) or of SciPy's LHS draws fails here.
    _, frame = design.lhs(300, 42)
    np.testing.assert_allclose(
        frame.iloc[0].to_numpy(),
        [24.70810805776474, 33.64879751536226, 12.628212513088526,
         17.341403519225082, 4.832465812168988], rtol=0, atol=1e-12)


def test_lhs_is_stratified_and_in_domain():
    unit, frame = design.lhs(50, 7)
    assert design.lhs_strata_complete(unit)
    for p, (lo, hi) in config.DOMAIN.items():
        assert frame[p].between(lo, hi).all()


def test_strata_check_detects_a_collapsed_column():
    unit, _ = design.lhs(20, 7)
    unit[:, 2] = 0.5
    assert not design.lhs_strata_complete(unit)


def test_assert_disjoint_names_the_offending_set():
    _, a = design.lhs(20, 1)
    with pytest.raises(design.DesignError, match="'twin'"):
        design.assert_disjoint(a, {"twin": a.copy()}, 1e-3)


def test_assert_disjoint_catches_internal_duplicates():
    _, a = design.lhs(20, 1)
    doubled = pd.concat([a, a.iloc[:1]], ignore_index=True)
    with pytest.raises(design.DesignError, match="inside the set"):
        design.assert_disjoint(doubled, {}, 1e-3)


def test_build_lhs_roles(root):
    frame, info = design.build("seed")
    assert len(frame) == 300 and info == {"seeds": {"seed_lhs_design": 42}}
    assert _hash(design.build("seed")[0]) == _hash(frame)
    with pytest.raises(FileNotFoundError, match="seed_lhs_300.csv"):
        design.build("lhs-extension")
    _write_design("seed", frame)
    extension, _ = design.build("lhs-extension")
    assert design.nearest_distance(extension, frame).min() > config.MIN_DISTANCE


# --- screener -------------------------------------------------------------------------

@pytest.fixture
def small_screener(monkeypatch):
    monkeypatch.setitem(config.SCREENER, "n_estimators", 20)
    monkeypatch.setitem(config.SCREENER, "cv_folds", 3)


def test_screener_uses_named_seeds(small_screener):
    model = design.screener_model(oob=True)
    assert model.random_state == config.SEEDS["screener_fit"] and model.oob_score


def test_fit_screener_is_deterministic_and_reports_diagnostics(small_screener):
    frame = data.add_features(synthetic_final(80, 5))
    a, diagnostics = design.fit_screener(frame)
    b, _ = design.fit_screener(frame)
    X = data.add_features(synthetic_final(10, 9))[list(config.FEATURES)]
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert diagnostics["n_train"] == 80 and 0 < diagnostics["oob_r2_log"] <= 1
    assert {"oof_rmse", "oof_rmse_selection_range", "oof_spearman"} <= set(diagnostics)


# --- guided max-min ---------------------------------------------------------------------

def _guided(rng_seed=42):
    return design.guided_maxmin(FormulaModel(), rng=np.random.default_rng(rng_seed), n_pool=6000,
                                n_bins=5, per_bin=4, drawn_per_bin=6, selection_range=(0.5, 6.0))


def test_guided_fills_every_bin_with_predicted_fos_inside_it():
    frame = _guided()
    assert frame["bin"].value_counts().sort_index().tolist() == [4] * 5
    edges = design.bin_edges(5, (0.5, 6.0))
    for b, group in frame.groupby("bin"):
        assert group["FOS_pred"].gt(edges[b]).all() and group["FOS_pred"].le(edges[b + 1]).all()


def test_guided_draws_the_pool_first_from_the_stream():
    # The order contract of `guided_stream`: every selected point is a row of
    # the pool drawn first from a fresh generator with the same seed.
    frame = _guided(42)
    pool = design.uniform_pool(np.random.default_rng(42), 6000)
    merged = frame.merge(pool, on=list(config.PARAMS), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_guided_is_deterministic_and_seed_sensitive():
    assert _hash(_guided(42)) == _hash(_guided(42))
    assert _hash(_guided(42)) != _hash(_guided(43))


def test_guided_refuses_thin_bins():
    with pytest.raises(design.DesignError, match="fewer than"):
        design.guided_maxmin(FormulaModel(), rng=np.random.default_rng(0), n_pool=200, n_bins=5,
                             per_bin=4, drawn_per_bin=60, selection_range=(0.5, 6.0))


def test_greedy_maxmin_picks_spread_points():
    X = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0], [0.0, 1.0]])
    chosen = design.greedy_maxmin(X, 3, np.random.default_rng(0))
    assert len(set(chosen.tolist())) == 3


# --- stratified random ------------------------------------------------------------------

def _stratified(pool_seed=1, selection_seed=2, references=None):
    return design.stratified_random(FormulaModel(), pool_rng=np.random.default_rng(pool_seed),
                                    selection_rng=np.random.default_rng(selection_seed), n_pool=6000,
                                    n_bins=5, per_bin=3, selection_range=(0.5, 6.0),
                                    references=references or {}, min_distance=1e-3)


def test_stratified_fills_bins_with_closed_first_bin():
    frame = _stratified()
    assert frame["bin"].value_counts().sort_index().tolist() == [3] * 5
    assert frame["FOS_pred"].between(0.5, 6.0).all()


def test_stratified_uses_both_streams_independently():
    base = _hash(_stratified(1, 2))
    assert _hash(_stratified(1, 2)) == base
    assert _hash(_stratified(1, 3)) != base
    assert _hash(_stratified(9, 2)) != base


def test_stratified_filters_candidates_near_references():
    first = _stratified()
    again = _stratified(references={"previous": first})
    assert design.nearest_distance(again, first).min() > 1e-3


# --- screener roles through build() -------------------------------------------------------

@pytest.fixture
def small_designs(root, monkeypatch, small_screener):
    monkeypatch.setattr(config, "N_GUIDED_POOL", 6000)
    monkeypatch.setattr(config, "N_TEST_POOL", 6000)
    monkeypatch.setattr(config, "N_BINS", 5)
    monkeypatch.setattr(config, "GUIDED_PER_BIN", 4)
    monkeypatch.setattr(config, "GUIDED_DRAWN_PER_BIN", 6)
    monkeypatch.setattr(config, "TEST_PER_BIN", 3)
    shrink_roles(monkeypatch, {"guided": 20, "test-stratified": 15})
    seed_design, _ = design.build("seed")
    _write_design("seed", seed_design)
    write_final("seed", synthetic_final(300, 1))


def test_screener_roles_are_deterministic_and_record_the_screener(small_designs):
    guided, info = design.build("guided")
    assert _hash(design.build("guided")[0]) == _hash(guided)
    assert info["seeds"] == {"guided_stream": 42}
    assert info["screener"]["trained_on_sha256"] == data.sha256_file(config.final_path("seed"))
    for role in ("guided", "lhs-extension", "test-natural"):
        _write_design(role, design.build(role)[0])
    stratified, _ = design.build("test-stratified")
    assert len(stratified) == 15 and set(stratified.columns) == {*config.PARAMS, "FOS_pred", "bin"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_design.py -q`
Expected: `ImportError: cannot import name 'design'`.

- [ ] **Step 3: Write `sgs/design.py`**

```python
"""Sampling designs: parameter sets before the solver.

Three procedures, one per ``SetRole.design``:
  lhs                Latin hypercube over the input domain;
  guided-maxmin      screener-predicted FOS bins, greedy max-min inside a bin
                     (the guided training selection);
  stratified-random  screener-predicted FOS bins, random selection inside a
                     bin (the target-stratified test set).
The screener is refitted in memory whenever a screener role is built; the fit
takes seconds and is deterministic.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from sgs import config, data


class DesignError(RuntimeError):
    """A design violates one of its own invariants."""


# --- LHS ------------------------------------------------------------------------

def _bounds():
    lo = np.array([config.DOMAIN[p][0] for p in config.PARAMS], dtype=float)
    hi = np.array([config.DOMAIN[p][1] for p in config.PARAMS], dtype=float)
    return lo, hi


def lhs(n: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    # ``seed=``, not ``rng=``: in SciPy 1.18 the two keywords seed different
    # generators, and the published designs were drawn with ``seed=``.
    sampler = qmc.LatinHypercube(d=len(config.PARAMS), seed=seed)
    unit = sampler.random(n)
    lo, hi = _bounds()
    return unit, pd.DataFrame(qmc.scale(unit, lo, hi), columns=list(config.PARAMS))


def lhs_strata_complete(unit: np.ndarray) -> bool:
    """One point in each of the n strata along every dimension."""
    n = len(unit)
    strata = np.floor(np.clip(unit, 0.0, np.nextafter(1.0, 0.0)) * n).astype(int)
    return all(len(np.unique(strata[:, j])) == n for j in range(unit.shape[1]))


# --- disjointness -----------------------------------------------------------------

def nearest_distance(points: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    return cKDTree(data.normalize(reference)).query(data.normalize(points), k=1)[0]


def assert_disjoint(frame: pd.DataFrame, references: Mapping[str, pd.DataFrame],
                    min_distance: float) -> None:
    for name, reference in references.items():
        d = nearest_distance(frame, reference)
        if d.min() <= min_distance:
            raise DesignError(f"{int((d <= min_distance).sum())} point(s) within {min_distance} "
                              f"of {name!r} (closest {d.min():.2e})")
    unit = data.normalize(frame)
    self_d = cKDTree(unit).query(unit, k=2)[0][:, 1]
    if self_d.min() <= min_distance:
        raise DesignError(f"duplicate points inside the set (closest pair {self_d.min():.2e})")


# --- screener -----------------------------------------------------------------------

def screener_model(*, oob: bool) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=config.SCREENER["n_estimators"], oob_score=oob,
                                 random_state=config.SEEDS["screener_fit"], n_jobs=-1)


def fit_screener(frame: pd.DataFrame):
    """Fit on log(FOS) of ``frame`` (features included); return model and diagnostics."""
    X = frame[list(config.FEATURES)]
    fos = frame["FOS"].to_numpy(float)
    y = np.log(fos)
    model = screener_model(oob=True).fit(X, y)
    cv = KFold(config.SCREENER["cv_folds"], shuffle=True, random_state=config.SEEDS["screener_cv"])
    oof = np.exp(cross_val_predict(screener_model(oob=False), X, y, cv=cv))
    lo, hi = config.SELECTION_RANGE
    inside = (fos >= lo) & (fos <= hi)
    diagnostics = {
        "n_train": int(len(frame)),
        "oob_r2_log": float(model.oob_score_),
        "oof_rmse": float(np.sqrt(np.mean((oof - fos) ** 2))),
        "oof_rmse_selection_range": float(np.sqrt(np.mean((oof[inside] - fos[inside]) ** 2))),
        "oof_spearman": float(spearmanr(oof, fos).statistic),
    }
    return model, diagnostics


# --- screener-driven designs --------------------------------------------------------------

def uniform_pool(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Uniform candidates over the domain, one column at a time in PARAMS order."""
    return pd.DataFrame({p: rng.uniform(config.DOMAIN[p][0], config.DOMAIN[p][1], n)
                         for p in config.PARAMS})


def predict_fos(model, frame: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(data.add_features(frame)[list(config.FEATURES)]))


def greedy_maxmin(Xs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy max-min selection of k rows from a random start."""
    n = len(Xs)
    if k >= n:
        return np.arange(n)
    chosen = [int(rng.integers(n))]
    dmin = np.linalg.norm(Xs - Xs[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Xs - Xs[nxt], axis=1))
    return np.array(chosen)


def bin_edges(n_bins: int, selection_range) -> np.ndarray:
    lo, hi = selection_range
    return np.linspace(lo, hi, n_bins + 1)


def guided_maxmin(model, *, rng, n_pool, n_bins, per_bin, drawn_per_bin, selection_range) -> pd.DataFrame:
    """The guided training selection.

    One generator, in this order: the pool, then per bin a max-min draw of
    ``drawn_per_bin`` candidates and a permutation of which the first
    ``per_bin`` are kept. Bins are (lo, hi] on screener-predicted FOS.
    """
    params = list(config.PARAMS)
    pool = uniform_pool(rng, n_pool)
    pool["FOS_pred"] = predict_fos(model, pool)
    edges = bin_edges(n_bins, selection_range)
    lo, hi = selection_range
    inside = pool[(pool["FOS_pred"] >= lo) & (pool["FOS_pred"] <= hi)]
    counts = pd.cut(inside["FOS_pred"], edges).value_counts().sort_index()
    thin = counts[counts < drawn_per_bin]
    if len(thin):
        raise DesignError(f"bins with fewer than {drawn_per_bin} candidates:\n{thin.to_string()}")

    mu, sd = pool[params].mean(), pool[params].std()
    chosen, bins = [], []
    for b, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        candidates = pool[(pool["FOS_pred"] > a) & (pool["FOS_pred"] <= z)]
        Xs = ((candidates[params] - mu) / sd).to_numpy()
        drawn = candidates.index.to_numpy()[greedy_maxmin(Xs, drawn_per_bin, rng)]
        kept = drawn[rng.permutation(len(drawn))[:per_bin]]
        chosen.extend(kept.tolist())
        bins.extend([b] * len(kept))
    out = pool.loc[chosen, params + ["FOS_pred"]].reset_index(drop=True)
    out["bin"] = bins
    return out


def stratified_random(model, *, pool_rng, selection_rng, n_pool, n_bins, per_bin,
                      selection_range, references, min_distance) -> pd.DataFrame:
    """The target-stratified test set: random selection within each bin.

    An independent pool (its own generator) is screened; candidates outside
    the selection range or within ``min_distance`` of any reference point are
    dropped; ``per_bin`` candidates are then drawn at random from each bin on
    a separate generator. The first bin is closed [lo, hi], the others (lo, hi].
    """
    params = list(config.PARAMS)
    lo, hi = selection_range
    pool = uniform_pool(pool_rng, n_pool)
    pool["FOS_pred"] = predict_fos(model, pool)
    pool = pool[pool["FOS_pred"].between(lo, hi, inclusive="both")].reset_index(drop=True)
    if references:
        reference = pd.concat([r[params] for r in references.values()], ignore_index=True)
        pool = pool.loc[nearest_distance(pool, reference) > min_distance].reset_index(drop=True)

    edges = bin_edges(n_bins, selection_range)
    chosen, bins = [], []
    for b, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        lower = pool["FOS_pred"] >= a if b == 0 else pool["FOS_pred"] > a
        candidates = pool.loc[lower & (pool["FOS_pred"] <= z)]
        if len(candidates) < per_bin:
            raise DesignError(f"bin {b} [{a:.3f}, {z:.3f}] has only {len(candidates)} candidates")
        picked = selection_rng.choice(len(candidates), per_bin, replace=False)
        chosen.extend(candidates.index.to_numpy()[picked].tolist())
        bins.extend([b] * per_bin)
    out = pool.loc[chosen, params + ["FOS_pred"]].reset_index(drop=True)
    out["bin"] = bins
    return out


# --- roles ------------------------------------------------------------------------

def read_design(role: str) -> pd.DataFrame:
    path = config.design_path(role)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; generate role {role!r} first")
    return pd.read_csv(path)


def build(role: str) -> tuple[pd.DataFrame, dict]:
    """The design of a role, and what produced it (seeds, screener diagnostics)."""
    spec = config.ROLES[role]
    references = {r: read_design(r) for r in spec.disjoint_from}
    seeds = [config.SEEDS[name] for name in spec.seeds]
    info = {"seeds": dict(zip(spec.seeds, seeds))}

    if spec.design == "lhs":
        unit, frame = lhs(spec.n, seeds[0])
        if not lhs_strata_complete(unit):
            raise DesignError(f"{role}: LHS stratification incomplete")
    else:
        trained_on = config.SCREENER["trained_on"]
        model, diagnostics = fit_screener(data.load_role(trained_on))
        info["screener"] = {**diagnostics, "trained_on": trained_on,
                            "trained_on_sha256": data.sha256_file(config.final_path(trained_on))}
        if spec.design == "guided-maxmin":
            frame = guided_maxmin(model, rng=np.random.default_rng(seeds[0]),
                                  n_pool=config.N_GUIDED_POOL, n_bins=config.N_BINS,
                                  per_bin=config.GUIDED_PER_BIN,
                                  drawn_per_bin=config.GUIDED_DRAWN_PER_BIN,
                                  selection_range=config.SELECTION_RANGE)
        elif spec.design == "stratified-random":
            frame = stratified_random(model, pool_rng=np.random.default_rng(seeds[0]),
                                      selection_rng=np.random.default_rng(seeds[1]),
                                      n_pool=config.N_TEST_POOL, n_bins=config.N_BINS,
                                      per_bin=config.TEST_PER_BIN,
                                      selection_range=config.SELECTION_RANGE,
                                      references=references, min_distance=config.MIN_DISTANCE)
        else:
            raise ValueError(f"unknown design procedure {spec.design!r} for role {role!r}")

    if len(frame) != spec.n:
        raise DesignError(f"{role}: {len(frame)} points, expected {spec.n}")
    assert_disjoint(frame, references, config.MIN_DISTANCE)
    return frame, info
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/design.py tests/test_design.py
git commit -m "Add the three design procedures and the in-memory screener

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: `solver.py` — xslope/Bishop wrapper, parallel set runner, repair

A port of `run_analyses.py` and `run_repair.py` (reference copies in `$REF`). What changes: all values come from `config.SOLVER` and `config.REPAIR_VARIANTS`; the template is cached under `artifacts/solver/`; one analysis becomes one row `dict` keyed by `data.RESULT_COLUMNS`; a set runs in a process pool and is written once at the end.

**Files:**
- Create: `sgs/solver.py`, `tests/test_solver.py`
- Modify: `tests/helpers.py` (append `fake_analyse`, `boundary_row`)

**Interfaces:**
- Consumes: `config.SOLVER`, `config.REPAIR_VARIANTS`, `config.solver_template_path()`, `data.RESULT_COLUMNS`.
- Produces:
  - `model_dims(H, angle, ext_factor=None, depth_factor=None) -> (slope, slope_length, extension, depth)`
  - `check_boundary(surface_left, surface_right, surface_depth, geometry, H) -> (bool, list[str])`
  - `calculate_fos(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None) -> dict` (keys `fos, surface_left, surface_right, surface_depth, cx, cy, radius, at_boundary, boundary_flags, search_converged, n_circles_tested, n_valid_surfaces, n_unsolved, warnings, error, runtime_s`; `fos` is None on solver error)
  - `analysis_row(combo, analysis, ext_factor=None, depth_factor=None) -> dict`, `analyse(combo) -> dict` (module level, pickles under `spawn`)
  - `run_set(combos, *, workers, label, fn=None) -> list[dict]` (rows in design order; `fn` defaults to `analyse` looked up at call time)
  - `read_rows(path) -> (list[dict], list[str])`, `write_rows(path, rows, columns) -> None` (tmp file then `os.replace`)
  - `variant_label(ext_factor, depth_factor) -> str`, `repair_rows(rows, calculate=None) -> dict` (mutates rows, adds `repair_variant` and `excluded`, returns summary with keys `n_rows, n_boundary, n_repaired, n_unresolved, n_solver_error`)

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
def fake_analyse(combo):
    """Stands in for solver.analyse: a solved, boundary-free row. Module level
    so that a spawned worker can import it."""
    angle, phi, c, gamma, H = combo
    row = {column: "" for column in data.RESULT_COLUMNS}
    row.update(angle=angle, phi=phi, c=c, gamma=gamma, H=H, FOS=round(c / H, 6),
               At_Boundary=False, Search_Converged=True, Runtime_s=0.0)
    return row


def boundary_row(fos="1.5", boundary="False", c=10.0):
    """A results row as read back from CSV (all strings)."""
    row = {column: "" for column in data.RESULT_COLUMNS}
    row.update(angle="45.0", phi="30.0", c=str(c), gamma="18.0", H="10.0", FOS=fos, At_Boundary=boundary)
    return row
```

- [ ] **Step 2: Write the failing tests `tests/test_solver.py`**

```python
import math

import numpy as np
import pytest

from sgs import config, data, solver
from tests.helpers import boundary_row, fake_analyse


def test_model_dims_matches_the_adaptive_rule():
    slope, L, ext, depth = solver.model_dims(10.0, 45.0)
    assert math.isclose(L, 10.0)
    assert math.isclose(ext, 15.0)          # max(1.5*H, 0.5*L)
    assert math.isclose(depth, 10.0)        # max(H, 0.5*L)
    _, _, ext2, depth2 = solver.model_dims(10.0, 45.0, 2.0, 3.0)
    assert math.isclose(ext2, 30.0) and math.isclose(depth2, 30.0)


def test_boundary_flags():
    geometry = {"depth": 10.0, "slope_length": 10.0, "extension": 15.0}
    assert solver.check_boundary(-9.95, 5.0, -2.0, geometry, 10.0) == (True, ["left_boundary"])
    assert solver.check_boundary(0.0, 24.95, -9.95, geometry, 10.0) == (True, ["right_boundary", "bottom_boundary"])
    assert solver.check_boundary(0.0, 20.0, -5.0, geometry, 10.0) == (False, [])


def _analysis(fos=1.5, at_boundary=False):
    return {"fos": fos, "surface_left": 0.1, "surface_right": 9.0, "surface_depth": -1.0,
            "cx": 2.0, "cy": 8.0, "radius": 9.0, "at_boundary": at_boundary,
            "boundary_flags": [], "search_converged": True, "n_circles_tested": 10,
            "n_valid_surfaces": 5, "n_unsolved": 0, "warnings": "", "error": "",
            "runtime_s": 1.23456}


def test_analysis_row_schema_for_solved_and_failed_cases():
    combo = (45.0, 30.0, 10.0, 18.0, 10.0)
    solved = solver.analysis_row(combo, _analysis())
    assert list(solved) == data.RESULT_COLUMNS
    assert solved["FOS"] == 1.5 and solved["Error"] == "" and solved["Runtime_s"] == 1.235
    failed = solver.analysis_row(combo, {**_analysis(fos=None), "error": "RuntimeError: x"})
    assert failed["FOS"] == "" and failed["At_Boundary"] == "" and failed["Error"] == "RuntimeError: x"
    assert failed["slope_length"] == 10.0


def _combos(n=7):
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(lo, hi, n) for lo, hi in config.DOMAIN.values()])


@pytest.mark.parametrize("workers", [1, 2])
def test_run_set_keeps_design_order(workers):
    combos = _combos()
    rows = solver.run_set(combos, workers=workers, label="t", fn=fake_analyse)
    np.testing.assert_allclose([[r[p] for p in config.PARAMS] for r in rows], combos)


def test_rows_round_trip_through_csv(tmp_path):
    rows = solver.run_set(_combos(3), workers=1, label="t", fn=fake_analyse)
    path = tmp_path / "out" / "r.csv"
    solver.write_rows(path, rows, data.RESULT_COLUMNS)
    back, columns = solver.read_rows(path)
    assert columns == data.RESULT_COLUMNS and len(back) == 3
    assert not list(path.parent.glob(".*.tmp"))


# --- repair ---------------------------------------------------------------------------

def fake_calculate(frees_at):
    """Returns a boundary-free surface only on the given variant."""
    def calculate(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None):
        free = (ext_factor, depth_factor) == frees_at
        return {**_analysis(fos=1.2, at_boundary=not free), "runtime_s": 0.1}
    return calculate


def test_label():
    assert solver.variant_label(2.0, None) == "ext2.0x_depth1x"


def test_rows_without_problems_are_left_alone():
    rows = [boundary_row(), boundary_row()]
    summary = solver.repair_rows(rows, calculate=fake_calculate(None))
    assert summary["n_boundary"] == 0 and summary["n_repaired"] == 0
    assert all(r["repair_variant"] == "" and r["excluded"] == "" for r in rows)


def test_boundary_row_is_repaired_by_the_first_freeing_variant():
    rows = [boundary_row(boundary="True")]
    summary = solver.repair_rows(rows, calculate=fake_calculate((None, 2.0)))
    assert summary["n_repaired"] == 1
    assert rows[0]["repair_variant"] == "ext1x_depth2.0x"
    assert rows[0]["FOS"] == 1.2 and rows[0]["At_Boundary"] is False and rows[0]["excluded"] == ""


def test_unfreeable_row_is_excluded():
    rows = [boundary_row(boundary="True")]
    summary = solver.repair_rows(rows, calculate=fake_calculate(("never", "never")))
    assert summary["n_unresolved"] == 1 and rows[0]["excluded"] == "unresolved"


def test_solver_error_is_excluded_not_repaired():
    rows = [boundary_row(fos="")]
    summary = solver.repair_rows(rows, calculate=fake_calculate(None))
    assert summary["n_solver_error"] == 1 and rows[0]["excluded"] == "solver_error"


@pytest.mark.slow
def test_one_real_analysis():
    row = solver.analyse((24.70810805776474, 33.64879751536226, 12.628212513088526,
                          17.341403519225082, 4.832465812168988))
    # First row of the seed design: FOS 3.171339 in the reference data, not at boundary.
    assert row["Error"] == ""
    assert abs(row["FOS"] - 3.171339) < 1e-4
    assert row["At_Boundary"] is False
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_solver.py -q`
Expected: `ImportError: cannot import name 'solver'`.

- [ ] **Step 4: Write `sgs/solver.py`**

```python
"""xslope/Bishop: one homogeneous, dry, unloaded slope per analysis; a set of
analyses in a process pool; boundary repair on enlarged domains.

Every model is built in memory; the version-pinned xslope workbook supplies
only a valid input schema. Solver settings come from ``config.SOLVER``.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import math
import multiprocessing as mp
import os
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

from sgs import config

_BASE_MODEL = None


# --- the template and the model -------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_template() -> Path:
    """Return the verified, version-pinned xslope workbook, downloading it once."""
    settings = config.SOLVER
    path = config.solver_template_path()
    if path.exists() and _sha256(path) == settings.template_sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    print(f"Downloading xslope {settings.package_version} input template...")
    urllib.request.urlretrieve(settings.template_url, path)
    actual = _sha256(path)
    if actual != settings.template_sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"xslope template checksum mismatch: expected "
                           f"{settings.template_sha256}, got {actual}")
    return path


def get_base_model():
    """Load the template once per process, after checking the solver version."""
    global _BASE_MODEL
    if _BASE_MODEL is None:
        import xslope
        from xslope.fileio import load_slope_data

        installed = getattr(xslope, "__version__", "unknown")
        if installed != config.SOLVER.package_version:
            raise RuntimeError(f"xslope=={config.SOLVER.package_version} required; "
                               f"installed version is {installed}")
        _BASE_MODEL = load_slope_data(ensure_template())
    return _BASE_MODEL


def model_dims(H, angle, ext_factor=None, depth_factor=None):
    settings = config.SOLVER
    slope = math.tan(math.radians(angle))
    slope_length = H / slope
    extension = max(settings.extension_h_factor * H, settings.extension_l_factor * slope_length)
    depth = max(H, settings.depth_l_factor * slope_length)
    if ext_factor:
        extension *= ext_factor
    if depth_factor:
        depth *= depth_factor
    return slope, slope_length, extension, depth


def initialize_model(H, angle, phi, c, gamma, ext_factor=None, depth_factor=None):
    """Construct one homogeneous, dry xslope model entirely in memory."""
    from shapely.geometry import Polygon
    from xslope.fileio import build_ground_surface_from_polygons
    from xslope.generators import generate_starting_circles
    from xslope.mesh import build_polygons

    slope, slope_length, extension, depth = model_dims(H, angle, ext_factor, depth_factor)
    model = deepcopy(get_base_model())

    ground = [(-depth, 0.0), (0.0, 0.0), (slope_length, H), (slope_length + extension, H)]
    model["max_depth"] = -depth
    model["profile_lines"] = [{"coords": ground, "mat_id": 0, "size": None}]

    raw_polygons = build_polygons(slope_data={"profile_lines": model["profile_lines"],
                                              "max_depth": model["max_depth"]})
    model["polygons"] = [{"polygon": Polygon(item["coords"]), "mat_id": int(item["mat_id"]),
                          "size": item.get("size")} for item in raw_polygons]
    model["ground_surface"], model["domain_polygon"] = build_ground_surface_from_polygons(model["polygons"])

    model["materials"][0].update(name="Soil", option="mc", gamma=float(gamma), c=float(c),
                                 phi=float(phi), ru=0.0, u="none")

    # The dry, unloaded problem is explicit rather than inherited from the
    # schema workbook.
    model.update(piezo_line=[], piezo_line2=[], piezo_phreatic=False, piezo_phreatic2=False,
                 dloads=[], dloads2=[], dload_dirs=[], dload2_dirs=[], reinforce_lines=[],
                 reinforcement_lines=[], pile_lines=[], line_loads=[], k_seismic=0.0,
                 tcrack_depth=0.0, tcrack_water=0.0, circular=True, non_circ=[])

    circle_report = generate_starting_circles(model, report=True)
    if not circle_report["circles"]:
        raise RuntimeError("xslope could not generate a starting circle: "
                           + circle_report.get("reason", "unknown reason"))
    model["circles"] = circle_report["circles"]

    # Toe-side exit range and crest-side entry range.
    geometry = {
        "slope": slope, "slope_length": slope_length, "extension": extension, "depth": depth,
        "exit_range": (-depth, slope_length / 3.0),
        "entry_range": (2.0 * slope_length / 3.0, slope_length + extension),
    }
    return model, geometry


def surface_limits(best):
    """Bounds of the clipped critical arc; xslope's ``Depth`` is an elevation."""
    failure_surface = best.get("failure_surface")
    if failure_surface is None or failure_surface.is_empty:
        raise RuntimeError("xslope did not return the critical failure surface")
    left, _, right, _ = map(float, failure_surface.bounds)
    return left, right, float(best["Depth"])


def check_boundary(surface_left, surface_right, surface_depth, geometry, H):
    """Flag a critical surface that reaches an analysis-domain boundary."""
    domain_left = -geometry["depth"]
    domain_right = geometry["slope_length"] + geometry["extension"]
    domain_bottom = -geometry["depth"]
    tol = config.SOLVER.boundary_tol_h_frac * H
    flags = []
    if surface_left <= domain_left + tol:
        flags.append("left_boundary")
    if surface_right >= domain_right - tol:
        flags.append("right_boundary")
    if surface_depth <= domain_bottom + tol:
        flags.append("bottom_boundary")
    return bool(flags), flags


def calculate_fos(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None) -> dict:
    """One xslope/Bishop circular search. ``fos`` is None on solver error."""
    from xslope.search import circular_search

    settings = config.SOLVER
    started = time.perf_counter()
    result = {"fos": None, "surface_left": None, "surface_right": None, "surface_depth": None,
              "cx": None, "cy": None, "radius": None, "at_boundary": None, "boundary_flags": [],
              "search_converged": False, "n_circles_tested": 0, "n_valid_surfaces": 0,
              "n_unsolved": 0, "warnings": "", "error": "", "runtime_s": None}
    try:
        model, geometry = initialize_model(H, angle, phi, c, gamma, ext_factor, depth_factor)
        unsolved = {}
        with contextlib.redirect_stdout(io.StringIO()):
            fs_cache, converged, _, circle_cache = circular_search(
                model, settings.method,
                num_slices=settings.num_slices,
                seed=settings.search_start,
                tol=settings.search_tol,
                fs_tol=settings.search_fs_tol,
                max_iter=settings.search_max_iter,
                min_grid_frac=settings.search_min_grid_frac,
                diagnostic=False,
                entry_range=geometry["entry_range"],
                exit_range=geometry["exit_range"],
                unsolved_out=unsolved,
            )
        if not fs_cache:
            raise RuntimeError("xslope circular search returned no valid surface")
        best = fs_cache[0]
        cy = float(best["Yo"])
        surface_left, surface_right, surface_depth = surface_limits(best)
        at_boundary, flags = check_boundary(surface_left, surface_right, surface_depth, geometry, H)
        solver_result = best.get("solver_result") or {}
        result.update(
            fos=float(best["FS"]), surface_left=surface_left, surface_right=surface_right,
            surface_depth=surface_depth, cx=float(best["Xo"]), cy=cy,
            radius=cy - float(best["Depth"]), at_boundary=at_boundary, boundary_flags=flags,
            search_converged=bool(converged), n_circles_tested=len(circle_cache),
            n_valid_surfaces=len(fs_cache), n_unsolved=int(unsolved.get("unsolved", 0)),
            warnings=" | ".join(map(str, solver_result.get("warnings", []))),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["runtime_s"] = time.perf_counter() - started
    return result


def analysis_row(combo, analysis: dict, ext_factor=None, depth_factor=None) -> dict:
    """One result row, keyed by ``data.RESULT_COLUMNS``."""
    angle, phi, c, gamma, H = (float(v) for v in combo)
    _, slope_length, extension, depth = model_dims(H, angle, ext_factor, depth_factor)
    solved = analysis["fos"] is not None

    def r6(value):
        return round(value, 6) if solved else ""

    return {
        "angle": angle, "phi": phi, "c": c, "gamma": gamma, "H": H,
        "FOS": r6(analysis["fos"]),
        "Surface_L": r6(analysis["surface_left"]),
        "Surface_R": r6(analysis["surface_right"]),
        "Surface_Depth": r6(analysis["surface_depth"]),
        "Center_X": r6(analysis["cx"]),
        "Center_Y": r6(analysis["cy"]),
        "Radius": r6(analysis["radius"]),
        "At_Boundary": analysis["at_boundary"] if solved else "",
        "slope_length": round(slope_length, 6),
        "extension": round(extension, 6),
        "depth": round(depth, 6),
        "Search_Converged": analysis["search_converged"] if solved else False,
        "N_Circles_Tested": analysis["n_circles_tested"],
        "N_Valid_Surfaces": analysis["n_valid_surfaces"],
        "N_Unsolved": analysis["n_unsolved"],
        "Runtime_s": round(analysis["runtime_s"], 3),
        "Warnings": analysis["warnings"],
        "Error": "" if solved else analysis["error"],
    }


def analyse(combo) -> dict:
    """Solve one parameter combination ``(angle, phi, c, gamma, H)``."""
    angle, phi, c, gamma, H = (float(v) for v in combo)
    return analysis_row(combo, calculate_fos(angle, phi, c, gamma, H))


# --- a set of analyses ----------------------------------------------------------------

def run_set(combos, *, workers: int, label: str, fn=None) -> list[dict]:
    """Solve every combination, in design order. Nothing is written here: a set
    of 300 takes minutes, so an interrupted set is simply rerun."""
    fn = fn or analyse
    combos = [tuple(float(v) for v in combo) for combo in combos]
    rows = []

    def progress(index, row):
        state = row["Error"] or f"FOS={row['FOS']}" + (" AT BOUNDARY" if row["At_Boundary"] is True else "")
        print(f"[{label} {index}/{len(combos)}] {state}", flush=True)

    if workers == 1:
        for index, combo in enumerate(combos, 1):
            rows.append(fn(combo))
            progress(index, rows[-1])
    else:
        # spawn: portable to macOS, and no solver state inherited from the parent.
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, row in enumerate(pool.imap(fn, combos), 1):
                rows.append(row)
                progress(index, row)
    return rows


def read_rows(path) -> tuple[list[dict], list[str]]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path, rows, columns) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# --- repair -------------------------------------------------------------------------

def variant_label(ext_factor, depth_factor) -> str:
    return f"ext{ext_factor or 1}x_depth{depth_factor or 1}x"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def repair_rows(rows: list[dict], calculate=None) -> dict:
    """Recompute boundary-truncated rows on the enlarged domains of
    ``config.REPAIR_VARIANTS``, in order, until the surface sits inside the
    domain. Rows no variant frees are excluded as ``unresolved``; rows the
    solver could not analyse are excluded as ``solver_error``."""
    calculate = calculate or calculate_fos
    for row in rows:
        row["repair_variant"] = ""
        row["excluded"] = ""
    n_boundary = n_repaired = n_error = 0
    for index, row in enumerate(rows):
        if _float(row["FOS"]) is None:
            row["excluded"] = "solver_error"
            n_error += 1
            continue
        if str(row["At_Boundary"]).strip().lower() != "true":
            continue
        n_boundary += 1
        combo = [float(row[p]) for p in config.PARAMS]
        for ext_factor, depth_factor in config.REPAIR_VARIANTS:
            label = variant_label(ext_factor, depth_factor)
            analysis = calculate(*combo, ext_factor, depth_factor)
            if analysis["fos"] is None or analysis["at_boundary"]:
                print(f"  row {index + 1}: {label} did not free the surface")
                continue
            row.update(analysis_row(combo, analysis, ext_factor, depth_factor))
            row["repair_variant"] = label
            n_repaired += 1
            print(f"  row {index + 1}: repaired with {label}, FOS={analysis['fos']:.4f}")
            break
        else:
            row["excluded"] = "unresolved"
            print(f"  row {index + 1}: no variant freed the surface; excluded")
    return {"n_rows": len(rows), "n_boundary": n_boundary, "n_repaired": n_repaired,
            "n_unresolved": n_boundary - n_repaired, "n_solver_error": n_error}
```

- [ ] **Step 5: Run tests, including the slow one once**

Run: `.venv/bin/pytest tests/test_solver.py -q`
Expected: all pass, 1 deselected.

Run: `.venv/bin/pytest tests/test_solver.py -q -m slow`
Expected: 1 passed, in about 10 s. It uses the template cached in `artifacts/solver/` by Task 2.

- [ ] **Step 6: Commit**

```bash
git add sgs/solver.py tests/test_solver.py tests/helpers.py
git commit -m "Port the xslope/Bishop wrapper, a parallel set runner and boundary repair

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: `cli.py` — `generate` and the subcommands

The whole CLI is written here. Handlers for `fit`, `evaluate`, `report` and `learning-curve` import their modules lazily, so the parser works before those modules exist (Tasks 9, 11, 12, 13 provide them).

**Files:**
- Create: `sgs/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `design.build`, `solver.run_set/read_rows/write_rows/repair_rows`, `data.sha256_file/read_metadata/record/environment/RESULT_COLUMNS/FINAL_COLUMNS`.
- Produces:
  - `GenerateError(RuntimeError)`, `generate(role, *, workers=1, force=False) -> dict` (the `run_metadata.json["datasets"][role]` entry)
  - `build_parser()`, `main(argv=None)`
  - Later modules must provide exactly: `models.fit_all(*, force)` (Task 9), `evaluate.run()` (Task 11), `report.run()` (Task 12), `models.learning_curve(*, force)` and `evaluate.run_learning_curve()` (Task 13).
- `run_metadata.json["datasets"][role]` keys: `n, seeds, design_sha256, results_sha256, final_sha256, repair, environment, updated`, plus `screener` for screener roles.

- [ ] **Step 1: Write the failing tests `tests/test_cli.py`**

```python
import pytest

from sgs import cli, config, data, solver
from tests.helpers import fake_analyse


def test_every_command_parses():
    parser = cli.build_parser()
    for argv in (["generate"], ["generate", "--role", "guided", "--workers", "4", "--force"],
                 ["fit", "--force"], ["evaluate"], ["report"], ["learning-curve", "--force"],
                 ["all", "--workers", "2"]):
        assert callable(parser.parse_args(argv).handler)


def test_unknown_role_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--role", "nope"])


def test_no_scientific_flags_exist():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--seed", "1"])


def test_workers_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["generate", "--role", "seed", "--workers", "0"])


@pytest.fixture
def fake_solver(root, monkeypatch):
    calls = []

    def counting(combo):
        calls.append(combo)
        return fake_analyse(combo)

    monkeypatch.setattr(solver, "analyse", counting)
    return calls


def test_generate_writes_three_files_and_records_them(fake_solver):
    entry = cli.generate("seed")
    for path in (config.design_path("seed"), config.results_path("seed"), config.final_path("seed")):
        assert path.exists()
    assert len(fake_solver) == 300
    assert list(data.read_table(config.final_path("seed")).columns) == data.FINAL_COLUMNS
    recorded = data.read_metadata()["datasets"]["seed"]
    assert recorded == entry
    assert recorded["design_sha256"] == data.sha256_file(config.design_path("seed"))
    assert recorded["seeds"] == {"seed_lhs_design": 42} and recorded["repair"]["n_boundary"] == 0


def test_generate_reuses_existing_outputs(fake_solver):
    cli.generate("seed")
    results = data.sha256_file(config.results_path("seed"))
    cli.generate("seed")
    assert len(fake_solver) == 300
    assert data.sha256_file(config.results_path("seed")) == results


def test_repair_is_redone_from_untouched_results(fake_solver):
    cli.generate("seed")
    results = data.sha256_file(config.results_path("seed"))
    config.final_path("seed").unlink()
    cli.generate("seed")
    assert len(fake_solver) == 300 and config.final_path("seed").exists()
    assert data.sha256_file(config.results_path("seed")) == results


def test_changed_design_is_refused_without_force(fake_solver, monkeypatch):
    cli.generate("seed")
    monkeypatch.setitem(config.SEEDS, "seed_lhs_design", 43)
    with pytest.raises(cli.GenerateError, match="differs"):
        cli.generate("seed")
    cli.generate("seed", force=True)
    assert len(fake_solver) == 600


def test_results_from_another_design_are_refused(fake_solver):
    cli.generate("seed")
    entry = data.read_metadata()["datasets"]["seed"]
    data.record("datasets", "seed", {**entry, "design_sha256": "0" * 64})
    with pytest.raises(cli.GenerateError, match="different design"):
        cli.generate("seed")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: `ImportError: cannot import name 'cli'`.

- [ ] **Step 3: Write `sgs/cli.py`**

```python
"""sgs — the surrogate-guided sampling pipeline.

  sgs generate        design → solve → repair, for every role or one
  sgs fit             tuned searches with nested OOF residuals, cached
  sgs evaluate        metrics on the test sets (seconds)
  sgs report          booktabs tables and figures
  sgs learning-curve  optional; RF with fixed hyper-parameters
  sgs all             generate, fit, evaluate, report

Scientific parameters live in sgs/config.py; the command line only selects a
role and controls execution.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import platform
import sys

from sgs import config, data, design, solver

REFERENCE_PYTHON = "3.14.4"


class GenerateError(RuntimeError):
    """Existing outputs of a role do not match what would be produced now."""


def generate(role: str, *, workers: int = 1, force: bool = False) -> dict:
    """Design, solve and repair one role; reuse outputs that still match."""
    spec = config.ROLES[role]
    frame, info = design.build(role)
    design_path, results_path, final_path = (config.design_path(role), config.results_path(role),
                                             config.final_path(role))

    text = frame.to_csv(index=False)
    if design_path.exists() and design_path.read_text() != text and not force:
        raise GenerateError(f"{design_path.name} differs from the design the current configuration "
                            f"and data produce; rerun with --force to regenerate {role!r}")
    if not design_path.exists() or design_path.read_text() != text:
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design_path.write_text(text)
    design_sha = data.sha256_file(design_path)

    previous = data.read_metadata().get("datasets", {}).get(role, {})
    solved = False
    if results_path.exists() and not force:
        if previous.get("design_sha256") != design_sha:
            raise GenerateError(f"{results_path.name} was solved from a different design; "
                                f"rerun with --force to solve {role!r} again")
        print(f"[{role}] results up to date")
    else:
        rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=role)
        solver.write_rows(results_path, rows, data.RESULT_COLUMNS)
        solved = True

    if solved or force or not final_path.exists():
        rows, columns = solver.read_rows(results_path)
        if columns != data.RESULT_COLUMNS:
            raise GenerateError(f"{results_path.name} does not match the result schema")
        repair = solver.repair_rows(rows)
        solver.write_rows(final_path, rows, data.FINAL_COLUMNS)
        print(f"[{role}] repair: {repair}")
    else:
        repair = previous["repair"]
        print(f"[{role}] final up to date")

    entry = {"n": spec.n, **info, "design_sha256": design_sha,
             "results_sha256": data.sha256_file(results_path),
             "final_sha256": data.sha256_file(final_path), "repair": repair,
             "environment": data.environment(),
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    data.record("datasets", role, entry)
    return entry


def _generate(args):
    roles = [args.role] if getattr(args, "role", None) else list(config.ROLE_ORDER)
    for role in roles:
        generate(role, workers=args.workers, force=args.force)


def _fit(args):
    from sgs import models
    models.fit_all(force=args.force)


def _evaluate(args):
    from sgs import evaluate
    evaluate.run()


def _report(args):
    from sgs import report
    report.run()


def _learning_curve(args):
    from sgs import evaluate, models
    models.learning_curve(force=args.force)
    evaluate.run_learning_curve()


def _all(args):
    _generate(args)
    _fit(args)
    _evaluate(args)
    _report(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, help, *, role=False, workers=False, force=False):
        p = sub.add_parser(name, help=help)
        if role:
            p.add_argument("--role", choices=tuple(config.ROLES), default=None,
                           help="one role (default: all, in dependency order)")
        if workers:
            p.add_argument("--workers", type=int, default=1, help="solver processes")
        if force:
            p.add_argument("--force", action="store_true",
                           help="regenerate outputs instead of reusing or refusing them")
        p.set_defaults(handler=handler)

    command("generate", _generate, "design, solve and repair the data sets", role=True, workers=True, force=True)
    command("fit", _fit, "tuned searches with nested OOF residuals", force=True)
    command("evaluate", _evaluate, "metrics on the test sets")
    command("report", _report, "booktabs tables and figures from outputs/metrics")
    command("learning-curve", _learning_curve, "optional: RF learning curve with fixed hyper-parameters",
            force=True)
    command("all", _all, "generate, fit, evaluate, report", workers=True, force=True)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be at least 1")
    if platform.python_version() != REFERENCE_PYTHON:
        print(f"warning: running on Python {platform.python_version()}; "
              f"the reference environment is {REFERENCE_PYTHON}", file=sys.stderr)
    args.handler(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

Run: `.venv/bin/sgs --help`
Expected: the six subcommands are listed.

- [ ] **Step 5: Commit**

```bash
git add sgs/cli.py tests/test_cli.py
git commit -m "Add the sgs CLI and the generate command

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: Generate the five data sets

**About an hour of solver time, unattended** (under 15 minutes per set of 300 with two workers).

**Files:**
- Produces: `data/designs/*.csv`, `data/results/*.csv`, `data/final/*.csv` (ignored), `data/run_metadata.json` (tracked)

- [ ] **Step 1: Generate**

```bash
.venv/bin/sgs generate --workers 2 2>&1 | tee /tmp/generate.log
```
Expected: for each of the five roles, solver progress lines, then `repair: {...}`. More workers are fine if the machine has the cores. If the run is interrupted, rerun the same command: finished roles print `results up to date` and `final up to date`.

- [ ] **Step 2: Validate the sets**

```bash
.venv/bin/python -c "from sgs import config, data; data.validate_sets(config.ROLE_ORDER); print('all sets valid and pairwise disjoint')"
```
Expected: `all sets valid and pairwise disjoint`. A `DataError` here means something upstream is wrong: stop and report it to the author.

- [ ] **Step 3: Descriptive comparison with the reference copies (for the author, not a gate)**

```bash
.venv/bin/python - <<'EOF'
import numpy as np
from sgs import config, data
REF = "/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09/"
OLD = {"seed": "initial_lhs_300_bishop.csv", "lhs-extension": "results_lhs_bishop.csv",
       "guided": "results_uniform_bishop.csv", "test-natural": "results_test_bishop.csv",
       "test-stratified": "results_test_150_bishop.csv"}
edges = np.linspace(*config.SELECTION_RANGE, config.N_BINS + 1)
for role, name in OLD.items():
    new, old = data.read_table(config.final_path(role)), data.read_table(REF + name)
    if "excluded" not in old:
        old["excluded"] = ""
    print(f"\n== {role}")
    for label, frame in (("new", new), ("ref", old)):
        fos = data.usable(frame)["FOS"].to_numpy(float)
        print(f"  {label}: n={len(frame)} excluded={len(frame) - len(fos)} "
              f"min={fos.min():.3f} median={np.median(fos):.3f} max={fos.max():.3f} "
              f"bins={np.histogram(fos, edges)[0].tolist()}")
    dparam = np.abs(new[list(config.PARAMS)].to_numpy(float) - old[list(config.PARAMS)].to_numpy(float)).max()
    print(f"  max |Δparam| vs reference: {dparam:.2e}")
EOF
```
Expected, per the Task 1 probe: `seed`, `lhs-extension` and `test-natural` have max |Δparam| ≤ 5e-7; `guided` and `test-stratified` differ, because the screener differs slightly from Colab. Distributions (n, median, bin occupancy) should be comparable. Record the output in the task report for the author.

- [ ] **Step 4: Commit the metadata**

```bash
git add data/run_metadata.json
git status --short data/     # must list only run_metadata.json
git commit -m "Generate the five data sets through the pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: `models.py` — tuned searches, nested OOF residuals, the fit cache

**Files:**
- Create: `sgs/models.py`, `tests/test_models.py`
- Modify: `tests/conftest.py` (add the `tiny_protocol` fixture)

**Interfaces:**
- Consumes: `data.load_arm`, `data.validate_sets`, `data.sha256_file`, `data.record`, `data.environment`, `config.GRIDS/ESTIMATOR_FIXED/CV_FOLDS/SEEDS/TARGET/fit_fingerprint`.
- Produces:
  - `FitArtifact` dataclass: `family, arm, scoring, target, model, best_params: dict, cv_results: dict, train_ids: ndarray[str], oof_pred: ndarray` (log scale), `oof_residual: ndarray` (`y_log − oof_pred`, signed), `fold: ndarray[int]`; property `scores -> |oof_residual|`
  - `estimator(family)`, `scorer(scoring)`, `cv_splitter() -> KFold`, `searcher(family, scoring) -> GridSearchCV`, `fit_arm(family, arm, scoring) -> FitArtifact`
  - `StaleFit(RuntimeError)`, `training_hashes(arm) -> dict[role, sha256]`, `save_fit(artifact) -> None`, `fit_problems(stored: dict) -> list[str]`, `load_fit(family, arm, scoring) -> FitArtifact`, `all_units() -> list[(family, arm, scoring)]`, `fit_all(*, force=False, units=None) -> None`
  - Fit file content: `{"artifact": FitArtifact, "fingerprint": str, "data": {role: sha256}, "environment": dict}`.

- [ ] **Step 1: Add the shared fixture to `tests/conftest.py`**

```python
@pytest.fixture
def tiny_protocol(root, monkeypatch):
    """Small grids and three small training sets, for fast fitting tests."""
    from tests.helpers import shrink_roles, synthetic_final, write_final

    monkeypatch.setattr(config, "GRIDS", {"rf": {"min_samples_leaf": [1, 5]},
                                          "gbm": {"max_iter": [20], "min_samples_leaf": [3, 10]}})
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20},
                                                    "gbm": {"early_stopping": False}})
    monkeypatch.setattr(config, "CV_FOLDS", 3)
    shrink_roles(monkeypatch, {"seed": 60, "lhs-extension": 60, "guided": 60})
    for i, role in enumerate(("seed", "lhs-extension", "guided")):
        write_final(role, synthetic_final(60, 10 + i))
```

- [ ] **Step 2: Write the failing tests `tests/test_models.py`**

```python
import joblib
import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from sgs import config, data, models
from tests.helpers import synthetic_final, write_final


def test_estimators_are_single_threaded_and_seeded():
    rf = models.estimator("rf")
    assert rf.n_jobs == 1 and rf.random_state == config.SEEDS["protocol_estimator"]
    assert models.estimator("gbm").random_state == config.SEEDS["protocol_estimator"]


def test_search_is_parallel_and_uses_the_named_split():
    search = models.searcher("rf", "log-rmse")
    assert isinstance(search, GridSearchCV) and search.n_jobs == -1
    assert search.cv.random_state == config.SEEDS["protocol_cv_split"]
    assert search.cv.n_splits == config.CV_FOLDS
    assert search.scoring == "neg_root_mean_squared_error"


def test_fos_scale_scorer():
    assert models._fos_rmse(np.log([1.0, 2.0]), np.log([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))
    assert models.scorer("fos-rmse") is not None
    with pytest.raises(ValueError):
        models.scorer("r2")


def test_fit_arm_artifact(tiny_protocol):
    art = models.fit_arm("rf", "C", "log-rmse")
    assert (art.family, art.arm, art.scoring, art.target) == ("rf", "C", "log-rmse", "log")
    assert len(art.train_ids) == 120 and art.train_ids[0] == "seed:0" and art.train_ids[-1] == "guided:59"
    assert set(art.fold) == {0, 1, 2}
    assert set(art.best_params) == {"min_samples_leaf"}
    assert len(art.cv_results["params"]) == 2
    assert np.all(art.scores >= 0) and art.scores.shape == (120,)


def test_oof_residuals_come_from_the_search_object(tiny_protocol, monkeypatch):
    seen = []
    real = models.cross_val_predict

    def spy(estimator, *args, **kwargs):
        seen.append(estimator)
        return real(estimator, *args, **kwargs)

    monkeypatch.setattr(models, "cross_val_predict", spy)
    models.fit_arm("gbm", "A", "log-rmse")
    assert len(seen) == 1 and isinstance(seen[0], GridSearchCV)


def test_fit_arm_is_deterministic(tiny_protocol):
    a = models.fit_arm("gbm", "C0", "log-rmse")
    b = models.fit_arm("gbm", "C0", "log-rmse")
    np.testing.assert_array_equal(a.oof_pred, b.oof_pred)
    assert a.best_params == b.best_params


def test_saved_fit_loads_while_fresh(tiny_protocol):
    models.save_fit(models.fit_arm("rf", "A", "log-rmse"))
    stored = joblib.load(config.fit_path("rf", "A", "log-rmse"))
    assert stored["data"] == {"seed": data.sha256_file(config.final_path("seed"))}
    assert models.load_fit("rf", "A", "log-rmse").arm == "A"


def test_load_fit_fails_on_a_changed_grid(tiny_protocol, monkeypatch):
    models.save_fit(models.fit_arm("rf", "A", "log-rmse"))
    monkeypatch.setitem(config.GRIDS, "rf", {"min_samples_leaf": [1, 2]})
    with pytest.raises(models.StaleFit, match="protocol configuration changed"):
        models.load_fit("rf", "A", "log-rmse")


def test_load_fit_fails_on_changed_training_data(tiny_protocol):
    models.save_fit(models.fit_arm("rf", "A", "log-rmse"))
    write_final("seed", synthetic_final(60, 99))
    with pytest.raises(models.StaleFit, match="training data 'seed' changed"):
        models.load_fit("rf", "A", "log-rmse")


def test_load_fit_names_a_missing_file(tiny_protocol):
    with pytest.raises(FileNotFoundError, match="sgs fit"):
        models.load_fit("rf", "A", "log-rmse")


def test_fit_all_skips_fresh_units_and_refuses_stale_ones(tiny_protocol, monkeypatch):
    unit = [("rf", "C", "log-rmse")]
    models.fit_all(units=unit)
    assert data.read_metadata()["fits"]["rf__C__log-rmse"]["n_train"] == 120
    calls = []
    real = models.fit_arm
    monkeypatch.setattr(models, "fit_arm", lambda *a: calls.append(a) or real(*a))
    models.fit_all(units=unit)
    assert calls == []
    write_final("guided", synthetic_final(60, 77))
    with pytest.raises(models.StaleFit, match="--force"):
        models.fit_all(units=unit)
    models.fit_all(units=unit, force=True)
    assert calls == [("rf", "C", "log-rmse")]


def test_fit_all_validates_the_training_sets(tiny_protocol):
    write_final("guided", synthetic_final(60, 10))       # identical to the seed set
    with pytest.raises(data.DataError, match="not disjoint"):
        models.fit_all(units=[("rf", "C", "log-rmse")])


def test_all_units_cover_the_declared_axes():
    assert len(models.all_units()) == len(config.FAMILIES) * len(config.ARMS) * len(config.ACTIVE_SCORINGS)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_models.py -q`
Expected: `ImportError: cannot import name 'models'`.

- [ ] **Step 4: Write `sgs/models.py`**

```python
"""The tuned learners: one fixed grid per family, identical for every arm.

Hyper-parameters are selected by K-fold CV on the arm's training data only;
no test set is touched. Out-of-fold residuals come from a nested CV:
``cross_val_predict`` receives the search object, so each OOF prediction is
made with hyper-parameters selected without its fold.

Each (family, arm, scoring) fit is cached in artifacts/fits/ together with a
fingerprint of its protocol configuration and the hashes of its training
data; a fit whose configuration or data changed is refused.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from sgs import config, data


@dataclass
class FitArtifact:
    family: str
    arm: str
    scoring: str
    target: str                  # "log": predictions are exponentiated at evaluation
    model: object                # refit on the whole arm with best_params
    best_params: dict
    cv_results: dict             # every grid point: params, mean and std CV score
    train_ids: np.ndarray
    oof_pred: np.ndarray         # nested out-of-fold predictions, target scale
    oof_residual: np.ndarray     # y_target - oof_pred, signed, target scale
    fold: np.ndarray             # outer fold of each training row

    @property
    def scores(self) -> np.ndarray:
        """Raw conformity scores |y - ŷ| on the target scale."""
        return np.abs(self.oof_residual)


# --- the protocol -----------------------------------------------------------------

def estimator(family: str):
    fixed = config.ESTIMATOR_FIXED[family]
    random_state = config.SEEDS["protocol_estimator"]
    if family == "rf":
        return RandomForestRegressor(**fixed, random_state=random_state, n_jobs=1)
    if family == "gbm":
        return HistGradientBoostingRegressor(**fixed, random_state=random_state)
    raise ValueError(f"unknown family {family!r}")


def _fos_rmse(y_log, p_log) -> float:
    return float(np.sqrt(np.mean((np.exp(p_log) - np.exp(y_log)) ** 2)))


def scorer(scoring: str):
    if scoring == "log-rmse":
        return "neg_root_mean_squared_error"
    if scoring == "fos-rmse":
        return make_scorer(_fos_rmse, greater_is_better=False)
    raise ValueError(f"unknown scoring {scoring!r}")


def cv_splitter() -> KFold:
    return KFold(config.CV_FOLDS, shuffle=True, random_state=config.SEEDS["protocol_cv_split"])


def searcher(family: str, scoring: str) -> GridSearchCV:
    return GridSearchCV(estimator(family), config.GRIDS[family], cv=cv_splitter(),
                        scoring=scorer(scoring), n_jobs=-1)


def fit_arm(family: str, arm: str, scoring: str) -> FitArtifact:
    if config.TARGET != "log":
        raise ValueError(f"unsupported target convention {config.TARGET!r}")
    frame = data.load_arm(arm)
    X = frame[list(config.FEATURES)]
    y = np.log(frame["FOS"].to_numpy(float))

    search = searcher(family, scoring).fit(X, y)
    outer = cv_splitter()
    oof = cross_val_predict(searcher(family, scoring), X, y, cv=outer, n_jobs=1)
    fold = np.empty(len(y), dtype=int)
    for k, (_, held_out) in enumerate(outer.split(X)):
        fold[held_out] = k

    cv = search.cv_results_
    return FitArtifact(
        family=family, arm=arm, scoring=scoring, target=config.TARGET,
        model=search.best_estimator_, best_params=dict(search.best_params_),
        cv_results={"params": [dict(p) for p in cv["params"]],
                    "mean_test_score": cv["mean_test_score"].tolist(),
                    "std_test_score": cv["std_test_score"].tolist()},
        train_ids=frame["row_id"].to_numpy(str), oof_pred=oof, oof_residual=y - oof, fold=fold,
    )


# --- the fit cache ------------------------------------------------------------------

class StaleFit(RuntimeError):
    """A cached fit no longer matches its configuration or training data."""


def training_hashes(arm: str) -> dict:
    return {role: data.sha256_file(config.final_path(role)) for role in config.ARMS[arm].roles}


def _dump(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, tmp)
    os.replace(tmp, path)


def save_fit(artifact: FitArtifact) -> None:
    _dump(config.fit_path(artifact.family, artifact.arm, artifact.scoring), {
        "artifact": artifact,
        "fingerprint": config.fit_fingerprint(artifact.family, artifact.scoring),
        "data": training_hashes(artifact.arm),
        "environment": data.environment(),
    })


def fit_problems(stored: dict) -> list[str]:
    artifact = stored["artifact"]
    problems = []
    if stored["fingerprint"] != config.fit_fingerprint(artifact.family, artifact.scoring):
        problems.append("protocol configuration changed (grid, fixed parameters, CV, seeds, target or features)")
    now = training_hashes(artifact.arm)
    for role in sorted(set(now) | set(stored["data"])):
        if stored["data"].get(role) != now.get(role):
            problems.append(f"training data {role!r} changed")
    return problems


def load_fit(family: str, arm: str, scoring: str) -> FitArtifact:
    path = config.fit_path(family, arm, scoring)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs fit`")
    stored = joblib.load(path)
    problems = fit_problems(stored)
    if problems:
        raise StaleFit(f"{path.name} is stale: {'; '.join(problems)}. Run `sgs fit --force`.")
    return stored["artifact"]


def all_units() -> list[tuple[str, str, str]]:
    return [(f, a, s) for s in config.ACTIVE_SCORINGS for f in config.FAMILIES for a in config.ARMS]


def fit_all(*, force: bool = False, units=None) -> None:
    units = list(units or all_units())
    data.validate_sets(sorted({role for _, arm, _ in units for role in config.ARMS[arm].roles}))
    for index, (family, arm, scoring) in enumerate(units, 1):
        path = config.fit_path(family, arm, scoring)
        label = f"[fit {index}/{len(units)}] {family} {arm} {scoring}"
        if path.exists():
            problems = fit_problems(joblib.load(path))
            if not problems:
                print(f"{label}: up to date")
                continue
            if not force:
                raise StaleFit(f"{path.name} is stale: {'; '.join(problems)}. "
                               "Rerun with --force to refit.")
        print(label, flush=True)
        artifact = fit_arm(family, arm, scoring)
        save_fit(artifact)
        print(f"  best {artifact.best_params}")
        data.record("fits", f"{family}__{arm}__{scoring}", {
            "best_params": artifact.best_params, "n_train": int(len(artifact.train_ids)),
            "fingerprint": config.fit_fingerprint(family, scoring), "data": training_hashes(arm),
            "environment": data.environment(),
            "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sgs/models.py tests/test_models.py tests/conftest.py
git commit -m "Add the symmetric tuning protocol and a fit cache that refuses stale fits

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: Run the searches; retire the environment probe

**Hours. Runs once; nothing downstream refits.** 8 units: 2 families × 4 arms × 1 active scoring; each unit is one search plus a five-fold nested `cross_val_predict`, so 48 grid searches in total.

- [ ] **Step 1: Fit**

```bash
.venv/bin/sgs fit 2>&1 | tee /tmp/fit.log
```
Expected: `[fit k/8] ...` lines with `best {...}`. If interrupted, rerun the same command: finished units print `up to date`.

- [ ] **Step 2: Compare the selected hyper-parameters with the Colab table (descriptive)**

```bash
.venv/bin/python -c "
from sgs import data
for key, entry in sorted(data.read_metadata()['fits'].items()):
    print(f\"{key:22} n={entry['n_train']:4}  {entry['best_params']}\")
"
```
Compare with the table in `docs/environment-probe-2026-09.md` and put the comparison in the task report. Differences are legitimate: the data were regenerated and the boundary rows repaired.

- [ ] **Step 3: Retire the probe and commit**

```bash
git rm -q docs/environment-probe-2026-09.md tools/probe_environment.py
git add data/run_metadata.json
git commit -m "Fit all arms locally; retire the environment probe

The acceptance target is determinism of the local run, not agreement with
the earlier Colab selections.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 11: `evaluate.py` — metrics, paired comparisons, `sgs evaluate`

**Files:**
- Create: `sgs/evaluate.py`, `tests/test_evaluate.py`
- Modify: `tests/helpers.py` (append stubs)

**Interfaces:**
- Consumes: `models.FitArtifact`, `models.load_fit`, `data.TestSet`, `data.load_test`, `data.validate_sets`, `data.sha256_file`, `data.environment`, `data.record`.
- Produces:
  - `Prediction(family, arm, scoring, test, ids, y_true, y_pred)` (frozen), `predict(fit, test) -> Prediction`
  - `rmse(y, p)`, `global_metrics(y, p) -> dict` (`n, r2, rmse, mae, bias, max_abs_error`)
  - `zones(edges) -> list[(label, lo, hi)]` (labels `"FOS < 1"`, `"1–1.3"`, `"1.3–3"`, `"3–6"`, `"FOS ≥ 6"`), `zone_metrics(y, p, edges) -> list[dict(zone, lo, hi, n, rmse, mae, bias)]`, `minimax(zone_rows) -> dict(zone, rmse)`
  - `conformal_quantile(scores, alpha)`, `conformal(pred, scores, *, alpha, edges) -> dict` (`alpha, q_log, factor, n_calibration, coverage, mean_width, zones, procedure`)
  - `SUBSETS = ("all", "selection-range")`, `subset_mask(y, subset, selection_range)`
  - `MisalignedPair(ValueError)`, `UnregisteredComparison(KeyError)`, `registered(first, second, registry)`, `wilcoxon_abs(p1, p2, *, registry, mask=None) -> dict` (`first, second, n, median_d, p_d_lt_0, statistic, pvalue, alternative, hypothesised, reading`), `paired_bootstrap(p1, p2, *, registry, n_boot, seed, ci, mask=None) -> dict` (`delta_rmse, ci, ci_level, p_delta_lt_0, n_boot, limitation`)
  - `EvalResult` with `to_json()`, `evaluate(fits, test, *, alpha, zones, comparisons, selection_range, n_boot, bootstrap_seed, bootstrap_ci) -> EvalResult`, `run() -> None`
- Metrics JSON (read by `report.py`): `{"test", "scoring", "n", "fos_range", "arms": {family: {arm: {"global", "zones", "minimax", "conformal", "best_params", "n_train"}}}, "comparisons": [{"family", "subset", "first", "second", "wilcoxon", "bootstrap"}], "predictions": {"row_id", "y_true", "<family>__<arm>": [...]}, "notes": {...}, "provenance": {"final_sha256", "fit_fingerprints", "environment"}}`.

- [ ] **Step 1: Append the stubs to `tests/helpers.py`**

```python
class OffsetModel:
    """log(FOS) of the formula, perturbed row-wise by a fixed pattern."""

    def __init__(self, scale):
        self.scale = scale

    def predict(self, X):
        return np.log(synthetic_fos(X)) + self.scale * np.sin(np.arange(len(X)) * 1.7)


ARM_SCALES = {"A": 0.30, "B": 0.20, "C": 0.10, "C0": 0.25}


def stub_fit(family, arm, scale, scoring="log-rmse", n_train=40):
    from sgs.models import FitArtifact
    rng = np.random.default_rng(int(scale * 1000) + len(arm))
    return FitArtifact(family=family, arm=arm, scoring=scoring, target="log", model=OffsetModel(scale),
                       best_params={}, cv_results={},
                       train_ids=np.array([f"seed:{i}" for i in range(n_train)]),
                       oof_pred=np.zeros(n_train), oof_residual=rng.normal(0.0, 0.1, n_train),
                       fold=np.arange(n_train) % 5)


def stub_fits(family="rf", scoring="log-rmse"):
    return {f"{family}__{arm}__{scoring}": stub_fit(family, arm, scale, scoring)
            for arm, scale in ARM_SCALES.items()}


def stub_test(n=60, seed=3, name="test-natural"):
    frame = data.add_features(synthetic_final(n, seed))
    return data.TestSet(name=name, X=frame[list(config.FEATURES)].reset_index(drop=True),
                        y=frame["FOS"].to_numpy(float), ids=np.array([f"{name}:{i}" for i in range(n)]))


EVAL_KWARGS = dict(alpha=0.10, zones=config.ZONE_EDGES, comparisons=config.COMPARISONS,
                   selection_range=config.SELECTION_RANGE, n_boot=300, bootstrap_seed=1,
                   bootstrap_ci=0.95)


def write_all_finals(monkeypatch, n=60):
    """Small, valid, pairwise-disjoint final sets for every role."""
    shrink_roles(monkeypatch, {role: n for role in config.ROLE_ORDER})
    for i, role in enumerate(config.ROLE_ORDER):
        write_final(role, synthetic_final(n, 200 + i))
```

- [ ] **Step 2: Write the failing tests `tests/test_evaluate.py`**

```python
import json
import math

import numpy as np
import pytest

from sgs import config, evaluate, models
from tests.helpers import EVAL_KWARGS, stub_fit, stub_fits, stub_test, ARM_SCALES, write_all_finals


def _pred(y_true, y_pred, arm="C", family="rf", test="test-natural", ids=None):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ids = np.array([f"{test}:{i}" for i in range(len(y_true))]) if ids is None else np.asarray(ids)
    return evaluate.Prediction(family=family, arm=arm, scoring="log-rmse", test=test,
                               ids=ids, y_true=y_true, y_pred=y_pred)


# --- primitives -------------------------------------------------------------------------

def test_global_metrics():
    m = evaluate.global_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert m["mae"] == pytest.approx(2 / 3) and m["bias"] == pytest.approx(2 / 3)
    assert m["max_abs_error"] == 2.0 and m["r2"] == pytest.approx(1 - 4 / 2) and m["n"] == 3


def test_zone_labels_and_half_open_bounds():
    labels = [z[0] for z in evaluate.zones(config.ZONE_EDGES)]
    assert labels == ["FOS < 1", "1–1.3", "1.3–3", "3–6", "FOS ≥ 6"]
    rows = evaluate.zone_metrics(np.array([1.0, 1.3, 6.0]), np.array([1.0, 1.3, 6.0]), config.ZONE_EDGES)
    assert [r["n"] for r in rows] == [0, 1, 1, 0, 1]
    assert rows[0]["rmse"] is None and rows[0]["lo"] is None and rows[-1]["hi"] is None


def test_minimax_is_the_worst_zone_rmse():
    y = np.array([0.8, 0.9, 1.1, 1.2, 2.0, 2.5, 4.0, 7.0])
    p = y + np.array([0.1, -0.1, 0.0, 0.0, 0.5, -0.5, 0.2, 0.0])
    worst = evaluate.minimax(evaluate.zone_metrics(y, p, config.ZONE_EDGES))
    assert worst == {"zone": "1.3–3", "rmse": pytest.approx(0.5)}


def test_conformal_quantile_is_an_order_statistic():
    scores = np.arange(1, 21, dtype=float)                 # n = 20
    # k = min(ceil(21 * 0.9), 20) = 19 -> the 19th smallest value
    assert evaluate.conformal_quantile(scores, 0.10) == 19.0
    assert np.quantile(scores, 19 / 20) != 19.0            # interpolation would differ
    assert evaluate.conformal_quantile(np.array([3.0, 1.0, 2.0]), 0.10) == 3.0   # k capped at n


def test_conformal_intervals_are_multiplicative():
    y_pred = np.array([1.0, 2.0, 2.5, 8.0])
    y_true = y_pred * np.exp([0.1, -0.2, 0.6, -0.05])   # zones 1–1.3, 1.3–3, 3–6, ≥6
    out = evaluate.conformal(_pred(y_true, y_pred), np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]),
                             alpha=0.10, edges=config.ZONE_EDGES)
    assert out["q_log"] == 0.5 and out["factor"] == pytest.approx(math.exp(0.5))
    assert out["coverage"] == 0.75                          # |0.6| > 0.5 is the only miss
    assert "no formal guarantee" in out["procedure"]
    assert [z["n"] for z in out["zones"]] == [0, 1, 1, 1, 1]


def test_subset_mask():
    y = np.array([0.4, 0.5, 3.0, 6.0, 6.1])
    assert evaluate.subset_mask(y, "all", (0.5, 6.0)).all()
    assert evaluate.subset_mask(y, "selection-range", (0.5, 6.0)).tolist() == [False, True, True, True, False]
    with pytest.raises(ValueError):
        evaluate.subset_mask(y, "band", (0.5, 6.0))


def test_predict_exponentiates_log_models():
    test = stub_test(10)
    pred = evaluate.predict(stub_fit("rf", "C", 0.0), test)
    np.testing.assert_allclose(pred.y_pred, test.y)
    assert pred.test == "test-natural" and (pred.ids == test.ids).all()


# --- paired comparisons -------------------------------------------------------------------

REG = config.COMPARISONS


def _pair(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.5, 6.0, n)
    better = _pred(y, y + rng.normal(0, 0.05, n), arm="C")
    worse = _pred(y, y + rng.normal(0, 0.30, n), arm="B")
    return better, worse


def test_registry_lookup_is_directional():
    assert evaluate.registered("C", "B", REG).alternative == "less"
    with pytest.raises(evaluate.UnregisteredComparison):
        evaluate.registered("B", "C", REG)
    with pytest.raises(evaluate.UnregisteredComparison):
        evaluate.registered("A", "B", REG)


def test_wilcoxon_on_differences_of_absolute_errors():
    c, b = _pair()
    out = evaluate.wilcoxon_abs(c, b, registry=REG)
    d = np.abs(c.y_pred - c.y_true) - np.abs(b.y_pred - b.y_true)
    assert out["median_d"] == pytest.approx(np.median(d)) and out["median_d"] < 0
    assert out["p_d_lt_0"] == pytest.approx(np.mean(d < 0))
    assert out["alternative"] == "less" and out["hypothesised"] and out["pvalue"] < 0.01
    assert "smaller absolute errors" in out["reading"]


def test_wilcoxon_is_invariant_to_row_order():
    c, b = _pair()
    perm = np.random.default_rng(1).permutation(len(c.ids))
    shuffle = lambda p: _pred(p.y_true[perm], p.y_pred[perm], arm=p.arm, ids=p.ids[perm])
    a1 = evaluate.wilcoxon_abs(c, b, registry=REG)
    a2 = evaluate.wilcoxon_abs(shuffle(c), shuffle(b), registry=REG)
    assert a1["statistic"] == a2["statistic"] and a1["pvalue"] == a2["pvalue"]


@pytest.mark.parametrize("break_it, message", [
    (lambda b: _pred(b.y_true[:-1], b.y_pred[:-1], arm="B"), "identifiers"),
    (lambda b: _pred(b.y_true, b.y_pred, arm="B", ids=b.ids[::-1]), "identifiers"),
    (lambda b: _pred(b.y_true, b.y_pred, arm="B", test="test-stratified"), "test set"),
    (lambda b: _pred(b.y_true, b.y_pred, arm="B", family="gbm"), "vertical"),
    (lambda b: _pred(b.y_true + 1, b.y_pred, arm="B"), "observed FOS"),
])
def test_misaligned_pairs_are_refused(break_it, message):
    c, b = _pair()
    with pytest.raises(evaluate.MisalignedPair, match=message):
        evaluate.wilcoxon_abs(c, break_it(b), registry=REG)
    with pytest.raises(evaluate.MisalignedPair, match=message):
        evaluate.paired_bootstrap(c, break_it(b), registry=REG, n_boot=50, seed=1, ci=0.95)


def test_bootstrap_is_reproducible_and_states_its_limitation():
    c, b = _pair()
    one = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=500, seed=7, ci=0.95)
    two = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=500, seed=7, ci=0.95)
    other = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=500, seed=8, ci=0.95)
    assert one == two and one["ci"] != other["ci"]
    assert one["delta_rmse"] == pytest.approx(evaluate.rmse(c.y_true, c.y_pred) - evaluate.rmse(b.y_true, b.y_pred))
    assert one["ci"][0] <= one["delta_rmse"] <= one["ci"][1] and one["ci"][1] < 0
    assert "test-sampling variability only" in one["limitation"]


def test_mask_restricts_the_test():
    c, b = _pair()
    mask = c.y_true <= 3.0
    assert evaluate.wilcoxon_abs(c, b, registry=REG, mask=mask)["n"] == int(mask.sum())


# --- evaluate() and run() -------------------------------------------------------------------

def test_evaluate_covers_every_arm_and_registered_comparison():
    fits = {**stub_fits("rf"), **stub_fits("gbm")}
    result = evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)
    assert set(result.arms) == {"rf", "gbm"}
    assert set(result.arms["rf"]) == {"A", "B", "C", "C0"}
    assert len(result.comparisons) == 2 * len(config.COMPARISONS) * len(evaluate.SUBSETS)
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["predictions"]["row_id"][0] == "test-natural:0"
    assert "rf__C0" in payload["predictions"]
    assert "test-sampling variability only" in payload["notes"]["bootstrap"]


def test_evaluate_refuses_a_missing_arm():
    fits = stub_fits("rf")
    fits.pop("rf__C0__log-rmse")
    with pytest.raises(ValueError, match="C0"):
        evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)


def test_evaluate_refuses_mixed_scorings():
    fits = {**stub_fits("rf"), **stub_fits("gbm", scoring="fos-rmse")}
    with pytest.raises(ValueError, match="one scoring"):
        evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)


def test_run_writes_metrics_with_provenance(root, monkeypatch):
    write_all_finals(monkeypatch)
    for family in config.FAMILIES:
        for arm, scale in ARM_SCALES.items():
            models.save_fit(stub_fit(family, arm, scale))
    monkeypatch.setattr(config, "N_BOOTSTRAP", 200)
    evaluate.run()
    for test in config.TEST_ROLES:
        metrics = json.loads(config.metrics_path(test, "log-rmse").read_text())
        assert metrics["test"] == test and set(metrics["arms"]["gbm"]) == {"A", "B", "C", "C0"}
        assert set(metrics["provenance"]["final_sha256"]) == set(config.ROLE_ORDER)
        assert set(metrics["provenance"]["fit_fingerprints"]) == {"rf__log-rmse", "gbm__log-rmse"}
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_evaluate.py -q`
Expected: `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 4: Write `sgs/evaluate.py`**

```python
"""Metrics on a test set. Pure functions: everything arrives as an argument.

Only ``run`` (and ``run_learning_curve``, Task 13) read files: they load fits
and test sets and pass them in explicitly. Predictions carry their test name
and row identifiers, so errors from one test set can never be paired with
observations from another.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import stats

from sgs import config, data, models
from sgs.models import FitArtifact

SUBSETS = ("all", "selection-range")
CONFORMAL_PROCEDURE = ("Cross-conformal-style: nested-CV out-of-fold residuals on log(FOS), "
                       "multiplicative intervals around the refit model. Coverage is reported "
                       "empirically; no formal guarantee is claimed.")
BOOTSTRAP_LIMITATION = ("Paired bootstrap over test rows: covers test-sampling variability only, "
                        "not seed or selection variability.")


# --- predictions and point metrics --------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    family: str
    arm: str
    scoring: str
    test: str
    ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def predict(fit: FitArtifact, test: data.TestSet) -> Prediction:
    if fit.target != "log":
        raise ValueError(f"unknown target convention {fit.target!r}")
    raw = fit.model.predict(test.X[list(config.FEATURES)])
    return Prediction(family=fit.family, arm=fit.arm, scoring=fit.scoring, test=test.name,
                      ids=test.ids, y_true=test.y, y_pred=np.exp(raw))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def global_metrics(y, p) -> dict:
    e = p - y
    return {"n": int(len(y)),
            "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse": rmse(y, p),
            "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()),
            "max_abs_error": float(np.abs(e).max())}


def zones(edges) -> list[tuple[str, float, float]]:
    bounds = [-math.inf, *edges, math.inf]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo == -math.inf:
            label = f"FOS < {hi:g}"
        elif hi == math.inf:
            label = f"FOS ≥ {lo:g}"
        else:
            label = f"{lo:g}–{hi:g}"
        out.append((label, lo, hi))
    return out


def _zone_mask(y, lo, hi):
    return (y >= lo) & (y < hi)


def _finite(value):
    return None if math.isinf(value) else float(value)


def zone_metrics(y, p, edges) -> list[dict]:
    rows = []
    for label, lo, hi in zones(edges):
        mask = _zone_mask(y, lo, hi)
        n = int(mask.sum())
        row = {"zone": label, "lo": _finite(lo), "hi": _finite(hi), "n": n,
               "rmse": None, "mae": None, "bias": None}
        if n:
            e = p[mask] - y[mask]
            row.update(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.abs(e).mean()),
                       bias=float(e.mean()))
        rows.append(row)
    return rows


def minimax(zone_rows) -> dict:
    worst = max((r for r in zone_rows if r["n"]), key=lambda r: r["rmse"])
    return {"zone": worst["zone"], "rmse": worst["rmse"]}


def conformal_quantile(scores, alpha: float) -> float:
    """The k-th smallest score, k = min(ceil((n+1)(1-alpha)), n)."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    n = len(ordered)
    if n == 0:
        raise ValueError("no calibration scores")
    k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(ordered[k - 1])


def conformal(pred: Prediction, scores, *, alpha: float, edges) -> dict:
    q = conformal_quantile(scores, alpha)
    factor = math.exp(q)
    lo, hi = pred.y_pred / factor, pred.y_pred * factor
    covered = (pred.y_true >= lo) & (pred.y_true <= hi)
    width = hi - lo
    per_zone = []
    for label, zlo, zhi in zones(edges):
        mask = _zone_mask(pred.y_true, zlo, zhi)
        n = int(mask.sum())
        per_zone.append({"zone": label, "n": n,
                         "coverage": float(covered[mask].mean()) if n else None,
                         "mean_width": float(width[mask].mean()) if n else None})
    return {"alpha": alpha, "q_log": q, "factor": factor, "n_calibration": int(len(scores)),
            "coverage": float(covered.mean()), "mean_width": float(width.mean()),
            "zones": per_zone, "procedure": CONFORMAL_PROCEDURE}


def subset_mask(y, subset: str, selection_range) -> np.ndarray:
    if subset == "all":
        return np.ones(len(y), dtype=bool)
    if subset == "selection-range":
        lo, hi = selection_range
        return (y >= lo) & (y <= hi)
    raise ValueError(f"unknown subset {subset!r}")


# --- paired comparisons ---------------------------------------------------------------------

class MisalignedPair(ValueError):
    """Two predictions that do not describe the same rows of the same test set."""


class UnregisteredComparison(KeyError):
    """A comparison absent from config.COMPARISONS."""


def registered(first: str, second: str, registry) -> config.Comparison:
    for comparison in registry:
        if (comparison.first, comparison.second) == (first, second):
            return comparison
    raise UnregisteredComparison(f"{first} vs {second} is not in config.COMPARISONS; "
                                 "the alternative cannot be chosen at reporting time")


def _check_aligned(p1: Prediction, p2: Prediction) -> None:
    if p1.test != p2.test:
        raise MisalignedPair(f"predictions come from different test sets ({p1.test}, {p2.test})")
    if (p1.family, p1.scoring) != (p2.family, p2.scoring):
        raise MisalignedPair("comparisons are vertical: same family and scoring, different arms")
    if len(p1.ids) != len(p2.ids) or not np.array_equal(p1.ids, p2.ids):
        raise MisalignedPair("rows are not paired: identifiers differ")
    if not np.array_equal(p1.y_true, p2.y_true):
        raise MisalignedPair("rows are not paired: observed FOS differs")


def _mask(p: Prediction, mask):
    return np.ones(len(p.ids), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


def wilcoxon_abs(p1: Prediction, p2: Prediction, *, registry, mask=None) -> dict:
    """Wilcoxon signed-rank on d = |e1| - |e2| with the registered alternative."""
    _check_aligned(p1, p2)
    comparison = registered(p1.arm, p2.arm, registry)
    m = _mask(p1, mask)
    d = np.abs(p1.y_pred[m] - p1.y_true[m]) - np.abs(p2.y_pred[m] - p2.y_true[m])
    result = stats.wilcoxon(d, alternative=comparison.alternative)
    return {"first": p1.arm, "second": p2.arm, "n": int(m.sum()),
            "median_d": float(np.median(d)), "p_d_lt_0": float(np.mean(d < 0)),
            "statistic": float(result.statistic), "pvalue": float(result.pvalue),
            "alternative": comparison.alternative, "hypothesised": comparison.hypothesised,
            "reading": f"d = |e_{p1.arm}| - |e_{p2.arm}|; d < 0 means {p1.arm} has smaller absolute errors"}


def paired_bootstrap(p1: Prediction, p2: Prediction, *, registry, n_boot: int, seed: int,
                     ci: float, mask=None) -> dict:
    """Percentile interval for RMSE(first) - RMSE(second), rows resampled jointly."""
    _check_aligned(p1, p2)
    registered(p1.arm, p2.arm, registry)
    m = _mask(p1, mask)
    y, a, b = p1.y_true[m], p1.y_pred[m], p2.y_pred[m]
    draws = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    delta = (np.sqrt(((a[draws] - y[draws]) ** 2).mean(axis=1))
             - np.sqrt(((b[draws] - y[draws]) ** 2).mean(axis=1)))
    tail = (1 - ci) / 2
    return {"delta_rmse": rmse(y, a) - rmse(y, b),
            "ci": [float(np.quantile(delta, tail)), float(np.quantile(delta, 1 - tail))],
            "ci_level": ci, "p_delta_lt_0": float(np.mean(delta < 0)), "n_boot": int(n_boot),
            "limitation": BOOTSTRAP_LIMITATION}


# --- one test set, every arm ----------------------------------------------------------------

@dataclass
class EvalResult:
    test: str
    scoring: str
    n: int
    fos_range: tuple[float, float]
    arms: dict
    comparisons: list
    predictions: dict

    def to_json(self) -> dict:
        return {
            "test": self.test, "scoring": self.scoring, "n": self.n,
            "fos_range": list(self.fos_range), "arms": self.arms,
            "comparisons": self.comparisons, "predictions": self.predictions,
            "notes": {
                "wilcoxon": "Signed-rank test on d = |e_first| - |e_second|; d < 0 means the first "
                            "arm has smaller absolute errors. Alternatives are fixed in config.COMPARISONS.",
                "bootstrap": BOOTSTRAP_LIMITATION,
                "conformal": CONFORMAL_PROCEDURE,
            },
        }


def evaluate(fits: Mapping[str, FitArtifact], test: data.TestSet, *, alpha, zones, comparisons,
             selection_range, n_boot, bootstrap_seed, bootstrap_ci) -> EvalResult:
    scorings = {fit.scoring for fit in fits.values()}
    if len(scorings) != 1:
        raise ValueError(f"evaluate takes fits of exactly one scoring, got {sorted(scorings)}")
    (scoring,) = scorings

    preds, arms = {}, {}
    for fit in fits.values():
        pred = predict(fit, test)
        preds[(fit.family, fit.arm)] = pred
        zone_rows = zone_metrics(pred.y_true, pred.y_pred, zones)
        arms.setdefault(fit.family, {})[fit.arm] = {
            "global": global_metrics(pred.y_true, pred.y_pred),
            "zones": zone_rows,
            "minimax": minimax(zone_rows),
            "conformal": conformal(pred, fit.scores, alpha=alpha, edges=zones),
            "best_params": fit.best_params,
            "n_train": int(len(fit.train_ids)),
        }

    rows = []
    for family in sorted(arms):
        for comparison in comparisons:
            first = preds.get((family, comparison.first))
            second = preds.get((family, comparison.second))
            if first is None or second is None:
                raise ValueError(f"{family}: {comparison.first} vs {comparison.second} needs both arms; "
                                 "every arm, C0 included, must be evaluated")
            for subset in SUBSETS:
                mask = subset_mask(test.y, subset, selection_range)
                rows.append({
                    "family": family, "subset": subset,
                    "first": comparison.first, "second": comparison.second,
                    "wilcoxon": wilcoxon_abs(first, second, registry=comparisons, mask=mask),
                    "bootstrap": paired_bootstrap(first, second, registry=comparisons, n_boot=n_boot,
                                                  seed=bootstrap_seed, ci=bootstrap_ci, mask=mask),
                })

    predictions = {"row_id": test.ids.tolist(), "y_true": test.y.tolist()}
    for (family, arm), pred in sorted(preds.items()):
        predictions[f"{family}__{arm}"] = pred.y_pred.tolist()
    return EvalResult(test=test.name, scoring=scoring, n=int(len(test.y)),
                      fos_range=(float(test.y.min()), float(test.y.max())),
                      arms=arms, comparisons=rows, predictions=predictions)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def run() -> None:
    """Every active scoring on every test set; every family and arm, C0 included."""
    data.validate_sets(config.ROLE_ORDER)
    provenance_data = {role: data.sha256_file(config.final_path(role)) for role in config.ROLE_ORDER}
    for scoring in config.ACTIVE_SCORINGS:
        fits = {f"{f}__{a}__{scoring}": models.load_fit(f, a, scoring)
                for f in config.FAMILIES for a in config.ARMS}
        for test_role in config.TEST_ROLES:
            result = evaluate(fits, data.load_test(test_role), alpha=config.ALPHA,
                              zones=config.ZONE_EDGES, comparisons=config.COMPARISONS,
                              selection_range=config.SELECTION_RANGE, n_boot=config.N_BOOTSTRAP,
                              bootstrap_seed=config.SEEDS["bootstrap_resampling"],
                              bootstrap_ci=config.BOOTSTRAP_CI)
            payload = result.to_json()
            payload["provenance"] = {
                "final_sha256": provenance_data,
                "fit_fingerprints": {f"{f}__{scoring}": config.fit_fingerprint(f, scoring)
                                     for f in config.FAMILIES},
                "environment": data.environment(),
            }
            path = config.metrics_path(test_role, scoring)
            _write_json(path, payload)
            print(f"[evaluate] wrote {path.name}")
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Run on the real fits and commit**

```bash
.venv/bin/sgs evaluate
```
Expected: `wrote test-natural__log-rmse.json` and `wrote test-stratified__log-rmse.json`, in seconds.

```bash
git add sgs/evaluate.py tests/test_evaluate.py tests/helpers.py outputs/metrics/
git commit -m "Add evaluation: zone metrics, order-statistic conformal, Wilcoxon on
absolute-error differences and a paired bootstrap

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 12: `report.py`, the narrative notebook, final verification

**Files:**
- Create: `sgs/report.py`, `tests/test_report.py`, `notebooks/surrogate_guided_sampling.ipynb`
- Modify: `tests/conftest.py` (Agg backend), `README.md` (notebook section)

**Interfaces:**
- Consumes: only `outputs/metrics/*.json` and `config` (arm order, ablation flags, budgets).
- Produces: `arm_label(arm)`, `load_metrics(test, scoring)`, `load_learning_curve()`, `booktabs(frame, *, caption, label, digits=3) -> str`, `global_frame(metrics)`, `zone_frame(metrics, family)`, `comparison_frame(metrics)`, `conformal_frame(metrics)`, `budget_frame()`, `fig_zone_rmse(metrics, family)`, `fig_learning_curve(lc, test)`, `run()`.
- Learning-curve metrics JSON (written in Task 13, read here): `{"procedure": str, "rows": [{"family", "curve", "budget", "test", "n_repeats", "rmse_mean", "rmse_std", "rmse_selection_mean", "rmse_selection_std"}]}`.

- [ ] **Step 1: Make matplotlib headless in tests — prepend to `tests/conftest.py`**

```python
import matplotlib

matplotlib.use("Agg")
```

- [ ] **Step 2: Write the failing tests `tests/test_report.py`**

```python
import json

import pandas as pd

from sgs import config, evaluate, report
from tests.helpers import EVAL_KWARGS, stub_fits, stub_test


def _write_metrics():
    result = evaluate.evaluate({**stub_fits("rf"), **stub_fits("gbm")}, stub_test(), **EVAL_KWARGS)
    path = config.metrics_path("test-natural", "log-rmse")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json()))
    return result.to_json()


def test_booktabs_structure_and_escaping():
    tex = report.booktabs(pd.DataFrame({"arm": ["C_0"], "RMSE": [0.12345], "n": [None]}),
                          caption="Demo.", label="tab:demo")
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"C\_0 & 0.123 & --" in tex and r"\label{tab:demo}" in tex


def test_budget_table_excludes_the_ablation():
    frame = report.budget_frame()
    assert frame["arm"].tolist() == ["A", "B", "C"]
    assert frame["solver budget"].tolist() == [300, 600, 600]


def test_every_results_table_includes_the_ablation(root):
    metrics = _write_metrics()
    assert "C0 (ablation)" in report.global_frame(metrics)["arm"].tolist()
    assert "C0 (ablation)" in report.zone_frame(metrics, "rf").columns
    assert "C0 (ablation)" in report.conformal_frame(metrics)["arm"].tolist()
    assert "C vs C0" in report.comparison_frame(metrics)["comparison"].tolist()


def test_learning_curve_figure_reads_the_rows():
    rows = [{"family": "rf", "curve": curve, "budget": b, "test": "test-natural", "n_repeats": 2,
             "rmse_mean": 1.0 / b, "rmse_std": 0.01, "rmse_selection_mean": 0.5 / b,
             "rmse_selection_std": 0.01}
            for curve in ("lhs", "guided") for b in (100, 200)]
    fig = report.fig_learning_curve({"rows": rows}, "test-natural")
    assert len(fig.axes[0].lines) == 2


def test_run_writes_tables_and_figures(root):
    _write_metrics()
    report.run()
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    assert {"budget.tex", "test-natural__log-rmse__global.tex", "test-natural__log-rmse__comparisons.tex",
            "test-natural__log-rmse__conformal.tex", "test-natural__log-rmse__zones_rf.tex"} <= tables
    figure = config.figures_dir() / "test-natural__log-rmse__zones_gbm.pdf"
    assert figure.read_bytes()[:4] == b"%PDF"
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_report.py -q`
Expected: `ImportError: cannot import name 'report'`.

- [ ] **Step 4: Write `sgs/report.py`**

```python
"""Booktabs tables and figures for the manuscript.

Reads only outputs/metrics/ and recomputes no metric, so no number reaches a
table without having passed through ``evaluate``. Every arm appears in every
results table, C0 included and labelled as the ablation; the budget table
lists budget arms only.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sgs import config


def arm_label(arm: str) -> str:
    return f"{arm} (ablation)" if config.ARMS[arm].ablation else arm


def _read(path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs evaluate`")
    return json.loads(path.read_text())


def load_metrics(test: str, scoring: str) -> dict:
    return _read(config.metrics_path(test, scoring))


def load_learning_curve() -> dict:
    return _read(config.learning_curve_metrics_path())


def _escape(text: str) -> str:
    for raw, safe in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(raw, safe)
    return text


def _cell(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return _escape(str(value))


def booktabs(frame: pd.DataFrame, *, caption: str, label: str, digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{l{'r' * (len(columns) - 1)}}}", r"\toprule",
             " & ".join(_escape(str(c)) for c in columns) + r" \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(_cell(v, digits) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def global_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for family, arms in metrics["arms"].items():
        for arm in config.ARMS:
            m = arms[arm]
            g = m["global"]
            rows.append({"family": family, "arm": arm_label(arm), "R2": g["r2"], "RMSE": g["rmse"],
                         "MAE": g["mae"], "bias": g["bias"], "max |e|": g["max_abs_error"],
                         "minimax": m["minimax"]["rmse"], "worst zone": m["minimax"]["zone"]})
    return pd.DataFrame(rows)


def zone_frame(metrics: dict, family: str) -> pd.DataFrame:
    arms = metrics["arms"][family]
    reference = arms[next(iter(config.ARMS))]["zones"]
    table = {"zone": [z["zone"] for z in reference], "n": [z["n"] for z in reference]}
    for arm in config.ARMS:
        table[arm_label(arm)] = [z["rmse"] for z in arms[arm]["zones"]]
    return pd.DataFrame(table)


def comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for c in metrics["comparisons"]:
        w, b = c["wilcoxon"], c["bootstrap"]
        rows.append({"family": c["family"], "subset": c["subset"],
                     "comparison": f"{c['first']} vs {c['second']}", "alternative": w["alternative"],
                     "n": w["n"], "median d": w["median_d"], "P(d<0)": w["p_d_lt_0"], "p": w["pvalue"],
                     "ΔRMSE": b["delta_rmse"], "CI low": b["ci"][0], "CI high": b["ci"][1]})
    return pd.DataFrame(rows)


def conformal_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for family, arms in metrics["arms"].items():
        for arm in config.ARMS:
            c = arms[arm]["conformal"]
            row = {"family": family, "arm": arm_label(arm), "factor": c["factor"],
                   "coverage": c["coverage"], "mean width": c["mean_width"]}
            for z in c["zones"]:
                row[f"cov {z['zone']}"] = z["coverage"]
                row[f"width {z['zone']}"] = z["mean_width"]
            rows.append(row)
    return pd.DataFrame(rows)


def budget_frame() -> pd.DataFrame:
    return pd.DataFrame([{"arm": arm, "training sets": " + ".join(spec.roles), "solver budget": spec.budget}
                         for arm, spec in config.ARMS.items() if not spec.ablation])


def fig_zone_rmse(metrics: dict, family: str):
    frame = zone_frame(metrics, family)
    labels = [arm_label(a) for a in config.ARMS]
    x = np.arange(len(frame))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, label in enumerate(labels):
        ax.bar(x + i * width, frame[label].astype(float), width, label=label)
    ax.set_xticks(x + width * (len(labels) - 1) / 2, frame["zone"])
    ax.set_ylabel("RMSE (FOS)")
    ax.set_title(f"{metrics['test']} — {family}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_learning_curve(lc: dict, test: str):
    rows = [r for r in lc["rows"] if r["test"] == test]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for curve in sorted({r["curve"] for r in rows}):
        points = sorted((r for r in rows if r["curve"] == curve), key=lambda r: r["budget"])
        budgets = [r["budget"] for r in points]
        for ax, key in zip(axes, ("rmse", "rmse_selection")):
            mean = np.array([r[f"{key}_mean"] for r in points])
            std = np.array([r[f"{key}_std"] for r in points])
            ax.plot(budgets, mean, "o-", label=curve)
            ax.fill_between(budgets, mean - std, mean + std, alpha=0.15)
    axes[0].set_title("Global RMSE")
    axes[1].set_title("RMSE in the selection range")
    for ax in axes:
        ax.set_xlabel("Solver budget")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"RMSE on {test} (RF, fixed hyper-parameters)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def _save_table(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_figure(path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No creation date, so an unchanged figure is byte-identical in git.
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def run() -> None:
    tables, figures = config.tables_dir(), config.figures_dir()
    _save_table(tables / "budget.tex", booktabs(
        budget_frame(), caption="Budget-fair arms. C0 is an ablation and carries no budget.",
        label="tab:budget"))

    for scoring in config.ACTIVE_SCORINGS:
        for test in config.TEST_ROLES:
            if not config.metrics_path(test, scoring).exists():
                print(f"[report] {config.metrics_path(test, scoring).name} missing; run sgs evaluate")
                continue
            metrics = load_metrics(test, scoring)
            stem = f"{test}__{scoring}"
            _save_table(tables / f"{stem}__global.tex", booktabs(
                global_frame(metrics), caption=f"Global metrics on {test}.", label=f"tab:{stem}:global"))
            _save_table(tables / f"{stem}__comparisons.tex", booktabs(
                comparison_frame(metrics), caption=f"Paired comparisons on {test}: Wilcoxon on "
                "d = |e1| - |e2| (smaller absolute errors) and bootstrap ΔRMSE.",
                label=f"tab:{stem}:comparisons"))
            _save_table(tables / f"{stem}__conformal.tex", booktabs(
                conformal_frame(metrics), caption=f"Cross-conformal-style intervals on {test}, "
                "empirical coverage.", label=f"tab:{stem}:conformal"))
            for family in metrics["arms"]:
                _save_table(tables / f"{stem}__zones_{family}.tex", booktabs(
                    zone_frame(metrics, family), caption=f"Zone RMSE on {test}, {family}.",
                    label=f"tab:{stem}:zones:{family}"))
                _save_figure(figures / f"{stem}__zones_{family}.pdf", fig_zone_rmse(metrics, family))

    if config.learning_curve_metrics_path().exists():
        lc = load_learning_curve()
        for test in config.TEST_ROLES:
            _save_figure(figures / f"learning_curve__{test}.pdf", fig_learning_curve(lc, test))
    print(f"[report] wrote {tables} and {figures}")
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Generate the notebook** (run once; the script itself is not committed)

```bash
.venv/bin/python - <<'EOF'
import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = [
    md("""
# Surrogate-guided sampling for slope-stability surrogates — results

Narrative and figures only. Every number shown here is read from `outputs/metrics/`,
written by `sgs evaluate`; nothing in this notebook fits a model or computes a metric.
To regenerate the numbers, run the pipeline (see README).
"""),
    code("""
# On Colab: fetch the repository; the metrics are tracked in it.
import os, subprocess, sys

if "google.colab" in sys.modules and not os.path.exists("sgs"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/bteodoru/surrogate-guided-slope-stability.git"], check=True)
    os.chdir("surrogate-guided-slope-stability")
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
"""),
    code("""
from IPython.display import display
from sgs import config, report

SCORING = config.ACTIVE_SCORINGS[0]
"""),
    md("""
## Budget-fair arms

A, B and C are compared at their solver budgets; C0 (guided points only) is an ablation
and appears in every results table below, but not in this one.
"""),
    code("display(report.budget_frame())"),
    md("""
## Results per test set

The two test sets are two deployment scenarios: `test-natural` samples the input space,
`test-stratified` samples the selection range of FOS uniformly.
"""),
    code("""
for test in config.TEST_ROLES:
    metrics = report.load_metrics(test, SCORING)
    print(f"== {test}: n = {metrics['n']}, FOS {metrics['fos_range'][0]:.2f}–{metrics['fos_range'][1]:.2f}")
    display(report.global_frame(metrics).round(3))
    for family in metrics["arms"]:
        display(report.zone_frame(metrics, family).round(3))
        report.fig_zone_rmse(metrics, family)
"""),
    md("""
## Paired comparisons

Wilcoxon signed-rank on d = |e₁| − |e₂| (read as *smaller absolute errors*), with the
alternative fixed per pair in `config.COMPARISONS`, and a paired bootstrap of ΔRMSE,
which covers test-sampling variability only.
"""),
    code("""
for test in config.TEST_ROLES:
    display(report.comparison_frame(report.load_metrics(test, SCORING)).round(4))
"""),
    md("""
## Intervals

Cross-conformal-style multiplicative intervals from nested-CV out-of-fold residuals;
coverage is reported empirically, without a formal guarantee.
"""),
    code("""
for test in config.TEST_ROLES:
    display(report.conformal_frame(report.load_metrics(test, SCORING)).round(3))
"""),
    md("""
## Learning curve (optional analysis)

RF with fixed hyper-parameters; present only after `sgs learning-curve`.
"""),
    code("""
if config.learning_curve_metrics_path().exists():
    lc = report.load_learning_curve()
    for test in config.TEST_ROLES:
        report.fig_learning_curve(lc, test)
"""),
]

notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}
with open("notebooks/surrogate_guided_sampling.ipynb", "w") as stream:
    json.dump(notebook, stream, indent=1, ensure_ascii=False)
    stream.write("\n")
EOF
```

- [ ] **Step 7: Append a notebook section to `README.md`**

```markdown
## Notebook

`notebooks/surrogate_guided_sampling.ipynb` holds the narrative and figures. It imports
`sgs.report` and reads `outputs/metrics/`; it fits nothing and computes no metric. On
Colab its first cell clones the repository, which already contains the metrics.
```

- [ ] **Step 8: Generate the real report and verify end to end**

```bash
.venv/bin/sgs report
ls outputs/tables outputs/figures
.venv/bin/pytest -q
.venv/bin/pytest -q -m slow
.venv/bin/sgs all --workers 2
git status --short
```
Expected: `budget.tex`, three tables plus two zone tables per test set, and zone figures. All tests pass, and the slow solver test passes. `sgs all` reports every data set and fit as up to date, rewrites the metrics, tables and figures, and `git status` shows no change to `outputs/` or `data/run_metadata.json` beyond the `updated` timestamps: the run is deterministic. No CSV or joblib file appears in `git status`.

- [ ] **Step 9: Commit**

```bash
git add sgs/report.py tests/test_report.py tests/conftest.py notebooks/surrogate_guided_sampling.ipynb README.md outputs/ data/run_metadata.json
git commit -m "Add booktabs tables, figures and the narrative notebook

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 13 (optional): RF learning curve with fixed hyper-parameters

Spec §8. RF only, `ESTIMATOR_FIXED["rf"]` plus scikit-learn defaults, declared in Methods. Not part of `sgs all`.

**Files:**
- Modify: `sgs/models.py` (append), `sgs/evaluate.py` (append), `tests/test_models.py` (append), `tests/test_evaluate.py` (append)

**Interfaces:**
- Consumes: `data.load_role/load_test/validate_sets/sha256_file`, `config.LC_*`, `config.learning_curve_fingerprint`, `config.learning_curve_path`.
- Produces:
  - In `models.py`: `SHARED = "seed-only"`, `lc_estimator()`, `unit_rng(budget, repeat)`, `subsample(seed_df, extra_df, budget, rng)`, `lc_plan(n_seed) -> list[(key, budget, repeats)]`, `lc_roles() -> list[str]`, `lc_state() -> dict`, `run_lc_budget(key, budget, repeats, seed_df, extras, tests) -> dict`, `learning_curve(*, force=False)`, `load_lc_units() -> dict[(curve, budget), list[unit]]`
  - In `evaluate.py`: `learning_curve_metrics(units, tests, *, selection_range) -> dict` (the JSON layout in Task 12), `run_learning_curve()`
  - Learning-curve file content: `{"key", "budget", "state": {"fingerprint", "data"}, "units": [{"repeat", "n_train", "train_ids", "predictions": {test: {"ids", "y_pred"}}}]}`.

- [ ] **Step 1: Append failing tests to `tests/test_models.py`**

```python
# --- learning curve ---------------------------------------------------------------------

import pandas as pd

from tests.helpers import shrink_roles


@pytest.fixture
def tiny_curve(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 40, "lhs-extension": 60, "guided": 60,
                               "test-natural": 30, "test-stratified": 30})
    for i, role in enumerate(("seed", "lhs-extension", "guided", "test-natural", "test-stratified")):
        write_final(role, synthetic_final(config.ROLES[role].n, 300 + i))
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20}, "gbm": {}})
    monkeypatch.setattr(config, "LC_BUDGETS", (20, 40, 60))
    monkeypatch.setattr(config, "LC_REPEATS", 2)


def test_lc_plan_shares_budgets_up_to_the_seed_size():
    assert models.lc_plan(300) == [("seed-only", 100, 10), ("seed-only", 200, 10), ("seed-only", 300, 1),
                                   ("lhs", 400, 10), ("guided", 400, 10), ("lhs", 500, 10),
                                   ("guided", 500, 10), ("lhs", 600, 10), ("guided", 600, 10)]


def test_subsample_semantics():
    seed_df = pd.DataFrame({"row_id": [f"s{i}" for i in range(10)]})
    extra = pd.DataFrame({"row_id": [f"e{i}" for i in range(10)]})
    small = models.subsample(seed_df, extra, 4, models.unit_rng(4, 0))
    assert len(small) == 4 and small["row_id"].str.startswith("s").all()
    assert models.subsample(seed_df, extra, 10, models.unit_rng(10, 0)) is seed_df
    large = models.subsample(seed_df, extra, 13, models.unit_rng(13, 0))
    assert len(large) == 13 and set(seed_df["row_id"]) <= set(large["row_id"])


def test_unit_rng_is_keyed_on_budget_and_repeat():
    draw = lambda b, r: models.unit_rng(b, r).integers(0, 10**9)
    assert draw(100, 0) == draw(100, 0)
    assert len({draw(100, 0), draw(100, 1), draw(200, 0)}) == 3


def test_lc_estimator_is_fixed_and_seeded():
    rf = models.lc_estimator()
    assert rf.random_state == config.SEEDS["protocol_estimator"]
    assert rf.max_features == 1.0 and rf.min_samples_leaf == 1


def test_learning_curve_writes_one_file_per_budget_and_refuses_stale(tiny_curve, monkeypatch):
    models.learning_curve()
    names = sorted(p.name for p in (config.ROOT / "artifacts" / "learning_curve").glob("*.joblib"))
    assert names == ["rf__guided__n60.joblib", "rf__lhs__n60.joblib",
                     "rf__seed-only__n20.joblib", "rf__seed-only__n40.joblib"]
    units = models.load_lc_units()
    assert units[("lhs", 20)] is units[("guided", 20)]
    assert len(units[("lhs", 40)]) == 1 and len(units[("lhs", 60)]) == 2
    monkeypatch.setattr(config, "LC_REPEATS", 3)
    with pytest.raises(models.StaleFit, match="--force"):
        models.learning_curve()
    with pytest.raises(models.StaleFit):
        models.load_lc_units()
    models.learning_curve(force=True)
    assert len(models.load_lc_units()[("lhs", 60)]) == 3
```

- [ ] **Step 2: Append failing tests to `tests/test_evaluate.py`**

```python
# --- learning-curve metrics -------------------------------------------------------------

def test_learning_curve_metrics_average_over_repeats():
    test = stub_test(20)
    good = {"repeat": 0, "predictions": {"test-natural": {"ids": test.ids, "y_pred": test.y}}}
    off = {"repeat": 1, "predictions": {"test-natural": {"ids": test.ids, "y_pred": test.y + 1.0}}}
    rows = evaluate.learning_curve_metrics({("lhs", 100): [good, off]}, {"test-natural": test},
                                           selection_range=config.SELECTION_RANGE)["rows"]
    assert len(rows) == 1 and rows[0]["n_repeats"] == 2 and rows[0]["family"] == "rf"
    assert rows[0]["rmse_mean"] == pytest.approx(0.5) and rows[0]["rmse_std"] == pytest.approx(0.5)


def test_learning_curve_metrics_refuse_misaligned_predictions():
    test = stub_test(20)
    unit = {"repeat": 0, "predictions": {"test-natural": {"ids": test.ids[::-1], "y_pred": test.y}}}
    with pytest.raises(evaluate.MisalignedPair):
        evaluate.learning_curve_metrics({("lhs", 100): [unit]}, {"test-natural": test},
                                        selection_range=config.SELECTION_RANGE)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_models.py tests/test_evaluate.py -q`
Expected: the new tests fail with `AttributeError` (`lc_plan`, `learning_curve_metrics`).

- [ ] **Step 4: Append to `sgs/models.py`**

```python
# --- learning curve (optional; RF with fixed hyper-parameters) ---------------------------

SHARED = "seed-only"


def lc_estimator() -> RandomForestRegressor:
    """ESTIMATOR_FIXED['rf'] plus scikit-learn defaults; not tuned (spec §8)."""
    return RandomForestRegressor(**config.ESTIMATOR_FIXED["rf"],
                                 random_state=config.SEEDS["protocol_estimator"], n_jobs=-1)


def unit_rng(budget: int, repeat: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(config.SEEDS["learning_curve_subsampling"],
                                      spawn_key=(budget, repeat))
    return np.random.default_rng(sequence)


def subsample(seed_df, extra_df, budget: int, rng):
    n_seed = len(seed_df)
    if budget < n_seed:
        return seed_df.iloc[rng.choice(n_seed, budget, replace=False)]
    if budget == n_seed:
        return seed_df
    take = extra_df.iloc[rng.choice(len(extra_df), budget - n_seed, replace=False)]
    return pd.concat([seed_df, take], ignore_index=True)


def lc_plan(n_seed: int) -> list[tuple[str, int, int]]:
    """Budgets up to the seed size do not depend on the strategy: fitted once.
    At the seed size every repeat would be identical: one repeat."""
    out = []
    for budget in config.LC_BUDGETS:
        repeats = 1 if budget == n_seed else config.LC_REPEATS
        if budget <= n_seed:
            out.append((SHARED, budget, repeats))
        else:
            out.extend((strategy, budget, repeats) for strategy in config.LC_STRATEGIES)
    return out


def lc_roles() -> list[str]:
    return [config.LC_BASE_ROLE, *config.LC_STRATEGIES.values(), *config.TEST_ROLES]


def lc_state() -> dict:
    return {"fingerprint": config.learning_curve_fingerprint(),
            "data": {role: data.sha256_file(config.final_path(role)) for role in lc_roles()}}


def run_lc_budget(key, budget, repeats, seed_df, extras, tests) -> dict:
    units = []
    for repeat in range(repeats):
        extra = seed_df.iloc[:0] if key == SHARED else extras[key]
        frame = subsample(seed_df, extra, budget, unit_rng(budget, repeat))
        model = lc_estimator().fit(frame[list(config.FEATURES)], np.log(frame["FOS"].to_numpy(float)))
        units.append({
            "repeat": repeat, "n_train": int(len(frame)), "train_ids": frame["row_id"].to_numpy(str),
            "predictions": {name: {"ids": test.ids, "y_pred": np.exp(model.predict(test.X))}
                            for name, test in tests.items()},
        })
    return {"key": key, "budget": budget, "units": units}


def learning_curve(*, force: bool = False) -> None:
    roles = lc_roles()
    data.validate_sets(roles)
    seed_df = data.load_role(config.LC_BASE_ROLE)
    extras = {strategy: data.load_role(role) for strategy, role in config.LC_STRATEGIES.items()}
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    state = lc_state()
    for key, budget, repeats in lc_plan(len(seed_df)):
        path = config.learning_curve_path(key, budget)
        if path.exists():
            if joblib.load(path)["state"] == state:
                print(f"[learning-curve] {path.name}: up to date")
                continue
            if not force:
                raise StaleFit(f"{path.name} was fitted under another configuration or other data; "
                               "rerun with --force")
        print(f"[learning-curve] {key} n={budget} x{repeats}", flush=True)
        _dump(path, {**run_lc_budget(key, budget, repeats, seed_df, extras, tests), "state": state})


def load_lc_units() -> dict:
    state = lc_state()
    out = {}
    for key, budget, _ in lc_plan(len(data.load_role(config.LC_BASE_ROLE))):
        path = config.learning_curve_path(key, budget)
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing; run `sgs learning-curve`")
        stored = joblib.load(path)
        if stored["state"] != state:
            raise StaleFit(f"{path.name} is stale; run `sgs learning-curve --force`")
        for curve in (tuple(config.LC_STRATEGIES) if key == SHARED else (key,)):
            out[(curve, budget)] = stored["units"]
    return out
```

Add `import pandas as pd` to the imports at the top of `sgs/models.py`.

- [ ] **Step 5: Append to `sgs/evaluate.py`**

```python
# --- learning curve -------------------------------------------------------------------------

LEARNING_CURVE_PROCEDURE = ("Random forest with fixed hyper-parameters (config.ESTIMATOR_FIXED['rf'] "
                            "plus scikit-learn defaults), not tuned; declared in Methods.")


def learning_curve_metrics(units: Mapping, tests: Mapping[str, data.TestSet], *, selection_range) -> dict:
    """Mean and spread of RMSE over repeats, per (curve, budget, test)."""
    rows = []
    for (curve, budget), unit_list in sorted(units.items()):
        for name, test in tests.items():
            mask = subset_mask(test.y, "selection-range", selection_range)
            overall, inside = [], []
            for unit in unit_list:
                pred = unit["predictions"][name]
                if not np.array_equal(pred["ids"], test.ids):
                    raise MisalignedPair(f"learning-curve predictions for {name} are not aligned with the test set")
                overall.append(rmse(test.y, pred["y_pred"]))
                inside.append(rmse(test.y[mask], pred["y_pred"][mask]))
            rows.append({"family": "rf", "curve": curve, "budget": int(budget), "test": name,
                         "n_repeats": len(unit_list),
                         "rmse_mean": float(np.mean(overall)), "rmse_std": float(np.std(overall)),
                         "rmse_selection_mean": float(np.mean(inside)),
                         "rmse_selection_std": float(np.std(inside))})
    return {"procedure": LEARNING_CURVE_PROCEDURE, "rows": rows}


def run_learning_curve() -> None:
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    result = learning_curve_metrics(models.load_lc_units(), tests, selection_range=config.SELECTION_RANGE)
    path = config.learning_curve_metrics_path()
    _write_json(path, result)
    print(f"[evaluate] wrote {path.name}")
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 7: Run it and regenerate the report**

```bash
.venv/bin/sgs learning-curve 2>&1 | tee /tmp/lc.log
.venv/bin/sgs report
```
Expected: 9 files under `artifacts/learning_curve/` (3 shared budgets, 6 strategy budgets), `wrote learning_curve.json`, then two `learning_curve__*.pdf` figures.

- [ ] **Step 8: Commit**

```bash
git add sgs/models.py sgs/evaluate.py tests/test_models.py tests/test_evaluate.py outputs/
git commit -m "Add the optional RF learning curve with fixed hyper-parameters

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Open items for the author (not tasks)

- `N_GUIDED_POOL`, `N_TEST_POOL`, `MIN_DISTANCE`, the log-scale selection criterion, the cross-conformal-style procedure and the fixed-parameter RF learning curve each need a sentence in Methods.
- Every published figure is regenerated from the new data. Compare the Task 8 descriptive output and the Task 10 hyper-parameter table with the numbers in circulation, and retire the old ones.
- The permutation-importance section of the old notebook is not ported: the spec lists no feature-importance analysis. If it returns, it belongs in `evaluate.py` (computed from fits and a test set), not in the notebook.
- Where `artifacts/fits/` lives at publication is still undecided. Nothing blocks on it.
