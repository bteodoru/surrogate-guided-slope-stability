# Pipeline Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic notebook and the standalone xslope scripts with an installable package `sgs` whose stages (design → solve → repair → fit-screener → check → fit → learning-curve → evaluate → report) are explicit, carry hash manifests, and fail loudly on stale inputs.

**Architecture:** A library core (`sgs/*.py`) with one module per responsibility, a single `argparse` CLI (`sgs <stage>`), and a guarded producer (`stage.produce`) that every stage uses: it verifies input provenance, refuses stale outputs unless `--force`, writes payloads atomically and writes a `*.manifest.json` sidecar. `config.py` is the only source of scientific values, hashed by section so that a change invalidates only the artifacts that depend on it.

**Tech Stack:** Python 3.14.4, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, shapely 2.1.2, openpyxl 3.1.5, xslope 0.5.2, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-09-18-pipeline-reorganization-design.md` — read it before starting any task. `CLAUDE.md` (rewritten in Task 2) holds the protocol disciplines.

## Global Constraints

- Python reference environment: **3.14.4** (`.venv/bin/python`). Every command below uses `.venv/bin/...`.
- No scientific value is defined outside `sgs/config.py`: seeds, ranges, grids, pool sizes, thresholds, zone edges, α. No scientific value is accepted on the command line; `--role` only selects a config entry.
- Execution flags are exactly: `--dry-run`, `--force`, `--resume`, `--workers`, `--only-family`, `--only-arm`, `--only-scoring` (plus `--role`).
- `data/` starts empty; every file in it is produced by a stage. CSV payloads under `data/` and `*.joblib`/`*.xlsx` under `artifacts/` are git-ignored; every `*.manifest.json` is tracked. `outputs/` is tracked.
- `data/results/` is immutable raw solver output; `repair` never rewrites its input.
- A stage that finds changed inputs fails and names what changed; `--force` is the only way past it.
- Config is hashed per section: `domain`, `design`, `screener`, `protocol`, `evaluation`, `learning_curve`.
- Target is `log(FOS)` everywhere; exponentiation only at the evaluation boundary.
- Estimators get `n_jobs=1`; the search gets `n_jobs=-1`.
- C₀ appears in every analysis. No threshold-misclassification code anywhere (no false-stable counts, no classification metrics, no comparison of a prediction against a fixed FOS).
- Conformal quantile is an order statistic `k = min(ceil((n+1)(1-α)), n)`; call the procedure *cross-conformal-style*, never claim a guarantee.
- Wilcoxon is on `d = |e₁| − |e₂|` with the per-pair alternative from `config.COMPARISONS`; phrase results as *smaller absolute errors*.
- The FOS interval [0.5, 6.0] is the **selection range** in code and prose — never "band", "near failure" or "critical band".
- `qmc.LatinHypercube` is always called with the `seed=` keyword. In SciPy 1.18 `seed=42` and `rng=42` produce different designs; the published ones were drawn with `seed=`.
- Reference copies of the current data live outside the repo in `REF=/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09`. Nothing in the pipeline reads from there; only `tools/` scripts do, for the author's descriptive comparison.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Deviations from the spec (small, deliberate; review before starting)

1. **Sixth config section `learning_curve`.** Budgets, repeats and the subsampling seed would otherwise live in `evaluation` or `protocol`, so changing α would invalidate hours of learning-curve fits — exactly the unrelated invalidation §4 argues against.
2. **Two protocol seeds added to `SEEDS`**: `protocol_cv_split` (42) and `protocol_estimator` (42). The spec's `SEEDS` block omits them, but the KFold shuffling and estimator `random_state` are seeds and must be named by role.
3. **`SetRole` gains `stem` and `disjoint_from`.** `stem` is the file name; `disjoint_from` lists the roles whose designs a role must stay away from. For `lhs` and `guided-maxmin` this is an assertion; for `stratified-random` it is the pre-selection filter the notebook applied. It also fixes the order in which designs have to be produced.
4. **Two extra modules.** `sgs/stage.py` holds `Options` and the guarded `produce()` shared by every stage. `sgs/solve.py` holds the batch/resume/parallel runner, so `solver.py` stays a thin xslope wrapper.
5. **Screener path is `artifacts/screeners/{name}.joblib`**, not `artifacts/screener.joblib`, because the registry is keyed by name.
6. **`requires-python = ">=3.12"`** in `pyproject.toml`, while 3.14.4 is pinned by `.python-version` and `requirements.lock`, and the CLI warns on any other version. An exact `==3.14.4` would stop the Colab notebook from installing the package at all.
7. **Learning curve is tuned with the same `searcher()`** as the arms (spec §10 phase 8 cost estimate implies this). The old notebook used an untuned RF. For budgets ≤ the seed size both curves share the same subsample, so those units are fitted once. At budget = seed size there is one repeat, because every repeat would be identical.

---

## File structure

```
pyproject.toml, requirements.lock, .python-version, .gitignore
CLAUDE.md, README.md, data/README.md
sgs/__init__.py        empty docstring; importing sgs imports nothing heavy
sgs/config.py          scientific truth + section hashing + path helpers
sgs/manifest.py        sha256, manifest read/write, staleness, provenance checks
sgs/stage.py           Options, produce(): guarded, atomic, manifest-writing
sgs/data.py            schema, derived feature, loading roles/arms/test sets
sgs/design.py          lhs, guided-maxmin, stratified-random, design stage
sgs/solver.py          xslope/Bishop wrapper (one analysis → one row)
sgs/solve.py           solve stage: batches, resume, parallel parts, merge
sgs/repair.py          repair stage
sgs/screener.py        preliminary RF: fit once, load thereafter
sgs/checks.py          per-set checks, pairwise disjointness, receipt
sgs/protocol.py        FitArtifact, searcher(), fit_arm()
sgs/fitting.py         fit stage, load_fit()
sgs/learning_curve.py  learning-curve stage
sgs/evaluate.py        pure metrics + evaluate stage
sgs/report.py          booktabs tables, figures
sgs/cli.py             subcommands, run-all
tests/helpers.py       synthetic data, FormulaModel, fixture writers
tests/conftest.py      `root` fixture (redirects config.ROOT to tmp)
tests/test_*.py        one file per module + discipline + notebook purity
tools/probe_environment.py   phase 0 only; deleted in Task 20
tools/describe_sets.py       author's descriptive comparison vs reference copies
notebooks/surrogate_guided_sampling.ipynb   narrative only
```

---

## Phase 0 — environment probe

### Task 1: Probe the local scikit-learn environment against the Colab results

**Files:**
- Create: `tools/probe_environment.py`
- Create: `docs/environment-probe-2026-09.md`
- External: copy the whole of `xslope/` and `surrogate_guided_sampling.ipynb` to `$REF`

**Interfaces:**
- Consumes: reference CSVs in `$REF` (old names).
- Produces: `docs/environment-probe-2026-09.md` with reference and observed hyper-parameters; the scikit-learn/joblib versions used in every later task.

- [ ] **Step 1: Create a working branch**

```bash
git checkout -b pipeline-reorganization
```

- [ ] **Step 2: Copy the reference material out of the repository (copy, not move — Task 2 deletes the originals)**

```bash
REF=/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09
mkdir -p "$REF"
rsync -a --exclude __pycache__ xslope/ "$REF/"
cp surrogate_guided_sampling.ipynb "$REF/surrogate_guided_sampling_published.ipynb"
ls "$REF"
```
Expected: the five `*_bishop.csv` data files, their `_part*`/`_backup`/` copy` residue, `run_analyses.py`, `run_repair.py`, `xslope_arrays.py`, `surrogate_guided_sampling_rev.ipynb`, `surrogate_guided_sampling_published.ipynb`, `xslope_simple1_v0.5.2.xlsx`.

- [ ] **Step 3: Install the analysis stack into the reference venv**

```bash
.venv/bin/pip install scikit-learn==1.9.1 joblib==1.6.0 pytest==9.1.1
.venv/bin/python -c "import sklearn, joblib; print(sklearn.__version__, joblib.__version__)"
```
Expected: `1.9.1 1.6.0`

- [ ] **Step 4: Write the reference record, taken from the stored outputs of `xslope/surrogate_guided_sampling_rev.ipynb` cell 38**

Create `docs/environment-probe-2026-09.md`:

```markdown
# Environment probe, September 2026

Temporary record for phase 0 of the pipeline reorganization. Deleted at the end
of phase 6 (Task 20), once the local fits have replaced the Colab ones.

## Reference: hyper-parameters selected on Colab

Source: stored output of cell 38 in `surrogate_guided_sampling_rev.ipynb`
(reference copy in `reference-data-2026-09/`). One `GridSearchCV` per
(family, arm), 5-fold `KFold(shuffle=True, random_state=42)`,
`scoring="neg_root_mean_squared_error"` on log(FOS).

| family | arm | n   | max_features | min_samples_leaf | learning_rate | max_iter | max_leaf_nodes |
|--------|-----|-----|--------------|------------------|---------------|----------|----------------|
| RF     | A   | 300 | 1.0          | 1                |               |          |                |
| RF     | C0  | 300 | 1.0          | 2                |               |          |                |
| RF     | C   | 600 | 1.0          | 1                |               |          |                |
| RF     | B   | 600 | 1.0          | 1                |               |          |                |
| GBM    | A   | 300 |              | 3                | 0.06          | 600      | 7              |
| GBM    | C0  | 300 |              | 5                | 0.03          | 600      | 7              |
| GBM    | C   | 600 |              | 3                | 0.06          | 600      | 7              |
| GBM    | B   | 600 |              | 3                | 0.10          | 600      | 7              |

## Observed locally

Filled in from the output of `tools/probe_environment.py` (Task 1, step 6).
```

- [ ] **Step 5: Write the probe script**

Create `tools/probe_environment.py`. It deliberately reproduces the notebook's code paths (not `sgs`, which does not exist yet), including its literals:

```python
"""Phase-0 environment probe. Temporary: deleted at the end of phase 6.

Re-runs locally the two computations whose results are stored in the notebook:
the hyper-parameter selection of the tuned searches, and the screener-driven
designs. Reads the reference copies of the CSV files, never the repository.

Usage: .venv/bin/python tools/probe_environment.py <reference-dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

REF = Path(sys.argv[1])
PARAMS = ["angle", "phi", "c", "gamma", "H"]
FEATURES = ["H", "angle", "phi", "c", "gamma", "gamma_H_c"]
RANGES = {"angle": (10.0, 65.0), "phi": (5.0, 40.0), "c": (1.0, 75.0),
          "gamma": (14.0, 22.0), "H": (1.0, 25.0)}
LO = np.array([RANGES[p][0] for p in PARAMS])
HI = np.array([RANGES[p][1] for p in PARAMS])

EXPECTED = {
    ("RF", "A"): {"max_features": 1.0, "min_samples_leaf": 1},
    ("RF", "C0"): {"max_features": 1.0, "min_samples_leaf": 2},
    ("RF", "C"): {"max_features": 1.0, "min_samples_leaf": 1},
    ("RF", "B"): {"max_features": 1.0, "min_samples_leaf": 1},
    ("GBM", "A"): {"learning_rate": 0.06, "max_iter": 600, "max_leaf_nodes": 7, "min_samples_leaf": 3},
    ("GBM", "C0"): {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 7, "min_samples_leaf": 5},
    ("GBM", "C"): {"learning_rate": 0.06, "max_iter": 600, "max_leaf_nodes": 7, "min_samples_leaf": 3},
    ("GBM", "B"): {"learning_rate": 0.10, "max_iter": 600, "max_leaf_nodes": 7, "min_samples_leaf": 3},
}


def load(name):
    df = pd.read_csv(REF / name, dtype={"excluded": "string"})
    if "excluded" in df.columns:
        df = df[df["excluded"].fillna("").str.strip() == ""]
    df = df[df["FOS"].notna()].reset_index(drop=True)
    df["gamma_H_c"] = df["gamma"] * df["H"] / df["c"]
    return df


def normalize(frame):
    return (frame[PARAMS].to_numpy(float) - LO) / (HI - LO)


def greedy_maxmin(Xs, k, rng):
    n = len(Xs)
    if k >= n:
        return np.arange(n)
    sel = [int(rng.integers(n))]
    dmin = np.linalg.norm(Xs - Xs[sel[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(dmin.argmax())
        sel.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Xs - Xs[nxt], axis=1))
    return np.array(sel)


def compare_inputs(label, produced, reference):
    diff = np.abs(produced[PARAMS].to_numpy(float) - reference[PARAMS].to_numpy(float))
    print(f"  {label}: n={len(produced)} vs {len(reference)}, max |Δparam| = {diff.max():.2e}")


print(f"scikit-learn {sklearn.__version__}")
seed = load("initial_lhs_300_bishop.csv")
lhs_ext = load("results_lhs_bishop.csv")
guided = load("results_uniform_bishop.csv")
test_natural_ref = load("results_test_bishop.csv")
test_150_ref = load("results_test_150_bishop.csv")

# --- 1. designs ---------------------------------------------------------------
print("\n== Designs (reproduced vs reference inputs) ==")
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42, n_jobs=-1)
rf.fit(seed[FEATURES], np.log(seed["FOS"]))
print(f"  screener OOB R² (log): {rf.oob_score_:.4f}   (notebook: 0.9567)")

rng = np.random.default_rng(42)
pool = pd.DataFrame({k: rng.uniform(lo, hi, 200_000) for k, (lo, hi) in RANGES.items()})
pool["gamma_H_c"] = pool["gamma"] * pool["H"] / pool["c"]
pool["FOS_pred"] = np.exp(rf.predict(pool[FEATURES]))
edges = np.linspace(0.5, 6.0, 16)
mu, sd = pool[PARAMS].mean(), pool[PARAMS].std()
train_idx = []
for lo, hi in zip(edges[:-1], edges[1:]):
    cand = pool[(pool["FOS_pred"] > lo) & (pool["FOS_pred"] <= hi)]
    sel = cand.index.values[greedy_maxmin(((cand[PARAMS] - mu) / sd).values, 30, rng)]
    perm = rng.permutation(len(sel))
    train_idx += list(sel[perm[:20]])
set_guided = pool.loc[train_idx, PARAMS].reset_index(drop=True)
compare_inputs("guided", set_guided, pd.read_csv(REF / "results_uniform_bishop.csv"))

set_lhs = pd.DataFrame(qmc.scale(qmc.LatinHypercube(d=5, seed=20260917).random(300), LO, HI), columns=PARAMS)
compare_inputs("lhs-extension", set_lhs, pd.read_csv(REF / "results_lhs_bishop.csv"))
set_natural = pd.DataFrame(qmc.scale(qmc.LatinHypercube(d=5, seed=20260920).random(300), LO, HI), columns=PARAMS)
compare_inputs("test-natural", set_natural, pd.read_csv(REF / "results_test_bishop.csv"))

for name, frame, refs in [
    ("lhs-extension", set_lhs, [seed]),
    ("guided", set_guided, [seed]),
    ("test-natural", set_natural, [seed, set_guided, set_lhs]),
]:
    ref = pd.concat([r[PARAMS] for r in refs], ignore_index=True)
    d = cKDTree(normalize(ref)).query(normalize(frame))[0]
    print(f"  min normalized distance {name} -> earlier sets: {d.min():.5f}")

pool_rng, sel_rng = np.random.default_rng(20260918), np.random.default_rng(20260919)
tp = pd.DataFrame({p: pool_rng.uniform(*RANGES[p], 200_000) for p in PARAMS})
tp["gamma_H_c"] = tp["gamma"] * tp["H"] / tp["c"]
tp["FOS_pred"] = np.exp(rf.predict(tp[FEATURES]))
tp = tp[tp["FOS_pred"].between(0.5, 6.0, inclusive="both")].reset_index(drop=True)
ref = pd.concat([seed[PARAMS], set_guided, set_lhs, set_natural], ignore_index=True)
tp = tp.loc[cKDTree(normalize(ref)).query(normalize(tp))[0] > 1e-3].reset_index(drop=True)
chosen = []
for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    mask = (tp["FOS_pred"] >= lo) if b == 0 else (tp["FOS_pred"] > lo)
    cand = tp.loc[mask & (tp["FOS_pred"] <= hi)]
    chosen += cand.index.to_numpy()[sel_rng.choice(len(cand), 10, replace=False)].tolist()
compare_inputs("test-stratified", tp.loc[chosen, PARAMS].reset_index(drop=True),
               pd.read_csv(REF / "results_test_150_bishop.csv"))

# --- 2. hyper-parameter selection -------------------------------------------------
print("\n== Hyper-parameters (local vs Colab) ==")
arms = {"A": seed, "C0": guided, "C": pd.concat([seed, guided], ignore_index=True),
        "B": pd.concat([seed, lhs_ext], ignore_index=True)}
cv = KFold(5, shuffle=True, random_state=42)
grids = {
    "RF": (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=1),
           {"max_features": [0.3, 0.5, 1.0], "min_samples_leaf": [1, 2, 5]}),
    "GBM": (HistGradientBoostingRegressor(random_state=42, early_stopping=False),
            {"learning_rate": [0.03, 0.06, 0.10], "max_leaf_nodes": [7, 15, 31],
             "min_samples_leaf": [3, 5, 10, 20], "max_iter": [300, 600]}),
}
agree = True
for family, (est, grid) in grids.items():
    for arm, df in arms.items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(df[FEATURES], np.log(df["FOS"]))
        same = gs.best_params_ == EXPECTED[(family, arm)]
        agree &= same
        print(f"  {family:3} {arm:2} n={len(df)}  {gs.best_params_}  {'==' if same else '!= ' + str(EXPECTED[(family, arm)])}")
print(f"\nAll selections agree with Colab: {agree}")
```

- [ ] **Step 6: Run the probe**

```bash
.venv/bin/python tools/probe_environment.py "$REF" 2>&1 | tee /tmp/probe.txt
```
Expected, if the environment agrees:
- `max |Δparam|` ≤ 5e-7 for all four designs. The reference inputs went through 6-decimal arrays, so 5e-7 is the rounding floor.
- Every minimum normalized distance > 0.001.
- `All selections agree with Colab: True`

If a design differs by more than 5e-7, or a selection differs, **stop and report to the author**. Do not continue to Task 2. A design mismatch means the screener or the RNG path differs, and that would change every screener-driven set.

- [ ] **Step 7: Record the outcome**

Paste the full probe output under "Observed locally" in `docs/environment-probe-2026-09.md`, inside a fenced block. Then add one line stating the verdict, e.g. `Verdict: identical designs and hyper-parameter selections on scikit-learn 1.9.1 / Python 3.14.4.`

- [ ] **Step 8: Commit**

```bash
git add tools/probe_environment.py docs/environment-probe-2026-09.md
git commit -m "Probe the local scikit-learn environment against the Colab runs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 1 — repository restructure

### Task 2: Restructure the tree, rewrite CLAUDE.md and README, add data/README.md

**Files:**
- Delete: `xslope/` (untracked), `surrogate_guided_sampling.ipynb` (tracked)
- Create: `data/README.md`, `.python-version`, `artifacts/solver/xslope_simple1_v0.5.2.xlsx` (untracked copy)
- Modify: `.gitignore`, `README.md`
- Create (currently untracked, rewritten): `CLAUDE.md`

**Interfaces:**
- Produces: the directory layout of spec §3; ignore rules that every later task relies on.

The HYRCAN part of spec phase 1 is already done: commit `de4b65a` removed `hyrcan/`, and the tag `hyrcan-final` exists. The stray `git init` inside `xslope/` is already gone. Verify before continuing:

```bash
git tag -l hyrcan-final        # prints hyrcan-final
test ! -e hyrcan && test ! -e xslope/.git && echo ok
```

- [ ] **Step 1: Confirm the reference copy is complete before deleting anything**

```bash
REF=/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09
diff <(cd xslope && ls | grep -v __pycache__ | sort) <(ls "$REF" | grep -v published | sort)
```
Expected: no output.

- [ ] **Step 2: Remove the old material and create the empty layout**

```bash
mkdir -p artifacts/solver
cp xslope/xslope_simple1_v0.5.2.xlsx artifacts/solver/
rm -rf xslope
git rm -q surrogate_guided_sampling.ipynb
mkdir -p sgs tests tools notebooks data/designs data/results data/final data/checks outputs/metrics outputs/tables outputs/figures
echo "3.14.4" > .python-version
```
The `.xlsx` copy is only a cache. `solver.ensure_template()` (Task 9) checks it against the pinned SHA-256 and downloads it again if it is missing or wrong.

- [ ] **Step 3: Replace `.gitignore`**

```gitignore
.DS_Store
__pycache__/
*.pyc
*.egg-info/
.venv/
.pytest_cache/

# Regenerable payloads; their *.manifest.json sidecars stay tracked.
data/**/*.csv
artifacts/**/*.joblib
artifacts/**/*.xlsx

# In-flight stage outputs and resume parts.
data/**/.*.tmp*
artifacts/**/.*.tmp*
outputs/**/.*.tmp*
```

- [ ] **Step 4: Write `data/README.md`**

```markdown
# data/

Everything in this directory is produced by a pipeline stage (`sgs ...`). Each
payload has a `*.manifest.json` sidecar recording its content hash, the hashes
of the configuration sections and input files that produced it, library
versions and wall time. Payloads are not tracked in git; manifests are.

| directory   | stage    | content                                              |
|-------------|----------|------------------------------------------------------|
| `designs/`  | `design` | parameter sets, before the solver                    |
| `results/`  | `solve`  | raw xslope/Bishop output, never modified afterwards  |
| `final/`    | `repair` | boundary rows repaired; consumed by the analysis     |
| `checks/`   | `check`  | `receipt.json`: per-set and pairwise validation      |

## File names and the Zenodo deposit

The first Zenodo versions (concept DOI 10.5281/zenodo.20706202) used the old
names below. Two of them are misleading: `results_uniform_bishop.csv` holds
the *guided* set, and `results_test_bishop.csv` holds the *natural* test.

| old name (Zenodo ≤ previous version) | new name                   | role                        |
|--------------------------------------|----------------------------|-----------------------------|
| `initial_lhs_300_bishop.csv`         | `seed_lhs_300.csv`         | seed, arm A                 |
| `results_lhs_bishop.csv`             | `lhs_extension_300.csv`    | LHS extension, arm B        |
| `results_uniform_bishop.csv`         | `guided_300.csv`           | guided selection, C and C₀  |
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

| Arm | Training data                  | Solver budget      |
|-----|--------------------------------|--------------------|
| A   | seed LHS-300                   | 300                |
| B   | seed LHS-300 + LHS-300         | 600                |
| C   | seed LHS-300 + guided-300      | 600                |
| C₀  | guided-300 only                | ablation, no budget |

Two test sets represent two deployment scenarios: `test-natural` (input-space
LHS, 300) and `test-stratified` (random selection within predicted-FOS bins,
150), each disjoint from every other set.

## Running the pipeline

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/sgs run-all --dry-run      # show what would run
.venv/bin/sgs run-all --workers 6    # everything; solver stages take hours
```

Stages can also be run one at a time; `sgs --help` lists them. Every stage
refuses to overwrite a stale artifact without `--force` and names what changed.
All scientific parameters live in `sgs/config.py`; none are accepted on the
command line.

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
thing. Both are disjoint from every other set by normalized distance; `sgs
check` asserts it.

## Non-negotiable disciplines

1. **C₀ appears in every analysis**, including robustness checks. Omitting it
   is selective reporting.
2. **No threshold misclassification analysis anywhere** — no false-stable
   counts, no FOS = 1.0 crossing metrics, no classification metrics. This is a
   regression study; an engineer treats FOS 0.9 and 1.1 as effectively the
   same, and a binary count penalises a 0.05 error like a 0.5 one. Zone RMSE,
   minimax and conformal coverage/width in the low-FOS zone carry the
   engineering argument instead. `tests/test_discipline.py` enforces this.
3. **Never duplicate a design call.** Every seed lives in `config.SEEDS`, named
   by role, and is referenced nowhere else as a literal.
   `tests/test_discipline.py` fails on integer literals assigned to or passed
   as a seed/random_state outside `config.py`. Always call
   `qmc.LatinHypercube(..., seed=...)`, never `rng=`: in SciPy 1.18 they
   produce different designs.
4. **No claim of model-agnosticism.** Two tuned tree ensembles support
   "consistent across two independently tuned tree-ensemble families" and
   nothing stronger.
5. **No formal conformal guarantee.** Residuals are nested-CV out-of-fold
   residuals from several fold models; test predictions come from a refit
   model. Call it _cross-conformal-style_ and report coverage empirically. A
   CV+ implementation (Barber et al. 2023) would earn a citable guarantee.
6. **No scientific parameter on the command line.** `--role` selects a config
   entry; everything else is execution control.

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

## Statistics

- Wilcoxon on `d = |e₁| − |e₂|`, reporting median d, P(d<0) and the per-pair
  alternative from `config.COMPARISONS`. Phrase as _smaller absolute errors_,
  never _smaller RMSE_. Comparisons absent from the registry are refused.
- Conformal quantile is an order statistic: `k = min(ceil((n+1)(1−α)), n)`.
- Paired bootstrap for ΔRMSE covers test-sampling variability only, not seed
  or selection variability; the limitation is written into the output.

## Pipeline

`sgs <stage>`; see `README.md`. Every artifact has a `*.manifest.json`; a stage
whose inputs or config sections changed fails and names the change; `--force`
is the only way past. `data/results/` is immutable; `repair` writes
`data/final/`. `evaluate` is pure and reads fits and a test set passed as
arguments. `report` reads only `outputs/metrics/`. The notebook imports from
`sgs` and holds narrative and figures only — no computation.

## Performance notes

Estimators get `n_jobs=1`; the search gets `n_jobs=-1`. The reverse
oversubscribes cores. The 48 searches of `sgs fit` are test-independent, so
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
Expected: `D surrogate_guided_sampling.ipynb`, modified `.gitignore` and `README.md`, untracked `CLAUDE.md`, `.python-version`, `data/README.md`; nothing under `artifacts/`, because the `.xlsx` is ignored.

```bash
git add -A .gitignore README.md CLAUDE.md .python-version data/README.md surrogate_guided_sampling.ipynb
git commit -m "Restructure the repository around the sgs pipeline

The standalone xslope scripts, the old CSV files and the notebooks move to an
external reference directory; data/ starts empty. CLAUDE.md now describes the
protocol only.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Package skeleton and `config.py`

**Files:**
- Create: `pyproject.toml`, `requirements.lock`, `sgs/__init__.py`, `sgs/config.py`
- Create: `tests/__init__.py` (empty), `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces (used by every later task): everything named in `config.py` below. In particular `config.ROOT` (monkeypatched by tests), `config.sections()`, `config.section_hash(name) -> str`, and the path helpers `design_path(role)`, `results_path(role)`, `final_path(role)`, `receipt_path()`, `screener_path(name)`, `solver_template_path()`, `fit_path(family, arm, scoring)`, `learning_curve_path(family, key, scoring, budget)`, `metrics_path(test, scoring)`, `learning_curve_metrics_path(scoring)`, `tables_dir()`, `figures_dir()`. Every module reads config as `from sgs import config` and uses `config.X` at call time, never `from sgs.config import X`. Tests depend on this when they monkeypatch.

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

- [ ] **Step 4: Write the failing tests `tests/test_config.py`**

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
    used |= {"protocol_cv_split", "protocol_estimator",
             "bootstrap_resampling", "learning_curve_subsampling"}
    assert used == set(config.SEEDS)


def test_role_order_covers_roles_and_respects_dependencies():
    assert set(config.ROLE_ORDER) == set(config.ROLES)
    position = {role: i for i, role in enumerate(config.ROLE_ORDER)}
    for role, spec in config.ROLES.items():
        for earlier in spec.disjoint_from:
            assert position[earlier] < position[role]
        if spec.screener:
            trained_on = config.SCREENERS[spec.screener].trained_on
            assert position[trained_on] < position[role]


def test_design_sizes_match_role_sizes():
    assert config.ROLES["guided"].n == config.N_BINS * config.GUIDED_PER_BIN
    assert config.ROLES["test-stratified"].n == config.N_BINS * config.TEST_PER_BIN
    assert config.GUIDED_DRAWN_PER_BIN >= config.GUIDED_PER_BIN


def test_arms_are_compositions_of_roles():
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


def test_section_hash_is_stable():
    assert config.section_hash("design") == config.section_hash("design")
    assert len(config.section_hash("design")) == 64


def test_section_hash_changes_only_its_own_section(monkeypatch):
    before = {name: config.section_hash(name) for name in config.sections()}
    monkeypatch.setitem(config.GRIDS["gbm"], "max_iter", [300, 600, 900])
    after = {name: config.section_hash(name) for name in config.sections()}
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {"protocol"}


def test_active_scorings_do_not_enter_any_hash(monkeypatch):
    before = {name: config.section_hash(name) for name in config.sections()}
    monkeypatch.setattr(config, "ACTIVE_SCORINGS", ("log-rmse", "fos-rmse"))
    assert before == {name: config.section_hash(name) for name in config.sections()}


def test_unknown_section_raises():
    with pytest.raises(KeyError):
        config.section_hash("nope")


def test_paths_follow_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert config.final_path("guided") == tmp_path / "data" / "final" / "guided_300.csv"
    assert config.fit_path("rf", "C0", "log-rmse") == tmp_path / "artifacts" / "fits" / "rf__C0__log-rmse.joblib"


def test_roles_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ROLES["seed"].n = 1
```

- [ ] **Step 5: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: collection error `ImportError: cannot import name 'config' from 'sgs'`.

- [ ] **Step 6: Write `sgs/config.py`**

```python
"""The only source of scientific truth for the study.

Every seed, range, grid and threshold is defined here, named by role, and
referenced from nowhere else as a literal. Values are grouped into sections;
each stage declares which sections it depends on, and a change to a section
invalidates only the artifacts that depend on it (see manifest.py).

Modules read this file as ``from sgs import config`` and use ``config.X`` at
call time, so tests can redirect ``ROOT`` and shrink sizes with monkeypatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Section "domain": the physical problem and how FOS is computed for it
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section "design": sampling procedures and their seeds
# ---------------------------------------------------------------------------

SELECTION_RANGE = (0.5, 6.0)
N_BINS = 15

# Candidate pool screened for the guided training selection. Declared study
# parameter (Methods): it is how the method finds candidates in each bin.
N_GUIDED_POOL = 200_000
GUIDED_PER_BIN = 20
# Greedy max-min draws 30 candidates per bin and a random 20 are kept. The
# other 10 were the share of an earlier, non-independent test set; they are
# still drawn so the RNG stream, and with it the training selection, is
# reproduced, and are then discarded.
GUIDED_DRAWN_PER_BIN = 30

# Independent pool for the target-stratified test set (declared in Methods).
N_TEST_POOL = 200_000
TEST_PER_BIN = 10

# Normalized-domain distance at or below which two points count as the same
# point (declared in Methods).
MIN_DISTANCE = 1e-3

SEEDS = {
    # design
    "seed_lhs_design": 42,
    "lhs_extension_design": 20260917,
    # One stream for the guided role: the candidate pool is drawn first and
    # the bin selection continues on the same generator. Splitting it into two
    # seeds would change the draws; tests/test_design.py pins the order.
    "guided_stream": 42,
    "test_stratified_pool": 20260918,
    "test_stratified_selection": 20260919,
    "test_natural_design": 20260920,
    # protocol
    "protocol_cv_split": 42,       # inner and outer KFold shuffling
    "protocol_estimator": 42,      # random_state of every tuned estimator
    # evaluation
    "bootstrap_resampling": 20260921,
    # learning curve: entropy of a SeedSequence keyed on (budget, repeat)
    "learning_curve_subsampling": 20260922,
}


@dataclass(frozen=True)
class SetRole:
    n: int
    design: str                      # "lhs" | "guided-maxmin" | "stratified-random"
    seeds: tuple[str, ...]           # keys of SEEDS, in the order the procedure uses them
    screener: str | None             # key of SCREENERS, or None
    stem: str                        # file name without extension
    disjoint_from: tuple[str, ...] = ()   # roles whose designs this one must avoid


# A test set uses random selection within each predicted-FOS bin, not greedy
# max-min: it must be representative of the target distribution, not
# diversified toward the edges of the domain like the guided training set.
ROLES = {
    "seed": SetRole(
        n=300, design="lhs", seeds=("seed_lhs_design",), screener=None,
        stem="seed_lhs_300"),
    "lhs-extension": SetRole(
        n=300, design="lhs", seeds=("lhs_extension_design",), screener=None,
        stem="lhs_extension_300", disjoint_from=("seed",)),
    "guided": SetRole(
        n=300, design="guided-maxmin", seeds=("guided_stream",), screener="primary",
        stem="guided_300", disjoint_from=("seed",)),
    "test-natural": SetRole(
        n=300, design="lhs", seeds=("test_natural_design",), screener=None,
        stem="test_natural_300", disjoint_from=("seed", "lhs-extension", "guided")),
    "test-stratified": SetRole(
        n=150, design="stratified-random",
        seeds=("test_stratified_pool", "test_stratified_selection"), screener="primary",
        stem="test_stratified_150",
        disjoint_from=("seed", "lhs-extension", "guided", "test-natural")),
}

# Generation order: each role follows the roles it depends on.
ROLE_ORDER = ("seed", "lhs-extension", "guided", "test-natural", "test-stratified")


# ---------------------------------------------------------------------------
# Section "screener": the preliminary surrogate(s)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Screener:
    trained_on: str
    family: str
    params: dict = field(default_factory=dict)
    seeds: dict = field(default_factory=dict)
    cv_folds: int = 5    # out-of-fold diagnostic only


# Fixed hyper-parameters, not a tuned search: a screening instrument for the
# candidate pool, not one of the models under comparison. Its seeds share the
# value 42 with guided_stream but are independent of that stream.
SCREENERS = {
    "primary": Screener(trained_on="seed", family="rf",
                        params={"n_estimators": 500},
                        seeds={"fit": 42, "cv": 42}),
}


# ---------------------------------------------------------------------------
# Section "protocol": the tuned learners and the arms they are fitted on
# ---------------------------------------------------------------------------

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
SEARCH_KIND = "grid"

# Selection criterion. "log-rmse" is RMSE on the trained (log) target, the
# declared choice; "fos-rmse" is the sensitivity check. Each scoring has its
# own fit artifacts, so ACTIVE_SCORINGS is not part of any section hash.
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


# ---------------------------------------------------------------------------
# Section "evaluation"
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section "learning_curve"
# ---------------------------------------------------------------------------

LC_BASE_ROLE = "seed"
LC_STRATEGIES = {"lhs": "lhs-extension", "guided": "guided"}
LC_BUDGETS = (100, 200, 300, 400, 500, 600)
LC_REPEATS = 10


# ---------------------------------------------------------------------------
# Section hashing
# ---------------------------------------------------------------------------

def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _seeds(*names):
    return {name: SEEDS[name] for name in names}


def sections() -> dict[str, dict]:
    """The configuration, grouped into independently hashed sections."""
    design_seeds = sorted({s for role in ROLES.values() for s in role.seeds})
    return {
        "domain": {
            "params": PARAMS, "domain": DOMAIN, "features": FEATURES,
            "target": TARGET, "solver": SOLVER, "repair_variants": REPAIR_VARIANTS,
        },
        "design": {
            "selection_range": SELECTION_RANGE, "n_bins": N_BINS,
            "n_guided_pool": N_GUIDED_POOL, "guided_per_bin": GUIDED_PER_BIN,
            "guided_drawn_per_bin": GUIDED_DRAWN_PER_BIN,
            "n_test_pool": N_TEST_POOL, "test_per_bin": TEST_PER_BIN,
            "min_distance": MIN_DISTANCE, "roles": ROLES,
            "seeds": _seeds(*design_seeds),
        },
        "screener": {"screeners": SCREENERS},
        "protocol": {
            "families": FAMILIES, "estimator_fixed": ESTIMATOR_FIXED,
            "grids": GRIDS, "cv_folds": CV_FOLDS, "search_kind": SEARCH_KIND,
            "arms": ARMS, "seeds": _seeds("protocol_cv_split", "protocol_estimator"),
        },
        "evaluation": {
            "test_roles": TEST_ROLES, "zone_edges": ZONE_EDGES, "alpha": ALPHA,
            "n_bootstrap": N_BOOTSTRAP, "bootstrap_ci": BOOTSTRAP_CI,
            "comparisons": COMPARISONS, "selection_range": SELECTION_RANGE,
            "seeds": _seeds("bootstrap_resampling"),
        },
        "learning_curve": {
            "base_role": LC_BASE_ROLE, "strategies": LC_STRATEGIES,
            "budgets": LC_BUDGETS, "repeats": LC_REPEATS,
            "seeds": _seeds("learning_curve_subsampling"),
        },
    }


def section_hash(name: str) -> str:
    payload = json.dumps(_jsonable(sections()[name]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Paths (functions, so tests can redirect ROOT)
# ---------------------------------------------------------------------------

def design_path(role: str) -> Path:
    return ROOT / "data" / "designs" / f"{ROLES[role].stem}.csv"


def results_path(role: str) -> Path:
    return ROOT / "data" / "results" / f"{ROLES[role].stem}.csv"


def final_path(role: str) -> Path:
    return ROOT / "data" / "final" / f"{ROLES[role].stem}.csv"


def receipt_path() -> Path:
    return ROOT / "data" / "checks" / "receipt.json"


def screener_path(name: str) -> Path:
    return ROOT / "artifacts" / "screeners" / f"{name}.joblib"


def solver_template_path() -> Path:
    return ROOT / "artifacts" / "solver" / f"xslope_simple1_v{SOLVER.package_version}.xlsx"


def fit_path(family: str, arm: str, scoring: str) -> Path:
    return ROOT / "artifacts" / "fits" / f"{family}__{arm}__{scoring}.joblib"


def learning_curve_path(family: str, key: str, scoring: str, budget: int) -> Path:
    return ROOT / "artifacts" / "learning_curve" / f"{family}__{key}__{scoring}__n{budget}.joblib"


def metrics_path(test: str, scoring: str) -> Path:
    return ROOT / "outputs" / "metrics" / f"{test}__{scoring}.json"


def learning_curve_metrics_path(scoring: str) -> Path:
    return ROOT / "outputs" / "metrics" / f"learning_curve__{scoring}.json"


def tables_dir() -> Path:
    return ROOT / "outputs" / "tables"


def figures_dir() -> Path:
    return ROOT / "outputs" / "figures"
```

- [ ] **Step 7: Write `tests/__init__.py` (empty) and `tests/conftest.py`**

```python
import pytest

from sgs import config


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Redirect every pipeline path into a temporary directory."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.lock sgs/__init__.py sgs/config.py tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "Add the sgs package and its configuration

config.py is the only source of scientific values, grouped into sections
that are hashed independently.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Discipline tests (seed literals, forbidden vocabulary)

These go in before any other module so that every later task is written under them.

**Files:**
- Create: `tests/test_discipline.py`

**Interfaces:**
- Consumes: the files under `sgs/` (all except `config.py`) and under `notebooks/`.
- Produces: `seed_violations(source: str) -> list[str]` and `threshold_violations(source: str) -> list[str]`, used only inside this test file.

- [ ] **Step 1: Write the test file**

```python
"""Tests that enforce the protocol disciplines on the source itself."""

import ast
import json
import re
from pathlib import Path

import pytest

import sgs

SGS_DIR = Path(sgs.__file__).parent
ROOT = SGS_DIR.parent
MODULES = sorted(p for p in SGS_DIR.glob("*.py") if p.name != "config.py")

SEED_NAME = re.compile(r"seed|random_state", re.IGNORECASE)
RNG_CALLS = {"LatinHypercube", "default_rng", "RandomState", "check_random_state"}
CLASSIFICATION_METRICS = {
    "accuracy_score", "balanced_accuracy_score", "precision_score", "recall_score",
    "f1_score", "fbeta_score", "confusion_matrix", "classification_report",
    "roc_auc_score", "roc_curve", "precision_recall_curve", "matthews_corrcoef",
    "cohen_kappa_score", "average_precision_score",
}
FORBIDDEN_WORDS = re.compile(r"false[\s_-]?stable", re.IGNORECASE)


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


def threshold_violations(source):
    found = []
    if FORBIDDEN_WORDS.search(source):
        found.append("forbidden vocabulary: false stable")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn.metrics"):
            for alias in node.names:
                if alias.name in CLASSIFICATION_METRICS:
                    found.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.Attribute) and node.attr in CLASSIFICATION_METRICS:
            found.append(f"line {node.lineno}: uses {node.attr}")
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            has_number = any(isinstance(o, ast.Constant) and isinstance(o.value, (int, float))
                             and not isinstance(o.value, bool) for o in operands)
            mentions_pred = any("pred" in ast.unparse(o).lower() for o in operands
                                if not isinstance(o, ast.Constant))
            if has_number and mentions_pred:
                found.append(f"line {node.lineno}: prediction compared with a fixed value")
    return found


def _notebook_sources():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        code = "\n".join(
            "\n".join(line for line in "".join(c["source"]).splitlines()
                      if not line.lstrip().startswith(("!", "%")))
            for c in cells if c["cell_type"] == "code")
        yield path, code


# --- the checkers themselves ------------------------------------------------------

@pytest.mark.parametrize("snippet", [
    "seed = 42",
    "RANDOM_STATE: int = 0",
    "f(random_state=0)",
    "qmc.LatinHypercube(42)",
    "np.random.default_rng(7)",
    "def f(seed=1): pass",
    "cfg['seed'] = 3",
])
def test_seed_checker_catches(snippet):
    assert seed_violations(snippet)


@pytest.mark.parametrize("snippet", [
    "seed = config.SEEDS['seed_lhs_design']",
    "f(random_state=config.SEEDS['protocol_estimator'], n_jobs=1)",
    "qmc.LatinHypercube(d=len(config.PARAMS), seed=seed)",
    "np.random.default_rng(config.SEEDS['guided_stream'])",
    "n_seed = len(frame)",
])
def test_seed_checker_allows(snippet):
    assert not seed_violations(snippet)


@pytest.mark.parametrize("snippet", [
    "n_false_stable = 3",
    "from sklearn.metrics import confusion_matrix",
    "sklearn.metrics.f1_score(a, b)",
    "bad = (y_pred >= 1.0) & (y < 1.0)",
    "flag = preds[name] < 1",
])
def test_threshold_checker_catches(snippet):
    assert threshold_violations(snippet)


@pytest.mark.parametrize("snippet", [
    "mask = (pool['FOS_pred'] > lo) & (pool['FOS_pred'] <= hi)",
    "share = np.mean(d < 0)",
    "from sklearn.metrics import make_scorer",
])
def test_threshold_checker_allows(snippet):
    assert not threshold_violations(snippet)


# --- the actual disciplines ------------------------------------------------------

@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_seed_literal_outside_config(path):
    assert seed_violations(path.read_text()) == []


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_threshold_misclassification_in_package(path):
    assert threshold_violations(path.read_text()) == []


def test_no_threshold_misclassification_in_notebooks():
    for path, code in _notebook_sources():
        assert threshold_violations(code) == [], path.name
```

- [ ] **Step 2: Run**

Run: `.venv/bin/pytest tests/test_discipline.py -q`
Expected: all pass. The checker tests exercise the snippets; the module tests pass over `__init__.py` alone, and the notebook test passes vacuously, since `notebooks/` is empty until Task 26.

- [ ] **Step 3: Commit**

```bash
git add tests/test_discipline.py
git commit -m "Enforce the seed and no-misclassification disciplines in tests

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `manifest.py` — hashing, sidecars, staleness

**Files:**
- Create: `sgs/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: `config.ROOT`, `config.section_hash`.
- Produces:
  - `SUFFIX = ".manifest.json"`, `manifest_path(payload: Path) -> Path`
  - `sha256_file(path) -> str`, `environment() -> dict`
  - `write_manifest(payload, *, stage: str, sections: Iterable[str], inputs: Mapping[str, Path], wall_time_s: float, extra: dict | None = None) -> dict`
  - `read_manifest(payload) -> dict | None`
  - `staleness(payload, *, sections, inputs) -> list[str]` (empty = fresh)
  - `verify_provenance(path) -> None` (raises `FileNotFoundError` / `MissingProvenance`)
  - `should_run(payload, *, sections, inputs, force: bool) -> bool` (raises `StaleArtifact`)
  - `require_fresh(payload, *, sections, inputs) -> None` (raises `MissingProvenance` / `StaleArtifact`)
  - exceptions `StaleArtifact(RuntimeError)`, `MissingProvenance(RuntimeError)`

- [ ] **Step 1: Write the failing tests `tests/test_manifest.py`**

```python
import json

import numpy as np
import pytest

from sgs import config, manifest


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_manifest_records_hashes_and_environment(root):
    src = _write(root / "data" / "a.csv", "x\n1\n")
    out = _write(root / "data" / "b.csv", "y\n2\n")
    record = manifest.write_manifest(out, stage="demo", sections=("domain",),
                                     inputs={"a": src}, wall_time_s=1.23456,
                                     extra={"n": np.int64(3)})
    on_disk = json.loads(manifest.manifest_path(out).read_text())
    assert on_disk == record
    assert record["sha256"] == manifest.sha256_file(out)
    assert record["config"] == {"domain": config.section_hash("domain")}
    assert record["inputs"]["a"]["path"] == "data/a.csv"
    assert record["environment"]["python"]
    assert record["extra"] == {"n": 3}
    assert record["wall_time_s"] == 1.235


def test_fresh_payload_has_no_staleness(root):
    src = _write(root / "a.csv", "1")
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=("design",), inputs={"a": src}, wall_time_s=0)
    assert manifest.staleness(out, sections=("design",), inputs={"a": src}) == []


def test_changed_input_is_named(root):
    src = _write(root / "a.csv", "1")
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=(), inputs={"a": src}, wall_time_s=0)
    src.write_text("changed")
    reasons = manifest.staleness(out, sections=(), inputs={"a": src})
    assert len(reasons) == 1 and reasons[0].startswith("input 'a' changed")


def test_changed_section_is_named(root, monkeypatch):
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=("protocol",), inputs={}, wall_time_s=0)
    monkeypatch.setitem(config.GRIDS["rf"], "min_samples_leaf", [1])
    assert manifest.staleness(out, sections=("protocol",), inputs={}) == ["config section 'protocol' changed"]


def test_unrelated_section_change_does_not_stale(root, monkeypatch):
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=("design",), inputs={}, wall_time_s=0)
    monkeypatch.setitem(config.GRIDS["rf"], "min_samples_leaf", [1])
    assert manifest.staleness(out, sections=("design",), inputs={}) == []


def test_modified_payload_is_detected(root):
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=(), inputs={}, wall_time_s=0)
    out.write_text("tampered")
    assert "payload modified after its manifest was written" in manifest.staleness(out, sections=(), inputs={})


def test_changed_dependency_set_is_named(root):
    src = _write(root / "a.csv", "1")
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=("domain",), inputs={}, wall_time_s=0)
    reasons = manifest.staleness(out, sections=("domain", "design"), inputs={"a": src})
    assert "stage now depends on config section 'design'" in reasons
    assert "new input 'a'" in reasons


def test_should_run(root):
    out = root / "b.csv"
    assert manifest.should_run(out, sections=(), inputs={}, force=False)
    _write(out, "2")
    manifest.write_manifest(out, stage="demo", sections=(), inputs={}, wall_time_s=0)
    assert not manifest.should_run(out, sections=(), inputs={}, force=False)
    out.write_text("tampered")
    with pytest.raises(manifest.StaleArtifact, match="--force"):
        manifest.should_run(out, sections=(), inputs={}, force=False)
    assert manifest.should_run(out, sections=(), inputs={}, force=True)


def test_verify_provenance(root):
    path = root / "x.csv"
    with pytest.raises(FileNotFoundError):
        manifest.verify_provenance(path)
    _write(path, "1")
    with pytest.raises(manifest.MissingProvenance, match="no manifest"):
        manifest.verify_provenance(path)
    manifest.write_manifest(path, stage="demo", sections=(), inputs={}, wall_time_s=0)
    manifest.verify_provenance(path)


def test_require_fresh(root):
    out = _write(root / "b.csv", "2")
    manifest.write_manifest(out, stage="demo", sections=("domain",), inputs={}, wall_time_s=0)
    manifest.require_fresh(out, sections=("domain",), inputs={})
    with pytest.raises(manifest.StaleArtifact):
        manifest.require_fresh(out, sections=("domain", "design"), inputs={})
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_manifest.py -q`
Expected: `ImportError: cannot import name 'manifest'`.

- [ ] **Step 3: Write `sgs/manifest.py`**

```python
"""Provenance sidecars: hashing, manifest read/write, staleness detection.

Every payload ``x`` has a sidecar ``x.manifest.json`` recording its content
hash, the hashes of the config sections and input files that produced it,
library versions and wall time. A payload whose recorded hashes no longer
match is stale, and the stage that owns it refuses to proceed without --force.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping

from sgs import config

SUFFIX = ".manifest.json"
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "shapely", "xslope")


class StaleArtifact(RuntimeError):
    """An artifact exists but no longer matches its configuration or inputs."""


class MissingProvenance(RuntimeError):
    """A file is used as an input but has no valid manifest."""


def manifest_path(payload: Path) -> Path:
    payload = Path(payload)
    return payload.with_name(payload.name + SUFFIX)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(config.ROOT).resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_default(value):
    if hasattr(value, "item"):          # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):        # numpy array
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_manifest(payload, *, stage: str, sections: Iterable[str],
                   inputs: Mapping[str, Path], wall_time_s: float,
                   extra: dict | None = None) -> dict:
    payload = Path(payload)
    record = {
        "stage": stage,
        "payload": payload.name,
        "sha256": sha256_file(payload),
        "config": {name: config.section_hash(name) for name in sorted(sections)},
        "inputs": {name: {"path": _relative(path), "sha256": sha256_file(path)}
                   for name, path in sorted(inputs.items())},
        "environment": environment(),
        "wall_time_s": round(float(wall_time_s), 3),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "extra": extra or {},
    }
    text = json.dumps(record, indent=2, sort_keys=True, default=_json_default)
    _atomic_write_text(manifest_path(payload), text + "\n")
    return json.loads(text)


def read_manifest(payload) -> dict | None:
    path = manifest_path(Path(payload))
    if not path.exists():
        return None
    return json.loads(path.read_text())


def staleness(payload, *, sections: Iterable[str], inputs: Mapping[str, Path]) -> list[str]:
    """Reasons why an existing payload no longer matches; empty when fresh."""
    payload = Path(payload)
    record = read_manifest(payload)
    if record is None:
        return ["no manifest: the payload has no recorded provenance"]
    reasons = []
    if sha256_file(payload) != record["sha256"]:
        reasons.append("payload modified after its manifest was written")

    sections = set(sections)
    recorded = record["config"]
    for name in sorted(sections | set(recorded)):
        if name not in sections:
            reasons.append(f"stage no longer depends on config section {name!r}")
        elif name not in recorded:
            reasons.append(f"stage now depends on config section {name!r}")
        elif recorded[name] != config.section_hash(name):
            reasons.append(f"config section {name!r} changed")

    recorded_inputs = record["inputs"]
    for name in sorted(set(inputs) | set(recorded_inputs)):
        if name not in inputs:
            reasons.append(f"input {name!r} no longer used")
        elif name not in recorded_inputs:
            reasons.append(f"new input {name!r}")
        elif not Path(inputs[name]).exists():
            reasons.append(f"input {name!r} missing")
        else:
            was = recorded_inputs[name]["sha256"]
            now = sha256_file(inputs[name])
            if now != was:
                reasons.append(f"input {name!r} changed ({was[:8]} -> {now[:8]})")
    return reasons


def verify_provenance(path) -> None:
    """Fail unless ``path`` exists, has a manifest, and still matches it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{_relative(path)} does not exist; run the stage that produces it")
    record = read_manifest(path)
    if record is None:
        raise MissingProvenance(f"{_relative(path)} has no manifest; only pipeline-produced files may be used")
    if sha256_file(path) != record["sha256"]:
        raise MissingProvenance(f"{_relative(path)} was modified after its manifest was written")


def should_run(payload, *, sections, inputs, force: bool) -> bool:
    """True if the payload is missing, or stale and ``force`` is set."""
    payload = Path(payload)
    if not payload.exists():
        return True
    reasons = staleness(payload, sections=sections, inputs=inputs)
    if not reasons:
        return False
    if force:
        return True
    raise StaleArtifact(f"{_relative(payload)} is stale: " + "; ".join(reasons)
                        + ". Re-run with --force to regenerate it.")


def require_fresh(payload, *, sections, inputs) -> None:
    """Fail loudly if an artifact about to be consumed is stale."""
    verify_provenance(payload)
    reasons = staleness(payload, sections=sections, inputs=inputs)
    if reasons:
        raise StaleArtifact(f"{_relative(Path(payload))} is stale: " + "; ".join(reasons)
                            + ". Re-run the stage that produces it.")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_manifest.py tests/test_discipline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/manifest.py tests/test_manifest.py
git commit -m "Add manifest sidecars with per-section staleness detection

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `stage.py` (guarded producer) and the CLI

**Files:**
- Create: `sgs/stage.py`, `sgs/cli.py`, `tests/test_stage.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `manifest.*`.
- Produces:
  - `Options(dry_run=False, force=False, resume=False, workers=1, only_family=None, only_arm=None, only_scoring=None)` (frozen dataclass)
  - `produce(payload, *, stage: str, sections, inputs: Mapping[str, Path], opts: Options, build: Callable[[Path], dict | None], resumable: bool = False) -> bool`. It builds into `.{name}.tmp`, then `os.replace`s it into place and writes the manifest. It returns True if the payload was (re)built. On failure it deletes the tmp file, unless `resumable`.
  - `cli.build_parser()`, `cli.main(argv=None)`. Each handler imports its stage module lazily, so the CLI works before later modules exist. The stage entry points it calls, which later tasks must provide exactly:
    `design.stage(role, opts)`, `solve.stage(role, opts)`, `repair.stage(role, opts)`, `screener.stage(opts, name=None)`, `checks.stage(opts)`, `fitting.stage(opts)`, `learning_curve.stage(opts)`, `evaluate.stage(opts, role=None)`, `report.stage(opts)`.

- [ ] **Step 1: Write the failing tests `tests/test_stage.py`**

```python
import pytest

from sgs import manifest
from sgs.stage import Options, produce


def _input(root, name="in.csv", text="1"):
    path = root / name
    path.write_text(text)
    manifest.write_manifest(path, stage="fixture", sections=(), inputs={}, wall_time_s=0)
    return path


def _builder(calls, text="out"):
    def build(tmp):
        calls.append(tmp)
        tmp.write_text(text)
        return {"n": 1}
    return build


def test_builds_once_then_skips(root):
    src, out, calls = _input(root), root / "out.csv", []
    assert produce(out, stage="t", sections=(), inputs={"a": src}, opts=Options(), build=_builder(calls))
    assert out.read_text() == "out"
    assert manifest.read_manifest(out)["extra"] == {"n": 1}
    assert not produce(out, stage="t", sections=(), inputs={"a": src}, opts=Options(), build=_builder(calls))
    assert len(calls) == 1
    assert calls[0].name == ".out.csv.tmp"


def test_changed_input_fails_without_force(root):
    src, out, calls = _input(root), root / "out.csv", []
    produce(out, stage="t", sections=(), inputs={"a": src}, opts=Options(), build=_builder(calls))
    src.write_text("2")
    manifest.write_manifest(src, stage="fixture", sections=(), inputs={}, wall_time_s=0)
    with pytest.raises(manifest.StaleArtifact, match="input 'a' changed"):
        produce(out, stage="t", sections=(), inputs={"a": src}, opts=Options(), build=_builder(calls))
    assert produce(out, stage="t", sections=(), inputs={"a": src}, opts=Options(force=True),
                   build=_builder(calls, "rebuilt"))
    assert out.read_text() == "rebuilt"


def test_input_without_manifest_is_refused(root):
    src = root / "raw.csv"
    src.write_text("1")
    with pytest.raises(manifest.MissingProvenance):
        produce(root / "out.csv", stage="t", sections=(), inputs={"a": src}, opts=Options(), build=_builder([]))


def test_dry_run_writes_nothing(root, capsys):
    out = root / "out.csv"
    assert not produce(out, stage="t", sections=(), inputs={}, opts=Options(dry_run=True), build=_builder([]))
    assert not out.exists()
    assert "dry run" in capsys.readouterr().out


def test_dry_run_reports_missing_inputs(root, capsys):
    produce(root / "out.csv", stage="t", sections=(), inputs={"a": root / "nope.csv"},
            opts=Options(dry_run=True), build=_builder([]))
    assert "waits for a" in capsys.readouterr().out


def test_failed_build_leaves_no_payload(root):
    out = root / "out.csv"

    def build(tmp):
        tmp.write_text("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        produce(out, stage="t", sections=(), inputs={}, opts=Options(), build=build)
    assert not out.exists() and not (root / ".out.csv.tmp").exists()


def test_resumable_build_keeps_its_checkpoint(root):
    out = root / "out.csv"

    def build(tmp):
        tmp.write_text("partial")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        produce(out, stage="t", sections=(), inputs={}, opts=Options(), build=build, resumable=True)
    assert (root / ".out.csv.tmp").read_text() == "partial"
```

- [ ] **Step 2: Write `tests/test_cli.py`**

```python
import pytest

from sgs import cli


def test_every_stage_has_a_subcommand():
    parser = cli.build_parser()
    for argv in (["design", "--role", "seed"], ["solve", "--role", "guided", "--workers", "4", "--resume"],
                 ["repair", "--role", "test-natural"], ["fit-screener"], ["check"],
                 ["fit", "--only-family", "rf", "--only-arm", "C0", "--only-scoring", "log-rmse"],
                 ["learning-curve", "--only-family", "gbm"], ["evaluate", "--role", "test-stratified"],
                 ["report"], ["run-all", "--dry-run", "--workers", "6"]):
        args = parser.parse_args(argv)
        assert callable(args.handler)


def test_unknown_role_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["design", "--role", "nope"])


def test_evaluate_role_is_restricted_to_test_sets():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evaluate", "--role", "seed"])


def test_no_scientific_flags_exist():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["design", "--role", "seed", "--seed", "1"])


def test_workers_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["solve", "--role", "seed", "--workers", "0"])


def test_options_are_carried():
    args = cli.build_parser().parse_args(["fit", "--force", "--only-arm", "C"])
    opts = cli._options(args)
    assert opts.force and opts.only_arm == "C" and not opts.dry_run and opts.workers == 1
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_stage.py tests/test_cli.py -q`
Expected: `ImportError` for `sgs.stage` and `sgs.cli`.

- [ ] **Step 4: Write `sgs/stage.py`**

```python
"""Execution control shared by every stage: options and the guarded producer."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from sgs import manifest


@dataclass(frozen=True)
class Options:
    dry_run: bool = False
    force: bool = False
    resume: bool = False
    workers: int = 1
    only_family: str | None = None
    only_arm: str | None = None
    only_scoring: str | None = None


def produce(payload, *, stage: str, sections: Iterable[str], inputs: Mapping[str, Path],
            opts: Options, build: Callable[[Path], dict | None], resumable: bool = False) -> bool:
    """Produce ``payload`` unless it is fresh; return True if it was (re)built.

    ``build(tmp)`` writes the payload to ``tmp`` and returns extra manifest
    fields (or None). The payload is moved into place only after ``build``
    returns, so an interrupted stage never leaves a partial file that looks
    complete. A resumable build keeps ``tmp`` on failure as its checkpoint.
    """
    payload = Path(payload)
    sections = tuple(sections)
    if opts.dry_run:
        missing = [name for name, path in inputs.items() if not Path(path).exists()]
        if missing:
            print(f"[{stage}] dry run: {payload.name} waits for {', '.join(missing)}")
            return False
    for path in inputs.values():
        manifest.verify_provenance(path)
    if not manifest.should_run(payload, sections=sections, inputs=inputs, force=opts.force):
        print(f"[{stage}] up to date: {payload.name}")
        return False
    if opts.dry_run:
        print(f"[{stage}] dry run: would produce {payload.name}")
        return False

    payload.parent.mkdir(parents=True, exist_ok=True)
    tmp = payload.with_name(f".{payload.name}.tmp")
    started = time.perf_counter()
    try:
        extra = build(tmp)
        os.replace(tmp, payload)
    except BaseException:
        if not resumable:
            tmp.unlink(missing_ok=True)
        raise
    manifest.write_manifest(payload, stage=stage, sections=sections, inputs=inputs,
                            wall_time_s=time.perf_counter() - started, extra=extra)
    print(f"[{stage}] wrote {payload.name}")
    return True
```

- [ ] **Step 5: Write `sgs/cli.py`**

```python
"""sgs — the surrogate-guided sampling pipeline, one subcommand per stage.

Scientific parameters live in sgs/config.py; the command line only selects a
role and controls execution.
"""

from __future__ import annotations

import argparse
import platform
import sys

from sgs import config
from sgs.stage import Options

REFERENCE_PYTHON = "3.14.4"


def _options(args) -> Options:
    return Options(
        dry_run=args.dry_run,
        force=args.force,
        resume=getattr(args, "resume", False),
        workers=getattr(args, "workers", 1),
        only_family=getattr(args, "only_family", None),
        only_arm=getattr(args, "only_arm", None),
        only_scoring=getattr(args, "only_scoring", None),
    )


def _design(args, opts):
    from sgs import design
    design.stage(args.role, opts)


def _solve(args, opts):
    from sgs import solve
    solve.stage(args.role, opts)


def _repair(args, opts):
    from sgs import repair
    repair.stage(args.role, opts)


def _fit_screener(args, opts):
    from sgs import screener
    screener.stage(opts)


def _check(args, opts):
    from sgs import checks
    checks.stage(opts)


def _fit(args, opts):
    from sgs import fitting
    fitting.stage(opts)


def _learning_curve(args, opts):
    from sgs import learning_curve
    learning_curve.stage(opts)


def _evaluate(args, opts):
    from sgs import evaluate
    evaluate.stage(opts, role=args.role)


def _report(args, opts):
    from sgs import report
    report.stage(opts)


def _run_all(args, opts):
    from sgs import checks, design, evaluate, fitting, learning_curve, repair, report, screener, solve
    for role in config.ROLE_ORDER:
        spec = config.ROLES[role]
        if spec.screener:
            screener.stage(opts, name=spec.screener)
        design.stage(role, opts)
        solve.stage(role, opts)
        repair.stage(role, opts)
    checks.stage(opts)
    fitting.stage(opts)
    learning_curve.stage(opts)
    evaluate.stage(opts)
    report.stage(opts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, help, *, role=None, resume=False, filters=()):
        p = sub.add_parser(name, help=help)
        p.add_argument("--dry-run", action="store_true", help="report what would run, run nothing")
        p.add_argument("--force", action="store_true", help="regenerate stale artifacts instead of failing")
        if role == "any":
            p.add_argument("--role", required=True, choices=tuple(config.ROLES))
        elif role == "test":
            p.add_argument("--role", default=None, choices=config.TEST_ROLES,
                           help="one test set (default: all)")
        if resume:
            p.add_argument("--resume", action="store_true", help="continue a partial solver run")
            p.add_argument("--workers", type=int, default=1, help="solver processes")
        for name_ in filters:
            p.add_argument(f"--only-{name_}", default=None,
                           help=f"restrict to one {name_} declared in config")
        p.set_defaults(handler=handler)

    command("design", _design, "draw the parameter set of a role", role="any")
    command("solve", _solve, "run xslope/Bishop on a role's design", role="any", resume=True)
    command("repair", _repair, "recompute boundary-truncated rows on enlarged domains", role="any")
    command("fit-screener", _fit_screener, "fit the preliminary surrogate")
    command("check", _check, "validate final sets and pairwise disjointness")
    command("fit", _fit, "tuned search + nested OOF residuals per (family, arm, scoring)",
            filters=("family", "arm", "scoring"))
    command("learning-curve", _learning_curve, "refit on budget subsamples",
            filters=("family", "scoring"))
    command("evaluate", _evaluate, "metrics on the test sets", role="test", filters=("scoring",))
    command("report", _report, "booktabs tables and figures from outputs/metrics")
    command("run-all", _run_all, "every stage, in order", resume=True, filters=("scoring",))
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be at least 1")
    if platform.python_version() != REFERENCE_PYTHON:
        print(f"warning: running on Python {platform.python_version()}; "
              f"the reference environment is {REFERENCE_PYTHON}", file=sys.stderr)
    args.handler(args, _options(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

Run: `.venv/bin/sgs --help`
Expected: the ten subcommands are listed.

- [ ] **Step 7: Commit**

```bash
git add sgs/stage.py sgs/cli.py tests/test_stage.py tests/test_cli.py
git commit -m "Add the guarded stage producer and the sgs CLI

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 2 — design, solve, repair; the seed chain

### Task 7: `data.py` — schema, derived feature, loading

**Files:**
- Create: `sgs/data.py`, `tests/helpers.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `config`, `manifest.verify_provenance`.
- Produces:
  - `RESULT_COLUMNS: list[str]`, `META_COLUMNS = ["repair_variant", "excluded"]`, `FINAL_COLUMNS`, `EXCLUSION_REASONS = ("solver_error", "unresolved")`
  - `add_features(frame) -> DataFrame` (copy, adds `gamma_H_c`) — the **only** definition of the derived feature
  - `read_table(path) -> DataFrame` (text columns as strings, empty = `""`)
  - `is_true(series) -> Series[bool]`, `usable(frame) -> DataFrame`
  - `load_role(role) -> DataFrame` — usable rows of `final/<role>`, with `row_id = "<role>:<row index in the file>"` and features
  - `load_arm(arm) -> DataFrame`
  - `TestSet(name, X, y, ids)` (frozen dataclass, `__test__ = False`), `load_test(role) -> TestSet`
  - `normalize(frame) -> ndarray` (params scaled to [0, 1] by `config.DOMAIN`)
- `tests/helpers.py` produces `synthetic_fos(frame)`, `FormulaModel`, `synthetic_final(n, seed)`, `write_payload(path, frame, sections=(), inputs=None)`, `write_final(role, frame)`.

- [ ] **Step 1: Write `tests/helpers.py`**

```python
"""Synthetic fixtures. The formula is not a slope-stability model; it only
gives a smooth, positive, FOS-like response over the real domain."""

import numpy as np
import pandas as pd

from sgs import config, data, manifest


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


def write_payload(path, frame, sections=(), inputs=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    manifest.write_manifest(path, stage="fixture", sections=sections,
                            inputs=inputs or {}, wall_time_s=0.0)
    return path


def write_final(role, frame):
    return write_payload(config.final_path(role), frame)
```

- [ ] **Step 2: Write the failing tests `tests/test_data.py`**

```python
import numpy as np
import pytest

from sgs import config, data, manifest
from tests.helpers import synthetic_final, write_final


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
    assert list(loaded.columns[-1:]) == ["gamma_H_c"]


def test_load_role_requires_provenance(root):
    path = config.final_path("seed")
    path.parent.mkdir(parents=True)
    synthetic_final(3, 0).to_csv(path, index=False)
    with pytest.raises(manifest.MissingProvenance):
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
    import pandas as pd
    assert data.is_true(pd.Series(["True", "false", "", " TRUE "])).tolist() == [True, False, False, True]


def test_normalize_maps_domain_to_unit_cube():
    import pandas as pd
    corners = pd.DataFrame([{p: lo for p, (lo, _) in config.DOMAIN.items()},
                            {p: hi for p, (_, hi) in config.DOMAIN.items()}])
    np.testing.assert_allclose(data.normalize(corners), [[0] * 5, [1] * 5])
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_data.py -q`
Expected: `ImportError: cannot import name 'data'`.

- [ ] **Step 4: Write `sgs/data.py`**

```python
"""Loading, the schema contract, and the one definition of the derived feature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sgs import config, manifest

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


def load_role(role: str) -> pd.DataFrame:
    """Usable rows of a final set, with row identifiers and features."""
    path = config.final_path(role)
    manifest.verify_provenance(path)
    frame = read_table(path)
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


def normalize(frame: pd.DataFrame) -> np.ndarray:
    lo = np.array([config.DOMAIN[p][0] for p in config.PARAMS], dtype=float)
    hi = np.array([config.DOMAIN[p][1] for p in config.PARAMS], dtype=float)
    return (frame[list(config.PARAMS)].to_numpy(float) - lo) / (hi - lo)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sgs/data.py tests/helpers.py tests/test_data.py
git commit -m "Add data loading with the schema contract and one derived-feature definition

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `design.py` — LHS designs and the design stage

**Files:**
- Create: `sgs/design.py`, `tests/test_design.py`

**Interfaces:**
- Consumes: `config`, `data.normalize`, `data.add_features`, `manifest.verify_provenance`, `stage.produce`.
- Produces:
  - `DesignError(RuntimeError)`
  - `lhs(n: int, seed: int) -> tuple[ndarray, DataFrame]` (unit design, physical frame with `config.PARAMS` columns)
  - `lhs_strata_complete(unit) -> bool`
  - `nearest_distance(points, reference) -> ndarray`, `assert_disjoint(frame, references: Mapping[str, DataFrame], min_distance: float) -> None`
  - `read_design(role) -> DataFrame`, `build(role) -> DataFrame`, `stage(role, opts)`
  - `_BUILDERS: dict[str, Callable[[str, SetRole, dict], DataFrame]]` — Tasks 14 and 15 add `"guided-maxmin"` and `"stratified-random"`.
  - Design CSV columns: `config.PARAMS` (+ `FOS_pred`, `bin` for screener designs).

- [ ] **Step 1: Write the failing tests `tests/test_design.py`**

```python
import hashlib

import numpy as np
import pandas as pd
import pytest

from sgs import config, design, manifest
from sgs.stage import Options


def _hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


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


@pytest.mark.parametrize("role", ["seed", "lhs-extension"])
def test_lhs_roles_are_deterministic(root, role):
    design.stage("seed", Options())
    if role != "seed":
        design.stage(role, Options())
    on_disk = manifest.sha256_file(config.design_path(role))
    assert _hash(design.build(role)) == on_disk


def test_stage_skips_when_fresh_and_records_provenance(root, capsys):
    design.stage("seed", Options())
    design.stage("seed", Options())
    assert "up to date" in capsys.readouterr().out
    record = manifest.read_manifest(config.design_path("seed"))
    assert set(record["config"]) == {"domain", "design"}
    assert record["extra"]["n"] == 300


def test_stage_requires_designs_it_must_avoid(root):
    with pytest.raises(FileNotFoundError):
        design.stage("lhs-extension", Options())
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
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from sgs import config, data, manifest
from sgs.stage import Options, produce


class DesignError(RuntimeError):
    """A design violates one of its own invariants."""


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


def read_design(role: str) -> pd.DataFrame:
    path = config.design_path(role)
    manifest.verify_provenance(path)
    return pd.read_csv(path)


def _build_lhs(role, spec, references):
    (seed_name,) = spec.seeds
    unit, frame = lhs(spec.n, config.SEEDS[seed_name])
    if not lhs_strata_complete(unit):
        raise DesignError(f"{role}: LHS stratification incomplete")
    return frame


_BUILDERS: dict[str, Callable] = {
    "lhs": _build_lhs,
}


def build(role: str) -> pd.DataFrame:
    spec = config.ROLES[role]
    if spec.design not in _BUILDERS:
        raise ValueError(f"unknown design procedure {spec.design!r} for role {role!r}")
    references = {r: read_design(r) for r in spec.disjoint_from}
    frame = _BUILDERS[spec.design](role, spec, references)
    if len(frame) != spec.n:
        raise DesignError(f"{role}: {len(frame)} points, expected {spec.n}")
    assert_disjoint(frame, references, config.MIN_DISTANCE)
    return frame


def stage(role: str, opts: Options) -> None:
    spec = config.ROLES[role]
    sections = ("domain", "design") + (("screener",) if spec.screener else ())
    inputs = {f"design:{r}": config.design_path(r) for r in spec.disjoint_from}
    if spec.screener:
        inputs[f"screener:{spec.screener}"] = config.screener_path(spec.screener)

    def build_to(tmp):
        frame = build(role)
        frame.to_csv(tmp, index=False)
        return {"n": len(frame), "columns": list(frame.columns)}

    produce(config.design_path(role), stage=f"design:{role}", sections=sections,
            inputs=inputs, opts=opts, build=build_to)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/design.py tests/test_design.py
git commit -m "Add LHS designs and the design stage

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: `solver.py` — the xslope/Bishop wrapper

A port of `run_analyses.py` (reference copy in `$REF`). Only what changes: all values come from `config.SOLVER`; the template is cached under `artifacts/solver/`; one analysis becomes one row `dict` keyed by `data.RESULT_COLUMNS`.

**Files:**
- Create: `sgs/solver.py`, `tests/test_solver.py`

**Interfaces:**
- Consumes: `config.SOLVER`, `config.solver_template_path()`, `data.RESULT_COLUMNS`.
- Produces:
  - `model_dims(H, angle, ext_factor=None, depth_factor=None) -> (slope, slope_length, extension, depth)`
  - `check_boundary(surface_left, surface_right, surface_depth, geometry: dict, H) -> (bool, list[str])`
  - `calculate_fos(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None) -> dict`. Keys: `fos, surface_left, surface_right, surface_depth, cx, cy, radius, at_boundary, boundary_flags, search_converged, n_circles_tested, n_valid_surfaces, n_unsolved, warnings, error, runtime_s`. `fos` is None on solver error.
  - `analysis_row(combo, analysis, ext_factor=None, depth_factor=None) -> dict` keyed by `data.RESULT_COLUMNS`
  - `analyse(combo) -> dict` — module-level, so it pickles under the `spawn` start method

- [ ] **Step 1: Write the failing tests `tests/test_solver.py`**

```python
import math

import pytest

from sgs import config, data, solver


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


@pytest.mark.slow
def test_one_real_analysis():
    row = solver.analyse((24.70810805776474, 33.64879751536226, 12.628212513088526,
                          17.341403519225082, 4.832465812168988))
    # Seed row 1 of the reference data: FOS 3.171339, not at boundary.
    assert row["Error"] == ""
    assert abs(row["FOS"] - 3.171339) < 1e-4
    assert row["At_Boundary"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_solver.py -q`
Expected: `ImportError: cannot import name 'solver'`.

- [ ] **Step 3: Write `sgs/solver.py`**

```python
"""xslope/Bishop wrapper: one homogeneous, dry, unloaded slope per analysis.

Every model is built in memory; the version-pinned xslope workbook supplies
only a valid input schema. Solver settings come from ``config.SOLVER``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

from sgs import config

_BASE_MODEL = None


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
```

- [ ] **Step 4: Run tests, including the slow one once**

Run: `.venv/bin/pytest tests/test_solver.py -q`
Expected: 3 passed, 1 deselected.

Run: `.venv/bin/pytest tests/test_solver.py -q -m slow`
Expected: 1 passed, in about 10 s. It uses the cached template in `artifacts/solver/`.

- [ ] **Step 5: Commit**

```bash
git add sgs/solver.py tests/test_solver.py
git commit -m "Port the xslope/Bishop wrapper into sgs.solver

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: `solve.py` — the solve stage (batches, resume, parallel parts)

A port of `run_batch`/`run_parallel`/`merge_parts` from `run_analyses.py`. Rows are dicts; parts are hidden files next to the stage's tmp payload and are deleted after a verified merge.

**Files:**
- Create: `sgs/solve.py`, `tests/test_solve.py`

**Interfaces:**
- Consumes: `solver.analyse`, `data.RESULT_COLUMNS`, `config.design_path/results_path`, `stage.produce(..., resumable=True)`.
- Produces: `read_rows(path) -> (list[dict], list[str])`, `validate_prefix(rows, combos, where)`, `run_batch(combos, outfile, label, analyse=solver.analyse)`, `part_path(outfile, k, n) -> Path`, `split_contiguous(combos, n_parts) -> list[ndarray]`, `merge_parts(part_specs, outfile, all_combos)`, `run_parallel(combos, outfile, label, workers, analyse=solver.analyse)`, `stage(role, opts)`.

- [ ] **Step 1: Write the failing tests `tests/test_solve.py`**

```python
import numpy as np
import pytest

from sgs import config, data, manifest, solve
from sgs.stage import Options
from tests.helpers import write_payload


def fake_analyse(combo):
    angle, phi, c, gamma, H = combo
    row = {column: "" for column in data.RESULT_COLUMNS}
    row.update(angle=angle, phi=phi, c=c, gamma=gamma, H=H, FOS=round(c / H, 6),
               At_Boundary=False, Search_Converged=True, Runtime_s=0.0)
    return row


def _combos(n=6):
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(lo, hi, n) for lo, hi in config.DOMAIN.values()])


def test_batch_writes_every_row(tmp_path):
    out = tmp_path / "r.csv"
    solve.run_batch(_combos(), out, "t", analyse=fake_analyse)
    rows, columns = solve.read_rows(out)
    assert columns == data.RESULT_COLUMNS and len(rows) == 6


def test_batch_resumes_where_it_stopped(tmp_path):
    combos = _combos()
    out = tmp_path / "r.csv"
    solve.run_batch(combos[:2], out, "t", analyse=fake_analyse)
    calls = []

    def counting(combo):
        calls.append(combo)
        return fake_analyse(combo)

    solve.run_batch(combos, out, "t", analyse=counting)
    assert len(calls) == 4 and len(solve.read_rows(out)[0]) == 6


def test_resume_refuses_another_design(tmp_path):
    out = tmp_path / "r.csv"
    solve.run_batch(_combos()[:2], out, "t", analyse=fake_analyse)
    other = _combos()[::-1].copy()
    with pytest.raises(RuntimeError, match="mismatch at row 1"):
        solve.run_batch(other, out, "t", analyse=fake_analyse)


def test_split_and_merge_preserve_order(tmp_path):
    combos = _combos(7)
    out = tmp_path / ".r.csv.tmp"
    chunks = solve.split_contiguous(combos, 3)
    assert [len(c) for c in chunks] == [3, 2, 2]
    specs = [(solve.part_path(out, k, 3), chunk) for k, chunk in enumerate(chunks, 1)]
    for path, chunk in specs:
        solve.run_batch(chunk, path, "p", analyse=fake_analyse)
    solve.merge_parts(specs, out, combos)
    rows, _ = solve.read_rows(out)
    np.testing.assert_allclose([[float(r[p]) for p in config.PARAMS] for r in rows], combos)
    assert not any(path.exists() for path, _ in specs)


def test_stage_writes_results_with_manifest(root, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(solve.solver, "analyse", fake_analyse)
    write_payload(config.design_path("seed"), pd.DataFrame(_combos(), columns=list(config.PARAMS)))
    solve.stage("seed", Options())
    record = manifest.read_manifest(config.results_path("seed"))
    assert record["inputs"]["design"]["path"] == "data/designs/seed_lhs_300.csv"
    assert record["extra"]["n"] == 6 and record["config"].keys() == {"domain"}


def test_stage_refuses_partial_output_without_resume(root, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(solve.solver, "analyse", fake_analyse)
    write_payload(config.design_path("seed"), pd.DataFrame(_combos(), columns=list(config.PARAMS)))
    tmp = config.results_path("seed").with_name(".seed_lhs_300.csv.tmp")
    solve.run_batch(_combos()[:2], tmp, "t", analyse=fake_analyse)
    with pytest.raises(RuntimeError, match="--resume"):
        solve.stage("seed", Options())
    solve.stage("seed", Options(resume=True))
    assert len(solve.read_rows(config.results_path("seed"))[0]) == 6
```

The stage uses `solver.analyse` looked up at call time (`solve.solver.analyse`), so the monkeypatch reaches it.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_solve.py -q`
Expected: `ImportError: cannot import name 'solve'`.

- [ ] **Step 3: Write `sgs/solve.py`**

```python
"""The solve stage: a design in, immutable raw solver output out.

Each completed analysis is appended and fsync'ed immediately, so an
interrupted run resumes with --resume. With --workers K the design is split
into K contiguous parts solved in separate processes and merged, in order,
after every part is verified against its slice of the design.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sgs import config, data, solver
from sgs.stage import Options, produce


def read_rows(path) -> tuple[list[dict], list[str]]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def validate_prefix(rows, combos, where) -> None:
    """Refuse rows that do not belong to the requested design, in order."""
    if len(rows) > len(combos):
        raise RuntimeError(f"{where} has more rows than the design")
    for index, row in enumerate(rows):
        previous = np.array([float(row[p]) for p in config.PARAMS])
        if not np.allclose(previous, combos[index], rtol=0.0, atol=1e-9):
            raise RuntimeError(f"cannot resume {where}: parameter mismatch at row {index + 1}")


def _write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=data.RESULT_COLUMNS).writeheader()
        stream.flush()
        os.fsync(stream.fileno())


def append_row(path: Path, row: dict) -> None:
    with path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=data.RESULT_COLUMNS).writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def run_batch(combos, outfile, label, analyse=None) -> None:
    analyse = analyse or solver.analyse
    combos = np.asarray(combos, dtype=float)
    outfile = Path(outfile)
    if outfile.exists():
        rows, columns = read_rows(outfile)
        if columns != data.RESULT_COLUMNS:
            raise RuntimeError(f"cannot resume {outfile.name}: schema differs")
        validate_prefix(rows, combos, outfile.name)
        done = len(rows)
        print(f"[{label}] resume: {done}/{len(combos)} already computed")
    else:
        _write_header(outfile)
        done = 0
    for index in range(done, len(combos)):
        row = analyse(combos[index])
        append_row(outfile, row)
        state = row["Error"] or f"FOS={row['FOS']}" + (" AT BOUNDARY" if row["At_Boundary"] is True else "")
        print(f"[{label} {index + 1}/{len(combos)}] {state}", flush=True)


def part_path(outfile, k: int, n: int) -> Path:
    outfile = Path(outfile)
    return outfile.with_name(f"{outfile.name}.part{k}of{n}.csv")


def split_contiguous(combos, n_parts: int) -> list[np.ndarray]:
    combos = np.asarray(combos, dtype=float)
    if len(combos) == 0:
        raise ValueError("cannot split an empty design")
    n_parts = min(int(n_parts), len(combos))
    return [combos[idx] for idx in np.array_split(np.arange(len(combos)), n_parts)]


def merge_parts(part_specs, outfile, all_combos) -> None:
    merged = []
    for path, expected in part_specs:
        rows, columns = read_rows(path)
        if columns != data.RESULT_COLUMNS or len(rows) != len(expected):
            raise RuntimeError(f"cannot merge {Path(path).name}: incomplete or wrong schema")
        validate_prefix(rows, expected, Path(path).name)
        merged.extend(rows)
    validate_prefix(merged, np.asarray(all_combos, dtype=float), "merged output")
    if len(merged) != len(all_combos):
        raise RuntimeError(f"merge produced {len(merged)} rows, expected {len(all_combos)}")
    with Path(outfile).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=data.RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
        stream.flush()
        os.fsync(stream.fileno())
    for path, _ in part_specs:
        Path(path).unlink()


def run_parallel(combos, outfile, label, workers, analyse=None) -> None:
    analyse = analyse or solver.analyse
    chunks = split_contiguous(combos, workers)
    specs = [(part_path(outfile, k, len(chunks)), chunk) for k, chunk in enumerate(chunks, 1)]
    context = mp.get_context("spawn")    # portable to macOS; no inherited solver state
    processes = []
    for k, (path, chunk) in enumerate(specs, 1):
        process = context.Process(target=run_batch, args=(chunk, path, f"{label}:part{k}/{len(chunks)}", analyse),
                                  name=f"{label}:part{k}")
        process.start()
        processes.append(process)
    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(f"{process.name} (exit={process.exitcode})")
    if failed:
        raise RuntimeError("solver workers failed: " + ", ".join(failed)
                           + "; completed parts are kept for --resume")
    merge_parts(specs, outfile, combos)


def stage(role: str, opts: Options) -> None:
    design_path = config.design_path(role)

    def build(tmp: Path):
        parts = sorted(tmp.parent.glob(f"{tmp.name}.part*.csv"))
        if (tmp.exists() or parts) and not opts.resume:
            raise RuntimeError(f"partial solver output for {role!r} exists; pass --resume to "
                               f"continue it, or delete {tmp.name}* to start over")
        if parts and not all(p.name.endswith(f"of{opts.workers}.csv") for p in parts):
            raise RuntimeError("partial output was written with a different --workers; resume with the same value")
        if opts.workers > 1 and tmp.exists():
            raise RuntimeError("partial output was written sequentially; resume with --workers 1")
        combos = pd.read_csv(design_path)[list(config.PARAMS)].to_numpy(float)
        if opts.workers == 1:
            run_batch(combos, tmp, role, analyse=solver.analyse)
        else:
            run_parallel(combos, tmp, role, opts.workers, analyse=solver.analyse)
        rows, _ = read_rows(tmp)
        return {"n": len(rows),
                "n_solver_error": sum(1 for r in rows if r["FOS"] == ""),
                "n_at_boundary": sum(1 for r in rows if r["At_Boundary"].strip().lower() == "true"),
                "workers": opts.workers}

    produce(config.results_path(role), stage=f"solve:{role}", sections=("domain",),
            inputs={"design": design_path}, opts=opts, build=build, resumable=True)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/solve.py tests/test_solve.py
git commit -m "Add the solve stage with resumable and parallel batches

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: `repair.py` — the repair stage

A port of `run_repair.py`. It reads from `data/results/` and writes to `data/final/`, and never rewrites its input.

**Files:**
- Create: `sgs/repair.py`, `tests/test_repair.py`

**Interfaces:**
- Consumes: `solver.calculate_fos`, `solver.analysis_row`, `solve.read_rows`, `config.REPAIR_VARIANTS`, `data.RESULT_COLUMNS/FINAL_COLUMNS`.
- Produces: `variant_label(ext_factor, depth_factor) -> str`, `repair_rows(rows: list[dict], calculate=None) -> dict` (mutates rows, returns summary), `stage(role, opts)`.

- [ ] **Step 1: Write the failing tests `tests/test_repair.py`**

```python
import csv

import pytest

from sgs import config, data, manifest, repair
from sgs.stage import Options


def _row(fos="1.5", boundary="False", c=10.0):
    row = {column: "" for column in data.RESULT_COLUMNS}
    row.update(angle="45.0", phi="30.0", c=str(c), gamma="18.0", H="10.0", FOS=fos, At_Boundary=boundary)
    return row


def fake_calculate(frees_at):
    """Returns a boundary-free surface only on the given variant."""
    def calculate(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None):
        free = (ext_factor, depth_factor) == frees_at
        return {"fos": 1.2, "surface_left": 0.0, "surface_right": 1.0, "surface_depth": -1.0,
                "cx": 0.0, "cy": 1.0, "radius": 2.0, "at_boundary": not free, "boundary_flags": [],
                "search_converged": True, "n_circles_tested": 1, "n_valid_surfaces": 1,
                "n_unsolved": 0, "warnings": "", "error": "", "runtime_s": 0.1}
    return calculate


def test_label():
    assert repair.variant_label(2.0, None) == "ext2.0x_depth1x"


def test_rows_without_problems_are_copied():
    rows = [_row(), _row()]
    summary = repair.repair_rows(rows, calculate=fake_calculate(None))
    assert summary["copied_unchanged"] and summary["n_boundary"] == 0
    assert all(r["repair_variant"] == "" and r["excluded"] == "" for r in rows)


def test_boundary_row_is_repaired_by_the_first_freeing_variant():
    rows = [_row(boundary="True")]
    summary = repair.repair_rows(rows, calculate=fake_calculate((None, 2.0)))
    assert summary["n_repaired"] == 1
    assert rows[0]["repair_variant"] == "ext1x_depth2.0x"
    assert rows[0]["FOS"] == 1.2 and rows[0]["At_Boundary"] is False and rows[0]["excluded"] == ""


def test_unfreeable_row_is_excluded():
    rows = [_row(boundary="True")]
    summary = repair.repair_rows(rows, calculate=fake_calculate(("never", "never")))
    assert summary["n_unresolved"] == 1 and rows[0]["excluded"] == "unresolved"


def test_solver_error_is_excluded_not_repaired():
    rows = [_row(fos="")]
    summary = repair.repair_rows(rows, calculate=fake_calculate(None))
    assert summary["n_solver_error"] == 1 and rows[0]["excluded"] == "solver_error"


def test_stage_writes_final_and_leaves_results_untouched(root, monkeypatch):
    monkeypatch.setattr(repair.solver, "calculate_fos", fake_calculate((2.0, None)))
    results = config.results_path("seed")
    results.parent.mkdir(parents=True)
    with results.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=data.RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows([_row(), _row(boundary="True", c=20.0)])
    manifest.write_manifest(results, stage="fixture", sections=(), inputs={}, wall_time_s=0)
    before = manifest.sha256_file(results)
    repair.stage("seed", Options())
    assert manifest.sha256_file(results) == before
    final = data.read_table(config.final_path("seed"))
    assert list(final.columns) == data.FINAL_COLUMNS
    assert final["repair_variant"].tolist() == ["", "ext2.0x_depth1x"]
    assert manifest.read_manifest(config.final_path("seed"))["extra"]["n_repaired"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_repair.py -q`
Expected: `ImportError: cannot import name 'repair'`.

- [ ] **Step 3: Write `sgs/repair.py`**

```python
"""The repair stage: boundary-truncated rows recomputed on enlarged domains.

A critical surface that reaches the analysis domain was cut short by the model
extent, so its FOS is not the true minimum. Each such row is recomputed with
the same solver and search settings on the enlarged domains of
``config.REPAIR_VARIANTS``, in order, until the surface sits inside the
domain. Rows no variant frees are excluded as ``unresolved``; rows the solver
could not analyse are excluded as ``solver_error``. The input in
data/results/ is never modified.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from sgs import config, data, solve, solver
from sgs.stage import Options, produce


def variant_label(ext_factor, depth_factor) -> str:
    return f"ext{ext_factor or 1}x_depth{depth_factor or 1}x"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def repair_rows(rows: list[dict], calculate=None) -> dict:
    calculate = calculate or solver.calculate_fos
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
            row.update(solver.analysis_row(combo, analysis, ext_factor, depth_factor))
            row["repair_variant"] = label
            n_repaired += 1
            print(f"  row {index + 1}: repaired with {label}, FOS={analysis['fos']:.4f}")
            break
        else:
            row["excluded"] = "unresolved"
            print(f"  row {index + 1}: no variant freed the surface; excluded")
    return {"n_rows": len(rows), "n_boundary": n_boundary, "n_repaired": n_repaired,
            "n_unresolved": n_boundary - n_repaired, "n_solver_error": n_error,
            "copied_unchanged": n_boundary == 0 and n_error == 0}


def stage(role: str, opts: Options) -> None:
    results = config.results_path(role)

    def build(tmp: Path):
        rows, columns = solve.read_rows(results)
        if columns != data.RESULT_COLUMNS:
            raise RuntimeError(f"{results.name} does not match the result schema")
        summary = repair_rows(rows)
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data.FINAL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[repair:{role}] {summary}")
        return summary

    produce(config.final_path(role), stage=f"repair:{role}", sections=("domain",),
            inputs={"results": results}, opts=opts, build=build)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/repair.py tests/test_repair.py
git commit -m "Add the repair stage writing data/final from immutable results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Run the seed chain and compare it descriptively

**EXPENSIVE: about 300 solver analyses, unattended.**

**Files:**
- Create: `tools/describe_sets.py`
- Produces (untracked payloads, tracked manifests): `data/designs/seed_lhs_300.csv`, `data/results/seed_lhs_300.csv`, `data/final/seed_lhs_300.csv`

**Interfaces:**
- Produces: `tools/describe_sets.py <reference-dir>` prints, for every final set present, the columns n, excluded, min, median, max FOS, occupancy of the 15 selection-range bins and max |Δparam|, next to the reference copy under its old name. The author reuses it in Task 16.

- [ ] **Step 1: Write `tools/describe_sets.py`**

```python
"""Descriptive comparison of regenerated final sets with the reference copies.

For the author's eyes only: nothing in the pipeline reads the reference
directory. Usage: .venv/bin/python tools/describe_sets.py <reference-dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sgs import config, data

OLD_NAMES = {
    "seed": "initial_lhs_300_bishop.csv",
    "lhs-extension": "results_lhs_bishop.csv",
    "guided": "results_uniform_bishop.csv",
    "test-natural": "results_test_bishop.csv",
    "test-stratified": "results_test_150_bishop.csv",
}


def describe(frame):
    kept = data.usable(frame)
    fos = kept["FOS"].to_numpy(float)
    edges = np.linspace(*config.SELECTION_RANGE, config.N_BINS + 1)
    occupancy = np.histogram(fos, bins=edges)[0].tolist()
    return {"n": len(frame), "excluded": len(frame) - len(kept), "min": fos.min(),
            "median": np.median(fos), "max": fos.max(), "bins": occupancy}


reference_dir = Path(sys.argv[1])
for role in config.ROLE_ORDER:
    path = config.final_path(role)
    if not path.exists():
        continue
    new = data.read_table(path)
    old = data.read_table(reference_dir / OLD_NAMES[role])
    if "excluded" not in old:
        old["excluded"] = ""
    a, b = describe(new), describe(old)
    dparam = np.abs(new[list(config.PARAMS)].to_numpy(float) - old[list(config.PARAMS)].to_numpy(float)).max()
    print(f"\n== {role} ({path.name} vs {OLD_NAMES[role]}) — max |Δparam| {dparam:.2e}")
    for key in ("n", "excluded", "min", "median", "max"):
        print(f"  {key:9} new {a[key]!s:>10}   ref {b[key]!s:>10}")
    print(f"  bins new {a['bins']}\n  bins ref {b['bins']}")
    same_rows = np.isclose(new["FOS"].to_numpy(float), old["FOS"].to_numpy(float), atol=1e-6, equal_nan=True)
    print(f"  rows with identical FOS (±1e-6): {int(same_rows.sum())}/{len(new)}")
```

- [ ] **Step 2: Run the chain**

```bash
.venv/bin/sgs design --role seed
.venv/bin/sgs solve --role seed --workers 6
.venv/bin/sgs repair --role seed
```
If interrupted, re-run `solve` with `--resume --workers 6`. The worker count must stay the same.

Expected: `wrote seed_lhs_300.csv` three times; the repair summary names how many boundary rows were found and repaired.

- [ ] **Step 3: Descriptive comparison**

```bash
.venv/bin/python tools/describe_sets.py "$REF"
```
Expected: `max |Δparam|` = 0, since the seed design is regenerated bit-identically. Most rows should have identical FOS: the search is deterministic, so any difference comes from repair. Report any row-level FOS differences to the author. This is a descriptive check, not a gate.

- [ ] **Step 4: Commit (tool and manifests; payloads are ignored)**

```bash
git add tools/describe_sets.py data/designs/*.manifest.json data/results/*.manifest.json data/final/*.manifest.json
git status --short data/     # must list only *.manifest.json
git commit -m "Regenerate the seed chain through the pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 3 — the screener

### Task 13: `screener.py` and `sgs fit-screener`

**Files:**
- Create: `sgs/screener.py`, `tests/test_screener.py`

**Interfaces:**
- Consumes: `data.load_role`, `config.SCREENERS`, `manifest.require_fresh`, `stage.produce`.
- Produces: `SECTIONS = ("domain", "screener")`, `inputs_for(name) -> dict[str, Path]`, `build_model(spec, *, oob: bool) -> RandomForestRegressor`, `fit(name) -> (model, diagnostics: dict)`, `load(name) -> fitted model` (fails on stale), `stage(opts, name=None)`.

- [ ] **Step 1: Write the failing tests `tests/test_screener.py`**

```python
import joblib
import numpy as np
import pytest

from sgs import config, manifest, screener
from sgs.stage import Options
from tests.helpers import synthetic_final, write_final


@pytest.fixture
def small_screener(root, monkeypatch):
    spec = config.SCREENERS["primary"]
    small = type(spec)(trained_on=spec.trained_on, family="rf", params={"n_estimators": 20},
                       seeds=dict(spec.seeds), cv_folds=3)
    monkeypatch.setitem(config.SCREENERS, "primary", small)
    write_final("seed", synthetic_final(80, 5))
    return small


def test_model_uses_named_seeds(small_screener):
    model = screener.build_model(small_screener, oob=True)
    assert model.random_state == small_screener.seeds["fit"] and model.oob_score


def test_non_rf_family_is_refused(small_screener):
    with pytest.raises(ValueError):
        screener.build_model(type(small_screener)(trained_on="seed", family="gbm"), oob=False)


def test_stage_fits_on_log_fos_and_records_diagnostics(small_screener):
    screener.stage(Options())
    path = config.screener_path("primary")
    diagnostics = manifest.read_manifest(path)["extra"]
    assert diagnostics["n_train"] == 80 and 0 < diagnostics["oob_r2_log"] <= 1
    model = screener.load("primary")
    assert np.all(np.exp(model.predict(np.zeros((1, len(config.FEATURES))) + 10)) > 0)


def test_fit_is_deterministic(small_screener):
    a, _ = screener.fit("primary")
    b, _ = screener.fit("primary")
    probe = synthetic_final(10, 9)
    from sgs import data
    X = data.add_features(probe)[list(config.FEATURES)]
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_load_refuses_a_stale_screener(small_screener, monkeypatch):
    screener.stage(Options())
    monkeypatch.setitem(config.SCREENERS, "primary",
                        type(small_screener)(trained_on="seed", family="rf",
                                             params={"n_estimators": 21}, seeds=dict(small_screener.seeds)))
    with pytest.raises(manifest.StaleArtifact, match="screener"):
        screener.load("primary")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_screener.py -q`
Expected: `ImportError: cannot import name 'screener'`.

- [ ] **Step 3: Write `sgs/screener.py`**

```python
"""The preliminary surrogate: fitted once on its training role, loaded thereafter.

Fixed hyper-parameters, not a tuned search. It screens the candidate pools of
the guided and target-stratified designs; it is not one of the models under
comparison.
"""

from __future__ import annotations

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from sgs import config, data, manifest
from sgs.stage import Options, produce

SECTIONS = ("domain", "screener")


def inputs_for(name: str) -> dict:
    trained_on = config.SCREENERS[name].trained_on
    return {f"final:{trained_on}": config.final_path(trained_on)}


def build_model(spec: config.Screener, *, oob: bool) -> RandomForestRegressor:
    if spec.family != "rf":
        raise ValueError(f"screener family {spec.family!r} is not supported")
    return RandomForestRegressor(**spec.params, oob_score=oob,
                                 random_state=spec.seeds["fit"], n_jobs=-1)


def fit(name: str):
    spec = config.SCREENERS[name]
    frame = data.load_role(spec.trained_on)
    X = frame[list(config.FEATURES)]
    fos = frame["FOS"].to_numpy(float)
    y = np.log(fos)
    model = build_model(spec, oob=True).fit(X, y)

    cv = KFold(spec.cv_folds, shuffle=True, random_state=spec.seeds["cv"])
    oof = np.exp(cross_val_predict(build_model(spec, oob=False), X, y, cv=cv))
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


def load(name: str):
    path = config.screener_path(name)
    manifest.require_fresh(path, sections=SECTIONS, inputs=inputs_for(name))
    return joblib.load(path)


def stage(opts: Options, name: str | None = None) -> None:
    for screener_name in ([name] if name else list(config.SCREENERS)):
        def build(tmp, screener_name=screener_name):
            model, diagnostics = fit(screener_name)
            joblib.dump(model, tmp)
            print(f"[fit-screener:{screener_name}] {diagnostics}")
            return diagnostics

        produce(config.screener_path(screener_name), stage=f"fit-screener:{screener_name}",
                sections=SECTIONS, inputs=inputs_for(screener_name), opts=opts, build=build)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Fit the real screener**

```bash
.venv/bin/sgs fit-screener
```
Expected: `oob_r2_log` close to the 0.9567 the notebook reported. The values are identical if repair left the seed FOS identical to the reference.

- [ ] **Step 6: Commit**

```bash
git add sgs/screener.py tests/test_screener.py artifacts/screeners/primary.joblib.manifest.json
git commit -m "Add the preliminary screener stage

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 4 — the remaining four roles

### Task 14: `guided-maxmin` design

**Files:**
- Modify: `sgs/design.py` (add helpers and register the builder)
- Modify: `tests/test_design.py` (append tests)

**Interfaces:**
- Consumes: `screener.load(name)` → an object with `.predict(X_features) -> log FOS`.
- Produces: `uniform_pool(rng, n) -> DataFrame`, `predict_fos(model, frame) -> ndarray`, `greedy_maxmin(Xs, k, rng) -> ndarray`, `bin_edges(n_bins, selection_range) -> ndarray`, `guided_maxmin(model, *, rng, n_pool, n_bins, per_bin, drawn_per_bin, selection_range) -> DataFrame` (columns `PARAMS + ["FOS_pred", "bin"]`), `_BUILDERS["guided-maxmin"]`.

- [ ] **Step 1: Append failing tests to `tests/test_design.py`**

```python
# --- screener-driven designs -------------------------------------------------------

import dataclasses

import joblib

from tests.helpers import FormulaModel, synthetic_final, write_final, write_payload


@pytest.fixture
def small_designs(root, monkeypatch):
    """Shrink the screener designs and install a formula screener artifact."""
    monkeypatch.setattr(config, "N_GUIDED_POOL", 6000)
    monkeypatch.setattr(config, "N_TEST_POOL", 6000)
    monkeypatch.setattr(config, "N_BINS", 5)
    monkeypatch.setattr(config, "GUIDED_PER_BIN", 4)
    monkeypatch.setattr(config, "GUIDED_DRAWN_PER_BIN", 6)
    monkeypatch.setattr(config, "TEST_PER_BIN", 3)
    roles = dict(config.ROLES)
    roles["guided"] = dataclasses.replace(roles["guided"], n=20)
    roles["test-stratified"] = dataclasses.replace(roles["test-stratified"], n=15)
    monkeypatch.setattr(config, "ROLES", roles)
    write_final("seed", synthetic_final(50, 1))
    path = config.screener_path("primary")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(FormulaModel(), path)
    manifest.write_manifest(path, stage="fixture", sections=("domain", "screener"),
                            inputs={"final:seed": config.final_path("seed")}, wall_time_s=0)


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


def test_guided_role_is_deterministic_through_the_stage(small_designs):
    design.stage("seed", Options())
    design.stage("guided", Options())
    assert _hash(design.build("guided")) == manifest.sha256_file(config.design_path("guided"))
    assert "screener" in manifest.read_manifest(config.design_path("guided"))["config"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_design.py -q`
Expected: new tests fail with `AttributeError: module 'sgs.design' has no attribute 'guided_maxmin'`.

- [ ] **Step 3: Add to `sgs/design.py`**

Add `from sgs import screener` to the imports. Insert the following above `_BUILDERS`:

```python
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


def _build_guided(role, spec, references):
    (stream,) = spec.seeds
    return guided_maxmin(screener.load(spec.screener),
                         rng=np.random.default_rng(config.SEEDS[stream]),
                         n_pool=config.N_GUIDED_POOL, n_bins=config.N_BINS,
                         per_bin=config.GUIDED_PER_BIN, drawn_per_bin=config.GUIDED_DRAWN_PER_BIN,
                         selection_range=config.SELECTION_RANGE)
```

Register it:

```python
_BUILDERS: dict[str, Callable] = {
    "lhs": _build_lhs,
    "guided-maxmin": _build_guided,
}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/design.py tests/test_design.py
git commit -m "Add the guided max-min design

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: `stratified-random` design

**Files:**
- Modify: `sgs/design.py`, `tests/test_design.py`

**Interfaces:**
- Produces: `stratified_random(model, *, pool_rng, selection_rng, n_pool, n_bins, per_bin, selection_range, references: Mapping[str, DataFrame], min_distance) -> DataFrame` (columns `PARAMS + ["FOS_pred", "bin"]`), `_BUILDERS["stratified-random"]`.

- [ ] **Step 1: Append failing tests to `tests/test_design.py`**

```python
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


def test_every_role_is_deterministic_through_the_stage(small_designs):
    for role in config.ROLE_ORDER:
        design.stage(role, Options())
    for role in config.ROLE_ORDER:
        assert _hash(design.build(role)) == manifest.sha256_file(config.design_path(role)), role
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_design.py -q`
Expected: new tests fail with `AttributeError: ... 'stratified_random'`.

- [ ] **Step 3: Add to `sgs/design.py` above `_BUILDERS`**

```python
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


def _build_stratified(role, spec, references):
    pool_seed, selection_seed = spec.seeds
    return stratified_random(screener.load(spec.screener),
                             pool_rng=np.random.default_rng(config.SEEDS[pool_seed]),
                             selection_rng=np.random.default_rng(config.SEEDS[selection_seed]),
                             n_pool=config.N_TEST_POOL, n_bins=config.N_BINS,
                             per_bin=config.TEST_PER_BIN, selection_range=config.SELECTION_RANGE,
                             references=references, min_distance=config.MIN_DISTANCE)
```

Register it:

```python
_BUILDERS: dict[str, Callable] = {
    "lhs": _build_lhs,
    "guided-maxmin": _build_guided,
    "stratified-random": _build_stratified,
}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/design.py tests/test_design.py
git commit -m "Add the target-stratified random design

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: Run the four remaining chains

**EXPENSIVE: about 1050 solver analyses, unattended — the largest single block of machine time in the project.**

- [ ] **Step 1: Designs first (cheap), then compare inputs before spending solver time**

```bash
for role in lhs-extension guided test-natural test-stratified; do .venv/bin/sgs design --role $role; done
.venv/bin/python - <<'EOF'
import numpy as np, pandas as pd
from sgs import config
REF = "/Users/bogdanteodoru/Documents/scoala/lucrari/FOS_by_ML/reference-data-2026-09/"
old = {"lhs-extension": "results_lhs_bishop.csv", "guided": "results_uniform_bishop.csv",
       "test-natural": "results_test_bishop.csv", "test-stratified": "results_test_150_bishop.csv"}
for role, name in old.items():
    a = pd.read_csv(config.design_path(role))[list(config.PARAMS)].to_numpy()
    b = pd.read_csv(REF + name)[list(config.PARAMS)].to_numpy()
    print(role, a.shape, "max |Δparam|", np.abs(a - b).max())
EOF
```
Expected: `max |Δparam|` ≤ 5e-7 for all four, the rounding of the old 6-decimal arrays. If `guided` or `test-stratified` differ by more, the screener changed. That happens only if repair changed seed FOS values relative to the reference. **Report to the author before solving.** It is a legitimate consequence of the regeneration, but it has to be seen before ~1050 analyses are spent on it.

- [ ] **Step 2: Solve and repair each role (unattended)**

```bash
for role in lhs-extension guided test-natural test-stratified; do
  .venv/bin/sgs solve --role $role --workers 6 && .venv/bin/sgs repair --role $role || break
done
```
After an interruption, re-run the same loop with `--resume` added to `solve`.

- [ ] **Step 3: Descriptive comparison**

```bash
.venv/bin/python tools/describe_sets.py "$REF"
```
Expected: n, min, median, max and bin occupancy close to the reference copies. Differences are confined to rows the previous run left boundary-truncated; the old run had eleven across these four sets. Record the output for the author.

- [ ] **Step 4: Commit manifests**

```bash
git add data/designs/*.manifest.json data/results/*.manifest.json data/final/*.manifest.json
git status --short data/
git commit -m "Regenerate the extension, guided and test chains through the pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 5 — check

### Task 17: `checks.py`, the receipt, and `sgs check`

**Files:**
- Create: `sgs/checks.py`, `tests/test_checks.py`

**Interfaces:**
- Consumes: `data.read_table/normalize/is_true/FINAL_COLUMNS/EXCLUSION_REASONS`, `manifest.sha256_file/verify_provenance/write_manifest`.
- Produces: `Check(name, passed, detail)`, `ReceiptError(RuntimeError)`, `check_set(frame, role) -> list[Check]`, `check_pair(a, b) -> Check`, `pairs_among(roles) -> list[tuple[str, str]]`, `load_receipt() -> dict`, `update_receipt(roles) -> dict`, `require(roles, pairs=()) -> None`, `stage(opts)`.
- Receipt shape: `{"sets": {<sha>: {"role", "file", "passed", "checks": [...], "config": {...}, "checked"}}, "pairs": {"<sha>|<sha>": {"roles", "passed", "detail", "config", "checked"}}}`. An entry is reused only while its file hashes **and** its `config` stamp (the `domain` and `design` section hashes) match.

- [ ] **Step 1: Write the failing tests `tests/test_checks.py`**

```python
import numpy as np
import pytest

from sgs import checks, config, data
from sgs.stage import Options
from tests.helpers import synthetic_final, write_final


@pytest.fixture
def clean_sets(root):
    for i, role in enumerate(config.ROLE_ORDER):
        write_final(role, synthetic_final(config.ROLES[role].n, 100 + i))


def _failed(frame, role="seed"):
    return {c.name for c in checks.check_set(frame, role) if not c.passed}


def test_clean_set_passes():
    assert _failed(synthetic_final(300, 1)) == set()


@pytest.mark.parametrize("name, corrupt", [
    ("schema", lambda f: f.drop(columns="Error")),
    ("row_count", lambda f: f.iloc[:-1]),
    ("domain", lambda f: f.assign(angle=np.r_[70.0, f["angle"].to_numpy()[1:]])),
    ("exclusion_reasons", lambda f: f.assign(excluded=["odd"] + [""] * (len(f) - 1))),
    ("fos_present", lambda f: f.assign(FOS=np.r_[np.nan, f["FOS"].to_numpy()[1:]])),
    ("no_untreated_boundary", lambda f: f.assign(At_Boundary=["True"] + ["False"] * (len(f) - 1))),
    ("no_internal_duplicates", lambda f: f.assign(**{p: np.r_[f[p].iloc[1], f[p].to_numpy()[1:]] for p in config.PARAMS})),
])
def test_each_violation_is_named(name, corrupt):
    assert name in _failed(corrupt(synthetic_final(300, 1)))


def test_excluded_rows_are_not_held_to_fos_or_boundary_rules():
    frame = synthetic_final(300, 1)
    frame.loc[0, ["FOS", "excluded"]] = [np.nan, "solver_error"]
    frame.loc[1, ["At_Boundary", "excluded"]] = ["True", "unresolved"]
    assert _failed(frame) == set()


def test_overlapping_pair_fails():
    a = synthetic_final(50, 1)
    assert not checks.check_pair(a, a.copy()).passed
    assert checks.check_pair(a, synthetic_final(50, 2)).passed


def test_stage_writes_a_complete_receipt(clean_sets):
    checks.stage(Options())
    receipt = checks.load_receipt()
    assert len(receipt["sets"]) == 5 and len(receipt["pairs"]) == 10
    assert all(entry["passed"] for entry in receipt["sets"].values())
    checks.require(list(config.ROLE_ORDER), checks.pairs_among(config.ROLE_ORDER))


def test_regenerating_one_set_rechecks_only_that_set(clean_sets, monkeypatch):
    checks.stage(Options())
    write_final("guided", synthetic_final(300, 999))
    with pytest.raises(checks.ReceiptError, match="guided: not checked"):
        checks.require(["guided"])
    calls = {"set": 0, "pair": 0}
    real_set, real_pair = checks.check_set, checks.check_pair
    monkeypatch.setattr(checks, "check_set", lambda *a: calls.__setitem__("set", calls["set"] + 1) or real_set(*a))
    monkeypatch.setattr(checks, "check_pair", lambda *a: calls.__setitem__("pair", calls["pair"] + 1) or real_pair(*a))
    checks.stage(Options())
    assert calls == {"set": 1, "pair": 4}
    assert len(checks.load_receipt()["sets"]) == 5


def test_failing_set_fails_the_stage_after_recording_it(clean_sets):
    frame = synthetic_final(300, 100)
    frame.loc[0, "At_Boundary"] = "True"
    write_final("seed", frame)
    with pytest.raises(checks.ReceiptError, match="seed"):
        checks.stage(Options())
    with pytest.raises(checks.ReceiptError, match="no_untreated_boundary"):
        checks.require(["seed"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_checks.py -q`
Expected: `ImportError: cannot import name 'checks'`.

- [ ] **Step 3: Write `sgs/checks.py`**

```python
"""Validation of the final sets: per-set checks and pairwise disjointness.

``sgs check`` writes data/checks/receipt.json. Per-set entries are keyed on
the hash of the file in data/final/, pairwise entries on the sorted pair of
hashes, so regenerating one set invalidates its entry and the pairs that
contain it, and nothing else. Stages that consume data call ``require``.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass

from scipy.spatial import cKDTree

from sgs import config, data, manifest
from sgs.stage import Options


class ReceiptError(RuntimeError):
    """The receipt does not vouch for the data about to be used."""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def check_set(frame, role: str) -> list[Check]:
    spec = config.ROLES[role]
    schema_ok = list(frame.columns) == data.FINAL_COLUMNS
    result = [Check("schema", schema_ok, "final schema" if schema_ok else f"columns {list(frame.columns)}")]
    if not schema_ok:
        return result
    result.append(Check("row_count", len(frame) == spec.n, f"{len(frame)} rows, expected {spec.n}"))
    outside = {p: int((~frame[p].between(lo, hi)).sum()) for p, (lo, hi) in config.DOMAIN.items()}
    result.append(Check("domain", not any(outside.values()), f"rows outside the domain: {outside}"))
    reasons = frame["excluded"]
    unknown = sorted(set(reasons) - {"", *data.EXCLUSION_REASONS})
    result.append(Check("exclusion_reasons", not unknown,
                        f"unknown reasons {unknown}" if unknown else f"{int((reasons != '').sum())} excluded"))
    kept = reasons == ""
    bad_fos = int((kept & ~(frame["FOS"] > 0)).sum())
    result.append(Check("fos_present", bad_fos == 0, f"{bad_fos} kept rows without a positive FOS"))
    boundary = int((kept & data.is_true(frame["At_Boundary"])).sum())
    result.append(Check("no_untreated_boundary", boundary == 0, f"{boundary} kept rows at the domain boundary"))
    unit = data.normalize(frame)
    closest = float(cKDTree(unit).query(unit, k=2)[0][:, 1].min())
    result.append(Check("no_internal_duplicates", closest > config.MIN_DISTANCE, f"closest pair {closest:.2e}"))
    return result


def check_pair(a, b) -> Check:
    closest = float(cKDTree(data.normalize(b)).query(data.normalize(a), k=1)[0].min())
    return Check("disjoint", closest > config.MIN_DISTANCE, f"closest cross pair {closest:.2e}")


def pairs_among(roles) -> list[tuple[str, str]]:
    return list(itertools.combinations(sorted(roles), 2))


def _pair_key(h1: str, h2: str) -> str:
    return "|".join(sorted((h1, h2)))


def _config_stamp() -> dict:
    return {name: config.section_hash(name) for name in ("domain", "design")}


def _hashes(roles) -> dict:
    out = {}
    for role in roles:
        path = config.final_path(role)
        manifest.verify_provenance(path)
        out[role] = manifest.sha256_file(path)
    return out


def load_receipt() -> dict:
    path = config.receipt_path()
    if not path.exists():
        return {"sets": {}, "pairs": {}}
    return json.loads(path.read_text())


def update_receipt(roles) -> dict:
    """Reuse entries still valid for the current files; check the rest."""
    previous, stamp = load_receipt(), _config_stamp()
    hashes = _hashes(roles)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    frames = {}

    def frame(role):
        if role not in frames:
            frames[role] = data.read_table(config.final_path(role))
        return frames[role]

    sets = {}
    for role in roles:
        entry = previous["sets"].get(hashes[role])
        if entry is None or entry["role"] != role or entry["config"] != stamp:
            result = check_set(frame(role), role)
            entry = {"role": role, "file": config.final_path(role).name,
                     "passed": all(c.passed for c in result),
                     "checks": [asdict(c) for c in result], "config": stamp, "checked": now}
        sets[hashes[role]] = entry
    pairs = {}
    for a, b in pairs_among(roles):
        key = _pair_key(hashes[a], hashes[b])
        entry = previous["pairs"].get(key)
        if entry is None or entry["config"] != stamp:
            result = check_pair(frame(a), frame(b))
            entry = {"roles": [a, b], "passed": result.passed, "detail": result.detail,
                     "config": stamp, "checked": now}
        pairs[key] = entry
    return {"sets": sets, "pairs": pairs}


def require(roles, pairs=()) -> None:
    """Fail unless the receipt vouches for every set and pair, as they are now."""
    receipt, stamp = load_receipt(), _config_stamp()
    hashes = _hashes(set(roles) | {r for pair in pairs for r in pair})
    problems = []
    for role in roles:
        entry = receipt["sets"].get(hashes[role])
        if entry is None or entry["config"] != stamp:
            problems.append(f"{role}: not checked in its current state")
        elif not entry["passed"]:
            failed = [c["name"] for c in entry["checks"] if not c["passed"]]
            problems.append(f"{role}: failed {', '.join(failed)}")
    for a, b in pairs:
        entry = receipt["pairs"].get(_pair_key(hashes[a], hashes[b]))
        if entry is None or entry["config"] != stamp:
            problems.append(f"{a}/{b}: disjointness not checked")
        elif not entry["passed"]:
            problems.append(f"{a}/{b}: not disjoint ({entry['detail']})")
    if problems:
        raise ReceiptError("the receipt does not cover the data: " + "; ".join(problems)
                           + ". Run `sgs check`.")


def stage(opts: Options) -> None:
    roles = list(config.ROLE_ORDER)
    missing = [r for r in roles if not config.final_path(r).exists()]
    if opts.dry_run:
        print(f"[check] dry run: would check {len(roles)} sets"
              + (f"; waits for {', '.join(missing)}" if missing else ""))
        return
    if missing:
        raise FileNotFoundError(f"final sets missing: {', '.join(missing)}")
    started = time.perf_counter()
    receipt = update_receipt(roles)
    path = config.receipt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

    failures = []
    for entry in receipt["sets"].values():
        print(f"[check] {'PASS' if entry['passed'] else 'FAIL'} {entry['role']}")
        if not entry["passed"]:
            failures.append(entry["role"] + ": " + ", ".join(c["name"] for c in entry["checks"] if not c["passed"]))
    for entry in receipt["pairs"].values():
        print(f"[check] {'PASS' if entry['passed'] else 'FAIL'} {' / '.join(entry['roles'])} ({entry['detail']})")
        if not entry["passed"]:
            failures.append(" / ".join(entry["roles"]) + " not disjoint")
    manifest.write_manifest(path, stage="check", sections=("domain", "design"),
                            inputs={f"final:{r}": config.final_path(r) for r in roles},
                            wall_time_s=time.perf_counter() - started,
                            extra={"passed": not failures})
    if failures:
        raise ReceiptError("checks failed: " + "; ".join(failures))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Run on the real data**

```bash
.venv/bin/sgs check
```
Expected: 5 set lines and 10 pair lines, all `PASS`. A `FAIL` here means something upstream is wrong (spec §10, phase 5). Stop and report; do not proceed to fitting.

- [ ] **Step 6: Commit**

```bash
git add sgs/checks.py tests/test_checks.py data/checks/receipt.json data/checks/receipt.json.manifest.json
git commit -m "Add the check stage and its hash-keyed receipt

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 6 — protocol and fitting

### Task 18: `protocol.py` — `searcher()` and `fit_arm()`

**Files:**
- Create: `sgs/protocol.py`, `tests/test_protocol.py`
- Modify: `tests/conftest.py` (add the `tiny_protocol` fixture)

**Interfaces:**
- Consumes: `data.load_arm`, `config.GRIDS/ESTIMATOR_FIXED/CV_FOLDS/SEARCH_KIND/SEEDS/TARGET`.
- Produces:
  - `FitArtifact` dataclass: `family, arm, scoring, target, model, best_params: dict, cv_results: dict, train_ids: ndarray[str], oof_pred: ndarray` (log scale), `oof_residual: ndarray` (`y_log − oof_pred`, signed), `fold: ndarray[int]`; property `scores -> |oof_residual|`
  - `estimator(family)`, `scorer(scoring)`, `cv_splitter() -> KFold`, `searcher(family, scoring) -> GridSearchCV`, `fit_arm(family, arm, scoring) -> FitArtifact`

- [ ] **Step 1: Add the shared fixture to `tests/conftest.py`**

```python
@pytest.fixture
def tiny_protocol(root, monkeypatch):
    """Small grids and three small training sets, for fast fitting tests."""
    from tests.helpers import synthetic_final, write_final

    monkeypatch.setattr(config, "GRIDS", {"rf": {"min_samples_leaf": [1, 5]},
                                          "gbm": {"max_iter": [20], "min_samples_leaf": [3, 10]}})
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20},
                                                    "gbm": {"early_stopping": False}})
    monkeypatch.setattr(config, "CV_FOLDS", 3)
    for i, role in enumerate(("seed", "lhs-extension", "guided")):
        write_final(role, synthetic_final(60, 10 + i))
```

- [ ] **Step 2: Write the failing tests `tests/test_protocol.py`**

```python
import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from sgs import config, protocol


def test_estimators_are_single_threaded_and_seeded():
    rf = protocol.estimator("rf")
    assert rf.n_jobs == 1 and rf.random_state == config.SEEDS["protocol_estimator"]
    assert protocol.estimator("gbm").random_state == config.SEEDS["protocol_estimator"]


def test_search_is_parallel_and_uses_the_named_split():
    search = protocol.searcher("rf", "log-rmse")
    assert isinstance(search, GridSearchCV) and search.n_jobs == -1
    assert search.cv.random_state == config.SEEDS["protocol_cv_split"]
    assert search.cv.n_splits == config.CV_FOLDS
    assert search.scoring == "neg_root_mean_squared_error"


def test_fos_scale_scorer():
    assert protocol._fos_rmse(np.log([1.0, 2.0]), np.log([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))
    assert protocol.scorer("fos-rmse") is not None
    with pytest.raises(ValueError):
        protocol.scorer("r2")


def test_unknown_search_kind_is_refused(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_KIND", "randomized")
    with pytest.raises(ValueError):
        protocol.searcher("rf", "log-rmse")


def test_fit_arm_artifact(tiny_protocol):
    art = protocol.fit_arm("rf", "C", "log-rmse")
    assert (art.family, art.arm, art.scoring, art.target) == ("rf", "C", "log-rmse", "log")
    assert len(art.train_ids) == 120 and art.train_ids[0] == "seed:0" and art.train_ids[-1] == "guided:59"
    assert set(art.fold) == {0, 1, 2}
    assert set(art.best_params) == {"min_samples_leaf"}
    assert len(art.cv_results["params"]) == 2
    assert np.all(art.scores >= 0) and art.scores.shape == (120,)


def test_oof_residuals_come_from_the_search_object(tiny_protocol, monkeypatch):
    seen = []
    real = protocol.cross_val_predict

    def spy(estimator, *args, **kwargs):
        seen.append(estimator)
        return real(estimator, *args, **kwargs)

    monkeypatch.setattr(protocol, "cross_val_predict", spy)
    protocol.fit_arm("gbm", "A", "log-rmse")
    assert len(seen) == 1 and isinstance(seen[0], GridSearchCV)


def test_fit_arm_is_deterministic(tiny_protocol):
    a = protocol.fit_arm("gbm", "C0", "log-rmse")
    b = protocol.fit_arm("gbm", "C0", "log-rmse")
    np.testing.assert_array_equal(a.oof_pred, b.oof_pred)
    assert a.best_params == b.best_params
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_protocol.py -q`
Expected: `ImportError: cannot import name 'protocol'`.

- [ ] **Step 4: Write `sgs/protocol.py`**

```python
"""The tuning protocol: one fixed grid per family, identical for every arm.

Hyper-parameters are selected by K-fold CV on the arm's training data only;
no test set is touched. Out-of-fold residuals come from a nested CV:
``cross_val_predict`` receives the search object, so each OOF prediction is
made with hyper-parameters selected without its fold.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    if config.SEARCH_KIND != "grid":
        raise ValueError(f"search kind {config.SEARCH_KIND!r} is not implemented")
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
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sgs/protocol.py tests/test_protocol.py tests/conftest.py
git commit -m "Add the symmetric tuning protocol with nested OOF residuals

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 19: `fitting.py` — the fit stage

**Files:**
- Create: `sgs/fitting.py`, `tests/test_fitting.py`

**Interfaces:**
- Consumes: `protocol.fit_arm`, `checks.require/pairs_among`, `stage.produce`, `manifest.require_fresh`.
- Produces: `SECTIONS = ("domain", "protocol")`, `select(declared, only, what) -> tuple`, `units(opts) -> list[(family, arm, scoring)]`, `fit_inputs(arm) -> dict`, `stage(opts)`, `load_fit(family, arm, scoring) -> FitArtifact` (fails on stale).

- [ ] **Step 1: Write the failing tests `tests/test_fitting.py`**

```python
import pytest

from sgs import config, fitting, manifest
from sgs.stage import Options


def test_units_cover_the_declared_axes():
    assert len(fitting.units(Options())) == len(config.FAMILIES) * len(config.ARMS) * len(config.ACTIVE_SCORINGS)
    assert fitting.units(Options(only_family="gbm", only_arm="C0")) == [("gbm", "C0", "log-rmse")]


def test_filters_must_name_declared_values():
    with pytest.raises(ValueError, match="only-arm 'D'"):
        fitting.units(Options(only_arm="D"))
    with pytest.raises(ValueError, match="only-scoring 'fos-rmse'"):
        fitting.units(Options(only_scoring="fos-rmse"))   # declared, but not active


def test_stage_requires_the_receipt_then_checkpoints_each_unit(tiny_protocol, monkeypatch):
    required = []
    monkeypatch.setattr(fitting.checks, "require", lambda roles, pairs: required.append((roles, pairs)))
    fitting.stage(Options(only_family="rf", only_arm="C"))
    assert required == [(["guided", "seed"], [("guided", "seed")])]
    path = config.fit_path("rf", "C", "log-rmse")
    record = manifest.read_manifest(path)
    assert set(record["inputs"]) == {"final:seed", "final:guided"}
    assert record["extra"]["n_train"] == 120
    assert fitting.load_fit("rf", "C", "log-rmse").arm == "C"


def test_load_fit_fails_loudly_on_a_changed_grid(tiny_protocol, monkeypatch):
    monkeypatch.setattr(fitting.checks, "require", lambda *a: None)
    fitting.stage(Options(only_family="rf", only_arm="A"))
    monkeypatch.setitem(config.GRIDS, "rf", {"min_samples_leaf": [1, 2]})
    with pytest.raises(manifest.StaleArtifact, match="protocol"):
        fitting.load_fit("rf", "A", "log-rmse")
    with pytest.raises(manifest.StaleArtifact, match="--force"):
        fitting.stage(Options(only_family="rf", only_arm="A"))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_fitting.py -q`
Expected: `ImportError: cannot import name 'fitting'`.

- [ ] **Step 3: Write `sgs/fitting.py`**

```python
"""The fit stage: one tuned search plus nested OOF residuals per unit.

A unit is (family, arm, scoring) and is checkpointed as soon as it finishes;
re-running skips fresh units. Nothing here touches a test set, so evaluating
on any test set afterwards costs seconds.
"""

from __future__ import annotations

import joblib

from sgs import checks, config, manifest, protocol
from sgs.stage import Options, produce

SECTIONS = ("domain", "protocol")


def select(declared, only, what) -> tuple:
    declared = tuple(declared)
    if only is None:
        return declared
    if only not in declared:
        raise ValueError(f"--only-{what} {only!r} is not declared in config ({', '.join(declared)})")
    return (only,)


def units(opts: Options) -> list[tuple[str, str, str]]:
    families = select(config.FAMILIES, opts.only_family, "family")
    arms = select(config.ARMS, opts.only_arm, "arm")
    scorings = select(config.ACTIVE_SCORINGS, opts.only_scoring, "scoring")
    return [(f, a, s) for f in families for a in arms for s in scorings]


def fit_inputs(arm: str) -> dict:
    return {f"final:{role}": config.final_path(role) for role in config.ARMS[arm].roles}


def stage(opts: Options) -> None:
    selected = units(opts)
    roles = sorted({role for _, arm, _ in selected for role in config.ARMS[arm].roles})
    if not opts.dry_run:
        checks.require(roles, checks.pairs_among(roles))
    for index, (family, arm, scoring) in enumerate(selected, 1):
        print(f"[fit {index}/{len(selected)}] {family} {arm} {scoring}", flush=True)

        def build(tmp, family=family, arm=arm, scoring=scoring):
            artifact = protocol.fit_arm(family, arm, scoring)
            joblib.dump(artifact, tmp)
            print(f"  best {artifact.best_params}")
            return {"best_params": artifact.best_params, "n_train": int(len(artifact.train_ids))}

        produce(config.fit_path(family, arm, scoring), stage="fit", sections=SECTIONS,
                inputs=fit_inputs(arm), opts=opts, build=build)


def load_fit(family: str, arm: str, scoring: str) -> protocol.FitArtifact:
    path = config.fit_path(family, arm, scoring)
    manifest.require_fresh(path, sections=SECTIONS, inputs=fit_inputs(arm))
    return joblib.load(path)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/fitting.py tests/test_fitting.py
git commit -m "Add the checkpointed fit stage

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 20: Run the 48 searches; retire the environment probe

**EXPENSIVE: hours. Runs once; nothing downstream refits.**

- [ ] **Step 1: Dry run, then the real run**

```bash
.venv/bin/sgs fit --dry-run
.venv/bin/sgs fit 2>&1 | tee /tmp/fit.log
```
Expected dry run: 8 `would produce` lines, one per (family, arm). The real run prints `best {...}` for each unit. If it is interrupted, re-run the same command: units already finished print `up to date`.

- [ ] **Step 2: Descriptive comparison of the selected hyper-parameters with the Colab reference**

```bash
.venv/bin/python - <<'EOF'
import json, glob
for path in sorted(glob.glob("artifacts/fits/*.manifest.json")):
    record = json.load(open(path))
    print(f"{record['payload']:32} n={record['extra']['n_train']:4}  {record['extra']['best_params']}")
EOF
```
Compare with the table in `docs/environment-probe-2026-09.md`. Differences are **legitimate**, because the data were regenerated and the boundary rows repaired (spec §10, phase 6 corollary). From here on, the acceptance target is determinism of the local run, not agreement with Colab. Report the comparison to the author.

- [ ] **Step 3: Retire the probe (end of phase 6)**

```bash
git rm -q docs/environment-probe-2026-09.md tools/probe_environment.py
git add artifacts/fits/*.manifest.json
git commit -m "Fit all arms locally; retire the environment probe

The 48 searches ran in the reference environment. The probe record has served
its comparison: from here the acceptance target is determinism of the local
run, not agreement with the earlier Colab selections.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 7 — evaluate

### Task 21: Metric primitives — global, zones, minimax, conformal

**Files:**
- Create: `sgs/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `protocol.FitArtifact`, `data.TestSet`.
- Produces:
  - `Prediction(family, arm, scoring, test, ids, y_true, y_pred)` (frozen)
  - `predict(fit, test) -> Prediction` (exponentiates according to `fit.target`)
  - `rmse(y, p) -> float`, `global_metrics(y, p) -> dict` with keys `n, r2, rmse, mae, bias, max_abs_error`
  - `zones(edges) -> list[(label, lo, hi)]` (half-open `[lo, hi)`; labels `"FOS < 1"`, `"1–1.3"`, …, `"FOS ≥ 6"`)
  - `zone_metrics(y, p, edges) -> list[dict(zone, lo, hi, n, rmse, mae, bias)]` (`None` metrics for empty zones, `None` for infinite bounds)
  - `minimax(zone_rows) -> dict(zone, rmse)`
  - `conformal_quantile(scores, alpha) -> float` (order statistic)
  - `conformal(pred, scores, *, alpha, edges) -> dict` with keys `alpha, q_log, factor, n_calibration, coverage, mean_width, zones: [dict(zone, n, coverage, mean_width)], procedure`
  - `subset_mask(y, subset, selection_range) -> ndarray[bool]`, `SUBSETS = ("all", "selection-range")`

- [ ] **Step 1: Write the failing tests `tests/test_evaluate.py`**

```python
import math

import numpy as np
import pytest

from sgs import config, evaluate


def _pred(y_true, y_pred, arm="C", family="rf", test="test-natural", ids=None):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ids = np.array([f"{test}:{i}" for i in range(len(y_true))]) if ids is None else np.asarray(ids)
    return evaluate.Prediction(family=family, arm=arm, scoring="log-rmse", test=test,
                               ids=ids, y_true=y_true, y_pred=y_pred)


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
    rows = evaluate.zone_metrics(y, p, config.ZONE_EDGES)
    worst = evaluate.minimax(rows)
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
    from tests.helpers import stub_fit, stub_test
    test = stub_test(10)
    pred = evaluate.predict(stub_fit("rf", "C", 0.0), test)
    np.testing.assert_allclose(pred.y_pred, test.y)
    assert pred.test == "test-natural" and (pred.ids == test.ids).all()
```

- [ ] **Step 2: Append the stub helpers to `tests/helpers.py`**

```python
class OffsetModel:
    """log(FOS) of the formula, perturbed row-wise by a fixed pattern."""

    def __init__(self, scale):
        self.scale = scale

    def predict(self, X):
        return np.log(synthetic_fos(X)) + self.scale * np.sin(np.arange(len(X)) * 1.7)


ARM_SCALES = {"A": 0.30, "B": 0.20, "C": 0.10, "C0": 0.25}


def stub_fit(family, arm, scale, scoring="log-rmse", n_train=40):
    from sgs.protocol import FitArtifact
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
    from sgs.data import TestSet
    frame = data.add_features(synthetic_final(n, seed))
    return TestSet(name=name, X=frame[list(config.FEATURES)].reset_index(drop=True),
                   y=frame["FOS"].to_numpy(float), ids=np.array([f"{name}:{i}" for i in range(n)]))
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_evaluate.py -q`
Expected: `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 4: Write `sgs/evaluate.py` (primitives)**

```python
"""Metrics on a test set. Pure functions: everything arrives as an argument.

Only ``stage`` (at the bottom) reads files; it loads fits and one test set
and passes them in explicitly. Predictions carry their test name and row
identifiers, so errors from one test set can never be paired with
observations from another.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import stats

from sgs import checks, config, data, fitting
from sgs.protocol import FitArtifact
from sgs.stage import Options, produce

SUBSETS = ("all", "selection-range")
CONFORMAL_PROCEDURE = ("Cross-conformal-style: nested-CV out-of-fold residuals on log(FOS), "
                       "multiplicative intervals around the refit model. Coverage is reported "
                       "empirically; no formal guarantee is claimed.")


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
    raw = fit.model.predict(test.X[list(config.FEATURES)])
    if fit.target != "log":
        raise ValueError(f"unknown target convention {fit.target!r}")
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
```

The imports of `checks`, `fitting`, `json`, `Mapping`, `stats`, `Options` and `produce` are used by Tasks 22–23. Keep them.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sgs/evaluate.py tests/test_evaluate.py tests/helpers.py
git commit -m "Add metric primitives with an order-statistic conformal quantile

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 22: Paired comparisons — Wilcoxon on |e₁|−|e₂| and paired bootstrap

**Files:**
- Modify: `sgs/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces: `MisalignedPair(ValueError)`, `UnregisteredComparison(KeyError)`, `BOOTSTRAP_LIMITATION: str`, `registered(first, second, registry) -> Comparison`, `wilcoxon_abs(p1, p2, *, registry, mask=None) -> dict` with keys `first, second, n, median_d, p_d_lt_0, statistic, pvalue, alternative, hypothesised, reading`, and `paired_bootstrap(p1, p2, *, registry, n_boot, seed, ci, mask=None) -> dict` with keys `delta_rmse, ci, ci_level, p_delta_lt_0, n_boot, limitation`.

- [ ] **Step 1: Append failing tests to `tests/test_evaluate.py`**

```python
# --- paired comparisons ----------------------------------------------------------------

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


def test_mask_restricts_both_tests():
    c, b = _pair()
    mask = c.y_true <= 3.0
    assert evaluate.wilcoxon_abs(c, b, registry=REG, mask=mask)["n"] == int(mask.sum())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_evaluate.py -q`
Expected: new tests fail with `AttributeError: ... 'registered'`.

- [ ] **Step 3: Add to `sgs/evaluate.py` (after `subset_mask`)**

```python
BOOTSTRAP_LIMITATION = ("Paired bootstrap over test rows: covers test-sampling variability only, "
                        "not seed or selection variability.")


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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sgs/evaluate.py tests/test_evaluate.py
git commit -m "Add Wilcoxon on absolute-error differences and a paired bootstrap

Both refuse pairs that are not aligned on the same test rows and comparisons
absent from the registry.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 23: `evaluate()` and `sgs evaluate`

**Files:**
- Modify: `sgs/evaluate.py`, `tests/test_evaluate.py`, `tests/helpers.py`

**Interfaces:**
- Produces:
  - `EvalResult(test, scoring, n, fos_range, arms, comparisons, predictions)` with `to_json() -> dict`
  - `evaluate(fits: Mapping[str, FitArtifact], test: TestSet, *, alpha, zones, comparisons, selection_range, n_boot, bootstrap_seed, bootstrap_ci) -> EvalResult`
  - `stage(opts, role=None)` writing `outputs/metrics/{test}__{scoring}.json`
- Metrics JSON layout (read by `report.py`): `{"test", "scoring", "n", "fos_range", "arms": {family: {arm: {"global", "zones", "minimax", "conformal", "best_params", "n_train"}}}, "comparisons": [{"family", "subset", "first", "second", "wilcoxon", "bootstrap"}], "predictions": {"row_id", "y_true", "<family>__<arm>": [...]}, "notes": {...}}`

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
EVAL_KWARGS = dict(alpha=0.10, zones=config.ZONE_EDGES, comparisons=config.COMPARISONS,
                   selection_range=config.SELECTION_RANGE, n_boot=300, bootstrap_seed=1,
                   bootstrap_ci=0.95)


def write_stub_fits(scoring="log-rmse"):
    """Stub fit artifacts on disk, with manifests that load_fit accepts."""
    import joblib
    from sgs import fitting
    for family in config.FAMILIES:
        for arm, scale in ARM_SCALES.items():
            path = config.fit_path(family, arm, scoring)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(stub_fit(family, arm, scale, scoring), path)
            manifest.write_manifest(path, stage="fixture", sections=fitting.SECTIONS,
                                    inputs=fitting.fit_inputs(arm), wall_time_s=0.0)
```

- [ ] **Step 2: Append failing tests to `tests/test_evaluate.py`**

```python
# --- evaluate() and the stage ------------------------------------------------------------

import json

from sgs.stage import Options
from tests.helpers import EVAL_KWARGS, stub_fits, stub_test, synthetic_final, write_final, write_stub_fits


def test_evaluate_covers_every_arm_and_registered_comparison():
    fits = {**stub_fits("rf"), **stub_fits("gbm")}
    result = evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)
    assert set(result.arms) == {"rf", "gbm"}
    assert set(result.arms["rf"]) == {"A", "B", "C", "C0"}
    assert len(result.comparisons) == 2 * len(config.COMPARISONS) * len(evaluate.SUBSETS)
    first = result.comparisons[0]
    assert {first["first"], first["second"]} <= set(config.ARMS)
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


def test_stage_writes_metrics_with_provenance(root, monkeypatch):
    for i, role in enumerate(config.ROLE_ORDER):
        write_final(role, synthetic_final(60, 200 + i))
    write_stub_fits()
    monkeypatch.setattr(config, "N_BOOTSTRAP", 200)
    monkeypatch.setattr(evaluate.checks, "require", lambda *a, **k: None)
    evaluate.stage(Options(), role="test-natural")
    path = config.metrics_path("test-natural", "log-rmse")
    metrics = json.loads(path.read_text())
    assert metrics["test"] == "test-natural" and set(metrics["arms"]["gbm"]) == {"A", "B", "C", "C0"}
    record = json.loads(path.with_name(path.name + ".manifest.json").read_text())
    assert "fit:rf__C0" in record["inputs"] and "final:test-natural" in record["inputs"]
    assert set(record["config"]) == {"domain", "protocol", "evaluation"}


def test_stage_refuses_arm_or_family_filters():
    with pytest.raises(ValueError, match="C0 included"):
        evaluate.stage(Options(only_arm="C"))
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_evaluate.py -q`
Expected: new tests fail with `AttributeError: ... 'evaluate'`.

- [ ] **Step 4: Append to `sgs/evaluate.py`**

```python
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


EVAL_SECTIONS = ("domain", "protocol", "evaluation")


def stage(opts: Options, role: str | None = None) -> None:
    if opts.only_family or opts.only_arm:
        raise ValueError("evaluate always covers every family and every arm, C0 included")
    test_roles = (role,) if role else config.TEST_ROLES
    scorings = fitting.select(config.ACTIVE_SCORINGS, opts.only_scoring, "scoring")
    training = sorted({r for arm in config.ARMS.values() for r in arm.roles})
    for scoring in scorings:
        for test_role in test_roles:
            inputs = {f"fit:{f}__{a}": config.fit_path(f, a, scoring)
                      for f in config.FAMILIES for a in config.ARMS}
            inputs[f"final:{test_role}"] = config.final_path(test_role)

            def build(tmp, scoring=scoring, test_role=test_role):
                checks.require([test_role, *training], [(test_role, r) for r in training])
                fits = {f"{f}__{a}__{scoring}": fitting.load_fit(f, a, scoring)
                        for f in config.FAMILIES for a in config.ARMS}
                result = evaluate(fits, data.load_test(test_role), alpha=config.ALPHA,
                                  zones=config.ZONE_EDGES, comparisons=config.COMPARISONS,
                                  selection_range=config.SELECTION_RANGE,
                                  n_boot=config.N_BOOTSTRAP,
                                  bootstrap_seed=config.SEEDS["bootstrap_resampling"],
                                  bootstrap_ci=config.BOOTSTRAP_CI)
                tmp.write_text(json.dumps(result.to_json(), indent=2) + "\n")
                return {"n_test": result.n, "fos_range": list(result.fos_range)}

            produce(config.metrics_path(test_role, scoring), stage=f"evaluate:{test_role}",
                    sections=EVAL_SECTIONS, inputs=inputs, opts=opts, build=build)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Run on the real fits**

```bash
.venv/bin/sgs evaluate
```
Expected: `wrote test-natural__log-rmse.json` and `wrote test-stratified__log-rmse.json`, in seconds.

- [ ] **Step 7: Commit**

```bash
git add sgs/evaluate.py tests/test_evaluate.py tests/helpers.py outputs/metrics/
git commit -m "Add the evaluate stage: pure metrics from fits and one test set

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 8 — learning curve

### Task 24: `learning_curve.py`, `sgs learning-curve`, and its metrics

**Files:**
- Create: `sgs/learning_curve.py`, `tests/test_learning_curve.py`
- Modify: `sgs/evaluate.py` (add `learning_curve_metrics` and call it from `stage`)

**Interfaces:**
- Consumes: `protocol.searcher`, `data.load_role/load_test`, `checks.require/pairs_among`, `fitting.select`.
- Produces:
  - `SECTIONS = ("domain", "protocol", "learning_curve")`, `SHARED = "seed-only"`
  - `unit_rng(budget, repeat) -> Generator`, `subsample(seed_df, extra_df, budget, rng) -> DataFrame`
  - `plan(n_seed) -> list[(key, budget, repeats)]`
  - `run_budget(family, key, budget, repeats, scoring, seed_df, extras, tests) -> dict` (`{"family", "key", "budget", "scoring", "units": [{"repeat", "n_train", "train_ids", "best_params", "predictions": {test: {"ids", "y_pred"}}}]}`)
  - `artifact_paths(scoring, families=None) -> list[Path]`, `available(scoring) -> bool`
  - `load_units(scoring, families=None) -> dict[(family, curve, budget), list[unit]]` (shared units are listed under every curve)
  - `stage(opts)`
  - In `evaluate.py`: `learning_curve_metrics(units, tests, *, selection_range) -> {"rows": [...]}` with row keys `family, curve, budget, test, n_repeats, rmse_mean, rmse_std, rmse_selection_mean, rmse_selection_std`

- [ ] **Step 1: Write the failing tests `tests/test_learning_curve.py`**

```python
import numpy as np
import pandas as pd
import pytest

from sgs import config, evaluate, learning_curve as lc
from sgs.stage import Options
from tests.helpers import synthetic_final, write_final


@pytest.fixture
def tiny_curve(tiny_protocol, monkeypatch):
    write_final("seed", synthetic_final(40, 10))          # overrides the 60-row seed
    for i, role in enumerate(config.TEST_ROLES):
        write_final(role, synthetic_final(30, 50 + i))
    monkeypatch.setattr(config, "LC_BUDGETS", (20, 40, 60))
    monkeypatch.setattr(config, "LC_REPEATS", 2)
    monkeypatch.setattr(lc.checks, "require", lambda *a, **k: None)


def test_plan_shares_budgets_up_to_the_seed_size():
    assert lc.plan(300) == [("seed-only", 100, 10), ("seed-only", 200, 10), ("seed-only", 300, 1),
                            ("lhs", 400, 10), ("guided", 400, 10), ("lhs", 500, 10),
                            ("guided", 500, 10), ("lhs", 600, 10), ("guided", 600, 10)]


def test_subsample_semantics():
    seed_df = pd.DataFrame({"row_id": [f"s{i}" for i in range(10)]})
    extra = pd.DataFrame({"row_id": [f"e{i}" for i in range(10)]})
    small = lc.subsample(seed_df, extra, 4, lc.unit_rng(4, 0))
    assert len(small) == 4 and small["row_id"].str.startswith("s").all()
    assert lc.subsample(seed_df, extra, 10, lc.unit_rng(10, 0)) is seed_df
    large = lc.subsample(seed_df, extra, 13, lc.unit_rng(13, 0))
    assert len(large) == 13 and set(seed_df["row_id"]) <= set(large["row_id"])


def test_unit_rng_is_keyed_on_budget_and_repeat():
    draw = lambda b, r: lc.unit_rng(b, r).integers(0, 10**9)
    assert draw(100, 0) == draw(100, 0)
    assert len({draw(100, 0), draw(100, 1), draw(200, 0)}) == 3


def test_stage_writes_one_checkpoint_per_budget_and_metrics_follow(tiny_curve):
    lc.stage(Options(only_family="rf"))
    paths = lc.artifact_paths("log-rmse", families=("rf",))
    assert [p.name for p in paths] == ["rf__seed-only__log-rmse__n20.joblib",
                                       "rf__seed-only__log-rmse__n40.joblib",
                                       "rf__lhs__log-rmse__n60.joblib",
                                       "rf__guided__log-rmse__n60.joblib"]
    units = lc.load_units("log-rmse", families=("rf",))
    assert units[("rf", "lhs", 20)] is units[("rf", "guided", 20)]
    assert len(units[("rf", "lhs", 40)]) == 1 and len(units[("rf", "lhs", 60)]) == 2

    from sgs import data
    tests = {r: data.load_test(r) for r in config.TEST_ROLES}
    rows = evaluate.learning_curve_metrics(units, tests, selection_range=config.SELECTION_RANGE)["rows"]
    shared = [r for r in rows if r["budget"] == 20 and r["test"] == "test-natural"]
    assert len(shared) == 2 and shared[0]["rmse_mean"] == shared[1]["rmse_mean"]
    assert all(r["rmse_std"] == 0.0 for r in rows if r["budget"] == 40)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_learning_curve.py -q`
Expected: `ImportError: cannot import name 'learning_curve'`.

- [ ] **Step 3: Write `sgs/learning_curve.py`**

```python
"""The learning curve: the only analysis that refits in a loop.

For each budget, the training set is the seed (subsampled when the budget is
smaller) plus a random subsample of one strategy's additional points. Every
subsample is refit with the same tuned ``searcher`` as the arms. Subsample
generators are keyed on (budget, repeat), so repeats at different budgets never
share draws, and both strategies see the same draw at the same key. Budgets up
to the seed size do not depend on the strategy and are fitted once.

Predictions on every test set are stored with each unit; ``evaluate`` turns
them into RMSE. The cache lives apart from artifacts/fits/.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sgs import checks, config, data, fitting, manifest, protocol
from sgs.stage import Options, produce

SECTIONS = ("domain", "protocol", "learning_curve")
SHARED = "seed-only"


def unit_rng(budget: int, repeat: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(config.SEEDS["learning_curve_subsampling"],
                                      spawn_key=(budget, repeat))
    return np.random.default_rng(sequence)


def subsample(seed_df: pd.DataFrame, extra_df: pd.DataFrame, budget: int, rng) -> pd.DataFrame:
    n_seed = len(seed_df)
    if budget < n_seed:
        return seed_df.iloc[rng.choice(n_seed, budget, replace=False)]
    if budget == n_seed:
        return seed_df
    take = extra_df.iloc[rng.choice(len(extra_df), budget - n_seed, replace=False)]
    return pd.concat([seed_df, take], ignore_index=True)


def plan(n_seed: int) -> list[tuple[str, int, int]]:
    out = []
    for budget in config.LC_BUDGETS:
        repeats = 1 if budget == n_seed else config.LC_REPEATS
        if budget <= n_seed:
            out.append((SHARED, budget, repeats))
        else:
            out.extend((strategy, budget, repeats) for strategy in config.LC_STRATEGIES)
    return out


def _roles() -> list[str]:
    return [config.LC_BASE_ROLE, *config.LC_STRATEGIES.values(), *config.TEST_ROLES]


def _inputs() -> dict:
    return {f"final:{role}": config.final_path(role) for role in _roles()}


def run_budget(family, key, budget, repeats, scoring, seed_df, extras, tests) -> dict:
    units = []
    for repeat in range(repeats):
        extra = seed_df.iloc[:0] if key == SHARED else extras[key]
        frame = subsample(seed_df, extra, budget, unit_rng(budget, repeat))
        X = frame[list(config.FEATURES)]
        y = np.log(frame["FOS"].to_numpy(float))
        search = protocol.searcher(family, scoring).fit(X, y)
        units.append({
            "repeat": repeat, "n_train": int(len(frame)),
            "train_ids": frame["row_id"].to_numpy(str),
            "best_params": dict(search.best_params_),
            "predictions": {name: {"ids": test.ids, "y_pred": np.exp(search.best_estimator_.predict(test.X))}
                            for name, test in tests.items()},
        })
    return {"family": family, "key": key, "budget": budget, "scoring": scoring, "units": units}


def artifact_paths(scoring: str, families=None) -> list[Path]:
    n_seed = len(data.load_role(config.LC_BASE_ROLE))
    return [config.learning_curve_path(family, key, scoring, budget)
            for family in (families or config.FAMILIES)
            for key, budget, _ in plan(n_seed)]


def available(scoring: str) -> bool:
    return all(path.exists() for path in artifact_paths(scoring))


def load_units(scoring: str, families=None) -> dict:
    n_seed = len(data.load_role(config.LC_BASE_ROLE))
    out = {}
    for family in (families or config.FAMILIES):
        for key, budget, _ in plan(n_seed):
            path = config.learning_curve_path(family, key, scoring, budget)
            manifest.require_fresh(path, sections=SECTIONS, inputs=_inputs())
            units = joblib.load(path)["units"]
            for curve in (tuple(config.LC_STRATEGIES) if key == SHARED else (key,)):
                out[(family, curve, budget)] = units
    return out


def stage(opts: Options) -> None:
    families = fitting.select(config.FAMILIES, opts.only_family, "family")
    scorings = fitting.select(config.ACTIVE_SCORINGS, opts.only_scoring, "scoring")
    missing = [name for name, path in _inputs().items() if not path.exists()]
    if missing:
        if opts.dry_run:
            print(f"[learning-curve] dry run: waits for {', '.join(missing)}")
            return
        raise FileNotFoundError(f"learning curve needs {', '.join(missing)}")
    roles = _roles()
    if not opts.dry_run:
        checks.require(roles, checks.pairs_among(roles))
    seed_df = data.load_role(config.LC_BASE_ROLE)
    extras = {strategy: data.load_role(role) for strategy, role in config.LC_STRATEGIES.items()}
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    for scoring in scorings:
        for family in families:
            for key, budget, repeats in plan(len(seed_df)):
                def build(tmp, family=family, key=key, budget=budget, repeats=repeats, scoring=scoring):
                    result = run_budget(family, key, budget, repeats, scoring, seed_df, extras, tests)
                    joblib.dump(result, tmp)
                    return {"repeats": repeats, "budget": budget}

                produce(config.learning_curve_path(family, key, scoring, budget),
                        stage=f"learning-curve:{family}:{key}:n{budget}", sections=SECTIONS,
                        inputs=_inputs(), opts=opts, build=build)
```

- [ ] **Step 4: Add the learning-curve metrics to `sgs/evaluate.py`**

Add `from sgs import learning_curve` to the imports. Insert the function below after `evaluate()`:

```python
def learning_curve_metrics(units: Mapping, tests: Mapping[str, data.TestSet], *, selection_range) -> dict:
    """Mean and spread of RMSE over repeats, per (family, curve, budget, test)."""
    rows = []
    for (family, curve, budget), unit_list in sorted(units.items()):
        for name, test in tests.items():
            mask = subset_mask(test.y, "selection-range", selection_range)
            overall, inside = [], []
            for unit in unit_list:
                pred = unit["predictions"][name]
                if not np.array_equal(pred["ids"], test.ids):
                    raise MisalignedPair(f"learning-curve predictions for {name} are not aligned with the test set")
                overall.append(rmse(test.y, pred["y_pred"]))
                inside.append(rmse(test.y[mask], pred["y_pred"][mask]))
            rows.append({"family": family, "curve": curve, "budget": int(budget), "test": name,
                         "n_repeats": len(unit_list),
                         "rmse_mean": float(np.mean(overall)), "rmse_std": float(np.std(overall)),
                         "rmse_selection_mean": float(np.mean(inside)),
                         "rmse_selection_std": float(np.std(inside))})
    return {"rows": rows}
```

At the end of `stage`, inside `for scoring in scorings:` and after the inner test-role loop, append:

```python
        if role is None and learning_curve.available(scoring):
            lc_inputs = {f"lc:{p.stem}": p for p in learning_curve.artifact_paths(scoring)}
            lc_inputs.update({f"final:{r}": config.final_path(r) for r in config.TEST_ROLES})

            def build_lc(tmp, scoring=scoring):
                tests = {r: data.load_test(r) for r in config.TEST_ROLES}
                result = learning_curve_metrics(learning_curve.load_units(scoring), tests,
                                                selection_range=config.SELECTION_RANGE)
                tmp.write_text(json.dumps(result, indent=2) + "\n")
                return {"n_rows": len(result["rows"])}

            produce(config.learning_curve_metrics_path(scoring), stage="evaluate:learning-curve",
                    sections=("domain", "evaluation", "learning_curve"), inputs=lc_inputs,
                    opts=opts, build=build_lc)
        elif role is None:
            print(f"[evaluate] learning curve for {scoring} not available yet; run sgs learning-curve")
```

`learning_curve.available()` calls `data.load_role("seed")`. The evaluate stage test in Task 23 writes a seed final, so that test keeps passing.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Run the learning curve (EXPENSIVE: hours, comparable to Task 20), then its metrics**

```bash
.venv/bin/sgs learning-curve --dry-run
.venv/bin/sgs learning-curve 2>&1 | tee /tmp/lc.log
.venv/bin/sgs evaluate
```
Expected: 18 checkpoints for the default config: 3 shared plus 6 strategy budgets, for 2 families. Then `wrote learning_curve__log-rmse.json`. The two test-metric files report `up to date`.

- [ ] **Step 7: Commit**

```bash
git add sgs/learning_curve.py sgs/evaluate.py tests/test_learning_curve.py artifacts/learning_curve/*.manifest.json outputs/metrics/
git commit -m "Add the tuned learning curve with its own cache and metrics

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 9 — report and notebook

### Task 25: `report.py` — booktabs tables and figures

**Files:**
- Create: `sgs/report.py`, `tests/test_report.py`
- Modify: `tests/conftest.py` (select the Agg backend)

**Interfaces:**
- Consumes: only `outputs/metrics/*.json` (via `load_metrics`, `load_learning_curve`) and `config` (arm order, ablation flags, budgets).
- Produces: `arm_label(arm)`, `load_metrics(test, scoring)`, `load_learning_curve(scoring)`, `booktabs(frame, *, caption, label, digits=3) -> str`, `global_frame(metrics)`, `zone_frame(metrics, family)`, `comparison_frame(metrics)`, `conformal_frame(metrics)`, `budget_frame()`, `fig_zone_rmse(metrics, family) -> Figure`, `fig_learning_curve(lc, test, family) -> Figure`, `stage(opts)`.

- [ ] **Step 1: Make matplotlib headless in tests — prepend to `tests/conftest.py`**

```python
import matplotlib

matplotlib.use("Agg")
```

- [ ] **Step 2: Write the failing tests `tests/test_report.py`**

```python
import json

from sgs import config, evaluate, manifest, report
from sgs.stage import Options
from tests.helpers import EVAL_KWARGS, stub_fits, stub_test


def _write_metrics():
    result = evaluate.evaluate({**stub_fits("rf"), **stub_fits("gbm")}, stub_test(), **EVAL_KWARGS)
    path = config.metrics_path("test-natural", "log-rmse")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json()))
    manifest.write_manifest(path, stage="fixture", sections=(), inputs={}, wall_time_s=0)
    return result.to_json()


def test_booktabs_structure_and_escaping():
    import pandas as pd
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


def test_stage_writes_tables_and_figures_with_manifests(root):
    _write_metrics()
    report.stage(Options())
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    assert {"budget.tex", "test-natural__log-rmse__global.tex", "test-natural__log-rmse__comparisons.tex",
            "test-natural__log-rmse__conformal.tex", "test-natural__log-rmse__zones_rf.tex"} <= tables
    figure = config.figures_dir() / "test-natural__log-rmse__zones_gbm.pdf"
    assert figure.read_bytes()[:4] == b"%PDF"
    assert manifest.read_manifest(figure)["inputs"]["metrics"]["path"].endswith("test-natural__log-rmse.json")
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

from sgs import config, manifest
from sgs.stage import Options, produce


def arm_label(arm: str) -> str:
    return f"{arm} (ablation)" if config.ARMS[arm].ablation else arm


def load_metrics(test: str, scoring: str) -> dict:
    path = config.metrics_path(test, scoring)
    manifest.verify_provenance(path)
    return json.loads(path.read_text())


def load_learning_curve(scoring: str) -> dict:
    path = config.learning_curve_metrics_path(scoring)
    manifest.verify_provenance(path)
    return json.loads(path.read_text())


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


def fig_learning_curve(lc: dict, test: str, family: str):
    rows = [r for r in lc["rows"] if r["test"] == test and r["family"] == family]
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
    axes[0].set_ylabel(f"RMSE on {test}")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


SECTIONS = ("protocol", "evaluation")


def stage(opts: Options) -> None:
    tables, figures = config.tables_dir(), config.figures_dir()

    def table(path, make, inputs):
        def build(tmp):
            tmp.write_text(make())
        produce(path, stage="report", sections=SECTIONS, inputs=inputs, opts=opts, build=build)

    def figure(path, make, inputs):
        def build(tmp):
            fig = make()
            fig.savefig(tmp, format="pdf")
            plt.close(fig)
        produce(path, stage="report", sections=SECTIONS, inputs=inputs, opts=opts, build=build)

    table(tables / "budget.tex", lambda: booktabs(
        budget_frame(), caption="Budget-fair arms. C0 is an ablation and carries no budget.",
        label="tab:budget"), {})

    for scoring in config.ACTIVE_SCORINGS:
        for test in config.TEST_ROLES:
            path = config.metrics_path(test, scoring)
            if not path.exists():
                print(f"[report] {path.name} missing; run sgs evaluate")
                continue
            metrics = load_metrics(test, scoring)
            stem, inputs = f"{test}__{scoring}", {"metrics": path}
            table(tables / f"{stem}__global.tex", lambda m=metrics, s=stem: booktabs(
                global_frame(m), caption=f"Global metrics on {test}.", label=f"tab:{s}:global"), inputs)
            table(tables / f"{stem}__comparisons.tex", lambda m=metrics, s=stem: booktabs(
                comparison_frame(m), caption=f"Paired comparisons on {test}: Wilcoxon on "
                "d = |e1| - |e2| (smaller absolute errors) and bootstrap ΔRMSE.",
                label=f"tab:{s}:comparisons"), inputs)
            table(tables / f"{stem}__conformal.tex", lambda m=metrics, s=stem: booktabs(
                conformal_frame(m), caption=f"Cross-conformal-style intervals on {test}, "
                "empirical coverage.", label=f"tab:{s}:conformal"), inputs)
            for family in metrics["arms"]:
                table(tables / f"{stem}__zones_{family}.tex", lambda m=metrics, f=family, s=stem: booktabs(
                    zone_frame(m, f), caption=f"Zone RMSE on {test}, {f}.", label=f"tab:{s}:zones:{f}"), inputs)
                figure(figures / f"{stem}__zones_{family}.pdf",
                       lambda m=metrics, f=family: fig_zone_rmse(m, f), inputs)

        lc_path = config.learning_curve_metrics_path(scoring)
        if lc_path.exists():
            lc = load_learning_curve(scoring)
            for test in config.TEST_ROLES:
                for family in config.FAMILIES:
                    figure(figures / f"learning_curve__{test}__{family}__{scoring}.pdf",
                           lambda t=test, f=family: fig_learning_curve(lc, t, f), {"metrics": lc_path})
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Generate the real report**

```bash
.venv/bin/sgs report
ls outputs/tables outputs/figures
```
Expected: `budget.tex`; four tables plus two zone tables per test set; zone figures; and learning-curve figures for 2 tests × 2 families.

- [ ] **Step 7: Commit**

```bash
git add sgs/report.py tests/test_report.py tests/conftest.py outputs/
git commit -m "Add booktabs tables and figures read from outputs/metrics

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 26: Narrative notebook, notebook purity test, final verification

**Files:**
- Create: `notebooks/surrogate_guided_sampling.ipynb`, `tests/test_notebook.py`
- Modify: `README.md` (add a notebook section)

**Interfaces:**
- Consumes: `sgs.config`, `sgs.report`, `sgs.manifest` only.

- [ ] **Step 1: Write the failing purity test `tests/test_notebook.py`**

```python
"""The notebook holds narrative and figures only: no fitting, no metrics."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
FORBIDDEN_CALLS = {"fit", "fit_transform", "fit_predict", "partial_fit", "cross_val_predict", "cross_val_score"}
COMPUTING_MODULES = {"protocol", "fitting", "learning_curve", "evaluate", "screener", "design", "solve",
                     "solver", "repair", "checks"}


def _code(path):
    cells = json.loads(path.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            lines = "".join(cell["source"]).splitlines()
            yield "\n".join(l for l in lines if not l.lstrip().startswith(("!", "%")))


def test_the_narrative_notebook_exists():
    assert [p.name for p in NOTEBOOKS] == ["surrogate_guided_sampling.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_pure(path):
    for source in _code(path):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("sklearn"), alias.name
                    assert alias.name.removeprefix("sgs.") not in COMPUTING_MODULES, alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("sklearn"), module
                if module == "sgs":
                    assert not {a.name for a in node.names} & COMPUTING_MODULES, module
                assert module.removeprefix("sgs.") not in COMPUTING_MODULES, module
            elif isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in FORBIDDEN_CALLS, name
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_notebook.py -q`
Expected: `test_the_narrative_notebook_exists` FAILS (no notebook).

- [ ] **Step 3: Check the Zenodo "latest version" endpoint before writing it into the notebook**

```bash
curl -sL https://zenodo.org/api/records/20706202/versions/latest | .venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print(r['id'], [f['key'] for f in r['files']])"
```
Expected: a record id and the file list of the newest version. If the call fails or returns something other than the newest version, ask the author for the correct record URL and use it for `ZENODO_LATEST` below. Until the corrected version is published, the listed files still carry the old names. The download loop then finds nothing, and the metrics in `outputs/` still render.

- [ ] **Step 4: Generate the notebook** (run once; the script itself is not committed)

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
To regenerate the numbers, run the pipeline (`sgs run-all`, see README).
"""),
    code("""
# On Colab: fetch the repository (code, manifests, metrics) and the final data sets.
import json, os, sys, subprocess, urllib.request

ZENODO_LATEST = "https://zenodo.org/api/records/20706202/versions/latest"

if "google.colab" in sys.modules and not os.path.exists("sgs"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/bteodoru/surrogate-guided-slope-stability.git"], check=True)
    os.chdir("surrogate-guided-slope-stability")
    sys.path.insert(0, os.getcwd())
    from sgs import config, manifest
    record = json.load(urllib.request.urlopen(ZENODO_LATEST))
    for entry in record["files"]:
        for role in config.ROLES:
            target = config.final_path(role)
            if entry["key"] == target.name:
                target.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(entry["links"]["self"], target)
                manifest.verify_provenance(target)   # the tracked manifest vouches for the download
elif os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.dirname(os.getcwd()))
"""),
    code("""
from IPython.display import display
from sgs import config, report

SCORING = config.ACTIVE_SCORINGS[0]
"""),
    md("""
## Budget-fair arms

A, B and C share the same solver budget logic; C0 (guided points only) is an ablation
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
    md("## Learning curve"),
    code("""
lc = report.load_learning_curve(SCORING)
for test in config.TEST_ROLES:
    for family in config.FAMILIES:
        report.fig_learning_curve(lc, test, family)
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

- [ ] **Step 5: Run the purity and discipline tests**

Run: `.venv/bin/pytest tests/test_notebook.py tests/test_discipline.py -q`
Expected: all pass. The discipline test now also scans the notebook.

- [ ] **Step 6: Append a notebook section to `README.md`**

```markdown
## Notebook

`notebooks/surrogate_guided_sampling.ipynb` holds the narrative and figures. It imports
`sgs.report` and reads `outputs/metrics/`; it fits nothing and computes no metric
(`tests/test_notebook.py` enforces this). On Colab its first cell clones the repository
and downloads the final data sets from Zenodo, checking each file against its tracked
manifest.
```

- [ ] **Step 7: Full verification**

```bash
.venv/bin/pytest -q
.venv/bin/pytest -q -m slow
.venv/bin/sgs run-all --dry-run
git status --short
```
Expected: all tests pass. The slow solver test passes. `run-all --dry-run` reports `up to date` for every stage, except `check`, which always re-evaluates and so prints its dry-run line. `git status` shows only the new notebook, test and README changes, and no CSV or joblib files.

- [ ] **Step 8: Commit**

```bash
git add notebooks/surrogate_guided_sampling.ipynb tests/test_notebook.py README.md
git commit -m "Add the narrative notebook and enforce that it computes nothing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Finish the branch**

Use superpowers:finishing-a-development-branch to decide how `pipeline-reorganization` is integrated into `main`.

---

## Open items for the author (not tasks)

- `N_GUIDED_POOL`, `N_TEST_POOL` and `MIN_DISTANCE` are declared study parameters. Each needs a sentence in Methods; the pool size also needs a note on how much the result depends on it.
- The selection criterion (RMSE on log FOS) and the cross-conformal-style procedure need to be stated in Methods as declared choices.
- Every published figure is regenerated from the new data. Compare the outputs of `tools/describe_sets.py`, Task 20 and Task 23 against the numbers in circulation, and retire the old ones.
- The permutation-importance section of the old notebook is not ported: the spec lists no feature-importance analysis. If it returns, it belongs in `evaluate.py` (computed from fits and a test set), not in the notebook.
- Where `artifacts/fits/` lives at publication is still undecided. Nothing blocks on it.
- The Zenodo release of the regenerated data (new version under the concept DOI, with the name mapping from `data/README.md`) follows this plan.
