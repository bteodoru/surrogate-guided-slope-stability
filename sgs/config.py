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
    # Circular-search window, as fractions of L measured from the toe:
    #   exit range  = (-depth, exit_range_l_frac * L)             toe side
    #   entry range = (entry_range_l_frac * L, L + extension)     crest side
    exit_range_l_frac: float = 1 / 3
    entry_range_l_frac: float = 2 / 3
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

# Seed-aware guided variant (role "guided-log", declared 2026-09-19 before any
# solver run). The aim is a training set, seed plus guided, that is even in
# FOS, not a guided set that is even on its own. The role's budget is
# water-filled against the seed's solved FOS in N_BINS geometric bins on
# SELECTION_RANGE (log(FOS) is the trained response). Inside a bin the points
# are drawn at random, not by max-min: max-min picks the corners of the domain
# (54% of the linear guided set has >= 2 parameters within 5% of a bound,
# against 9% for LHS), while a random draw keeps the natural distribution of
# the parameters given FOS.

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
    # The seed-aware log(FOS) guided variant: pool first, then the bin
    # selection, on one generator.
    "guided_log_stream": 20260923,
}


@dataclass(frozen=True)
class SetRole:
    n: int
    design: str                           # "lhs" | "guided-maxmin" | "stratified-random" | "guided-log-deficit"
    seeds: tuple[str, ...]                # keys of SEEDS, in the order the procedure uses them
    stem: str                             # file name without extension
    screener: bool = False                # built from screener predictions
    disjoint_from: tuple[str, ...] = ()   # roles whose designs this one must avoid
    # Solved roles behind the screener and the water-filling deficit; empty
    # means SCREENER["trained_on"]. A sequential batch lists every earlier set.
    trained_on: tuple[str, ...] = ()


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
    # Declared after both test sets so that it avoids them; every set used
    # together must be pairwise disjoint (data.validate_sets).
    "guided-log": SetRole(n=300, design="guided-log-deficit", seeds=("guided_log_stream",),
                          stem="surrogate_guided_log_300", screener=True,
                          disjoint_from=("seed", "lhs-extension", "guided", "test-natural",
                                         "test-stratified")),
}

# Generation order: each role follows the roles it depends on.
ROLE_ORDER = ("seed", "lhs-extension", "guided", "test-natural", "test-stratified", "guided-log")

# The preliminary surrogate. Fixed hyper-parameters, not a tuned search: it
# screens the candidate pools and is not one of the models under comparison.
SCREENER = {"trained_on": "seed", "n_estimators": 500, "cv_folds": 5}


def screener_roles(role: str) -> tuple[str, ...]:
    """The solved roles the screener of ``role`` is trained on, and whose FOS
    the guided-log deficit is computed against."""
    return ROLES[role].trained_on or (SCREENER["trained_on"],)


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
        "max_leaf_nodes": [2, 3, 4, 7],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_iter": [2000, 4000, 8000],
    },
}

CV_FOLDS = 5

# Selection criterion. "log-rmse" is RMSE on the trained (log) target, the
# declared choice; "fos-rmse" is the sensitivity check. Each scoring has its
# own fit files.
SCORINGS = ("log-rmse", "fos-rmse")
ACTIVE_SCORINGS = ("log-rmse", "fos-rmse")


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
    "C-log": Arm(roles=("seed", "guided-log"), budget=600, ablation=False),
    "C0-log": Arm(roles=("guided-log",), budget=None, ablation=True),
}


# --- evaluation -------------------------------------------------------------------

TEST_ROLES = ("test-natural", "test-stratified")

# Reporting zones, half-open [lo, hi): FOS < 0.5 | 0.5–1 | 1–2 | 2–6 | FOS ≥ 6.
# 0.5 and 6 are the ends of the selection range, 1 is limiting equilibrium and
# 2 mirrors 0.5 around 1 in log space. Below 1 the slope is unstable, 1–2 is
# where design decisions are taken, above 2 the reserve is comfortable. No
# normative threshold (1.3, 1.5) is an edge. Declared 2026-09-19, replacing
# (1.0, 1.3, 3.0, 6.0); a reporting change, no design or fit depends on it.
# Minimax is the RMSE of the worst zone.
ZONE_EDGES = (0.5, 1.0, 2.0, 6.0)

# Primary criterion: log-RMSE on the decision range [0.5, 2), the two zones
# around equilibrium. Declared 2026-09-19, after the C-log results and before
# C-seq was run. It is built only from the lower end of the
# selection range and its mirror around 1 in log space; its errors on the
# existing arms were not computed before this declaration.
DECISION_RANGE = (0.5, 2.0)

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
# and C-log against B and A, not for the comparisons involving an ablation or
# between the two binning rules. The C-log entries were declared on
# 2026-09-19, before the guided-log set was solved.
COMPARISONS = (
    Comparison("C", "B", alternative="less", hypothesised=True),
    Comparison("C", "A", alternative="less", hypothesised=True),
    Comparison("C", "C0", alternative="two-sided", hypothesised=False),
    Comparison("C0", "B", alternative="two-sided", hypothesised=False),
    Comparison("C-log", "B", alternative="less", hypothesised=True),
    Comparison("C-log", "A", alternative="less", hypothesised=True),
    Comparison("C-log", "C", alternative="two-sided", hypothesised=False),
    Comparison("C-log", "C0-log", alternative="two-sided", hypothesised=False),
    Comparison("C0-log", "B", alternative="two-sided", hypothesised=False),
)


# --- learning curve (optional; fixed hyper-parameters, both families) ---------------

LC_BASE_ROLE = "seed"
LC_STRATEGIES = {"lhs": "lhs-extension", "guided": "guided"}
LC_BUDGETS = (100, 200, 300, 400, 500, 600)
LC_REPEATS = 10
LC_FAMILIES = ("rf", "gbm")

# Fixed hyper-parameters of the learning curve, on top of ESTIMATOR_FIXED; not
# tuned per point. RF: scikit-learn defaults, which are also what the grid
# selects on every arm. GBM: the grid point selected by CV on the seed alone
# (arm A). The seed is shared by both curves, so the choice favours neither
# strategy; GBM library defaults (max_iter=100, min_samples_leaf=20) are not
# neutral at small n.
LC_PARAMS = {
    "rf": {},
    "gbm": {"learning_rate": 0.06, "max_iter": 8000, "max_leaf_nodes": 3, "min_samples_leaf": 3},
}


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


def learning_curve_fingerprint(family: str) -> str:
    return _digest({
        "family": family, "base": LC_BASE_ROLE, "strategies": LC_STRATEGIES, "budgets": LC_BUDGETS,
        "repeats": LC_REPEATS, "fixed": ESTIMATOR_FIXED[family], "params": LC_PARAMS[family],
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


def learning_curve_path(family: str, key: str, budget: int) -> Path:
    return ROOT / "artifacts" / "learning_curve" / f"{family}__{key}__n{budget}.joblib"


def metrics_path(test: str, scoring: str) -> Path:
    return ROOT / "outputs" / "metrics" / f"{test}__{scoring}.json"


def learning_curve_metrics_path() -> Path:
    return ROOT / "outputs" / "metrics" / "learning_curve.json"


def tables_dir() -> Path:
    return ROOT / "outputs" / "tables"


def figures_dir() -> Path:
    return ROOT / "outputs" / "figures"


def inspect_path(role: str, stage: str, row: int, view: str) -> Path:
    return ROOT / "artifacts" / "inspect" / f"{ROLES[role].stem}_{stage}_row{row}_{view}.png"
