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
