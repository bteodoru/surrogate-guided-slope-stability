"""Do the replicated results depend on test points whose critical circle enters the crest
almost vertically?

Such a circle has its centre within 0.01 H of crest height. On it the last slice is steep,
and its FOS depends on how the program discretizes that slice (docs/solver-crosscheck/README.md,
V06). This drops those test points and recomputes, from the stored replication predictions
(r1-r5, nothing refitted):
  - every arm against B, Welch 95% interval of the mean log-RMSE difference, on the decision
    range and its two halves;
  - the relocation gate, median L < 0 < median H per learner x test cell.
On the full test sets it reproduces outputs/metrics/replications.json. Run from the repository
root:  PYTHONPATH=. python docs/crest-entry-check/check.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sgs import config, data

NEAR = 0.01                                  # |centre height - H| < NEAR * H
ARMS = ("C-log", "C-oracle", "C-logP", "C-seq", "C-logG",
        "C0-log", "C0-oracle", "C0-logP", "C0-seq", "C0-logG")
SUBSETS = {"decision-range": (0.5, 2.0), "decision-lower": (0.5, 1.0), "decision-upper": (1.0, 2.0)}
FILES = {"test-natural": "data/final/test_natural_300.csv",
         "test-stratified": "data/final/test_stratified_150.csv"}


def log_rmse(pred, y, mask):
    return float(np.sqrt(np.mean((np.log(pred[mask]) - np.log(y[mask])) ** 2)))


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = se ** 4 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = stats.t.ppf(0.975, df) * se
    d = a.mean() - b.mean()
    return d, d - half, d + half


def verdict(lo, hi):
    return "below 0" if hi < 0 else ("above 0" if lo > 0 else "contains 0")


def main():
    rep = json.load(open("outputs/metrics/replications.json"))
    conf = rep["confirmatory"]
    units = {(f, r): joblib.load(config.replication_fit_path(f, f"r{r}"))["arms"]
             for f in ("rf", "gbm") for r in conf}
    near = {}
    for name, path in FILES.items():
        t = pd.read_csv(path)
        near[name] = (((t.Center_Y - t.H) / t.H).abs() < NEAR).to_numpy()
    rows, gates = [], []
    for name in FILES:
        y = np.asarray(data.load_test(name).y, float)
        print(f"{name}: {near[name].sum()} of {len(y)} points near-vertical")
        for f in ("rf", "gbm"):
            for drop in (False, True):
                keep = ~near[name] if drop else np.ones(len(y), bool)
                for sub, (lo, hi) in SUBSETS.items():
                    m = (y >= lo) & (y < hi) & keep
                    vals = {a: [log_rmse(units[(f, r)][a]["predictions"][name], y, m) for r in conf]
                            for a in ("B",) + ARMS}
                    for a in ARMS:
                        d, l, h = welch(vals[a], vals["B"])
                        rows.append(dict(family=f, test=name, subset=sub, dropped=drop, arm=a,
                                         n=int(m.sum()), diff=d, lo=l, hi=h, verdict=verdict(l, h)))
                for a in ARMS[:5]:
                    L, H = [], []
                    for r in conf:
                        P = units[(f, r)]
                        for side, (lo, hi) in (("L", (0.5, 1.0)), ("H", (1.0, 2.0))):
                            m = (y >= lo) & (y < hi) & keep
                            v = np.log(log_rmse(P[a]["predictions"][name], y, m)
                                       / log_rmse(P["B"]["predictions"][name], y, m))
                            (L if side == "L" else H).append(v)
                    gates.append(dict(family=f, test=name, dropped=drop, arm=a,
                                      L=np.median(L), H=np.median(H),
                                      gate=bool(np.median(L) < 0 < np.median(H))))
    rows, gates = pd.DataFrame(rows), pd.DataFrame(gates)
    key = ["family", "test", "subset", "arm"]
    j = rows[~rows.dropped].set_index(key).join(rows[rows.dropped].set_index(key), rsuffix="_dropped")
    changed = j[j.verdict != j.verdict_dropped]
    print(f"\nverdicts changed: {len(changed)} of {len(j)}")
    print(changed[["n", "diff", "lo", "hi", "n_dropped", "diff_dropped", "lo_dropped", "hi_dropped"]]
          .round(4).to_string())
    print("\nrelocation gate passes (cells out of 4):")
    print(gates.groupby(["arm", "dropped"]).gate.sum().unstack().to_string())
    rows.to_csv("docs/crest-entry-check/intervals.csv", index=False)
    gates.to_csv("docs/crest-entry-check/gates.csv", index=False)


if __name__ == "__main__":
    main()
