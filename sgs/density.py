"""The density model of the revision (docs/revision-declaration.md §3).

Per learner x test cell, ln E_Z = alpha_{Z,cell} - a ln n_Z, with E_Z the log-RMSE on
evaluation zone Z ([lo, hi)), n_Z the training points in Z, and the slope a common to
every zone and cell. Fitted by ordinary least squares on results that already exist:
the replicated arms r1..r5 (and A, the seed alone, once: it is the same in every
realization), the zones [0.5, 1), [1, 2) and [2, 6). Reported pooled and per learner,
each with a 95% percentile interval from a bootstrap over realizations.

Reads ``outputs/metrics/replications.json`` for the zone errors and the stored
replication fits for the training FOS; writes the ``density`` entry of
``outputs/metrics/revision.json``. Nothing here reads a revision set.
"""

from __future__ import annotations

import json

import numpy as np

from sgs import config, replicate

ARMS = ("A", "B", "C-log", "C0-log", "C-seq", "C0-seq", "C-logG", "C0-logG",
        "C-oracle", "C0-oracle", "C-logP", "C0-logP")
ZONES = ("0.5–1", "1–2", "2–6")
ZONE_BOUNDS = {"0.5–1": (0.5, 1.0), "1–2": (1.0, 2.0), "2–6": (2.0, 6.0)}
N_BOOTSTRAP = 2_000


def revision_metrics_path():
    return config.ROOT / "outputs" / "metrics" / "revision.json"


def observations() -> list[dict]:
    """One row per (family, test, arm, realization, zone): the zone's log-RMSE and its
    training count. A enters once, from r0; every other arm from r1..r5."""
    values = json.loads(config.replication_metrics_path().read_text())["values"]
    units = replicate.load_units()
    rows = []
    for v in values:
        r = v["realization"]
        if v["arm"] not in ARMS or (v["arm"] == "A") != (r == 0):
            continue
        train = units[(v["family"], r)]["arms"][v["arm"]]["train_fos"]
        zones = {z["zone"]: z for z in v["zones"]}
        for name in ZONES:
            lo, hi = ZONE_BOUNDS[name]
            n = int(((train >= lo) & (train < hi)).sum())
            if n == 0 or zones[name]["n"] == 0:
                continue
            rows.append({"family": v["family"], "test": v["test"], "arm": v["arm"], "realization": r,
                         "zone": name, "n_train": n, "log_rmse": zones[name]["log_rmse"]})
    return rows


def fit_slope(rows: list[dict]) -> float:
    """OLS slope -a of ln E on ln n, with one intercept per (zone, family, test)."""
    groups = sorted({(r["zone"], r["family"], r["test"]) for r in rows})
    index = {g: i for i, g in enumerate(groups)}
    X = np.zeros((len(rows), len(groups) + 1))
    y = np.empty(len(rows))
    for i, r in enumerate(rows):
        X[i, index[(r["zone"], r["family"], r["test"])]] = 1.0
        X[i, -1] = np.log(r["n_train"])
        y[i] = np.log(r["log_rmse"])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(-coef[-1])


def bootstrap(rows: list[dict], rng: np.random.Generator) -> np.ndarray:
    """Resample the five realizations with replacement (A, from r0, kept as is)."""
    reals = list(range(1, config.REP_REALIZATIONS + 1))
    by_r = {r: [x for x in rows if x["realization"] == r] for r in [0, *reals]}
    out = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        draw = rng.choice(reals, size=len(reals), replace=True)
        sample = by_r[0] + [x for r in draw for x in by_r[r]]
        out[b] = fit_slope(sample)
    return out


def estimate() -> dict:
    rows = observations()
    rng = np.random.default_rng(config.SEEDS["rev_density_bootstrap"])
    out = {"procedure": __doc__.strip(), "n_obs": len(rows), "estimates": {}}
    for label, subset in (("pooled", rows),
                          *((f, [r for r in rows if r["family"] == f]) for f in config.REP_FAMILIES)):
        a = fit_slope(subset)
        boot = bootstrap(subset, rng)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["estimates"][label] = {"a": a, "ci95": [float(lo), float(hi)], "n_obs": len(subset)}
    return out


def write(entry_key: str, entry: dict) -> None:
    path = revision_metrics_path()
    current = json.loads(path.read_text()) if path.exists() else {}
    current[entry_key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")


def run() -> dict:
    result = estimate()
    write("density", result)
    return result
