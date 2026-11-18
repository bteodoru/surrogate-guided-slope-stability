"""The revision's smooth learner (docs/revision-declaration.md §9).

A Gaussian process, ConstantKernel x Matern(nu=2.5, one length scale per feature) +
WhiteKernel, on the five parameters and ln(gamma*H/c), standardized on each training set;
target log FOS with ``normalize_y=True``; hyper-parameters by marginal likelihood on each
arm's training data, with ``N_RESTARTS`` optimizer restarts from SEEDS["rev_gp_restarts"].
The same procedure for every arm, old and new, with its ablations, r1..r5.

Each realization is cached in ``artifacts/revision/gp/``. Readings, for the GP alone, on
both fixed test sets: the level against B on the decision range (Welch, as in the
replications; level over the two cells) and the relocation gate. The declared prediction:
the gate passes for C-log in at least 1 of the 2 test sets. Wording: "consistent across two
tree-ensemble families and a Gaussian process", never model-agnostic.
Results: the ``gp`` entry of ``outputs/metrics/revision.json``.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sgs import config, density, evaluate, models, replicate
from sgs import revision_large as rl
from sgs import revision_robust as rr

FAMILY = "gp"
N_RESTARTS = 5
INPUTS = (*config.PARAMS, "gamma_H_c")


def _inputs(X):
    out = np.asarray(X[list(INPUTS)], dtype=float).copy()
    out[:, -1] = np.log(out[:, -1])
    return out


def estimator():
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(len(INPUTS)), nu=2.5)
              + WhiteKernel(1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=N_RESTARTS,
                                  random_state=config.SEEDS["rev_gp_restarts"])
    return make_pipeline(FunctionTransformer(_inputs), StandardScaler(), gp)


def fit_path(r: int):
    return config.ROOT / "artifacts" / "revision" / "gp" / f"gp__r{r}.joblib"


def _state(r: int) -> dict:
    return {"kernel": "C*Matern(2.5, ARD)+White", "restarts": N_RESTARTS,
            "seed": config.SEEDS["rev_gp_restarts"], "inputs": INPUTS,
            "data": replicate.data_hashes(r)}


def fit_all() -> dict:
    # a length scale at its upper bound means the GP found that input irrelevant; it is
    # visible in the stored kernel, so the warning is not repeated in the log
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    tests = {**rr._tests(), **rl.large_tests()}
    units = {}
    for r in rr.REALS:
        path, state = fit_path(r), _state(r)
        if path.exists() and joblib.load(path)["state"] == state:
            units[(FAMILY, r)] = joblib.load(path)
            continue
        arms = {}
        for arm, frame in rr.frames(r).items():
            print(f"[gp] {arm} r{r} (n={len(frame)})", flush=True)
            model = estimator().fit(frame, np.log(frame["FOS"].to_numpy(float)))
            arms[arm] = {"n_train": int(len(frame)), "kernel": str(model[-1].kernel_),
                         "log_marginal_likelihood": float(model[-1].log_marginal_likelihood_value_),
                         "predictions": {n: np.exp(model.predict(t.X)) for n, t in tests.items()}}
        stored = {"state": state, "arms": arms}
        path.parent.mkdir(parents=True, exist_ok=True)
        models._dump(path, stored)
        units[(FAMILY, r)] = stored
    return units


def readings(vals, arms) -> dict:
    out = {}
    for arm in arms:
        per = {}
        for t in config.TEST_ROLES:
            a = [vals[(FAMILY, t, arm, r)]["decision"] for r in rr.REALS]
            b = [vals[(FAMILY, t, "B", r)]["decision"] for r in rr.REALS]
            iv = evaluate.welch_interval(a, b, config.REP_CI)
            L = float(np.median([np.log(vals[(FAMILY, t, arm, r)]["L"] / vals[(FAMILY, t, "B", r)]["L"])
                                 for r in rr.REALS]))
            H = float(np.median([np.log(vals[(FAMILY, t, arm, r)]["H"] / vals[(FAMILY, t, "B", r)]["H"])
                                 for r in rr.REALS]))
            per[t] = {"diff": iv["diff"], "ci": iv["ci"], "verdict": rr._verdict(iv["ci"]),
                      "L": L, "H": H, "gate": bool(L < 0 < H)}
        out[arm] = {"cells": per, "gate_passes": sum(c["gate"] for c in per.values()),
                    "level": evaluate.oracle_level([{"diff": c["diff"], "ci": c["ci"]} for c in per.values()],
                                                   len(config.TEST_ROLES))}
    return out


def run() -> dict:
    units = fit_all()
    vals = rr.values(units)
    arms = sorted({k[2] for k in vals} - {"B"})
    r = readings(vals, arms)
    decision_b = {t: float(np.mean([vals[(FAMILY, t, "B", x)]["decision"] for x in rr.REALS]))
                  for t in config.TEST_ROLES}
    entry = {"readings": r, "B_decision_log_rmse": decision_b,
             "prediction_C-log_gate": r["C-log"]["gate_passes"] >= 1,
             "kernels": {f"{a}/r{x}": units[(FAMILY, x)]["arms"][a]["kernel"]
                         for x in rr.REALS for a in units[(FAMILY, x)]["arms"]}}
    density.write("gp", entry)
    return entry
