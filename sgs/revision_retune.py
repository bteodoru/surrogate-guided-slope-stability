"""The revision's independent retuning (docs/revision-declaration.md §8.1).

B, C-log and C-seq with their ablations (C0-log, C0-seq), r1..r5, each retuned by the
protocol's grid search on its own training data: ``config.GRIDS``, ``CV_FOLDS`` folds
(``models.cv_splitter``), RMSE on the log target. The search is the protocol's
(``models.searcher``) with ``n_jobs`` capped at ``N_JOBS``, which is execution control:
it changes no result. No nested CV: the retuned model only predicts the test sets.

Each (family, arm, realization) is cached on its own in ``artifacts/revision/retune/``,
with the selected hyper-parameters, so an interruption loses one unit. The reading, per
learner x test cell, is "holds" or "differs" against the baseline (the LC_PARAMS fits),
for the decision-range verdict against B and for the relocation gate, as in §8.2-8.6.
Results: the ``retune`` entry of ``outputs/metrics/revision.json``.
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from sgs import config, data, density, models, replicate
from sgs import revision_large as rl
from sgs import revision_robust as rr

ARMS = ("B", "C-log", "C0-log", "C-seq", "C0-seq")
N_JOBS = 5


def fit_path(family: str, arm: str, r: int):
    return config.ROOT / "artifacts" / "revision" / "retune" / f"{family}__{arm}__r{r}.joblib"


def _state(family: str, arm: str, r: int) -> dict:
    return {"fingerprint": config.replication_fingerprint(family), "grid": config.GRIDS[family],
            "folds": config.CV_FOLDS, "scoring": "log-rmse",
            "data": replicate.data_hashes(r), "arm": arm}


def fit_unit(family: str, arm: str, r: int, frame, tests) -> dict:
    path, state = fit_path(family, arm, r), _state(family, arm, r)
    if path.exists():
        stored = joblib.load(path)
        if stored["state"] == state:
            return stored
    print(f"[retune] {family} {arm} r{r} (n={len(frame)})", flush=True)
    search = models.searcher(family, "log-rmse").set_params(n_jobs=N_JOBS)
    search.fit(frame[list(config.FEATURES)], np.log(frame["FOS"].to_numpy(float)))
    model = search.best_estimator_
    out = {"state": state, "best_params": {k: (v.item() if hasattr(v, "item") else v)
                                           for k, v in search.best_params_.items()},
           "best_cv_log_rmse": float(-search.best_score_), "n_train": int(len(frame)),
           "predictions": {n: np.exp(model.predict(t.X)) for n, t in tests.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    models._dump(path, out)
    return out


def fit_all() -> dict:
    tests = {**rr._tests(), **rl.large_tests()}
    units = {}
    for family in ("rf", "gbm"):                      # the fast family first
        for r in rr.REALS:
            frames = replicate.arm_frames(r)
            units[(family, r)] = {"arms": {arm: fit_unit(family, arm, r, frames[arm], tests) for arm in ARMS}}
    return units


def run() -> dict:
    units = fit_all()
    vals = rr.values(units)
    arms = [a for a in ARMS if a != "B"]
    readings = rr.cell_readings(vals, arms)
    base = json.loads(density.revision_metrics_path().read_text())["robustness"]["baseline"]
    params = {f"{f}/{a}/r{r}": u["arms"][a]["best_params"] for (f, r), u in units.items() for a in ARMS}
    entry = {"readings": readings, "against_baseline": rr.compare(base, readings),
             "best_params": params,
             "best_cv_log_rmse": {f"{f}/{a}/r{r}": u["arms"][a]["best_cv_log_rmse"]
                                  for (f, r), u in units.items() for a in ARMS},
             "n_jobs": N_JOBS}
    density.write("retune", entry)
    return entry
