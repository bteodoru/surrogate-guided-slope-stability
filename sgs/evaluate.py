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
