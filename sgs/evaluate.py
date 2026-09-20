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

SUBSETS = ("all", "selection-range", "decision-range")
# Paired comparisons run on both scales: log is the trained response and the
# primary criterion, FOS the secondary one.
SCALES = ("log", "fos")
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


def on_scale(values, scale: str) -> np.ndarray:
    if scale == "log":
        return np.log(np.asarray(values, dtype=float))
    if scale == "fos":
        return np.asarray(values, dtype=float)
    raise ValueError(f"unknown scale {scale!r}")


def global_metrics(y, p) -> dict:
    e = p - y
    return {"n": int(len(y)),
            "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "rmse": rmse(y, p),
            "log_rmse": rmse(np.log(y), np.log(p)),
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
               "rmse": None, "log_rmse": None, "mae": None, "bias": None}
        if n:
            e = p[mask] - y[mask]
            row.update(rmse=float(np.sqrt(np.mean(e ** 2))),
                       log_rmse=rmse(np.log(y[mask]), np.log(p[mask])),
                       mae=float(np.abs(e).mean()), bias=float(e.mean()))
        rows.append(row)
    return rows


def minimax(zone_rows, key: str = "rmse") -> dict:
    worst = max((r for r in zone_rows if r["n"]), key=lambda r: r[key])
    return {"zone": worst["zone"], key: worst[key]}


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


def subset_mask(y, subset: str, selection_range, decision_range=None) -> np.ndarray:
    if subset == "all":
        return np.ones(len(y), dtype=bool)
    if subset == "selection-range":
        lo, hi = selection_range
        return (y >= lo) & (y <= hi)
    if subset == "decision-range":
        if decision_range is None:
            raise ValueError("the decision-range subset needs decision_range")
        lo, hi = decision_range
        return (y >= lo) & (y < hi)
    raise ValueError(f"unknown subset {subset!r}")


def subset_metrics(y, p, selection_range, decision_range) -> dict:
    out = {}
    for subset in SUBSETS:
        m = subset_mask(y, subset, selection_range, decision_range)
        n = int(m.sum())
        out[subset] = {"n": n,
                       "rmse": rmse(y[m], p[m]) if n else None,
                       "log_rmse": rmse(np.log(y[m]), np.log(p[m])) if n else None}
    return out


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


def wilcoxon_abs(p1: Prediction, p2: Prediction, *, registry, mask=None, scale: str = "fos") -> dict:
    """Wilcoxon signed-rank on d = |e1| - |e2| with the registered alternative;
    errors on the given scale (log: e = log p - log y)."""
    _check_aligned(p1, p2)
    comparison = registered(p1.arm, p2.arm, registry)
    m = _mask(p1, mask)
    y = on_scale(p1.y_true[m], scale)
    d = np.abs(on_scale(p1.y_pred[m], scale) - y) - np.abs(on_scale(p2.y_pred[m], scale) - y)
    result = stats.wilcoxon(d, alternative=comparison.alternative)
    return {"first": p1.arm, "second": p2.arm, "scale": scale, "n": int(m.sum()),
            "median_d": float(np.median(d)), "p_d_lt_0": float(np.mean(d < 0)),
            "statistic": float(result.statistic), "pvalue": float(result.pvalue),
            "alternative": comparison.alternative, "hypothesised": comparison.hypothesised,
            "reading": f"d = |e_{p1.arm}| - |e_{p2.arm}|; d < 0 means {p1.arm} has smaller absolute errors"}


def paired_bootstrap(p1: Prediction, p2: Prediction, *, registry, n_boot: int, seed: int,
                     ci: float, mask=None, scale: str = "fos") -> dict:
    """Percentile interval for RMSE(first) - RMSE(second) on the given scale, rows resampled jointly."""
    _check_aligned(p1, p2)
    registered(p1.arm, p2.arm, registry)
    m = _mask(p1, mask)
    y, a, b = (on_scale(v[m], scale) for v in (p1.y_true, p1.y_pred, p2.y_pred))
    draws = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    delta = (np.sqrt(((a[draws] - y[draws]) ** 2).mean(axis=1))
             - np.sqrt(((b[draws] - y[draws]) ** 2).mean(axis=1)))
    tail = (1 - ci) / 2
    return {"scale": scale, "delta_rmse": rmse(y, a) - rmse(y, b),
            "ci": [float(np.quantile(delta, tail)), float(np.quantile(delta, 1 - tail))],
            "ci_level": ci, "p_delta_lt_0": float(np.mean(delta < 0)), "n_boot": int(n_boot),
            "limitation": BOOTSTRAP_LIMITATION}


# --- replication statistics ------------------------------------------------------------------

def welch_interval(a, b, level: float) -> dict:
    """Interval for mean(a) - mean(b) from independent samples, unequal variances."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff, se = float(a.mean() - b.mean()), float(np.sqrt(va + vb))
    if se == 0.0:
        return {"diff": diff, "se": 0.0, "df": None, "t": None, "ci": [diff, diff], "level": level}
    df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))
    t = float(stats.t.ppf(0.5 + level / 2, df))
    return {"diff": diff, "se": se, "df": df, "t": t, "ci": [diff - t * se, diff + t * se], "level": level}


def paired_interval(a, b, level: float) -> dict:
    """Interval for the mean of a - b over naturally paired observations."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    diff, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.5 + level / 2, n - 1))
    return {"diff": diff, "se": se, "df": n - 1, "t": t, "ci": [diff - t * se, diff + t * se], "level": level}


def prob_less(a, b) -> float:
    """Share of all (a_i, b_j) pairs with a_i < b_j, ties counted 1/2. Descriptive:
    the pairs are not independent observations."""
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    return float(np.mean((a < b) + 0.5 * (a == b)))


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
             selection_range, decision_range, n_boot, bootstrap_seed, bootstrap_ci) -> EvalResult:
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
            "minimax_log": minimax(zone_rows, key="log_rmse"),
            "subsets": subset_metrics(pred.y_true, pred.y_pred, selection_range, decision_range),
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
                mask = subset_mask(test.y, subset, selection_range, decision_range)
                for scale in SCALES:
                    rows.append({
                        "family": family, "subset": subset, "scale": scale,
                        "first": comparison.first, "second": comparison.second,
                        "wilcoxon": wilcoxon_abs(first, second, registry=comparisons, mask=mask,
                                                 scale=scale),
                        "bootstrap": paired_bootstrap(first, second, registry=comparisons,
                                                      n_boot=n_boot, seed=bootstrap_seed,
                                                      ci=bootstrap_ci, mask=mask, scale=scale),
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
                              selection_range=config.SELECTION_RANGE,
                              decision_range=config.DECISION_RANGE, n_boot=config.N_BOOTSTRAP,
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


# --- learning curve -------------------------------------------------------------------------

LEARNING_CURVE_PROCEDURE = ("Fixed hyper-parameters per family, not tuned per point; declared in Methods. "
                            "RF: scikit-learn defaults. GBM: the grid point selected by CV on the seed "
                            "alone (config.LC_PARAMS).")


def learning_curve_metrics(units: Mapping, tests: Mapping[str, data.TestSet], *, selection_range,
                           edges) -> dict:
    """Mean and spread over repeats, per (family, curve, budget, test): RMSE on the FOS
    scale (global, selection range, per zone) and on the log scale the model is trained on.
    FOS-scale RMSE is dominated by the few highest-FOS test points; the log scale and the
    zones show whether a change is global or confined to the tail."""
    rows = []
    for (family, curve, budget), unit_list in sorted(units.items()):
        for name, test in tests.items():
            inside = subset_mask(test.y, "selection-range", selection_range)
            zone_masks = [(label, _zone_mask(test.y, lo, hi)) for label, lo, hi in zones(edges)]
            scores = {"rmse": [], "rmse_selection": [], "log_rmse": [], "log_rmse_selection": []}
            per_zone = {label: [] for label, _ in zone_masks}
            for unit in unit_list:
                pred = unit["predictions"][name]
                if not np.array_equal(pred["ids"], test.ids):
                    raise MisalignedPair(f"learning-curve predictions for {name} are not aligned with the test set")
                p = pred["y_pred"]
                scores["rmse"].append(rmse(test.y, p))
                scores["rmse_selection"].append(rmse(test.y[inside], p[inside]))
                scores["log_rmse"].append(rmse(np.log(test.y), np.log(p)))
                scores["log_rmse_selection"].append(rmse(np.log(test.y[inside]), np.log(p[inside])))
                for label, mask in zone_masks:
                    if mask.any():
                        per_zone[label].append(rmse(test.y[mask], p[mask]))
            row = {"family": family, "curve": curve, "budget": int(budget), "test": name,
                   "n_repeats": len(unit_list)}
            for key, values in scores.items():
                row[f"{key}_mean"], row[f"{key}_std"] = float(np.mean(values)), float(np.std(values))
            row["zones"] = []
            for label, mask in zone_masks:
                values = per_zone[label]
                row["zones"].append({"zone": label, "n": int(mask.sum()),
                                     "rmse_mean": float(np.mean(values)) if values else None,
                                     "rmse_std": float(np.std(values)) if values else None})
            rows.append(row)
    return {"procedure": LEARNING_CURVE_PROCEDURE, "rows": rows}


def run_learning_curve() -> None:
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    result = learning_curve_metrics(models.load_lc_units(), tests, selection_range=config.SELECTION_RANGE,
                                    edges=config.ZONE_EDGES)
    path = config.learning_curve_metrics_path()
    _write_json(path, result)
    print(f"[evaluate] wrote {path.name}")
