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


_zone_bounds = zones


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


def prob_less_paired(a, b) -> float:
    """Share of within-realization pairs with a_i < b_i, ties counted 1/2. Descriptive:
    the pairs are not independent observations."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
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


# --- replications ---------------------------------------------------------------------------

REPLICATION_PROCEDURE = (
    "Common fixed hyper-parameters (config.LC_PARAMS, chosen by CV on the seed alone) for every arm and "
    "realization; conclusions are conditional on the seed, the test sets and these hyper-parameters. "
    "r0 = the existing sets (exploratory, reported apart); r1.. = new realizations (confirmatory). "
    "Independent arms: Welch interval; arms sharing points within a realization: paired interval. "
    "P(first<second) is descriptive (the pairs are not independent): for Welch comparisons it is taken "
    "over all n x n realization pairs and, for paired comparisons, over the n within-realization pairs; "
    "ties count 1/2 in both. "
    "Scales: log-RMSE (absolute) and log(log-RMSE) (relative). An interval containing 0 is inconclusive, "
    "not a tie. These intervals cover variability between designs only. Because both test sets are held "
    "fixed, test-sampling error is a component common to every realization: it does not widen the spread "
    "between them and does not average out over them. It is covered separately by the paired bootstrap of "
    "`sgs evaluate`, and the two components are never combined. "
    "The oracle block (C-oracle against C-logP, confirmatory realizations only, no r0) follows the "
    "readings declared in docs/oracle-declaration.md before the pool was solved; 'attributable to the "
    "screener's error, in this design' is used only where the paired dR_O interval excludes 0, on the "
    "side of the reading, in at least three of four cells.")


def interaction_rows(*, series, tests, measures, interactions, level: float) -> list[dict]:
    """Learner-contrast rows. The paired method is a difference-in-differences inside
    the realization; it is used where the two arms share a realization's pool."""
    rows = []
    for test in tests:
        for measure in measures:
            for spec in interactions:
                deltas = {}
                for arm in (spec.first, spec.second):
                    rf, gbm = series("rf", test, arm, measure), series("gbm", test, arm, measure)
                    deltas[arm] = None if rf is None or gbm is None else list(np.log(rf) - np.log(gbm))
                if deltas[spec.first] is None or deltas[spec.second] is None:
                    continue
                interval = (paired_interval(deltas[spec.first], deltas[spec.second], level)
                            if spec.method == "paired"
                            else welch_interval(deltas[spec.first], deltas[spec.second], level))
                rows.append({"test": test, "measure": measure, "first": spec.first,
                             "second": spec.second, "method": spec.method, **interval,
                             "first_values": [float(x) for x in deltas[spec.first]],
                             "second_values": [float(x) for x in deltas[spec.second]]})
    return rows


def _measure_value(row: dict, measure: str):
    if measure in row["subsets"]:
        return row["subsets"][measure]["log_rmse"]
    return next(z["log_rmse"] for z in row["zones"] if z["zone"] == measure)


def relocation_reading(cells: dict) -> dict:
    """Classify the screener swap's effect on relocation (spec §10.2).

    Relocation is declared present only where the arm gains below 1 and loses in
    1-2 against B; R alone would also call a uniform loss a relocation. The labels
    apply only after that gate; everything else is mixed/inconclusive.
    """
    margin = float(np.log(config.RELOCATION_MARGIN))
    need = config.RELOCATION_MIN_CELLS
    per_cell = {}
    for cell, values in cells.items():
        L = float(np.median(values["L_g"]))
        H = float(np.median(values["H_g"]))
        per_cell[cell] = {"L": L, "H": H, "R": H - L, "dR": float(np.median(values["dR"]))}
    gate = sum(1 for v in per_cell.values() if v["L"] < 0 < v["H"])
    if gate >= need:
        if sum(1 for v in per_cell.values() if v["dR"] >= margin) >= need:
            reading = "larger"
        elif sum(1 for v in per_cell.values() if abs(v["dR"]) < margin) >= need:
            reading = "similar"
        else:
            reading = "mixed/inconclusive"
    elif sum(1 for v in per_cell.values() if v["R"] <= -margin) >= need:
        reading = "reverses"
    elif sum(1 for v in per_cell.values() if abs(v["R"]) < margin) >= need:
        reading = "vanishes"
    else:
        reading = "mixed/inconclusive"
    return {"reading": reading, "gate_cells": gate, "margin": margin,
            "min_cells": need, "cells": {f"{f}/{t}": v for (f, t), v in per_cell.items()}}


def _relocation_cells(*, series, families, tests, zones, arm="C-logG", reference="C-log",
                      base="B") -> dict:
    """Per-(family, test) series keyed as relocation_reading expects (spec §10.2).

    L_g, H_g are `arm`'s own log-ratios against `base` in [0.5, 1) and [1, 2), per
    realization: they feed the gate and the level R. dR is the paired, per-realization
    difference R(arm) - R(reference), each R taken against the same `base`; `base`
    cancels in it exactly. `series` must return every confirmatory realization, in the
    same order, for the pairing to hold. A cell is omitted where any of the six series
    is unavailable."""
    bounds = _zone_bounds(zones)
    low = next(label for label, lo, hi in bounds if lo == 0.5 and hi == 1.0)
    high = next(label for label, lo, hi in bounds if lo == 1.0 and hi == 2.0)

    def log_ratio(family, test, x, zone):
        a, b = series(family, test, x, zone), series(family, test, base, zone)
        if a is None or b is None:
            return None
        if len(a) != len(b):
            raise MisalignedPair(f"relocation {family}/{test}: {x} and {base} differ in realizations")
        return [float(u) - float(v) for u, v in zip(np.log(a), np.log(b))]

    out = {}
    for family in families:
        for test in tests:
            l_g, h_g = log_ratio(family, test, arm, low), log_ratio(family, test, arm, high)
            l_c, h_c = log_ratio(family, test, reference, low), log_ratio(family, test, reference, high)
            if None in (l_g, h_g, l_c, h_c):
                continue
            if len(l_g) != len(l_c):
                raise MisalignedPair(f"relocation {family}/{test}: {arm} and {reference} differ in realizations")
            d_r = [(hg - lg) - (hc - lc) for hg, lg, hc, lc in zip(h_g, l_g, h_c, l_c)]
            out[(family, test)] = {"L_g": l_g, "H_g": h_g, "dR": d_r}
    return out


# --- the oracle (spec 2026-09-23 oracle-reference-pool §9) -------------------------------------

def oracle_level(intervals: list[dict], n_cells: int) -> int:
    """Level against B on the decision range: 3 every one of n_cells entirely below 0;
    2 one cell entirely below 0; 1 one cell with a negative mean; 0 otherwise."""
    if len(intervals) == n_cells and all(iv["ci"][1] < 0 for iv in intervals):
        return 3
    if any(iv["ci"][1] < 0 for iv in intervals):
        return 2
    if any(iv["diff"] < 0 for iv in intervals):
        return 1
    return 0


def oracle_density(rho_z: float, medians: list[float]) -> dict:
    """The density prediction for one zone, at the direction and magnitude levels."""
    margin, need, ratio = float(np.log(config.RELOCATION_MARGIN)), config.RELOCATION_MIN_CELLS, \
        config.ORACLE_DENSITY_RATIO
    if rho_z >= ratio:
        sign, predicted = -1.0, "negative"
    elif rho_z <= 1 / ratio:
        sign, predicted = 1.0, "positive"
    else:
        return {"rho": rho_z, "predicted": None, "direction_met": None, "magnitude_met": None,
                "medians": medians}
    direction = sum(1 for x in medians if np.sign(x) == sign) >= need
    magnitude = direction and sum(1 for x in medians if np.sign(x) == sign and abs(x) >= margin) >= need
    return {"rho": rho_z, "predicted": predicted, "direction_met": bool(direction),
            "magnitude_met": bool(magnitude), "medians": medians}


def oracle_rho(values: list[dict], confirm: list[int], zones) -> dict:
    """rho_Z = median over realizations of n_Z(C-oracle) / n_Z(C-logP), from the training
    sets (seed + selection) in the evaluation zones [lo, hi): L = [0.5, 1), H = [1, 2)."""
    bounds = _zone_bounds(zones)
    index = {z: next(i for i, (_, lo, hi) in enumerate(bounds) if (lo, hi) == edge)
             for z, edge in (("L", (0.5, 1.0)), ("H", (1.0, 2.0)))}
    counts = {(v["arm"], v["realization"]): v["train_zone_counts"] for v in values}
    per = {z: [counts[("C-oracle", r)][i] / counts[("C-logP", r)][i] for r in confirm]
           for z, i in index.items()}
    return {"L": float(np.median(per["L"])), "H": float(np.median(per["H"])), "per_realization": per}


def oracle_reading(cells: dict, rho: dict, level: float) -> dict:
    """Step P, groups A/B and C, and the density predictions (spec §9.2-9.3). First
    matching row, in order, within each group. Attribution needs the paired dR_O
    interval to exclude 0 in >= 3/4 cells, on the side of the reading (above 0 for A1,
    below 0 for A2, B2 and B3); equivalence needs it inside (-m, m)."""
    m, need = float(np.log(config.RELOCATION_MARGIN)), config.RELOCATION_MIN_CELLS
    n_cells = len(config.REP_FAMILIES) * len(config.TEST_ROLES)
    zeros = lambda xs: [0.0] * len(xs)
    per = {}
    for cell, v in cells.items():
        R_o = [h - l for h, l in zip(v["H_o"], v["L_o"])]
        R_p = [h - l for h, l in zip(v["H_p"], v["L_p"])]
        dR = [a - b for a, b in zip(R_o, R_p)]
        dL = list(np.log(v["E_oL"]) - np.log(v["E_pL"]))
        dH = list(np.log(v["E_oH"]) - np.log(v["E_pH"]))
        per[cell] = {"L_o": float(np.median(v["L_o"])), "H_o": float(np.median(v["H_o"])),
                     "L_p": float(np.median(v["L_p"])), "H_p": float(np.median(v["H_p"])),
                     "R_o": float(np.median(R_o)), "dR": float(np.median(dR)),
                     "dL": float(np.median(dL)), "dH": float(np.median(dH)),
                     "dR_interval": paired_interval(R_o, R_p, level),
                     "R_o_interval": paired_interval(R_o, zeros(R_o), level),
                     "dL_interval": paired_interval(dL, zeros(dL), level),
                     "dH_interval": paired_interval(dH, zeros(dH), level),
                     "dec_o_vs_b": welch_interval(v["dec_o"], v["dec_b"], level),
                     "dec_p_vs_b": welch_interval(v["dec_p"], v["dec_b"], level),
                     "dec_o_vs_p": paired_interval(v["dec_o"], v["dec_p"], level),
                     "values": {"R_o": R_o, "R_p": R_p, "dR": dR, "dL": dL, "dH": dH}}
    count = lambda pred: sum(1 for c in per.values() if pred(c))
    inside = lambda iv: -m < iv["ci"][0] and iv["ci"][1] < m
    logp_gate = count(lambda c: c["L_p"] < 0 < c["H_p"]) >= need
    oracle_gate = count(lambda c: c["L_o"] < 0 < c["H_o"]) >= need
    if oracle_gate:
        if count(lambda c: c["dR"] >= m) >= need:
            row = "A1"
        elif count(lambda c: c["dR"] <= -m) >= need:
            row = "A2"
        elif count(lambda c: inside(c["dR_interval"])) >= need:
            row = "A3"
        elif count(lambda c: abs(c["dR"]) < m) >= need:
            row = "A4"
        else:
            row = "A5"
    elif not logp_gate:
        row = "B1"
    elif count(lambda c: inside(c["R_o_interval"])) >= need:
        row = "B2"
    elif count(lambda c: c["R_o"] <= -m) >= need:
        row = "B3"
    elif count(lambda c: abs(c["R_o"]) < m) >= need:
        row = "B4"
    else:
        row = "B5"
    # Attribution needs the interval to exclude 0 on the side of the reading, not
    # merely somewhere: an A1 "larger" cannot be attributed from intervals below 0.
    # B2/B3 fire only after step P passed (B1 comes first), so C-logP relocates and
    # the attributable direction is C-oracle relocating less: dR_O < 0.
    side = {"A1": 1, "A2": -1, "B2": -1, "B3": -1}.get(row)
    excludes_on_side = lambda iv: iv["ci"][0] > 0 if side > 0 else iv["ci"][1] < 0
    attribution = side is not None and count(lambda c: excludes_on_side(c["dR_interval"])) >= need
    level_o = oracle_level([c["dec_o_vs_b"] for c in per.values()], n_cells)
    level_p = oracle_level([c["dec_p_vs_b"] for c in per.values()], n_cells)
    if (level_o == 3 and len(per) == n_cells and level_p < 3
            and all(c["dec_o_vs_p"]["ci"][1] < 0 for c in per.values())):
        c_row = "C1"
    elif level_o == 3:
        c_row = "C2"
    else:
        c_row = "C3"
    density = {z: oracle_density(rho[z], [c[f"d{z}"] for c in per.values()]) for z in ("L", "H")}
    return {"step_p": {"logp_gate": logp_gate,
                       "medians": {f"{f}/{t}": [c["L_p"], c["H_p"]] for (f, t), c in per.items()}},
            "oracle_gate": oracle_gate, "row": row, "attribution": bool(attribution), "c_row": c_row,
            "levels": {"C-oracle": level_o, "C-logP": level_p}, "density": density,
            "margin": m, "min_cells": need, "cells": {f"{f}/{t}": c for (f, t), c in per.items()}}


def _oracle_cells(*, series, families, tests, zones, base="B") -> dict:
    """Per (family, test): the lists oracle_reading takes, over the confirmatory
    realizations in order. A cell is omitted where any series is unavailable."""
    bounds = _zone_bounds(zones)
    low = next(label for label, lo, hi in bounds if lo == 0.5 and hi == 1.0)
    high = next(label for label, lo, hi in bounds if lo == 1.0 and hi == 2.0)
    out = {}
    for family in families:
        for test in tests:
            got = {(arm, z): series(family, test, arm, z)
                   for arm in ("C-oracle", "C-logP", base) for z in (low, high, "decision-range")}
            if any(v is None for v in got.values()):
                continue
            ratio = lambda arm, z: [float(a - b) for a, b in zip(np.log(got[(arm, z)]), np.log(got[(base, z)]))]
            out[(family, test)] = {
                "L_o": ratio("C-oracle", low), "H_o": ratio("C-oracle", high),
                "L_p": ratio("C-logP", low), "H_p": ratio("C-logP", high),
                "E_oL": got[("C-oracle", low)], "E_oH": got[("C-oracle", high)],
                "E_pL": got[("C-logP", low)], "E_pH": got[("C-logP", high)],
                "dec_o": got[("C-oracle", "decision-range")], "dec_p": got[("C-logP", "decision-range")],
                "dec_b": got[(base, "decision-range")]}
    return out


def replication_metrics(units: Mapping, tests: Mapping[str, data.TestSet], *, zones, bins, selection_range,
                        decision_range, level, welch, paired, interactions) -> dict:
    values = []
    for (family, r), stored in sorted(units.items()):
        for arm, unit in stored["arms"].items():
            for name, test in tests.items():
                if not np.array_equal(stored["test_ids"][name], test.ids):
                    raise MisalignedPair(f"replication {family} r{r}: predictions for {name} are not aligned")
                p = np.asarray(unit["predictions"][name], dtype=float)
                train = np.asarray(unit["train_fos"], dtype=float)
                values.append({
                    "family": family, "test": name, "arm": arm, "realization": int(r),
                    "subsets": subset_metrics(test.y, p, selection_range, decision_range),
                    "zones": zone_metrics(test.y, p, zones),
                    "bins": zone_metrics(test.y, p, bins),
                    "train_counts": [int(_zone_mask(train, lo, hi).sum()) for _, lo, hi in _zone_bounds(bins)],
                    "train_zone_counts": [int(_zone_mask(train, lo, hi).sum()) for _, lo, hi in _zone_bounds(zones)],
                })
    realizations = sorted({v["realization"] for v in values})
    confirm = [r for r in realizations if r >= 1]
    measures = [*SUBSETS, *(label for label, _, _ in _zone_bounds(zones))]
    index = {(v["family"], v["test"], v["arm"], v["realization"]): v for v in values}
    families = sorted({v["family"] for v in values})

    def series(family, test, arm, measure):
        out = [_measure_value(index[(family, test, arm, r)], measure) for r in confirm
               if (family, test, arm, r) in index]
        return None if len(out) < 2 or any(x is None for x in out) else [float(x) for x in out]

    summaries, comparisons, inter = [], [], []
    for family in families:
        for test in tests:
            for arm in sorted({v["arm"] for v in values}):
                for measure in measures:
                    s = series(family, test, arm, measure)
                    r0 = index.get((family, test, arm, 0))
                    summaries.append({"family": family, "test": test, "arm": arm, "measure": measure,
                                      "r0": None if r0 is None else _measure_value(r0, measure),
                                      "values": s, "mean": None if s is None else float(np.mean(s)),
                                      "sd": None if s is None else float(np.std(s, ddof=1)),
                                      "min": None if s is None else float(min(s)),
                                      "max": None if s is None else float(max(s))})
            for measure in measures:
                for method, pairs in (("welch", welch), ("paired", paired)):
                    for first, second in pairs:
                        a, b = series(family, test, first, measure), series(family, test, second, measure)
                        if a is None or b is None:
                            continue
                        for scale, (x, y) in (("log-rmse", (a, b)),
                                              ("relative", (list(np.log(a)), list(np.log(b))))):
                            interval = (welch_interval(x, y, level) if method == "welch"
                                        else paired_interval(x, y, level))
                            p_less = (prob_less(x, y) if method == "welch"
                                      else prob_less_paired(x, y))
                            comparisons.append({"family": family, "test": test, "measure": measure,
                                                "first": first, "second": second, "method": method,
                                                "scale": scale, **interval, "p_first_less": p_less,
                                                "first_values": a, "second_values": b})
    if {"rf", "gbm"} <= set(families):
        inter = interaction_rows(series=series, tests=tests, measures=measures,
                                 interactions=interactions, level=level)
    def complete_series(family, test, arm, measure):
        # dR pairs realizations by position: take an arm only when it has every one.
        out = series(family, test, arm, measure)
        return out if out is not None and len(out) == len(confirm) else None

    relocation = {"arm": "C-logG", "reference": "C-log", "base": "B",
                  **relocation_reading(_relocation_cells(series=complete_series, families=families,
                                                         tests=tests, zones=zones))}
    oracle = None
    if {"C-oracle", "C-logP"} <= {v["arm"] for v in values}:
        rho = oracle_rho(values, confirm, zones)
        oracle = {"arm": "C-oracle", "reference": "C-logP", "base": "B", "rho": rho,
                  **oracle_reading(_oracle_cells(series=complete_series, families=families, tests=tests,
                                                 zones=zones), rho, level)}
    return {"procedure": REPLICATION_PROCEDURE, "realizations": realizations, "confirmatory": confirm,
            "values": values, "summaries": summaries, "comparisons": comparisons, "interactions": inter,
            "relocation": relocation, "oracle": oracle}


def run_replications() -> None:
    from sgs import design, replicate
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    result = replication_metrics(
        replicate.load_units(), tests, zones=config.ZONE_EDGES,
        bins=tuple(design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)),
        selection_range=config.SELECTION_RANGE, decision_range=config.DECISION_RANGE, level=config.REP_CI,
        welch=config.REP_WELCH, paired=config.REP_PAIRED, interactions=config.REP_INTERACTIONS)
    path = config.replication_metrics_path()
    _write_json(path, result)
    print(f"[evaluate] wrote {path.name}")
