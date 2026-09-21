"""Descriptive structural artifacts: per-bin training density beside per-bin
error, and the fidelity of the delivered allocation.

Design: docs/superpowers/specs/2026-09-20-structural-analysis-design.md.
Procedure declared in docs/predictions-and-outcomes.md before this module
existed. Nothing here computes a test or states an inferential claim; Welch
intervals are reused only to visualize variability between designs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sgs import config, data, design, evaluate, replicate


class StructuralError(RuntimeError):
    """An input artifact contradicts what the design guarantees."""


def EDGES() -> np.ndarray:
    return design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)


def bin_labels() -> list[str]:
    edges = EDGES()
    return [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]


def check_realizations(found, where: str) -> list[int]:
    """The confirmatory realizations must all be present.

    A missing row would otherwise produce an interval over four realizations
    that looks exactly like one over five.
    """
    expected = set(range(1, config.REP_REALIZATIONS + 1))
    missing = sorted(expected - set(found))
    if missing:
        raise StructuralError(f"{where}: realizations {missing} are missing")
    return sorted(expected)


def _split(train_counts):
    """A 17-bin row as (below 0.5, the 15 in-range bins, at or above 6)."""
    counts = np.asarray(train_counts, dtype=int)
    if len(counts) != config.N_BINS + 2:
        raise StructuralError(f"expected {config.N_BINS + 2} bins, got {len(counts)}")
    return int(counts[0]), counts[1:-1], int(counts[-1])


def _confirmatory(rep: dict) -> list[dict]:
    return [row for row in rep["values"] if 1 <= row["realization"] <= config.REP_REALIZATIONS]


def density_rows(rep: dict) -> dict[tuple[str, int], np.ndarray]:
    """(arm, realization) -> in-range training counts.

    The training set depends on neither family nor test, so the four cells must
    agree; disagreement means the artifact was built from mismatched sets.
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for row in _confirmatory(rep):
        key = (row["arm"], row["realization"])
        _, inside, _ = _split(row["train_counts"])
        if key in out and not np.array_equal(out[key], inside):
            raise StructuralError(f"train_counts differ between cells for {key}")
        out[key] = inside
    return out


def outside_density_rows(rep: dict) -> dict[tuple[str, int], tuple[int, int]]:
    """(arm, realization) -> training points below 0.5 and at or above 6.

    Spillover: the gate is on a predicted FOS, so these points were never
    intended. They are a property of the training design, not of the test error.
    """
    out: dict[tuple[str, int], tuple[int, int]] = {}
    for row in _confirmatory(rep):
        key = (row["arm"], row["realization"])
        below, _, above = _split(row["train_counts"])
        if key in out and out[key] != (below, above):
            raise StructuralError(f"outside train_counts differ between cells for {key}")
        out[key] = (below, above)
    return out


PRIOR_COUNT_KEYS = ("seed_counts", "trained_counts")


def _prior_counts(allocation: dict) -> np.ndarray:
    """The bin counts the screener's training data already held.

    The main pipeline wrote `seed_counts` for guided-log and `trained_counts`
    for the sequential batches; the replications wrote `trained_counts`
    throughout. Exactly one must be present.
    """
    present = [key for key in PRIOR_COUNT_KEYS if key in allocation]
    if len(present) != 1:
        raise StructuralError(f"prior counts: expected one of {PRIOR_COUNT_KEYS}, found {present}")
    return np.asarray(allocation[present[0]], dtype=int)


def _metadata() -> dict:
    return json.loads(config.metadata_path().read_text())


def _sets_of(realization: int) -> dict[str, list[tuple[str, dict]]]:
    """arm -> [(label, allocation provenance)], in generation order.

    The only function that knows realization 0 is the main pipeline.
    """
    meta = _metadata()
    if realization == 0:
        roles = {"C-log": ["guided-log"], "C-seq": list(replicate.SEQ_ROLES_R0)}
        return {arm: [(role, meta["datasets"][role]["allocation"]) for role in names]
                for arm, names in roles.items()}
    keys = replicate.set_keys(realization)
    grouped = {"C-log": [k for k, s, _ in keys if s == "guided-log"],
               "C-seq": [k for k, s, _ in keys if s == "guided-seq"]}
    return {arm: [(key, meta["replications"][key]["allocation"]) for key in names]
            for arm, names in grouped.items()}


def _solved(realization: int, label: str) -> pd.DataFrame:
    return data.load_role(label) if realization == 0 else replicate.load_set(label)


def _achieved_counts(realization: int, label: str) -> np.ndarray:
    """In-range bin counts of the true FOS the solver returned for one set."""
    frame = _solved(realization, label)
    return design.bin_counts(frame["FOS"].to_numpy(dtype=float), EDGES())


def allocation_rows(realization: int) -> dict[str, dict]:
    """Per arm: what the prior data held, what was asked for, what was delivered.

    C-seq sums over its batches: each one reallocates against what the earlier
    batches actually returned, so only the sum is the strategy's row.
    """
    out = {}
    for arm, sets in _sets_of(realization).items():
        seed = np.zeros(config.N_BINS, dtype=int)
        intended = np.zeros(config.N_BINS, dtype=int)
        achieved = np.zeros(config.N_BINS, dtype=int)
        for i, (label, allocation) in enumerate(sets):
            if i == 0:
                seed = _prior_counts(allocation)
            intended = intended + np.asarray(allocation["added"], dtype=int)
            achieved = achieved + _achieved_counts(realization, label)
        deviation = achieved - intended
        out[arm] = {"seed": seed.tolist(), "intended": intended.tolist(),
                    "achieved": achieved.tolist(), "deviation": deviation.tolist(),
                    "abs_deviation": int(np.abs(deviation).sum())}
    return out


def batch_allocation_rows(realization: int) -> list[dict]:
    """The C-seq batches, each on its own and cumulatively. The cumulative view
    shows whether retraining corrects a deviation or only moves it."""
    rows = []
    cum_intended = np.zeros(config.N_BINS, dtype=int)
    cum_achieved = np.zeros(config.N_BINS, dtype=int)
    for batch, (label, allocation) in enumerate(_sets_of(realization)["C-seq"], start=1):
        intended = np.asarray(allocation["added"], dtype=int)
        achieved = _achieved_counts(realization, label)
        cum_intended, cum_achieved = cum_intended + intended, cum_achieved + achieved
        rows.append({"batch": batch, "seed": _prior_counts(allocation).tolist(),
                     "intended": intended.tolist(), "achieved": achieved.tolist(),
                     "cumulative_intended": cum_intended.tolist(),
                     "cumulative_achieved": cum_achieved.tolist()})
    return rows


def _design_of(realization: int, label: str) -> pd.DataFrame:
    """The design rows, carrying FOS_pred and the screener's bin."""
    return design.read_design(label) if realization == 0 else replicate.read_set_design(label)


def _true_bin(fos: np.ndarray) -> np.ndarray:
    """Bin of a true FOS: 0..N_BINS-1 inside, N_BINS below 0.5, N_BINS+1 above 6.

    The convention is the generator's, not `evaluate`'s: `design.bin_counts` and
    `design.guided_log_deficit` treat the bins as (lo, hi] with the first closed
    and the last closed at 6, so FOS exactly 6 is the last in-range bin. It must
    match, because this array is compared against the design's own `bin` column
    and against `achieved`, which comes from `bin_counts`. Everything read out of
    `replications.json` uses the other convention: both its `bins` and its
    `train_counts` are built with `evaluate._zone_mask`, i.e. [lo, hi), so there
    a value of exactly 6 falls in `FOS >= 6` and a value on an interior edge goes
    to the bin on its right. The conventions differ only on an exact edge, which
    a continuous solver output reaches with probability zero;
    `test_true_bin_boundaries` pins this module's choice down.
    """
    edges = EDGES()
    out = np.full(len(fos), config.N_BINS, dtype=int)
    out[fos > edges[-1]] = config.N_BINS + 1
    inside = (fos >= edges[0]) & (fos <= edges[-1])
    out[inside] = np.maximum(np.searchsorted(edges, fos[inside], side="left") - 1, 0)
    return out


def displacement_rows(realization: int) -> dict[str, dict]:
    """Where the screener's selected points actually landed.

    Conditioned on selection: these points were chosen because they were
    predicted inside the range and inside a deficit bin. The shift statistics
    cover only the points that stayed inside [0.5, 6]; the rest are `outside`.
    """
    params = list(config.PARAMS)
    out = {}
    for arm, sets in _sets_of(realization).items():
        if not sets:
            continue
        predicted, actual = [], []
        for label, _ in sets:
            d, f = _design_of(realization, label), _solved(realization, label)
            if len(d) != len(f) or not np.allclose(d[params].to_numpy(dtype=float),
                                                   f[params].to_numpy(dtype=float)):
                raise StructuralError(f"{label}: design and final are not aligned row for row")
            predicted.append(d["bin"].to_numpy(dtype=int))
            actual.append(_true_bin(f["FOS"].to_numpy(dtype=float)))
        predicted, actual = np.concatenate(predicted), np.concatenate(actual)
        inside = actual < config.N_BINS
        shift = np.abs(actual[inside] - predicted[inside])
        matrix = np.zeros((config.N_BINS, config.N_BINS + 2), dtype=int)
        np.add.at(matrix, (predicted, actual), 1)
        out[arm] = {"n": int(len(predicted)),
                    "stayed": float(np.mean(predicted == actual)),
                    "mean_abs_shift": float(shift.mean()) if len(shift) else 0.0,
                    "median_abs_shift": float(np.median(shift)) if len(shift) else 0.0,
                    "outside": float(np.mean(~inside)),
                    "matrix": matrix.tolist()}
    return out


ARMS_VS_B = ("C-log", "C-seq")
# Both sides of these rows come from replications.json, whose bins and
# train_counts are built with evaluate._zone_mask, i.e. [lo, hi). So the outer
# bin is "at or above 6" for the density and for the error alike. The generator's
# (lo, hi] convention governs the allocation and displacement rows instead.
OUTSIDE_SIDES = {"below 0.5": 0, "at or above 6": config.N_BINS + 1}


def _ordered(rows, family, test, arm) -> list[dict]:
    """The rows of one cell and arm, validated, in realization order."""
    chosen = [r for r in rows if r["family"] == family and r["test"] == test and r["arm"] == arm]
    where = f"{family}/{test}/{arm}"
    seen = [r["realization"] for r in chosen]
    if len(seen) != len(set(seen)):
        raise StructuralError(f"{where}: more than one row for a realization")
    check_realizations(seen, where)
    return sorted(chosen, key=lambda r: r["realization"])


REQUIRED_ARMS = ("B", *ARMS_VS_B)


def _require_arms(found, where: str) -> None:
    """B and both guided arms must be present.

    Deriving the arm list from whatever the artifact happens to contain would let
    a whole arm vanish in silence: the analysis would simply not mention C-seq.
    """
    missing = [arm for arm in REQUIRED_ARMS if arm not in set(found)]
    if missing:
        raise StructuralError(f"{where}: arms {missing} are missing")


def _check_by_arm(keyed, where: str) -> list[int]:
    """Every required arm must carry every confirmatory realization.

    Checking the union would pass when C-log is missing r5 but B and C-seq are
    not, and the omission would surface later as a KeyError.
    """
    _require_arms({a for a, _ in keyed}, where)
    for arm in REQUIRED_ARMS:
        check_realizations({r for a, r in keyed if a == arm}, f"{where}/{arm}")
    return sorted(range(1, config.REP_REALIZATIONS + 1))


def _bin_series(ordered_rows, index: int, key: str) -> np.ndarray:
    """One value per realization; `index` counts in-range bins, offset by the
    below-range bin at position 0."""
    return np.array([r["bins"][index + 1][key] for r in ordered_rows], dtype=float)


def error_rows(rep: dict) -> list[dict]:
    """Per (family, test, arm, in-range bin): the difference in mean log-RMSE
    against B and a descriptive Welch interval over the five realizations."""
    values, rows = _confirmatory(rep), []
    _require_arms({r["arm"] for r in values}, "errors")
    for family in config.REP_FAMILIES:
        for test in config.TEST_ROLES:
            for arm in ARMS_VS_B:
                arm_rows = _ordered(values, family, test, arm)
                base_rows = _ordered(values, family, test, "B")
                for b in range(config.N_BINS):
                    counts = set(_bin_series(arm_rows, b, "n").tolist()
                                 + _bin_series(base_rows, b, "n").tolist())
                    if len(counts) != 1:
                        raise StructuralError(
                            f"{test} bin {b}: test count differs between arms or "
                            f"realizations ({sorted(counts)}); it depends only on the test set")
                    test_n = int(counts.pop())
                    interval = evaluate.welch_interval(_bin_series(arm_rows, b, "log_rmse"),
                                                       _bin_series(base_rows, b, "log_rmse"),
                                                       config.REP_CI)
                    rows.append({"family": family, "test": test, "arm": arm, "bin": b,
                                 "test_n": test_n,
                                 "marked": test_n < config.STRUCTURAL_MIN_TEST_N, **interval})
    return rows


def density_diff_rows(rep: dict) -> list[dict]:
    """Per (arm, in-range bin): the difference in mean training count against B,
    with a descriptive Welch interval. Raw counts; no family, no test."""
    density, rows = density_rows(rep), []
    realizations = _check_by_arm(density, "density")
    for arm in ARMS_VS_B:
        for b in range(config.N_BINS):
            a = np.array([density[(arm, r)][b] for r in realizations], dtype=float)
            base = np.array([density[("B", r)][b] for r in realizations], dtype=float)
            rows.append({"arm": arm, "bin": b, **evaluate.welch_interval(a, base, config.REP_CI)})
    return rows


def outside_rows(rep: dict) -> list[dict]:
    """The two bins outside the selection range, on both sides of the question.

    `kind="density"` is the spillover itself: how many training points landed
    outside although nothing was allocated there. `kind="error"` is the test
    error in those bins. A table that reported only the second would describe
    tail error, not spillover.
    """
    values, rows = _confirmatory(rep), []
    outside = outside_density_rows(rep)
    realizations = _check_by_arm(outside, "outside density")
    for arm in ARMS_VS_B:
        for side, index in OUTSIDE_SIDES.items():
            column = 0 if index == 0 else 1
            a = np.array([outside[(arm, r)][column] for r in realizations], dtype=float)
            base = np.array([outside[("B", r)][column] for r in realizations], dtype=float)
            rows.append({"kind": "density", "arm": arm, "side": side,
                         **evaluate.welch_interval(a, base, config.REP_CI)})
            for family in config.REP_FAMILIES:
                for test in config.TEST_ROLES:
                    arm_rows = _ordered(values, family, test, arm)
                    base_rows = _ordered(values, family, test, "B")
                    ea = np.array([r["bins"][index]["log_rmse"] for r in arm_rows], dtype=float)
                    eb = np.array([r["bins"][index]["log_rmse"] for r in base_rows], dtype=float)
                    rows.append({"kind": "error", "family": family, "test": test, "arm": arm,
                                 "side": side, **evaluate.welch_interval(ea, eb, config.REP_CI)})
    return rows


def concordance(error_rows_, density_diff_rows_) -> list[dict]:
    """k of n bins where the two differences have opposite signs.

    The criterion is (delta density) x (delta log-RMSE) < 0, in both directions.
    The primary count covers the unmarked bins only, because no sentence may rest
    on a thin bin; the `_all` fields repeat it over all 15 as a descriptive
    sensitivity. Bins where either difference is exactly zero leave the
    denominator. A count, with no test attached.
    """
    density = {(r["arm"], r["bin"]): r["diff"] for r in density_diff_rows_}
    tally: dict[tuple[str, str, str], dict] = {}
    for row in error_rows_:
        cell = tally.setdefault((row["family"], row["test"], row["arm"]),
                                {"k": 0, "n": 0, "zero": 0, "k_all": 0, "n_all": 0,
                                 "zero_all": 0, "thin": 0})
        e, d = row["diff"], density[(row["arm"], row["bin"])]
        opposite = int(e * d < 0)
        if e == 0.0 or d == 0.0:
            cell["zero_all"] += 1
        else:
            cell["n_all"] += 1
            cell["k_all"] += opposite
        if row["marked"]:
            cell["thin"] += 1
            continue
        if e == 0.0 or d == 0.0:
            cell["zero"] += 1
        else:
            cell["n"] += 1
            cell["k"] += opposite
    return [{"family": f, "test": t, "arm": a, **cell} for (f, t, a), cell in tally.items()]


FIGURE_CAPTION = (
    "Per-bin training density and per-bin error, both as a difference against B, over the five "
    "confirmatory realizations. Every interval, on the density panel and on the error panels "
    "alike, visualizes variability between designs and constitutes no test; the fifteen intervals "
    "are a diagnostic, not fifteen tests. The pattern is consistent with density-driven "
    "relocation; it is not evidence that density caused it. The small numbers above the error "
    "panels are the test points per bin. Hollow markers, on the error panels, are bins with "
    "fewer than {min_n} of them; they are shown but carry no statement in the text. Both the "
    "densities and the errors in this figure are read from the replication metrics and use the "
    "half-open reporting zones; the allocation and displacement tables use the generator's "
    "binning instead, which differs only on an exact bin edge.")

PROCEDURE = (
    "Descriptive. No new hypothesis test or inferential claim is introduced; Welch intervals are "
    "reused solely to visualize between-design uncertainty at bin level. Confirmatory "
    "realizations r1..r5; r0 is reported apart and never averaged into a confirmatory mean. "
    "15 geometric bins inside the selection range; the two bins outside have no defined intended "
    "density and appear in the supplementary rows, as spillover and tail behaviour, never as "
    "truncation. Bins with fewer than STRUCTURAL_MIN_TEST_N test points are marked, never "
    "dropped, and the concordance count is primarily over the unmarked bins. Concordance is the "
    "count of bins with (delta density) x (delta log-RMSE) < 0; bins with either difference "
    "exactly zero leave the denominator. achieved - intended is the net distortion of the "
    "requested bin-count distribution, not the screener's prediction error; the displacement rows "
    "describe the screener on the points it selected, conditioned on selection, and their shift "
    "statistics cover only points whose true FOS stayed inside the selection range.")


def build() -> dict:
    rep = json.loads(config.replication_metrics_path().read_text())
    errors = error_rows(rep)
    densities = density_diff_rows(rep)
    everything = (0, *range(1, config.REP_REALIZATIONS + 1))
    return {
        "procedure": PROCEDURE,
        "figure_caption": FIGURE_CAPTION.format(min_n=config.STRUCTURAL_MIN_TEST_N),
        "bins": {"edges": EDGES().tolist(), "labels": bin_labels()},
        "min_test_n": config.STRUCTURAL_MIN_TEST_N,
        "confirmatory": list(range(1, config.REP_REALIZATIONS + 1)),
        "errors": errors,
        "densities": densities,
        "outside": outside_rows(rep),
        "concordance": concordance(errors, densities),
        "allocation": {str(r): allocation_rows(r) for r in everything},
        "batches": {str(r): batch_allocation_rows(r) for r in everything},
        "displacement": {str(r): displacement_rows(r) for r in everything},
    }


def run() -> None:
    payload = build()
    path = config.structural_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[structural] wrote {path.name}")
