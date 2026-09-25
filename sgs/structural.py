"""Descriptive structural artifacts: per-bin training density beside per-bin
error, and the fidelity of the delivered allocation.

Design: docs/superpowers/specs/2026-09-20-structural-analysis-design.md.
Procedure in docs/predictions-and-outcomes.md, committed before this module
was committed, on replication results that were already computed; it is not a
preregistration, and the limit of its blinding is stated there. Nothing here computes a test or states an inferential claim; Welch
intervals are reused only to visualize variability between designs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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


def check_edges(rep: dict) -> None:
    """Every row of replications.json must carry this module's 17 bins.

    Its bins are indexed by position here; were the artifact built on other
    edges, every row would still line up and every label would be wrong. The
    open ends are stored as null.
    """
    edges = EDGES().tolist()
    expected = list(zip([None, *edges], [*edges, None]))

    def same(a, b):
        return a is None and b is None or (a is not None and b is not None and np.isclose(a, b))

    for row in rep["values"]:
        got = [(b["lo"], b["hi"]) for b in row["bins"]]
        if len(got) != len(expected) or not all(same(g[0], e[0]) and same(g[1], e[1])
                                                for g, e in zip(got, expected)):
            raise StructuralError(f"{row['arm']}/r{row['realization']}: bin edges differ from EDGES()")


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
# The allocation and displacement rows follow the generator's (lo, hi]: the
# outer bin is strictly above 6.
ALLOCATION_OUTSIDE = ("< 0.5", "> 6")


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
        roles = {"C-log": ["guided-log"], "C-logG": ["guided-log-g"],
                 "C-seq": list(replicate.SEQ_ROLES_R0)}
        return {arm: [(role, meta["datasets"][role]["allocation"]) for role in names]
                for arm, names in roles.items()}
    by = replicate.keys_by_strategy(realization)
    grouped = {"C-log": by["guided-log"], "C-logG": by["guided-log-g"], "C-seq": by["guided-seq"],
               "C-oracle": by["guided-oracle"], "C-logP": by["guided-log-p"]}
    return {arm: [(key, meta["replications"][key]["allocation"]) for key in names]
            for arm, names in grouped.items()}


def _budget_of(realization: int, label: str) -> int:
    """The budget n for a given realization and set label."""
    meta = _metadata()
    if realization == 0:
        return meta["datasets"][label]["n"]
    return meta["replications"][label]["n"]


def _verify_allocation(label: str, realization: int, allocation: dict) -> None:
    """Verify that allocation.added matches the deficit_allocation recomputation.

    Raises StructuralError if metadata or recomputation diverges.
    """
    prior = _prior_counts(allocation)
    budget = _budget_of(realization, label)
    recomputed = design.deficit_allocation(prior, budget)
    stored = np.asarray(allocation["added"], dtype=int)
    if not np.array_equal(stored, recomputed):
        raise StructuralError(
            f"{label}: allocation.added {stored.tolist()} does not match "
            f"deficit_allocation({prior.tolist()}, {budget}) = {recomputed.tolist()}")


def _solved(realization: int, label: str) -> pd.DataFrame:
    return data.load_role(label) if realization == 0 else replicate.load_set(label)


def _achieved_counts(realization: int, label: str) -> np.ndarray:
    """In-range bin counts of the true FOS the solver returned for one set."""
    frame = _solved(realization, label)
    return design.bin_counts(frame["FOS"].to_numpy(dtype=float), EDGES())


def _achieved_outside(realization: int, label: str) -> np.ndarray:
    """Points of one set whose true FOS left the range: [< 0.5, > 6], the
    generator's convention, so they and the in-range counts add up to the set."""
    true = _true_bin(_solved(realization, label)["FOS"].to_numpy(dtype=float))
    return np.array([int((true == config.N_BINS).sum()), int((true == config.N_BINS + 1).sum())])


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
        outside = np.zeros(2, dtype=int)
        for i, (label, allocation) in enumerate(sets):
            _verify_allocation(label, realization, allocation)
            if i == 0:
                seed = _prior_counts(allocation)
            intended = intended + np.asarray(allocation["added"], dtype=int)
            achieved = achieved + _achieved_counts(realization, label)
            outside = outside + _achieved_outside(realization, label)
        deviation = achieved - intended
        # abs_deviation stays over the 15 in-range bins, as declared; the points
        # that left the range, where nothing was intended, are reported beside it.
        out[arm] = {"seed": seed.tolist(), "intended": intended.tolist(),
                    "achieved": achieved.tolist(), "deviation": deviation.tolist(),
                    "abs_deviation": int(np.abs(deviation).sum()),
                    "achieved_outside": dict(zip(ALLOCATION_OUTSIDE, outside.tolist()))}
    return out


def allocation_gap_paired(realizations=range(1, config.REP_REALIZATIONS + 1)) -> list[dict]:
    """Stage 1 of the screener swap: the paired per-realization difference of
    sum |achieved - intended| between C-logG and C-log, GBM minus RF, over the 15
    in-range bins. Descriptive net distortion of the requested distribution, never
    prediction error, and never phrased as the screener causing it."""
    out = []
    for r in realizations:
        rows = allocation_rows(r)
        if "C-logG" not in rows or "C-log" not in rows:
            continue
        gbm, rf = rows["C-logG"]["abs_deviation"], rows["C-log"]["abs_deviation"]
        out.append({"realization": r, "gbm": gbm, "rf": rf, "delta": gbm - rf})
    return out


def shared_candidate_rows(realizations=(0, *range(1, config.REP_REALIZATIONS + 1))) -> list[dict]:
    """Per realization: how many of guided-log-g's points guided-log also drew.

    Both screeners choose from the same pool (guided_log_stream at r0; guided-log's
    replication draw stream, via config.REP_SEED_STREAM, at r1..) against the same
    intended allocation; only the screener differs, so overlap is expected but not
    guaranteed. A shared point is one within config.MIN_DISTANCE of a guided-log
    point in the normalized parameter space, the same criterion data.validate_sets
    uses for disjointness; realizations missing either set are skipped."""
    out = []
    for r in realizations:
        sets = _sets_of(r)
        log_labels = [label for label, _ in sets.get("C-log", [])]
        logg_labels = [label for label, _ in sets.get("C-logG", [])]
        if not log_labels or not logg_labels:
            continue
        log_frame = pd.concat([_solved(r, label) for label in log_labels], ignore_index=True)
        logg_frame = pd.concat([_solved(r, label) for label in logg_labels], ignore_index=True)
        distances = cKDTree(data.normalize(log_frame)).query(data.normalize(logg_frame), k=1)[0]
        shared = int((distances <= config.MIN_DISTANCE).sum())
        out.append({"realization": r, "n": int(len(logg_frame)), "shared": shared})
    return out


def pool_overlap_rows(realizations=range(1, config.REP_REALIZATIONS + 1)) -> list[dict]:
    """Per realization: how many rows C-oracle and C-logP both drew from their shared
    sub-pool, by pool row. They are alternatives, never combined; overlap is expected."""
    out = []
    for r in realizations:
        o = set(replicate.read_set_design(f"guided_oracle_r{r}")["pool_row"])
        p = set(replicate.read_set_design(f"guided_log_p_r{r}")["pool_row"])
        out.append({"realization": r, "n": len(o), "shared": len(o & p)})
    return out


def batch_allocation_rows(realization: int) -> list[dict]:
    """The C-seq batches, each on its own and cumulatively. The cumulative view
    shows whether retraining corrects a deviation or only moves it."""
    rows = []
    cum_intended = np.zeros(config.N_BINS, dtype=int)
    cum_achieved = np.zeros(config.N_BINS, dtype=int)
    for batch, (label, allocation) in enumerate(_sets_of(realization)["C-seq"], start=1):
        _verify_allocation(label, realization, allocation)
        intended = np.asarray(allocation["added"], dtype=int)
        achieved = _achieved_counts(realization, label)
        cum_intended, cum_achieved = cum_intended + intended, cum_achieved + achieved
        rows.append({"batch": batch, "seed": _prior_counts(allocation).tolist(),
                     "intended": intended.tolist(), "achieved": achieved.tolist(),
                     "cumulative_intended": cum_intended.tolist(),
                     "cumulative_achieved": cum_achieved.tolist(),
                     "achieved_outside": dict(zip(ALLOCATION_OUTSIDE,
                                                  _achieved_outside(realization, label).tolist()))})
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


ARMS_VS_B = ("C-log", "C-seq", "C-logG", "C-oracle", "C-logP")
# Arms whose design bins come from a screener's prediction. C-oracle bins on solved
# FOS: its matrix is the identity by construction and describes no screener.
SCREENED_ARMS = tuple(arm for arm in ARMS_VS_B if arm != "C-oracle")
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


def _require_arms(found, where: str, arms=REQUIRED_ARMS) -> None:
    """B and both guided arms (or the arms passed) must be present.

    Deriving the arm list from whatever the artifact happens to contain would let
    a whole arm vanish in silence: the analysis would simply not mention C-seq.
    """
    missing = [arm for arm in arms if arm not in set(found)]
    if missing:
        raise StructuralError(f"{where}: arms {missing} are missing")


def _check_by_arm(keyed, where: str, arms=REQUIRED_ARMS) -> list[int]:
    """Every required arm must carry every confirmatory realization.

    Checking the union would pass when C-log is missing r5 but B and C-seq are
    not, and the omission would surface later as a KeyError.
    """
    _require_arms({a for a, _ in keyed}, where, arms)
    for arm in arms:
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
                    counts = {r["bins"][index]["n"] for r in (*arm_rows, *base_rows)}
                    if len(counts) != 1:
                        raise StructuralError(
                            f"{test} {side}: test count differs between arms or realizations "
                            f"({sorted(counts)}); it depends only on the test set")
                    test_n = int(counts.pop())
                    rows.append({"kind": "error", "family": family, "test": test, "arm": arm,
                                 "side": side, "test_n": test_n,
                                 "marked": test_n < config.STRUCTURAL_MIN_TEST_N,
                                 **evaluate.welch_interval(ea, eb, config.REP_CI)})
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


# --- the C0 ablations (Amendment 1, docs/predictions-and-outcomes.md) ------------------------
# An extension decided after the review of Tasks 0-4, not part of the original
# declaration. None of it enters the main figure or the concordance count.

# C0 arm -> the C arm whose guided points it trains on, in the same realization.
ABLATIONS = {"C0-log": "C-log", "C0-seq": "C-seq", "C0-logG": "C-logG",
             "C0-oracle": "C-oracle", "C0-logP": "C-logP"}
ABLATION_ARMS = ("B", *ARMS_VS_B, *ABLATIONS)


def reporting_bin_labels() -> list[str]:
    """The 17 bins of replications.json, [lo, hi): below, the 15 in range, at or above."""
    return ["below 0.5", *bin_labels(), "at or above 6"]


def _declared_intervals() -> None:
    """The interval type is the one config declares, not a choice made here."""
    for c0, c in ABLATIONS.items():
        if (c0, "B") not in config.REP_WELCH:
            raise StructuralError(f"({c0!r}, 'B') is not declared in REP_WELCH")
        if (c, c0) not in config.REP_PAIRED:
            raise StructuralError(f"({c!r}, {c0!r}) is not declared in REP_PAIRED")


def _full_counts(rep: dict) -> dict[tuple[str, int], np.ndarray]:
    """(arm, realization) -> the 17-bin training counts, cell-independent."""
    out: dict[tuple[str, int], np.ndarray] = {}
    for row in _confirmatory(rep):
        key = (row["arm"], row["realization"])
        below, inside, above = _split(row["train_counts"])
        counts = np.array([below, *inside, above], dtype=int)
        if key in out and not np.array_equal(out[key], counts):
            raise StructuralError(f"train_counts differ between cells for {key}")
        out[key] = counts
    return out


def ablation_budgets(rep: dict) -> dict[str, int]:
    """Training-set size per arm, read from the artifact. It must not vary
    between realizations; the C0 rows are labelled by it."""
    counts, out = _full_counts(rep), {}
    for arm in ABLATION_ARMS:
        sizes = {int(v.sum()) for (a, _), v in counts.items() if a == arm}
        if len(sizes) != 1:
            raise StructuralError(f"{arm}: training-set size varies between realizations: {sorted(sizes)}")
        out[arm] = sizes.pop()
    return out


def ablation_error_rows(rep: dict) -> list[dict]:
    """Per (family, test, C0 arm, reporting bin): log-RMSE against B by a Welch
    interval and against the corresponding C by a paired one, over r1..r5.

    Both comparisons join arms trained on different budgets, and every row says
    so. The paired rows follow config.REP_PAIRED's order: C - C0.
    """
    _declared_intervals()
    values, rows = _confirmatory(rep), []
    _require_arms({r["arm"] for r in values}, "ablation errors", ABLATION_ARMS)
    labels = reporting_bin_labels()
    for family in config.REP_FAMILIES:
        for test in config.TEST_ROLES:
            base_rows = _ordered(values, family, test, "B")
            for c0, c in ABLATIONS.items():
                c0_rows = _ordered(values, family, test, c0)
                c_rows = _ordered(values, family, test, c)
                if [r["realization"] for r in c0_rows] != [r["realization"] for r in c_rows]:
                    raise StructuralError(f"{family}/{test}: {c} and {c0} cover different realizations")
                for index, label in enumerate(labels):
                    def series(ordered, key):
                        return np.array([r["bins"][index][key] for r in ordered], dtype=float)
                    counts = set(np.concatenate([series(base_rows, "n"), series(c0_rows, "n"),
                                                 series(c_rows, "n")]).tolist())
                    if len(counts) != 1:
                        raise StructuralError(
                            f"{test} reporting bin {index}: test count differs between arms or "
                            f"realizations ({sorted(counts)}); it depends only on the test set")
                    test_n = int(counts.pop())
                    common = {"family": family, "test": test, "arm": c0, "reporting_bin": index,
                              "label": label, "test_n": test_n,
                              "marked": test_n < config.STRUCTURAL_MIN_TEST_N, "budget_differs": True}
                    rows.append({**common, "first": c0, "second": "B", "interval": "welch",
                                 **evaluate.welch_interval(series(c0_rows, "log_rmse"),
                                                           series(base_rows, "log_rmse"), config.REP_CI)})
                    rows.append({**common, "first": c, "second": c0, "interval": "paired",
                                 **evaluate.paired_interval(series(c_rows, "log_rmse"),
                                                            series(c0_rows, "log_rmse"), config.REP_CI)})
    return rows


def ablation_fraction_rows(rep: dict) -> list[dict]:
    """Per (C0 arm, reporting bin): the training fraction n_bin / N_total, C0 - B,
    with a Welch interval over r1..r5. Fractions, because raw counts would carry
    the budget difference. They enter no concordance and no summary."""
    _declared_intervals()
    counts, rows = _full_counts(rep), []
    realizations = _check_by_arm(counts, "ablation fractions", ABLATION_ARMS)

    def fractions(arm):
        return np.array([counts[(arm, r)] / counts[(arm, r)].sum() for r in realizations], dtype=float)

    base = fractions("B")
    for c0 in ABLATIONS:
        a = fractions(c0)
        for index, label in enumerate(reporting_bin_labels()):
            rows.append({"arm": c0, "first": c0, "second": "B", "interval": "welch",
                         "budget_differs": True, "reporting_bin": index, "label": label,
                         "first_mean": float(a[:, index].mean()),
                         "second_mean": float(base[:, index].mean()),
                         **evaluate.welch_interval(a[:, index], base[:, index], config.REP_CI)})
    return rows


ABLATION_PROCEDURE = (
    "Amendment 1, 2026-09-21: an extension decided after the review of Tasks 0-4, not part of the "
    "original procedure. It was fixed on replication results already computed: the bin-level C0 "
    "results had existed since 075edb7. It is not a preregistration; that they had not been "
    "inspected is the author's statement and cannot be verified from the repository. Supplementary; it "
    "enters neither the main figure nor the concordance count. For C0-log and C0-seq, over r1..r5 "
    "and the 17 reporting bins of the replication metrics ([lo, hi)): log-RMSE against B by a Welch "
    "interval, and against the corresponding C by a paired interval (C - C0), because within a "
    "realization C0 trains on exactly the guided points C adds to the seed. Every row joins arms "
    "trained on different budgets and is labelled so; it is not a like-for-like comparison. The "
    "training distribution is compared as fractions n_bin / N_total, never raw counts. Thin bins are "
    "marked, never dropped. The intervals visualize between-design variability and constitute no "
    "test; an interval containing 0 is inconclusive, not a tie. r0 does not enter. C0-logG "
    "(2026-09-23) follows the same procedure, extended to the GBM-screener arm. C0-oracle and "
    "C0-logP follow the same procedure, on the pool-backed arms (Amendment 2, "
    "docs/predictions-and-outcomes.md). C-oracle, C-logP and their C0 ablations exist only in "
    "r1-r5; they have no r0.")

SHARED_ROWS_NOTE = (
    "C0-log, C0-seq, C0-logG, C0-oracle and C0-logP train on the same guided points as C-log, "
    "C-seq, C-logG, C-oracle and C-logP in every realization, so their allocation, batch and "
    "displacement rows are those of C, shared rather than duplicated.")

ORIGINAL_C0_NOTE = (
    "The original C0 (arm C0, role guided) exists only as r0 in the main pipeline. It was never "
    "replicated, so it has no replication interval, and it predates the log-deficit rule and stores "
    "no allocation provenance, so it has no allocation row. Nothing is computed for it here.")


ABLATION_CAPTION = (
    "Supplementary ablation. C0-log, C0-seq, C0-logG, C0-oracle and C0-logP train on the guided "
    "points alone, without the seed. Training-set sizes: {budgets}. Red: C0 minus B, 95% Welch "
    "interval. Purple: C minus the corresponding C0, 95% paired interval over the realizations, "
    "because within a realization C0 trains on exactly the guided points that C adds to the seed. "
    "The two differences are oriented oppositely; in both, a negative value means that the "
    "first-named arm has the smaller log-RMSE. Every comparison joins arms trained on different "
    "budgets and is not a like-for-like comparison. Means over the five confirmatory realizations, "
    "on the half-open reporting zones; grey shading marks the two bins outside the selection "
    "range, and hollow markers the bins with fewer than {min_n} test points. The intervals "
    "visualize variability between designs and constitute no test; an interval containing 0 is "
    "inconclusive, not a tie. C-oracle, C-logP and their C0 ablations exist only in r1-r5; they "
    "have no r0.")


def _budget_text(budgets: dict) -> str:
    return ", ".join(f"{arm} {n}" for arm, n in budgets.items())


def ablation(rep: dict) -> dict:
    budgets = ablation_budgets(rep)
    return {"procedure": ABLATION_PROCEDURE,
            "caption": ABLATION_CAPTION.format(budgets=_budget_text(budgets),
                                               min_n=config.STRUCTURAL_MIN_TEST_N),
            "labels": reporting_bin_labels(),
            "budgets": budgets,
            "errors": ablation_error_rows(rep),
            "fractions": ablation_fraction_rows(rep),
            "shared_rows": {**ABLATIONS, "note": SHARED_ROWS_NOTE},
            "original_c0": ORIGINAL_C0_NOTE}


FIGURE_CAPTION = (
    "Per-bin training density (raw counts; B, C-log, C-seq, C-logG, C-oracle and C-logP all train "
    "on {budget} points) and "
    "per-bin log-RMSE, each as the difference between an arm's mean over the five confirmatory "
    "realizations and B's, with 95% Welch intervals. A negative error difference means a smaller "
    "log-RMSE than B. Every interval, on the density panel and on the error panels alike, "
    "visualizes variability between designs and constitutes no test; read together they are a "
    "per-bin diagnostic, not a set of tests. The pattern is consistent with density-driven "
    "relocation; it is not evidence that density caused it. The small numbers above the upper "
    "error panels are the test points per bin. Hollow markers, on the error panels, are bins with "
    "fewer than {min_n} of them; they are shown but carry no statement in the text. Both the "
    "densities and the errors in this figure are read from the replication metrics and use the "
    "half-open reporting zones; the allocation and displacement tables use the generator's "
    "binning instead, which differs only on an exact bin edge. C-oracle, C-logP and their C0 "
    "ablations exist only in r1-r5; they have no r0.")

# The same sentence closes the allocation and displacement captions in report.
ORACLE_SENTENCE = ("C-oracle bins on the solved FOS and meets the intended allocation by "
                   "construction; its rows are a check, not a finding. What known bin membership "
                   "changes in test accuracy is read only from the paired contrast C-oracle "
                   "against C-logP (Table tab:rep:oracle), in this design.")

MATRIX_CAPTION = (
    "Predicted bin (rows) against true bin (columns) for the points the screener selected, C-log, "
    "C-seq, C-logG and C-logP, pooled over the confirmatory realizations r1-r5; r0 is excluded. "
    "Each row is normalized by the number of points selected into that predicted bin, given beside "
    "its label; a predicted bin into which nothing was selected is shown grey. Bins follow the "
    "generator, (lo, hi], so the outer columns are FOS < 0.5 and FOS > 6. The matrices describe the "
    "screener only on the points it selected, which were predicted inside the selection range and "
    "inside a deficit bin; they are not its accuracy over the candidate pool. Off-diagonal mass "
    "includes compensating errors, which leave achieved minus intended untouched. C-oracle has no "
    "matrix: it bins on the solved FOS, so every selected point stays in its bin by construction. "
    + ORACLE_SENTENCE)

BATCHES_CAPTION = (
    "C-seq batch by batch: the intended count per bin and the deviation, achieved minus intended, "
    "then the cumulative deviation after batches 2 and 3 (cum.); mean ± sd over r1-r5, r0 (exploratory) "
    "apart. Each batch reallocates against what the earlier batches actually returned; the "
    "cumulative columns show whether retraining corrects a deviation or moves it. The outside rows "
    "count points whose true FOS left the selection range, where nothing was intended (generator "
    "bins, outer bin FOS > 6). " + ORACLE_SENTENCE)

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
    "statistics cover only points whose true FOS stayed inside the selection range. C-oracle, "
    "C-logP and their C0 ablations exist only in r1-r5; they have no r0.")


def provenance(realizations) -> dict:
    """SHA-256 of every input: the replication metrics, the run metadata, and
    the design and final file of every guided set read for allocation and
    displacement. `report` refuses an artifact built from another
    replications.json."""
    sets = {}
    for r in realizations:
        for arm_sets in _sets_of(r).values():
            for label, _ in arm_sets:
                design_path = (config.design_path(label) if r == 0
                               else config.replication_path("designs", label))
                final_path = (config.final_path(label) if r == 0
                              else config.replication_path("final", label))
                sets[label] = {"realization": r, "design": data.sha256_file(design_path),
                               "final": data.sha256_file(final_path)}
    return {"replications_sha256": data.sha256_file(config.replication_metrics_path()),
            "run_metadata_sha256": data.sha256_file(config.metadata_path()),
            "sets_sha256": sets}


def build() -> dict:
    rep = json.loads(config.replication_metrics_path().read_text())
    check_edges(rep)
    errors = error_rows(rep)
    densities = density_diff_rows(rep)
    everything = (0, *range(1, config.REP_REALIZATIONS + 1))
    return {
        "procedure": PROCEDURE,
        "figure_caption": FIGURE_CAPTION.format(min_n=config.STRUCTURAL_MIN_TEST_N,
                                                budget=ablation_budgets(rep)["B"]),
        "matrix_caption": MATRIX_CAPTION,
        "batches_caption": BATCHES_CAPTION,
        "bins": {"edges": EDGES().tolist(), "labels": bin_labels()},
        "min_test_n": config.STRUCTURAL_MIN_TEST_N,
        "confirmatory": list(range(1, config.REP_REALIZATIONS + 1)),
        "errors": errors,
        "densities": densities,
        "outside": outside_rows(rep),
        "concordance": concordance(errors, densities),
        "allocation": {str(r): allocation_rows(r) for r in everything},
        "screener_swap": {"allocation_gap_paired": allocation_gap_paired(),
                          "shared_candidates": shared_candidate_rows()},
        "batches": {str(r): batch_allocation_rows(r) for r in everything},
        "displacement": {str(r): displacement_rows(r) for r in everything},
        "ablation": ablation(rep),
        "oracle": {"shared_rows": pool_overlap_rows()},
        "provenance": provenance(everything),
    }


def run() -> None:
    payload = build()
    path = config.structural_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[structural] wrote {path.name}")
