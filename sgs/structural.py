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
