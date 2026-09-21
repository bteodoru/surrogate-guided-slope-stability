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
