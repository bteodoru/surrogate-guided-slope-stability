import json

import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design, replicate, structural


def test_edges_are_the_declared_fifteen_bins():
    edges = structural.EDGES()
    assert len(edges) == config.N_BINS + 1
    assert edges[0] == config.SELECTION_RANGE[0] and edges[-1] == config.SELECTION_RANGE[1]
    ratios = edges[1:] / edges[:-1]
    assert np.allclose(ratios, ratios[0])          # geometric


def test_bin_labels_match_the_edges():
    labels = structural.bin_labels()
    assert len(labels) == config.N_BINS and labels[0].startswith("0.50")


def test_check_realizations_refuses_a_missing_one():
    assert structural.check_realizations({1, 2, 3, 4, 5}, "cell") == [1, 2, 3, 4, 5]
    with pytest.raises(structural.StructuralError, match=r"\[3\] are missing"):
        structural.check_realizations({1, 2, 4, 5}, "cell")


def _rep_stub(train_counts_by_arm, *, realizations=(1, 2, 3, 4, 5), below=0, above=0):
    """Shaped like the real artifact: 17 bins, the first and last outside the range."""
    values = []
    for family in config.REP_FAMILIES:
        for test in config.TEST_ROLES:
            for arm, counts in train_counts_by_arm.items():
                for r in realizations:
                    values.append({"family": family, "test": test, "arm": arm, "realization": r,
                                   "train_counts": [below, *counts, above],
                                   "bins": [{"n": 20, "log_rmse": 0.1} for _ in range(config.N_BINS + 2)]})
    return {"values": values}


def test_density_rows_are_read_from_the_artifact_and_are_cell_independent():
    counts = list(range(1, config.N_BINS + 1))
    rows = structural.density_rows(_rep_stub({"B": counts, "C-log": counts, "C-seq": counts}))
    assert rows[("B", 1)].tolist() == counts
    assert len(rows) == 15                          # three arms x five realizations, not sixty


def test_density_rows_refuse_a_cell_dependent_density():
    rep = _rep_stub({"B": list(range(1, config.N_BINS + 1))})
    rep["values"][0]["train_counts"][3] += 1
    with pytest.raises(structural.StructuralError, match="train_counts differ"):
        structural.density_rows(rep)


def test_outside_density_rows_read_the_two_outer_bins():
    rep = _rep_stub({"C-log": [1] * config.N_BINS}, below=45, above=19)
    assert structural.outside_density_rows(rep)[("C-log", 1)] == (45, 19)
