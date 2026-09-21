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


def test_prior_counts_accepts_both_spellings_and_refuses_ambiguity():
    counts = [1, 2, 3]
    assert structural._prior_counts({"seed_counts": counts}).tolist() == counts
    assert structural._prior_counts({"trained_counts": counts}).tolist() == counts
    with pytest.raises(structural.StructuralError, match="prior counts"):
        structural._prior_counts({"seed_counts": counts, "trained_counts": counts})
    with pytest.raises(structural.StructuralError, match="prior counts"):
        structural._prior_counts({"added": counts})


def test_allocation_rows_use_the_stored_provenance(monkeypatch):
    seed_counts = [2] * config.N_BINS
    added = [4] * config.N_BINS
    achieved = list(added)
    achieved[0] -= 1                 # one point landed one bin higher than asked
    achieved[1] += 1
    monkeypatch.setattr(structural, "_sets_of",
                        lambda r: {"C-log": [("k", {"seed_counts": seed_counts, "added": added})]})
    monkeypatch.setattr(structural, "_achieved_counts",
                        lambda realization, label: np.array(achieved))

    rows = structural.allocation_rows(1)["C-log"]
    assert rows["seed"] == seed_counts and rows["intended"] == added
    assert rows["achieved"] == achieved
    assert rows["deviation"][0] == -1 and rows["deviation"][1] == 1
    assert rows["abs_deviation"] == 2


def test_allocation_deviation_is_zero_when_the_screener_was_exact(monkeypatch):
    added = [4] * config.N_BINS
    monkeypatch.setattr(structural, "_sets_of",
                        lambda r: {"C-log": [("k", {"trained_counts": [0] * config.N_BINS,
                                                    "added": added})]})
    monkeypatch.setattr(structural, "_achieved_counts",
                        lambda realization, label: np.array(added))
    assert structural.allocation_rows(1)["C-log"]["abs_deviation"] == 0


def test_sets_of_reads_r0_from_the_main_pipeline_roles():
    sets = structural._sets_of(0)
    assert [label for label, _ in sets["C-log"]] == ["guided-log"]
    assert [label for label, _ in sets["C-seq"]] == ["guided-seq-1", "guided-seq-2", "guided-seq-3"]


def test_allocation_rows_work_for_r0_including_the_seed_counts_spelling():
    """C-log at r0 stores its prior counts as seed_counts; a literal key raises."""
    rows = structural.allocation_rows(0)
    assert set(rows) == {"C-log", "C-seq"}
    assert len(rows["C-log"]["intended"]) == config.N_BINS
    # The budget is not a named constant; it is the metadata entry's own n.
    meta = json.loads(config.metadata_path().read_text())
    assert sum(rows["C-log"]["intended"]) == meta["datasets"]["guided-log"]["n"]
    assert sum(rows["C-seq"]["intended"]) == sum(
        meta["datasets"][role]["n"] for role in replicate.SEQ_ROLES_R0)


def test_stored_allocation_equals_the_generator_recomputation():
    """intended = deficit_allocation(prior counts, n), as build_design computed it."""
    meta = json.loads(config.metadata_path().read_text())
    for section in ("replications", "datasets"):
        for key, entry in meta[section].items():
            if "allocation" not in entry:
                continue
            prior = structural._prior_counts(entry["allocation"])
            recomputed = design.deficit_allocation(prior, entry["n"])
            assert recomputed.tolist() == entry["allocation"]["added"], f"{section}:{key}"


def test_selected_equals_intended_for_every_guided_set():
    """The generator refuses a thin bin and otherwise draws exactly allocation[b],
    so every stored design reproduces its own allocation."""
    meta = json.loads(config.metadata_path().read_text())
    for key, entry in meta["replications"].items():
        if entry["strategy"] == "lhs":
            continue
        frame = replicate.read_set_design(key)
        selected = np.bincount(frame["bin"].to_numpy(dtype=int), minlength=config.N_BINS)
        assert selected.tolist() == entry["allocation"]["added"], key
    for role in ("guided-log", *replicate.SEQ_ROLES_R0):
        frame = design.read_design(role)
        selected = np.bincount(frame["bin"].to_numpy(dtype=int), minlength=config.N_BINS)
        assert selected.tolist() == meta["datasets"][role]["allocation"]["added"], role
