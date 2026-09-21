import json

import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design, evaluate, replicate, structural


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
    seed_counts = np.array([2] * config.N_BINS, dtype=int)
    # Use the correct allocation computed by deficit_allocation
    budget = 100
    added = design.deficit_allocation(seed_counts, budget).tolist()
    achieved = list(added)
    achieved[0] -= 1                 # one point landed one bin higher than asked
    achieved[1] += 1
    monkeypatch.setattr(structural, "_sets_of",
                        lambda r: {"C-log": [("k", {"seed_counts": seed_counts.tolist(), "added": added})]})
    monkeypatch.setattr(structural, "_achieved_counts",
                        lambda realization, label: np.array(achieved))
    monkeypatch.setattr(structural, "_achieved_outside",
                        lambda realization, label: np.array([3, 1]))
    # Mock _budget_of to return the expected budget for allocation verification
    monkeypatch.setattr(structural, "_budget_of", lambda r, l: budget)
    # Mock _metadata to support the allocation verification
    monkeypatch.setattr(structural, "_metadata",
                        lambda: {"replications": {"k": {"n": budget}}, "datasets": {}})

    rows = structural.allocation_rows(1)["C-log"]
    assert rows["achieved_outside"] == {"< 0.5": 3, "> 6": 1}
    assert rows["seed"] == seed_counts.tolist() and rows["intended"] == added
    assert rows["achieved"] == achieved
    assert rows["deviation"][0] == -1 and rows["deviation"][1] == 1
    assert rows["abs_deviation"] == 2


def test_allocation_deviation_is_zero_when_the_screener_was_exact(monkeypatch):
    seed_counts = np.array([0] * config.N_BINS, dtype=int)
    budget = 100
    added = design.deficit_allocation(seed_counts, budget).tolist()
    monkeypatch.setattr(structural, "_sets_of",
                        lambda r: {"C-log": [("k", {"trained_counts": seed_counts.tolist(),
                                                    "added": added})]})
    monkeypatch.setattr(structural, "_achieved_counts",
                        lambda realization, label: np.array(added))
    monkeypatch.setattr(structural, "_achieved_outside",
                        lambda realization, label: np.array([0, 0]))
    monkeypatch.setattr(structural, "_budget_of", lambda r, l: budget)
    monkeypatch.setattr(structural, "_metadata",
                        lambda: {"replications": {"k": {"n": budget}}, "datasets": {}})
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


def test_displacement_counts_shifts_and_refuses_misaligned_frames(monkeypatch):
    edges = structural.EDGES()
    mid = (edges[:-1] + edges[1:]) / 2
    params = list(config.PARAMS)
    design_frame = pd.DataFrame({p: np.arange(4, dtype=float) for p in params})
    design_frame["bin"] = [0, 1, 2, 3]
    design_frame["FOS_pred"] = mid[[0, 1, 2, 3]]
    final_frame = design_frame[params].copy()
    # stayed, moved up one, moved down two, landed above 6
    final_frame["FOS"] = [mid[0], mid[2], mid[0], 7.0]

    monkeypatch.setattr(structural, "_design_of", lambda realization, label: design_frame)
    monkeypatch.setattr(structural, "_solved", lambda realization, label: final_frame)
    monkeypatch.setattr(structural, "_sets_of", lambda r: {"C-log": [("k", {})], "C-seq": []})

    row = structural.displacement_rows(1)["C-log"]
    assert row["n"] == 4
    assert row["stayed"] == 0.25
    assert row["outside"] == 0.25
    # the out-of-range point is excluded from the shift statistics
    assert row["mean_abs_shift"] == pytest.approx((0 + 1 + 2) / 3)
    assert np.array(row["matrix"]).shape == (config.N_BINS, config.N_BINS + 2)
    assert np.array(row["matrix"]).sum() == 4

    misaligned = final_frame.copy()
    misaligned.loc[0, params[0]] += 1.0
    monkeypatch.setattr(structural, "_solved", lambda realization, label: misaligned)
    with pytest.raises(structural.StructuralError, match="not aligned"):
        structural.displacement_rows(1)


def test_true_bin_boundaries_follow_the_generator():
    """The spec's boundary test. _true_bin must agree with design.bin_counts,
    which is what produced both the design's bins and the achieved counts:
    bins are (lo, hi], the first closed, the last closed at 6."""
    edges = structural.EDGES()
    lo, hi = config.SELECTION_RANGE
    assert structural._true_bin(np.array([lo]))[0] == 0
    assert structural._true_bin(np.array([hi]))[0] == config.N_BINS - 1     # exactly 6 is in range
    assert structural._true_bin(np.array([lo - 1e-9]))[0] == config.N_BINS
    assert structural._true_bin(np.array([hi + 1e-9]))[0] == config.N_BINS + 1
    for b, edge in enumerate(edges[1:-1], start=1):
        assert structural._true_bin(np.array([edge]))[0] == b - 1           # (lo, hi]
    # and it agrees, bin for bin, with the counts the generator would produce
    sample = np.array([lo, *edges[1:-1], hi])
    counted = design.bin_counts(sample, edges)
    from collections import Counter
    tally = Counter(structural._true_bin(sample).tolist())
    assert [tally.get(b, 0) for b in range(config.N_BINS)] == counted.tolist()


def _error_stub(diff_by_arm, *, thin_bin=None, realizations=(1, 2, 3, 4, 5)):
    values = []
    for family in config.REP_FAMILIES:
        for test in config.TEST_ROLES:
            for arm, err in diff_by_arm.items():
                for r in realizations:
                    bins = [{"n": 20, "log_rmse": err} for _ in range(config.N_BINS + 2)]
                    if thin_bin is not None:
                        bins[thin_bin + 1] = {"n": 3, "log_rmse": err}
                    values.append({"family": family, "test": test, "arm": arm, "realization": r,
                                   "train_counts": [0, *([1] * config.N_BINS), 0], "bins": bins})
    return {"values": values}


def test_error_rows_compare_each_arm_against_B_without_pairing():
    stub = _error_stub({"B": 0.20, "C-log": 0.10, "C-seq": 0.15})
    rows = [r for r in structural.error_rows(stub)
            if r["arm"] == "C-log" and r["family"] == "rf" and r["test"] == config.TEST_ROLES[0]]
    assert len(rows) == config.N_BINS
    assert rows[0]["diff"] == pytest.approx(-0.10)
    assert rows[0]["ci"] == [pytest.approx(-0.10), pytest.approx(-0.10)]   # zero variance
    assert rows[0]["marked"] is False


def test_error_rows_mark_thin_bins_without_dropping_them():
    stub = _error_stub({"B": 0.2, "C-log": 0.1, "C-seq": 0.15}, thin_bin=0)
    rows = [r for r in structural.error_rows(stub) if r["arm"] == "C-log"]
    assert len({r["bin"] for r in rows}) == config.N_BINS      # none dropped
    assert all(r["marked"] for r in rows if r["bin"] == 0)


def test_error_rows_refuse_a_missing_arm():
    """A whole arm must not be able to disappear in silence."""
    rep = _error_stub({"B": 0.2, "C-log": 0.1})          # no C-seq at all
    with pytest.raises(structural.StructuralError, match=r"\['C-seq'\] are missing"):
        structural.error_rows(rep)


def test_error_rows_refuse_a_missing_realization():
    rep = _error_stub({"B": 0.2, "C-log": 0.1, "C-seq": 0.15}, realizations=(1, 2, 3, 4))
    with pytest.raises(structural.StructuralError, match=r"\[5\] are missing"):
        structural.error_rows(rep)


def test_error_rows_refuse_a_test_count_that_differs_between_arms():
    rep = _error_stub({"B": 0.2, "C-log": 0.1, "C-seq": 0.15})
    rep["values"][0]["bins"][1] = {"n": 99, "log_rmse": 0.2}
    with pytest.raises(structural.StructuralError, match="test count"):
        structural.error_rows(rep)


def test_concordance_separates_unmarked_bins_from_all_bins():
    errors = [{"family": "rf", "test": "t", "arm": "C-log", "bin": b, "diff": d, "marked": m}
              for b, (d, m) in enumerate([(-0.1, False), (+0.1, False), (+0.1, True), (0.0, False)])]
    densities = [{"arm": "C-log", "bin": b, "diff": d}
                 for b, d in enumerate([+5.0, -5.0, -5.0, +5.0])]
    got = structural.concordance(errors, densities)[0]
    assert (got["k"], got["n"], got["zero"]) == (2, 2, 1)       # unmarked only
    assert (got["k_all"], got["n_all"]) == (3, 3)               # the marked bin agrees too
    assert got["thin"] == 1


def test_build_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "structural_metrics_path", lambda: tmp_path / "structural.json")
    structural.run()
    first = (tmp_path / "structural.json").read_bytes()
    structural.run()
    assert (tmp_path / "structural.json").read_bytes() == first


# --- the C0 ablations (Amendment 1) -----------------------------------------------------------

def _ablation_stub(*, drop_arm=None, shuffle=False):
    """Errors vary by realization, so a Welch and a paired interval differ;
    C0 trains on half of B's budget with the same shape."""
    budgets = {"B": 2, "C-log": 2, "C-seq": 2, "C0-log": 1, "C0-seq": 1}
    level = {"B": 0.20, "C-log": 0.15, "C-seq": 0.16, "C0-log": 0.30, "C0-seq": 0.35}
    values = []
    for family in config.REP_FAMILIES:
        for test in config.TEST_ROLES:
            for arm in budgets:
                if arm == drop_arm:
                    continue
                for r in (1, 2, 3, 4, 5):
                    # C and C0 move together within a realization: a shared-points effect
                    err = level[arm] + 0.01 * r * (1.0 if arm != "B" else -0.5)
                    values.append({"family": family, "test": test, "arm": arm, "realization": r,
                                   "train_counts": [budgets[arm] * (1 + (b % 3)) for b in range(config.N_BINS + 2)],
                                   "bins": [{"n": 20, "log_rmse": err} for _ in range(config.N_BINS + 2)]})
    if shuffle:
        values = values[::-1]
    return {"values": values}


def test_ablation_covers_both_c0_variants_on_all_reporting_bins():
    rows = structural.ablation_error_rows(_ablation_stub())
    assert {r["arm"] for r in rows} == {"C0-log", "C0-seq"}
    per = len(config.REP_FAMILIES) * len(config.TEST_ROLES) * (config.N_BINS + 2) * 2
    for arm in ("C0-log", "C0-seq"):
        assert len([r for r in rows if r["arm"] == arm]) == per
    assert all(r["budget_differs"] for r in rows)


def test_ablation_uses_welch_against_B_and_paired_against_C():
    stub = _ablation_stub()
    rows = structural.ablation_error_rows(stub)
    b = np.array([0.20 - 0.005 * r for r in (1, 2, 3, 4, 5)])
    c0 = np.array([0.30 + 0.01 * r for r in (1, 2, 3, 4, 5)])
    c = np.array([0.15 + 0.01 * r for r in (1, 2, 3, 4, 5)])
    pick = lambda second: next(r for r in rows if r["arm"] == "C0-log" and r["second"] == second
                               and r["reporting_bin"] == 0)
    vs_b, vs_c = pick("B"), pick("C0-log")
    assert vs_b["interval"] == "welch" and vs_b["first"] == "C0-log"
    assert vs_b["ci"] == pytest.approx(evaluate.welch_interval(c0, b, config.REP_CI)["ci"])
    assert vs_c["interval"] == "paired" and vs_c["first"] == "C-log"          # REP_PAIRED order
    assert vs_c["ci"] == pytest.approx(evaluate.paired_interval(c, c0, config.REP_CI)["ci"])
    assert vs_c["df"] == 4
    assert vs_c["se"] == pytest.approx(0.0)       # the shared-points effect cancels in the pairs


def test_ablation_pairs_by_realization_whatever_the_row_order():
    ordered = structural.ablation_error_rows(_ablation_stub())
    shuffled = structural.ablation_error_rows(_ablation_stub(shuffle=True))
    assert ordered == shuffled


@pytest.mark.parametrize("arm", ["C0-log", "C0-seq"])
def test_ablation_refuses_a_missing_c0_variant(arm):
    with pytest.raises(structural.StructuralError, match=rf"\['{arm}'\] are missing"):
        structural.ablation_error_rows(_ablation_stub(drop_arm=arm))
    with pytest.raises(structural.StructuralError, match=rf"\['{arm}'\] are missing"):
        structural.ablation_fraction_rows(_ablation_stub(drop_arm=arm))


def test_ablation_refuses_an_undeclared_interval_type(monkeypatch):
    monkeypatch.setattr(config, "REP_PAIRED", (("C-log", "C0-log"),))
    with pytest.raises(structural.StructuralError, match="REP_PAIRED"):
        structural.ablation_error_rows(_ablation_stub())


def test_ablation_fractions_remove_the_budget():
    """Same shape, half the budget: raw counts differ, fractions do not."""
    rows = structural.ablation_fraction_rows(_ablation_stub())
    assert {r["arm"] for r in rows} == {"C0-log", "C0-seq"}
    assert all(r["interval"] == "welch" and r["second"] == "B" for r in rows)
    assert all(r["diff"] == pytest.approx(0.0) for r in rows)
    for arm in ("C0-log", "C0-seq"):
        assert sum(r["first_mean"] for r in rows if r["arm"] == arm) == pytest.approx(1.0)


def test_ablation_budgets_are_read_from_the_artifact():
    budgets = structural.ablation_budgets(_ablation_stub())
    assert budgets["C0-log"] * 2 == budgets["B"]


def test_c0_stays_out_of_the_main_figure_and_the_concordance():
    stub = _ablation_stub()
    errors = structural.error_rows(stub)
    densities = structural.density_diff_rows(stub)
    concordance = structural.concordance(errors, densities)
    for rows in (errors, densities, concordance):
        assert {r["arm"] for r in rows} == set(structural.ARMS_VS_B)


def test_built_artifact_carries_both_c0_variants_and_the_shared_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "structural_metrics_path", lambda: tmp_path / "structural.json")
    structural.run()
    out = json.loads((tmp_path / "structural.json").read_text())
    section = out["ablation"]
    assert {r["arm"] for r in section["errors"]} == {"C0-log", "C0-seq"}
    assert {r["arm"] for r in section["fractions"]} == {"C0-log", "C0-seq"}
    assert {r["interval"] for r in section["errors"] if r["second"] == "B"} == {"welch"}
    assert {r["interval"] for r in section["errors"] if r["second"] != "B"} == {"paired"}
    assert section["shared_rows"]["C0-log"] == "C-log" and section["shared_rows"]["C0-seq"] == "C-seq"
    assert "only as r0" in section["original_c0"]
    assert {r["arm"] for r in out["concordance"]} == set(structural.ARMS_VS_B)


# --- review fixes: r0 contamination, outside rows, edges, provenance ----------------------------

def _with_r0(rep, value=9.9, count=999):
    """Append r0 rows with absurd values for every arm and cell already present."""
    extra = []
    for row in rep["values"]:
        if row["realization"] != 1:
            continue
        clone = json.loads(json.dumps(row))
        clone["realization"] = 0
        clone["train_counts"] = [count] * len(clone["train_counts"])
        for b in clone["bins"]:
            b["log_rmse"] = value
        extra.append(clone)
    return {"values": rep["values"] + extra}


def test_r0_rows_change_no_confirmatory_output():
    clean = _ablation_stub()
    dirty = _with_r0(_ablation_stub())
    for fn in (structural.error_rows, structural.density_diff_rows, structural.outside_rows,
               structural.ablation_error_rows, structural.ablation_fraction_rows,
               structural.ablation_budgets):
        assert fn(dirty) == fn(clean), fn.__name__


def test_allocation_rows_report_the_points_that_left_the_range(monkeypatch):
    """Achieved inside plus achieved outside is the whole set."""
    seed_counts = np.array([0] * config.N_BINS, dtype=int)
    budget = 20 * config.N_BINS
    added = design.deficit_allocation(seed_counts, budget).tolist()
    achieved = [16] * config.N_BINS
    monkeypatch.setattr(structural, "_sets_of",
                        lambda r: {"C-log": [("k", {"trained_counts": seed_counts.tolist(),
                                                    "added": added})]})
    monkeypatch.setattr(structural, "_achieved_counts", lambda realization, label: np.array(achieved))
    monkeypatch.setattr(structural, "_achieved_outside", lambda realization, label: np.array([45, 15]))
    monkeypatch.setattr(structural, "_budget_of", lambda r, l: budget)
    monkeypatch.setattr(structural, "_metadata",
                        lambda: {"replications": {"k": {"n": budget}}, "datasets": {}})
    row = structural.allocation_rows(1)["C-log"]
    assert sum(row["achieved"]) + sum(row["achieved_outside"].values()) == sum(row["intended"])


def test_achieved_outside_uses_the_generator_convention(monkeypatch):
    lo, hi = config.SELECTION_RANGE
    frame = pd.DataFrame({"FOS": [lo - 1e-9, lo, hi, hi + 1e-9, 7.0]})
    monkeypatch.setattr(structural, "_solved", lambda realization, label: frame)
    assert structural._achieved_outside(1, "k").tolist() == [1, 2]      # 6 exactly is inside


def test_outside_error_rows_carry_test_n_and_mark_thin_bins():
    rep = _ablation_stub()
    for row in rep["values"]:
        row["bins"][0]["n"] = 3                         # below 0.5: thin
    rows = [r for r in structural.outside_rows(rep) if r["kind"] == "error"]
    below = [r for r in rows if r["side"] == "below 0.5"]
    above = [r for r in rows if r["side"] == "at or above 6"]
    assert all(r["test_n"] == 3 and r["marked"] for r in below)
    assert all(r["test_n"] == 20 and not r["marked"] for r in above)


def _with_edges(rep, edges):
    bounds = list(zip([None, *edges], [*edges, None]))
    for row in rep["values"]:
        for b, (lo, hi) in zip(row["bins"], bounds):
            b["lo"], b["hi"] = lo, hi
    return rep


def test_check_edges_accepts_the_module_bins_and_refuses_others():
    edges = structural.EDGES().tolist()
    structural.check_edges(_with_edges(_ablation_stub(), edges))
    shifted = [edges[0], edges[1] * 1.01, *edges[2:]]
    with pytest.raises(structural.StructuralError, match="bin edges differ"):
        structural.check_edges(_with_edges(_ablation_stub(), shifted))


def test_build_records_the_provenance_of_its_inputs():
    out = structural.build()
    assert out["provenance"]["replications_sha256"] == data.sha256_file(config.replication_metrics_path())
    assert out["provenance"]["run_metadata_sha256"] == data.sha256_file(config.metadata_path())
    assert "guided-log" in out["provenance"]["sets_sha256"]
    assert len(out["provenance"]["sets_sha256"]) == 4 * (config.REP_REALIZATIONS + 1)


def test_captions_state_the_budgets_and_the_interval_level():
    out = structural.build()
    assert "95% Welch" in out["figure_caption"] and "raw counts" in out["figure_caption"]
    caption = out["ablation"]["caption"]
    assert "C0-log 300" in caption and "B 600" in caption
    assert "Amendment" not in caption and "Task" not in caption      # paper text, not process


def test_allocation_rows_verifies_allocation_against_recomputation(monkeypatch):
    """allocation_rows checks that allocation.added == deficit_allocation(prior, budget)."""
    # This should pass: allocation is correct
    rows = structural.allocation_rows(0)
    assert "C-log" in rows

    # Corrupt the allocation in metadata and verify it raises
    original_metadata = structural._metadata()
    corrupt_meta = json.loads(json.dumps(original_metadata))  # deep copy
    corrupt_meta["datasets"]["guided-log"]["allocation"]["added"][0] = 999
    monkeypatch.setattr(structural, "_metadata", lambda: corrupt_meta)
    with pytest.raises(structural.StructuralError, match="allocation.added.*does not match"):
        structural.allocation_rows(0)


def test_batch_allocation_rows_verifies_allocation_against_recomputation(monkeypatch):
    """batch_allocation_rows checks that each batch's allocation is correct."""
    original_metadata = structural._metadata()
    batch_key = [k for k in original_metadata["replications"].keys() if k.startswith("guided_seq_r1")]
    if batch_key:
        batch_key = batch_key[0]
        # This should pass: allocation is correct
        rows = structural.batch_allocation_rows(1)
        assert len(rows) > 0

        # Corrupt the allocation and verify it raises
        corrupt_meta = json.loads(json.dumps(original_metadata))  # deep copy
        corrupt_meta["replications"][batch_key]["allocation"]["added"][0] = 999
        monkeypatch.setattr(structural, "_metadata", lambda: corrupt_meta)
        with pytest.raises(structural.StructuralError, match="allocation.added.*does not match"):
            structural.batch_allocation_rows(1)


def test_build_includes_matrix_and_batches_captions():
    """build() includes the matrix_caption and batches_caption keys."""
    out = structural.build()
    assert "matrix_caption" in out
    assert "batches_caption" in out
    assert "{arm}" in out["matrix_caption"]  # template placeholder
    assert "oracle" in out["batches_caption"]  # should mention oracle limitation
