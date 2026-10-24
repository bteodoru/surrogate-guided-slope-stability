import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.container import BarContainer

from sgs import config, data, evaluate, report
from tests.helpers import EVAL_KWARGS, stub_fits, stub_test


def _write_metrics():
    result = evaluate.evaluate({**stub_fits("rf"), **stub_fits("gbm")}, stub_test(), **EVAL_KWARGS)
    path = config.metrics_path("test-natural", "log-rmse")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json()))
    return result.to_json()


def test_booktabs_structure_and_escaping():
    tex = report.booktabs(pd.DataFrame({"arm": ["C_0"], "RMSE": [0.12345], "n": [None]}),
                          caption="Demo.", label="tab:demo")
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"C\_0 & 0.123 & --" in tex and r"\label{tab:demo}" in tex


def test_booktabs_wraps_a_column_ragged_right():
    tex = report.booktabs(pd.DataFrame({"arm": ["B"], "how": ["Latin hypercube"], "n": [600]}),
                          caption="Demo.", label="tab:demo", left=2, wrap={"how": "3cm"})
    assert r"\begin{tabular}{llr}" in tex
    assert r"B & \parbox[t]{3cm}{\raggedright Latin hypercube\strut} & 600 \\" in tex


def test_arms_table_lists_every_arm_and_every_ablation():
    from sgs import replicate
    frame = report.budget_frame()
    expected = [*config.ARMS, *replicate.POOL_ARMS]
    assert sorted(frame["arm"]) == sorted(report.arm_label(a) for a in expected)
    assert len(frame) == len(set(frame["arm"]))
    ablations = frame.loc[frame["group"] == "Ablations", "arm"].tolist()
    assert sorted(ablations) == sorted(report.arm_label(a) for a in expected if a.startswith("C0"))
    assert list(dict.fromkeys(frame["group"])) == ["Budget arms", "Pool arms", "Ablations"]


def test_arms_table_budgets_match_config():
    from sgs import replicate
    rows = report.budget_frame().set_index("arm")
    for arm, spec in config.ARMS.items():
        assert rows.at[report.arm_label(arm), "analyses"] == spec.budget
        if spec.ablation:
            assert pd.isna(rows.at[report.arm_label(arm), "analyses"])
    for arm in replicate.POOL_ARMS:
        cell = rows.at[report.arm_label(arm), "analyses"]
        if report.is_ablation(arm):
            assert pd.isna(cell)
        else:
            assert cell == f"{config.ROLES['seed'].n + config.REP_EXTENSION}†"


def test_arms_table_realizations():
    from sgs import replicate
    new = tuple(range(1, config.REP_REALIZATIONS + 1))
    assert report.arm_realizations("A") == (0,)
    assert report.arm_realizations("C") == report.arm_realizations("C0") == (0,)
    for arm in replicate.POOL_ARMS:
        assert report.arm_realizations(arm) == new
    for arm in replicate.REP_ARMS:
        if arm in config.ARMS and arm != "A":
            assert report.arm_realizations(arm) == (0, *new)


def test_arms_table_realizations_match_the_replication_metrics():
    path = config.replication_metrics_path()
    if not path.exists():
        pytest.skip("no replication metrics")
    values = json.loads(path.read_text())["values"]
    for arm in {v["arm"] for v in values}:
        assert {v["realization"] for v in values if v["arm"] == arm} == set(report.arm_realizations(arm))


def test_every_arm_needs_a_description(monkeypatch):
    monkeypatch.delitem(report.ARM_TEXT, "C-logP")
    with pytest.raises(KeyError, match="C-logP"):
        report.budget_frame()


def test_comparison_frame_formats_tiny_pvalues():
    metrics = {"comparisons": [
        {"family": "rf", "subset": "global", "scale": "fos", "first": "C", "second": "C0",
         "wilcoxon": {"alternative": "less", "n": 10, "median_d": -0.1, "p_d_lt_0": 0.9,
                      "pvalue": 0.00005},
         "bootstrap": {"delta_rmse": -0.2, "ci": [-0.3, -0.1]}},
        {"family": "rf", "subset": "global", "scale": "fos", "first": "B", "second": "A",
         "wilcoxon": {"alternative": "less", "n": 10, "median_d": -0.05, "p_d_lt_0": 0.7,
                      "pvalue": 0.042},
         "bootstrap": {"delta_rmse": -0.1, "ci": [-0.2, 0.0]}},
    ]}
    frame = report.comparison_frame(metrics)
    assert frame["p"].tolist() == ["<0.001", "0.042"]


def test_every_results_table_includes_the_ablation(root):
    metrics = _write_metrics()
    assert "C₀" in report.global_frame(metrics)["arm"].tolist()
    assert "C₀" in report.zone_frame(metrics, "rf").columns
    assert "C₀" in report.conformal_frame(metrics)["arm"].tolist()
    assert "C vs C0" in report.comparison_frame(metrics)["comparison"].tolist()
    assert {"log-RMSE", "minimax log"} <= set(report.global_frame(metrics).columns)


def test_learning_curve_figure_reads_the_rows():
    rows = [{"family": family, "curve": curve, "budget": b, "test": "test-natural", "n_repeats": 2,
             "rmse_mean": 1.0 / b, "rmse_std": 0.01, "rmse_selection_mean": 0.5 / b,
             "rmse_selection_std": 0.01, "log_rmse_mean": 0.1 / b, "log_rmse_std": 0.001,
             "log_rmse_selection_mean": 0.05 / b, "log_rmse_selection_std": 0.001,
             "zones": [{"zone": "FOS < 1", "n": 5, "rmse_mean": 0.2, "rmse_std": 0.0},
                       {"zone": "FOS ≥ 6", "n": 3, "rmse_mean": 2.0 / b, "rmse_std": 0.1}]}
            for family in ("rf", "gbm") for curve in ("lhs", "guided") for b in (100, 200)]
    fig = report.fig_learning_curve({"rows": rows}, "test-natural", "gbm")
    assert len(fig.axes) == 4 and all(len(ax.lines) == 2 for ax in fig.axes)
    assert "GBM" in fig.get_suptitle() and "FOS ≥ 6" in fig.axes[3].get_title()


def test_run_writes_tables_and_figures(root):
    _write_metrics()
    report.run()
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    assert {"budget.tex", "test-natural__log-rmse__global.tex", "test-natural__log-rmse__comparisons.tex",
            "test-natural__log-rmse__conformal.tex", "test-natural__log-rmse__zones_rf.tex",
            "test-natural__log-rmse__zones_log_rf.tex", "test-natural__log-rmse__subsets.tex"} <= tables
    for name in ("test-natural__log-rmse__zones_gbm.pdf", "test-natural__log-rmse__zones_log_gbm.pdf"):
        assert (config.figures_dir() / name).read_bytes()[:4] == b"%PDF"


def test_subset_frame_has_every_arm_and_both_scales(root):
    metrics = _write_metrics()
    frame = report.subset_frame(metrics)
    assert len(frame) == len(config.FAMILIES) * len(config.ARMS)
    assert {"log-RMSE decision-range", "RMSE decision-range", "log-RMSE all"} <= set(frame.columns)
    assert "C₀" in frame["arm"].tolist()


def test_scoring_frames_compare_the_selection_criteria():
    log = evaluate.evaluate({**stub_fits("rf"), **stub_fits("gbm")}, stub_test(), **EVAL_KWARGS).to_json()
    fos = evaluate.evaluate({**stub_fits("rf", scoring="fos-rmse"), **stub_fits("gbm", scoring="fos-rmse")},
                            stub_test(), **EVAL_KWARGS).to_json()
    frame = report.scoring_frame({"log-rmse": log, "fos-rmse": fos})
    assert len(frame) == len(config.FAMILIES) * len(config.ARMS)
    assert "C₀" in frame["arm"].tolist()
    for scoring in ("log-rmse", "fos-rmse"):
        for family in config.FAMILIES:
            ranks = frame.loc[frame["family"] == family, f"rank {scoring}"]
            assert sorted(ranks) == list(range(1, len(config.ARMS) + 1))
    for family in config.FAMILIES:
        params = report.scoring_params_frame({"log-rmse": log, "fos-rmse": fos}, family)
        names = sorted(log["arms"][family]["A"]["best_params"])
        # one column per hyper-parameter under each scoring, the arm first
        assert list(params.columns) == ["arm", *((s, report.PARAM_HEADERS.get(n, n))
                                                 for s in ("log-rmse", "fos-rmse") for n in names)]
        first = params.iloc[0]
        for n in names:
            assert first[("log-rmse", report.PARAM_HEADERS.get(n, n))] == \
                report._param(log["arms"][family]["A"]["best_params"][n])


def test_run_writes_scoring_tables_when_both_scorings_are_active(root, monkeypatch):
    monkeypatch.setattr(config, "ACTIVE_SCORINGS", ("log-rmse", "fos-rmse"))
    for scoring in config.ACTIVE_SCORINGS:
        fits = {**stub_fits("rf", scoring=scoring), **stub_fits("gbm", scoring=scoring)}
        path = config.metrics_path("test-natural", scoring)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS).to_json()))
    report.run()
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    assert {"test-natural__scorings.tex", "scorings_params.tex"} <= tables
    assert "test-stratified__scorings.tex" not in tables


def _write_replications(root):
    from tests.test_evaluate import REP_KW, _fake_units
    tests = {"test-natural": stub_test(80, 3), "test-stratified": stub_test(50, 4, name="test-stratified")}
    kw = {**REP_KW, "bins": (0.5, 1.0, 2.0, 6.0)}
    rep = evaluate.replication_metrics(_fake_units(tests), tests, **kw)
    path = config.replication_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep))
    return rep


def test_replication_frames_show_every_value_and_the_ablations(root):
    rep = _write_replications(root)
    frame = report.replication_value_frame(rep, "test-natural", "decision-range")
    assert {"r0 (exploratory)", "r1", "r2", "r3", "mean", "sd"} <= set(frame.columns)
    assert {"C0-log", "C0-seq"} <= set(frame["arm"])
    comp = report.replication_comparison_frame(rep, "test-natural")
    assert {"CI low", "CI high", "P(first<second)", "method", "scale"} <= set(comp.columns)
    assert set(comp["measure"]) <= {"decision-range", "0.5–1", "1–2", "selection-range", "all"}
    assert len(report.replication_interaction_frame(rep)) > 0


def test_run_writes_replication_outputs_when_present(root):
    _write_metrics()
    _write_replications(root)
    report.run()
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    assert {"replications__test-natural__values.tex", "replications__test-natural__comparisons.tex",
            "replications__interactions.tex"} <= tables
    assert (config.figures_dir() / "replications__test-natural__gbm.pdf").read_bytes()[:4] == b"%PDF"


def test_replication_bin_figure_has_ablations_and_no_warning(root, recwarn):
    rep = _write_replications(root)
    fig = report.fig_replication_bins(rep, "test-natural", "gbm")
    labels = {line.get_label() for line in fig.axes[0].lines}
    assert {"C0-log", "C0-seq"} <= labels
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --- structural (Task 5) -----------------------------------------------------

def _stub_with_pool_arms(by_realization, pool_row):
    """`by_realization` (a per-r0/r1/r2 dict of arm rows, like structural's own
    "allocation"/"displacement" sections) with `pool_row` (C-oracle's and C-logP's
    row) merged into r1 and r2 only -- they have no r0, as in the real artifact."""
    return {r: {**arms, **(pool_row if r != "0" else {})} for r, arms in by_realization.items()}


def _struct_stub():
    labels = ["0.50-1.00", "1.00-6.00"]
    errors = [{"family": f, "test": t, "arm": a, "bin": b, "diff": -0.01, "ci": [-0.02, 0.0],
               "test_n": 20 if b == 0 else 3, "marked": b == 1}
              for f in config.REP_FAMILIES for t in config.TEST_ROLES
              for a in ("C-log", "C-seq", "C-logG") for b in (0, 1)]
    return {
        "bins": {"edges": [0.5, 1.0, 6.0], "labels": labels},
        "min_test_n": 10, "confirmatory": [1, 2],
        "figure_caption": "caption text",
        "matrix_caption": "matrix for {arm}",
        "batches_caption": "batches caption text",
        "densities": [{"arm": a, "bin": b, "diff": 3.0, "ci": [1.0, 5.0]}
                      for a in ("C-log", "C-seq", "C-logG") for b in (0, 1)],
        "errors": errors,
        "outside": [{"kind": "density", "arm": "C-log", "side": "below 0.5",
                     "diff": 20.0, "ci": [10.0, 30.0]}],
        "concordance": [],
        # r1 and r2 differ, so there is a spread; r0 is absurd, so a leak shows.
        # C-oracle and C-logP (the pool arms) have no r0, as in the real artifact.
        "allocation": _stub_with_pool_arms(
            {r: {arm: {"seed": [2, 2], "intended": [4, 4],
                      "achieved": [3 + k, 5 - k], "deviation": [-1 + k, 1 - k],
                      "abs_deviation": 2 * abs(1 - k),
                      "achieved_outside": {"< 0.5": k, "> 6": 0}}
                 for arm in ("C-log", "C-seq", "C-logG")}
             for r, k in (("0", 90), ("1", 0), ("2", 1))},
            {arm: {"seed": [0, 0], "intended": [4, 4], "achieved": [4, 4], "deviation": [0, 0],
                  "abs_deviation": 0, "achieved_outside": {"< 0.5": 0, "> 6": 0}}
             for arm in ("C-oracle", "C-logP")}),
        "batches": {r: [{"batch": j, "seed": [0, 0], "intended": [2, 2], "achieved": [1 + k, 3],
                         "cumulative_intended": [2 * j, 2 * j], "cumulative_achieved": [j + j * k, 3 * j],
                         "achieved_outside": {"< 0.5": k, "> 6": 0}}
                        for j in (1, 2)]
                    for r, k in (("0", 90), ("1", 0), ("2", 1))},
        "displacement": _stub_with_pool_arms(
            {r: {"C-log": {"n": 4, "stayed": 0.5, "mean_abs_shift": 0.5,
                          "median_abs_shift": 0.0, "outside": 0.25,
                          "matrix": [[2, 0, 1, 1], [1, 0, 1, 0]]},
                "C-seq": {"n": 4, "stayed": 0.5, "mean_abs_shift": 0.5,
                         "median_abs_shift": 0.0, "outside": 0.25,
                         "matrix": [[1, 1, 0, 2], [0, 1, 1, 0]]},
                "C-logG": {"n": 4, "stayed": 0.5, "mean_abs_shift": 0.5,
                          "median_abs_shift": 0.0, "outside": 0.25,
                          "matrix": [[2, 0, 1, 1], [1, 0, 1, 0]]}}
             for r in ("0", "1", "2")},
            {"C-logP": {"n": 4, "stayed": 0.5, "mean_abs_shift": 0.5,
                       "median_abs_shift": 0.0, "outside": 0.25,
                       "matrix": [[1, 1, 0, 2], [0, 1, 1, 0]]},
             "C-oracle": {"n": 4, "stayed": 1.0, "mean_abs_shift": 0.0,
                         "median_abs_shift": 0.0, "outside": 0.0,
                         "matrix": [[0, 4, 0, 0], [0, 0, 4, 0]]}}),
        "screener_swap": {
            "allocation_gap_paired": [{"realization": 1, "gbm": 40, "rf": 55, "delta": -15},
                                      {"realization": 2, "gbm": 30, "rf": 50, "delta": -20}],
            "shared_candidates": [{"realization": 0, "n": 300, "shared": 300},
                                  {"realization": 1, "n": 300, "shared": 210},
                                  {"realization": 2, "n": 300, "shared": 198}],
        },
    }


def _ablation_stub():
    """Shaped like structural.ablation(): two C0 arms, three reporting bins
    (below range, one in range, above range), one welch and one paired row each."""
    labels = ["below 0.5", "1.00-2.00", "at or above 6"]
    ablations = {"C0-log": "C-log", "C0-seq": "C-seq"}
    errors = []
    for f in config.REP_FAMILIES:
        for t in config.TEST_ROLES:
            for c0, c in ablations.items():
                for i, label in enumerate(labels):
                    n = 20 if i != 1 else 3
                    common = {"family": f, "test": t, "arm": c0, "reporting_bin": i, "label": label,
                              "test_n": n, "marked": n < 10, "budget_differs": True}
                    errors.append({**common, "first": c0, "second": "B", "interval": "welch",
                                   "diff": -0.02, "ci": [-0.05, 0.01]})
                    errors.append({**common, "first": c, "second": c0, "interval": "paired",
                                   "diff": 0.03, "ci": [0.0, 0.06]})
    fractions = [{"arm": c0, "first": c0, "second": "B", "interval": "welch",
                  "budget_differs": True, "reporting_bin": i, "label": label,
                  "first_mean": 0.2, "second_mean": 0.1, "diff": 0.1, "ci": [0.05, 0.15]}
                 for c0 in ablations for i, label in enumerate(labels)]
    return {
        "procedure": "ablation procedure text",
        "labels": labels,
        "budgets": {"B": 600, "C-log": 600, "C-seq": 600, "C0-log": 300, "C0-seq": 300},
        "errors": errors,
        "fractions": fractions,
        "shared_rows": {"C0-log": "C-log", "C0-seq": "C-seq", "note": "shared rows note text"},
        "original_c0": "original c0 note text",
    }


def _struct_with_ablation():
    struct = _struct_stub()
    struct["ablation"] = _ablation_stub()
    return struct


def test_structural_figure_has_one_density_panel_above_four_error_panels():
    fig = report.fig_structural(_struct_stub())
    assert len(fig.axes) == 5
    assert all(any(line.get_ydata()[0] == 0 for line in ax.lines) for ax in fig.axes)


def test_structural_figure_uses_one_colour_per_arm_for_bars_and_markers():
    """errorbar(fmt="none") and scatter do not advance the property cycle in step,
    so an implicit colour gives the second arm's bars the first arm's colour and
    turns its marked markers grey. Six collections, three colours (one per arm), is
    the fix."""
    assert len({report.STRUCTURAL_COLORS[a] for a in ("C-log", "C-seq", "C-logG")}) == 3
    fig = report.fig_structural(_struct_stub())
    error_panel = fig.axes[1]          # bin 0 solid, bin 1 marked, for every arm
    colours = {tuple(round(float(v), 3) for v in coll.get_edgecolor()[0])
               for coll in error_panel.collections if len(coll.get_edgecolor())}
    assert len(colours) == 3, "solid and hollow markers of one arm must share its colour"


def test_structural_matrix_figure_is_row_normalized():
    fig = report.fig_structural_matrix(_struct_stub(), "C-log")
    image = fig.axes[0].images[0]
    shares = image.get_array()
    row_totals = shares.sum(axis=1)
    # the stub's two realizations (r1, r2) both have row totals > 0 for every row
    assert np.allclose(row_totals, 1.0)


def test_structural_allocation_frame_keeps_r0_apart_and_reports_a_spread():
    frame = report.structural_allocation_frame(_struct_stub(), "C-log")
    columns = [c.lower() for c in frame.columns]
    assert "selected" not in columns and "intended" in columns
    assert {"intended r0", "achieved r0", "deviation r0"} <= set(columns)
    first = frame.iloc[0]
    assert first["achieved"] == "3.5 ± 0.7"             # r1, r2 only: 3 and 4
    assert first["achieved r0"] == "93"                 # r0 apart, never averaged in
    assert "93" not in first["achieved"]
    outside = frame[frame["bin"] == "< 0.5 (outside)"].iloc[0]
    assert outside["intended"] == "0" and outside["achieved r0"] == "90"
    assert frame.iloc[-1]["bin"].startswith("Σ|deviation|")


def test_structural_batch_frame_averages_over_the_confirmatory_realizations():
    frame = report.structural_batch_frame(_struct_stub())
    assert set(frame["set"]) == {"r1-r5", "r0 (exploratory)"}
    assert {"int. 1", "dev. 1", "int. 2", "dev. 2", "cum. 2"} <= set(frame.columns)
    confirmatory = frame[(frame["set"] == "r1-r5") & (frame["bin"] == "0.50-1.00")].iloc[0]
    assert confirmatory["int. 1"] == "2"                # the intended count is visible
    assert confirmatory["dev. 1"] == "-0.5 ± 0.7"       # r1: -1, r2: 0; r0's +89 stays out
    assert confirmatory["cum. 2"] == "-1.0 ± 1.4"
    r0 = frame[(frame["set"] == "r0 (exploratory)") & (frame["bin"] == "0.50-1.00")].iloc[0]
    assert r0["dev. 1"] == "89"


def test_structural_batch_frame_outside_rows_intend_nothing():
    frame = report.structural_batch_frame(_struct_stub())
    outside = frame[(frame["set"] == "r1-r5") & (frame["bin"] == "< 0.5 (outside)")].iloc[0]
    assert outside["int. 1"] == "0"
    assert outside["dev. 1"] == "0.5 ± 0.7"             # achieved outside: r1 0, r2 1
    assert outside["cum. 2"] == "1.0 ± 1.4"


def test_structural_displacement_frame_keeps_r0_apart():
    frame = report.structural_displacement_frame(_struct_stub())
    assert set(frame["set"]) == {"r1-r5", "r0 (exploratory)"}
    assert {"C-log", "C-seq"} <= set(frame["arm"])


def test_structural_outside_frame_reads_kind_and_side():
    frame = report.structural_outside_frame(_struct_stub())
    assert set(frame["kind"]) == {"density"}
    assert "below 0.5" in set(frame["side"])


def test_structural_ablation_figure_has_a_row_per_c0_arm_and_a_column_per_cell():
    fig = report.fig_structural_ablation(_struct_with_ablation())
    expected = 2 * len(config.REP_FAMILIES) * len(config.TEST_ROLES)  # two C0 arms
    assert len(fig.axes) == expected


def test_structural_ablation_figure_uses_explicit_colours_by_comparison():
    """A single arm's panel draws two comparisons (C0-B welch, C-C0 paired); an
    implicit colour cycle would give them the same colour as a second arm's
    panel elsewhere in the grid, erasing which comparison is which."""
    assert report.STRUCTURAL_ABLATION_COLORS["welch"] != report.STRUCTURAL_ABLATION_COLORS["paired"]
    fig = report.fig_structural_ablation(_struct_with_ablation())
    panel = fig.axes[0]
    colours = {tuple(round(float(v), 3) for v in coll.get_edgecolor()[0])
               for coll in panel.collections if len(coll.get_edgecolor())}
    assert len(colours) == 2


def test_structural_ablation_fraction_frame_reports_fractions_not_counts():
    frame = report.structural_ablation_fraction_frame(_struct_with_ablation())
    columns = [c.lower() for c in frame.columns]
    assert not any("count" in c for c in columns)
    assert {"c0 fraction", "b fraction", "diff"} <= set(columns)
    assert set(frame["arm"]) == {"C0-log", "C0-seq"}


def test_interaction_frame_declares_the_method():
    # REPLICATION_MEASURES has no bare "decision" entry; "decision-range" is the
    # actual member, so that is what the row must carry to survive the filter.
    assert "decision-range" in report.REPLICATION_MEASURES
    rep = {"interactions": [
        {"test": "test-natural", "measure": "decision-range", "first": "C-logG", "second": "C-log",
         "method": "paired", "diff": -0.1, "ci": [-0.2, 0.0], "df": 4},
    ]}
    frame = report.replication_interaction_frame(rep)
    assert "method" in frame.columns and frame.loc[0, "method"] == "paired"
    assert frame.loc[0, "contrast"] == "C-logG vs C-log"


def test_booktabs_escapes_the_caption():
    text = report.booktabs(pd.DataFrame({"a": [1]}), caption="n_bin / N_total, 95% & more",
                           label="tab:x")
    assert r"\caption{n\_bin / N\_total, 95\% \& more}" in text


def test_run_writes_structural_outputs_when_present(root, monkeypatch):
    monkeypatch.setattr(report, "check_structural_fresh", lambda struct: None)
    struct = _struct_with_ablation()
    struct["concordance"] = [{"family": "rf", "test": "test-natural", "arm": "C-log", "k": 5,
                              "n": 7, "zero": 0, "thin": 8, "k_all": 9, "n_all": 15, "zero_all": 0}]
    struct["ablation"]["caption"] = "ablation caption with n_bin"
    path = config.structural_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(struct))
    report.run()
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    figures = {p.name for p in config.figures_dir().glob("*.pdf")}
    assert {"structural__caption.tex", "structural__allocation_c-log.tex",
            "structural__concordance.tex", "structural__matrix_caption.tex",
            "structural__batches_r0.tex",
            "structural__allocation_c-seq.tex", "structural__batches.tex",
            "structural__displacement.tex", "structural__outside.tex",
            "structural__shared_candidates.tex",
            "structural__ablation_caption.tex", "structural__ablation_fractions.tex",
            "structural__original_c0.tex"} <= tables
    assert {"structural__bins.pdf", "structural__matrix_c-log.pdf",
            "structural__matrix_c-seq.pdf", "structural__ablation.pdf"} <= figures
    note = (config.tables_dir() / "structural__allocation_c-log.tex").read_text()
    assert struct["ablation"]["shared_rows"]["note"] in note
    batches_note = (config.tables_dir() / "structural__batches.tex").read_text()
    assert struct["ablation"]["shared_rows"]["note"] in batches_note
    displacement_note = (config.tables_dir() / "structural__displacement.tex").read_text()
    assert struct["ablation"]["shared_rows"]["note"] in displacement_note
    original = (config.tables_dir() / "structural__original_c0.tex").read_text()
    assert original.strip() == struct["ablation"]["original_c0"]
    caption = (config.tables_dir() / "structural__ablation_caption.tex").read_text()
    assert caption.strip() == r"ablation caption with n\_bin"          # the caption, escaped
    assert struct["ablation"]["procedure"] not in caption
    for name in ("allocation_c-log", "batches", "displacement"):
        text = (config.tables_dir() / f"structural__{name}.tex").read_text()
        assert text.rstrip().endswith(r"\end{table}"), name       # the note sits inside the float


def _set_labels():
    """The full manifest: guided-log, guided-log-g and every C-seq batch, for r0 and r1..rN."""
    labels = {0: ["guided-log", "guided-log-g",
                  *(f"guided-seq-{b}" for b in range(1, len(config.REP_SEQ_BATCHES) + 1))]}
    for r in range(1, config.REP_REALIZATIONS + 1):
        labels[r] = [f"guided_log_r{r}", f"guided_log_g_r{r}",
                     *(f"guided_seq_r{r}_b{b}" for b in range(1, len(config.REP_SEQ_BATCHES) + 1))]
    return labels


def _fresh_inputs(root_dir):
    """replications.json, the metadata and every guided set's design and final
    on disk, with the provenance that describes them. `files` names the ones
    the parametrized test changes."""
    rep, meta = config.replication_metrics_path(), config.metadata_path()
    sets = {}
    for path in (rep, meta):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n")
    for r, labels in _set_labels().items():
        for label in labels:
            paths = ({"design": config.design_path(label), "final": config.final_path(label)} if r == 0
                     else {"design": config.replication_path("designs", label),
                           "final": config.replication_path("final", label)})
            for kind, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{label} {kind}\n")
            sets[label] = {"realization": r, **{k: data.sha256_file(v) for k, v in paths.items()}}
    files = {"rep": rep, "meta": meta,
             "design0": config.design_path("guided-log"), "final0": config.final_path("guided-log"),
             "design1": config.replication_path("designs", "guided_log_r1"),
             "final1": config.replication_path("final", "guided_log_r1")}
    provenance = {"replications_sha256": data.sha256_file(rep),
                  "run_metadata_sha256": data.sha256_file(meta), "sets_sha256": sets}
    return files, {"provenance": provenance}


@pytest.mark.parametrize("changed, message", [
    ("rep", "another replications"), ("meta", "another run_metadata"),
    ("design0", "another guided-log design"), ("final0", "another guided-log final"),
    ("design1", "another guided_log_r1 design"), ("final1", "another guided_log_r1 final")])
def test_structural_freshness_refuses_every_changed_input(root, changed, message):
    files, struct = _fresh_inputs(root)
    report.check_structural_fresh(struct)                   # intact: passes
    files[changed].write_text("changed\n")
    with pytest.raises(report.StaleStructuralError, match=message):
        report.check_structural_fresh(struct)


def test_structural_freshness_refuses_a_set_without_its_realization(root):
    _, struct = _fresh_inputs(root)
    del struct["provenance"]["sets_sha256"]["guided-log"]["realization"]
    with pytest.raises(report.StaleStructuralError, match="no realization"):
        report.check_structural_fresh(struct)


def test_run_checks_structural_freshness_before_rendering(root, monkeypatch):
    struct = _struct_with_ablation()
    path = config.structural_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(struct))
    seen = []

    def refuse(given):
        seen.append(given)
        raise report.StaleStructuralError("stale")

    monkeypatch.setattr(report, "check_structural_fresh", refuse)
    with pytest.raises(report.StaleStructuralError, match="stale"):
        report.run()
    assert seen and seen[0]["figure_caption"] == struct["figure_caption"]
    assert not (config.figures_dir() / "structural__bins.pdf").exists()


@pytest.mark.parametrize("damage, message", [
    (lambda sets: sets.clear(), "no set manifest"),
    (lambda sets: sets.pop("guided_seq_r3_b2"), "incomplete"),
    (lambda sets: sets.pop("guided-log"), "incomplete"),
    (lambda sets: [sets.pop(k) for k in [k for k, v in sets.items() if v["realization"] == 5]],
     "incomplete"),
    (lambda sets: sets.update({"stray": {"realization": 9, "design": "0", "final": "0"}}),
     "unknown realizations"),
])
def test_structural_freshness_refuses_an_incomplete_manifest(root, damage, message):
    _, struct = _fresh_inputs(root)
    report.check_structural_fresh(struct)                   # complete: passes
    damage(struct["provenance"]["sets_sha256"])
    with pytest.raises(report.StaleStructuralError, match=message):
        report.check_structural_fresh(struct)


def _with_pool_sets(struct):
    """Add C-oracle and C-logP sets for r1..rN only, as an artifact built after Task 13 has."""
    sets = struct["provenance"]["sets_sha256"]
    for r in range(1, config.REP_REALIZATIONS + 1):
        for label in (f"guided_oracle_r{r}", f"guided_log_p_r{r}"):
            paths = {"design": config.replication_path("designs", label),
                     "final": config.replication_path("final", label)}
            for kind, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{label} {kind}\n")
            sets[label] = {"realization": r, **{k: data.sha256_file(v) for k, v in paths.items()}}
    struct["oracle"] = {"shared_rows": []}
    return struct


def test_manifest_count_before_the_pool_arms_exist(root):
    # An artifact without the oracle block (every artifact before Task 13) is
    # complete with the guided sets alone, even though REP_STRATEGIES lists the pool.
    _, struct = _fresh_inputs(root)
    report.check_structural_fresh(struct)


def test_manifest_count_with_the_pool_arms_has_no_r0(root):
    _, struct = _fresh_inputs(root)
    report.check_structural_fresh(_with_pool_sets(struct))          # r0 has 5, r1..rN have 7
    struct["provenance"]["sets_sha256"].pop("guided_log_p_r2")
    with pytest.raises(report.StaleStructuralError, match="incomplete"):
        report.check_structural_fresh(struct)


def test_manifest_with_an_oracle_block_but_no_pool_sets_is_incomplete(root):
    _, struct = _fresh_inputs(root)
    struct["oracle"] = {"shared_rows": []}
    with pytest.raises(report.StaleStructuralError, match="incomplete"):
        report.check_structural_fresh(struct)


def test_structural_freshness_refuses_a_missing_manifest_field(root):
    _, struct = _fresh_inputs(root)
    del struct["provenance"]["sets_sha256"]
    with pytest.raises(report.StaleStructuralError, match="no set manifest"):
        report.check_structural_fresh(struct)


def test_the_committed_artifact_passes_the_freshness_check():
    """The real manifest is complete and matches the files on disk."""
    report.check_structural_fresh(report.load_structural())


def test_structural_freshness_refuses_missing_provenance_fields(root):
    with pytest.raises(report.StaleStructuralError, match="no provenance"):
        report.check_structural_fresh({})
    for field, message in (("replications_sha256", "missing replications_sha256"),
                           ("run_metadata_sha256", "missing run_metadata_sha256")):
        _, struct = _fresh_inputs(root)
        del struct["provenance"][field]
        with pytest.raises(report.StaleStructuralError, match=message):
            report.check_structural_fresh(struct)


def test_structural_freshness_refuses_a_missing_input_file(root):
    files, struct = _fresh_inputs(root)
    for name in ("meta", "final1"):
        files, struct = _fresh_inputs(root)
        files[name].unlink()
        with pytest.raises(report.StaleStructuralError, match="missing"):
            report.check_structural_fresh(struct)


# --- Task 13: the pool arms (C-oracle, C-logP) everywhere -------------------------------------

def test_oracle_reading_frame_names_the_rows_and_the_attribution():
    rep = json.loads(config.replication_metrics_path().read_text())
    frame = report.oracle_reading_frame(rep)
    assert {"cell", "median L C-oracle", "median H C-oracle", "median dR_O", "dR_O CI low",
            "dR_O CI high"} <= set(frame.columns)
    summary = frame[frame["cell"] == "reading"].iloc[0]
    assert summary["median dR_O"] in {"A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"}


def _struct_without_r0(arm):
    """The committed structural artifact with `arm` copied from C-log into r1..rN
    only, as the pool arms are: present in every confirmatory realization, absent at r0."""
    struct = json.loads(config.structural_metrics_path().read_text())
    for r in map(str, struct["confirmatory"]):
        struct["allocation"][r][arm] = struct["allocation"][r]["C-log"]
        struct["displacement"][r][arm] = struct["displacement"][r]["C-log"]
    assert arm not in struct["allocation"]["0"] and arm not in struct["displacement"]["0"]
    return struct


def test_allocation_frame_marks_the_missing_r0():
    frame = report.structural_allocation_frame(_struct_without_r0("C-oracle"), "C-oracle")
    assert set(frame["intended r0"]) == {"--"} and set(frame["deviation r0"]) == {"--"}
    assert (frame["deviation"] != "--").all()                  # r1-r5 are all there


def test_displacement_frame_keeps_an_arm_absent_at_r0():
    # Taking the arms from r0 would drop C-oracle and C-logP without a trace.
    frame = report.structural_displacement_frame(_struct_without_r0("C-logP"))
    rows = frame[frame["arm"] == "C-logP"]
    assert rows["set"].tolist() == ["r1-r5", "r0 (exploratory)"]
    assert set(rows.iloc[1].drop(["arm", "set"])) == {"--"}


# --- the layout of the tables (2026-09-29): fit the page, lose nothing -------------------------

import re
from collections import Counter

_NUMBER = re.compile(r"^[-+]?\d+\.\d+$")


def _table_numbers(tex: str) -> Counter:
    """The cells of every body row that are a plain decimal number."""
    cells = Counter()
    for line in tex.splitlines():
        if line.endswith(r"\\") and not line.startswith(("\\caption", "\\multicolumn")):
            for cell in line[:-2].split("&"):
                if _NUMBER.match(cell.strip()):
                    cells[cell.strip()] += 1
    return cells


def _frame_numbers(frame: pd.DataFrame, digits: int = 3) -> Counter:
    return Counter(f"{v:.{digits}f}" for row in frame.itertuples(index=False) for v in row
                   if isinstance(v, (float, np.floating)) and not np.isnan(v))


def test_arms_are_written_as_the_manuscript_writes_them():
    assert report.arm_label("C0-logG") == "C₀-logG" and report.arm_label("C-log") == "C-log"
    assert report._escape("C0-log vs C0") == "C$_0$-log vs C$_0$"
    assert report._escape("C0 fraction; RC0 stays") == "C$_0$ fraction; RC0 stays"
    assert report._escape(report.Tex(r"$R^2$")) == r"$R^2$"          # raw LaTeX passes as is


def test_the_ablation_note_names_every_ablation_once():
    note = report.ablation_note()
    assert all(report.arm_label(arm) in note for arm, spec in config.ARMS.items() if spec.ablation)
    assert note.count("C₀-log,") == 1 and "(ablation)" not in note
    assert report.ablation_note(["C0"]).startswith("C₀ is an ablation")


def test_grouped_headers_and_family_blocks():
    frame = report.regroup(pd.DataFrame({"family": ["rf", "rf", "gbm"], "arm": ["A", "C0", "A"],
                                         "x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]}),
                           {"family": "family", "arm": "arm", "x": ("scale", "x"), "y": ("scale", "y")})
    tex = report.booktabs(frame, caption="Demo.", label="tab:demo", block="family")
    assert r"\multicolumn{2}{c}{scale}" in tex and r"\cmidrule(lr){2-3}" in tex
    assert r"\multicolumn{3}{l}{\textit{RF}} \\" in tex and r"\multicolumn{3}{l}{\textit{GBM}} \\" in tex
    assert "family" not in tex.split(r"\midrule")[0].split(r"\label")[1]    # the block column leaves
    assert r"C$_0$ & 0.200 & 2.000" in tex
    assert r"\begin{tabular}{lrr}" in tex


def test_longtable_repeats_its_head_and_keeps_every_value():
    frame = pd.DataFrame({"family": ["rf"] * 3, "subset": ["all"] * 3, "d": [0.1, -0.2, 0.3]})
    tex = report.longtable(frame, caption="Inventory.", label="tab:inv", block="family")
    assert tex.count("subset & d") == 2                       # first head, then every page
    assert tex.index(r"\endfirsthead") < tex.index(r"\endhead") < tex.index(r"\endlastfoot")
    assert r"\begin{longtable}{lr}" in tex and r"\caption{Inventory.}\label{tab:inv} \\" in tex
    assert _table_numbers(tex) == _frame_numbers(frame)


def test_the_restructured_tables_keep_every_value(root):
    metrics = _write_metrics()
    cases = [(report.global_table(metrics, "test-natural", "s"), report.global_frame(metrics)),
             (report.subset_table(metrics, "test-natural", "s"), report.subset_frame(metrics)),
             (report.comparison_table(metrics, "test-natural", "s"), report.comparison_frame(metrics)),
             (report.conformal_tables(metrics, "test-natural", "s"), report.conformal_frame(metrics))]
    for tex, frame in cases:
        assert _table_numbers(tex) == _frame_numbers(frame)


def test_conformal_is_two_tables_with_the_zones_as_columns(root):
    metrics = _write_metrics()
    tex = report.conformal_tables(metrics, "test-natural", "s")
    assert tex.count(r"\begin{table}") == 2
    assert r"\label{tab:s:conformal}" in tex and r"\label{tab:s:conformal:width}" in tex
    zones = [z["zone"] for z in metrics["arms"]["rf"]["A"]["conformal"]["zones"]]
    coverage, width = tex.split(r"\end{table}")[:2]
    for part in (coverage, width):
        assert all(report._escape(z) in part for z in zones)
    assert "factor" in width and "factor" not in coverage.split(r"\label")[1]


def test_scoring_params_table_has_one_panel_per_family(root):
    metrics = _write_metrics()
    tex = report.scoring_params_table({"log-rmse": metrics, "fos-rmse": metrics})
    assert tex.count(r"\begin{tabular}") == len(config.FAMILIES)
    assert tex.count(r"\label{tab:scorings:params}") == 1
    assert r"\textit{RF}" in tex and r"\textit{GBM}" in tex and "=" not in tex.split(r"\label")[1]


def test_oracle_table_splits_the_reading_with_its_columns():
    rep = json.loads(config.replication_metrics_path().read_text())
    tex = report.oracle_table(rep)
    medians, intervals = tex.split(r"\par\medskip")
    assert "gate passes" in medians and "rho" not in medians
    assert rep["oracle"]["row"] in intervals and r"$\rho_L$" in intervals
    assert _table_numbers(tex) == _frame_numbers(report.oracle_reading_frame(rep))


def test_oracle_table_typesets_its_labels_and_keeps_its_values():
    rep = json.loads(config.replication_metrics_path().read_text())
    tex = report.oracle_table(rep)
    for raw in ("dR\\_O", "dL\\_O", "dH\\_O", "rho\\_", "gbm/", "rf/", "test-natural"):
        assert raw not in tex, raw
    for typeset in (r"{$dR_O$}", r"{$dL_O$}", r"{$dH_O$}", r"$\rho_H$", "median $L$", "GBM, natural",
                    "RF, stratified", r"$dR_O = R(\mbox{C-oracle}) - R(\mbox{C-logP})$"):
        assert typeset in tex, typeset
    for rho in (rep["oracle"]["rho"]["L"], rep["oracle"]["rho"]["H"]):
        assert f"{rho:.3f}" in tex


def test_split_tables_drop_only_what_is_constant():
    frame = pd.DataFrame({"kind": ["density", "error", "error"], "arm": ["C-log"] * 3,
                          "test": ["--", "test-natural", "test-stratified"], "diff": [1.0, 2.0, 3.0]})
    tex = report.booktabs_split(frame, "kind", caption="Out.", label="tab:out")
    density, error = tex.split(r"\end{table}")[:2]
    assert r"\label{tab:out:density}" in density and "test" not in density.split(r"\label")[1]
    assert "This table: density." in density and "test-natural" in error
    assert _table_numbers(tex) == _frame_numbers(frame)


def test_zone_figure_keeps_its_legend():
    metrics = evaluate.evaluate({**stub_fits("rf"), **stub_fits("gbm")}, stub_test(), **EVAL_KWARGS).to_json()
    fig = report.fig_zone_rmse(metrics, "rf")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "C0 (ablation)" in labels                      # figures are revised in their own pass


def test_hyper_parameters_keep_the_point_of_a_whole_float():
    assert report._param(1.0) == "1.0"                    # max_features: every feature, not one
    assert report._param(0.5) == "0.5" and report._param(0.06) == "0.06"
    assert report._param(8000) == "8000" and report._param(1) == "1"
    metrics = {"arms": {"rf": {"A": {"best_params": {"max_features": 1.0}}}}}
    assert "max_features, a fraction of the features, 1.0 = all" in report.param_legend({"log-rmse": metrics})


def test_forest_plot_reads_every_declared_row_against_b():
    rep = json.loads(config.replication_metrics_path().read_text())
    frame = report.forest_frame(rep)
    arms = [a for _, _, block in report.FOREST_BLOCKS for a in block]
    # discipline 1: every C0 variant with a replicated row against B; A has none
    c0 = {a for a in config.ARMS if config.ARMS[a].ablation and a != "C0"} | {"C0-oracle", "C0-logP"}
    assert c0 <= set(arms) and "A" not in arms and "B" not in arms
    assert set(frame["arm"]) == set(arms)
    assert set(frame["measure"]) == {m for m, _ in report.FOREST_MEASURES}
    assert set(zip(frame["family"], frame["test"])) == set(report.FOREST_CELLS)
    assert len(frame) == len(arms) * len(report.FOREST_MEASURES) * len(report.FOREST_CELLS)
    assert (frame["level"] == 0.95).all()
    row = frame[(frame["arm"] == "C-oracle") & (frame["measure"] == "decision-range")
                & (frame["family"] == "rf") & (frame["test"] == "test-natural")].iloc[0]
    source = next(c for c in rep["comparisons"] if c["first"] == "C-oracle" and c["second"] == "B"
                  and c["scale"] == "log-rmse" and c["measure"] == "decision-range"
                  and c["family"] == "rf" and c["test"] == "test-natural")
    assert (row["diff"], row["ci low"], row["ci high"]) == (source["diff"], *source["ci"])
    fig = report.fig_forest(rep)
    assert [ax.get_title() for ax in fig.axes] == [t for _, t in report.FOREST_MEASURES]
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [report.arm_label(a) for a in arms]
    assert len(fig.legends[0].get_texts()) == len(report.FOREST_CELLS)


def test_forest_plot_refuses_a_missing_ablation():
    rep = json.loads(config.replication_metrics_path().read_text())
    rep["comparisons"] = [c for c in rep["comparisons"] if c["first"] != "C0-logG"]
    with pytest.raises(KeyError, match="C0-logG"):
        report.forest_frame(rep)


def test_forest_caption_states_what_the_intervals_are_not():
    caption = report.forest_caption(json.loads(config.replication_metrics_path().read_text()))
    for phrase in ("r1-r5", "95% Welch interval", "never a tie or evidence of equivalence",
                   "A is absent", "C and the original C0 are absent", "different budgets", "have no r0"):
        assert phrase in caption
    assert "significan" not in caption.lower() and "balanced" not in caption
    assert "Zones are half-open, [lo, hi)" in caption and "Bins" not in caption


def test_allocation_figure_draws_the_allocation_table_values():
    struct = json.loads(config.structural_metrics_path().read_text())
    fig = report.fig_structural_allocation(struct)
    assert [ax.get_title(loc="left") for ax in fig.axes] == list(report.ALLOCATION_FIGURE_ARMS)
    for ax, arm in zip(fig.axes, report.ALLOCATION_FIGURE_ARMS):
        table = report.structural_allocation_frame(struct, arm).iloc[:-1]        # Σ row is not drawn
        bars = next(c for c in ax.containers if isinstance(c, BarContainer))
        assert [str(int(p.get_height())) for p in bars] == table["intended"].tolist()
        means = ax.lines[0].get_ydata()
        assert np.allclose(means, [float(a.split(" ± ")[0]) for a in table["achieved"]], atol=0.05)
    caption = report.structural_allocation_caption(struct)
    assert "(lo, hi]" in caption and "not a prediction error" in caption
    assert "C$_0$-oracle" in caption and "C$_0$-logP" in caption               # the shared rows, discipline 1
    assert r"\cref{tab:rep:oracle}" in caption and "Table tab:rep:oracle" not in caption
    assert "the arm it ablates" in caption and "those of C," not in caption
