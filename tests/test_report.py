import json

import numpy as np
import pandas as pd

from sgs import config, evaluate, report
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


def test_budget_table_excludes_the_ablation():
    frame = report.budget_frame()
    assert frame["arm"].tolist() == ["A", "B", "C", "C-log", "C-seq"]
    assert frame["solver budget"].tolist() == [300, 600, 600, 600, 600]


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
    assert "C0 (ablation)" in report.global_frame(metrics)["arm"].tolist()
    assert "C0 (ablation)" in report.zone_frame(metrics, "rf").columns
    assert "C0 (ablation)" in report.conformal_frame(metrics)["arm"].tolist()
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
    assert "C0 (ablation)" in frame["arm"].tolist()


def test_scoring_frames_compare_the_selection_criteria():
    log = evaluate.evaluate({**stub_fits("rf"), **stub_fits("gbm")}, stub_test(), **EVAL_KWARGS).to_json()
    fos = evaluate.evaluate({**stub_fits("rf", scoring="fos-rmse"), **stub_fits("gbm", scoring="fos-rmse")},
                            stub_test(), **EVAL_KWARGS).to_json()
    frame = report.scoring_frame({"log-rmse": log, "fos-rmse": fos})
    assert len(frame) == len(config.FAMILIES) * len(config.ARMS)
    assert "C0 (ablation)" in frame["arm"].tolist()
    for scoring in ("log-rmse", "fos-rmse"):
        for family in config.FAMILIES:
            ranks = frame.loc[frame["family"] == family, f"rank {scoring}"]
            assert sorted(ranks) == list(range(1, len(config.ARMS) + 1))
    params = report.scoring_params_frame({"log-rmse": log, "fos-rmse": fos})
    assert list(params.columns) == ["family", "arm", "log-rmse", "fos-rmse"]


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

def _struct_stub():
    labels = ["0.50-1.00", "1.00-6.00"]
    errors = [{"family": f, "test": t, "arm": a, "bin": b, "diff": -0.01, "ci": [-0.02, 0.0],
               "test_n": 20 if b == 0 else 3, "marked": b == 1}
              for f in config.REP_FAMILIES for t in config.TEST_ROLES
              for a in ("C-log", "C-seq") for b in (0, 1)]
    return {
        "bins": {"edges": [0.5, 1.0, 6.0], "labels": labels},
        "min_test_n": 10, "confirmatory": [1, 2],
        "figure_caption": "caption text",
        "densities": [{"arm": a, "bin": b, "diff": 3.0, "ci": [1.0, 5.0]}
                      for a in ("C-log", "C-seq") for b in (0, 1)],
        "errors": errors,
        "outside": [{"kind": "density", "arm": "C-log", "side": "below 0.5",
                     "diff": 20.0, "ci": [10.0, 30.0]}],
        "concordance": [],
        "allocation": {r: {"C-log": {"seed": [2, 2], "intended": [4, 4], "achieved": [3, 5],
                                     "deviation": [-1, 1], "abs_deviation": 2},
                           "C-seq": {"seed": [1, 1], "intended": [3, 3], "achieved": [2, 4],
                                    "deviation": [-1, 1], "abs_deviation": 2}}
                       for r in ("0", "1", "2")},
        "batches": {r: [{"batch": 1, "seed": [0, 0], "intended": [2, 2], "achieved": [1, 3],
                         "cumulative_intended": [2, 2], "cumulative_achieved": [1, 3]}]
                    for r in ("0", "1", "2")},
        "displacement": {r: {"C-log": {"n": 4, "stayed": 0.5, "mean_abs_shift": 0.5,
                                       "median_abs_shift": 0.0, "outside": 0.25,
                                       "matrix": [[2, 0, 1, 1], [1, 0, 1, 0]]},
                             "C-seq": {"n": 4, "stayed": 0.5, "mean_abs_shift": 0.5,
                                      "median_abs_shift": 0.0, "outside": 0.25,
                                      "matrix": [[1, 1, 0, 2], [0, 1, 1, 0]]}}
                         for r in ("0", "1", "2")},
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
    turns its marked markers grey. Four collections, two colours, is the fix."""
    assert report.STRUCTURAL_COLORS["C-log"] != report.STRUCTURAL_COLORS["C-seq"]
    fig = report.fig_structural(_struct_stub())
    error_panel = fig.axes[1]          # bin 0 solid, bin 1 marked, for both arms
    colours = {tuple(round(float(v), 3) for v in coll.get_edgecolor()[0])
               for coll in error_panel.collections if len(coll.get_edgecolor())}
    assert len(colours) == 2, "solid and hollow markers of one arm must share its colour"


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
    assert any("sd" in c for c in columns)
    assert set(frame["set"]) == {"r1-r5", "r0 (exploratory)"}


def test_structural_batch_frame_averages_over_the_confirmatory_realizations():
    frame = report.structural_batch_frame(_struct_stub())
    assert set(frame["set"]) == {"r1-r5", "r0 (exploratory)"}
    assert "cum. intended" in frame.columns


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


def test_run_writes_structural_outputs_when_present(root):
    struct = _struct_with_ablation()
    path = config.structural_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(struct))
    report.run()
    tables = {p.name for p in config.tables_dir().glob("*.tex")}
    figures = {p.name for p in config.figures_dir().glob("*.pdf")}
    assert {"structural__caption.tex", "structural__allocation_c-log.tex",
            "structural__allocation_c-seq.tex", "structural__batches.tex",
            "structural__displacement.tex", "structural__outside.tex",
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
    assert struct["ablation"]["procedure"] in caption and "300" in caption and "600" in caption
