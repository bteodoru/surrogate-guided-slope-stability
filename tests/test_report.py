import json

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
    assert frame["arm"].tolist() == ["A", "B", "C", "C-log"]
    assert frame["solver budget"].tolist() == [300, 600, 600, 600]


def test_comparison_frame_formats_tiny_pvalues():
    metrics = {"comparisons": [
        {"family": "rf", "subset": "global", "first": "C", "second": "C0",
         "wilcoxon": {"alternative": "less", "n": 10, "median_d": -0.1, "p_d_lt_0": 0.9,
                      "pvalue": 0.00005},
         "bootstrap": {"delta_rmse": -0.2, "ci": [-0.3, -0.1]}},
        {"family": "rf", "subset": "global", "first": "B", "second": "A",
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
            "test-natural__log-rmse__conformal.tex", "test-natural__log-rmse__zones_rf.tex"} <= tables
    figure = config.figures_dir() / "test-natural__log-rmse__zones_gbm.pdf"
    assert figure.read_bytes()[:4] == b"%PDF"


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
