"""Booktabs tables and figures for the manuscript.

Reads only outputs/metrics/ and recomputes no metric, so no number reaches a
table without having passed through ``evaluate``. Every arm appears in every
results table, C0 included and labelled as the ablation; the budget table
lists budget arms only.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sgs import config


def arm_label(arm: str) -> str:
    return f"{arm} (ablation)" if config.ARMS[arm].ablation else arm


def _read(path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs evaluate`")
    return json.loads(path.read_text())


def load_metrics(test: str, scoring: str) -> dict:
    return _read(config.metrics_path(test, scoring))


def load_learning_curve() -> dict:
    return _read(config.learning_curve_metrics_path())


def _escape(text: str) -> str:
    for raw, safe in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(raw, safe)
    return text


def _cell(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return _escape(str(value))


def booktabs(frame: pd.DataFrame, *, caption: str, label: str, digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{l{'r' * (len(columns) - 1)}}}", r"\toprule",
             " & ".join(_escape(str(c)) for c in columns) + r" \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(_cell(v, digits) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def global_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for family, arms in metrics["arms"].items():
        for arm in config.ARMS:
            m = arms[arm]
            g = m["global"]
            rows.append({"family": family, "arm": arm_label(arm), "R2": g["r2"], "RMSE": g["rmse"],
                         "log-RMSE": g["log_rmse"], "MAE": g["mae"], "bias": g["bias"],
                         "max |e|": g["max_abs_error"], "minimax": m["minimax"]["rmse"],
                         "worst zone": m["minimax"]["zone"],
                         "minimax log": m["minimax_log"]["log_rmse"]})
    return pd.DataFrame(rows)


def zone_frame(metrics: dict, family: str, key: str = "rmse") -> pd.DataFrame:
    arms = metrics["arms"][family]
    reference = arms[next(iter(config.ARMS))]["zones"]
    table = {"zone": [z["zone"] for z in reference], "n": [z["n"] for z in reference]}
    for arm in config.ARMS:
        table[arm_label(arm)] = [z[key] for z in arms[arm]["zones"]]
    return pd.DataFrame(table)


def subset_frame(metrics: dict) -> pd.DataFrame:
    """Log- and FOS-scale RMSE per arm on every subset; the decision range carries the primary criterion."""
    rows = []
    for family, arms in metrics["arms"].items():
        for arm in config.ARMS:
            row = {"family": family, "arm": arm_label(arm)}
            for subset, m in arms[arm]["subsets"].items():
                row[f"log-RMSE {subset}"] = m["log_rmse"]
            for subset, m in arms[arm]["subsets"].items():
                row[f"RMSE {subset}"] = m["rmse"]
            rows.append(row)
    return pd.DataFrame(rows)


def _format_pvalue(p: float) -> str:
    """Render a p-value for the tables: an order of magnitude below the
    table's usual 3 decimals would otherwise print as an uninformative 0.000."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for c in metrics["comparisons"]:
        w, b = c["wilcoxon"], c["bootstrap"]
        rows.append({"family": c["family"], "subset": c["subset"], "scale": c["scale"],
                     "comparison": f"{c['first']} vs {c['second']}", "alternative": w["alternative"],
                     "n": w["n"], "median d": w["median_d"], "P(d<0)": w["p_d_lt_0"],
                     "p": _format_pvalue(w["pvalue"]),
                     "ΔRMSE": b["delta_rmse"], "CI low": b["ci"][0], "CI high": b["ci"][1]})
    return pd.DataFrame(rows)


def conformal_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for family, arms in metrics["arms"].items():
        for arm in config.ARMS:
            c = arms[arm]["conformal"]
            row = {"family": family, "arm": arm_label(arm), "factor": c["factor"],
                   "coverage": c["coverage"], "mean width": c["mean_width"]}
            for z in c["zones"]:
                row[f"cov {z['zone']}"] = z["coverage"]
                row[f"width {z['zone']}"] = z["mean_width"]
            rows.append(row)
    return pd.DataFrame(rows)


def scoring_frame(by_scoring: dict[str, dict]) -> pd.DataFrame:
    """The selection-criterion sensitivity check on one test set: global RMSE, minimax
    and within-family rank of every arm under each scoring."""
    rows = []
    for family in config.FAMILIES:
        ranks = {}
        for scoring, metrics in by_scoring.items():
            rmse = {arm: metrics["arms"][family][arm]["global"]["rmse"] for arm in config.ARMS}
            ranks[scoring] = {arm: 1 + sorted(rmse.values()).index(value) for arm, value in rmse.items()}
        for arm in config.ARMS:
            row = {"family": family, "arm": arm_label(arm)}
            for scoring, metrics in by_scoring.items():
                m = metrics["arms"][family][arm]
                row[f"RMSE {scoring}"] = m["global"]["rmse"]
                row[f"minimax {scoring}"] = m["minimax"]["rmse"]
                row[f"rank {scoring}"] = ranks[scoring][arm]
            rows.append(row)
    return pd.DataFrame(rows)


def _params(params: dict) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in sorted(params.items()))


def scoring_params_frame(by_scoring: dict[str, dict]) -> pd.DataFrame:
    """Hyper-parameters selected under each scoring; they do not depend on the test set."""
    rows = []
    for family in config.FAMILIES:
        for arm in config.ARMS:
            row = {"family": family, "arm": arm_label(arm)}
            for scoring, metrics in by_scoring.items():
                row[scoring] = _params(metrics["arms"][family][arm]["best_params"])
            rows.append(row)
    return pd.DataFrame(rows)


def budget_frame() -> pd.DataFrame:
    return pd.DataFrame([{"arm": arm, "training sets": " + ".join(spec.roles), "solver budget": spec.budget}
                         for arm, spec in config.ARMS.items() if not spec.ablation])


def fig_zone_rmse(metrics: dict, family: str, key: str = "rmse"):
    frame = zone_frame(metrics, family, key)
    labels = [arm_label(a) for a in config.ARMS]
    x = np.arange(len(frame))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, label in enumerate(labels):
        ax.bar(x + i * width, frame[label].astype(float), width, label=label)
    ax.set_xticks(x + width * (len(labels) - 1) / 2, frame["zone"])
    ax.set_ylabel("RMSE (log FOS)" if key == "log_rmse" else "RMSE (FOS)")
    ax.set_title(f"{metrics['test']} — {family}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_learning_curve(lc: dict, test: str, family: str):
    """FOS-scale RMSE (global, selection range), log-scale RMSE (the trained response)
    and RMSE in the highest zone, where FOS-scale errors are largest."""
    rows = [r for r in lc["rows"] if r["test"] == test and r["family"] == family]
    top_zone = rows[0]["zones"][-1]["zone"]
    panels = [
        ("Global RMSE", "RMSE (FOS)", lambda r: (r["rmse_mean"], r["rmse_std"])),
        ("RMSE in the selection range", "RMSE (FOS)",
         lambda r: (r["rmse_selection_mean"], r["rmse_selection_std"])),
        ("Global RMSE of log(FOS)", "RMSE (log FOS)", lambda r: (r["log_rmse_mean"], r["log_rmse_std"])),
        (f"RMSE in zone {top_zone}", "RMSE (FOS)",
         lambda r: (r["zones"][-1]["rmse_mean"], r["zones"][-1]["rmse_std"])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for curve in sorted({r["curve"] for r in rows}):
        points = sorted((r for r in rows if r["curve"] == curve), key=lambda r: r["budget"])
        budgets = [r["budget"] for r in points]
        for ax, (_, _, value) in zip(axes.flat, panels):
            mean, std = (np.array(v, dtype=float) for v in zip(*(value(r) for r in points)))
            ax.plot(budgets, mean, "o-", label=curve)
            ax.fill_between(budgets, mean - std, mean + std, alpha=0.15)
    for ax, (title, ylabel, _) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Solver budget")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{test}, {family.upper()} with fixed hyper-parameters")
    fig.tight_layout()
    return fig


def _save_table(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_figure(path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No creation date, so an unchanged figure is byte-identical in git.
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


REPLICATION_MEASURES = ("decision-range", "0.5–1", "1–2", "selection-range", "all")


def load_replications() -> dict:
    return _read(config.replication_metrics_path())


def replication_value_frame(rep: dict, test: str, measure: str) -> pd.DataFrame:
    """Every realization's log-RMSE on one measure; r0 apart, as exploratory."""
    rows = []
    for s in rep["summaries"]:
        if s["test"] != test or s["measure"] != measure:
            continue
        row = {"family": s["family"], "arm": s["arm"], "r0 (exploratory)": s["r0"]}
        for r, v in zip(rep["confirmatory"], s["values"] or [None] * len(rep["confirmatory"])):
            row[f"r{r}"] = v
        row.update(mean=s["mean"], sd=s["sd"])
        rows.append(row)
    return pd.DataFrame(rows)


def replication_comparison_frame(rep: dict, test: str) -> pd.DataFrame:
    rows = [{"family": c["family"], "measure": c["measure"], "comparison": f"{c['first']} vs {c['second']}",
             "method": c["method"], "scale": c["scale"], "diff": c["diff"], "CI low": c["ci"][0],
             "CI high": c["ci"][1], "df": c["df"], "P(first<second)": c["p_first_less"]}
            for c in rep["comparisons"] if c["test"] == test and c["measure"] in REPLICATION_MEASURES]
    return pd.DataFrame(rows)


def replication_interaction_frame(rep: dict) -> pd.DataFrame:
    return pd.DataFrame([{"test": i["test"], "measure": i["measure"],
                          "contrast": f"{i['first']} vs {i['second']}", "diff": i["diff"],
                          "CI low": i["ci"][0], "CI high": i["ci"][1], "df": i["df"]}
                         for i in rep["interactions"] if i["measure"] in REPLICATION_MEASURES])


def fig_replication_bins(rep: dict, test: str, family: str):
    """Diagnostic: log-RMSE per design bin (confirmatory mean, min-max across
    realizations), test points per bin and the mean training count per bin."""
    arms = ("B", "C-log", "C-seq", "C0-log", "C0-seq")
    rows = [v for v in rep["values"] if v["test"] == test and v["family"] == family
            and v["realization"] in rep["confirmatory"]]
    labels = [b["zone"] for b in rows[0]["bins"]]
    x = np.arange(len(labels))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=(2, 1))
    for arm in arms:
        arm_rows = [v for v in rows if v["arm"] == arm]
        if not arm_rows:
            continue
        ablation = arm.startswith("C0")
        style = "o--" if ablation else "o-"
        grid = np.ma.masked_invalid(np.array(
            [[np.nan if b["log_rmse"] is None else b["log_rmse"] for b in v["bins"]] for v in arm_rows],
            dtype=float))
        top.plot(x, grid.mean(axis=0).filled(np.nan), style, label=arm)
        if not ablation:
            top.fill_between(x, grid.min(axis=0).filled(np.nan), grid.max(axis=0).filled(np.nan), alpha=0.15)
        counts = np.array([v["train_counts"] for v in arm_rows], dtype=float)
        bottom.plot(x, counts.mean(axis=0), style, label=f"{arm} training")
    bottom.bar(x, [b["n"] for b in rows[0]["bins"]], alpha=0.3, color="grey", label="test points")
    top.set_ylabel("log-RMSE")
    top.set_title(f"{test}, {family.upper()}: mean over realizations {rep['confirmatory']}, min–max for budget arms")
    top.legend(frameon=False)
    bottom.set_ylabel("points per bin")
    bottom.legend(frameon=False, fontsize=8)
    bottom.set_xticks(x, labels, rotation=60, fontsize=7)
    fig.tight_layout()
    return fig


def run() -> None:
    tables, figures = config.tables_dir(), config.figures_dir()
    _save_table(tables / "budget.tex", booktabs(
        budget_frame(), caption="Budget-fair arms. C0 is an ablation and carries no budget.",
        label="tab:budget"))

    for scoring in config.ACTIVE_SCORINGS:
        for test in config.TEST_ROLES:
            if not config.metrics_path(test, scoring).exists():
                print(f"[report] {config.metrics_path(test, scoring).name} missing; run sgs evaluate")
                continue
            metrics = load_metrics(test, scoring)
            stem = f"{test}__{scoring}"
            _save_table(tables / f"{stem}__global.tex", booktabs(
                global_frame(metrics), caption=f"Global metrics on {test}.", label=f"tab:{stem}:global"))
            _save_table(tables / f"{stem}__comparisons.tex", booktabs(
                comparison_frame(metrics), caption=f"Paired comparisons on {test}: Wilcoxon on "
                "d = |e1| - |e2| (smaller absolute errors) and bootstrap ΔRMSE.",
                label=f"tab:{stem}:comparisons"))
            _save_table(tables / f"{stem}__conformal.tex", booktabs(
                conformal_frame(metrics), caption=f"Cross-conformal-style intervals on {test}, "
                "empirical coverage.", label=f"tab:{stem}:conformal"))
            _save_table(tables / f"{stem}__subsets.tex", booktabs(
                subset_frame(metrics), caption=f"RMSE on {test} by subset, log and FOS scale. "
                "The decision range [0.5, 2) carries the primary criterion (log scale).",
                label=f"tab:{stem}:subsets"))
            for family in metrics["arms"]:
                for key, tag in (("rmse", "zones"), ("log_rmse", "zones_log")):
                    scale = "log FOS" if key == "log_rmse" else "FOS"
                    _save_table(tables / f"{stem}__{tag}_{family}.tex", booktabs(
                        zone_frame(metrics, family, key), caption=f"Zone RMSE ({scale}) on {test}, {family}.",
                        label=f"tab:{stem}:{tag}:{family}"))
                    _save_figure(figures / f"{stem}__{tag}_{family}.pdf", fig_zone_rmse(metrics, family, key))

    if len(config.ACTIVE_SCORINGS) > 1:
        by_test = {test: {s: load_metrics(test, s) for s in config.ACTIVE_SCORINGS
                          if config.metrics_path(test, s).exists()} for test in config.TEST_ROLES}
        complete = {t: m for t, m in by_test.items() if len(m) == len(config.ACTIVE_SCORINGS)}
        for test, by_scoring in complete.items():
            _save_table(tables / f"{test}__scorings.tex", booktabs(
                scoring_frame(by_scoring), caption=f"Selection-criterion sensitivity on {test}: "
                "global RMSE, minimax and within-family rank under each scoring.",
                label=f"tab:{test}:scorings"))
        if complete:
            _save_table(tables / "scorings_params.tex", booktabs(
                scoring_params_frame(next(iter(complete.values()))),
                caption="Hyper-parameters selected by CV under each scoring.",
                label="tab:scorings:params"))

    if config.learning_curve_metrics_path().exists():
        lc = load_learning_curve()
        for test in config.TEST_ROLES:
            for family in sorted({r["family"] for r in lc["rows"]}):
                _save_figure(figures / f"learning_curve__{test}__{family}.pdf",
                             fig_learning_curve(lc, test, family))
    if config.replication_metrics_path().exists():
        rep = load_replications()
        for test in config.TEST_ROLES:
            _save_table(tables / f"replications__{test}__values.tex", booktabs(
                replication_value_frame(rep, test, "decision-range"),
                caption=f"Log-RMSE on the decision range, {test}, every realization; r0 = existing sets "
                        "(exploratory). Common fixed hyper-parameters.", label=f"tab:rep:{test}:values"))
            _save_table(tables / f"replications__{test}__comparisons.tex", booktabs(
                replication_comparison_frame(rep, test),
                caption=f"Replicated comparisons on {test}: Welch (independent arms) or paired intervals, "
                        "log-RMSE and relative scale; P(first<second) is descriptive.",
                label=f"tab:rep:{test}:comparisons"))
            for family in config.REP_FAMILIES:
                _save_figure(figures / f"replications__{test}__{family}.pdf", fig_replication_bins(rep, test, family))
        _save_table(tables / "replications__interactions.tex", booktabs(
            replication_interaction_frame(rep), caption="Learner interaction: Welch interval on "
            "log(log-RMSE_rf) - log(log-RMSE_gbm) between arms.", label="tab:rep:interactions"))
    print(f"[report] wrote {tables} and {figures}")
