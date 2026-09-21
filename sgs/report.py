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


# --- structural (Task 5): the figure, the heatmaps and the tables -----------

STRUCTURAL_CELLS = (("gbm", "test-natural"), ("gbm", "test-stratified"),
                    ("rf", "test-natural"), ("rf", "test-stratified"))

# Explicit, because errorbar(fmt="none") and scatter do not advance the property
# cycle in step: in a two-arm loop the second arm's bars keep the first arm's colour.
STRUCTURAL_COLORS = {"C-log": "tab:blue", "C-seq": "tab:orange"}


def load_structural() -> dict:
    return _read(config.structural_metrics_path())


def _structural_errorbar(ax, rows, colour, *, x_key="bin", label=None):
    x = [r[x_key] for r in rows]
    y = [r["diff"] for r in rows]
    lo = [r["diff"] - r["ci"][0] for r in rows]
    hi = [r["ci"][1] - r["diff"] for r in rows]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", elinewidth=0.8, capsize=2, ecolor=colour)
    solid = [(xi, yi) for xi, yi, r in zip(x, y, rows) if not r.get("marked", False)]
    hollow = [(xi, yi) for xi, yi, r in zip(x, y, rows) if r.get("marked", False)]
    if solid:
        ax.scatter(*zip(*solid), s=14, color=colour, label=label, zorder=3)
    if hollow:
        # same colour, hollow face: a marked bin keeps its arm's identity
        ax.scatter(*zip(*hollow), s=14, facecolors="none", edgecolors=colour, zorder=3)


def fig_structural(struct: dict):
    """One density panel above a 2x2 grid of error panels (family x test),
    every arm keeping one colour for both its bars and its markers."""
    labels = struct["bins"]["labels"]
    arms = sorted({r["arm"] for r in struct["errors"]})
    fig = plt.figure(figsize=(9.0, 7.5))
    grid = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.25)

    top = fig.add_subplot(grid[0, :])
    for arm in arms:
        rows = [r for r in struct["densities"] if r["arm"] == arm]
        _structural_errorbar(top, rows, STRUCTURAL_COLORS[arm], label=arm)
    top.axhline(0.0, linewidth=0.8, color="black")
    top.set_ylabel("Δ training points vs B")
    top.set_xticks(range(len(labels)))
    top.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    top.legend(fontsize=7, frameon=False)

    for i, (family, test) in enumerate(STRUCTURAL_CELLS):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        for arm in arms:
            rows = [r for r in struct["errors"]
                    if r["family"] == family and r["test"] == test and r["arm"] == arm]
            _structural_errorbar(ax, rows, STRUCTURAL_COLORS[arm], label=arm)
        ax.axhline(0.0, linewidth=0.8, color="black")
        ax.set_title(f"{family.upper()}, {test}", fontsize=8)
        ax.set_ylabel("Δ log-RMSE vs B", fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=5)
        if i // 2 == 0:                       # once per test column
            counts = {r["bin"]: r["test_n"] for r in struct["errors"]
                      if r["family"] == family and r["test"] == test}
            for b, n in sorted(counts.items()):
                ax.annotate(str(n), (b, ax.get_ylim()[1]), fontsize=4, ha="center", va="bottom",
                            color="dimgray", annotation_clip=False)
    return fig


def fig_structural_matrix(struct: dict, arm: str):
    """Predicted bin against true bin, row-normalized, pooled over the
    confirmatory realizations only.

    This is where compensating errors live: two points swapping bins leave
    achieved - intended untouched and show up here as off-diagonal mass.
    """
    labels = struct["bins"]["labels"]
    confirmatory = [str(r) for r in struct["confirmatory"]]
    stacked = np.array([struct["displacement"][r][arm]["matrix"] for r in confirmatory],
                       dtype=float).sum(axis=0)
    totals = stacked.sum(axis=1, keepdims=True)
    shares = np.divide(stacked, totals, out=np.zeros_like(stacked), where=totals > 0)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    image = ax.imshow(shares, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("true bin")
    ax.set_ylabel("predicted bin")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xticks(range(len(labels) + 2))
    # > 6, not >= 6: this matrix is built with _true_bin, the generator's (lo, hi]
    ax.set_xticklabels([*labels, "< 0.5", "> 6"], rotation=60, ha="right", fontsize=5)
    ax.set_title(f"{arm}: where the selected points landed, r1-r5 pooled", fontsize=9)
    fig.colorbar(image, ax=ax, label="share of the predicted bin")
    fig.tight_layout()
    return fig


def structural_allocation_frame(struct: dict, arm: str) -> pd.DataFrame:
    """Mean and sd over the confirmatory realizations, with r0 on its own rows."""
    labels = struct["bins"]["labels"]
    confirmatory = [str(r) for r in struct["confirmatory"]]
    frames = []
    for name, keys in (("r1-r5", confirmatory), ("r0 (exploratory)", ["0"])):
        block = {"set": name, "bin": labels}
        for column in ("seed", "intended", "achieved", "deviation"):
            stacked = np.array([struct["allocation"][k][arm][column] for k in keys], dtype=float)
            block[column] = stacked.mean(axis=0)
            block[f"{column} sd"] = (stacked.std(axis=0, ddof=1) if len(keys) > 1
                                     else np.full(len(labels), np.nan))
        frame = pd.DataFrame(block)
        # The set total goes on its own row. Broadcasting it down the bin column
        # would read as a per-bin quantity, which it is not.
        totals = np.array([struct["allocation"][k][arm]["abs_deviation"] for k in keys],
                          dtype=float)
        total_row = {column: np.nan for column in frame.columns}
        total_row.update({"set": name, "bin": "Σ|deviation| (whole set)",
                          "deviation": totals.mean(),
                          "deviation sd": totals.std(ddof=1) if len(keys) > 1 else np.nan})
        frames.append(pd.concat([frame, pd.DataFrame([total_row])], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def structural_batch_frame(struct: dict) -> pd.DataFrame:
    """C-seq batch by batch and cumulatively, averaged over r1-r5, r0 apart."""
    labels = struct["bins"]["labels"]
    confirmatory = [str(r) for r in struct["confirmatory"]]
    rows = []
    for name, keys in (("r1-r5", confirmatory), ("r0 (exploratory)", ["0"])):
        n_batches = len(struct["batches"][keys[0]])
        for index in range(n_batches):
            entries = [struct["batches"][k][index] for k in keys]
            for b, label in enumerate(labels):
                row = {"set": name, "batch": entries[0]["batch"], "bin": label}
                for column, key in (("intended", "intended"), ("achieved", "achieved"),
                                    ("cum. intended", "cumulative_intended"),
                                    ("cum. achieved", "cumulative_achieved")):
                    series = np.array([e[key][b] for e in entries], dtype=float)
                    row[column] = float(series.mean())
                    # the spread is the whole point of having five realizations
                    row[f"{column} sd"] = (float(series.std(ddof=1)) if len(entries) > 1
                                           else float("nan"))
                rows.append(row)
    return pd.DataFrame(rows)


def structural_displacement_frame(struct: dict) -> pd.DataFrame:
    """Mean over the confirmatory realizations, with r0 kept as its own rows."""
    confirmatory = {str(r) for r in struct["confirmatory"]}
    rows = []
    for realization, per_arm in struct["displacement"].items():
        if realization != "0" and realization not in confirmatory:
            continue
        for arm, row in per_arm.items():
            rows.append({"set": "r0 (exploratory)" if realization == "0" else "r1-r5",
                         "arm": arm, "n": row["n"], "stayed": row["stayed"],
                         "mean |Δbin|": row["mean_abs_shift"],
                         "median |Δbin|": row["median_abs_shift"], "outside": row["outside"]})
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["arm", "set"], as_index=False)
    means = grouped.mean(numeric_only=True)
    spreads = grouped.std(numeric_only=True, ddof=1).drop(columns=["arm", "set"])
    spreads.columns = [f"{column} sd" for column in spreads.columns]
    return pd.concat([means, spreads], axis=1).sort_values(["arm", "set"])


def structural_outside_frame(struct: dict) -> pd.DataFrame:
    rows = [{"kind": r["kind"], "arm": r["arm"], "side": r["side"],
             "family": r.get("family", "—"), "test": r.get("test", "—"),
             "diff": r["diff"], "ci lo": r["ci"][0], "ci hi": r["ci"][1]}
            for r in struct["outside"]]
    return pd.DataFrame(rows).sort_values(["kind", "arm", "side"])


# --- structural (Task 5 addendum): the C0 ablations --------------------------
# Amendment 1, docs/predictions-and-outcomes.md. None of this enters the main
# figure above or its concordance count; every row here joins arms trained on
# different budgets, and both the panels and the tables say so.

# Explicit by comparison, not by arm: a single panel already draws both
# comparisons for one C0 arm, so a property cycle would conflate them.
STRUCTURAL_ABLATION_COLORS = {"welch": "tab:red", "paired": "tab:purple"}


def _ablation_budget_sentence(budgets: dict) -> str:
    parts = ", ".join(f"{arm} ({n})" for arm, n in budgets.items())
    return f"Budgets: {parts}. Every comparison above joins arms trained on different budgets."


def fig_structural_ablation(struct: dict):
    """One row per C0 arm, one column per family x test cell: the C0 - B
    (Welch) and C - C0 (paired) differences per reporting bin, with the two
    bins outside the selection range shaded apart from the fifteen inside it."""
    ablation = struct["ablation"]
    labels = ablation["labels"]
    budgets = ablation["budgets"]
    arms = sorted({r["arm"] for r in ablation["errors"]})
    fig, axes = plt.subplots(len(arms), len(STRUCTURAL_CELLS),
                             figsize=(11.0, 2.8 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        c_arm = next(r["first"] for r in ablation["errors"]
                    if r["arm"] == arm and r["interval"] == "paired")
        for j, (family, test) in enumerate(STRUCTURAL_CELLS):
            ax = axes[i][j]
            cell = [r for r in ablation["errors"]
                    if r["arm"] == arm and r["family"] == family and r["test"] == test]
            welch = sorted((r for r in cell if r["interval"] == "welch"),
                           key=lambda r: r["reporting_bin"])
            paired = sorted((r for r in cell if r["interval"] == "paired"),
                            key=lambda r: r["reporting_bin"])
            _structural_errorbar(ax, welch, STRUCTURAL_ABLATION_COLORS["welch"],
                                 x_key="reporting_bin", label=f"{arm} ({budgets[arm]}) - B ({budgets['B']})")
            _structural_errorbar(ax, paired, STRUCTURAL_ABLATION_COLORS["paired"],
                                 x_key="reporting_bin",
                                 label=f"{c_arm} ({budgets[c_arm]}) - {arm} ({budgets[arm]})")
            ax.axhline(0.0, linewidth=0.8, color="black")
            # the two bins outside the selection range, set apart visually
            ax.axvspan(-0.5, 0.5, color="lightgrey", alpha=0.3, zorder=0)
            ax.axvspan(len(labels) - 1.5, len(labels) - 0.5, color="lightgrey", alpha=0.3, zorder=0)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=4)
            if i == 0:
                ax.set_title(f"{family.upper()}, {test}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{arm} ({budgets[arm]}) vs {c_arm} ({budgets[c_arm]})/B ({budgets['B']})",
                             fontsize=6)
        axes[i][0].legend(fontsize=5, frameon=False)
    fig.suptitle("C0 ablations: different budgets throughout (Δ log-RMSE)", fontsize=9)
    fig.tight_layout()
    return fig


def structural_ablation_fraction_frame(struct: dict) -> pd.DataFrame:
    """Per C0 arm and reporting bin: the mean C0 fraction, the mean B
    fraction, and their difference. Fractions only, never raw counts, which
    would carry the budget difference between C0 (300) and B (600)."""
    rows = [{"arm": r["arm"], "bin": r["label"], "C0 fraction": r["first_mean"],
             "B fraction": r["second_mean"], "diff": r["diff"],
             "ci lo": r["ci"][0], "ci hi": r["ci"][1]}
            for r in struct["ablation"]["fractions"]]
    return pd.DataFrame(rows)


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
                        "(exploratory). Common fixed hyper-parameters. C0-log and C0-seq are ablations "
                        "without the seed and carry no budget: they train on 300 points against the 600 "
                        "of B, C-log and C-seq.", label=f"tab:rep:{test}:values"))
            _save_table(tables / f"replications__{test}__comparisons.tex", booktabs(
                replication_comparison_frame(rep, test),
                caption=f"Replicated comparisons on {test}: Welch (independent arms) or paired intervals, "
                        "log-RMSE and relative scale; P(first<second) is descriptive. C0-log and C0-seq "
                        "are ablations without the seed and carry no budget: they train on 300 points "
                        "against the 600 of B, C-log and C-seq.",
                label=f"tab:rep:{test}:comparisons"))
            for family in config.REP_FAMILIES:
                _save_figure(figures / f"replications__{test}__{family}.pdf", fig_replication_bins(rep, test, family))
        _save_table(tables / "replications__interactions.tex", booktabs(
            replication_interaction_frame(rep), caption="Learner interaction: Welch interval on "
            "log(log-RMSE_rf) - log(log-RMSE_gbm) between arms.", label="tab:rep:interactions"))

    if config.structural_metrics_path().exists():
        struct = load_structural()
        shared_note = struct["ablation"]["shared_rows"]["note"]
        _save_figure(figures / "structural__bins.pdf", fig_structural(struct))
        _save_table(tables / "structural__caption.tex", struct["figure_caption"] + "\n")
        for arm in ("C-log", "C-seq"):
            _save_figure(figures / f"structural__matrix_{arm.lower()}.pdf",
                         fig_structural_matrix(struct, arm))
            _save_table(tables / f"structural__allocation_{arm.lower()}.tex", booktabs(
                structural_allocation_frame(struct, arm),
                caption=f"Allocation fidelity for {arm}: mean and sd over the confirmatory "
                        "realizations, r0 apart and exploratory. 'deviation' is achieved minus "
                        "intended, the net distortion of the requested bin-count distribution "
                        "and not the screener's prediction error. Selected equals intended by "
                        "construction and is not shown.",
                label=f"tab:structural:allocation:{arm.lower()}") + "\n" + shared_note + "\n")
        _save_table(tables / "structural__batches.tex", booktabs(
            structural_batch_frame(struct),
            caption="C-seq batch by batch and cumulatively, averaged over the confirmatory "
                    "realizations with r0 apart. Each batch reallocates against what the earlier "
                    "batches actually returned; the cumulative columns show whether retraining "
                    "corrects a deviation or moves it.",
            label="tab:structural:batches") + "\n" + shared_note + "\n")
        _save_table(tables / "structural__displacement.tex", booktabs(
            structural_displacement_frame(struct),
            caption="The screener on the points it selected: share staying in the predicted bin, "
                    "absolute displacement in bins, share landing outside [0.5, 6]. Conditioned "
                    "on selection, so this is its accuracy where it was used, not over the pool. "
                    "The displacement columns cover only points whose true FOS stayed inside the "
                    "selection range; the rest are counted by 'outside'.",
            label="tab:structural:displacement") + "\n" + shared_note + "\n")
        _save_table(tables / "structural__outside.tex", booktabs(
            structural_outside_frame(struct),
            caption="Outside the selection range, where nothing was allocated. 'density' rows are "
                    "the spillover itself: training points that landed there although the gate is "
                    "on a predicted FOS. 'error' rows are the test error in those bins. "
                    "Differences against B with descriptive Welch intervals. Both sides are read "
                    "from the replication metrics, so both use the half-open reporting zones and "
                    "the outer bin is FOS >= 6.",
            label="tab:structural:outside"))
        # Task 5 addendum (Amendment 1): the C0 ablations. Supplementary; they
        # never enter the main figure or the concordance count above.
        _save_figure(figures / "structural__ablation.pdf", fig_structural_ablation(struct))
        _save_table(tables / "structural__ablation_fractions.tex", booktabs(
            structural_ablation_fraction_frame(struct),
            caption="C0 training-distribution fractions n_bin / N_total against B (Welch), by "
                    "reporting bin. Fractions only, never raw counts, because C0 trains on a "
                    "different budget (300) than B (600).",
            label="tab:structural:ablation:fractions"))
        _save_table(tables / "structural__ablation_caption.tex",
                    struct["ablation"]["procedure"] + "\n\n"
                    + _ablation_budget_sentence(struct["ablation"]["budgets"]) + "\n")
        _save_table(tables / "structural__original_c0.tex", struct["ablation"]["original_c0"] + "\n")
    else:
        print("[report] structural.json missing; run sgs structural")
    print(f"[report] wrote {tables} and {figures}")
