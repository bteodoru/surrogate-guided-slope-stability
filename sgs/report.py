"""Booktabs tables and figures for the manuscript.

Reads only outputs/metrics/ and recomputes no metric, so no number reaches a
table without having passed through ``evaluate``. Every arm appears in every
results table, C0 included, written C$_0$ as in the manuscript and named as an
ablation in the caption; the budget table lists budget arms only. Tables fit the
text width; the comparison inventories are longtables for the Supplementary.
"""

from __future__ import annotations

import json
import math
import re

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

from sgs import config, data

_C0 = re.compile(r"\bC0\b")


def arm_label(arm: str) -> str:
    """The arm as the manuscript writes it, C₀ with a subscript; the ablations
    are named once in each caption (`ablation_note`), not in every cell."""
    return _C0.sub("C₀", arm)


def _legend_label(arm: str) -> str:
    """The figures' legend, unchanged: figures are revised in a separate pass."""
    return f"{arm} (ablation)" if config.ARMS[arm].ablation else arm


def _join(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def ablation_note(arms=None) -> str:
    """The sentence that marks the ablations in a caption, once."""
    names = [arm_label(a) for a in (arms or [a for a, s in config.ARMS.items() if s.ablation])]
    if len(names) == 1:
        return f"{names[0]} is an ablation: it trains on the added points without the seed and carries no budget."
    return f"{_join(names)} are ablations: they train on the added points without the seed and carry no budget."


def _read(path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs evaluate`")
    return json.loads(path.read_text())


def load_metrics(test: str, scoring: str) -> dict:
    return _read(config.metrics_path(test, scoring))


def load_learning_curve() -> dict:
    return _read(config.learning_curve_metrics_path())


class Tex(str):
    """A header or cell already written in LaTeX: never escaped."""


def _escape(text) -> str:
    if isinstance(text, Tex):
        return str(text)
    text = _C0.sub("C₀", str(text))
    for raw, safe in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(raw, safe)
    # the manuscript's C$_0$, not a Unicode subscript the text font may lack
    return text.replace("₀", "$_0$")


def _cell(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return _escape(value)


SMALL = (r"\small", r"\setlength{\tabcolsep}{4pt}")


def _columns(names) -> pd.Index:
    """Column labels kept as given: a (group, name) pair stays one label, not a MultiIndex level."""
    return pd.Index(list(names), dtype=object, tupleize_cols=False)


def _frame(rows: list[dict]) -> pd.DataFrame:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    return pd.DataFrame([[r.get(c) for c in columns] for r in rows], columns=_columns(columns))


def regroup(frame: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """The columns of `frame` named in `columns`, in that order, relabelled
    (a label may be a (group, name) pair). Selects and renames only: no value changes."""
    out = frame[list(columns)].copy()
    out.columns = _columns(columns.values())
    return out


def _header(columns) -> list[str]:
    """One header row; two when a column is a (group, name) pair: the group
    spans its columns and is underlined with \\cmidrule."""
    groups = [c[0] if isinstance(c, tuple) else None for c in columns]
    names = [c[1] if isinstance(c, tuple) else c for c in columns]
    lines = []
    if any(groups):
        top, rules, i = [], [], 0
        while i < len(columns):
            j = i
            while groups[i] is not None and j + 1 < len(columns) and groups[j + 1] == groups[i]:
                j += 1
            if groups[i] is None:
                top.append("")
            else:
                top.append(rf"\multicolumn{{{j - i + 1}}}{{c}}{{{_escape(groups[i])}}}")
                rules.append(rf"\cmidrule(lr){{{i + 1}-{j + 1}}}")
            i = j + 1
        lines += [" & ".join(top) + r" \\", " ".join(rules)]
    lines.append(" & ".join(_escape(n) for n in names) + r" \\")
    return lines


def _body(frame: pd.DataFrame, digits: int, block: str | None) -> tuple[list, list[str]]:
    """The rows; with `block`, that column leaves the table and its value heads
    each run of rows instead (a learner family: RF, GBM)."""
    columns = [c for c in frame.columns if c != block]
    positions = [list(frame.columns).index(c) for c in columns]
    where = list(frame.columns).index(block) if block is not None else None
    lines, current = [], object()
    for row in frame.itertuples(index=False):
        if where is not None and row[where] != current:
            current = row[where]
            name = current.upper() if current in config.FAMILIES else current
            if lines:
                lines.append(r"\addlinespace")
            lines.append(rf"\multicolumn{{{len(columns)}}}{{l}}{{\textit{{{_escape(name)}}}}} \\")
        lines.append(" & ".join(_cell(row[p], digits) for p in positions) + r" \\")
    return columns, lines


def _spec(n: int, left: int) -> str:
    return "l" * left + "r" * (n - left)


def _tabular(frame: pd.DataFrame, *, digits: int = 3, block: str | None = None, left: int = 1) -> list[str]:
    columns, body = _body(frame, digits, block)
    return [rf"\begin{{tabular}}{{{_spec(len(columns), left)}}}", r"\toprule", *_header(columns),
            r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]


def booktabs(frame: pd.DataFrame, *, caption: str, label: str, digits: int = 3,
             small: bool = False, block: str | None = None, left: int = 1) -> str:
    lines = [r"\begin{table}[htbp]", r"\centering", *(SMALL if small else ()),
             rf"\caption{{{_escape(caption)}}}", rf"\label{{{label}}}",
             *_tabular(frame, digits=digits, block=block, left=left), r"\end{table}"]
    return "\n".join(lines) + "\n"


def booktabs_panels(panels, *, caption: str, label: str, digits: int = 3, small: bool = False) -> str:
    """Several tabulars under one caption and label, stacked; each panel is
    (title or None, frame, keyword arguments of `_tabular`)."""
    lines = [r"\begin{table}[htbp]", r"\centering", *(SMALL if small else ()),
             rf"\caption{{{_escape(caption)}}}", rf"\label{{{label}}}"]
    for i, (title, frame, kwargs) in enumerate(panels):
        if i:
            lines.append(r"\par\medskip")
        if title:
            lines += [rf"\textit{{{_escape(title)}}}\par\smallskip"]
        lines += _tabular(frame, **{"digits": digits, **kwargs})
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def booktabs_split(frame: pd.DataFrame, by: str, *, caption: str, label: str,
                   part=lambda value: f"This table: {value}.", **kwargs) -> str:
    """One float per value of `by`, in order of appearance; `by` leaves the table and
    the caption names the part. A column that is '--' throughout a part is dropped there."""
    parts = []
    for value in dict.fromkeys(frame[by]):
        rows = frame[frame[by] == value].drop(columns=by)
        rows = rows.loc[:, [c for c in rows.columns if not (rows[c].astype(str) == "--").all()]]
        parts.append(booktabs(rows, caption=f"{caption} {part(value)}", label=f"{label}:{value}", **kwargs))
    return "".join(parts)


def longtable(frame: pd.DataFrame, *, caption: str, label: str, digits: int = 3,
              small: bool = True, block: str | None = None, left: int = 1, colsep: int = 4) -> str:
    """An inventory for the Supplementary material: it may run over several
    pages, never wider than the text; the head repeats on every page."""
    columns, body = _body(frame, digits, block)
    head, n = _header(columns), len(columns)
    lines = [r"\begingroup", *((r"\small", rf"\setlength{{\tabcolsep}}{{{colsep}pt}}") if small else ()),
             r"\setlength{\LTcapwidth}{\textwidth}",
             rf"\begin{{longtable}}{{{_spec(n, left)}}}",
             rf"\caption{{{_escape(caption)}}}\label{{{label}}} \\",
             r"\toprule", *head, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{{n}}}{{l}}{{\textit{{Table \thetable, continued}}}} \\",
             r"\toprule", *head, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{n}}}{{r}}{{\textit{{continued on the next page}}}} \\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot",
             *body, r"\end{longtable}", r"\endgroup"]
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


# Short headers for the hyper-parameters, spelled out in the caption.
PARAM_HEADERS = {"max_features": "max feat.", "min_samples_leaf": "min leaf", "learning_rate": "rate",
                 "max_iter": "iter.", "max_leaf_nodes": "leaves"}


# Glosses for a header whose values could be misread: max_features=1.0 is every feature, not one.
PARAM_NOTES = {"max_features": "a fraction of the features, 1.0 = all"}


def _param(value) -> str:
    """An integer as an integer; a float keeps its point, so 1.0 never reads as the integer 1."""
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def scoring_params_frame(by_scoring: dict[str, dict], family: str) -> pd.DataFrame:
    """Hyper-parameters of one family selected under each scoring, one column per
    hyper-parameter; they do not depend on the test set."""
    arms = {scoring: metrics["arms"][family] for scoring, metrics in by_scoring.items()}
    names = sorted({k for a in arms.values() for m in a.values() for k in m["best_params"]})
    rows = []
    for arm in config.ARMS:
        row = {"arm": arm_label(arm)}
        for scoring, by_arm in arms.items():
            params = by_arm[arm]["best_params"]
            for name in names:
                row[(scoring, PARAM_HEADERS.get(name, name))] = _param(params[name]) if name in params else None
        rows.append(row)
    return _frame(rows)


def param_legend(by_scoring: dict[str, dict]) -> str:
    """'rate: learning_rate; ...' for the headers actually used."""
    names = sorted({k for metrics in by_scoring.values() for arms in metrics["arms"].values()
                    for m in arms.values() for k in m["best_params"]})
    return "; ".join(f"{PARAM_HEADERS[n]}: {n}" + (f", {PARAM_NOTES[n]}" if n in PARAM_NOTES else "")
                     for n in names if n in PARAM_HEADERS)


def budget_frame() -> pd.DataFrame:
    return pd.DataFrame([{"arm": arm, "training sets": " + ".join(spec.roles), "solver budget": spec.budget}
                         for arm, spec in config.ARMS.items() if not spec.ablation])


def fig_zone_rmse(metrics: dict, family: str, key: str = "rmse"):
    frame = zone_frame(metrics, family, key)
    arms = list(config.ARMS)
    x = np.arange(len(frame))
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, arm in enumerate(arms):
        ax.bar(x + i * width, frame[arm_label(arm)].astype(float), width, label=_legend_label(arm))
    ax.set_xticks(x + width * (len(arms) - 1) / 2, frame["zone"])
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
                          "contrast": f"{i['first']} vs {i['second']}", "method": i["method"],
                          "diff": i["diff"], "CI low": i["ci"][0], "CI high": i["ci"][1], "df": i["df"]}
                         for i in rep["interactions"] if i["measure"] in REPLICATION_MEASURES])


def oracle_reading_frame(rep: dict) -> pd.DataFrame:
    """The declared oracle reading: per cell, the medians and the paired intervals of
    dR_O, dL_O and dH_O (all three are declared in spec §9.2 and §10), then one summary
    row with the reading, the attribution flag, group C and the levels."""
    o = rep["oracle"]
    ci = lambda iv: f"[{iv['ci'][0]:+.3f}, {iv['ci'][1]:+.3f}]"
    rows = [{"cell": cell, "median L C-oracle": c["L_o"], "median H C-oracle": c["H_o"],
             "median L C-logP": c["L_p"], "median H C-logP": c["H_p"], "median dR_O": c["dR"],
             "dR_O CI low": c["dR_interval"]["ci"][0], "dR_O CI high": c["dR_interval"]["ci"][1],
             "median dL_O": c["dL"], "dL_O CI": ci(c["dL_interval"]),
             "median dH_O": c["dH"], "dH_O CI": ci(c["dH_interval"])}
            for cell, c in o["cells"].items()]
    rows.append({"cell": "reading", "median L C-oracle": f"gate {'passes' if o['oracle_gate'] else 'fails'}",
                 "median L C-logP": f"gate {'passes' if o['step_p']['logp_gate'] else 'fails'}",
                 "median dR_O": o["row"],
                 "dR_O CI low": "attributable" if o["attribution"] else "no attribution",
                 "dR_O CI high": o["c_row"],
                 "median dL_O": f"rho_L {o['rho']['L']:.3f}", "median dH_O": f"rho_H {o['rho']['H']:.3f}"})
    return pd.DataFrame(rows)


def fig_replication_bins(rep: dict, test: str, family: str):
    """Diagnostic: log-RMSE per design bin (confirmatory mean, min-max across
    realizations), test points per bin and the mean training count per bin."""
    arms = ("B", "C-log", "C-seq", "C-logG", "C-oracle", "C-logP",
           "C0-log", "C0-seq", "C0-logG", "C0-oracle", "C0-logP")
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


# --- the primary criterion as a forest plot ---------------------------------
# Every row against B, log scale, Welch over r1-r5. A, C and the original C0 have
# no replications and no interval, so they are not rows; r0 enters no interval,
# so it is not drawn.

# (title, second line or None, arms); the C0 block names the budgets it joins.
FOREST_BLOCKS = (("Main-text arms", None, ("C-log", "C-oracle", "C-logP")),
                 ("Sensitivity arms (Supplementary)", None, ("C-seq", "C-logG")),
                 ("C₀ ablations", "different budgets: 300 vs 600 points",
                  ("C0-log", "C0-oracle", "C0-logP", "C0-seq", "C0-logG")))
FOREST_MEASURES = (("decision-range", "Decision range [0.5, 2)"),
                   ("0.5–1", "Lower half [0.5, 1)"), ("1–2", "Upper half [1, 2)"))
FOREST_CELLS = (("rf", "test-natural"), ("rf", "test-stratified"),
                ("gbm", "test-natural"), ("gbm", "test-stratified"))

# Okabe-Ito, colour-blind safe; the family is also carried by the marker shape
# and the test set by the fill, so neither depends on colour.
FOREST_FAMILY_STYLE = {"rf": ("o", "#0072B2"), "gbm": ("s", "#D55E00")}
FOREST_HOLLOW = {"test-natural": False, "test-stratified": True}


def forest_frame(rep: dict) -> pd.DataFrame:
    """The rows of the forest plot: each declared arm against B, log scale, on the
    decision range and its two halves, in the four learner x test cells. A row
    missing from the metrics, duplicated or not a Welch interval is refused."""
    found = {}
    for c in rep["comparisons"]:
        if c["second"] == "B" and c["scale"] == "log-rmse":
            key = (c["first"], c["measure"], c["family"], c["test"])
            if key in found:
                raise ValueError(f"comparison {key} vs B appears twice in the replication metrics")
            found[key] = c
    rows, missing = [], []
    for block, _, arms in FOREST_BLOCKS:
        for arm in arms:
            for measure, _ in FOREST_MEASURES:
                for family, test in FOREST_CELLS:
                    c = found.get((arm, measure, family, test))
                    if c is None:
                        missing.append((arm, measure, family, test))
                        continue
                    if c["method"] != "welch":
                        raise ValueError(f"{arm} vs B on {measure}, {family}, {test} is not a Welch interval")
                    rows.append({"block": block, "arm": arm, "measure": measure, "family": family,
                                 "test": test, "diff": c["diff"], "ci low": c["ci"][0],
                                 "ci high": c["ci"][1], "level": c["level"]})
    if missing:
        raise KeyError(f"the forest plot needs comparisons absent from the replication metrics: {missing}")
    return pd.DataFrame(rows)


def fig_forest(rep: dict):
    """Arm minus B, log-RMSE, on the decision range and its two halves; one row
    per arm in three blocks, the four cells as dodged points in each row. Sized
    for the DCE text width (144 mm), so it is included at \\textwidth unscaled."""
    frame = forest_frame(rep)
    dodge = {cell: (i - (len(FOREST_CELLS) - 1) / 2) * 0.19 for i, cell in enumerate(FOREST_CELLS)}
    # one unit per arm, one more above each block for its title (two lines take more)
    rows, titles, starts, y = {}, [], [], 0.0
    for block, note, arms in FOREST_BLOCKS:
        height = 1.0 if note is None else 1.5
        starts.append(y)
        titles.append((block, note, y))
        y += height
        for arm in arms:
            rows[arm] = y
            y += 1.0
    fig, axes = plt.subplots(1, len(FOREST_MEASURES), sharey=True, figsize=(5.67, 5.4))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.93, bottom=0.165, wspace=0.12)
    for ax, (measure, title) in zip(axes, FOREST_MEASURES):
        for i, yi in enumerate(rows.values()):
            if i % 2 == 0:
                ax.axhspan(yi - 0.5, yi + 0.5, color="0.94", zorder=0, linewidth=0)
        for (family, test), dy in dodge.items():
            cell = frame[(frame["measure"] == measure) & (frame["family"] == family) & (frame["test"] == test)]
            marker, colour = FOREST_FAMILY_STYLE[family]
            ys = [rows[a] + dy for a in cell["arm"]]
            ax.errorbar(cell["diff"], ys, xerr=[cell["diff"] - cell["ci low"], cell["ci high"] - cell["diff"]],
                        fmt="none", ecolor=colour, elinewidth=0.9, capsize=0, zorder=2)
            ax.scatter(cell["diff"], ys, marker=marker, s=14, linewidths=0.9, edgecolors=colour,
                       facecolors="white" if FOREST_HOLLOW[test] else colour, zorder=3)
        ax.axvline(0.0, color="black", linewidth=0.8, zorder=1)
        for start in starts[1:]:
            ax.axhline(start - 0.5, color="0.55", linewidth=0.6)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis="x", color="0.85", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(y - 0.5, -0.5)
    axes[0].set_yticks(list(rows.values()), [arm_label(a) for a in rows], fontsize=7.5)
    edge = blended_transform_factory(fig.transFigure, axes[0].transData)
    # from the figure's left edge, a title may run past the label column and over the zero line
    behind = {"facecolor": "white", "edgecolor": "none", "pad": 0.6}
    for block, note, yt in titles:
        axes[0].text(0.01, yt, block, transform=edge, fontsize=7.5, fontweight="bold",
                     va="center", ha="left", bbox=behind, zorder=4)
        if note is not None:
            axes[0].text(0.01, yt + 0.5, note, transform=edge, fontsize=7, va="center", ha="left",
                         bbox=behind, zorder=4)
    realizations = f"r{rep['confirmatory'][0]}–r{rep['confirmatory'][-1]}"
    fig.supxlabel(f"Mean log-RMSE over {realizations}: arm − B (negative: smaller than B)", fontsize=8, y=0.075)
    handles = [plt.Line2D([], [], linestyle="none", marker=FOREST_FAMILY_STYLE[f][0], markersize=5,
                          markeredgewidth=0.9, markeredgecolor=FOREST_FAMILY_STYLE[f][1],
                          markerfacecolor="white" if FOREST_HOLLOW[t] else FOREST_FAMILY_STYLE[f][1],
                          label=f"{f.upper()}, {t}") for f, t in FOREST_CELLS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    return fig


def forest_caption(rep: dict) -> str:
    realizations = f"r{rep['confirmatory'][0]}-r{rep['confirmatory'][-1]}"
    return (
        "Primary criterion: log-RMSE against B on the decision range, FOS in [0.5, 2), and on its two "
        f"halves, [0.5, 1) and [1, 2). Each point is the mean over the confirmatory realizations "
        f"{realizations} of the arm's log-RMSE minus B's, with its 95% Welch interval; a negative "
        "difference means a smaller log-RMSE than B. Each row carries the four learner-by-test cells: "
        "circles RF, squares GBM; filled markers test-natural, hollow markers test-stratified. The rows "
        "share one x axis within each column; the columns have their own. The intervals cover "
        "variability between designs, conditional on the fixed seed, the fixed test sets and the common "
        "hyper-parameters fixed from the seed; they constitute no test. An interval containing 0 is "
        "inconclusive, never a tie or evidence of equivalence. C-seq and C-logG are sensitivity arms, "
        "reported in Supplementary. The C0 rows are ablations: they train on the 300 guided points "
        "without the seed, against the 600 of B, so they join different budgets and are not a "
        "like-for-like comparison. A is absent: it has no replications, only its existing set (r0), "
        "and therefore no interval; C and the original C0 are absent for the same reason, as they "
        "were never replicated. r0 is exploratory and enters no interval, so it is not drawn; "
        "C-oracle, C-logP and their C0 ablations have no r0. Bins are half-open, [lo, hi), as "
        "everywhere the replication metrics are read.")


# --- structural (Task 5): the figure, the heatmaps and the tables -----------

STRUCTURAL_CELLS = (("gbm", "test-natural"), ("gbm", "test-stratified"),
                    ("rf", "test-natural"), ("rf", "test-stratified"))

# Explicit, because errorbar(fmt="none") and scatter do not advance the property
# cycle in step: in a two-arm loop the second arm's bars keep the first arm's colour.
STRUCTURAL_COLORS = {"C-log": "tab:blue", "C-seq": "tab:orange", "C-logG": "tab:green",
                     "C-oracle": "tab:red", "C-logP": "tab:brown"}

# Arms drawn in one panel are dodged by this much, in bin units, so their bars
# and markers do not occlude each other.
STRUCTURAL_DODGE = 0.3


class StaleStructuralError(RuntimeError):
    """structural.json was built from another replications.json."""


def load_structural() -> dict:
    return _read(config.structural_metrics_path())


def check_structural_fresh(struct: dict) -> None:
    """Refuse to render an artifact whose inputs have since changed.

    Verifies hashes of all inputs stored in provenance: the replication metrics,
    the run metadata, and the design and final files of every guided set.
    """
    provenance = struct.get("provenance", {})
    if not provenance:
        raise StaleStructuralError("structural.json carries no provenance; rerun `sgs structural`")

    # Check replications.json
    recorded_rep = provenance.get("replications_sha256")
    if recorded_rep is None:
        raise StaleStructuralError("structural.json provenance missing replications_sha256; rerun `sgs structural`")
    path = config.replication_metrics_path()
    if not path.exists():
        raise StaleStructuralError(f"{path.name} is missing; structural.json cannot be verified")
    if data.sha256_file(path) != recorded_rep:
        raise StaleStructuralError(f"structural.json was built from another {path.name}; "
                                   "rerun `sgs structural`")

    # Check run_metadata.json
    recorded_meta = provenance.get("run_metadata_sha256")
    if recorded_meta is None:
        raise StaleStructuralError("structural.json provenance missing run_metadata_sha256; rerun `sgs structural`")
    meta_path = config.metadata_path()
    if not meta_path.exists():
        raise StaleStructuralError(f"{meta_path.name} is missing; structural.json cannot be verified")
    if data.sha256_file(meta_path) != recorded_meta:
        raise StaleStructuralError(f"structural.json was built from another {meta_path.name}; "
                                   "rerun `sgs structural`")

    # The design and final file of every guided set read for allocation and
    # displacement. The realization is stored with the hash, so the path does
    # not depend on how a label is spelled. The manifest itself must be
    # complete: a missing or shortened list would otherwise verify nothing.
    sets = provenance.get("sets_sha256")
    if not isinstance(sets, dict) or not sets:
        raise StaleStructuralError("structural.json provenance carries no set manifest; "
                                   "rerun `sgs structural`")
    for label, hashes in sets.items():
        if not isinstance(hashes, dict) or hashes.get("realization") is None:
            raise StaleStructuralError(f"provenance of {label} carries no realization; "
                                       "rerun `sgs structural`")
    # Per realization: one set per guided strategy and one per C-seq batch, r0 included.
    # The pool-backed strategies have no r0, and are expected only in an artifact
    # built after their sets exist: structural.build() writes the "oracle" block
    # exactly when it reads them, so the artifact itself says which manifest it carries.
    with_pool = "oracle" in struct

    def expected(r: int) -> int:
        return sum(len(config.REP_SEQ_BATCHES) if s == "guided-seq" else 1
                   for s in config.REP_STRATEGIES
                   if s != "lhs" and (s not in config.REP_POOL_STRATEGIES or (with_pool and r >= 1)))

    found = {r: sum(1 for h in sets.values() if h["realization"] == r)
             for r in range(config.REP_REALIZATIONS + 1)}
    unknown = sorted({h["realization"] for h in sets.values()} - set(found), key=str)
    short = {r: n for r, n in found.items() if n != expected(r)}
    if unknown or short:
        raise StaleStructuralError(
            f"structural.json set manifest is incomplete: expected "
            f"{ {r: expected(r) for r in found} } sets per realization, found {short or 'the right counts'}"
            f"{f' and unknown realizations {unknown}' if unknown else ''}; rerun `sgs structural`")
    for label, hashes in sets.items():
        realization = hashes["realization"]
        paths = ({"design": config.design_path(label), "final": config.final_path(label)}
                 if realization == 0 else
                 {"design": config.replication_path("designs", label),
                  "final": config.replication_path("final", label)})
        for kind, file_path in paths.items():
            recorded_hash = hashes.get(kind)
            if recorded_hash is None:
                raise StaleStructuralError(f"provenance of {label} carries no {kind} hash; "
                                           "rerun `sgs structural`")
            if not file_path.exists():
                raise StaleStructuralError(f"{label} {kind} file is missing; "
                                           "structural.json cannot be verified")
            if data.sha256_file(file_path) != recorded_hash:
                raise StaleStructuralError(f"structural.json was built from another {label} "
                                           f"{kind} file; rerun `sgs structural`")


def _offsets(names) -> dict:
    names = list(names)
    return {name: (i - (len(names) - 1) / 2) * STRUCTURAL_DODGE for i, name in enumerate(names)}


def _structural_errorbar(ax, rows, colour, *, x_key="bin", label=None, offset=0.0):
    x = [r[x_key] + offset for r in rows]
    y = [r["diff"] for r in rows]
    lo = [r["diff"] - r["ci"][0] for r in rows]
    hi = [r["ci"][1] - r["diff"] for r in rows]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", elinewidth=0.8, capsize=2, ecolor=colour)
    solid = [(xi, yi) for xi, yi, r in zip(x, y, rows) if not r.get("marked", False)]
    hollow = [(xi, yi) for xi, yi, r in zip(x, y, rows) if r.get("marked", False)]
    if solid:
        ax.scatter(*zip(*solid), s=16, color=colour, label=label, zorder=3)
    if hollow:
        # same colour, hollow face: a marked bin keeps its arm's identity
        ax.scatter(*zip(*hollow), s=16, facecolors="none", edgecolors=colour, zorder=3,
                   label=None if solid else label)


def fig_structural(struct: dict):
    """One density panel above a 2x2 grid of error panels (family x test),
    every arm keeping one colour for both its bars and its markers."""
    labels = struct["bins"]["labels"]
    arms = sorted({r["arm"] for r in struct["errors"]})
    offsets = _offsets(arms)
    fig = plt.figure(figsize=(10.0, 9.0))
    grid = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.22)

    top = fig.add_subplot(grid[0, :])
    for arm in arms:
        rows = [r for r in struct["densities"] if r["arm"] == arm]
        _structural_errorbar(top, rows, STRUCTURAL_COLORS[arm], label=arm, offset=offsets[arm])
    top.axhline(0.0, linewidth=0.8, color="black")
    top.set_ylabel("Δ training points vs B", fontsize=8)
    top.set_xticks(range(len(labels)))
    top.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    top.tick_params(axis="y", labelsize=7)
    top.legend(fontsize=8, frameon=False)

    for i, (family, test) in enumerate(STRUCTURAL_CELLS):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        for arm in arms:
            rows = [r for r in struct["errors"]
                    if r["family"] == family and r["test"] == test and r["arm"] == arm]
            _structural_errorbar(ax, rows, STRUCTURAL_COLORS[arm], label=arm, offset=offsets[arm])
        ax.axhline(0.0, linewidth=0.8, color="black")
        ax.set_title(f"{family.upper()}, {test}", fontsize=9, pad=14 if i // 2 == 0 else 6)
        ax.set_ylabel("Δ log-RMSE vs B", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        if i // 2 == 1:
            ax.set_xlabel("FOS bin (selection range)", fontsize=8)
        if i // 2 == 0:                       # once per test column
            counts = {r["bin"]: r["test_n"] for r in struct["errors"]
                      if r["family"] == family and r["test"] == test}
            for b, n in sorted(counts.items()):
                ax.annotate(str(n), (b, 1.0), xycoords=("data", "axes fraction"), fontsize=6,
                            ha="center", va="bottom", color="dimgray", annotation_clip=False)
    return fig


def fig_structural_matrix(struct: dict, arm: str):
    """Predicted bin against true bin, row-normalized, pooled over the
    confirmatory realizations only.

    This is where compensating errors live: two points swapping bins leave
    achieved - intended untouched and show up here as off-diagonal mass.
    A predicted bin that selected nothing is masked, not drawn as zero shares.
    """
    labels = struct["bins"]["labels"]
    n_bins = len(labels)
    confirmatory = [str(r) for r in struct["confirmatory"]]
    stacked = np.array([struct["displacement"][r][arm]["matrix"] for r in confirmatory],
                       dtype=float).sum(axis=0)
    # stored order: the in-range bins, then < 0.5, then > 6; drawn in FOS order
    order = [n_bins, *range(n_bins), n_bins + 1]
    stacked = stacked[:, order]
    totals = stacked.sum(axis=1, keepdims=True)
    shares = np.ma.masked_where(np.repeat(totals == 0, stacked.shape[1], axis=1),
                                np.divide(stacked, totals, out=np.zeros_like(stacked),
                                          where=totals > 0))
    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgrey")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(shares, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xlabel("true bin", fontsize=8)
    ax.set_ylabel("predicted bin (n selected, r1-r5 pooled)", fontsize=8)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"{label} (n={int(t)})" for label, t in zip(labels, totals[:, 0])],
                       fontsize=6.5)
    ax.set_xticks(range(n_bins + 2))
    # > 6, not >= 6: this matrix is built with _true_bin, the generator's (lo, hi]
    ax.set_xticklabels(["< 0.5", *labels, "> 6"], rotation=45, ha="right", fontsize=6.5)
    ax.set_title(f"{arm}: where the selected points landed, r1-r5 pooled", fontsize=9)
    fig.colorbar(image, ax=ax, label="share of the predicted bin")
    fig.tight_layout()
    return fig


def _mean_sd(values, digits: int = 1) -> str:
    """'mean ± sd' over the confirmatory realizations; a plain integer when
    every realization agrees on one."""
    values = np.asarray(values, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if sd == 0.0 and mean == round(mean):
        return str(int(round(mean)))
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _plain(value, digits: int = 1) -> str:
    return str(int(value)) if float(value) == round(float(value)) else f"{value:.{digits}f}"


STRUCTURAL_OUTSIDE_LABELS = ("< 0.5", "> 6")


def structural_allocation_frame(struct: dict, arm: str) -> pd.DataFrame:
    """Bins as rows; r1-r5 as mean ± sd, r0 in its own columns, never averaged in.

    The two rows outside the range carry the points that left it; nothing was
    intended there, so their deviation is their achieved count. The last row is
    Σ|deviation| over the 15 in-range bins, per realization, then averaged.

    C-oracle and C-logP (and their C0 ablations) have no r0: `struct["allocation"]["0"]`
    carries no entry for them, and every r0 cell reads as "--" instead.
    """
    labels = struct["bins"]["labels"]
    confirmatory = [str(r) for r in struct["confirmatory"]]
    rows_c = [struct["allocation"][k][arm] for k in confirmatory]
    row_0 = struct["allocation"]["0"].get(arm)
    r0 = (lambda value: "--") if row_0 is None else _plain
    rows = []
    for b, label in enumerate(labels):
        rows.append({"bin": label,
                     "seed": _mean_sd([r["seed"][b] for r in rows_c]),
                     "intended": _mean_sd([r["intended"][b] for r in rows_c]),
                     "achieved": _mean_sd([r["achieved"][b] for r in rows_c]),
                     "deviation": _mean_sd([r["deviation"][b] for r in rows_c]),
                     "intended r0": r0(row_0 and row_0["intended"][b]),
                     "achieved r0": r0(row_0 and row_0["achieved"][b]),
                     "deviation r0": r0(row_0 and row_0["deviation"][b])})
    for side in STRUCTURAL_OUTSIDE_LABELS:
        achieved = _mean_sd([r["achieved_outside"][side] for r in rows_c])
        rows.append({"bin": f"{side} (outside)", "seed": "--", "intended": "0",
                     "achieved": achieved, "deviation": achieved,
                     "intended r0": "0" if row_0 is not None else "--",
                     "achieved r0": r0(row_0 and row_0["achieved_outside"][side]),
                     "deviation r0": r0(row_0 and row_0["achieved_outside"][side])})
    blank = "" if row_0 is not None else "--"
    rows.append({"bin": "Σ|deviation|, 15 bins", "seed": "", "intended": "", "achieved": "",
                 "deviation": _mean_sd([r["abs_deviation"] for r in rows_c]),
                 "intended r0": blank, "achieved r0": blank,
                 "deviation r0": r0(row_0 and row_0["abs_deviation"])})
    return pd.DataFrame(rows)


ALLOCATION_FIGURE_ARMS = ("C-log", "C-oracle", "C-logP")


def fig_structural_allocation(struct: dict):
    """Intended against achieved bin counts, one panel per arm, from the same rows as
    `structural_allocation_frame`: r1-r5 only, achieved as mean ± sd, the generator's
    (lo, hi] bins (the first closed at 0.5), then the two bins outside the selection range, where nothing was
    intended. Descriptive; sized for the DCE text width (144 mm)."""
    labels = [*struct["bins"]["labels"], *STRUCTURAL_OUTSIDE_LABELS]
    n_in = len(struct["bins"]["labels"])
    confirmatory = [str(r) for r in struct["confirmatory"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(ALLOCATION_FIGURE_ARMS), 1, sharex=True, sharey=True, figsize=(5.67, 5.6))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.17, hspace=0.32)
    for ax, arm in zip(axes, ALLOCATION_FIGURE_ARMS):
        rows = [struct["allocation"][k][arm] for k in confirmatory]
        intended = np.array([*rows[0]["intended"], 0, 0], dtype=float)
        if any(r["intended"] != rows[0]["intended"] for r in rows):
            raise ValueError(f"{arm}: the intended allocation differs between realizations")
        achieved = np.array([[*r["achieved"], *(r["achieved_outside"][s] for s in STRUCTURAL_OUTSIDE_LABELS)]
                             for r in rows], dtype=float)
        mean, sd = achieved.mean(axis=0), achieved.std(axis=0, ddof=1)
        ax.axvspan(n_in - 0.5, len(labels) - 0.5, color="0.92", zorder=0, linewidth=0)
        ax.bar(x, intended, width=0.7, facecolor="white", edgecolor="0.35", hatch="////", linewidth=0.7,
               label="intended", zorder=1)
        ax.errorbar(x, mean, yerr=sd, fmt="o", markersize=3.5, color="#0072B2", elinewidth=0.9,
                    capsize=1.5, label=f"achieved, mean ± sd over r{confirmatory[0]}–r{confirmatory[-1]}",
                    zorder=3)
        ax.set_title(arm, fontsize=8, loc="left")
        ax.set_ylabel("points per bin", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.85", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right", fontsize=6.5)
    axes[-1].set_xlabel("FOS bin, (lo, hi]; the shaded pair lies outside the selection range", fontsize=7.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.99))
    return fig


def structural_allocation_caption(struct: dict) -> str:
    realizations = f"r{struct['confirmatory'][0]}-r{struct['confirmatory'][-1]}"
    shared = struct["ablation"]["shared_rows"]["note"]
    return (
        "Intended against achieved allocation of the 300 added points for C-log, C-oracle and C-logP: "
        "per bin, the intended count (hatched) and the achieved count as the mean ± sd over the "
        f"confirmatory realizations {realizations}; r0 is not drawn, and C-oracle and C-logP have none. "
        "The intended allocation is the same in every realization. Descriptive only: achieved minus "
        "intended is the net distortion of the requested bin-count distribution, not a prediction "
        "error, and two points that swap bins leave it unchanged. Bins follow the generator, "
        "(lo, hi] with the first bin closed at 0.5, as the allocation tables do; the two shaded bins, "
        "FOS < 0.5 and FOS > 6, lie outside the selection range, where nothing was intended. The counts are those "
        "of the allocation tables. C-oracle bins on the solved FOS and meets the intended allocation by "
        "construction; its panel is a check, not a finding. What known bin membership changes in test "
        "accuracy is read only from the paired contrast C-oracle against C-logP (Table tab:rep:oracle), "
        f"in this design. {shared}")


def structural_batch_frame(struct: dict) -> pd.DataFrame:
    """C-seq batch by batch: intended and deviation (achieved - intended) per
    batch, then the cumulative deviation; bins as rows, r1-r5 as mean ± sd,
    then r0 as its own block. Outside the range nothing was intended, so there
    the deviation is the achieved count."""
    labels = struct["bins"]["labels"]
    confirmatory = [str(r) for r in struct["confirmatory"]]
    rows = []
    for name, keys in (("r1-r5", confirmatory), ("r0 (exploratory)", ["0"])):
        batches = [struct["batches"][k] for k in keys]
        n_batches = len(batches[0])
        fmt = _mean_sd if len(keys) > 1 else (lambda v: _plain(v[0]))
        for b, label in enumerate([*labels, *STRUCTURAL_OUTSIDE_LABELS]):
            side = STRUCTURAL_OUTSIDE_LABELS[b - len(labels)] if b >= len(labels) else None

            def intended(entry):
                return 0 if side else entry["intended"][b]

            def deviation(entry):
                return (entry["achieved_outside"][side] if side
                        else entry["achieved"][b] - entry["intended"][b])

            row = {"set": name, "bin": f"{label} (outside)" if side else label}
            for j in range(n_batches):
                row[f"int. {j + 1}"] = fmt([intended(s[j]) for s in batches])
                row[f"dev. {j + 1}"] = fmt([deviation(s[j]) for s in batches])
            for j in range(1, n_batches):
                row[f"cum. {j + 1}"] = fmt([sum(deviation(s[i]) for i in range(j + 1))
                                                    for s in batches])
            rows.append(row)
    return pd.DataFrame(rows)


def structural_displacement_frame(struct: dict) -> pd.DataFrame:
    """Per arm: r1-r5 as mean ± sd, then r0 on its own row.

    Arms come from the union over every realization, not from r0 alone: C-oracle
    and C-logP (and their C0 ablations) have no r0 entry, and taking the arms
    from r0 would drop them from this table without a trace. Their r0 row reads
    "--" in every field. The r1-r5 row still requires the arm in every
    confirmatory realization; a KeyError there is a real defect, not papered over.
    """
    confirmatory = [str(r) for r in struct["confirmatory"]]
    arms = sorted({a for k in struct["displacement"] for a in struct["displacement"][k]})
    fields = (("n", "n", 0), ("stayed", "stayed", 2), ("mean |Δbin|", "mean_abs_shift", 2),
              ("median |Δbin|", "median_abs_shift", 2), ("outside", "outside", 2))
    rows = []
    for arm in arms:
        rows.append({"arm": arm, "set": "r1-r5",
                     **{name: _mean_sd([struct["displacement"][k][arm][key] for k in confirmatory],
                                       digits) for name, key, digits in fields}})
        row_0 = struct["displacement"]["0"].get(arm)
        if row_0 is None:
            rows.append({"arm": arm, "set": "r0 (exploratory)",
                         **{name: "--" for name, _, _ in fields}})
        else:
            rows.append({"arm": arm, "set": "r0 (exploratory)",
                         **{name: _plain(row_0[key], max(digits, 1)) if key == "n"
                            else f"{row_0[key]:.{digits}f}" for name, key, digits in fields}})
    return pd.DataFrame(rows)


def structural_outside_frame(struct: dict) -> pd.DataFrame:
    rows = [{"kind": r["kind"], "arm": r["arm"], "side": r["side"],
             "family": r.get("family", "--"), "test": r.get("test", "--"),
             "test n": r.get("test_n", "--"),
             "thin": ("yes" if r["marked"] else "no") if "marked" in r else "--",
             "diff": r["diff"], "ci lo": r["ci"][0], "ci hi": r["ci"][1]}
            for r in struct["outside"]]
    return pd.DataFrame(rows).sort_values(["kind", "arm", "side"])


def structural_shared_candidates_frame(struct: dict) -> pd.DataFrame:
    """Per realization: how many of guided-log-g's points guided-log also drew,
    from the same candidate pool but a different screener (structural.MIN_DISTANCE)."""
    rows = [{"realization": "r0 (exploratory)" if r["realization"] == 0 else f"r{r['realization']}",
             "n": r["n"], "shared candidates (same pool, different screener)": r["shared"]}
            for r in struct["screener_swap"]["shared_candidates"]]
    return pd.DataFrame(rows)


def structural_concordance_frame(struct: dict) -> pd.DataFrame:
    rows = [{"family": r["family"], "test": r["test"], "arm": r["arm"],
             "k / n (unmarked)": f"{r['k']} / {r['n']}", "zero (unmarked)": r["zero"],
             "thin": r["thin"], "k / n (all 15)": f"{r['k_all']} / {r['n_all']}",
             "zero (all 15)": r["zero_all"]}
            for r in struct["concordance"]]
    return pd.DataFrame(rows)


# --- structural (Task 5 addendum): the C0 ablations --------------------------
# Amendment 1, docs/predictions-and-outcomes.md. None of this enters the main
# figure above or its concordance count; every row here joins arms trained on
# different budgets, and both the panels and the tables say so.

# Explicit by comparison, not by arm: a single panel already draws both
# comparisons for one C0 arm, so a property cycle would conflate them.
STRUCTURAL_ABLATION_COLORS = {"welch": "tab:red", "paired": "tab:purple"}


def fig_structural_ablation(struct: dict):
    """One row per C0 arm, one column per family x test cell: the C0 - B
    (Welch) and C - C0 (paired) differences per reporting bin, with the two
    bins outside the selection range shaded apart from the fifteen inside it."""
    ablation = struct["ablation"]
    labels = ablation["labels"]
    budgets = ablation["budgets"]
    arms = sorted({r["arm"] for r in ablation["errors"]})
    offsets = _offsets(("welch", "paired"))
    fig, axes = plt.subplots(len(arms), len(STRUCTURAL_CELLS), sharex=True,
                             figsize=(13.0, 3.4 * len(arms) + 1.2), squeeze=False)
    for i, arm in enumerate(arms):
        c_arm = next(r["first"] for r in ablation["errors"]
                     if r["arm"] == arm and r["interval"] == "paired")
        for j, (family, test) in enumerate(STRUCTURAL_CELLS):
            ax = axes[i][j]
            cell = [r for r in ablation["errors"]
                    if r["arm"] == arm and r["family"] == family and r["test"] == test]
            for kind, first, second in (("welch", arm, "B"), ("paired", c_arm, arm)):
                rows = sorted((r for r in cell if r["interval"] == kind),
                              key=lambda r: r["reporting_bin"])
                _structural_errorbar(ax, rows, STRUCTURAL_ABLATION_COLORS[kind],
                                     x_key="reporting_bin", offset=offsets[kind],
                                     label=f"{first} ({budgets[first]}) − {second} "
                                           f"({budgets[second]}), {kind}")
            ax.axhline(0.0, linewidth=0.8, color="black")
            # the two bins outside the selection range, set apart visually
            ax.axvspan(-0.5, 0.5, color="lightgrey", alpha=0.4, zorder=0)
            ax.axvspan(len(labels) - 1.5, len(labels) - 0.5, color="lightgrey", alpha=0.4, zorder=0)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_title(f"{arm}: {family.upper()}, {test}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("Δ log-RMSE", fontsize=8)
        axes[i][0].legend(fontsize=7, frameon=False, loc="best")
    for ax in axes[-1]:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_xlabel("reporting bin", fontsize=8)
    fig.suptitle("C0 ablations, different budgets throughout; negative = first-named arm has "
                 "the smaller log-RMSE", fontsize=9.5)
    fig.tight_layout()
    return fig


def structural_ablation_fraction_frame(struct: dict) -> pd.DataFrame:
    """Per C0 arm and reporting bin: the mean C0 fraction, the mean B
    fraction, and their difference. Fractions only, never raw counts, which
    would carry the budget difference."""
    rows = [{"arm": r["arm"], "bin": r["label"], "C0 fraction": r["first_mean"],
             "B fraction": r["second_mean"], "diff": r["diff"],
             "ci lo": r["ci"][0], "ci hi": r["ci"][1]}
            for r in struct["ablation"]["fractions"]]
    return pd.DataFrame(rows)


def _structural_text(text: str) -> str:
    """A free-standing caption or note, escaped as booktabs escapes its cells."""
    return _escape(text) + "\n"


# --- the tables as the manuscript prints them -----------------------------------
# Each only selects, relabels, blocks or splits a frame above; no value changes.

def _zone_columns(frame: pd.DataFrame, prefix: str) -> dict:
    """'<prefix><zone>' columns, relabelled as the zone under a 'by zone' group."""
    return {c: ("by zone", c[len(prefix):]) for c in frame.columns
            if isinstance(c, str) and c.startswith(prefix)}


def global_table(metrics: dict, test: str, stem: str) -> str:
    frame = global_frame(metrics)
    columns = {c: (Tex(r"$R^2$") if c == "R2" else c) for c in frame.columns}
    return booktabs(regroup(frame, columns), caption=f"Global metrics on {test}. {ablation_note()}",
                    label=f"tab:{stem}:global", block="family", small=True)


def comparison_table(metrics: dict, test: str, stem: str) -> str:
    frame = comparison_frame(metrics)
    columns = {"family": "family", "subset": "subset", "scale": "scale", "comparison": "comparison",
               "alternative": "alternative",
               "n": ("Wilcoxon on d", "n"), "median d": ("Wilcoxon on d", "median"),
               "P(d<0)": ("Wilcoxon on d", "P(d<0)"), "p": ("Wilcoxon on d", "p"),
               **{c: ("bootstrap", c) for c in ("ΔRMSE", "CI low", "CI high")}}
    return longtable(regroup(frame, columns),
                     caption=f"Paired comparisons on {test}: Wilcoxon on d = |e1| - |e2| (smaller absolute "
                             f"errors) and bootstrap ΔRMSE. {ablation_note()}",
                     label=f"tab:{stem}:comparisons", block="family", left=4, colsep=3)


def conformal_tables(metrics: dict, test: str, stem: str) -> str:
    """Coverage and mean width as two tables, each with the zones as columns, so
    the arms stay rows as in every other table and each quantity reads along the
    FOS axis in one row."""
    frame = conformal_frame(metrics)
    caption = f"Cross-conformal-style intervals on {test}, empirical coverage."
    coverage = regroup(frame, {"family": "family", "arm": "arm", "coverage": "all",
                               **_zone_columns(frame, "cov ")})
    width = regroup(frame, {"family": "family", "arm": "arm", "factor": "factor", "mean width": "all",
                            **_zone_columns(frame, "width ")})
    return (booktabs(coverage, caption=f"{caption} Coverage over the whole test set and by reporting zone. "
                                       f"{ablation_note()}",
                     label=f"tab:{stem}:conformal", block="family")
            + booktabs(width, caption=f"{caption} Mean interval width over the whole test set and by "
                                      "reporting zone; the interval is [prediction / factor, prediction x "
                                      f"factor]. {ablation_note()}",
                       label=f"tab:{stem}:conformal:width", block="family"))


def subset_table(metrics: dict, test: str, stem: str) -> str:
    frame = subset_frame(metrics)
    columns = {"family": "family", "arm": "arm"}
    for scale in ("log-RMSE", "RMSE"):
        columns.update({c: (scale, c[len(scale) + 1:].replace("-", " ")) for c in frame.columns
                        if c.startswith(f"{scale} ")})
    return booktabs(regroup(frame, columns),
                    caption=f"RMSE on {test} by subset, log and FOS scale. The decision range [0.5, 2) "
                            f"carries the primary criterion (log scale). {ablation_note()}",
                    label=f"tab:{stem}:subsets", block="family")


def scoring_table(by_scoring: dict[str, dict], test: str) -> str:
    frame = scoring_frame(by_scoring)
    columns = {"family": "family", "arm": "arm"}
    for scoring in by_scoring:
        columns.update({f"{m} {scoring}": (scoring, m) for m in ("RMSE", "minimax", "rank")})
    return booktabs(regroup(frame, columns),
                    caption=f"Selection-criterion sensitivity on {test}: global RMSE, minimax and "
                            f"within-family rank under each scoring. {ablation_note()}",
                    label=f"tab:{test}:scorings", block="family")


def scoring_params_table(by_scoring: dict[str, dict]) -> str:
    """RF and GBM as two panels: their hyper-parameters differ."""
    return booktabs_panels(
        [(family.upper(), scoring_params_frame(by_scoring, family), {}) for family in config.FAMILIES],
        caption="Hyper-parameters selected by CV under each scoring, one column per hyper-parameter "
                f"(scikit-learn names: {param_legend(by_scoring)}). {ablation_note()}",
        label="tab:scorings:params")


def _stack(text: str) -> Tex:
    """Two words on two lines, right-aligned, with no package beyond LaTeX's own."""
    first, rest = text.split(" ", 1)
    return Tex(rf"\shortstack[r]{{{_escape(first)}\\{_escape(rest)}}}")


def oracle_table(rep: dict) -> str:
    """The oracle reading in two panels: the medians of L and H per arm, then
    dR_O, dL_O and dH_O with their intervals. The reading row is split with its columns."""
    frame = oracle_reading_frame(rep)
    medians = regroup(frame, {"cell": "cell", **{f"median {q} {arm}": (arm, f"median {q}")
                                                 for arm in ("C-oracle", "C-logP") for q in "LH"}})
    intervals = regroup(frame, {"cell": "cell", "median dR_O": ("dR_O", "median"),
                                "dR_O CI low": ("dR_O", "CI low"), "dR_O CI high": ("dR_O", "CI high"),
                                "median dL_O": ("dL_O", "median"), "dL_O CI": ("dL_O", "CI"),
                                "median dH_O": ("dH_O", "median"), "dH_O CI": ("dH_O", "CI")})
    reading = intervals.index[intervals["cell"] == "reading"]
    for column in intervals.columns[1:]:
        value = intervals.at[reading[0], column] if len(reading) else None
        if isinstance(value, str) and " " in value:
            intervals.at[reading[0], column] = _stack(value)
    return booktabs_panels(
        [(None, medians, {}), (None, intervals, {})],
        caption="The oracle, read as declared before the pool was solved: C-oracle against "
                "C-logP on the same sub-pool, r1-r5, medians per learner x test cell and the "
                "paired 95% interval of dR_O = R(C-oracle) - R(C-logP). 'Attributable to the "
                "screener's error, in this design' only where that interval excludes 0 in at "
                "least three of four cells; an interval containing 0 is inconclusive. C0-oracle "
                "and C0-logP are reported in the replication tables.",
        label="tab:rep:oracle", small=True)


def run() -> None:
    from sgs import structural
    tables, figures = config.tables_dir(), config.figures_dir()
    _save_table(tables / "budget.tex", booktabs(
        budget_frame(), caption="Budget-fair arms. C0 is an ablation and carries no budget. "
        "G: GBM screener; the learner is named separately.",
        label="tab:budget"))

    for scoring in config.ACTIVE_SCORINGS:
        for test in config.TEST_ROLES:
            if not config.metrics_path(test, scoring).exists():
                print(f"[report] {config.metrics_path(test, scoring).name} missing; run sgs evaluate")
                continue
            metrics = load_metrics(test, scoring)
            stem = f"{test}__{scoring}"
            _save_table(tables / f"{stem}__global.tex", global_table(metrics, test, stem))
            _save_table(tables / f"{stem}__comparisons.tex", comparison_table(metrics, test, stem))
            _save_table(tables / f"{stem}__conformal.tex", conformal_tables(metrics, test, stem))
            _save_table(tables / f"{stem}__subsets.tex", subset_table(metrics, test, stem))
            for family in metrics["arms"]:
                for key, tag in (("rmse", "zones"), ("log_rmse", "zones_log")):
                    scale = "log FOS" if key == "log_rmse" else "FOS"
                    _save_table(tables / f"{stem}__{tag}_{family}.tex", booktabs(
                        zone_frame(metrics, family, key),
                        caption=f"Zone RMSE ({scale}) on {test}, {family}. {ablation_note()}",
                        label=f"tab:{stem}:{tag}:{family}", small=True))
                    _save_figure(figures / f"{stem}__{tag}_{family}.pdf", fig_zone_rmse(metrics, family, key))

    if len(config.ACTIVE_SCORINGS) > 1:
        by_test = {test: {s: load_metrics(test, s) for s in config.ACTIVE_SCORINGS
                          if config.metrics_path(test, s).exists()} for test in config.TEST_ROLES}
        complete = {t: m for t, m in by_test.items() if len(m) == len(config.ACTIVE_SCORINGS)}
        for test, by_scoring in complete.items():
            _save_table(tables / f"{test}__scorings.tex", scoring_table(by_scoring, test))
        if complete:
            _save_table(tables / "scorings_params.tex", scoring_params_table(next(iter(complete.values()))))

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
                        "(exploratory). Common fixed hyper-parameters. C0-log, C0-seq, C0-logG, "
                        "C0-oracle and C0-logP are ablations without the seed and carry no budget: they "
                        "train on 300 points against the 600 of B, C-log, C-seq, C-logG, C-oracle and "
                        "C-logP. C-oracle, C-logP, C0-oracle and C0-logP have no r0 (shown as '--').",
                label=f"tab:rep:{test}:values"))
            _save_table(tables / f"replications__{test}__comparisons.tex", longtable(
                replication_comparison_frame(rep, test),
                caption=f"Replicated comparisons on {test}: Welch (independent arms) or paired intervals, "
                        "log-RMSE and relative scale; P(first<second) is descriptive. C0-log, C0-seq, "
                        "C0-logG, C0-oracle and C0-logP are ablations without the seed and carry no "
                        "budget: they train on 300 points against the 600 of B, C-log, C-seq, C-logG, "
                        "C-oracle and C-logP. C-oracle, C-logP, C0-oracle and C0-logP have no r0 (shown "
                        "as '--').",
                label=f"tab:rep:{test}:comparisons", block="family", left=4))
            for family in config.REP_FAMILIES:
                _save_figure(figures / f"replications__{test}__{family}.pdf", fig_replication_bins(rep, test, family))
        _save_table(tables / "replications__interactions.tex", booktabs_split(
            replication_interaction_frame(rep), "test", caption="Learner interaction on "
            "log(log-RMSE_rf) − log(log-RMSE_gbm) between arms; Welch, or paired where the two "
            "arms share a realization's candidate pool and intended allocation.",
            label="tab:rep:interactions"))
        if rep.get("oracle"):
            _save_table(tables / "replications__oracle.tex", oracle_table(rep))
            # The forest plot declares the pool arms; metrics built before the pool have none.
            _save_figure(figures / "replications__decision_forest.pdf", fig_forest(rep))
            _save_table(tables / "replications__decision_forest_caption.tex",
                        _structural_text(forest_caption(rep)))

    if config.structural_metrics_path().exists():
        struct = load_structural()
        check_structural_fresh(struct)
        ablation = struct["ablation"]
        shared = ablation["shared_rows"]["note"]
        oracle = structural.ORACLE_SENTENCE
        _save_figure(figures / "structural__bins.pdf", fig_structural(struct))
        _save_table(tables / "structural__caption.tex", _structural_text(struct["figure_caption"]))
        _save_table(tables / "structural__concordance.tex", booktabs(
            structural_concordance_frame(struct),
            caption="Concordance: the number of in-range bins in which the difference in "
                    "training density and the difference in log-RMSE against B have opposite "
                    "signs, k of n. Primary over the bins with at least "
                    f"{struct['min_test_n']} test points; over all 15 as a descriptive "
                    "sensitivity. Bins where either difference is exactly zero leave the "
                    "denominator. A count, with no test attached.",
            label="tab:structural:concordance", small=True))
        for arm in structural.SCREENED_ARMS:
            # C-oracle bins on the solved FOS, not a screener's prediction: it gets no matrix.
            _save_figure(figures / f"structural__matrix_{arm.lower()}.pdf",
                         fig_structural_matrix(struct, arm))
        for arm in structural.ARMS_VS_B:
            oracle_note = (" C-oracle bins on the solved FOS: every deviation is 0 by "
                           "construction, a check, not a finding.") if arm == "C-oracle" else ""
            _save_table(tables / f"structural__allocation_{arm.lower()}.tex", booktabs(
                structural_allocation_frame(struct, arm),
                caption=f"Allocation fidelity for {arm}: mean ± sd over the confirmatory "
                        "realizations r1-r5, with r0 (exploratory) in its own columns. "
                        "'deviation' is achieved minus intended, the net distortion of the "
                        "requested bin-count distribution and not the screener's prediction "
                        "error. The two outside rows count the points whose true FOS left the "
                        "selection range, where nothing was intended; the last row sums "
                        "|deviation| over the 15 in-range bins. Bins follow the generator, "
                        "(lo, hi], so the outer bin is FOS > 6. Selected equals intended by "
                        f"construction and is not shown. {oracle} {shared}{oracle_note}",
                label=f"tab:structural:allocation:{arm.lower()}", small=True))
        _save_figure(figures / "structural__allocation.pdf", fig_structural_allocation(struct))
        _save_table(tables / "structural__allocation_caption.tex",
                    _structural_text(structural_allocation_caption(struct)))
        _save_table(tables / "structural__matrix_caption.tex",
                    _structural_text(struct["matrix_caption"]))
        # Two floats, r1-r5 and r0: r0 stays visibly apart and each fits the text width.
        batches = structural_batch_frame(struct)
        for stem, name, note in (("batches", "r1-r5", ""),
                                 ("batches_r0", "r0 (exploratory)",
                                  " This table is r0 alone: exploratory, never averaged in.")):
            _save_table(tables / f"structural__{stem}.tex", booktabs(
                batches[batches["set"] == name].drop(columns="set"),
                caption=f"{struct['batches_caption']}{note} {shared}",
                label=f"tab:structural:{stem.replace('_', ':')}", small=True))
        _save_table(tables / "structural__displacement.tex", booktabs(
            structural_displacement_frame(struct),
            caption="The screener on the points it selected: share staying in the predicted bin, "
                    "absolute displacement in bins, share landing outside [0.5, 6]; mean ± sd over "
                    "r1-r5, r0 (exploratory) apart. Conditioned on selection, so this is its "
                    "accuracy where it was used, not over the pool. The displacement columns "
                    "cover only points whose true FOS stayed inside the selection range; the "
                    f"rest are counted by 'outside'. C-oracle's row is a construction check "
                    f"(stayed = 1, shift 0), not a description of a screener. {oracle} {shared}",
            label="tab:structural:displacement", small=True))
        _save_table(tables / "structural__outside.tex", booktabs_split(
            structural_outside_frame(struct), "kind",
            caption="Outside the selection range, where nothing was allocated. 'density' rows are "
                    "the spillover itself: training points that landed there because the gate is "
                    "on a predicted FOS. 'error' rows are the test error in those bins, with the "
                    "number of test points; rows with fewer than "
                    f"{struct['min_test_n']} are thin and carry no statement in the text. "
                    "Differences against B with descriptive 95% Welch intervals over r1-r5. Both "
                    "sides are read from the replication metrics, so both use the half-open "
                    "reporting zones and the outer bin is FOS >= 6.",
            label="tab:structural:outside", part=lambda kind: f"This table: the '{kind}' rows.",
            small=True))
        _save_table(tables / "structural__shared_candidates.tex", booktabs(
            structural_shared_candidates_frame(struct),
            caption="Shared candidates (same pool, different screener): how many of guided-log-g's "
                    "points guided-log also drew, per realization, by normalized distance. Both "
                    "screeners choose, in each realization, from the same candidate pool against the "
                    "same intended allocation; only the screener differs.",
            label="tab:structural:shared_candidates"))
        # Task 5 addendum (Amendment 1): the C0 ablations. Supplementary; they
        # never enter the main figure or the concordance count above.
        budgets = ablation["budgets"]
        _save_figure(figures / "structural__ablation.pdf", fig_structural_ablation(struct))
        _save_table(tables / "structural__ablation_fractions.tex", longtable(
            structural_ablation_fraction_frame(struct),
            caption="C0 training distribution against B, as the fraction of the training set in "
                    "each reporting bin, with descriptive 95% Welch intervals over r1-r5. "
                    "Fractions only, never raw counts, because C0 trains on "
                    f"{budgets['C0-log']} points and B on {budgets['B']}.",
            label="tab:structural:ablation:fractions", left=2))
        _save_table(tables / "structural__ablation_caption.tex", _structural_text(ablation["caption"]))
        _save_table(tables / "structural__original_c0.tex", _structural_text(ablation["original_c0"]))
    else:
        print("[report] structural.json missing; run sgs structural")
    print(f"[report] wrote {tables} and {figures}")
