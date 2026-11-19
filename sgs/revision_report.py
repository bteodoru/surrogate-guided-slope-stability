"""LaTeX tables of the revision's results, read only from ``outputs/metrics/`` (replications.json
and revision.json) and written, additively, to ``outputs/tables/revision__*.tex``.

Cells are in the paper's order: GBM natural, GBM stratified, RF natural, RF stratified.
Every ablation is listed with its arm.
"""

from __future__ import annotations

import json

from sgs import config

CELLS = (("gbm", "test-natural"), ("gbm", "test-stratified"), ("rf", "test-natural"), ("rf", "test-stratified"))
LARGE = {"test-natural": "test-natural-large", "test-stratified": "test-stratified-large"}
ARMS = ("C-log", "C0-log", "C-seq", "C0-seq", "C-logG", "C0-logG", "C-oracle", "C0-oracle",
        "C-logP", "C0-logP", "C-seqF", "C0-seqF", "C-seqP", "C0-seqP", "D-oracle", "D0-oracle", "D-P", "D0-P")
SEED_ARMS = {"C-logP": "C-logP", "C0-logP": "C0-logP", "D-P": "D-P", "D0-P": "D0-P"}


def _tex_arm(arm: str) -> str:
    return arm.replace("C0", "C$_0$").replace("D0", "D$_0$")


def _load():
    rep = json.loads(config.replication_metrics_path().read_text())
    rev = json.loads((config.ROOT / "outputs" / "metrics" / "revision.json").read_text())
    return rep, rev


def _level(cells) -> int:
    """The declared level from (diff, ci) per cell: 3 all below 0, 2 one below 0, 1 one negative mean."""
    if all(ci[1] < 0 for _, ci in cells):
        return 3
    if any(ci[1] < 0 for _, ci in cells):
        return 2
    if any(d < 0 for d, _ in cells):
        return 1
    return 0


def _write(name: str, text: str) -> None:
    path = config.tables_dir() / f"{name}.tex"
    path.write_text(text)
    print(f"[revision-report] {path.name}")


def levels_table(rep, rev) -> str:
    base = rev["robustness"]["baseline"]
    boot = {(c["family"], c["test"], c["first"]): c for c in rev["bootstrap"]["comparisons"]
            if c["tests"] == "fixed" and c["measure"] == "decision-range"}
    large = rev["large_tests"]["readings"]
    seed = rev["seed_study"]["readings"]
    gp = rev["gp"]["readings"]
    rows = []
    for arm in ARMS:
        b = base[arm]
        below = sum(1 for c in b["cells"].values() if c["ci"][1] < 0)
        bc = [(boot[(f, t, arm)]["diff"], boot[(f, t, arm)]["bootstrap_ci"]) for f, t in CELLS]
        lg = large["levels"][arm]
        lbelow = sum(1 for c in lg["cells"].values() if c["ci"][1] < 0)
        if arm in SEED_ARMS:
            s = seed["levels"][arm]
            sb = sum(1 for c in s["cells"].values() if c["ci"][1] < 0)
            seed_txt = f"{s['level']} ({sb})"
            seed_gate = f"{seed['gate'][arm]['passes']}/4"
        else:
            seed_txt, seed_gate = "--", "--"
        gate_fixed = f"{b['gate_passes']}/4"
        gate_large = f"{large['gate'][arm]['passes']}/4"
        rows.append(f"{_tex_arm(arm)} & {b['level']} ({below}) & {_level(bc)} ({sum(1 for _, ci in bc if ci[1] < 0)}) "
                    f"& {lg['level']} ({lbelow}) & {seed_txt} & {gp[arm]['level']} "
                    f"& {gate_fixed} & {gate_large} & {seed_gate} & {gp[arm]['gate_passes']}/2 \\\\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Every arm against B, across the analyses: the level on the decision range "
        r"(3: interval entirely below 0 in every cell; 2: in at least one cell; 1: a negative mean in at least one "
        r"cell; 0: otherwise), with the number of cells whose interval lies entirely below 0 in brackets, and the "
        r"number of cells in which the relocation gate passes. Fixed: the fixed test sets, Welch intervals over "
        r"$r_1$--$r_5$ (the declared readings); bootstrap: the same contrasts with the two-level bootstrap interval; "
        r"large: T-natL and T-stratL, Welch; seeds: paired over the five seeds; GP: the Gaussian process on the fixed "
        r"test sets, two cells. The ablations train on 300 points against B's 600 and are not budget arms.}",
        r"\label{tab:rev:levels}",
        r"\begin{tabular}{lccccc|cccc}", r"\toprule",
        r" & \multicolumn{5}{c|}{Level against B (cells below 0)} & \multicolumn{4}{c}{Relocation gate} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-10}",
        r"Arm & fixed & bootstrap & large & seeds & GP & fixed & large & seeds & GP \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def _fmt(d, ci) -> str:
    v = f"{d:+.4f}".replace("+", "$+$").replace("-", "$-$")
    return rf"\textbf{{{v}}}" if (ci[0] > 0 or ci[1] < 0) else v


def whole_selection_table(rep, rev) -> str:
    comps = {(c["family"], c["test"], c["measure"], c["first"]): c
             for c in rep["comparisons"] + rev["fixed_tests"]["comparisons"]
             if c["second"] == "B" and c["scale"] == "log-rmse"}
    rows = []
    for arm in ARMS:
        cells = []
        for measure in ("all", "selection-range"):
            for f, t in CELLS:
                c = comps.get((f, t, measure, arm))
                cells.append("--" if c is None else _fmt(c["diff"], c["ci"]))
        rows.append(f"{_tex_arm(arm)} & " + " & ".join(cells) + r" \\")
    head = " & ".join(["GBM nat.", "GBM str.", "RF nat.", "RF str."] * 2)
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{The whole test set and the selection range: each arm's log-RMSE minus B's, mean over "
        r"$r_1$--$r_5$, on the fixed test sets. Bold: the 95\% Welch interval excludes 0. The intervals are in "
        r"\cref{S-tab:rev:whole-selection-intervals}. The ablations train on 300 points against B's 600.}",
        r"\label{tab:rev:whole-selection}",
        r"\begin{tabular}{lcccc|cccc}", r"\toprule",
        r" & \multicolumn{4}{c|}{Whole test set} & \multicolumn{4}{c}{Selection range, FOS $\in [0.5, 6]$} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        f"Arm & {head} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def _iv(d, ci) -> str:
    return f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]".replace("-", "$-$").replace("+", "$+$")


def whole_selection_intervals_table(rep, rev) -> str:
    """The intervals behind tab:rev:whole-selection, every arm and ablation, for the supplement."""
    comps = {(c["family"], c["test"], c["measure"], c["first"]): c
             for c in rep["comparisons"] + rev["fixed_tests"]["comparisons"]
             if c["second"] == "B" and c["scale"] == "log-rmse"}
    rows = []
    for arm in ARMS:
        for f, t in CELLS:
            cell = f"{f.upper()}, {t.split('-')[1]}"
            vals = [comps.get((f, t, m, arm)) for m in ("all", "selection-range")]
            txt = " & ".join("--" if c is None else _iv(c["diff"], c["ci"]) for c in vals)
            rows.append(f"{_tex_arm(arm) if (f, t) == CELLS[0] else ''} & {cell} & {txt} \\\\")
        rows.append(r"\addlinespace")
    return "\n".join([
        r"\begin{longtable}{llcc}",
        r"\caption{The whole test set and the selection range: log-RMSE of every arm minus B's, mean over "
        r"$r_1$--$r_5$ on the fixed test sets, with the 95\% Welch interval. The means are those of \cref{M-tab:rev:whole-selection}.}\label{tab:rev:whole-selection-intervals}\\",
        r"\toprule", r"Arm & Cell & Whole test set & Selection range \\", r"\midrule", r"\endfirsthead",
        r"\toprule", r"Arm & Cell & Whole test set & Selection range \\", r"\midrule", r"\endhead",
        *rows[:-1], r"\bottomrule", r"\end{longtable}", ""])


def decision_intervals_table(rep, rev) -> str:
    base = rev["robustness"]["baseline"]
    boot = {(c["family"], c["test"], c["first"]): c for c in rev["bootstrap"]["comparisons"]
            if c["tests"] == "fixed" and c["measure"] == "decision-range"}
    large = rev["large_tests"]["readings"]["levels"]

    def iv(d, ci):
        return f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]".replace("-", "$-$").replace("+", "$+$")

    rows = []
    for arm in ARMS:
        for f, t in CELLS:
            b = base[arm]["cells"][f"{f}/{t}"]
            bs = boot[(f, t, arm)]
            lg = large[arm]["cells"][f"{f}/{LARGE[t]}"]
            cell = f"{f.upper()}, {t.split('-')[1]}"
            rows.append(f"{_tex_arm(arm) if (f, t) == CELLS[0] else ''} & {cell} & {iv(b['diff'], b['ci'])} & "
                        f"{iv(bs['diff'], bs['bootstrap_ci'])} & {iv(lg['diff'], lg['ci'])} \\\\")
        rows.append(r"\addlinespace")
    return "\n".join([
        r"\begin{longtable}{llccc}",
        r"\caption{Decision-range log-RMSE of every arm minus B's, with 95\% intervals: Welch on the fixed test "
        r"sets (the declared readings), the two-level bootstrap on the same sets, and Welch on the large test sets "
        r"(T-natL, T-stratL).}\label{tab:rev:decision-intervals}\\",
        r"\toprule", r"Arm & Cell & Fixed, Welch & Fixed, bootstrap & Large, Welch \\", r"\midrule", r"\endfirsthead",
        r"\toprule", r"Arm & Cell & Fixed, Welch & Fixed, bootstrap & Large, Welch \\", r"\midrule", r"\endhead",
        *rows[:-1], r"\bottomrule", r"\end{longtable}", ""])


def robustness_table(rep, rev) -> str:
    r = rev["robustness"]
    rt = rev["retune"]["against_baseline"]
    rows = []
    for arm in ARMS:
        base = r["baseline"][arm]["level"]
        tf = r["three-features"]["readings"][arm]["level"]
        nc = r["no-crest-entry"]["readings"][arm]["level"]
        diff3 = sum(1 for c in r["three-features"]["against_baseline"][arm]["cells"].values() if c["verdict"] == "differs")
        diffc = sum(1 for c in r["no-crest-entry"]["against_baseline"][arm]["cells"].values() if c["verdict"] == "differs")
        ret = f"{rt[arm]['level'][1]}" if arm in rt else "--"
        alt = " / ".join(str(r["alternative_ranges"][k][arm]["level"]) for k in ("[2/3, 1.5)", "[0.4, 2.5)"))
        rows.append(f"{_tex_arm(arm)} & {base} & {ret} & {tf} ({diff3}) & {nc} ({diffc}) & {alt} \\\\")
    rw = r["reweight"]

    def cells(entry):
        return " & ".join(f"{entry['cells'][f'{f}/{t}']['diff']:+.4f}".replace("-", "$-$").replace("+", "$+$")
                          + ("" if entry["cells"][f"{f}/{t}"]["verdict"] == "contains 0"
                             else (r"$^{<}$" if entry["cells"][f"{f}/{t}"]["verdict"] == "below 0" else r"$^{>}$"))
                          for f, t in CELLS)
    rwrows = [f"B-wC $-$ B & {cells(rw['vs_B']['B-wC'])} \\\\", f"B-wD $-$ B & {cells(rw['vs_B']['B-wD'])} \\\\",
              f"B-wC $-$ C-logP & {cells(rw['vs_guided']['B-wC'])} \\\\",
              f"B-wD $-$ D-P & {cells(rw['vs_guided']['B-wD'])} \\\\"]
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Robustness checks on the fixed test sets. Top: each arm's level against B on the decision range "
        r"under the baseline, the independent retuning, the three governing variables and the training sets without "
        r"crest entries (in brackets, the number of cells whose verdict differs from the baseline), and on the two "
        r"alternative decision ranges. Bottom: density reweighting, decision-range log-RMSE differences "
        r"(superscript $<$ or $>$: the interval lies entirely below or above 0).}",
        r"\label{tab:rev:robustness}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Arm & baseline & retuned & three variables & no crest entries & $[2/3, 1.5)$ / $[0.4, 2.5)$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\par\medskip",
        r"\begin{tabular}{lcccc}", r"\toprule", r" & GBM nat. & GBM str. & RF nat. & RF str. \\", r"\midrule",
        *rwrows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def bias_table(rep, rev) -> str:
    m = rev["signed_bias"]["means"]
    rows = []
    for arm in ("B", *ARMS):
        vals = []
        for zone in ("0.5–1", "1–2"):
            for f, t in CELLS:
                vals.append(f"{m[f'{f}/{t}/{arm}/{zone}']:+.3f}".replace("-", "$-$").replace("+", "$+$"))
        rows.append(f"{_tex_arm(arm)} & " + " & ".join(vals) + r" \\")
    head = " & ".join(["GBM nat.", "GBM str.", "RF nat.", "RF str."] * 2)
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Signed bias on the two halves of the decision range: the mean of $\ln(\widehat{\mathrm{FOS}}/"
        r"\mathrm{FOS})$ over the test points of the zone, mean over $r_1$--$r_5$, fixed test sets. A positive value "
        r"is an over-prediction of the FOS, on the unconservative side. Post hoc and descriptive, added in revision; "
        r"no test is made on it. The natural and stratified test sets hold 33 and 12 points on $[0.5, 1)$, and 94 and "
        r"24 on $[1, 2)$.}",
        r"\label{tab:rev:bias}",
        r"\begin{tabular}{lcccc|cccc}", r"\toprule",
        r" & \multicolumn{4}{c|}{$[0.5, 1)$} & \multicolumn{4}{c}{$[1, 2)$} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        f"Arm & {head} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def run() -> None:
    rep, rev = _load()
    _write("revision__levels", levels_table(rep, rev))
    _write("revision__whole_selection", whole_selection_table(rep, rev))
    _write("revision__whole_selection_intervals", whole_selection_intervals_table(rep, rev))
    _write("revision__decision_intervals", decision_intervals_table(rep, rev))
    _write("revision__robustness", robustness_table(rep, rev))
    _write("revision__bias", bias_table(rep, rev))
