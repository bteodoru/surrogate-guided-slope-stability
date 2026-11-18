"""The revision's robustness checks (docs/revision-declaration.md §8.2-§8.6).

None of them changes a declared reading. Each is reported per learner x test cell as
"holds" or "differs" against the baseline reading, with every ablation:
  - the decision-range verdict of an arm against B (95% interval below 0, containing 0,
    above 0; Welch, as in the replications);
  - the relocation gate (median L < 0 < median H over r1..r5).

§8.2 refits every replicated arm, and the revision's arms, on angle, phi and gamma*H/c only;
§8.3 drops from every training set the rows whose critical circle enters the crest
(|Center_Y - H| < 0.01 H, the criterion of docs/crest-entry-check/check.py) and refits;
§8.4 refits B r1..r5 with weights that bring its binned training distribution to C-log's and
to D's intended one (B-wC, B-wD); §8.5 sets A against every ablation (all 300 points) from
the fits that exist; §8.6 reads the levels of every arm on two other decision ranges.
Refits use LC_PARAMS on log(FOS), like the replications; they go to artifacts/revision/robust/.
Results: the ``robustness`` entry of outputs/metrics/revision.json.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sgs import config, data, density, design, evaluate, models, replicate
from sgs import revision_fit as rf
from sgs import revision_pool as rp

CELLS = [(f, t) for f in config.REP_FAMILIES for t in config.TEST_ROLES]
REALS = range(1, config.REP_REALIZATIONS + 1)
GATE_ARMS = ("C-log", "C-seq", "C-logG", "C-oracle", "C-logP", "D-oracle", "D-P", "C-seqF", "C-seqP")
THREE = ("angle", "phi", "gamma_H_c")
NEAR_CREST = 0.01
ALT_RANGES = {"[2/3, 1.5)": (2 / 3, 1.5), "[0.4, 2.5)": (0.4, 2.5)}
LOWER, UPPER = (0.5, 1.0), (1.0, 2.0)


def _tests():
    return {role: data.load_test(role) for role in config.TEST_ROLES}


def _log_rmse(pred, y, lo, hi) -> float:
    m = (y >= lo) & (y < hi)
    return float(np.sqrt(np.mean((np.log(pred[m]) - np.log(y[m])) ** 2)))


def _verdict(ci) -> str:
    return "below 0" if ci[1] < 0 else ("above 0" if ci[0] > 0 else "contains 0")


# --- training frames ---------------------------------------------------------------------

def frames(r: int) -> dict[str, pd.DataFrame]:
    """Every arm's training frame in realization r: the replications' and the revision's."""
    return {**replicate.arm_frames(r), **rf.arm_frames(r)}


def crest_entry(frame: pd.DataFrame) -> np.ndarray:
    return (((frame["Center_Y"] - frame["H"]) / frame["H"]).abs() < NEAR_CREST).to_numpy()


def _fit_predict(family, frame, features, tests, weight=None) -> dict:
    model = models.lc_estimator(family)
    model.fit(frame[list(features)], np.log(frame["FOS"].to_numpy(float)),
              **({} if weight is None else {"sample_weight": weight}))
    return {"n_train": int(len(frame)), "train_fos": frame["FOS"].to_numpy(float),
            "predictions": {n: np.exp(model.predict(t.X[list(features)])) for n, t in tests.items()}}


def _cached(check: str, family: str, r: int, build) -> dict:
    path = config.ROOT / "artifacts" / "revision" / "robust" / f"{check}__{family}__r{r}.joblib"
    state = {"fingerprint": config.replication_fingerprint(family), "check": check}
    if path.exists():
        stored = joblib.load(path)
        if stored["state"] == state:
            return stored["arms"]
    print(f"[robust] {check} {family} r{r}", flush=True)
    arms = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    models._dump(path, {"state": state, "arms": arms})
    return arms


# --- weights (§8.4) ----------------------------------------------------------------------

def _regions(fos, edges):
    """Index of each point's region: 0 below edges[0], 1..k the bins (lo, hi] (first closed),
    k+1 above edges[-1]."""
    fos = np.asarray(fos, dtype=float)
    out = np.where(fos < edges[0], 0, len(edges))
    inside = (fos >= edges[0]) & (fos <= edges[-1])
    out[inside] = np.maximum(np.searchsorted(edges, fos[inside], side="left") - 1, 0) + 1
    return out


def intended(edges, allocation) -> np.ndarray:
    """Counts per region of the arm's intended training set: the seed plus the allocation in
    the bins, the seed alone outside them."""
    seed = data.load_role("seed")["FOS"].to_numpy(float)
    have = np.bincount(_regions(seed, edges), minlength=len(edges) + 1)
    have[1:len(edges)] += np.asarray(allocation)
    return have


def weights(fos, edges, target) -> tuple[np.ndarray, list[int]]:
    """target count / achieved count, per region; the regions with a target and no point
    are returned (they cannot be weighted)."""
    reg = _regions(fos, edges)
    achieved = np.bincount(reg, minlength=len(target))
    unweighted = [int(i) for i in range(len(target)) if target[i] > 0 and achieved[i] == 0]
    w = np.asarray(target, float)[reg] / achieved[reg]
    return w, unweighted


def targets() -> dict:
    """B-wC: C-log's intended distribution (15 selection bins and the two regions outside);
    B-wD: D's (4 bins on the decision range and the two regions outside)."""
    from sgs import pool
    sel, d = rp.sel_edges(), rp.d_edges()
    return {"B-wC": (sel, intended(sel, pool.allocation())), "B-wD": (d, intended(d, np.asarray(config.REV_D_ALLOCATION)))}


# --- the units of each check ------------------------------------------------------------

def baseline_units() -> dict:
    return {k: v for k, v in rf.merged_units().items() if k[1] >= 1}


def check_units(check: str) -> tuple[dict, dict]:
    tests, units, info = _tests(), {}, {}
    tg = targets() if check == "reweight" else None
    for family in config.REP_FAMILIES:
        for r in REALS:
            def build():
                fr = frames(r)
                if check == "three-features":
                    return {a: _fit_predict(family, f, THREE, tests) for a, f in fr.items()}
                if check == "no-crest-entry":
                    return {a: _fit_predict(family, f.loc[~crest_entry(f)].reset_index(drop=True),
                                            config.FEATURES, tests) for a, f in fr.items()}
                b = fr["B"]
                out = {}
                for name, (edges, target) in tg.items():
                    w, _ = weights(b["FOS"].to_numpy(float), edges, target)
                    out[name] = _fit_predict(family, b, config.FEATURES, tests, weight=w)
                return out
            units[(family, r)] = {"arms": _cached(check, family, r, build)}
            if check == "no-crest-entry":
                fr = frames(r)
                info[f"r{r}"] = {a: int(crest_entry(f).sum()) for a, f in fr.items()}
            if check == "reweight":
                b = frames(r)["B"]["FOS"].to_numpy(float)
                info[f"r{r}"] = {n: weights(b, e, t)[1] for n, (e, t) in tg.items()}
    return units, info


# --- readings ---------------------------------------------------------------------------

def values(units, decision=(0.5, 2.0)) -> dict:
    tests, out = _tests(), {}
    for (family, r), u in units.items():
        for arm, a in u["arms"].items():
            for name, t in tests.items():
                y, p = np.asarray(t.y, float), np.asarray(a["predictions"][name], float)
                out[(family, name, arm, r)] = {"decision": _log_rmse(p, y, *decision),
                                               "L": _log_rmse(p, y, *LOWER), "H": _log_rmse(p, y, *UPPER)}
    return out


def cell_readings(vals, arms, against_b=None) -> dict:
    """Per arm and cell: the decision-range interval against B, its verdict, the gate."""
    against_b = against_b or {}
    out = {}
    for arm in arms:
        per = {}
        for f, t in CELLS:
            a = [vals[(f, t, arm, r)]["decision"] for r in REALS]
            ref = against_b.get(arm, "B")
            b = [vals[(f, t, ref, r)]["decision"] for r in REALS]
            iv = (evaluate.paired_interval(a, b, config.REP_CI) if arm.startswith("B-w")
                  else evaluate.welch_interval(a, b, config.REP_CI))
            L = float(np.median([np.log(vals[(f, t, arm, r)]["L"] / vals[(f, t, "B", r)]["L"]) for r in REALS]))
            H = float(np.median([np.log(vals[(f, t, arm, r)]["H"] / vals[(f, t, "B", r)]["H"]) for r in REALS]))
            per[f"{f}/{t}"] = {"diff": iv["diff"], "ci": iv["ci"], "verdict": _verdict(iv["ci"]),
                               "L": L, "H": H, "gate": bool(L < 0 < H)}
        out[arm] = {"cells": per, "level": evaluate.oracle_level(
            [{"diff": c["diff"], "ci": c["ci"]} for c in per.values()], len(CELLS)),
            "gate_passes": sum(c["gate"] for c in per.values())}
    return out


def compare(base: dict, other: dict) -> dict:
    out = {}
    for arm, o in other.items():
        cells = {}
        for cell, c in o["cells"].items():
            b = base[arm]["cells"][cell]
            v = "holds" if c["verdict"] == b["verdict"] else "differs"
            g = ("holds" if c["gate"] == b["gate"] else "differs") if arm in GATE_ARMS else None
            cells[cell] = {"verdict": v, "gate": g, "base_verdict": b["verdict"], "verdict_now": c["verdict"]}
        out[arm] = {"cells": cells, "level": [base[arm]["level"], o["level"]],
                    "gate_passes": [base[arm]["gate_passes"], o["gate_passes"]] if arm in GATE_ARMS else None}
    return out


def size_matched(units_r0, base_vals) -> dict:
    """§8.5: every ablation (r1..r5) against A (one deterministic fit on the seed, r0).
    A has a single value per cell, so Welch reduces to the one-sample t interval of
    ablation − A over the five realizations; stated as such."""
    tests, out = _tests(), {}
    ablations = sorted({k[2] for k in base_vals if "0" in k[2]})
    for abl in ablations:
        per = {}
        for f, t in CELLS:
            y = np.asarray(tests[t].y, float)
            a_val = _log_rmse(np.asarray(units_r0[(f, 0)]["arms"]["A"]["predictions"][t], float), y, 0.5, 2.0)
            x = [base_vals[(f, t, abl, r)]["decision"] for r in REALS]
            iv = evaluate.paired_interval(x, [a_val] * len(x), config.REP_CI)
            per[f"{f}/{t}"] = {"A": a_val, "diff": iv["diff"], "ci": iv["ci"], "verdict": _verdict(iv["ci"])}
        out[abl] = per
    return out


def run() -> dict:
    base_units = baseline_units()
    base_vals = values(base_units)
    arms = sorted({k[2] for k in base_vals} - {"B"})
    base = cell_readings(base_vals, arms)
    # reproduction check of the baseline against the published replication metrics
    import json
    rep = {(v["family"], v["test"], v["arm"], v["realization"]): v["subsets"]["decision-range"]["log_rmse"]
           for v in json.loads(config.replication_metrics_path().read_text())["values"]}
    worst = max(abs(base_vals[k]["decision"] - rep[k]) for k in base_vals if k in rep)
    entry = {"baseline": base, "baseline_reproduction_max_abs": worst}
    for check in ("three-features", "no-crest-entry"):
        units, info = check_units(check)
        vals = values(units)
        entry[check] = {"readings": cell_readings(vals, arms),
                        "against_baseline": compare(base, cell_readings(vals, arms)), "info": info}
    units, info = check_units("reweight")
    vals = {**base_vals, **values(units)}
    entry["reweight"] = {"vs_B": cell_readings(vals, ("B-wC", "B-wD")),
                         "vs_guided": cell_readings(vals, ("B-wC", "B-wD"),
                                                    against_b={"B-wC": "C-logP", "B-wD": "D-P"}),
                         "targets": {k: {"edges": e.tolist(), "counts": v.tolist()} for k, (e, v) in targets().items()},
                         "unweighted_regions": info}
    entry["size_matched"] = size_matched(replicate.load_units(), base_vals)
    entry["alternative_ranges"] = {}
    for label, rng in ALT_RANGES.items():
        v = values(base_units, decision=rng)
        entry["alternative_ranges"][label] = {a: {"level": c["level"],
                                                  "cells": {k: {"diff": x["diff"], "ci": x["ci"]}
                                                            for k, x in c["cells"].items()}}
                                              for a, c in cell_readings(v, arms).items()}
    density.write("robustness", entry)
    return entry
