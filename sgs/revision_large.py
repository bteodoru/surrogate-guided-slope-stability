"""The revision's large test sets (docs/revision-declaration.md §6).

T-natL is the solved LHS of 5,000 points. T-stratL is drawn from the second solved LHS
(6,000 points): 30 points at random within each of the 15 selection bins, the geometric
bins of the selection range on which C-log allocates (``revision_pool.sel_edges``),
binned on the **solved** FOS as the generator bins, (lo, hi] with the first closed. A
bin with fewer than 30 points takes all of them, and this is recorded. The bin type is
not named in §6; "selection bins" is read as it is used in §8.4, and the choice was made
on 2026-11-18 before any arm was evaluated on the large sets.

Every replicated arm, old and new, r1..r5, is refitted exactly as stored (LC_PARAMS on
log(FOS), the same estimator and seed) and predicts both large sets; the refit is checked
to reproduce the stored predictions on the fixed test sets bit for bit, so it stands for
the stored fit. Fits go to ``artifacts/revision/large/``; the metrics and readings (levels
against B, relocation gate, zone intervals, and the comparison with the fixed-set
verdicts) to the ``large_tests`` entry of ``outputs/metrics/revision.json``.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from sgs import config, data, density, design, evaluate, models, revision
from sgs import revision_fit as rf
from sgs import revision_pool as rp
from sgs import revision_robust as rr

NATURAL, STRATIFIED = "test-natural-large", "test-stratified-large"
FIXED_OF = {NATURAL: "test-natural", STRATIFIED: "test-stratified"}
SOURCE = "test_stratified_large_source"
ZONES = ("FOS < 0.5", "0.5–1", "1–2", "2–6", "FOS ≥ 6")


def stratified_path():
    return config.ROOT / "data" / "revision" / "derived" / "test_stratified_large.csv"


def _with_ids(key: str) -> pd.DataFrame:
    frame = data.add_features(revision.load_set(key))
    frame.insert(0, "row_id", [f"rev:{key}:{i}" for i in range(len(frame))])
    return frame


def build_stratified() -> dict:
    """Draw T-stratL from the solved source and write it; returns the draw's record."""
    source = _with_ids(SOURCE)
    fos = source["FOS"].to_numpy(float)
    edges = rp.sel_edges()
    have = design.bin_counts(fos, edges)
    allocation = np.minimum(have, config.REV_TEST_STRAT_PER_BIN)
    rng = np.random.default_rng(config.SEEDS["rev_test_stratified_large_selection"])
    chosen, bins = design.select_in_bins(fos, edges, allocation, rng)
    out = source.iloc[chosen].reset_index(drop=True)
    out.insert(1, "bin", bins)
    text = out.to_csv(index=False)
    path = stratified_path()
    if path.exists():
        if data.sha256_file(path) != data.text_sha256(text):
            raise data.DataError(f"{path.name} exists and differs from the draw produced now")
    else:
        data.write_text_atomic(path, text)
        data.protect(path)
    return {"edges": edges.tolist(), "source_counts": have.tolist(), "allocation": allocation.tolist(),
            "short_bins": [int(b) for b in np.flatnonzero(have < config.REV_TEST_STRAT_PER_BIN)],
            "n": int(len(out)), "sha256": data.sha256_file(path)}


def _test(name: str, frame: pd.DataFrame) -> data.TestSet:
    return data.TestSet(name=name, X=frame[list(config.FEATURES)].reset_index(drop=True),
                        y=frame["FOS"].to_numpy(float), ids=frame["row_id"].to_numpy(str))


def large_tests() -> dict[str, data.TestSet]:
    strat = data.read_table(stratified_path())
    return {NATURAL: _test(NATURAL, _with_ids("test_natural_large")), STRATIFIED: _test(STRATIFIED, strat)}


def fit_path(family: str, r: int):
    return config.ROOT / "artifacts" / "revision" / "large" / f"{family}__r{r}.joblib"


def _state(family: str, r: int) -> dict:
    parts = sorted(k for k in data.read_metadata(config.revision_metadata_path())["sets"]
                   if k.startswith("test_natural_large_p"))
    return {"fingerprint": config.replication_fingerprint(family),
            "stratified": data.sha256_file(stratified_path()),
            "natural": {k: data.sha256_file(config.revision_path("final", k)) for k in parts}}


def fit_all() -> dict:
    """Refit and predict the large sets; returns the worst deviation of the refit from the
    stored predictions on the fixed test sets (0 when the refit reproduces them)."""
    tests, fixed = large_tests(), {role: data.load_test(role) for role in config.TEST_ROLES}
    stored = rf.merged_units()
    worst = 0.0
    for family in config.REP_FAMILIES:
        for r in range(1, config.REP_REALIZATIONS + 1):
            path, state = fit_path(family, r), _state(family, r)
            if path.exists() and joblib.load(path)["state"] == state:
                print(f"[large] {path.name}: up to date")
                worst = max(worst, joblib.load(path)["reproduction_max_abs"])
                continue
            print(f"[large] fit {family} r{r}", flush=True)
            arms, dev = {}, 0.0
            for arm, frame in rr.frames(r).items():
                for name, test in tests.items():
                    if design.nearest_distance(frame, test.X).min() <= config.MIN_DISTANCE:
                        raise data.DataError(f"{arm} r{r} is not disjoint from {name}")
                model = models.lc_estimator(family).fit(frame[list(config.FEATURES)],
                                                        np.log(frame["FOS"].to_numpy(float)))
                for name, test in fixed.items():
                    again = np.exp(model.predict(test.X))
                    dev = max(dev, float(np.max(np.abs(again - stored[(family, r)]["arms"][arm]["predictions"][name]))))
                arms[arm] = {"n_train": int(len(frame)), "train_ids": frame["row_id"].to_numpy(str),
                             "train_fos": frame["FOS"].to_numpy(float),
                             "predictions": {n: np.exp(model.predict(t.X)) for n, t in tests.items()}}
            path.parent.mkdir(parents=True, exist_ok=True)
            models._dump(path, {"state": state, "family": family, "realization": r,
                                "reproduction_max_abs": dev,
                                "test_ids": {n: t.ids for n, t in tests.items()}, "arms": arms})
            worst = max(worst, dev)
    return {"reproduction_max_abs": worst}


def units() -> dict:
    out = {}
    for family in config.REP_FAMILIES:
        for r in range(1, config.REP_REALIZATIONS + 1):
            stored = joblib.load(fit_path(family, r))
            if stored["state"] != _state(family, r):
                raise models.StaleFit(f"{fit_path(family, r).name} is stale; rerun the large-set fits")
            out[(family, r)] = stored
    return out


def _verdict(ci) -> str:
    return "below 0" if ci[1] < 0 else ("above 0" if ci[0] > 0 else "contains 0")


def readings(result: dict, arms) -> dict:
    comp = {(c["family"], c["test"], c["measure"], c["first"], c["second"], c["scale"]): c
            for c in result["comparisons"]}
    vals = {(v["family"], v["test"], v["arm"], v["realization"]): v for v in result["values"]}
    reals = result["confirmatory"]
    cells = [(f, t) for f in config.REP_FAMILIES for t in (NATURAL, STRATIFIED)]
    fixed = json.loads(density.revision_metrics_path().read_text())["robustness"]["baseline"]
    out = {"levels": {}, "gate": {}, "zones_vs_B": {}, "against_fixed": {}}
    for arm in arms:
        ivs = [comp[(f, t, "decision-range", arm, "B", "log-rmse")] for f, t in cells]
        out["levels"][arm] = {"level": evaluate.oracle_level(ivs, len(cells)),
                              "cells": {f"{f}/{t}": {"diff": iv["diff"], "ci": iv["ci"]}
                                        for (f, t), iv in zip(cells, ivs)}}
        gate = {}
        for f, t in cells:
            def ratio(z):
                return [float(np.log(evaluate._measure_value(vals[(f, t, arm, r)], z)
                                     / evaluate._measure_value(vals[(f, t, "B", r)], z))) for r in reals]
            gate[f"{f}/{t}"] = {"L": float(np.median(ratio("0.5–1"))), "H": float(np.median(ratio("1–2")))}
        out["gate"][arm] = {"cells": gate, "passes": sum(1 for g in gate.values() if g["L"] < 0 < g["H"])}
        out["zones_vs_B"][arm] = {f"{f}/{t}": {z: {"diff": comp[(f, t, z, arm, "B", "log-rmse")]["diff"],
                                                    "ci": comp[(f, t, z, arm, "B", "log-rmse")]["ci"]}
                                               for z in ZONES if (f, t, z, arm, "B", "log-rmse") in comp}
                                  for f, t in cells}
        per = {}
        for (f, t), iv in zip(cells, ivs):
            base = fixed[arm]["cells"][f"{f}/{FIXED_OF[t]}"]
            g = gate[f"{f}/{t}"]
            now_gate = bool(g["L"] < 0 < g["H"])
            per[f"{f}/{t}"] = {"verdict_fixed": base["verdict"], "verdict_large": _verdict(iv["ci"]),
                               "verdict": "holds" if base["verdict"] == _verdict(iv["ci"]) else "differs",
                               "gate_fixed": base["gate"], "gate_large": now_gate,
                               "gate": "holds" if base["gate"] == now_gate else "differs"}
        out["against_fixed"][arm] = {"cells": per, "level": [fixed[arm]["level"], out["levels"][arm]["level"]],
                                     "gate_passes": [fixed[arm]["gate_passes"], out["gate"][arm]["passes"]]}
    return out


def run() -> dict:
    draw = build_stratified()
    reproduction = fit_all()
    tests = large_tests()
    u = units()
    arms = sorted(set(u[(config.REP_FAMILIES[0], 1)]["arms"]) - {"B"})
    result = evaluate.replication_metrics(
        u, tests, zones=config.ZONE_EDGES,
        bins=tuple(design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)),
        selection_range=config.SELECTION_RANGE, decision_range=config.DECISION_RANGE, level=config.REP_CI,
        welch=tuple((a, "B") for a in arms), paired=(), interactions=())
    entry = {"stratified_draw": draw, **reproduction,
             "test_sizes": {n: int(len(t.y)) for n, t in tests.items()},
             "test_zone_counts": {n: {z: int(c) for z, c in zip(ZONES, np.bincount(
                 np.searchsorted(config.ZONE_EDGES, t.y, side="right"), minlength=len(ZONES)))}
                 for n, t in tests.items()},
             "comparisons": result["comparisons"], "values": result["values"],
             "readings": readings(result, arms)}
    density.write("large_tests", entry)
    return entry
