"""The revision's seed study (docs/revision-declaration.md §5).

One realization per seed. Seeds 1..4 are the new LHS seeds; seed 42, with its r1 sets
(B r1, C-logP r1 and D-P r1 from sub-pool 1), is the fifth, carried here as unit 5. Every
arm is fitted with LC_PARAMS on log(FOS), no tuning, like the replications. Intervals
are paired over the five seeds (a seed's arms share that seed). The fits go to
``artifacts/revision/seeds/``; the metrics and readings to the ``seed_study`` entry of
``outputs/metrics/revision.json``.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sgs import config, data, density, design, evaluate, models, revision
from sgs import revision_fit as rf
from sgs import revision_pool as rp

ARMS = ("B", "C-logP", "C0-logP", "D-P", "D0-P")
PAIRED = (("C-logP", "B"), ("D-P", "B"), ("C0-logP", "B"), ("D0-P", "B"),
          ("C-logP", "C0-logP"), ("D-P", "D0-P"), ("D-P", "C-logP"))
LEVEL_ARMS = ("C-logP", "D-P", "C0-logP", "D0-P")
UNIT_OF_42 = 5          # seed 42, its r1 sets


def fit_path(family: str, s: int):
    return config.ROOT / "artifacts" / "revision" / "seeds" / f"{family}__s{s}.joblib"


def _solved(key: str) -> pd.DataFrame:
    frame = data.add_features(revision.load_set(key))
    frame.insert(0, "row_id", [f"rev:{key}:{i}" for i in range(len(frame))])
    return frame


def arm_frames(s: int) -> dict[str, pd.DataFrame]:
    seed = _solved(f"seed_s{s}")
    logp, dp = rp.load_set(f"logp_s{s}"), rp.load_set(f"dp_s{s}")
    cols = ["row_id", *config.FEATURES, "FOS"]
    return {"B": pd.concat([seed, _solved(f"extension_s{s}")], ignore_index=True)[cols],
            "C-logP": pd.concat([seed[cols], logp[cols]], ignore_index=True), "C0-logP": logp[cols],
            "D-P": pd.concat([seed[cols], dp[cols]], ignore_index=True), "D0-P": dp[cols]}


def _state(family: str, s: int) -> dict:
    paths = [config.final_path(role) for role in config.TEST_ROLES]
    paths += [config.revision_path("final", k) for k in (f"logp_s{s}", f"dp_s{s}")]
    meta = data.read_metadata(config.revision_metadata_path())["sets"]
    parts = sorted(k for k in meta if k.startswith((f"seed_s{s}_p", f"extension_s{s}_p")))
    return {"fingerprint": config.replication_fingerprint(family),
            "data": {**{p.name: data.sha256_file(p) for p in paths},
                     **{k: data.sha256_file(config.revision_path("final", k)) for k in parts}}}


def fit_all() -> None:
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    for family in config.REP_FAMILIES:
        for s in range(1, config.REV_SEEDS + 1):
            path, state = fit_path(family, s), _state(family, s)
            if path.exists() and joblib.load(path)["state"] == state:
                print(f"[seeds] {path.name}: up to date")
                continue
            print(f"[seeds] fit {family} s{s}", flush=True)
            arms = {}
            for arm, frame in arm_frames(s).items():
                for name, test in tests.items():
                    if design.nearest_distance(frame, test.X).min() <= config.MIN_DISTANCE:
                        raise data.DataError(f"{arm} s{s} is not disjoint from {name}")
                model = models.lc_estimator(family).fit(frame[list(config.FEATURES)],
                                                        np.log(frame["FOS"].to_numpy(float)))
                arms[arm] = {"n_train": int(len(frame)), "train_ids": frame["row_id"].to_numpy(str),
                             "train_fos": frame["FOS"].to_numpy(float),
                             "predictions": {n: np.exp(model.predict(t.X)) for n, t in tests.items()}}
            path.parent.mkdir(parents=True, exist_ok=True)
            models._dump(path, {"state": state, "family": family, "seed": s,
                                "test_ids": {n: t.ids for n, t in tests.items()}, "arms": arms})


def units() -> dict:
    out = {}
    r1 = rf.merged_units()
    for family in config.REP_FAMILIES:
        for s in range(1, config.REV_SEEDS + 1):
            stored = joblib.load(fit_path(family, s))
            if stored["state"] != _state(family, s):
                raise models.StaleFit(f"{fit_path(family, s).name} is stale; run the seed fits")
            out[(family, s)] = stored
        old = r1[(family, 1)]
        out[(family, UNIT_OF_42)] = {"test_ids": old["test_ids"],
                                       "arms": {a: old["arms"][a] for a in ARMS}}
    return out


def readings(result: dict) -> dict:
    comp = {(c["family"], c["test"], c["measure"], c["first"], c["second"], c["scale"]): c
            for c in result["comparisons"]}
    vals = {(v["family"], v["test"], v["arm"], v["realization"]): v for v in result["values"]}
    seeds = result["confirmatory"]
    cells = [(f, t) for f in config.REP_FAMILIES for t in config.TEST_ROLES]
    out = {"levels": {}, "gate": {}}
    for arm in LEVEL_ARMS:
        ivs = [comp[(f, t, "decision-range", arm, "B", "log-rmse")] for f, t in cells]
        out["levels"][arm] = {"level": evaluate.oracle_level(ivs, len(cells)),
                              "cells": {f"{f}/{t}": {"diff": iv["diff"], "ci": iv["ci"]}
                                        for (f, t), iv in zip(cells, ivs)}}
        gate = {}
        for f, t in cells:
            def ratio(z):
                return [float(np.log(evaluate._measure_value(vals[(f, t, arm, s)], z)
                                     / evaluate._measure_value(vals[(f, t, "B", s)], z))) for s in seeds]
            lo, hi = ratio("0.5–1"), ratio("1–2")
            gate[f"{f}/{t}"] = {"L": float(np.median(lo)), "H": float(np.median(hi)),
                                "L_per_seed": lo, "H_per_seed": hi}
        out["gate"][arm] = {"cells": gate, "passes": sum(1 for g in gate.values() if g["L"] < 0 < g["H"])}
    out["prediction_direction_C-logP"] = out["gate"]["C-logP"]["passes"] >= config.RELOCATION_MIN_CELLS
    return out


def run() -> dict:
    fit_all()
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    result = evaluate.replication_metrics(
        units(), tests, zones=config.ZONE_EDGES,
        bins=tuple(design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)),
        selection_range=config.SELECTION_RANGE, decision_range=config.DECISION_RANGE, level=config.REP_CI,
        welch=(), paired=PAIRED, interactions=())
    entry = {"units": {"1-4": "new seeds s1..s4 (rev_seed_lhs_s)", str(UNIT_OF_42): "seed 42, r1 sets"},
             "comparisons": result["comparisons"], "values": result["values"], "readings": readings(result)}
    density.write("seed_study", entry)
    return entry
