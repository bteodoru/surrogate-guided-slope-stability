"""Fits and readings of the revision's pool-backed arms on the fixed test sets
(docs/revision-declaration.md §2-§4).

The new arms of realization r are fitted exactly as replicate fits its arms: the
estimator of the family with LC_PARAMS, on log(FOS), no tuning. The fits go to
``artifacts/revision/``. The metrics reuse ``evaluate.replication_metrics`` on the union
of the published replication units (r1..r5, read only) and the new ones, and are written
to the ``fixed_tests`` entry of ``outputs/metrics/revision.json``, with the declared
readings: the level against B on the decision range, the relocation gate, the zone
intervals, and the density prediction's direction and calibration (§3).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from sgs import config, data, density, design, evaluate, models, replicate
from sgs import revision_pool as rp

NEW_ARMS = ("D-oracle", "D0-oracle", "D-P", "D0-P", "C-seqF", "C0-seqF", "C-seqP", "C0-seqP")
WELCH = (("D-oracle", "B"), ("D-P", "B"), ("D0-oracle", "B"), ("D0-P", "B"),
         ("C-seqF", "B"), ("C-seqP", "B"), ("C0-seqF", "B"), ("C0-seqP", "B"),
         ("C-seq", "C-seqP"))
PAIRED = (("D-oracle", "D-P"), ("D-P", "C-logP"), ("D-oracle", "C-oracle"),
          ("D-oracle", "D0-oracle"), ("D-P", "D0-P"),
          ("C-seqF", "C-logP"), ("C-seqP", "C-seqF"), ("C-seqF", "C0-seqF"), ("C-seqP", "C0-seqP"))
LEVEL_ARMS = ("D-oracle", "D-P", "C-seqF", "C-seqP", "D0-oracle", "D0-P", "C0-seqF", "C0-seqP")


def fit_path(family: str, r: int):
    return config.ROOT / "artifacts" / "revision" / f"{family}__r{r}.joblib"


def arm_frames(r: int) -> dict[str, pd.DataFrame]:
    seed = data.load_role("seed")
    added = rp.arm_sets(r)
    out = {}
    for ablation, frame in added.items():
        out[ablation.replace("0", "", 1)] = pd.concat([seed, frame], ignore_index=True)
        out[ablation] = frame
    return out


def _state(family: str, r: int) -> dict:
    keys = [f"d_oracle_r{r}", f"d_p_r{r}", *(f"{s}_r{r}_b{b}" for s in ("seq_f", "seq_p") for b in (1, 2, 3))]
    paths = [config.final_path(role) for role in ("seed", *config.TEST_ROLES)]
    paths += [config.revision_path("final", k) for k in keys]
    return {"fingerprint": config.replication_fingerprint(family),
            "data": {p.name: data.sha256_file(p) for p in paths}}


def fit_all() -> None:
    data.validate_sets(("seed", *config.TEST_ROLES))
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    for family in config.REP_FAMILIES:
        for r in range(1, config.REP_REALIZATIONS + 1):
            path, state = fit_path(family, r), _state(family, r)
            if path.exists() and joblib.load(path)["state"] == state:
                print(f"[revision] {path.name}: up to date")
                continue
            print(f"[revision] fit {family} r{r}", flush=True)
            arms = {}
            for arm, frame in arm_frames(r).items():
                # every set used together is pairwise disjoint from the test sets
                for name, test in tests.items():
                    d = design.nearest_distance(frame, test.X)
                    if d.min() <= config.MIN_DISTANCE:
                        raise data.DataError(f"{arm} r{r} is not disjoint from {name}")
                model = models.lc_estimator(family).fit(frame[list(config.FEATURES)],
                                                        np.log(frame["FOS"].to_numpy(float)))
                arms[arm] = {"n_train": int(len(frame)), "train_ids": frame["row_id"].to_numpy(str),
                             "train_fos": frame["FOS"].to_numpy(float),
                             "predictions": {n: np.exp(model.predict(t.X)) for n, t in tests.items()}}
            path.parent.mkdir(parents=True, exist_ok=True)
            models._dump(path, {"state": state, "family": family, "realization": r,
                                "test_ids": {n: t.ids for n, t in tests.items()}, "arms": arms})


def merged_units() -> dict:
    units = {k: v for k, v in replicate.load_units().items() if k[1] >= 1}
    for (family, r), stored in units.items():
        new = joblib.load(fit_path(family, r))
        if new["state"] != _state(family, r):
            raise models.StaleFit(f"{fit_path(family, r).name} is stale; run the revision fits")
        for name, ids in new["test_ids"].items():
            if not np.array_equal(ids, stored["test_ids"][name]):
                raise evaluate.MisalignedPair(f"revision {family} r{r}: test ids differ")
        stored = {**stored, "arms": {**stored["arms"], **new["arms"]}}
        units[(family, r)] = stored
    return units


def readings(result: dict, stage1: dict) -> dict:
    """The declared readings for the new arms, per learner x test cell."""
    comp = {(c["family"], c["test"], c["measure"], c["first"], c["second"], c["scale"]): c
            for c in result["comparisons"]}
    vals = {(v["family"], v["test"], v["arm"], v["realization"]): v for v in result["values"]}
    reals = result["confirmatory"]
    cells = [(f, t) for f in config.REP_FAMILIES for t in config.TEST_ROLES]
    zones = ("FOS < 0.5", "0.5–1", "1–2", "2–6", "FOS ≥ 6")
    out = {"levels": {}, "gate": {}, "zones_vs_B": {}, "density": {}}
    for arm in LEVEL_ARMS:
        ivs = [comp[(f, t, "decision-range", arm, "B", "log-rmse")] for f, t in cells]
        out["levels"][arm] = {"level": evaluate.oracle_level(ivs, len(cells)),
                              "cells": {f"{f}/{t}": {"diff": iv["diff"], "ci": iv["ci"]}
                                        for (f, t), iv in zip(cells, ivs)}}
        gate = {}
        for f, t in cells:
            ratio = lambda z: [float(np.log(evaluate._measure_value(vals[(f, t, arm, r)], z)
                                            / evaluate._measure_value(vals[(f, t, "B", r)], z))) for r in reals]
            gate[f"{f}/{t}"] = {"L": float(np.median(ratio("0.5–1"))), "H": float(np.median(ratio("1–2")))}
        out["gate"][arm] = {"cells": gate, "passes": sum(1 for g in gate.values() if g["L"] < 0 < g["H"])}
        out["zones_vs_B"][arm] = {f"{f}/{t}": {z: {"diff": comp[(f, t, z, arm, "B", "log-rmse")]["diff"],
                                                    "ci": comp[(f, t, z, arm, "B", "log-rmse")]["ci"]}
                                               for z in zones}
                                  for f, t in cells}
    margin = float(np.log(config.RELOCATION_MARGIN))
    for arm in ("D-oracle", "D-P"):
        pred = np.median(np.array(stage1["density_prediction"][arm]["predicted_log_ratio_vs_B"],
                                  dtype=float), axis=0)
        per_zone = {}
        for zi, z in enumerate(zones):
            obs = []
            for f, t in cells:
                b_mean = np.mean([evaluate._measure_value(vals[(f, t, "B", r)], z) for r in reals])
                o = [np.log(evaluate._measure_value(vals[(f, t, arm, r)], z) / b_mean) for r in reals]
                obs.append(float(np.median(o)))
            p = float(pred[zi])
            per_zone[z] = {"predicted": p, "observed_medians": obs,
                           "direction_met": sum(1 for x in obs if np.sign(x) == np.sign(p)) >= 3,
                           "calibration_met": sum(1 for x in obs if abs(x - p) < margin) >= 3}
        dr = {}
        for f, t in cells:
            b_mean = np.mean([vals[(f, t, "B", r)]["subsets"]["decision-range"]["log_rmse"] for r in reals])
            o = float(np.median([np.log(vals[(f, t, arm, r)]["subsets"]["decision-range"]["log_rmse"] / b_mean)
                                 for r in reals]))
            p = stage1["density_prediction"][arm]["decision_range"][f"{f}/{t}"]["predicted_log_ratio"]
            dr[f"{f}/{t}"] = {"predicted": p, "observed_median": o, "within_margin": abs(o - p) < margin}
        per_zone["decision-range"] = {"cells": dr,
                                      "direction_met": sum(1 for v in dr.values()
                                                           if np.sign(v["observed_median"]) == np.sign(v["predicted"])) >= 3,
                                      "calibration_met": sum(1 for v in dr.values() if v["within_margin"]) >= 3}
        out["density"][arm] = per_zone
    return out


def run() -> dict:
    fit_all()
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    result = evaluate.replication_metrics(
        merged_units(), tests, zones=config.ZONE_EDGES,
        bins=tuple(design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)),
        selection_range=config.SELECTION_RANGE, decision_range=config.DECISION_RANGE, level=config.REP_CI,
        welch=WELCH, paired=PAIRED, interactions=())
    stage1 = json.loads(density.revision_metrics_path().read_text())["stage1"]
    entry = {"comparisons": [c for c in result["comparisons"]
                             if c["first"] in NEW_ARMS or c["second"] in NEW_ARMS or c["first"] == "C-seq"],
             "values": [v for v in result["values"] if v["arm"] in NEW_ARMS],
             "readings": readings(result, stage1)}
    density.write("fixed_tests", entry)
    return entry
