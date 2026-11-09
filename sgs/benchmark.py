"""The curated deposit's FOS baseline: one reference model on the main task.

  sgs benchmark     read data/curated/, fit, write docs/curated-benchmark/baseline.json

Main task (data/curated/README.md): predict FOS from the five inputs, train on
the `train` split, report on each test set the RMSE of log(FOS), overall and per
reporting zone. The model is a reference point and a usage example, not a tuned
result: a random forest with library defaults, trained on log(FOS).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sgs import config, curate, data, evaluate


def output_path() -> Path:
    return config.ROOT / "docs" / "curated-benchmark" / "baseline.json"


def load(base: Path) -> tuple[pd.DataFrame, pd.Series]:
    """The simulations and their split, read from the CSV files as a user would."""
    sims = pd.read_csv(base / "data" / "simulations.csv")
    splits = pd.read_csv(base / "data" / "splits.csv").set_index("simulation_id")["split"]
    return sims.set_index("simulation_id"), splits


def model() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=config.SEEDS["benchmark_baseline"], n_jobs=1)


def run_baseline(sims: pd.DataFrame, splits: pd.Series) -> dict:
    inputs = list(curate.ID_COLUMNS)
    train = sims.loc[splits[splits == "train"].index]
    estimator = model().fit(train[inputs], np.log(train["FOS"].to_numpy(float)))
    results = {}
    for test in config.TEST_ROLES:
        frame = sims.loc[splits[splits == test].index]
        y = frame["FOS"].to_numpy(float)
        p = np.exp(estimator.predict(frame[inputs]))
        results[test] = {"overall": evaluate.global_metrics(y, p),
                         "zones": evaluate.zone_metrics(y, p, config.ZONE_EDGES)}
    return {"task": "FOS from (" + ", ".join(inputs) + "); train on split 'train'; "
                    "zones on the solved FOS, [lo, hi)",
            "model": {"class": "sklearn.ensemble.RandomForestRegressor",
                      "params": "library defaults", "random_state": "SEEDS['benchmark_baseline']",
                      "target": "log(FOS)", "n_train": len(train)},
            "results": results}


def run() -> dict:
    base = curate.out_dir()
    provenance = json.loads((base / "metadata" / "provenance.json").read_text())
    sims, splits = load(base)
    out = run_baseline(sims, splits)
    out["deposit"] = {"simulations_sha256": data.sha256_file(base / "data" / "simulations.csv"),
                      "splits_sha256": data.sha256_file(base / "data" / "splits.csv"),
                      "curated_at_commit": provenance["code"]["commit"]}
    out["code"] = curate._git_state()
    out["environment"] = data.environment()
    path = output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_text_atomic(path, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for test, r in out["results"].items():
        zones = ", ".join(f"{z['zone']}: {z['log_rmse']:.4f} (n={z['n']})" for z in r["zones"] if z["n"])
        print(f"[benchmark] {test}: log-RMSE {r['overall']['log_rmse']:.4f}; {zones}")
    return out
