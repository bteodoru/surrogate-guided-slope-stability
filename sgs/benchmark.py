"""The curated deposit's FOS baselines: reference models on the benchmark tasks.

  sgs benchmark     read data/curated/, fit, write docs/curated-benchmark/v1.1/

Main task (data/curated/README.md): predict FOS from the five inputs, train on
the `train` split, report on each test split the RMSE of log(FOS), overall and per
reporting zone. Version 1.1 adds the grouped split (`split_grouped`, held-out cells
of the dimensionless space) and two reference points that need no fitting:

  baseline.json           random forest, library defaults, on log(FOS): the fixed
                          test splits and the grouped split
  nearest_neighbour.json  distances to the nearest training analysis (train→train,
                          test→train) in the dimensionless space (angle, phi,
                          ln c/(gamma H)) and in the five normalized inputs; and the
                          1-NN predictor in each space, on the same splits

The models are reference points and usage examples, not tuned results. Version 1.0's
record, docs/curated-benchmark/baseline.json, is kept as it was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

from sgs import config, curate, data, evaluate

# The share of test analyses whose nearest training analysis is closer than this quantile of
# the train→train nearest-neighbour distances is reported beside the distance quantiles.
NEAR_QUANTILE = 0.05
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def output_dir() -> Path:
    return config.ROOT / "docs" / "curated-benchmark" / "v1.1"


def output_path() -> Path:
    """Version 1.0's record (the RF baseline on the two fixed test sets); not rewritten."""
    return config.ROOT / "docs" / "curated-benchmark" / "baseline.json"


def load(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The simulations and their splits, read from the CSV files as a user would."""
    sims = pd.read_csv(base / "data" / "simulations.csv")
    splits = pd.read_csv(base / "data" / "splits.csv").set_index("simulation_id")
    return sims.set_index("simulation_id"), splits


def tasks(splits: pd.DataFrame) -> dict[str, tuple[pd.Index, dict[str, pd.Index]]]:
    """task -> (training ids, {test name: test ids}): the fixed splits, then the grouped one."""
    fixed = splits["split"]
    grouped = splits["split_grouped"]
    return {"fixed": (fixed[fixed == "train"].index,
                      {t: fixed[fixed == t].index for t in curate.TEST_SPLITS if (fixed == t).any()}),
            "grouped": (grouped[grouped == "train"].index, {"test": grouped[grouped == "test"].index})}


def model() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=config.SEEDS["benchmark_baseline"], n_jobs=1)


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"overall": evaluate.global_metrics(y, p), "zones": evaluate.zone_metrics(y, p, config.ZONE_EDGES)}


def run_baseline(sims: pd.DataFrame, splits: pd.DataFrame) -> dict:
    inputs = list(curate.ID_COLUMNS)
    results = {}
    for task, (train_ids, tests) in tasks(splits).items():
        train = sims.loc[train_ids]
        estimator = model().fit(train[inputs], np.log(train["FOS"].to_numpy(float)))
        results[task] = {"n_train": len(train), "tests": {}}
        for name, ids in tests.items():
            frame = sims.loc[ids]
            results[task]["tests"][name] = _scores(frame["FOS"].to_numpy(float),
                                                   np.exp(estimator.predict(frame[inputs])))
    return {"task": "FOS from (" + ", ".join(inputs) + "); zones on the solved FOS, [lo, hi)",
            "model": {"class": "sklearn.ensemble.RandomForestRegressor", "params": "library defaults",
                      "random_state": "SEEDS['benchmark_baseline']", "target": "log(FOS)",
                      "preprocessing": "none: the five inputs as stored, unscaled"},
            "results": results}


def spaces(sims: pd.DataFrame) -> dict[str, np.ndarray]:
    """The two spaces, each in the unit cube: the dimensionless one (3D) and the five inputs (5D)."""
    return {"3d": curate.dimensionless_unit(sims),
            "5d": data.normalize(sims)}


def _quantiles(d: np.ndarray) -> dict:
    return {f"q{round(100 * q):02d}": float(np.quantile(d, q)) for q in QUANTILES}


def run_nearest_neighbour(sims: pd.DataFrame, splits: pd.DataFrame) -> dict:
    position = pd.Series(np.arange(len(sims)), index=sims.index)
    fos = sims["FOS"].to_numpy(float)
    out = {"spaces": {"3d": "angle, phi, ln c/(gamma H), each scaled to [0, 1] by the bounds of the domain",
                      "5d": "H, angle, phi, c, gamma, each scaled to [0, 1] by the bounds of the domain"},
           "near_quantile": NEAR_QUANTILE, "results": {}}
    for space, coords in spaces(sims).items():
        out["results"][space] = {}
        for task, (train_ids, tests) in tasks(splits).items():
            train = position[train_ids].to_numpy()
            tree = cKDTree(coords[train])
            self_d = tree.query(coords[train], k=2)[0][:, 1]
            near = float(np.quantile(self_d, NEAR_QUANTILE))
            entry = {"n_train": int(len(train)), "train_to_train": _quantiles(self_d), "tests": {}}
            for name, ids in tests.items():
                test = position[ids].to_numpy()
                d, j = tree.query(coords[test], k=1)
                entry["tests"][name] = {"test_to_train": _quantiles(d),
                                        "share_within_near_quantile": float(np.mean(d < near)),
                                        "one_nn": _scores(fos[test], fos[train][j])}
            out["results"][space][task] = entry
    return out


def _record(sims_path: Path, splits_path: Path, provenance: dict) -> dict:
    return {"deposit": {"simulations_sha256": data.sha256_file(sims_path),
                        "splits_sha256": data.sha256_file(splits_path),
                        "curated_version": provenance.get("version", "1.0"),
                        "curated_at_commit": provenance["code"]["commit"]},
            "code": curate._git_state(), "environment": data.environment()}


def _write(name: str, payload: dict) -> Path:
    path = output_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _print(label: str, scores: dict) -> None:
    zones = ", ".join(f"{z['zone']}: {z['log_rmse']:.4f} (n={z['n']})" for z in scores["zones"] if z["n"])
    print(f"[benchmark] {label}: log-RMSE {scores['overall']['log_rmse']:.4f}; {zones}")


def run_nn_only() -> dict:
    """The distances and the 1-NN predictor alone (seconds); writes nearest_neighbour.json."""
    base = curate.out_dir()
    provenance = json.loads((base / "metadata" / "provenance.json").read_text())
    sims, splits = load(base)
    out = run_nearest_neighbour(sims, splits)
    out.update(_record(base / "data" / "simulations.csv", base / "data" / "splits.csv", provenance))
    _write("nearest_neighbour.json", out)
    for space, tasks_ in out["results"].items():
        for task, entry in tasks_.items():
            for name, r in entry["tests"].items():
                _print(f"1-NN {space} {task}/{name}", r["one_nn"])
    return out


def run() -> dict:
    base = curate.out_dir()
    provenance = json.loads((base / "metadata" / "provenance.json").read_text())
    sims, splits = load(base)
    record = _record(base / "data" / "simulations.csv", base / "data" / "splits.csv", provenance)
    nn = run_nearest_neighbour(sims, splits)
    _write("nearest_neighbour.json", {**nn, **record})
    out = run_baseline(sims, splits)
    _write("baseline.json", {**out, **record})
    for task, r in out["results"].items():
        for name, scores in r["tests"].items():
            _print(f"RF {task}/{name}", scores)
    return out
