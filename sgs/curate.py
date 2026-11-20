"""The curated deposit: the published data reorganized for reuse, derived from
doi:10.5281/zenodo.23618253 and adding no analysis.

  sgs curate        read data/, verify it against run_metadata.json, write data/curated/

Three tables, each as CSV and Parquet:
  simulations   one row per distinct analysis (30 237), keyed by simulation_id
  membership    one row per row of the published deposit (33 250): which set,
                file and row each simulation comes from
  splits        the benchmark split of every simulation

simulation_id depends on the five inputs only, never on row order (ID_RECIPE).
Everything under data/ is read, never written, apart from data/curated/; the
published deposit's hashes in run_metadata.json are checked before anything is
built, and nothing is built from a file that does not match them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess
from importlib import metadata as _metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sgs import config, data, replicate, revision_fit

# The source deposit: version 1.1 holds every file of version 1.0 unchanged and adds data/revision/.
SOURCE_DOI = "10.5281/zenodo.24374247"          # version 1.1 of the source deposit
SOURCE_VERSION = "1.1"
SOURCE_V10_DOI = "10.5281/zenodo.23618253"      # version 1.0, the files outside data/revision/
SOURCE_CONCEPT_DOI = "10.5281/zenodo.23618252"  # all versions of the source deposit
CURATED_VERSION = "1.1"
CURATED_DOI = None                               # version 1.1 of this deposit: reserved with its Zenodo draft
CURATED_V10_DOI = "10.5281/zenodo.24088012"      # version 1.0 (published 2026-11-10)
CURATED_CONCEPT_DOI = "10.5281/zenodo.24088011"  # all versions of this deposit
CODE_DOI = "10.5281/zenodo.23727658"          # concept DOI of the code (all versions)
CODE_RELEASE = "v1.3.0"
CODE_RELEASE_DOI = None                          # the version DOI of CODE_RELEASE, once archived

# The id recipe, stated once; the README prints it from here.
ID_COLUMNS = ("H", "angle", "phi", "c", "gamma")
ID_SEPARATOR = "|"
ID_PREFIX = "sim_"
ID_HEX = 16
# Seven significant digits: a step of at most 1e-5 on these inputs, far below the
# 3.6e-3 by which two distinct points differ in at least one input (MIN_DISTANCE in
# the normalized domain), and coarse enough that a last-digit parsing difference
# cannot move an id. stable_ids() checks the margin on the data.
ID_DIGITS = 7
ID_ULP_MARGIN = 1024
ID_RECIPE = (f"{ID_PREFIX} + the first {ID_HEX} hexadecimal characters of SHA-256 over the UTF-8 text "
             f"{ID_SEPARATOR!r}.join(f'{{float(x):.{ID_DIGITS}g}}' for x in ({', '.join(ID_COLUMNS)}))")

OUTPUT_COLUMNS = [c for c in data.FINAL_COLUMNS if c != "excluded"]   # excluded is empty everywhere
INPUTS = list(ID_COLUMNS)
# Columns compared between the rows of one simulation; Runtime_s is wall-clock time.
DETERMINISTIC = [c for c in OUTPUT_COLUMNS if c != "Runtime_s"]

_FLOAT = ["H", "angle", "phi", "c", "gamma", "FOS", "Surface_L", "Surface_R", "Surface_Min_Y",
          "Center_X", "Center_Y", "Radius", "Circle_Nadir_Y", "slope_length", "extension", "depth",
          "Runtime_s"]
_INT = ["N_Circles_Tested", "N_Valid_Surfaces", "N_Unsolved", "Tension_Slices", "cell_id"]
_BOOL = ["At_Boundary", "Search_Converged", "Crest_Entry"]
# version 1.1 columns that are empty where they do not apply
_FLOAT_OPTIONAL = ["Min_Normal_Force", "FOS_converged"]
_INT_OPTIONAL = ["N_Slices", "Min_Normal_Force_Slice", "Converged_Max_Iter"]

SPLITS = ("train", "test-natural", "test-stratified", "test-natural-large", "unassigned")
# The test splits: the study's two fixed test sets and, from version 1.1, the revision's large
# natural test set (T-natL). Each must be disjoint from every other simulation.
TEST_SPLITS = (*config.TEST_ROLES, "test-natural-large")
_SPLIT_SETS = (("train", "pool"), ("test-natural", "test-natural"), ("test-stratified", "test-stratified"),
               ("test-natural-large", "test-natural-large"))

# design labels for the replicated strategies; the fixed roles carry their own (config.ROLES)
_STRATEGY_DESIGN = {"lhs": "lhs", "guided-log": "guided-log-deficit", "guided-log-g": "guided-log-deficit",
                    "guided-seq": "guided-log-deficit", "guided-oracle": "pool-selection",
                    "guided-log-p": "pool-selection"}

# --- version 1.1: the revision's sets (data/revision/, source deposit version 1.1) -----------
# Solved sets: (revision key, curated set name, design). Pool-backed sets: revision strategy ->
# curated set name; every one is a copy of pool rows. The large stratified test set is a subset
# of its solved source, read from data/revision/derived/.
_REV_SOLVED = {"seed": ("seed-study-seed", "lhs"), "extension": ("seed-study-lhs-extension", "lhs"),
               "test_natural_large": ("test-natural-large", "lhs"),
               "test_stratified_large_source": ("test-stratified-large-source", "lhs")}
_REV_POOL_SETS = {"d-oracle": "guided-d-oracle", "d-p": "guided-d-p", "seq-f": "guided-seq-f",
                  "seq-p": "guided-seq-p", "logp-seed": "seed-study-guided-log-p",
                  "dp-seed": "seed-study-guided-d-p"}
REV_STRATIFIED = ("test-stratified-large", "stratified-solved")
SUBSET_INPUT_ULP = 4
_MEMBERSHIP_FIELDS = ("origin", "set", "realization", "batch", "design", "is_copy", "seed_index")


class CurateError(RuntimeError):
    """The source data are not the published deposit, or a curated invariant fails."""


def out_dir() -> Path:
    return config.ROOT / "data" / "curated"


def _id_text(values) -> str:
    return ID_SEPARATOR.join(f"{float(v):.{ID_DIGITS}g}" for v in values)


def simulation_id(values) -> str:
    return ID_PREFIX + hashlib.sha256(_id_text(values).encode()).hexdigest()[:ID_HEX]


def stable_ids(simulations: pd.DataFrame) -> dict:
    """Every id must survive a change of up to ID_ULP_MARGIN units in the last place
    of any input, so that a reader's float parser cannot change it."""
    moved = 0
    for column in INPUTS:
        values = simulations[column].astype(float).to_numpy()
        step = ID_ULP_MARGIN * np.spacing(values)
        base = [f"{v:.{ID_DIGITS}g}" for v in values]
        for shifted in (values - step, values + step):
            moved += sum(f"{v:.{ID_DIGITS}g}" != b for v, b in zip(shifted, base))
    if moved:
        raise CurateError(f"{moved} inputs lie within {ID_ULP_MARGIN} ulp of a rounding boundary "
                          f"at {ID_DIGITS} significant digits")
    return {"significant_digits": ID_DIGITS, "ulp_margin": ID_ULP_MARGIN, "inputs_moved": 0}


# --- the sources ------------------------------------------------------------------

def sources() -> list[dict]:
    """Every final file of the published deposit, in canonical order: the pool, the
    fixed sets (config.ROLE_ORDER), the replications (r1..r5, generation order). The
    first row of a simulation in this order is the one simulations.csv keeps."""
    meta = data.read_metadata()
    out = []
    for key in sorted(k for k in meta.get("pool", {}) if k != "stop"):
        out.append({"path": config.pool_path("final", int(key[1:])), "origin": "pool", "set": "pool",
                    "realization": None, "batch": None, "seed_index": None, "design": "uniform-iid",
                    "is_copy": False, "sha256": meta["pool"][key]["final_sha256"]})
    for role in config.ROLE_ORDER:
        spec = config.ROLES[role]
        fixed = role == "seed" or role in config.TEST_ROLES
        batch = int(role[-1]) if role.startswith("guided-seq-") else None
        out.append({"path": config.final_path(role), "origin": "fixed", "set": role,
                    "realization": None if fixed else 0, "batch": batch, "seed_index": None,
                    "design": spec.design, "is_copy": False,
                    "sha256": meta["datasets"][role]["final_sha256"]})
    for r in range(1, config.REP_REALIZATIONS + 1):
        for key, strategy, batch in replicate.set_keys(r):
            out.append({"path": config.replication_path("final", key), "origin": "replication",
                        "set": strategy, "realization": r, "batch": batch or None, "seed_index": None,
                        "design": _STRATEGY_DESIGN[strategy],
                        "is_copy": strategy in config.REP_POOL_STRATEGIES,
                        "sha256": meta["replications"][key]["final_sha256"]})
    return out + revision_sources()


def revision_sources() -> list[dict]:
    """Version 1.1: the revision's files (source deposit version 1.1), after the published
    study's. The solved sets in generation order (config.rev_solved_sets), each in its
    recorded parts; then the pool-backed sets as recorded; last, the large stratified test
    set, a subset of its solved source (``subset_of``)."""
    meta = data.read_metadata(config.revision_metadata_path())
    solved, pooled = meta.get("sets", {}), meta.get("pool_sets", {})
    out = []
    for key, n, _ in config.rev_solved_sets():
        stem, _, s = key.rpartition("_s") if key.startswith(("seed_s", "extension_s")) else (key, "", "")
        name, design_label = _REV_SOLVED[stem]
        parts = sorted(k for k in solved if k == key or k.startswith(f"{key}_p"))
        if sum(solved[p]["n"] for p in parts) != n:
            raise CurateError(f"revision set {key}: the recorded parts do not add up to {n} rows")
        for part in parts:
            out.append({"path": config.revision_path("final", part), "origin": "revision", "set": name,
                        "realization": None, "batch": None, "seed_index": int(s) if s else None,
                        "design": design_label, "is_copy": False, "sha256": solved[part]["final_sha256"]})
    for key, entry in pooled.items():
        if entry.get("source") not in (None, "pool") or entry["strategy"] not in _REV_POOL_SETS:
            raise CurateError(f"revision pool set {key}: unexpected source or strategy")
        seed_set = entry.get("seed_set")
        out.append({"path": config.revision_path("final", key), "origin": "revision",
                    "set": _REV_POOL_SETS[entry["strategy"]], "realization": entry.get("realization"),
                    "batch": entry.get("batch") or None,
                    "seed_index": int(seed_set.rpartition("_s")[2]) if seed_set else None,
                    "design": "pool-selection", "is_copy": True, "sha256": entry["final_sha256"]})
    from sgs import revision_large
    # the draw's hash as the revision recorded it (revision_large.build_stratified)
    draw = json.loads((config.ROOT / "outputs" / "metrics" / "revision.json").read_text())
    out.append({"path": revision_large.stratified_path(), "origin": "revision", "set": REV_STRATIFIED[0],
                "realization": None, "batch": None, "seed_index": None, "design": REV_STRATIFIED[1],
                "is_copy": True, "sha256": draw["large_tests"]["stratified_draw"]["sha256"],
                "subset_of": "test-stratified-large-source", "row_prefix": f"rev:{revision_large.SOURCE}:"})
    return out


def _relative(path: Path) -> str:
    return path.relative_to(config.ROOT).as_posix()


def read_sources(entries: list[dict]) -> pd.DataFrame:
    """All rows, as the strings stored in the files, with their membership fields."""
    frames, problems, subsets = [], [], []
    for entry in entries:
        path = Path(entry["path"])
        if not path.exists():
            problems.append(f"{_relative(path)}: missing")
            continue
        if data.sha256_file(path) != entry["sha256"]:
            problems.append(f"{_relative(path)}: SHA-256 differs from the recorded hash")
            continue
        if "subset_of" in entry:
            subsets.append(entry)
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(frame.columns) != data.FINAL_COLUMNS:
            problems.append(f"{_relative(path)}: not the final schema")
            continue
        frame.insert(0, "source_row", np.arange(len(frame)))
        frame.insert(0, "source_file", _relative(path))
        for field in _MEMBERSHIP_FIELDS:
            frame[field] = entry[field]
        frames.append(frame)
    if problems:
        raise CurateError("; ".join(problems))
    frames += [_subset_rows(entry, frames) for entry in subsets]
    rows = pd.concat(frames, ignore_index=True)
    if (rows["excluded"] != "").any():
        raise CurateError("an excluded row is present; the curated schema assumes none")
    rows.insert(0, "simulation_id", [simulation_id(v) for v in rows[INPUTS].itertuples(index=False)])
    return rows


def _subset_rows(entry: dict, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """A derived file whose rows are rows of an already read set, named by their row_id
    (``<row_prefix><index>``, the 0-based row of the set with its parts concatenated). Each
    row is taken from that set, as stored, after checking that the derived file holds the
    same inputs and outputs; its membership points to the derived file."""
    path = Path(entry["path"])
    derived = pd.read_csv(path, dtype=str, keep_default_na=False)
    source = pd.concat([f for f in frames if f["set"].iat[0] == entry["subset_of"]], ignore_index=True)
    ids = derived["row_id"]
    if not ids.str.startswith(entry["row_prefix"]).all():
        raise CurateError(f"{_relative(path)}: a row_id outside {entry['row_prefix']}")
    index = ids.str.removeprefix(entry["row_prefix"]).astype(int).to_numpy()
    if not index.size or index.max() >= len(source) or len(set(index)) != len(index):
        raise CurateError(f"{_relative(path)}: row_ids do not name distinct rows of {entry['subset_of']}")
    picked = source.iloc[index].reset_index(drop=True)
    # The derived file was written from floats parsed with pandas' default reader, so an input
    # may differ in its last digit (at most SUBSET_INPUT_ULP units in the last place); the id,
    # and every output, must be the same.
    given, stored = derived[INPUTS].astype(float).to_numpy(), picked[INPUTS].astype(float).to_numpy()
    if (np.abs(given - stored) > SUBSET_INPUT_ULP * np.spacing(stored)).any():
        raise CurateError(f"{_relative(path)}: inputs differ from the rows of {entry['subset_of']}")
    if [simulation_id(v) for v in derived[INPUTS].itertuples(index=False)] != \
            [simulation_id(v) for v in picked[INPUTS].itertuples(index=False)]:
        raise CurateError(f"{_relative(path)}: ids differ from the rows of {entry['subset_of']}")
    numeric = [c for c in data.FINAL_COLUMNS if (c in _FLOAT or c in _INT) and c not in INPUTS]
    if not np.array_equal(derived[numeric].astype(float).to_numpy(), picked[numeric].astype(float).to_numpy()):
        raise CurateError(f"{_relative(path)}: outputs differ from the rows of {entry['subset_of']}")
    text = [c for c in data.FINAL_COLUMNS if c not in _FLOAT and c not in _INT]
    if not (derived[text].to_numpy() == picked[text].to_numpy()).all():
        raise CurateError(f"{_relative(path)}: text fields differ from the rows of {entry['subset_of']}")
    picked["source_file"] = _relative(path)
    picked["source_row"] = np.arange(len(picked))
    for field in _MEMBERSHIP_FIELDS:
        picked[field] = entry[field]
    return picked


# --- the tables -------------------------------------------------------------------

def _check_groups(rows: pd.DataFrame) -> dict:
    """The rows of one simulation must agree on every deterministic output; report
    the duplicates that are separate solver runs, and the copies."""
    keys = rows.groupby("simulation_id")[INPUTS].nunique()
    if (keys > 1).any().any():
        raise CurateError("an id collision: two different input sets share a simulation_id")
    varying = rows.groupby("simulation_id")[DETERMINISTIC].nunique()
    bad = varying.columns[(varying > 1).any()].tolist()
    if bad:
        raise CurateError(f"rows of one simulation disagree on {bad}")
    runs = rows.loc[~rows["is_copy"]]
    counts = runs["simulation_id"].value_counts()
    repeated = counts[counts > 1].index
    pairs = []
    for sim in sorted(repeated):
        group = runs.loc[runs["simulation_id"] == sim]
        runtimes = group["Runtime_s"].astype(float)
        pairs.append({"simulation_id": sim, "rows": [f"{f}:{r}" for f, r in
                                                     zip(group["source_file"], group["source_row"])],
                      "runtime_s": runtimes.tolist()})
    copies = rows.loc[rows["is_copy"]]
    uncovered = sorted(set(copies["simulation_id"]) - set(runs["simulation_id"]))
    if uncovered:
        raise CurateError(f"{len(uncovered)} copied rows have no solver run in the deposit")
    return {"solver_runs": len(runs), "copied_rows": len(copies),
            "repeated_solver_runs": pairs,
            "identical_outputs": "every output column agrees across the rows of each simulation, "
                                 "Runtime_s excepted (wall-clock time)"}


def build(rows: pd.DataFrame, reruns: dict | None = None) -> dict:
    """The three tables and their checks; ``reruns`` defaults to every recorded rerun (all_reruns)."""
    checks = _check_groups(rows)
    simulations = rows.drop_duplicates("simulation_id", keep="first")[["simulation_id", *OUTPUT_COLUMNS]]
    simulations = simulations.reset_index(drop=True)
    membership = rows[["simulation_id", "source_file", "source_row", *_MEMBERSHIP_FIELDS]].reset_index(drop=True)
    for column in ("realization", "batch", "seed_index"):
        membership[column] = membership[column].astype("Int64")
    derived, checks["derived_columns"] = derived_columns(
        simulations, membership, all_reruns() if reruns is None else reruns)
    simulations = pd.concat([simulations, derived], axis=1)

    split = pd.Series("unassigned", index=simulations["simulation_id"])
    for name, set_name in _SPLIT_SETS:
        ids = membership.loc[membership["set"] == set_name, "simulation_id"].unique()
        if (split[ids] != "unassigned").any():
            raise CurateError(f"{name}: simulations already assigned to another split")
        split[ids] = name
    splits = split.rename("split").reset_index()
    splits["cell_id"], splits["split_grouped"], checks["grouped_split"] = grouped_split(simulations, splits)
    checks["disjointness"] = disjointness(simulations, splits)
    checks["id_stability"] = stable_ids(simulations)
    return {"simulations": simulations, "membership": membership, "splits": splits, "checks": checks}


def disjointness(simulations: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Minimum normalized distance from the training split, and from every non-test
    simulation, to each test split, and between every two test splits; all must exceed
    config.MIN_DISTANCE. A test split absent from the data is skipped."""
    inputs = simulations.set_index("simulation_id")[INPUTS].astype(float)
    by = splits.set_index("simulation_id")["split"]
    tests = [t for t in TEST_SPLITS if (by == t).any()]
    trees = {t: cKDTree(data.normalize(inputs.loc[by[by == t].index])) for t in tests}
    out = {}
    for test in tests:
        for name, mask in (("train", by == "train"), ("non_test", ~by.isin(TEST_SPLITS))):
            closest = float(trees[test].query(data.normalize(inputs.loc[by[mask].index]), k=1)[0].min())
            if closest <= config.MIN_DISTANCE:
                raise CurateError(f"{name} is not disjoint from {test}: {closest:.2e}")
            out[f"{name}__{test}"] = {"n": int(mask.sum()), "min_normalized_distance": closest}
    for i, first in enumerate(tests):
        for second in tests[i + 1:]:
            closest = float(trees[first].query(data.normalize(inputs.loc[by[by == second].index]), k=1)[0].min())
            if closest <= config.MIN_DISTANCE:
                raise CurateError(f"{first} is not disjoint from {second}: {closest:.2e}")
            out[f"{first}__{second}"] = {"min_normalized_distance": closest}
    out["threshold"] = config.MIN_DISTANCE
    return out


# --- the grouped split (version 1.1) ----------------------------------------------------------

GROUPED_SPLITS = ("train", "test", "unassigned")


def dimensionless_bounds() -> np.ndarray:
    """(3, 2): the bounds of angle, phi and ln c/(gamma H) that DOMAIN implies."""
    d = config.DOMAIN
    group = (np.log(d["c"][0] / (d["gamma"][1] * d["H"][1])), np.log(d["c"][1] / (d["gamma"][0] * d["H"][0])))
    return np.array([d["angle"], d["phi"], group], dtype=float)


def dimensionless_unit(frame: pd.DataFrame) -> np.ndarray:
    """(n, 3): angle, phi and ln c/(gamma H), each scaled to [0, 1] by dimensionless_bounds."""
    x = frame[INPUTS].astype(float)
    raw = np.column_stack([x["angle"], x["phi"], np.log(x["c"] / (x["gamma"] * x["H"]))])
    bounds = dimensionless_bounds()
    return (raw - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def cell_ids(unit: np.ndarray) -> np.ndarray:
    """Cell of each point in the grid of CURATED_GROUPED_CELLS per axis:
    i_angle * k^2 + i_phi * k + i_group, each index in 0..k-1 (the upper bound in the last cell)."""
    k = config.CURATED_GROUPED_CELLS
    if (unit < 0).any() or (unit > 1).any():
        raise CurateError("a point outside the dimensionless bounds of the domain")
    index = np.minimum(np.floor(unit * k).astype(int), k - 1)
    return index[:, 0] * k * k + index[:, 1] * k + index[:, 2]


def held_out_cells() -> np.ndarray:
    k = config.CURATED_GROUPED_CELLS
    n_cells = k ** 3
    rng = np.random.default_rng(config.SEEDS["curated_grouped_cells"])
    return np.sort(rng.choice(n_cells, size=round(config.CURATED_GROUPED_SHARE * n_cells), replace=False))


def grouped_split(simulations: pd.DataFrame, splits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """cell_id of every simulation, and split_grouped: the pool's analyses in a held-out cell
    are 'test', the others 'train'; every other simulation is 'unassigned'."""
    cells = cell_ids(dimensionless_unit(simulations))
    held = held_out_cells()
    pool = (splits.set_index("simulation_id")["split"].reindex(simulations["simulation_id"]) == "train").to_numpy()
    out = np.where(pool, np.where(np.isin(cells, held), "test", "train"), "unassigned")
    k = config.CURATED_GROUPED_CELLS
    counts = np.bincount(cells[pool], minlength=k ** 3)
    summary = {"space": "angle, phi, ln c/(gamma H), each scaled to [0, 1] by the bounds of the domain",
               "bounds": dimensionless_bounds().tolist(), "cells_per_axis": k,
               "held_out_cells": held.tolist(), "seed_key": "curated_grouped_cells",
               "n": {name: int((out == name).sum()) for name in GROUPED_SPLITS},
               "pool_per_cell": {"min": int(counts.min()), "median": float(np.median(counts)),
                                 "max": int(counts.max()), "empty": int((counts == 0).sum())}}
    return cells, out, summary


# --- derived columns (version 1.1) ------------------------------------------------------------
# Every one is computed from the stored outputs or taken from a committed record; none needs a
# solver run, and FOS itself is never changed.

DERIVED_COLUMNS = ["Crest_Entry", "Tension_Slices", "N_Slices", "Min_Normal_Force", "Min_Normal_Force_Slice",
                   "FOS_converged", "Converged_Max_Iter"]

# The circle enters the crest almost vertically: its centre lies within this fraction of H of
# crest height (the definition of revision_robust.crest_entry, docs/crest-entry-check/).
CREST_ENTRY_FRACTION = 0.01

# xslope 0.5.2 (solve.py) reports base tension as this one warning; nothing else appears.
_TENSION = re.compile(r"base tension on (\d+) of (\d+) slices \(worst N' = (-?\d+\.\d) at slice (\d+)\)")

# The non-converged searches solved again to convergence, outside the pipeline
# (docs/non-converged-check.md, table "Results"): (source file, 0-based row) ->
# (stored FOS at 50 iterations, converged FOS, smallest tested max_iter that converged).
NON_CONVERGED_RERUNS = {
    ("data/final/lhs_continued_300.csv", 85): ("3.628919", "3.628214", 200),
    ("data/final/test_stratified_150.csv", 65): ("2.770601", "2.770490", 100),
    ("data/replications/final/guided_log_r1.csv", 75): ("0.566515", "0.566506", 100),
    ("data/replications/final/guided_log_r5.csv", 115): ("0.830120", "0.829919", 100),
    ("data/replications/final/lhs_r1.csv", 186): ("1.100943", "1.100866", 100),
    ("data/replications/final/guided_log_g_r4.csv", 94): ("0.652006", "0.651774", 100),
    ("data/replications/final/guided_log_g_r5.csv", 157): ("0.965329", "0.965159", 100),
    ("data/replications/final/guided_log_g_r5.csv", 242): ("4.578877", "4.578877", 100),
    ("data/pool/final/pool_c001.csv", 131): ("2.086449", "2.086426", 100),
    ("data/pool/final/pool_c001.csv", 937): ("2.178500", "2.178282", 100),
    ("data/pool/final/pool_c003.csv", 376): ("3.154162", "3.153745", 200),
    ("data/pool/final/pool_c004.csv", 266): ("1.703435", "1.703366", 100),
    ("data/pool/final/pool_c005.csv", 910): ("2.051945", "2.051924", 100),
    ("data/pool/final/pool_c006.csv", 544): ("2.163868", "2.163847", 100),
    ("data/pool/final/pool_c011.csv", 947): ("2.762275", "2.762080", 100),
    ("data/pool/final/pool_c012.csv", 658): ("0.707920", "0.707519", 100),
    ("data/pool/final/pool_c015.csv", 965): ("2.927884", "2.927800", 100),
    ("data/pool/final/pool_c016.csv", 572): ("4.251066", "4.251066", 100),
    ("data/pool/final/pool_c017.csv", 471): ("2.598604", "2.598597", 100),
    ("data/pool/final/pool_c020.csv", 115): ("0.638141", "0.638141", 100),
    ("data/pool/final/pool_c020.csv", 621): ("1.453592", "1.452597", 200),
    ("data/pool/final/pool_c022.csv", 716): ("4.359769", "4.359493", 100),
}


def revision_rerun_path() -> Path:
    return config.ROOT / "docs" / "non-converged-check-revision" / "results.csv"


def revision_reruns(path: Path | None = None) -> dict:
    """The revision's non-converged searches solved again (docs/non-converged-check-revision/),
    in the form of NON_CONVERGED_RERUNS. The converged value is the FOS at the smallest tested
    max_iter whose search converged; every larger max_iter must give the same FOS, and 50 must
    reproduce the stored row. A case that converged at no tested limit is left out."""
    runs = pd.read_csv(path or revision_rerun_path(), dtype=str, keep_default_na=False)
    out = {}
    for (file, row), group in runs.groupby(["file", "row"], sort=False):
        group = group.assign(max_iter=group["max_iter"].astype(int)).sort_values("max_iter")
        stored = group["stored_fos"].iat[0]
        base = group.loc[group["max_iter"] == config.SOLVER.search_max_iter]
        if len(base) != 1 or float(base["fos"].iat[0]) != float(stored) or base["converged"].iat[0] != "False":
            raise CurateError(f"rerun {file}:{row}: the stored limit does not reproduce the stored row")
        done = group.loc[group["converged"] == "True"]
        if done.empty:
            continue
        first = done.iloc[0]
        if (group.loc[group["max_iter"] > first["max_iter"], "fos"].astype(float) != float(first["fos"])).any():
            raise CurateError(f"rerun {file}:{row}: FOS changes after the search converged")
        out[(file, int(row))] = (stored, first["fos"], int(first["max_iter"]))
    return out


def all_reruns() -> dict:
    return {**NON_CONVERGED_RERUNS, **revision_reruns()}


def tension_fields(warnings: pd.Series) -> pd.DataFrame:
    """Tension_Slices, the slice count of the surface, Min_Normal_Force (kN/m, one decimal
    as xslope prints it) and its slice, parsed from Warnings; 0 and empty when no slice base
    is in tension. xslope's slice count is SOLVER.num_slices adjusted to the geometry (29-34
    here), and it is reported only in this note."""
    parsed = warnings.str.fullmatch(_TENSION.pattern)
    other = (warnings != "") & ~parsed
    if other.any():
        raise CurateError(f"{int(other.sum())} Warnings other than xslope's base-tension note: "
                          f"{warnings[other].iloc[0]!r}")
    parts = warnings.str.extract(_TENSION.pattern).fillna("")
    return pd.DataFrame({"Tension_Slices": parts[0].where(parsed, "0"), "N_Slices": parts[1],
                         "Min_Normal_Force": parts[2], "Min_Normal_Force_Slice": parts[3]},
                        index=warnings.index)


def derived_columns(simulations: pd.DataFrame, membership: pd.DataFrame,
                    reruns: dict) -> tuple[pd.DataFrame, dict]:
    """The version 1.1 columns, as strings like the rest of the CSV, and their summary."""
    centre = simulations["Center_Y"].astype(float)
    height = simulations["H"].astype(float)
    crest = ((centre - height) / height).abs() < CREST_ENTRY_FRACTION
    out = pd.DataFrame({"Crest_Entry": np.where(crest, "True", "False")}, index=simulations.index)
    out = out.join(tension_fields(simulations["Warnings"]))

    position = pd.Series(simulations.index, index=simulations["simulation_id"])
    converged, max_iter = pd.Series("", index=simulations.index), pd.Series("", index=simulations.index)
    for key, (stored, value, iterations) in reruns.items():
        named = membership.loc[(membership["source_file"] == key[0]) & (membership["source_row"] == key[1]),
                               "simulation_id"].unique()
        if len(named) != 1:
            raise CurateError(f"non-converged rerun {key}: no such source row")
        i = position[named[0]]
        if simulations.at[i, "Search_Converged"] != "False" or float(simulations.at[i, "FOS"]) != float(stored):
            raise CurateError(f"non-converged rerun {key}: the row is not the recorded non-converged case")
        converged[i], max_iter[i] = value, str(iterations)
    out["FOS_converged"], out["Converged_Max_Iter"] = converged, max_iter
    flagged = simulations["Search_Converged"] == "False"
    if ((converged != "") & ~flagged).any():
        raise CurateError("a converged value for a search that converged")
    missing = simulations.loc[flagged & (converged == ""), "simulation_id"]
    rows = membership.loc[membership["simulation_id"].isin(missing) & ~membership["is_copy"]]
    fos = simulations["FOS"].astype(float)
    delta = (converged[converged != ""].astype(float) - fos[converged != ""]) / fos[converged != ""]
    tension = out["Tension_Slices"].astype(int)
    worst = pd.to_numeric(out["Min_Normal_Force"], errors="coerce")
    summary = {
        "crest_entry": {"definition": f"|Center_Y - H| / H < {CREST_ENTRY_FRACTION:g}",
                        "n": int(crest.sum()), "share": float(crest.mean())},
        "tension": {"n_with_tension": int((tension > 0).sum()), "share": float((tension > 0).mean()),
                    "slices_in_tension": {str(k): int(v) for k, v in tension.value_counts().sort_index().items()},
                    "worst_normal_force_kN_per_m": {q: float(worst.quantile(p)) for q, p in
                                                    (("min", 0.0), ("q05", 0.05), ("median", 0.5),
                                                     ("q95", 0.95), ("max", 1.0))},
                    "slice_of_worst": {str(k): int(v) for k, v in
                                       out.loc[tension > 0, "Min_Normal_Force_Slice"].value_counts().items()}},
        "non_converged": {"n": int(flagged.sum()), "with_converged_value": int((converged != "").sum()),
                          "max_relative_change": float(delta.abs().max()) if len(delta) else None,
                          "without_converged_value": [f"{f}:{r}" for f, r in
                                                      zip(rows["source_file"], rows["source_row"])]},
    }
    return out, summary


def typed(table: pd.DataFrame) -> pd.DataFrame:
    """Column types for Parquet; the CSV keeps the deposit's strings."""
    out = table.copy()
    for column in out.columns:
        if column in _FLOAT:
            out[column] = out[column].astype(float)
        elif column in _INT:
            out[column] = out[column].astype("int64")
        elif column in _FLOAT_OPTIONAL:
            out[column] = pd.to_numeric(out[column].replace("", np.nan)).astype(float)
        elif column in _INT_OPTIONAL:
            out[column] = pd.to_numeric(out[column].replace("", np.nan)).astype("Int64")
        elif column in _BOOL:
            if not out[column].isin(["True", "False"]).all():
                raise CurateError(f"{column}: values other than True/False")
            out[column] = out[column] == "True"
        elif column in ("Warnings", "Error", "repair_variant", "simulation_id", "source_file", "origin",
                        "set", "design", "split", "split_grouped"):
            out[column] = out[column].astype("string")
    return out


# --- the data dictionary --------------------------------------------------------------

_DICTIONARY = [
    ("simulations", "simulation_id", "string", "", "identifier derived from the five inputs (README)"),
    ("simulations", "angle", "float64", "degree", "slope angle from the horizontal, 10–65"),
    ("simulations", "phi", "float64", "degree", "effective friction angle, 5–40"),
    ("simulations", "c", "float64", "kPa", "effective cohesion, 1–75"),
    ("simulations", "gamma", "float64", "kN/m3", "unit weight, 14–22"),
    ("simulations", "H", "float64", "m", "slope height, 1–25"),
    ("simulations", "FOS", "float64", "", "factor of safety of the critical circle, Bishop simplified"),
    ("simulations", "Surface_L", "float64", "m", "left x bound of the critical slip surface, clipped "
                                                 "to the ground surface (toe at x = 0)"),
    ("simulations", "Surface_R", "float64", "m", "right x bound of the critical slip surface, clipped "
                                                 "to the ground surface"),
    ("simulations", "Surface_Min_Y", "float64", "m", "lowest point of the critical slip surface (toe at y = 0)"),
    ("simulations", "Center_X", "float64", "m", "x of the centre of the critical circle"),
    ("simulations", "Center_Y", "float64", "m", "y of the centre of the critical circle"),
    ("simulations", "Radius", "float64", "m", "radius of the critical circle"),
    ("simulations", "Circle_Nadir_Y", "float64", "m", "lowest point of the full circle, Center_Y − Radius; "
                                                      "below Surface_Min_Y for toe circles"),
    ("simulations", "At_Boundary", "bool", "", "the critical surface lies within a set fraction of H of the "
                                               "analysis-domain edge; False in every row after repair"),
    ("simulations", "slope_length", "float64", "m", "horizontal length of the slope face, H / tan(angle)"),
    ("simulations", "extension", "float64", "m", "extent of the analysis domain beyond the crest"),
    ("simulations", "depth", "float64", "m", "depth of the analysis domain below the toe; the domain "
                                             "extends by the same length to the left of the toe"),
    ("simulations", "Search_Converged", "bool", "", "the circular search met its tolerance before the "
                                                    "iteration limit"),
    ("simulations", "N_Circles_Tested", "int64", "", "trial circles evaluated by the search"),
    ("simulations", "N_Valid_Surfaces", "int64", "", "trial circles for which a FOS was obtained"),
    ("simulations", "N_Unsolved", "int64", "", "trial circles the solver left unsolved (xslope's count)"),
    ("simulations", "Runtime_s", "float64", "s", "solver wall-clock time; platform-dependent"),
    ("simulations", "Warnings", "string", "", "solver warnings, empty if none"),
    ("simulations", "Error", "string", "", "solver error, empty in every row"),
    ("simulations", "repair_variant", "string", "", "empty, or the enlarged domain on which a boundary "
                                                    "case was solved again (e.g. ext2.0x_depth1x)"),
    ("simulations", "Crest_Entry", "bool", "", "the centre of the critical circle lies within 0.01 H of "
                                               "crest height, so the circle enters the crest almost "
                                               "vertically (version 1.1; computed from Center_Y and H)"),
    ("simulations", "Tension_Slices", "int64", "", "slices whose effective base normal force N' is negative "
                                                   "(base in tension); 0 if none (version 1.1; from Warnings)"),
    ("simulations", "N_Slices", "Int64", "", "slices of the critical surface as xslope discretized it (the "
                                             "nominal 30, adjusted to the geometry); reported only with a "
                                             "base-tension warning, empty otherwise (version 1.1; from Warnings)"),
    ("simulations", "Min_Normal_Force", "float64", "kN/m", "most negative effective base normal force N' "
                                                           "per metre run, to 0.1 as xslope prints it; empty "
                                                           "when no base is in tension (version 1.1; from Warnings)"),
    ("simulations", "Min_Normal_Force_Slice", "Int64", "", "slice of Min_Normal_Force, numbered from the toe "
                                                           "side, 1 to N_Slices; empty when no base is in "
                                                           "tension (version 1.1; from Warnings)"),
    ("simulations", "FOS_converged", "float64", "", "FOS of a non-converged search solved again outside the "
                                                    "pipeline with a larger iteration limit; empty otherwise. "
                                                    "FOS keeps the stored value (version 1.1)"),
    ("simulations", "Converged_Max_Iter", "Int64", "", "the smallest tested iteration limit (100, 200, 400) "
                                                       "at which that search converged; empty otherwise "
                                                       "(version 1.1)"),
    ("membership", "simulation_id", "string", "", "the simulation this row of the source deposit holds"),
    ("membership", "source_file", "string", "", "file in the source deposit, final/ stage; the designs/ "
                                                "and results/ files of the same name hold the same row"),
    ("membership", "source_row", "int64", "", "0-based data row in source_file (header excluded)"),
    ("membership", "origin", "string", "", "pool | fixed (data/) | replication (data/replications/) | "
                                           "revision (data/revision/, source deposit version 1.1)"),
    ("membership", "set", "string", "", "the set's name in the study's code: a role for fixed sets, "
                                        "a strategy for replications, pool for the reference pool; "
                                        "for the revision, the names in the README"),
    ("membership", "realization", "Int64", "", "0 for the fixed added sets, 1–5 for the replications and the "
                                               "revision's replicated sets; empty for the seed, the test "
                                               "sets, the pool and the seed study"),
    ("membership", "batch", "Int64", "", "1–3 for the sequential batches, empty otherwise"),
    ("membership", "design", "string", "", "how the point was chosen: uniform-iid, lhs, guided-maxmin, "
                                           "guided-log-deficit, stratified-random, pool-selection, "
                                           "stratified-solved"),
    ("membership", "is_copy", "bool", "", "the row was copied from the pool, or from another set of the "
                                          "deposit (test-stratified-large), not solved again"),
    ("membership", "seed_index", "Int64", "", "1–4 for the revision's seed study (four further Latin "
                                              "hypercube seeds), empty otherwise (version 1.1)"),
    ("splits", "simulation_id", "string", "", "every simulation, once"),
    ("splits", "split", "string", "", "train (the uniform i.i.d. pool) | test-natural | test-stratified | "
                                      "test-natural-large (version 1.1) | unassigned (study designs, not "
                                      "uniform; usable for training)"),
    ("splits", "cell_id", "int64", "", "cell of the 5 × 5 × 5 grid in the dimensionless space (README): "
                                       "25 i_angle + 5 i_phi + i_group (version 1.1)"),
    ("splits", "split_grouped", "string", "", "the grouped split of the pool: test (a held-out cell) | train "
                                              "(the other cells) | unassigned (not in the pool) (version 1.1)"),
]


def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(_DICTIONARY, columns=["table", "column", "type", "unit", "description"])


# --- writing --------------------------------------------------------------------------

def _git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "describe": git("describe", "--tags", "--always"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _versions() -> dict:
    out = data.environment()
    out["packages"]["pyarrow"] = _metadata.version("pyarrow")
    # No version of sgs itself: an editable install keeps the metadata of when it was
    # installed. The commit in provenance["code"] identifies the code.
    return out


def write(tables: dict, entries: list[dict], target: Path) -> dict:
    """Write the curated deposit into ``target``; return the provenance record."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "data").mkdir(exist_ok=True)
    (target / "metadata").mkdir(exist_ok=True)
    for name in ("simulations", "membership", "splits"):
        table = tables[name]
        table.to_csv(target / "data" / f"{name}.csv", index=False)
        typed(table).to_parquet(target / "data" / f"{name}.parquet", index=False, engine="pyarrow")
    data_dictionary().to_csv(target / "metadata" / "data_dictionary.csv", index=False)
    (target / "LICENSE").write_text(LICENSE)
    (target / "README.md").write_text(readme(tables))

    # Hidden files (a .DS_Store left by the file manager, for one) are not part of the deposit, and the
    # version 1.0 provenance listed one that was never uploaded; they are left out of the record.
    files = sorted(p for p in target.rglob("*") if p.is_file() and p.name != "provenance.json"
                   and not any(part.startswith(".") for part in p.relative_to(target).parts))
    provenance = {
        "version": CURATED_VERSION,
        "derived_from": f"https://doi.org/{SOURCE_DOI}",
        "derived_from_version": SOURCE_VERSION,
        "relation": "IsDerivedFrom",
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": "PYTHONPATH=. python -m sgs.cli curate",
        "code": {"repository": "https://github.com/bteodoru/surrogate-guided-slope-stability",
                 "concept_doi": f"https://doi.org/{CODE_DOI}", **_git_state()},
        "environment": _versions(),
        "solver": {"package": "xslope", "version": config.SOLVER.package_version,
                   "method": "Bishop simplified"},
        "simulation_id": ID_RECIPE,
        "zone_edges": list(config.ZONE_EDGES),
        "counts": {"source_rows": len(tables["membership"]), "simulations": len(tables["simulations"]),
                   "splits": tables["splits"]["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict()},
        "checks": tables["checks"],
        "source_files": [{"path": _relative(Path(e["path"])), "sha256": e["sha256"]} for e in entries],
        "files": [{"path": p.relative_to(target).as_posix(), "sha256": data.sha256_file(p)} for p in files],
    }
    data.write_text_atomic(target / "metadata" / "provenance.json",
                           json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    return provenance


def run() -> dict:
    entries = sources()
    tables = build(read_sources(entries))
    provenance = write(tables, entries, out_dir())
    counts = provenance["counts"]
    print(f"[curate] {counts['source_rows']} source rows -> {counts['simulations']} simulations; "
          f"splits {counts['splits']}; written to {_relative(out_dir())}")
    return provenance


# --- texts ------------------------------------------------------------------------------

LICENSE = """\
Creative Commons Attribution 4.0 International (CC BY 4.0)

The data in this deposit are licensed under CC BY 4.0.
Legal code: https://creativecommons.org/licenses/by/4.0/legalcode

They are derived from doi:10.5281/zenodo.23618253 (CC BY 4.0), by the same authors.
"""


# The fixed role whose design each replicated strategy repeats (replicate.arm_frames composes
# the arms the same way); the pool-backed strategies have no fixed role, only their own arms.
_STRATEGY_ROLES = {"lhs": ("lhs-extension",), "guided-log": ("guided-log",),
                   "guided-log-g": ("guided-log-g",), "guided-seq": replicate.SEQ_ROLES_R0}
_POOL_STRATEGY_ARMS = {"guided-oracle": ("C-oracle", "C0-oracle"), "guided-log-p": ("C-logP", "C0-logP")}


# The revision's sets (docs/revision-declaration.md): the arms each enters. The seed-study
# arms are named with _s; seed_index says which of the four further seeds a row belongs to.
_REV_ARMS = {
    "seed-study-seed": "B_s, C-logP_s, D-P_s (seed study)",
    "seed-study-lhs-extension": "B_s (seed study)",
    "seed-study-guided-log-p": "C-logP_s, C0-logP_s (seed study)",
    "seed-study-guided-d-p": "D-P_s, D0-P_s (seed study)",
    "guided-d-oracle": "D-oracle, D0-oracle",
    "guided-d-p": "D-P, D0-P",
    "guided-seq-f": "C-seqF, C0-seqF",
    "guided-seq-p": "C-seqP, C0-seqP",
    "test-natural-large": "large test set of every replicated arm (T-natL), read beside the fixed ones",
    "test-stratified-large-source": "none; test-stratified-large is drawn from it",
    "test-stratified-large": "large test set of every replicated arm (T-stratL), read beside the fixed ones",
}


def _arms_of_role(role: str) -> list[str]:
    return [name for name, arm in config.ARMS.items() if role in arm.roles]


def arm_rows(membership: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(origin, set, arms) for the README: the training sets of the study each set enters."""
    if set(sum(_POOL_STRATEGY_ARMS.values(), ())) != set(replicate.POOL_ARMS):
        raise CurateError("the pool-backed arms differ from replicate.POOL_ARMS")
    rows = []
    for origin, name in membership[["origin", "set"]].drop_duplicates().itertuples(index=False):
        if name in config.TEST_ROLES:
            arms = "test set of every arm"
        elif name == "pool":
            arms = "none; the pool-backed sets are drawn from it"
        elif origin == "fixed":
            arms = ", ".join(_arms_of_role(name))
            if name == "seed":                    # the pool-backed arms exist in the replications only
                arms += "; in the replications also " + ", ".join(
                    a for a in replicate.POOL_ARMS if not config.ARMS.get(a) and not a.startswith("C0"))
                arms += "; in the revision also " + ", ".join(
                    a for a in revision_fit.NEW_ARMS if not a.startswith(("C0", "D0")))
        elif origin == "revision":
            arms = _REV_ARMS[name]
        elif name in _POOL_STRATEGY_ARMS:
            arms = ", ".join(_POOL_STRATEGY_ARMS[name])
        else:
            arms = ", ".join(dict.fromkeys(a for r in _STRATEGY_ROLES[name] for a in _arms_of_role(r)))
        rows.append((origin, name, arms))
    return rows


def _count(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def _scientific(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa} × 10" + str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def readme(tables: dict) -> str:
    sims, mem, splits = tables["simulations"], tables["membership"], tables["splits"]
    checks = tables["checks"]
    counts = splits["split"].value_counts()
    dis = checks["disjointness"]
    repeats = checks["repeated_solver_runs"]
    repeat_sets = sorted({m for p in repeats for m in mem.loc[mem["simulation_id"] == p["simulation_id"],
                                                               "set"]})
    edges = config.ZONE_EDGES
    zones = " | ".join([f"FOS < {edges[0]:g}"] + [f"[{a:g}, {b:g})" for a, b in zip(edges, edges[1:])]
                       + [f"FOS ≥ {edges[-1]:g}"])
    n = len(sims)
    non_test = n - sum(counts.get(t, 0) for t in TEST_SPLITS)
    arm_table = "\n".join(f"| `{name}` | {origin} | {arms} |" for origin, name, arms in arm_rows(mem))
    ablations = ", ".join([name for name, arm in config.ARMS.items() if arm.ablation]
                          + [name for name in replicate.POOL_ARMS if name.startswith("C0")]
                          + [a for a in revision_fit.NEW_ARMS if a.startswith(("C0", "D0"))]
                          + ["C0-logP_s", "D0-P_s"])
    revision = mem.loc[mem["origin"] == "revision"]
    copies = mem.loc[mem["is_copy"]]
    derived, grouped = checks["derived_columns"], checks["grouped_split"]
    conv = derived["non_converged"]
    fos, converged = sims["FOS"].astype(float), sims["FOS_converged"]
    rerun = converged != ""
    increased = int((converged[rerun].astype(float) > fos[rerun]).sum())
    missing = conv["without_converged_value"]
    if missing:
        sets = sorted(set(mem.loc[mem["source_file"].isin({m.rsplit(":", 1)[0] for m in missing}), "set"]))
        not_rerun = (f" The other {len(missing)}, in `{'` and `'.join(sets)}`, were not solved again; "
                     "their `FOS_converged` is empty.")
    else:
        not_rerun = ""
    tension = derived["tension"]
    in_tension = sims["Tension_Slices"] != "0"
    not_last = int((in_tension & (sims["Min_Normal_Force_Slice"] != sims["N_Slices"])).sum())
    first = int((in_tension & (sims["Min_Normal_Force_Slice"] == "1")).sum())
    crest = derived["crest_entry"]
    tests_disjoint = "\n".join(
        f"| `{t}` | {dis[f'train__{t}']['min_normalized_distance']:.3f} | "
        f"{dis[f'non_test__{t}']['min_normalized_distance']:.3f} |" for t in TEST_SPLITS if f"train__{t}" in dis)
    return README.format(
        arm_table=arm_table, ablations=ablations, tests_disjoint=tests_disjoint,
        n_sim=_count(n), n_rows=_count(len(mem)), n_copies=_count(len(copies)),
        n_copies_pool=_count((copies["set"] != REV_STRATIFIED[0]).sum()),
        n_copies_strat=_count((copies["set"] == REV_STRATIFIED[0]).sum()),
        n_runs=_count(checks["solver_runs"]),
        n_rev_runs=_count((~revision["is_copy"]).sum()), n_rev_rows=_count(len(revision)),
        n_repeat=len(repeats), repeat_sets=" and ".join(repeat_sets),
        n_train=_count(counts.get("train", 0)), n_tn=_count(counts.get("test-natural", 0)),
        n_ts=_count(counts.get("test-stratified", 0)), n_tnl=_count(counts.get("test-natural-large", 0)),
        n_tsl=_count((mem["set"] == REV_STRATIFIED[0]).sum()),
        n_un=_count(counts.get("unassigned", 0)), n_nontest=_count(non_test),
        threshold=config.MIN_DISTANCE, zones=zones,
        n_repaired=_count((sims["repair_variant"] != "").sum()),
        n_nonconv=conv["n"],
        n_rerun_text=(f"All {conv['n']} were" if conv["with_converged_value"] == conv["n"]
                      else f"{conv['with_converged_value']} of them were"),
        max_rel=_scientific(conv["max_relative_change"]) if conv["with_converged_value"] else "–",
        increase=("never increased" if not increased else f"increased in {increased} cases"),
        not_rerun=not_rerun,
        n_tension=_count(tension["n_with_tension"]), tension_share=f"{100 * tension['share']:.0f}%",
        tension_median=f"{tension['worst_normal_force_kN_per_m']['median']:.1f}".replace("-", "−"),
        n_not_last=_count(not_last), n_first=_count(first),
        n_crest=_count(crest["n"]), crest_share=f"{100 * crest['share']:.1f}%",
        k=grouped["cells_per_axis"], n_cells=grouped["cells_per_axis"] ** 3,
        n_held=len(grouped["held_out_cells"]), share=f"{100 * config.CURATED_GROUPED_SHARE:.0f}%",
        g_train=_count(grouped["n"]["train"]), g_test=_count(grouped["n"]["test"]),
        g_lo=f"{grouped['bounds'][2][0]:.3f}", g_hi=f"{grouped['bounds'][2][1]:.3f}",
        g_min=grouped["pool_per_cell"]["min"], g_max=grouped["pool_per_cell"]["max"],
        id_columns=", ".join(ID_COLUMNS), id_sep=ID_SEPARATOR, id_prefix=ID_PREFIX, id_hex=ID_HEX,
        id_digits=ID_DIGITS, id_margin=ID_ULP_MARGIN,
        source=SOURCE_DOI, source_v10=SOURCE_V10_DOI, source_concept=SOURCE_CONCEPT_DOI,
        source_version=SOURCE_VERSION, version=CURATED_VERSION,
        curated=(f"doi:{CURATED_DOI}" if CURATED_DOI else "version DOI to be added when the version is reserved"),
        curated_v10=CURATED_V10_DOI, curated_concept=CURATED_CONCEPT_DOI,
        release=CODE_RELEASE, code_doi=CODE_DOI,
        release_doi=(f"doi:{CODE_RELEASE_DOI}" if CODE_RELEASE_DOI
                     else "version DOI to be added when the release is archived"))


README = """\
# A curated dataset of slope stability analyses of homogeneous drained slopes using the Bishop simplified method

Version {version}. {n_sim} Bishop simplified analyses of homogeneous, drained slopes, organized for
reuse: one table of distinct analyses, a table linking each to the published files it comes from, and
fixed benchmark splits.

**This deposit adds no analysis.** It is derived from version {source_version} of doi:{source_concept}
(doi:{source}, CC BY 4.0), which holds the same analyses as they were organized for a study of sampling
strategies for surrogate models. Version {source_version} of that deposit holds every file of version
1.0 (doi:{source_v10}) unchanged and adds, under `data/revision/`, the sets solved for the study's
revision. Every value here is read from the files of that deposit, after checking each file against
the SHA-256 recorded there.

## What changed in version {version}

Version 1.0 (doi:{curated_v10}) is kept whole: every simulation, membership row and split of version
1.0 appears here with the same values, and the new columns are appended after the old ones.

- **{n_rev_runs} further analyses**, solved for the study's revision: a seed study (four further Latin
  hypercube seeds, each with its own Latin hypercube extension), a large natural test set
  (`test-natural-large`) and the Latin hypercube from which a large stratified test set was drawn.
  They are {n_rev_rows} rows of `membership` (`origin = revision`), with the revision's sets drawn from
  the reference pool, which add no analysis.
- **A further test split,** `test-natural-large` ({n_tnl} analyses).
- **A grouped split** of the reference pool in the dimensionless space (`splits.cell_id`,
  `splits.split_grouped`), for testing interpolation across regions with no training neighbour.
- **New columns:** `Crest_Entry`; the base-tension fields parsed from `Warnings` (`Tension_Slices`,
  `N_Slices`, `Min_Normal_Force`, `Min_Normal_Force_Slice`); the converged value of the non-converged
  searches (`FOS_converged`, `Converged_Max_Iter`); `membership.seed_index`.

## Files

| file | content |
|---|---|
| `data/simulations.csv`, `.parquet` | {n_sim} distinct analyses: inputs, factor of safety, critical circle, search diagnostics, derived flags |
| `data/membership.csv`, `.parquet` | {n_rows} rows, one per row of the source deposit: set, file and row |
| `data/splits.csv`, `.parquet` | the benchmark split of each analysis, its grid cell and its grouped split |
| `metadata/data_dictionary.csv` | type, unit and meaning of every column |
| `metadata/provenance.json` | source files and their SHA-256, code version, checks, SHA-256 of every file here |
| `LICENSE` | CC BY 4.0 |

The CSV files keep the values as written in the source deposit, at full float64 precision. The
Parquet files hold the same tables with typed columns.

## The model

A homogeneous slope in one Mohr–Coulomb soil, drained, with no water table and no external load. The
toe is at the origin (0, 0); the ground is flat on both sides of the slope and the same soil continues
below the toe. Bishop simplified method, circular slip surfaces, 30 slices, xslope 0.5.2. Inputs:
H 1–25 m, slope angle 10–65°, φ 5–40°, c 1–75 kPa, γ 14–22 kN/m³.

## simulation_id

The id depends only on the five inputs, never on file or row order:

    text = "{id_sep}".join(f"{{float(x):.{id_digits}g}}" for x in ({id_columns}))
    simulation_id = "{id_prefix}" + sha256(text.encode("utf-8")).hexdigest()[:{id_hex}]

Each input is written with {id_digits} significant digits before hashing. Two analyses with the same
five inputs therefore share one id. Distinct analyses keep distinct ids: they differ in at least one
input by far more than the rounding step.

The rounding also makes the id independent of how a file is read. No input in this dataset lies within
{id_margin} units in the last place of a rounding boundary, so the ids come out the same with
`pd.read_csv` (default or `float_precision="round_trip"`), Python's `csv` module, or the Parquet files.

In the source deposit, the `designs/` files of `guided_oracle_rN` and `guided_log_p_rN` hold inputs that
differ from their `final/` rows by at most 7 × 10⁻¹⁵, the last digit of a float64; so do the inputs of
`data/revision/derived/test_stratified_large.csv` against the rows it was drawn from. The `final/` rows
are exact copies of the pool's, and every output agrees exactly. Both give the same id.

## Distinct analyses and membership

The source deposit has {n_rows} rows, from {n_runs} solver runs; {n_sim} are distinct analyses.

- **Copies.** {n_copies} rows (`is_copy = True`) were not solved again: {n_copies_pool} were copied from
  the reference pool (the pool-selection sets), and the {n_copies_strat} rows of `test-stratified-large`
  are a subset of `test-stratified-large-source`. Their outputs are identical to the rows they copy.
- **Repeated analyses.** {n_repeat} analyses were solved twice, in `{repeat_sets}`. The two sets
  draw from the same candidate pool by design, and the two screeners sometimes chose the same
  candidate. The inputs are bit-identical, and every output is identical except `Runtime_s`, which is
  wall-clock time.

`simulations` keeps one row per id: the first in the order pool, fixed sets, replications, revision.
`membership` keeps every source row, so the original sets can be rebuilt by joining on
`simulation_id`.

Design labels (`membership.design`) say how a point was chosen; only `uniform-iid` (the pool) is a
random sample of the domain. `lhs` sets are Latin hypercube designs. `guided-maxmin`,
`guided-log-deficit` and `pool-selection` were chosen from predicted or computed FOS to reach a target
distribution of FOS. `stratified-random` is a test set stratified on predicted FOS; `stratified-solved`
(`test-stratified-large`) takes up to 30 analyses at random in each of 15 geometric bins of solved FOS
on [0.5, 6] from `test-stratified-large-source`.

## Correspondence with the study

The sets are the designs of the study that generated the data (arXiv:2610.38815). There, each training
set ("arm") joins the seed to one added set; an arm whose name has a 0 ({ablations}) is the added
set alone, an ablation. A set of origin `replication`, or of origin `revision` with a realization,
enters the arms of its realization only; a seed-study set enters the arms of its seed
(`seed_index`) only.

| set | origin | arms of the study it enters |
|---|---|---|
{arm_table}

## Benchmark splits

| split | analyses | content |
|---|---|---|
| `train` | {n_train} | the reference pool, uniform i.i.d. over the domain |
| `test-natural` | {n_tn} | Latin hypercube over the domain |
| `test-stratified` | {n_ts} | stratified on predicted FOS |
| `test-natural-large` | {n_tnl} | Latin hypercube over the domain (version 1.1) |
| `unassigned` | {n_un} | the study's other designs; not uniform, usable as extra training data |

`test-stratified-large` ({n_tsl} analyses, stratified on solved FOS) is not a split of its own: its
analyses are `unassigned`, and `membership` names them.

The test splits are disjoint from every other analysis. Minimum distance in the domain normalized to the
unit hypercube (the threshold for "the same point" is {threshold:g}):

| test split | to `train` | to all {n_nontest} non-test analyses |
|---|---|---|
{tests_disjoint}

**Main task:** predict FOS from the five inputs; train on `train`, report on each test split the RMSE
of log(FOS), overall and per FOS zone: {zones}. The two smaller test sets are those of the study;
`test-natural-large` gives tighter estimates of the same natural-distribution error.

### Grouped split

Random splits of a dense uniform sample put a near training neighbour beside almost every test point.
The factor of safety of this model depends mainly on the slope angle, φ and the dimensionless group
c/(γH), so neighbours in that space count, not in the five raw inputs. `splits.split_grouped` holds out
whole regions of that space:

- the space (angle, φ, ln c/(γH)), each axis scaled to [0, 1] by the bounds the domain implies
  (ln c/(γH) from {g_lo} to {g_hi});
- cut into {k} × {k} × {k} = {n_cells} cells (`splits.cell_id`); every cell holds pool analyses
  ({g_min} to {g_max});
- {n_held} cells ({share}), drawn with a declared seed, held out: the pool's analyses in them are
  `test` ({g_test}), those in the other cells `train` ({g_train}); every other analysis is
  `unassigned`.

It tests interpolation across gaps with no training neighbour in the cell. It does not hold out a
compact region, which would mix extrapolation with a change of regime (frictional against cohesive
slopes).

## Quality control

- **Boundary repair:** {n_repaired} analyses had a critical surface at the analysis-domain edge and were
  solved again on progressively enlarged domains (`repair_variant`).
- **Non-converged searches:** {n_nonconv} analyses stopped at the iteration limit
  (`Search_Converged = False`); `FOS` keeps the stored value. {n_rerun_text} solved again with
  larger iteration limits, outside the pipeline (`FOS_converged`, `Converged_Max_Iter`): FOS moved by at
  most {max_rel} (relative) and {increase}.{not_rerun}
- **Base tension:** in {n_tension} analyses ({tension_share}) the effective normal force N' at the base
  of at least one slice of the critical circle is negative (`Tension_Slices`; median of the most
  negative N', {tension_median} kN/m). In all but {n_not_last} of them the most negative N' is at the
  last slice, at the crest end of the circle ({n_first} at the first, at the toe end). The model has no
  tension crack, and FOS is reported as solved.
- **Crest entries:** {n_crest} analyses ({crest_share}) have `Crest_Entry = True`: the centre of the
  critical circle lies within 0.01 H of crest height, so the circle enters the crest almost vertically.
  On such circles FOS depends on how a program discretizes the steep last slice.
- **No analysis excluded.**
- **Independent check with another program.** 15 analyses of `test-natural`, 3 per FOS zone, were
  solved blind in Slide2 9.044 (Rocscience), Bishop simplified, with its own circular search. The
  checker received the inputs and the geometry, not the xslope FOS.
  - The FOS agree within 0.6% in 14 of the 15 cases. The median absolute difference is 0.26%.
  - xslope gave the lower FOS in 13 cases, so Slide2's search found no lower critical circle in those.
  - The exception (1.8%, at FOS 0.94) is a steep, strongly cohesive slope. Its critical circle enters
    the crest almost vertically, where the two programs discretize the last slice differently; on the
    same circle they converge from opposite sides as the slices get finer.
  - The cases, both programs' values and the analysis of the exception are in
    `docs/solver-crosscheck/` of the code repository.

## Code and regeneration

This deposit is produced by `sgs curate` in https://github.com/bteodoru/surrogate-guided-slope-stability,
release {release} ({release_doi}; all versions: doi:{code_doi}). Unpack the `data.zip` of version
{source_version} of the source deposit in the repository root, then:

    python3.14 -m venv .venv
    .venv/bin/pip install -r requirements.lock
    .venv/bin/pip install --no-deps -e .
    PYTHONPATH=. .venv/bin/sgs curate

Instead of the lock file, `.venv/bin/pip install -e ".[curate]"` installs the package with the Parquet
writer. `sgs curate` checks every source file against the SHA-256 in `run_metadata.json` (and
`data/revision/run_metadata.json`) before it builds anything, and writes `data/curated/`.
`metadata/provenance.json` records the commit it ran on.

## Licence and citation

CC BY 4.0. Please cite this deposit, doi:{curated_concept} (all versions; this is version {version},
{curated}), and the source deposit, doi:{source_concept} (version {source_version}, doi:{source}).
"""
