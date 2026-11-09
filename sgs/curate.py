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
import subprocess
from importlib import metadata as _metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sgs import config, data, replicate

SOURCE_DOI = "10.5281/zenodo.23618253"
CODE_DOI = "10.5281/zenodo.23727658"          # concept DOI of the code (all versions)
CODE_RELEASE = "v1.1.0"
CODE_RELEASE_DOI = None                       # the version DOI of CODE_RELEASE, once the release exists

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
_INT = ["N_Circles_Tested", "N_Valid_Surfaces", "N_Unsolved"]
_BOOL = ["At_Boundary", "Search_Converged"]

SPLITS = ("train", "test-natural", "test-stratified", "unassigned")

# design labels for the replicated strategies; the fixed roles carry their own (config.ROLES)
_STRATEGY_DESIGN = {"lhs": "lhs", "guided-log": "guided-log-deficit", "guided-log-g": "guided-log-deficit",
                    "guided-seq": "guided-log-deficit", "guided-oracle": "pool-selection",
                    "guided-log-p": "pool-selection"}


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
                    "realization": None, "batch": None, "design": "uniform-iid", "is_copy": False,
                    "sha256": meta["pool"][key]["final_sha256"]})
    for role in config.ROLE_ORDER:
        spec = config.ROLES[role]
        fixed = role == "seed" or role in config.TEST_ROLES
        batch = int(role[-1]) if role.startswith("guided-seq-") else None
        out.append({"path": config.final_path(role), "origin": "fixed", "set": role,
                    "realization": None if fixed else 0, "batch": batch, "design": spec.design,
                    "is_copy": False, "sha256": meta["datasets"][role]["final_sha256"]})
    for r in range(1, config.REP_REALIZATIONS + 1):
        for key, strategy, batch in replicate.set_keys(r):
            out.append({"path": config.replication_path("final", key), "origin": "replication",
                        "set": strategy, "realization": r, "batch": batch or None,
                        "design": _STRATEGY_DESIGN[strategy],
                        "is_copy": strategy in config.REP_POOL_STRATEGIES,
                        "sha256": meta["replications"][key]["final_sha256"]})
    return out


def _relative(path: Path) -> str:
    return path.relative_to(config.ROOT).as_posix()


def read_sources(entries: list[dict]) -> pd.DataFrame:
    """All rows, as the strings stored in the files, with their membership fields."""
    frames, problems = [], []
    for entry in entries:
        path = Path(entry["path"])
        if not path.exists():
            problems.append(f"{_relative(path)}: missing")
            continue
        if data.sha256_file(path) != entry["sha256"]:
            problems.append(f"{_relative(path)}: SHA-256 differs from run_metadata.json")
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(frame.columns) != data.FINAL_COLUMNS:
            problems.append(f"{_relative(path)}: not the final schema")
            continue
        frame.insert(0, "source_row", np.arange(len(frame)))
        frame.insert(0, "source_file", _relative(path))
        for field in ("origin", "set", "realization", "batch", "design", "is_copy"):
            frame[field] = entry[field]
        frames.append(frame)
    if problems:
        raise CurateError("; ".join(problems))
    rows = pd.concat(frames, ignore_index=True)
    if (rows["excluded"] != "").any():
        raise CurateError("an excluded row is present; the curated schema assumes none")
    rows.insert(0, "simulation_id", [simulation_id(v) for v in rows[INPUTS].itertuples(index=False)])
    return rows


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


def build(rows: pd.DataFrame) -> dict:
    checks = _check_groups(rows)
    simulations = rows.drop_duplicates("simulation_id", keep="first")[["simulation_id", *OUTPUT_COLUMNS]]
    simulations = simulations.reset_index(drop=True)
    membership = rows[["simulation_id", "source_file", "source_row", "origin", "set", "realization",
                       "batch", "design", "is_copy"]].reset_index(drop=True)
    membership["realization"] = membership["realization"].astype("Int64")
    membership["batch"] = membership["batch"].astype("Int64")

    split = pd.Series("unassigned", index=simulations["simulation_id"])
    for name, sets in (("train", ["pool"]), ("test-natural", ["test-natural"]),
                       ("test-stratified", ["test-stratified"])):
        ids = membership.loc[membership["set"].isin(sets), "simulation_id"].unique()
        if (split[ids] != "unassigned").any():
            raise CurateError(f"{name}: simulations already assigned to another split")
        split[ids] = name
    splits = split.rename("split").reset_index()
    checks["disjointness"] = disjointness(simulations, splits)
    checks["id_stability"] = stable_ids(simulations)
    return {"simulations": simulations, "membership": membership, "splits": splits, "checks": checks}


def disjointness(simulations: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Minimum normalized distance from the training split, and from every non-test
    simulation, to each test set; both must exceed config.MIN_DISTANCE."""
    inputs = simulations.set_index("simulation_id")[INPUTS].astype(float)
    by = splits.set_index("simulation_id")["split"]
    out = {}
    for test in config.TEST_ROLES:
        tree = cKDTree(data.normalize(inputs.loc[by[by == test].index]))
        for name, mask in (("train", by == "train"), ("non_test", ~by.isin(config.TEST_ROLES))):
            closest = float(tree.query(data.normalize(inputs.loc[by[mask].index]), k=1)[0].min())
            if closest <= config.MIN_DISTANCE:
                raise CurateError(f"{name} is not disjoint from {test}: {closest:.2e}")
            out[f"{name}__{test}"] = {"n": int(mask.sum()), "min_normalized_distance": closest}
    out["threshold"] = config.MIN_DISTANCE
    return out


def typed(table: pd.DataFrame) -> pd.DataFrame:
    """Column types for Parquet; the CSV keeps the deposit's strings."""
    out = table.copy()
    for column in out.columns:
        if column in _FLOAT:
            out[column] = out[column].astype(float)
        elif column in _INT:
            out[column] = out[column].astype("int64")
        elif column in _BOOL:
            if not out[column].isin(["True", "False"]).all():
                raise CurateError(f"{column}: values other than True/False")
            out[column] = out[column] == "True"
        elif column in ("Warnings", "Error", "repair_variant", "simulation_id", "source_file", "origin",
                        "set", "design", "split"):
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
    ("simulations", "depth", "float64", "m", "extent of the analysis domain below the toe, and to the "
                                             "left of it"),
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
    ("membership", "simulation_id", "string", "", "the simulation this row of the source deposit holds"),
    ("membership", "source_file", "string", "", "file in the source deposit, final/ stage; the designs/ "
                                                "and results/ files of the same name hold the same row"),
    ("membership", "source_row", "int64", "", "0-based data row in source_file (header excluded)"),
    ("membership", "origin", "string", "", "pool | fixed (data/) | replication (data/replications/)"),
    ("membership", "set", "string", "", "the set's name in the study's code: a role for fixed sets, "
                                        "a strategy for replications, pool for the reference pool"),
    ("membership", "realization", "Int64", "", "0 for the fixed added sets, 1–5 for the replications; "
                                               "empty for the seed, the test sets and the pool"),
    ("membership", "batch", "Int64", "", "1–3 for the sequential batches, empty otherwise"),
    ("membership", "design", "string", "", "how the point was chosen: uniform-iid, lhs, guided-maxmin, "
                                           "guided-log-deficit, stratified-random, pool-selection"),
    ("membership", "is_copy", "bool", "", "the row was copied from the pool, not solved again"),
    ("splits", "simulation_id", "string", "", "every simulation, once"),
    ("splits", "split", "string", "", "train (the uniform i.i.d. pool) | test-natural | test-stratified | "
                                      "unassigned (study designs, not uniform; usable for training)"),
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

    files = sorted(p for p in target.rglob("*") if p.is_file() and p.name != "provenance.json")
    provenance = {
        "derived_from": f"https://doi.org/{SOURCE_DOI}",
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
    non_test = n - counts.get("test-natural", 0) - counts.get("test-stratified", 0)
    return README.format(
        n_sim=f"{n:,}", n_rows=f"{len(mem):,}", n_copies=f"{checks['copied_rows']:,}",
        n_repeat=len(repeats), repeat_sets=" and ".join(repeat_sets),
        n_train=f"{counts.get('train', 0):,}", n_tn=f"{counts.get('test-natural', 0):,}",
        n_ts=f"{counts.get('test-stratified', 0):,}", n_un=f"{counts.get('unassigned', 0):,}",
        n_nontest=f"{non_test:,}",
        d_train_tn=dis["train__test-natural"]["min_normalized_distance"],
        d_train_ts=dis["train__test-stratified"]["min_normalized_distance"],
        d_all_tn=dis["non_test__test-natural"]["min_normalized_distance"],
        d_all_ts=dis["non_test__test-stratified"]["min_normalized_distance"],
        threshold=config.MIN_DISTANCE, zones=zones,
        n_repaired=int((sims["repair_variant"] != "").sum()),
        n_nonconv=int((sims["Search_Converged"] != "True").sum()),
        id_columns=", ".join(ID_COLUMNS), id_sep=ID_SEPARATOR, id_prefix=ID_PREFIX, id_hex=ID_HEX,
        id_digits=ID_DIGITS, id_margin=ID_ULP_MARGIN, source=SOURCE_DOI,
        release=CODE_RELEASE, code_doi=CODE_DOI,
        release_doi=(f"doi:{CODE_RELEASE_DOI}" if CODE_RELEASE_DOI
                     else "version DOI to be added when the release is archived"))


README = """\
# A curated dataset of slope stability analyses of homogeneous drained slopes using the Bishop simplified method

{n_sim} Bishop simplified analyses of homogeneous, drained slopes, organized for reuse: one table of
distinct analyses, a table linking each to the published files it comes from, and fixed benchmark
splits.

**This deposit adds no analysis.** It is derived from doi:{source} (CC BY 4.0), which holds the same
analyses as they were organized for a study of sampling strategies for surrogate models. Every value
here is read from the `final/` files of that deposit, after checking each file against the SHA-256
recorded there.

## Files

| file | content |
|---|---|
| `data/simulations.csv`, `.parquet` | {n_sim} distinct analyses: inputs, factor of safety, critical circle, search diagnostics |
| `data/membership.csv`, `.parquet` | {n_rows} rows, one per row of the source deposit: set, file and row |
| `data/splits.csv`, `.parquet` | the benchmark split of each analysis |
| `metadata/data_dictionary.csv` | type, unit and meaning of every column |
| `metadata/provenance.json` | source files and their SHA-256, code version, checks, SHA-256 of every file here |
| `LICENSE` | CC BY 4.0 |

The CSV files keep the values as written in the source deposit, at full float64 precision. The
Parquet files hold the same tables with typed columns.

## The model

A homogeneous slope in one Mohr–Coulomb soil, drained, with no water table and no external load. The
toe is at (0, 0) and the crest at (L, H); the ground is flat on both sides and the same soil continues
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
differ from their `final/` rows by at most 7 × 10⁻¹⁵, the last digit of a float64. The `final/` rows
are exact copies of the pool's. Both give the same id.

## Distinct analyses and membership

The source deposit has {n_rows} rows; {n_sim} are distinct analyses.

- **Copies.** {n_copies} rows (`is_copy = True`; sets `guided-oracle` and `guided-log-p`) were copied
  from the reference pool and not solved again. Their outputs are identical to the pool's.
- **Repeated analyses.** {n_repeat} analyses were solved twice, in `{repeat_sets}`. The two sets
  draw from the same candidate pool by design, and the two screeners sometimes chose the same
  candidate. The inputs are bit-identical, and every output is identical except `Runtime_s`, which is
  wall-clock time.

`simulations` keeps one row per id: the first in the order pool, fixed sets, replications.
`membership` keeps every source row, so the original sets can be rebuilt by joining on
`simulation_id`.

Design labels (`membership.design`) say how a point was chosen; only `uniform-iid` (the pool) is a
random sample of the domain. `lhs` sets are Latin hypercube designs. `guided-maxmin`,
`guided-log-deficit` and `pool-selection` were chosen from predicted or computed FOS to reach a target
distribution of FOS. `stratified-random` is a test set stratified on predicted FOS.

## Benchmark splits

| split | analyses | content |
|---|---|---|
| `train` | {n_train} | the reference pool, uniform i.i.d. over the domain |
| `test-natural` | {n_tn} | Latin hypercube over the domain |
| `test-stratified` | {n_ts} | stratified on predicted FOS |
| `unassigned` | {n_un} | the study's other designs; not uniform, usable as extra training data |

Both test sets are disjoint from every other analysis. The minimum distance in the domain normalized
to the unit hypercube is {d_train_tn:.3f} (train to test-natural) and {d_train_ts:.3f} (train to
test-stratified). Over all {n_nontest} non-test analyses it is {d_all_tn:.3f} and {d_all_ts:.3f}. The
threshold for "the same point" is {threshold:g}.

**Main task:** predict FOS from the five inputs; train on `train`, report on each test set the RMSE
of log(FOS), overall and per FOS zone: {zones}.

## Quality control

- **Boundary repair:** {n_repaired} analyses had a critical surface at the analysis-domain edge and were solved
  again on progressively enlarged domains (`repair_variant`).
- **Non-converged searches:** {n_nonconv} analyses stopped at the iteration limit (`Search_Converged = False`).
  Solved again to convergence, their FOS moved by at most 6.8 × 10⁻⁴ (relative).
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
release {release} ({release_doi}; all versions: doi:{code_doi}). Unpack the source deposit's `data.zip`
in the repository root, then:

    python3.14 -m venv .venv
    .venv/bin/pip install -r requirements.lock
    .venv/bin/pip install --no-deps -e .
    PYTHONPATH=. .venv/bin/sgs curate

Instead of the lock file, `.venv/bin/pip install -e ".[curate]"` installs the package with the Parquet
writer. `sgs curate` checks every source file against the SHA-256 in `run_metadata.json` before it
builds anything, and writes `data/curated/`. `metadata/provenance.json` records the commit it ran on.

## Licence and citation

CC BY 4.0. Please cite this deposit, and the source deposit doi:{source}.
"""
