"""Schema, the one definition of the derived feature, loading, validation of
the final sets, and the run metadata file."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
from dataclasses import dataclass
from importlib import metadata as _metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sgs import config

RESULT_COLUMNS = [
    *config.PARAMS,
    "FOS",
    "Surface_L", "Surface_R", "Surface_Depth",
    "Center_X", "Center_Y", "Radius",
    "At_Boundary",
    "slope_length", "extension", "depth",
    "Search_Converged", "N_Circles_Tested", "N_Valid_Surfaces", "N_Unsolved",
    "Runtime_s", "Warnings", "Error",
]
META_COLUMNS = ["repair_variant", "excluded"]
FINAL_COLUMNS = RESULT_COLUMNS + META_COLUMNS
EXCLUSION_REASONS = ("solver_error", "unresolved")

_TEXT_COLUMNS = ("At_Boundary", "Search_Converged", "Warnings", "Error", "repair_variant", "excluded")
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "shapely", "xslope")


class DataError(RuntimeError):
    """A final set, or a pair of them, violates the data contract."""


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``gamma_H_c = gamma * H / c``. Defined only here."""
    out = frame.copy()
    out["gamma_H_c"] = out["gamma"] * out["H"] / out["c"]
    return out


def read_table(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={c: "string" for c in _TEXT_COLUMNS})
    for column in _TEXT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].fillna("")
    return frame


def is_true(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower() == "true"


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[(frame["excluded"] == "") & frame["FOS"].notna()]


def normalize(frame: pd.DataFrame) -> np.ndarray:
    lo = np.array([config.DOMAIN[p][0] for p in config.PARAMS], dtype=float)
    hi = np.array([config.DOMAIN[p][1] for p in config.PARAMS], dtype=float)
    return (frame[list(config.PARAMS)].to_numpy(float) - lo) / (hi - lo)


def _final_table(role: str) -> pd.DataFrame:
    path = config.final_path(role)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs generate --role {role}`")
    return read_table(path)


def load_role(role: str) -> pd.DataFrame:
    """Usable rows of a final set, with row identifiers and features."""
    frame = _final_table(role)
    frame.insert(0, "row_id", [f"{role}:{i}" for i in range(len(frame))])
    return add_features(usable(frame)).reset_index(drop=True)


def load_arm(arm: str) -> pd.DataFrame:
    return pd.concat([load_role(role) for role in config.ARMS[arm].roles], ignore_index=True)


@dataclass(frozen=True)
class TestSet:
    __test__ = False       # not a pytest class

    name: str
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray


def load_test(role: str) -> TestSet:
    if role not in config.TEST_ROLES:
        raise ValueError(f"{role!r} is not a test role ({', '.join(config.TEST_ROLES)})")
    frame = load_role(role)
    return TestSet(name=role,
                   X=frame[list(config.FEATURES)].reset_index(drop=True),
                   y=frame["FOS"].to_numpy(float),
                   ids=frame["row_id"].to_numpy(str))


# --- validation -----------------------------------------------------------------

def set_problems(frame: pd.DataFrame, role: str) -> list[str]:
    """Violations of the data contract by one final set; empty when clean."""
    if list(frame.columns) != FINAL_COLUMNS:
        return [f"schema: columns {list(frame.columns)}"]
    problems = []
    expected = config.ROLES[role].n
    if len(frame) != expected:
        problems.append(f"row_count: {len(frame)} rows, expected {expected}")
    outside = {p: int((~frame[p].between(lo, hi)).sum()) for p, (lo, hi) in config.DOMAIN.items()}
    if any(outside.values()):
        problems.append(f"domain: rows outside the domain {outside}")
    unknown = sorted(set(frame["excluded"]) - {"", *EXCLUSION_REASONS})
    if unknown:
        problems.append(f"exclusion_reasons: unknown {unknown}")
    kept = frame["excluded"] == ""
    bad_fos = int((kept & ~(frame["FOS"] > 0)).sum())
    if bad_fos:
        problems.append(f"fos_present: {bad_fos} kept rows without a positive FOS")
    boundary = int((kept & is_true(frame["At_Boundary"])).sum())
    if boundary:
        problems.append(f"no_untreated_boundary: {boundary} kept rows at the domain boundary")
    unit = normalize(frame)
    closest = float(cKDTree(unit).query(unit, k=2)[0][:, 1].min())
    if closest <= config.MIN_DISTANCE:
        problems.append(f"no_internal_duplicates: closest pair {closest:.2e}")
    return problems


def validate_sets(roles) -> None:
    """Fail unless every set is clean and every pair is disjoint."""
    frames = {role: _final_table(role) for role in roles}
    problems = [f"{role}: {p}" for role, frame in frames.items() for p in set_problems(frame, role)]
    for a, b in itertools.combinations(sorted(frames), 2):
        closest = float(cKDTree(normalize(frames[b])).query(normalize(frames[a]), k=1)[0].min())
        if closest <= config.MIN_DISTANCE:
            problems.append(f"{a}/{b}: not disjoint (closest cross pair {closest:.2e})")
    if problems:
        raise DataError("; ".join(problems))


# --- provenance -----------------------------------------------------------------

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def _json_default(value):
    if hasattr(value, "item") and np.ndim(value) == 0:   # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):                          # numpy array
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_metadata() -> dict:
    path = config.metadata_path()
    return json.loads(path.read_text()) if path.exists() else {}


def record(section: str, key: str, entry: dict) -> None:
    """Set ``run_metadata.json[section][key] = entry`` and write the file."""
    meta = read_metadata()
    meta.setdefault(section, {})[key] = entry
    path = config.metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)
